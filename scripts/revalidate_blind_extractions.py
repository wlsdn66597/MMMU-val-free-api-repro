"""Revalidate cached reference-blind judge outputs on CPU; never generate new answers."""
import argparse
from collections import Counter
import json
from pathlib import Path

from mmmu_repro import blind_evaluation
from mmmu_repro.blind_evaluation import POLICY, build_prompt, extraction_view, parse_output
from mmmu_repro.common import RunLock, digest, file_hash, lock_manifest, read_jsonl, save_json
from mmmu_repro.judge import checked_predictions
if __package__:
    from .evaluate_blind_local_judge import write_report
else:
    from evaluate_blind_local_judge import write_report


def run(args):
    infer, cached, out = (Path(value).resolve() for value in
                          (args.inference_dir, args.cached_dir, args.output_dir))
    for source in (infer, cached):
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("Use a new output directory separate from both source directories")
    if (cached / ".running.lock").exists():
        raise ValueError("Cached extraction is locked; wait until its process has finished")
    manifest_path = cached / "blind_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("policy") not in ("blind-final-answer-v1", POLICY):
        raise ValueError("Only reference-blind extraction caches can be revalidated")
    scope = manifest["scope"]
    if scope not in ("all", "open", "multiple-choice"):
        raise ValueError("Unknown cached scope")
    rows = [row for row in checked_predictions(infer)
            if scope == "all" or row["question_type"] == scope]
    views = {row["id"]: extraction_view(row) for row in rows}
    source_sha = digest([dict(id=row["id"], input=views[row["id"]]) for row in rows])
    if source_sha != manifest["inputs_sha256"]:
        raise ValueError("Cached inputs do not match saved VLM responses")
    cache_path = cached / "extractions.jsonl"
    original = read_jsonl(cache_path)
    records, seen, transitions = [], set(), Counter()
    for old in original:
        sample_id = old["id"]
        if sample_id not in views or sample_id in seen:
            raise ValueError("Duplicate or unknown cached extraction ID")
        view = views[sample_id]
        if old["input_sha256"] != digest(view) or old["prompt_sha256"] != digest(build_prompt(view)):
            raise ValueError("Cached input or generation prompt changed; cannot reuse this output")
        parsed = parse_output(old["raw_judge_output"], old["finish_reason"], view)
        # Whitelist cache fields: never carry references or correctness into extraction.
        record = {key: old[key] for key in ("id", "input_sha256", "prompt_sha256",
                   "raw_judge_output", "finish_reason", "output_tokens")}
        record.update(parsed)
        records.append(record)
        seen.add(sample_id)
        transitions[f"{old['status']} -> {parsed['status']}"] += 1
    if seen != set(views):
        raise ValueError("Cached extraction is incomplete; no final accuracy reported")
    identity = dict(policy=POLICY, inputs_sha256=source_sha, scope=scope,
                    policy_source_sha256=file_hash(blind_evaluation.__file__),
                    script_sha256=file_hash(__file__), mode="cpu_cached_revalidation",
                    cache_path=str(cached), cache_manifest=manifest,
                    cache_manifest_sha256=file_hash(manifest_path),
                    cache_records_sha256=file_hash(cache_path))
    with RunLock(out):
        lock_manifest(out, "blind_manifest.json", identity)
        target = out / "extractions.jsonl"
        if target.exists() and read_jsonl(target) != records:
            raise ValueError("Destination records differ; use a new output directory")
        temporary = out / "extractions.jsonl.tmp"
        temporary.write_text("".join(json.dumps(record, ensure_ascii=False)+"\n"
                                     for record in records), encoding="utf-8")
        temporary.replace(target)
        summary = write_report(out, rows, records, scope)
        summary["revalidation"] = dict(mode="cpu_cached_revalidation", new_model_calls=0,
                                        status_transitions=dict(transitions))
        save_json(out / "summary.json", summary)
        save_json(out / "progress.json", dict(state="complete", completed=len(records), n=len(rows)))
        with (out / "report.md").open("a", encoding="utf-8") as report:
            report.write("\nCPU revalidation of unchanged cached judge outputs; new model calls: 0.\n")
            report.write("Only presentation markup, unambiguous option labels/text, and supported "
                         "answer substrings or format-equivalent scalar expressions are normalized. "
                         "Missing quotes, conflicting choices, and unsupported answer rewrites remain unparsed.\n")
            for transition, count in sorted(transitions.items()):
                report.write(f"- {transition}: {count}\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--cached-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
