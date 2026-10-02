"""Extract saved final answers without gold, then score them separately."""
import argparse
from collections import Counter
import json
from pathlib import Path

from mmmu_repro import blind_evaluation
from mmmu_repro.blind_evaluation import POLICY, build_prompt, extraction_view, parse_output, score_records
from mmmu_repro.common import RunLock, append_jsonl, digest, file_hash, lock_manifest, read_jsonl, save_json
from mmmu_repro.judge import checked_predictions
if __package__:
    from .evaluate_local_model_judge import rope_settings, token_prompt
    from .prepare_local_judge_model import resolve
else:
    from evaluate_local_model_judge import rope_settings, token_prompt
    from prepare_local_judge_model import resolve


def validate_records(rows, records):
    sources = {row["id"]: row for row in rows}
    seen = set()
    for record in records:
        row = sources.get(record["id"])
        if row is None or record["id"] in seen:
            raise ValueError("Duplicate or unknown blind extraction")
        view = extraction_view(row)
        if record["input_sha256"] != digest(view) or record["prompt_sha256"] != digest(build_prompt(view)):
            raise ValueError("Stale blind extraction")
        checked = parse_output(record["raw_judge_output"], record["finish_reason"], view)
        if any(record.get(key) != value for key, value in checked.items()):
            raise ValueError("Saved blind extraction does not match validated model output")
        seen.add(row["id"])
    return seen


def write_report(out, rows, records, scope):
    scored = score_records(rows, records)
    def group(items):
        n = len(items)
        exact = sum(item["score"]["exact_correct"] for item in items)
        precision = sum(item["score"]["precision_correct"] for item in items)
        return dict(n=n, exact_correct=exact, precision_correct=precision,
                    exact_accuracy_pct=round(100*exact/n, 4) if n else None,
                    precision_accuracy_pct=round(100*precision/n, 4) if n else None,
                    unparsed=sum(item["answer"] is None for item in items),
                    statuses=dict(Counter(item["status"] for item in items)),
                    unit_review_required=sum(item["score"]["unit_review_required"] for item in items))
    summary = dict(policy=POLICY, scope=scope, full_validation=scope == "all", state="complete",
                   references_sha256=digest([dict(id=row["id"], reference=row["answer"]) for row in rows]),
                   overall=group(scored),
                   multiple_choice=group([item for item in scored if item["question_type"] == "multiple-choice"]),
                   open=group([item for item in scored if item["question_type"] == "open"]))
    save_json(out / "summary.json", summary)
    temporary = out / "scored_records.jsonl.tmp"
    temporary.write_text("".join(json.dumps(item, ensure_ascii=False)+"\n" for item in scored), encoding="utf-8")
    temporary.replace(out / "scored_records.jsonl")
    lines = ["# Reference-blind local final-answer extraction", "",
             "All selected responses were extracted again; no legacy Qwen rule result is reused.",
             "The extractor receives no reference answers or correctness labels. No VLM regeneration or API calls.", "",
             "| Group | N | Exact correct | Exact accuracy % | Precision correct | Precision accuracy % | Unparsed |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("overall", "multiple_choice", "open"):
        value = summary[name]
        lines.append(f"| {name} | {value['n']} | {value['exact_correct']} | {value['exact_accuracy_pct']} | "
                     f"{value['precision_correct']} | {value['precision_accuracy_pct']} | {value['unparsed']} |")
    lines.extend(["", "Exact: normalized text aliases or exact scalar arithmetic equality.",
                  "Precision: also allows scalar answers rounded to a decimal reference's written precision (half-up). "
                  "Integer and fractional references stay exact; no relative tolerance is used.",
                  "Aliases in a serialized list are compared individually. Only extracted final answers are scored, "
                  "never numbers/words anywhere in the full reasoning.",
                  "Units are not converted. An explicit reference unit must match; if the reference has no unit, "
                  "recognized answer units are stripped from magnitude matching and flagged for review. Percent is converted to a ratio.",
                  f"Unit review cases: {summary['overall']['unit_review_required']}",
                  "Unparsed/invalid JSON/unverifiable evidence count as wrong. Model extraction is still fallible; inspect evidence.",
                  "This is an additional evaluation policy, not the official Qwen judge score."])
    (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return summary


def run(args):
    infer, out = Path(args.inference_dir).resolve(), Path(args.output_dir).resolve()
    if out == infer or out in infer.parents or infer in out.parents:
        raise ValueError("Choose an output directory separate from inference")
    all_rows = checked_predictions(infer)
    rows = [row for row in all_rows if args.scope == "all" or row["question_type"] == args.scope]
    views = {row["id"]: extraction_view(row) for row in rows}
    source_sha = digest([dict(id=row["id"], input=views[row["id"]]) for row in rows])
    engine_options = rope_settings(args.max_model_len, args.rope_factor)
    manifest_path = out / "blind_manifest.json"
    records_path = out / "extractions.jsonl"
    if args.score_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["inputs_sha256"] != source_sha or manifest["policy"] != POLICY or \
                manifest["policy_source_sha256"] != file_hash(blind_evaluation.__file__):
            raise ValueError("Inputs or policy changed; use a new output directory")
        with RunLock(out):
            records = read_jsonl(records_path)
            validate_records(rows, records)
            return write_report(out, rows, records, args.scope)
    snapshot, state = resolve(args.model, args.revision, download=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    prompts = {key: token_prompt(tokenizer, build_prompt(view)) for key, view in views.items()}
    maximum = max(len(prompt["prompt_token_ids"]) for prompt in prompts.values())
    print(f"[preflight] n={len(rows)} max_input_tokens={maximum} required_context="
          f"{maximum+args.max_new_tokens} configured={args.max_model_len}", flush=True)
    if maximum + args.max_new_tokens > args.max_model_len:
        raise ValueError("Input exceeds context; no truncation performed")
    if args.preflight_only:
        return dict(n=len(rows), max_input_tokens=maximum, required_context=maximum+args.max_new_tokens)
    identity = dict(policy=POLICY, inputs_sha256=source_sha,
                    policy_source_sha256=file_hash(blind_evaluation.__file__), script_sha256=file_hash(__file__),
                    model_snapshot=str(snapshot), model_config_sha256=file_hash(snapshot / "config.json"),
                    max_model_len=args.max_model_len, rope_factor=args.rope_factor,
                    max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, seed=args.seed,
                    temperature=0, thinking=False, scope=args.scope)
    with RunLock(out):
        lock_manifest(out, "blind_manifest.json", identity)
        records = read_jsonl(records_path)
        seen = validate_records(rows, records)
        todo = [row for row in rows if row["id"] not in seen]
        if todo:
            from vllm import LLM, SamplingParams
            llm = LLM(model=str(snapshot), dtype="bfloat16", max_model_len=args.max_model_len,
                      gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.batch_size,
                      seed=args.seed, trust_remote_code=False, enforce_eager=True, **engine_options)
            sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=args.seed)
            for start in range(0, len(todo), args.batch_size):
                batch = todo[start:start+args.batch_size]
                outputs = llm.generate([prompts[row["id"]] for row in batch], sampling, use_tqdm=False)
                if len(outputs) != len(batch):
                    raise ValueError("Unexpected model output count")
                for row, output in zip(batch, outputs):
                    if len(output.outputs) != 1:
                        raise ValueError("Expected one extraction per response")
                    completion = output.outputs[0]
                    view = views[row["id"]]
                    extracted = parse_output(completion.text, completion.finish_reason, view)
                    record = dict(id=row["id"], input_sha256=digest(view),
                                  prompt_sha256=digest(build_prompt(view)), **extracted,
                                  raw_judge_output=completion.text, finish_reason=completion.finish_reason,
                                  output_tokens=len(completion.token_ids))
                    append_jsonl(records_path, record)
                    records.append(record)
                save_json(out / "progress.json", dict(state="running", completed=len(records), n=len(rows)))
                print(f"[blind extract] {len(records)}/{len(rows)}", flush=True)
        report = write_report(out, rows, records, args.scope)
        save_json(out / "progress.json", dict(state="complete", completed=len(rows), n=len(rows)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=("all", "open", "multiple-choice"), default="all")
    parser.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--rope-factor", type=float, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--score-only", action="store_true", help="Score saved extractions without loading the model")
    args = parser.parse_args()
    if args.max_model_len < 128 or not 1 <= args.rope_factor <= 4 or not 16 <= args.max_new_tokens <= 1024 or \
            not 1 <= args.batch_size <= 32 or not 0.1 <= args.gpu_memory_utilization <= 0.95:
        parser.error("Invalid context, token, batch, or memory setting")
    try:
        print(json.dumps(run(args), indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
