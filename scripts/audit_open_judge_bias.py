"""Check whether open-answer extraction changes when the reference moves from A to B.

This reuses saved VLM responses. It never generates new VLM answers or calls an API.
"""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from mmmu_repro.common import RunLock, append_jsonl, digest, file_hash, lock_manifest, read_jsonl, save_json
from mmmu_repro.judge import checked_predictions, extract, extraction_input
from mmmu_repro.vendor.qwen_extract import build_prompt, can_infer

if __package__:
    from .evaluate_local_model_judge import parse_letter, rope_settings, token_prompt, validate_saved
    from .prepare_local_judge_model import resolve
else:
    from evaluate_local_model_judge import parse_letter, rope_settings, token_prompt, validate_saved
    from prepare_local_judge_model import resolve


def swapped_input(row):
    if row["question_type"] != "open":
        raise ValueError("Position audit accepts open questions only")
    choices = {"A": "Other Answers", "B": str(row["answer"])}
    prediction = str(row["raw_response"]).split("</think>")[-1].strip()
    options = "There are several options: \n" + "".join(f"{k}. {v}\n" for k, v in choices.items())
    return choices, prediction, build_prompt(row["question"], options, prediction)


def swapped_rule(row):
    choices, prediction, _ = swapped_input(row)
    return can_infer(prediction, deepcopy(choices)) or None


def comparison(rows, original_records, swapped_records):
    original_by_id = {record["id"]: record for record in original_records}
    swapped_by_id = {record["id"]: record for record in swapped_records}
    cases = []
    for row in rows:
        original = extract(row)
        if original["resolved"]:
            original_letter, original_method = original["extracted_answer"], "rule"
        else:
            record = original_by_id.get(row["id"])
            if record is None:
                raise ValueError(f"Original local judge is incomplete for {row['id']}")
            original_letter, original_method = record["extracted_answer"], "local_model"
        rule_letter = swapped_rule(row)
        if rule_letter is not None:
            swapped_letter, swapped_method = rule_letter, "rule"
        else:
            record = swapped_by_id.get(row["id"])
            swapped_letter, swapped_method = (record["extracted_answer"], "local_model") if record else (None, "pending")
        cases.append(dict(id=row["id"], subject=row["subject"], reference=str(row["answer"]),
                          response_characters=len(row["raw_response"]),
                          response_tail=str(row["raw_response"])[-1200:],
                          original_method=original_method, original_letter=original_letter,
                          original_correct=original_letter == "A",
                          swapped_method=swapped_method, swapped_letter=swapped_letter,
                          swapped_correct=swapped_letter == "B" if swapped_method != "pending" else None,
                          always_A=original_letter == "A" and swapped_letter == "A",
                          always_B=original_letter == "B" and swapped_letter == "B"))
    complete = all(case["swapped_method"] != "pending" for case in cases)
    counts = Counter((case["original_correct"], case["swapped_correct"]) for case in cases if case["swapped_correct"] is not None)
    return dict(n=len(cases), complete=complete,
                original_correct=sum(case["original_correct"] for case in cases),
                swapped_correct=sum(case["swapped_correct"] is True for case in cases) if complete else None,
                original_methods=dict(Counter(case["original_method"] for case in cases)),
                swapped_methods=dict(Counter(case["swapped_method"] for case in cases)),
                original_letters=dict(Counter(str(case["original_letter"]) for case in cases)),
                swapped_letters=dict(Counter(str(case["swapped_letter"]) for case in cases)),
                original_only=counts[(True, False)] if complete else None,
                swapped_only=counts[(False, True)] if complete else None,
                stable_correct=counts[(True, True)] if complete else None,
                stable_wrong=counts[(False, False)] if complete else None,
                always_A=sum(case["always_A"] for case in cases) if complete else None,
                always_B=sum(case["always_B"] for case in cases) if complete else None), cases


def check_original_settings(manifest, args, snapshot):
    expected = dict(max_model_len=args.max_model_len, max_new_tokens=args.max_new_tokens,
                    rope_factor=args.rope_factor, temperature=0, thinking=False,
                    model_config_sha256=file_hash(snapshot / "config.json"))
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Original local judge used a different {key}; match its settings")
    if manifest.get("revision") != snapshot.name:
        raise ValueError("Original local judge used a different model snapshot")


def run(args):
    engine_options = rope_settings(args.max_model_len, args.rope_factor)
    infer, original, out = (Path(path).resolve() for path in
                            (args.inference_dir, args.original_local_dir, args.output_dir))
    for source in (infer, original):
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("Audit output must be separate from source directories")
    all_rows = checked_predictions(infer)
    rows = [row for row in all_rows if row["question_type"] == "open"]
    if len(rows) != 53:
        raise ValueError("Expected all 53 MMMU-val open questions")
    original_records = read_jsonl(original / "extractions.jsonl")
    validate_saved(all_rows, original_records)
    manifest = json.loads((original / "local_model_judge_manifest.json").read_text(encoding="utf-8"))
    if manifest["predictions_sha256"] != file_hash(infer / "predictions.jsonl"):
        raise ValueError("Original local judge belongs to different VLM predictions")
    snapshot, state = resolve(args.model, args.revision, download=False)
    if state == "local-path":
        raise ValueError("Use the same Hugging Face model ID as the original local judge")
    check_original_settings(manifest, args, snapshot)
    # Make sure every rule-unparsed original open answer has a saved local result.
    comparison(rows, original_records, [])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    pending = [row for row in rows if swapped_rule(row) is None]
    prompts = {row["id"]: token_prompt(tokenizer, swapped_input(row)[2]) for row in pending}
    maximum = max((len(prompt["prompt_token_ids"]) for prompt in prompts.values()), default=0)
    print(f"[preflight] open=53 swapped_pending={len(pending)} max_input_tokens={maximum} "
          f"required_context={maximum + args.max_new_tokens} configured={args.max_model_len}", flush=True)
    if maximum + args.max_new_tokens > args.max_model_len:
        raise ValueError("Swapped judge input exceeds context; no truncation performed")
    if args.preflight_only:
        return dict(open=53, swapped_pending=len(pending), required_context=maximum + args.max_new_tokens)
    identity = dict(predictions_sha256=file_hash(infer / "predictions.jsonl"),
                    original_manifest_sha256=file_hash(original / "local_model_judge_manifest.json"),
                    original_extractions_sha256=file_hash(original / "extractions.jsonl"),
                    script_sha256=file_hash(__file__), model_snapshot=str(snapshot),
                    max_model_len=args.max_model_len, max_new_tokens=args.max_new_tokens,
                    rope_factor=args.rope_factor, temperature=0, thinking=False,
                    question_scope="all-53-open", reference_position="B")
    with RunLock(out):
        lock_manifest(out, "manifest.json", identity)
        saved_path = out / "swapped_extractions.jsonl"
        records = read_jsonl(saved_path)
        pending_by_id = {row["id"]: row for row in pending}
        seen = set()
        for record in records:
            row = pending_by_id.get(record["id"])
            if row is None or record["id"] in seen or record["source_sha256"] != digest(row) or \
                    record["prompt_sha256"] != digest(swapped_input(row)[2]):
                raise ValueError("Duplicate, unknown, or stale swapped judge result")
            if record["extracted_answer"] not in ("A", "B", "Z", None):
                raise ValueError("Invalid swapped judge output")
            seen.add(row["id"])
        todo = [row for row in pending if row["id"] not in seen]
        if todo:
            from vllm import LLM, SamplingParams
            llm = LLM(model=str(snapshot), dtype="bfloat16", max_model_len=args.max_model_len,
                      gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.batch_size,
                      trust_remote_code=False, enforce_eager=True, **engine_options)
            sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
            for start in range(0, len(todo), args.batch_size):
                batch = todo[start:start + args.batch_size]
                outputs = llm.generate([prompts[row["id"]] for row in batch], sampling, use_tqdm=False)
                for row, output in zip(batch, outputs):
                    completion = output.outputs[0]
                    letter = parse_letter(completion.text, {"A", "B"}) if completion.finish_reason == "stop" else None
                    record = dict(id=row["id"], source_sha256=digest(row),
                                  prompt_sha256=digest(swapped_input(row)[2]),
                                  extracted_answer=letter, raw_judge_output=completion.text,
                                  finish_reason=completion.finish_reason,
                                  output_tokens=len(completion.token_ids))
                    append_jsonl(saved_path, record)
                    records.append(record)
                print(f"[swapped judge] {len(records)}/{len(pending)}", flush=True)
        report, cases = comparison(rows, original_records, records)
        save_json(out / "summary.json", report)
        with (out / "cases.jsonl").open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")
        lines = ["# Open-answer judge position audit", "",
                 "Same 53 saved VLM responses; same Qwen rules and local model; reference moved from A to B.",
                 "No VLM regeneration or API calls. The judge receives the reference only during evaluation.", "",
                 f"- Original reference=A correct: {report['original_correct']}/53",
                 f"- Swapped reference=B correct: {report['swapped_correct']}/53",
                 f"- Correct in both: {report['stable_correct']}",
                 f"- Correct only with reference=A: {report['original_only']}",
                 f"- Correct only with reference=B: {report['swapped_only']}",
                 f"- Chose A in both orientations: {report['always_A']}",
                 f"- Chose B in both orientations: {report['always_B']}",
                 f"- Original letters: {report['original_letters']}",
                 f"- Swapped letters: {report['swapped_letters']}", "",
                 "Original-only correct IDs: " + ", ".join(case["id"] for case in cases
                    if case["original_correct"] and case["swapped_correct"] is False),
                 "Always-A IDs: " + ", ".join(case["id"] for case in cases if case["always_A"]), "",
                 "Inspect cases.jsonl, especially original_only and always_A. Position sensitivity is diagnostic; "
                 "it does not by itself prove which extraction is semantically correct."]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--original-local-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--rope-factor", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.max_model_len < 128 or not 1 <= args.max_new_tokens <= 128 or \
            not 1 <= args.batch_size <= 32 or not 0.1 <= args.gpu_memory_utilization <= 0.95 or \
            not 1 <= args.rope_factor <= 4:
        parser.error("Invalid context, token, batch, or memory setting")
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
