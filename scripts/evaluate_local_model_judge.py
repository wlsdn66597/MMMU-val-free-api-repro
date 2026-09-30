"""Apply a separate local Qwen3-8B judge to Qwen-rule-unparsed saved responses."""
import argparse
from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path

from mmmu_repro.common import RunLock, append_jsonl, digest, file_hash, lock_manifest, read_jsonl, save_json
from mmmu_repro.judge import checked_predictions, extract, extraction_input
if __package__:
    from .prepare_local_judge_model import resolve
else:
    from prepare_local_judge_model import resolve


def parse_letter(text, valid):
    value = text.strip()
    return value if value in valid | {"Z"} else None


def pending_rows(rows):
    return [row for row in rows if not extract(row)["resolved"]]


def token_prompt(tokenizer, prompt):
    """Pass precisely the IDs counted in preflight to vLLM, without re-tokenizing decoded text."""
    encoded = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_dict=True)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, (list, tuple)) and len(ids) == 1 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    if not isinstance(ids, (list, tuple)) or not ids or any(not isinstance(value, int) for value in ids):
        raise TypeError("Tokenizer must return nonempty integer input_ids; preflight stopped")
    return {"prompt_token_ids": list(ids)}


def rope_settings(max_model_len, rope_factor):
    if max_model_len > 32768:
        if rope_factor < max_model_len / 32768:
            raise ValueError("Context above 32768 requires explicit YaRN factor covering max-model-len")
        # vLLM 0.28 accepts RoPE extension through HF config overrides.
        return {"hf_overrides": {"rope_parameters": {
            "rope_type": "yarn", "factor": rope_factor,
            "original_max_position_embeddings": 32768, "rope_theta": 1000000}}}
    if rope_factor != 1:
        raise ValueError("YaRN factor is unnecessary for context <= 32768")
    return {}


def validate_saved(rows, records):
    pending = {row["id"]: row for row in pending_rows(rows)}
    seen = set()
    for record in records:
        key = record["id"]
        if key in seen or key not in pending or record["source_sha256"] != digest(pending[key]):
            raise ValueError("Duplicate, unknown, or stale local-judge result")
        if record["prompt_sha256"] != digest(extraction_input(pending[key])[2]):
            raise ValueError("Local judge prompt changed")
        valid = set(extraction_input(pending[key])[0]) | {"Z"}
        if record["extracted_answer"] is not None and record["extracted_answer"] not in valid:
            raise ValueError("Invalid saved local-judge answer")
        seen.add(key)
    return seen


def build_report(rows, records, api_records):
    saved = {record["id"]: record for record in records}
    api = {record["id"]: record for record in api_records if record.get("method") == "judge"}
    if len(api) != sum(record.get("method") == "judge" for record in api_records):
        raise ValueError("Duplicate API extraction")
    source = {row["id"]: row for row in rows}
    for key, record in api.items():
        if key not in source or record["source_sha256"] != digest(source[key]):
            raise ValueError("Stale or unknown saved API extraction")

    def group(items):
        rule_correct = local_correct = api_correct = local_done = api_done = 0
        invalid = Counter()
        disagree = compared = 0
        for row in items:
            original = extract(row)
            if original["resolved"]:
                rule_correct += original["correct"] is True
                local_correct += original["correct"] is True
                api_correct += original["correct"] is True
                continue
            local = saved.get(row["id"])
            judged = api.get(row["id"])
            if local:
                local_done += 1
                valid = local["extracted_answer"]
                if valid is None:
                    invalid[local["finish_reason"]] += 1
                gold = row["answer"] if row["question_type"] == "multiple-choice" else "A"
                local_correct += valid == gold
            if judged:
                api_done += 1
                api_correct += judged["correct"] is True
            if local and judged:
                compared += 1
                disagree += local["extracted_answer"] != judged["extracted_answer"]
        n = len(items)
        pending = sum(not extract(row)["resolved"] for row in items)
        rate = lambda value: round(100 * value / n, 4) if n else None
        return dict(n=n, original_rule_correct=rule_correct,
                    original_rule_accuracy_pct=rate(rule_correct), pending_local=pending-local_done,
                    local_correct=local_correct, local_accuracy_pct=rate(local_correct) if local_done == pending else None,
                    invalid_local=dict(invalid), api_correct=api_correct,
                    api_accuracy_pct=rate(api_correct) if api_done == pending else None,
                    local_api_compared=compared, local_api_disagreements=disagree)

    report = {"overall": group(rows)}
    for kind in ("multiple-choice", "open"):
        report[kind] = group([row for row in rows if row["question_type"] == kind])
    report["local_calls"] = len(records)
    report["api_snapshot_records"] = len(api_records)
    report["notes"] = [
        "Only responses unresolved by the pinned Qwen rules are passed to the local judge.",
        "The official Qwen prompt is used. Open questions include the reference as choice A at evaluation time.",
        "A local judge extracts/matches a saved answer; it does not generate a new VLM answer.",
        "Agreement with the API is diagnostic, not proof of extraction correctness.",
    ]
    return report


def run(args):
    engine_options = rope_settings(args.max_model_len, args.rope_factor)
    infer = Path(args.inference_dir).resolve()
    out = Path(args.output_dir).resolve()
    if out == infer or out in infer.parents or infer in out.parents:
        raise ValueError("Output must be separate from inference")
    if args.judge_dir:
        judge_dir = Path(args.judge_dir).resolve()
        if out == judge_dir or out in judge_dir.parents or judge_dir in out.parents:
            raise ValueError("Output must be separate from API judge")
    rows = checked_predictions(infer)
    pending = pending_rows(rows)
    snapshot, state = resolve(args.model, args.revision, download=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    prompts = {}
    lengths = []
    for row in pending:
        prompt = extraction_input(row)[2]
        tokens_prompt = token_prompt(tokenizer, prompt)
        lengths.append(len(tokens_prompt["prompt_token_ids"]))
        prompts[row["id"]] = tokens_prompt
    maximum = max(lengths, default=0)
    print(f"[preflight] pending={len(pending)} max_input_tokens={maximum} "
          f"required_context={maximum+args.max_new_tokens} configured={args.max_model_len}", flush=True)
    if maximum + args.max_new_tokens > args.max_model_len:
        raise ValueError("Judge input exceeds context; no truncation performed. Use a model/context configuration that fits.")
    if args.preflight_only:
        return dict(pending=len(pending), max_input_tokens=maximum,
                    required_context=maximum+args.max_new_tokens, model_snapshot=str(snapshot))

    identity = dict(model=args.model, revision=snapshot.name if state != "local-path" else str(snapshot),
                    model_config_sha256=file_hash(snapshot / "config.json"),
                    predictions_sha256=file_hash(infer / "predictions.jsonl"),
                    script_sha256=file_hash(__file__), max_model_len=args.max_model_len,
                    max_new_tokens=args.max_new_tokens, temperature=0, thinking=False,
                    rope_factor=args.rope_factor,
                    prompt_policy="official-qwen-option-match-v1")
    api_records = read_jsonl(Path(args.judge_dir) / "extractions.jsonl") if args.judge_dir else []
    with RunLock(out):
        lock_manifest(out, "local_model_judge_manifest.json", identity)
        saved_path = out / "extractions.jsonl"
        records = read_jsonl(saved_path)
        seen = validate_saved(rows, records)
        todo = [row for row in pending if row["id"] not in seen]
        if todo:
            from vllm import LLM, SamplingParams
            llm = LLM(model=str(snapshot), dtype="bfloat16", max_model_len=args.max_model_len,
                      gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.batch_size,
                      trust_remote_code=False, enforce_eager=True, **engine_options)
            sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
            for start in range(0, len(todo), args.batch_size):
                batch = todo[start:start+args.batch_size]
                outputs = llm.generate([prompts[row["id"]] for row in batch], sampling, use_tqdm=False)
                for row, output in zip(batch, outputs):
                    completion = output.outputs[0]
                    choices = set(extraction_input(row)[0])
                    answer = parse_letter(completion.text, choices) if completion.finish_reason == "stop" else None
                    record = dict(id=row["id"], source_sha256=digest(row),
                                  prompt_sha256=digest(extraction_input(row)[2]),
                                  extracted_answer=answer, raw_judge_output=completion.text,
                                  finish_reason=completion.finish_reason,
                                  output_tokens=len(completion.token_ids))
                    append_jsonl(saved_path, record)
                    records.append(record)
                print(f"[local judge] {len(records)}/{len(pending)}", flush=True)
        report = build_report(rows, records, api_records)
        report["model_snapshot"] = str(snapshot)
        report["max_input_tokens"] = maximum
        save_json(out / "summary.json", report)
        lines = ["# Local model answer extraction", "",
                 "Saved VLM answers; Qwen rules first; Qwen3-8B only for unresolved cases.", "",
                 "| Group | N | Rule accuracy | Rule + local model | Rule + API | Pending local | API disagreements |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for key in ("overall", "multiple-choice", "open"):
            value = report[key]
            lines.append(f"| {key} | {value['n']} | {value['original_rule_accuracy_pct']} | "
                         f"{value['local_accuracy_pct']} | {value['api_accuracy_pct']} | "
                         f"{value['pending_local']} | {value['local_api_disagreements']} |")
        lines.extend(["", "Open-question prompts include the reference as choice A at evaluation time.",
                      "API disagreements require manual inspection; either judge may be wrong."])
        (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--judge-dir", help="Read cached API results for comparison; no API call")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--rope-factor", type=float, default=1.0,
                        help="Explicit Qwen3 YaRN factor when max-model-len exceeds native 32768")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.max_model_len < 128 or not 1 <= args.max_new_tokens <= 128 or not 1 <= args.batch_size <= 32 or not 0.1 <= args.gpu_memory_utilization <= 0.95 or not 1 <= args.rope_factor <= 4:
        parser.error("Invalid context, token, batch, or memory setting")
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
