"""Compare rule-only and rule-plus-API scoring of one fixed inference run.

This is deliberately outside src/mmmu_repro: reading a report during an active
judge run must not change that run's locked source hash or scoring policy.
"""

import argparse
import json
from pathlib import Path

from mmmu_repro.common import digest, read_jsonl
from mmmu_repro.judge import checked_predictions, extract


def compare(rows, records):
    by_id = {row["id"]: row for row in rows}
    seen = set()
    api = {}
    rule = {row["id"]: extract(row) for row in rows}
    for record in records:
        item_id = record["id"]
        if item_id in seen or item_id not in by_id:
            raise ValueError("Duplicate or unknown extraction ID")
        seen.add(item_id)
        if record.get("source_sha256") != digest(by_id[item_id]):
            raise ValueError(f"Extraction source mismatch: {item_id}")
        if record["method"] == "rule":
            if not rule[item_id]["resolved"] or record["extracted_answer"] != rule[item_id]["extracted_answer"]:
                raise ValueError(f"Rule extraction mismatch: {item_id}")
        elif record["method"] == "judge":
            if rule[item_id]["resolved"] or not record["resolved"] or not isinstance(record["correct"], bool):
                raise ValueError(f"Invalid API extraction: {item_id}")
            gold = by_id[item_id]["answer"] if by_id[item_id]["question_type"] == "multiple-choice" else "A"
            choices = set(by_id[item_id]["choices"]) if by_id[item_id]["question_type"] == "multiple-choice" else {"A", "B"}
            if record.get("extracted_answer") not in choices | {"Z"} or \
               record["correct"] != (record["extracted_answer"] == gold):
                raise ValueError(f"API correctness mismatch: {item_id}")
            api[item_id] = record
        else:
            raise ValueError(f"Unknown extraction method: {item_id}")

    def group(items):
        n = len(items)
        parsed = sum(rule[row["id"]]["resolved"] for row in items)
        before_correct = sum(rule[row["id"]]["correct"] is True for row in items)
        api_items = [api[row["id"]] for row in items if row["id"] in api]
        api_correct = sum(record["correct"] for record in api_items)
        remaining = n - parsed - len(api_items)
        known_correct = before_correct + api_correct
        return {
            "n": n,
            "before_api": {
                "rule_parsed": parsed,
                "unparsed_counted_wrong": n - parsed,
                "correct": before_correct,
                "accuracy_pct": round(100 * before_correct / n, 4),
            },
            "api": {
                "processed": len(api_items),
                "correct": api_correct,
                "wrong": len(api_items) - api_correct,
                "remaining": remaining,
            },
            "after_api": {
                "known_correct": known_correct,
                "accuracy_pct": round(100 * known_correct / n, 4) if remaining == 0 else None,
                "current_minimum_pct": round(100 * known_correct / n, 4),
            },
        }

    return {
        "comparison": "Same Qwen responses; API extracts an answer only when the public rule could not.",
        "denominator": "Every validation question remains in the denominator; unparsed means wrong before API.",
        "overall": group(rows),
        "multiple_choice": group([row for row in rows if row["question_type"] == "multiple-choice"]),
        "open": group([row for row in rows if row["question_type"] == "open"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--judge-dir", required=True)
    args = parser.parse_args()
    rows = checked_predictions(args.inference_dir)
    records = read_jsonl(Path(args.judge_dir) / "extractions.jsonl")
    print(json.dumps(compare(rows, records), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
