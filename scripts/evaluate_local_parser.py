"""Offline, gold-blind fallback parsing. Does not call a model or edit judge data."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import re
import unicodedata

from mmmu_repro.common import RunLock, digest, file_hash, lock_manifest, read_jsonl, save_json
from mmmu_repro.judge import checked_predictions, extract

POLICY = "explicit-final-fallback-v1"
MARKER = re.compile(r"\b(?:the\s+)?(?:(?:final|correct)\s+(?:answer|choice|option)|answer|ans)"
                    r"(?:\s+is\b\s*[:：=]?|\s*[:：=])\s*", re.I)
HEADING = re.compile(r"^(?:(?:the\s+)?(?:final|correct)\s+(?:answer|choice|option)|answer)$", re.I)
UNCERTAIN = re.compile(r"\b(?:maybe|perhaps|possibly|probably|if|suppose|assume|might|could|guess)\b", re.I)
REFUSAL = re.compile(r"\b(?:cannot|can't|unable|unknown|uncertain|unsure|insufficient|undetermined)\b", re.I)


def clean(text):
    text = unicodedata.normalize("NFKC", str(text)).strip()
    text = re.sub(r"\*\*|__|`", "", text)
    return text.strip(" \t$*")


def boxes(text):
    """Balanced-brace box payloads, including nested LaTeX wrappers."""
    result = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth:
            if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
                depth += 1
            elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
                depth -= 1
            i += 1
        if depth == 0:
            result.append((match.start(), i, text[start:i - 1]))
    return result


def unwrap(text):
    text = clean(clean(text).rstrip(".。"))
    if (text.startswith(r"\(") and text.endswith(r"\)")) or (text.startswith(r"\[") and text.endswith(r"\]")):
        text = clean(text[2:-2])
    for _ in range(4):
        match = re.fullmatch(r"\\(?:text|mathrm|mathbf|boxed)\s*\{(.*)\}", text, re.S)
        if not match:
            break
        text = clean(clean(match.group(1)).rstrip(".。"))
    return text


def open_key(text):
    """Surface normalization and exact scalar numeric equality; no unit conversion."""
    text = unwrap(text).strip().rstrip(".").strip()
    latex_fraction = re.fullmatch(r"\\(?:dfrac|frac)\s*\{([+-]?\d+)\}\s*\{([+-]?\d+)\}", text)
    if latex_fraction:
        text = "/".join(latex_fraction.groups())
    numeric = text.replace(",", "") if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text) else text
    if re.fullmatch(r"[+-]?(?:\d+/\d+|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", numeric):
        try:
            return ("number", str(Fraction(numeric)))
        except (ValueError, ZeroDivisionError):
            pass
    return ("text", re.sub(r"\s+", " ", text))


def score_open(prediction, reference):
    # Called only AFTER extraction; the extractor never receives the reference.
    refs = reference if isinstance(reference, list) else [reference]
    return any(open_key(prediction) == open_key(ref) for ref in refs)


def mc_value(text, choices):
    text = unwrap(text)
    exact_text = [key for key, value in choices.items()
                  if clean(value).rstrip(".").casefold() == text.rstrip(".").casefold()]
    if len(exact_text) == 1:
        return exact_text[0]
    label = re.match(r"^(?:(?:option|choice)\s+)?(?:\(([A-Za-z])\)|\[([A-Za-z])\]|([A-Z])|([a-z])(?=[.)\s]*$))(.*)$", text)
    if not label:
        return None
    value = next(part for part in label.groups()[:4] if part).upper()
    rest = label.group(5).strip()
    if value not in choices:
        return None
    # A leading article ('A bird ...') and alternative lists are not answers.
    if rest and not re.match(r"^[.。,:;!\-–—(\)]|^because\b", rest, re.I):
        return None
    alternatives = re.sub(r"^[.。,:;!\-–—\s]+", "", rest)
    if re.match(r"^(?:or|and|/|either)\b", alternatives, re.I) or re.match(r"^[A-Z](?:\b|[.)])", alternatives):
        return None
    description = alternatives.strip("()[] .")
    described = [key for key, option in choices.items() if clean(option).rstrip(".").casefold() == description.casefold()]
    if described and value not in described:
        return None
    return value


def parse_local(raw_response, question_type, choices, finish_reason):
    """Inputs deliberately exclude question ID, reference answer and API result."""
    text = str(raw_response).split("</think>")[-1].strip()
    if "<think>" in text:
        return dict(answer=None, reason="unfinished_think", evidence="")
    if finish_reason == "length":
        return dict(answer=None, reason="length_limited", evidence="")
    if not text:
        return dict(answer=None, reason="empty", evidence="")
    lines = [clean(line).lstrip("# ") for line in text.splitlines() if clean(line).strip("# ")]
    candidates = []
    for i, line in enumerate(lines):
        matches = list(MARKER.finditer(line))
        if HEADING.fullmatch(line):
            tail = lines[i + 1] if i + 1 < len(lines) else ""
            candidates.append((tail, "answer_heading"))
        for marker in matches:
            before = line[:marker.start()]
            if UNCERTAIN.search(before) or re.search(r"[\"'>]", before):
                candidates.append(("", "speculative_marker"))
                continue
            tail = line[marker.end():].strip()
            if not tail and i + 1 < len(lines):
                tail = lines[i + 1]
            candidates.append((tail, "explicit_answer"))
    box_values = boxes(text)
    if box_values:
        # Only terminal boxes: a boxed intermediate computation is not a final answer.
        for start, end, value in box_values:
            tail = clean(text[end:]).strip(" .。,!;\\()[]")
            if not tail:
                candidates.append((value, "terminal_box"))
    if not candidates and lines:
        last = lines[-1]
        if question_type == "multiple-choice":
            valid = mc_value(last, choices)
            # Unmarked terminal lines must be exact letters or exact option text.
            preceding_option = len(lines) > 1 and (re.match(r"^[\[(]?[A-Z][\]).:]?(?:$|\s)", lines[-2]) or any(
                clean(value).rstrip(".").casefold() == lines[-2].rstrip(".").casefold() for value in choices.values()))
            if not preceding_option and (re.fullmatch(r"(?:\(|\[)?[A-Za-z][)\]]?[.。]?", last) or any(
                    clean(value).rstrip(".").casefold() == last.rstrip(".").casefold() for value in choices.values())):
                if valid:
                    candidates.append((last, "terminal_exact"))
        elif len(lines) == 1 and len(last) <= 120:
            # Accept only an unmarked scalar, not arbitrary explanatory prose.
            if open_key(last)[0] == "number":
                candidates.append((last, "exact_scalar"))
    if not candidates:
        return dict(answer=None, reason="no_explicit_final", evidence="")
    parsed = []
    for value, mode in candidates:
        value = unwrap(value)
        if not value or (REFUSAL.search(value) and not (
                question_type == "multiple-choice" and any(clean(v).casefold() == value.casefold() for v in choices.values()))):
            return dict(answer=None, reason="refusal_or_ambiguous_marker", evidence=value[:400])
        if UNCERTAIN.search(value) or re.search(r"\b(?:or|either)\b", value, re.I):
            return dict(answer=None, reason="speculative_or_alternative", evidence=value[:400])
        if question_type == "multiple-choice":
            answer = mc_value(value, choices)
        else:
            answer = value if len(value) <= 200 and "\n" not in value and not re.search(
                r"\b(?:because|therefore|since|explanation)\b", value, re.I) else None
        if answer is None:
            return dict(answer=None, reason="unrecognized_final", evidence=value[:400])
        parsed.append((answer, mode, value))
    keys = {open_key(a) if question_type == "open" else a for a, _, _ in parsed}
    if len(keys) != 1:
        return dict(answer=None, reason="conflicting_answers", evidence=" | ".join(v for _, _, v in parsed)[:400])
    answer, mode, evidence = parsed[-1]
    return dict(answer=answer, reason=mode, evidence=evidence[:400])


def analyze(rows, api_records):
    sources = {row["id"]: row for row in rows}
    api = {}
    seen = set()
    for saved in api_records:
        item_id = saved["id"]
        if item_id in seen or item_id not in sources or saved["source_sha256"] != digest(sources[item_id]):
            raise ValueError("Duplicate/unknown/stale API extraction")
        seen.add(item_id)
        if saved["method"] not in ("rule", "judge"):
            raise ValueError("Unknown saved extraction method")
        if saved["method"] == "judge":
            row = sources[item_id]
            gold = row["answer"] if row["question_type"] == "multiple-choice" else "A"
            choices = set(row["choices"]) if row["question_type"] == "multiple-choice" else {"A", "B"}
            if not saved["resolved"] or saved["extracted_answer"] not in choices | {"Z"} or saved["correct"] != (saved["extracted_answer"] == gold):
                raise ValueError("Invalid API extraction")
            api[item_id] = saved
    records = []
    for row in rows:
        base = extract(row)
        fallback = dict(answer=None, reason="already_rule_resolved", evidence="")
        if not base["resolved"]:
            fallback = parse_local(row["raw_response"], row["question_type"], row["choices"], row["finish_reason"])
        recovered = fallback["answer"] is not None
        correct = base["correct"] if base["resolved"] else (
            (score_open(fallback["answer"], row["answer"]) if row["question_type"] == "open"
             else fallback["answer"] == row["answer"]) if recovered else False)
        judge = api.get(row["id"])
        if base["resolved"] and judge:
            raise ValueError("API record unexpectedly overrides an original rule result")
        disagreement = None
        if recovered and judge:
            disagreement = (fallback["answer"] != judge["extracted_answer"] if row["question_type"] == "multiple-choice"
                            else correct != judge["correct"])
        records.append(dict(id=row["id"], subject=row["subject"], question_type=row["question_type"],
                            finish_reason=row["finish_reason"], reference_answer=row["answer"],
                            original_rule_answer=base["extracted_answer"], original_rule_resolved=base["resolved"],
                            original_rule_correct=base["correct"] is True,
                            fallback=fallback, recovered=recovered, local_correct=bool(correct),
                            api_answer=judge["extracted_answer"] if judge else None,
                            api_correct=judge["correct"] if judge else None,
                            api_disagreement=disagreement, raw_tail=row["raw_response"][-1600:]))

    def group(items):
        n = len(items)
        parsed = sum(r["original_rule_resolved"] for r in items)
        before_correct = sum(r["original_rule_correct"] for r in items)
        recovered = sum(r["recovered"] for r in items)
        local_correct = sum(r["local_correct"] for r in items)
        api_done = sum(r["api_correct"] is not None for r in items)
        api_correct = before_correct + sum(r["api_correct"] is True for r in items)
        rate = lambda k: round(100 * k / n, 4) if n else None
        return dict(n=n,
                    original_rule=dict(correct=before_correct, accuracy_pct=rate(before_correct), unparsed=n-parsed),
                    additional_parser=dict(recovered=recovered, recovered_correct=local_correct-before_correct,
                                           recovered_wrong=recovered-(local_correct-before_correct),
                                           reasons=dict(Counter(r["fallback"]["reason"] for r in items if not r["original_rule_resolved"]))),
                    rule_plus_local=dict(correct=local_correct, accuracy_pct=rate(local_correct), unparsed=n-parsed-recovered),
                    rule_plus_api=dict(processed=api_done, pending=n-parsed-api_done, known_correct=api_correct,
                                       accuracy_pct=rate(api_correct) if n-parsed-api_done == 0 else None),
                    local_vs_api=dict(compared=sum(r["api_disagreement"] is not None for r in items),
                                      disagreements=sum(r["api_disagreement"] is True for r in items)))

    summary = dict(parser_policy=POLICY, overall=group(records),
                   multiple_choice=group([r for r in records if r["question_type"] == "multiple-choice"]),
                   open=group([r for r in records if r["question_type"] == "open"]),
                   original_unparsed_by_finish_reason=dict(Counter(r["finish_reason"] for r in records if not r["original_rule_resolved"])),
                   notes=["No new inference or API calls. All questions stay in the accuracy denominator.",
                          "Local fallback only receives response/type/choices/finish reason; never reference labels or API results.",
                          "Open fallback uses conservative surface/numeric matching after extraction; this is not the API semantic scorer.",
                          "Agreement with API does not establish parser correctness; inspect disagreements and recovered samples."])
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--judge-dir", help="Optional: read saved API results for comparison only")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    for path in (args.inference_dir, args.judge_dir):
        if path:
            source = Path(path).resolve()
            if out == source or out in source.parents or source in out.parents:
                parser.error("Choose a separate output directory beside inference/judge")
    rows = checked_predictions(args.inference_dir)
    api_path = Path(args.judge_dir) / "extractions.jsonl" if args.judge_dir else None
    api_records = read_jsonl(api_path) if api_path else []
    summary, records = analyze(rows, api_records)
    with RunLock(out):
        lock_manifest(out, "local_parser_manifest.json", dict(parser_policy=POLICY,
            parser_sha256=file_hash(__file__), predictions_sha256=file_hash(Path(args.inference_dir)/"predictions.jsonl")))
        summary["api_snapshot_records"] = len(api_records)
        summary["api_snapshot_sha256"] = digest(api_records)
        save_json(out / "summary.json", summary)
        for filename, selected in (("records.jsonl", records),
                ("unresolved.jsonl", [r for r in records if not r["original_rule_resolved"] and not r["recovered"]]),
                ("api_disagreements.jsonl", [r for r in records if r["api_disagreement"] is True])):
            target = out / filename
            temp = target.with_suffix(".tmp")
            temp.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in selected), encoding="utf-8")
            temp.replace(target)
        lines = ["# Additional local parser — offline comparison", "",
                 "All questions remain in the denominator. Unparsed answers count as wrong.", "",
                 "| Group | N | Rule accuracy | Rule unparsed | Recovered | Rule + local accuracy | Remaining | Rule + API accuracy |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in ("overall", "multiple_choice", "open"):
            g = summary[name]
            lines.append(f"| {name} | {g['n']} | {g['original_rule']['accuracy_pct']} | {g['original_rule']['unparsed']} | "
                         f"{g['additional_parser']['recovered']} | {g['rule_plus_local']['accuracy_pct']} | "
                         f"{g['rule_plus_local']['unparsed']} | {g['rule_plus_api']['accuracy_pct']} |")
        lines.extend(["", "API accuracy is null until the required API extractions are complete.",
                      "Open fallback is scored by conservative surface/numeric matching, not API semantic matching.",
                      "API disagreements are diagnostic, not proof that either parser is right."])
        (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
