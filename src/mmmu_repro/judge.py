from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

from .common import append_jsonl, digest, file_hash, lock_manifest, read_jsonl, save_json, source_hash
from .vendor.qwen_extract import can_infer, build_prompt


@dataclass(frozen=True)
class JudgeConfig:
    api_base: str = "https://copa.codyssey.kr/v1"
    model: str = "gpt-5.4-mini"
    key_env: str = "CODYSSEY_API_KEY"
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    temperature: float | None = None
    retries: int = 3

    def validate(self):
        parts = urlsplit(self.api_base)
        if parts.scheme != "https" or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("API base must be an HTTPS URL without embedded credentials/query/fragment")
        if (self.max_tokens is not None and self.max_tokens < 1) or not 1 <= self.retries <= 5:
            raise ValueError("Invalid judge token/retry limit")
        return self


def extraction_input(row):
    # Reference answer is used ONLY after VLM inference, never in its input.
    choices = (deepcopy(row["choices"]) if row["question_type"] == "multiple-choice"
               else {"A": str(row["answer"]), "B": "Other Answers"})
    prediction = str(row["raw_response"]).split("</think>")[-1].strip()
    options = "There are several options: \n" + "".join(f"{k}. {v}\n" for k, v in choices.items())
    prompt = build_prompt(row["question"], options, prediction)
    return choices, prediction, prompt


class JudgeClient:
    def __init__(self, config, session=None, sleep=time.sleep):
        import requests
        self.config = config.validate()
        self.key = os.environ.get(config.key_env, "").strip()
        if not self.key:
            raise ValueError(f"Set {config.key_env} in your shell; never put the key in Git or CLI arguments")
        self.session = session or requests.Session()
        self.sleep = sleep

    def payload(self, prompt):
        c = self.config
        payload = dict(model=c.model, messages=[dict(role="user", content=prompt)])
        if c.max_tokens is not None:
            payload["max_completion_tokens"] = c.max_tokens
        if c.temperature is not None:
            payload["temperature"] = c.temperature
        if c.reasoning_effort is not None:
            payload["reasoning_effort"] = c.reasoning_effort
        return payload

    def __call__(self, prompt):
        import requests
        c = self.config
        endpoint = c.api_base.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        for attempt in range(c.retries):
            try:
                response = self.session.post(endpoint, json=self.payload(prompt),
                    headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                    timeout=(15, 180), allow_redirects=False)
            except (requests.Timeout, requests.ConnectionError):
                if attempt + 1 == c.retries:
                    raise RuntimeError("Judge connection/timeout failed; no answer was assigned") from None
                self.sleep(min(2 ** attempt, 8))
                continue
            if response.status_code in (429, 500, 502, 503, 504) and attempt + 1 < c.retries:
                self.sleep(min(2 ** attempt, 8))
                continue
            if response.status_code != 200:
                # Only expose structured diagnostic labels, never a raw gateway body.
                details = []
                try:
                    error = response.json().get("error", {})
                    if isinstance(error, dict):
                        for field in ("code", "param", "type"):
                            value = error.get(field)
                            if isinstance(value, str) and value and len(value) <= 80 and \
                               all(ch.isalnum() or ch in "_.-/" for ch in value) and self.key not in value:
                                details.append(f"{field}={value}")
                except (ValueError, TypeError, AttributeError):
                    pass
                suffix = (" (" + ", ".join(details) + ")") if details else ""
                raise RuntimeError(f"Judge HTTP {response.status_code}{suffix}; check endpoint/model/key/parameter support. "
                                   "No automatic parameter changes or random answers.")
            try:
                value = response.json()
                item = value["choices"][0]
                text = item["message"]["content"]
                if item.get("finish_reason") != "stop" or not isinstance(text, str) or not text.strip():
                    raise ValueError("Incomplete judge output")
                usage = value.get("usage") or {}
                usage = {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if k in usage}
                return dict(text=text, model=value.get("model", c.model), usage=usage,
                            finish_reason=item["finish_reason"], attempts=attempt + 1)
            except (KeyError, IndexError, ValueError, TypeError):
                raise RuntimeError("Invalid/empty/truncated judge response; no answer was assigned") from None
        raise RuntimeError("Judge request exhausted")


def extract(row, client=None):
    choices, prediction, prompt = extraction_input(row)
    answer = can_infer(prediction, deepcopy(choices)) or None
    result = dict(id=row["id"], subject=row["subject"], question_type=row["question_type"],
                  source_sha256=digest(row), method="rule", judge=None)
    if answer is None:
        result.update(method="pending_judge", judge_prompt_sha256=digest(prompt))
        if client is not None:
            response = client(prompt)
            # Stricter than upstream's heuristic parser for the judge's own output.
            # Z is a legitimate unmatched answer, not a reason to keep sampling.
            answer = response["text"].strip()
            if answer not in set(choices) | {"Z"}:
                raise RuntimeError(f"Judge did not return one valid letter for item {row['id']}")
            result.update(method="judge", judge=response)
    gold = row["answer"] if row["question_type"] == "multiple-choice" else "A"
    result.update(extracted_answer=answer, resolved=answer is not None,
                  correct=(answer == gold) if answer is not None else None)
    return result


def checked_predictions(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
    rows = read_jsonl(directory / "predictions.jsonl")
    ids = manifest["selected_ids"]
    if progress.get("state") != "complete" or [r["id"] for r in rows] != ids:
        raise ValueError("Inference is incomplete or IDs/order differ from its manifest")
    if len(rows) != 900 or manifest.get("diagnostic"):
        raise ValueError("Final scoring requires the full 900-question run; diagnostic inference is not a baseline")
    from .data import validate_coverage
    validate_coverage(rows)
    if progress.get("predictions_sha256") != file_hash(directory / "predictions.jsonl"):
        raise ValueError("Predictions changed since inference completion")
    return rows


def plan(rows):
    pending = []
    for row in rows:
        result = extract(row)
        if not result["resolved"]:
            prompt = extraction_input(row)[2]
            pending.append(dict(id=row["id"], characters=len(prompt), judge_prompt=prompt))
    return dict(n=len(rows), rule_resolved=len(rows)-len(pending), pending_judge=len(pending),
                total_judge_prompt_characters=sum(r["characters"] for r in pending),
                max_judge_prompt_characters=max((r["characters"] for r in pending), default=0)), pending


def summarize(rows, records):
    lookup = {r["id"]: r for r in records}
    def group(items):
        correct = sum(lookup.get(r["id"], {}).get("correct") is True for r in items)
        resolved = sum(lookup.get(r["id"], {}).get("resolved", False) for r in items)
        return dict(n=len(items), correct=correct, resolved=resolved,
                    accuracy_pct=100*correct/len(items) if resolved == len(items) else None)
    complete = len(records) == len(rows) and all(r["resolved"] for r in records)
    subjects = {s: group([r for r in rows if r["subject"] == s]) for s in sorted({r["subject"] for r in rows})}
    usage_rows = [r["judge"]["usage"] for r in records if r.get("judge")]
    unknown_usage = sum("total_tokens" not in u for u in usage_rows)
    return dict(state="complete" if complete else "incomplete", **group(rows),
                macro_accuracy_pct=sum(g["accuracy_pct"] for g in subjects.values())/len(subjects) if complete else None,
                subjects=subjects,
                question_types={k: group([r for r in rows if r["question_type"] == k]) for k in ("multiple-choice", "open")},
                rule_extractions=sum(r["method"] == "rule" for r in records),
                api_extractions=len(usage_rows),
                api_total_tokens_reported=sum(u.get("total_tokens", 0) for u in usage_rows),
                api_usage_missing=unknown_usage,
                api_usage_note="Provider-reported successful calls only; gateway credit conversion and failed/retried calls may differ")


def evaluate(inference_dir, output_dir, config, max_api_calls=900):
    rows = checked_predictions(inference_dir)
    out = Path(output_dir)
    lock_manifest(out, "judge_manifest.json", dict(source_sha256=file_hash(Path(inference_dir)/"predictions.jsonl"),
                  config=asdict(config), code_sha256=source_hash(),
                  policy="qwen-rules-and-prompt; strict-api-letter; Z-final; no-random-fallback-v1"))
    records = read_jsonl(out / "extractions.jsonl")
    seen = set()
    source = {r["id"]: r for r in rows}
    for r in records:
        if r["id"] in seen or r["id"] not in source or r["source_sha256"] != digest(source[r["id"]]) or not r["resolved"]:
            raise ValueError("Corrupt/duplicate/stale extraction cache")
        seen.add(r["id"])
    client, calls = None, 0
    start = time.perf_counter()
    try:
        for i, row in enumerate(rows, 1):
            if row["id"] in seen:
                continue
            result = extract(row)
            if not result["resolved"]:
                if calls >= max_api_calls:
                    continue  # Still record remaining free rule extractions.
                if client is None:
                    client = JudgeClient(config)
                result = extract(row, client)
                calls += 1
            records.append(result)
            append_jsonl(out / "extractions.jsonl", result)
            report = summarize(rows, records)
            save_json(out / "summary.json", report)
            print(f"[judge] {len(records)}/{len(rows)} resolved; API this session={calls}", flush=True)
    except BaseException as e:
        report = summarize(rows, records)
        report.update(state="failed", error_type=type(e).__name__)
        save_json(out / "summary.json", report)
        raise
    report = summarize(rows, records)
    report["this_session_seconds"] = time.perf_counter()-start
    save_json(out / "summary.json", report)
    if report["state"] == "complete":
        md = ["# MMMU validation — free generation + API extraction", "",
              f"- Questions: {report['n']} (847 multiple-choice + 53 open)",
              f"- Accuracy: {report['accuracy_pct']:.4f}% ({report['correct']}/900)",
              f"- Subject macro accuracy: {report['macro_accuracy_pct']:.4f}%",
              f"- Judge model: `{config.model}`", f"- API extractions: {report['api_extractions']}",
              f"- Reported successful-call tokens: {report['api_total_tokens_reported']}", "",
              "This is a Qwen public-pipeline adaptation with a different judge model, not an exact reproduction of the paper score."]
        (out / "report.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    return report
