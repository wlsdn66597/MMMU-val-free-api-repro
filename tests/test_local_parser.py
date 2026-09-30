import inspect
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluate_local_parser import parse_local, score_open, analyze, main
from mmmu_repro.common import digest, append_jsonl, file_hash, save_json
from mmmu_repro.judge import extract


class LocalParserTests(unittest.TestCase):
    choices = {"A": "red", "B": "blue", "C": "green", "D": "yellow"}

    def test_explicit_formats_after_option_discussion(self):
        for tail in ("Final answer: **B**", "The correct answer is (B).", "Answer: option B",
                     "### Final Answer\n\nB", r"$\boxed{\text{B}}$", "\n(B)", "Answer: blue",
                     r"Final answer: $\boxed{B}$.", r"Final answer: \(\boxed{B}\)."):
            with self.subTest(tail=tail):
                raw = "Consider (A) and (B). Both need checking.\n" + tail
                self.assertEqual(parse_local(raw, "multiple-choice", self.choices, "stop")["answer"], "B")

    def test_refuses_ambiguous_speculative_and_incomplete_responses(self):
        for raw in ("Compare (A) with (B), and then examine (C).", "Maybe the answer is B.",
                    "Final answer: B or C", "Final answer: B / C", "Final answer: B\nFinal answer: C",
                    "Answer: B\nAnswer: Cannot determine", "Answer: A bird is shown.",
                    "> Final answer: B", "A\nB\nC", "Final answer: B. red", "Answer: invalid"):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_local(raw, "multiple-choice", self.choices, "stop")["answer"])
        self.assertIsNone(parse_local("Final answer: B", "multiple-choice", self.choices, "length")["answer"])
        self.assertIsNone(parse_local("<think>Final answer: B", "multiple-choice", self.choices, "stop")["answer"])
        self.assertEqual(parse_local("<think>A or C</think>Final answer: B", "multiple-choice", self.choices, "stop")["answer"], "B")

    def test_open_extraction_and_conservative_scoring(self):
        value = parse_local(r"Work on it. Final answer: \boxed{\frac{1}{2}}", "open", {}, "stop")["answer"]
        self.assertEqual(value, r"\frac{1}{2}")
        self.assertTrue(score_open(value, "0.5"))
        self.assertTrue(score_open("1,000", "1000"))
        self.assertFalse(score_open("5 cm", "5 m"))
        self.assertFalse(score_open("-2", "2"))
        self.assertFalse(score_open("Co", "CO"))
        self.assertIsNone(parse_local("Try 2, then 4. The graph crosses at 5.", "open", {}, "stop")["answer"])

    def test_reference_and_api_are_not_parser_inputs(self):
        self.assertEqual(list(inspect.signature(parse_local).parameters),
                         ["raw_response", "question_type", "choices", "finish_reason"])
        base = dict(id="q", subject="test", question_type="multiple-choice", question="Question?",
                    choices=self.choices, answer="B", raw_response="Discuss (A) and (B). Final answer: B", finish_reason="stop")
        self.assertFalse(extract(base)["resolved"])
        with patch("mmmu_repro.judge.JudgeClient", side_effect=AssertionError("No API call permitted")):
            one, records = analyze([base], [])
            other, changed = analyze([dict(base, answer="C")], [])
        self.assertEqual(records[0]["fallback"], changed[0]["fallback"])
        self.assertEqual(one["overall"]["additional_parser"]["recovered_correct"], 1)
        self.assertEqual(other["overall"]["additional_parser"]["recovered_wrong"], 1)
        self.assertIsNone(one["overall"]["rule_plus_api"]["accuracy_pct"])
        api_record = dict(id="q", source_sha256=digest(base), method="judge", resolved=True,
                          extracted_answer="C", correct=False)
        summary, combined = analyze([base], [api_record])
        self.assertTrue(combined[0]["api_disagreement"])
        self.assertEqual(summary["overall"]["rule_plus_api"]["accuracy_pct"], 0)
        self.assertEqual(summary["overall"]["rule_plus_local"]["accuracy_pct"], 100)
        with self.assertRaises(ValueError):
            analyze([base], [dict(api_record, source_sha256="bad")])

    def test_full_run_offline_report_and_repeat_leave_inference_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            infer, out = root / "inference", root / "local"
            infer.mkdir()
            rows = []
            for i in range(900):
                is_mc = i < 847
                rows.append(dict(id=str(i), subject=f"s{i//30}", question_type="multiple-choice" if is_mc else "open",
                                 question="What is shown?", choices=self.choices if is_mc else {},
                                 answer="B" if is_mc else "42", finish_reason="stop",
                                 raw_response="Compare (A) and (B). Final answer: B" if i < 2 else "B" if is_mc else "42"))
                append_jsonl(infer / "predictions.jsonl", rows[-1])
            source_sha = file_hash(infer / "predictions.jsonl")
            save_json(infer / "manifest.json", dict(selected_ids=[r["id"] for r in rows], diagnostic=False))
            save_json(infer / "progress.json", dict(state="complete", predictions_sha256=source_sha))
            argv = ["local-parser", "--inference-dir", str(infer), "--output-dir", str(out)]
            with patch("sys.argv", argv), redirect_stdout(io.StringIO()), \
                 patch("requests.sessions.Session.request", side_effect=AssertionError("Offline report cannot use network")):
                main()
                main()
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["overall"]["original_rule"]["unparsed"], 2)
            self.assertEqual(summary["overall"]["rule_plus_local"]["correct"], 900)
            self.assertEqual(summary["overall"]["rule_plus_api"]["pending"], 2)
            self.assertEqual(len((out / "records.jsonl").read_text().splitlines()), 900)
            self.assertEqual((out / "unresolved.jsonl").read_text(), "")
            self.assertEqual(file_hash(infer / "predictions.jsonl"), source_sha)


if __name__ == "__main__":
    unittest.main()
