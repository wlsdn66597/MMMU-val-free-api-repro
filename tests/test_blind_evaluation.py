import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from mmmu_repro.blind_evaluation import build_prompt, extraction_view, parse_output, score_open
from scripts.evaluate_blind_local_judge import run


class BlindEvaluationTests(unittest.TestCase):
    def row(self, **values):
        row = dict(id="one", subject="Math", question_type="open", question="Value?",
                   choices={}, answer="6.5", raw_response="0.0252", finish_reason="stop")
        return dict(row, **values)

    def test_gold_and_metadata_cannot_change_prompt(self):
        row = self.row(answer="SECRET_GOLD_SENTINEL", correct=True, api_answer="SECRET_API_SENTINEL")
        view = extraction_view(row)
        prompt = build_prompt(view)
        self.assertNotIn("SECRET", prompt)
        self.assertEqual(prompt, build_prompt(extraction_view(self.row(answer="another answer"))))
        with self.assertRaises(ValueError):
            build_prompt(dict(view, answer="gold"))

    def test_verbatim_evidence_and_option_consistency(self):
        view = extraction_view(self.row())
        self.assertEqual(parse_output('{"answer":"0.0252","evidence":"0.0252"}', "stop", view)["answer"], "0.0252")
        self.assertIsNone(parse_output('{"answer":"6.5","evidence":"6.5"}', "stop", view)["answer"])
        self.assertIsNone(parse_output('{"answer":"6.5","evidence":"0.0252"}', "stop", view)["answer"])
        self.assertIsNone(parse_output('{"answer":"0.0252","evidence":"0.0252"}', "length", view)["answer"])
        mc = extraction_view(self.row(question_type="multiple-choice", choices={"A":"red", "B":"blue"}, raw_response="Final answer: B"))
        self.assertIsNone(parse_output('{"answer":"A","evidence":"B"}', "stop", mc)["answer"])
        self.assertEqual(parse_output('{"answer":"B","evidence":"B"}', "stop", mc)["answer"], "B")

    def test_scalars_aliases_rounding_and_no_substring_scoring(self):
        self.assertFalse(score_open("0.0252", "6.5")["precision_correct"])
        self.assertFalse(score_open("100/3 V", "20")["precision_correct"])
        self.assertFalse(score_open("First trimester", "embryonic")["precision_correct"])
        self.assertFalse(score_open("cis-1-chloro-2-methylcyclohexane", "trans-1-chloro-4-methylcyclohexane")["precision_correct"])
        sqrt = score_open(r"\boxed{2\sqrt{2}} \text{A}", "2.83")
        self.assertFalse(sqrt["exact_correct"])
        self.assertTrue(sqrt["precision_correct"])
        self.assertTrue(sqrt["unit_review_required"])
        self.assertTrue(score_open(r"\frac{24}{7} ft/s", "['24/7', '3.429']")["exact_correct"])
        self.assertTrue(score_open("MgS", "['$MgS$', 'MgS']")["exact_correct"])
        self.assertTrue(score_open("Region A", "A")["exact_correct"])
        self.assertFalse(score_open("10.41", "10")["precision_correct"])
        self.assertFalse(score_open("1460", "1464")["precision_correct"])
        self.assertFalse(score_open("2.83 mA", "2.83 A")["precision_correct"])
        self.assertFalse(score_open("__import__('os').system('bad')", "1")["precision_correct"])

    def test_all_responses_are_extracted_resume_and_cpu_rescore(self):
        rows = [self.row(id="mc", question_type="multiple-choice", choices={"A":"red", "B":"blue"},
                         answer="B", raw_response="Final answer: B"), self.row(id="open")]
        calls = []
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                prompt = messages[0]["content"]
                calls.append(prompt)
                return {"input_ids": [ord(char) for char in prompt]}
        class LLM:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def generate(self, prompts, sampling, **kwargs):
                values = [("B", "B"), ("0.0252", "0.0252")]
                return [SimpleNamespace(outputs=[SimpleNamespace(
                    text=json.dumps(dict(answer=a, evidence=e)), finish_reason="stop", token_ids=[1])])
                    for a, e in values]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            args = argparse.Namespace(inference_dir=str(root / "infer"), output_dir=str(root / "out"),
                scope="all", model="fake", revision="main", max_model_len=65536, rope_factor=2,
                max_new_tokens=256, batch_size=4, seed=3407, gpu_memory_utilization=.9,
                preflight_only=False, score_only=False)
            transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()))
            vllm = SimpleNamespace(LLM=LLM, SamplingParams=lambda **kwargs: kwargs)
            with patch("scripts.evaluate_blind_local_judge.checked_predictions", return_value=rows), \
                    patch("scripts.evaluate_blind_local_judge.resolve", return_value=(snapshot, "cached")), \
                    patch.dict("sys.modules", {"transformers": transformers, "vllm": vllm}):
                report = run(args)
                self.assertEqual(report["overall"]["n"], 2)
                self.assertEqual(report["overall"]["exact_correct"], 1)
                self.assertEqual(len(calls), 2)
                # No gold-based rule result was retained for either response.
                saved = [json.loads(line) for line in (root / "out" / "extractions.jsonl").read_text().splitlines()]
                self.assertEqual(saved[1]["answer"], "0.0252")
                self.assertNotIn("reference", saved[1])
                with patch.object(vllm, "LLM", side_effect=AssertionError("Must resume without model")):
                    self.assertEqual(run(args)["overall"]["exact_correct"], 1)
                args.score_only = True
                with patch("scripts.evaluate_blind_local_judge.resolve", side_effect=AssertionError("CPU rescore")):
                    self.assertEqual(run(args)["overall"]["exact_correct"], 1)


if __name__ == "__main__":
    unittest.main()
