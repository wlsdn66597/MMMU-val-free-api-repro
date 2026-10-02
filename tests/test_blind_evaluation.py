import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from mmmu_repro.blind_evaluation import build_prompt, extraction_view, parse_output, score_open
from mmmu_repro.common import digest, read_jsonl, save_json
from scripts.evaluate_blind_local_judge import run
from scripts.revalidate_blind_extractions import run as revalidate


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

    def test_supported_answer_formatting_without_gold(self):
        def extract(answer, evidence, response, **values):
            return parse_output(json.dumps(dict(answer=answer, evidence=evidence)), "stop",
                                extraction_view(self.row(raw_response=response, **values)))
        self.assertEqual(extract("c", "Answer: c", "Answer: c")["answer"], "c")
        self.assertEqual(extract("$8", "✅ **Answer: $8**", "✅ **Answer: $8**")["answer"], "8")
        self.assertEqual(extract(r"$\boxed{1000}$", "✅ Final Answer: **$1,000**",
                                 "✅ Final Answer: **$1,000**")["status"], "extracted")
        self.assertEqual(extract("24/7", r"\boxed{\frac{24}{7}}",
                                 r"\boxed{\frac{24}{7}}")["status"], "extracted")
        self.assertEqual(extract("2√2 A", r"\boxed{2\sqrt{2}} \text{A}",
                                 r"\boxed{2\sqrt{2}} \text{A}")["status"], "extracted")
        display = "✅ Final Answer:\n\n" + r"\[\boxed{2\sqrt{2}} \text{ A}\]"
        self.assertEqual(extract("2√2 A", display, display)["status"], "extracted")
        self.assertEqual(extract("10.41 V", r"Final Answer: $$\boxed{10.41 \, \text{V}}$$",
                                 "### Final Answer:\n\n$$\n" + r"\boxed{10.41 \, \text{V}}" + "\n$$")["status"], "extracted")
        # An invented quote is still rejected even when its numeric answer occurs.
        self.assertIsNone(extract("65", "Thus, the final answer is $65", "$65")["answer"])
        self.assertIsNone(extract("1", "10", "10")["answer"])
        self.assertIsNone(extract("3", "10.3", "10.3")["answer"])
        self.assertIsNone(extract("3", "sqrt(3)", "sqrt(3)")["answer"])
        self.assertIsNone(extract("3", "0.7 * 1.27 * 0.2", "0.7 * 1.27 * 0.2")["answer"])
        self.assertIsNone(extract("3 mA", "3 A", "3 A")["answer"])
        choices = {"A": "$8", "B": "$12,000"}
        self.assertEqual(extract("B. $12,000", "Dividends = **12,000**", "Dividends = **12,000**",
                                 question_type="multiple-choice", choices=choices)["answer"], "B")
        self.assertIsNone(extract("A. $12,000", "12,000", "12,000",
                                 question_type="multiple-choice", choices=choices)["answer"])
        self.assertIsNone(extract("A", "Final Answer: **B. $12,000**", "Final Answer: **B. $12,000**",
                                 question_type="multiple-choice", choices=choices)["answer"])

    def test_concise_scoring_stays_separate_from_reasoning(self):
        self.assertTrue(score_open("The industry's price-to-earnings (P₀/E₁) ratio is 30.", "30.0")["exact_correct"])
        self.assertTrue(score_open("60 Mbps", "60")["exact_correct"])
        self.assertTrue(score_open(r"\boxed{\dfrac{19}{3} \mu\text{F}}", "6.333")["precision_correct"])
        self.assertEqual(score_open("24/7", "['24/7','3.429']")["matched_reference"], "24/7")
        self.assertFalse(score_open("The current is 20 A; but the final answer is 100/3 V", "20")["precision_correct"])
        self.assertFalse(score_open("The answer might be 20 or 30", "20")["precision_correct"])

    def test_cpu_revalidation_preserves_cache_and_rejects_stale_input(self):
        row = self.row(raw_response="✅ **Answer: $8**", answer="8")
        view = extraction_view(row)
        raw = json.dumps(dict(answer="$8", evidence="✅ **Answer: $8**"), ensure_ascii=False)
        record = dict(id=row["id"], input_sha256=digest(view), prompt_sha256=digest(build_prompt(view)),
                      answer=None, evidence="✅ **Answer: $8**", status="answer_not_verbatim",
                      raw_judge_output=raw, finish_reason="stop", output_tokens=20)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cached = root / "cached"
            cached.mkdir()
            source = cached / "extractions.jsonl"
            source.write_text(json.dumps(record, ensure_ascii=False)+"\n", encoding="utf-8")
            original = source.read_bytes()
            save_json(cached / "blind_manifest.json", dict(policy="blind-final-answer-v1", scope="all",
                      inputs_sha256=digest([dict(id=row["id"], input=view)])))
            args = argparse.Namespace(inference_dir=str(root/"infer"), cached_dir=str(cached), output_dir=str(root/"v2"))
            with patch("scripts.revalidate_blind_extractions.checked_predictions", return_value=[row]), \
                 patch("scripts.evaluate_blind_local_judge.resolve", side_effect=AssertionError("No model")):
                report = revalidate(args)
                self.assertEqual(report["overall"]["exact_correct"], 1)
                self.assertEqual(report["revalidation"]["new_model_calls"], 0)
                self.assertEqual(source.read_bytes(), original)
                self.assertNotIn("reference", read_jsonl(root/"v2"/"extractions.jsonl")[0])
                self.assertEqual(revalidate(args)["overall"]["exact_correct"], 1)
                row["raw_response"] = "Changed response"
                with self.assertRaisesRegex(ValueError, "inputs"):
                    revalidate(args)

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
