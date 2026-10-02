import unittest

from scripts.audit_open_judge_bias import comparison, swapped_input, swapped_rule
from mmmu_repro.judge import extraction_input


class OpenJudgeBiasTests(unittest.TestCase):
    def test_swapped_prompt_changes_only_reference_position(self):
        row = dict(id="open", subject="Math", question_type="open", question="What value?",
                   choices={}, answer="42", raw_response="The value is forty-two.")
        original = extraction_input(row)[2]
        choices, prediction, swapped = swapped_input(row)
        self.assertEqual(choices, {"A": "Other Answers", "B": "42"})
        self.assertEqual(prediction, "The value is forty-two.")
        self.assertIn("A. 42\nB. Other Answers", original)
        self.assertIn("A. Other Answers\nB. 42", swapped)
        self.assertIsNone(swapped_rule(row))

    def test_paired_report_detects_always_a(self):
        base = dict(subject="Math", question_type="open", question="What value?", choices={}, answer="42")
        exact = dict(base, id="exact", raw_response="42")
        ambiguous = dict(base, id="ambiguous", raw_response="The value is forty-two.")
        self.assertEqual(swapped_rule(exact), "B")
        report, cases = comparison([exact, ambiguous],
                                   [dict(id="ambiguous", extracted_answer="A")],
                                   [dict(id="ambiguous", extracted_answer="A")])
        self.assertTrue(report["complete"])
        self.assertEqual(report["original_correct"], 2)
        self.assertEqual(report["swapped_correct"], 1)
        self.assertEqual(report["stable_correct"], 1)
        self.assertEqual(report["original_only"], 1)
        self.assertEqual(report["always_A"], 1)
        self.assertEqual(cases[1]["swapped_method"], "local_model")


if __name__ == "__main__":
    unittest.main()
