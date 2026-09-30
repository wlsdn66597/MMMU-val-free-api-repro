import unittest

from scripts.report_api_effect import compare
from mmmu_repro.common import digest


def row(item_id, kind, raw, answer):
    return dict(id=item_id, subject="test", question_type=kind, question="What is it?",
                choices={"A": "red", "B": "blue"} if kind == "multiple-choice" else {},
                raw_response=raw, answer=answer)


class ApiEffectTests(unittest.TestCase):
    def test_rule_before_api_and_partial_then_final_api(self):
        rows = [row("1", "multiple-choice", "A", "A"),
                row("2", "multiple-choice", "unclear", "B"),
                row("3", "open", "unclear", "42")]
        partial = compare(rows, [])
        self.assertEqual(partial["overall"]["before_api"]["correct"], 1)
        self.assertEqual(partial["overall"]["api"]["remaining"], 2)
        self.assertIsNone(partial["overall"]["after_api"]["accuracy_pct"])
        records = [dict(id="2", source_sha256=digest(rows[1]), method="judge",
                        resolved=True, extracted_answer="B", correct=True),
                   dict(id="3", source_sha256=digest(rows[2]), method="judge",
                        resolved=True, extracted_answer="B", correct=False)]
        final = compare(rows, records)
        self.assertEqual(final["overall"]["before_api"]["accuracy_pct"], 33.3333)
        self.assertEqual(final["overall"]["api"]["correct"], 1)
        self.assertEqual(final["overall"]["after_api"]["accuracy_pct"], 66.6667)
        self.assertEqual(final["open"]["after_api"]["accuracy_pct"], 0.0)
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            compare(rows, [dict(records[0], source_sha256="wrong")])


if __name__ == "__main__":
    unittest.main()
