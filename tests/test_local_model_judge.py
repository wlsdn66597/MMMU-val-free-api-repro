import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_local_model_judge import (build_report, parse_letter, pending_rows,
                                               rope_settings, token_prompt, validate_saved)
from scripts.prepare_local_judge_model import complete, resolve
from mmmu_repro.common import digest
from mmmu_repro.judge import extraction_input


class LocalModelJudgeTests(unittest.TestCase):
    def test_preflight_ids_are_exact_ids_sent_to_engine(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return [1, 2, 3]

            def decode(self, *args, **kwargs):
                raise AssertionError("Decoding would cause re-tokenization")

        tokenizer = Tokenizer()
        self.assertEqual(token_prompt(tokenizer, "original prompt"), {"prompt_token_ids": [1, 2, 3]})
        self.assertEqual(tokenizer.messages[0]["content"], "original prompt")
        self.assertFalse(tokenizer.kwargs["enable_thinking"])
        self.assertEqual(rope_settings(32768, 1), {})
        self.assertEqual(rope_settings(65536, 2)["rope_scaling"]["factor"], 2)
        with self.assertRaises(ValueError):
            rope_settings(65536, 1)

    def test_model_file_check_requires_every_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                (root / name).write_text("{}", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map":{"a":"a.safetensors","b":"b.safetensors"}}', encoding="utf-8")
            (root / "a.safetensors").write_bytes(b"a")
            self.assertFalse(complete(root))
            (root / "b.safetensors").write_bytes(b"b")
            self.assertTrue(complete(root))
            path, state = resolve(str(root))
            self.assertEqual(path, root.resolve())
            self.assertEqual(state, "local-path")

    def test_only_unparsed_are_judged_and_records_are_source_locked(self):
        choices = {"A": "red", "B": "blue", "C": "green", "D": "yellow"}
        base = dict(subject="Biology", question_type="multiple-choice", question="Color?",
                    choices=choices, answer="B", finish_reason="stop")
        rows = [dict(base, id="one", raw_response="B"),
                dict(base, id="two", raw_response="Compare A and B; answer B.")]
        self.assertEqual([row["id"] for row in pending_rows(rows)], ["two"])
        row = rows[1]
        record = dict(id="two", source_sha256=digest(row),
                      prompt_sha256=digest(extraction_input(row)[2]), extracted_answer="B",
                      finish_reason="stop")
        self.assertEqual(validate_saved(rows, [record]), {"two"})
        self.assertEqual(build_report(rows, [record], [])["overall"]["local_accuracy_pct"], 100)
        with self.assertRaises(ValueError):
            validate_saved(rows, [dict(record, source_sha256="wrong")])
        with self.assertRaises(ValueError):
            validate_saved(rows, [record, record])

    def test_strict_single_letter_and_open_official_prompt(self):
        self.assertEqual(parse_letter(" B\n", {"A", "B"}), "B")
        self.assertIsNone(parse_letter("The answer is B", {"A", "B"}))
        self.assertIsNone(parse_letter("C", {"A", "B"}))
        row = dict(id="open", subject="Math", question_type="open", question="Value?",
                   choices={}, answer="42", raw_response="The result is forty-two.", finish_reason="stop")
        _, _, prompt = extraction_input(row)
        self.assertIn("A. 42", prompt)
        self.assertIn("B. Other Answers", prompt)


if __name__ == "__main__":
    unittest.main()
