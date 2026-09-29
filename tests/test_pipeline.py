import base64
from contextlib import redirect_stdout
import csv
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mmmu_repro.common import append_jsonl, file_hash, lock_manifest, save_json
from mmmu_repro.config import Config, check_budget
from mmmu_repro.data import load_rows, text_prompt, validate_coverage, messages_for, download_data
from mmmu_repro.judge import JudgeClient, JudgeConfig, extract, extraction_input, evaluate, plan
from mmmu_repro.cli import parser


def sample(i=0, kind="multiple-choice"):
    return dict(id=str(i), subject=f"Subject_{i//30}", question_type=kind, question="What is shown?",
                choices={"A": "red", "B": "blue"} if kind == "multiple-choice" else {},
                answer="A" if kind == "multiple-choice" else "42",
                raw_response="A" if kind == "multiple-choice" else "42",
                inference_seconds=0.1, finish_reason="stop", output_tokens=2)


def full_rows():
    return [sample(i, "multiple-choice" if i < 847 else "open") for i in range(900)]


class PromptAndDataTests(unittest.TestCase):
    def test_mc_prompt_and_hint_exact_whitespace(self):
        r = sample()
        r["hint"] = "Look carefully."
        self.assertEqual(text_prompt(r), "Hint: Look carefully.\nQuestion: What is shown?\nOptions:\nA. red\nB. blue\nPlease select the correct answer from the options above.")

    def test_open_prompt_does_not_expose_gold(self):
        r = sample(kind="open")
        self.assertEqual(text_prompt(r), "Question: What is shown?")
        self.assertNotIn("42", text_prompt(r))
        choices, _, prompt = extraction_input(r)
        self.assertEqual(choices, {"A": "42", "B": "Other Answers"})
        self.assertIn("A. 42", prompt)

    def test_coverage_rejects_missing_duplicate_or_unbalanced(self):
        rows = full_rows()
        self.assertEqual(validate_coverage(rows)["n"], 900)
        for bad in (rows[:-1], rows[:-1]+[rows[0]]):
            with self.assertRaises(ValueError):
                validate_coverage(bad)
        rows[0]["subject"] = "extra"
        with self.assertRaises(ValueError):
            validate_coverage(rows)

    def test_tsv_filters_dev_and_resolves_image_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"data.tsv"
            fields = ["index", "split", "question", "answer", "category", "A", "B", "image"]
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
                writer.writeheader()
                writer.writerow(dict(index="dev_image", split="dev", question="dev", answer="A", category="dev", A="red", B="blue", image="x"*100))
                for r in full_rows():
                    writer.writerow(dict(index=r["id"], split="validation", question=r["question"], answer=r["answer"],
                                         category=r["subject"], A=r["choices"].get("A", ""), B=r["choices"].get("B", ""), image="dev_image"))
            with patch("mmmu_repro.data.KNOWN_MD5", {file_hash(path,"md5"): "test-only fixture"}):
                rows, info = load_rows(path)
            self.assertEqual(len(rows), 900)
            self.assertEqual(info["source_rows"], 901)
            self.assertEqual(rows[0]["images"], ["x"*100])
            self.assertEqual(rows[-1]["question_type"], "open")
            with self.assertRaisesRegex(ValueError, "Unrecognized"):
                load_rows(path)

    def test_images_are_prefix_and_cache_is_rebuilt(self):
        from PIL import Image
        blob = io.BytesIO()
        Image.new("RGB", (3, 4), "red").save(blob, format="PNG")
        row = dict(sample(), images=[base64.b64encode(blob.getvalue()).decode()])
        with tempfile.TemporaryDirectory() as tmp:
            messages, hashes = messages_for(row, tmp, Config())
            self.assertEqual([c["type"] for c in messages[0]["content"]], ["image", "text"])
            target = Path(messages[0]["content"][0]["image"])
            target.write_bytes(b"corrupt")
            _, again = messages_for(row, tmp, Config())
            self.assertEqual(hashes, again)

    def test_http_fallback_rejects_wrong_sha_even_if_md5_is_known(self):
        import requests
        response=Mock()
        response.__enter__=Mock(return_value=response)
        response.__exit__=Mock(return_value=False)
        response.iter_content.return_value=[b"unexpected content"]
        with tempfile.TemporaryDirectory() as tmp, \
             patch("requests.get",side_effect=[requests.exceptions.SSLError("expired"),response]), \
             patch("mmmu_repro.data.file_hash",side_effect=lambda p,algorithm="sha256": "521afc0f3bf341e6654327792781644d" if algorithm=="md5" else "wrong"), \
             redirect_stdout(io.StringIO()):
            path=Path(tmp)/"MMMU_DEV_VAL.tsv"
            with self.assertRaisesRegex(ValueError,"checksum"):
                download_data(path)
            self.assertFalse(path.exists())


class ContextTests(unittest.TestCase):
    def test_exact_boundary_and_overflow(self):
        c = replace(Config(), max_model_len=40000)
        self.assertEqual(check_budget(7232,c), 40000)
        with self.assertRaisesRegex(ValueError,"Context too small"):
            check_budget(7233,c)
        with self.assertRaises(ValueError):
            replace(c,max_model_len=16384).validate()

    def test_sampling_seed_profile_and_cli(self):
        self.assertEqual(Config().sampling()["seed"],3407)
        self.assertEqual(Config().sampling()["temperature"],0.7)
        self.assertEqual(replace(Config(),profile="paper-text-4b-experimental").sampling()["temperature"],1.0)
        args = parser().parse_args(["infer","--max-model-len","65536","--seed","3407"])
        self.assertEqual(args.max_model_len,65536)

    def test_manifest_rejects_config_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_manifest(tmp,"manifest.json", {"seed":3407})
            lock_manifest(tmp,"manifest.json", {"seed":3407})
            with self.assertRaises(ValueError):
                lock_manifest(tmp,"manifest.json", {"seed":42})


class InferenceTests(unittest.TestCase):
    def test_generation_resume_uses_saved_prefix(self):
        from mmmu_repro.inference import run_inference
        rows = [sample(0),sample(1)]
        infos = [dict(id=r["id"],input_tokens=10) for r in rows]
        preflight = (rows,object(),dict(items=infos,maximum_input_tokens=10))
        completion = SimpleNamespace(text="A",token_ids=[1],finish_reason="stop")
        engine = Mock()
        engine.generate.return_value = [SimpleNamespace(prompt_token_ids=list(range(10)),outputs=[completion])]
        backend = SimpleNamespace(LLM=Mock(return_value=engine),SamplingParams=lambda **kw:kw)
        sampler = Mock()
        sampler.finish.return_value = {}
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules",{"vllm":backend}), \
             patch("mmmu_repro.inference.preflight",return_value=preflight), \
             patch("mmmu_repro.inference.GPUSampler",return_value=sampler), \
             patch("mmmu_repro.inference.prepared_input",side_effect=lambda r,*a: ({},infos[int(r["id"])])), \
             redirect_stdout(io.StringIO()):
            first=run_inference(Config(),"unused",tmp,limit=2)
            again=run_inference(Config(),"unused",tmp,limit=2)
            self.assertEqual(first["state"],"complete")
            self.assertEqual(again["n"],2)
            self.assertEqual(engine.generate.call_count,2)
            backend.LLM.assert_called_once()
            self.assertEqual(engine.generate.call_args.kwargs["sampling_params"]["seed"],3407)

    def test_rejected_config_preserves_complete_status(self):
        from mmmu_repro.inference import run_inference
        sampler=Mock()
        sampler.finish.return_value={}
        with tempfile.TemporaryDirectory() as tmp, \
             patch("mmmu_repro.inference.GPUSampler",return_value=sampler), \
             patch("mmmu_repro.inference.preflight",side_effect=ValueError("Settings changed")):
            save_json(Path(tmp)/"progress.json",{"state":"complete","completed":900})
            with self.assertRaises(ValueError):
                run_inference(Config(),"unused",tmp)
            self.assertEqual(json.loads((Path(tmp)/"progress.json").read_text())["state"],"complete")


class JudgeTests(unittest.TestCase):
    def test_rules_skip_api_and_missing_is_pending(self):
        client = Mock(side_effect=AssertionError("must not call"))
        self.assertTrue(extract(sample(),client)["correct"])
        self.assertTrue(extract(sample(kind="open"),client)["correct"])
        self.assertIsNone(extract(dict(sample(),raw_response="unclear"))["correct"])

    def test_z_final_and_no_semantic_retry(self):
        client = Mock(return_value=dict(text="Z",model="mock",usage={}))
        row = dict(sample(),raw_response="unclear")
        self.assertFalse(extract(row,client)["correct"])
        client.assert_called_once()
        with self.assertRaises(RuntimeError):
            extract(row,lambda p: dict(text="A or B"))

    def test_transport_exact_payload_and_no_redirect(self):
        response = Mock(status_code=200)
        response.json.return_value = dict(model="returned-version", choices=[dict(finish_reason="stop",message=dict(content="A"))],
                                          usage=dict(prompt_tokens=10,completion_tokens=1,total_tokens=11))
        session = Mock()
        session.post.return_value = response
        with patch.dict(os.environ,{"CODYSSEY_API_KEY":"unit-test-placeholder"}):
            client = JudgeClient(JudgeConfig(),session=session)
            result = client("prompt")
        kwargs = session.post.call_args.kwargs
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["json"],{"model":"gpt-5.4-mini","messages":[{"role":"user","content":"prompt"}]})
        with patch.dict(os.environ,{"CODYSSEY_API_KEY":"unit-test-placeholder"}):
            configured = JudgeClient(replace(JudgeConfig(),max_tokens=4096,reasoning_effort="none",temperature=0.0),session=session).payload("prompt")
        self.assertEqual(configured["max_completion_tokens"],4096)
        self.assertEqual(configured["reasoning_effort"],"none")
        self.assertEqual(configured["temperature"],0.0)
        self.assertEqual(result["usage"]["total_tokens"],11)
        self.assertEqual(session.post.call_args.args[0],"https://copa.codyssey.kr/v1/chat/completions")

    def test_401_never_retried_or_dumped(self):
        session = Mock()
        session.post.return_value = Mock(status_code=401)
        with patch.dict(os.environ,{"CODYSSEY_API_KEY":"unit-test-placeholder"}):
            client = JudgeClient(JudgeConfig(),session=session)
            with self.assertRaisesRegex(RuntimeError,"HTTP 401") as error:
                client("prompt")
        session.post.assert_called_once()
        self.assertNotIn("unit-test-placeholder",str(error.exception))

    def test_400_reports_safe_gateway_code_without_secret(self):
        response = Mock(status_code=400)
        response.json.return_value = {"error":{"code":"unsupported_parameter","param":"reasoning_effort",
                                                "message":"unit-test-placeholder"}}
        session = Mock()
        session.post.return_value = response
        with patch.dict(os.environ,{"CODYSSEY_API_KEY":"unit-test-placeholder"}):
            with self.assertRaisesRegex(RuntimeError,"param=reasoning_effort") as error:
                JudgeClient(JudgeConfig(),session=session)("prompt")
        self.assertNotIn("unit-test-placeholder",str(error.exception))

    def test_length_response_is_not_scored(self):
        response = Mock(status_code=200)
        response.json.return_value = dict(choices=[dict(finish_reason="length",message=dict(content="A"))])
        session = Mock()
        session.post.return_value = response
        with patch.dict(os.environ,{"CODYSSEY_API_KEY":"unit-test-placeholder"}):
            with self.assertRaisesRegex(RuntimeError,"truncated"):
                JudgeClient(JudgeConfig(),session=session)("prompt")

    def test_bad_endpoint_rejected(self):
        for base in ("http://example.com", "https://user:secret@example.com", "https://example.com?key=secret"):
            with self.assertRaises(ValueError):
                JudgeConfig(api_base=base).validate()

    def test_900_item_pilot_resume_and_no_duplicate_api(self):
        rows=full_rows()
        rows[0]["raw_response"]="unclear"
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            infer=root/"inference"
            infer.mkdir()
            for row in rows:
                append_jsonl(infer/"predictions.jsonl",row)
            save_json(infer/"manifest.json",dict(selected_ids=[r["id"] for r in rows],diagnostic=False))
            save_json(infer/"progress.json",dict(state="complete",predictions_sha256=file_hash(infer/"predictions.jsonl")))
            summary,pending=plan(rows)
            self.assertEqual(summary["pending_judge"],1)
            self.assertEqual(pending[0]["id"],"0")
            out=root/"judge"
            with redirect_stdout(io.StringIO()):
                result=evaluate(infer,out,JudgeConfig(),max_api_calls=0)
            self.assertEqual(result["state"],"incomplete")
            self.assertIsNone(result["accuracy_pct"])
            client=Mock(return_value=dict(text="A",model="mock",usage=dict(total_tokens=12)))
            with patch("mmmu_repro.judge.JudgeClient",return_value=client),redirect_stdout(io.StringIO()):
                result=evaluate(infer,out,JudgeConfig())
                again=evaluate(infer,out,JudgeConfig())
            client.assert_called_once()
            self.assertEqual(result["accuracy_pct"],100)
            self.assertEqual(again["api_extractions"],1)
            self.assertEqual(result["api_total_tokens_reported"],12)
            self.assertTrue((out/"report.md").exists())
            with self.assertRaises(ValueError):
                evaluate(infer,out,replace(JudgeConfig(),model="other"))


if __name__ == "__main__":
    unittest.main()
