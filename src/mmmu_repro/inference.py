from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time

from .common import append_jsonl, digest, file_hash, lock_manifest, read_jsonl, save_json, source_hash
from .config import check_budget
from .data import download_data, load_rows, messages_for, text_prompt


def model_identity(config):
    path = Path(config.model_path).expanduser()
    if not path.exists():
        return dict(repo=config.model_path, revision=config.model_revision)
    if not (path / "config.json").exists():
        raise ValueError("Local model path must contain config.json")
    # Hash all weights too: a local path must not silently switch checkpoints on resume.
    files = sorted(p for p in path.rglob("*") if p.is_file() and
                   (p.suffix in (".json", ".safetensors", ".bin", ".model", ".jinja") or p.name == "merges.txt"))
    print("Hashing local checkpoint files for reproducibility...", flush=True)
    return dict(declared_revision=config.model_revision,
                files={str(p.relative_to(path)): file_hash(p) for p in files})


def environment():
    versions = {}
    for name in ("torch", "vllm", "transformers", "qwen-vl-utils", "Pillow", "numpy", "requests"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(python=platform.python_version(), platform=platform.platform(), packages=versions)


def prepared_input(row, processor, cache, config):
    from qwen_vl_utils import process_vision_info
    messages, image_hashes = messages_for(row, cache, config)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True, return_video_metadata=True)
    if videos is not None:
        raise ValueError("MMMU image evaluation must not produce video inputs")
    kwargs = kwargs or {}
    encoded = processor(text=[text], images=images, return_tensors="pt", **kwargs)
    tokens = int(encoded["input_ids"].shape[-1])
    check_budget(tokens, config)
    item = dict(prompt=text, multi_modal_data={"image": images}, mm_processor_kwargs=kwargs)
    info = dict(id=row["id"], input_tokens=tokens, image_sha256=image_hashes,
                prompt_sha256=digest(text), question_type=row["question_type"], subject=row["subject"])
    return item, info


def preflight(config, data_root, output_dir, limit=0, download=True):
    from transformers import AutoProcessor
    config.validate()
    out = Path(output_dir)
    source = Path(data_root).expanduser()
    path = source if source.suffix.lower() == ".tsv" else source / "MMMU_DEV_VAL.tsv"
    if not path.exists():
        if not download:
            raise ValueError("Data TSV not found")
        download_data(path)
    rows, dataset = load_rows(path)
    selected = rows[:limit] if limit else rows
    identity = dict(config=config.as_dict(), dataset=dataset,
                    selected_ids=[r["id"] for r in selected], diagnostic=bool(limit),
                    model=model_identity(config), code_sha256=source_hash())
    lock_manifest(out, "manifest.json", identity)
    runtime = environment()
    # Strict resume: changing numerical libraries can change sampling / image processing.
    lock_manifest(out, "environment.json", runtime)
    freeze = subprocess.run([os.sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if freeze.returncode == 0:
        (out / "requirements.freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    processor = AutoProcessor.from_pretrained(config.model_path, revision=config.model_revision)
    manifest_items = []
    maximum = 0
    for i, row in enumerate(selected, 1):
        _, info = prepared_input(row, processor, out / "images", config)
        manifest_items.append(info)
        maximum = max(maximum, info["input_tokens"])
        if i == 1 or i % 25 == 0 or i == len(selected):
            print(f"[preflight] {i}/{len(selected)} checked; max input={maximum}", flush=True)
    result = dict(n=len(selected), maximum_input_tokens=maximum,
                  required_context=maximum+config.max_tokens, max_model_len=config.max_model_len,
                  full_validation=not bool(limit), items=manifest_items)
    existing = out / "preflight.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8")) != result:
        raise ValueError("Input processing changed since previous preflight")
    save_json(existing, result)
    return selected, processor, result


class GPUSampler:
    def __init__(self):
        self.stop_event = threading.Event()
        self.peaks = {}
        self.samples = 0

    def sample(self):
        while not self.stop_event.is_set():
            try:
                proc = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=3)
                if proc.returncode == 0:
                    for line in proc.stdout.splitlines():
                        index, memory = [x.strip() for x in line.split(",")]
                        self.peaks[index] = max(self.peaks.get(index, 0), int(memory))
                    self.samples += 1
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            self.stop_event.wait(1)

    def start(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=4)
        return dict(peak_whole_device_used_mib=self.peaks, samples=self.samples,
                    scope="all devices reported by nvidia-smi; includes other processes; 1s sampling")


def run_inference(config, data_root, output_dir, limit=0):
    started = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    previous_progress = out / "progress.json"
    identity_accepted = False
    gpu = GPUSampler()
    gpu.start()
    try:
        rows, processor, checked = preflight(config, data_root, out, limit)
        identity_accepted = True
        info_by_id = {r["id"]: r for r in checked["items"]}
        predictions = out / "predictions.jsonl"
        saved = read_jsonl(predictions)
        expected = [r["id"] for r in rows]
        if [r["id"] for r in saved] != expected[:len(saved)]:
            raise ValueError("Saved predictions are not an exact prefix of selected IDs")
        for row, record in zip(rows, saved):
            expected_source = dict(id=row["id"], question=row["question"], answer=row["answer"],
                                   choices=row["choices"], subject=row["subject"], question_type=row["question_type"])
            if any(record.get(k) != v for k, v in expected_source.items()) or record["input"] != info_by_id[row["id"]]:
                raise ValueError("Resume prediction source/input mismatch")
        if len(saved) < len(rows):
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            from vllm import LLM, SamplingParams
            save_json(out / "progress.json", dict(state="loading_model", completed=len(saved), n=len(rows)))
            load_start = time.perf_counter()
            llm = LLM(model=config.model_path, revision=config.model_revision,
                      tokenizer_revision=config.model_revision, seed=config.seed, dtype=config.dtype,
                      max_model_len=config.max_model_len, gpu_memory_utilization=config.gpu_memory_utilization,
                      tensor_parallel_size=config.tensor_parallel_size, max_num_seqs=config.max_num_seqs,
                      max_num_batched_tokens=config.max_num_batched_tokens,
                      limit_mm_per_prompt={"image": config.max_images}, generation_config="vllm",
                      enable_chunked_prefill=True, enforce_eager=True)
            model_load_seconds = time.perf_counter()-load_start
            for row in rows[len(saved):]:
                request, info = prepared_input(row, processor, out / "images", config)
                if info != info_by_id[row["id"]]:
                    raise ValueError("Input changed after preflight")
                tick = time.perf_counter()
                outputs = llm.generate([request], sampling_params=SamplingParams(**config.sampling()), use_tqdm=False)
                elapsed = time.perf_counter()-tick
                if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                    raise ValueError("Expected exactly one VLM response per question")
                result = outputs[0]
                answer = result.outputs[0]
                if result.prompt_token_ids is None:
                    raise ValueError("vLLM did not expose actual prompt tokens")
                actual = len(result.prompt_token_ids)
                check_budget(actual, config)
                if actual != info["input_tokens"]:
                    raise ValueError(f"HF/vLLM input token mismatch for {row['id']}: {info['input_tokens']} vs {actual}")
                if answer.finish_reason not in ("stop", "length"):
                    raise ValueError("Unexpected VLM finish reason")
                if answer.finish_reason == "length" and len(answer.token_ids) != config.max_tokens:
                    raise ValueError("Output was cut below the requested token cap")
                record = dict(id=row["id"], subject=row["subject"], question_type=row["question_type"],
                              question=row["question"], choices=row["choices"], answer=row["answer"],
                              prompt_text=text_prompt(row), input=info, raw_response=answer.text,
                              finish_reason=answer.finish_reason, output_tokens=len(answer.token_ids),
                              inference_seconds=elapsed)
                append_jsonl(predictions, record)
                saved.append(record)
                save_json(out / "progress.json", dict(state="running", completed=len(saved), n=len(rows), last_id=row["id"]))
                print(f"[infer] {len(saved)}/{len(rows)} tokens={record['output_tokens']} "
                      f"finish={answer.finish_reason} seconds={elapsed:.2f}", flush=True)
        else:
            model_load_seconds = 0.0
        summary = dict(state="complete", n=len(rows), full_validation=not bool(limit),
                       inference_seconds=sum(r["inference_seconds"] for r in saved),
                       length_limited=sum(r["finish_reason"] == "length" for r in saved),
                       mean_output_tokens=sum(r["output_tokens"] for r in saved)/len(saved),
                       max_input_tokens=checked["maximum_input_tokens"],
                       this_session_model_load_seconds=model_load_seconds,
                       predictions_sha256=file_hash(predictions))
        save_json(out / "inference_summary.json", summary)
        save_json(out / "progress.json", dict(summary, completed=len(rows)))
        return summary
    except BaseException as e:
        # A rejected configuration must not invalidate an already completed run.
        if identity_accepted or not previous_progress.exists():
            save_json(previous_progress, dict(state="failed", completed=len(saved), error_type=type(e).__name__))
        raise
    finally:
        append_jsonl(out / "sessions.jsonl", dict(elapsed_seconds=time.perf_counter()-started, gpu=gpu.finish()))
