from dataclasses import asdict, dataclass

QWEN_COMMIT = "96588727e44c78b25ba03ea03b8e12f7e64fd0da"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
PROFILES = {
    "qwen-mmmu": dict(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5),
    # Section 5.11 describes TEXT-CENTRIC tasks, not a confirmed MMMU recipe.
    "paper-text-4b-experimental": dict(temperature=1.0, top_p=1.0, top_k=40, presence_penalty=2.0),
}


@dataclass(frozen=True)
class Config:
    model_path: str = "Qwen/Qwen3-VL-4B-Instruct"
    model_revision: str = MODEL_REVISION
    profile: str = "qwen-mmmu"
    seed: int = 3407
    max_tokens: int = 32768
    max_model_len: int = 128000
    min_pixels: int = 1003520
    max_pixels: int = 4014080
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 2048
    tensor_parallel_size: int = 1
    max_images: int = 10
    dtype: str = "bfloat16"

    def validate(self):
        if self.profile not in PROFILES:
            raise ValueError("Unknown profile")
        if not 0 < self.max_tokens < self.max_model_len:
            raise ValueError("max-model-len must exceed max-tokens (input + output share context)")
        if not 0 < self.min_pixels <= self.max_pixels:
            raise ValueError("Invalid pixel limits")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu-memory-utilization must be in (0, 1)")
        if min(self.max_num_seqs, self.max_num_batched_tokens, self.tensor_parallel_size, self.max_images) < 1:
            raise ValueError("Engine limits must be positive")
        if not self.model_revision:
            raise ValueError("Pin model-revision; for local snapshots it is a declared revision")
        return self

    def sampling(self):
        return dict(PROFILES[self.profile], repetition_penalty=1.0, seed=self.seed,
                    max_tokens=self.max_tokens, n=1, stop_token_ids=[])

    def as_dict(self):
        return dict(asdict(self), sampling=self.sampling(), qwen_source_commit=QWEN_COMMIT,
                    prompt="qwen-mmmu-instruct-no-added-cot", generation_calls_per_question=1)


def check_budget(input_tokens, config):
    required = input_tokens + config.max_tokens
    if required > config.max_model_len:
        raise ValueError(f"Context too small: input={input_tokens} + output={config.max_tokens} "
                         f"= {required} > max-model-len={config.max_model_len}. "
                         "Increase context in a new run; no silent truncation is allowed.")
    return required
