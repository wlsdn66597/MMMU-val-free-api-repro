# Sources and reproducibility boundaries

- Qwen3-VL source commit: `96588727e44c78b25ba03ea03b8e12f7e64fd0da`
- [Public MMMU inference code](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)
- [Public MMMU extraction code](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/eval_utils.py)
- [Public Instruct command](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/infer_instruct.sh)
- [Evaluation Reproduction README](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/README.md#evaluation-reproduction)
- [Technical report v1](https://arxiv.org/html/2511.21631v1): §5.2 covers multimodal reasoning, §5.11 covers text-centric evaluation.
- [VLMEvalKit dataset checksums](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/image_mcq.py): `MMMUDataset` uses MD5 `585e8ad75e73f75dcad265dfd0417d64`. Qwen's pinned loader uses `521afc0f3bf341e6654327792781644d`. Both versions are explicitly recorded; no claim of byte identity.
- Codyssey endpoint/authentication were provided from the user's console examples: `POST https://copa.codyssey.kr/v1/chat/completions`, `Authorization: Bearer <virtual-key>`. No key or private console contents are distributed.

## Preserved

`vendor/qwen_extract.py` contains the unchanged `can_infer_option`, `can_infer_text`, `can_infer`, and `build_prompt` functions from the pinned Qwen source. They are Apache-2.0 licensed. Inference uses the same hint/question/options strings, image-first layout, processor chat template, pixel limits, and no added CoT. Official `infer_instruct.sh` does not pass `--use-cot`. Output is unconstrained, one completion per question, maximum 32768 tokens.

Open question reformatting with A=reference/B=Other Answers happens only in the evaluator, matching Qwen's `MMMU_preproc`. Rules are tried before the judge. The judge prompt includes Qwen's two extraction examples and full generated answer.

## Explicit changes

| Component | Public code | This repository |
|---|---|---|
| Seed | MMMU code hardcodes 42; top-level README says 3407 | 3407 in LLM and per-request SamplingParams |
| Split | DEV_VAL loaded; combined and split metrics | validation 900 only, balanced coverage enforced |
| Batch | all requests passed to generate | one request at a time for memory and checkpointing |
| Engine | default runtime choices | BF16, eager, chunked prefill, explicit limits |
| Judge | script names gpt-3.5-turbo-0125 | gpt-5.4-mini through Codyssey; default request has only model and messages, as in the user's console example |
| API retry | nested retries; random fallback | bounded transport retry; error stops, successful calls cached |
| Judge output | heuristics; Z causes repeated attempts | exactly one valid letter or Z; Z is final unmatched |
| Unknown dataset hash | downloaded without post-download verification | fail; accepted known revisions recorded |
| Token limit | requested ceiling | all inputs checked before GPU generation, actual vLLM tokens cross-checked |

These choices do not establish equivalence to the private pipeline used for the published 67.40% score. The model name returned by a gateway is recorded, but a gateway alias cannot by itself prove a frozen backend snapshot. Transport retries may incur duplicate provider billing after a timeout; successful semantic results are never sampled again to improve scores.

## Sampling scope correction

The report's small-Instruct recipe (temperature=1.0, top_p=1.0, top_k=40, presence_penalty=2.0) appears under §5.11 **Text-Centric Tasks**. It is not sufficient evidence of MMMU generation settings. The default uses the published MMMU/README recipe (0.7/0.8/20/1.5). The text-task recipe is exposed only as `paper-text-4b-experimental` and labeled as such.

## Runtime validation

The actual 80,999,563-byte TSV was downloaded and checked against VLMEvalKit's published MD5 `585e8ad75e73f75dcad265dfd0417d64`. Its SHA256 is `cae0445a60c2a22f29cb9df43d2d97671baa8d5921df5d8667f4f3b1f643a478`. The source contains 1050 rows; filtering validation yielded exactly 900=847+53, with 30 questions in each of 30 `l2-category` subjects. Because the public dataset host presented an expired TLS certificate, the downloader can fall back to the same public HTTP artifact only with this exact SHA256. Judge requests always require HTTPS and never follow redirects.

CPU tests exercise prompt layout, official extraction functions, coverage, context budgets, API errors, strict judge output, safe resume and full 900-item mock evaluation. They do not substitute for an actual 4090 inference run or authenticated Codyssey test. Run `api-check` before GPU inference; no credentials are embedded or read during the CPU test suite.
