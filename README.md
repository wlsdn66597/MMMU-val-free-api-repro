# MMMU-val free generation + API evaluation

Qwen3-VL-4B-Instruct로 MMMU validation 900문항(객관식 847 + 주관식 53)을 자유 생성하고, Qwen 공개 규칙·프롬프트와 Codyssey judge API로 평가합니다. 추론은 로컬 GPU에서 문항당 1회 수행합니다. 규칙 추출이 불가능한 답변만 API에 전달합니다.

## 설치

Linux, NVIDIA GPU, Python 3.10 이상이 필요합니다. 아래 GPU 의존성은 기존 실험 환경의 버전에 맞춘 직접 의존성 핀입니다. 이 저장소의 새 파이프라인은 CPU 테스트를 통과했으며, 전체 GPU 실행과 실제 Codyssey 호출은 사용자의 환경에서 확인해야 합니다. 24GB RTX 4090에서 128000 문맥이 수용된다고 보장하지 않습니다.

```bash
git clone https://github.com/wlsdn66597/MMMU-val-free-api-repro.git
cd MMMU-val-free-api-repro
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m mmmu_repro --help
```

모델·데이터는 기본값으로 다운로드합니다. 이미 다운로드한 모델 snapshot이나 `MMMU_DEV_VAL.tsv`가 있으면 `--model-path`와 `--data-root`로 지정할 수 있습니다. Hugging Face의 MMMU 원본 parquet 디렉터리와 이 TSV는 서로 다른 입력 형식입니다.

## 키 입력 후 전체 실행

Codyssey에서 발급한 virtual key를 입력합니다. 입력은 화면과 명령 이력에 남지 않습니다. 키를 파일·Git·명령 인자에 넣지 않습니다.

이미 `.env` 파일에 `CODYSSEY_API_KEY=...`를 저장했다면 `set -a; source .env; set +a`로 현재 셸에 불러올 수 있습니다. `.env.example`은 빈 예시 파일이며 Git 추적 대상이므로 키를 넣거나 불러오지 마세요. 이 프로그램은 `.env`를 자동으로 읽지 않습니다.

```bash
read -rsp 'Codyssey API key: ' CODYSSEY_API_KEY; echo
export CODYSSEY_API_KEY
mkdir -p logs
nohup bash scripts/run_mmmu_val_free.sh \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --data-root data \
  --output-root results/qwen_free3407 \
  --seed 3407 \
  --max-tokens 32768 \
  --max-model-len 128000 \
  > logs/qwen_free3407.log 2>&1 < /dev/null &
echo $! > logs/qwen_free3407.pid
```

실행 순서: 소액의 synthetic API 확인 1회 → 전체 900문항 입력·문맥 검사 → 로컬 Qwen 추론 → API 필요 문항 집계 → 규칙/API 추출 및 채점. API 확인은 실제 요청 형식과 모델 접근성을 검증합니다. 파라미터를 지원하지 않으면 장시간 GPU 추론 전에 중단합니다. Codyssey 모델별 토큰 차감 계수는 해당 서비스 정책을 따릅니다.

원하는 로컬 경로로 지정하려면 다음과 같이 실행합니다.

```bash
MODEL_SNAPSHOT="/your/model/snapshot"
MMMU_TSV="/your/data/MMMU_DEV_VAL.tsv"
bash scripts/run_mmmu_val_free.sh \
  --model-path "$MODEL_SNAPSHOT" \
  --data-root "$MMMU_TSV" \
  --output-root results/qwen_free3407
```

## 고정 기본 설정

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3-VL-4B-Instruct` |
| 모델 revision | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| Qwen 코드 commit | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` |
| 샘플링 | temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.5 |
| seed | 3407 — 엔진과 문항별 SamplingParams 모두 지정 |
| 출력 상한 | 32768 |
| 총 문맥 상한 | 128000 — 이미지·텍스트 입력 + 출력 |
| 이미지 min/max pixels | 1,003,520 / 4,014,080 |
| 추론 | BF16, vLLM.generate, 자유 생성 1회, 추가 CoT 없음 |
| 메모리 운영 설정 | gpu_memory_utilization 0.90, max_num_seqs 1, max_num_batched_tokens 2048, chunked prefill, eager |
| judge endpoint | `https://copa.codyssey.kr/v1/chat/completions` |
| judge | `gpt-5.4-mini`; 기본 요청은 Codyssey 콘솔 예시와 동일한 `model`·`messages` 두 필드 |

논문 §5.11의 4B 설정(1.0 / 1.0 / 40 / 2.0)은 **Text-Centric Tasks**에 제시되어 있으므로 MMMU 공식 설정으로 간주하지 않습니다. 비교가 필요하면 새 결과 디렉터리에서 `--profile paper-text-4b-experimental`을 명시합니다. 기본값은 MMMU 공개 코드·README의 샘플링이며, 사용자 지정 seed 3407과 대체 judge 사용 등의 차이는 [출처와 변경점](docs/provenance.md)에 적었습니다. 공개 점수 67.40%의 완전 재현을 보장하는 저장소가 아닙니다.

Judge의 `--judge-max-tokens`, `--reasoning-effort`, `--judge-temperature`는 Codyssey 게이트웨이가 해당 필드를 지원한다고 확인한 경우에만 명시합니다. 옵션을 명시하면 평가 설정과 캐시 식별자에 기록됩니다. API가 400을 반환할 때는 안전한 `error.code`·`error.param`·`error.type`만 표시하고 키와 원본 오류 본문은 출력하지 않습니다.

## 문맥·VRAM 확인

API 키 없이 전체 입력을 검사할 수 있습니다. processor 다운로드/이미지 처리 시간은 소요되지만 GPU 엔진은 로드하지 않습니다.

```bash
bash scripts/run_mmmu_val_free.sh --stage preflight \
  --output-root results/context_check128k --max-model-len 128000
```

`inference/preflight.json`의 `maximum_input_tokens`, `required_context`를 확인합니다. 모든 문항에서 `input_tokens + 32768 <= max_model_len`을 요구하며, 이미지 수 제한과 실제 vLLM 입력 토큰도 검증합니다. 입력 삭제·자동 출력 축소를 하지 않습니다.

128000 설정에서 KV cache/메모리 부족이 발생하면, `required_context <= 65536`인지 확인하고 **새 결과 디렉터리**에서 `--max-model-len 65536`을 시도할 수 있습니다. 이것도 24GB에서의 성공을 보장하지 않습니다. 수용되지 않으면 GPU 자원을 늘려야 합니다. 문맥 설정을 바꾸면 그 값을 기록합니다.

## 단계별 실행 및 재개

```bash
# API 키 없이 추론까지. 같은 설정·같은 디렉터리로 다시 실행하면 완료 문항은 건너뜁니다.
bash scripts/run_mmmu_val_free.sh --stage infer --output-root results/qwen_free3407

# 추론 완료 후 원격 호출 없이 API 필요 문항/프롬프트 길이를 집계합니다.
python -m mmmu_repro judge-plan

# API 요청 형식과 인증을 synthetic 문항 1건으로 확인합니다.
python -m mmmu_repro api-check

# 먼저 최대 20건의 실제 API 추출. 나머지 문항은 미완료로 남습니다.
python -m mmmu_repro judge --max-api-calls 20

# 확인 후 나머지 평가. 완료된 추출 결과는 재사용합니다.
python -m mmmu_repro judge
```

환경변수 키가 없는 경우에도 `judge-plan`은 실행됩니다. `--max-api-calls`는 이번 실행의 성공한 문항 호출 수 제한입니다. 네트워크/429/5xx 재시도, 별도 `api-check`, 서비스 내부 과금 계수는 포함하지 않으므로 절대 지출 한도가 아닙니다. API 반환 usage를 기록하지만 공급자가 usage를 생략하면 추측하지 않고 표시합니다.

개발용 GPU smoke test는 `python -m mmmu_repro infer --limit 2 --output-dir results/smoke/inference`로 실행합니다. 전체 900개 coverage를 확인한 뒤 앞 2개만 생성하며, 이 결과로 최종 baseline 점수를 내지 않습니다.

## 상태·결과

```bash
tail -n 30 logs/qwen_free3407.log
cat results/qwen_free3407/inference/progress.json
wc -l results/qwen_free3407/inference/predictions.jsonl
cat results/qwen_free3407/inference/inference_summary.json
cat results/qwen_free3407/judge/summary.json
cat results/qwen_free3407/judge/report.md
```

같은 Qwen 답변을 **API 호출 전(로컬 규칙만)**과 **API 추출 후**로 나눠 보려면 다음을 실행합니다. 원격 요청은 하지 않습니다. 채점 중에는 `after_api.accuracy_pct`가 `null`이고, `current_minimum_pct`는 현재 확인된 정답만 반영한 하한입니다. `before_api`는 규칙으로 읽지 못한 답을 전체 900문항 분모에서 오답으로 계산합니다.

```bash
python scripts/report_api_effect.py \
  --inference-dir results/qwen_free3407/inference \
  --judge-dir results/qwen_free3407/judge
```

`overall`, `multiple_choice`, `open`에서 규칙 추출 수·API 대상/처리/정답/남은 수와 전후 정확도를 확인할 수 있습니다. API는 모델 답을 수정하지 않으므로 전후 차이는 **답 추출·채점 범위의 차이**입니다.

## 추가 로컬 파서 비교 (API 호출 없음)

공개 규칙은 답변 전체에서 여러 선택지 문자가 발견되면 미파싱으로 남길 수 있습니다. `scripts/evaluate_local_parser.py`는 저장된 900개 답변을 읽어 **공개 규칙이 미파싱한 문항에만** 추가 파서를 적용하는 별도 실험입니다. 새 추론·API 요청·의존성 설치 없이 실행합니다.

```bash
python scripts/evaluate_local_parser.py \
  --inference-dir results/qwen_free3407_ctx65536/inference \
  --output-dir results/qwen_free3407_ctx65536/local_parser_v1

cat results/qwen_free3407_ctx65536/local_parser_v1/report.md
```

API 결과와도 비교하려면 같은 명령에 `--judge-dir results/qwen_free3407_ctx65536/judge`를 추가합니다. 완료된 API 기록만 읽으며, API 처리가 끝나지 않았으면 API 최종 정확도는 `null`입니다. 같은 파서·같은 입력으로 다시 실행하면 이 별도 보고서만 갱신됩니다. 파서 코드나 입력이 바뀌면 새로운 출력 디렉터리를 사용합니다.

- 객관식: `Final answer: B`, `Answer: (B)`, 마지막의 `\boxed{B}`, 마지막 줄의 독립된 선택지 문자·정확한 선택지 문구를 인식합니다. 복수 답·추측·거절·출력 잘림은 보수적으로 미파싱 상태로 둡니다. 풀이에서 마지막에 언급된 선택지를 임의로 고르지 않습니다.
- 주관식: 명시된 최종 답이나 마지막 boxed 식을 먼저 추출합니다. 정답은 추출 함수에 전달하지 않습니다. 이후 문자열과 간단한 숫자·분수의 정확한 동치로 채점합니다. 단위 변환·복잡한 수식 동치는 지원하지 않으므로 API 의미 비교와 채점 범위가 다릅니다.
- `report.md`, `summary.json`: 기존 규칙 / 규칙+추가 파서 / 규칙+API 점수, 미파싱 수, 추가 추출 중 정답·오답 수, 출력 종료 원인별 기존 미파싱 수.
- `records.jsonl`: 문항별 추출 방법·근거·정답 여부와 원문 마지막 1600자. `unresolved.jsonl`: 여전히 미파싱인 문항. `api_disagreements.jsonl`: 추가 파서와 API가 다른 문항(객관식은 선택지, 주관식은 정오 판단 비교).

이 파서는 공개 Qwen 규칙에 대한 추가 실험으로 따로 보고합니다. 실제 답변에서 복구율과 추출 오류를 확인해야 하며, 미파싱 감소만으로 추출이 정확하다고 판단하지 않습니다. API와의 불일치는 검토 대상이며 어느 쪽이 맞는지의 증거 자체는 아닙니다. 평가·추론 본체를 수정하지 않으므로 기존 채점의 재개 설정에 영향을 주지 않습니다.

## 별도 로컬 모델 judge (Qwen3-8B)

저장된 VLM 응답에 먼저 기존 공개 규칙을 적용하고, **미파싱 응답에만** 텍스트 모델 `Qwen/Qwen3-8B`를 호출합니다. VLM을 다시 실행하거나 Codyssey API를 호출하지 않습니다. 주관식은 공개 judge와 같이 평가 시점에 정답을 선택지 A, `Other Answers`를 B로 넣어 의미 일치를 묻습니다. 따라서 추가 규칙 파서의 주관식 표면 일치 점수와 평가 기준이 다릅니다. GPT-3.5 Turbo와 같은 모델이나 점수를 재현한다는 뜻은 아닙니다.

먼저 GPU 컴퓨터의 **현재 Hugging Face 캐시**를 확인합니다. `HF_HOME`을 사용했다면 모델 다운로드와 추론에서 같은 값을 유지하세요. 없거나 파일이 불완전하면 두 번째 명령이 다운로드합니다. 모델 파일은 Git에 넣지 않습니다.

```bash
python scripts/prepare_local_judge_model.py
python scripts/prepare_local_judge_model.py --download-if-missing
```

첫 명령이 `state: cached`이면 두 번째 명령은 생략할 수 있습니다. 로컬 모델 디렉터리가 별도로 있다면 두 명령과 아래 평가 명령에 `--model /path/to/Qwen3-8B`를 지정하세요. `snapshot`과 revision은 결과 manifest에 남습니다.

```bash
# 모델 로드 전 전체 judge 입력 길이 확인. 초과하면 답변을 자르지 않고 중단합니다.
python scripts/evaluate_local_model_judge.py \
  --inference-dir results/qwen_free3407_ctx65536/inference \
  --judge-dir results/qwen_free3407_ctx65536/judge \
  --output-dir results/qwen_free3407_ctx65536/local_model_qwen3_8b_v1 \
  --preflight-only

# preflight 성공 후: GPU 모델 로드, 미파싱 문항만 추출, API 기록과 비교
python scripts/evaluate_local_model_judge.py \
  --inference-dir results/qwen_free3407_ctx65536/inference \
  --judge-dir results/qwen_free3407_ctx65536/judge \
  --output-dir results/qwen_free3407_ctx65536/local_model_qwen3_8b_v1

cat results/qwen_free3407_ctx65536/local_model_qwen3_8b_v1/report.md
```

기본 문맥 상한은 32768, 출력 상한은 16, 온도는 0, Qwen3 thinking은 비활성화합니다. 필요한 문맥 길이가 32768을 넘으면 자동으로 자르거나 최대값을 올리지 않습니다. Qwen3-8B의 32768 초과 입력은 YaRN 등 별도 설정과 GPU 메모리 확인이 필요합니다. 원본 VLM 추론이 GPU에서 완전히 끝난 뒤 실행하세요. 문항별 로컬 judge 응답은 `extractions.jsonl`에 즉시 저장되어 같은 설정으로 중단 지점부터 재개할 수 있습니다. `summary.json`과 `report.md`는 규칙, 규칙+로컬 모델, 규칙+API 점수를 같은 900문항 분모로 비교합니다. 무효 형식·출력 길이 제한은 미파싱으로 남고 오답에 포함합니다. 결과의 API 불일치 사례를 수동 검토해야 합니다.

v1에서는 preflight가 만든 토큰을 문자열로 다시 디코딩하고 vLLM이 재토큰화하여, 사전 검사보다 실제 입력 토큰이 많아질 수 있었습니다. v2는 **같은 토큰 ID를 검사와 vLLM 입력에 모두 사용**합니다. v1 실행이 실패했다면 `git pull` 후 `local_model_qwen3_8b_v2`처럼 새 결과 디렉터리를 사용하세요. v2 사전 검사가 32768 초과를 보고하면 그 입력은 실제로 기본 문맥에 들어가지 않습니다.

v2에서 `max_input_tokens=2`로 표시된 경우는 tokenizer가 반환한 `input_ids` 대신 딕셔너리 키를 센 오류입니다. 수정된 버전은 `input_ids`를 명시적으로 꺼내 정수 토큰만 허용합니다. 이 값으로 모델을 실행하지 말고 `git pull` 후 `local_model_qwen3_8b_v3`처럼 새 결과 디렉터리에서 preflight를 다시 수행하세요.

그 경우 24GB GPU에서는 [공식 4비트 AWQ 모델](https://huggingface.co/Qwen/Qwen3-8B-AWQ)과 명시적인 YaRN 확장을 별도 실험으로 사용할 수 있습니다. 아래 명령은 새 모델의 캐시를 확인하고 없으면 다운로드한 뒤, 원문을 자르지 않고 65536 문맥으로 검사·실행합니다. 먼저 preflight의 `required_context`가 65536 이하인지 확인하세요. AWQ 양자화와 YaRN은 원래 BF16 8B와 다른 평가 설정입니다.

vLLM 0.28.0은 `LLM(..., rope_scaling=...)` 인자를 받지 않습니다. 현재 스크립트는 해당 버전의 [문맥 확장 방식](https://docs.vllm.ai/en/v0.28.0/features/context_extension/)인 `hf_overrides={"rope_parameters": ...}`를 사용합니다. `unexpected keyword argument 'rope_scaling'` 오류가 났다면 `git pull` 후 새 출력 폴더로 실행하세요.

```bash
python scripts/prepare_local_judge_model.py --model Qwen/Qwen3-8B-AWQ || \
python scripts/prepare_local_judge_model.py --model Qwen/Qwen3-8B-AWQ --download-if-missing

python scripts/evaluate_local_model_judge.py \
  --model Qwen/Qwen3-8B-AWQ --max-model-len 65536 --rope-factor 2 \
  --inference-dir results/qwen_free3407_ctx65536/inference \
  --judge-dir results/qwen_free3407_ctx65536/judge \
  --output-dir results/qwen_free3407_ctx65536/local_model_qwen3_8b_awq_yarn_v1 \
  --preflight-only

python scripts/evaluate_local_model_judge.py \
  --model Qwen/Qwen3-8B-AWQ --max-model-len 65536 --rope-factor 2 \
  --inference-dir results/qwen_free3407_ctx65536/inference \
  --judge-dir results/qwen_free3407_ctx65536/judge \
  --output-dir results/qwen_free3407_ctx65536/local_model_qwen3_8b_awq_yarn_v1
```

### 주관식 정답 위치 민감도 점검

공개 Qwen 방식은 주관식 채점 시 참조 정답을 항상 선택지 A에 둡니다. 아래 점검은 **저장된 동일한 VLM 답변 53개**를 사용하여 참조 정답을 B로 옮기고 `Other Answers`를 A에 둔 상태에서 규칙 추출과 미파싱 대상 로컬 모델 추출을 다시 적용합니다. VLM 추론과 API 호출은 발생하지 않습니다. 원본 로컬 judge와 동일한 모델 스냅샷·문맥·출력 설정을 검사하고, 정답 위치만 바뀐 쌍의 판정을 비교합니다.

```bash
python scripts/audit_open_judge_bias.py \
  --inference-dir results/qwen_free3407_tok4096_ctx65536/inference \
  --original-local-dir results/qwen_free3407_tok4096_ctx65536/local_model_awq_yarn_v1 \
  --output-dir results/qwen_free3407_tok4096_ctx65536/open_position_audit_v1 \
  --preflight-only

python scripts/audit_open_judge_bias.py \
  --inference-dir results/qwen_free3407_tok4096_ctx65536/inference \
  --original-local-dir results/qwen_free3407_tok4096_ctx65536/local_model_awq_yarn_v1 \
  --output-dir results/qwen_free3407_tok4096_ctx65536/open_position_audit_v1

cat results/qwen_free3407_tok4096_ctx65536/open_position_audit_v1/report.md
```

`original_only`는 정답이 A일 때만 맞은 문항 수, `swapped_only`는 B일 때만 맞은 문항 수입니다. `always_A`는 위치를 바꿔도 A를 고른 문항 수입니다. `cases.jsonl`에 문항별 원본·교체 판정과 답변 끝부분을 기록하므로 위치에 민감한 사례를 직접 검토할 수 있습니다. 위치 민감도만으로 어느 판정이 의미상 옳은지는 확정할 수 없습니다. 8192토큰 실행도 같은 명령에서 세 경로의 `tok4096`을 `tok8192`로 바꿔 점검할 수 있습니다.

### 정답을 보여주지 않는 별도 최종 답 추출

`evaluate_blind_local_judge.py`는 저장된 응답 전체를 다시 추출합니다. 기존 Qwen 규칙의 판정을 재사용하지 않습니다. 객관식은 질문·원래 선택지·응답, 주관식은 질문·응답만 추출 모델에 전달합니다. 입력 필드를 명시적으로 제한하여 정답·이전 채점·API 판정은 전달하지 않습니다. 추출 모델은 문제를 새로 풀지 않고 응답에 명시된 최종 답을 JSON으로 반환하며, 원문에 실제 있는 인용 근거를 검증합니다. 주관식 답을 새로 계산하거나 임의로 고쳐 쓰면 무효 처리합니다.

기본값은 기존 캐시의 `Qwen/Qwen3-8B-AWQ`, YaRN 2, 문맥 65536, thinking 비활성화, 온도 0, seed 3407입니다. 인용 근거를 포함하는 JSON을 받으므로 추출 모델의 출력 한도는 256입니다. 이는 VLM의 출력 토큰 설정을 바꾸지 않습니다. 추가 의존성 설치·API 키·VLM 재추론 없이 실행합니다.

```bash
# 먼저 주관식 53문항의 입력 길이를 검사합니다.
python scripts/evaluate_blind_local_judge.py \
  --inference-dir results/qwen_free3407_tok4096_ctx65536/inference \
  --output-dir results/qwen_free3407_tok4096_ctx65536/blind_open_v1 \
  --scope open --preflight-only

python scripts/evaluate_blind_local_judge.py \
  --inference-dir results/qwen_free3407_tok4096_ctx65536/inference \
  --output-dir results/qwen_free3407_tok4096_ctx65536/blind_open_v1 \
  --scope open

# 900문항 전체 평가: 별도 폴더를 사용합니다.
python scripts/evaluate_blind_local_judge.py \
  --inference-dir results/qwen_free3407_tok4096_ctx65536/inference \
  --output-dir results/qwen_free3407_tok4096_ctx65536/blind_all_v1

cat results/qwen_free3407_tok4096_ctx65536/blind_all_v1/report.md

# 추출 완료 뒤 CPU에서 같은 추출을 재채점할 수 있습니다.
python scripts/evaluate_blind_local_judge.py \
  --inference-dir results/qwen_free3407_tok4096_ctx65536/inference \
  --output-dir results/qwen_free3407_tok4096_ctx65536/blind_all_v1 --score-only
```

`extractions.jsonl`에는 정답을 포함하지 않는 추출 결과·인용·원문·상태를 저장합니다. `scored_records.jsonl`에서만 정답과 비교하며, `report.md`는 정확한 수치/표면 일치 점수와 기준답의 소수 정밀도에 맞춘 점수를 각각 보여줍니다. 여러 허용답을 나타내는 리스트 문자열은 개별 답으로 비교합니다. 분수·제곱근은 제한된 산술 문법으로 처리합니다. 소수 기준답은 표시된 소수 정밀도로 반올림(half-up)을 허용하는 별도 점수를 제공하고, 정수·분수 기준답에는 정확한 수치 일치를 요구합니다. 상대 오차 허용은 없습니다.

#### 기존 blind 추출의 표기 오류를 CPU에서 재검증

v1 검증은 `C. $12,000`처럼 문자와 내용이 함께 나온 객관식 답과 `answer="c", evidence="Answer: c"`처럼 답을 포함하는 인용도 미파싱 처리했습니다. v2에서는 원래 선택지 내용과 일치하는 문자·내용 조합을 문자로 정규화하고, 주관식 답이 검증된 인용 안에 독립된 값으로 들어 있는 경우를 허용합니다. Markdown/LaTeX 표시 차이와 동일한 수치 표현도 처리합니다. 단일 최종 답 문장에서는 수치 답을 분리할 수 있지만, 풀이 전체에서 정답과 같은 수치를 검색하는 방식은 사용하지 않습니다. 원문에 없는 인용·선택지와 충돌하는 문자·근거 없는 값은 계속 미파싱 처리합니다.

아래 명령은 **기존 judge 원문을 그대로 재사용**합니다. GPU 모델 로딩, VLM 재추론, 새 judge 호출, API 호출 없이 CPU에서 수행합니다. 원래 생성 프롬프트와 입력 해시를 검증하고 이전 manifest와 상태 변화도 기록합니다. 원본 결과는 보존하며 반드시 별도 출력 폴더를 지정합니다. `--cached-dir`에는 `blind_manifest.json`과 `extractions.jsonl`이 있는 기존 blind 결과 폴더를 넣습니다.

```bash
git pull
RUN=results/qwen_free3407_tok4096_ctx65536

python scripts/revalidate_blind_extractions.py \
  --inference-dir "$RUN/inference" \
  --cached-dir "$RUN/blind_all_v1" \
  --output-dir "$RUN/blind_all_v2_revalidated"

cat "$RUN/blind_all_v2_revalidated/report.md"
```

주관식만 추출했던 폴더도 같은 명령으로 재검증할 수 있으며 범위는 기존 manifest에서 읽습니다. 코드가 바뀐 v1 결과에 `--score-only`를 직접 적용하면 소스 해시 검사로 중단되므로 위 재검증 명령을 사용합니다. 완전한 원본 응답이 필요하여 `scored_records.jsonl` 파일만으로 새 정확도를 확정하지 않습니다. 표기 정규화 후에도 추출 모델의 잘못된 결론 선택과 인용 오류는 남을 수 있으므로 미파싱 상태와 개별 근거를 함께 살펴봐야 합니다.

v3는 객관식 답과 원래 선택지 양쪽의 끝 마침표에 같은 정규화를 적용합니다. `B. Lateral geniculate.`와 선택지 `Lateral geniculate.`처럼 동일한 내용이 v2에서 거부된 비교 오류를 수정합니다. v2 결과가 이미 있으면 `--cached-dir "$RUN/blind_all_v2_revalidated" --output-dir "$RUN/blind_all_v3_revalidated"`로 원래 judge 원문을 다시 재검증할 수 있습니다. 답이 같은지 정답으로 판단하는 방식이나 인용 검사는 변경하지 않습니다.

단위 변환은 수행하지 않습니다. 기준답에 단위가 있으면 일치해야 하고, 기준답이 단위 없는 수치이면 인식 가능한 응답 단위를 제외한 크기를 비교하되 단위 검토 대상으로 기록합니다. 백분율은 비율로 변환합니다. 미파싱·무효 JSON·원문에 없는 인용은 오답에 포함하고 상태별 수를 보고합니다. 인용이 있다고 추출이 항상 옳은 것은 아니므로 결과를 수동 검토해야 합니다. 이 점수는 정답 추출과 채점을 분리한 추가 평가이며 기존 공식 방식 점수와 구분해 기록합니다.

- `manifest.json`: 데이터 해시·선택 ID·모델·sampling·소스 해시. 변경된 설정으로 같은 폴더에 이어 쓰지 않습니다.
- `preflight.json`: 문항별 입력 토큰·프롬프트/이미지 해시와 필요 문맥.
- `predictions.jsonl`: Qwen 원문, 종료 원인, 실제 입출력 토큰, 문항별 추론 초.
- `environment.json`, `requirements.freeze.txt`: 실제 설치 환경.
- `sessions.jsonl`: 호출 실행별 경과 초와 1초 간격 전체 GPU 사용량 표본. 다른 프로세스 메모리가 포함될 수 있습니다.
- `extractions.jsonl`: 규칙/API 여부, judge 응답·사용량·실제 반환 모델, 최종 선택지.
- `summary.json`: 전체·분야별·객관식/주관식 정확도. 미완료 상태에서는 최종 정확도를 출력하지 않습니다.

외부 강제 종료 뒤 `.running.lock`이 남으면 먼저 프로세스가 종료됐는지 확인하고 그 lock 파일만 제거한 뒤 재실행합니다. JSONL 마지막 줄이 강제 종료로 손상됐으면 파일을 백업하고 그 미완성 마지막 줄만 복구해야 합니다. 다른 완료 문항을 다시 생성하지 않습니다.

## 평가 범위와 기록

`MMMU_DEV_VAL.tsv`에서 validation만 선택합니다. dev split은 최종 점수에 합치지 않습니다. 30분야 × 30문항, 중복 ID, 객관식/주관식 수를 검사합니다. 두 개의 알려진 배포 파일 MD5를 인식하며 실제 사용한 버전과 SHA256을 기록합니다. 알 수 없는 버전은 중단합니다. 배포 서버의 HTTPS 인증서 오류가 재현되어, HTTPS 실패 시 같은 서버의 공개 HTTP 파일을 **고정 SHA256까지 일치하는 경우에만** 사용합니다. API 키는 데이터 다운로드에 사용하지 않습니다. 다운로드 경로는 `.source.json`에 기록합니다. 이 경로도 실패하면 검증된 로컬 TSV를 지정합니다.

주관식은 **평가 단계에서만** 정답을 A, `Other Answers`를 B로 구성하는 Qwen 방식을 사용합니다. 정답은 VLM 추론 프롬프트에 들어가지 않습니다. 원래 Qwen 응답을 judge가 수정하는 흐름이 아니며, 생성된 응답에 대응하는 선택지를 추출합니다.

judge 모델 교체, seed 적용, 순차 추론, 장애 시 중단, `Z`를 최종 미일치로 인정하는 정책은 공개 원본과의 차이입니다. judge가 문자를 추출했다고 그 문항이 정답인 것은 아닙니다. 저장된 최종 선택지를 정답과 비교합니다. 모델/프롬프트를 바꿔 점수가 높은 조합을 반복 선택하는 용도로 평가셋을 사용하지 않습니다.

## CPU 검증

```bash
# 별도 CPU 환경에서는 GPU requirements 대신 패키지 자체만 설치
python -m pip install -e .
python -m unittest discover -s tests -v
```

라이선스와 원본 출처: [provenance](docs/provenance.md), [Qwen Apache-2.0](src/mmmu_repro/vendor/QWEN_LICENSE).
