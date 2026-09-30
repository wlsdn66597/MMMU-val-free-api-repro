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
