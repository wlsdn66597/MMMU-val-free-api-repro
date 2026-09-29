#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="results/qwen_free3407"
STAGE="all"
API_BASE="https://copa.codyssey.kr/v1"
JUDGE_MODEL="gpt-5.4-mini"
MAX_API_CALLS=900
INFER_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --api-base) API_BASE="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --max-api-calls) MAX_API_CALLS="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: bash scripts/run_mmmu_val_free.sh [--stage all|infer|judge|preflight] [--output-root DIR] [--model-path HF_ID_OR_PATH] [--data-root TSV_OR_DIR] [--max-model-len 128000] [other infer options]"
      exit 0 ;;
    *) INFER_ARGS+=("$1"); shift ;;
  esac
done
case "$STAGE" in all|infer|judge|preflight) ;; *) echo "Invalid stage: $STAGE" >&2; exit 2 ;; esac
if [[ "$STAGE" == all || "$STAGE" == judge ]]; then
  : "${CODYSSEY_API_KEY:?Set CODYSSEY_API_KEY in the shell first}"
  "$PYTHON" -m mmmu_repro api-check --api-base "$API_BASE" --judge-model "$JUDGE_MODEL"
fi
if [[ "$STAGE" == preflight ]]; then
  "$PYTHON" -m mmmu_repro preflight --output-dir "$OUTPUT_ROOT/inference" "${INFER_ARGS[@]}"
fi
if [[ "$STAGE" == all || "$STAGE" == infer ]]; then
  "$PYTHON" -u -m mmmu_repro infer --output-dir "$OUTPUT_ROOT/inference" "${INFER_ARGS[@]}"
fi
if [[ "$STAGE" == all || "$STAGE" == judge ]]; then
  "$PYTHON" -m mmmu_repro judge-plan --inference-dir "$OUTPUT_ROOT/inference" --output-dir "$OUTPUT_ROOT/judge"
  "$PYTHON" -u -m mmmu_repro judge --inference-dir "$OUTPUT_ROOT/inference" --output-dir "$OUTPUT_ROOT/judge" \
    --api-base "$API_BASE" --judge-model "$JUDGE_MODEL" --max-api-calls "$MAX_API_CALLS"
fi
