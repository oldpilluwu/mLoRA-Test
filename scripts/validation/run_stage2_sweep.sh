#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
DEVICE="${DEVICE:-cuda:0}"
PRECISION="${PRECISION:-int8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUT_ROOT="${OUT_ROOT:-validation_runs/stage2}"

TASKS="${TASKS:-32}"
DATA_SIZE="${DATA_SIZE:-128}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-4}"
CUTOFF_LEN="${CUTOFF_LEN:-1024}"
RANK="${RANK:-8}"

CONCURRENCY="${CONCURRENCY:-1 2 4 8 16 32}"
WORKLOADS="${WORKLOADS:-homogeneous rank_heterogeneous seq_heterogeneous batch_heterogeneous}"
HETERO_RANKS="${HETERO_RANKS:-4,8,16,32,64,128}"
HETERO_SEQ_WORDS="${HETERO_SEQ_WORDS:-16,64,256,768}"
HETERO_MINI_BATCH_SIZES="${HETERO_MINI_BATCH_SIZES:-1,2,4,8}"

mkdir -p "${OUT_ROOT}"

python scripts/validation/generate_validation_configs.py \
  --out-dir "${OUT_ROOT}/configs" \
  --workloads ${WORKLOADS} \
  --concurrency ${CONCURRENCY} \
  --tasks "${TASKS}" \
  --data-size "${DATA_SIZE}" \
  --num-epochs "${NUM_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --mini-batch-size "${MINI_BATCH_SIZE}" \
  --cutoff-len "${CUTOFF_LEN}" \
  --rank "${RANK}" \
  --seq-words 64 \
  --hetero-ranks "${HETERO_RANKS}" \
  --hetero-seq-words "${HETERO_SEQ_WORDS}" \
  --hetero-mini-batch-sizes "${HETERO_MINI_BATCH_SIZES}"

python scripts/validation/run_fixed_baselines.py \
  --base-model "${BASE_MODEL}" \
  --config-dir "${OUT_ROOT}/configs" \
  --output-dir "${OUT_ROOT}/results" \
  --device "${DEVICE}" \
  --precision "${PRECISION}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" \
  --continue-on-error

python scripts/validation/summarize_jsonl.py \
  --trace-dir "${OUT_ROOT}/results/jsonl" \
  --output "${OUT_ROOT}/results/summary.csv"

echo "Stage 2 sweep complete."
echo "Summary: ${OUT_ROOT}/results/summary.csv"
