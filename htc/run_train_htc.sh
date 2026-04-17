#!/bin/bash
set -euo pipefail

echo "[CHTC] Job started on $(hostname)"
echo "[CHTC] Working dir: $(pwd)"

RUN_NAME="${1:-run_default}"
DATA_CSV="${2:-Data/apod_preloaded_dataset.csv}"
IMAGES_DIR="${3:-images}"
CHECKPOINT_ROOT="${4:-checkpoints}"
EPOCHS="${5:-50}"
IMAGE_SIZE="${6:-96}"
N_LEVELS="${7:-4}"
N_STEPS="${8:-8}"
SAMPLES="${9:-1000}"
LR="${10:-1e-5}"
COND_DIM="${11:-256}"
BATCH_SIZE="${12:-32}"

CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}"
FLOW_CKPT_NAME="${RUN_NAME}-glow-ckpt.pth"
LLM_CKPT_NAME="${RUN_NAME}-llm-ckpt.pth"

mkdir -p "${IMAGES_DIR}" "${CHECKPOINT_DIR}" logs

echo "[CHTC] Downloading preloaded images to ${IMAGES_DIR}"
python3 scripts/download_preloaded_images.py \
  --csv_path "${DATA_CSV}" \
  --output_dir "${IMAGES_DIR}" \
  --splits train val test \
  --url_col best_url

echo "[CHTC] Starting training"
python3 train.py \
  --data_csv "${DATA_CSV}" \
  --images_dir "${IMAGES_DIR}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --samples "${SAMPLES}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --device cuda \
  --cond_dim "${COND_DIM}" \
  --n_levels "${N_LEVELS}" \
  --n_steps "${N_STEPS}" \
  --image_size "${IMAGE_SIZE}" \
  --flow_ckpt_name "${FLOW_CKPT_NAME}" \
  --llm_ckpt_name "${LLM_CKPT_NAME}"

echo "[CHTC] Training finished. Checkpoints in ${CHECKPOINT_DIR}"
