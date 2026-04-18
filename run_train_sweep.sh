#!/bin/bash
# =============================================================================
# run_train_sweep.sh — HTCondor executable for sweep jobs
# Arguments passed from train_sweep.sub via HTCondor variables:
#   $1 = run_name
#   $2 = epochs
#   $3 = image_size
#   $4 = n_levels
#   $5 = n_steps
#   $6 = samples
#   $7 = lr
#   $8 = cond_dim
#   $9 = batch_size
# =============================================================================

set -e

# ── Read arguments ────────────────────────────────────────────────────────────
RUN_NAME=$1
EPOCHS=$2
IMAGE_SIZE=$3
N_LEVELS=$4
N_STEPS=$5
SAMPLES=$6
LR=$7
COND_DIM=$8
BATCH_SIZE=$9

echo "============================================================"
echo "[INFO] Job started at:     $(date)"
echo "[INFO] Running on host:    $(hostname)"
echo "[INFO] Working directory:  $(pwd)"
echo "[INFO] Run name:           $RUN_NAME"
echo "------------------------------------------------------------"
echo "[INFO] Hyperparameters:"
echo "       epochs     = $EPOCHS"
echo "       image_size = $IMAGE_SIZE"
echo "       n_levels   = $N_LEVELS"
echo "       n_steps    = $N_STEPS"
echo "       samples    = $SAMPLES"
echo "       lr         = $LR"
echo "       cond_dim   = $COND_DIM"
echo "       batch_size = $BATCH_SIZE"
echo "============================================================"

# ── Python check ─────────────────────────────────────────────────────────────
echo "[INFO] Python binary:  $(which python3)"
echo "[INFO] Python version: $(python3 --version)"

# ── Verify critical files ─────────────────────────────────────────────────────
echo "[INFO] Checking required files..."
for f in train.py Data/apod_preloaded_dataset.csv; do
    if [ ! -e "$f" ]; then
        echo "[ERROR] Required file missing: $f"
        exit 1
    else
        echo "[OK]   Found: $f"
    fi
done

if [ ! -d "Data/images" ]; then
    echo "[ERROR] Data/images directory not found."
    exit 1
else
    IMG_COUNT=$(ls Data/images/ | wc -l)
    echo "[OK]   Data/images/ found — $IMG_COUNT files"
fi

# ── GPU check ─────────────────────────────────────────────────────────────────
echo "[INFO] Checking GPU..."
python3 -c "
import torch
print(f'[INFO] CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[INFO] GPU: {torch.cuda.get_device_name(0)}')
    print(f'[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('[WARNING] No GPU — training will be slow')
"

# ── Pre-create checkpoints folder BEFORE training starts ─────────────────────
# This ensures HTCondor can always transfer it back even if training crashes
mkdir -p checkpoints
echo "[INFO] checkpoints/ folder ready"

# ── Run training ──────────────────────────────────────────────────────────────
echo "[INFO] Launching train.py for run: $RUN_NAME"
echo "============================================================"

python3 train.py \
    --data_csv        ./Data/apod_preloaded_dataset.csv \
    --images_dir      ./Data/images \
    --checkpoint_dir  ./checkpoints \
    --epochs          $EPOCHS \
    --image_size      $IMAGE_SIZE \
    --n_levels        $N_LEVELS \
    --n_steps         $N_STEPS \
    --samples         $SAMPLES \
    --lr              $LR \
    --cond_dim        $COND_DIM \
    --batch_size      $BATCH_SIZE \
    --device          cuda \
    --flow_ckpt_name  ${RUN_NAME}-glow-ckpt.pth \
    --llm_ckpt_name   ${RUN_NAME}-llm-ckpt.pth

EXIT_CODE=$?

echo "============================================================"
echo "[INFO] train.py exited with code: $EXIT_CODE"
echo "[INFO] Job finished at: $(date)"
echo "============================================================"

exit $EXIT_CODE