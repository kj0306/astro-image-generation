#!/bin/bash
# =============================================================================
# run_train.sh — HTCondor executable for astro-image-generation training
# HTCondor captures stdout → .out file and stderr → .err file automatically.
# Do NOT redirect output yourself (no > train.log).
# =============================================================================

set -e  # Exit immediately if any command fails

echo "============================================================"
echo "[INFO] Job started at: $(date)"
echo "[INFO] Running on host: $(hostname)"
echo "[INFO] Working directory: $(pwd)"
echo "[INFO] User: $(whoami)"
echo "============================================================"

# ── Python / Conda environment ────────────────────────────────────────────────
# Option A (preferred): activate your conda environment.
# Adjust the path below to match where conda is installed on CHTC nodes,
# or the path to a portable conda env tarball you've transferred.
#
# Example if using a tarball (uncomment and adapt):
# mkdir -p project-env
# tar -xzf project-env.tar.gz -C project-env
# export PATH="$(pwd)/project-env/bin:$PATH"
# export PYTHONPATH="$(pwd):$PYTHONPATH"

# Example if conda is available on the node (uncomment and adapt):
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate project-env

echo "[INFO] Python binary: $(which python3)"
echo "[INFO] Python version: $(python3 --version)"

# ── Verify critical files are present ────────────────────────────────────────
echo "[INFO] Checking required files..."

for f in train.py Data/apod_preloaded_dataset.csv; do
    if [ ! -e "$f" ]; then
        echo "[ERROR] Required file missing: $f"
        exit 1
    else
        echo "[OK]   Found: $f"
    fi
done

if [ ! -d "images" ]; then
    echo "[ERROR] Images directory not found. Check transfer_input_files or /staging path."
    exit 1
else
    IMG_COUNT=$(ls images/ | wc -l)
    echo "[OK]   images/ directory found — $IMG_COUNT files"
fi

# ── GPU check ─────────────────────────────────────────────────────────────────
echo "[INFO] Checking GPU availability..."
python3 -c "
import torch
print(f'[INFO] CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[INFO] GPU: {torch.cuda.get_device_name(0)}')
else:
    print('[WARNING] No GPU detected — training will be slow on CPU')
"

# ── Run training ──────────────────────────────────────────────────────────────
echo "[INFO] Launching train.py..."
echo "============================================================"

python3 train.py \
    --data_csv   ./Data/apod_preloaded_dataset.csv \
    --images_dir ./images \
    --checkpoint_dir ./checkpoints \
    --samples    1000 \
    --epochs     50 \
    --batch_size 32 \
    --lr         1e-5 \
    --device     cuda \
    --cond_dim   256 \
    --n_levels   4 \
    --n_steps    8 \
    --image_size 96

EXIT_CODE=$?

echo "============================================================"
echo "[INFO] train.py exited with code: $EXIT_CODE"
echo "[INFO] Job finished at: $(date)"
echo "============================================================"

exit $EXIT_CODE