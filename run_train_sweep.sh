#!/bin/bash
set -e

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
echo "[INFO] Run name:           $RUN_NAME"
echo "[INFO] epochs=$EPOCHS image_size=$IMAGE_SIZE n_levels=$N_LEVELS n_steps=$N_STEPS"
echo "[INFO] samples=$SAMPLES lr=$LR cond_dim=$COND_DIM batch_size=$BATCH_SIZE"
echo "============================================================"
echo "[INFO] Python: $(python3 --version)"
echo "[INFO] Working dir: $(pwd)"
echo "[INFO] Files present: $(ls)"

# ── Rebuild Data/ module structure ────────────────────────────────────────────
echo "[INFO] Setting up Data/ module structure..."
mkdir -p Data
[ -f "dataset.py" ] && cp dataset.py Data/dataset.py
touch Data/__init__.py
echo "[OK] Data/ module ready"

# ── Rebuild models/ module structure ─────────────────────────────────────────
echo "[INFO] Setting up models/ module structure..."
mkdir -p models
[ -f "llm_encoder.py" ] && cp llm_encoder.py models/llm_encoder.py
[ -f "flow_model.py" ]  && cp flow_model.py  models/flow_model.py
touch models/__init__.py
echo "[OK] models/ module ready"
echo "[INFO] models/ contains: $(ls models/)"

# ── Verify CSV ────────────────────────────────────────────────────────────────
[ ! -f "apod_preloaded_dataset.csv" ] && echo "[ERROR] CSV not found" && exit 1
echo "[OK] CSV found"

# ── Download images ───────────────────────────────────────────────────────────
echo "[INFO] Downloading training images..."
mkdir -p Data/images

python3 - << 'PYEOF'
import pandas as pd
import requests
import os
from tqdm import tqdm

df = pd.read_csv("./apod_preloaded_dataset.csv")
df = df[df["split"] == "train"].dropna(subset=["best_url"])
print(f"[INFO] Downloading up to {len(df)} training images...")

save_dir = "./Data/images"
os.makedirs(save_dir, exist_ok=True)

failed = 0
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img_id = int(row['img_index'])
    save_path = f"{save_dir}/{img_id}.jpg"
    if os.path.exists(save_path):
        continue
    try:
        response = requests.get(row["best_url"], timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
    except:
        failed += 1

count = len(os.listdir(save_dir))
print(f"[INFO] Images ready: {count}, failed: {failed}")
PYEOF

echo "[OK] Image download complete"

# ── GPU check ─────────────────────────────────────────────────────────────────
python3 -c "
import torch
print(f'[INFO] CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[INFO] GPU: {torch.cuda.get_device_name(0)}')
    print(f'[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ── Pre-create checkpoints ────────────────────────────────────────────────────
mkdir -p checkpoints
echo "[INFO] checkpoints/ ready"

# ── Run training ──────────────────────────────────────────────────────────────
echo "[INFO] Launching train.py for run: $RUN_NAME"
echo "============================================================"

python3 train.py \
    --data_csv        ./apod_preloaded_dataset.csv \
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
