#!/bin/bash
# =============================================================================
# test_local.sh — Quick smoke test to verify everything works BEFORE submitting
# Runs on the CHTC access point directly (no GPU, tiny settings, ~2-3 minutes)
#
# Run with:   bash test_local.sh
# =============================================================================

set -e

echo "============================================================"
echo "[TEST] Smoke test started at: $(date)"
echo "[TEST] This uses tiny settings just to verify code runs"
echo "[TEST] No GPU needed — runs on CPU"
echo "============================================================"

# ── Step 1: Check all imports work ───────────────────────────────────────────
echo ""
echo "[TEST] Step 1 — Checking all imports..."
python3 -c "
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
print('[OK] torch, pandas, PIL, tqdm, torchvision all imported')

from Data.dataset import AstroDataset
print('[OK] AstroDataset imported')

from models.llm_encoder import TextEncoder
print('[OK] TextEncoder imported')

from models.flow_model import GlowModel
print('[OK] GlowModel imported')

print('[OK] All imports successful')
"

# ── Step 2: Check CSV loads correctly ────────────────────────────────────────
echo ""
echo "[TEST] Step 2 — Checking CSV loads..."
python3 -c "
import pandas as pd
df = pd.read_csv('./Data/apod_preloaded_dataset.csv')
print(f'[OK] CSV loaded — {len(df)} total rows')
train = df[df.split == 'train']
print(f'[OK] Train split — {len(train)} rows')
print(f'[OK] Columns: {list(df.columns)}')
print(f'[OK] Sample img_index values: {list(train.img_index.head(5))}')
"

# ── Step 3: Check images exist and are readable ───────────────────────────────
echo ""
echo "[TEST] Step 3 — Checking images are readable..."
python3 -c "
import os
from PIL import Image
import pandas as pd

df = pd.read_csv('./Data/apod_preloaded_dataset.csv')
train = df[df.split == 'train'].head(5)

images_dir = './Data/images'
found = 0
for _, row in train.iterrows():
    img_id = str(row['img_index'])
    for ext in ['.jpg', '.png', '.jpeg']:
        path = f'{images_dir}/{img_id}{ext}'
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            print(f'[OK] Loaded image {img_id}{ext} — size: {img.size}')
            found += 1
            break
    else:
        print(f'[WARNING] Image not found for img_index={img_id}')

print(f'[OK] {found}/5 sample images readable')
"

# ── Step 4: Check model initializes ──────────────────────────────────────────
echo ""
echo "[TEST] Step 4 — Checking model initializes with tiny settings..."
python3 -c "
import torch
from models.llm_encoder import TextEncoder
from models.flow_model import GlowModel

device = torch.device('cpu')
print('[INFO] Using CPU for test')

# Tiny settings for fast test
image_size = 64
cond_dim   = 128
n_levels   = 2
n_steps    = 2

text_enc = TextEncoder(out_dim=cond_dim).to(device)
print(f'[OK] TextEncoder initialized')

flow = GlowModel(
    img_shape=(3, image_size, image_size),
    cond_dim=cond_dim,
    n_levels=n_levels,
    n_steps=n_steps,
).to(device)
print(f'[OK] GlowModel initialized')
"

# ── Step 5: Run full mini training loop ──────────────────────────────────────
echo ""
echo "[TEST] Step 5 — Running mini training loop (10 samples, 2 epochs)..."
echo "[TEST] This is the real test — if this passes, HTCondor job will work"
echo ""

python3 train.py \
    --data_csv       ./Data/apod_preloaded_dataset.csv \
    --images_dir     ./Data/images \
    --checkpoint_dir ./test_checkpoints \
    --samples        10 \
    --epochs         2 \
    --batch_size     2 \
    --image_size     64 \
    --n_levels       2 \
    --n_steps        2 \
    --cond_dim       128 \
    --lr             1e-4 \
    --device         cpu

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================================"
    echo "[TEST] ✅ ALL TESTS PASSED — Safe to submit to HTCondor!"
    echo "[TEST] Checkpoints saved to ./test_checkpoints/"
    ls ./test_checkpoints/
    echo "============================================================"
else
    echo "============================================================"
    echo "[TEST] ❌ TEST FAILED — Fix errors above before submitting"
    echo "============================================================"
fi

exit $EXIT_CODE
