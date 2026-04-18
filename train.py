from Data.dataset import AstroDataset
from models.llm_encoder import TextEncoder
from models.flow_model import GlowModel

from tqdm import tqdm
import pandas as pd
import os
import torch
import argparse
from torch.utils.data import DataLoader
from time import time as tick
import datetime


def filter_missing_images(df, images_dir):
    """Remove rows where the image file does not exist on disk."""
    print(f"[INFO] Checking which images exist in {images_dir}...", flush=True)
    valid_indices = []
    for _, row in df.iterrows():
        img_id = str(int(row['img_index']))
        exists = any(
            os.path.exists(f"{images_dir}/{img_id}{ext}")
            for ext in ['.jpg', '.png', '.jpeg']
        )
        if exists:
            valid_indices.append(row.name)

    df_filtered = df.loc[valid_indices].reset_index(drop=True)
    removed = len(df) - len(df_filtered)
    print(f"[INFO] Images found:   {len(df_filtered)}", flush=True)
    print(f"[INFO] Images missing: {removed} (skipped)", flush=True)
    return df_filtered


def train(
    df,
    samples=3000,
    epochs=100,
    batch_size=32,
    lr=1e-5,
    device="cuda",
    cond_dim=128,
    n_levels=3,
    n_steps=8,
    images_dir="images",
    image_size=96,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}", flush=True)

    df_sub = df.sample(min(samples, len(df)), ignore_index=True, random_state=42)
    print(f"[INFO] Sampled {len(df_sub)} rows from dataset", flush=True)

    dataset = AstroDataset(df_sub, images_dir=images_dir, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"[INFO] DataLoader created — {len(loader)} batches per epoch", flush=True)

    text_enc = TextEncoder(out_dim=cond_dim).to(device)
    flow = GlowModel(
        img_shape=(3, image_size, image_size),
        cond_dim=cond_dim,
        n_levels=n_levels,
        n_steps=n_steps,
    ).to(device)
    print(f"[INFO] Models initialised (TextEncoder + GlowModel)", flush=True)

    optimizer = torch.optim.Adam(
        list(flow.parameters()) + list(text_enc.proj.parameters()), lr=lr
    )

    print(f"[INFO] Starting training: {epochs} epochs, batch_size={batch_size}, lr={lr}", flush=True)

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        flow.train()

        for batch_idx, (images, texts) in enumerate(loader):
            images = images.to(device)
            images = images + torch.rand_like(images) / 256.0
            texts = list(texts)

            context = text_enc(texts)
            loss = -flow.log_prob(images, context).mean()

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx == 0 and epoch % 10 == 0:
                print(
                    f"[DEBUG] Epoch {epoch+1} | Batch 0 | "
                    f"images.shape={images.shape} | "
                    f"context.shape={context.shape} | "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        avg_loss = total_loss / len(loader)
        print(f"[TRAIN] Epoch {epoch+1}/{epochs} — avg_loss={avg_loss:.4f}", flush=True)

    print("[INFO] Training loop complete", flush=True)
    return flow, text_enc


def save_model(path, flow, llm, flow_model_name='glow-ckpt.pth', llm_model_name='llm-ckpt.pth'):
    os.makedirs(path, exist_ok=True)
    flow_path = os.path.join(path, flow_model_name)
    llm_path = os.path.join(path, llm_model_name)

    torch.save(flow.state_dict(), flow_path)
    print(f"[INFO] Flow model saved → {flow_path}", flush=True)

    torch.save(llm.state_dict(), llm_path)
    print(f"[INFO] LLM encoder saved → {llm_path}", flush=True)


def load_glow_models(path,
                     flow_model_name='glow-ckpt.pth',
                     llm_model_name='llm-ckpt.pth',
                     img_shape=(3, 96, 96),
                     cond_dim=128,
                     n_levels=3,
                     n_steps=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow = GlowModel(img_shape=img_shape, cond_dim=cond_dim, n_levels=n_levels, n_steps=n_steps)
    llm = TextEncoder(out_dim=cond_dim)

    flow_path = os.path.join(path, flow_model_name)
    llm_path = os.path.join(path, llm_model_name)

    if os.path.exists(flow_path):
        flow.load_state_dict(torch.load(flow_path, map_location=device))
        print(f"[INFO] Loaded Glow weights from {flow_path}", flush=True)
    else:
        print(f"[WARNING] Glow checkpoint not found at {flow_path}", flush=True)

    if os.path.exists(llm_path):
        llm.load_state_dict(torch.load(llm_path, map_location=device))
        print(f"[INFO] Loaded LLM weights from {llm_path}", flush=True)
    else:
        print(f"[WARNING] LLM checkpoint not found at {llm_path}", flush=True)

    flow.to(device).eval()
    llm.to(device).eval()
    return flow, llm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train text-conditioned Glow model.")
    parser.add_argument("--data_csv",        default="./Data/apod_preloaded_dataset.csv")
    parser.add_argument("--images_dir",      default="images")
    parser.add_argument("--checkpoint_dir",  default="./checkpoints")
    parser.add_argument("--samples",         type=int,   default=1000)
    parser.add_argument("--epochs",          type=int,   default=50)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-5)
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--cond_dim",        type=int,   default=256)
    parser.add_argument("--n_levels",        type=int,   default=4)
    parser.add_argument("--n_steps",         type=int,   default=8)
    parser.add_argument("--image_size",      type=int,   default=96)
    parser.add_argument("--flow_ckpt_name",  default="glow-ckpt.pth")
    parser.add_argument("--llm_ckpt_name",   default="llm-ckpt.pth")
    args = parser.parse_args()

    # ── Startup diagnostics ───────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("[INFO] Job started", flush=True)
    print(f"[INFO] Hostname     : {os.uname().nodename}", flush=True)
    print(f"[INFO] Working dir  : {os.getcwd()}", flush=True)
    print(f"[INFO] Arguments    : {vars(args)}", flush=True)
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[INFO] GPU          : {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[INFO] VRAM total   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)
    else:
        print("[WARNING] No GPU found — falling back to CPU. Training will be very slow.", flush=True)
    print("=" * 60, flush=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading CSV: {args.data_csv}", flush=True)
    if not os.path.exists(args.data_csv):
        raise FileNotFoundError(f"[ERROR] CSV not found: {args.data_csv}")
    nasa = pd.read_csv(args.data_csv)
    print(f"[INFO] CSV loaded — shape: {nasa.shape}", flush=True)

    if 'split' not in nasa.columns:
        raise ValueError("[ERROR] CSV has no 'split' column.")
    nasa_train = nasa[nasa.split == 'train']
    print(f"[INFO] Training rows before filtering: {len(nasa_train)}", flush=True)

    if not os.path.isdir(args.images_dir):
        raise FileNotFoundError(f"[ERROR] Images directory not found: {args.images_dir}")

    # ── Filter out missing images ─────────────────────────────────────────────
    nasa_train = filter_missing_images(nasa_train, args.images_dir)
    print(f"[INFO] Training rows after filtering: {len(nasa_train)}", flush=True)

    if len(nasa_train) == 0:
        raise RuntimeError("[ERROR] No valid training images found after filtering!")

    # ── Train ────────────────────────────────────────────────────────────────
    t0 = tick()
    flow, text_enc = train(
        nasa_train,
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        cond_dim=args.cond_dim,
        n_levels=args.n_levels,
        n_steps=args.n_steps,
        images_dir=args.images_dir,
        image_size=args.image_size,
    )
    elapsed = datetime.timedelta(seconds=int(tick() - t0))
    print(f"[INFO] Total train time: {elapsed}", flush=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    save_model(
        args.checkpoint_dir,
        flow=flow,
        llm=text_enc,
        flow_model_name=args.flow_ckpt_name,
        llm_model_name=args.llm_ckpt_name,
    )
    print("[INFO] Job finished successfully", flush=True)
