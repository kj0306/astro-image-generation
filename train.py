from Data.dataset import AstroDataset
from models.llm_encoder import TextEncoder
# Assuming you saved the Glow architecture in glow_model.py
from models.flow_model import GlowModel 
from tqdm import tqdm
import pandas as pd
import os
import torch
import argparse
from torch.utils.data import DataLoader
from time import time as tick
import datetime

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
    
    df_sub = df.sample(min(samples, len(df)), ignore_index=True, random_state=42)
    dataset = AstroDataset(df_sub, images_dir=images_dir, image_size=image_size)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    text_enc = TextEncoder(out_dim=cond_dim).to(device)
    flow = GlowModel(
        img_shape=(3, image_size, image_size),
        cond_dim=cond_dim,
        n_levels=n_levels,
        n_steps=n_steps,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(flow.parameters()) + list(text_enc.proj.parameters()), lr=lr
    )

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        flow.train()
        
        for images, texts in loader:
            images = images.to(device)
            images = images + torch.rand_like(images) / 256.0
            
            texts = list(texts)           
            context = text_enc(texts)     # [B, 128]

            loss = -flow.log_prob(images, context).mean()

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}")

    return flow, text_enc

def save_model(path, flow, llm, flow_model_name='glow-ckpt.pth', llm_model_name='llm-ckpt.pth'):
    os.makedirs(path, exist_ok=True)
    flow_path = os.path.join(path, flow_model_name)
    llm_path = os.path.join(path, llm_model_name)

    torch.save(flow.state_dict(), flow_path)
    torch.save(llm.state_dict(), llm_path)
    print(f"Models saved to {path}")

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
        print("Loaded Glow weights.")
    
    if os.path.exists(llm_path):
        llm.load_state_dict(torch.load(llm_path, map_location=device))
        print("Loaded LLM weights.")
    
    flow.to(device).eval()
    llm.to(device).eval()
    return flow, llm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train text-conditioned Glow model.")
    parser.add_argument("--data_csv", default="./Data/apod_preloaded_dataset.csv")
    parser.add_argument("--images_dir", default="images")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cond_dim", type=int, default=256)
    parser.add_argument("--n_levels", type=int, default=4)
    parser.add_argument("--n_steps", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--flow_ckpt_name", default="glow-ckpt.pth")
    parser.add_argument("--llm_ckpt_name", default="llm-ckpt.pth")
    args = parser.parse_args()

    nasa = pd.read_csv(args.data_csv)
    nasa_train = nasa[nasa.split == 'train']
    
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
    
    print(f'Train time: {datetime.timedelta(seconds=int(tick() - t0))}')
    
    checkpoint_dir = args.checkpoint_dir
    save_model(
        checkpoint_dir,
        flow=flow,
        llm=text_enc,
        flow_model_name=args.flow_ckpt_name,
        llm_model_name=args.llm_ckpt_name,
    )