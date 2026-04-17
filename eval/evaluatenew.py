import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from Data.dataset import AstroDataset
from models.llm_encoder import TextEncoder
from models.flow_model import ConditionalFlow

# ===== metrics =====
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import clip
from PIL import Image


# ===============================
# load model
# ===============================
def load_model(flow_path, llm_path, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    text_enc = TextEncoder(out_dim=128).to(device)
    flow = ConditionalFlow().to(device)

    flow.load_state_dict(torch.load(flow_path, map_location=device))
    text_enc.load_state_dict(torch.load(llm_path, map_location=device))

    flow.eval()
    text_enc.eval()

    return flow, text_enc


# ===============================
# CLIP score
# ===============================
def compute_clip_score(images, texts, model, preprocess, device="cuda"):
    scores = []

    for img, text in zip(images, texts):
        # 🔥 truncate text (avoid 77 token error)
        text = text[:200]

        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype("uint8")

        img = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
        text_token = clip.tokenize([text], truncate=True).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(img)
            txt_feat = model.encode_text(text_token)

        sim = torch.cosine_similarity(img_feat, txt_feat).item()
        scores.append(sim)

    return sum(scores) / len(scores)


# ===============================
# evaluation
def evaluatenew(df, flow, text_enc, batch_size=16, device="cuda", use_zero_context=False):
    
    device = torch.device(device)


    dataset = AstroDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=50).to(device)

    # 🔥 load CLIP once
    model_clip, preprocess = clip.load("ViT-B/32", device=device)

    total_ll = 0
    clip_scores = []

    #for i, (images, _) in enumerate(tqdm(loader)):
        # ===== move to GPU =====
        #images = images.to(device)

        ## ===== use TITLE instead of explanation =====
        #batch_df = df.iloc[i * batch_size:(i + 1) * batch_size]
        #texts = batch_df["title"].fillna("").tolist()
    for images, texts in tqdm(loader):
        images = images.to(device)
        texts = list(texts)

        # ===== context =====
        context = text_enc(texts).to(device)
        
        if use_zero_context:
            #texts = [""]*len(texts)
            texts = ["galaxy"] * len(texts)
            context = text_enc(texts).to(device)



        # ===== log likelihood =====
        ll = flow.log_prob(images, context).mean()
        total_ll += ll.item()

        # ===== sampling =====
        fake = flow.sample(len(texts), context)

        # ===== convert to uint8 for FID/KID =====
        images_uint8 = (images * 255).clamp(0, 255).to(torch.uint8)
        fake_uint8 = (fake * 255).clamp(0, 255).to(torch.uint8)

        # ===== FID / KID =====
        fid.update(images_uint8, real=True)
        fid.update(fake_uint8, real=False)

        kid.update(images_uint8, real=True)
        kid.update(fake_uint8, real=False)

        # ===== CLIP =====
        clip_score = compute_clip_score(
            fake, texts, model_clip, preprocess, device
        )
        clip_scores.append(clip_score)

    fid_score = fid.compute()
    kid_score = kid.compute()[0]
    clip_score = sum(clip_scores) / len(clip_scores)
    avg_ll = total_ll / len(loader)

    print("\n===== Evaluation =====")
    print(f"FID: {fid_score:.4f}")
    print(f"KID: {kid_score:.4f}")
    print(f"CLIP: {clip_score:.4f}")
    print(f"Log-Likelihood: {avg_ll:.4f}")

    return fid_score, kid_score, clip_score, avg_ll


# ===============================
# run
# ===============================
if __name__ == "__main__":
    device = "cuda"

    df = pd.read_csv("./Data/apod_clean_urls.csv")

    # 🔥 use TEST set (important)
    df_test = df[df["split"] == "test"].dropna(subset=["hdurl", "title"])

    flow, text_enc = load_model(
        "./checkpoints/flow-ckpt.pth",
        "./checkpoints/llm-ckpt.pth",
        device=device
    )

    evaluate(df_test, flow, text_enc, device=device)