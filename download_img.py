import pandas as pd
import requests
import os
from tqdm import tqdm

# read input
df = pd.read_csv("./Data/apod_clean_urls.csv")

# only download test
df = df[df["split"] == "test"].dropna(subset=["best_url"])

# save
save_dir = "./Data/images"
os.makedirs(save_dir, exist_ok=True)

def download_image(idx, url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        with open(f"{save_dir}/{idx}.jpg", "wb") as f:
            f.write(response.content)

    except Exception as e:
        print(f"❌ failed: {idx}")

# download
for idx, row in tqdm(df.iterrows(), total=len(df)):
    download_image(idx, row["best_url"])

print("✅ Done downloading!")