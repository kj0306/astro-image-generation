import argparse
import os
from io import BytesIO

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download APOD images locally by img_index for preloaded training."
    )
    parser.add_argument("--csv_path", default="./Data/apod_preloaded_dataset.csv")
    parser.add_argument("--output_dir", default="./images")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to download",
    )
    parser.add_argument("--url_col", default="best_url")
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--retries", type=int, default=2)
    return parser.parse_args()


def download_and_convert_png(url, out_path, timeout, retries):
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img.save(out_path, format="PNG")
            return True
        except Exception:
            if attempt == retries:
                return False
    return False


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    df = df[df["split"].isin(args.splits)].dropna(subset=[args.url_col, "img_index"])

    ok = 0
    skipped = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_id = str(row["img_index"])
        url = row[args.url_col]
        out_path = os.path.join(args.output_dir, f"{image_id}.png")

        if os.path.exists(out_path):
            skipped += 1
            continue

        success = download_and_convert_png(
            url=url, out_path=out_path, timeout=args.timeout, retries=args.retries
        )
        if success:
            ok += 1
        else:
            failed += 1

    print("Download complete.")
    print(f"Downloaded: {ok}")
    print(f"Skipped (already existed): {skipped}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
