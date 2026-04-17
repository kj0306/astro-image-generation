import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AstroDataset(Dataset):
    def __init__(self, df, transform=None, images_dir="images", image_size=96):
        self.df = df.reset_index(drop=True)
        self.url_col = 'hdurl'
        self.text_col = 'explanation'
        self.image_index_col = 'img_index'
        self.images_dir = images_dir
        self.image_size = image_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def _fetch_image(self, url):
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        if hasattr(img, 'n_frames') and img.n_frames > 1:  # handle GIFs
            img.seek(0)
        return img.convert("RGB")
    
    def _fetch_preloaded_image(self, image_id):
        '''
        This function looks for images stored inside ./images/<img_index>.png, replacing the _fetch_image() function which downloads from URLs.
        '''
        image_id = str(image_id)
        candidate_paths = [
            f"{self.images_dir}/{image_id}.png",
            f"{self.images_dir}/{image_id}.jpg",
            f"{self.images_dir}/{image_id}.jpeg",
        ]
        for path in candidate_paths:
            try:
                img = Image.open(path)
                return img.convert('RGB')
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            f"Could not find preloaded image {image_id} in {self.images_dir} "
            "(looked for .png/.jpg/.jpeg)."
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        url = row[self.url_col]
        text = row[self.text_col]
        image_id = row[self.image_index_col]
        image = self._fetch_preloaded_image(image_id) # self._fetch_image(url)
        image = self.transform(image)
        return image, text