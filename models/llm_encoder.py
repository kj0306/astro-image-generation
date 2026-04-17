from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", out_dim=128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder   = AutoModel.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(768, out_dim)

    def forward(self, texts):
        tokens = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=64, return_tensors="pt"
        ).to(next(self.proj.parameters()).device)
        with torch.no_grad():
            out = self.encoder(**tokens).last_hidden_state[:, 0] # cls token
        return self.proj(out)        