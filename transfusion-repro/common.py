import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import CocoCaptions


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Encoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        self.vae.eval()
        with torch.no_grad():
            return self.vae.encode(image * 2 - 1).latent_dist.sample() * self.vae.config.scaling_factor


class Decoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent):
        self.vae.eval()
        with torch.no_grad():
            decoded = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        return (decoded / 2 + 0.5).clamp(0, 1)


class CocoTextImageDataset:
    def __init__(self, root, split, tokenizer, image_size=256, max_text_length=128, limit=None):
        root = Path(root)
        annotation = root / "annotations" / f"captions_{split}2017.json"
        image_folder = root / f"{split}2017"
        if not annotation.is_file() or not image_folder.is_dir():
            raise FileNotFoundError(f"Expected {annotation} and {image_folder}; see data/README.md")
        self.dataset = CocoCaptions(str(image_folder), str(annotation))
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.length = min(len(self.dataset), limit) if limit is not None else len(self.dataset)
        if self.length < 1:
            raise ValueError("Dataset is empty")
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image, captions = self.dataset[index]
        tokens = self.tokenizer.encode(captions[0], add_special_tokens=False,
                                       truncation=True, max_length=self.max_text_length)
        return torch.tensor(tokens, dtype=torch.long), self.transform(image.convert("RGB"))


def multimodal_batch(samples):
    return [[tokens, (0, image)] for tokens, image in samples]


def move_batch(batch, device):
    return [[tokens.to(device), (kind, image.to(device))] for tokens, (kind, image) in batch]


def build_model(config, device):
    from diffusers import AutoencoderKL
    from transformers import AutoTokenizer
    from transfusion_pytorch import Transfusion

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
    vae = AutoencoderKL.from_pretrained(config["vae"]).to(device).eval()
    vae.requires_grad_(False)
    dim = config["dim"]
    model = Transfusion(
        num_text_tokens=len(tokenizer), dim_latent=4, channel_first_latent=True,
        modality_default_shape=(config["image_size"] // 8,) * 2, modality_num_dim=2,
        modality_encoder=Encoder(vae), modality_decoder=Decoder(vae),
        pre_post_transformer_enc_dec=(nn.Conv2d(4, dim, 3, 2, 1),
                                      nn.ConvTranspose2d(dim, 4, 3, 2, 1, output_padding=1)),
        fallback_to_default_shape_if_invalid=True,
        transformer=dict(dim=dim, depth=config["depth"], heads=config["heads"], dim_head=dim // config["heads"]),
        flow_loss_weight=config["flow_loss_weight"], text_loss_weight=config["text_loss_weight"],
    ).to(device)
    return model, tokenizer


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or not {"model", "config", "optimizer", "step"}.issubset(checkpoint):
        raise ValueError("Use a checkpoint written by the current main.py; legacy state-dict files have no matching run configuration")
    return checkpoint


def plot_losses(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    figure, axis = plt.subplots(figsize=(8, 4))
    for field in ("total", "text", "flow"):
        axis.plot([int(row["step"]) for row in rows], [float(row[field]) for row in rows], label=field)
    axis.set(xlabel="Training step", ylabel="Loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(Path(path).with_name("loss_curve.png"), dpi=150)
    plt.close(figure)


def save_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2), encoding="utf-8")
