# eval_coco_val.py
import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.datasets import CocoCaptions

from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer

from transfusion_pytorch import Transfusion


# -------------------------------------------------
# Argument parser
# -------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--coco-root",
        type=str,
        required=True,
        help="COCO root path (e.g., ./data/coco)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Which COCO split to evaluate on (train or val). "
             "If you don't have val2017 images, keep 'train'.",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=256,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    p.add_argument(
        "--ckpt",
        type=str,
        default="results_coco_single/model_step_5000.pt",
        help="Path to trained model checkpoint",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default="./results_coco_single",
        help="Where to save eval logs (val_loss.txt 등)",
    )
    return p.parse_args()


# -------------------------------------------------
# Tokenizer (TinyLlama – same as training)
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# -------------------------------------------------
# VAE encoder/decoder wrappers (same as training)
# -------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        # image: [0,1] -> [-1, 1]
        with torch.no_grad():
            latent = self.vae.encode(image * 2 - 1).latent_dist.sample()
        return 0.18215 * latent


class Decoder(nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        latents = (1 / 0.18215) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        return (image / 2 + 0.5).clamp(0, 1)


# -------------------------------------------------
# COCO text–image dataset (train/val 공용)
# -------------------------------------------------
class CocoTextImageDataset(Dataset):
    def __init__(self, coco_root, split="train", image_size=256):
        import os

        assert split in ["train", "val"]
        img_dir = "train2017" if split == "train" else "val2017"
        ann_file = f"captions_{split}2017.json"

        image_root = os.path.join(coco_root, img_dir)
        ann_path = os.path.join(coco_root, "annotations", ann_file)

        self.ds = CocoCaptions(root=image_root, annFile=ann_path)
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, captions = self.ds[idx]
        caption = captions[0]

        image = image.convert("RGB")
        image = self.transform(image)

        enc = tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        text_tokens = enc["input_ids"].squeeze(0)

        return text_tokens, image


# -------------------------------------------------
# Main eval loop
# -------------------------------------------------
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using device: {device}")

    # 1) VAE (동결 + eval 모드)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.requires_grad_(False)
    vae.eval()
    vae.to(device)

    encoder = Encoder(vae)
    decoder = Decoder(vae)

    # 2) Transfusion model (train 때와 아키텍처 동일해야 함)
    model = Transfusion(
        num_text_tokens=tokenizer.vocab_size,
        dim_latent=4,
        channel_first_latent=True,
        modality_default_shape=(32, 32),  # 256 / 8
        modality_encoder=encoder,
        modality_decoder=decoder,
        pre_post_transformer_enc_dec=(
            nn.Conv2d(4, 768, 3, 2, 1),
            nn.ConvTranspose2d(768, 4, 3, 2, 1, output_padding=1),
        ),
        add_pos_emb=False,
        modality_num_dim=2,
        fallback_to_default_shape_if_invalid=True,
        transformer=dict(
            dim=768,
            depth=12,
            dim_head=64,
            heads=12,
        ),
    ).to(device)

    # 3) Checkpoint load
    print(f"[Eval] Loading checkpoint from: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    # train 스크립트에서 model.state_dict()만 저장했다고 가정
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    # 4) Dataset & DataLoader
    dataset = CocoTextImageDataset(
        coco_root=args.coco_root,
        split=args.split,
        image_size=args.image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 5) Eval loop
    total_loss = 0.0
    total_cnt = 0

    print(f"[Eval] Dataset size: {len(dataset)} images, split={args.split}")

    with torch.no_grad():
        for step, (text_tokens, images) in enumerate(loader):
            text_tokens = text_tokens.to(device)
            images = images.to(device)

            loss = model(text_tokens, images)
            bs = images.size(0)

            total_loss += loss.item() * bs
            total_cnt += bs

            if (step + 1) % 20 == 0:
                print(
                    f"[Eval] Batch {step+1}/{len(loader)} "
                    f"loss={loss.item():.4f}"
                )

    avg_loss = total_loss / max(total_cnt, 1)
    print(f"[Eval] Avg loss on {args.split} set (N={total_cnt}): {avg_loss:.4f}")

    # 6) Save result
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    out_path = results_dir / f"val_loss_{args.split}.txt"
    with open(out_path, "w") as f:
        f.write(f"{avg_loss:.6f}\n")

    print(f"[Eval] Saved avg loss to: {out_path}")


if __name__ == "__main__":
    main()
