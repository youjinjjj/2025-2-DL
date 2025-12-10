#!/usr/bin/env python
import argparse
import os
from pathlib import Path
from shutil import rmtree

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import torchvision.transforms as T
from torchvision.datasets import CocoCaptions
from torchvision.utils import save_image

from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer
from accelerate import Accelerator

from transfusion_pytorch import Transfusion



def parse_args():
    p = argparse.ArgumentParser(description="Transfusion COCO training (fresh + resume)")

    p.add_argument("--coco-root", type=str, required=True,
                   help="COCO root path (e.g., ./data/coco)")
    p.add_argument("--coco-ann", type=str,
                   default="annotations/captions_train2017.json",
                   help="Relative annotations path from coco-root")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=5000,
                   help="Total training steps. <=0 means run until manually stopped.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--results-dir", type=str, default="./results_coco_single")

    # 논문 λ 실험용
    p.add_argument("--flow-loss-weight", type=float, default=1.0,
                   help="λ: weight for image(flow) loss")
    p.add_argument("--text-loss-weight", type=float, default=1.0,
                   help="weight for LM(text) loss")

    # Resume 관련
    p.add_argument("--resume-step", type=int, default=0,
                   help="Resume from model_step_{resume_step}.pt (0 = start from scratch)")
    p.add_argument("--sample-every", type=int, default=500,
                   help="Generate sample every N steps")
    p.add_argument("--save-every", type=int, default=5000,
                   help="Save checkpoint every N steps")

    return p.parse_args()



tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


class Encoder(nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.vae = vae

    def forward(self, image):
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



class CocoTextImageDataset(Dataset):
    def __init__(self, coco_root, ann_rel_path, image_size=256):
        image_root = os.path.join(coco_root, "train2017")
        ann_path = os.path.join(coco_root, ann_rel_path)

        self.ds = CocoCaptions(root=image_root, annFile=ann_path)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

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
            max_length=128
        )
        text_tokens = enc["input_ids"].squeeze(0)

        return text_tokens, image


def cycle(dl):
    while True:
        for batch in dl:
            yield batch



def main():
    args = parse_args()
    accelerator = Accelerator()

    results_dir = Path(args.results_dir)


    if accelerator.is_main_process and args.resume_step == 0:
        rmtree(results_dir, ignore_errors=True)
    if accelerator.is_main_process:
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Train] Results dir: {results_dir}")
        print(f"[Train] flow_loss_weight = {args.flow_loss_weight}, text_loss_weight = {args.text_loss_weight}")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device)

    encoder = Encoder(vae)
    decoder = Decoder(vae)

    # --- Transfusion model ---
    model = Transfusion(
        num_text_tokens=tokenizer.vocab_size,
        dim_latent=4,
        channel_first_latent=True,
        modality_default_shape=(32, 32),
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
    )

    dataset = CocoTextImageDataset(
        coco_root=args.coco_root,
        ann_rel_path=args.coco_ann,
        image_size=args.image_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    ema_model = None
    if accelerator.is_main_process and hasattr(accelerator.unwrap_model(model), "create_ema"):
        ema_model = accelerator.unwrap_model(model).create_ema(0.9)

    start_step = 0
    if args.resume_step > 0:
        ckpt = results_dir / f"model_step_{args.resume_step}.pt"
        if ckpt.exists():
            print(f"[Train] Resuming from {ckpt}")
            state = torch.load(str(ckpt), map_location="cpu")
            accelerator.unwrap_model(model).load_state_dict(state)
            start_step = args.resume_step
        else:
            print(f"[Train] WARNING: checkpoint {ckpt} not found, starting from scratch")

    model.train()
    data_iter = cycle(dataloader)

    step = start_step
    while True:
        step += 1
        if args.max_steps > 0 and step > args.max_steps:
            if accelerator.is_main_process:
                print(f"[Train] Reached max_steps={args.max_steps}, stopping.")
            break

        text_tokens, images = next(data_iter)
        loss = model(text_tokens, images)

        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if accelerator.is_main_process:
            if ema_model is not None:
                ema_model.update()

            if step % 10 == 0:
                print(f"[Train] Step {step}: Loss = {loss.item():.4f}")

            if step % args.sample_every == 0:
                eval_model = ema_model if ema_model is not None else accelerator.unwrap_model(model)
                eval_model.eval()
                try:
                    with torch.no_grad():
                        sample = eval_model.sample()
                    cnt = 0
                    for item in sample:
                        if isinstance(item, torch.Tensor) and item.ndim == 4:
                            save_path = results_dir / f"{step}_sample_{cnt}.png"
                            save_image(item, str(save_path))
                            cnt += 1
                    if cnt == 0:
                        print("[Train] Warning: no image tensor in sample() output.")
                except Exception as e:
                    print(f"[Train] Sampling failed at step {step}: {e}")
                finally:
                    model.train()

            if step % args.save_every == 0:
                ckpt_path = results_dir / f"model_step_{step}.pt"
                accelerator.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
                print(f"[Train] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
