import argparse
import os
from pathlib import Path
import glob
import re

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import torchvision.transforms as T
from torchvision.utils import save_image
from torchvision.datasets import CocoCaptions

from diffusers.models import AutoencoderKL
from transformers import AutoTokenizer
from accelerate import Accelerator

from transfusion_pytorch import Transfusion, print_modality_sample



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", type=str, required=True, help="COCO root path")
    p.add_argument("--coco-ann", type=str, default="annotations/captions_train2017.json")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--results-dir", type=str, default="./results_coco_ddp")

    p.add_argument("--flow-loss-weight", type=float, default=1.0,
                   help="weight λ for image (flow) loss")

    p.add_argument("--resume-step", type=int, default=0,
                   help="step to resume from (0 = start fresh)")
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
            max_length=128,
        )
        text_tokens = enc["input_ids"].squeeze(0)

        return text_tokens, image


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def find_latest_checkpoint(results_dir: Path):
    pattern = str(results_dir / "model_step_*.pt")
    ckpts = glob.glob(pattern)
    if not ckpts:
        return 0, None

    step_re = re.compile(r"model_step_(\d+)\.pt")
    steps = []
    for p in ckpts:
        m = step_re.search(os.path.basename(p))
        if m:
            steps.append((int(m.group(1)), p))

    if not steps:
        return 0, None

    steps.sort()
    last_step, last_path = steps[-1]
    return last_step, last_path


def main():
    args = parse_args()
    accelerator = Accelerator()

    results_folder = Path(args.results_dir)
    results_folder.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(f"[Info] results_dir = {results_folder}")

    # VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    vae.requires_grad_(False)
    vae.eval()
    vae.to(accelerator.device)

    encoder = Encoder(vae)
    decoder = Decoder(vae)

    # Transfusion 모델
    model = Transfusion(
        num_text_tokens = tokenizer.vocab_size,
        dim_latent = 4,
        channel_first_latent = True,
        modality_default_shape = (32, 32),
        modality_encoder = encoder,
        modality_decoder = decoder,
        pre_post_transformer_enc_dec = (
            nn.Conv2d(4, 768, 3, 2, 1),
            nn.ConvTranspose2d(768, 4, 3, 2, 1, output_padding=1),
        ),
        add_pos_emb = False,
        modality_num_dim = 2,
        fallback_to_default_shape_if_invalid = True,
        transformer = dict(
            dim = 768,
            depth = 12,
            dim_head = 64,
            heads = 12,
        ),

        flow_loss_weight = args.flow_loss_weight,
        text_loss_weight = 1.0,
    )


    dataset = CocoTextImageDataset(
        coco_root=args.coco_root,
        ann_rel_path=args.coco_ann,
        image_size=args.image_size,
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

    iter_dl = cycle(dataloader)

    # EMA
    ema_model = None
    if accelerator.is_main_process:
        try:
            ema_model = accelerator.unwrap_model(model).create_ema(0.9)
        except AttributeError:
            print("Warning: create_ema not found. Skipping EMA.")
            ema_model = None


    current_step, resume_path = find_latest_checkpoint(results_folder)

    if resume_path is not None and current_step > 0:
        if accelerator.is_main_process:
            print(f"[Resume] Found checkpoint {resume_path} (step={current_step})")
        state_dict = torch.load(resume_path, map_location="cpu")
        accelerator.unwrap_model(model).load_state_dict(state_dict)
    else:
        if accelerator.is_main_process:
            print("[Resume] No checkpoint found. Starting from scratch.")
        current_step = 0

    model.train()


    max_steps = args.max_steps  

    try:
        while True:
            current_step += 1

            if max_steps > 0 and current_step > max_steps:
                if accelerator.is_main_process:
                    print(f"[Info] Reached max_steps={max_steps}. Stopping training.")
                break

            text_tokens, images = next(iter_dl)
            loss = model(text_tokens, images)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if accelerator.is_main_process:
                if ema_model:
                    ema_model.update()

                if current_step % 10 == 0:
                    print(f"Step {current_step}: Loss = {loss.item():.4f}")


                if current_step % 500 == 0:
                    print(f"Generating Sample at step {current_step}...")
                    eval_model = ema_model if ema_model else accelerator.unwrap_model(model)
                    eval_model.eval()
                    with torch.no_grad():
                        try:
                            sample = eval_model.sample()
                            cnt = 0
                            for item in sample:
                                if isinstance(item, torch.Tensor) and item.ndim == 4:
                                    save_path = results_folder / f"{current_step}_sample_{cnt}.png"
                                    save_image(item, str(save_path))
                                    cnt += 1
                            if cnt == 0:
                                print("Warning: No image tensor found in sample().")
                        except Exception as e:
                            print(f"Sampling Failed: {e}")
                    model.train()


                if current_step % 5000 == 0:
                    ckpt_path = results_folder / f"model_step_{current_step}.pt"
                    accelerator.save(
                        accelerator.unwrap_model(model).state_dict(), ckpt_path
                    )
                    print(f"[Save] Model saved to {ckpt_path}")

    except KeyboardInterrupt:
        if accelerator.is_main_process:
            print(f"\n[Interrupted] Training stopped by user at step {current_step}.")
            ckpt_path = results_folder / f"model_step_{current_step}.pt"
            accelerator.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
            print(f"[Save] Final model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
