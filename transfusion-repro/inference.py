import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from common import build_model, load_checkpoint, save_json, seed_everything


def image_tensors(sample):
    for item in sample:
        if isinstance(item, tuple) and len(item) == 2:
            _, image = item
            if isinstance(image, torch.Tensor) and image.ndim in (3, 4):
                yield image


def generate(args):
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.ckpt, device)
    if checkpoint["config"]["smoke_test"]:
        raise ValueError("A synthetic smoke-test checkpoint has no image decoder")
    model, tokenizer = build_model(checkpoint["config"], device)
    model.load_state_dict(checkpoint["model"])
    checkpoint = None
    model.eval()
    seed_everything(args.seed)
    prompt = torch.tensor(tokenizer.encode(args.prompt, add_special_tokens=False), dtype=torch.long, device=device)
    if args.force_image:
        prompt = torch.cat([prompt, torch.tensor([model.som_ids[0]], device=device)])
    with torch.no_grad():
        output = model.sample(prompt=prompt, max_length=args.max_length, modality_steps=args.modality_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for count, image in enumerate(image_tensors(output), start=1):
        save_image(image.detach().cpu(), args.output_dir / f"sample_{count}.png")
    text = [item[item < len(tokenizer)].cpu().tolist() for item in output
            if isinstance(item, torch.Tensor) and item.dtype == torch.long]
    save_json(args.output_dir / "generation.json", {"prompt": args.prompt, "seed": args.seed,
              "forced_image_start": args.force_image, "images_saved": count,
              "text": [tokenizer.decode(ids) for ids in text]})
    print(f"Images saved: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--prompt", default="A cat sitting on a chair")
    parser.add_argument("--output-dir", type=Path, default=Path("results/samples"))
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--modality-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-image", action="store_true")
    generate(parser.parse_args())
