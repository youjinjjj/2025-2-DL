import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import CocoTextImageDataset, build_model, load_checkpoint, move_batch, multimodal_batch, save_json, seed_everything


def evaluate(args):
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.ckpt, device)
    config = checkpoint["config"]
    if config["smoke_test"]:
        raise ValueError("A synthetic smoke-test checkpoint cannot be evaluated on COCO")
    model, tokenizer = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    checkpoint_step = checkpoint["step"]
    checkpoint = None
    model.eval()
    dataset = CocoTextImageDataset(args.coco_root, args.split, tokenizer, config["image_size"],
                                  config["max_text_length"], args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=multimodal_batch)
    seed_everything(args.seed)
    total = {"total": 0.0, "text": 0.0, "flow": 0.0}
    with torch.no_grad():
        for batch in loader:
            loss, breakdown = model(move_batch(batch, device), return_breakdown=True)
            values = {"total": loss.item(), "text": breakdown.text.item(),
                      "flow": sum(value.item() for value in breakdown.flow)}
            for name in total:
                total[name] += values[name] * len(batch)
    result = {"split": args.split, "samples": len(dataset), "seed": args.seed,
              "checkpoint_step": checkpoint_step, "flow_loss_weight": config["flow_loss_weight"],
              **{name: value / len(dataset) for name, value in total.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    evaluate(parser.parse_args())
