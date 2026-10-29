import argparse
import csv
import random
import re
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import (
    CocoTextImageDataset, build_model, load_checkpoint, move_batch, multimodal_batch,
    plot_losses, save_json, seed_everything,
)


def arguments():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path)
    parser.add_argument("--results-dir", type=Path, default=root / "results" / "lambda1")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--flow-loss-weight", type=float, default=1.0)
    parser.add_argument("--text-loss-weight", type=float, default=1.0)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--max-text-length", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--model-output-clean", action="store_true")
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.max_steps, args.save_every, args.dim, args.depth, args.heads, args.threads) < 1:
        parser.error("batch size, steps and model dimensions must be positive")
    if args.image_size < 16 or args.image_size % 16 or args.dim % args.heads:
        parser.error("image-size must be a multiple of 16 and dim must be divisible by heads")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.lr <= 0 or args.max_text_length < 1 or args.num_workers < 0:
        parser.error("lr and max-text-length must be positive; num-workers must be nonnegative")
    if args.keep_checkpoints < 1:
        parser.error("keep-checkpoints must be positive")
    if min(args.flow_loss_weight, args.text_loss_weight) < 0 or args.flow_loss_weight + args.text_loss_weight == 0:
        parser.error("loss weights must be nonnegative and at least one must be positive")
    if not args.smoke_test and args.resume is None and args.coco_root is None:
        parser.error("--coco-root is required for COCO training")
    return args


def train(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.resume, device) if args.resume else None
    resuming = checkpoint is not None
    config = checkpoint["config"] if checkpoint else {
        "tokenizer": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "vae": "stabilityai/sd-vae-ft-mse",
        "image_size": args.image_size, "dim": args.dim, "depth": args.depth, "heads": args.heads,
        "flow_loss_weight": args.flow_loss_weight, "text_loss_weight": args.text_loss_weight,
        "max_text_length": args.max_text_length, "seed": args.seed, "smoke_test": args.smoke_test,
        "batch_size": args.batch_size, "lr": args.lr, "limit": args.limit,
        "model_output_clean": args.model_output_clean,
    }
    seed_everything(config["seed"])
    if config["smoke_test"]:
        from transfusion_pytorch import Transfusion
        model = Transfusion(
            num_text_tokens=32, dim_latent=4, channel_first_latent=True,
            modality_default_shape=(2, 2), modality_num_dim=2,
            flow_loss_weight=config["flow_loss_weight"], text_loss_weight=config["text_loss_weight"],
            model_output_clean=config.get("model_output_clean", True),
            transformer=dict(dim=32, depth=2, heads=2, dim_head=16),
        ).to(device)
        generator = torch.Generator().manual_seed(config["seed"])
        dataset = [(torch.randint(0, 32, (8,), generator=generator), torch.randn(4, 2, 2, generator=generator))
                   for _ in range(8)]
    else:
        if args.coco_root is None:
            raise ValueError("--coco-root is required when resuming a COCO run")
        model, tokenizer = build_model(config, device)
        dataset = CocoTextImageDataset(args.coco_root, "train", tokenizer, config["image_size"],
                                      config["max_text_length"], config["limit"])
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config["lr"])
    step, epoch, offset = 0, 0, 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, epoch, offset = checkpoint["step"], checkpoint["epoch"], checkpoint["batch_offset"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
        checkpoint = None
    if step >= args.max_steps:
        raise ValueError("max-steps must exceed the checkpoint step")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.results_dir / "training.csv"
    if log_path.exists() and not resuming:
        raise FileExistsError("training.csv already exists; use --resume or a new results directory")
    if log_path.exists() and resuming:
        with log_path.open(encoding="utf-8", newline="") as previous:
            rows = list(csv.DictReader(previous))
        if rows and int(rows[-1]["step"]) != step:
            raise ValueError("Log and checkpoint steps differ; use a new results directory")
    config["dataset_size"] = len(dataset)
    save_json(args.results_dir / "config.json", config)
    started = time.monotonic()
    start_step = step
    stopped = False

    def progress(state):
        elapsed = time.monotonic() - started
        rate = elapsed / max(1, step - start_step)
        save_json(args.results_dir / "progress.json", {
            "state": state, "step": step, "target_steps": args.max_steps,
            "dataset_size": len(dataset),
            "images_seen": epoch * len(dataset) + min(offset * config["batch_size"], len(dataset)),
            "elapsed_seconds": elapsed, "seconds_per_step": rate,
            "estimated_seconds_remaining": max(0, args.max_steps - step) * rate,
            "flow_loss_weight": config["flow_loss_weight"],
        })

    def save():
        path = args.results_dir / f"model_step_{step}.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                    "step": step, "epoch": epoch, "batch_offset": offset,
                    "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, path)
        checkpoints = []
        for candidate in args.results_dir.glob("model_step_*.pt"):
            match = re.fullmatch(r"model_step_(\d+)\.pt", candidate.name)
            if match and candidate.resolve().parent == args.results_dir.resolve():
                checkpoints.append((int(match.group(1)), candidate))
        for _, candidate in sorted(checkpoints)[:-args.keep_checkpoints]:
            candidate.unlink()
        return path

    with log_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "total", "text", "flow"])
        if stream.tell() == 0:
            writer.writeheader()
        model.train()
        while step < args.max_steps and not stopped:
            loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,
                                generator=torch.Generator().manual_seed(config["seed"] + epoch),
                                num_workers=args.num_workers, collate_fn=multimodal_batch)
            start_offset = offset
            for batch_index, batch in enumerate(loader):
                if batch_index < start_offset:
                    continue
                optimizer.zero_grad()
                loss, breakdown = model(move_batch(batch, device), return_breakdown=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Loss is not finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                step += 1
                offset = batch_index + 1
                row = {"step": step, "total": loss.item(), "text": breakdown.text.item(),
                       "flow": sum(value.item() for value in breakdown.flow)}
                writer.writerow(row)
                stream.flush()
                if step % 10 == 0 or config["smoke_test"] or step == 1:
                    print(row, flush=True)
                    progress("training")
                if step % args.save_every == 0:
                    save()
                if step >= args.max_steps:
                    break
                if args.stop_file is not None and args.stop_file.exists():
                    stopped = True
                    break
            if step < args.max_steps and not stopped:
                epoch += 1
                offset = 0
    path = save()
    plot_losses(log_path)
    progress("stopped" if stopped else "completed")
    print(f"Saved {path}")


if __name__ == "__main__":
    train(arguments())
