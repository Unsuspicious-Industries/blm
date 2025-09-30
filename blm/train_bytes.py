#!/usr/bin/env python3
"""CLI script to train a byte-level GPT using PyTorch."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data import ByteSequenceDataset, DatasetStats, build_dataloader
from .gpt import BytesGPT, BytesGPTConfig

try:  # PyTorch 2.0+
    from torch.amp import autocast as torch_autocast, GradScaler as TorchGradScaler  # type: ignore[attr-defined]

    def create_grad_scaler(use_amp: bool) -> TorchGradScaler:
        try:
            return TorchGradScaler(device_type="cuda", enabled=use_amp)
        except TypeError:  # pragma: no cover - older signature
            return TorchGradScaler(enabled=use_amp)

    def autocast_ctx(use_amp: bool, device_type: str):
        try:
            return torch_autocast(device_type=device_type, enabled=use_amp)
        except TypeError:  # pragma: no cover
            return torch_autocast(enabled=use_amp)

except ImportError:  # Fallback for older versions
    from torch.cuda.amp import autocast as torch_autocast, GradScaler as TorchGradScaler  # type: ignore

    def create_grad_scaler(use_amp: bool) -> TorchGradScaler:
        return TorchGradScaler(enabled=use_amp)

    def autocast_ctx(use_amp: bool, device_type: str):
        return torch_autocast(enabled=use_amp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=str, help="Directory containing raw files to train on")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Directory to store checkpoints")
    parser.add_argument("--block-size", type=int, default=512, help="Context window size")
    parser.add_argument("--stride", type=int, default=None, help="Stride between training examples (defaults to block size)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping value (0 to disable)")
    parser.add_argument("--n-layer", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-embd", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Fraction of sequences reserved for validation")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision (CUDA only)")
    parser.add_argument("--compile", action="store_true", help="Compile the model with torch.compile (requires PyTorch 2.x)")
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU even if CUDA is available")
    parser.add_argument("--save-best", action="store_true", help="Only keep the best checkpoint")
    parser.add_argument("--save-every", type=int, default=0, help="Save a checkpoint every N epochs (0 to disable)")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def split_dataset(dataset: ByteSequenceDataset, val_ratio: float, seed: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset | None]:
    if val_ratio <= 0:
        return dataset, None
    val_size = max(1, int(len(dataset) * val_ratio))
    if val_size >= len(dataset):
        return dataset, None
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


def save_metadata(path: Path, args: argparse.Namespace, config: BytesGPTConfig, stats: DatasetStats, epoch: int, metrics: Dict[str, float]) -> None:
    payload = {
        "args": vars(args),
        "config": asdict(config),
        "dataset": asdict(stats),
        "epoch": epoch,
        "metrics": metrics,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    use_amp = args.amp and device.type == "cuda"

    dataset = ByteSequenceDataset(args.data_dir, block_size=args.block_size, stride=args.stride)
    stats = dataset.stats()
    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio, args.seed)

    train_loader = build_dataloader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)  # type: ignore[arg-type]
    val_loader: DataLoader | None = None
    if val_dataset is not None:
        val_loader = build_dataloader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)  # type: ignore[arg-type]

    config = BytesGPTConfig(
        block_size=args.block_size,
        vocab_size=256,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = BytesGPT(config).to(device)
    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[misc]
        except AttributeError as exc:  # pragma: no cover - torch<2.0
            raise RuntimeError("torch.compile requires PyTorch 2.0 or later") from exc

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    scaler = create_grad_scaler(use_amp)

    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    history: list[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            inputs = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(use_amp, device.type):
                logits, loss = model(inputs, targets)
            if loss is None:
                raise RuntimeError("Model forward pass did not return a loss")

            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        train_loss = running_loss / max(1, len(train_loader))

        val_loss = None
        if val_loader is not None:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["input_ids"].to(device)
                    targets = batch["labels"].to(device)
                    _, loss = model(inputs, targets)
                    if loss is None:
                        continue
                    total += loss.item()
            val_loss = total / max(1, len(val_loader))

        metrics: Dict[str, float] = {"train_loss": train_loss}
        if val_loss is not None:
            metrics["val_loss"] = val_loss

        history.append({"epoch": epoch, **metrics})
        tqdm.write(
            f"Epoch {epoch}: train_loss={train_loss:.4f}" + (f", val_loss={val_loss:.4f}" if val_loss is not None else "")
        )

        save_checkpoint = False
        ckpt_name = f"epoch{epoch:03d}.pt"

        if val_loss is None:
            save_checkpoint = True
            if args.save_best:
                ckpt_name = "last.pt"
        else:
            if args.save_best:
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint = True
                    ckpt_name = "best.pt"
            else:
                if val_loss < best_val:
                    best_val = val_loss
                if args.save_every and epoch % args.save_every == 0:
                    save_checkpoint = True

        if not args.save_best:
            if args.save_every == 0:
                save_checkpoint = True
            elif epoch == args.epochs:
                save_checkpoint = True

        if save_checkpoint:
            ckpt_path = checkpoint_dir / ckpt_name
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": asdict(config),
                    "args": vars(args),
                    "epoch": epoch,
                    "metrics": metrics,
                },
                ckpt_path,
            )
            save_metadata(checkpoint_dir / "metadata.json", args, config, stats, epoch, metrics)

            if args.save_best and ckpt_name not in {"best.pt", "last.pt"}:
                ckpt_path.unlink(missing_ok=True)

    history_path = checkpoint_dir / "training_log.json"
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    tqdm.write(f"Training complete. Checkpoints stored in {checkpoint_dir.resolve()}")


if __name__ == "__main__":
    main()
