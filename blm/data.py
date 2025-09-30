from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .gpt import encode


@dataclass(slots=True)
class DatasetStats:
    total_tokens: int
    num_sequences: int


class ByteSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Concatenate byte files into a corpus and expose sliding-window sequences."""

    def __init__(
        self,
        root: str | Path,
        block_size: int,
        stride: int | None = None,
        extensions: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.block_size = block_size
        self.stride = stride or block_size
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        buffer = bytearray()
        allowed_exts = {ext.lower() for ext in extensions} if extensions else None
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if allowed_exts and path.suffix.lower() not in allowed_exts:
                continue
            data = path.read_bytes()
            if data:
                buffer.extend(data)

        if not buffer:
            raise ValueError(f"No bytes found in directory: {self.root}")

        self.data = torch.tensor(buffer, dtype=torch.long)
        self.total_tokens = self.data.numel()
        self.max_start = self.total_tokens - (self.block_size + 1)
        if self.max_start < 0:
            raise ValueError(
                "Corpus is shorter than block_size + 1; add more data or reduce block_size"
            )

        self._length = self.max_start // self.stride + 1

    def __len__(self) -> int:  # type: ignore[override]
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        if index < 0 or index >= self._length:
            raise IndexError(index)
        start = min(index * self.stride, self.max_start)
        end = start + self.block_size
        x = self.data[start:end]
        y = self.data[start + 1 : end + 1]
        return x.clone(), y.clone()

    def stats(self) -> DatasetStats:
        return DatasetStats(total_tokens=self.total_tokens, num_sequences=len(self))


def build_dataloader(
    dataset: ByteSequenceDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    """Create a :class:`DataLoader` with sensible defaults for byte data."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    def _collate(batch: Iterable[tuple[torch.Tensor, torch.Tensor]]):
        inputs, targets = zip(*batch)
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        return {
            "input_ids": inputs,
            "labels": targets,
            "attention_mask": torch.ones_like(inputs, dtype=torch.bool),
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
    )


def tokenize_bytes(data: bytes | bytearray | memoryview | str) -> torch.Tensor:
    """Tokenize arbitrary string/bytes into a 1D tensor of byte IDs."""

    encoded = encode(data)
    return torch.tensor(encoded, dtype=torch.long)


__all__ = ["ByteSequenceDataset", "build_dataloader", "tokenize_bytes", "DatasetStats"]