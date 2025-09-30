from __future__ import annotations

import torch
import torch.nn.functional as F


def cross_entropy_loss(outputs: torch.Tensor, targets: torch.Tensor, debug: bool = False) -> torch.Tensor:
    """Compute token-level cross entropy for language modeling."""

    if outputs.ndim != 3:
        raise ValueError("outputs must have shape (batch, seq_len, vocab_size)")
    if targets.ndim != 2:
        raise ValueError("targets must have shape (batch, seq_len)")

    logits = outputs.reshape(-1, outputs.size(-1))
    labels = targets.reshape(-1)
    log_probs = F.log_softmax(logits, dim=-1)

    if debug:
        print(f"outputs: {outputs.detach().cpu().numpy()}")
        print(f"targets: {targets.detach().cpu().numpy()}")
        print(f"probs: {F.softmax(logits, dim=-1).detach().cpu().numpy()}")
        print(f"log_probs: {log_probs.detach().cpu().numpy()}")

    loss = -log_probs[torch.arange(logits.size(0), device=logits.device), labels]

    if debug:
        print(f"loss: {loss.detach().cpu().numpy()}")

    return loss.mean()


if __name__ == "__main__":
    outputs = torch.tensor([[-20.0, -10.0, 10.0], [0.9, 0.05, 0.05]])
    outputs = outputs.unsqueeze(0)
    targets = torch.tensor([[2, 0]])
    loss = cross_entropy_loss(outputs, targets, debug=True)
    print(loss.item())
