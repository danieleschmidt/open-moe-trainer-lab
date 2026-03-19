#!/usr/bin/env python3
"""
Train a small MoE language model from scratch.

Demonstrates:
- Building a MoEModel with top-k routing
- A minimal training loop with load-balancing loss
- Router analytics (expert utilization, routing entropy)
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from moe_lab.models import MoEModel


# ── Config ──────────────────────────────────────────────────────────────────
VOCAB_SIZE     = 1000
HIDDEN_SIZE    = 128
NUM_EXPERTS    = 8
EXPERTS_PER_TOKEN = 2
NUM_LAYERS     = 4
SEQ_LEN        = 32
BATCH_SIZE     = 16
NUM_STEPS      = 200
LR             = 3e-4
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


def make_synthetic_data(n_samples=512):
    """Synthetic token sequences (random LM task)."""
    tokens = torch.randint(0, VOCAB_SIZE, (n_samples, SEQ_LEN + 1))
    return TensorDataset(tokens[:, :-1], tokens[:, 1:])


def main():
    print(f"Device: {DEVICE}")

    # Build model
    model = MoEModel(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_experts=NUM_EXPERTS,
        experts_per_token=EXPERTS_PER_TOKEN,
        num_layers=NUM_LAYERS,
        num_attention_heads=4,
        max_position_embeddings=64,
        moe_layers=list(range(1, NUM_LAYERS, 2)),  # every other layer is MoE
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"MoE layers: {sorted(model.moe_layers)}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(make_synthetic_data(), batch_size=BATCH_SIZE, shuffle=True)
    data_iter = iter(loader)

    model.train()
    for step in range(1, NUM_STEPS + 1):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)

        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x)
        logits = model.lm_head(out.last_hidden_state)

        task_loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        total_loss = task_loss + out.load_balancing_loss

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(
                f"Step {step:4d}/{NUM_STEPS}  "
                f"loss={task_loss.item():.4f}  "
                f"lb_loss={out.load_balancing_loss.item():.5f}  "
                f"entropy={out.routing_info.entropy:.3f}  "
                f"load_var={out.routing_info.load_variance:.5f}"
            )

    # ── Router analytics ────────────────────────────────────────────────────
    print("\n── Expert utilization (final batch) ──")
    model.eval()
    with torch.no_grad():
        x_sample, _ = next(iter(loader))
        out = model(x_sample.to(DEVICE), return_routing_info=True)

    # Check each MoE layer's expert pool
    for layer_idx in sorted(model.moe_layers):
        layer = model.layers[layer_idx]
        routing_info = out.routing_info  # aggregated across layers

    print(f"Routing entropy  : {out.routing_info.entropy:.4f}  (higher = more diverse)")
    print(f"Load variance    : {out.routing_info.load_variance:.6f}  (lower = more balanced)")
    print("\nDone.")


if __name__ == "__main__":
    main()
