# open-moe-trainer-lab

A focused PyTorch toolkit for training and analyzing **Mixture of Experts (MoE)** language models. Implements top-k gating, load-balancing losses, and router analytics — the core machinery behind Mixtral, Switch Transformer, and similar architectures.

---

## What it does

- **MoE layer** — top-k router (token → k experts) + weighted expert aggregation
- **Load balancing** — auxiliary loss + router z-loss to keep expert utilization balanced
- **Router analytics** — routing entropy, per-expert load variance, utilization tracking
- **Multiple router types** — `TopKRouter`, `SwitchRouter` (top-1 with capacity), `ExpertChoice`
- **Expert variants** — standard FFN or GLU (SwiGLU-style)
- **Training utilities** — `MoETrainer` with gradient accumulation, warmup scheduling, mixed precision

---

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

---

## Quick start

```python
import torch
from moe_lab.models import MoEModel

model = MoEModel(
    vocab_size=32000,
    hidden_size=768,
    num_experts=8,
    experts_per_token=2,       # top-2 routing
    num_layers=12,
    num_attention_heads=12,
    max_position_embeddings=2048,
    moe_layers=[1, 3, 5, 7, 9, 11],  # every other layer is MoE
)

input_ids = torch.randint(0, 32000, (2, 64))
out = model(input_ids, return_routing_info=True)

print(out.last_hidden_state.shape)   # (2, 64, 768)
print(out.load_balancing_loss)        # auxiliary loss for training
print(out.routing_info.entropy)       # routing diversity
print(out.routing_info.load_variance) # expert balance
```

### Training loop

```python
import torch.nn as nn

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

out = model(input_ids)
logits = model.lm_head(out.last_hidden_state)  # (batch, seq, vocab)

task_loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
total_loss = task_loss + out.load_balancing_loss  # include routing loss

optimizer.zero_grad()
total_loss.backward()
optimizer.step()
```

See [`examples/train_moe.py`](examples/train_moe.py) for a full working training loop.

---

## Architecture

```
MoEModel
├── embed_tokens  (nn.Embedding)
├── embed_positions (nn.Embedding)
├── layers
│   ├── [0] TransformerEncoderLayer   (dense FFN)
│   ├── [1] MoELayer                  ← top-k router + expert pool
│   │       ├── TopKRouter (linear gate → softmax → top-k)
│   │       └── ExpertPool (N independent FFNs)
│   ├── [2] TransformerEncoderLayer
│   └── ...
├── ln_f (LayerNorm)
└── lm_head (Linear → vocab)
```

**Routing**: Each token's hidden state is projected to `num_experts` logits. Top-k are selected; softmax over the selected logits gives mixing weights. Each chosen expert processes the token independently; outputs are weighted and summed.

**Load balancing**: Auxiliary loss penalises deviation from uniform expert assignment. Router z-loss penalises large logit magnitudes (improves stability).

---

## Example output

Running `python examples/train_moe.py` (2.7M param model, 200 steps, CUDA):

```
Device: cuda
Model parameters: 2,770,432
MoE layers: [1, 3]

Step    1/200  loss=6.9370  lb_loss=0.00212  entropy=2.056  load_var=0.00002
Step   50/200  loss=6.7461  lb_loss=0.00212  entropy=2.052  load_var=0.00002
Step  100/200  loss=6.4281  lb_loss=0.00212  entropy=2.043  load_var=0.00003
Step  200/200  loss=6.0246  lb_loss=0.00216  entropy=2.004  load_var=0.00004

Routing entropy  : 1.9997  (higher = more diverse)
Load variance    : 0.000069  (lower = more balanced)
```

---

## Running tests

```bash
pytest tests/
```

34 tests covering routers, experts, the MoE layer, full model forward/backward, and a training integration test.

---

## Project layout

```
moe_lab/
├── models/
│   ├── router.py        # TopKRouter, SwitchRouter, ExpertChoice
│   ├── expert.py        # Expert, GLUExpert, ExpertPool
│   ├── moe_model.py     # MoELayer, MoEModel
│   └── architectures.py # SwitchTransformer, MixtralModel, CustomMoE
├── training/
│   ├── trainer.py       # MoETrainer
│   └── distributed.py   # distributed training utilities
├── analytics/
│   ├── monitor.py       # RouterMonitor
│   └── analyzer.py      # RouterAnalyzer
└── inference/
    └── optimized.py     # OptimizedMoEModel
```

---

## References

- [Switch Transformers](https://arxiv.org/abs/2101.03961) — Fedus et al. 2021
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Jiang et al. 2024
- [ST-MoE](https://arxiv.org/abs/2202.08906) — Zoph et al. 2022 (z-loss)
- [Expert Choice Routing](https://arxiv.org/abs/2202.09368) — Zhou et al. 2022

---

## License

MIT
