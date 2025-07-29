# API Reference Guide

Complete API reference for Open MoE Trainer Lab.

## Quick Start

```python
from moe_lab import MoEModel, MoETrainer

# Create and train a model
model = MoEModel(num_experts=8, experts_per_token=2)
trainer = MoETrainer(model)
trainer.train(dataset)
```

## Core Classes

### MoEModel

```python
class MoEModel(nn.Module):
    """Main MoE model implementation."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_experts: int = 8,
        experts_per_token: int = 2,
        expert_hidden_size: int = None,
        router_type: str = "top_k",
        load_balancing_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
        dropout: float = 0.1,
        **kwargs
    )
```

**Parameters:**
- `vocab_size`: Vocabulary size
- `hidden_size`: Hidden dimension size
- `num_layers`: Number of transformer layers
- `num_experts`: Total number of experts
- `experts_per_token`: Number of experts per token (top-k)
- `router_type`: Routing algorithm ("top_k", "expert_choice", "switch")
- `load_balancing_loss_coef`: Load balancing loss coefficient
- `router_z_loss_coef`: Router z-loss coefficient

**Methods:**

#### forward()
```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    return_routing_info: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]
```

#### get_expert_weights()
```python
def get_expert_weights(self, layer_idx: int = None) -> Dict[int, torch.Tensor]
```

#### set_expert_parallel_group()
```python
def set_expert_parallel_group(self, group: torch.distributed.ProcessGroup) -> None
```

### MoETrainer

```python
class MoETrainer:
    """High-level trainer for MoE models."""
    
    def __init__(
        self,
        model: MoEModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        **kwargs
    )
```

**Methods:**

#### train()
```python
def train(
    self,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    warmup_steps: int = 1000,
    eval_steps: int = 500,
    save_steps: int = 1000,
    output_dir: str = "./outputs",
    **kwargs
) -> TrainingResult
```

#### evaluate()
```python
def evaluate(
    self,
    eval_dataset: Dataset,
    batch_size: int = 16,
    num_samples: Optional[int] = None
) -> EvalResult
```

#### save_model()
```python
def save_model(self, path: str, save_optimizer: bool = True) -> None
```

## Router Classes

### TopKRouter

```python
class TopKRouter(nn.Module):
    """Top-k routing implementation."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        router_bias: bool = False,
        router_jitter_noise: float = 0.0
    )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]
```

### ExpertChoiceRouter

```python
class ExpertChoiceRouter(nn.Module):
    """Expert-choice routing implementation."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        drop_tokens: bool = True
    )
```

### SwitchRouter

```python
class SwitchRouter(nn.Module):
    """Switch Transformer routing implementation."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        capacity_factor: float = 1.25,
        drop_tokens: bool = True,
        jitter_noise: float = 0.1
    )
```

## Expert Implementations

### FeedForwardExpert

```python
class FeedForwardExpert(nn.Module):
    """Standard feedforward expert."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        dropout: float = 0.1
    )
```

### GLUExpert

```python
class GLUExpert(nn.Module):
    """Gated Linear Unit expert."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "swiglu"
    )
```

## Distributed Training

### DistributedMoETrainer

```python
class DistributedMoETrainer(MoETrainer):
    """Distributed training for MoE models."""
    
    def __init__(
        self,
        model: MoEModel,
        world_size: int,
        expert_parallel_size: int = 1,
        model_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        **kwargs
    )
```

### ExpertParallelism

```python
class ExpertParallelism:
    """Expert parallelism utilities."""
    
    @staticmethod
    def parallelize_experts(
        model: MoEModel,
        expert_parallel_size: int
    ) -> MoEModel
    
    @staticmethod
    def gather_expert_outputs(
        expert_outputs: torch.Tensor,
        expert_parallel_group: torch.distributed.ProcessGroup
    ) -> torch.Tensor
```

## Inference Classes

### OptimizedMoEModel

```python
class OptimizedMoEModel:
    """Optimized model for inference."""
    
    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        max_experts_in_memory: int = 4,
        expert_cache_size: int = 8
    )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor
```

### DynamicExpertLoader

```python
class DynamicExpertLoader:
    """Dynamic expert loading for memory efficiency."""
    
    def __init__(
        self,
        model: MoEModel,
        cache_size: int = 8,
        eviction_policy: str = "lru",
        prefetch_strategy: str = "usage_based"
    )
```

## Analytics and Visualization

### RouterAnalyzer

```python
class RouterAnalyzer:
    """Analyze routing decisions and expert utilization."""
    
    def __init__(self, model: MoEModel)
    
    def analyze_routing(
        self,
        inputs: torch.Tensor,
        return_heatmap: bool = False
    ) -> RoutingAnalysis
    
    def compute_expert_specialization(
        self,
        dataset: Dataset,
        num_samples: int = 10000
    ) -> Dict[int, Dict[str, float]]
```

### ExpertMonitor

```python
class ExpertMonitor:
    """Monitor expert utilization in real-time."""
    
    def __init__(self, model: MoEModel)
    
    def start_monitoring(self) -> None
    def stop_monitoring(self) -> Dict[str, Any]
    def get_current_stats(self) -> MonitoringStats
```

### TrainingVisualizer

```python
class TrainingVisualizer:
    """Visualize training progress and expert behavior."""
    
    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: str = None
    ) -> None
    
    def plot_expert_utilization(
        self,
        utilization_data: Dict[int, float],
        save_path: str = None
    ) -> None
```

## Utility Functions

### Model Loading

```python
def load_pretrained_moe(
    model_name_or_path: str,
    **kwargs
) -> MoEModel

def save_moe_checkpoint(
    model: MoEModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    path: str
) -> None
```

### Data Processing

```python
def prepare_moe_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    num_proc: int = 4
) -> Dataset

def collate_moe_batch(
    batch: List[Dict[str, Any]]
) -> Dict[str, torch.Tensor]
```

## Configuration Classes

### MoEConfig

```python
@dataclass
class MoEConfig:
    """Configuration for MoE models."""
    
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_experts: int = 8
    experts_per_token: int = 2
    router_type: str = "top_k"
    load_balancing_loss_coef: float = 0.01
    
    @classmethod
    def from_json(cls, json_path: str) -> "MoEConfig"
    
    def to_dict(self) -> Dict[str, Any]
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    output_dir: str = "./outputs"
    num_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
```

## Exception Classes

```python
class MoEError(Exception):
    """Base exception for MoE-related errors."""

class RouterError(MoEError):
    """Routing-related errors."""

class ExpertError(MoEError):
    """Expert-related errors."""

class DistributedError(MoEError):
    """Distributed training errors."""
```

## Constants and Enums

```python
class RouterType(Enum):
    TOP_K = "top_k"
    EXPERT_CHOICE = "expert_choice"
    SWITCH = "switch"

class ActivationType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SWIGLU = "swiglu"
    GEGLU = "geglu"

# Default values
DEFAULT_HIDDEN_SIZE = 768
DEFAULT_NUM_EXPERTS = 8
DEFAULT_EXPERTS_PER_TOKEN = 2
MAX_EXPERTS_SUPPORTED = 1024
```

## Type Hints

```python
from typing import Dict, List, Optional, Tuple, Union, Any
from torch import Tensor
from torch.nn import Module

# Common type aliases
ExpertWeights = Dict[int, Tensor]
RoutingInfo = Dict[str, Any]
ExpertOutputs = Tuple[Tensor, RoutingInfo]
TrainingMetrics = Dict[str, float]
```

---

For examples and tutorials, see the [Examples Directory](../examples/) and [Getting Started Guide](GETTING_STARTED.md).