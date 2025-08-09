"""Advanced distributed training for MoE models with expert parallelism."""

import os
import logging
import threading
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from contextlib import contextmanager
from enum import Enum

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import torch.multiprocessing as mp

from ..models import MoEModel
from ..utils.monitoring import get_metrics_collector
from ..utils.error_handling import MoETrainingError, ErrorSeverity

logger = logging.getLogger(__name__)


class DistributedStrategy(Enum):
    """Distributed training strategies."""
    DATA_PARALLEL = "data_parallel"
    EXPERT_PARALLEL = "expert_parallel" 
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"
    FSDP = "fsdp"


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    strategy: DistributedStrategy
    world_size: int
    local_rank: int
    expert_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    static_graph: bool = False
    sync_batch_norm: bool = True
    
    def __post_init__(self):
        # Validate parallel sizes
        total_size = self.expert_parallel_size * self.pipeline_parallel_size * self.data_parallel_size
        if total_size != self.world_size:
            raise ValueError(f"Product of parallel sizes ({total_size}) must equal world_size ({self.world_size})")


class ExpertParallelWrapper(nn.Module):
    """Wrapper for expert-parallel MoE models."""
    
    def __init__(
        self,
        model: MoEModel,
        expert_parallel_group: dist.ProcessGroup,
        local_expert_indices: List[int]
    ):
        super().__init__()
        self.model = model
        self.expert_parallel_group = expert_parallel_group
        self.local_expert_indices = local_expert_indices
        self.expert_parallel_rank = dist.get_rank(expert_parallel_group)
        self.expert_parallel_size = dist.get_world_size(expert_parallel_group)
        
        # Distribute experts across processes
        self._distribute_experts()
        
        logger.info(f"ExpertParallel: rank {self.expert_parallel_rank}, experts {local_expert_indices}")
        
    def _distribute_experts(self):
        """Distribute expert parameters across processes."""
        for layer_idx, layer in enumerate(self.model.layers):
            if hasattr(layer, 'experts') and hasattr(layer.experts, 'experts'):
                expert_pool = layer.experts.experts
                distributed_experts = nn.ModuleList()
                
                # Keep only local experts
                for expert_idx in range(len(expert_pool)):
                    if expert_idx in self.local_expert_indices:
                        distributed_experts.append(expert_pool[expert_idx])
                    else:
                        # Create empty placeholder to maintain indexing
                        distributed_experts.append(None)
                        
                layer.experts.experts = distributed_experts
                
    def forward(self, *args, **kwargs):
        """Forward pass with expert communication."""
        # Custom forward logic to handle distributed experts
        return self._distributed_forward(*args, **kwargs)
        
    def _distributed_forward(self, input_ids, attention_mask=None, **kwargs):
        """Distributed forward pass with expert communication."""
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings (replicated across all processes)
        hidden_states = self.model.embed_tokens(input_ids)
        
        # Process through layers
        for layer_idx, layer in enumerate(self.model.layers):
            if hasattr(layer, 'experts'):
                # MoE layer - needs expert communication
                hidden_states = self._process_moe_layer(hidden_states, layer, layer_idx)
            else:
                # Regular layer - process locally
                hidden_states = layer(hidden_states, attention_mask)
                
        # Final layer norm and head
        hidden_states = self.model.layer_norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        
        return logits
        
    def _process_moe_layer(self, hidden_states, layer, layer_idx):
        """Process MoE layer with expert parallelism."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Router computation (replicated)
        router_output = layer.router(hidden_states)
        expert_indices = router_output.selected_experts  # [batch_size, seq_len, k]
        expert_weights = router_output.expert_weights    # [batch_size, seq_len, k]
        
        # Prepare for all-to-all communication
        tokens_per_expert = self._count_tokens_per_expert(expert_indices)
        
        # All-to-all scatter to distribute tokens to expert processes
        scattered_tokens, token_counts = self._all_to_all_scatter(
            hidden_states, expert_indices, expert_weights
        )
        
        # Process tokens through local experts
        expert_outputs = []
        for expert_idx in self.local_expert_indices:
            expert_tokens = scattered_tokens.get(expert_idx, torch.empty(0, hidden_size))
            if expert_tokens.numel() > 0 and layer.experts.experts[expert_idx] is not None:
                expert_output = layer.experts.experts[expert_idx](expert_tokens)
                expert_outputs.append(expert_output)
            else:
                expert_outputs.append(torch.empty(0, hidden_size))
                
        # All-to-all gather to collect expert outputs
        gathered_outputs = self._all_to_all_gather(expert_outputs, token_counts)
        
        # Combine outputs with routing weights
        combined_output = self._combine_expert_outputs(
            gathered_outputs, expert_indices, expert_weights, (batch_size, seq_len, hidden_size)
        )
        
        return combined_output
        
    def _count_tokens_per_expert(self, expert_indices):
        """Count tokens assigned to each expert."""
        flat_indices = expert_indices.flatten()
        expert_counts = torch.zeros(self.model.num_experts, dtype=torch.long, device=expert_indices.device)
        
        for expert_idx in range(self.model.num_experts):
            expert_counts[expert_idx] = (flat_indices == expert_idx).sum()
            
        return expert_counts
        
    def _all_to_all_scatter(self, hidden_states, expert_indices, expert_weights):
        """Scatter tokens to appropriate expert processes."""
        # Simplified implementation - in practice would use optimized all-to-all
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten tokens
        flat_tokens = hidden_states.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        flat_indices = expert_indices.view(-1)  # [batch_size * seq_len * k]
        
        # Group tokens by expert
        scattered_tokens = {}
        for expert_idx in range(self.model.num_experts):
            mask = flat_indices == expert_idx
            expert_tokens = flat_tokens[mask]
            scattered_tokens[expert_idx] = expert_tokens
            
        # In real implementation, this would involve actual all-to-all communication
        # For now, simulate by keeping only local expert tokens
        local_scattered = {}
        token_counts = {}
        
        for expert_idx in self.local_expert_indices:
            if expert_idx in scattered_tokens:
                local_scattered[expert_idx] = scattered_tokens[expert_idx]
                token_counts[expert_idx] = scattered_tokens[expert_idx].shape[0]
            else:
                local_scattered[expert_idx] = torch.empty(0, hidden_size, device=hidden_states.device)
                token_counts[expert_idx] = 0
                
        return local_scattered, token_counts
        
    def _all_to_all_gather(self, expert_outputs, token_counts):
        """Gather expert outputs from all processes."""
        # Simplified implementation - in practice would use optimized all-to-all
        # For now, just return local expert outputs
        return {i: output for i, output in enumerate(expert_outputs)}
        
    def _combine_expert_outputs(self, expert_outputs, expert_indices, expert_weights, output_shape):
        """Combine expert outputs with routing weights."""
        batch_size, seq_len, hidden_size = output_shape
        combined = torch.zeros(batch_size * seq_len, hidden_size, device=expert_indices.device)
        
        # This is a simplified combination - real implementation would be more complex
        flat_indices = expert_indices.view(-1)
        flat_weights = expert_weights.view(-1)
        
        for i, (expert_idx, weight) in enumerate(zip(flat_indices, flat_weights)):
            if expert_idx.item() in expert_outputs:
                expert_output = expert_outputs[expert_idx.item()]
                if i < len(expert_output):
                    combined[i] += weight * expert_output[i % len(expert_output)]
                    
        return combined.view(batch_size, seq_len, hidden_size)


class DistributedMoETrainer:
    """Distributed trainer for MoE models with multiple parallelism strategies."""
    
    def __init__(
        self,
        model: MoEModel,
        config: DistributedConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.current_step = 0
        
        # Setup distributed environment
        self._setup_distributed()
        
        # Wrap model based on strategy
        self.distributed_model = self._wrap_model()
        
        # Performance tracking
        self.step_times = []
        self.communication_times = []
        self.computation_times = []
        
        logger.info(f"Initialized DistributedMoETrainer with strategy: {config.strategy}")
        
    def _setup_distributed(self):
        """Setup distributed training environment."""
        if not dist.is_initialized():
            raise RuntimeError("Distributed training requires torch.distributed to be initialized")
            
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = self.config.local_rank
        
        # Create process groups for different parallelism types
        self._create_process_groups()
        
    def _create_process_groups(self):
        """Create process groups for different types of parallelism."""
        # Expert parallel groups
        if self.config.expert_parallel_size > 1:
            self.expert_parallel_groups = []
            for i in range(0, self.world_size, self.config.expert_parallel_size):
                group_ranks = list(range(i, min(i + self.config.expert_parallel_size, self.world_size)))
                group = dist.new_group(group_ranks)
                if self.rank in group_ranks:
                    self.expert_parallel_group = group
                    self.expert_parallel_rank = group_ranks.index(self.rank)
                self.expert_parallel_groups.append(group)
        else:
            self.expert_parallel_group = None
            self.expert_parallel_rank = 0
            
        # Pipeline parallel groups  
        if self.config.pipeline_parallel_size > 1:
            self.pipeline_parallel_groups = []
            # Create groups for pipeline stages
            for stage in range(self.config.pipeline_parallel_size):
                group_ranks = list(range(stage, self.world_size, self.config.pipeline_parallel_size))
                group = dist.new_group(group_ranks)
                if self.rank in group_ranks:
                    self.pipeline_parallel_group = group
                    self.pipeline_stage_id = stage
                self.pipeline_parallel_groups.append(group)
        else:
            self.pipeline_parallel_group = None
            self.pipeline_stage_id = 0
            
        # Data parallel groups
        if self.config.data_parallel_size > 1:
            # Data parallel group spans remaining dimensions
            self.data_parallel_group = dist.new_group()  # Default group
        else:
            self.data_parallel_group = None
            
    def _wrap_model(self):
        """Wrap model with appropriate distributed wrapper."""
        if self.config.strategy == DistributedStrategy.DATA_PARALLEL:
            return self._wrap_data_parallel()
        elif self.config.strategy == DistributedStrategy.EXPERT_PARALLEL:
            return self._wrap_expert_parallel()
        elif self.config.strategy == DistributedStrategy.HYBRID:
            return self._wrap_hybrid_parallel()
        elif self.config.strategy == DistributedStrategy.FSDP:
            return self._wrap_fsdp()
        else:
            raise ValueError(f"Unsupported distributed strategy: {self.config.strategy}")
            
    def _wrap_data_parallel(self):
        """Wrap model with standard data parallelism."""
        return DDP(
            self.model.to(self.local_rank),
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=self.config.find_unused_parameters,
            bucket_cap_mb=self.config.bucket_cap_mb,
            static_graph=self.config.static_graph
        )
        
    def _wrap_expert_parallel(self):
        """Wrap model with expert parallelism."""
        # Distribute experts across processes
        experts_per_process = self.model.num_experts // self.config.expert_parallel_size
        start_expert = self.expert_parallel_rank * experts_per_process
        end_expert = start_expert + experts_per_process
        
        if self.expert_parallel_rank == self.config.expert_parallel_size - 1:
            # Last process gets remaining experts
            end_expert = self.model.num_experts
            
        local_expert_indices = list(range(start_expert, end_expert))
        
        return ExpertParallelWrapper(
            self.model.to(self.local_rank),
            self.expert_parallel_group,
            local_expert_indices
        )
        
    def _wrap_hybrid_parallel(self):
        """Wrap model with hybrid parallelism (expert + data parallel)."""
        # First apply expert parallel
        model = self._wrap_expert_parallel()
        
        # Then wrap with data parallel if needed
        if self.config.data_parallel_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                process_group=self.data_parallel_group,
                find_unused_parameters=self.config.find_unused_parameters
            )
            
        return model
        
    def _wrap_fsdp(self):
        """Wrap model with Fully Sharded Data Parallel."""
        auto_wrap_policy = size_based_auto_wrap_policy  # Can be customized
        
        return FSDP(
            self.model.to(self.local_rank),
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.local_rank,
            sync_module_states=True,
            sharding_strategy=FSDP.FULL_SHARD  # Can be configured
        )
        
    @contextmanager
    def timing_context(self, name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        yield
        elapsed = time.time() - start_time
        
        if name == "communication":
            self.communication_times.append(elapsed)
        elif name == "computation":
            self.computation_times.append(elapsed)
            
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Execute single distributed training step."""
        step_start = time.time()
        
        # Move batch to device
        batch = {k: v.to(self.local_rank) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
                
        # Forward pass
        with self.timing_context("computation"):
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.distributed_model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            else:
                outputs = self.distributed_model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
            # Scale loss by gradient accumulation steps
            loss = loss / self.config.gradient_accumulation_steps
            
        # Backward pass
        with self.timing_context("computation"):
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
        # Optimizer step (if at accumulation boundary)
        step_metrics = {}
        if (self.current_step + 1) % self.config.gradient_accumulation_steps == 0:
            with self.timing_context("communication"):
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    
                self.optimizer.zero_grad()
                
            # Collect metrics
            step_metrics.update({
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gradient_norm': self._compute_gradient_norm()
            })
            
        # Track timing
        step_time = time.time() - step_start
        self.step_times.append(step_time)
        self.current_step += 1
        
        step_metrics.update({
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'step_time': step_time,
            'rank': self.rank
        })
        
        return step_metrics
        
    def _compute_gradient_norm(self) -> float:
        """Compute global gradient norm across all parameters."""
        total_norm = 0.0
        for p in self.distributed_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                
        total_norm = total_norm ** (1. / 2)
        
        # Reduce across processes for accurate global norm
        if self.config.strategy in [DistributedStrategy.DATA_PARALLEL, DistributedStrategy.FSDP]:
            norm_tensor = torch.tensor(total_norm, device=self.local_rank)
            dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM)
            total_norm = norm_tensor.item() / self.world_size
            
        return total_norm


def setup_distributed_training(
    rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl"
):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    torch.cuda.set_device(rank)
    logger.info(f"Initialized distributed training: rank {rank}/{world_size}")


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed training")