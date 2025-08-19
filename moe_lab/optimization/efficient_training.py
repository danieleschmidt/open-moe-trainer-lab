"""Efficient training optimizations for MoE models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from typing import Dict, List, Tuple, Optional, Any, Callable
import math
import time
from contextlib import contextmanager

from ..models import MoEModel
from ..training import MoETrainer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class GradientAccumulationOptimizer:
    """Optimizer wrapper for efficient gradient accumulation."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 1,
        clip_grad_norm: Optional[float] = None,
        clip_grad_value: Optional[float] = None
    ):
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        
        self.current_step = 0
        self.accumulated_loss = 0.0
        
    def step(self, loss: torch.Tensor) -> bool:
        """Accumulate gradients and step when ready."""
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.accumulated_loss += loss.item()
        self.current_step += 1
        
        # Step when accumulation is complete
        if self.current_step % self.accumulation_steps == 0:
            # Apply gradient clipping
            if self.clip_grad_norm is not None:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self._get_parameters(), 
                    self.clip_grad_norm
                )
                
            if self.clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(
                    self._get_parameters(),
                    self.clip_grad_value
                )
                
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            avg_loss = self.accumulated_loss / self.accumulation_steps
            self.accumulated_loss = 0.0
            
            return True, avg_loss
            
        return False, self.accumulated_loss / self.current_step
        
    def _get_parameters(self):
        """Get parameters from optimizer."""
        params = []
        for group in self.optimizer.param_groups:
            params.extend(group['params'])
        return params


class DynamicBatchSizeScheduler:
    """Dynamic batch size scheduling for memory efficiency."""
    
    def __init__(
        self,
        initial_batch_size: int = 16,
        max_batch_size: int = 128,
        min_batch_size: int = 4,
        growth_factor: float = 1.5,
        shrink_factor: float = 0.8,
        patience: int = 10,
        memory_threshold: float = 0.9
    ):
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor
        self.patience = patience
        self.memory_threshold = memory_threshold
        
        self.oom_count = 0
        self.success_count = 0
        
    def should_increase_batch_size(self) -> bool:
        """Check if batch size should be increased."""
        return (
            self.success_count >= self.patience and
            self.current_batch_size < self.max_batch_size and
            self._get_memory_usage() < self.memory_threshold
        )
        
    def should_decrease_batch_size(self) -> bool:
        """Check if batch size should be decreased."""
        return (
            self.oom_count > 0 or
            self._get_memory_usage() > self.memory_threshold
        )
        
    def update_batch_size(self, success: bool = True, oom: bool = False) -> int:
        """Update batch size based on training feedback."""
        if oom:
            self.oom_count += 1
            self.success_count = 0
            # Aggressive reduction on OOM
            self.current_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.5)
            )
            logger.warning(f"OOM detected, reducing batch size to {self.current_batch_size}")
            
        elif success:
            self.oom_count = 0
            self.success_count += 1
            
            if self.should_increase_batch_size():
                new_batch_size = min(
                    self.max_batch_size,
                    int(self.current_batch_size * self.growth_factor)
                )
                if new_batch_size > self.current_batch_size:
                    self.current_batch_size = new_batch_size
                    self.success_count = 0
                    logger.info(f"Increasing batch size to {self.current_batch_size}")
                    
        return self.current_batch_size
        
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage ratio."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_memory = torch.cuda.get_device_properties(0).total_memory
            return allocated / max_memory
        return 0.0


class SelectiveActivationCheckpointing:
    """Selective gradient checkpointing for memory efficiency."""
    
    def __init__(
        self,
        model: MoEModel,
        checkpoint_ratio: float = 0.5,
        memory_threshold: float = 0.8
    ):
        self.model = model
        self.checkpoint_ratio = checkpoint_ratio
        self.memory_threshold = memory_threshold
        
        self.checkpointed_layers = set()
        self._setup_checkpointing()
        
    def _setup_checkpointing(self):
        """Setup gradient checkpointing for selected layers."""
        total_layers = len(self.model.layers)
        num_checkpoint = int(total_layers * self.checkpoint_ratio)
        
        # Checkpoint every nth layer
        step = max(1, total_layers // num_checkpoint)
        
        for i in range(0, total_layers, step):
            if i < total_layers:
                self.checkpointed_layers.add(i)
                layer = self.model.layers[i]
                
                # Enable gradient checkpointing
                if hasattr(layer, 'gradient_checkpointing'):
                    layer.gradient_checkpointing = True
                else:
                    # Wrap layer with checkpointing
                    self._wrap_layer_with_checkpointing(layer, i)
                    
    def _wrap_layer_with_checkpointing(self, layer: nn.Module, layer_idx: int):
        """Wrap layer with gradient checkpointing."""
        original_forward = layer.forward
        
        def checkpointed_forward(*args, **kwargs):
            if self.model.training and self._should_checkpoint():
                return torch.utils.checkpoint.checkpoint(
                    original_forward, *args, **kwargs
                )
            else:
                return original_forward(*args, **kwargs)
                
        layer.forward = checkpointed_forward
        
    def _should_checkpoint(self) -> bool:
        """Decide whether to use checkpointing based on memory usage."""
        if not torch.cuda.is_available():
            return True
            
        memory_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        return memory_usage > self.memory_threshold
        
    def adaptive_checkpointing(self, current_memory_usage: float):
        """Adaptively adjust checkpointing based on memory pressure."""
        if current_memory_usage > 0.9:
            # High memory pressure - checkpoint more layers
            additional_layers = min(2, len(self.model.layers) // 4)
            for i, layer in enumerate(self.model.layers):
                if i not in self.checkpointed_layers and len(self.checkpointed_layers) < len(self.model.layers) // 2:
                    self.checkpointed_layers.add(i)
                    self._wrap_layer_with_checkpointing(layer, i)
                    additional_layers -= 1
                    if additional_layers <= 0:
                        break
                        
        elif current_memory_usage < 0.6:
            # Low memory pressure - can reduce checkpointing
            if len(self.checkpointed_layers) > 1:
                layer_to_remove = max(self.checkpointed_layers)
                self.checkpointed_layers.remove(layer_to_remove)


class EfficientDataLoading:
    """Efficient data loading with prefetching and caching."""
    
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        self.dataloader = None
        self._setup_dataloader()
        
    def _setup_dataloader(self):
        """Setup optimized dataloader."""
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True  # For consistent batch sizes
        )
        
    def get_dataloader(self) -> DataLoader:
        """Get the optimized dataloader."""
        return self.dataloader
        
    def update_batch_size(self, new_batch_size: int):
        """Update batch size and recreate dataloader."""
        if new_batch_size != self.batch_size:
            self.batch_size = new_batch_size
            self._setup_dataloader()


class AdaptiveLossScaling:
    """Adaptive loss scaling for mixed precision training."""
    
    def __init__(
        self,
        initial_scale: float = 2**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        dynamic: bool = True
    ):
        self.scale = initial_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.dynamic = dynamic
        
        self._growth_tracker = 0
        self._overflow_tracker = 0
        
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        return loss * self.scale
        
    def unscale_gradients(self, optimizer: torch.optim.Optimizer) -> bool:
        """Unscale gradients and check for overflow."""
        inv_scale = 1.0 / self.scale
        
        found_inf = False
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.mul_(inv_scale)
                    if torch.isinf(param.grad.data).any() or torch.isnan(param.grad.data).any():
                        found_inf = True
                        
        return found_inf
        
    def update_scale(self, found_inf: bool):
        """Update loss scale based on overflow detection."""
        if found_inf:
            self._overflow_tracker += 1
            if self.dynamic:
                self.scale *= self.backoff_factor
                self._growth_tracker = 0
                logger.warning(f"Gradient overflow detected, reducing scale to {self.scale}")
        else:
            self._growth_tracker += 1
            if self.dynamic and self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0
                logger.info(f"Increasing loss scale to {self.scale}")
                
    def get_scale(self) -> float:
        """Get current loss scale."""
        return self.scale


class EfficientMoETrainer(MoETrainer):
    """Enhanced MoE trainer with efficiency optimizations."""
    
    def __init__(
        self,
        model: MoEModel,
        enable_dynamic_batching: bool = True,
        enable_adaptive_checkpointing: bool = True,
        enable_adaptive_scaling: bool = True,
        max_batch_size: int = 128,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        
        self.enable_dynamic_batching = enable_dynamic_batching
        self.enable_adaptive_checkpointing = enable_adaptive_checkpointing
        self.enable_adaptive_scaling = enable_adaptive_scaling
        
        # Initialize optimizations
        if enable_dynamic_batching:
            self.batch_scheduler = DynamicBatchSizeScheduler(
                max_batch_size=max_batch_size
            )
            
        if enable_adaptive_checkpointing:
            self.checkpointing = SelectiveActivationCheckpointing(model)
            
        if enable_adaptive_scaling and self.use_mixed_precision:
            self.adaptive_scaler = AdaptiveLossScaling()
            
        # Performance tracking
        self.performance_stats = {
            'batch_sizes': [],
            'step_times': [],
            'memory_usage': [],
            'throughput': []
        }
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        step_idx: int
    ) -> Dict[str, float]:
        """Optimized training step."""
        start_time = time.time()
        
        try:
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                outputs = self.model(**batch, return_routing_info=True)
                
                # Compute losses
                logits = self.model.lm_head(outputs.last_hidden_state)
                targets = batch['input_ids'][:, 1:].contiguous()
                logits = logits[:, :-1].contiguous()
                
                lm_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1)
                )
                
                # Add auxiliary losses
                aux_loss = outputs.load_balancing_loss if outputs.load_balancing_loss is not None else 0.0
                router_z_loss = outputs.router_z_loss if outputs.router_z_loss is not None else 0.0
                
                total_loss = lm_loss + self.aux_loss_coef * aux_loss + self.router_z_loss_coef * router_z_loss
                
            # Backward pass with scaling
            if self.use_mixed_precision:
                if hasattr(self, 'adaptive_scaler'):
                    scaled_loss = self.adaptive_scaler.scale_loss(total_loss)
                    scaled_loss.backward()
                    
                    # Check for gradient overflow
                    found_inf = self.adaptive_scaler.unscale_gradients(optimizer)
                    self.adaptive_scaler.update_scale(found_inf)
                    
                    if not found_inf:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        optimizer.step()
                else:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(optimizer)
                    self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                
            optimizer.zero_grad()
            
            # Update dynamic batch size
            if self.enable_dynamic_batching:
                self.batch_scheduler.update_batch_size(success=True)
                
            # Track performance
            step_time = time.time() - start_time
            self.performance_stats['step_times'].append(step_time)
            
            if torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated() / (1024**3)  # GB
                self.performance_stats['memory_usage'].append(memory_usage)
                
                # Adaptive checkpointing based on memory
                if self.enable_adaptive_checkpointing:
                    memory_ratio = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                    self.checkpointing.adaptive_checkpointing(memory_ratio)
                    
            batch_size = batch['input_ids'].size(0)
            seq_len = batch['input_ids'].size(1)
            throughput = (batch_size * seq_len) / step_time
            self.performance_stats['throughput'].append(throughput)
            self.performance_stats['batch_sizes'].append(batch_size)
            
            return {
                'loss': lm_loss.item(),
                'aux_loss': aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss,
                'router_z_loss': router_z_loss.item() if isinstance(router_z_loss, torch.Tensor) else router_z_loss,
                'step_time': step_time,
                'throughput': throughput,
                'memory_gb': memory_usage if torch.cuda.is_available() else 0.0
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("OOM detected during training step")
                
                # Clear cache and reduce batch size
                torch.cuda.empty_cache()
                
                if self.enable_dynamic_batching:
                    new_batch_size = self.batch_scheduler.update_batch_size(success=False, oom=True)
                    
                # Return failure metrics
                return {
                    'loss': float('inf'),
                    'aux_loss': 0.0,
                    'router_z_loss': 0.0,
                    'step_time': time.time() - start_time,
                    'throughput': 0.0,
                    'memory_gb': 0.0,
                    'oom': True
                }
            else:
                raise e
                
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get training performance summary."""
        if not self.performance_stats['step_times']:
            return {}
            
        import numpy as np
        
        return {
            'mean_step_time': np.mean(self.performance_stats['step_times']),
            'mean_throughput': np.mean(self.performance_stats['throughput']),
            'mean_memory_usage_gb': np.mean(self.performance_stats['memory_usage']) if self.performance_stats['memory_usage'] else 0.0,
            'mean_batch_size': np.mean(self.performance_stats['batch_sizes']),
            'total_steps': len(self.performance_stats['step_times']),
            'throughput_std': np.std(self.performance_stats['throughput']),
            'step_time_std': np.std(self.performance_stats['step_times'])
        }
        
    @contextmanager
    def profile_training(self):
        """Context manager for training profiling."""
        if torch.cuda.is_available():
            torch.cuda.profiler.start()
            
        start_time = time.time()
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.profiler.stop()
                
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f} seconds")
            
            # Print performance summary
            summary = self.get_performance_summary()
            if summary:
                logger.info("Performance Summary:")
                for key, value in summary.items():
                    logger.info(f"  {key}: {value:.4f}")


class DistributedOptimizations:
    """Optimizations for distributed MoE training."""
    
    def __init__(self, model: MoEModel):
        self.model = model
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
    def setup_expert_parallelism(self, expert_parallel_size: int = None):
        """Setup expert parallelism across devices."""
        if not dist.is_initialized():
            logger.warning("Distributed training not initialized")
            return
            
        if expert_parallel_size is None:
            expert_parallel_size = min(self.world_size, self.model.num_experts)
            
        # Create expert parallel groups
        experts_per_device = self.model.num_experts // expert_parallel_size
        
        for i, layer in enumerate(self.model.layers):
            if i in self.model.moe_layers:
                # Assign experts to devices
                start_expert = self.rank * experts_per_device
                end_expert = min(start_expert + experts_per_device, self.model.num_experts)
                
                # This would require modifying the expert pool to support distributed experts
                logger.info(f"Rank {self.rank}: handling experts {start_expert}-{end_expert-1}")
                
    def allreduce_gradients(self, model_parameters):
        """All-reduce gradients across devices."""
        if not dist.is_initialized() or self.world_size == 1:
            return
            
        for param in model_parameters:
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size
                
    def broadcast_model(self):
        """Broadcast model parameters from rank 0."""
        if not dist.is_initialized() or self.world_size == 1:
            return
            
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)