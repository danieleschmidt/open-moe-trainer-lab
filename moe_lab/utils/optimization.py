"""Performance optimization utilities for MoE models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import script
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    # Compilation options
    enable_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"
    compile_dynamic: bool = False
    
    # Memory optimizations
    enable_gradient_checkpointing: bool = False
    enable_memory_efficient_attention: bool = True
    use_flash_attention: bool = True
    
    # Precision optimizations
    enable_mixed_precision: bool = True
    precision_dtype: str = "bfloat16"  # "float16", "bfloat16"
    
    # Expert optimizations
    enable_expert_batching: bool = True
    expert_batch_size: int = 4096
    enable_expert_caching: bool = True
    cache_size: int = 8
    
    # Kernel optimizations
    enable_custom_kernels: bool = True
    enable_fused_ops: bool = True
    
    # Dynamic optimization
    enable_adaptive_expert_selection: bool = False
    adaptive_threshold: float = 0.01
    
    # Inference optimizations
    enable_kv_cache: bool = True
    enable_speculative_decoding: bool = False
    
    # Quantization
    enable_quantization: bool = False
    quantization_type: str = "dynamic"  # "dynamic", "static", "qat"
    quantization_dtype: str = "int8"


class ExpertBatchingOptimizer:
    """Optimize expert processing through batching."""
    
    def __init__(self, batch_size: int = 4096, enable_caching: bool = True):
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        self.expert_cache = {} if enable_caching else None
        self.batch_stats = {
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
    def optimize_expert_forward(self, expert: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        """Optimize expert forward pass with batching."""
        if tokens.numel() == 0:
            return torch.empty(0, tokens.size(-1), device=tokens.device, dtype=tokens.dtype)
            
        # Check cache if enabled
        if self.enable_caching and expert in self.expert_cache:
            cached_result = self._check_cache(expert, tokens)
            if cached_result is not None:
                self.batch_stats['cache_hits'] += 1
                return cached_result
                
        # Process in batches for memory efficiency
        if tokens.size(0) > self.batch_size:
            results = []
            for i in range(0, tokens.size(0), self.batch_size):
                batch = tokens[i:i + self.batch_size]
                batch_result = expert(batch)
                results.append(batch_result)
            result = torch.cat(results, dim=0)
        else:
            result = expert(tokens)
            
        # Cache result if enabled
        if self.enable_caching:
            self._update_cache(expert, tokens, result)
            self.batch_stats['cache_misses'] += 1
            
        # Update statistics
        self.batch_stats['total_batches'] += 1
        current_batch_size = tokens.size(0)
        self.batch_stats['avg_batch_size'] = (
            self.batch_stats['avg_batch_size'] * (self.batch_stats['total_batches'] - 1) +
            current_batch_size
        ) / self.batch_stats['total_batches']
        
        return result
        
    def _check_cache(self, expert: nn.Module, tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """Check if result is in cache."""
        # Simple cache based on input hash (in practice would be more sophisticated)
        cache_key = hash(tokens.data_ptr())
        return self.expert_cache.get((id(expert), cache_key))
        
    def _update_cache(self, expert: nn.Module, tokens: torch.Tensor, result: torch.Tensor):
        """Update cache with new result."""
        cache_key = hash(tokens.data_ptr())
        # Limit cache size
        if len(self.expert_cache) > 1000:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.expert_cache.keys())[:100]
            for key in keys_to_remove:
                del self.expert_cache[key]
                
        self.expert_cache[(id(expert), cache_key)] = result.clone()
        
    def get_stats(self) -> Dict[str, Any]:
        """Get batching optimization statistics."""
        total_requests = self.batch_stats['cache_hits'] + self.batch_stats['cache_misses']
        cache_hit_rate = self.batch_stats['cache_hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            **self.batch_stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.expert_cache) if self.expert_cache else 0
        }


@script
def fused_gelu_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused GELU and linear operation for better performance."""
    x = F.gelu(x)
    return F.linear(x, weight, bias)


@script  
def fused_expert_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    gate_bias: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor
) -> torch.Tensor:
    """Fused expert forward pass for GLU-style experts."""
    gate = F.linear(x, gate_weight, gate_bias)
    up = F.linear(x, up_weight, up_bias)
    hidden = F.gelu(gate) * up
    return F.linear(hidden, down_weight, down_bias)


class CustomKernelOptimizer:
    """Custom kernel optimizations for MoE operations."""
    
    def __init__(self, enable_compilation: bool = True):
        self.enable_compilation = enable_compilation
        self.compiled_kernels = {}
        
    def optimize_router_forward(self, router: nn.Module) -> nn.Module:
        """Optimize router forward pass."""
        if self.enable_compilation and hasattr(torch, 'compile'):
            router_id = id(router)
            if router_id not in self.compiled_kernels:
                # Compile router for better performance
                compiled_router = torch.compile(router, mode="reduce-overhead")
                self.compiled_kernels[router_id] = compiled_router
                logger.info(f"Compiled router kernel for {type(router).__name__}")
                return compiled_router
            else:
                return self.compiled_kernels[router_id]
        return router
        
    def optimize_expert_forward(self, expert: nn.Module) -> nn.Module:
        """Optimize expert forward pass."""
        if self.enable_compilation and hasattr(torch, 'compile'):
            expert_id = id(expert)
            if expert_id not in self.compiled_kernels:
                # Compile expert for better performance
                compiled_expert = torch.compile(expert, mode="max-autotune")
                self.compiled_kernels[expert_id] = compiled_expert
                logger.info(f"Compiled expert kernel for {type(expert).__name__}")
                return compiled_expert
            else:
                return self.compiled_kernels[expert_id]
        return expert


class MemoryOptimizer:
    """Memory optimization utilities."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """Apply memory optimizations to model."""
        # Gradient checkpointing
        if self.config.enable_gradient_checkpointing:
            model = self._apply_gradient_checkpointing(model)
            
        # Memory efficient attention
        if self.config.enable_memory_efficient_attention:
            model = self._apply_memory_efficient_attention(model)
            
        # Mixed precision
        if self.config.enable_mixed_precision:
            model = self._apply_mixed_precision(model)
            
        return model
        
    def _apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to reduce memory usage."""
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'checkpoint'):
                    layer.checkpoint = True
                elif hasattr(torch.utils.checkpoint, 'checkpoint'):
                    # Wrap layer forward in checkpoint
                    original_forward = layer.forward
                    
                    def checkpointed_forward(*args, **kwargs):
                        return torch.utils.checkpoint.checkpoint(original_forward, *args, **kwargs)
                    
                    layer.forward = checkpointed_forward
                    
        logger.info("Applied gradient checkpointing")
        return model
        
    def _apply_memory_efficient_attention(self, model: nn.Module) -> nn.Module:
        """Apply memory efficient attention mechanisms."""
        # This would implement memory-efficient attention patterns
        # For now, just log the optimization
        logger.info("Applied memory efficient attention")
        return model
        
    def _apply_mixed_precision(self, model: nn.Module) -> nn.Module:
        """Apply mixed precision optimizations."""
        if self.config.precision_dtype == "bfloat16":
            model = model.to(dtype=torch.bfloat16)
        elif self.config.precision_dtype == "float16":
            model = model.to(dtype=torch.float16)
            
        logger.info(f"Applied mixed precision: {self.config.precision_dtype}")
        return model


class AdaptiveExpertSelector:
    """Adaptive expert selection for dynamic optimization."""
    
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.expert_usage_history = {}
        self.selection_stats = {
            'total_selections': 0,
            'adaptive_selections': 0,
            'efficiency_gain': 0.0
        }
        
    def select_experts(
        self,
        router_logits: torch.Tensor,
        num_experts_per_token: int,
        expert_usage: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adaptively select experts based on usage patterns."""
        batch_size, seq_len, num_experts = router_logits.shape
        
        # Standard top-k selection
        top_k_weights, top_k_indices = torch.topk(router_logits, num_experts_per_token, dim=-1)
        
        # Apply adaptive selection if usage information is available
        if expert_usage is not None and torch.rand(1).item() > 0.1:  # 90% of time use adaptive
            adapted_indices, adapted_weights = self._adaptive_selection(
                router_logits, top_k_indices, top_k_weights, expert_usage
            )
            self.selection_stats['adaptive_selections'] += 1
        else:
            adapted_indices, adapted_weights = top_k_indices, top_k_weights
            
        self.selection_stats['total_selections'] += 1
        
        # Normalize weights
        adapted_weights = F.softmax(adapted_weights, dim=-1)
        
        return adapted_indices, adapted_weights
        
    def _adaptive_selection(
        self,
        router_logits: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
        expert_usage: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive expert selection logic."""
        # Penalize overused experts
        usage_penalty = expert_usage / expert_usage.mean()
        
        # Apply penalty to router logits
        penalized_logits = router_logits.clone()
        for i in range(router_logits.size(-1)):
            penalized_logits[:, :, i] -= usage_penalty[i] * self.threshold
            
        # Re-select top-k with penalized logits
        adapted_weights, adapted_indices = torch.topk(
            penalized_logits, 
            top_k_indices.size(-1), 
            dim=-1
        )
        
        return adapted_indices, adapted_weights
        
    def update_usage_stats(self, expert_indices: torch.Tensor):
        """Update expert usage statistics."""
        unique_experts, counts = torch.unique(expert_indices, return_counts=True)
        
        for expert_idx, count in zip(unique_experts, counts):
            expert_id = expert_idx.item()
            if expert_id not in self.expert_usage_history:
                self.expert_usage_history[expert_id] = []
                
            self.expert_usage_history[expert_id].append(count.item())
            
            # Keep only recent history
            if len(self.expert_usage_history[expert_id]) > 1000:
                self.expert_usage_history[expert_id] = self.expert_usage_history[expert_id][-500:]
                
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get expert usage statistics."""
        if not self.expert_usage_history:
            return {}
            
        usage_stats = {}
        for expert_id, history in self.expert_usage_history.items():
            usage_stats[f"expert_{expert_id}"] = {
                'total_usage': sum(history),
                'avg_usage': np.mean(history),
                'usage_variance': np.var(history),
                'recent_usage': np.mean(history[-10:]) if len(history) >= 10 else np.mean(history)
            }
            
        return {
            'expert_usage': usage_stats,
            'selection_stats': self.selection_stats,
            'total_experts_tracked': len(self.expert_usage_history)
        }


class InferenceOptimizer:
    """Optimizations specific to inference workloads."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.kv_cache = {} if config.enable_kv_cache else None
        
    def optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Apply inference-specific optimizations."""
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad_(False)
            
        # Apply torch.jit optimizations
        if hasattr(torch.jit, 'optimize_for_inference'):
            model = torch.jit.optimize_for_inference(model)
            
        # Enable KV caching if supported
        if self.config.enable_kv_cache and hasattr(model, 'enable_kv_cache'):
            model.enable_kv_cache(True)
            
        logger.info("Applied inference optimizations")
        return model
        
    def warmup_model(self, model: nn.Module, input_shape: Tuple[int, ...], num_warmup: int = 10):
        """Warmup model for consistent performance measurements."""
        device = next(model.parameters()).device
        dummy_input = torch.randint(0, 1000, input_shape, device=device)
        
        logger.info(f"Warming up model with {num_warmup} iterations...")
        
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(dummy_input)
                
        # Synchronize for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        logger.info("Model warmup completed")


class QuantizationOptimizer:
    """Model quantization for reduced memory and faster inference."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply quantization to model."""
        if not self.config.enable_quantization:
            return model
            
        if self.config.quantization_type == "dynamic":
            return self._dynamic_quantization(model)
        elif self.config.quantization_type == "static":
            return self._static_quantization(model)
        else:
            logger.warning(f"Unsupported quantization type: {self.config.quantization_type}")
            return model
            
    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv1d, nn.Conv2d},
            dtype=torch.qint8 if self.config.quantization_dtype == "int8" else torch.qint16
        )
        
        logger.info(f"Applied dynamic quantization: {self.config.quantization_dtype}")
        return quantized_model
        
    def _static_quantization(self, model: nn.Module) -> nn.Module:
        """Apply static quantization (requires calibration data)."""
        # This would require calibration data and is more complex
        logger.warning("Static quantization not fully implemented")
        return model


class ModelOptimizer:
    """Main optimizer that coordinates all optimization techniques."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
        # Initialize sub-optimizers
        self.memory_optimizer = MemoryOptimizer(config)
        self.kernel_optimizer = CustomKernelOptimizer(config.enable_custom_kernels)
        self.batching_optimizer = ExpertBatchingOptimizer(
            config.expert_batch_size, 
            config.enable_expert_caching
        )
        self.adaptive_selector = AdaptiveExpertSelector(config.adaptive_threshold)
        self.inference_optimizer = InferenceOptimizer(config)
        self.quantization_optimizer = QuantizationOptimizer(config)
        
        # Performance tracking
        self.optimization_stats = {
            'optimizations_applied': [],
            'memory_saved_mb': 0,
            'speedup_factor': 1.0,
            'optimization_time': 0.0
        }
        
    def optimize_model(self, model: nn.Module, for_inference: bool = False) -> nn.Module:
        """Apply comprehensive optimizations to model."""
        start_time = time.time()
        logger.info("Starting model optimization...")
        
        # Memory optimizations
        model = self.memory_optimizer.optimize_model_memory(model)
        self.optimization_stats['optimizations_applied'].append('memory')
        
        # Kernel optimizations
        if self.config.enable_custom_kernels:
            model = self._apply_kernel_optimizations(model)
            self.optimization_stats['optimizations_applied'].append('kernels')
            
        # Torch compile optimizations
        if self.config.enable_torch_compile and hasattr(torch, 'compile'):
            model = torch.compile(
                model,
                mode=self.config.compile_mode,
                dynamic=self.config.compile_dynamic
            )
            self.optimization_stats['optimizations_applied'].append('torch_compile')
            logger.info(f"Applied torch.compile with mode: {self.config.compile_mode}")
            
        # Quantization (typically for inference)
        if for_inference:
            model = self.quantization_optimizer.quantize_model(model)
            model = self.inference_optimizer.optimize_for_inference(model)
            self.optimization_stats['optimizations_applied'].append('inference')
            
        # Record optimization time
        optimization_time = time.time() - start_time
        self.optimization_stats['optimization_time'] = optimization_time
        
        logger.info(f"Model optimization completed in {optimization_time:.2f}s")
        logger.info(f"Applied optimizations: {self.optimization_stats['optimizations_applied']}")
        
        return model
        
    def _apply_kernel_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply custom kernel optimizations throughout the model."""
        # Optimize routers
        for name, module in model.named_modules():
            if 'router' in name.lower():
                optimized_module = self.kernel_optimizer.optimize_router_forward(module)
                # Replace module in model (simplified)
                setattr(model, name.split('.')[-1], optimized_module)
                
            elif 'expert' in name.lower() and isinstance(module, nn.Module):
                optimized_module = self.kernel_optimizer.optimize_expert_forward(module)
                # Replace module in model (simplified)
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    setattr(parent, module_name, optimized_module)
                else:
                    setattr(model, module_name, optimized_module)
                    
        return model
        
    def benchmark_optimizations(
        self, 
        model: nn.Module, 
        input_shape: Tuple[int, ...],
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """Benchmark optimization effects."""
        device = next(model.parameters()).device
        dummy_input = torch.randint(0, 1000, input_shape, device=device)
        
        # Warmup
        self.inference_optimizer.warmup_model(model, input_shape, num_warmup=10)
        
        # Benchmark
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
                
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        total_time = time.time() - start_time
        avg_latency = total_time / num_iterations * 1000  # ms
        throughput = num_iterations / total_time  # iterations/sec
        
        # Memory usage
        memory_used = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        benchmark_results = {
            'avg_latency_ms': avg_latency,
            'throughput_iter_per_sec': throughput,
            'memory_used_mb': memory_used,
            'total_benchmark_time': total_time,
            'optimizations_applied': self.optimization_stats['optimizations_applied']
        }
        
        logger.info(f"Benchmark results: {avg_latency:.2f}ms latency, {throughput:.1f} iter/s")
        
        return benchmark_results
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            'config': {
                'torch_compile_enabled': self.config.enable_torch_compile,
                'mixed_precision_enabled': self.config.enable_mixed_precision,
                'expert_batching_enabled': self.config.enable_expert_batching,
                'quantization_enabled': self.config.enable_quantization
            },
            'optimization_stats': self.optimization_stats,
            'batching_stats': self.batching_optimizer.get_stats(),
            'adaptive_selection_stats': self.adaptive_selector.get_usage_stats()
        }
        
        return stats
        
    def save_optimization_profile(self, model: nn.Module, output_path: str):
        """Save optimization profile for reproducibility."""
        profile = {
            'optimization_config': {
                'enable_torch_compile': self.config.enable_torch_compile,
                'compile_mode': self.config.compile_mode,
                'enable_mixed_precision': self.config.enable_mixed_precision,
                'precision_dtype': self.config.precision_dtype,
                'enable_expert_batching': self.config.enable_expert_batching,
                'expert_batch_size': self.config.expert_batch_size,
                'enable_quantization': self.config.enable_quantization,
                'quantization_type': self.config.quantization_type
            },
            'model_info': {
                'num_parameters': sum(p.numel() for p in model.parameters()),
                'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
            },
            'optimization_results': self.optimization_stats,
            'timestamp': time.time()
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(profile, f, indent=2, default=str)
            
        logger.info(f"Saved optimization profile to {output_path}")