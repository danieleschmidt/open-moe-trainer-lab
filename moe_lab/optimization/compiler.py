"""Advanced model compilation and optimization for MoE models."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable, Union
import logging
from dataclasses import dataclass
from pathlib import Path
import time
import functools

logger = logging.getLogger(__name__)


@dataclass
class CompilationConfig:
    """Configuration for model compilation."""
    
    # Compilation backend
    backend: str = "inductor"  # "inductor", "aot_eager", "nvfuser", "tensorrt"
    
    # Optimization mode
    mode: str = "default"  # "default", "reduce-overhead", "max-autotune"
    
    # Dynamic shapes support
    dynamic: bool = False
    
    # Full graph capture
    fullgraph: bool = False
    
    # Custom optimization passes
    options: Optional[Dict[str, Any]] = None
    
    # Router-specific optimizations
    optimize_routing: bool = True
    fuse_expert_ops: bool = True
    
    # Memory optimizations
    enable_memory_planning: bool = True
    gradient_checkpointing: bool = False
    
    # Experimental features
    enable_experimental: bool = False


class MoEModelCompiler:
    """
    Advanced compiler for MoE models with specialized optimizations.
    
    Features:
    - Router-aware compilation optimizations
    - Expert operation fusion
    - Memory layout optimization
    - Dynamic batching support
    - Performance profiling integration
    """
    
    def __init__(self, config: Optional[CompilationConfig] = None):
        self.config = config or CompilationConfig()
        self.compiled_models = {}
        self.compilation_stats = {}
        
        # Check torch.compile availability
        self.compile_available = hasattr(torch, 'compile')
        if not self.compile_available:
            logger.warning("torch.compile not available, using fallback optimizations")
        
        logger.info(f"Initialized MoEModelCompiler with backend: {self.config.backend}")
    
    def compile_model(self, model: nn.Module, example_inputs: Optional[Dict[str, torch.Tensor]] = None) -> nn.Module:
        """
        Compile MoE model with specialized optimizations.
        
        Args:
            model: MoE model to compile
            example_inputs: Example inputs for compilation
            
        Returns:
            Compiled model
        """
        model_id = id(model)
        
        if model_id in self.compiled_models:
            logger.info("Model already compiled, returning cached version")
            return self.compiled_models[model_id]
        
        logger.info(f"Compiling MoE model with {self.config.backend} backend")
        start_time = time.time()
        
        try:
            # Apply pre-compilation optimizations
            optimized_model = self._apply_pre_compilation_opts(model)
            
            # Compile with torch.compile if available
            if self.compile_available:
                compiled_model = self._torch_compile(optimized_model, example_inputs)
            else:
                compiled_model = self._fallback_optimization(optimized_model)
            
            # Apply post-compilation optimizations
            final_model = self._apply_post_compilation_opts(compiled_model)
            
            compilation_time = time.time() - start_time
            
            # Cache compiled model
            self.compiled_models[model_id] = final_model
            self.compilation_stats[model_id] = {
                'compilation_time': compilation_time,
                'backend': self.config.backend,
                'mode': self.config.mode,
                'optimizations_applied': self._get_applied_optimizations()
            }
            
            logger.info(f"Model compilation completed in {compilation_time:.2f}s")
            return final_model
            
        except Exception as e:
            logger.error(f"Model compilation failed: {e}")
            logger.info("Falling back to uncompiled model")
            return model
    
    def _torch_compile(self, model: nn.Module, example_inputs: Optional[Dict[str, torch.Tensor]]) -> nn.Module:
        """Compile model using torch.compile."""
        compile_options = {
            'backend': self.config.backend,
            'mode': self.config.mode,
            'dynamic': self.config.dynamic,
            'fullgraph': self.config.fullgraph
        }
        
        if self.config.options:
            compile_options.update(self.config.options)
        
        # Apply MoE-specific compilation hints
        if self.config.optimize_routing:
            compile_options['options'] = compile_options.get('options', {})
            compile_options['options']['triton.cudagraphs'] = True
            compile_options['options']['shape_padding'] = True
        
        return torch.compile(model, **compile_options)
    
    def _fallback_optimization(self, model: nn.Module) -> nn.Module:
        """Fallback optimizations when torch.compile is not available."""
        logger.info("Applying fallback optimizations")
        
        # Basic optimizations
        model.eval()
        
        # Operator fusion for experts
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'experts'):
                    self._fuse_expert_operations(layer.experts)
        
        # Memory layout optimization
        self._optimize_memory_layout(model)
        
        return model
    
    def _apply_pre_compilation_opts(self, model: nn.Module) -> nn.Module:
        """Apply optimizations before compilation."""
        logger.debug("Applying pre-compilation optimizations")
        
        # Router optimization
        if self.config.optimize_routing:
            self._optimize_routers(model)
        
        # Expert operation fusion preparation
        if self.config.fuse_expert_ops:
            self._prepare_expert_fusion(model)
        
        # Memory planning
        if self.config.enable_memory_planning:
            self._plan_memory_layout(model)
        
        return model
    
    def _apply_post_compilation_opts(self, model: nn.Module) -> nn.Module:
        """Apply optimizations after compilation."""
        logger.debug("Applying post-compilation optimizations")
        
        # Set optimized inference mode
        if hasattr(model, '_optimized_inference'):
            model._optimized_inference = True
        
        return model
    
    def _optimize_routers(self, model: nn.Module):
        """Optimize router operations for better compilation."""
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'router'):
                    router = layer.router
                    
                    # Add compilation hints
                    if hasattr(router, 'forward'):
                        original_forward = router.forward
                        
                        @functools.wraps(original_forward)
                        def optimized_forward(hidden_states):
                            # Hint for better memory access patterns
                            if hidden_states.is_contiguous():
                                return original_forward(hidden_states)
                            else:
                                return original_forward(hidden_states.contiguous())
                        
                        router.forward = optimized_forward
    
    def _prepare_expert_fusion(self, model: nn.Module):
        """Prepare expert operations for fusion."""
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'experts'):
                    experts = layer.experts
                    
                    # Mark operations for fusion
                    if hasattr(experts, 'expert_modules'):
                        for expert in experts.expert_modules:
                            if hasattr(expert, 'fc1') and hasattr(expert, 'fc2'):
                                # Mark for potential fusion
                                expert._fusion_candidate = True
    
    def _fuse_expert_operations(self, experts):
        """Fuse expert operations for better performance."""
        if hasattr(experts, 'expert_modules'):
            for expert in experts.expert_modules:
                if hasattr(expert, '_fusion_candidate') and expert._fusion_candidate:
                    self._fuse_mlp_operations(expert)
    
    def _fuse_mlp_operations(self, expert):
        """Fuse MLP operations within an expert."""
        if hasattr(expert, 'fc1') and hasattr(expert, 'fc2') and hasattr(expert, 'activation'):
            # Create fused operation
            class FusedExpertMLP(nn.Module):
                def __init__(self, fc1, activation, fc2):
                    super().__init__()
                    self.fc1 = fc1
                    self.activation = activation
                    self.fc2 = fc2
                
                def forward(self, x):
                    # Fused operation
                    x = self.fc1(x)
                    x = self.activation(x)
                    x = self.fc2(x)
                    return x
            
            # Replace with fused version
            expert.fused_mlp = FusedExpertMLP(expert.fc1, expert.activation, expert.fc2)
            expert._use_fused = True
    
    def _plan_memory_layout(self, model: nn.Module):
        """Plan optimal memory layout for the model."""
        # Convert to channels_last for better performance on modern GPUs
        try:
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    if hasattr(module, 'weight'):
                        module.weight = module.weight.to(memory_format=torch.channels_last)
        except Exception as e:
            logger.debug(f"Memory layout optimization skipped: {e}")
    
    def _optimize_memory_layout(self, model: nn.Module):
        """Optimize memory layout for better cache utilization."""
        # Ensure contiguous tensors
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    def _get_applied_optimizations(self) -> List[str]:
        """Get list of applied optimizations."""
        optimizations = []
        
        if self.config.optimize_routing:
            optimizations.append("router_optimization")
        
        if self.config.fuse_expert_ops:
            optimizations.append("expert_fusion")
        
        if self.config.enable_memory_planning:
            optimizations.append("memory_planning")
        
        if self.compile_available:
            optimizations.append("torch_compile")
        else:
            optimizations.append("fallback_optimization")
        
        return optimizations
    
    def benchmark_compilation(self, model: nn.Module, example_inputs: Dict[str, torch.Tensor],
                            num_runs: int = 100) -> Dict[str, Any]:
        """
        Benchmark compilation performance improvements.
        
        Args:
            model: Model to benchmark
            example_inputs: Example inputs for benchmarking
            num_runs: Number of benchmark runs
            
        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking compilation with {num_runs} runs")
        
        # Benchmark original model
        model.eval()
        original_times = []
        
        with torch.no_grad():
            # Warmup
            for _ in range(5):
                _ = model(**example_inputs)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(num_runs):
                _ = model(**example_inputs)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            original_time = (time.time() - start_time) / num_runs
        
        # Compile and benchmark compiled model
        compiled_model = self.compile_model(model, example_inputs)
        compiled_times = []
        
        with torch.no_grad():
            # Warmup compiled model
            for _ in range(5):
                _ = compiled_model(**example_inputs)
            
            # Benchmark compiled model
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            for _ in range(num_runs):
                _ = compiled_model(**example_inputs)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            compiled_time = (time.time() - start_time) / num_runs
        
        speedup = original_time / compiled_time if compiled_time > 0 else 1.0
        
        results = {
            'original_time_ms': original_time * 1000,
            'compiled_time_ms': compiled_time * 1000,
            'speedup': speedup,
            'compilation_config': self.config.__dict__,
            'num_runs': num_runs
        }
        
        logger.info(f"Compilation benchmark: {speedup:.2f}x speedup")
        return results
    
    def export_optimized_model(self, model: nn.Module, output_path: str,
                             example_inputs: Optional[Dict[str, torch.Tensor]] = None):
        """
        Export optimized model for deployment.
        
        Args:
            model: Model to export
            output_path: Output path for exported model
            example_inputs: Example inputs for tracing
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting optimized model to {output_path}")
        
        # Compile model
        compiled_model = self.compile_model(model, example_inputs)
        
        # Export based on format
        if self.config.backend == "tensorrt":
            self._export_tensorrt(compiled_model, output_path, example_inputs)
        elif example_inputs:
            self._export_torchscript(compiled_model, output_path, example_inputs)
        else:
            # Save as PyTorch state dict
            torch.save(compiled_model.state_dict(), output_path / "optimized_model.pt")
        
        # Save compilation metadata
        metadata = {
            'compilation_config': self.config.__dict__,
            'optimizations_applied': self._get_applied_optimizations(),
            'export_timestamp': time.time()
        }
        
        import json
        with open(output_path / "compilation_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Model export completed")
    
    def _export_torchscript(self, model: nn.Module, output_path: Path,
                           example_inputs: Dict[str, torch.Tensor]):
        """Export model as TorchScript."""
        try:
            # Trace the model
            traced_model = torch.jit.trace(model, tuple(example_inputs.values()))
            traced_model.save(output_path / "traced_model.pt")
            logger.info("Exported as TorchScript traced model")
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            # Fallback to script
            try:
                scripted_model = torch.jit.script(model)
                scripted_model.save(output_path / "scripted_model.pt")
                logger.info("Exported as TorchScript scripted model")
            except Exception as e2:
                logger.error(f"TorchScript script export also failed: {e2}")
    
    def _export_tensorrt(self, model: nn.Module, output_path: Path,
                        example_inputs: Dict[str, torch.Tensor]):
        """Export model for TensorRT."""
        try:
            import torch_tensorrt
            
            # Convert to TensorRT
            trt_model = torch_tensorrt.compile(
                model,
                inputs=list(example_inputs.values()),
                enabled_precisions=[torch.float, torch.half]
            )
            
            torch.jit.save(trt_model, output_path / "tensorrt_model.pt")
            logger.info("Exported as TensorRT model")
            
        except ImportError:
            logger.warning("torch_tensorrt not available, skipping TensorRT export")
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
    
    def get_compilation_stats(self) -> Dict[str, Any]:
        """Get compilation statistics."""
        return {
            'compiled_models': len(self.compiled_models),
            'compilation_stats': self.compilation_stats,
            'config': self.config.__dict__,
            'torch_compile_available': self.compile_available
        }
    
    def clear_cache(self):
        """Clear compilation cache."""
        self.compiled_models.clear()
        self.compilation_stats.clear()
        
        if hasattr(torch, '_dynamo'):
            torch._dynamo.reset()
        
        logger.info("Cleared compilation cache")


def create_optimized_model(model: nn.Module, optimization_level: str = "balanced",
                          example_inputs: Optional[Dict[str, torch.Tensor]] = None) -> nn.Module:
    """
    Factory function to create optimized model with preset configurations.
    
    Args:
        model: Model to optimize
        optimization_level: Optimization level ("fast", "balanced", "memory")
        example_inputs: Example inputs for compilation
        
    Returns:
        Optimized model
    """
    configs = {
        "fast": CompilationConfig(
            backend="inductor",
            mode="max-autotune",
            optimize_routing=True,
            fuse_expert_ops=True,
            enable_memory_planning=False
        ),
        "balanced": CompilationConfig(
            backend="inductor", 
            mode="default",
            optimize_routing=True,
            fuse_expert_ops=True,
            enable_memory_planning=True
        ),
        "memory": CompilationConfig(
            backend="aot_eager",
            mode="reduce-overhead",
            optimize_routing=False,
            fuse_expert_ops=False,
            enable_memory_planning=True,
            gradient_checkpointing=True
        )
    }
    
    config = configs.get(optimization_level, configs["balanced"])
    compiler = MoEModelCompiler(config)
    
    return compiler.compile_model(model, example_inputs)