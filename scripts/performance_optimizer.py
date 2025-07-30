#!/usr/bin/env python3
"""
Advanced Performance Optimization Script for Open MoE Trainer Lab

This script provides comprehensive performance optimization capabilities including:
- GPU memory optimization
- Distributed training configuration
- Expert caching strategies
- Performance profiling and monitoring
- Auto-tuning for different hardware configurations
"""

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil
import torch


class PerformanceOptimizer:
    """Advanced performance optimization for MoE training and inference."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "performance_config.json"
        self.hardware_info = self._detect_hardware()
        self.optimization_strategies = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('performance_optimization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _detect_hardware(self) -> Dict:
        """Detect and analyze hardware configuration."""
        hardware = {
            "platform": platform.platform(),
            "cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "gpu_count": 0,
            "gpu_memory_gb": 0,
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
        }
        
        if torch.cuda.is_available():
            hardware["gpu_count"] = torch.cuda.device_count()
            hardware["gpu_names"] = [torch.cuda.get_device_name(i) 
                                   for i in range(torch.cuda.device_count())]
            hardware["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            hardware["cuda_version"] = torch.version.cuda
            
        return hardware
    
    def generate_optimal_config(self) -> Dict:
        """Generate optimal configuration based on detected hardware."""
        config = {
            "hardware_profile": self._classify_hardware(),
            "training_config": self._optimize_training_config(),
            "inference_config": self._optimize_inference_config(),
            "memory_config": self._optimize_memory_config(),
            "distributed_config": self._optimize_distributed_config(),
            "caching_config": self._optimize_caching_config(),
            "monitoring_config": self._generate_monitoring_config(),
        }
        
        self.logger.info(f"Generated optimization config for {config['hardware_profile']} hardware")
        return config
    
    def _classify_hardware(self) -> str:
        """Classify hardware into performance tiers."""
        gpu_memory = self.hardware_info.get("gpu_memory_gb", 0)
        gpu_count = self.hardware_info.get("gpu_count", 0)
        system_memory = self.hardware_info.get("memory_gb", 0)
        
        if gpu_count >= 8 and gpu_memory >= 40:
            return "enterprise_multi_gpu"
        elif gpu_count >= 4 and gpu_memory >= 24:
            return "high_end_multi_gpu"
        elif gpu_count >= 2 and gpu_memory >= 16:
            return "mid_range_multi_gpu"
        elif gpu_count == 1 and gpu_memory >= 12:
            return "single_gpu_high_memory"
        elif gpu_count == 1 and gpu_memory >= 8:
            return "single_gpu_standard"
        elif system_memory >= 32:
            return "cpu_only_high_memory"
        else:
            return "resource_constrained"
    
    def _optimize_training_config(self) -> Dict:
        """Optimize training configuration based on hardware."""
        hardware_profile = self._classify_hardware()
        
        configs = {
            "enterprise_multi_gpu": {
                "batch_size": 64,
                "gradient_accumulation_steps": 2,
                "num_experts": 64,
                "experts_per_token": 4,
                "hidden_size": 4096,
                "num_layers": 32,
                "use_amp": True,
                "amp_dtype": "bfloat16",
                "expert_parallel_size": min(8, self.hardware_info["gpu_count"]),
                "model_parallel_size": 2,
                "pipeline_parallel_size": 2,
            },
            "high_end_multi_gpu": {
                "batch_size": 32,
                "gradient_accumulation_steps": 4,
                "num_experts": 32,
                "experts_per_token": 2,
                "hidden_size": 2048,
                "num_layers": 24,
                "use_amp": True,
                "amp_dtype": "float16",
                "expert_parallel_size": min(4, self.hardware_info["gpu_count"]),
                "model_parallel_size": 2,
                "pipeline_parallel_size": 1,
            },
            "mid_range_multi_gpu": {
                "batch_size": 16,
                "gradient_accumulation_steps": 4,
                "num_experts": 16,
                "experts_per_token": 2,
                "hidden_size": 1024,
                "num_layers": 16,
                "use_amp": True,
                "amp_dtype": "float16",
                "expert_parallel_size": self.hardware_info["gpu_count"],
                "model_parallel_size": 1,
                "pipeline_parallel_size": 1,
            },
            "single_gpu_high_memory": {
                "batch_size": 8,
                "gradient_accumulation_steps": 8,
                "num_experts": 8,
                "experts_per_token": 2,
                "hidden_size": 768,
                "num_layers": 12,
                "use_amp": True,
                "amp_dtype": "float16",
                "expert_parallel_size": 1,
                "model_parallel_size": 1,
                "pipeline_parallel_size": 1,
            },
            "single_gpu_standard": {
                "batch_size": 4,
                "gradient_accumulation_steps": 16,
                "num_experts": 4,
                "experts_per_token": 2,
                "hidden_size": 512,
                "num_layers": 8,
                "use_amp": True,
                "amp_dtype": "float16",
                "expert_parallel_size": 1,
                "model_parallel_size": 1,
                "pipeline_parallel_size": 1,
            }
        }
        
        return configs.get(hardware_profile, configs["single_gpu_standard"])
    
    def _optimize_inference_config(self) -> Dict:
        """Optimize inference configuration."""
        gpu_memory = self.hardware_info.get("gpu_memory_gb", 0)
        
        if gpu_memory >= 40:
            expert_cache_size = 16
            preload_experts = 8
        elif gpu_memory >= 24:
            expert_cache_size = 8
            preload_experts = 4
        elif gpu_memory >= 12:
            expert_cache_size = 4
            preload_experts = 2
        else:
            expert_cache_size = 2
            preload_experts = 1
            
        return {
            "use_expert_cache": True,
            "expert_cache_size_gb": expert_cache_size,
            "cache_policy": "weighted_lru",
            "preload_top_k_experts": preload_experts,
            "use_torch_compile": torch.__version__ >= "2.0.0",
            "compile_backend": "inductor",
            "use_quantization": gpu_memory < 16,
            "quantization_method": "gptq" if gpu_memory >= 8 else "dynamic",
            "quantization_bits": 4 if gpu_memory < 12 else 8,
        }
    
    def _optimize_memory_config(self) -> Dict:
        """Optimize memory usage configuration."""
        total_memory = self.hardware_info.get("memory_gb", 0)
        gpu_memory = self.hardware_info.get("gpu_memory_gb", 0)
        
        return {
            "cpu_memory_limit_gb": max(4, total_memory * 0.8),
            "gpu_memory_fraction": 0.9 if gpu_memory > 0 else 0,
            "enable_memory_pool": True,
            "gradient_checkpointing": gpu_memory < 24,
            "offload_optimizer": gpu_memory < 16,
            "offload_parameters": gpu_memory < 12,
            "max_split_size_mb": 128 if gpu_memory >= 16 else 64,
        }
    
    def _optimize_distributed_config(self) -> Dict:
        """Optimize distributed training configuration."""
        gpu_count = self.hardware_info.get("gpu_count", 0)
        
        if gpu_count <= 1:
            return {"enabled": False}
            
        return {
            "enabled": True,
            "backend": "nccl" if gpu_count > 0 else "gloo",
            "init_method": "env://",
            "world_size": gpu_count,
            "find_unused_parameters": False,
            "ddp_bucket_cap_mb": 25,
            "gradient_as_bucket_view": True,
            "static_graph": True,
        }
    
    def _optimize_caching_config(self) -> Dict:
        """Optimize caching strategies."""
        available_memory = self.hardware_info.get("available_memory_gb", 0)
        
        return {
            "model_cache_size_gb": min(8, available_memory * 0.3),
            "dataset_cache_size_gb": min(16, available_memory * 0.4),
            "expert_weight_cache_size_gb": min(4, available_memory * 0.2),
            "routing_cache_size_mb": 512,
            "enable_disk_cache": True,
            "cache_compression": True,
            "cache_eviction_policy": "lru",
        }
    
    def _generate_monitoring_config(self) -> Dict:
        """Generate monitoring configuration."""
        return {
            "enable_profiling": True,
            "profile_memory": True,
            "profile_compute": True,
            "monitor_gpu_utilization": self.hardware_info["cuda_available"],
            "metrics_collection_interval": 30,
            "performance_logging": True,
            "enable_tensorboard": True,
            "enable_wandb": True,
            "alert_thresholds": {
                "memory_usage_percent": 90,
                "gpu_utilization_percent": 95,
                "temperature_celsius": 85,
            }
        }
    
    def apply_optimizations(self, config: Dict) -> bool:
        """Apply performance optimizations to the system."""
        try:
            self._apply_torch_optimizations(config)
            self._apply_system_optimizations(config)
            self._apply_cuda_optimizations(config)
            
            self.logger.info("Successfully applied performance optimizations")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply optimizations: {e}")
            return False
    
    def _apply_torch_optimizations(self, config: Dict):
        """Apply PyTorch-specific optimizations."""
        memory_config = config.get("memory_config", {})
        
        # Set memory fraction
        if torch.cuda.is_available() and memory_config.get("gpu_memory_fraction"):
            torch.cuda.set_per_process_memory_fraction(
                memory_config["gpu_memory_fraction"]
            )
        
        # Enable memory pool
        if memory_config.get("enable_memory_pool"):
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # Set thread counts
        if config.get("training_config", {}).get("expert_parallel_size", 1) > 1:
            torch.set_num_threads(min(4, self.hardware_info["physical_cpu_count"]))
    
    def _apply_system_optimizations(self, config: Dict):
        """Apply system-level optimizations."""
        # Set OMP threads
        os.environ["OMP_NUM_THREADS"] = str(
            min(8, self.hardware_info["physical_cpu_count"])
        )
        
        # Set MKL threads
        os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
        
        # Enable memory debugging if needed
        memory_config = config.get("memory_config", {})
        if memory_config.get("gradient_checkpointing"):
            os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    
    def _apply_cuda_optimizations(self, config: Dict):
        """Apply CUDA-specific optimizations."""
        if not torch.cuda.is_available():
            return
            
        # Set CUDA launch blocking for debugging
        if config.get("monitoring_config", {}).get("enable_profiling"):
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        
        # Set NCCL debug level
        distributed_config = config.get("distributed_config", {})
        if distributed_config.get("enabled"):
            os.environ["NCCL_DEBUG"] = "WARN"
            
        # Enable CuDNN benchmark
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    def benchmark_configuration(self, config: Dict) -> Dict:
        """Benchmark the current configuration."""
        self.logger.info("Starting performance benchmark...")
        
        results = {
            "hardware_info": self.hardware_info,
            "config": config,
            "benchmarks": {},
            "timestamp": str(pd.Timestamp.now()),
        }
        
        # Memory benchmark
        results["benchmarks"]["memory"] = self._benchmark_memory()
        
        # Compute benchmark
        if torch.cuda.is_available():
            results["benchmarks"]["gpu_compute"] = self._benchmark_gpu_compute()
            
        # Training benchmark
        results["benchmarks"]["training"] = self._benchmark_training(config)
        
        self.logger.info("Benchmark completed")
        return results
    
    def _benchmark_memory(self) -> Dict:
        """Benchmark memory performance."""
        import time
        
        # Allocate and time memory operations
        start_time = time.time()
        
        # System memory test
        test_size = min(1024**3, self.hardware_info["available_memory_gb"] * 0.1 * 1024**3)
        test_array = bytearray(int(test_size))
        memory_alloc_time = time.time() - start_time
        
        # GPU memory test if available
        gpu_alloc_time = 0
        if torch.cuda.is_available():
            start_time = time.time()
            gpu_tensor = torch.randn(1000, 1000, device='cuda')
            gpu_alloc_time = time.time() - start_time
            del gpu_tensor
            
        del test_array
        
        return {
            "system_memory_alloc_time": memory_alloc_time,
            "gpu_memory_alloc_time": gpu_alloc_time,
            "memory_bandwidth_gbps": test_size / memory_alloc_time / 1024**3 if memory_alloc_time > 0 else 0,
        }
    
    def _benchmark_gpu_compute(self) -> Dict:
        """Benchmark GPU compute performance."""
        if not torch.cuda.is_available():
            return {}
            
        import time
        
        device = torch.device('cuda')
        
        # Matrix multiplication benchmark
        start_time = time.time()
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        
        for _ in range(100):
            c = torch.matmul(a, b)
            
        torch.cuda.synchronize()
        compute_time = time.time() - start_time
        
        # Memory bandwidth test
        start_time = time.time()
        large_tensor = torch.randn(10000, 10000, device=device)
        torch.cuda.synchronize()
        memory_time = time.time() - start_time
        
        return {
            "compute_time_100_matmuls": compute_time,
            "memory_bandwidth_test_time": memory_time,
            "estimated_tflops": (100 * 2 * 1024**3) / compute_time / 1e12,
        }
    
    def _benchmark_training(self, config: Dict) -> Dict:
        """Benchmark training performance with given config."""
        # Simplified training benchmark
        training_config = config.get("training_config", {})
        
        return {
            "estimated_tokens_per_second": self._estimate_tokens_per_second(training_config),
            "estimated_memory_usage_gb": self._estimate_memory_usage(training_config),
            "configuration_score": self._score_configuration(config),
        }
    
    def _estimate_tokens_per_second(self, training_config: Dict) -> float:
        """Estimate training throughput."""
        base_throughput = 1000  # Base tokens per second
        
        # Scale by hardware
        hardware_multiplier = {
            "enterprise_multi_gpu": 16.0,
            "high_end_multi_gpu": 8.0,
            "mid_range_multi_gpu": 4.0,
            "single_gpu_high_memory": 2.0,
            "single_gpu_standard": 1.0,
            "cpu_only_high_memory": 0.1,
            "resource_constrained": 0.05,
        }
        
        hardware_profile = self._classify_hardware()
        multiplier = hardware_multiplier.get(hardware_profile, 1.0)
        
        return base_throughput * multiplier
    
    def _estimate_memory_usage(self, training_config: Dict) -> float:
        """Estimate memory usage for configuration."""
        hidden_size = training_config.get("hidden_size", 768)
        num_experts = training_config.get("num_experts", 8)
        num_layers = training_config.get("num_layers", 12)
        batch_size = training_config.get("batch_size", 16)
        
        # Rough estimation
        param_memory = (hidden_size * hidden_size * num_experts * num_layers * 4) / 1024**3
        activation_memory = (batch_size * 2048 * hidden_size * 4) / 1024**3
        
        return param_memory + activation_memory * 2  # Factor of 2 for gradients
    
    def _score_configuration(self, config: Dict) -> float:
        """Score the configuration for the current hardware."""
        hardware_profile = self._classify_hardware()
        training_config = config.get("training_config", {})
        
        # Score based on resource utilization
        memory_score = min(1.0, self._estimate_memory_usage(training_config) / 
                          self.hardware_info.get("gpu_memory_gb", 1))
        
        throughput_score = self._estimate_tokens_per_second(training_config) / 10000
        
        return (memory_score * 0.4 + throughput_score * 0.6) * 100
    
    def save_config(self, config: Dict, path: Optional[str] = None):
        """Save configuration to file."""
        path = path or self.config_path
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
            
        self.logger.info(f"Configuration saved to {path}")
    
    def load_config(self, path: Optional[str] = None) -> Dict:
        """Load configuration from file."""
        path = path or self.config_path
        
        try:
            with open(path, 'r') as f:
                config = json.load(f)
                
            self.logger.info(f"Configuration loaded from {path}")
            return config
            
        except FileNotFoundError:
            self.logger.warning(f"Configuration file not found: {path}")
            return {}


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Performance Optimizer for Open MoE Trainer Lab")
    
    parser.add_argument("--generate", action="store_true",
                       help="Generate optimal configuration")
    parser.add_argument("--apply", action="store_true",
                       help="Apply optimizations")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run performance benchmark")
    parser.add_argument("--config", type=str, default="performance_config.json",
                       help="Configuration file path")
    parser.add_argument("--output", type=str,
                       help="Output file for results")
    
    args = parser.parse_args()
    
    optimizer = PerformanceOptimizer(args.config)
    
    if args.generate:
        config = optimizer.generate_optimal_config()
        optimizer.save_config(config, args.config)
        print(f"Generated configuration saved to {args.config}")
        
        # Print summary
        print(f"\\nHardware Profile: {config['hardware_profile']}")
        print(f"Recommended batch size: {config['training_config']['batch_size']}")
        print(f"Number of experts: {config['training_config']['num_experts']}")
        
    if args.apply:
        config = optimizer.load_config(args.config)
        if config:
            success = optimizer.apply_optimizations(config)
            print(f"Optimizations applied: {'Success' if success else 'Failed'}")
        else:
            print("No configuration found. Run --generate first.")
            
    if args.benchmark:
        config = optimizer.load_config(args.config) or optimizer.generate_optimal_config()
        results = optimizer.benchmark_configuration(config)
        
        output_path = args.output or "benchmark_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Benchmark results saved to {output_path}")
        
        # Print summary
        benchmarks = results.get("benchmarks", {})
        if "training" in benchmarks:
            print(f"\\nEstimated throughput: {benchmarks['training']['estimated_tokens_per_second']:.0f} tokens/sec")
            print(f"Configuration score: {benchmarks['training']['configuration_score']:.1f}/100")


if __name__ == "__main__":
    # Try to import pandas for timestamps, fall back to datetime if not available
    try:
        import pandas as pd
    except ImportError:
        import datetime
        
        class pd:
            @staticmethod
            def Timestamp():
                return datetime.datetime.now()
    
    main()