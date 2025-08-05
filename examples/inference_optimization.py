#!/usr/bin/env python3
"""
MoE Inference Optimization Example

This example demonstrates how to optimize MoE models for inference using:
- Expert caching strategies
- Selective expert loading
- Model compilation
- Performance benchmarking

The example shows how to take a trained MoE model and optimize it for production deployment.
"""

import torch
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from moe_lab import MoEModel
from moe_lab.inference import OptimizedMoEModel
from moe_lab.inference.caching import ExpertCache, create_expert_cache
from moe_lab.utils.logging import setup_logging


class InferenceBenchmark:
    """Comprehensive inference benchmarking suite."""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.results = {}
    
    def benchmark_throughput(self, batch_sizes: List[int], seq_lengths: List[int], num_runs: int = 10):
        """Benchmark inference throughput across different configurations."""
        print("🚀 Benchmarking throughput...")
        
        results = {}
        
        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                print(f"  Testing batch_size={batch_size}, seq_length={seq_length}")
                
                # Create sample input
                input_ids = torch.randint(0, 32000, (batch_size, seq_length)).to(self.device)
                
                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = self.model(input_ids)
                
                # Measure
                torch.cuda.synchronize() if self.device.startswith('cuda') else None
                start_time = time.time()
                
                for _ in range(num_runs):
                    with torch.no_grad():
                        _ = self.model(input_ids)
                
                torch.cuda.synchronize() if self.device.startswith('cuda') else None
                end_time = time.time()
                
                total_time = end_time - start_time
                tokens_per_second = (batch_size * seq_length * num_runs) / total_time
                
                results[(batch_size, seq_length)] = {
                    'tokens_per_second': tokens_per_second,
                    'latency_ms': (total_time / num_runs) * 1000,
                    'throughput_factor': tokens_per_second / (batch_size * seq_length)
                }
        
        self.results['throughput'] = results
        return results
    
    def benchmark_memory(self, batch_sizes: List[int]):
        """Benchmark memory usage."""
        print("💾 Benchmarking memory usage...")
        
        results = {}
        
        for batch_size in batch_sizes:
            # Clear cache
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            # Measure baseline
            if hasattr(torch.cuda, 'memory_allocated'):
                baseline_memory = torch.cuda.memory_allocated(self.device)
            else:
                baseline_memory = 0
            
            # Create input and run model
            input_ids = torch.randint(0, 32000, (batch_size, 512)).to(self.device)
            
            with torch.no_grad():
                _ = self.model(input_ids)
            
            if hasattr(torch.cuda, 'memory_allocated'):
                peak_memory = torch.cuda.max_memory_allocated(self.device)
                memory_used = (peak_memory - baseline_memory) / (1024**3)  # GB
            else:
                memory_used = 0  # Fallback for non-CUDA
            
            results[batch_size] = {
                'memory_gb': memory_used,
                'memory_per_sample': memory_used / batch_size if batch_size > 0 else 0
            }
        
        self.results['memory'] = results
        return results
    
    def benchmark_expert_utilization(self, num_samples: int = 1000):
        """Analyze expert utilization patterns."""
        print("👥 Analyzing expert utilization...")
        
        expert_counts = {}
        total_tokens = 0
        
        for _ in range(num_samples):
            # Random input
            batch_size = np.random.randint(1, 5)
            seq_length = np.random.randint(50, 513)
            input_ids = torch.randint(0, 32000, (batch_size, seq_length)).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_ids, return_routing_info=True)
            
            # Track expert usage
            if output.routing_info and output.routing_info.selected_experts is not None:
                selected_experts = output.routing_info.selected_experts
                for layer_experts in selected_experts:
                    for expert_id in layer_experts.flatten():
                        expert_id = expert_id.item()
                        expert_counts[expert_id] = expert_counts.get(expert_id, 0) + 1
            
            total_tokens += batch_size * seq_length
        
        # Calculate utilization statistics
        if expert_counts:
            expert_ids = list(expert_counts.keys())
            expert_usage = list(expert_counts.values())
            
            utilization_stats = {
                'total_experts_used': len(expert_ids),
                'mean_usage': np.mean(expert_usage),
                'std_usage': np.std(expert_usage),
                'usage_variance': np.var(expert_usage),
                'most_used_expert': max(expert_ids, key=lambda x: expert_counts[x]),
                'least_used_expert': min(expert_ids, key=lambda x: expert_counts[x]),
                'load_balance_coefficient': np.std(expert_usage) / np.mean(expert_usage) if np.mean(expert_usage) > 0 else 0
            }
        else:
            utilization_stats = {'error': 'No expert usage data collected'}\n        
        self.results['expert_utilization'] = utilization_stats
        return utilization_stats
    
    def generate_report(self, save_path: str = None):
        """Generate comprehensive benchmark report."""
        print("📊 Generating benchmark report...")
        
        report = {
            'model_info': {
                'model_type': type(self.model).__name__,
                'device': self.device,
                'num_parameters': sum(p.numel() for p in self.model.parameters())
            },
            'benchmarks': self.results
        }
        
        # Print summary
        print("\n" + "="*60)
        print("📈 BENCHMARK REPORT SUMMARY")
        print("="*60)
        
        if 'throughput' in self.results:
            throughput_data = self.results['throughput']
            max_throughput = max(data['tokens_per_second'] for data in throughput_data.values())
            print(f"Peak throughput: {max_throughput:.0f} tokens/second")
        
        if 'memory' in self.results:
            memory_data = self.results['memory']
            max_memory = max(data['memory_gb'] for data in memory_data.values())
            print(f"Peak memory usage: {max_memory:.2f} GB")
        
        if 'expert_utilization' in self.results:
            util_data = self.results['expert_utilization']
            if 'load_balance_coefficient' in util_data:
                print(f"Load balance coefficient: {util_data['load_balance_coefficient']:.3f}")
        
        print("="*60 + "\n")
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📋 Report saved to: {save_path}")
        
        return report


def demonstrate_expert_caching():
    """Demonstrate expert caching strategies."""
    print("\n🗄️  EXPERT CACHING DEMONSTRATION")
    print("-" * 50)
    
    # Create different cache configurations
    cache_configs = {
        'small': {'capacity_gb': 2.0, 'max_experts': 16, 'policy': 'lru'},
        'medium': {'capacity_gb': 4.0, 'max_experts': 32, 'policy': 'weighted_lru'},
        'large': {'capacity_gb': 8.0, 'max_experts': 64, 'policy': 'weighted_lru'},
        'adaptive': {'capacity_gb': 6.0, 'max_experts': 48, 'cache_type': 'adaptive'}
    }
    
    for name, config in cache_configs.items():
        print(f"\n🎯 Testing {name} cache configuration:")
        
        cache_type = config.pop('cache_type', 'standard')
        cache = create_expert_cache(cache_type=cache_type, **config)
        
        # Simulate expert usage patterns
        num_experts = 64
        mock_experts = {i: torch.nn.Linear(512, 2048) for i in range(num_experts)}
        
        # Simulate requests with usage patterns
        requests = []
        # Popular experts (Zipf distribution)
        popular_experts = list(range(8))  # First 8 are popular
        normal_experts = list(range(8, 32))  # Next 24 are normal
        rare_experts = list(range(32, num_experts))  # Rest are rare
        
        for _ in range(1000):
            if np.random.random() < 0.6:  # 60% popular
                expert_id = np.random.choice(popular_experts)
            elif np.random.random() < 0.9:  # 30% normal
                expert_id = np.random.choice(normal_experts)
            else:  # 10% rare
                expert_id = np.random.choice(rare_experts)
            
            requests.append(expert_id)
        
        # Process requests
        hits = 0
        for expert_id in requests:
            expert = cache.get(expert_id)
            if expert is not None:
                hits += 1
            else:
                # Cache miss - load expert
                cache.put(expert_id, mock_experts[expert_id], 
                         weight=2.0 if expert_id in popular_experts else 1.0)
        
        stats = cache.get_stats()
        print(f"  Hit rate: {stats.hit_rate:.1%}")
        print(f"  Cache size: {stats.cache_size} experts")
        print(f"  Memory usage: {stats.memory_usage_gb:.2f} GB")
        print(f"  Evictions: {stats.evictions}")


def main():
    """Main inference optimization demonstration."""
    print("🎯 MoE Inference Optimization Example")
    
    # Setup
    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create a sample model (in practice, you'd load a trained model)
    print("🧠 Creating sample MoE model...")
    model_config = {
        'vocab_size': 32000,
        'hidden_size': 768,
        'num_experts': 16,
        'experts_per_token': 2,
        'num_layers': 8,
        'num_attention_heads': 12,
        'max_position_embeddings': 2048,
        'moe_layers': [1, 3, 5, 7]
    }
    
    base_model = MoEModel(**model_config)
    base_model.to(device)
    base_model.eval()
    
    print(f"📊 Model parameters: {sum(p.numel() for p in base_model.parameters()):,}")
    
    # Optimization 1: Create optimized model with caching
    print("\n⚡ Creating optimized model with expert caching...")
    optimized_model = OptimizedMoEModel(base_model)
    
    # Setup expert cache
    cache = create_expert_cache(
        cache_type="adaptive",
        capacity_gb=4.0,
        max_experts=32,
        preload_top_k=8
    )
    optimized_model.set_expert_cache(cache)
    
    # Optimization 2: Model compilation (if available)
    print("🔧 Attempting model compilation...")
    try:
        if hasattr(torch, 'compile'):
            compiled_model = torch.compile(optimized_model, mode='reduce-overhead')
            print("✅ Model compiled successfully")
        else:
            compiled_model = optimized_model
            print("ℹ️  torch.compile not available, using uncompiled model")
    except Exception as e:
        compiled_model = optimized_model
        print(f"⚠️  Compilation failed: {e}")
    
    # Benchmark different configurations
    models_to_test = {
        'baseline': base_model,
        'optimized': optimized_model,
        'compiled': compiled_model
    }
    
    batch_sizes = [1, 4, 8, 16]
    seq_lengths = [128, 256, 512]
    
    benchmark_results = {}
    
    for model_name, model in models_to_test.items():
        print(f"\n📊 Benchmarking {model_name} model...")
        
        benchmark = InferenceBenchmark(model, device)
        
        # Throughput benchmark
        throughput_results = benchmark.benchmark_throughput(
            batch_sizes=batch_sizes,
            seq_lengths=seq_lengths,
            num_runs=5
        )
        
        # Memory benchmark
        memory_results = benchmark.benchmark_memory(batch_sizes)
        
        # Expert utilization (only for base model to avoid redundancy)
        if model_name == 'baseline':
            utilization_results = benchmark.benchmark_expert_utilization()
        
        # Generate report
        report_path = f"benchmark_report_{model_name}.json"
        report = benchmark.generate_report(report_path)
        benchmark_results[model_name] = report
    
    # Compare results
    print("\n📈 PERFORMANCE COMPARISON")
    print("="*70)
    
    # Find best throughput for each configuration
    for (batch_size, seq_length) in [(1, 256), (8, 512)]:
        print(f"\nBatch size {batch_size}, Sequence length {seq_length}:")
        print("-" * 50)
        
        for model_name in models_to_test.keys():
            if 'throughput' in benchmark_results[model_name]['benchmarks']:
                config_key = (batch_size, seq_length)
                if config_key in benchmark_results[model_name]['benchmarks']['throughput']:
                    result = benchmark_results[model_name]['benchmarks']['throughput'][config_key]
                    print(f"{model_name:>12}: {result['tokens_per_second']:>8.0f} tokens/sec, "
                          f"{result['latency_ms']:>6.1f}ms latency")
    
    # Expert caching demonstration
    demonstrate_expert_caching()
    
    # Generation example with different optimization levels
    print("\n📝 TEXT GENERATION COMPARISON")
    print("-" * 50)
    
    prompt = "The future of artificial intelligence will"
    input_ids = torch.randint(0, 1000, (1, 10)).to(device)  # Mock tokenization
    
    for model_name, model in models_to_test.items():
        print(f"\n🎯 {model_name} model generation:")
        
        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=20,
                temperature=0.8,
                do_sample=True
            )
        generation_time = time.time() - start_time
        
        print(f"  Generation time: {generation_time:.3f}s")
        print(f"  Tokens generated: {output_ids.shape[1] - input_ids.shape[1]}")
        print(f"  Speed: {(output_ids.shape[1] - input_ids.shape[1]) / generation_time:.1f} tokens/sec")
    
    # Save comprehensive comparison
    comparison_report = {
        'timestamp': time.time(),
        'device': str(device),
        'model_config': model_config,
        'benchmark_results': benchmark_results,
        'optimization_summary': {
            'expert_caching': True,
            'model_compilation': hasattr(torch, 'compile'),
            'adaptive_caching': True
        }
    }
    
    import json
    with open('inference_optimization_report.json', 'w') as f:
        json.dump(comparison_report, f, indent=2, default=str)
    
    print("\n🎉 Inference optimization example completed!")
    print("📁 Check detailed reports in current directory")
    print("🚀 Key optimizations demonstrated:")
    print("   • Expert caching with LRU and adaptive policies")
    print("   • Selective expert loading")
    print("   • Model compilation (when available)")
    print("   • Comprehensive performance benchmarking")


if __name__ == "__main__":
    main()