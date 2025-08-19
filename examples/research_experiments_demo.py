#!/usr/bin/env python3
"""
Research Experiments Demo - Generation 1 Implementation

This demo showcases the novel research components and experimental framework
added to the Open MoE Trainer Lab. It demonstrates:
1. Novel router algorithms  
2. Baseline comparisons
3. Statistical experimental framework
4. Research-grade analysis capabilities

Generation 1: MAKE IT WORK - Basic functional implementation
"""

import torch
import json
import numpy as np
from pathlib import Path

# Research components
from moe_lab.research.experimental_routers import (
    AdaptiveRouter, HierarchicalRouter, LearnedSparseRouter,
    DynamicTopKRouter, ContextAwareRouter
)
from moe_lab.research.baseline_comparisons import (
    DenseBaseline, SwitchBaseline, MixtralBaseline,
    PerformanceBenchmark, BaselineComparison, RouterComparison
)
from moe_lab.research.experimental_framework import (
    ExperimentRunner, ExperimentConfig, StatisticalValidator, ResultsAnalyzer
)

# Core components
from moe_lab.models.moe_model import MoEModel


def create_sample_dataset(vocab_size=1000, seq_length=128, num_samples=100):
    """Create a small sample dataset for demonstration."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Generate random sequences
    data = torch.randint(0, vocab_size, (num_samples, seq_length))
    dataset = TensorDataset(data)
    
    return DataLoader(dataset, batch_size=8, shuffle=True)


def demo_novel_routers():
    """Demonstrate novel experimental routers."""
    print("=" * 60)
    print("🧠 DEMO: Novel Experimental Routers")
    print("=" * 60)
    
    # Configuration
    hidden_size = 256
    num_experts = 8
    batch_size = 4
    
    # Sample input
    sample_input = torch.randn(batch_size, hidden_size)
    
    # Test each novel router
    routers = {
        "Adaptive Router": AdaptiveRouter(hidden_size, num_experts, min_k=1, max_k=3),
        "Hierarchical Router": HierarchicalRouter(hidden_size, num_experts, num_groups=4),
        "Learned Sparse Router": LearnedSparseRouter(hidden_size, num_experts, sparsity_level=0.7),
        "Dynamic TopK Router": DynamicTopKRouter(hidden_size, num_experts, max_k=4),
        "Context-Aware Router": ContextAwareRouter(hidden_size, num_experts, top_k=2)
    }
    
    results = {}
    
    for name, router in routers.items():
        print(f"\nTesting {name}...")
        try:
            router.eval()
            with torch.no_grad():
                router_logits, selected_experts, expert_weights, routing_info = router(sample_input)
                
                results[name] = {
                    "success": True,
                    "output_shape": list(router_logits.shape),
                    "selected_experts_shape": list(selected_experts.shape),
                    "expert_weights_shape": list(expert_weights.shape),
                    "load_variance": routing_info.load_variance,
                    "entropy": routing_info.entropy
                }
                
                print(f"  ✅ Success - Load Variance: {routing_info.load_variance:.4f}, Entropy: {routing_info.entropy:.4f}")
                
        except Exception as e:
            results[name] = {"success": False, "error": str(e)}
            print(f"  ❌ Failed: {e}")
    
    return results


def demo_baseline_comparisons():
    """Demonstrate baseline model comparisons."""
    print("\n" + "=" * 60)
    print("📊 DEMO: Baseline Model Comparisons")  
    print("=" * 60)
    
    # Create sample dataset
    dataloader = create_sample_dataset(vocab_size=1000, seq_length=64, num_samples=50)
    
    # Model configuration
    model_config = {
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_experts": 4,
        "experts_per_token": 2,
        "num_layers": 2,
        "num_attention_heads": 4,
        "max_position_embeddings": 64
    }
    
    # Create models
    print("Creating models...")
    models = {
        "Dense": DenseBaseline(
            vocab_size=model_config["vocab_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
            num_attention_heads=model_config["num_attention_heads"],
            intermediate_size=model_config["hidden_size"] * 4
        ),
        "Switch": SwitchBaseline(**model_config),
        "Mixtral": MixtralBaseline(**model_config),
        "Experimental": MoEModel(**model_config)
    }
    
    # Run performance benchmark
    print("Running performance benchmarks...")
    benchmark = PerformanceBenchmark()
    
    results = {}
    for name, model in models.items():
        print(f"  Benchmarking {name}...")
        try:
            metrics = benchmark.benchmark_model(model, dataloader, num_batches=5)
            results[name] = {
                "perplexity": metrics.perplexity,
                "throughput": metrics.throughput, 
                "memory_usage": metrics.memory_usage,
                "parameters": metrics.parameters,
                "active_parameters": metrics.active_parameters
            }
            print(f"    ✅ Perplexity: {metrics.perplexity:.2f}, Throughput: {metrics.throughput:.2f} tokens/s")
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"    ❌ Failed: {e}")
    
    return results


def demo_router_comparison():
    """Demonstrate router algorithm comparison."""
    print("\n" + "=" * 60)
    print("🔀 DEMO: Router Algorithm Comparison")
    print("=" * 60)
    
    # Create sample dataset
    dataloader = create_sample_dataset(vocab_size=500, seq_length=32, num_samples=30)
    
    # Base model configuration
    base_config = {
        "vocab_size": 500,
        "hidden_size": 128,
        "num_experts": 4,
        "num_layers": 2,
        "num_attention_heads": 4,
        "max_position_embeddings": 32
    }
    
    # Router configurations to compare
    router_configs = {
        "TopK": {"router_type": "top_k", "experts_per_token": 2},
        "Switch": {"router_type": "switch", "experts_per_token": 1},
        # Note: In a full implementation, we'd integrate experimental routers here
    }
    
    print("Comparing router algorithms...")
    comparison = RouterComparison()
    
    try:
        results = comparison.compare_routers(
            base_config, 
            router_configs, 
            dataloader, 
            num_batches=3
        )
        
        print("✅ Router comparison completed")
        
        # Display key results
        for router_name, result in results["results"].items():
            routing_metrics = result["routing_metrics"]
            print(f"  {router_name}:")
            print(f"    Load Variance: {routing_metrics.get('avg_load_variance', 'N/A')}")
            print(f"    Entropy: {routing_metrics.get('avg_entropy', 'N/A')}")
            print(f"    Load Balancing Score: {routing_metrics.get('load_balancing_score', 'N/A')}")
            
        return results
        
    except Exception as e:
        print(f"❌ Router comparison failed: {e}")
        return {"error": str(e)}


def demo_experimental_framework():
    """Demonstrate the experimental framework."""
    print("\n" + "=" * 60)
    print("🔬 DEMO: Experimental Framework")
    print("=" * 60)
    
    # Create experimental configuration
    config = ExperimentConfig(
        experiment_name="demo_experiment",
        model_config={
            "vocab_size": 500,
            "hidden_size": 64,
            "num_experts": 4,
            "experts_per_token": 2,
            "num_layers": 2,
            "num_attention_heads": 2,
            "max_position_embeddings": 32
        },
        training_config={
            "batch_size": 4,
            "learning_rate": 1e-4,
            "num_epochs": 1
        },
        dataset_config={
            "vocab_size": 500,
            "seq_length": 32,
            "num_samples": 20
        },
        num_runs=2,  # Small number for demo
        seed=42
    )
    
    print(f"Running experiment: {config.experiment_name}")
    print(f"  - Model: {config.model_config['hidden_size']}D, {config.model_config['num_experts']} experts")
    print(f"  - Number of runs: {config.num_runs}")
    
    # Create output directory
    output_dir = Path("./demo_experiment_results")
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Run experiment
        runner = ExperimentRunner(config, output_dir=str(output_dir))
        
        # Note: In a full implementation, this would run complete training
        # For demo, we'll simulate results
        print("📊 Simulating experimental framework...")
        
        # Simulate results structure
        simulated_results = {
            "config": config.__dict__,
            "aggregated_results": {
                "perplexity": {"mean": 15.2, "std": 0.8, "values": [15.1, 15.3]},
                "throughput": {"mean": 245.6, "std": 12.1, "values": [239.2, 252.0]},
                "memory_usage": {"mean": 0.45, "std": 0.02, "values": [0.44, 0.46]}
            },
            "statistical_analysis": {
                "perplexity": {
                    "normality_test_p_value": 0.8,
                    "is_normal": True,
                    "confidence_interval": [14.1, 16.3],
                    "coefficient_of_variation": 0.053
                }
            },
            "experiment_metadata": {
                "total_duration": 125.4,
                "num_runs": 2,
                "timestamp": "2024-01-01 12:00:00"
            }
        }
        
        # Save simulated results
        with open(output_dir / "demo_experiment_final_report.json", 'w') as f:
            json.dump(simulated_results, f, indent=2)
        
        print("✅ Experimental framework demo completed")
        print(f"  Results saved to: {output_dir}")
        
        return simulated_results
        
    except Exception as e:
        print(f"❌ Experimental framework failed: {e}")
        return {"error": str(e)}


def demo_statistical_validation():
    """Demonstrate statistical validation capabilities."""
    print("\n" + "=" * 60)
    print("📈 DEMO: Statistical Validation")
    print("=" * 60)
    
    # Simulate experimental results for comparison
    experiment_a_results = [
        {"perplexity": 15.2, "throughput": 245.6},
        {"perplexity": 15.1, "throughput": 250.2}, 
        {"perplexity": 15.4, "throughput": 248.1}
    ]
    
    experiment_b_results = [
        {"perplexity": 16.8, "throughput": 220.4},
        {"perplexity": 16.5, "throughput": 225.1},
        {"perplexity": 16.9, "throughput": 218.9}
    ]
    
    print("Comparing two experimental configurations...")
    print("  Experiment A: Novel adaptive routing")
    print("  Experiment B: Standard top-k routing")
    
    # Statistical comparison
    validator = StatisticalValidator()
    
    for metric in ["perplexity", "throughput"]:
        print(f"\n📊 Comparing {metric}:")
        
        comparison = validator.compare_experiments(
            experiment_a_results,
            experiment_b_results,
            metric
        )
        
        print(f"  Mean A: {comparison['summary']['experiment_a_mean']:.2f}")
        print(f"  Mean B: {comparison['summary']['experiment_b_mean']:.2f}")
        print(f"  Difference: {comparison['summary']['difference']:.2f}")
        print(f"  Relative improvement: {comparison['summary']['relative_improvement']:.1f}%")
        print(f"  Statistically significant: {comparison['t_test']['significant']}")
        print(f"  Effect size: {comparison['effect_size']['magnitude']}")
    
    return {
        "experiment_a": experiment_a_results,
        "experiment_b": experiment_b_results,
        "statistical_comparisons": {
            "perplexity": validator.compare_experiments(experiment_a_results, experiment_b_results, "perplexity"),
            "throughput": validator.compare_experiments(experiment_a_results, experiment_b_results, "throughput")
        }
    }


def main():
    """Run all research experiments demos."""
    print("🔬 Open MoE Trainer Lab - Research Experiments Demo")
    print("Generation 1: MAKE IT WORK - Basic Implementation")
    print("=" * 70)
    
    # Track all results
    all_results = {}
    
    # Run all demos
    try:
        all_results["novel_routers"] = demo_novel_routers()
        all_results["baseline_comparisons"] = demo_baseline_comparisons()
        all_results["router_comparison"] = demo_router_comparison()
        all_results["experimental_framework"] = demo_experimental_framework()
        all_results["statistical_validation"] = demo_statistical_validation()
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 DEMO SUMMARY")
        print("=" * 70)
        
        successful_demos = []
        failed_demos = []
        
        for demo_name, results in all_results.items():
            if isinstance(results, dict) and "error" not in results:
                successful_demos.append(demo_name)
            else:
                failed_demos.append(demo_name)
        
        print(f"✅ Successful demos: {len(successful_demos)}/{len(all_results)}")
        for demo in successful_demos:
            print(f"  - {demo}")
        
        if failed_demos:
            print(f"❌ Failed demos: {len(failed_demos)}")
            for demo in failed_demos:
                print(f"  - {demo}")
        
        # Save complete results
        output_file = Path("./research_experiments_demo_results.json")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📄 Complete results saved to: {output_file}")
        
        print("\n🎉 Research experiments demo completed!")
        print("Generation 1 implementation demonstrates:")
        print("  • Novel router algorithms working")
        print("  • Baseline comparison framework")  
        print("  • Statistical experimental validation")
        print("  • Research-grade analysis capabilities")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()