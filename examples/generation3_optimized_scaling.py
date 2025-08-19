"""
Generation 3 Optimized Scaling Demo
===================================

Advanced optimization and auto-scaling capabilities for MoE models.
Demonstrates performance optimization, adaptive routing, and auto-scaling.
"""

import torch
import numpy as np
import time
import json
from typing import Dict, Any, List

from moe_lab import (
    MoEModel, 
    MoETrainer,
    RouterMonitor,
    RouterAnalyzer
)
from moe_lab.optimization.adaptive_routing import (
    AdaptiveRouter,
    DynamicCapacityRouter,
    HierarchicalRouter,
    MetaLearningRouter
)
from moe_lab.optimization.efficient_training import (
    EfficientMoETrainer,
    DynamicBatchSizeScheduler,
    SelectiveActivationCheckpointing,
    AdaptiveLossScaling
)
from moe_lab.optimization.auto_scaling import (
    AutoScaler,
    ResourceMonitor,
    AutoScalingMoEModel
)
from moe_lab.utils.logging import get_logger

logger = get_logger(__name__)


def create_sample_dataset(vocab_size: int = 1000, seq_len: int = 64, num_samples: int = 200):
    """Create sample dataset for testing."""
    
    class SampleDataset:
        def __init__(self, vocab_size, seq_len, num_samples):
            self.vocab_size = vocab_size
            self.seq_len = seq_len
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
            return {'input_ids': input_ids.unsqueeze(0)}
            
    return SampleDataset(vocab_size, seq_len, num_samples)


def demo_adaptive_routing():
    """Demonstrate adaptive routing algorithms."""
    logger.info("=== Adaptive Routing Demo ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_size = 256
    num_experts = 8
    
    # Test different adaptive routers
    routers = {
        'adaptive': AdaptiveRouter(hidden_size, num_experts),
        'dynamic_capacity': DynamicCapacityRouter(hidden_size, num_experts),
        'hierarchical': HierarchicalRouter(hidden_size, num_experts, num_groups=4),
        'meta_learning': MetaLearningRouter(hidden_size, num_experts)
    }
    
    # Move to device
    for router in routers.values():
        router.to(device)
        
    # Test input
    test_input = torch.randn(32, hidden_size, device=device)
    
    results = {}
    
    for name, router in routers.items():
        logger.info(f"\nTesting {name} router...")
        
        router.eval()
        start_time = time.time()
        
        with torch.no_grad():
            logits, experts, weights, routing_info = router(test_input)
            
        routing_time = time.time() - start_time
        
        # Collect statistics
        results[name] = {
            'routing_time_ms': routing_time * 1000,
            'load_variance': routing_info.load_variance,
            'entropy': routing_info.entropy,
            'expert_utilization': self._compute_expert_utilization(routing_info.selected_experts, num_experts)
        }
        
        logger.info(f"  Routing time: {routing_time*1000:.2f}ms")
        logger.info(f"  Load variance: {routing_info.load_variance:.4f}")
        logger.info(f"  Entropy: {routing_info.entropy:.3f}")
        
        # Test adaptation for adaptive router
        if name == 'adaptive' and hasattr(router, 'get_routing_stats'):
            router.train()
            for _ in range(10):
                _, _, _, _ = router(test_input)
                
            stats = router.get_routing_stats()
            logger.info(f"  Adapted temperature: {stats['temperature']:.3f}")
            results[name]['adapted_temperature'] = stats['temperature']
            
    return results


def _compute_expert_utilization(selected_experts: torch.Tensor, num_experts: int) -> Dict[int, float]:
    """Compute expert utilization."""
    experts_flat = selected_experts.flatten()
    total_selections = len(experts_flat)
    
    utilization = {}
    for expert_idx in range(num_experts):
        count = (experts_flat == expert_idx).sum().item()
        utilization[expert_idx] = count / total_selections if total_selections > 0 else 0.0
        
    return utilization


def demo_efficient_training():
    """Demonstrate efficient training optimizations."""
    logger.info("\n=== Efficient Training Demo ===")
    
    # Create model with optimizations
    model = MoEModel(
        vocab_size=1000,
        hidden_size=256,
        num_experts=8,
        experts_per_token=2,
        num_layers=4
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create datasets
    train_dataset = create_sample_dataset(vocab_size=1000, seq_len=32, num_samples=100)
    eval_dataset = create_sample_dataset(vocab_size=1000, seq_len=32, num_samples=20)
    
    # Test different training optimizations
    training_configs = [
        {
            'name': 'standard',
            'enable_dynamic_batching': False,
            'enable_adaptive_checkpointing': False,
            'enable_adaptive_scaling': False
        },
        {
            'name': 'optimized',
            'enable_dynamic_batching': True,
            'enable_adaptive_checkpointing': True,
            'enable_adaptive_scaling': True
        }
    ]
    
    results = {}
    
    for config in training_configs:
        logger.info(f"\nTesting {config['name']} training...")
        
        # Reset model weights
        for param in model.parameters():
            param.data.normal_(0, 0.02)
            
        # Create trainer
        if config['name'] == 'optimized':
            trainer = EfficientMoETrainer(
                model=model,
                enable_dynamic_batching=config['enable_dynamic_batching'],
                enable_adaptive_checkpointing=config['enable_adaptive_checkpointing'],
                enable_adaptive_scaling=config['enable_adaptive_scaling'],
                max_batch_size=64,
                logging_steps=10,
                output_dir=f"./checkpoints_{config['name']}"
            )
        else:
            trainer = MoETrainer(
                model=model,
                logging_steps=10,
                output_dir=f"./checkpoints_{config['name']}"
            )
            
        # Train with performance monitoring
        start_time = time.time()
        
        if hasattr(trainer, 'profile_training'):
            with trainer.profile_training():
                result = trainer.train(
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    batch_size=16,
                    learning_rate=1e-3,
                    num_epochs=2
                )
        else:
            result = trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                batch_size=16,
                learning_rate=1e-3,
                num_epochs=2
            )
            
        training_time = time.time() - start_time
        
        # Collect results
        training_result = {
            'training_time': training_time,
            'final_loss': result.loss,
            'aux_loss': result.aux_loss
        }
        
        # Get performance summary if available
        if hasattr(trainer, 'get_performance_summary'):
            performance = trainer.get_performance_summary()
            training_result.update(performance)
            
        results[config['name']] = training_result
        
        logger.info(f"  Training time: {training_time:.2f}s")
        logger.info(f"  Final loss: {result.loss:.4f}")
        
        if 'mean_throughput' in training_result:
            logger.info(f"  Mean throughput: {training_result['mean_throughput']:.1f} tokens/sec")
            
        if 'mean_memory_usage_gb' in training_result:
            logger.info(f"  Mean memory usage: {training_result['mean_memory_usage_gb']:.2f} GB")
            
    # Compare results
    if 'standard' in results and 'optimized' in results:
        standard = results['standard']
        optimized = results['optimized']
        
        speedup = standard['training_time'] / optimized['training_time']
        logger.info(f"\nOptimization Results:")
        logger.info(f"  Training speedup: {speedup:.2f}x")
        
        if 'mean_throughput' in optimized:
            throughput_improvement = optimized['mean_throughput'] / standard.get('mean_throughput', 1)
            logger.info(f"  Throughput improvement: {throughput_improvement:.2f}x")
            
    return results


def demo_auto_scaling():
    """Demonstrate auto-scaling capabilities."""
    logger.info("\n=== Auto-Scaling Demo ===")
    
    # Create auto-scaling model
    model = AutoScalingMoEModel(
        vocab_size=1000,
        hidden_size=128,
        num_experts=4,
        experts_per_token=2,
        num_layers=2,
        enable_auto_scaling=True,
        max_experts=12
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize resource monitoring
    resource_monitor = ResourceMonitor(monitoring_interval=0.5)
    resource_monitor.start_monitoring()
    
    # Initialize auto-scaler
    auto_scaler = AutoScaler(
        model=model,
        min_batch_size=4,
        max_batch_size=32,
        min_experts=2,
        max_experts=12,
        scaling_interval=5.0
    )
    
    results = {
        'scaling_decisions': [],
        'performance_metrics': [],
        'resource_usage': []
    }
    
    try:
        # Simulate different workload scenarios
        scenarios = [
            {'name': 'low_load', 'batch_size': 4, 'duration': 10},
            {'name': 'high_load', 'batch_size': 24, 'duration': 10},
            {'name': 'variable_load', 'batch_size': 'variable', 'duration': 15}
        ]
        
        for scenario in scenarios:
            logger.info(f"\nSimulating {scenario['name']} scenario...")
            
            scenario_start = time.time()
            step_count = 0
            
            while time.time() - scenario_start < scenario['duration']:
                # Determine batch size
                if scenario['batch_size'] == 'variable':
                    batch_size = np.random.choice([4, 8, 16, 24, 32])
                else:
                    batch_size = scenario['batch_size']
                    
                # Create batch
                test_input = torch.randint(0, 1000, (batch_size, 32), device=device)
                
                # Measure performance
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model(test_input, return_routing_info=True)
                    
                inference_time = time.time() - start_time
                throughput = (batch_size * 32) / inference_time
                latency = inference_time * 1000  # ms
                
                # Get resource metrics
                metrics = resource_monitor.get_latest_metrics()
                if metrics:
                    metrics.throughput = throughput
                    metrics.latency = latency
                    results['resource_usage'].append({
                        'scenario': scenario['name'],
                        'step': step_count,
                        'cpu_percent': metrics.cpu_percent,
                        'memory_percent': metrics.memory_percent,
                        'gpu_memory_used': metrics.gpu_memory_used,
                        'throughput': throughput,
                        'latency': latency
                    })
                    
                # Check for scaling decision
                if auto_scaler.should_scale():
                    decision = auto_scaler.make_scaling_decision(throughput, latency)
                    
                    results['scaling_decisions'].append({
                        'scenario': scenario['name'],
                        'step': step_count,
                        'action': decision.action,
                        'target_batch_size': decision.target_batch_size,
                        'target_num_experts': decision.target_num_experts,
                        'confidence': decision.confidence,
                        'reasoning': decision.reasoning
                    })
                    
                    if decision.action != "no_change":
                        logger.info(f"  Scaling decision: {decision.action}")
                        logger.info(f"    Target batch size: {decision.target_batch_size}")
                        logger.info(f"    Target experts: {decision.target_num_experts}")
                        logger.info(f"    Confidence: {decision.confidence:.2f}")
                        logger.info(f"    Reasoning: {decision.reasoning}")
                        
                        # Apply scaling (simulation)
                        auto_scaler.apply_scaling_decision(decision)
                        
                results['performance_metrics'].append({
                    'scenario': scenario['name'],
                    'step': step_count,
                    'batch_size': batch_size,
                    'throughput': throughput,
                    'latency': latency,
                    'current_experts': auto_scaler.current_num_experts
                })
                
                step_count += 1
                time.sleep(0.1)  # Brief pause
                
            logger.info(f"  Completed {scenario['name']} with {step_count} steps")
            
    finally:
        resource_monitor.stop_monitoring()
        
    # Analyze results
    logger.info("\nAuto-Scaling Results:")
    
    total_decisions = len(results['scaling_decisions'])
    scale_up_decisions = sum(1 for d in results['scaling_decisions'] if d['action'] == 'scale_up')
    scale_down_decisions = sum(1 for d in results['scaling_decisions'] if d['action'] == 'scale_down')
    
    logger.info(f"  Total scaling decisions: {total_decisions}")
    logger.info(f"  Scale up decisions: {scale_up_decisions}")
    logger.info(f"  Scale down decisions: {scale_down_decisions}")
    
    if results['performance_metrics']:
        avg_throughput = np.mean([m['throughput'] for m in results['performance_metrics']])
        avg_latency = np.mean([m['latency'] for m in results['performance_metrics']])
        logger.info(f"  Average throughput: {avg_throughput:.1f} tokens/sec")
        logger.info(f"  Average latency: {avg_latency:.1f} ms")
        
    return results


def demo_comprehensive_optimization():
    """Demonstrate comprehensive optimization pipeline."""
    logger.info("\n=== Comprehensive Optimization Demo ===")
    
    # Create optimized model with all features
    model = AutoScalingMoEModel(
        vocab_size=1000,
        hidden_size=256,
        num_experts=6,
        experts_per_token=2,
        num_layers=3,
        enable_auto_scaling=True,
        max_experts=16
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create comprehensive dataset
    train_dataset = create_sample_dataset(vocab_size=1000, seq_len=48, num_samples=150)
    eval_dataset = create_sample_dataset(vocab_size=1000, seq_len=48, num_samples=30)
    
    # Setup efficient trainer with all optimizations
    trainer = EfficientMoETrainer(
        model=model,
        enable_dynamic_batching=True,
        enable_adaptive_checkpointing=True,
        enable_adaptive_scaling=True,
        max_batch_size=64,
        aux_loss_coef=0.01,
        router_z_loss_coef=0.001,
        logging_steps=5,
        eval_steps=20,
        output_dir="./comprehensive_checkpoints"
    )
    
    # Setup monitoring
    monitor = RouterMonitor(model, window_size=200)
    analyzer = RouterAnalyzer(model)
    
    # Initialize auto-scaling
    model.start_auto_scaling()
    
    try:
        logger.info("Starting comprehensive optimized training...")
        
        # Training with full optimization pipeline
        with monitor.track():
            with trainer.profile_training():
                result = trainer.train(
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    batch_size=16,
                    learning_rate=2e-4,
                    num_epochs=3
                )
                
        # Collect comprehensive results
        final_stats = monitor.get_stats()
        performance_summary = trainer.get_performance_summary()
        
        # Post-training analysis
        test_input = torch.randint(0, 1000, (8, 48), device=device)
        routing_analysis = analyzer.analyze_batch({'input_ids': test_input})
        
        comprehensive_results = {
            'training_result': {
                'final_loss': result.loss,
                'aux_loss': result.aux_loss,
                'router_z_loss': result.router_z_loss
            },
            'performance_metrics': performance_summary,
            'routing_metrics': {
                'final_load_variance': final_stats.load_variance,
                'final_entropy': final_stats.entropy,
                'final_throughput': final_stats.throughput_tokens_per_sec,
                'expert_utilization': final_stats.expert_utilization
            },
            'optimization_analysis': {
                'batch_utilization': routing_analysis['expert_utilization'],
                'routing_efficiency': routing_analysis['entropy'] / np.log(model.num_experts)
            }
        }
        
        # Report comprehensive results
        logger.info("Comprehensive Optimization Results:")
        logger.info(f"  Final training loss: {result.loss:.4f}")
        logger.info(f"  Mean training throughput: {performance_summary.get('mean_throughput', 0):.1f} tokens/sec")
        logger.info(f"  Mean memory usage: {performance_summary.get('mean_memory_usage_gb', 0):.2f} GB")
        logger.info(f"  Final routing entropy: {final_stats.entropy:.3f}")
        logger.info(f"  Load balance variance: {final_stats.load_variance:.4f}")
        
        # Expert utilization analysis
        logger.info("Expert Utilization Distribution:")
        for expert_idx, util in final_stats.expert_utilization.items():
            logger.info(f"    Expert {expert_idx}: {util:.2%}")
            
        # Performance comparison metrics
        if performance_summary:
            logger.info("Performance Statistics:")
            logger.info(f"  Mean step time: {performance_summary.get('mean_step_time', 0):.3f}s")
            logger.info(f"  Throughput std: {performance_summary.get('throughput_std', 0):.1f}")
            logger.info(f"  Total training steps: {performance_summary.get('total_steps', 0)}")
            
        return comprehensive_results
        
    finally:
        model.stop_auto_scaling()


def main():
    """Run all Generation 3 optimization demos."""
    logger.info("Starting Generation 3 Optimized Scaling Demo")
    logger.info("=" * 70)
    
    results = {}
    
    try:
        # Adaptive routing demo
        routing_results = demo_adaptive_routing()
        results['adaptive_routing'] = routing_results
        
        # Efficient training demo
        training_results = demo_efficient_training()
        results['efficient_training'] = training_results
        
        # Auto-scaling demo
        scaling_results = demo_auto_scaling()
        results['auto_scaling'] = scaling_results
        
        # Comprehensive optimization demo
        comprehensive_results = demo_comprehensive_optimization()
        results['comprehensive_optimization'] = comprehensive_results
        
        logger.info("\n" + "=" * 70)
        logger.info("Generation 3 Optimized Scaling Demo Completed Successfully!")
        logger.info("Advanced Capabilities Demonstrated:")
        logger.info("✓ Adaptive routing algorithms (temperature, capacity, hierarchical)")
        logger.info("✓ Efficient training with dynamic batching and checkpointing")
        logger.info("✓ Auto-scaling with resource monitoring and decision making")
        logger.info("✓ Meta-learning routing adaptation")
        logger.info("✓ Comprehensive optimization pipeline integration")
        logger.info("✓ Real-time performance monitoring and analysis")
        logger.info("✓ Memory-aware adaptive optimizations")
        logger.info("✓ Load balancing and expert utilization optimization")
        
        return results
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    results = main()
    
    # Save results
    try:
        # Convert results to JSON-serializable format
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = value
            else:
                json_results[key] = str(value)
                
        with open('generation3_optimized_scaling_results.json', 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
            
        print("Results saved to generation3_optimized_scaling_results.json")
        
    except Exception as e:
        print(f"Failed to save results: {e}")