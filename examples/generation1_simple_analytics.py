"""
Generation 1 Simple Analytics Demo
==================================

Simple demonstration of MoE analytics and monitoring capabilities.
This shows the basic router monitoring and cost analysis features.
"""

import torch
import numpy as np
import time
from typing import Dict, Any

from moe_lab import (
    MoEModel, 
    MoETrainer, 
    RouterMonitor, 
    RouterAnalyzer, 
    MoECostAnalyzer
)
from moe_lab.utils.logging import get_logger

logger = get_logger(__name__)


def create_sample_data(vocab_size: int = 1000, seq_len: int = 128, num_samples: int = 100):
    """Create sample dataset for testing."""
    
    class SampleDataset:
        def __init__(self, vocab_size, seq_len, num_samples):
            self.vocab_size = vocab_size
            self.seq_len = seq_len  
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            # Generate random sequence
            input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
            return {'input_ids': input_ids.unsqueeze(0)}
            
    return SampleDataset(vocab_size, seq_len, num_samples)


def demo_router_monitoring():
    """Demonstrate router monitoring capabilities."""
    logger.info("=== Router Monitoring Demo ===")
    
    # Create simple MoE model
    model = MoEModel(
        vocab_size=1000,
        hidden_size=256,
        num_experts=8,
        experts_per_token=2,
        num_layers=4,
        num_attention_heads=8
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create sample data
    dataset = create_sample_data(vocab_size=1000, seq_len=64, num_samples=10)
    
    # Initialize router monitor
    monitor = RouterMonitor(model, window_size=50)
    
    # Start monitoring
    with monitor.track():
        logger.info("Running model with monitoring...")
        
        for i in range(10):
            batch = dataset[i]
            input_ids = batch['input_ids'].to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids, return_routing_info=True)
                
            # Record routing decisions
            monitor.record_routing(outputs.routing_info, batch_size=1)
            
            time.sleep(0.1)  # Simulate processing time
            
    # Get statistics
    stats = monitor.get_stats()
    
    logger.info("Monitoring Results:")
    logger.info(f"  Load Variance: {stats.load_variance:.4f}")
    logger.info(f"  Drop Rate: {stats.drop_rate:.2%}")
    logger.info(f"  Routing Entropy: {stats.entropy:.3f}")
    logger.info(f"  Throughput: {stats.throughput_tokens_per_sec:.1f} tokens/sec")
    logger.info(f"  Memory Usage: {stats.memory_usage_mb:.1f} MB")
    
    # Expert utilization
    logger.info("Expert Utilization:")
    for expert_idx, util in stats.expert_utilization.items():
        logger.info(f"  Expert {expert_idx}: {util:.2%}")
        
    return stats


def demo_router_analysis():
    """Demonstrate router analysis capabilities."""
    logger.info("\n=== Router Analysis Demo ===")
    
    # Create model
    model = MoEModel(
        vocab_size=1000,
        hidden_size=256,
        num_experts=6,
        experts_per_token=2,
        num_layers=3
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create analyzer
    analyzer = RouterAnalyzer(model)
    
    # Create test input
    test_input = {
        'input_ids': torch.randint(0, 1000, (2, 32)).to(device)
    }
    
    # Analyze routing decisions
    analysis = analyzer.analyze_batch(test_input, return_attention_maps=True)
    
    logger.info("Batch Analysis Results:")
    logger.info(f"  Batch Size: {analysis['batch_size']}")
    logger.info(f"  Sequence Length: {analysis['sequence_length']}")
    logger.info(f"  Load Variance: {analysis['load_variance']:.4f}")
    logger.info(f"  Routing Entropy: {analysis['entropy']:.3f}")
    
    logger.info("Expert Utilization in Batch:")
    for expert_idx, util in analysis['expert_utilization'].items():
        logger.info(f"  Expert {expert_idx}: {util:.2%}")
        
    # Test routing consistency
    consistency = analyzer.analyze_routing_consistency(test_input, num_trials=5)
    
    logger.info("Routing Consistency:")
    logger.info(f"  Mean Position Consistency: {consistency['mean_position_consistency']:.3f}")
    logger.info(f"  Routing Variance: {consistency['routing_variance']:.4f}")
    logger.info(f"  Expert Usage Variance: {consistency['expert_usage_variance']:.2f}")
    
    return analysis


def demo_cost_analysis():
    """Demonstrate cost analysis capabilities."""
    logger.info("\n=== Cost Analysis Demo ===")
    
    # Create model
    model = MoEModel(
        vocab_size=32000,
        hidden_size=512,
        num_experts=16,
        experts_per_token=2,
        num_layers=12
    )
    
    # Create cost analyzer
    cost_analyzer = MoECostAnalyzer(model, hardware_profile="a100_80gb")
    
    # Analyze costs for different configurations
    configs = [
        {'batch_size': 1, 'sequence_length': 512},
        {'batch_size': 8, 'sequence_length': 1024},
        {'batch_size': 32, 'sequence_length': 2048}
    ]
    
    logger.info("Cost Analysis Results:")
    
    for i, config in enumerate(configs):
        cost_report = cost_analyzer.analyze(**config)
        
        logger.info(f"\nConfiguration {i+1} (B={config['batch_size']}, S={config['sequence_length']}):")
        logger.info(f"  FLOPs per token: {cost_report.flops_per_token:.2e}")
        logger.info(f"  Memory bandwidth: {cost_report.memory_bandwidth_gb:.2f} GB")
        logger.info(f"  Throughput: {cost_report.throughput:.1f} tokens/sec")
        logger.info(f"  Memory usage: {cost_report.memory_usage_mb:.1f} MB")
        logger.info(f"  Inference latency: {cost_report.inference_latency_ms:.2f} ms")
        
    # Compare with dense model
    dense_comparison = cost_analyzer.compare_with_dense(
        hidden_size=512,
        num_layers=12
    )
    
    logger.info("\nDense Model Comparison:")
    logger.info(f"  Compute reduction: {dense_comparison['compute_reduction']:.1%}")
    logger.info(f"  Parameter reduction: {dense_comparison['parameter_reduction']:.1%}")
    logger.info(f"  Dense FLOPs/token: {dense_comparison['dense_flops_per_token']:.2e}")
    logger.info(f"  MoE FLOPs/token: {dense_comparison['moe_flops_per_token']:.2e}")
    
    return cost_report


def run_simple_training_with_analytics():
    """Demonstrate training with analytics integration."""
    logger.info("\n=== Training with Analytics Demo ===")
    
    # Create model and data
    model = MoEModel(
        vocab_size=1000,
        hidden_size=128,
        num_experts=4,
        experts_per_token=2,
        num_layers=2
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create datasets
    train_dataset = create_sample_data(vocab_size=1000, seq_len=32, num_samples=20)
    eval_dataset = create_sample_data(vocab_size=1000, seq_len=32, num_samples=5)
    
    # Initialize trainer
    trainer = MoETrainer(
        model=model,
        aux_loss_coef=0.01,
        router_z_loss_coef=0.001,
        logging_steps=5,
        eval_steps=10
    )
    
    # Initialize monitoring
    monitor = RouterMonitor(model)
    
    # Train with monitoring
    with monitor.track():
        logger.info("Training with analytics...")
        
        result = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            batch_size=4,
            learning_rate=1e-3,
            num_epochs=2
        )
        
    # Get final statistics
    final_stats = monitor.get_stats()
    
    logger.info("Training Results:")
    logger.info(f"  Final Loss: {result.loss:.4f}")
    logger.info(f"  Aux Loss: {result.aux_loss:.4f}")
    logger.info(f"  Router Z Loss: {result.router_z_loss:.4f}")
    
    logger.info("Final Router Statistics:")
    logger.info(f"  Load Variance: {final_stats.load_variance:.4f}")
    logger.info(f"  Routing Entropy: {final_stats.entropy:.3f}")
    logger.info(f"  Average Throughput: {final_stats.throughput_tokens_per_sec:.1f} tokens/sec")
    
    return result, final_stats


def main():
    """Run all Generation 1 analytics demos."""
    logger.info("Starting Generation 1 Analytics Demo")
    logger.info("=" * 50)
    
    results = {}
    
    try:
        # Router monitoring demo
        monitoring_stats = demo_router_monitoring()
        results['monitoring'] = monitoring_stats
        
        # Router analysis demo
        analysis_results = demo_router_analysis()
        results['analysis'] = analysis_results
        
        # Cost analysis demo
        cost_results = demo_cost_analysis()
        results['cost'] = cost_results
        
        # Training with analytics demo
        training_results, training_stats = run_simple_training_with_analytics()
        results['training'] = {
            'training_result': training_results,
            'final_stats': training_stats
        }
        
        logger.info("\n" + "=" * 50)
        logger.info("Generation 1 Analytics Demo Completed Successfully!")
        logger.info("Key Capabilities Demonstrated:")
        logger.info("✓ Real-time router monitoring")
        logger.info("✓ Routing decision analysis") 
        logger.info("✓ Cost and performance analysis")
        logger.info("✓ Training integration with analytics")
        logger.info("✓ Expert utilization tracking")
        logger.info("✓ Hardware-aware cost estimation")
        
        return results
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    results = main()
    
    # Save results
    import json
    
    # Convert results to JSON-serializable format
    json_results = {}
    for key, value in results.items():
        if hasattr(value, '__dict__'):
            json_results[key] = value.__dict__
        else:
            json_results[key] = str(value)
            
    with open('generation1_analytics_results.json', 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
        
    print("Results saved to generation1_analytics_results.json")