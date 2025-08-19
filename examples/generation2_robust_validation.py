"""
Generation 2 Robust Validation Demo
===================================

Comprehensive validation and error handling for MoE models.
Demonstrates robust error handling, validation, and monitoring.
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
from moe_lab.validation import (
    MoEModelValidator,
    RoutingValidator, 
    TrainingValidator
)
from moe_lab.utils.logging import get_logger

logger = get_logger(__name__)


def create_test_model(config: Dict[str, Any]) -> MoEModel:
    """Create test model with given configuration."""
    return MoEModel(
        vocab_size=config.get('vocab_size', 1000),
        hidden_size=config.get('hidden_size', 256),
        num_experts=config.get('num_experts', 8),
        experts_per_token=config.get('experts_per_token', 2),
        num_layers=config.get('num_layers', 4),
        num_attention_heads=config.get('num_attention_heads', 8)
    )


def create_sample_dataset(vocab_size: int = 1000, seq_len: int = 64, num_samples: int = 100):
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


def demo_model_validation():
    """Demonstrate comprehensive model validation."""
    logger.info("=== Model Validation Demo ===")
    
    # Test with valid model
    logger.info("Testing valid model configuration...")
    valid_config = {
        'vocab_size': 1000,
        'hidden_size': 256,
        'num_experts': 8,
        'experts_per_token': 2,
        'num_layers': 4
    }
    
    valid_model = create_test_model(valid_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    valid_model = valid_model.to(device)
    
    validator = MoEModelValidator(valid_model)
    result = validator.validate_model()
    
    logger.info(f"Valid Model Validation: {'PASSED' if result.is_valid else 'FAILED'}")
    logger.info(f"  Errors: {len(result.errors)}")
    logger.info(f"  Warnings: {len(result.warnings)}")
    logger.info(f"  Total Parameters: {result.metrics.get('total_parameters', 'N/A')}")
    
    if result.errors:
        for error in result.errors:
            logger.error(f"  ERROR: {error}")
            
    if result.warnings:
        for warning in result.warnings[:3]:  # Show first 3 warnings
            logger.warning(f"  WARNING: {warning}")
            
    # Test with problematic model
    logger.info("\nTesting problematic model configuration...")
    
    try:
        # Create model with potential issues
        problematic_config = {
            'vocab_size': 1000,
            'hidden_size': 64,  # Very small
            'num_experts': 2,   # Very few experts
            'experts_per_token': 3,  # More than num_experts (invalid)
            'num_layers': 1
        }
        
        problematic_model = create_test_model(problematic_config)
        problematic_model = problematic_model.to(device)
        
        validator = MoEModelValidator(problematic_model)
        result = validator.validate_model()
        
        logger.info(f"Problematic Model Validation: {'PASSED' if result.is_valid else 'FAILED'}")
        logger.info(f"  Errors: {len(result.errors)}")
        logger.info(f"  Warnings: {len(result.warnings)}")
        
        if result.errors:
            for error in result.errors:
                logger.error(f"  ERROR: {error}")
                
    except Exception as e:
        logger.info(f"Expected validation failure: {str(e)}")
        
    # Test expert utilization validation
    logger.info("\nTesting expert utilization validation...")
    
    utilization_result = validator.validate_expert_utilization(num_samples=50)
    
    logger.info(f"Utilization Validation: {'PASSED' if utilization_result.is_valid else 'FAILED'}")
    if 'expert_utilizations' in utilization_result.metrics:
        utils = utilization_result.metrics['expert_utilizations']
        logger.info(f"  Expert Utilizations: {[f'{u:.3f}' for u in utils]}")
        logger.info(f"  Utilization Variance: {utilization_result.metrics.get('utilization_variance', 'N/A'):.4f}")
        
    return result


def demo_routing_validation():
    """Demonstrate routing behavior validation."""
    logger.info("\n=== Routing Validation Demo ===")
    
    # Create model for routing tests
    model = create_test_model({
        'vocab_size': 1000,
        'hidden_size': 128,
        'num_experts': 6,
        'experts_per_token': 2,
        'num_layers': 3
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize routing validator
    routing_validator = RoutingValidator(model, tolerance=0.1)
    
    # Run comprehensive routing validation
    logger.info("Running comprehensive routing validation...")
    
    result = routing_validator.validate_routing_behavior(
        num_samples=100,
        sequence_lengths=[16, 32, 64]
    )
    
    logger.info(f"Routing Validation: {'PASSED' if result.is_valid else 'FAILED'}")
    logger.info(f"  Load Balance Score: {result.load_balance_score:.3f}")
    logger.info(f"  Routing Efficiency: {result.routing_efficiency:.3f}")
    logger.info(f"  Expert Specialization: {result.expert_specialization_score:.3f}")
    logger.info(f"  Errors: {len(result.errors)}")
    logger.info(f"  Warnings: {len(result.warnings)}")
    
    # Show detailed metrics
    if 'expert_utilizations' in result.detailed_metrics:
        utils = result.detailed_metrics['expert_utilizations']
        logger.info(f"  Expert Utilizations: {[f'{u:.3f}' for u in utils]}")
        
    if 'utilization_cv' in result.detailed_metrics:
        cv = result.detailed_metrics['utilization_cv']
        logger.info(f"  Utilization Coefficient of Variation: {cv:.3f}")
        
    if 'normalized_entropy' in result.detailed_metrics:
        entropy = result.detailed_metrics['normalized_entropy']
        logger.info(f"  Normalized Routing Entropy: {entropy:.3f}")
        
    # Show routing issues
    if result.errors:
        logger.info("  Routing Errors:")
        for error in result.errors:
            logger.error(f"    - {error}")
            
    if result.warnings:
        logger.info("  Routing Warnings:")
        for warning in result.warnings[:3]:
            logger.warning(f"    - {warning}")
            
    return result


def demo_training_validation():
    """Demonstrate training validation."""
    logger.info("\n=== Training Validation Demo ===")
    
    # Create model and trainer
    model = create_test_model({
        'vocab_size': 1000,
        'hidden_size': 128,
        'num_experts': 4,
        'experts_per_token': 2,
        'num_layers': 2
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    trainer = MoETrainer(
        model=model,
        aux_loss_coef=0.01,
        router_z_loss_coef=0.001,
        max_grad_norm=1.0,
        logging_steps=5
    )
    
    # Initialize training validator
    training_validator = TrainingValidator(model, trainer)
    
    # Validate training setup
    logger.info("Validating training setup...")
    
    setup_result = training_validator.validate_training_setup()
    
    logger.info(f"Training Setup Validation: {'PASSED' if setup_result.is_valid else 'FAILED'}")
    logger.info(f"  Errors: {len(setup_result.errors)}")
    logger.info(f"  Warnings: {len(setup_result.warnings)}")
    
    if 'trainable_parameters' in setup_result.training_metrics:
        trainable = setup_result.training_metrics['trainable_parameters']
        total = setup_result.training_metrics['total_parameters']
        logger.info(f"  Trainable Parameters: {trainable:,} / {total:,}")
        
    if setup_result.errors:
        for error in setup_result.errors:
            logger.error(f"  SETUP ERROR: {error}")
            
    # Validate training convergence
    logger.info("\nValidating training convergence...")
    
    convergence_result = training_validator.validate_training_convergence(
        num_steps=30,
        batch_size=4,
        learning_rate=1e-3
    )
    
    logger.info(f"Training Convergence Validation: {'PASSED' if convergence_result.is_valid else 'FAILED'}")
    logger.info(f"  Convergence Score: {convergence_result.convergence_score:.3f}")
    logger.info(f"  Stability Score: {convergence_result.stability_score:.3f}")
    logger.info(f"  Efficiency Score: {convergence_result.efficiency_score:.3f}")
    
    # Show training metrics
    metrics = convergence_result.training_metrics
    if 'loss_reduction' in metrics:
        logger.info(f"  Loss Reduction: {metrics['loss_reduction']:.3f}")
        
    if 'loss_cv' in metrics:
        logger.info(f"  Loss Coefficient of Variation: {metrics['loss_cv']:.3f}")
        
    if 'total_gradient_norm' in metrics:
        logger.info(f"  Total Gradient Norm: {metrics['total_gradient_norm']:.3f}")
        
    # Memory metrics
    if 'backward_memory_mb' in metrics:
        logger.info(f"  Peak Memory Usage: {metrics['backward_memory_mb']:.1f} MB")
        
    if convergence_result.errors:
        logger.info("  Training Errors:")
        for error in convergence_result.errors:
            logger.error(f"    - {error}")
            
    return convergence_result


def demo_robust_training_with_validation():
    """Demonstrate robust training with integrated validation."""
    logger.info("\n=== Robust Training with Validation Demo ===")
    
    # Create model and data
    model = create_test_model({
        'vocab_size': 1000,
        'hidden_size': 128,
        'num_experts': 4,
        'experts_per_token': 2,
        'num_layers': 2
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    train_dataset = create_sample_dataset(vocab_size=1000, seq_len=32, num_samples=40)
    eval_dataset = create_sample_dataset(vocab_size=1000, seq_len=32, num_samples=10)
    
    # Initialize validators
    model_validator = MoEModelValidator(model)
    routing_validator = RoutingValidator(model)
    
    # Pre-training validation
    logger.info("Pre-training validation...")
    
    pre_model_result = model_validator.validate_model()
    if not pre_model_result.is_valid:
        logger.error("Model validation failed before training!")
        for error in pre_model_result.errors:
            logger.error(f"  - {error}")
        return None
        
    pre_routing_result = routing_validator.validate_routing_behavior(num_samples=20)
    logger.info(f"Pre-training routing efficiency: {pre_routing_result.routing_efficiency:.3f}")
    
    # Setup trainer with validation
    trainer = MoETrainer(
        model=model,
        aux_loss_coef=0.01,
        router_z_loss_coef=0.001,
        logging_steps=5,
        eval_steps=10,
        output_dir="./robust_checkpoints"
    )
    
    # Setup monitoring
    monitor = RouterMonitor(model, window_size=100)
    
    # Training with monitoring and validation
    logger.info("Starting robust training with monitoring...")
    
    try:
        with monitor.track():
            result = trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                batch_size=4,
                learning_rate=1e-3,
                num_epochs=2
            )
            
        # Post-training validation
        logger.info("Post-training validation...")
        
        post_model_result = model_validator.validate_model()
        post_routing_result = routing_validator.validate_routing_behavior(num_samples=30)
        
        # Get final monitoring stats
        final_stats = monitor.get_stats()
        
        # Collect all results
        validation_summary = {
            'training_result': {
                'final_loss': result.loss,
                'aux_loss': result.aux_loss,
                'router_z_loss': result.router_z_loss
            },
            'pre_training_validation': {
                'model_valid': pre_model_result.is_valid,
                'routing_efficiency': pre_routing_result.routing_efficiency,
                'load_balance_score': pre_routing_result.load_balance_score
            },
            'post_training_validation': {
                'model_valid': post_model_result.is_valid,
                'routing_efficiency': post_routing_result.routing_efficiency,
                'load_balance_score': post_routing_result.load_balance_score
            },
            'final_monitoring': {
                'load_variance': final_stats.load_variance,
                'routing_entropy': final_stats.entropy,
                'throughput': final_stats.throughput_tokens_per_sec,
                'expert_utilization': final_stats.expert_utilization
            }
        }
        
        # Report summary
        logger.info("Training Validation Summary:")
        logger.info(f"  Training completed successfully: {result.loss:.4f} final loss")
        logger.info(f"  Model validation: {post_model_result.is_valid}")
        logger.info(f"  Routing efficiency improvement: {pre_routing_result.routing_efficiency:.3f} → {post_routing_result.routing_efficiency:.3f}")
        logger.info(f"  Load balancing improvement: {pre_routing_result.load_balance_score:.3f} → {post_routing_result.load_balance_score:.3f}")
        logger.info(f"  Final routing entropy: {final_stats.entropy:.3f}")
        logger.info(f"  Training throughput: {final_stats.throughput_tokens_per_sec:.1f} tokens/sec")
        
        return validation_summary
        
    except Exception as e:
        logger.error(f"Robust training failed: {str(e)}")
        return None


def demo_error_recovery():
    """Demonstrate error detection and recovery."""
    logger.info("\n=== Error Recovery Demo ===")
    
    # Test with intentionally problematic configurations
    problematic_configs = [
        {
            'name': 'Invalid experts_per_token',
            'config': {'experts_per_token': 10, 'num_experts': 4}
        },
        {
            'name': 'Zero hidden_size',
            'config': {'hidden_size': 0}
        },
        {
            'name': 'Negative num_layers',
            'config': {'num_layers': -1}
        }
    ]
    
    recovery_results = []
    
    for test_case in problematic_configs:
        logger.info(f"\nTesting error recovery for: {test_case['name']}")
        
        try:
            # Create problematic model
            base_config = {
                'vocab_size': 1000,
                'hidden_size': 128,
                'num_experts': 4,
                'experts_per_token': 2,
                'num_layers': 2
            }
            base_config.update(test_case['config'])
            
            model = create_test_model(base_config)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Validate and attempt recovery
            validator = MoEModelValidator(model)
            result = validator.validate_model()
            
            recovery_info = {
                'test_case': test_case['name'],
                'validation_passed': result.is_valid,
                'errors_detected': len(result.errors),
                'warnings': len(result.warnings),
                'recovery_attempted': False,
                'recovery_successful': False
            }
            
            if not result.is_valid:
                logger.warning(f"  Detected {len(result.errors)} errors:")
                for error in result.errors[:2]:
                    logger.warning(f"    - {error}")
                    
                # Attempt automatic recovery (simplified)
                logger.info("  Attempting automatic recovery...")
                recovery_info['recovery_attempted'] = True
                
                # Simple recovery strategies
                if 'experts_per_token' in test_case['config']:
                    if base_config['experts_per_token'] > base_config['num_experts']:
                        base_config['experts_per_token'] = min(2, base_config['num_experts'])
                        logger.info(f"    Corrected experts_per_token to {base_config['experts_per_token']}")
                        
                # Re-test after recovery
                try:
                    recovered_model = create_test_model(base_config)
                    recovered_model = recovered_model.to(device)
                    
                    recovery_result = validator.validate_model()
                    if recovery_result.is_valid:
                        recovery_info['recovery_successful'] = True
                        logger.info("    Recovery successful!")
                    else:
                        logger.warning("    Recovery failed")
                        
                except Exception as recovery_error:
                    logger.error(f"    Recovery attempt failed: {str(recovery_error)}")
                    
            else:
                logger.info("  No errors detected")
                
            recovery_results.append(recovery_info)
            
        except Exception as e:
            logger.error(f"  Test case failed with exception: {str(e)}")
            recovery_results.append({
                'test_case': test_case['name'],
                'validation_passed': False,
                'errors_detected': 1,
                'warnings': 0,
                'recovery_attempted': False,
                'recovery_successful': False,
                'exception': str(e)
            })
            
    return recovery_results


def main():
    """Run all Generation 2 robust validation demos."""
    logger.info("Starting Generation 2 Robust Validation Demo")
    logger.info("=" * 60)
    
    results = {}
    
    try:
        # Model validation demo
        model_result = demo_model_validation()
        results['model_validation'] = model_result
        
        # Routing validation demo
        routing_result = demo_routing_validation()
        results['routing_validation'] = routing_result
        
        # Training validation demo
        training_result = demo_training_validation()
        results['training_validation'] = training_result
        
        # Robust training demo
        robust_training_result = demo_robust_training_with_validation()
        results['robust_training'] = robust_training_result
        
        # Error recovery demo
        error_recovery_result = demo_error_recovery()
        results['error_recovery'] = error_recovery_result
        
        logger.info("\n" + "=" * 60)
        logger.info("Generation 2 Robust Validation Demo Completed Successfully!")
        logger.info("Key Capabilities Demonstrated:")
        logger.info("✓ Comprehensive model validation")
        logger.info("✓ Routing behavior analysis and validation")
        logger.info("✓ Training convergence and stability validation")
        logger.info("✓ Integrated validation during training")
        logger.info("✓ Error detection and recovery mechanisms")
        logger.info("✓ Memory usage and performance validation")
        logger.info("✓ Expert utilization balance validation")
        logger.info("✓ Gradient flow and optimization validation")
        
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
            if hasattr(value, '__dict__'):
                json_results[key] = value.__dict__
            elif isinstance(value, dict):
                json_results[key] = value
            else:
                json_results[key] = str(value)
                
        with open('generation2_robust_validation_results.json', 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
            
        print("Results saved to generation2_robust_validation_results.json")
        
    except Exception as e:
        print(f"Failed to save results: {e}")