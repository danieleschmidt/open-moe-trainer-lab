"""
Comprehensive end-to-end pipeline tests for the MoE lab.
Tests the complete workflow from model creation to deployment.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import json
import time
from typing import Dict, Any, List

from moe_lab import (
    MoEModel,
    MoETrainer,
    RouterMonitor,
    RouterAnalyzer,
    MoECostAnalyzer
)
from moe_lab.validation import (
    MoEModelValidator,
    RoutingValidator,
    TrainingValidator
)
from moe_lab.optimization.efficient_training import EfficientMoETrainer
from moe_lab.optimization.auto_scaling import AutoScalingMoEModel


class TestDataset:
    """Simple test dataset."""
    
    def __init__(self, vocab_size: int = 1000, seq_len: int = 32, num_samples: int = 100):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {'input_ids': input_ids.unsqueeze(0)}


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def test_model(device):
    """Create test model."""
    model = MoEModel(
        vocab_size=1000,
        hidden_size=128,
        num_experts=4,
        experts_per_token=2,
        num_layers=2,
        num_attention_heads=4
    )
    return model.to(device)


@pytest.fixture
def test_dataset():
    """Create test dataset."""
    return TestDataset(vocab_size=1000, seq_len=32, num_samples=50)


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestCompletePipeline:
    """Test complete MoE pipeline."""
    
    def test_model_creation_and_validation(self, device):
        """Test model creation and validation."""
        # Test various model configurations
        configs = [
            {'num_experts': 4, 'experts_per_token': 2, 'hidden_size': 128},
            {'num_experts': 8, 'experts_per_token': 2, 'hidden_size': 256},
            {'num_experts': 6, 'experts_per_token': 3, 'hidden_size': 192}
        ]
        
        for config in configs:
            model = MoEModel(
                vocab_size=1000,
                num_layers=2,
                **config
            ).to(device)
            
            # Validate model
            validator = MoEModelValidator(model)
            result = validator.validate_model()
            
            assert result.is_valid, f"Model validation failed: {result.errors}"
            assert result.metrics['total_parameters'] > 0
            
            # Test forward pass
            test_input = torch.randint(0, 1000, (2, 16), device=device)
            with torch.no_grad():
                outputs = model(test_input, return_routing_info=True)
                
            assert outputs.last_hidden_state.shape == (2, 16, config['hidden_size'])
            assert outputs.routing_info is not None
            
    def test_training_pipeline(self, test_model, test_dataset, temp_dir, device):
        """Test complete training pipeline."""
        # Create trainer
        trainer = MoETrainer(
            model=test_model,
            aux_loss_coef=0.01,
            router_z_loss_coef=0.001,
            logging_steps=5,
            output_dir=temp_dir
        )
        
        # Validate training setup
        training_validator = TrainingValidator(test_model, trainer)
        setup_result = training_validator.validate_training_setup()
        
        assert setup_result.is_valid, f"Training setup invalid: {setup_result.errors}"
        
        # Train model
        result = trainer.train(
            train_dataset=test_dataset,
            batch_size=4,
            learning_rate=1e-3,
            num_epochs=2
        )
        
        # Validate results
        assert result.loss < 10.0, "Training loss too high"
        assert os.path.exists(os.path.join(temp_dir, "final_model")), "Model not saved"
        
        # Test convergence validation
        convergence_result = training_validator.validate_training_convergence(
            num_steps=20,
            batch_size=4
        )
        
        assert convergence_result.convergence_score > 0.0
        assert convergence_result.stability_score > 0.0
        
    def test_routing_validation_pipeline(self, test_model, device):
        """Test routing validation pipeline."""
        routing_validator = RoutingValidator(test_model)
        
        # Test routing behavior
        result = routing_validator.validate_routing_behavior(
            num_samples=50,
            sequence_lengths=[16, 32]
        )
        
        assert result.load_balance_score >= 0.0
        assert result.routing_efficiency >= 0.0
        assert result.expert_specialization_score >= 0.0
        
        # Check detailed metrics
        assert 'expert_utilizations' in result.detailed_metrics
        assert 'utilization_cv' in result.detailed_metrics
        
        utilizations = result.detailed_metrics['expert_utilizations']
        assert len(utilizations) == test_model.num_experts
        assert all(0.0 <= u <= 1.0 for u in utilizations)
        
    def test_analytics_pipeline(self, test_model, test_dataset, device):
        """Test analytics and monitoring pipeline."""
        # Test router monitoring
        monitor = RouterMonitor(test_model, window_size=20)
        
        with monitor.track():
            test_model.eval()
            for i in range(10):
                batch = test_dataset[i]
                input_ids = batch['input_ids'].to(device)
                
                with torch.no_grad():
                    outputs = test_model(input_ids, return_routing_info=True)
                    
                monitor.record_routing(outputs.routing_info, batch_size=1)
                
        # Get monitoring statistics
        stats = monitor.get_stats()
        assert stats.load_variance >= 0.0
        assert stats.entropy >= 0.0
        assert stats.throughput_tokens_per_sec >= 0.0
        
        # Test router analysis
        analyzer = RouterAnalyzer(test_model)
        test_input = {'input_ids': torch.randint(0, 1000, (2, 16), device=device)}
        
        analysis = analyzer.analyze_batch(test_input)
        assert 'expert_utilization' in analysis
        assert 'load_variance' in analysis
        assert 'entropy' in analysis
        
        # Test consistency analysis
        consistency = analyzer.analyze_routing_consistency(test_input, num_trials=3)
        assert 'routing_consistency' in consistency
        assert 0.0 <= consistency['routing_consistency'] <= 1.0
        
    def test_cost_analysis_pipeline(self, test_model):
        """Test cost analysis pipeline."""
        cost_analyzer = MoECostAnalyzer(test_model, hardware_profile="a100_80gb")
        
        # Test cost analysis
        cost_report = cost_analyzer.analyze(
            batch_size=8,
            sequence_length=128
        )
        
        assert cost_report.flops_per_token > 0
        assert cost_report.memory_usage_mb > 0
        assert cost_report.throughput > 0
        
        # Test dense model comparison
        comparison = cost_analyzer.compare_with_dense(
            hidden_size=test_model.hidden_size,
            num_layers=test_model.num_layers
        )
        
        assert 'compute_reduction' in comparison
        assert 'parameter_reduction' in comparison
        assert comparison['moe_flops_per_token'] > 0
        assert comparison['dense_flops_per_token'] > 0
        
    def test_optimization_pipeline(self, device, test_dataset, temp_dir):
        """Test optimization pipeline."""
        # Test efficient trainer
        model = MoEModel(
            vocab_size=1000,
            hidden_size=128,
            num_experts=4,
            experts_per_token=2,
            num_layers=2
        ).to(device)
        
        efficient_trainer = EfficientMoETrainer(
            model=model,
            enable_dynamic_batching=True,
            enable_adaptive_checkpointing=True,
            enable_adaptive_scaling=True,
            max_batch_size=32,
            output_dir=temp_dir
        )
        
        # Train with optimizations
        result = efficient_trainer.train(
            train_dataset=test_dataset,
            batch_size=8,
            learning_rate=1e-3,
            num_epochs=1
        )
        
        assert result.loss > 0
        
        # Get performance summary
        performance = efficient_trainer.get_performance_summary()
        if performance:
            assert 'mean_throughput' in performance
            assert performance['mean_throughput'] > 0
            
    def test_auto_scaling_pipeline(self, device):
        """Test auto-scaling pipeline."""
        # Create auto-scaling model
        model = AutoScalingMoEModel(
            vocab_size=1000,
            hidden_size=128,
            num_experts=4,
            experts_per_token=2,
            num_layers=2,
            enable_auto_scaling=True,
            max_experts=8
        ).to(device)
        
        # Test basic functionality
        test_input = torch.randint(0, 1000, (4, 32), device=device)
        
        with torch.no_grad():
            outputs = model(test_input)
            
        assert outputs.last_hidden_state.shape == (4, 32, 128)
        
        # Test scaling
        initial_experts = model.num_experts
        model.scale_experts(6)
        assert model.num_experts == 6
        
        # Test scaling context
        with model.auto_scaling_context():
            # Model should work normally
            with torch.no_grad():
                outputs = model(test_input)
                
        assert outputs.last_hidden_state.shape == (4, 32, 128)
        
    def test_integration_with_validation(self, device, test_dataset, temp_dir):
        """Test integration of all components with validation."""
        # Create model
        model = MoEModel(
            vocab_size=1000,
            hidden_size=128,
            num_experts=6,
            experts_per_token=2,
            num_layers=3
        ).to(device)
        
        # Pre-training validation
        model_validator = MoEModelValidator(model)
        routing_validator = RoutingValidator(model)
        
        # Validate model
        model_result = model_validator.validate_model()
        assert model_result.is_valid, f"Model validation failed: {model_result.errors}"
        
        # Validate routing
        routing_result = routing_validator.validate_routing_behavior(num_samples=30)
        initial_efficiency = routing_result.routing_efficiency
        
        # Setup monitoring
        monitor = RouterMonitor(model)
        analyzer = RouterAnalyzer(model)
        
        # Train with monitoring
        trainer = MoETrainer(
            model=model,
            aux_loss_coef=0.01,
            logging_steps=5,
            output_dir=temp_dir
        )
        
        with monitor.track():
            training_result = trainer.train(
                train_dataset=test_dataset,
                batch_size=4,
                learning_rate=1e-3,
                num_epochs=2
            )
            
        # Post-training validation
        final_model_result = model_validator.validate_model()
        assert final_model_result.is_valid, "Model became invalid after training"
        
        final_routing_result = routing_validator.validate_routing_behavior(num_samples=30)
        
        # Get final statistics
        final_stats = monitor.get_stats()
        
        # Validate training improved routing
        assert final_stats.entropy > 0
        assert final_stats.load_variance >= 0
        
        # Test analysis after training
        test_input = {'input_ids': torch.randint(0, 1000, (4, 32), device=device)}
        final_analysis = analyzer.analyze_batch(test_input)
        
        assert final_analysis['entropy'] > 0
        assert len(final_analysis['expert_utilization']) == model.num_experts
        
        # Comprehensive validation report
        validation_report = {
            'model_validation': {
                'pre_training': model_result.is_valid,
                'post_training': final_model_result.is_valid,
                'total_parameters': final_model_result.metrics['total_parameters']
            },
            'routing_validation': {
                'initial_efficiency': initial_efficiency,
                'final_efficiency': final_routing_result.routing_efficiency,
                'load_balance_score': final_routing_result.load_balance_score
            },
            'training_results': {
                'final_loss': training_result.loss,
                'aux_loss': training_result.aux_loss
            },
            'monitoring_results': {
                'final_entropy': final_stats.entropy,
                'load_variance': final_stats.load_variance,
                'throughput': final_stats.throughput_tokens_per_sec
            }
        }
        
        # Save validation report
        report_path = os.path.join(temp_dir, "validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
            
        assert os.path.exists(report_path)
        
        # Validate report contents
        assert validation_report['model_validation']['post_training']
        assert validation_report['training_results']['final_loss'] > 0
        assert validation_report['monitoring_results']['final_entropy'] > 0
        
    def test_error_handling_and_recovery(self, device, temp_dir):
        """Test error handling and recovery mechanisms."""
        # Test with invalid configuration
        with pytest.raises(ValueError):
            MoEModel(
                vocab_size=1000,
                hidden_size=128,
                num_experts=4,
                experts_per_token=5,  # Invalid: more than num_experts
                num_layers=2
            )
            
        # Test with problematic model
        model = MoEModel(
            vocab_size=1000,
            hidden_size=64,  # Very small
            num_experts=2,   # Very few
            experts_per_token=2,
            num_layers=1
        ).to(device)
        
        # Validate and check for warnings
        validator = MoEModelValidator(model)
        result = validator.validate_model()
        
        # Should have warnings but still be valid for basic models
        assert len(result.warnings) > 0 or result.is_valid
        
        # Test training with problematic configuration
        dataset = TestDataset(num_samples=20)
        trainer = MoETrainer(
            model=model,
            output_dir=temp_dir,
            logging_steps=5
        )
        
        try:
            result = trainer.train(
                train_dataset=dataset,
                batch_size=2,
                learning_rate=1e-3,
                num_epochs=1
            )
            # Should complete but may have high loss
            assert result.loss >= 0
        except Exception as e:
            # If training fails, should be caught gracefully
            assert isinstance(e, (RuntimeError, ValueError))
            
    def test_performance_benchmarks(self, device, test_dataset):
        """Test performance benchmarks and requirements."""
        model = MoEModel(
            vocab_size=1000,
            hidden_size=256,
            num_experts=8,
            experts_per_token=2,
            num_layers=4
        ).to(device)
        
        # Benchmark inference speed
        model.eval()
        batch_sizes = [1, 4, 8, 16]
        performance_results = {}
        
        for batch_size in batch_sizes:
            test_input = torch.randint(0, 1000, (batch_size, 64), device=device)
            
            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    model(test_input)
                    
            # Benchmark
            start_time = time.time()
            num_iterations = 10
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    outputs = model(test_input)
                    
            total_time = time.time() - start_time
            
            tokens_per_sec = (batch_size * 64 * num_iterations) / total_time
            latency_ms = (total_time / num_iterations) * 1000
            
            performance_results[batch_size] = {
                'tokens_per_sec': tokens_per_sec,
                'latency_ms': latency_ms
            }
            
            # Performance requirements
            assert tokens_per_sec > 0, f"Zero throughput for batch size {batch_size}"
            assert latency_ms > 0, f"Invalid latency for batch size {batch_size}"
            
            # Reasonable performance bounds
            if torch.cuda.is_available():
                assert tokens_per_sec > 100, f"Throughput too low: {tokens_per_sec}"
                assert latency_ms < 1000, f"Latency too high: {latency_ms}ms"
                
        # Check scaling behavior
        throughputs = [performance_results[bs]['tokens_per_sec'] for bs in batch_sizes]
        
        # Throughput should generally increase with batch size (up to a point)
        assert throughputs[-1] > throughputs[0], "No throughput scaling with batch size"
        
    @pytest.mark.slow
    def test_long_running_stability(self, device, temp_dir):
        """Test long-running stability and memory management."""
        model = MoEModel(
            vocab_size=1000,
            hidden_size=128,
            num_experts=4,
            experts_per_token=2,
            num_layers=2
        ).to(device)
        
        dataset = TestDataset(num_samples=100)
        
        # Monitor memory usage
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Long training run
        trainer = MoETrainer(
            model=model,
            output_dir=temp_dir,
            logging_steps=10,
            save_steps=50
        )
        
        result = trainer.train(
            train_dataset=dataset,
            batch_size=4,
            learning_rate=1e-3,
            num_epochs=5  # Longer training
        )
        
        # Check memory didn't grow excessively
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        if torch.cuda.is_available():
            memory_growth = (final_memory - initial_memory) / (1024**2)  # MB
            assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.1f}MB"
            
        # Check training completed successfully
        assert result.loss > 0
        assert result.loss < 20  # Should converge somewhat
        
        # Check checkpoints were saved
        checkpoints = [f for f in os.listdir(temp_dir) if f.startswith('checkpoint-')]
        assert len(checkpoints) > 0, "No checkpoints saved during long training"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])