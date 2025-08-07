#!/usr/bin/env python3
"""
Complete implementation test for MoE Trainer Lab.
This test verifies all core functionality is working properly.
"""

import os
import sys
import json
import tempfile
import logging
from pathlib import Path
import torch
import pytest
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from moe_lab.models import MoEModel
from moe_lab.training import MoETrainer
from moe_lab.inference import OptimizedMoEModel
from moe_lab.data.datasets import create_sample_dataset, create_domain_dataset
from moe_lab.data.collators import MoEDataCollator
from moe_lab.utils.logging import setup_logging
from moe_lab.utils.validation import ConfigValidator, validate_and_suggest
from moe_lab.utils.error_handling import (
    setup_error_handling, with_error_handling, MoETrainingError, 
    CheckpointManager, GradientMonitor
)

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMoEImplementation:
    """Comprehensive test suite for MoE implementation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield tmp_dir
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'model': {
                'vocab_size': 1000,
                'hidden_size': 256,
                'num_experts': 4,
                'experts_per_token': 2,
                'num_layers': 6,
                'num_attention_heads': 8,
                'max_position_embeddings': 512,
                'moe_layers': [1, 3, 5],
                'aux_loss_coef': 0.01,
                'z_loss_coef': 0.001
            },
            'training': {
                'learning_rate': 3e-4,
                'per_device_train_batch_size': 4,
                'num_epochs': 2,
                'gradient_accumulation_steps': 2,
                'max_grad_norm': 1.0,
                'warmup_steps': 10,
                'logging_steps': 5,
                'save_steps': 20,
                'eval_steps': 15,
                'fp16': True
            },
            'data': {
                'max_seq_length': 128,
                'num_workers': 2
            }
        }
    
    def test_model_creation(self, sample_config):
        """Test MoE model creation and basic functionality."""
        logger.info("Testing MoE model creation...")
        
        model_config = sample_config['model']
        model = MoEModel(**model_config)
        
        # Test model structure
        assert model.hidden_size == model_config['hidden_size']
        assert model.num_experts == model_config['num_experts']
        assert model.experts_per_token == model_config['experts_per_token']
        assert len(model.moe_layers) == len(model_config['moe_layers'])
        
        # Test forward pass
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        assert outputs.last_hidden_state.shape == (batch_size, seq_len, model_config['hidden_size'])
        assert outputs.routing_info is not None
        assert outputs.load_balancing_loss is not None
        
        logger.info("✅ Model creation test passed")
    
    def test_data_creation(self):
        """Test data creation and processing."""
        logger.info("Testing data creation...")
        
        # Test sample dataset
        sample_dataset = create_sample_dataset(num_samples=50, max_length=64)
        assert len(sample_dataset) == 50
        
        sample = sample_dataset[0]
        assert 'input_ids' in sample
        assert 'attention_mask' in sample
        assert 'labels' in sample
        
        # Test domain dataset
        domain_dataset = create_domain_dataset(
            domains=['math', 'science'],
            samples_per_domain=25,
            max_length=64
        )
        assert len(domain_dataset) == 50
        
        # Test collator
        collator = MoEDataCollator(max_length=64)
        batch = collator([sample_dataset[i] for i in range(4)])
        
        assert batch['input_ids'].shape[0] == 4
        assert batch['attention_mask'].shape[0] == 4
        assert batch['labels'].shape[0] == 4
        
        logger.info("✅ Data creation test passed")
    
    def test_training_basic(self, sample_config, temp_dir):
        """Test basic training functionality."""
        logger.info("Testing basic training...")
        
        # Create model and data
        model = MoEModel(**sample_config['model'])
        train_dataset = create_sample_dataset(num_samples=20, max_length=32)
        
        # Setup trainer
        trainer = MoETrainer(
            model=model,
            output_dir=temp_dir,
            logging_steps=5,
            save_steps=50,
            eval_steps=50,
            use_mixed_precision=False  # Disable for CPU testing
        )
        
        # Run short training
        result = trainer.train(
            train_dataset=train_dataset,
            batch_size=2,
            learning_rate=1e-3,
            num_epochs=1
        )
        
        assert result.loss > 0
        assert len(trainer.history['train_loss']) > 0
        
        # Test saving
        save_path = Path(temp_dir) / "test_model"
        trainer.save_model(str(save_path))
        assert (save_path / "pytorch_model.bin").exists()
        assert (save_path / "training_state.json").exists()
        
        logger.info("✅ Basic training test passed")
    
    def test_inference_optimization(self, sample_config, temp_dir):
        """Test inference optimization."""
        logger.info("Testing inference optimization...")
        
        # Create and save model
        model = MoEModel(**sample_config['model'])
        save_path = Path(temp_dir) / "model"
        save_path.mkdir(exist_ok=True)
        
        torch.save(model.state_dict(), save_path / "pytorch_model.bin")
        with open(save_path / "config.json", 'w') as f:
            json.dump(sample_config['model'], f)
        
        # Load optimized model
        optimized_model = OptimizedMoEModel.from_pretrained(
            str(save_path),
            device_map="cpu",
            torch_dtype=torch.float32,
            expert_cache_size=2
        )
        
        # Test inference
        input_ids = torch.randint(0, 100, (1, 16))
        with torch.no_grad():
            outputs = optimized_model(input_ids)
        
        assert outputs.last_hidden_state.shape[0] == 1
        
        # Test generation
        generated = optimized_model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            do_sample=False
        )
        assert generated.shape[1] > input_ids.shape[1]
        
        # Test performance stats
        stats = optimized_model.get_performance_stats()
        assert 'total_inference_steps' in stats
        assert 'cache_stats' in stats
        
        logger.info("✅ Inference optimization test passed")
    
    def test_configuration_validation(self, sample_config):
        """Test configuration validation."""
        logger.info("Testing configuration validation...")
        
        # Test valid config
        is_valid, _ = validate_and_suggest(sample_config, silent=True)
        assert is_valid
        
        # Test invalid config
        invalid_config = sample_config.copy()
        invalid_config['model']['hidden_size'] = -1
        
        is_valid, _ = validate_and_suggest(invalid_config, silent=True)
        assert not is_valid
        
        # Test specific validators
        model_result = ConfigValidator.validate_model_config(sample_config['model'])
        assert model_result.is_valid
        
        training_result = ConfigValidator.validate_training_config(sample_config['training'])
        assert training_result.is_valid
        
        logger.info("✅ Configuration validation test passed")
    
    def test_error_handling(self, temp_dir):
        """Test error handling system."""
        logger.info("Testing error handling...")
        
        # Setup error handler
        error_handler = setup_error_handling(
            log_file=os.path.join(temp_dir, "errors.log"),
            max_retries=2
        )
        
        # Test error handling decorator
        @with_error_handling(ValueError, recovery_suggestion="Check your input values")
        def failing_function(should_fail=True):
            if should_fail:
                raise ValueError("Test error")
            return "success"
        
        # Test handling
        try:
            failing_function(should_fail=True)
            assert False, "Should have raised error"
        except MoETrainingError as e:
            assert e.recovery_suggestion is not None
        
        # Test retry mechanism
        call_count = 0
        def sometimes_failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Temporary failure")
            return "success"
        
        result = error_handler.retry_with_backoff(sometimes_failing_func, backoff_factor=0.1)
        assert result == "success"
        assert call_count == 2
        
        # Test error summary
        summary = error_handler.get_error_summary()
        assert summary['total_errors'] > 0
        
        logger.info("✅ Error handling test passed")
    
    def test_checkpoint_management(self, temp_dir):
        """Test checkpoint management."""
        logger.info("Testing checkpoint management...")
        
        checkpoint_manager = CheckpointManager(temp_dir)
        
        # Test save/load
        test_state = {'model': torch.randn(10, 10), 'step': 100}
        
        checkpoint_path = checkpoint_manager.save_checkpoint(test_state, step=100)
        assert os.path.exists(checkpoint_path)
        
        loaded_state = checkpoint_manager.load_checkpoint(checkpoint_path)
        assert loaded_state['step'] == 100
        assert torch.allclose(loaded_state['model'], test_state['model'])
        
        logger.info("✅ Checkpoint management test passed")
    
    def test_gradient_monitoring(self, sample_config):
        """Test gradient monitoring."""
        logger.info("Testing gradient monitoring...")
        
        model = MoEModel(**sample_config['model'])
        monitor = GradientMonitor(clip_threshold=5.0)
        
        # Simulate training step
        input_ids = torch.randint(0, 100, (2, 16))
        outputs = model(input_ids)
        loss = outputs.last_hidden_state.mean()
        loss.backward()
        
        # Check gradients
        grad_stats = monitor.check_gradients(model)
        
        assert 'total_norm' in grad_stats
        assert 'param_count' in grad_stats
        assert grad_stats['param_count'] > 0
        
        if grad_stats['anomalies']:
            logger.warning(f"Gradient anomalies detected: {grad_stats['anomalies']}")
        
        logger.info("✅ Gradient monitoring test passed")
    
    def test_complete_workflow(self, sample_config, temp_dir):
        """Test complete training workflow."""
        logger.info("Testing complete workflow...")
        
        # 1. Create model and data
        model = MoEModel(**sample_config['model'])
        train_dataset = create_sample_dataset(num_samples=30, max_length=32)
        eval_dataset = create_sample_dataset(num_samples=10, max_length=32)
        
        # 2. Setup trainer with monitoring
        trainer = MoETrainer(
            model=model,
            output_dir=temp_dir,
            logging_steps=3,
            eval_steps=10,
            save_steps=15,
            use_mixed_precision=False
        )
        
        # 3. Train model
        result = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            batch_size=2,
            learning_rate=5e-4,
            num_epochs=1
        )
        
        assert result.loss > 0
        
        # 4. Save model
        model_path = Path(temp_dir) / "final_model"
        trainer.save_model(str(model_path))
        
        # 5. Load for inference
        optimized_model = OptimizedMoEModel.from_pretrained(
            str(model_path),
            device_map="cpu",
            torch_dtype=torch.float32
        )
        
        # 6. Test inference
        test_input = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            outputs = optimized_model(test_input)
        
        assert outputs.last_hidden_state.shape[0] == 1
        
        # 7. Test generation
        generated = optimized_model.generate(
            test_input,
            max_new_tokens=5,
            temperature=1.0,
            do_sample=False
        )
        
        assert generated.shape[1] > test_input.shape[1]
        
        logger.info("✅ Complete workflow test passed")
    
    def run_all_tests(self):
        """Run all tests with proper setup."""
        logger.info("🚀 Starting comprehensive MoE implementation tests...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config = {
                'model': {
                    'vocab_size': 1000,
                    'hidden_size': 256,
                    'num_experts': 4,
                    'experts_per_token': 2,
                    'num_layers': 6,
                    'num_attention_heads': 8,
                    'max_position_embeddings': 512,
                    'moe_layers': [1, 3, 5],
                    'aux_loss_coef': 0.01,
                    'z_loss_coef': 0.001
                },
                'training': {
                    'learning_rate': 3e-4,
                    'per_device_train_batch_size': 4,
                    'num_epochs': 2,
                    'gradient_accumulation_steps': 2,
                    'max_grad_norm': 1.0,
                    'warmup_steps': 10,
                    'logging_steps': 5,
                    'save_steps': 20,
                    'eval_steps': 15,
                    'fp16': False  # Disable for CPU testing
                },
                'data': {
                    'max_seq_length': 128,
                    'num_workers': 2
                }
            }
            
            try:
                self.test_model_creation(sample_config)
                self.test_data_creation()
                self.test_training_basic(sample_config, temp_dir)
                self.test_inference_optimization(sample_config, temp_dir)
                self.test_configuration_validation(sample_config)
                self.test_error_handling(temp_dir)
                self.test_checkpoint_management(temp_dir)
                self.test_gradient_monitoring(sample_config)
                self.test_complete_workflow(sample_config, temp_dir)
                
                logger.info("🎉 ALL TESTS PASSED! MoE implementation is working correctly.")
                return True
                
            except Exception as e:
                logger.error(f"❌ Test failed: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False


def main():
    """Main test runner."""
    print("=" * 80)
    print("🧠 MoE Trainer Lab - Complete Implementation Test")
    print("=" * 80)
    
    tester = TestMoEImplementation()
    success = tester.run_all_tests()
    
    print("=" * 80)
    if success:
        print("✅ ALL TESTS PASSED - Implementation is production-ready!")
        print("🚀 Ready for Generation 3 (Scale) optimizations")
        return 0
    else:
        print("❌ TESTS FAILED - Issues need to be resolved")
        return 1


if __name__ == "__main__":
    exit(main())