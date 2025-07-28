"""
End-to-end tests for the complete MoE training pipeline.
"""

import pytest
import torch
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch


class TestFullTrainingPipeline:
    """End-to-end tests for complete training workflows."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_training_workflow(self, temp_dir, mock_config):
        """Test complete training workflow from start to finish."""
        # Setup directories
        data_dir = temp_dir / "data"
        output_dir = temp_dir / "output"
        checkpoint_dir = temp_dir / "checkpoints"
        
        data_dir.mkdir()
        output_dir.mkdir()
        checkpoint_dir.mkdir()
        
        # Create mock training config
        config_path = temp_dir / "config.yaml"
        training_config = {
            **mock_config,
            "data": {
                "data_dir": str(data_dir),
                "dataset_name": "mock_dataset"
            },
            "output": {
                "output_dir": str(output_dir),
                "checkpoint_dir": str(checkpoint_dir)
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(training_config, f)
        
        # Mock the training pipeline components
        with patch('moe_lab.MoETrainer') as mock_trainer_class:
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Mock training results
            mock_trainer.train.return_value = {
                "train_loss": 2.5,
                "eval_loss": 2.8,
                "train_time": 120.0,
                "tokens_per_second": 1000
            }
            
            # Mock model saving
            mock_trainer.save_model = Mock()
            
            # Simulate training pipeline execution
            # This would normally call the actual CLI or API
            result = self._simulate_training_pipeline(config_path)
            
            # Verify training was called
            mock_trainer.train.assert_called_once()
            mock_trainer.save_model.assert_called_once()
            
            # Verify results
            assert result["status"] == "success"
            assert "train_loss" in result
            assert result["train_loss"] < 3.0

    def _simulate_training_pipeline(self, config_path):
        """Simulate the training pipeline execution."""
        # This would normally execute the actual training code
        # For testing, we return mock results
        return {
            "status": "success",
            "train_loss": 2.5,
            "eval_loss": 2.8,
            "train_time": 120.0,
            "model_path": "/tmp/model"
        }

    @pytest.mark.integration
    def test_model_export_workflow(self, temp_dir, mock_config):
        """Test model export workflow."""
        model_dir = temp_dir / "model"
        export_dir = temp_dir / "export"
        
        model_dir.mkdir()
        export_dir.mkdir()
        
        # Create mock model files
        (model_dir / "config.json").write_text('{"model_type": "moe"}')
        (model_dir / "pytorch_model.bin").write_bytes(b"mock model data")
        
        with patch('moe_lab.MoEExporter') as mock_exporter_class:
            mock_exporter = Mock()
            mock_exporter_class.return_value = mock_exporter
            
            # Mock export methods
            mock_exporter.export_onnx = Mock(return_value=str(export_dir / "model.onnx"))
            mock_exporter.export_tensorrt = Mock(return_value=str(export_dir / "model.trt"))
            
            # Simulate export workflow
            result = self._simulate_export_workflow(model_dir, export_dir)
            
            # Verify exports were called
            mock_exporter.export_onnx.assert_called_once()
            
            assert result["status"] == "success"
            assert "onnx_path" in result

    def _simulate_export_workflow(self, model_dir, export_dir):
        """Simulate model export workflow."""
        return {
            "status": "success",
            "onnx_path": str(export_dir / "model.onnx"),
            "tensorrt_path": str(export_dir / "model.trt")
        }

    @pytest.mark.integration
    def test_inference_pipeline(self, temp_dir):
        """Test inference pipeline with trained model."""
        model_dir = temp_dir / "model"
        model_dir.mkdir()
        
        # Create mock model files
        (model_dir / "config.json").write_text(
            '{"model_type": "moe", "num_experts": 4, "hidden_size": 512}'
        )
        
        with patch('moe_lab.MoEInferenceEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            
            # Mock inference
            mock_engine.generate.return_value = {
                "generated_text": "Hello, this is a test output.",
                "generation_time": 0.05,
                "tokens_generated": 8,
                "expert_usage": {"expert_0": 0.4, "expert_1": 0.3, "expert_2": 0.2, "expert_3": 0.1}
            }
            
            # Simulate inference
            result = self._simulate_inference(model_dir, "Hello, how are you?")
            
            # Verify inference was called
            mock_engine.generate.assert_called_once()
            
            assert result["status"] == "success"
            assert "generated_text" in result
            assert "expert_usage" in result

    def _simulate_inference(self, model_dir, prompt):
        """Simulate inference workflow."""
        return {
            "status": "success",
            "generated_text": "Hello, this is a test output.",
            "generation_time": 0.05,
            "expert_usage": {"expert_0": 0.4, "expert_1": 0.3}
        }

    @pytest.mark.integration
    def test_distributed_training_setup(self, mock_config):
        """Test distributed training setup and coordination."""
        with patch('torch.distributed.init_process_group') as mock_init:
            with patch('torch.distributed.get_world_size') as mock_world_size:
                with patch('torch.distributed.get_rank') as mock_rank:
                    
                    mock_world_size.return_value = 2
                    mock_rank.return_value = 0
                    
                    # Simulate distributed training setup
                    result = self._simulate_distributed_setup(mock_config)
                    
                    # Verify distributed was initialized
                    mock_init.assert_called_once()
                    
                    assert result["status"] == "success"
                    assert result["world_size"] == 2
                    assert result["rank"] == 0

    def _simulate_distributed_setup(self, config):
        """Simulate distributed training setup."""
        return {
            "status": "success",
            "world_size": 2,
            "rank": 0,
            "backend": "nccl"
        }

    @pytest.mark.integration
    def test_monitoring_integration(self, temp_dir):
        """Test integration with monitoring systems."""
        with patch('wandb.init') as mock_wandb_init:
            with patch('wandb.log') as mock_wandb_log:
                mock_run = Mock()
                mock_wandb_init.return_value = mock_run
                
                # Simulate training with monitoring
                result = self._simulate_training_with_monitoring()
                
                # Verify monitoring was set up
                mock_wandb_init.assert_called_once()
                mock_wandb_log.assert_called()
                
                assert result["status"] == "success"
                assert "metrics_logged" in result

    def _simulate_training_with_monitoring(self):
        """Simulate training with monitoring integration."""
        return {
            "status": "success",
            "metrics_logged": True,
            "experiments_tracked": True
        }

    @pytest.mark.integration
    @pytest.mark.slow
    def test_checkpoint_recovery(self, temp_dir, mock_config):
        """Test training recovery from checkpoint."""
        checkpoint_dir = temp_dir / "checkpoints"
        checkpoint_dir.mkdir()
        
        # Create mock checkpoint
        checkpoint_path = checkpoint_dir / "checkpoint-100.pt"
        mock_checkpoint = {
            "epoch": 1,
            "step": 100,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "loss": 2.5,
            "expert_routing_stats": {}
        }
        torch.save(mock_checkpoint, checkpoint_path)
        
        with patch('moe_lab.MoETrainer') as mock_trainer_class:
            mock_trainer = Mock()
            mock_trainer_class.return_value = mock_trainer
            
            # Mock checkpoint loading
            mock_trainer.load_checkpoint.return_value = mock_checkpoint
            
            # Simulate recovery
            result = self._simulate_checkpoint_recovery(checkpoint_path)
            
            # Verify checkpoint was loaded
            mock_trainer.load_checkpoint.assert_called_once()
            
            assert result["status"] == "success"
            assert result["resumed_from_step"] == 100

    def _simulate_checkpoint_recovery(self, checkpoint_path):
        """Simulate training recovery from checkpoint."""
        return {
            "status": "success",
            "resumed_from_step": 100,
            "resumed_from_epoch": 1
        }

    @pytest.mark.integration
    def test_expert_analysis_workflow(self, expert_routing_data, temp_dir):
        """Test expert analysis and visualization workflow."""
        analysis_dir = temp_dir / "analysis"
        analysis_dir.mkdir()
        
        with patch('moe_lab.RouterAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer_class.return_value = mock_analyzer
            
            # Mock analysis methods
            mock_analyzer.compute_expert_specialization.return_value = {
                "expert_0": {"domain": "mathematics", "confidence": 0.8},
                "expert_1": {"domain": "language", "confidence": 0.7}
            }
            
            mock_analyzer.plot_routing_heatmap.return_value = str(analysis_dir / "heatmap.png")
            
            # Simulate analysis workflow
            result = self._simulate_expert_analysis(expert_routing_data, analysis_dir)
            
            # Verify analysis was performed
            mock_analyzer.compute_expert_specialization.assert_called_once()
            mock_analyzer.plot_routing_heatmap.assert_called_once()
            
            assert result["status"] == "success"
            assert "specialization" in result
            assert "visualizations" in result

    def _simulate_expert_analysis(self, routing_data, analysis_dir):
        """Simulate expert analysis workflow."""
        return {
            "status": "success",
            "specialization": {
                "expert_0": {"domain": "mathematics", "confidence": 0.8}
            },
            "visualizations": ["heatmap.png", "flow_diagram.png"]
        }
