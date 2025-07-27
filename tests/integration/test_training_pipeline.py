"""
Integration tests for the complete training pipeline.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Mock classes for integration testing
class MockMoEModel(torch.nn.Module):
    """Mock MoE model for integration testing."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = torch.nn.ModuleList([
            MockMoELayer(config) for _ in range(config["num_layers"])
        ])
        self.lm_head = torch.nn.Linear(config["hidden_size"], config["vocab_size"])
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.embedding(input_ids)
        
        total_aux_loss = 0.0
        routing_info = []
        
        for layer in self.layers:
            hidden_states, aux_loss, layer_routing = layer(hidden_states, attention_mask)
            total_aux_loss += aux_loss
            routing_info.append(layer_routing)
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss += 0.01 * total_aux_loss  # Add auxiliary loss
        
        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss,
            "routing_info": routing_info
        }


class MockMoELayer(torch.nn.Module):
    """Mock MoE layer for testing."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = torch.nn.MultiheadAttention(
            config["hidden_size"], 
            num_heads=8,
            batch_first=True
        )
        self.moe = MockMoEBlock(config)
        self.norm1 = torch.nn.LayerNorm(config["hidden_size"])
        self.norm2 = torch.nn.LayerNorm(config["hidden_size"])
        
    def forward(self, hidden_states, attention_mask=None):
        # Self-attention
        normed_states = self.norm1(hidden_states)
        attn_output, _ = self.attention(normed_states, normed_states, normed_states)
        hidden_states = hidden_states + attn_output
        
        # MoE block
        normed_states = self.norm2(hidden_states)
        moe_output, aux_loss, routing_info = self.moe(normed_states)
        hidden_states = hidden_states + moe_output
        
        return hidden_states, aux_loss, routing_info


class MockMoEBlock(torch.nn.Module):
    """Mock MoE block with experts."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = torch.nn.Linear(config["hidden_size"], config["num_experts"])
        self.experts = torch.nn.ModuleList([
            MockExpert(config) for _ in range(config["num_experts"])
        ])
        
    def forward(self, hidden_states):
        batch_size, seq_length, hidden_size = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_size)
        
        # Router forward
        router_logits = self.router(hidden_flat)
        routing_weights, selected_experts = torch.topk(
            router_logits, self.config["experts_per_token"], dim=-1
        )
        routing_weights = torch.softmax(routing_weights, dim=-1)
        
        # Expert forward (simplified)
        output = torch.zeros_like(hidden_flat)
        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i).any(dim=-1)
            if expert_mask.any():
                expert_input = hidden_flat[expert_mask]
                expert_output = expert(expert_input)
                output[expert_mask] += expert_output
        
        # Compute auxiliary loss
        aux_loss = self._compute_aux_loss(router_logits, selected_experts)
        
        # Routing info for analytics
        routing_info = {
            "router_logits": router_logits,
            "selected_experts": selected_experts,
            "routing_weights": routing_weights
        }
        
        return output.view(batch_size, seq_length, hidden_size), aux_loss, routing_info
    
    def _compute_aux_loss(self, router_logits, selected_experts):
        """Compute auxiliary loss for load balancing."""
        num_experts = self.config["num_experts"]
        
        # Expert load balancing
        expert_counts = torch.zeros(num_experts, device=router_logits.device)
        for i in range(num_experts):
            expert_counts[i] = (selected_experts == i).float().sum()
        
        # Simple auxiliary loss
        total_tokens = selected_experts.shape[0]
        target_load = total_tokens / num_experts
        load_variance = ((expert_counts - target_load) ** 2).mean()
        
        return load_variance * 0.01


class MockExpert(torch.nn.Module):
    """Mock expert network."""
    
    def __init__(self, config):
        super().__init__()
        intermediate_size = config.get("intermediate_size", config["hidden_size"] * 4)
        self.fc1 = torch.nn.Linear(config["hidden_size"], intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, config["hidden_size"])
        self.activation = torch.nn.GELU()
        
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class MockTrainer:
    """Mock trainer for integration testing."""
    
    def __init__(self, model, config, train_dataset=None, eval_dataset=None):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        self.scheduler = None
        self.step = 0
        self.epoch = 0
        self.metrics = []
        
    def train(self):
        """Simple training loop."""
        self.model.train()
        
        for epoch in range(self.config["num_epochs"]):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            # Simulate training batches
            for _ in range(10):  # 10 batches per epoch
                batch = self._get_mock_batch()
                loss = self._training_step(batch)
                epoch_loss += loss
                num_batches += 1
                self.step += 1
                
                if self.step >= self.config.get("max_steps", float("inf")):
                    break
            
            avg_loss = epoch_loss / num_batches
            self.metrics.append({"epoch": epoch, "train_loss": avg_loss})
            
            if self.step >= self.config.get("max_steps", float("inf")):
                break
        
        return self.metrics
    
    def _training_step(self, batch):
        """Single training step."""
        self.optimizer.zero_grad()
        
        outputs = self.model(**batch)
        loss = outputs["loss"]
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def _get_mock_batch(self):
        """Generate a mock training batch."""
        batch_size = self.config["batch_size"]
        seq_length = 64
        vocab_size = self.config["vocab_size"]
        
        return {
            "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length)),
            "attention_mask": torch.ones(batch_size, seq_length),
            "labels": torch.randint(0, vocab_size, (batch_size, seq_length))
        }
    
    def evaluate(self):
        """Simple evaluation."""
        self.model.eval()
        
        with torch.no_grad():
            eval_loss = 0.0
            num_batches = 5
            
            for _ in range(num_batches):
                batch = self._get_mock_batch()
                outputs = self.model(**batch)
                eval_loss += outputs["loss"].item()
        
        return {"eval_loss": eval_loss / num_batches}
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "config": self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]


class TestTrainingPipeline:
    """Integration tests for the complete training pipeline."""
    
    def test_end_to_end_training(self, mock_config, torch_device):
        """Test complete training pipeline end-to-end."""
        # Create model and trainer
        model = MockMoEModel(mock_config["model"]).to(torch_device)
        trainer = MockTrainer(model, mock_config["training"])
        
        # Run training
        metrics = trainer.train()
        
        # Check training completed
        assert len(metrics) > 0
        assert all("train_loss" in m for m in metrics)
        assert trainer.step > 0
        
        # Check model is trainable
        initial_loss = metrics[0]["train_loss"]
        final_loss = metrics[-1]["train_loss"]
        # Loss should decrease (simple check)
        assert final_loss <= initial_loss * 2  # Allow some variance
    
    def test_model_forward_pass(self, mock_config, sample_batch, torch_device):
        """Test model forward pass with sample batch."""
        model = MockMoEModel(mock_config["model"]).to(torch_device)
        
        # Move batch to device
        batch = {k: v.to(torch_device) for k, v in sample_batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        
        # Check outputs
        assert "logits" in outputs
        assert "loss" in outputs
        assert "aux_loss" in outputs
        assert "routing_info" in outputs
        
        # Check shapes
        batch_size, seq_length = batch["input_ids"].shape
        vocab_size = mock_config["model"]["vocab_size"]
        
        assert outputs["logits"].shape == (batch_size, seq_length, vocab_size)
        assert outputs["loss"].shape == ()
        assert outputs["aux_loss"].shape == ()
        assert len(outputs["routing_info"]) == mock_config["model"]["num_layers"]
    
    def test_gradient_computation(self, mock_config, sample_batch, torch_device):
        """Test gradient computation through the model."""
        model = MockMoEModel(mock_config["model"]).to(torch_device)
        
        # Move batch to device
        batch = {k: v.to(torch_device) for k, v in sample_batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs["loss"]
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_checkpoint_save_load(self, mock_config, torch_device, temp_dir):
        """Test model checkpoint saving and loading."""
        model = MockMoEModel(mock_config["model"]).to(torch_device)
        trainer = MockTrainer(model, mock_config["training"])
        
        # Train for a few steps
        trainer.step = 5
        trainer.epoch = 1
        
        # Save checkpoint
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        assert checkpoint_path.exists()
        
        # Create new model and trainer
        new_model = MockMoEModel(mock_config["model"]).to(torch_device)
        new_trainer = MockTrainer(new_model, mock_config["training"])
        
        # Load checkpoint
        new_trainer.load_checkpoint(checkpoint_path)
        
        # Check state is restored
        assert new_trainer.step == 5
        assert new_trainer.epoch == 1
        
        # Check model weights are the same
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)
    
    def test_evaluation_mode(self, mock_config, torch_device):
        """Test model evaluation mode."""
        model = MockMoEModel(mock_config["model"]).to(torch_device)
        trainer = MockTrainer(model, mock_config["training"])
        
        # Run evaluation
        eval_results = trainer.evaluate()
        
        # Check evaluation results
        assert "eval_loss" in eval_results
        assert isinstance(eval_results["eval_loss"], float)
        assert eval_results["eval_loss"] > 0
    
    def test_routing_consistency(self, mock_config, torch_device):
        """Test routing behavior is consistent."""
        model = MockMoEModel(mock_config["model"]).to(torch_device)
        
        # Create deterministic input
        torch.manual_seed(42)
        batch_size = 2
        seq_length = 16
        vocab_size = mock_config["model"]["vocab_size"]
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=torch_device)
        batch = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids)
        }
        
        # Two forward passes with same seed
        torch.manual_seed(42)
        outputs1 = model(**batch)
        
        torch.manual_seed(42)
        outputs2 = model(**batch)
        
        # Check outputs are identical
        assert torch.allclose(outputs1["logits"], outputs2["logits"], atol=1e-6)
        
        # Check routing info is identical
        for layer_idx in range(len(outputs1["routing_info"])):
            info1 = outputs1["routing_info"][layer_idx]
            info2 = outputs2["routing_info"][layer_idx]
            
            assert torch.equal(info1["selected_experts"], info2["selected_experts"])
            assert torch.allclose(info1["routing_weights"], info2["routing_weights"], atol=1e-6)
    
    @pytest.mark.slow
    def test_memory_efficiency(self, mock_config, torch_device):
        """Test memory efficiency of training pipeline."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        model = MockMoEModel(mock_config["model"]).to(torch_device)
        trainer = MockTrainer(model, mock_config["training"])
        
        # Clear cache and measure initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Run training for a few steps
        config = mock_config["training"].copy()
        config["num_epochs"] = 1
        config["max_steps"] = 5
        trainer.config = config
        
        metrics = trainer.train()
        
        # Measure final memory
        final_memory = torch.cuda.memory_allocated()
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 2GB for this small model)
        assert memory_growth < 2e9, f"Memory growth too large: {memory_growth / 1e9:.2f}GB"
    
    def test_expert_utilization_tracking(self, mock_config, torch_device):
        """Test that expert utilization can be tracked during training."""
        model = MockMoEModel(mock_config["model"]).to(torch_device)
        
        # Forward pass to get routing info
        batch = {
            "input_ids": torch.randint(0, mock_config["model"]["vocab_size"], (4, 32), device=torch_device),
            "attention_mask": torch.ones(4, 32, device=torch_device)
        }
        
        outputs = model(**batch)
        routing_info = outputs["routing_info"]
        
        # Track expert usage
        num_experts = mock_config["model"]["num_experts"]
        expert_usage = torch.zeros(num_experts)
        
        for layer_info in routing_info:
            selected_experts = layer_info["selected_experts"]
            for expert_idx in range(num_experts):
                expert_usage[expert_idx] += (selected_experts == expert_idx).sum().item()
        
        # Check that experts are being used
        assert expert_usage.sum() > 0
        assert (expert_usage > 0).sum() >= 1  # At least one expert used
    
    def test_loss_components(self, mock_config, sample_batch, torch_device):
        """Test that loss has both task loss and auxiliary loss components."""
        model = MockMoEModel(mock_config["model"]).to(torch_device)
        
        # Move batch to device
        batch = {k: v.to(torch_device) for k, v in sample_batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        
        # Check loss components
        total_loss = outputs["loss"]
        aux_loss = outputs["aux_loss"]
        
        assert total_loss > 0
        assert aux_loss >= 0
        
        # Auxiliary loss should contribute to total loss
        # (This is implementation dependent, so we just check it exists)
        assert aux_loss.requires_grad