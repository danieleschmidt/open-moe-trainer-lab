"""
Tests for distributed and parallel training functionality.
"""

import pytest
import torch
import torch.distributed as dist
from unittest.mock import Mock, patch, MagicMock
import os


class MockDistributedMoEModel(torch.nn.Module):
    """Mock distributed MoE model for testing."""
    
    def __init__(self, config, world_size=1, expert_parallel_size=1):
        super().__init__()
        self.config = config
        self.world_size = world_size
        self.expert_parallel_size = expert_parallel_size
        
        # Simplified model for testing
        self.embedding = torch.nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.expert_layers = torch.nn.ModuleList([
            MockDistributedMoELayer(config, expert_parallel_size) 
            for _ in range(config["num_layers"])
        ])
        self.lm_head = torch.nn.Linear(config["hidden_size"], config["vocab_size"])
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.embedding(input_ids)
        
        total_aux_loss = 0.0
        
        for layer in self.expert_layers:
            hidden_states, aux_loss = layer(hidden_states)
            total_aux_loss += aux_loss
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss += 0.01 * total_aux_loss
        
        return {
            "logits": logits,
            "loss": loss,
            "aux_loss": total_aux_loss
        }


class MockDistributedMoELayer(torch.nn.Module):
    """Mock distributed MoE layer."""
    
    def __init__(self, config, expert_parallel_size=1):
        super().__init__()
        self.config = config
        self.expert_parallel_size = expert_parallel_size
        
        # Router is replicated across all processes
        self.router = torch.nn.Linear(config["hidden_size"], config["num_experts"])
        
        # Experts are distributed across expert parallel group
        experts_per_rank = config["num_experts"] // expert_parallel_size
        self.local_experts = torch.nn.ModuleList([
            MockExpert(config) for _ in range(experts_per_rank)
        ])
        
        self.norm = torch.nn.LayerNorm(config["hidden_size"])
    
    def forward(self, hidden_states):
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Router forward
        router_logits = self.router(hidden_states.view(-1, hidden_size))
        routing_weights, selected_experts = torch.topk(
            router_logits, self.config["experts_per_token"], dim=-1
        )
        routing_weights = torch.softmax(routing_weights, dim=-1)
        
        # Simulate expert parallel communication
        expert_output = self._simulate_expert_parallel_forward(
            hidden_states.view(-1, hidden_size), selected_experts, routing_weights
        )
        
        # Compute auxiliary loss
        aux_loss = self._compute_aux_loss(router_logits, selected_experts)
        
        output = expert_output.view(batch_size, seq_length, hidden_size)
        return self.norm(output), aux_loss
    
    def _simulate_expert_parallel_forward(self, hidden_states, selected_experts, routing_weights):
        """Simulate expert parallel forward pass."""
        output = torch.zeros_like(hidden_states)
        
        # In real implementation, this would involve all-to-all communication
        # Here we simulate by using local experts only
        experts_per_rank = len(self.local_experts)
        
        for i, expert in enumerate(self.local_experts):
            # Check if this expert is selected
            expert_mask = (selected_experts == i).any(dim=-1)
            if expert_mask.any():
                expert_input = hidden_states[expert_mask]
                expert_out = expert(expert_input)
                # Weight by routing weights (simplified)
                weights = routing_weights[expert_mask]
                expert_weights = weights[selected_experts[expert_mask] == i].mean(dim=-1, keepdim=True)
                output[expert_mask] += expert_out * expert_weights
        
        return output
    
    def _compute_aux_loss(self, router_logits, selected_experts):
        """Compute auxiliary loss for load balancing."""
        num_experts = self.config["num_experts"]
        
        # Expert counts
        expert_counts = torch.zeros(num_experts, device=router_logits.device)
        for i in range(num_experts):
            expert_counts[i] = (selected_experts == i).float().sum()
        
        # Load balancing loss
        total_tokens = selected_experts.shape[0]
        target_load = total_tokens / num_experts
        load_variance = ((expert_counts - target_load) ** 2).mean()
        
        return load_variance * 0.01


class MockExpert(torch.nn.Module):
    """Mock expert for distributed testing."""
    
    def __init__(self, config):
        super().__init__()
        hidden_size = config["hidden_size"]
        intermediate_size = config.get("intermediate_size", hidden_size * 4)
        
        self.fc1 = torch.nn.Linear(hidden_size, intermediate_size)
        self.fc2 = torch.nn.Linear(intermediate_size, hidden_size)
        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(0.1)
    
    def forward(self, x):
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class MockDistributedTrainer:
    """Mock distributed trainer for testing."""
    
    def __init__(self, model, config, rank=0, world_size=1):
        self.model = model
        self.config = config
        self.rank = rank
        self.world_size = world_size
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
        
        # Mock distributed setup
        self.is_distributed = world_size > 1
        if self.is_distributed:
            # In real implementation, this would wrap model with DDP
            self.model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    def train_step(self, batch):
        """Single distributed training step."""
        self.optimizer.zero_grad()
        
        outputs = self.model(**batch)
        loss = outputs["loss"]
        
        loss.backward()
        
        # Gradient synchronization is handled by DDP in real implementation
        if self.is_distributed:
            # Simulate gradient synchronization
            self._sync_gradients()
        
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "aux_loss": outputs["aux_loss"].item()
        }
    
    def _sync_gradients(self):
        """Simulate gradient synchronization across ranks."""
        # In real implementation, this is handled automatically by DDP
        # Here we just ensure gradients exist
        for param in self.model.parameters():
            if param.grad is not None:
                # Simulate all-reduce operation
                param.grad = param.grad.clone()


@pytest.mark.distributed
class TestDistributedTraining:
    """Tests for distributed training functionality."""
    
    def test_model_parallel_initialization(self, mock_config):
        """Test model initialization with parallelism."""
        world_size = 4
        expert_parallel_size = 2
        
        model = MockDistributedMoEModel(
            mock_config["model"], 
            world_size=world_size,
            expert_parallel_size=expert_parallel_size
        )
        
        # Check expert distribution
        total_experts = mock_config["model"]["num_experts"]
        experts_per_rank = total_experts // expert_parallel_size
        
        for layer in model.expert_layers:
            assert len(layer.local_experts) == experts_per_rank
    
    def test_expert_parallel_forward(self, mock_config, torch_device):
        """Test expert parallel forward pass."""
        expert_parallel_size = 2
        model = MockDistributedMoELayer(mock_config["model"], expert_parallel_size).to(torch_device)
        
        batch_size = 4
        seq_length = 32
        hidden_size = mock_config["model"]["hidden_size"]
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size, device=torch_device)
        
        output, aux_loss = model(hidden_states)
        
        # Check output shape
        assert output.shape == hidden_states.shape
        assert aux_loss.shape == ()
        assert aux_loss >= 0
    
    def test_distributed_training_step(self, mock_config, torch_device):
        """Test distributed training step."""
        rank = 0
        world_size = 2
        
        model = MockDistributedMoEModel(mock_config["model"], world_size=world_size).to(torch_device)
        trainer = MockDistributedTrainer(model, mock_config["training"], rank=rank, world_size=world_size)
        
        # Create sample batch
        batch = {
            "input_ids": torch.randint(0, mock_config["model"]["vocab_size"], (4, 32), device=torch_device),
            "attention_mask": torch.ones(4, 32, device=torch_device),
            "labels": torch.randint(0, mock_config["model"]["vocab_size"], (4, 32), device=torch_device)
        }
        
        # Training step
        metrics = trainer.train_step(batch)
        
        # Check metrics
        assert "loss" in metrics
        assert "aux_loss" in metrics
        assert metrics["loss"] > 0
        assert metrics["aux_loss"] >= 0
    
    def test_gradient_synchronization(self, mock_config, torch_device):
        """Test gradient synchronization simulation."""
        world_size = 2
        model = MockDistributedMoEModel(mock_config["model"], world_size=world_size).to(torch_device)
        trainer = MockDistributedTrainer(model, mock_config["training"], rank=0, world_size=world_size)
        
        # Create batch and run forward/backward
        batch = {
            "input_ids": torch.randint(0, mock_config["model"]["vocab_size"], (2, 16), device=torch_device),
            "attention_mask": torch.ones(2, 16, device=torch_device),
            "labels": torch.randint(0, mock_config["model"]["vocab_size"], (2, 16), device=torch_device)
        }
        
        trainer.optimizer.zero_grad()
        outputs = trainer.model(**batch)
        outputs["loss"].backward()
        
        # Check gradients exist before sync
        grad_norms_before = []
        for param in trainer.model.parameters():
            if param.grad is not None:
                grad_norms_before.append(param.grad.norm().item())
        
        # Simulate gradient sync
        trainer._sync_gradients()
        
        # Check gradients still exist after sync
        grad_norms_after = []
        for param in trainer.model.parameters():
            if param.grad is not None:
                grad_norms_after.append(param.grad.norm().item())
        
        assert len(grad_norms_before) == len(grad_norms_after)
        assert all(gn > 0 for gn in grad_norms_after)
    
    @pytest.mark.parametrize("expert_parallel_size", [1, 2, 4])
    def test_expert_parallelism_scaling(self, mock_config, expert_parallel_size, torch_device):
        """Test expert parallelism with different group sizes."""
        # Adjust number of experts to be divisible by parallel size
        total_experts = 8
        mock_config["model"]["num_experts"] = total_experts
        
        if total_experts % expert_parallel_size != 0:
            pytest.skip(f"Total experts {total_experts} not divisible by {expert_parallel_size}")
        
        model = MockDistributedMoELayer(mock_config["model"], expert_parallel_size).to(torch_device)
        
        expected_local_experts = total_experts // expert_parallel_size
        assert len(model.local_experts) == expected_local_experts
        
        # Test forward pass
        hidden_states = torch.randn(2, 16, mock_config["model"]["hidden_size"], device=torch_device)
        output, aux_loss = model(hidden_states)
        
        assert output.shape == hidden_states.shape
        assert aux_loss >= 0
    
    def test_communication_patterns(self, mock_config, torch_device):
        """Test communication patterns in expert parallelism."""
        expert_parallel_size = 2
        layer = MockDistributedMoELayer(mock_config["model"], expert_parallel_size).to(torch_device)
        
        batch_size = 4
        seq_length = 16
        hidden_size = mock_config["model"]["hidden_size"]
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size, device=torch_device)
        
        # In real implementation, we would track all-to-all communications
        # Here we just ensure the forward pass completes
        output, aux_loss = layer(hidden_states)
        
        # Check that output is reasonable
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert aux_loss >= 0
    
    def test_load_balancing_across_ranks(self, mock_config, torch_device):
        """Test load balancing with expert parallelism."""
        expert_parallel_size = 2
        layer = MockDistributedMoELayer(mock_config["model"], expert_parallel_size).to(torch_device)
        
        # Run multiple forward passes and check load distribution
        total_expert_usage = torch.zeros(mock_config["model"]["num_experts"])
        
        for _ in range(10):  # Multiple batches
            hidden_states = torch.randn(4, 16, mock_config["model"]["hidden_size"], device=torch_device)
            
            # Get routing decisions
            router_logits = layer.router(hidden_states.view(-1, mock_config["model"]["hidden_size"]))
            _, selected_experts = torch.topk(router_logits, mock_config["model"]["experts_per_token"], dim=-1)
            
            # Count expert usage
            for expert_idx in range(mock_config["model"]["num_experts"]):
                total_expert_usage[expert_idx] += (selected_experts == expert_idx).sum().item()
        
        # Check that load is somewhat balanced
        mean_usage = total_expert_usage.mean()
        max_usage = total_expert_usage.max()
        min_usage = total_expert_usage.min()
        
        # Allow for some imbalance but not extreme
        assert max_usage / (min_usage + 1e-8) < 10  # No expert should be 10x more used than another


@pytest.mark.slow
@pytest.mark.distributed
class TestDistributedPerformance:
    """Performance tests for distributed training."""
    
    def test_memory_usage_scaling(self, mock_config, torch_device):
        """Test memory usage scales appropriately with expert parallelism."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        # Test different expert parallel sizes
        expert_parallel_sizes = [1, 2]
        memory_usage = {}
        
        for ep_size in expert_parallel_sizes:
            if mock_config["model"]["num_experts"] % ep_size != 0:
                continue
            
            torch.cuda.empty_cache()
            
            model = MockDistributedMoEModel(
                mock_config["model"], 
                expert_parallel_size=ep_size
            ).to(torch_device)
            
            # Measure memory
            memory_usage[ep_size] = torch.cuda.memory_allocated()
            
            # Clean up
            del model
            torch.cuda.empty_cache()
        
        # Memory usage should decrease with more parallelism (fewer local experts)
        if len(memory_usage) > 1:
            sizes = sorted(memory_usage.keys())
            for i in range(1, len(sizes)):
                # Allow some variance, but generally should use less memory
                assert memory_usage[sizes[i]] <= memory_usage[sizes[i-1]] * 1.1
    
    def test_compute_efficiency(self, mock_config, torch_device):
        """Test compute efficiency with expert parallelism."""
        expert_parallel_sizes = [1, 2]
        times = {}
        
        for ep_size in expert_parallel_sizes:
            if mock_config["model"]["num_experts"] % ep_size != 0:
                continue
            
            model = MockDistributedMoELayer(mock_config["model"], ep_size).to(torch_device)
            hidden_states = torch.randn(8, 64, mock_config["model"]["hidden_size"], device=torch_device)
            
            # Warmup
            for _ in range(5):
                _ = model(hidden_states)
            
            # Time forward passes
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            import time
            start_time = time.time()
            
            for _ in range(10):
                _ = model(hidden_states)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            times[ep_size] = (end_time - start_time) / 10  # Average time per forward pass
        
        # Just check that timing completes without errors
        assert all(t > 0 for t in times.values())


class TestExpertParallelCommunication:
    """Tests for expert parallel communication patterns."""
    
    def test_all_to_all_simulation(self, mock_config, torch_device):
        """Test all-to-all communication simulation."""
        # This would test the actual all-to-all communication in a real implementation
        # Here we test the mock version
        
        layer = MockDistributedMoELayer(mock_config["model"], expert_parallel_size=2).to(torch_device)
        hidden_states = torch.randn(4, 16, mock_config["model"]["hidden_size"], device=torch_device)
        
        # Test that the communication simulation works
        flat_states = hidden_states.view(-1, mock_config["model"]["hidden_size"])
        router_logits = layer.router(flat_states)
        _, selected_experts = torch.topk(router_logits, mock_config["model"]["experts_per_token"], dim=-1)
        routing_weights = torch.softmax(torch.topk(router_logits, mock_config["model"]["experts_per_token"], dim=-1)[0], dim=-1)
        
        output = layer._simulate_expert_parallel_forward(flat_states, selected_experts, routing_weights)
        
        assert output.shape == flat_states.shape
        assert not torch.isnan(output).any()
    
    def test_routing_consistency_across_ranks(self, mock_config, torch_device):
        """Test that routing is consistent across different ranks."""
        # In a real distributed setup, routing decisions should be consistent
        # Here we test that the same input produces the same routing
        
        layer1 = MockDistributedMoELayer(mock_config["model"], expert_parallel_size=1).to(torch_device)
        layer2 = MockDistributedMoELayer(mock_config["model"], expert_parallel_size=2).to(torch_device)
        
        # Copy router weights to ensure same routing decisions
        layer2.router.load_state_dict(layer1.router.state_dict())
        
        hidden_states = torch.randn(2, 8, mock_config["model"]["hidden_size"], device=torch_device)
        
        # Get routing decisions from both
        with torch.no_grad():
            flat_states = hidden_states.view(-1, mock_config["model"]["hidden_size"])
            
            routing1 = layer1.router(flat_states)
            routing2 = layer2.router(flat_states)
            
            # Routing logits should be identical
            assert torch.allclose(routing1, routing2, atol=1e-6)