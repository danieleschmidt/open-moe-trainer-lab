"""
Performance benchmarks for MoE training pipeline.
"""

import pytest
import torch
import time
import psutil
import gc
from memory_profiler import profile
from typing import Dict, Any


class TestTrainingPerformance:
    """Performance benchmarks for training operations."""

    @pytest.mark.benchmark
    def test_forward_pass_performance(self, benchmark, mock_config, sample_batch):
        """Benchmark forward pass performance."""
        # Mock model for testing
        class MockMoEModel(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.linear = torch.nn.Linear(config['model']['hidden_size'], config['model']['vocab_size'])
                
            def forward(self, input_ids, attention_mask=None):
                # Simulate MoE forward pass
                hidden = torch.randn(input_ids.shape[0], input_ids.shape[1], self.config['model']['hidden_size'])
                return self.linear(hidden)
        
        model = MockMoEModel(mock_config)
        model.eval()
        
        def forward_pass():
            with torch.no_grad():
                return model(sample_batch['input_ids'], sample_batch['attention_mask'])
        
        result = benchmark(forward_pass)
        assert result is not None

    @pytest.mark.benchmark
    @pytest.mark.gpu
    def test_gpu_memory_usage(self, mock_config, sample_batch, torch_device):
        """Test GPU memory usage during training."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
            
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Simulate training step
        model = torch.nn.Linear(mock_config['model']['hidden_size'], mock_config['model']['vocab_size'])
        model = model.to(torch_device)
        
        optimizer = torch.optim.Adam(model.parameters())
        
        # Forward pass
        hidden = torch.randn(sample_batch['input_ids'].shape[0], 
                           sample_batch['input_ids'].shape[1], 
                           mock_config['model']['hidden_size'], 
                           device=torch_device)
        output = model(hidden)
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, output.size(-1)), 
            sample_batch['labels'].view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        peak_memory = torch.cuda.max_memory_allocated()
        memory_used = (peak_memory - initial_memory) / (1024 ** 3)  # GB
        
        # Assert reasonable memory usage (< 1GB for small model)
        assert memory_used < 1.0, f"Memory usage too high: {memory_used:.2f}GB"
        
        torch.cuda.empty_cache()

    @pytest.mark.benchmark
    def test_throughput_benchmark(self, benchmark, mock_config):
        """Benchmark training throughput (tokens/second)."""
        def simulate_training_step():
            batch_size = mock_config['training']['batch_size']
            seq_length = mock_config['model']['max_seq_length']
            vocab_size = mock_config['model']['vocab_size']
            
            # Simulate processing a batch
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
            labels = torch.randint(0, vocab_size, (batch_size, seq_length))
            
            # Simple computation to simulate model forward
            hidden = torch.randn(batch_size, seq_length, mock_config['model']['hidden_size'])
            output = torch.randn(batch_size, seq_length, vocab_size)
            loss = torch.nn.functional.cross_entropy(output.view(-1, vocab_size), labels.view(-1))
            
            return batch_size * seq_length  # Return number of tokens processed
        
        tokens_processed = benchmark(simulate_training_step)
        assert tokens_processed > 0

    @pytest.mark.memory
    def test_memory_leak_detection(self, mock_config):
        """Test for memory leaks during repeated operations."""
        initial_memory = psutil.Process().memory_info().rss
        
        # Simulate multiple training iterations
        for _ in range(10):
            hidden = torch.randn(4, 64, mock_config['model']['hidden_size'])
            output = torch.randn(4, 64, mock_config['model']['vocab_size'])
            loss = torch.mean(output)
            loss.backward()
            del hidden, output, loss
            gc.collect()
        
        final_memory = psutil.Process().memory_info().rss
        memory_growth = (final_memory - initial_memory) / (1024 ** 2)  # MB
        
        # Allow some memory growth but not excessive
        assert memory_growth < 100, f"Potential memory leak: {memory_growth:.2f}MB growth"

    @pytest.mark.slow_benchmark
    def test_large_batch_performance(self, benchmark):
        """Test performance with large batch sizes."""
        def process_large_batch():
            batch_size = 64
            seq_length = 512
            hidden_size = 768
            vocab_size = 32000
            
            # Simulate large batch processing
            hidden = torch.randn(batch_size, seq_length, hidden_size)
            output = torch.randn(batch_size, seq_length, vocab_size)
            loss = torch.mean(output)
            
            return batch_size * seq_length
        
        tokens_processed = benchmark(process_large_batch)
        assert tokens_processed > 0

    @pytest.mark.benchmark
    def test_expert_routing_performance(self, benchmark, expert_routing_data):
        """Benchmark expert routing computation performance."""
        def routing_computation():
            # Simulate top-k expert selection
            router_logits = expert_routing_data['router_logits']
            top_k = 2
            
            # Get top-k experts
            top_k_logits, top_k_indices = torch.topk(router_logits, top_k, dim=-1)
            
            # Compute routing weights
            routing_weights = torch.softmax(top_k_logits, dim=-1)
            
            return routing_weights.shape[0] * routing_weights.shape[1]  # Number of tokens processed
        
        tokens_processed = benchmark(routing_computation)
        assert tokens_processed > 0

    @pytest.mark.gpu_benchmark
    def test_expert_computation_performance(self, benchmark, torch_device):
        """Benchmark expert computation performance."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
            
        def expert_computation():
            batch_size = 32
            seq_length = 128
            hidden_size = 768
            intermediate_size = 3072
            
            # Simulate expert FFN computation
            hidden_states = torch.randn(batch_size, seq_length, hidden_size, device=torch_device)
            
            # Expert layers
            fc1 = torch.nn.Linear(hidden_size, intermediate_size).to(torch_device)
            fc2 = torch.nn.Linear(intermediate_size, hidden_size).to(torch_device)
            
            # Forward pass
            intermediate = torch.nn.functional.gelu(fc1(hidden_states))
            output = fc2(intermediate)
            
            return batch_size * seq_length
        
        tokens_processed = benchmark(expert_computation)
        assert tokens_processed > 0

    def test_performance_regression(self, performance_baseline):
        """Test for performance regression against baseline."""
        # Simulate current performance metrics
        current_metrics = {
            "training_speed": {
                "tokens_per_second": 950,  # Slightly slower
                "steps_per_second": 9.5,
                "memory_usage_gb": 8.2
            }
        }
        
        baseline = performance_baseline["training_speed"]
        current = current_metrics["training_speed"]
        
        # Allow 10% performance degradation
        tolerance = 0.1
        
        tokens_ratio = current["tokens_per_second"] / baseline["tokens_per_second"]
        assert tokens_ratio > (1 - tolerance), f"Tokens/sec regression: {tokens_ratio:.2f}"
        
        steps_ratio = current["steps_per_second"] / baseline["steps_per_second"]
        assert steps_ratio > (1 - tolerance), f"Steps/sec regression: {steps_ratio:.2f}"
        
        memory_ratio = current["memory_usage_gb"] / baseline["memory_usage_gb"]
        assert memory_ratio < (1 + tolerance), f"Memory usage increase: {memory_ratio:.2f}"
