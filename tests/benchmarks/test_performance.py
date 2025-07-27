"""
Performance benchmarks and regression tests.
"""

import pytest
import torch
import time
import psutil
import os
from typing import Dict, Any, List
from unittest.mock import Mock
import numpy as np


class BenchmarkResults:
    """Container for benchmark results."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.timings: Dict[str, List[float]] = {}
        self.memory_usage: Dict[str, float] = {}
        self.throughput: Dict[str, float] = {}
    
    def add_timing(self, name: str, time_seconds: float):
        """Add timing measurement."""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(time_seconds)
    
    def add_memory(self, name: str, memory_bytes: float):
        """Add memory measurement."""
        self.memory_usage[name] = memory_bytes
    
    def add_throughput(self, name: str, items_per_second: float):
        """Add throughput measurement."""
        self.throughput[name] = items_per_second
    
    def get_avg_timing(self, name: str) -> float:
        """Get average timing for a measurement."""
        return np.mean(self.timings[name]) if name in self.timings else 0.0
    
    def get_std_timing(self, name: str) -> float:
        """Get standard deviation for timing."""
        return np.std(self.timings[name]) if name in self.timings else 0.0


class MockBenchmarkModel(torch.nn.Module):
    """Mock model for benchmarking."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config["hidden_size"]
        self.num_experts = config["num_experts"]
        self.experts_per_token = config["experts_per_token"]
        
        # Model components
        self.embedding = torch.nn.Embedding(config["vocab_size"], self.hidden_size)
        self.router = torch.nn.Linear(self.hidden_size, self.num_experts)
        self.experts = torch.nn.ModuleList([
            self._create_expert() for _ in range(self.num_experts)
        ])
        self.lm_head = torch.nn.Linear(self.hidden_size, config["vocab_size"])
        
    def _create_expert(self):
        """Create a single expert network."""
        intermediate_size = self.config.get("intermediate_size", self.hidden_size * 4)
        return torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, intermediate_size),
            torch.nn.GELU(),
            torch.nn.Linear(intermediate_size, self.hidden_size),
            torch.nn.Dropout(0.1)
        )
    
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        batch_size, seq_length = input_ids.shape
        
        # Embedding
        hidden_states = self.embedding(input_ids)
        
        # MoE layer
        hidden_flat = hidden_states.view(-1, self.hidden_size)
        
        # Routing
        router_logits = self.router(hidden_flat)
        routing_weights, selected_experts = torch.topk(
            router_logits, self.experts_per_token, dim=-1
        )
        routing_weights = torch.softmax(routing_weights, dim=-1)
        
        # Expert computation
        expert_outputs = torch.zeros_like(hidden_flat)
        for i, expert in enumerate(self.experts):
            expert_mask = (selected_experts == i).any(dim=-1)
            if expert_mask.any():
                expert_input = hidden_flat[expert_mask]
                expert_output = expert(expert_input)
                # Simple aggregation for benchmarking
                expert_outputs[expert_mask] += expert_output
        
        # Reshape and final projection
        hidden_states = expert_outputs.view(batch_size, seq_length, self.hidden_size)
        logits = self.lm_head(hidden_states)
        
        return {
            "logits": logits,
            "routing_weights": routing_weights,
            "selected_experts": selected_experts
        }


class PerformanceBenchmark:
    """Performance benchmark runner."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.results = BenchmarkResults()
    
    def benchmark_forward_pass(self, model, batch, num_iterations=100, warmup=10):
        """Benchmark model forward pass."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(**batch)
        
        # Synchronize if using GPU
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                _ = model(**batch)
                
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        # Store results
        for t in times:
            self.results.add_timing("forward_pass", t)
        
        # Calculate throughput
        batch_size = batch["input_ids"].shape[0]
        seq_length = batch["input_ids"].shape[1]
        tokens_per_batch = batch_size * seq_length
        avg_time = np.mean(times)
        throughput = tokens_per_batch / avg_time
        
        self.results.add_throughput("tokens_per_second", throughput)
        
        return times
    
    def benchmark_memory_usage(self, model, batch):
        """Benchmark memory usage."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Forward pass
            with torch.no_grad():
                _ = model(**batch)
            
            peak_memory = torch.cuda.max_memory_allocated()
            current_memory = torch.cuda.memory_allocated()
            
            self.results.add_memory("peak_memory_bytes", peak_memory)
            self.results.add_memory("current_memory_bytes", current_memory)
        else:
            # CPU memory tracking
            process = psutil.Process()
            memory_before = process.memory_info().rss
            
            with torch.no_grad():
                _ = model(**batch)
            
            memory_after = process.memory_info().rss
            memory_used = memory_after - memory_before
            
            self.results.add_memory("memory_used_bytes", memory_used)
    
    def benchmark_training_step(self, model, batch, optimizer, num_iterations=50):
        """Benchmark training step performance."""
        model.train()
        
        times = []
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(**batch)
            
            # Simple loss for benchmarking
            loss = outputs["logits"].mean()
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        for t in times:
            self.results.add_timing("training_step", t)
        
        return times
    
    def benchmark_expert_utilization(self, model, batch, num_iterations=100):
        """Benchmark expert utilization patterns."""
        model.eval()
        
        expert_counts = torch.zeros(model.num_experts)
        total_tokens = 0
        
        with torch.no_grad():
            for _ in range(num_iterations):
                outputs = model(**batch)
                selected_experts = outputs["selected_experts"]
                
                # Count expert usage
                batch_size, seq_length = batch["input_ids"].shape
                tokens_this_batch = batch_size * seq_length
                total_tokens += tokens_this_batch
                
                for expert_idx in range(model.num_experts):
                    expert_counts[expert_idx] += (selected_experts == expert_idx).sum().item()
        
        # Calculate utilization metrics
        expert_utilization = expert_counts / total_tokens
        utilization_variance = expert_utilization.var().item()
        utilization_max = expert_utilization.max().item()
        utilization_min = expert_utilization.min().item()
        
        self.results.metrics["expert_utilization_variance"] = utilization_variance
        self.results.metrics["expert_utilization_max"] = utilization_max
        self.results.metrics["expert_utilization_min"] = utilization_min
        self.results.metrics["expert_utilization_ratio"] = utilization_max / (utilization_min + 1e-8)
        
        return expert_utilization


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_forward_pass_latency(self, mock_config, torch_device):
        """Test forward pass latency benchmark."""
        model = MockBenchmarkModel(mock_config["model"]).to(torch_device)
        
        batch = {
            "input_ids": torch.randint(0, mock_config["model"]["vocab_size"], (4, 128), device=torch_device),
            "attention_mask": torch.ones(4, 128, device=torch_device)
        }
        
        benchmark = PerformanceBenchmark(torch_device)
        times = benchmark.benchmark_forward_pass(model, batch, num_iterations=50, warmup=5)
        
        # Check that timing was recorded
        assert len(times) == 50
        assert all(t > 0 for t in times)
        
        # Check results
        avg_time = benchmark.results.get_avg_timing("forward_pass")
        std_time = benchmark.results.get_std_timing("forward_pass")
        
        assert avg_time > 0
        assert std_time >= 0
        
        # Performance expectation (adjust based on hardware)
        # This is just a sanity check
        assert avg_time < 1.0  # Should be less than 1 second per forward pass
    
    def test_memory_usage_benchmark(self, mock_config, torch_device):
        """Test memory usage benchmark."""
        model = MockBenchmarkModel(mock_config["model"]).to(torch_device)
        
        batch = {
            "input_ids": torch.randint(0, mock_config["model"]["vocab_size"], (8, 256), device=torch_device),
            "attention_mask": torch.ones(8, 256, device=torch_device)
        }
        
        benchmark = PerformanceBenchmark(torch_device)
        benchmark.benchmark_memory_usage(model, batch)
        
        # Check that memory usage was recorded
        if torch_device.type == "cuda":
            assert "peak_memory_bytes" in benchmark.results.memory_usage
            assert "current_memory_bytes" in benchmark.results.memory_usage
            assert benchmark.results.memory_usage["peak_memory_bytes"] > 0
        else:
            assert "memory_used_bytes" in benchmark.results.memory_usage
    
    def test_training_step_performance(self, mock_config, torch_device):
        """Test training step performance."""
        model = MockBenchmarkModel(mock_config["model"]).to(torch_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        batch = {
            "input_ids": torch.randint(0, mock_config["model"]["vocab_size"], (4, 64), device=torch_device),
            "attention_mask": torch.ones(4, 64, device=torch_device)
        }
        
        benchmark = PerformanceBenchmark(torch_device)
        times = benchmark.benchmark_training_step(model, batch, optimizer, num_iterations=20)
        
        # Check timing results
        assert len(times) == 20
        assert all(t > 0 for t in times)
        
        avg_time = benchmark.results.get_avg_timing("training_step")
        assert avg_time > 0
        assert avg_time < 5.0  # Should be reasonable
    
    def test_expert_utilization_benchmark(self, mock_config, torch_device):
        """Test expert utilization benchmark."""
        model = MockBenchmarkModel(mock_config["model"]).to(torch_device)
        
        batch = {
            "input_ids": torch.randint(0, mock_config["model"]["vocab_size"], (8, 32), device=torch_device),
            "attention_mask": torch.ones(8, 32, device=torch_device)
        }
        
        benchmark = PerformanceBenchmark(torch_device)
        utilization = benchmark.benchmark_expert_utilization(model, batch, num_iterations=50)
        
        # Check utilization results
        assert len(utilization) == mock_config["model"]["num_experts"]
        assert torch.all(utilization >= 0)
        assert torch.all(utilization <= 1)
        
        # Check metrics
        metrics = benchmark.results.metrics
        assert "expert_utilization_variance" in metrics
        assert "expert_utilization_max" in metrics
        assert "expert_utilization_min" in metrics
        assert "expert_utilization_ratio" in metrics
        
        assert metrics["expert_utilization_variance"] >= 0
        assert metrics["expert_utilization_ratio"] >= 1.0
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16])
    def test_scaling_with_batch_size(self, mock_config, batch_size, torch_device):
        """Test performance scaling with batch size."""
        model = MockBenchmarkModel(mock_config["model"]).to(torch_device)
        
        batch = {
            "input_ids": torch.randint(0, mock_config["model"]["vocab_size"], (batch_size, 64), device=torch_device),
            "attention_mask": torch.ones(batch_size, 64, device=torch_device)
        }
        
        benchmark = PerformanceBenchmark(torch_device)
        times = benchmark.benchmark_forward_pass(model, batch, num_iterations=20, warmup=3)
        
        avg_time = np.mean(times)
        throughput = benchmark.results.throughput["tokens_per_second"]
        
        # Basic checks
        assert avg_time > 0
        assert throughput > 0
        
        # Larger batch sizes should generally have better throughput
        # (though this may not always hold due to memory constraints)
        expected_tokens = batch_size * 64
        assert throughput > expected_tokens / 10  # At least 10 seconds per batch
    
    @pytest.mark.parametrize("seq_length", [32, 64, 128, 256])
    def test_scaling_with_sequence_length(self, mock_config, seq_length, torch_device):
        """Test performance scaling with sequence length."""
        model = MockBenchmarkModel(mock_config["model"]).to(torch_device)
        
        batch = {
            "input_ids": torch.randint(0, mock_config["model"]["vocab_size"], (4, seq_length), device=torch_device),
            "attention_mask": torch.ones(4, seq_length, device=torch_device)
        }
        
        benchmark = PerformanceBenchmark(torch_device)
        times = benchmark.benchmark_forward_pass(model, batch, num_iterations=20, warmup=3)
        
        avg_time = np.mean(times)
        
        # Longer sequences should take more time (roughly linear)
        assert avg_time > 0
        
        # Time should increase with sequence length, but we'll just check it's reasonable
        assert avg_time < seq_length / 100  # Sanity check: less than 10ms per token
    
    def test_performance_regression(self, mock_config, torch_device, performance_baseline):
        """Test for performance regression against baseline."""
        model = MockBenchmarkModel(mock_config["model"]).to(torch_device)
        
        batch = {
            "input_ids": torch.randint(0, mock_config["model"]["vocab_size"], (8, 128), device=torch_device),
            "attention_mask": torch.ones(8, 128, device=torch_device)
        }
        
        benchmark = PerformanceBenchmark(torch_device)
        
        # Benchmark forward pass
        benchmark.benchmark_forward_pass(model, batch, num_iterations=30, warmup=5)
        
        # Benchmark memory
        benchmark.benchmark_memory_usage(model, batch)
        
        # Check against baseline (allow some variance)
        throughput = benchmark.results.throughput["tokens_per_second"]
        baseline_throughput = performance_baseline["inference_speed"]["throughput_tokens_per_second"]
        
        # Allow 20% performance degradation
        assert throughput >= baseline_throughput * 0.8, f"Throughput regression: {throughput} < {baseline_throughput * 0.8}"
        
        # Memory usage check (if on GPU)
        if torch_device.type == "cuda":
            memory_used = benchmark.results.memory_usage["peak_memory_bytes"]
            baseline_memory = performance_baseline["inference_speed"]["memory_usage_gb"] * 1e9
            
            # Allow 50% memory increase
            assert memory_used <= baseline_memory * 1.5, f"Memory regression: {memory_used / 1e9:.2f}GB > {baseline_memory * 1.5 / 1e9:.2f}GB"


@pytest.mark.slow
class TestStressTests:
    """Stress tests for edge cases and limits."""
    
    def test_large_batch_stress(self, mock_config, torch_device):
        """Test with large batch sizes."""
        if torch_device.type == "cpu":
            pytest.skip("Large batch stress test requires GPU")
        
        model = MockBenchmarkModel(mock_config["model"]).to(torch_device)
        
        # Try increasingly large batches until we hit memory limits
        batch_sizes = [32, 64, 128]
        
        for batch_size in batch_sizes:
            try:
                batch = {
                    "input_ids": torch.randint(0, mock_config["model"]["vocab_size"], (batch_size, 128), device=torch_device),
                    "attention_mask": torch.ones(batch_size, 128, device=torch_device)
                }
                
                # Single forward pass to test memory
                with torch.no_grad():
                    outputs = model(**batch)
                
                # If we get here, the batch size worked
                assert outputs["logits"].shape[0] == batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # This is expected for large batches
                    break
                else:
                    raise
    
    def test_long_sequence_stress(self, mock_config, torch_device):
        """Test with very long sequences."""
        model = MockBenchmarkModel(mock_config["model"]).to(torch_device)
        
        # Test increasingly long sequences
        seq_lengths = [512, 1024, 2048]
        
        for seq_length in seq_lengths:
            try:
                batch = {
                    "input_ids": torch.randint(0, mock_config["model"]["vocab_size"], (2, seq_length), device=torch_device),
                    "attention_mask": torch.ones(2, seq_length, device=torch_device)
                }
                
                with torch.no_grad():
                    outputs = model(**batch)
                
                assert outputs["logits"].shape[1] == seq_length
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
                else:
                    raise
    
    def test_many_experts_stress(self, mock_config, torch_device):
        """Test with many experts."""
        # Test with more experts
        stress_config = mock_config["model"].copy()
        stress_config["num_experts"] = 32
        stress_config["experts_per_token"] = 4
        
        model = MockBenchmarkModel(stress_config).to(torch_device)
        
        batch = {
            "input_ids": torch.randint(0, stress_config["vocab_size"], (4, 64), device=torch_device),
            "attention_mask": torch.ones(4, 64, device=torch_device)
        }
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(**batch)
        
        # Check routing to many experts
        selected_experts = outputs["selected_experts"]
        assert selected_experts.max() < stress_config["num_experts"]
        assert selected_experts.shape[-1] == stress_config["experts_per_token"]
    
    def test_numerical_stability(self, mock_config, torch_device):
        """Test numerical stability with extreme inputs."""
        model = MockBenchmarkModel(mock_config["model"]).to(torch_device)
        model.eval()
        
        # Test with extreme values
        test_cases = [
            torch.zeros(2, 32, dtype=torch.long, device=torch_device),  # All zeros
            torch.full((2, 32), mock_config["model"]["vocab_size"] - 1, dtype=torch.long, device=torch_device),  # All max values
            torch.randint(0, mock_config["model"]["vocab_size"], (2, 32), device=torch_device) * 0 + mock_config["model"]["vocab_size"] // 2  # All middle values
        ]
        
        for test_input in test_cases:
            batch = {
                "input_ids": test_input,
                "attention_mask": torch.ones_like(test_input)
            }
            
            with torch.no_grad():
                outputs = model(**batch)
            
            # Check for NaN or Inf values
            assert not torch.isnan(outputs["logits"]).any(), "NaN values in output"
            assert not torch.isinf(outputs["logits"]).any(), "Inf values in output"
            assert not torch.isnan(outputs["routing_weights"]).any(), "NaN values in routing weights"
            assert not torch.isinf(outputs["routing_weights"]).any(), "Inf values in routing weights"