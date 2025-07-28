"""
Test utilities and helper functions for Open MoE Trainer Lab.

This module provides common utilities used across different test modules
to reduce code duplication and ensure consistent testing patterns.
"""

import os
import tempfile
import shutil
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from contextlib import contextmanager
import time
import psutil
import torch
import numpy as np
from unittest.mock import Mock, MagicMock


class TestTimer:
    """Context manager for timing test execution."""
    
    def __init__(self, name: str = "test"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time
    
    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed:.4f}s"


class MemoryTracker:
    """Track memory usage during tests."""
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None
        self.final_memory = None
    
    def __enter__(self):
        self.initial_memory = self._get_memory()
        self.peak_memory = self.initial_memory
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self._get_memory()
    
    def _get_memory(self) -> Dict[str, float]:
        """Get current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        memory = {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
        }
        
        # Add GPU memory if available
        if torch.cuda.is_available():
            memory['gpu_allocated'] = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            memory['gpu_reserved'] = torch.cuda.memory_reserved() / 1024 / 1024  # MB
        
        return memory
    
    def update_peak(self):
        """Update peak memory usage."""
        current = self._get_memory()
        if current['rss'] > self.peak_memory['rss']:
            self.peak_memory = current
    
    @property
    def memory_delta(self) -> Dict[str, float]:
        """Get memory usage delta."""
        if self.initial_memory is None or self.final_memory is None:
            return {}
        
        delta = {}
        for key in self.initial_memory:
            delta[f'{key}_delta'] = self.final_memory[key] - self.initial_memory[key]
        
        return delta


def create_mock_model_config(
    hidden_size: int = 512,
    num_experts: int = 8,
    experts_per_token: int = 2,
    num_layers: int = 12,
    vocab_size: int = 50000,
    max_seq_length: int = 2048
) -> Dict[str, Any]:
    """Create a mock model configuration for testing."""
    return {
        "hidden_size": hidden_size,
        "num_experts": num_experts,
        "experts_per_token": experts_per_token,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "max_seq_length": max_seq_length,
        "intermediate_size": hidden_size * 4,
        "num_attention_heads": hidden_size // 64,
        "dropout_prob": 0.1,
        "attention_dropout_prob": 0.1,
        "layer_norm_eps": 1e-12,
        "initializer_range": 0.02,
        "use_cache": True,
        "router_type": "top_k",
        "router_bias": False,
        "router_jitter_noise": 0.01,
        "load_balancing_loss_coef": 0.01,
        "router_z_loss_coef": 0.001,
    }


def create_mock_training_config(
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    warmup_steps: int = 1000,
    max_steps: Optional[int] = None,
    eval_steps: int = 500,
    save_steps: int = 1000,
    logging_steps: int = 100
) -> Dict[str, Any]:
    """Create a mock training configuration for testing."""
    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "warmup_steps": warmup_steps,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "max_steps": max_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "fp16": False,
        "gradient_accumulation_steps": 1,
        "dataloader_num_workers": 0,  # Single-threaded for tests
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 3,
        "seed": 42,
    }


def create_sample_batch(
    batch_size: int = 4,
    seq_length: int = 64,
    vocab_size: int = 50000,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """Create a sample batch for testing."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length), device=device),
        "attention_mask": torch.ones(batch_size, seq_length, device=device, dtype=torch.long),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_length), device=device),
    }


def create_mock_tokenizer(vocab_size: int = 50000) -> Mock:
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.vocab_size = vocab_size
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.bos_token_id = 2
    tokenizer.unk_token_id = 3
    tokenizer.mask_token_id = 4
    
    # Mock encoding/decoding methods
    tokenizer.encode.return_value = list(range(10))
    tokenizer.decode.return_value = "sample text"
    tokenizer.batch_encode_plus.return_value = {
        "input_ids": [[1, 2, 3, 4, 5]],
        "attention_mask": [[1, 1, 1, 1, 1]]
    }
    
    return tokenizer


def create_mock_dataset(size: int = 100, seq_length: int = 64) -> Mock:
    """Create a mock dataset for testing."""
    dataset = Mock()
    dataset.__len__.return_value = size
    
    def getitem(idx):
        return {
            "input_ids": torch.randint(0, 50000, (seq_length,)),
            "attention_mask": torch.ones(seq_length),
            "labels": torch.randint(0, 50000, (seq_length,)),
        }
    
    dataset.__getitem__.side_effect = getitem
    return dataset


@contextmanager
def temporary_checkpoint_dir():
    """Create a temporary directory for checkpoint testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_checkpoints_")
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@contextmanager
def mock_environment_variables(env_vars: Dict[str, str]):
    """Temporarily set environment variables for testing."""
    original_env = {}
    
    # Save original values and set new ones
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        yield
    finally:
        # Restore original values
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def assert_tensor_equal(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: Optional[str] = None
):
    """Assert that two tensors are equal within tolerance."""
    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        if msg is None:
            msg = f"Tensors not equal within tolerance (rtol={rtol}, atol={atol})"
        raise AssertionError(msg)


def assert_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    msg: Optional[str] = None
):
    """Assert that tensor has expected shape."""
    if tensor.shape != expected_shape:
        if msg is None:
            msg = f"Expected shape {expected_shape}, got {tensor.shape}"
        raise AssertionError(msg)


def assert_memory_usage_within_limit(
    memory_tracker: MemoryTracker,
    max_memory_mb: float,
    memory_type: str = "rss"
):
    """Assert that peak memory usage is within specified limit."""
    peak_memory = memory_tracker.peak_memory.get(memory_type, 0)
    if peak_memory > max_memory_mb:
        raise AssertionError(
            f"Peak {memory_type} memory usage {peak_memory:.2f}MB exceeds limit {max_memory_mb}MB"
        )


def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("No GPU available for this test")


def skip_if_no_distributed():
    """Skip test if distributed training is not available."""
    import pytest
    if not torch.distributed.is_available():
        pytest.skip("Distributed training not available")


def skip_if_insufficient_memory(required_gb: float):
    """Skip test if insufficient system memory."""
    import pytest
    available_gb = psutil.virtual_memory().available / (1024**3)
    if available_gb < required_gb:
        pytest.skip(f"Insufficient memory: {available_gb:.1f}GB < {required_gb}GB required")


def skip_if_insufficient_gpu_memory(required_gb: float):
    """Skip test if insufficient GPU memory."""
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")
    
    available_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if available_gb < required_gb:
        pytest.skip(f"Insufficient GPU memory: {available_gb:.1f}GB < {required_gb}GB required")


def create_expert_routing_data(
    batch_size: int = 8,
    seq_length: int = 64,
    num_experts: int = 8,
    experts_per_token: int = 2,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """Create sample expert routing data for testing analytics."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate routing decisions
    expert_indices = torch.randint(
        0, num_experts, 
        (batch_size, seq_length, experts_per_token), 
        device=device
    )
    
    # Generate routing weights (should sum to 1 per token)
    expert_weights = torch.rand(batch_size, seq_length, experts_per_token, device=device)
    expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
    
    # Generate router logits
    router_logits = torch.randn(batch_size, seq_length, num_experts, device=device)
    
    # Calculate expert utilization
    tokens_per_expert = torch.zeros(num_experts, device=device)
    for expert_idx in range(num_experts):
        tokens_per_expert[expert_idx] = (expert_indices == expert_idx).sum()
    
    return {
        "expert_indices": expert_indices,
        "expert_weights": expert_weights,
        "router_logits": router_logits,
        "num_experts": num_experts,
        "experts_per_token": experts_per_token,
        "tokens_per_expert": tokens_per_expert,
        "batch_size": batch_size,
        "seq_length": seq_length,
    }


def save_test_artifact(
    data: Any,
    filename: str,
    test_dir: Optional[Path] = None
) -> Path:
    """Save test artifact (results, logs, etc.) for debugging."""
    if test_dir is None:
        test_dir = Path("test_artifacts")
    test_dir.mkdir(exist_ok=True)
    
    filepath = test_dir / filename
    
    if isinstance(data, dict):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif isinstance(data, (list, tuple)):
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif isinstance(data, torch.Tensor):
        torch.save(data, filepath)
    else:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    return filepath


def load_test_artifact(filepath: Path) -> Any:
    """Load test artifact from file."""
    if filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.suffix in ['.pt', '.pth']:
        return torch.load(filepath, map_location='cpu')
    else:
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def generate_performance_report(
    test_results: Dict[str, Any],
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """Generate a performance report from test results."""
    report = {
        "timestamp": time.time(),
        "system_info": {
            "python_version": torch.__version__,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
        },
        "results": test_results
    }
    
    if torch.cuda.is_available():
        report["system_info"]["gpu_name"] = torch.cuda.get_device_name(0)
        report["system_info"]["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    return report