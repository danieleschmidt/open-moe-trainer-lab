"""
Pytest configuration and shared fixtures for Open MoE Trainer Lab tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

# Test configuration
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "distributed: marks tests as requiring distributed setup"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Mark tests automatically based on their location."""
    for item in items:
        # Mark tests in certain directories
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "distributed" in str(item.fspath):
            item.add_marker(pytest.mark.distributed)
        elif "benchmarks" in str(item.fspath):
            item.add_marker(pytest.mark.slow)


@pytest.fixture(scope="session")
def torch_device():
    """Get the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Mock configuration for testing."""
    return {
        "model": {
            "hidden_size": 512,
            "num_experts": 4,
            "experts_per_token": 2,
            "num_layers": 6,
            "vocab_size": 1000,
            "max_seq_length": 512
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 1e-4,
            "num_epochs": 1,
            "warmup_steps": 10,
            "load_balance_loss_coef": 0.01,
            "router_z_loss_coef": 0.001
        },
        "distributed": {
            "world_size": 1,
            "expert_parallel_size": 1,
            "data_parallel_size": 1
        }
    }


@pytest.fixture
def sample_batch(torch_device):
    """Create a sample batch for testing."""
    batch_size = 4
    seq_length = 64
    vocab_size = 1000
    
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length), device=torch_device),
        "attention_mask": torch.ones(batch_size, seq_length, device=torch_device),
        "labels": torch.randint(0, vocab_size, (batch_size, seq_length), device=torch_device)
    }


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.vocab_size = 1000
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.bos_token_id = 2
    tokenizer.encode.return_value = [2, 100, 200, 300, 1]  # Simple sequence
    tokenizer.decode.return_value = "Hello world"
    return tokenizer


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    dataset = MagicMock()
    dataset.__len__.return_value = 100
    dataset.__getitem__.return_value = {
        "input_ids": torch.randint(0, 1000, (64,)),
        "attention_mask": torch.ones(64),
        "labels": torch.randint(0, 1000, (64,))
    }
    return dataset


@pytest.fixture
def expert_routing_data():
    """Sample expert routing data for analytics testing."""
    batch_size = 8
    seq_length = 32
    num_experts = 4
    
    return {
        "expert_indices": torch.randint(0, num_experts, (batch_size, seq_length, 2)),
        "expert_weights": torch.rand(batch_size, seq_length, 2),
        "router_logits": torch.randn(batch_size, seq_length, num_experts),
        "expert_capacity": torch.tensor([16, 16, 16, 16]),  # Equal capacity
        "tokens_per_expert": torch.tensor([15, 17, 14, 16])  # Slightly unbalanced
    }


@pytest.fixture(scope="session")
def skip_if_no_gpu():
    """Skip test if no GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")


@pytest.fixture(scope="session") 
def skip_if_no_distributed():
    """Skip test if distributed training is not available."""
    if not torch.distributed.is_available():
        pytest.skip("Distributed training not available")


@pytest.fixture
def checkpoint_dir(temp_dir):
    """Create a temporary checkpoint directory."""
    ckpt_dir = temp_dir / "checkpoints"
    ckpt_dir.mkdir()
    return ckpt_dir


@pytest.fixture
def mock_wandb_run():
    """Mock wandb run for testing."""
    run = MagicMock()
    run.log = MagicMock()
    run.finish = MagicMock()
    run.id = "test_run_123"
    run.project = "test_project"
    return run


@pytest.fixture
def performance_baseline():
    """Performance baseline metrics for regression testing."""
    return {
        "training_speed": {
            "tokens_per_second": 1000,
            "steps_per_second": 10,
            "memory_usage_gb": 8.0
        },
        "inference_speed": {
            "latency_ms": 50,
            "throughput_tokens_per_second": 2000,
            "memory_usage_gb": 4.0
        },
        "model_quality": {
            "perplexity": 15.0,
            "bleu_score": 0.25,
            "rouge_l": 0.30
        }
    }


@pytest.fixture(autouse=True)
def set_deterministic():
    """Set deterministic behavior for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "WANDB_MODE": "disabled",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONPATH": "/workspace",
        "LOG_LEVEL": "DEBUG"
    }
    
    # Set environment variables
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield env_vars
    
    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def model_config_small():
    """Small model configuration for fast testing."""
    return {
        "hidden_size": 128,
        "num_experts": 2,
        "experts_per_token": 1,
        "num_layers": 2,
        "vocab_size": 100,
        "intermediate_size": 256,
        "max_seq_length": 32
    }


@pytest.fixture
def training_config_fast():
    """Fast training configuration for testing."""
    return {
        "batch_size": 2,
        "learning_rate": 1e-3,
        "num_epochs": 1,
        "max_steps": 5,
        "warmup_steps": 1,
        "eval_steps": 2,
        "save_steps": 3,
        "logging_steps": 1
    }