"""Shared pytest fixtures for open-moe-trainer-lab tests."""

import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_model_config():
    """Minimal config for fast tests."""
    return {
        "vocab_size": 500,
        "hidden_size": 64,
        "num_experts": 4,
        "experts_per_token": 2,
        "num_layers": 2,
        "num_attention_heads": 4,
        "max_position_embeddings": 32,
        "moe_layers": [1],
    }
