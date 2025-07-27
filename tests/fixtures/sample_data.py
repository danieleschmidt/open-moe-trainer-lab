"""
Sample data fixtures for testing.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
import tempfile
import json
from pathlib import Path


def create_sample_tokenized_dataset(
    vocab_size: int = 1000,
    num_samples: int = 100,
    seq_length: int = 128,
    seed: int = 42
) -> List[Dict[str, torch.Tensor]]:
    """Create a sample tokenized dataset for testing."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset = []
    for _ in range(num_samples):
        # Create random token sequences
        input_ids = torch.randint(0, vocab_size, (seq_length,))
        attention_mask = torch.ones(seq_length)
        
        # Random attention mask with some padding
        if np.random.random() < 0.3:  # 30% chance of padding
            pad_length = np.random.randint(1, seq_length // 4)
            attention_mask[-pad_length:] = 0
            input_ids[-pad_length:] = 0  # Pad token
        
        # Labels are the same as input_ids for language modeling
        labels = input_ids.clone()
        
        dataset.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        })
    
    return dataset


def create_routing_analytics_data(
    batch_size: int = 8,
    seq_length: int = 64,
    num_experts: int = 8,
    experts_per_token: int = 2,
    num_layers: int = 4
) -> Dict[str, Any]:
    """Create sample routing analytics data."""
    total_tokens = batch_size * seq_length
    
    routing_data = {
        "batch_size": batch_size,
        "seq_length": seq_length,
        "num_experts": num_experts,
        "experts_per_token": experts_per_token,
        "num_layers": num_layers,
        "layers": []
    }
    
    for layer_idx in range(num_layers):
        # Router logits for all tokens
        router_logits = torch.randn(total_tokens, num_experts)
        
        # Top-k expert selection
        routing_weights, selected_experts = torch.topk(router_logits, experts_per_token, dim=-1)
        routing_weights = torch.softmax(routing_weights, dim=-1)
        
        # Expert utilization statistics
        expert_counts = torch.zeros(num_experts)
        for expert_idx in range(num_experts):
            expert_counts[expert_idx] = (selected_experts == expert_idx).sum().float()
        
        # Load balancing metrics
        total_assignments = expert_counts.sum()
        expert_utilization = expert_counts / total_assignments
        load_variance = expert_utilization.var().item()
        
        # Router entropy (measure of routing diversity)
        router_probs = torch.softmax(router_logits, dim=-1)
        router_entropy = -(router_probs * torch.log(router_probs + 1e-8)).sum(dim=-1).mean().item()
        
        layer_data = {
            "layer_idx": layer_idx,
            "router_logits": router_logits,
            "selected_experts": selected_experts,
            "routing_weights": routing_weights,
            "expert_counts": expert_counts,
            "expert_utilization": expert_utilization,
            "load_variance": load_variance,
            "router_entropy": router_entropy,
            "total_tokens": total_tokens
        }
        
        routing_data["layers"].append(layer_data)
    
    return routing_data


def create_performance_baseline_data() -> Dict[str, Any]:
    """Create baseline performance data for regression testing."""
    return {
        "model_config": {
            "hidden_size": 768,
            "num_experts": 8,
            "experts_per_token": 2,
            "num_layers": 12,
            "vocab_size": 50000
        },
        "training_performance": {
            "tokens_per_second": 2500,
            "steps_per_second": 4.8,
            "memory_usage_gb": 12.5,
            "gradient_norm": 1.2,
            "loss_convergence_steps": 1000
        },
        "inference_performance": {
            "latency_ms": 45,
            "throughput_tokens_per_second": 3200,
            "memory_usage_gb": 8.2,
            "expert_cache_hit_rate": 0.85
        },
        "model_quality": {
            "perplexity": 12.5,
            "bleu_score": 0.28,
            "rouge_l": 0.32,
            "expert_specialization_score": 0.65
        },
        "scaling_metrics": {
            "expert_parallel_efficiency": 0.92,
            "data_parallel_efficiency": 0.88,
            "memory_scaling_factor": 0.75,
            "compute_scaling_factor": 0.95
        }
    }


def create_config_variants() -> List[Dict[str, Any]]:
    """Create different model configuration variants for testing."""
    base_config = {
        "hidden_size": 512,
        "num_experts": 4,
        "experts_per_token": 2,
        "num_layers": 6,
        "vocab_size": 1000,
        "intermediate_size": 2048,
        "max_seq_length": 512
    }
    
    variants = []
    
    # Small model for fast testing
    small_config = base_config.copy()
    small_config.update({
        "hidden_size": 128,
        "num_experts": 2,
        "experts_per_token": 1,
        "num_layers": 2,
        "intermediate_size": 512,
        "max_seq_length": 64
    })
    variants.append(("small", small_config))
    
    # Medium model
    medium_config = base_config.copy()
    variants.append(("medium", medium_config))
    
    # Large model with many experts
    large_config = base_config.copy()
    large_config.update({
        "hidden_size": 1024,
        "num_experts": 16,
        "experts_per_token": 4,
        "num_layers": 12,
        "intermediate_size": 4096
    })
    variants.append(("large", large_config))
    
    # Switch Transformer style (single expert per token)
    switch_config = base_config.copy()
    switch_config.update({
        "num_experts": 32,
        "experts_per_token": 1,
        "routing_strategy": "switch"
    })
    variants.append(("switch", switch_config))
    
    # Mixtral style (top-2 routing)
    mixtral_config = base_config.copy()
    mixtral_config.update({
        "num_experts": 8,
        "experts_per_token": 2,
        "routing_strategy": "topk"
    })
    variants.append(("mixtral", mixtral_config))
    
    return variants


def create_training_scenarios() -> List[Dict[str, Any]]:
    """Create different training scenarios for testing."""
    scenarios = []
    
    # Fast training for testing
    fast_scenario = {
        "name": "fast",
        "batch_size": 2,
        "learning_rate": 1e-3,
        "num_epochs": 1,
        "max_steps": 10,
        "warmup_steps": 2,
        "eval_steps": 5,
        "save_steps": 10,
        "logging_steps": 1,
        "gradient_accumulation_steps": 1
    }
    scenarios.append(fast_scenario)
    
    # Standard training
    standard_scenario = {
        "name": "standard",
        "batch_size": 8,
        "learning_rate": 5e-4,
        "num_epochs": 3,
        "max_steps": 1000,
        "warmup_steps": 100,
        "eval_steps": 100,
        "save_steps": 200,
        "logging_steps": 10,
        "gradient_accumulation_steps": 2
    }
    scenarios.append(standard_scenario)
    
    # Large scale training
    large_scenario = {
        "name": "large",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 10,
        "max_steps": 10000,
        "warmup_steps": 1000,
        "eval_steps": 500,
        "save_steps": 1000,
        "logging_steps": 50,
        "gradient_accumulation_steps": 4
    }
    scenarios.append(large_scenario)
    
    return scenarios


def create_mock_checkpoint_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create mock checkpoint data for testing."""
    checkpoint = {
        "model_state_dict": {},
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [{"lr": 1e-4, "weight_decay": 0.01}]
        },
        "scheduler_state_dict": {},
        "epoch": 5,
        "step": 1234,
        "best_metric": 2.45,
        "config": config,
        "training_args": {
            "batch_size": 16,
            "learning_rate": 1e-4,
            "num_epochs": 10
        },
        "metrics": {
            "train_loss": [3.2, 2.8, 2.5, 2.3, 2.1],
            "eval_loss": [2.9, 2.6, 2.4, 2.2, 2.0],
            "expert_utilization": [0.12, 0.11, 0.13, 0.12, 0.11],
            "router_entropy": [2.1, 2.0, 2.1, 2.0, 2.1]
        },
        "timestamp": "2025-01-27T10:30:00Z",
        "git_commit": "abc123def456",
        "python_version": "3.9.0",
        "torch_version": "2.0.0"
    }
    
    # Add mock model weights
    hidden_size = config["hidden_size"]
    num_experts = config["num_experts"]
    vocab_size = config["vocab_size"]
    
    # Simplified weight shapes
    checkpoint["model_state_dict"] = {
        "embedding.weight": torch.randn(vocab_size, hidden_size),
        "router.weight": torch.randn(num_experts, hidden_size),
        "router.bias": torch.randn(num_experts),
        "lm_head.weight": torch.randn(vocab_size, hidden_size),
        "lm_head.bias": torch.randn(vocab_size)
    }
    
    # Add expert weights
    intermediate_size = config.get("intermediate_size", hidden_size * 4)
    for i in range(num_experts):
        checkpoint["model_state_dict"][f"experts.{i}.fc1.weight"] = torch.randn(intermediate_size, hidden_size)
        checkpoint["model_state_dict"][f"experts.{i}.fc1.bias"] = torch.randn(intermediate_size)
        checkpoint["model_state_dict"][f"experts.{i}.fc2.weight"] = torch.randn(hidden_size, intermediate_size)
        checkpoint["model_state_dict"][f"experts.{i}.fc2.bias"] = torch.randn(hidden_size)
    
    return checkpoint


def create_evaluation_datasets() -> Dict[str, List[Dict[str, Any]]]:
    """Create evaluation datasets for different tasks."""
    datasets = {}
    
    # Language modeling dataset
    lm_data = []
    for i in range(50):
        text_length = np.random.randint(32, 128)
        input_ids = torch.randint(0, 1000, (text_length,))
        labels = input_ids.clone()
        
        lm_data.append({
            "input_ids": input_ids,
            "labels": labels,
            "text": f"Sample text {i} for language modeling task."
        })
    
    datasets["language_modeling"] = lm_data
    
    # Text completion dataset
    completion_data = []
    for i in range(30):
        prompt_length = np.random.randint(16, 64)
        completion_length = np.random.randint(16, 64)
        
        prompt_ids = torch.randint(0, 1000, (prompt_length,))
        completion_ids = torch.randint(0, 1000, (completion_length,))
        
        completion_data.append({
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "prompt_text": f"Prompt {i}:",
            "completion_text": f"This is completion {i} for the prompt."
        })
    
    datasets["text_completion"] = completion_data
    
    # Classification dataset (simplified)
    classification_data = []
    for i in range(40):
        text_length = np.random.randint(16, 100)
        input_ids = torch.randint(0, 1000, (text_length,))
        label = np.random.randint(0, 5)  # 5 classes
        
        classification_data.append({
            "input_ids": input_ids,
            "label": label,
            "text": f"Classification sample {i} with label {label}."
        })
    
    datasets["classification"] = classification_data
    
    return datasets


def save_test_fixtures(output_dir: Path):
    """Save test fixtures to disk for reuse."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample dataset
    dataset = create_sample_tokenized_dataset(num_samples=20)
    torch.save(dataset, output_dir / "sample_dataset.pt")
    
    # Save routing data
    routing_data = create_routing_analytics_data()
    torch.save(routing_data, output_dir / "routing_analytics.pt")
    
    # Save performance baseline
    baseline = create_performance_baseline_data()
    with open(output_dir / "performance_baseline.json", "w") as f:
        json.dump(baseline, f, indent=2)
    
    # Save config variants
    configs = create_config_variants()
    with open(output_dir / "config_variants.json", "w") as f:
        json.dump(dict(configs), f, indent=2)
    
    # Save training scenarios
    scenarios = create_training_scenarios()
    with open(output_dir / "training_scenarios.json", "w") as f:
        json.dump({s["name"]: s for s in scenarios}, f, indent=2)
    
    # Save evaluation datasets
    eval_datasets = create_evaluation_datasets()
    torch.save(eval_datasets, output_dir / "evaluation_datasets.pt")
    
    print(f"Test fixtures saved to {output_dir}")


def load_test_fixtures(fixtures_dir: Path) -> Dict[str, Any]:
    """Load test fixtures from disk."""
    fixtures = {}
    
    # Load dataset
    if (fixtures_dir / "sample_dataset.pt").exists():
        fixtures["sample_dataset"] = torch.load(fixtures_dir / "sample_dataset.pt")
    
    # Load routing data
    if (fixtures_dir / "routing_analytics.pt").exists():
        fixtures["routing_data"] = torch.load(fixtures_dir / "routing_analytics.pt")
    
    # Load performance baseline
    if (fixtures_dir / "performance_baseline.json").exists():
        with open(fixtures_dir / "performance_baseline.json") as f:
            fixtures["performance_baseline"] = json.load(f)
    
    # Load config variants
    if (fixtures_dir / "config_variants.json").exists():
        with open(fixtures_dir / "config_variants.json") as f:
            fixtures["config_variants"] = json.load(f)
    
    # Load training scenarios
    if (fixtures_dir / "training_scenarios.json").exists():
        with open(fixtures_dir / "training_scenarios.json") as f:
            fixtures["training_scenarios"] = json.load(f)
    
    # Load evaluation datasets
    if (fixtures_dir / "evaluation_datasets.pt").exists():
        fixtures["evaluation_datasets"] = torch.load(fixtures_dir / "evaluation_datasets.pt")
    
    return fixtures


# Pytest fixtures for use in tests
def pytest_sample_dataset():
    """Pytest fixture for sample dataset."""
    return create_sample_tokenized_dataset(num_samples=10, seq_length=32)


def pytest_routing_data():
    """Pytest fixture for routing analytics data."""
    return create_routing_analytics_data(batch_size=4, seq_length=16, num_experts=4)


def pytest_config_variants():
    """Pytest fixture for configuration variants."""
    return dict(create_config_variants())


def pytest_training_scenarios():
    """Pytest fixture for training scenarios."""
    scenarios = create_training_scenarios()
    return {s["name"]: s for s in scenarios}


if __name__ == "__main__":
    # Create and save test fixtures when run as script
    output_dir = Path("test_fixtures")
    save_test_fixtures(output_dir)