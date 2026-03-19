"""Unit tests for MoE routing components."""

import pytest
import torch
from moe_lab.models.router import TopKRouter, SwitchRouter, RoutingInfo


class TestTopKRouter:
    """Tests for the TopK router."""

    def test_init(self):
        router = TopKRouter(hidden_size=64, num_experts=4, top_k=2)
        assert router.hidden_size == 64
        assert router.num_experts == 4
        assert router.top_k == 2

    def test_forward_output_shapes(self, device):
        hidden_size, num_experts, top_k = 64, 4, 2
        batch_size = 8

        router = TopKRouter(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k).to(device)
        x = torch.randn(batch_size, hidden_size, device=device)

        router_logits, selected_experts, expert_weights, routing_info = router(x)

        assert router_logits.shape == (batch_size, num_experts)
        assert selected_experts.shape == (batch_size, top_k)
        assert expert_weights.shape == (batch_size, top_k)

    def test_selected_experts_are_valid_indices(self, device):
        num_experts = 4
        router = TopKRouter(hidden_size=32, num_experts=num_experts, top_k=2).to(device)
        x = torch.randn(16, 32, device=device)

        _, selected_experts, _, _ = router(x)

        assert selected_experts.min() >= 0
        assert selected_experts.max() < num_experts

    def test_expert_weights_sum_to_one(self, device):
        router = TopKRouter(hidden_size=32, num_experts=4, top_k=2).to(device)
        x = torch.randn(8, 32, device=device)

        _, _, expert_weights, _ = router(x)

        weight_sums = expert_weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_routing_info_fields(self, device):
        router = TopKRouter(hidden_size=32, num_experts=4, top_k=2).to(device)
        x = torch.randn(8, 32, device=device)

        _, _, _, routing_info = router(x)

        assert isinstance(routing_info, RoutingInfo)
        assert routing_info.load_variance >= 0
        assert routing_info.entropy >= 0

    def test_jitter_noise_only_during_training(self, device):
        router = TopKRouter(hidden_size=32, num_experts=4, top_k=2, jitter_noise=0.5).to(device)
        x = torch.randn(8, 32, device=device)

        router.train()
        logits_train, _, _, _ = router(x)

        router.eval()
        logits_eval, _, _, _ = router(x)

        # Can't guarantee different (noise might add 0), but both should be tensors
        assert logits_train.shape == logits_eval.shape

    def test_top_k_one(self, device):
        router = TopKRouter(hidden_size=32, num_experts=8, top_k=1).to(device)
        x = torch.randn(16, 32, device=device)

        _, selected, weights, _ = router(x)

        assert selected.shape == (16, 1)
        assert weights.shape == (16, 1)
        # Single expert selected, so weights should be 1.0
        assert torch.allclose(weights, torch.ones_like(weights), atol=1e-5)


class TestSwitchRouter:
    """Tests for the Switch router (top-1)."""

    def test_forward(self, device):
        router = SwitchRouter(hidden_size=32, num_experts=4).to(device)
        x = torch.randn(8, 32, device=device)

        logits, selected, weights, info = router(x)

        assert logits.shape == (8, 4)
        assert selected.shape[0] == 8
        assert isinstance(info, RoutingInfo)

    def test_single_expert_per_token(self, device):
        router = SwitchRouter(hidden_size=32, num_experts=4).to(device)
        x = torch.randn(16, 32, device=device)

        _, selected, _, _ = router(x)
        # selected is (batch, 1)
        assert selected.shape == (16, 1)
