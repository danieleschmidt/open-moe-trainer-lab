"""Unit tests for expert networks."""

import pytest
import torch
from moe_lab.models.expert import Expert, GLUExpert, ExpertPool


class TestExpert:
    def test_output_shape(self, device):
        expert = Expert(hidden_size=64, expert_hidden_size=256).to(device)
        x = torch.randn(8, 64, device=device)
        out = expert(x)
        assert out.shape == (8, 64)

    def test_residual_dimension(self, device):
        """Expert output has same dimension as input."""
        expert = Expert(hidden_size=128, expert_hidden_size=512).to(device)
        x = torch.randn(4, 128, device=device)
        assert expert(x).shape == x.shape

    def test_activations(self, device):
        for act in ("gelu", "relu", "swish"):
            expert = Expert(hidden_size=32, expert_hidden_size=128, activation=act).to(device)
            x = torch.randn(4, 32, device=device)
            out = expert(x)
            assert out.shape == (4, 32), f"Failed for activation {act}"

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError):
            Expert(hidden_size=32, expert_hidden_size=128, activation="tanh")


class TestGLUExpert:
    def test_output_shape(self, device):
        expert = GLUExpert(hidden_size=64, expert_hidden_size=256).to(device)
        x = torch.randn(8, 64, device=device)
        out = expert(x)
        assert out.shape == (8, 64)


class TestExpertPool:
    def test_init(self):
        pool = ExpertPool(num_experts=4, hidden_size=64, expert_hidden_size=256)
        assert len(pool.experts) == 4

    def test_forward_single_expert(self, device):
        pool = ExpertPool(num_experts=4, hidden_size=64, expert_hidden_size=256).to(device)
        x = torch.randn(1, 64, device=device)
        out = pool.forward(x, expert_idx=0)
        assert out.shape == (1, 64)

    def test_forward_all_routing(self, device):
        num_experts, top_k = 4, 2
        pool = ExpertPool(num_experts=num_experts, hidden_size=64, expert_hidden_size=256).to(device)

        batch_size = 6
        x = torch.randn(batch_size, 64, device=device)

        # Simulate top-2 routing
        selected = torch.randint(0, num_experts, (batch_size, top_k), device=device)
        weights = torch.softmax(torch.randn(batch_size, top_k, device=device), dim=-1)

        out = pool.forward_all(x, expert_weights=weights, selected_experts=selected)
        assert out.shape == (batch_size, 64)

    def test_expert_utilization(self, device):
        pool = ExpertPool(num_experts=4, hidden_size=32, expert_hidden_size=128).to(device)
        from moe_lab.models.router import RoutingInfo

        # Construct mock routing history
        selected = torch.randint(0, 4, (16, 2))
        routing_info = RoutingInfo(
            expert_weights=torch.rand(16, 2),
            selected_experts=selected,
            router_logits=torch.rand(16, 4),
            load_variance=0.01,
            entropy=1.5,
        )

        utilization = pool.get_expert_utilization([routing_info])
        assert len(utilization) == 4
        total = sum(utilization.values())
        assert abs(total - 1.0) < 1e-5

    def test_invalid_expert_index_raises(self, device):
        pool = ExpertPool(num_experts=4, hidden_size=32, expert_hidden_size=128).to(device)
        x = torch.randn(1, 32, device=device)
        with pytest.raises(ValueError):
            pool.forward(x, expert_idx=10)
