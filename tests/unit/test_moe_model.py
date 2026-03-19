"""Unit tests for the MoELayer and MoEModel."""

import pytest
import torch
from moe_lab.models.moe_model import MoELayer, MoEModel, MoEOutput


class TestMoELayer:
    def test_output_shape(self, device):
        layer = MoELayer(hidden_size=64, num_experts=4, experts_per_token=2).to(device)
        x = torch.randn(2, 8, 64, device=device)
        out, routing_info, loss = layer(x)
        assert out.shape == (2, 8, 64)

    def test_returns_scalar_loss(self, device):
        layer = MoELayer(hidden_size=64, num_experts=4, experts_per_token=2).to(device)
        x = torch.randn(2, 8, 64, device=device)
        _, _, loss = layer(x)
        assert loss.ndim == 0  # scalar

    def test_loss_is_non_negative(self, device):
        layer = MoELayer(hidden_size=64, num_experts=4, experts_per_token=2).to(device)
        x = torch.randn(2, 8, 64, device=device)
        _, _, loss = layer(x)
        assert loss.item() >= 0

    def test_gradients_flow(self, device):
        layer = MoELayer(hidden_size=32, num_experts=4, experts_per_token=2).to(device)
        x = torch.randn(2, 4, 32, device=device, requires_grad=True)
        out, _, loss = layer(x)
        total = out.sum() + loss
        total.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestMoEModel:
    def test_forward_output_type(self, device, small_model_config):
        model = MoEModel(**small_model_config).to(device)
        ids = torch.randint(0, small_model_config["vocab_size"], (2, 16), device=device)
        out = model(ids)
        assert isinstance(out, MoEOutput)

    def test_output_shape(self, device, small_model_config):
        model = MoEModel(**small_model_config).to(device)
        ids = torch.randint(0, small_model_config["vocab_size"], (2, 16), device=device)
        out = model(ids)
        assert out.last_hidden_state.shape == (2, 16, small_model_config["hidden_size"])

    def test_load_balancing_loss_present(self, device, small_model_config):
        model = MoEModel(**small_model_config).to(device)
        ids = torch.randint(0, small_model_config["vocab_size"], (2, 16), device=device)
        out = model(ids)
        assert out.load_balancing_loss is not None

    def test_routing_info_stats(self, device, small_model_config):
        model = MoEModel(**small_model_config).to(device)
        ids = torch.randint(0, small_model_config["vocab_size"], (2, 16), device=device)
        out = model(ids)
        assert out.routing_info.load_variance >= 0
        assert out.routing_info.entropy > 0

    def test_gradients_flow_through_model(self, device, small_model_config):
        model = MoEModel(**small_model_config).to(device)
        ids = torch.randint(0, small_model_config["vocab_size"], (2, 8), device=device)
        out = model(ids)
        loss = out.last_hidden_state.sum() + out.load_balancing_loss
        loss.backward()
        # Check at least some params have gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_parameter_count(self, device, small_model_config):
        model = MoEModel(**small_model_config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_return_routing_info(self, device, small_model_config):
        model = MoEModel(**small_model_config).to(device)
        ids = torch.randint(0, small_model_config["vocab_size"], (2, 16), device=device)
        out = model(ids, return_routing_info=True)
        assert out.expert_weights is not None

    def test_all_moe_layers(self, device):
        """Model with all layers as MoE."""
        model = MoEModel(
            vocab_size=100,
            hidden_size=32,
            num_experts=4,
            experts_per_token=1,
            num_layers=2,
            num_attention_heads=4,
            max_position_embeddings=16,
            moe_layers=[0, 1],
        ).to(device)
        ids = torch.randint(0, 100, (1, 8), device=device)
        out = model(ids)
        assert out.last_hidden_state.shape == (1, 8, 32)
