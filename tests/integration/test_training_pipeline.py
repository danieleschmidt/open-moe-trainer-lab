"""Integration tests for the training pipeline."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from moe_lab.models import MoEModel
from moe_lab.training import MoETrainer


class TestTrainingIntegration:
    """End-to-end training pipeline tests."""

    def _make_model_and_data(self, device, seq_len=8, batch_size=4, vocab_size=100):
        model = MoEModel(
            vocab_size=vocab_size,
            hidden_size=32,
            num_experts=4,
            experts_per_token=2,
            num_layers=2,
            num_attention_heads=4,
            max_position_embeddings=32,
            moe_layers=[1],
        ).to(device)

        # Simple language modelling: predict next token
        tokens = torch.randint(0, vocab_size, (batch_size * 4, seq_len + 1))
        input_ids = tokens[:, :-1]
        labels = tokens[:, 1:]
        dataset = TensorDataset(input_ids, labels)
        loader = DataLoader(dataset, batch_size=batch_size)
        return model, loader

    def test_single_step(self, device):
        model, loader = self._make_model_and_data(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        model.train()
        input_ids, labels = next(iter(loader))
        input_ids, labels = input_ids.to(device), labels.to(device)

        out = model(input_ids)
        # Project to vocab (use lm_head if present, else linear on the fly)
        logits = model.lm_head(out.last_hidden_state)

        task_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss = task_loss + out.load_balancing_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        assert not torch.isnan(total_loss)

    def test_loss_decreases_over_steps(self, device):
        """Loss should generally decrease over several training steps on same batch."""
        model, loader = self._make_model_and_data(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        model.train()
        input_ids, labels = next(iter(loader))
        input_ids, labels = input_ids.to(device), labels.to(device)

        losses = []
        for _ in range(10):
            out = model(input_ids)
            logits = model.lm_head(out.last_hidden_state)
            task_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss = task_loss + out.load_balancing_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses.append(total_loss.item())

        # Loss should decrease at least somewhat from first to last
        assert losses[-1] < losses[0], f"Loss didn't decrease: {losses}"

    def test_model_checkpoint_roundtrip(self, device, tmp_path, small_model_config):
        """Save and reload model weights."""
        model = MoEModel(**small_model_config).to(device)
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)

        model2 = MoEModel(**small_model_config).to(device)
        model2.load_state_dict(torch.load(path, weights_only=True))

        # State dicts must match exactly
        sd1 = model.state_dict()
        sd2 = model2.state_dict()
        assert set(sd1.keys()) == set(sd2.keys())
        for key in sd1:
            assert torch.equal(sd1[key], sd2[key]), f"Mismatch in {key}"
