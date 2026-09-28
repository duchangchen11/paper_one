from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_intention_scratch_matched import model_sha256
from scripts.analyze_intention_scratch_matched import paired_bootstrap_p1_minus_m0
from scripts.evaluate_intention_scratch_test import verify_protocol
from scripts.trajectory_preserving_utils import TrajectoryIntentDataset, set_seed
from src.models.intention_scratch_transformer import IntentionScratchTransformer
from src.models.trajectory_preserving_joint import TrajectoryPreservingJoint


def make_p1_and_m0(*, d_model=16, layers=1, dropout=0.0, max_obs_len=5):
    p1 = TrajectoryPreservingJoint(
        input_dim=8,
        scene_dim=6,
        d_model=d_model,
        nhead=4,
        num_layers=layers,
        pred_len=3,
        dropout=dropout,
        max_obs_len=max_obs_len,
        intent_input="target",
    )
    m0 = IntentionScratchTransformer(
        input_dim=8,
        d_model=d_model,
        nhead=4,
        num_layers=layers,
        dropout=dropout,
        max_obs_len=max_obs_len,
    )
    return p1, m0


def test_scratch_model_input_and_output_shapes_match_p1_target_context():
    p1, m0 = make_p1_and_m0()
    target = torch.randn(4, 5, 8)
    with torch.no_grad():
        p1_context = p1.backbone.temporal_encoder(
            p1.backbone.input_projection(target) + p1.backbone.position_embedding[:, :5]
        )[:, -1]
        m0_output = m0(target)
    assert m0_output["target_context"].shape == p1_context.shape == (4, 16)
    assert m0_output["intent_logit"].shape == (4,)


def test_scratch_encoder_and_intention_head_architecture_match_p1():
    p1, m0 = make_p1_and_m0()
    p1_state, m0_state = p1.state_dict(), m0.state_dict()
    mapping = {
        "backbone.input_projection.weight": "input_projection.weight",
        "backbone.input_projection.bias": "input_projection.bias",
        "backbone.position_embedding": "position_embedding",
    }
    for name in p1_state:
        if name.startswith("backbone.temporal_encoder."):
            mapping[name] = name.removeprefix("backbone.")
        elif name.startswith(("intent_adapter.", "intent_head.")):
            mapping[name] = name
    assert mapping
    assert all(left in p1_state and right in m0_state for left, right in mapping.items())
    assert all(tuple(p1_state[left].shape) == tuple(m0_state[right].shape) for left, right in mapping.items())
    assert not hasattr(m0, "scene_encoder")
    assert not hasattr(m0, "decoder")
    assert isinstance(m0.intent_adapter[0], nn.LayerNorm)
    assert isinstance(m0.intent_adapter[1], nn.Linear)
    assert isinstance(m0.intent_adapter[2], nn.GELU)
    assert isinstance(m0.intent_adapter[3], nn.Dropout)
    assert isinstance(m0.intent_head, nn.Linear)


def test_scratch_model_initialization_does_not_read_any_checkpoint(monkeypatch):
    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("M0 initialization must not load checkpoints")

    monkeypatch.setattr(torch, "load", forbidden_load)
    set_seed(42)
    first = IntentionScratchTransformer(d_model=16, nhead=4, num_layers=1, max_obs_len=5)
    first_hash = model_sha256(first)
    set_seed(42)
    second = IntentionScratchTransformer(d_model=16, nhead=4, num_layers=1, max_obs_len=5)
    assert first_hash == model_sha256(second)


def test_optimizer_updates_scratch_encoder_and_intention_head():
    set_seed(73)
    model = IntentionScratchTransformer(d_model=16, nhead=4, num_layers=1, max_obs_len=5)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    target = torch.randn(8, 5, 8)
    labels = torch.randint(0, 2, (8,), dtype=torch.float32)
    loss = nn.functional.binary_cross_entropy_with_logits(model(target)["intent_logit"], labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert any(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in model.named_parameters()
        if name.startswith(("input_projection.", "temporal_encoder."))
    )
    assert any(
        not torch.equal(parameter.detach(), before[name])
        for name, parameter in model.named_parameters()
        if name.startswith(("intent_adapter.", "intent_head."))
    )


def test_train_validation_dataset_guard_rejects_test_path_before_numpy_load(monkeypatch, tmp_path):
    calls = []

    def forbidden_load(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("test split must not be loaded during training")

    monkeypatch.setattr("scripts.trajectory_preserving_utils.np.load", forbidden_load)
    with pytest.raises(RuntimeError, match="Test split may only be loaded"):
        TrajectoryIntentDataset(tmp_path / "test.npz")
    assert calls == []


def test_seed_controls_initial_weights_and_natural_dataloader_order():
    def build(seed: int):
        set_seed(seed)
        model = IntentionScratchTransformer(d_model=16, nhead=4, num_layers=1, max_obs_len=5)
        dataset = TensorDataset(torch.arange(24))
        loader = DataLoader(
            dataset,
            batch_size=5,
            shuffle=True,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_workers=0,
        )
        order = torch.cat([batch[0] for batch in loader])
        return model_sha256(model), order

    hash_a, order_a = build(123)
    hash_b, order_b = build(123)
    hash_c, order_c = build(2024)
    assert hash_a == hash_b
    assert torch.equal(order_a, order_b)
    assert hash_a != hash_c
    assert not torch.equal(order_a, order_c)


def test_paired_bootstrap_delta_sign_is_p1_minus_m0():
    labels = np.tile(np.asarray([0, 1], dtype=np.int64), 6)
    scenes = np.repeat(np.asarray([f"scene-{i}" for i in range(6)]), 2)
    m0_probability = np.tile(np.asarray([0.9, 0.1]), 6)
    p1_probability = np.tile(np.asarray([0.1, 0.9]), 6)
    result = paired_bootstrap_p1_minus_m0(
        labels,
        m0_probability,
        p1_probability,
        scenes,
        repetitions=100,
        seed=11,
    )
    assert result["delta_roc_auc"]["mean"] == pytest.approx(1.0)
    assert result["delta_brier"]["mean"] < 0.0


def test_test_evaluator_rejects_unfrozen_protocol(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    checksum_path = tmp_path / "protocol.sha256"
    payload = json.dumps({"frozen": False, "m0_test_accessed_before_freeze": False}).encode()
    protocol_path.write_bytes(payload)
    checksum_path.write_text(hashlib.sha256(payload).hexdigest() + "  protocol.json\n")
    with pytest.raises(RuntimeError, match="valid frozen protocol"):
        verify_protocol(protocol_path, checksum_path)
