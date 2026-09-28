from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from torch import nn

from scripts.trajectory_preserving_utils import (
    backbone_sha256,
    load_backbone_state,
)
from scripts.evaluate_trajectory_preserving_joint_test import (
    load_test_archive_after_freeze,
)
from src.models.trajectory_preserving_joint import TrajectoryPreservingJoint
from src.models.trajectory_transformer import SceneTrajectoryTransformer


def make_pair(intent_input: str = "target"):
    standalone = SceneTrajectoryTransformer(
        input_dim=8,
        scene_dim=6,
        d_model=16,
        nhead=4,
        num_layers=1,
        pred_len=5,
        dropout=0.0,
        max_obs_len=7,
    )
    joint = TrajectoryPreservingJoint(
        input_dim=8,
        scene_dim=6,
        d_model=16,
        nhead=4,
        num_layers=1,
        pred_len=5,
        dropout=0.0,
        max_obs_len=7,
        intent_input=intent_input,
    )
    report = load_backbone_state(joint, standalone.state_dict())
    return standalone.eval(), joint, report


def test_pretrained_mapping_is_complete_and_shape_exact():
    standalone, joint, report = make_pair()
    assert report["complete"] is True
    assert report["total_expected_tensors"] == len(standalone.state_dict())
    assert report["loaded_tensors"] == len(standalone.state_dict())
    assert report["missing_tensors"] == []
    assert report["unexpected_tensors"] == []
    assert report["shape_mismatch_tensors"] == []
    for key, value in standalone.state_dict().items():
        torch.testing.assert_close(joint.backbone.state_dict()[key], value, rtol=0.0, atol=0.0)


def test_trajectory_path_matches_standalone_and_p1_p2_shapes():
    torch.manual_seed(91)
    target = torch.randn(4, 7, 8)
    scene = torch.randn(4, 6)
    standalone, p1, _ = make_pair("target")
    p2 = TrajectoryPreservingJoint(
        input_dim=8,
        scene_dim=6,
        d_model=16,
        nhead=4,
        num_layers=1,
        pred_len=5,
        dropout=0.0,
        max_obs_len=7,
        intent_input="target_scene",
    )
    load_backbone_state(p2, standalone.state_dict())
    with torch.no_grad():
        expected = standalone(target, scene)
        result1 = p1.eval()(target, scene)
        result2 = p2.eval()(target, scene)
    torch.testing.assert_close(result1["future_pred"], expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(result2["future_pred"], expected, rtol=0.0, atol=0.0)
    assert result1["future_pred"].shape == (4, 5, 2)
    assert result2["future_pred"].shape == (4, 5, 2)
    assert result1["intent_logit"].shape == (4,)
    assert result2["intent_logit"].shape == (4,)


@pytest.mark.parametrize("intent_input", ["target", "target_scene"])
def test_only_intention_branch_updates_and_backbone_remains_frozen(intent_input: str):
    torch.manual_seed(23)
    _, model, _ = make_pair(intent_input)
    model.train()
    backbone_before = backbone_sha256(model)
    intent_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.05
    )
    target = torch.randn(8, 7, 8)
    scene = torch.randn(8, 6)
    labels = torch.randint(0, 2, (8,), dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    loss = nn.functional.binary_cross_entropy_with_logits(model(target, scene)["intent_logit"], labels)
    loss.backward()
    optimizer.step()
    assert backbone_sha256(model) == backbone_before
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert any(
        not torch.equal(parameter.detach(), intent_before[name])
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def test_test_archive_is_not_loaded_before_protocol_freeze(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    checksum = tmp_path / "protocol.sha256"
    protocol.write_text('{"frozen": false}\n', encoding="utf-8")
    checksum.write_text(hashlib.sha256(protocol.read_bytes()).hexdigest() + "  protocol.json\n", encoding="utf-8")
    test_archive = tmp_path / "holdout_placeholder.npz"
    loaded = []

    def forbidden_numpy_load(*_args, **_kwargs):
        loaded.append(True)
        raise AssertionError("numpy.load must not be called before a valid freeze")

    monkeypatch.setattr(np, "load", forbidden_numpy_load)
    with pytest.raises(RuntimeError, match="valid frozen protocol"):
        load_test_archive_after_freeze(test_archive, protocol, checksum)
    assert loaded == []
