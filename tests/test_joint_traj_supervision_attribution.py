from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from scripts.joint_traj_supervision_utils import (
    SHARED_PREFIXES,
    compose_objective,
    guard_pretest_split,
    linear_norm,
    only_trajectory_weight_differs,
    validate_metrics_payload,
    write_json,
)
from src.models.joint_transformer_gate import JointTransformerSceneGate


def _tiny_forward():
    torch.manual_seed(7)
    model = JointTransformerSceneGate(
        input_dim=8, scene_dim=5, hidden_dim=16, pred_len=3,
        nhead=4, num_layers=1, dropout=0.0, gate_mode="uncertainty", max_obs_len=4,
    )
    model.train()
    target = torch.randn(4, 4, 8)
    neighbors = torch.randn(4, 2, 4, 4)
    neighbor_mask = torch.ones(4, 2)
    visible = torch.ones(4, 2, 4)
    scene = torch.randn(4, 5)
    output = model(target, neighbors, neighbor_mask, visible, scene)
    return model, output


def test_j0_keeps_joint_future_output_and_zeroes_only_trajectory_objective():
    model, output = _tiny_forward()
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    future = torch.randn(4, 3, 2)
    main = nn.functional.binary_cross_entropy_with_logits(output["intent_logit"], labels)
    prior = nn.functional.binary_cross_entropy_with_logits(output["prior_logit"], labels)
    trajectory = nn.functional.smooth_l1_loss(output["future_pred"], future)
    ambiguity = output["intent_logit"].square().mean()
    objective = compose_objective(
        main, prior, trajectory, ambiguity,
        prior_weight=0.5, traj_weight=0.0, ambiguous_weight=0.2,
    )
    intent_objective = main + 0.5 * prior + 0.2 * ambiguity
    assert output["future_pred"].shape == (4, 3, 2)
    assert torch.equal(objective, intent_objective)

    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    shared = [(name, param) for name, param in named if name.startswith(SHARED_PREFIXES)]
    traj_head = [(name, param) for name, param in named if name.startswith("traj_head.")]
    zero_traj_grads = torch.autograd.grad(0.0 * trajectory, [p for _, p in traj_head], retain_graph=True)
    total_grads = torch.autograd.grad(objective, [p for _, p in shared], retain_graph=True)
    intent_grads = torch.autograd.grad(intent_objective, [p for _, p in shared])
    zero_norm, _ = linear_norm(zero_traj_grads, [p for _, p in traj_head])
    total_norm, total_vec = linear_norm(total_grads, [p for _, p in shared])
    intent_norm, intent_vec = linear_norm(intent_grads, [p for _, p in shared])
    assert zero_norm == 0.0
    assert total_norm > 0.0
    assert torch.equal(total_vec, intent_vec)
    assert intent_norm == total_norm


def test_configuration_contract_allows_only_lambda_difference():
    j100 = {
        "seed": 42, "epochs": 15, "batch_size": 512,
        "optimizer": "AdamW", "traj_weight": 100.0,
    }
    j0 = copy.deepcopy(j100)
    j0["traj_weight"] = 0.0
    assert only_trajectory_weight_differs(j100, j0)
    j0["batch_size"] = 256
    assert not only_trajectory_weight_differs(j100, j0)


def test_pretest_split_guard_rejects_test_archive(tmp_path):
    with pytest.raises(RuntimeError, match="test.npz"):
        guard_pretest_split(tmp_path / "test.npz")
    guard_pretest_split(tmp_path / "val.npz")


def test_j0_metrics_require_full_validation_history_and_withheld_test():
    payload = {
        "seed": 42, "traj_weight": 0.0, "best_epoch": 2,
        "history": [{}, {}], "test": None,
        "initial_model_state_sha256": "abc",
    }
    validate_metrics_payload(payload, seed=42, traj_weight=0.0, epoch_count=2)
    with pytest.raises(ValueError, match="withhold test"):
        validate_metrics_payload({**payload, "test": {"intent_auc": 0.5}}, seed=42, traj_weight=0.0, epoch_count=2)
    with pytest.raises(ValueError, match="history"):
        validate_metrics_payload(payload, seed=42, traj_weight=0.0, epoch_count=3)


def test_result_serialization_round_trips_protocol_safe_metrics(tmp_path):
    payload = {
        "seed": 42,
        "traj_weight": 0.0,
        "best_epoch": 1,
        "history": [{"epoch": 1, "val": {"intent_auc": 0.75, "trajectory_ade_pixel": 12.0}}],
        "test": None,
        "initial_model_state_sha256": "abc123",
        "test_evaluation_status": "withheld_until_protocol_freeze",
    }
    path = tmp_path / "metrics.json"
    write_json(path, payload)
    import json

    assert json.loads(path.read_text(encoding="utf-8")) == payload
