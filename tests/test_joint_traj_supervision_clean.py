from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from scripts.joint_traj_supervision_clean_utils import (
    gradient_norm,
    guard_pretest_split,
    load_config,
    normalized_clean_contract,
    only_traj_weight_differs,
    sha256_state,
    write_json,
)
from scripts.run_joint_traj_supervision_clean import build_train_command, contract_for_arm
from scripts.train_joint_transformer_gate import (
    intent_auc_selection_decision,
    shared_named_parameters,
    validation_scheduler_monitor,
)
from src.models.joint_transformer_gate import JointTransformerSceneGate


def test_intent_auc_scheduler_monitor_does_not_read_trajectory_metrics():
    # ADE/FDE/loss are intentionally absent: intent-AUC mode must not need them.
    name, value = validation_scheduler_monitor("intent_auc", {"intent_auc": 0.8123, "intent_brier": 0.11})
    assert name == "intent_auc"
    assert value == pytest.approx(0.8123)


def test_intent_auc_selection_uses_only_auc_then_brier_tie_break():
    selected, reason, best_auc = intent_auc_selection_decision(0.80, 0.20, None, None, None)
    assert selected and reason == "first_valid_checkpoint" and best_auc == pytest.approx(0.80)
    selected, reason, best_auc = intent_auc_selection_decision(0.80005, 0.10, 0.80, 0.80, 0.20)
    assert selected and reason == "auc_within_1e-4_tie_lower_brier"
    selected, reason, _ = intent_auc_selection_decision(0.8002, 0.90, best_auc, 0.80005, 0.10)
    assert selected and reason == "higher_auc_outside_tie_tolerance"
    selected, _, _ = intent_auc_selection_decision(0.7998, 0.01, 0.8002, 0.8002, 0.90)
    assert not selected  # a lower Brier cannot overcome an AUC gap > 1e-4


def test_scheduler_mode_is_explicit_and_composite_mode_remains_available():
    metrics = {"intent_auc": 0.7, "intent_f1": 0.8, "trajectory_ade_pixel": 500.0}
    assert validation_scheduler_monitor("intent_auc", metrics) == ("intent_auc", 0.7)
    name, value = validation_scheduler_monitor("composite", metrics)
    assert name == "composite_auc_f1_ade"
    assert value == pytest.approx(0.7 + 0.08 - 5.0)


def test_shared_per_seed_initial_state_loads_bit_identically():
    torch.manual_seed(5)
    model = JointTransformerSceneGate(input_dim=8, scene_dim=4, hidden_dim=16, pred_len=3, nhead=4, num_layers=1)
    initial = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    digest = sha256_state(initial)
    j0 = JointTransformerSceneGate(input_dim=8, scene_dim=4, hidden_dim=16, pred_len=3, nhead=4, num_layers=1)
    j100 = JointTransformerSceneGate(input_dim=8, scene_dim=4, hidden_dim=16, pred_len=3, nhead=4, num_layers=1)
    j0.load_state_dict(initial, strict=True)
    j100.load_state_dict(initial, strict=True)
    assert sha256_state(j0.state_dict()) == digest == sha256_state(j100.state_dict())
    maximum = max(float((j0.state_dict()[key] - j100.state_dict()[key]).abs().max()) for key in initial)
    assert maximum == 0.0


def test_clean_arm_contracts_differ_only_by_trajectory_weight():
    j0 = {"seed": 42, "selection": "intent_auc", "scheduler": "intent_auc", "initial_state": "seed42.pt", "traj_weight": 0.0}
    j100 = {**j0, "traj_weight": 100.0}
    assert only_traj_weight_differs(j0, j100)
    changed = {**j100, "scheduler": "composite"}
    assert not only_traj_weight_differs(j0, changed)


def test_runner_uses_config_audit_contract_field_names():
    source = {
        "data_root": "data/processed/jaad_sequences_scene_15x15",
        "ambiguous_root": "data/processed/jaad_ambiguous_scene_15x15",
        "epochs": 15,
        "batch_size": 512,
        "hidden_dim": 128,
        "learning_rate": 1e-3,
        "prior_weight": 0.5,
        "ambiguous_weight": 0.2,
        "gate_mode": "uncertainty",
    }
    contract = normalized_clean_contract(
        source,
        seed=123,
        traj_weight=0.0,
        initial_state_path="checkpoints/joint_traj_supervision_clean/initial_state_seed123.pt",
    )
    config_audit = {"per_seed": {"123": {"J0_clean_contract": contract}}}
    resolved_contract = contract_for_arm(config_audit, 123, "J0_clean")
    command, output_root, _ = build_train_command("J0_clean", 123, 2, smoke=True, contract=resolved_contract)
    assert "--selection-mode" in command
    assert command[command.index("--selection-mode") + 1] == "intent_auc"
    assert output_root.as_posix().endswith("results/joint_traj_supervision_clean/smoke_test/J0_clean/seed123")


def test_j0_and_j100_both_forward_trajectory_but_only_j100_adds_trajectory_gradient():
    torch.manual_seed(11)
    model = JointTransformerSceneGate(input_dim=8, scene_dim=4, hidden_dim=16, pred_len=3, nhead=4, num_layers=1, dropout=0.0)
    model.train()
    target = torch.randn(5, 4, 8)
    neighbors = torch.randn(5, 2, 4, 4)
    mask = torch.ones(5, 2)
    scene = torch.randn(5, 4)
    output = model(target, neighbors, mask, mask, scene)
    assert output["future_pred"].shape == (5, 3, 2)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
    future_gt = torch.randn(5, 3, 2)
    intent = nn.functional.binary_cross_entropy_with_logits(output["intent_logit"], labels)
    trajectory = nn.functional.smooth_l1_loss(output["future_pred"], future_gt)
    shared = shared_named_parameters(model)
    params = [parameter for _, parameter in shared]
    g0 = torch.autograd.grad(0.0 * trajectory, params, retain_graph=True, allow_unused=True)
    g100 = torch.autograd.grad(100.0 * trajectory, params, retain_graph=True, allow_unused=True)
    gi = torch.autograd.grad(intent, params, allow_unused=True)
    zero_norm, _ = gradient_norm(g0, params)
    weighted_norm, _ = gradient_norm(g100, params)
    intent_norm, _ = gradient_norm(gi, params)
    assert trajectory.detach().item() > 0
    assert zero_norm == 0.0
    assert weighted_norm > 0.0
    assert intent_norm > 0.0


def test_test_split_guard_rejects_pre_freeze_test_archive(tmp_path):
    with pytest.raises(RuntimeError, match="test.npz"):
        guard_pretest_split(tmp_path / "test.npz")
    guard_pretest_split(tmp_path / "val.npz")


def test_result_json_serialization_round_trips_selection_and_history(tmp_path):
    payload = {"seed": 123, "history": [{"epoch": 1, "selection": {"selected_checkpoint": True, "tie_break_reason": "first_valid_checkpoint"}}], "test": None}
    path = tmp_path / "metrics.json"
    write_json(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
