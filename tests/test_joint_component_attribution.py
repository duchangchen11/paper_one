from __future__ import annotations

import torch
from torch import nn

from scripts.run_joint_component_attribution import legacy_forward
from scripts.train_joint_transformer_gate import (
    component_loss_weights,
    intent_auc_selection_decision,
    requested_split_names,
    validation_scheduler_monitor,
)
from src.models.joint_transformer_gate import JointTransformerSceneGate


def make_inputs(batch: int = 3):
    generator = torch.Generator().manual_seed(781)
    return (
        torch.randn(batch, 5, 8, generator=generator),
        torch.randn(batch, 2, 5, 4, generator=generator),
        torch.ones(batch, 2),
        torch.ones(batch, 2, 5),
        torch.randn(batch, 12, generator=generator),
    )


def make_model(**kwargs):
    return JointTransformerSceneGate(
        input_dim=8,
        scene_dim=12,
        hidden_dim=16,
        pred_len=3,
        nhead=4,
        num_layers=1,
        dropout=0.0,
        **kwargs,
    )


def test_all_enabled_matches_pre_ablation_forward_and_state_dict():
    torch.manual_seed(13)
    original = make_model()
    all_enabled = make_model(component_flags_all_enabled=True)
    all_enabled.load_state_dict(original.state_dict(), strict=True)
    original.eval()
    all_enabled.eval()
    inputs = make_inputs()
    with torch.inference_mode():
        expected_intent, expected_future = legacy_forward(original, *inputs)
        actual = all_enabled(*inputs)
    assert original.state_dict().keys() == all_enabled.state_dict().keys()
    assert torch.equal(expected_intent, actual["intent_logit"])
    assert torch.equal(expected_future, actual["future_pred"])


def test_no_scene_zeros_every_downstream_scene_context():
    model = make_model(component_flags_all_enabled=False, component_ablation="no_scene").eval()
    with torch.inference_mode():
        output = model(*make_inputs())
    assert torch.count_nonzero(output["effective_scene_context"]) == 0
    assert torch.isfinite(output["intent_logit"]).all()


def test_no_social_zeros_every_downstream_social_context():
    model = make_model(component_flags_all_enabled=False, component_ablation="no_social").eval()
    with torch.inference_mode():
        output = model(*make_inputs())
    assert torch.count_nonzero(output["effective_social_context"]) == 0
    assert torch.isfinite(output["intent_logit"]).all()


def test_no_proposal_loss_keeps_proposal_forward():
    model = make_model(component_flags_all_enabled=False, component_ablation="no_proposal_loss").eval()
    with torch.inference_mode():
        output = model(*make_inputs())
    assert "prior_logit" in output
    assert torch.isfinite(output["prior_logit"]).all()


def test_no_proposal_loss_sets_weighted_proposal_loss_to_zero():
    model = make_model(component_flags_all_enabled=False, component_ablation="no_proposal_loss").eval()
    prior_weight, _ = component_loss_weights("no_proposal_loss", 0.5, 0.2)
    labels = torch.tensor([0.0, 1.0, 0.0])
    with torch.inference_mode():
        output = model(*make_inputs())
        raw_loss = nn.functional.binary_cross_entropy_with_logits(output["prior_logit"], labels)
    assert prior_weight == 0.0
    assert prior_weight * raw_loss == 0.0
    assert hasattr(model, "proposal_fusion") and hasattr(model, "proposal_head")


def test_no_adaptive_gate_is_fixed_neutral():
    model = make_model(component_flags_all_enabled=False, component_ablation="no_adaptive_gate").eval()
    with torch.inference_mode():
        output = model(*make_inputs())
    assert torch.equal(output["gate"], torch.full_like(output["gate"], 0.5))


def test_no_adaptive_gate_keeps_both_fusion_branches():
    model = make_model(component_flags_all_enabled=False, component_ablation="no_adaptive_gate").eval()
    with torch.inference_mode():
        output = model(*make_inputs())
    assert output["effective_scene_context"].abs().sum() > 0
    assert output["effective_social_context"].abs().sum() > 0
    assert hasattr(model, "gate") and hasattr(model, "fusion")


def test_no_ambiguity_sets_only_ambiguity_weight_to_zero():
    prior_weight, ambiguous_weight = component_loss_weights("no_ambiguity", 0.5, 0.2)
    assert prior_weight == 0.5
    assert ambiguous_weight == 0.0
    assert component_loss_weights("full", 0.5, 0.2) == (0.5, 0.2)


def test_checkpoint_selection_uses_auc_with_brier_tie_break_only():
    selected, reason, _ = intent_auc_selection_decision(0.81, 0.18, None, None, None)
    assert selected
    assert reason == "first_valid_checkpoint"


def test_scheduler_monitors_validation_auc_only():
    metric = {"intent_auc": 0.81, "intent_brier": 0.18, "trajectory_ade_pixel": 9999.0}
    name, value = validation_scheduler_monitor("intent_auc", metric)
    assert (name, value) == ("intent_auc", 0.81)


def test_training_only_phase_does_not_request_test_split():
    assert requested_split_names(skip_test=True) == ("train", "val")
    assert "test" not in requested_split_names(skip_test=True)


def test_variant_state_serialization_round_trips_without_architecture_change(tmp_path):
    model = make_model(component_flags_all_enabled=False, component_ablation="no_scene")
    path = tmp_path / "component.pt"
    torch.save({"model": model.state_dict(), "args": {"component_ablation": "no_scene"}}, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = make_model(component_flags_all_enabled=False, component_ablation="no_scene")
    restored.load_state_dict(payload["model"], strict=True)
    assert payload["args"]["component_ablation"] == "no_scene"
    assert payload["model"].keys() == restored.state_dict().keys()
