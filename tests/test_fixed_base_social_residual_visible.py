from __future__ import annotations

import torch
import pytest

from src.models.fixed_base_social_residual import FixedBaseIntentModel
from src.models.fixed_base_social_residual_visible import FixedBaseSocialResidualVisible
from src.models.trajectory_transformer import SceneTrajectoryTransformer


def make_model(gate_mode: str = "always") -> FixedBaseSocialResidualVisible:
    backbone = SceneTrajectoryTransformer(
        input_dim=8,
        scene_dim=6,
        d_model=16,
        nhead=4,
        num_layers=1,
        pred_len=7,
        dropout=0.0,
        max_obs_len=8,
    )
    base = FixedBaseIntentModel(backbone, dropout=0.0)
    return FixedBaseSocialResidualVisible(base, gate_mode=gate_mode, dropout=0.0)


def make_inputs(batch: int = 2, neighbors: int = 3, obs_len: int = 5):
    torch.manual_seed(47)
    target = torch.randn(batch, obs_len, 8)
    scene = torch.randn(batch, 6)
    neighbors_tensor = torch.randn(batch, neighbors, obs_len, 4)
    neighbor_mask = torch.ones(batch, neighbors)
    visible = torch.ones(batch, neighbors, obs_len)
    visible[:, :, 1] = 0
    return target, scene, neighbors_tensor, neighbor_mask, visible


def test_n3_t5_visibility_features_keep_batch_neighbor_time_feature_order():
    batch, neighbors_count, obs_len = 2, 3, 5
    model = make_model().eval()
    target, scene, neighbors, neighbor_mask, visible = make_inputs(
        batch, neighbors_count, obs_len
    )
    captured = {}

    def inspect_gru_input(_module, inputs):
        captured["input"] = inputs[0].detach().clone()

    hook = model.neighbor_encoder.register_forward_pre_hook(inspect_gru_input)
    with torch.no_grad():
        model(target, scene, neighbors, neighbor_mask, visible)
    hook.remove()

    expected = torch.cat(
        [neighbors * visible.unsqueeze(-1), visible.unsqueeze(-1)], dim=-1
    ).reshape(batch * neighbors_count, obs_len, 5)
    assert captured["input"].shape == (batch * neighbors_count, obs_len, 5)
    torch.testing.assert_close(captured["input"], expected, rtol=0, atol=0)


def test_changing_invisible_raw_frame_does_not_change_social_representation():
    model = make_model().eval()
    target, scene, neighbors, neighbor_mask, visible = make_inputs()
    changed = neighbors.clone()
    changed[0, 1, 1] = torch.tensor([1000.0, -800.0, 600.0, -400.0])
    with torch.no_grad():
        original_output = model(target, scene, neighbors, neighbor_mask, visible)
        changed_output = model(target, scene, changed, neighbor_mask, visible)
    torch.testing.assert_close(
        original_output["social_context"], changed_output["social_context"], rtol=0, atol=0
    )


def test_changing_visible_frame_can_change_social_representation():
    model = make_model().eval()
    target, scene, neighbors, neighbor_mask, visible = make_inputs()
    changed = neighbors.clone()
    changed[0, 1, 2] += torch.tensor([20.0, -30.0, 40.0, -50.0])
    with torch.no_grad():
        original_output = model(target, scene, neighbors, neighbor_mask, visible)
        changed_output = model(target, scene, changed, neighbor_mask, visible)
    assert not torch.equal(original_output["social_context"], changed_output["social_context"])


def test_zero_neighbors_leave_final_logit_equal_to_base():
    model = make_model("always").eval()
    model.social_head[-1].bias.data.fill_(3.0)
    target, scene, neighbors, _, visible = make_inputs()
    neighbor_mask = torch.zeros(2, 3)
    with torch.no_grad():
        output = model(target, scene, neighbors, neighbor_mask, visible)
    assert torch.count_nonzero(output["effective_gate"]) == 0
    torch.testing.assert_close(output["final_logit"], output["base_logit"], rtol=0, atol=0)


def test_base_predictions_are_independent_of_all_visibility_social_inputs():
    model = make_model("uncertainty").eval()
    target, scene, neighbors, neighbor_mask, visible = make_inputs()
    changed_neighbors = torch.randn_like(neighbors) * 100
    changed_visible = torch.zeros_like(visible)
    changed_mask = torch.zeros_like(neighbor_mask)
    with torch.no_grad():
        first = model(target, scene, neighbors, neighbor_mask, visible)
        second = model(target, scene, changed_neighbors, changed_mask, changed_visible)
    for key in ("base_logit", "base_probability", "base_entropy", "future_pred"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_invalid_neighbor_mask_shapes_raise_value_error():
    model = make_model().eval()
    target, scene, neighbors, neighbor_mask, visible = make_inputs()
    with torch.no_grad(), pytest.raises(ValueError, match="neighbor_visible_mask"):
        model(target, scene, neighbors, neighbor_mask, visible[:, :, :-1])
    with torch.no_grad(), pytest.raises(ValueError, match="neighbor_mask"):
        model(target, scene, neighbors, neighbor_mask[:, :-1], visible)


def test_transformer_and_base_classifier_stay_frozen_in_training_mode():
    model = make_model("uncertainty")
    model.train()
    assert model.all_base_parameters_frozen
    assert not model.base_model.training
    assert not model.base_model.trajectory_backbone.training
    assert all(not p.requires_grad for p in model.base_model.parameters())
    assert all(p.requires_grad for p in model.neighbor_encoder.parameters())
    assert all(p.requires_grad for p in model.social_head.parameters())
