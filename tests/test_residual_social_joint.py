from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.residual_social_joint import ResidualSocialJointModel
from src.models.trajectory_transformer import SceneTrajectoryTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED123_CHECKPOINT = PROJECT_ROOT / "checkpoints/trajectory_transformer_scene_15x15_seed123.pt"
SEED123_TEST_DATA = PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15/test.npz"


def make_small_model(**kwargs) -> ResidualSocialJointModel:
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
    return ResidualSocialJointModel(
        input_dim=8,
        scene_dim=6,
        d_model=16,
        nhead=4,
        num_layers=1,
        pred_len=7,
        dropout=0.0,
        max_obs_len=8,
        trajectory_backbone=backbone,
        **kwargs,
    )


def make_inputs(batch=2, neighbors=3, obs_len=5):
    torch.manual_seed(19)
    target = torch.randn(batch, obs_len, 8)
    scene = torch.randn(batch, 6)
    neighbor = torch.arange(batch * neighbors * obs_len * 4, dtype=torch.float32).view(
        batch, neighbors, obs_len, 4
    )
    neighbor_mask = torch.ones(batch, neighbors)
    visible = torch.ones(batch, neighbors, obs_len)
    return target, scene, neighbor, neighbor_mask, visible


def make_pretrained_backbone(checkpoint_path: Path) -> tuple[SceneTrajectoryTransformer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    model = SceneTrajectoryTransformer(
        input_dim=8,
        scene_dim=512,
        d_model=int(saved_args.get("d_model", 128)),
        nhead=4,
        num_layers=int(saved_args.get("num_layers", 3)),
        pred_len=15,
        dropout=0.1,
        max_obs_len=15,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def test_pretrained_backbone_reproduces_seed123_prediction():
    if not SEED123_CHECKPOINT.is_file() or not SEED123_TEST_DATA.is_file():
        pytest.skip("local seed123 Transformer checkpoint/data are unavailable")
    backbone, _ = make_pretrained_backbone(SEED123_CHECKPOINT)
    arrays = np.load(SEED123_TEST_DATA, allow_pickle=False)
    target = torch.from_numpy(
        np.concatenate(
            [arrays["target_obs"][:2], arrays["target_abs_obs"][:2]], axis=-1
        ).astype(np.float32)
    )
    scene = torch.from_numpy(arrays["scene_feat"][:2].astype(np.float32))
    wrapper_backbone, checkpoint = make_pretrained_backbone(SEED123_CHECKPOINT)
    saved_args = checkpoint.get("args", {})
    wrapper = ResidualSocialJointModel(
        input_dim=8,
        scene_dim=512,
        d_model=int(saved_args.get("d_model", 128)),
        num_layers=int(saved_args.get("num_layers", 3)),
        pred_len=15,
        max_obs_len=15,
        gate_mode="none",
        enable_trajectory_residual=False,
        trajectory_backbone=wrapper_backbone,
    ).eval()
    neighbors = torch.zeros(2, 8, 15, 4)
    neighbor_mask = torch.zeros(2, 8)
    visible = torch.zeros(2, 8, 15)
    with torch.no_grad():
        expected = backbone(target, scene)
        output = wrapper(target, neighbors, neighbor_mask, visible, scene)
    torch.testing.assert_close(output["future_pred"], expected, rtol=0.0, atol=0.0)
    assert wrapper.trajectory_backbone_frozen


def test_none_gate_makes_trajectory_independent_of_neighbors():
    model = make_small_model(gate_mode="none", enable_trajectory_residual=True).eval()
    target, scene, neighbor_a, mask, visible = make_inputs()
    neighbor_b = neighbor_a.flip(dims=(1, 2, 3)) * -3.0
    with torch.no_grad():
        output_a = model(target, neighbor_a, mask, visible, scene)
        output_b = model(target, neighbor_b, mask, visible, scene)
    torch.testing.assert_close(output_a["future_pred"], output_b["future_pred"], rtol=0.0, atol=0.0)


def test_zero_initialized_trajectory_residual_preserves_base_prediction():
    model = make_small_model(gate_mode="always", enable_trajectory_residual=True).eval()
    target, scene, neighbor, mask, visible = make_inputs()
    final_layer = model.social_trajectory_residual_head[-1]
    assert torch.count_nonzero(final_layer.weight) == 0
    assert torch.count_nonzero(final_layer.bias) == 0
    with torch.no_grad():
        output = model(target, neighbor, mask, visible, scene)
    torch.testing.assert_close(output["future_pred"], output["base_future_pred"], rtol=0.0, atol=0.0)


def test_uncertainty_gate_is_monotonic_in_entropy():
    model = make_small_model(gate_mode="uncertainty")
    entropy = torch.tensor([0.05, 0.2, 0.5, 0.9])
    gate = model.uncertainty_to_gate(entropy)
    assert model.gate_slope.item() > 0.0
    assert 0.0 < model.gate_threshold.item() < 1.0
    assert torch.all(gate[1:] >= gate[:-1])


def test_prior_entropy_and_uncertainty_gate_do_not_depend_on_social_context():
    model = make_small_model(gate_mode="uncertainty").eval()
    target, scene, neighbor_a, mask, visible = make_inputs()
    neighbor_b = torch.randn_like(neighbor_a) * 100.0
    with torch.no_grad():
        output_a = model(target, neighbor_a, mask, visible, scene)
        output_b = model(target, neighbor_b, mask, visible, scene)
    torch.testing.assert_close(output_a["prior_logit"], output_b["prior_logit"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        output_a["normalized_entropy"], output_b["normalized_entropy"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(output_a["gate"], output_b["gate"], rtol=0.0, atol=0.0)


def test_neighbor_gru_receives_batch_neighbor_time_feature_order():
    model = make_small_model(gate_mode="always").eval()
    target, scene, neighbor, mask, visible = make_inputs(batch=2, neighbors=3, obs_len=5)
    captured = {}

    def inspect_input(_module, inputs):
        captured["input"] = inputs[0].detach().clone()

    hook = model.neighbor_encoder.register_forward_pre_hook(inspect_input)
    with torch.no_grad():
        model(target, neighbor, mask, visible, scene)
    hook.remove()
    expected = neighbor.reshape(2 * 3, 5, 4)
    assert captured["input"].shape == (6, 5, 4)
    torch.testing.assert_close(captured["input"], expected, rtol=0.0, atol=0.0)


def test_backbone_is_frozen_and_stays_in_eval_mode_during_training():
    model = make_small_model()
    model.train()
    assert model.trajectory_backbone_frozen
    assert not model.trajectory_backbone.training
    assert all(not parameter.requires_grad for parameter in model.trajectory_backbone.parameters())
