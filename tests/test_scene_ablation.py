from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_trajectory_transformer import SequenceWithImageSize, apply_scene_mode
from src.models.scene_ablation_intent import SceneAblationIntentModel
from src.models.trajectory_transformer import SceneTrajectoryTransformer


def _backbone() -> SceneTrajectoryTransformer:
    torch.manual_seed(321)
    return SceneTrajectoryTransformer(
        input_dim=8,
        scene_dim=6,
        d_model=16,
        nhead=4,
        num_layers=1,
        pred_len=5,
        dropout=0.0,
        max_obs_len=4,
    )


def _paired_models() -> tuple[SceneAblationIntentModel, SceneAblationIntentModel]:
    base = _backbone()
    target_only = SceneAblationIntentModel(copy.deepcopy(base), scene_mode="target_only", dropout=0.0)
    target_scene = SceneAblationIntentModel(copy.deepcopy(base), scene_mode="target_scene", dropout=0.0)
    target_only.eval()
    target_scene.eval()
    return target_only, target_scene


def test_intent_readouts_have_same_frozen_backbone_and_parameter_budget():
    target_only, target_scene = _paired_models()
    only_state = target_only.trajectory_backbone.state_dict()
    scene_state = target_scene.trajectory_backbone.state_dict()
    assert only_state.keys() == scene_state.keys()
    assert all(torch.equal(only_state[key], scene_state[key]) for key in only_state)
    assert target_only.trajectory_backbone_frozen
    assert target_scene.trajectory_backbone_frozen
    assert all(not p.requires_grad for p in target_only.trajectory_backbone.parameters())
    assert all(not p.requires_grad for p in target_scene.trajectory_backbone.parameters())
    assert target_only.base_fusion[0].in_features == target_scene.base_fusion[0].in_features
    assert sum(p.numel() for p in target_only.parameters()) == sum(
        p.numel() for p in target_scene.parameters()
    )


def test_target_only_intent_logit_is_invariant_to_scene_features():
    target_only, _ = _paired_models()
    target = torch.randn(3, 4, 8)
    scene_a = torch.randn(3, 6)
    scene_b = torch.randn(3, 6) * 100.0
    logit_a = target_only(target, scene_a)["base_raw_logit"]
    logit_b = target_only(target, scene_b)["base_raw_logit"]
    torch.testing.assert_close(logit_a, logit_b, rtol=0, atol=0)


def test_target_scene_readout_can_change_with_scene_features():
    _, target_scene = _paired_models()
    target = torch.randn(3, 4, 8)
    scene_a = torch.randn(3, 6)
    scene_b = scene_a + 5.0
    logit_a = target_scene(target, scene_a)["base_raw_logit"]
    logit_b = target_scene(target, scene_b)["base_raw_logit"]
    assert not torch.allclose(logit_a, logit_b)


def test_paired_readouts_have_identical_future_predictions():
    target_only, target_scene = _paired_models()
    target = torch.randn(3, 4, 8)
    scene = torch.randn(3, 6)
    pred_only = target_only(target, scene)["future_pred"]
    pred_scene = target_scene(target, scene)["future_pred"]
    torch.testing.assert_close(pred_only, pred_scene, rtol=0, atol=0)


def test_zero_scene_trajectory_mode_ignores_real_scene_features():
    backbone = _backbone().eval()
    assert hasattr(backbone, "scene_encoder")
    assert sum(p.numel() for p in backbone.parameters()) > sum(
        p.numel() for p in backbone.scene_encoder.parameters()
    )
    target = torch.randn(2, 4, 8)
    scene_a = torch.randn(2, 6)
    scene_b = torch.randn(2, 6) * 20
    pred_a = backbone(target, apply_scene_mode(scene_a, "zero"))
    pred_b = backbone(target, apply_scene_mode(scene_b, "zero"))
    torch.testing.assert_close(pred_a, pred_b, rtol=0, atol=0)


def test_shuffled_loader_keeps_image_size_attached_to_sample(tmp_path):
    count = 3
    scene = np.zeros((count, 6), dtype=np.float32)
    scene[:, 0] = np.arange(count)
    image_size = np.zeros((count, 2), dtype=np.float32)
    image_size[:, 0] = 100 + np.arange(count)
    path = tmp_path / "tiny_sequences.npz"
    np.savez(
        path,
        target_obs=np.zeros((count, 4, 4), dtype=np.float32),
        target_abs_obs=np.zeros((count, 4, 4), dtype=np.float32),
        future_gt=np.zeros((count, 5, 2), dtype=np.float32),
        neighbor_obs=np.zeros((count, 0, 4, 4), dtype=np.float32),
        neighbor_mask=np.zeros((count, 0), dtype=np.float32),
        neighbor_visible_mask=np.zeros((count, 0), dtype=np.float32),
        intent_label=np.zeros(count, dtype=np.float32),
        scene_feat=scene,
        image_size=image_size,
    )
    dataset = SequenceWithImageSize(path)
    shuffled = DataLoader(dataset, batch_size=3, sampler=[2, 0, 1])
    batch = next(iter(shuffled))
    sample_ids = batch["scene_feat"][:, 0]
    paired_widths = batch["image_size"][:, 0]
    torch.testing.assert_close(sample_ids, torch.tensor([2.0, 0.0, 1.0]))
    torch.testing.assert_close(paired_widths, torch.tensor([102.0, 100.0, 101.0]))
