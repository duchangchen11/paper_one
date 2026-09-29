from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts import run_scene_locality_audit as audit
from scripts import train_joint_transformer_gate as trainer


def test_scene_id_maps_to_same_video_stem(tmp_path):
    (tmp_path / "video_0171.mp4").touch()
    assert audit.resolve_video("video_0171", tmp_path) == tmp_path / "video_0171.mp4"


def test_frame_index_bounds_are_zero_based():
    assert audit.frame_in_bounds(0, 10)
    assert audit.frame_in_bounds(9, 10)
    assert not audit.frame_in_bounds(-1, 10)
    assert not audit.frame_in_bounds(10, 10)


def test_bbox_expansion_is_centered_and_scaled():
    roi = audit.expanded_bbox(np.asarray([0.5, 0.5, 0.2, 0.1]), 100, 80, 4)
    assert roi == (10, 24, 90, 56)


def test_bbox_expansion_clips_at_image_boundary():
    roi = audit.expanded_bbox(np.asarray([0.05, 0.5, 0.2, 0.2]), 100, 100, 4)
    assert roi[0] == 0
    assert 0 <= roi[1] < roi[3] <= 100


def test_local_and_background_masks_are_complementary():
    rgb = np.full((8, 10, 3), 200, dtype=np.uint8)
    roi = (2, 1, 7, 6)
    local, background = audit.mask_variants(rgb, roi)
    x1, y1, x2, y2 = roi
    assert np.array_equal(local[y1:y2, x1:x2], rgb[y1:y2, x1:x2])
    assert np.all(local[:y1] == audit.FILL_RGB)
    assert np.all(background[y1:y2, x1:x2] == audit.FILL_RGB)
    assert np.array_equal(background[:y1], rgb[:y1])


def test_r0_r1_r2_image_shapes_match():
    rgb = np.zeros((17, 23, 3), dtype=np.uint8)
    local, background = audit.mask_variants(rgb, (3, 4, 20, 12))
    assert rgb.shape == local.shape == background.shape


def test_masking_and_encoding_are_deterministic():
    rgb = np.arange(12 * 14 * 3, dtype=np.uint8).reshape(12, 14, 3)
    first = audit.mask_variants(rgb, (2, 2, 10, 9))
    second = audit.mask_variants(rgb, (2, 2, 10, 9))
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 512, bias=False))
    encoder.eval()
    tensor = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(3, 4, 4)
    a = audit.encode_batch(encoder, [tensor], torch.device("cpu"))
    b = audit.encode_batch(encoder, [tensor], torch.device("cpu"))
    assert np.array_equal(a, b)


def test_frozen_feature_extractor_contract_is_512d():
    class Dummy(torch.nn.Module):
        def forward(self, value):
            return torch.ones((value.shape[0], 512), device=value.device)

    result = audit.encode_batch(Dummy(), [torch.zeros(3, 4, 4), torch.ones(3, 4, 4)], torch.device("cpu"))
    audit.validate_features(result, 2)
    assert result.shape == (2, 512)


def test_test_split_is_rejected_before_archive_open(monkeypatch, tmp_path):
    def forbidden_load(*args, **kwargs):
        raise AssertionError("np.load must not be called for test")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(ValueError, match="Only train and val"):
        audit.load_split(tmp_path, "test")
    with pytest.raises(ValueError, match="restricted to train and val"):
        audit.extract_split(
            "test", {}, tmp_path, tmp_path, torch.nn.Identity(), lambda x: x,
            torch.device("cpu"),
        )
    assert trainer.requested_split_names(skip_test=True) == ("train", "val")


def test_trainer_scene_override_changes_only_scene_tensor(tmp_path):
    class DummyDataset:
        scene_feat = torch.zeros((2, 512), dtype=torch.float32)
        intent_label = torch.tensor([0.0, 1.0])

    dataset = DummyDataset()
    expected = np.ones((2, 512), dtype=np.float32)
    path = tmp_path / "features.npz"
    np.savez(path, local=expected)
    digest = trainer.apply_scene_feature_override(dataset, path, "local")
    assert torch.equal(dataset.scene_feat, torch.from_numpy(expected))
    assert torch.equal(dataset.intent_label, torch.tensor([0.0, 1.0]))
    assert len(digest) == 64


def test_trainer_rejects_scene_feature_shape_mismatch(tmp_path):
    class DummyDataset:
        scene_feat = torch.zeros((2, 512), dtype=torch.float32)

    path = tmp_path / "bad_features.npz"
    np.savez(path, local=np.zeros((2, 256), dtype=np.float32))
    with pytest.raises(ValueError, match="shape mismatch"):
        trainer.apply_scene_feature_override(DummyDataset(), path, "local")


def test_matched_initialization_report_is_exact(tmp_path):
    state = {"weight": torch.tensor([1.0, -2.0]), "bias": torch.tensor([0.5])}
    for seed in audit.SEEDS:
        torch.save({"model": state, "seed": seed, "sha256": "seed-state"}, tmp_path / f"initial_state_seed{seed}.pt")
    report = audit.initialization_report(tmp_path)
    assert report["all_arms_exactly_matched_within_seed"]
    assert report["max_abs_parameter_difference"] == 0


def test_sampler_fingerprints_match_for_all_arms():
    metrics = {
        arm: {
            str(seed): {
                "history": [
                    {"train": {"sampler_sha256": f"e{epoch}", "first_sample_indices": [1, 4, 2]}}
                    for epoch in range(1, 4)
                ]
            }
            for seed in audit.SEEDS
        }
        for arm in audit.ARMS
    }
    assert audit._verify_sampler(metrics)["all_matched"]
    metrics["B2"]["42"]["history"][0]["train"]["sampler_sha256"] = "different"
    assert not audit._verify_sampler(metrics)["all_matched"]
