import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.evaluate_trajectory_reliability_deconfounding import cluster_bootstrap_indices
from scripts.run_reliability_internal_replication import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PROTOCOL_SHA256,
    EXPECTED_TRAIN_NPZ_SHA256,
    MODEL_CONFIG,
    PRIMARY_SCORE,
    RANDOM_SEED,
    SCENE_MODE,
    VideoSubsetDataset,
    apply_motion_adjustment,
    begin_holdout_recovery,
    frozen_recovery_parameters,
    _fit_internal_val_adjustment,
    _decision_from_holdout,
    freeze_val_thresholds,
    indices_for_split,
    make_manifest_payload,
    ordered_sample_metadata_sha256,
    select_best_epoch,
    split_video_ids,
    validate_holdout_access_state,
    validate_recovery_hashes,
    write_json,
)


class TinyDataset(Dataset):
    def __init__(self, scene_ids):
        self.scene_ids = np.asarray(scene_ids).astype(str)

    def __len__(self):
        return len(self.scene_ids)

    def __getitem__(self, index):
        return {"index": torch.tensor(index)}


def test_video_split_has_no_overlap_and_covers_every_video():
    scene_ids = np.array([f"video_{index:03d}" for index in range(144) for _ in range(index % 4 + 1)])
    splits = split_video_ids(scene_ids)

    split_sets = {name: set(values) for name, values in splits.items()}
    assert split_sets["internal_train"].isdisjoint(split_sets["internal_val"])
    assert split_sets["internal_train"].isdisjoint(split_sets["internal_holdout"])
    assert split_sets["internal_val"].isdisjoint(split_sets["internal_holdout"])
    assert set.union(*split_sets.values()) == set(np.unique(scene_ids))
    assert [len(splits[name]) for name in ("internal_train", "internal_val", "internal_holdout")] == [101, 22, 21]


def test_fixed_seed_makes_manifest_identical():
    scene_ids = np.repeat(np.array(["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"]), 2)
    target_ids = np.tile(np.array(["p1", "p2"]), 10)

    manifest_a = make_manifest_payload(scene_ids, target_ids, "train-hash", seed=RANDOM_SEED)
    manifest_b = make_manifest_payload(scene_ids, target_ids, "train-hash", seed=RANDOM_SEED)

    assert manifest_a == manifest_b


def test_sample_indices_match_their_manifest_scene_ids():
    scene_ids = np.array(["v2", "v1", "v3", "v2", "v4", "v1"])
    allowed = ["v1", "v4"]
    indices = indices_for_split(scene_ids, allowed)

    assert set(scene_ids[indices]) == set(allowed)
    assert all(scene_ids[index] in allowed for index in indices)
    with pytest.raises(ValueError, match="absent"):
        indices_for_split(scene_ids, ["not-in-archive"])


def test_subset_loader_contains_only_its_assigned_videos():
    base = TinyDataset(["train-a", "val-a", "holdout-a", "train-b", "val-b"])
    indices = np.array([0, 3])
    subset = VideoSubsetDataset(base, indices, ["train-a", "train-b"])
    seen_indices = torch.cat(
        [batch["index"] for batch in DataLoader(subset, batch_size=2, shuffle=False)]
    ).tolist()

    assert set(subset.scene_ids) == {"train-a", "train-b"}
    assert seen_indices == [0, 3]


def test_checkpoint_selection_uses_only_internal_val_pixel_ade():
    history = [
        {"epoch": 1, "internal_val": {"trajectory_ade_pixel": 12.0}, "internal_holdout": {"trajectory_ade_pixel": 0.01}},
        {"epoch": 2, "internal_val": {"trajectory_ade_pixel": 10.0}, "internal_holdout": {"trajectory_ade_pixel": 9999.0}},
        {"epoch": 3, "internal_val": {"trajectory_ade_pixel": 11.0}, "internal_holdout": {"trajectory_ade_pixel": 0.0}},
    ]

    assert select_best_epoch(history) == 2


def test_high_error_thresholds_are_fixed_from_internal_val():
    val_ade = np.arange(100, dtype=float)
    val_fde = 2 * np.arange(100, dtype=float)

    thresholds = freeze_val_thresholds(val_ade, val_fde)

    assert thresholds["source_split"] == "internal_val"
    assert thresholds["high_ade_pixel_threshold"] == pytest.approx(np.quantile(val_ade, 0.8))
    assert thresholds["high_fde_pixel_threshold"] == pytest.approx(np.quantile(val_fde, 0.8))
    assert thresholds["holdout_top20_used_for_primary_label"] is False


def test_adjustment_fit_uses_only_internal_val_motion_and_u():
    val_motion = np.linspace(0, 100, 40)
    val_u = 0.5 + 0.03 * val_motion + 0.0001 * val_motion**2

    coefficients = _fit_internal_val_adjustment(val_motion, val_u)

    assert coefficients["fit_split"] == "internal_val"
    assert coefficients["ADE_or_FDE_used"] is False
    assert coefficients["holdout_used"] is False


def test_holdout_uses_frozen_adjustment_without_refitting():
    coefficients = {
        "b0": 0.2,
        "b1": 0.3,
        "b2": -0.04,
        "fit_split": "internal_val",
        "ADE_or_FDE_used": False,
        "future_gt_used": False,
        "holdout_used": False,
    }
    motion_holdout = np.array([2.0, 8.0, 18.0])
    u_holdout = np.array([0.7, 1.3, 2.9])
    frozen = {key: coefficients[key] for key in ("b0", "b1", "b2")}

    adjusted = apply_motion_adjustment(motion_holdout, u_holdout, coefficients)
    x = np.log1p(motion_holdout)
    expected = np.log1p(u_holdout) - (0.2 + 0.3 * x - 0.04 * x**2)

    np.testing.assert_allclose(adjusted, expected)
    assert {key: coefficients[key] for key in frozen} == frozen
    assert coefficients["fit_split"] == "internal_val"


def test_primary_score_and_training_protocol_are_fixed():
    assert PRIMARY_SCORE == "u_mean"
    assert SCENE_MODE == "zero"
    assert MODEL_CONFIG["input_dim"] == 8
    assert MODEL_CONFIG["d_model"] == 128
    assert MODEL_CONFIG["num_layers"] == 3
    assert MODEL_CONFIG["nhead"] == 4
    assert MODEL_CONFIG["epochs"] == 20


def test_cluster_bootstrap_keeps_all_samples_from_a_video_together():
    cluster_ids = np.array(["video-a", "video-a", "video-b", "video-c", "video-c", "video-c"])
    sampled_indices = cluster_bootstrap_indices(cluster_ids, np.random.default_rng(9124))
    multiplicities = np.bincount(sampled_indices, minlength=len(cluster_ids))

    for cluster in np.unique(cluster_ids):
        sample_weights = multiplicities[cluster_ids == cluster]
        assert np.all(sample_weights == sample_weights[0])


def test_started_marker_without_final_outputs_allows_only_explicit_recovery():
    record = {"phase": "PHASE D holdout access started", "started_at_utc": "t0"}
    outputs = {
        "internal_holdout_reliability_exists": False,
        "decision_exists": False,
        "cluster_bootstrap_exists": False,
        "risk_coverage_exists": False,
    }

    validate_holdout_access_state(record, outputs, recovery_requested=True)
    with pytest.raises(RuntimeError, match="already started"):
        validate_holdout_access_state(record, outputs, recovery_requested=False)


def test_recovery_rejects_a_changed_protocol_sha():
    with pytest.raises(RuntimeError, match="protocol SHA256"):
        validate_recovery_hashes(
            "wrong", EXPECTED_MANIFEST_SHA256, EXPECTED_TRAIN_NPZ_SHA256, EXPECTED_CHECKPOINT_SHA256
        )


def test_recovery_rejects_a_changed_manifest_sha():
    with pytest.raises(RuntimeError, match="manifest SHA256"):
        validate_recovery_hashes(
            EXPECTED_PROTOCOL_SHA256, "wrong", EXPECTED_TRAIN_NPZ_SHA256, EXPECTED_CHECKPOINT_SHA256
        )


def test_recovery_rejects_changed_checkpoint_hashes():
    changed = {**EXPECTED_CHECKPOINT_SHA256, "42": "wrong"}
    with pytest.raises(RuntimeError, match="Checkpoint SHA256"):
        validate_recovery_hashes(
            EXPECTED_PROTOCOL_SHA256, EXPECTED_MANIFEST_SHA256, EXPECTED_TRAIN_NPZ_SHA256, changed
        )


def test_completed_decision_cannot_be_recovered_or_rerun():
    record = {"phase": "PHASE D holdout evaluation completed after crash recovery", "status": "completed"}
    outputs = {
        "internal_holdout_reliability_exists": True,
        "decision_exists": True,
        "cluster_bootstrap_exists": True,
        "risk_coverage_exists": True,
    }

    with pytest.raises(RuntimeError, match="immutable"):
        validate_holdout_access_state(record, outputs, recovery_requested=True)


def test_recovery_audit_does_not_call_split_training_or_refitting(tmp_path, monkeypatch):
    import scripts.run_reliability_internal_replication as replication

    def forbidden(*_args, **_kwargs):
        pytest.fail("Recovery audit must not regenerate data splits, train, or refit protocol")

    for name in (
        "split_video_ids", "prepare_manifest", "train_all",
        "freeze_val_thresholds", "_fit_internal_val_adjustment",
    ):
        monkeypatch.setattr(replication, name, forbidden)
    access = {"phase": "PHASE D holdout access started", "started_at_utc": "initial"}
    outputs = {
        "internal_holdout_reliability_exists": False,
        "decision_exists": False,
        "cluster_bootstrap_exists": False,
        "risk_coverage_exists": False,
    }

    audit, updated = begin_holdout_recovery(
        tmp_path, access, EXPECTED_PROTOCOL_SHA256, EXPECTED_MANIFEST_SHA256,
        EXPECTED_TRAIN_NPZ_SHA256, EXPECTED_CHECKPOINT_SHA256, outputs,
    )

    assert audit["recovery_authorized"] is True
    assert audit["existing_access_record"] == access
    assert audit["final_outputs_present_before_recovery"] == outputs
    assert audit["recovery_reason"] == "interrupted computation after holdout access marker was written"
    assert audit["protocol_changed"] is False
    assert audit["checkpoint_changed"] is False
    assert audit["split_changed"] is False
    assert audit["threshold_changed"] is False
    assert audit["adjustment_changed"] is False
    assert audit["score_changed"] is False
    assert audit["decision_rule_changed"] is False
    assert updated["initial_started_at_utc"] == "initial"
    assert updated["status"] == "running"


def test_recovery_uses_frozen_thresholds_and_motion_coefficients_verbatim():
    frozen = {
        "high_error_thresholds": {"high_ade_pixel_threshold": 14.960247728473131,
                                  "high_fde_pixel_threshold": 26.810941923553397},
        "motion_adjustment": {"b0": 1.425159516871413, "b1": -0.26737580192935345,
                              "b2": 0.0574144343900067, "fit_split": "internal_val"},
        "motion_stratification_cutpoints": {"slow_medium_fast_q33_q67_pixel": [18.248287200927734, 58.66926066080728]},
    }
    thresholds, coefficients, cutpoints = frozen_recovery_parameters(frozen)
    assert thresholds is frozen["high_error_thresholds"]
    assert coefficients is frozen["motion_adjustment"]
    assert cutpoints is frozen["motion_stratification_cutpoints"]
    values = apply_motion_adjustment(np.array([10.0]), np.array([2.0]), coefficients)
    expected = np.log1p(2.0) - (
        coefficients["b0"] + coefficients["b1"] * np.log1p(10.0)
        + coefficients["b2"] * np.log1p(10.0) ** 2
    )
    assert values[0] == pytest.approx(expected)


def test_json_writer_replaces_target_atomically(tmp_path):
    target = tmp_path / "result.json"
    write_json(target, {"complete": True, "value": 7})

    assert target.exists()
    assert not target.with_name("result.json.tmp").exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"complete": True, "value": 7}


def test_sample_order_hash_tracks_scene_target_and_observation_end_frame():
    first = ordered_sample_metadata_sha256(np.array(["v1", "v2"]), np.array(["p1", "p2"]), np.array([10, 20]))
    reordered = ordered_sample_metadata_sha256(np.array(["v2", "v1"]), np.array(["p2", "p1"]), np.array([20, 10]))

    assert first["sample_count"] == 2
    assert first["rows"][0] == {"scene_id": "v1", "target_id": "p1", "obs_end_frame": 10}
    assert first["sha256"] != reordered["sha256"]


def test_frozen_decision_reads_twenty_percent_stratified_risk_row():
    report = {
        "partial_spearman_raw_u_given_motion": {"u_mean_vs_ade": 0.2, "u_mean_vs_fde": 0.1},
        "metrics": {"adjusted_u_mean": {"spearman_vs_ade": {"rho": 0.2}}},
        "motion_stratified_metrics": {"slow_medium_fast": [
            {"raw_u_vs_ade": {"rho": 0.2}, "adjusted_u_vs_ade": {"rho": 0.0}},
            {"raw_u_vs_ade": {"rho": 0.0}, "adjusted_u_vs_ade": {"rho": 0.2}},
            {"raw_u_vs_ade": {"rho": 0.0}, "adjusted_u_vs_ade": {"rho": 0.0}},
        ]},
    }
    risk = {"adjusted_u_mean": [
        {"nominal_coverage": 1.0, "ade": 30.0},
        {"nominal_coverage": 0.2, "ade": 9.0},
    ]}
    permutation = {"rows": [{
        "nominal_coverage": 0.2,
        "randomized_within_motion_mean_sample_std": {"ade": {"mean": 10.0}},
    }]}
    cluster = {"metrics": {"adjusted_u_mean": {"spearman_vs_ade": {"lower_95": 0.01}}}}

    decision = _decision_from_holdout(report, risk, permutation, cluster)

    assert decision["inputs"]["adjusted_stratified_20pct_ade"] == pytest.approx(9.0)
    assert decision["inputs"]["within_motion_random_20pct_ade_mean"] == pytest.approx(10.0)
    assert decision["decision"] == "REPLICATED"
