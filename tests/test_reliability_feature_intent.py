from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from scripts.reliability_feature_intent_utils import (
    VARIANT_FEATURES,
    apply_train_oof_normalization,
    bootstrap_comparison,
    fit_train_oof_normalization,
    get_model_inputs,
    validation_calibration,
)
from scripts.evaluate_reliability_feature_intent import summarize_strata
from src.models.reliability_feature_intent import ReliabilityFeatureIntent, parameter_count


def _synthetic_split(n: int = 12) -> dict[str, np.ndarray]:
    return {
        "target_obs": np.zeros((n, 15, 8), dtype=np.float32),
        "future_pred_mean": np.zeros((n, 15, 2), dtype=np.float32),
        "u_mean_pixel": np.linspace(0.0, 11.0, n, dtype=np.float32),
        "observed_motion_pixel": np.linspace(2.0, 24.0, n, dtype=np.float32),
        "adjusted_u": np.linspace(-1.0, 1.0, n, dtype=np.float32),
    }


def _initial_hash(model: ReliabilityFeatureIntent, prefix: str) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(prefix)
    }


def test_future_gt_and_trajectory_errors_are_not_model_inputs():
    signature = inspect.signature(ReliabilityFeatureIntent.forward)
    assert "future_gt" not in signature.parameters
    assert "ADE" not in signature.parameters
    assert "FDE" not in signature.parameters
    data = _synthetic_split()
    normalization = fit_train_oof_normalization(data, fit_split="official_train_oof")
    data["normalized_features"] = {
        key: apply_train_oof_normalization(data[key], normalization, key)
        for key in ("u_mean_pixel", "observed_motion_pixel", "adjusted_u")
    }
    before = get_model_inputs(data, "E")
    data["future_gt"] = np.full((len(data["target_obs"]), 15, 2), 1e6, dtype=np.float32)
    data["ADE"] = np.full(len(data["target_obs"]), 1e6, dtype=np.float32)
    data["FDE"] = np.full(len(data["target_obs"]), 1e6, dtype=np.float32)
    after = get_model_inputs(data, "E")
    assert all(
        (left is None and right is None) or np.array_equal(left, right)
        for left, right in zip(before, after)
    )
    model = ReliabilityFeatureIntent("E")
    with pytest.raises(TypeError):
        model(
            torch.zeros(2, 15, 8),
            torch.zeros(2, 15, 2),
            torch.zeros(2, 2),
            future_gt=torch.zeros(2, 15, 2),
        )


def test_normalization_is_fit_only_on_official_train_oof():
    data = _synthetic_split()
    normalization = fit_train_oof_normalization(data, fit_split="official_train_oof")
    assert normalization["fit_split"] == "official_train_oof"
    with pytest.raises(ValueError, match="may only be fit on official_train_oof"):
        fit_train_oof_normalization(data, fit_split="official_val")


def test_validation_and_test_application_do_not_mutate_train_normalization():
    train = _synthetic_split()
    normalization = fit_train_oof_normalization(train, fit_split="official_train_oof")
    snapshot = {
        key: dict(value) for key, value in normalization["features"].items()
    }
    validation_values = np.array([1e3, 2e3], dtype=np.float64)
    test_values = np.array([-100.0, 1e6], dtype=np.float64)
    val_scaled = apply_train_oof_normalization(
        validation_values, normalization, "u_mean_pixel"
    )
    test_scaled = apply_train_oof_normalization(test_values, normalization, "u_mean_pixel")
    assert np.isfinite(val_scaled).all() and np.isfinite(test_scaled).all()
    assert normalization["features"] == snapshot
    assert normalization["fit_split"] == "official_train_oof"


def test_c_d_e_reliability_dimensions_and_fusion_input_sizes():
    models = {variant: ReliabilityFeatureIntent(variant) for variant in ("C", "D", "E")}
    assert models["C"].reliability_dim == 1
    assert models["D"].reliability_dim == 1
    assert models["E"].reliability_dim == 2
    assert models["C"].fusion_input_dim == 224
    assert models["D"].fusion_input_dim == 224
    assert models["E"].fusion_input_dim == 224
    assert VARIANT_FEATURES["C"] == ("u_mean_pixel",)
    assert VARIANT_FEATURES["D"] == ("adjusted_u",)
    assert VARIANT_FEATURES["E"] == ("observed_motion_pixel", "adjusted_u")


def test_parameter_counts_follow_model_ablation_design():
    counts = {
        variant: parameter_count(ReliabilityFeatureIntent(variant))
        for variant in ("A", "B", "C", "D", "E", "D_no_future")
    }
    assert counts["A"] < counts["B"] < counts["C"]
    assert counts["C"] == counts["D"]
    assert counts["D"] < counts["E"]
    assert counts["A"] < counts["D_no_future"] < counts["D"]


def test_same_seed_initializes_observed_and_future_branches_identically():
    seed = 2024
    matched = {}
    for variant in ("B", "C", "D", "E"):
        torch.manual_seed(seed)
        model = ReliabilityFeatureIntent(variant)
        matched[variant] = (
            _initial_hash(model, "observed_encoder."),
            _initial_hash(model, "future_encoder."),
        )
    reference = matched["B"]
    for variant in ("C", "D", "E"):
        for branch_idx in (0, 1):
            assert matched[variant][branch_idx].keys() == reference[branch_idx].keys()
            assert all(
                torch.equal(matched[variant][branch_idx][key], reference[branch_idx][key])
                for key in reference[branch_idx]
            )


def test_temperature_and_threshold_are_recorded_as_validation_fitted():
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    logits = np.array([-2.0, -0.5, 0.2, 2.5, 0.3, 1.3])
    result = validation_calibration(logits, labels)
    assert result["temperature_fit_split"] == "official_val"
    assert result["threshold_fit_split"] == "official_val"
    assert result["temperature"] > 0
    assert 0 <= result["threshold"] <= 1


def test_video_cluster_bootstrap_uses_paired_samples():
    labels = np.tile(np.array([0, 1, 1, 0]), 3)
    probabilities = np.tile(np.array([0.1, 0.8, 0.7, 0.2]), 3)
    scenes = np.repeat(np.array(["v1", "v2", "v3"]), 4)
    result = bootstrap_comparison(labels, probabilities, probabilities.copy(), scenes)
    assert result["bootstrap_unit"] == "scene_id video cluster"
    assert result["requested_repetitions"] == 2000
    assert result["seed"] == 9124
    assert result["delta_roc_auc"]["mean"] == 0.0
    assert result["delta_brier"]["mean"] == 0.0
    assert result["delta_roc_auc"]["ci_percentile_95"] == {"lower_95": 0.0, "upper_95": 0.0}


def test_high_adjusted_u_maps_to_low_reliability_high_uncertainty():
    data = {
        "adjusted_u": np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]),
        "intent_label": np.array([0, 1, 0, 1, 0, 1]),
    }
    probabilities = {
        model: {seed: np.array([.1, .8, .2, .7, .3, .9]) for seed in (42, 123, 2024)}
        for model in ("A", "B", "C", "D", "E", "D_no_future")
    }
    result = summarize_strata(
        data,
        probabilities,
        {},
        {"reliability_tertile_cutpoints_train_oof_adjusted_u": [-1.0, 1.0]},
    )
    assert result["strata"]["high_reliability"]["sample_count"] == 2
    assert result["strata"]["low_reliability"]["sample_count"] == 3
    assert result["high_uncertainty_group_primary"]["sample_count"] == 3
