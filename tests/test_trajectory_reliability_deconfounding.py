import numpy as np
import pytest
from scipy.stats import rankdata

from scripts.evaluate_trajectory_reliability import prediction_diagnostics
from scripts.evaluate_trajectory_reliability_deconfounding import (
    apply_motion_adjustment,
    cluster_bootstrap_indices,
    fit_motion_adjustment,
    partial_spearman,
    select_stratified_indices,
)


def test_partial_spearman_matches_explicit_rank_residualization():
    motion = np.array([1, 4, 2, 8, 3, 7, 5, 6], dtype=float)
    uncertainty = np.array([4, 1, 7, 2, 8, 3, 6, 5], dtype=float)
    error = np.array([2, 7, 1, 8, 4, 6, 3, 5], dtype=float)

    ranks = [rankdata(values, method="average") for values in (uncertainty, error, motion)]
    design = np.column_stack((np.ones(len(motion)), ranks[2]))
    residual_u = ranks[0] - design @ np.linalg.lstsq(design, ranks[0], rcond=None)[0]
    residual_error = ranks[1] - design @ np.linalg.lstsq(design, ranks[1], rcond=None)[0]
    expected = np.corrcoef(residual_u, residual_error)[0, 1]

    assert partial_spearman(uncertainty, error, motion) == pytest.approx(expected)


def test_partial_spearman_is_zero_when_uncertainty_is_only_motion_proxy():
    motion = np.arange(1, 51, dtype=float)
    uncertainty = motion**2
    error = motion

    assert abs(partial_spearman(uncertainty, error, motion)) < 1e-10


def test_partial_spearman_retains_independent_reliability_signal():
    rng = np.random.default_rng(781)
    motion = np.linspace(0, 4, 500)
    reliability_signal = rng.normal(size=len(motion))
    uncertainty = motion + 2.5 * reliability_signal + 10
    error = 1.5 * motion + 3.0 * reliability_signal + 20

    assert partial_spearman(uncertainty, error, motion) > 0.5


def test_validation_polynomial_is_frozen_and_applied_without_refit_or_mutation():
    val_motion = np.array([0, 2, 5, 9, 14, 20], dtype=float)
    val_u = np.array([0.4, 0.7, 1.1, 1.8, 2.4, 3.2], dtype=float)
    test_motion = np.array([3, 8, 17], dtype=float)
    test_u = np.array([0.6, 1.9, 2.6], dtype=float)
    val_motion_before, val_u_before = val_motion.copy(), val_u.copy()
    test_motion_before, test_u_before = test_motion.copy(), test_u.copy()

    fit = fit_motion_adjustment(val_motion, val_u)
    frozen_coefficients = {key: fit[key] for key in ("b0", "b1", "b2")}
    actual = apply_motion_adjustment(test_motion, test_u, fit)
    x = np.log1p(test_motion)
    expected = np.log1p(test_u) - (
        fit["b0"] + fit["b1"] * x + fit["b2"] * x**2
    )

    assert fit["fit_split"] == "validation"
    assert fit["future_gt_not_used"] is True
    assert fit["trajectory_error_not_used"] is True
    np.testing.assert_allclose(actual, expected)
    assert {key: fit[key] for key in frozen_coefficients} == frozen_coefficients
    np.testing.assert_array_equal(val_motion, val_motion_before)
    np.testing.assert_array_equal(val_u, val_u_before)
    np.testing.assert_array_equal(test_motion, test_motion_before)
    np.testing.assert_array_equal(test_u, test_u_before)


def test_adjusted_u_formula_is_log_space_residual():
    coefficients = {"b0": 0.2, "b1": 0.3, "b2": -0.04}
    motion = np.array([4.0, 11.0])
    raw_u = np.array([1.0, 2.5])
    x = np.log1p(motion)
    expected = np.log1p(raw_u) - (
        0.2 + 0.3 * x - 0.04 * x**2
    )

    np.testing.assert_allclose(apply_motion_adjustment(motion, raw_u, coefficients), expected)


def test_motion_stratified_coverage_preserves_per_stratum_fraction():
    score = np.arange(10, dtype=float)
    strata = np.array([0] * 5 + [1] * 5)

    selected, counts = select_stratified_indices(score, strata, coverage=0.6)

    assert len(selected) == 6
    assert counts == {"0": 3, "1": 3}
    np.testing.assert_array_equal(np.sort(selected), [0, 1, 2, 5, 6, 7])


def test_video_cluster_bootstrap_keeps_each_cluster_together():
    cluster_ids = np.array(["video-a", "video-a", "video-b", "video-c", "video-c", "video-c"])
    selected = cluster_bootstrap_indices(cluster_ids, np.random.default_rng(9124))
    sample_weights = np.bincount(selected, minlength=len(cluster_ids))

    for cluster in np.unique(cluster_ids):
        weights = sample_weights[cluster_ids == cluster]
        assert np.all(weights == weights[0])
    assert len(np.unique(cluster_ids[selected])) <= len(np.unique(cluster_ids))


def test_prediction_diagnostics_does_not_mutate_model_output_arrays():
    predictions = np.arange(3 * 2 * 4 * 2, dtype=float).reshape(3, 2, 4, 2) / 100
    future = np.zeros((2, 4, 2), dtype=float)
    size = np.array([[640, 480], [1920, 1080]])
    predictions_before = predictions.copy()
    future_before = future.copy()
    size_before = size.copy()

    result = prediction_diagnostics(predictions, future, size)

    np.testing.assert_array_equal(predictions, predictions_before)
    np.testing.assert_array_equal(future, future_before)
    np.testing.assert_array_equal(size, size_before)
    assert "u_mean" in result["scores"]
