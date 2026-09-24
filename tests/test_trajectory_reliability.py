import numpy as np
import pytest

from scripts.evaluate_trajectory_reliability import prediction_diagnostics


def test_disagreement_is_invariant_to_model_order():
    rng = np.random.default_rng(17)
    predictions = rng.normal(size=(3, 4, 5, 2))
    future = rng.normal(size=(4, 5, 2))
    image_size = np.array([[640, 480], [1280, 720], [800, 600], [1920, 1080]])

    original = prediction_diagnostics(predictions, future, image_size)
    permuted = prediction_diagnostics(predictions[[2, 0, 1]], future, image_size)

    np.testing.assert_allclose(
        original["ensemble_prediction_pixel"], permuted["ensemble_prediction_pixel"]
    )
    np.testing.assert_allclose(original["sample_errors"]["ade_pixel"], permuted["sample_errors"]["ade_pixel"])
    for name in original["scores"]:
        np.testing.assert_allclose(original["scores"][name], permuted["scores"][name])


def test_identical_predictions_have_zero_disagreement():
    one_prediction = np.arange(24, dtype=np.float64).reshape(1, 2, 6, 2) / 100
    predictions = np.repeat(one_prediction, 3, axis=0)
    future = np.zeros((2, 6, 2), dtype=np.float64)
    image_size = np.array([[640, 480], [1280, 720]])

    result = prediction_diagnostics(predictions, future, image_size)

    for score in result["scores"].values():
        # Centering identical floating-point predictions can leave roundoff at ~1e-15 px.
        np.testing.assert_allclose(score, 0, atol=1e-12)


def test_changing_one_model_prediction_increases_disagreement():
    predictions = np.zeros((3, 2, 4, 2), dtype=np.float64)
    future = np.zeros((2, 4, 2), dtype=np.float64)
    image_size = np.array([[100, 80], [200, 160]])
    baseline = prediction_diagnostics(predictions, future, image_size)

    predictions[2, :, :, 0] = 0.1
    changed = prediction_diagnostics(predictions, future, image_size)

    for score in baseline["scores"]:
        np.testing.assert_allclose(baseline["scores"][score], 0, atol=1e-12)
        assert np.all(changed["scores"][score] > 0)


def test_pixel_scale_ade_fde_and_pairwise_score_formulas():
    # Each model predicts a constant normalized x offset of 0, 0.1, or 0.2.
    predictions = np.zeros((3, 2, 2, 2), dtype=np.float64)
    predictions[1, ..., 0] = 0.1
    predictions[2, ..., 0] = 0.2
    future = np.zeros((2, 2, 2), dtype=np.float64)
    image_size = np.array([[100, 200], [50, 80]], dtype=np.float64)

    result = prediction_diagnostics(predictions, future, image_size)

    # Ensemble means are x=10 px and x=5 px, for both horizons.
    np.testing.assert_allclose(result["sample_errors"]["ade_pixel"], [10, 5])
    np.testing.assert_allclose(result["sample_errors"]["fde_pixel"], [10, 5])
    np.testing.assert_allclose(result["ensemble_prediction_pixel"][:, :, 0], [[10, 10], [5, 5]])
    # For [0, 10, 20] px, mean pairwise distance is 40/3; for [0, 5, 10], it is 20/3.
    np.testing.assert_allclose(result["scores"]["u_pairwise"], [40 / 3, 20 / 3])
    np.testing.assert_allclose(result["scores"]["u_endpoint_pairwise"], [40 / 3, 20 / 3])


def test_prediction_diagnostics_rejects_misaligned_shapes():
    predictions = np.zeros((3, 2, 4, 2))
    future = np.zeros((2, 4, 2))

    with pytest.raises(ValueError, match="Incompatible"):
        prediction_diagnostics(predictions, future, np.ones((1, 2)))
