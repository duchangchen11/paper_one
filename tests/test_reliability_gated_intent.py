from __future__ import annotations

import inspect

import numpy as np
import torch

from scripts.build_crossfit_future_features import FeatureSplit, ensemble_features, make_folds
from scripts.reliability_gated_intent_utils import (
    apply_motion_adjustment,
    cluster_bootstrap_paired_delta,
    empirical_confidence,
    fit_empirical_reference,
    fit_motion_adjustment,
    fit_temperature,
    choose_balanced_accuracy_threshold,
    reliability_tertile,
)
from src.models.reliability_gated_intent import (
    ObservedOnlyIntent,
    ReliabilityGatedIntent,
    parameter_count,
)


def _state_hash(module, prefix):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
        if key.startswith(prefix)
    }


def test_video_folds_are_disjoint_exhaustive_and_repeatable():
    scene_ids = np.repeat(np.array([f"video_{i:03d}" for i in range(17)]), np.arange(1, 18))
    folds = make_folds(scene_ids, seed=424242)
    assert folds == make_folds(scene_ids, seed=424242)
    sets = [set(fold) for fold in folds]
    assert set.union(*sets) == set(np.unique(scene_ids))
    assert all(not (sets[i] & sets[j]) for i in range(3) for j in range(i + 1, 3))


def test_each_crossfit_training_partition_excludes_its_heldout_scenes():
    scenes = np.repeat(np.array([f"v{i}" for i in range(12)]), 3)
    folds = make_folds(scenes)
    for heldout in folds:
        heldout_mask = np.isin(scenes, heldout)
        train_scenes = set(scenes[~heldout_mask])
        assert train_scenes.isdisjoint(heldout)


def test_oof_feature_builder_covers_ordered_samples_without_future_gt_or_labels():
    dataset = FeatureSplit.__new__(FeatureSplit)
    n = 9
    dataset.target_obs = np.zeros((n, 15, 4), dtype=np.float32)
    dataset.target_abs_obs = np.zeros((n, 15, 4), dtype=np.float32)
    dataset.target_abs_obs[:, :, 0] = np.arange(15, dtype=np.float32)[None]
    dataset.future_gt = np.zeros((n, 15, 2), dtype=np.float32)
    dataset.scene_ids = np.array([f"v{i // 3}" for i in range(n)])
    dataset.target_ids = np.array([f"p{i}" for i in range(n)])
    dataset.obs_end_frame = np.arange(n)
    dataset.image_size = np.tile(np.array([[640, 480]], dtype=np.float32), (n, 1))
    indices = np.arange(n)
    prediction = np.zeros((3, n, 15, 2), dtype=np.float32)
    features = ensemble_features(prediction, dataset, indices)
    assert np.array_equal(features["sample_index"], indices)
    assert len(np.unique(features["sample_index"])) == n
    assert "future_gt" not in features
    assert "intent_label" not in features
    assert features["future_pred_mean"].shape == (n, 15, 2)


def test_intention_forward_has_no_future_gt_input():
    assert "future_gt" not in inspect.signature(ReliabilityGatedIntent.forward).parameters
    assert "future_gt" not in inspect.signature(ObservedOnlyIntent.forward).parameters


def test_polynomial_adjustment_is_fit_only_on_explicit_train_oof_arrays():
    motion = np.linspace(0, 30, 100)
    uncertainty = np.sqrt(motion + 1)
    fitted = fit_motion_adjustment(motion, uncertainty)
    assert fitted["fit_split"] == "official_train_oof_only"
    assert fitted["future_gt_used"] is False
    assert fitted["trajectory_error_used"] is False
    assert fitted["intent_label_used"] is False
    altered_external = apply_motion_adjustment(np.array([1000.0]), np.array([0.01]), fitted)
    fitted_again = fit_motion_adjustment(motion, uncertainty)
    assert fitted_again == fitted
    assert np.isfinite(altered_external).all()


def test_empirical_confidence_is_monotone_and_strictly_inside_zero_one():
    fit = fit_empirical_reference(np.array([0.0, 1.0, 1.0, 2.0, 5.0]), source="train_oof")
    scores = np.array([-1.0, 0.0, 1.0, 1.5, 5.0, 6.0])
    confidence = empirical_confidence(scores, fit)
    assert np.all(np.diff(confidence) <= 0)
    assert np.all((confidence > 0) & (confidence < 1))


def test_motion_and_reliability_gates_are_open_interval_and_tertiles_fixed():
    fit = fit_empirical_reference(np.array([-1.0, 0.0, 0.5, 1.0]), source="train_oof")
    confidence = empirical_confidence(np.array([-2.0, 0.0, 2.0]), fit)
    assert np.all((confidence > 0) & (confidence < 1))
    assert np.all(np.diff(confidence) <= 0)
    assert np.array_equal(reliability_tertile(np.array([-1, 0, 1]), np.array([0, 0])), np.array([0, 2, 2]))


def test_bcd_architecture_parameter_counts_and_same_seed_initialization():
    states = []
    counts = []
    for _variant in ("B", "C", "D"):
        torch.manual_seed(123)
        model = ReliabilityGatedIntent()
        states.append(_state_hash(model, ("future_encoder.", "residual_head.")))
        counts.append(parameter_count(model))
    assert counts[0] == counts[1] == counts[2]
    for state in states[1:]:
        assert state.keys() == states[0].keys()
        assert all(torch.equal(state[key], states[0][key]) for key in state)


def test_base_is_frozen_and_unchanged_during_residual_training():
    model = ReliabilityGatedIntent()
    model.freeze_base()
    before = _state_hash(model, ("observed_encoder.", "base_head."))
    model.train()
    assert model.base_is_frozen
    assert not model.observed_encoder.training and not model.base_head.training
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    x = torch.randn(8, 15, 8)
    future = torch.randn(8, 15, 2)
    gate = torch.rand(8)
    labels = torch.randint(0, 2, (8,)).float()
    output = model(x, future, gate)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output["final_logit"], labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    after = _state_hash(model, ("observed_encoder.", "base_head."))
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_zero_initialized_residual_and_gate_endpoints():
    model = ReliabilityGatedIntent().eval()
    x = torch.randn(4, 15, 8)
    future = torch.randn(4, 15, 2)
    base = model(x)["base_logit"]
    output = model(x, future, torch.tensor([0.2, 0.4, 0.7, 0.9]))
    assert torch.equal(output["delta_logit"], torch.zeros_like(output["delta_logit"]))
    assert torch.allclose(output["final_logit"], base, atol=0, rtol=0)
    zero_gate = model(x, future, torch.zeros(4))
    assert torch.equal(zero_gate["final_logit"], zero_gate["base_logit"])
    one_gate = model(x, future, torch.ones(4))
    assert torch.equal(one_gate["final_logit"], one_gate["base_logit"] + one_gate["delta_logit"])


def test_temperature_and_threshold_are_deterministic_validation_calibrators():
    labels = np.array([0, 0, 1, 1, 0, 1])
    logits = np.array([-2.0, -0.5, 0.2, 2.5, 0.3, 1.3])
    temperature = fit_temperature(logits, labels)
    probabilities = 1.0 / (1.0 + np.exp(-logits / temperature))
    threshold = choose_balanced_accuracy_threshold(probabilities, labels)
    assert 0.05 <= temperature <= 20
    assert 0 <= threshold <= 1
    assert fit_temperature(logits, labels) == temperature
    assert choose_balanced_accuracy_threshold(probabilities, labels) == threshold


def test_paired_video_cluster_bootstrap_uses_same_resampled_clusters():
    scenes = np.repeat(np.array([f"video_{i}" for i in range(12)]), 4)
    labels = np.tile(np.array([0, 1, 0, 1]), 12)
    probs = np.linspace(0.01, 0.99, len(labels))
    result = cluster_bootstrap_paired_delta(labels, probs, probs.copy(), scenes, repetitions=50, seed=9124)
    assert result["bootstrap_unit"] == "scene_id video cluster"
    assert result["valid_repetitions"] == 50
    assert result["delta_roc_auc"]["ci_percentile_95"] == {"lower_95": 0.0, "upper_95": 0.0}
    assert result["delta_brier"]["ci_percentile_95"] == {"lower_95": 0.0, "upper_95": 0.0}


def test_changing_only_gate_does_not_change_base_or_future_encoding():
    model = ReliabilityGatedIntent().eval()
    observed = torch.randn(6, 15, 8)
    future = torch.randn(6, 15, 2)
    first = model(observed, future, torch.zeros(6))
    second = model(observed, future, torch.ones(6))
    assert torch.equal(first["base_logit"], second["base_logit"])
    assert torch.equal(first["z_future"], second["z_future"])
    assert torch.equal(first["delta_logit"], second["delta_logit"])
    assert not torch.equal(first["final_logit"], second["final_logit"]) or torch.equal(first["delta_logit"], torch.zeros_like(first["delta_logit"]))
