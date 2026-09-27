"""Shared, deterministic transforms and evaluation helpers for the gated study."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def fit_motion_adjustment(motion: np.ndarray, uncertainty: np.ndarray) -> dict[str, Any]:
    motion = np.asarray(motion, dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if len(motion) != len(uncertainty) or len(motion) < 3:
        raise ValueError("motion and uncertainty must have equal length >= 3")
    if not np.isfinite(motion).all() or not np.isfinite(uncertainty).all():
        raise ValueError("motion and uncertainty must be finite")
    if np.any(motion < 0) or np.any(uncertainty < 0):
        raise ValueError("motion and uncertainty must be non-negative")
    x, y = np.log1p(motion), np.log1p(uncertainty)
    b2, b1, b0 = np.polyfit(x, y, deg=2)
    return {
        "b0": float(b0), "b1": float(b1), "b2": float(b2),
        "fit_split": "official_train_oof_only",
        "target": "log1p(u_mean_pixel)",
        "predictor": "log1p(observed_motion_pixel)",
        "degree": 2,
        "fit_sample_count": int(len(motion)),
        "fit_motion_sha256": array_sha256(motion),
        "fit_u_mean_sha256": array_sha256(uncertainty),
        "future_gt_used": False,
        "trajectory_error_used": False,
        "intent_label_used": False,
    }


def apply_motion_adjustment(
    motion: np.ndarray, uncertainty: np.ndarray, coefficients: dict[str, Any]
) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float64).reshape(-1)
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if len(motion) != len(uncertainty) or np.any(motion < 0) or np.any(uncertainty < 0):
        raise ValueError("motion and uncertainty must be non-negative and equal length")
    x = np.log1p(motion)
    fitted = coefficients["b0"] + coefficients["b1"] * x + coefficients["b2"] * x**2
    return np.log1p(uncertainty) - fitted


def fit_empirical_reference(values: np.ndarray, *, source: str) -> dict[str, Any]:
    reference = np.sort(np.asarray(values, dtype=np.float64).reshape(-1), kind="mergesort")
    if len(reference) == 0 or not np.isfinite(reference).all():
        raise ValueError("empirical CDF reference must be non-empty and finite")
    return {
        "reference": reference,
        "source": source,
        "sample_count": int(len(reference)),
        "reference_sha256": array_sha256(reference),
    }


def empirical_confidence(scores: np.ndarray, fitted: dict[str, Any]) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    reference = fitted["reference"]
    rank = np.searchsorted(reference, scores, side="right")
    confidence = 1.0 - (rank + 0.5) / (len(reference) + 1.0)
    if not ((confidence > 0).all() and (confidence < 1).all()):
        raise RuntimeError("Empirical confidence escaped the open unit interval")
    return confidence.astype(np.float32)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(logits) != len(labels) or not np.isfinite(logits).all():
        raise ValueError("temperature inputs are invalid")

    def objective(log_temperature: float) -> float:
        scaled = logits / np.exp(log_temperature)
        # Numerically stable mean BCE-with-logits.
        return float(np.mean(np.logaddexp(0.0, scaled) - labels * scaled))

    fit = minimize_scalar(objective, bounds=(np.log(0.05), np.log(20.0)), method="bounded", options={"xatol": 1e-10})
    return float(np.exp(fit.x))


def choose_balanced_accuracy_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(p) != len(y) or not np.isfinite(p).all():
        raise ValueError("threshold inputs are invalid")
    unique = np.unique(p)
    candidates = np.concatenate(([0.0], (unique[:-1] + unique[1:]) / 2.0, [1.0]))
    scores = np.asarray([balanced_accuracy_score(y, p >= threshold) for threshold in candidates])
    best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0.0, atol=1e-12))
    # Deterministic tie-break: prefer closest to 0.5, then lower threshold.
    selected = min(best.tolist(), key=lambda index: (abs(candidates[index] - 0.5), candidates[index]))
    return float(candidates[selected])


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(y) != len(p) or bins < 1:
        raise ValueError("ECE inputs are invalid")
    bin_id = np.minimum((p * bins).astype(np.int64), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = bin_id == index
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    pred = (p >= threshold).astype(np.int64)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_15_equal_width": expected_calibration_error(y, p, bins=15),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_positive": float(f1_score(y, pred, zero_division=0)),
        "negative_recall_specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "threshold": float(threshold),
        "sample_count": int(len(y)),
        "negative_count": int(np.sum(y == 0)),
        "positive_count": int(np.sum(y == 1)),
    }


def paired_delta_metrics(labels: np.ndarray, probability_a: np.ndarray, probability_b: np.ndarray, threshold_a: float, threshold_b: float) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    metrics_a = binary_metrics(y, probability_a, threshold_a)
    metrics_b = binary_metrics(y, probability_b, threshold_b)
    return {
        "delta_roc_auc": float(metrics_b["roc_auc"] - metrics_a["roc_auc"]),
        "delta_brier": float(metrics_b["brier"] - metrics_a["brier"]),
        "delta_balanced_accuracy": float(metrics_b["balanced_accuracy"] - metrics_a["balanced_accuracy"]),
    }


def mean_sample_std(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "n": int(len(array)),
    }


def cluster_bootstrap_paired_delta(
    labels: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    scene_ids: np.ndarray,
    *,
    repetitions: int = 2000,
    seed: int = 9124,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    pa = np.asarray(probability_a, dtype=np.float64).reshape(-1)
    pb = np.asarray(probability_b, dtype=np.float64).reshape(-1)
    scenes = np.asarray(scene_ids).astype(str).reshape(-1)
    if not (len(y) == len(pa) == len(pb) == len(scenes)):
        raise ValueError("bootstrap inputs have different lengths")
    unique_scenes = np.unique(scenes)
    rows = {scene: np.flatnonzero(scenes == scene) for scene in unique_scenes}
    rng = np.random.RandomState(seed)
    delta_auc: list[float] = []
    delta_brier: list[float] = []
    for _ in range(repetitions):
        draw = rng.choice(unique_scenes, size=len(unique_scenes), replace=True)
        indexes = np.concatenate([rows[scene] for scene in draw])
        if np.unique(y[indexes]).size < 2:
            continue
        delta_auc.append(float(roc_auc_score(y[indexes], pb[indexes]) - roc_auc_score(y[indexes], pa[indexes])))
        delta_brier.append(float(brier_score_loss(y[indexes], pb[indexes]) - brier_score_loss(y[indexes], pa[indexes])))

    def ci(values: list[float]) -> dict[str, float]:
        lower, upper = np.quantile(np.asarray(values), [0.025, 0.975])
        return {"lower_95": float(lower), "upper_95": float(upper)}

    if not delta_auc:
        raise RuntimeError("No valid paired video-cluster bootstrap samples")
    return {
        "bootstrap_unit": "scene_id video cluster",
        "seed": seed,
        "requested_repetitions": repetitions,
        "valid_repetitions": len(delta_auc),
        "video_count": int(len(unique_scenes)),
        "delta_roc_auc": {"mean": float(np.mean(delta_auc)), "ci_percentile_95": ci(delta_auc)},
        "delta_brier": {"mean": float(np.mean(delta_brier)), "ci_percentile_95": ci(delta_brier)},
    }


def reliability_tertile(adjusted_u: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    values = np.asarray(adjusted_u, dtype=np.float64).reshape(-1)
    bounds = np.asarray(cutpoints, dtype=np.float64).reshape(-1)
    if bounds.shape != (2,) or bounds[0] > bounds[1]:
        raise ValueError("tertile cutpoints must be an ordered pair")
    return np.searchsorted(bounds, values, side="right").astype(np.int8)


def train_defined_deciles(values: np.ndarray) -> np.ndarray:
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0.1, 0.9, 9), method="linear")
