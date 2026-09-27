"""Integrity, feature preparation, and paired evaluation helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.reliability_gated_intent_utils import (
    apply_motion_adjustment,
    binary_metrics,
    canonical_json,
    choose_balanced_accuracy_threshold,
    cluster_bootstrap_paired_delta,
    fit_temperature,
    mean_sample_std,
)

FEATURE_FILES = {
    "train_oof": "train_oof_features.npz",
    "val": "val_features.npz",
    "test_unlabeled": "test_features_unlabeled.npz",
}
RAW_FEATURES = ("u_mean_pixel", "observed_motion_pixel", "adjusted_u")
VARIANT_FEATURES = {
    "A": (),
    "B": (),
    "C": ("u_mean_pixel",),
    "D": ("adjusted_u",),
    "E": ("observed_motion_pixel", "adjusted_u"),
    "D_no_future": ("adjusted_u",),
}
VARIANT_NAMES = {
    "A": "Observed-only",
    "B": "Observed + future trajectory",
    "C": "Observed + future + raw reliability",
    "D": "Observed + future + motion-adjusted reliability",
    "E": "Observed + future + motion + adjusted reliability",
    "D_no_future": "Observed + adjusted reliability (no future)",
}
PRIMARY_COMPARISONS = (("B", "D"), ("A", "D"), ("C", "D"), ("D", "E"))
SEEDS = (42, 123, 2024)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_self_hash(payload: dict[str, Any], *, label: str) -> str:
    if "manifest_sha256" not in payload:
        raise ValueError(f"{label} is missing manifest_sha256")
    claimed = str(payload["manifest_sha256"])
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual = hashlib.sha256(canonical_json(body)).hexdigest()
    if actual != claimed:
        raise ValueError(f"{label} manifest SHA mismatch: {actual} != {claimed}")
    return actual


def verify_feature_cache(root: Path) -> dict[str, Any]:
    manifest_path = root / "feature_cache_manifest.json"
    crossfit_path = root / "crossfit_manifest.json"
    manifest = load_json(manifest_path)
    crossfit = load_json(crossfit_path)
    verify_self_hash(manifest, label="feature_cache_manifest")
    verify_self_hash(crossfit, label="crossfit_manifest")
    if manifest.get("crossfit_manifest_sha256") != crossfit.get("manifest_sha256"):
        raise ValueError("feature cache and crossfit manifest identities disagree")
    expected_cache = manifest.get("feature_cache_sha256", {})
    for filename in FEATURE_FILES.values():
        expected = expected_cache.get(filename)
        path = root / "cache" / filename
        if expected is None or not path.is_file():
            raise FileNotFoundError(f"manifest/cache entry missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"feature cache SHA mismatch for {filename}: {actual} != {expected}")
    if manifest.get("future_gt_saved") is not False:
        raise ValueError("feature cache manifest does not certify future_gt exclusion")
    if manifest.get("test_intent_label_read_or_saved") is not False:
        raise ValueError("unlabeled test cache was not certified label-free before freeze")
    return {
        "feature_cache_manifest_sha256": sha256_file(manifest_path),
        "feature_cache_manifest_embedded_sha256": manifest["manifest_sha256"],
        "crossfit_manifest_sha256": sha256_file(crossfit_path),
        "crossfit_manifest_embedded_sha256": crossfit["manifest_sha256"],
        "cache_file_sha256": {
            filename: sha256_file(root / "cache" / filename)
            for filename in FEATURE_FILES.values()
        },
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def prepare_splits(root: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    manifests = verify_feature_cache(root)
    splits = {
        name: load_npz(root / "cache" / filename)
        for name, filename in FEATURE_FILES.items()
    }
    required = {
        "sample_index",
        "scene_id",
        "target_obs",
        "future_pred_mean",
        "u_mean_pixel",
        "observed_motion_pixel",
    }
    for split_name, data in splits.items():
        missing = required - data.keys()
        if missing:
            raise ValueError(f"{split_name} cache missing fields: {sorted(missing)}")
        if "future_gt" in data or "ADE" in data or "FDE" in data or "ade" in data or "fde" in data:
            raise ValueError(f"forbidden trajectory ground-truth/error field in {split_name} cache")
        if data["target_obs"].shape[1:] != (15, 8):
            raise ValueError(f"unexpected target_obs shape in {split_name}")
        if data["future_pred_mean"].shape[1:] != (15, 2):
            raise ValueError(f"unexpected future_pred_mean shape in {split_name}")
        if split_name == "test_unlabeled":
            if "intent_label" in data or "crossing_label" in data:
                raise ValueError("pre-freeze test feature cache contains a label")
        elif "intent_label" not in data:
            raise ValueError(f"{split_name} cache has no intent_label")

    transform_root = root / "reliability_transform.json"
    old_protocol = load_json(root / "protocol_frozen.json")
    if sha256_file(transform_root) != old_protocol.get("reliability_transform_sha256"):
        raise ValueError("previously frozen motion-adjustment transform SHA mismatch")
    transform_payload = load_json(transform_root)
    polynomial = transform_payload["polynomial"]
    if polynomial.get("fit_split") != "official_train_oof_only":
        raise ValueError("motion-adjustment coefficients were not fit on official train OOF")
    for split_name, data in splits.items():
        data["adjusted_u"] = apply_motion_adjustment(
            data["observed_motion_pixel"], data["u_mean_pixel"], polynomial
        )
    manifests["reliability_transform_sha256"] = sha256_file(transform_root)
    manifests["reliability_adjustment_coefficients"] = polynomial
    return splits, manifests


def fit_train_oof_normalization(
    train: dict[str, np.ndarray], *, fit_split: str
) -> dict[str, Any]:
    if fit_split != "official_train_oof":
        raise ValueError("normalization may only be fit on official_train_oof")
    stats: dict[str, Any] = {"fit_split": "official_train_oof", "std_ddof": 0, "features": {}}
    for feature in RAW_FEATURES:
        values = np.asarray(train[feature], dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"non-finite/empty train OOF feature: {feature}")
        mean, std = float(values.mean()), float(values.std(ddof=0))
        if std <= 1e-12:
            raise ValueError(f"train OOF feature has zero scale: {feature}")
        stats["features"][feature] = {
            "mean": mean,
            "std": std,
            "sample_count": int(values.size),
        }
    return stats


def apply_train_oof_normalization(
    values: np.ndarray, normalization: dict[str, Any], feature: str
) -> np.ndarray:
    if normalization.get("fit_split") != "official_train_oof":
        raise ValueError("normalization must be fitted on official train OOF only")
    parameters = normalization["features"][feature]
    result = (np.asarray(values, dtype=np.float64) - parameters["mean"]) / parameters["std"]
    if not np.isfinite(result).all():
        raise ValueError(f"normalized feature {feature} contains non-finite values")
    return result.astype(np.float32)


def add_normalized_reliability(
    data: dict[str, np.ndarray], normalization: dict[str, Any]
) -> dict[str, np.ndarray]:
    normalized = {
        name: apply_train_oof_normalization(data[name], normalization, name)
        for name in RAW_FEATURES
    }
    return {**data, "normalized_features": normalized}


def get_model_inputs(
    data: dict[str, np.ndarray], variant: str
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if variant not in VARIANT_FEATURES:
        raise ValueError(f"unknown model variant: {variant}")
    obs = np.asarray(data["target_obs"], dtype=np.float32)
    future = np.asarray(data["future_pred_mean"], dtype=np.float32) if variant in {"B", "C", "D", "E"} else None
    selected = VARIANT_FEATURES[variant]
    reliability = None
    if selected:
        normalized = data["normalized_features"]
        reliability = np.stack([normalized[name] for name in selected], axis=-1).astype(np.float32)
    return obs, future, reliability


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "sample_count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q33": float(np.quantile(values, 1 / 3)),
        "q67": float(np.quantile(values, 2 / 3)),
        "q95": float(np.quantile(values, 0.95)),
    }


def validation_calibration(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    temperature = fit_temperature(logits, labels)
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(scaled, -60, 60)))
    threshold = choose_balanced_accuracy_threshold(probabilities, labels)
    return {
        "temperature": float(temperature),
        "temperature_fit_split": "official_val",
        "threshold": float(threshold),
        "threshold_fit_split": "official_val",
    }


def summarize_seed_metrics(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = ("roc_auc", "brier", "ece_15_equal_width", "balanced_accuracy", "f1_positive")
    return {metric: mean_sample_std([float(rows[str(seed)][metric]) for seed in SEEDS]) for metric in metrics}


def bootstrap_comparison(
    labels: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    scene_ids: np.ndarray,
) -> dict[str, Any]:
    return cluster_bootstrap_paired_delta(
        labels,
        p_a,
        p_b,
        scene_ids,
        repetitions=2000,
        seed=9124,
    )


def metric_pair_delta(
    labels: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    threshold_a: float,
    threshold_b: float,
) -> dict[str, float]:
    m_a = binary_metrics(labels, p_a, threshold_a)
    m_b = binary_metrics(labels, p_b, threshold_b)
    return {
        "delta_roc_auc": float(m_b["roc_auc"] - m_a["roc_auc"]),
        "delta_brier": float(m_b["brier"] - m_a["brier"]),
        "delta_balanced_accuracy": float(m_b["balanced_accuracy"] - m_a["balanced_accuracy"]),
    }


def short_delta_name(model_a: str, model_b: str) -> str:
    return f"{model_b}-{model_a}"
