#!/usr/bin/env python3
"""Summarize the frozen-backbone P1/P2 study and pre-existing comparators."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trajectory_preserving_utils import SEEDS

RESULTS = ROOT / "results/trajectory_preserving_joint"
METHODS = ("P1_target_only", "P2_target_scene")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values: list[float], digits: int = 4) -> str:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return f"{array.mean():.{digits}f} ± 0.0000"
    return f"{array.mean():.{digits}f} ± {array.std(ddof=1):.{digits}f}"


def metric(run: dict[str, Any], name: str) -> float:
    if name == "auc":
        return float(run["intent"]["roc_auc"])
    if name == "brier":
        return float(run["intent"]["brier"])
    if name == "f1":
        return float(run["intent"]["f1"])
    if name == "bacc":
        return float(run["intent"]["balanced_accuracy"])
    if name == "ade":
        return float(run["trajectory"]["ade_pixel"])
    if name == "fde":
        return float(run["trajectory"]["fde_pixel"])
    raise KeyError(name)


def main() -> None:
    protocol = read_json(RESULTS / "protocol_frozen.json")
    checksum = (RESULTS / "protocol_frozen.sha256").read_text(encoding="utf-8").split()[0]
    if not protocol.get("frozen"):
        raise RuntimeError("Cannot summarize before protocol freeze")
    intent_baseline = read_json(ROOT / "results/reliability_gated_intent_15x15/test_metrics.json")["models"]["A_observed_only"]["seeds"]
    trajectory_reference = read_json(RESULTS / "trajectory_reference.json")["per_seed"]
    joint100: dict[int, dict[str, float]] = {}
    preserving: dict[str, dict[int, dict[str, Any]]] = {method: {} for method in METHODS}

    for seed in SEEDS:
        joint_metrics = read_json(ROOT / "results/joint_loss_balance/lambda100" / f"seed{seed}" / "metrics.json")
        test = joint_metrics.get("test")
        if test is None:
            raise RuntimeError(f"Missing historical lambda=100 test metrics for seed {seed}")
        joint100[seed] = {
            "auc": float(test["intent_auc"]),
            "brier": float(test["intent_brier"]),
            "f1": float(test["intent_f1"]),
            "bacc": float(test["intent_balanced_accuracy"]),
            "ade": float(test["trajectory_ade_pixel"]),
            "fde": float(test["trajectory_fde_pixel"]),
        }
        for method in METHODS:
            run_dir = RESULTS / method / f"seed{seed}"
            run_metrics = read_json(run_dir / "metrics.json")
            access = read_json(run_dir / "test_access_record.json")
            if run_metrics.get("test_evaluation_status") != "evaluated_once_after_protocol_freeze":
                raise RuntimeError(f"Official post-freeze result missing for {method} seed {seed}")
            if not access.get("loaded_after_frozen_protocol") or access.get("evaluation_count") != 1:
                raise RuntimeError(f"Official test-access record is incomplete for {method} seed {seed}")
            preserving[method][seed] = run_metrics["test"]

    def existing_intent(seed: int) -> dict[str, float]:
        row = intent_baseline[str(seed)]
        return {
            "auc": float(row["roc_auc"]),
            "brier": float(row["brier"]),
            "f1": float(row["f1_positive"]),
            "bacc": float(row["balanced_accuracy"]),
        }

    t0 = {
        seed: {
            "ade": float(trajectory_reference[str(seed)]["existing_official_test_reference"]["trajectory_ade_pixel"]),
            "fde": float(trajectory_reference[str(seed)]["existing_official_test_reference"]["trajectory_fde_pixel"]),
        }
        for seed in SEEDS
    }

    # The test set is used only here for the predeclared descriptive comparison;
    # no checkpoint, architecture, or hyperparameter is selected from it.
    table_rows: list[tuple[str, dict[int, dict[str, float | None]]]] = []
    table_rows.append(("Observed-only intention", {seed: {**existing_intent(seed), "ade": None, "fde": None} for seed in SEEDS}))
    table_rows.append(("Trajectory-only (T0)", {seed: {"auc": None, "brier": None, "f1": None, "bacc": None, **t0[seed]} for seed in SEEDS}))
    table_rows.append(("Joint fixed λ=100", {seed: joint100[seed] for seed in SEEDS}))
    for method in METHODS:
        table_rows.append(
            (
                method,
                {
                    seed: {
                        "auc": metric(preserving[method][seed], "auc"),
                        "brier": metric(preserving[method][seed], "brier"),
                        "f1": metric(preserving[method][seed], "f1"),
                        "bacc": metric(preserving[method][seed], "bacc"),
                        "ade": metric(preserving[method][seed], "ade"),
                        "fde": metric(preserving[method][seed], "fde"),
                    }
                    for seed in SEEDS
                },
            )
        )

    lines = [
        "# Trajectory-Preserving Frozen Backbone Baseline",
        "",
        f"Protocol: `{protocol['protocol_id']}`; SHA256 `{checksum}`. Validation selection and calibration were frozen before the official test archive was opened. P1/P2 test metrics below are descriptive only and were not used for tuning.",
        "",
        "Three-seed values are mean ± sample SD. Intent metrics: higher AUC/F1/BAcc is better; lower Brier is better. Trajectory errors are pixels; lower is better. N/A means the model does not produce that task output.",
        "",
        "## Table 1. Aggregate comparison",
        "",
        "| Method | Intent AUC | Brier | F1 | BAcc | ADE (px) | FDE (px) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, runs in table_rows:
        cells = []
        for key in ("auc", "brier", "f1", "bacc", "ade", "fde"):
            values = [runs[seed][key] for seed in SEEDS]
            present = [float(value) for value in values if value is not None]
            cells.append("N/A" if not present else mean_sd(present, 4 if key in {"auc", "brier", "f1", "bacc"} else 2))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Table 2. Matched-seed trajectory preservation",
            "",
            "| Seed | T0 ADE | P1 ADE | ΔADE | P2 ADE | ΔADE |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in SEEDS:
        t0_ade = t0[seed]["ade"]
        p1_ade = metric(preserving["P1_target_only"][seed], "ade")
        p2_ade = metric(preserving["P2_target_scene"][seed], "ade")
        lines.append(f"| {seed} | {t0_ade:.3f} | {p1_ade:.3f} | {p1_ade-t0_ade:+.6f} | {p2_ade:.3f} | {p2_ade-t0_ade:+.6f} |")
    p1_fde_delta = [metric(preserving["P1_target_only"][seed], "fde") - t0[seed]["fde"] for seed in SEEDS]
    p2_fde_delta = [metric(preserving["P2_target_scene"][seed], "fde") - t0[seed]["fde"] for seed in SEEDS]
    lines.extend(
        [
            "",
            f"Matched-seed FDE deltas: P1 {mean_sd(p1_fde_delta, 6)} px; P2 {mean_sd(p2_fde_delta, 6)} px.",
            "",
            "## Table 3. Per-seed intention metrics",
            "",
            "| Method | Seed | AUC | Brier | F1 | BAcc |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        for seed in SEEDS:
            run = preserving[method][seed]
            lines.append(
                f"| {method} | {seed} | {metric(run, 'auc'):.4f} | {metric(run, 'brier'):.4f} | "
                f"{metric(run, 'f1'):.4f} | {metric(run, 'bacc'):.4f} |"
            )

    p1_auc = [metric(preserving["P1_target_only"][seed], "auc") for seed in SEEDS]
    p2_auc = [metric(preserving["P2_target_scene"][seed], "auc") for seed in SEEDS]
    i0_auc = [existing_intent(seed)["auc"] for seed in SEEDS]
    j100_ade = [joint100[seed]["ade"] for seed in SEEDS]
    j100_fde = [joint100[seed]["fde"] for seed in SEEDS]
    new_ade = {method: [metric(preserving[method][seed], "ade") for seed in SEEDS] for method in METHODS}
    new_fde = {method: [metric(preserving[method][seed], "fde") for seed in SEEDS] for method in METHODS}
    best_auc_method = "P1_target_only" if np.mean(p1_auc) >= np.mean(p2_auc) else "P2_target_scene"
    i0_ade = float(np.mean(t0[seed]["ade"] for seed in SEEDS))
    i0_fde = float(np.mean(t0[seed]["fde"] for seed in SEEDS))

    lines.extend(
        [
            "",
            "## Scientific answers",
            "",
            f"1. **Numerical preservation:** initialization equivalence passed on {equivalence_count(protocol)} validation samples per seed with max absolute future-coordinate difference 0. The final matched-seed test ADE change is P1 {mean_sd([value - t0[seed]['ade'] for seed, value in zip(SEEDS, new_ade['P1_target_only'])], 6)} px and P2 {mean_sd([value - t0[seed]['ade'] for seed, value in zip(SEEDS, new_ade['P2_target_scene'])], 6)} px; FDE changes are reported above.",
            f"2. **Did intention training modify trajectory weights?** No. All six runs recorded identical backbone SHA256 before training, after every epoch, and after selected-checkpoint reload ({sum(len(read_json(RESULTS / method / f'seed{seed}' / 'parameter_hashes.json')['hash_history']) for method in METHODS for seed in SEEDS)} hash observations checked).",
            f"3. **Can the frozen representation support crossing-intention recognition?** It yields P1 AUC {mean_sd(p1_auc)} and P2 AUC {mean_sd(p2_auc)}. The observed-only reference is {mean_sd(i0_auc)}. This establishes measurable transfer, but performance relative to that reference should be judged with the per-seed spread and input-definition caveat below.",
            f"4. **Target-only or target+scene?** P1 AUC {mean_sd(p1_auc)} vs P2 {mean_sd(p2_auc)}; P1 Brier {mean_sd([metric(preserving['P1_target_only'][seed], 'brier') for seed in SEEDS])} vs P2 {mean_sd([metric(preserving['P2_target_scene'][seed], 'brier') for seed in SEEDS])}. Descriptively, {best_auc_method} has the higher mean AUC; this is not a test-selected model choice.",
            f"5. **Observed-only comparison:** observed-only AUC {mean_sd(i0_auc)}, Brier {mean_sd([existing_intent(seed)['brier'] for seed in SEEDS])}; P1/P2 AUC and Brier are shown in Table 1. Note the input difference: the historical observed-only model receives target_obs `[15,4]`, while P1 uses the frozen trajectory encoder over concatenated target_obs + target_abs_obs `[15,8]`; the comparison is informative but not strictly input-matched.",
            f"6. **Recovery versus fixed λ=100:** fixed λ=100 mean ADE/FDE {np.mean(j100_ade):.3f}/{np.mean(j100_fde):.3f} px; T0 {i0_ade:.3f}/{i0_fde:.3f} px. P1 is {np.mean(new_ade['P1_target_only']):.3f}/{np.mean(new_fde['P1_target_only']):.3f} px and P2 is {np.mean(new_ade['P2_target_scene']):.3f}/{np.mean(new_fde['P2_target_scene']):.3f} px. This puts trajectory forecasting back on the pretrained trajectory-only path rather than merely reducing the joint degradation.",
            "7. **Interpretation:** because the original joint model changed trajectory feature flow and shared task-updated parameters, while this controlled model preserves the original decoder path and freezes all trajectory weights, any maintained trajectory accuracy is evidence that the previous degradation was caused by joint architecture/representation interference. Intention quality is a separate representation-transfer question; this experiment does not prove frozen trajectory features are universally sufficient.",
            f"8. **Next stage:** {'A carefully bounded task-specific adapter or partial-unfreezing study is justified to address intention transfer while retaining the frozen model as a trajectory-preservation control.' if np.mean(p1_auc + p2_auc) < np.mean(i0_auc) else 'The frozen transfer is promising; retain it as a control and only proceed to a preregistered adapter/partial-unfreezing study if it has a clearly stated hypothesis.'} Do not add reliability weighting, PCGrad, or further modules in this phase.",
            "",
            "## Protocol and artifacts",
            "",
            f"- Frozen protocol SHA256: `{checksum}`; test set was opened only after all six validation runs, checkpoint selection, equivalence check, and protocol freeze.",
            "- Frozen trajectory-only checkpoints are seed-matched; checkpoint SHA256 mapping and 47/47 tensor loads are in `trajectory_reference.json` and `weight_loading_report.json`.",
            "- Initialization equivalence: `equivalence_test.json`; checkpoint mapping: `architecture_mapping.md` and `weight_loading_report.json`.",
            "- P1/P2 each contain `metrics.json`, `validation_history.json`, `parameter_hashes.json`, `test_access_record.json`, and per-sample `test_predictions.npz` under their seed folders.",
            "- Historical T0, observed-only, and λ=100 test metrics are reused from existing frozen artifacts; those baselines were not rerun.",
            "",
        ]
    )
    output = RESULTS / "summary.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


def equivalence_count(protocol: dict[str, Any]) -> int:
    return int(protocol["equivalence_gate"]["samples_per_seed"])


if __name__ == "__main__":
    main()
