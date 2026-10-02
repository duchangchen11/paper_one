#!/usr/bin/env python3
"""Summarize the matched temporal baselines without loading any dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.mamba_jaad_utils import CONFIG_PATH, RESULTS_ROOT, SEEDS, write_json
from scripts.trajectory_preserving_utils import sha256_file
from src.models.intention_scratch_transformer import IntentionScratchTransformer


def describe(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "per_seed": dict(zip(map(str, SEEDS), map(float, values)))}


def read_runs(method):
    runs = []
    for seed in SEEDS:
        path = RESULTS_ROOT / method / f"seed{seed}" / "metrics_validation.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["test_accessed"] is False and data["all_losses_and_gradients_finite"] is True
        assert data["seed"] == seed
        history_path = path.parent / "validation_history.json"
        assert len(json.loads(history_path.read_text())["epochs"]) == 20
        runs.append(data)
    return runs


def compare_trajectory(transformer, mamba, rules):
    result = {"per_seed": {}}
    for tt, mt in zip(transformer, mamba):
        assert tt["data_provenance"] == mt["data_provenance"]
        assert tt["initialization"]["decoder_initialization_sha256"] == mt["initialization"]["decoder_initialization_sha256"]
        result["per_seed"][str(tt["seed"])] = {
            key: mt["selected_validation_metrics"][key] - tt["selected_validation_metrics"][key]
            for key in ("ade_pixel", "fde_pixel")
        }
    gates = []
    for key in ("ade_pixel", "fde_pixel"):
        tt = [r["selected_validation_metrics"][key] for r in transformer]
        mt = [r["selected_validation_metrics"][key] for r in mamba]
        relative = (float(np.mean(tt)) - float(np.mean(mt))) / float(np.mean(tt))
        improved_seeds = sum(m < t for t, m in zip(tt, mt))
        gate = relative >= rules["trajectory_minimum_mean_relative_improvement"] and improved_seeds >= rules["trajectory_minimum_improved_seed_count"]
        result[key] = {"tt": describe(tt), "mt": describe(mt), "delta_mt_minus_tt": describe(np.asarray(mt)-np.asarray(tt)),
                       "relative_mean_improvement": relative, "improved_seed_count": improved_seeds, "passes": gate}
        gates.append(gate)
    result["decision"] = "GO" if any(gates) else "STOP"
    return result


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    methods = ("trajectory_transformer_target", "trajectory_mamba", "intention_mamba")
    runs = {method: read_runs(method) for method in methods}
    sweep_path = RESULTS_ROOT / "trajectory_mamba_layers2"
    if sweep_path.exists():
        runs["trajectory_mamba_layers2"] = read_runs("trajectory_mamba_layers2")
    m0 = []
    for seed in SEEDS:
        path = ROOT / "results/intention_scratch_matched" / f"seed{seed}" / "metrics_validation.json"
        m0.append({"seed": seed, "metrics": json.loads(path.read_text())["selected_validation_raw_metrics"],
                   "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)})
    trajectory_comparison = {
        method: compare_trajectory(runs["trajectory_transformer_target"], runs[method], config["decision_rules"])
        for method in ("trajectory_mamba", "trajectory_mamba_layers2") if method in runs
    }
    mi_auc = [r["selected_validation_metrics"]["roc_auc"] for r in runs["intention_mamba"]]
    m0_auc = [r["metrics"]["roc_auc"] for r in m0]
    auc_delta = float(np.mean(mi_auc) - np.mean(m0_auc))
    intent_go = auc_delta >= -config["decision_rules"]["intention_maximum_mean_auc_degradation"]
    selected_mt = min(trajectory_comparison, key=lambda key: trajectory_comparison[key]["ade_pixel"]["mt"]["mean"])
    comparisons = {
        "trajectory": trajectory_comparison, "intention": {"mi_auc": describe(mi_auc), "m0_auc": describe(m0_auc),
        "mean_delta_mi_minus_m0": auc_delta, "decision": "GO" if intent_go else "STOP"},
        "selected_validation_trajectory_variant": selected_mt,
        "mamba_trajectory_decision": "GO" if any(c["decision"] == "GO" for c in trajectory_comparison.values()) else "STOP",
        "mamba_intention_decision": "GO" if intent_go else "STOP",
        "needs_layer2_sweep": trajectory_comparison["trajectory_mamba"]["decision"] == "STOP" and "trajectory_mamba_layers2" not in runs,
        "m0_validation_reference": m0, "test_accessed": False, "intent_guided_stage_started": False,
    }
    write_json(RESULTS_ROOT / "comparison.json", comparisons)
    lines = ["# JAAD Mamba baseline validation audit", "", "All new experiments use the existing train/val arrays only. No test data were opened. Existing frozen results are unchanged.", "",
             "## Protocol", "", "Target input: concat(target_obs, target_abs_obs), [B,15,8]. TT and MT share exactly the same decoder design and per-seed decoder initialization. All baselines train from scratch, with no scene, social, reliability, or intention conditioning in trajectory models.", "",
             "Trajectory: 20 epochs, batch 512, AdamW lr=1e-3/wd=1e-4, clip=5, SmoothL1, ReduceLROnPlateau(factor=0.5, patience=2), select minimum validation pixel ADE. Intention: 20 epochs, batch 256, natural shuffle, inverse-frequency weighted BCE; select highest raw AUC with the same 1e-4 Brier tie-break as M0.", "",
             "## Selected validation results", "", "Means ± sample SD across the three seeds; intention metrics in this table are raw (threshold=0.5).", "",
             "| Method | Parameters | Val AUC | Val Brier | Val ADE (px) | Val FDE (px) |", "|---|---:|---:|---:|---:|---:|"]
    def text_stat(values):
        s=describe(values)
        return f"{s['mean']:.4f} ± {s['std']:.4f}"
    m0_params = sum(p.numel() for p in IntentionScratchTransformer().parameters())
    lines.append(f"| M0 Transformer (existing) | {m0_params} | {text_stat(m0_auc)} | {text_stat([r['metrics']['brier'] for r in m0])} | — | — |")
    for method, items in runs.items():
        params = items[0]["parameter_count"]["total"]
        is_intent = method == "intention_mamba"
        metrics = [r["selected_validation_metrics"] for r in items]
        auc = text_stat([r["roc_auc"] for r in metrics]) if is_intent else "—"
        brier = text_stat([r["brier"] for r in metrics]) if is_intent else "—"
        ade = text_stat([r["ade_pixel"] for r in metrics]) if not is_intent else "—"
        fde = text_stat([r["fde_pixel"] for r in metrics]) if not is_intent else "—"
        lines.append(f"| {method} | {params} | {auc} | {brier} | {ade} | {fde} |")
    lines += ["", "## Per-seed trajectory comparison", "", "Delta = MT − TT; negative error delta favors MT.", "", "| Variant | Seed | TT ADE | TT FDE | MT ADE | MT FDE | Delta ADE | Delta FDE |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, comparison in trajectory_comparison.items():
        for tt, mt in zip(runs["trajectory_transformer_target"], runs[method]):
            t, m = tt["selected_validation_metrics"], mt["selected_validation_metrics"]
            lines.append(f"| {method} | {tt['seed']} | {t['ade_pixel']:.3f} | {t['fde_pixel']:.3f} | {m['ade_pixel']:.3f} | {m['fde_pixel']:.3f} | {m['ade_pixel']-t['ade_pixel']:.3f} | {m['fde_pixel']-t['fde_pixel']:.3f} |")
        ade, fde = comparison["ade_pixel"], comparison["fde_pixel"]
        lines += ["", f"{method}: mean delta ADE={ade['delta_mt_minus_tt']['mean']:.3f}px; FDE={fde['delta_mt_minus_tt']['mean']:.3f}px. Mean relative improvement: ADE={ade['relative_mean_improvement']:.2%} ({ade['improved_seed_count']}/3 seeds better), FDE={fde['relative_mean_improvement']:.2%} ({fde['improved_seed_count']}/3 better). Gate: **{comparison['decision']}**.", ""]
    lines += ["## Intention metrics", "", "| Seed | MI raw AUC | MI raw Brier | MI raw F1 | MI raw BAcc | MI raw Accuracy | M0 raw AUC |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for run, ref in zip(runs["intention_mamba"], m0):
        m=run["selected_validation_metrics"]
        lines.append(f"| {run['seed']} | {m['roc_auc']:.4f} | {m['brier']:.4f} | {m['f1']:.4f} | {m['balanced_accuracy']:.4f} | {m['accuracy']:.4f} | {ref['metrics']['roc_auc']:.4f} |")
    lines += ["", f"Mean MI − M0 validation AUC: {auc_delta:+.4f}. MI gate: **{'GO' if intent_go else 'STOP'}**; GO here means no degradation exceeding 0.02, not proof of superiority. Validation-calibrated metrics are also retained in each run and follow the same procedure as M0.", "",
              "## Speed and memory", "", "CUDA event timing after warmup, FP32, eval mode, excludes host-to-device transfers; measured 30 repetitions per seed. Batch 512 for TT/MT and 256 for MI. Sequence length is only 15; these measurements do not establish asymptotic-complexity benefits. The optional causal-conv1d package is absent; Mamba uses the officially supported PyTorch convolution fallback plus compiled selective-scan CUDA kernels. Timings describe this installation, not an optimally fused installation.", "", "| Method | Single sample (ms) | Batch latency (ms) | Peak train allocated GPU memory (MiB) |", "|---|---:|---:|---:|"]
    for method, items in runs.items():
        benchmarks=[json.loads((RESULTS_ROOT/method/f"seed{r['seed']}"/"inference_benchmark.json").read_text()) for r in items]
        lines.append(f"| {method} | {text_stat([b['measurements']['single_sample']['mean_ms'] for b in benchmarks])} | {text_stat([b['measurements']['batch']['mean_ms'] for b in benchmarks])} | {max(r['training_gpu_peak_allocated_mb'] for r in items):.1f} |")
    lines += ["", "## Decision and scope", "", f"- Mamba trajectory: **{comparisons['mamba_trajectory_decision']}**. Rule: ≥2% mean improvement in ADE or FDE with ≥2/3 seeds improving the same metric.",
              f"- Mamba intention: **{comparisons['mamba_intention_decision']}**. Rule: mean raw AUC must not decrease by more than 0.02 versus existing input-matched M0.",
              f"- Permitted two-layer trajectory check needed: {comparisons['needs_layer2_sweep']}.",
              "- All completed runs have finite losses/gradients/predictions; no NaN or Inf detected.",
              "- Hidden dimensions, input, decoder, seeds, data order, and trajectory optimizer/schedule are matched; total parameter counts differ between temporal encoder families.",
              "- Validation selection is exploratory. No paper contribution, intent-guidance benefit, or test generalization is established by this baseline stage.",
              "- Intent-Guided G1/G2/G3 and protocol freeze are deferred to a later instruction.", ""]
    (RESULTS_ROOT/"summary.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(comparisons,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
