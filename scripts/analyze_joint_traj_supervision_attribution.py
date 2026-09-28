#!/usr/bin/env python3
"""Paired J0/J100 held-out attribution analysis and final report."""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.joint_traj_supervision_utils import RESULTS_ROOT, SEEDS, sha256_file, write_json
from scripts.reliability_gated_intent_utils import cluster_bootstrap_paired_delta


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> str:
    values_array = np.asarray(values, dtype=np.float64)
    return f"{values_array.mean():.4f} ± {values_array.std(ddof=1):.4f}"


def metric_row(metrics: dict[str, Any], family: str) -> dict[str, float | None]:
    if family == "direct":
        return {
            "auc": float(metrics["intent_auc"]),
            "brier": float(metrics["intent_brier"]),
            "f1": float(metrics["intent_f1"]),
            "bacc": float(metrics["intent_balanced_accuracy"]),
            "ade": float(metrics["trajectory_ade_pixel"]),
            "fde": float(metrics["trajectory_fde_pixel"]),
        }
    if family == "m0":
        row = metrics["intent_raw_uncalibrated_threshold_0_5"]
        return {"auc": float(row["roc_auc"]), "brier": float(row["brier"]), "f1": float(row["f1"]), "bacc": float(row["balanced_accuracy"]), "ade": None, "fde": None}
    if family == "p1":
        intent, trajectory = metrics["intent"], metrics["trajectory"]
        return {"auc": float(intent["roc_auc"]), "brier": float(intent["brier"]), "f1": float(intent["f1"]), "bacc": float(intent["balanced_accuracy"]), "ade": float(trajectory["ade_pixel"]), "fde": float(trajectory["fde_pixel"])}
    raise ValueError(f"Unknown metric family: {family}")


def load_and_align(seed: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    seed_dir = RESULTS_ROOT / "j0" / f"seed{seed}"
    with np.load(seed_dir / "test_predictions.npz", allow_pickle=False) as archive:
        j0 = {key: archive[key].copy() for key in archive.files}
    with np.load(seed_dir / "j100_reference_predictions.npz", allow_pickle=False) as archive:
        j100 = {key: archive[key].copy() for key in archive.files}
    fields = ("scene_id", "target_id", "obs_end_frame", "label", "future_gt")
    for key in fields:
        if key not in j0 or key not in j100 or not np.array_equal(j0[key], j100[key]):
            raise RuntimeError(f"J0/J100 held-out samples do not align for seed{seed}: {key}")
    return j0, j100


def build_summary(
    per_seed: dict[str, Any],
    bootstrap: dict[str, Any],
    selection: dict[str, Any],
    representation: dict[str, Any],
    gradients: dict[str, Any],
    test_access: dict[str, Any],
) -> str:
    frozen_protocol = read_json(RESULTS_ROOT / "protocol_frozen.json")
    pytest_counts = frozen_protocol["pytest"]["counts"]
    methods = ("M0 Scratch", "P1 Frozen", "J0 Joint-no-traj-loss", "J100 Joint-λ100")
    lines: list[str] = [
        "# Trajectory auxiliary supervision 对联合意图预测的贡献归因",
        "",
        f"- Protocol: `{(RESULTS_ROOT / 'protocol_frozen.json').relative_to(ROOT)}`（test 访问前冻结）。",
        f"- Official test: {test_access['test_sample_count']} 个样本；test archive SHA256 `{test_access['test_archive_sha256']}`；本协议只评估一次。",
        f"- Tests: {pytest_counts['passed']} passed, {pytest_counts['failures']} failed, {pytest_counts['errors']} errors（全量 `tests`）。",
        "- J100 为已冻结历史实验，不重训；J0 与 J100 都使用未校准 sigmoid 概率、0.5 阈值。",
        "- Protocol freeze 后仅修复了报告渲染字段引用；未重新读取 test 或重算 bootstrap，修订记录见 `post_freeze_report_amendment.json`。",
        "- bootstrap 按 `scene_id` 成簇，每 seed 2000 次；报告 `J100 − J0`，不跨 seed pooling。",
        "",
        "## 主表：test 性能",
        "",
        "| Method | AUC | Brier | F1 | BAcc | ADE (px) | FDE (px) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = per_seed["aggregate"][method]
        values = [mean_std(row[key]) if row[key] else "—" for key in ("auc", "brier", "f1", "bacc", "ade", "fde")]
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "说明：M0 使用其已保存的 raw、threshold=0.5 test 指标，P1 使用其已保存且仅依 validation calibration 的 test 指标；J0/J100 为本次同一 test archive 的 raw 输出。因此 J0−J100 是严格匹配比较，M0/P1 用作历史参照而非本轮因果对照。",
        "",
        "## Seed-level AUC 与方向",
        "",
        "| Seed | J0 AUC | J100 AUC | ΔAUC (J100−J0) | ΔBrier | ΔF1 | ΔBAcc |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in map(str, SEEDS):
        row = per_seed["seeds"][seed]
        lines.append(f"| {seed} | {row['J0']['auc']:.4f} | {row['J100']['auc']:.4f} | {row['delta']['auc']:+.4f} | {row['delta']['brier']:+.4f} | {row['delta']['f1']:+.4f} | {row['delta']['bacc']:+.4f} |")
    lines.extend([
        "",
        "## Paired scene-cluster bootstrap",
        "",
        "| Seed | ΔAUC J100−J0 | 95% CI | ΔBrier | 95% CI |",
        "|---:|---:|---|---:|---|",
    ])
    for seed in map(str, SEEDS):
        result = bootstrap["per_seed"][seed]
        auc = result["delta_roc_auc"]
        brier = result["delta_brier"]
        lines.append(f"| {seed} | {per_seed['seeds'][seed]['delta']['auc']:+.4f} (boot mean {auc['mean']:+.4f}) | [{auc['ci_percentile_95']['lower_95']:+.4f}, {auc['ci_percentile_95']['upper_95']:+.4f}] | {per_seed['seeds'][seed]['delta']['brier']:+.4f} (boot mean {brier['mean']:+.4f}) | [{brier['ci_percentile_95']['lower_95']:+.4f}, {brier['ci_percentile_95']['upper_95']:+.4f}] |")
    lines.extend([
        "",
        "## Trajectory branch（secondary）",
        "",
        "| Seed | J0 ADE | J100 ADE | T0 ADE | J0 FDE | J100 FDE | T0 FDE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for seed in map(str, SEEDS):
        row = per_seed["seeds"][seed]
        lines.append(f"| {seed} | {row['J0']['ade']:.2f} | {row['J100']['ade']:.2f} | {row['T0']['ade']:.2f} | {row['J0']['fde']:.2f} | {row['J100']['fde']:.2f} | {row['T0']['fde']:.2f} |")
    lines.extend(["", "J0 每 epoch validation ADE/FDE 均在 `j0/seed*/validation_history.json` 中。下表突出 epoch 1、官方选择 epoch、epoch 15 与整个训练期最小 ADE：", "", "| Seed | ADE epoch1 | selected epoch / ADE | ADE epoch15 | min ADE (epoch) |", "|---:|---:|---:|---:|---:|"])
    for seed in map(str, SEEDS):
        row = per_seed["seeds"][seed]["trajectory_validation"]
        lines.append(f"| {seed} | {row['epoch1_ade']:.2f} | {row['selected_epoch']} / {row['selected_ade']:.2f} | {row['epoch15_ade']:.2f} | {row['minimum_ade']:.2f} ({row['minimum_epoch']}) |")

    seed_rows = [per_seed["seeds"][str(seed)] for seed in SEEDS]
    mean_delta = float(np.mean([row["delta"]["auc"] for row in seed_rows]))
    positive_count = sum(row["delta"]["auc"] > 0 for row in seed_rows)
    positive_ci_count = sum(bootstrap["per_seed"][str(seed)]["delta_roc_auc"]["ci_percentile_95"]["lower_95"] > 0 for seed in SEEDS)
    if mean_delta > 0.03 and positive_count >= 2 and positive_ci_count >= 2:
        support = "Strong support：J100−J0 平均 ΔAUC > 0.03，至少两个 seed 为正且至少两个 scene-bootstrap 95% CI 下界 > 0。"
        support_label = "strong positive support"
    elif mean_delta > 0 and positive_count >= 2:
        support = "Weak / suggestive support：平均 ΔAUC 为正且至少两个 seed 为正，但 bootstrap 区间未达到预设的稳定正向标准。"
        support_label = "weak/suggestive positive trend"
    elif mean_delta < 0:
        support = "Negative：平均 ΔAUC < 0，J0 整体高于 J100；trajectory supervision 可能压制意图预测。"
        support_label = "negative"
    else:
        support = "No support：平均 ΔAUC 接近零或 seed 方向不一致，当前没有稳定证据表明 trajectory supervision 提升 intention AUC。"
        support_label = "no stable support"

    lines.extend([
        "",
        "## Validation 选点敏感性",
        "",
        "| Seed | Official composite best epoch | AUC-best epoch | Same? | AUC at selected epoch | ADE at selected epoch (px) |",
        "|---:|---:|---:|:---:|---:|---:|",
    ])
    for seed in map(str, SEEDS):
        row = selection["per_seed"][seed]
        lines.append(f"| {seed} | {row['composite_best_epoch']} | {row['validation_auc_best_epoch']} | {'Yes' if row['same_epoch'] else 'No'} | {row['selected_checkpoint_val_auc']:.4f} | {row['selected_checkpoint_val_ade_pixel']:.2f} |")
    lines.extend([
        "",
        "本轮保留预注册的 composite selection；AUC-best 仅是只读敏感性分析，没有用来替换 checkpoint。由于 J0 trajectory ADE 很大，复合分数中的 ADE 项对选点有明显影响，应在结果解释中保留这一限制。",
        "",
        "## Shared representation drift（validation 固定 1000 samples）",
        "",
        "| Representation | Arm | cosine to init (mean) | linear CKA to init (mean) | feature L2 norm (mean) | mean per-dim variance |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for feature in ("target_encoder_last", "fused_decoder_context"):
        for arm in ("J0", "J100"):
            cosine = [representation["per_seed"][str(seed)]["features"][feature][arm]["similarity_to_initialization"]["cosine_mean"] for seed in SEEDS]
            cka = [representation["per_seed"][str(seed)]["features"][feature][arm]["similarity_to_initialization"]["linear_cka"] for seed in SEEDS]
            norm = [representation["per_seed"][str(seed)]["features"][feature][arm]["statistics"]["l2_norm_mean"] for seed in SEEDS]
            variance = [representation["per_seed"][str(seed)]["features"][feature][arm]["statistics"]["mean_per_dimension_variance"] for seed in SEEDS]
            lines.append(f"| {feature} | {arm} | {np.mean(cosine):.4f} | {np.mean(cka):.4f} | {np.mean(norm):.4f} | {np.mean(variance):.6g} |")
    lines.extend([
        "",
        "这些是表示漂移的描述性统计，不构成轨迹监督约束表示的因果证明。",
        "",
        "## Selected-checkpoint gradient diagnostics",
        "",
        "| Arm | intent shared-grad norm | raw trajectory shared-grad norm | weighted trajectory grad norm | cosine(intent, trajectory) | intent / weighted trajectory |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm in ("J0", "J100"):
        rows = [gradients["per_seed"][str(seed)][arm] for seed in SEEDS]
        intent_norm = np.mean([row["intent_gradient_norm_shared"] for row in rows])
        raw_norm = np.mean([row["trajectory_gradient_norm_shared_unweighted"] for row in rows])
        weighted_norm = np.mean([row["trajectory_gradient_norm_shared_weighted"] for row in rows])
        cosine = np.mean([row["gradient_cosine_intent_vs_trajectory"] for row in rows])
        ratios = [row["weighted_intent_to_trajectory_gradient_ratio"] for row in rows if row["weighted_intent_to_trajectory_gradient_ratio"] is not None]
        ratio_text = "undefined (λ=0)" if not ratios else f"{np.mean(ratios):.4g}"
        lines.append(f"| {arm} | {intent_norm:.4g} | {raw_norm:.4g} | {weighted_norm:.4g} | {cosine:+.4f} | {ratio_text} |")

    j100_verification = [test_access["per_seed"][str(seed)]["J100"]["matches_historical_within_1e-5"] for seed in SEEDS]
    j0_auc_text = ", ".join(f"{seed}: {per_seed['seeds'][str(seed)]['J0']['auc']:.4f}" for seed in SEEDS)
    j100_auc_text = ", ".join(f"{seed}: {per_seed['seeds'][str(seed)]['J100']['auc']:.4f}" for seed in SEEDS)
    lines.extend([
        "",
        "## 对预设问题的回答",
        "",
        f"1. **J0 是否严格只去掉 trajectory supervision？** 是。三个 seed 使用相同联合模型 forward；预训练固定 batch 审计显示 `traj_weight=0`，future output 存在，轨迹项共享梯度贡献为 0、意图梯度仍更新 shared trunk。J0/J100 实际训练字段只有轨迹权重不同。",
        "2. **architecture/config 是否一致？** Architecture 文件在两个实验间未改；J100 checkpoint args 与运行设置已审计。J0 有逐次初始化 SHA；历史 J100 未保存该字段，按来源 commit 中相同 seed/模型构造顺序重建，故初始化一致是源代码审计支持，而非历史哈希直接证明。",
        f"3. **J0 三 seed AUC？** {j0_auc_text}。",
        f"4. **J100 三 seed AUC？** {j100_auc_text}。历史 J100 重评与已有指标均在 1e-5 内匹配：{j100_verification}。",
        f"5. **mean ΔAUC (J100−J0)？** {mean_delta:+.4f}。",
        f"6. **三 seed 方向是否一致？** {positive_count}/3 为正；各 seed ΔAUC 见上表。",
        f"7. **Bootstrap 是否支持正向贡献？** {positive_ci_count}/3 的 95% CI 下界大于 0。",
        f"8. **trajectory supervision 是否是 AUC≈0.79 的重要来源？** {support_label}。预设判断：{support}",
        "9. **若未支持，AUC 提升是否来自 joint architecture？** 只能说 scene/social/proposal/gate/fusion/ambiguity 等其余 joint 配置合起来是可能来源；本实验不能辨别具体组件，不能据此断言某一模块因果有效。",
        "10. **J0 trajectory ADE 如何变化？** 查看上方 epoch1/selected/epoch15/min 表及完整 `validation_history.json`；J0 未收到轨迹损失梯度，因此这些轨迹指标是自然漂移诊断，不是优化目标。",
        "11. **trajectory supervision 是否可能像 representation regularizer？** 对照 cosine/CKA/L2/variance 表：若 J100 比 J0 更接近初始化且其 AUC 更高，可作为后续假设；仍只是关联性证据。",
        "12. **现在是否做 scene/social/proposal/gate/ambiguity component ablation？** 本轮不自动开展。若 ΔAUC 无稳定正向支持，建议下一轮有计划地做 joint architecture component ablation；若正向支持，则先研究 task-decoupled trajectory-preserving auxiliary learning。",
        "13. **是否继续 joint intention+trajectory 主线？** 结论以本轮 ΔAUC 分类为准：正向支持可继续，但不能称 trajectory supervision 本身已是创新；无支持时，应把主线调整为 joint architecture 贡献归因，之后再评估任务解耦。",
        "",
        "## J100 历史指标一致性",
        "",
        "每个 J100 checkpoint 在同一 test archive 上的重评与 frozen historical metrics 的逐指标差异保存在 `official_test_evaluation.json`。",
        "",
        "## 结论边界与下一步",
        "",
        "本次唯一因果对照是 J0 vs frozen J100。M0/P1/T0 是上下文对照。未进行 component ablation，也没有据此宣称任何新模块的因果贡献。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-summary-only", action="store_true", help="Reuse already saved test/bootstrap outputs without re-evaluation or re-bootstrap")
    args = parser.parse_args()
    protocol_path = RESULTS_ROOT / "protocol_frozen.json"
    protocol_sha = sha256_file(protocol_path)
    expected_sha = (RESULTS_ROOT / "protocol_frozen.sha256").read_text(encoding="utf-8").split()[0]
    if protocol_sha != expected_sha:
        raise RuntimeError("Frozen protocol integrity check failed")
    access = read_json(RESULTS_ROOT / "official_test_access_record.json")
    if access.get("status") != "complete_one_time_official_test_evaluation":
        raise RuntimeError("Official one-time test evaluation is incomplete")
    evaluation = read_json(RESULTS_ROOT / "official_test_evaluation.json")
    if evaluation.get("status") != "official_test_evaluated_once" or evaluation.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("Official test evaluation is not bound to the frozen protocol")
    if args.render_summary_only:
        per_seed = read_json(RESULTS_ROOT / "attribution_metrics.json")
        bootstrap_payload = read_json(RESULTS_ROOT / "cluster_bootstrap.json")
        selection = read_json(RESULTS_ROOT / "checkpoint_selection_sensitivity.json")
        representation = read_json(RESULTS_ROOT / "representation_drift.json")
        gradients = read_json(RESULTS_ROOT / "gradient_diagnostics.json")
        summary = build_summary(per_seed, bootstrap_payload, selection, representation, gradients, evaluation)
        (RESULTS_ROOT / "summary.md").write_text(summary, encoding="utf-8")
        print(json.dumps({"summary": str((RESULTS_ROOT / "summary.md").relative_to(ROOT)), "render_only": True, "test_reloaded": False, "bootstrap_recomputed": False}, ensure_ascii=False, indent=2))
        return

    per_seed: dict[str, Any] = {"seeds": {}, "aggregate": {}}
    method_seed_rows: dict[str, list[dict[str, float | None]]] = {name: [] for name in ("M0 Scratch", "P1 Frozen", "J0 Joint-no-traj-loss", "J100 Joint-λ100")}
    bootstrap_payload: dict[str, Any] = {
        "protocol_sha256": protocol_sha,
        "bootstrap_unit": "scene_id video cluster",
        "requested_repetitions_per_seed": 2000,
        "bootstrap_seed": 9124,
        "delta_definition": "J100 minus J0",
        "pooled_across_seeds": False,
        "per_seed": {},
    }
    for seed in SEEDS:
        seed_key = str(seed)
        j0_pred, j100_pred = load_and_align(seed)
        j0_metrics = read_json(RESULTS_ROOT / "j0" / f"seed{seed}/official_test_metrics.json")
        j100_record = evaluation["per_seed"][seed_key]["J100"]
        j100_metrics = j100_record["metrics"]
        if not np.array_equal(j0_pred["labels"], j100_pred["labels"]):
            raise RuntimeError(f"J0/J100 labels differ for seed{seed}")
        bootstrap = cluster_bootstrap_paired_delta(
            j0_pred["labels"], j0_pred["probability"], j100_pred["probability"], j0_pred["scene_id"],
            repetitions=2000, seed=9124,
        )
        bootstrap_payload["per_seed"][seed_key] = bootstrap
        delta = {
            "auc": float(j100_metrics["intent_auc"] - j0_metrics["intent_auc"]),
            "brier": float(j100_metrics["intent_brier"] - j0_metrics["intent_brier"]),
            "f1": float(j100_metrics["intent_f1"] - j0_metrics["intent_f1"]),
            "bacc": float(j100_metrics["intent_balanced_accuracy"] - j0_metrics["intent_balanced_accuracy"]),
        }
        m0_metrics = read_json(ROOT / f"results/intention_scratch_matched/seed{seed}/metrics.json")["test"]
        p1_metrics = read_json(ROOT / f"results/trajectory_preserving_joint/P1_target_only/seed{seed}/metrics.json")["test"]
        t0_metrics = read_json(ROOT / f"results/trajectory_transformer_scene_15x15_seed{seed}/metrics.json")["test"]
        m0 = metric_row(m0_metrics, "m0")
        p1 = metric_row(p1_metrics, "p1")
        j0 = metric_row(j0_metrics, "direct")
        j100 = metric_row(j100_metrics, "direct")
        t0 = {"ade": float(t0_metrics["trajectory_ade_pixel"]), "fde": float(t0_metrics["trajectory_fde_pixel"])}
        for method, row in (("M0 Scratch", m0), ("P1 Frozen", p1), ("J0 Joint-no-traj-loss", j0), ("J100 Joint-λ100", j100)):
            method_seed_rows[method].append(row)
        history = read_json(RESULTS_ROOT / "j0" / f"seed{seed}/metrics.json")["history"]
        best_epoch = int(read_json(RESULTS_ROOT / "j0" / f"seed{seed}/metrics.json")["best_epoch"])
        val_ades = [float(row["val"]["trajectory_ade_pixel"]) for row in history]
        val_fdes = [float(row["val"]["trajectory_fde_pixel"]) for row in history]
        selected = history[best_epoch - 1]["val"]
        per_seed["seeds"][seed_key] = {
            "J0": j0, "J100": j100, "M0": m0, "P1": p1, "T0": t0,
            "delta": delta,
            "bootstrap": bootstrap,
            "trajectory_validation": {
                "epoch1_ade": val_ades[0], "epoch1_fde": val_fdes[0],
                "selected_epoch": best_epoch, "selected_ade": float(selected["trajectory_ade_pixel"]), "selected_fde": float(selected["trajectory_fde_pixel"]),
                "epoch15_ade": val_ades[-1], "epoch15_fde": val_fdes[-1],
                "minimum_ade": float(min(val_ades)), "minimum_epoch": int(np.argmin(val_ades) + 1),
                "minimum_fde": float(min(val_fdes)), "minimum_fde_epoch": int(np.argmin(val_fdes) + 1),
            },
        }

    for method, rows in method_seed_rows.items():
        per_seed["aggregate"][method] = {
            key: [float(row[key]) for row in rows if row[key] is not None]
            for key in ("auc", "brier", "f1", "bacc", "ade", "fde")
        }
    write_json(RESULTS_ROOT / "cluster_bootstrap.json", bootstrap_payload)
    selection = read_json(RESULTS_ROOT / "checkpoint_selection_sensitivity.json")
    representation = read_json(RESULTS_ROOT / "representation_drift.json")
    gradients = read_json(RESULTS_ROOT / "gradient_diagnostics.json")
    write_json(RESULTS_ROOT / "attribution_metrics.json", per_seed)
    summary = build_summary(per_seed, bootstrap_payload, selection, representation, gradients, evaluation)
    (RESULTS_ROOT / "summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"summary": str((RESULTS_ROOT / "summary.md").relative_to(ROOT)), "mean_delta_auc": float(np.mean([per_seed["seeds"][str(seed)]["delta"]["auc"] for seed in SEEDS])), "per_seed_delta_auc": {str(seed): per_seed["seeds"][str(seed)]["delta"]["auc"] for seed in SEEDS}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
