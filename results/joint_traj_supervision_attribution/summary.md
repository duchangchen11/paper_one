# Trajectory auxiliary supervision 对联合意图预测的贡献归因

- Protocol: `results/joint_traj_supervision_attribution/protocol_frozen.json`（test 访问前冻结）。
- Official test: 18331 个样本；test archive SHA256 `23658152706d460493fa6146f6b401648a509cc8708ab5eb6c0892e300de1935`；本协议只评估一次。
- Tests: 124 passed, 0 failed, 0 errors（全量 `tests`）。
- J100 为已冻结历史实验，不重训；J0 与 J100 都使用未校准 sigmoid 概率、0.5 阈值。
- Protocol freeze 后仅修复了报告渲染字段引用；未重新读取 test 或重算 bootstrap，修订记录见 `post_freeze_report_amendment.json`。
- bootstrap 按 `scene_id` 成簇，每 seed 2000 次；报告 `J100 − J0`，不跨 seed pooling。

## 主表：test 性能

| Method | AUC | Brier | F1 | BAcc | ADE (px) | FDE (px) |
|---|---:|---:|---:|---:|---:|---:|
| M0 Scratch | 0.6732 ± 0.0079 | 0.2402 ± 0.0359 | 0.4882 ± 0.4578 | 0.5422 ± 0.0693 | — | — |
| P1 Frozen | 0.6878 ± 0.0262 | 0.2131 ± 0.0024 | 0.7633 ± 0.0211 | 0.6383 ± 0.0339 | 11.0107 ± 0.1785 | 19.4592 ± 0.4057 |
| J0 Joint-no-traj-loss | 0.7544 ± 0.0053 | 0.0975 ± 0.0042 | 0.9335 ± 0.0033 | 0.5820 ± 0.0368 | 527.1868 ± 102.3785 | 487.3561 ± 197.7073 |
| J100 Joint-λ100 | 0.7914 ± 0.0135 | 0.0994 ± 0.0004 | 0.9368 ± 0.0072 | 0.6239 ± 0.0249 | 15.8904 ± 1.4719 | 26.9228 ± 2.7406 |

说明：M0 使用其已保存的 raw、threshold=0.5 test 指标，P1 使用其已保存且仅依 validation calibration 的 test 指标；J0/J100 为本次同一 test archive 的 raw 输出。因此 J0−J100 是严格匹配比较，M0/P1 用作历史参照而非本轮因果对照。

## Seed-level AUC 与方向

| Seed | J0 AUC | J100 AUC | ΔAUC (J100−J0) | ΔBrier | ΔF1 | ΔBAcc |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.7540 | 0.7795 | +0.0255 | +0.0059 | +0.0044 | +0.0738 |
| 123 | 0.7493 | 0.8061 | +0.0568 | -0.0028 | -0.0017 | +0.0449 |
| 2024 | 0.7598 | 0.7886 | +0.0288 | +0.0025 | +0.0074 | +0.0072 |

## Paired scene-cluster bootstrap

| Seed | ΔAUC J100−J0 | 95% CI | ΔBrier | 95% CI |
|---:|---:|---|---:|---|
| 42 | +0.0255 (boot mean +0.0249) | [-0.0485, +0.1072] | +0.0059 (boot mean +0.0059) | [-0.0089, +0.0213] |
| 123 | +0.0568 (boot mean +0.0574) | [+0.0160, +0.1099] | -0.0028 (boot mean -0.0029) | [-0.0194, +0.0144] |
| 2024 | +0.0288 (boot mean +0.0295) | [-0.0448, +0.1114] | +0.0025 (boot mean +0.0022) | [-0.0145, +0.0181] |

## Trajectory branch（secondary）

| Seed | J0 ADE | J100 ADE | T0 ADE | J0 FDE | J100 FDE | T0 FDE |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 452.23 | 14.76 | 11.16 | 300.53 | 25.96 | 19.78 |
| 123 | 643.83 | 17.55 | 11.06 | 694.39 | 30.01 | 19.59 |
| 2024 | 485.50 | 15.36 | 10.81 | 467.15 | 24.79 | 19.00 |

J0 每 epoch validation ADE/FDE 均在 `j0/seed*/validation_history.json` 中。下表突出 epoch 1、官方选择 epoch、epoch 15 与整个训练期最小 ADE：

| Seed | ADE epoch1 | selected epoch / ADE | ADE epoch15 | min ADE (epoch) |
|---:|---:|---:|---:|---:|
| 42 | 460.16 | 1 / 460.16 | 664.93 | 460.16 (1) |
| 123 | 649.65 | 2 / 600.50 | 868.34 | 600.50 (2) |
| 2024 | 545.61 | 2 / 487.63 | 790.13 | 487.63 (2) |

## Validation 选点敏感性

| Seed | Official composite best epoch | AUC-best epoch | Same? | AUC at selected epoch | ADE at selected epoch (px) |
|---:|---:|---:|:---:|---:|---:|
| 42 | 1 | 3 | No | 0.8155 | 460.16 |
| 123 | 2 | 1 | No | 0.8578 | 600.50 |
| 2024 | 2 | 1 | No | 0.8368 | 487.63 |

本轮保留预注册的 composite selection；AUC-best 仅是只读敏感性分析，没有用来替换 checkpoint。由于 J0 trajectory ADE 很大，复合分数中的 ADE 项对选点有明显影响，应在结果解释中保留这一限制。

## Shared representation drift（validation 固定 1000 samples）

| Representation | Arm | cosine to init (mean) | linear CKA to init (mean) | feature L2 norm (mean) | mean per-dim variance |
|---|---|---:|---:|---:|---:|
| target_encoder_last | J0 | 0.1765 | 0.7852 | 16.5240 | 0.695373 |
| target_encoder_last | J100 | 0.2582 | 0.6790 | 19.0388 | 1.47496 |
| fused_decoder_context | J0 | 0.2523 | 0.4938 | 10.0897 | 0.198324 |
| fused_decoder_context | J100 | 0.1439 | 0.4758 | 11.0327 | 0.461606 |

这些是表示漂移的描述性统计，不构成轨迹监督约束表示的因果证明。

## Selected-checkpoint gradient diagnostics

| Arm | intent shared-grad norm | raw trajectory shared-grad norm | weighted trajectory grad norm | cosine(intent, trajectory) | intent / weighted trajectory |
|---|---:|---:|---:|---:|---:|
| J0 | 12.26 | 0.4706 | 0 | +0.0245 | undefined (λ=0) |
| J100 | 29.78 | 0.0003648 | 0.03648 | +0.1173 | 921.7 |

## 对预设问题的回答

1. **J0 是否严格只去掉 trajectory supervision？** 是。三个 seed 使用相同联合模型 forward；预训练固定 batch 审计显示 `traj_weight=0`，future output 存在，轨迹项共享梯度贡献为 0、意图梯度仍更新 shared trunk。J0/J100 实际训练字段只有轨迹权重不同。
2. **architecture/config 是否一致？** Architecture 文件在两个实验间未改；J100 checkpoint args 与运行设置已审计。J0 有逐次初始化 SHA；历史 J100 未保存该字段，按来源 commit 中相同 seed/模型构造顺序重建，故初始化一致是源代码审计支持，而非历史哈希直接证明。
3. **J0 三 seed AUC？** 42: 0.7540, 123: 0.7493, 2024: 0.7598。
4. **J100 三 seed AUC？** 42: 0.7795, 123: 0.8061, 2024: 0.7886。历史 J100 重评与已有指标均在 1e-5 内匹配：[True, True, True]。
5. **mean ΔAUC (J100−J0)？** +0.0370。
6. **三 seed 方向是否一致？** 3/3 为正；各 seed ΔAUC 见上表。
7. **Bootstrap 是否支持正向贡献？** 1/3 的 95% CI 下界大于 0。
8. **trajectory supervision 是否是 AUC≈0.79 的重要来源？** weak/suggestive positive trend。预设判断：Weak / suggestive support：平均 ΔAUC 为正且至少两个 seed 为正，但 bootstrap 区间未达到预设的稳定正向标准。
9. **若未支持，AUC 提升是否来自 joint architecture？** 只能说 scene/social/proposal/gate/fusion/ambiguity 等其余 joint 配置合起来是可能来源；本实验不能辨别具体组件，不能据此断言某一模块因果有效。
10. **J0 trajectory ADE 如何变化？** 查看上方 epoch1/selected/epoch15/min 表及完整 `validation_history.json`；J0 未收到轨迹损失梯度，因此这些轨迹指标是自然漂移诊断，不是优化目标。
11. **trajectory supervision 是否可能像 representation regularizer？** 对照 cosine/CKA/L2/variance 表：若 J100 比 J0 更接近初始化且其 AUC 更高，可作为后续假设；仍只是关联性证据。
12. **现在是否做 scene/social/proposal/gate/ambiguity component ablation？** 本轮不自动开展。若 ΔAUC 无稳定正向支持，建议下一轮有计划地做 joint architecture component ablation；若正向支持，则先研究 task-decoupled trajectory-preserving auxiliary learning。
13. **是否继续 joint intention+trajectory 主线？** 结论以本轮 ΔAUC 分类为准：正向支持可继续，但不能称 trajectory supervision 本身已是创新；无支持时，应把主线调整为 joint architecture 贡献归因，之后再评估任务解耦。

## J100 历史指标一致性

每个 J100 checkpoint 在同一 test archive 上的重评与 frozen historical metrics 的逐指标差异保存在 `official_test_evaluation.json`。

## 结论边界与下一步

本次唯一因果对照是 J0 vs frozen J100。M0/P1/T0 是上下文对照。未进行 component ablation，也没有据此宣称任何新模块的因果贡献。
