# Clean J0 单组件归因结果

协议 SHA256：`910dedc0eda685d211a816e7e65e4f116c238ad9e3f82ef0f90025babc1ff532`；15 个正式消融训练已在冻结前完成。所有 test 指标均在冻结后统一计算。

## 主结果

三 seed 分别训练/评估，表中为均值 ± sample SD；F1、BAcc、Accuracy 均使用固定阈值 0.5。ADE/FDE 不用于比较或模型选择。

| Method | AUC | Brier | F1 | BAcc | Accuracy |
|---|---:|---:|---:|---:|---:|
| M0 Target-only | 0.6732 ± 0.0079 | 0.2402 ± 0.0359 | 0.4882 ± 0.4578 | 0.5422 ± 0.0693 | 0.4760 ± 0.3726 |
| A0 Full Clean J0 | 0.7644 ± 0.0210 | 0.0953 ± 0.0032 | 0.9315 ± 0.0039 | 0.5896 ± 0.0211 | 0.8752 ± 0.0059 |
| A1 No Scene | 0.6736 ± 0.0194 | 0.1944 ± 0.0279 | 0.8194 ± 0.0828 | 0.6211 ± 0.0131 | 0.7196 ± 0.1010 |
| A2 No Social | 0.7559 ± 0.0099 | 0.1234 ± 0.0220 | 0.9012 ± 0.0333 | 0.6662 ± 0.0616 | 0.8323 ± 0.0479 |
| A3 No Proposal Loss | 0.7705 ± 0.0153 | 0.0973 ± 0.0046 | 0.9361 ± 0.0106 | 0.6383 ± 0.0331 | 0.8849 ± 0.0163 |
| A4 No Adaptive Gate | 0.7604 ± 0.0163 | 0.1096 ± 0.0118 | 0.9136 ± 0.0277 | 0.6038 ± 0.0396 | 0.8478 ± 0.0415 |
| A5 No Ambiguity | 0.7866 ± 0.0120 | 0.1026 ± 0.0065 | 0.9388 ± 0.0044 | 0.5756 ± 0.0372 | 0.8870 ± 0.0087 |

## 组件贡献与逐 seed bootstrap

组件贡献定义为 `Full − Ablation`，正值表示移除后 AUC 下降；bootstrap 文件同时保存原始 `Ablation − Full`。CI 是按 scene_id 配对 cluster bootstrap，2000 次/seed；五项比较仅作探索性归因，不做确认性显著性宣称。

| Removed component | Full AUC | Ablated AUC | Mean contribution Full−Ablated | Seeds Full>Ablated | Grade |
|---|---:|---:|---:|---:|---|
| No Scene | 0.7644 | 0.6736 | +0.0909 | 3/3 | moderate |
| No Social | 0.7644 | 0.7559 | +0.0085 | 2/3 | weak |
| No Proposal Loss | 0.7644 | 0.7705 | -0.0061 | 0/3 | harmful |
| No Adaptive Gate | 0.7644 | 0.7604 | +0.0041 | 2/3 | weak |
| No Ambiguity | 0.7644 | 0.7866 | -0.0222 | 0/3 | harmful |

| Removed component | Seed | ΔAUC Ablation−Full | 95% CI for Ablation−Full | ΔBrier Ablation−Full | 95% CI for ΔBrier |
|---|---:|---:|---:|---:|---:|
| A1 No Scene | 42 | -0.0828 | [-0.2082, +0.0513] | +0.1249 | [+0.0850, +0.1715] |
| A1 No Scene | 123 | -0.0900 | [-0.2495, +0.0941] | +0.0658 | [+0.0302, +0.1030] |
| A1 No Scene | 2024 | -0.0998 | [-0.2308, +0.0406] | +0.1064 | [+0.0671, +0.1414] |
| A2 No Social | 42 | -0.0162 | [-0.0593, +0.0184] | +0.0516 | [+0.0244, +0.0806] |
| A2 No Social | 123 | +0.0060 | [-0.0078, +0.0218] | +0.0221 | [+0.0093, +0.0349] |
| A2 No Social | 2024 | -0.0154 | [-0.1354, +0.0839] | +0.0103 | [-0.0042, +0.0279] |
| A3 No Proposal Loss | 42 | +0.0064 | [-0.0295, +0.0377] | -0.0025 | [-0.0165, +0.0092] |
| A3 No Proposal Loss | 123 | +0.0119 | [-0.0038, +0.0321] | +0.0028 | [-0.0009, +0.0074] |
| A3 No Proposal Loss | 2024 | +0.0000 | [-0.0681, +0.0685] | +0.0055 | [-0.0090, +0.0201] |
| A4 No Adaptive Gate | 42 | -0.0112 | [-0.0502, +0.0227] | +0.0274 | [+0.0059, +0.0539] |
| A4 No Adaptive Gate | 123 | +0.0037 | [-0.0008, +0.0083] | +0.0003 | [-0.0016, +0.0026] |
| A4 No Adaptive Gate | 2024 | -0.0047 | [-0.0939, +0.0805] | +0.0149 | [+0.0006, +0.0328] |
| A5 No Ambiguity | 42 | +0.0175 | [-0.0655, +0.0996] | +0.0131 | [-0.0105, +0.0420] |
| A5 No Ambiguity | 123 | +0.0329 | [-0.0376, +0.1087] | +0.0057 | [-0.0121, +0.0262] |
| A5 No Ambiguity | 2024 | +0.0161 | [-0.0378, +0.0681] | +0.0030 | [-0.0096, +0.0158] |

## 核心问题回答

1. 移除后 AUC 下降最大的是 **A1 No Scene**，平均贡献 `Full−Ablation=+0.0909`（等级：moderate）。
2. Scene 稳定贡献：moderate；3 seed 中 Full AUC 更高 3/3，mean contribution `+0.0909`。
3. Social 稳定贡献：weak；3 seed 中 Full AUC 更高 2/3，mean contribution `+0.0085`。
4. Proposal auxiliary loss：harmful；移除 supervision 后平均贡献 `-0.0061`。proposal forward/prior entropy/gate 均保留，因此此项只归因于 prior BCE supervision。
5. Adaptive gate 相对固定中性 `g=0.5`：weak；平均贡献 `+0.0041`。
6. Ambiguity regularization：harmful；移除后平均贡献 `-0.0222`。
7. 保留决策：保留达到 moderate/strong 且跨 seed 方向一致的组件；其余暂不宣称为核心创新。
8. 简化决策：weak 或 harmful 的 social/proposal/gate/ambiguity 项可列为简化候选，但本轮不实际删除组合模块。
9. `M0→J0` AUC gap 为 `0.0912`；No Scene 的 Full−Ablation 点估计为 `+0.0909`，几乎与该 gap 同量级。但 3/3 seed 的 AUC bootstrap CI 均跨 0，故目前不能据此确认 scene 单独解释了全部差距。
10. Interaction：存在单项贡献信号；单项消融不能识别组件间 interaction，本轮没有直接检验。
11. 下一步建议：**scene refinement**。不自动启动下一阶段。

## 协议与解释边界

- Frozen protocol SHA256: `910dedc0eda685d211a816e7e65e4f116c238ad9e3f82ef0f90025babc1ff532`。
- A0 使用已有 Clean J0 checkpoint/test predictions；没有重新训练 A0。
- A1–A5 每 seed 使用匹配的 Clean J0 initial state；正式训练 sampler fingerprints 与 A0 同 seed 完全一致。
- 所有 ablation 使用 15 epochs；checkpoint 和 scheduler 只依 raw validation AUC/Brier；trajectory ADE/FDE 仅保存 forward sanity，不解释、不用于选择。
- Bootstrap 按 seed 单独计算；不 pooling seeds、不对五项比较作多重比较校正。CI 跨零仅表示该 cluster-bootstrap 证据不确定，不是组件无效的证明。
- 详细数字见 `component_effects.json`、`cluster_bootstrap.json`；依赖与干预定义见 `component_dependency_map.md`、`component_definitions.json`。
