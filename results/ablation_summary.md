# JAAD 第一轮输入与门控消融记录

## 新版主实验：干净标签（crossing=0/1）

`crossing=-1` 已排除出主训练，并单独保存到 `data/processed/jaad_ambiguous/`。当前使用相对位置+速度输入，8 帧观测、12 帧预测。

| 模型 | Test AUC | Test balanced accuracy | Test F1 | ADE（归一化） |
|---|---:|---:|---:|---:|
| Target-only baseline | 0.512 | 0.500 | 0.931 | 0.0147 |
| Social gate: none | 0.479 | 0.500 | 0.931 | 0.0148 |
| Social gate: always | 0.642 | 0.581 | 0.857 | 0.0162 |
| Social gate: uncertainty | 0.641 | 0.581 | 0.857 | 0.0160 |
| Proposal-entropy gate + ambiguous supervision | 0.634 | 0.581 | 0.857 | 0.0156 |

### 新版结果解释

干净标签后，社会交互信息带来了明显提升；但当前不确定性门控与始终使用邻居的结果几乎相同，尚不能证明“熵门控”本身优于固定社会交互。因此，下一步必须优先改进或重新定义不确定性机制，而不是直接把当前结构写成已验证的创新。

模糊样本监督使候选分支的熵产生了样本间变化，但在当前单次训练中没有带来测试集提升。因此该机制暂时记录为候选实验，不作为最终模型结论。

训练配置：JAAD 官方 default video split；观测 8 帧、预测 12 帧；随机种子 42；类别均衡采样仅用于社会门控模型；指标来自未参与训练的 test split。

### Proposal-entropy 候选模型的多随机种子与概率校准

当前候选模型使用 `relative` 输入、`uncertainty` 门控和模糊样本辅助监督，在 seed=42/123/2024 上重复训练。下面报告均值±标准差：

| 指标 | 原始输出 | 验证集温度+偏置校准后 |
|---|---:|---:|
| Test AUC | 0.6350±0.0010 | 0.6350±0.0010 |
| Test balanced accuracy | 0.5810±0.0004 | 0.5000±0.0000* |
| Test F1 | 0.8567±0.0000 | 0.9310±0.0000* |
| Test Brier | 0.2348±0.0120 | 0.1124±0.0000 |
| Test ECE-10 | 0.3528±0.0173 | 0.0105±0.0001 |
| Test ADE（归一化） | 0.0161±0.0012 | — |
| Test FDE（归一化） | 0.0281±0.0010 | — |

校准显著改善概率可信度，但不改变 AUC；带 `*` 的分类指标使用 0.5 阈值，在 JAAD 干净标签的强类别不平衡下不适合作为主要结论。校准目前是独立后处理评估，不改变训练期的不确定性门控，因此下一步仍需验证“校准不确定性是否能提升门控决策”。

### 同一新架构下的公平门控对照

为避免把旧版模型与 proposal-entropy 版本混合比较，使用完全相同的数据、训练设置和 seed=42/123/2024，分别测试不使用邻居（`none`）、始终使用邻居（`always`）和不确定性门控（`uncertainty`）：

| 门控模式 | Test AUC | balanced accuracy | F1 | Brier | ECE-10 | ADE |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.5300±0.0313 | 0.5149±0.0210 | 0.8163±0.1621 | 0.2445±0.0028 | 0.3633±0.0041 | 0.0124±0.0027 |
| always | 0.6336±0.0019 | 0.5807±0.0000 | 0.8566±0.0000 | 0.2389±0.0087 | 0.3586±0.0122 | 0.0153±0.0004 |
| uncertainty | 0.6350±0.0010 | 0.5810±0.0004 | 0.8567±0.0000 | 0.2348±0.0120 | 0.3528±0.0173 | 0.0161±0.0012 |

当前证据支持“社会交互信息有效”，但只支持“当前不确定性门控与 always 基本相当”，不支持已经证明门控优于固定社会交互。后续若要形成论文贡献，需要进一步改进不确定性估计，或把贡献收缩为“模糊意图场景下的社会信息建模与概率校准”。

### 干净样本与模糊样本的不确定性诊断

使用三个候选模型种子，在 clean test 与 `crossing=-1` ambiguous test 之间进行区分测试：

| 不确定性信号 | ambiguous-vs-clean AUROC |
|---|---:|
| entropy | 0.5590±0.0016 |
| prior probability | 0.4821±0.0018 |
| gate | 0.5324±0.0058 |
| final intent probability | 0.4822±0.0018 |

clean/ambiguous 的平均熵分别为 `0.6890` 和 `0.6895`，平均 gate 几乎完全相同，说明当前 proposal 分支虽然数值上产生了轻微变化，但没有形成可用于识别模糊意图的有效不确定性。诊断脚本和逐种子结果保存在 `scripts/diagnose_uncertainty.py` 与 `results/multiseed/proposal_uncertainty_seed*/uncertainty_diagnostic.json`。

### MC Dropout 不确定性诊断

在同一组候选 checkpoint 上进行 10 次随机 dropout 前向推理，得到以下 ambiguous-vs-clean 区分结果：

| MC Dropout 信号 | AUROC |
|---|---:|
| predictive entropy | 0.5570±0.0007 |
| mutual information | 0.4779±0.0036 |
| probability standard deviation | 0.4787±0.0038 |
| gate standard deviation | 0.4579±0.0056 |

MC Dropout 没有实质改善模糊样本识别，说明当前问题不是简单更换熵计算方式即可解决。现阶段应保留社会交互模型作为有效基线，并重新设计模糊意图监督/输入特征；不把当前 uncertainty gate 或 MC Dropout 结果写成最终创新结论。

### 三状态意图分类试验

另行训练了一个显式三状态模型，将标签定义为“不过街 / 过街 / 标注模糊”。该模型在验证集上曾达到最高 ambiguity AUROC `0.748`，但独立测试集结果为：

| 指标 | Test 结果 |
|---|---:|
| Overall accuracy | 0.431 |
| Balanced accuracy | 0.381 |
| Macro-F1 | 0.304 |
| Ambiguous-F1 | 0.088 |
| Ambiguous AUROC | 0.478 |

三状态模型没有泛化，说明 `crossing=-1` 不适合作为普通第三行为类别直接预测。该试验代码和结果保存在 `src/models/tristate_social_gate.py`、`scripts/train_tristate_social_gate.py`、`results/tristate_pilot/` 和 `checkpoints/tristate_pilot.pt`，作为保留的探索分支。

### 绝对位置/尺度特征候选主模型

在相对位置+速度之外加入目标行人的绝对位置和框尺度（`relative_abs`，共 8 维输入），使用同一 proposal-entropy 架构和三个随机种子重新训练：

| 指标 | relative | relative_abs |
|---|---:|---:|
| Test AUC | 0.6350±0.0010 | 0.6758±0.0068 |
| balanced accuracy | 0.5810±0.0004 | 0.6246±0.0334 |
| F1 | 0.8567±0.0000 | 0.8040±0.0497 |
| ADE（归一化） | 0.0161±0.0012 | 0.0158±0.0003 |
| FDE（归一化） | 0.0281±0.0010 | 0.0271±0.0004 |

`relative_abs` 版本的验证集温度+偏置校准将 Test Brier 从 `0.2147±0.0053` 降至 `0.1075±0.0006`，ECE-10 从 `0.3161±0.0088` 降至 `0.0253±0.0066`。但其 ambiguous-vs-clean 熵 AUROC 为 `0.4734±0.0319`，仍不能说明熵可以识别模糊样本。因此暂定 `relative_abs` 社会交互模型为新的主模型候选，校准作为可靠性分析，不把熵门控作为已证实的核心创新。

### relative_abs 输入下的公平门控对照

在相同的 8 维目标输入、数据划分、训练设置和三个随机种子下，对比三种门控：

| 门控模式 | Test AUC | balanced accuracy | F1 | ADE | FDE |
|---|---:|---:|---:|---:|---:|
| none | 0.6156±0.0127 | 0.6033±0.0048 | 0.5669±0.0443 | 0.0157±0.0003 | 0.0269±0.0004 |
| always | 0.6833±0.0046 | 0.6190±0.0336 | 0.7800±0.0521 | 0.0155±0.0001 | 0.0270±0.0001 |
| uncertainty | 0.6758±0.0068 | 0.6246±0.0334 | 0.8040±0.0497 | 0.0158±0.0003 | 0.0271±0.0004 |

当前最稳妥的主模型是 `relative_abs + always-social`，或将 `relative_abs + uncertainty` 作为意图 F1 稍高但尚未证明门控有效的候选。无论选择哪一个，都必须明确社会交互带来了主要收益，不能声称熵门控优于固定社会交互。

对 `relative_abs + always-social` 主模型进行验证集温度+偏置校准后，三个种子的 Test AUC 为 `0.6833±0.0046`（不变），Brier 为 `0.1076±0.0007`，ECE-10 为 `0.0224±0.0065`。因此该模型目前具备较好的意图排序和概率可靠性，适合作为后续正式结果图与论文主表的起点。

### 修正邻居维度后的场景社会交互正式结果

检查数据构造器后确认，`neighbor_obs` 实际存储顺序为 `[B, N, T, F]`。修正前，`SceneSocialGate` 和 `UncertaintySocialGate` 将邻居数与观测时间维解释反了；由于当时 `N=T=8`，形状仍能运行，但喂给邻居 GRU 的序列语义错误。现已修正这两个模型、增加非方形维度回归测试，并在相同数据划分、训练设置和三个随机种子下重跑 `none`、`always`、`uncertainty`。数据审计、逐种子指标和完整统计见 [`results/neighbor_dimension_fix/summary.md`](neighbor_dimension_fix/summary.md) 与 [`summary.json`](neighbor_dimension_fix/summary.json)。

| 门控模式 | Test AUC | balanced accuracy | F1 | Brier | ADE（归一化） | FDE（归一化） |
|---|---:|---:|---:|---:|---:|---:|
| Scene + no-social | 0.7606±0.0196 | 0.6046±0.0507 | 0.9224±0.0080 | 0.1104±0.0106 | 0.0149±0.0001 | 0.0272±0.0003 |
| Scene + always-social | 0.7405±0.0329 | 0.6028±0.0645 | 0.9257±0.0065 | 0.1106±0.0101 | 0.0153±0.0003 | 0.0271±0.0003 |
| Scene + uncertainty gate | **0.7674±0.0179** | 0.6022±0.0315 | **0.9278±0.0049** | **0.1045±0.0073** | 0.0152±0.0007 | **0.0270±0.0005** |

不确定性门控的平均 AUC 比 no-social 高 0.0068，但仅在 3 个种子中的 1 个种子上高于 no-social；always-social 平均 AUC 比 no-social 低 0.0201。故当前结果尚不能证明社会交互稳定提升意图预测，也不能声称不确定性门控可靠优于 no-social。它相对 always-social 的 AUC 在 3/3 个种子上更高，可作为后续复验线索。轨迹指标与意图指标应分别报告，且三种门控的归一化 ADE/FDE 数值接近。

### 历史场景社会交互结果（LEGACY；不可用于最终结论）

> **LEGACY / INVALID FOR FINAL CONCLUSION:** 旧 scene-social 实验中的 `neighbor_obs` 存储顺序是 `[B, N, T, F]`，但两个社会模型把邻居维与时间维解释反了。旧实验 `obs_len=max_neighbors=8`，因此转置后形状仍可通过后续计算，未触发 shape exception。以下旧数值仅保留为历史审计记录，不再作为正式论文结论；正式结论以后续 neighborfix 重跑结果为准。

以下旧表格、基于旧表格的场景收益解释，以及旧 uncertainty 检查点的像素误差均为修正前结果，仅为审计历史保留，不应引用为本项目最终实验结论。

从每个视频的代表性首帧提取冻结 ResNet-18 的 512 维场景 embedding，并与目标轨迹、邻居轨迹共同输入模型。clean 与 ambiguous 数据均无缺失场景特征。

| 模型 | Test AUC | balanced accuracy | F1 | ADE | FDE |
|---|---:|---:|---:|---:|---:|
| Scene + no-social | 0.7508±0.0117 | 0.6037±0.0612 | 0.9182±0.0012 | 0.0147±0.0001 | 0.0268±0.0002 |
| Scene + always-social | 0.7431±0.0390 | 0.5959±0.0464 | 0.9196±0.0105 | 0.0148±0.0003 | 0.0268±0.0004 |
| Scene + uncertainty gate | 0.7620±0.0043 | 0.6500±0.0579 | 0.9204±0.0041 | 0.0154±0.0007 | 0.0273±0.0007 |

加入场景信息后，三种模型都明显提升，说明场景视觉特征是主要收益来源；`scene + uncertainty` 相比 `scene + no-social` 仍有小幅 AUC 增益，并且跨种子波动更小，说明场景条件下的自适应社会交互具有潜力。另一方面，scene+uncertainty 的 ambiguous-vs-clean entropy AUROC 仍只有 `0.5460±0.0076`，因此不能把 `crossing=-1` 直接当作不确定性真值。更准确的论文表述应是“场景条件下的自适应社会交互门控”，并在主表中同时保留 no-social 对照。

对 `scene + uncertainty gate` 的三个测试检查点，将归一化坐标误差还原到原始图像尺寸后重新计算像素单位轨迹误差：

| 指标 | seed=42 | seed=123 | seed=2024 | 均值±标准差 |
|---|---:|---:|---:|---:|
| ADE（pixel） | 26.55 | 27.62 | 29.18 | 27.78±1.32 |
| FDE（pixel） | 48.72 | 50.00 | 51.74 | 50.16±1.52 |

这里的 pixel 指图像像素距离，不是米；结果文件保存在各自的 `results/scene_uncertainty_seed*/pixel_metrics.json`。该统计口径更接近 JITP 在 JAAD 上使用边界框图像坐标计算 ADE/FDE 的方式，但只有在确认对方论文的预处理和坐标定义完全一致时，才能进行严格数值比较。

### 15→15 轨迹 Transformer 基线

为排除当前 GRU+MLP 轨迹头过弱造成的影响，重新建立了 15 帧观测、15 帧预测的数据，并训练 `SceneTrajectoryTransformer`。模型输入为 15 帧目标轨迹特征（相对位置/速度 + 绝对位置/尺度）和冻结场景特征，验证集按像素 ADE 选择最佳检查点。

| 模型 | Test ADE（pixel） | Test FDE（pixel） |
|---|---:|---:|
| SceneTrajectoryTransformer | **11.01±0.18** | **19.46±0.41** |
| JITP（论文报告） | 17.73 | 31.34 |

该结果来自 seed=42/123/2024 三次独立训练，分别保存在 `results/trajectory_transformer_scene_15x15_seed*/metrics.json`。它说明专门的 Transformer 轨迹分支具有较强预测能力；但目前它还是独立轨迹基线，还没有与意图分类头和不确定性社会门控联合训练，不能直接作为最终联合模型结论。

### Transformer 与联合门控的初步联合实验

将 Transformer 目标编码器、场景特征和社会门控直接合并后，seed=123 的联合模型测试结果为：AUC `0.7625`、ADE `25.03 pixel`、FDE `42.66 pixel`；提高轨迹损失权重后为 AUC `0.8055`、ADE `24.71 pixel`、FDE `43.10 pixel`。这明显差于独立 Transformer，说明直接共享融合特征会造成意图分支与轨迹分支之间的优化竞争。该结果暂不作为最终模型，后续应保留独立 Transformer 的轨迹主干，把社会门控以可控残差方式接入，并重新做联合消融。

| 模型 | 目标输入 | Test AUC | Test balanced accuracy | Test F1 | ADE（归一化） |
|---|---|---:|---:|---:|---:|
| Target-only baseline | 相对位置 + 速度 | 0.482 | 0.500 | 0.809 | 0.0145 |
| Uncertainty social gate | 相对位置 + 速度 | 0.568 | 0.551 | 0.730 | 0.0151 |
| Target-only baseline | 相对位置 + 速度 + 绝对位置/尺度 | 0.473 | 0.480 | 0.746 | 0.0144 |
| Uncertainty social gate | 相对位置 + 速度 + 绝对位置/尺度 | 0.514 | 0.500 | 0.676 | 0.0143 |

| 门控对照：不使用邻居 | 相对位置 + 速度 | 0.497 | 0.500 | 0.000 | 0.0146 |
| 门控对照：始终使用邻居 | 相对位置 + 速度 | 0.570 | 0.545 | 0.714 | 0.0151 |

门控机制的正式对照实验已预留：`gate_mode=none` 表示不使用邻居，`gate_mode=always` 表示始终使用邻居，`gate_mode=uncertainty` 表示由目标意图熵控制邻居贡献。

## 当前结论

1. 只使用目标历史运动时，意图分类接近随机，且 F1 受到类别比例影响，不能单独作为主要判断依据。
2. 加入邻居轨迹和社会交互门控后，相对特征版本的 AUC 和 balanced accuracy 提升，说明社会信息具有有效性。
3. 绝对位置/尺度特征在验证集上有提升，但在测试视频上没有稳定泛化，暂时保留为消融项，不作为默认最终输入。
4. 相对特征版本的先验熵仍接近最大值，说明门控的不确定性分支还需要进一步校准；当前结果是工程基线，不是论文最终结果。
