# Social protocol repair — seed123

本轮只排查 social residual 的采样先验偏移与邻居不可见帧处理；所有社交模型使用同一 fixed-base checkpoint，轨迹网络冻结。旧结果保留，新增结果使用独立目录。

## 七组对照

| Condition | Val AUC | Selected epoch | Test AUC | Brier | ECE | Paired BCE gain | Helped / hurt | Δlogit mean ± std | Shuffle ΔAUC* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 0.7991 | — | 0.8170 | 0.0953 | 0.0218 | +0.00000 | 0.0% / 0.0% | +0.0000 ± 0.0000 | — |
| 旧 balanced · Always | 0.7695 | 1 | 0.7944 | 0.1004 | 0.0425 | -0.01779 | 8.3% / 71.8% | -0.5425 ± 0.2429 | 0.0164 |
| 旧 balanced · Uncertainty | 0.7915 | 2 | 0.8064 | 0.1037 | 0.0583 | -0.02246 | 12.0% / 68.1% | -0.9038 ± 0.8116 | 0.0012 |
| Natural · Always | 0.7991 | 0 | 0.8170 | 0.0953 | 0.0218 | +0.00000 | 0.0% / 0.0% | +0.0000 ± 0.0000 | 0.0000 |
| Natural · Uncertainty | 0.7991 | 0 | 0.8170 | 0.0953 | 0.0218 | +0.00000 | 0.0% / 0.0% | +0.0000 ± 0.0000 | 0.0000 |
| Natural+visibility · Always | 0.7991 | 0 | 0.8170 | 0.0953 | 0.0218 | +0.00000 | 0.0% / 0.0% | +0.0000 ± 0.0000 | 0.0000 |
| Natural+visibility · Uncertainty | 0.7991 | 0 | 0.8170 | 0.0953 | 0.0218 | +0.00000 | 0.0% / 0.0% | +0.0000 ± 0.0000 | 0.0000 |

*Shuffle ΔAUC is shuffled minus real; negative values mean real neighbors scored higher. Base has no social input. Old balanced shuffle values come from the prior diagnostic; new natural/visibility values permute neighbor_obs, neighbor_mask, and neighbor_visible_mask together.

## Epoch-0 selection and sampling diagnosis

Natural always/uncertainty epoch-0 validation AUC: 0.7991. Both selected epoch **0** (best AUC 0.7991); visible always/uncertainty also both selected epoch **0**. The selected models therefore preserve the base logits exactly and produce zero paired BCE change.

Old balanced selected test residual-logit means: always -0.5425, uncertainty -0.9038. Natural selected means are exactly zero because epoch 0 was retained. The best nonzero natural epochs had positive residual means (always +1.3434, uncertainty +1.6327) but validation AUC below the base. This removes the selected model's negative residual offset by falling back to zero, not by finding a better social residual.

## Visibility strata

Among 14,682 test samples with at least one valid neighbor, mean valid-neighbor visible-frame ratio is 0.998141. Tercile cutpoints are q1=1.0000, q2=1.0000; they tie, so the middle bin is empty rather than splitting identical visibility values arbitrarily. 3,649 zero-neighbor samples are excluded.

| Visibility stratum | N | Base AUC / Brier | Visible Always AUC / Brier | Visible Uncertainty AUC / Brier | Paired BCE gains (A/U) |
|---|---:|---:|---:|---:|---:|
| low visibility | 600 | 0.8188 / 0.0869 | 0.8188 / 0.0869 | 0.8188 / 0.0869 | +0.0000 / +0.0000 |
| medium visibility | 0 | — | — | — | — / — |
| high visibility | 14082 | 0.8381 / 0.0784 | 0.8381 / 0.0784 | 0.8381 / 0.0784 | +0.0000 / +0.0000 |

## Neighbor-count strata

Cells show AUC / Brier. The ≥4-neighbor subgroup remains exploratory and was not used to tune a gate.

| Neighbors | N | Base (AUC/Brier) | Balanced Always · Uncertainty | Natural Always · Uncertainty | Visibility Always · Uncertainty |
|---|---:|---:|---:|---:|---:|
| 0 | 3649 | 0.7774 / 0.1623 | Always 0.7774 / 0.1623 · Unc 0.7774 / 0.1623 | Always 0.7774 / 0.1623 · Unc 0.7774 / 0.1623 | Always 0.7774 / 0.1623 · Unc 0.7774 / 0.1623 |
| 1 | 4565 | 0.8221 / 0.0830 | Always 0.8313 / 0.0942 · Unc 0.8407 / 0.1009 | Always 0.8221 / 0.0830 · Unc 0.8221 / 0.0830 | Always 0.8221 / 0.0830 · Unc 0.8221 / 0.0830 |
| 2-3 | 5454 | 0.8305 / 0.0659 | Always 0.8318 / 0.0752 · Unc 0.8170 / 0.0852 | Always 0.8305 / 0.0659 · Unc 0.8305 / 0.0659 | Always 0.8305 / 0.0659 · Unc 0.8305 / 0.0659 |
| >=4 | 4663 | 0.8571 / 0.0894 | Always 0.8718 / 0.0876 · Unc 0.8885 / 0.0822 | Always 0.8571 / 0.0894 · Unc 0.8571 / 0.0894 | Always 0.8571 / 0.0894 · Unc 0.8571 / 0.0894 |

## Decision

Stop current GRU social residual route: **True**. Natural+visibility uncertainty test AUC ≤ Base: True; paired BCE mean ≤ 0: True; real neighbor better than shuffled: False.

> current trajectory-based social representation does not provide reliable incremental intent information under this protocol.

Trajectory remains ADE 11.0585px / FDE 19.5931px, unchanged across all seven conditions. No extra seeds were run.

Detailed metrics and checkpoint-selection histories are in the companion JSON.
