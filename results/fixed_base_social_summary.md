# Fixed-base social residual: seed123 report

Test split: 18,331 samples. Both social models share the same fixed base checkpoint and calibrated entropy.

## Test metrics

| Model | AUC ↑ | BAcc ↑ | F1 ↑ | Brier ↓ | ECE ↓ | ADE (px) ↓ | FDE (px) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base intent | 0.8170 | 0.5149 | 0.9290 | 0.0953 | 0.0218 | 11.058 | 19.593 |
| Always social | 0.7944 | 0.5645 | 0.9224 | 0.1004 | 0.0425 | 11.058 | 19.593 |
| Uncertainty social | 0.8064 | 0.6383 | 0.9195 | 0.1037 | 0.0583 | 11.058 | 19.593 |

## Calibration and controlled comparison

Temperature fitted on validation only: **0.774683**. Validation Brier/ECE: 0.0950/0.1018 → 0.0929/0.0870. Test Brier/ECE: 0.0951/0.0334 → 0.0953/0.0218.

Base logits are exactly identical across none/always/uncertainty: **True** (maximum absolute difference 0). The Transformer and intent head remain frozen during social training.

| Social mode | Gate mean ± std | Delta-logit mean ± std | Mean paired BCE improvement ↑ | Helped | Hurt |
|---|---:|---:|---:|---:|---:|
| Always social | 1.0000 ± 0.0000 | -0.5425 ± 0.2429 | -0.01779 | 8.34% | 71.75% |
| Uncertainty social | 0.4051 ± 0.2874 | -0.9038 ± 0.8116 | -0.02246 | 12.01% | 68.08% |

## Shared-base entropy strata

Positive paired BCE improvement means the social model reduced that sample's BCE versus base.

| Entropy | N | Model | AUC | Brier | Mean paired BCE gain | Helped | Hurt |
|---|---:|---|---:|---:|---:|---:|---:|
| low | 6111 | Base intent | 0.8623 | 0.0121 | 0.00000 | 0.00% | 0.00% |
| low | 6111 | Always social | 0.5757 | 0.0128 | -0.01128 | 0.15% | 82.38% |
| low | 6111 | Uncertainty social | 0.7992 | 0.0122 | -0.00193 | 6.19% | 76.34% |
| medium | 6110 | Base intent | 0.6281 | 0.0722 | 0.00000 | 0.00% | 0.00% |
| medium | 6110 | Always social | 0.5616 | 0.0746 | -0.01267 | 5.48% | 74.06% |
| medium | 6110 | Uncertainty social | 0.5797 | 0.0744 | -0.00860 | 9.08% | 70.46% |
| high | 6110 | Base intent | 0.6267 | 0.2017 | 0.00000 | 0.00% | 0.00% |
| high | 6110 | Always social | 0.5793 | 0.2139 | -0.02942 | 19.39% | 58.82% |
| high | 6110 | Uncertainty social | 0.6121 | 0.2245 | -0.05687 | 20.77% | 57.45% |

## Neighbor-count strata

| Neighbors | N | Model | AUC | Brier | Mean paired BCE gain | Helped | Hurt |
|---|---:|---|---:|---:|---:|---:|---:|
| 0 | 3649 | Base intent | 0.7774 | 0.1623 | 0.00000 | 0.00% | 0.00% |
| 0 | 3649 | Always social | 0.7774 | 0.1623 | 0.00000 | 0.00% | 0.00% |
| 0 | 3649 | Uncertainty social | 0.7774 | 0.1623 | 0.00000 | 0.00% | 0.00% |
| 1 | 4565 | Base intent | 0.8221 | 0.0830 | 0.00000 | 0.00% | 0.00% |
| 1 | 4565 | Always social | 0.8313 | 0.0942 | -0.03515 | 10.19% | 89.81% |
| 1 | 4565 | Uncertainty social | 0.8407 | 0.1009 | -0.04625 | 18.27% | 81.73% |
| 2-3 | 5454 | Base intent | 0.8305 | 0.0659 | 0.00000 | 0.00% | 0.00% |
| 2-3 | 5454 | Always social | 0.8318 | 0.0752 | -0.02817 | 8.20% | 91.80% |
| 2-3 | 5454 | Uncertainty social | 0.8170 | 0.0852 | -0.04836 | 12.12% | 87.88% |
| >=4 | 4663 | Base intent | 0.8571 | 0.0894 | 0.00000 | 0.00% | 0.00% |
| >=4 | 4663 | Always social | 0.8718 | 0.0876 | -0.00258 | 13.23% | 86.77% |
| >=4 | 4663 | Uncertainty social | 0.8885 | 0.0822 | 0.01353 | 15.16% | 84.84% |

## Neighbor-shuffle diagnostic

Target, scene, and label were held fixed while complete neighbor tensors and masks were permuted across test samples.

| Model | Real AUC | Shuffled AUC | Δ AUC (shuffled-real) | Real Brier | Shuffled Brier | Δ Brier |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.8170 | 0.8170 | +0.0000 | 0.0953 | 0.0953 | +0.0000 |
| always | 0.7944 | 0.8108 | +0.0164 | 0.1004 | 0.0988 | -0.0016 |
| uncertainty | 0.8064 | 0.8076 | +0.0012 | 0.1037 | 0.1063 | +0.0026 |

## Decision

**Stop after seed123; do not run additional seeds or Stage B.** Both social AUCs are below the shared base; mean paired BCE improvements are negative overall and in the high-uncertainty stratum. The >=4-neighbor uncertainty subgroup has a positive mean but most samples are still hurt, so it is not sufficient positive evidence under the pre-registered stop rule.

Trajectory output is unchanged by social residuals: ADE 11.058px, FDE 19.593px. This is consistent with the frozen-trajectory design; social residuals only affect intent logits.

Interpretation: on this seed/test split, social residuals did not improve overall intent prediction. Uncertainty gating had a smaller AUC drop than always-on fusion, but it also remained below the base and its mean paired BCE gain was negative; therefore this is not evidence that social interaction improves prediction.
