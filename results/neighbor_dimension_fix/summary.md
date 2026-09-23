# Neighbor-dimension correction: final ablation summary

All entries are test-set mean ± sample standard deviation across seeds 42, 123, and 2024. The corrected runs use the same recorded training protocol and splits as the legacy runs; only the neighbor tensor axis interpretation in the two social models was corrected.

Lower is better for Brier, ADE, and FDE; higher is better for AUC, balanced accuracy, and F1. Gate and entropy means are descriptive.

## Corrected results

| Gate mode | AUC | Balanced accuracy | F1 | Brier | ADE (normalized) | FDE (normalized) | Gate mean | Entropy mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.7606 ± 0.0196 | 0.6046 ± 0.0507 | 0.9224 ± 0.0080 | 0.1104 ± 0.0106 | 0.0149 ± 0.0001 | 0.0272 ± 0.0003 | 0.0000 ± 0.0000 | 0.0971 ± 0.0036 |
| always | 0.7405 ± 0.0329 | 0.6028 ± 0.0645 | 0.9257 ± 0.0065 | 0.1106 ± 0.0101 | 0.0153 ± 0.0003 | 0.0271 ± 0.0003 | 1.0000 ± 0.0000 | 0.1392 ± 0.0451 |
| uncertainty | 0.7674 ± 0.0179 | 0.6022 ± 0.0315 | 0.9278 ± 0.0049 | 0.1045 ± 0.0073 | 0.0152 ± 0.0007 | 0.0270 ± 0.0005 | 0.7289 ± 0.0385 | 0.1541 ± 0.0522 |

## Legacy results (axis bug; historical comparison only)

| Gate mode | AUC | Balanced accuracy | F1 | Brier | ADE (normalized) | FDE (normalized) | Gate mean | Entropy mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.7508 ± 0.0144 | 0.6037 ± 0.0749 | 0.9182 ± 0.0015 | 0.1177 ± 0.0093 | 0.0147 ± 0.0001 | 0.0268 ± 0.0003 | 0.0000 ± 0.0000 | 0.1160 ± 0.0033 |
| always | 0.7431 ± 0.0478 | 0.5959 ± 0.0569 | 0.9196 ± 0.0128 | 0.1156 ± 0.0149 | 0.0148 ± 0.0004 | 0.0268 ± 0.0004 | 1.0000 ± 0.0000 | 0.1421 ± 0.0120 |
| uncertainty | 0.7620 ± 0.0053 | 0.6500 ± 0.0710 | 0.9204 ± 0.0050 | 0.1095 ± 0.0054 | 0.0154 ± 0.0008 | 0.0273 ± 0.0008 | 0.4694 ± 0.2521 | 0.2015 ± 0.0875 |

## Side-by-side: legacy vs neighborfix

Each cell is legacy → corrected, with mean ± sample standard deviation.

| Gate mode | AUC | Balanced accuracy | F1 | Brier | ADE (normalized) | FDE (normalized) |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.7508 ± 0.0144 → 0.7606 ± 0.0196 | 0.6037 ± 0.0749 → 0.6046 ± 0.0507 | 0.9182 ± 0.0015 → 0.9224 ± 0.0080 | 0.1177 ± 0.0093 → 0.1104 ± 0.0106 | 0.0147 ± 0.0001 → 0.0149 ± 0.0001 | 0.0268 ± 0.0003 → 0.0272 ± 0.0003 |
| always | 0.7431 ± 0.0478 → 0.7405 ± 0.0329 | 0.5959 ± 0.0569 → 0.6028 ± 0.0645 | 0.9196 ± 0.0128 → 0.9257 ± 0.0065 | 0.1156 ± 0.0149 → 0.1106 ± 0.0101 | 0.0148 ± 0.0004 → 0.0153 ± 0.0003 | 0.0268 ± 0.0004 → 0.0271 ± 0.0003 |
| uncertainty | 0.7620 ± 0.0053 → 0.7674 ± 0.0179 | 0.6500 ± 0.0710 → 0.6022 ± 0.0315 | 0.9204 ± 0.0050 → 0.9278 ± 0.0049 | 0.1095 ± 0.0054 → 0.1045 ± 0.0073 | 0.0154 ± 0.0008 → 0.0152 ± 0.0007 | 0.0273 ± 0.0008 → 0.0270 ± 0.0005 |

## Interpretation

- Corrected mean AUC: no-social 0.7606, always-social 0.7405, uncertainty gate 0.7674.
- Uncertainty gate versus no-social: mean AUC difference +0.0068; it is higher in 1/3 seeds. This is not a consistent per-seed improvement.
- Uncertainty gate versus always-social: mean AUC difference +0.0269; it is higher in 3/3 seeds.
- Always-social versus no-social: mean AUC difference -0.0201; it is higher in 1/3 seeds.
- Do not claim that social interaction or uncertainty gating reliably improves intent prediction from this three-seed ablation alone. Report all controls and seed variability.
- Treat trajectory ADE/FDE separately from intent metrics; the best intent AUC mode need not be the best trajectory mode.

## Run artifacts

Per-seed corrected `metrics.json` files are in `results/scene_{none,always,uncertainty}_neighborfix_seed{42,123,2024}/`. Checkpoints remain local and are not part of the repository.
