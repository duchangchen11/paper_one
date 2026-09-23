# 15×15 intent scene ablation

Primary comparison uses validation-selected calibration and a validation-selected balanced-accuracy threshold. Test threshold 0.5 BAcc/F1 are retained per run in JSON for historical comparability.

| Seed | Target-only AUC | Target+scene AUC | ΔAUC | Target-only Brier | Target+scene Brier | ΔBrier | Target-only ECE | Target+scene ECE | Target-only BAcc | Target+scene BAcc | Target-only F1 | Target+scene F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6817 | 0.6313 | -0.0504 | 0.1062 | 0.1265 | +0.0203 | 0.0227 | 0.1177 | 0.6121 | 0.5079 | 0.6918 | 0.9202 |
| 123 | 0.7173 | 0.8170 | +0.0996 | 0.1045 | 0.0955 | -0.0090 | 0.0228 | 0.0223 | 0.6660 | 0.7495 | 0.7781 | 0.8626 |
| 2024 | 0.6816 | 0.6553 | -0.0264 | 0.1067 | 0.1139 | +0.0072 | 0.0342 | 0.0942 | 0.6407 | 0.5826 | 0.7210 | 0.8586 |

- Target-only AUC: 0.6936 ± 0.0206; target+scene AUC: 0.7012 ± 0.1010.
- Mean paired ΔAUC: 0.0076 ± 0.0806; scene AUC higher in 1/3 seeds.
- Mean paired ΔBrier: 0.0062 ± 0.0146; lower Brier is better; Brier improved in 1/3 seeds.
- Conclusion: No stable intent benefit is supported: target+scene AUC is higher in 1/3 seeds and Brier improves in 1/3; the mean AUC difference is small relative to seed variability.

## Mean ± sample standard deviation

| Metric | Target-only | Target+scene | Paired Δ (scene−only) |
|---|---:|---:|---:|
| auc | 0.6936 ± 0.0206 | 0.7012 ± 0.1010 | 0.0076 ± 0.0806 |
| brier | 0.1058 ± 0.0012 | 0.1120 ± 0.0156 | 0.0062 ± 0.0146 |
| ece_10 | 0.0266 ± 0.0066 | 0.0780 ± 0.0497 | 0.0515 ± 0.0483 |
| balanced_accuracy | 0.6396 ± 0.0269 | 0.6133 ± 0.1237 | -0.0263 ± 0.0978 |
| f1 | 0.7303 ± 0.0439 | 0.8805 ± 0.0345 | 0.1502 ± 0.0728 |

## Validation-only calibration and thresholds

| Seed | Readout | Selected calibration | Threshold | Validation BAcc |
|---:|---|---|---:|---:|
| 42 | target_only | temperature+bias | 0.8761 | 0.6359 |
| 42 | target_scene | temperature+bias | 0.6389 | 0.8812 |
| 123 | target_only | temperature+bias | 0.8593 | 0.6904 |
| 123 | target_scene | temperature+bias | 0.8386 | 0.8293 |
| 2024 | target_only | temperature+bias | 0.8704 | 0.6482 |
| 2024 | target_scene | temperature+bias | 0.8879 | 0.8975 |

Temperature-only versus temperature+bias calibration; selection uses validation Brier (validation ECE breaks exact ties).

| Seed | Readout | Temp Brier | Temp ECE | Temp+bias Brier | Temp+bias ECE | Selected |
|---:|---|---:|---:|---:|---:|---|
| 42 | target_only | 0.2481 | 0.3812 | 0.1030 | 0.0075 | temperature+bias |
| 42 | target_scene | 0.1013 | 0.1402 | 0.0575 | 0.0338 | temperature+bias |
| 123 | target_only | 0.2456 | 0.3787 | 0.1019 | 0.0322 | temperature+bias |
| 123 | target_scene | 0.0929 | 0.0870 | 0.0924 | 0.0737 | temperature+bias |
| 2024 | target_only | 0.2407 | 0.3692 | 0.1027 | 0.0306 | temperature+bias |
| 2024 | target_scene | 0.1122 | 0.1544 | 0.0825 | 0.0659 | temperature+bias |

## Test threshold 0.5 compatibility metrics

| Seed | Readout | BAcc @ 0.5 | F1 @ 0.5 |
|---:|---|---:|---:|
| 42 | target_only | 0.5000 | 0.9324 |
| 42 | target_scene | 0.5008 | 0.9239 |
| 123 | target_only | 0.5000 | 0.9324 |
| 123 | target_scene | 0.5218 | 0.9290 |
| 2024 | target_only | 0.5000 | 0.9324 |
| 2024 | target_scene | 0.5757 | 0.9294 |

## Scene permutation diagnostic

See `scene_permutation_diagnostic.json` for real-, shuffled-, and zero-readout-scene AUC per seed. The permutation is test-only diagnostic (seed 9124), not used for training or selection.
