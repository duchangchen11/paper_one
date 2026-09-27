# Reliability-gated future trajectory intention experiment

Frozen protocol SHA256: `d4535b05241c7ed60eeac30e113ff04b80ebf8af78f79e80464b0e36fde4f4b9`
Official test samples: 18331

## Protocol integrity

- Test labels were read before freeze: `False`; released after freeze: `True`.
- A/B/C/D were evaluated together in one fixed test pass; thresholds and scalar temperatures were fit on official val only.
- Trajectory predictors were frozen for intention training; future_gt/ADE/FDE were not intention inputs, loss, gates, or selection criteria.

## Crossfit trajectory feature construction

All 24556 official train samples have OOF predictions: `True`.
| Fold | Videos | Samples | Seed | Best epoch | Val ADE px | Held-out ADE px | Held-out FDE px |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 48 | 7542 | 42 | 16 | 9.913 | 11.808 | 21.761 |
| 0 | 48 | 7542 | 123 | 20 | 10.046 | 12.070 | 21.824 |
| 0 | 48 | 7542 | 2024 | 16 | 10.241 | 12.267 | 22.880 |
| 1 | 48 | 7165 | 42 | 15 | 9.949 | 10.977 | 19.995 |
| 1 | 48 | 7165 | 123 | 19 | 10.186 | 11.166 | 19.818 |
| 1 | 48 | 7165 | 2024 | 18 | 10.073 | 11.232 | 20.715 |
| 2 | 48 | 9849 | 42 | 19 | 9.800 | 9.834 | 18.118 |
| 2 | 48 | 9849 | 123 | 17 | 10.109 | 10.217 | 17.522 |
| 2 | 48 | 9849 | 2024 | 19 | 9.748 | 9.688 | 17.500 |

## Feature distribution audit

| Split | N | ADE px | FDE px | u_mean mean/std/median px | Motion mean/std/median px | Adjusted-U mean/std/median |
|---|---:|---:|---:|---|---|---|
| Train OOF | 24556 | 10.523 | 19.406 | 2.464/1.056/2.174 | 55.193/53.793/36.734 | 0.000/0.162/-0.022 |
| Val | 2636 | 9.443 | 17.476 | 1.584/0.469/1.473 | 43.254/34.470/31.694 | -0.242/0.115/-0.250 |
| Test | 18331 | 10.472 | 19.111 | 1.847/0.751/1.555 | 52.264/48.175/36.555 | -0.184/0.152/-0.217 |

Reliability polynomial: b0=1.6033214, b1=-0.41989981, b2=0.080509202.
Motion and adjusted-U empirical CDFs were fit on official train OOF only; hashes are in `reliability_transform.json`.

## Observed-only baseline

A uses only the 15-frame observed target history; no future feature is passed to the model. Test mean ± SD: AUC 0.6688 ± 0.0093; Brier 0.2154 ± 0.0210; BAcc 0.6358 ± 0.0099; F1 0.7148 ± 0.0097.
## Always-future

B adds the ensemble-mean predicted future trajectory with a fixed gate of 1. Test mean ± SD: AUC 0.6686 ± 0.0185; Brier 0.2069 ± 0.0058; BAcc 0.6267 ± 0.0157; F1 0.6976 ± 0.0393.
## Motion-gated future

C uses the train-OOF motion empirical confidence as its fixed gate. Test mean ± SD: AUC 0.6758 ± 0.0083; Brier 0.2223 ± 0.0023; BAcc 0.6279 ± 0.0141; F1 0.6995 ± 0.0258.
## Reliability-gated future

D uses the train-OOF motion-adjusted disagreement confidence as its fixed gate. Test mean ± SD: AUC 0.6631 ± 0.0202; Brier 0.2143 ± 0.0057; BAcc 0.6172 ± 0.0187; F1 0.6968 ± 0.0809.
### Per-seed validation/test metrics

AUC, Brier, balanced accuracy, and positive F1:

| Model | Seed | Val AUC | Val Brier | Val BAcc | Val F1 | Test AUC | Test Brier | Test BAcc | Test F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_observed_only | 42 | 0.7007 | 0.2408 | 0.6984 | 0.6041 | 0.6584 | 0.2222 | 0.6468 | 0.7059 |
| A_observed_only | 123 | 0.7400 | 0.2453 | 0.7070 | 0.6206 | 0.6761 | 0.2323 | 0.6329 | 0.7252 |
| A_observed_only | 2024 | 0.7432 | 0.2191 | 0.6785 | 0.5887 | 0.6720 | 0.1919 | 0.6277 | 0.7134 |
| B_always_future | 42 | 0.5749 | 0.2266 | 0.6613 | 0.4878 | 0.6484 | 0.2005 | 0.6103 | 0.6523 |
| B_always_future | 123 | 0.7592 | 0.2248 | 0.7132 | 0.6927 | 0.6849 | 0.2080 | 0.6416 | 0.7184 |
| B_always_future | 2024 | 0.7428 | 0.2368 | 0.6788 | 0.6011 | 0.6726 | 0.2120 | 0.6282 | 0.7222 |
| C_motion_gate | 42 | 0.6602 | 0.2383 | 0.6997 | 0.5919 | 0.6702 | 0.2209 | 0.6365 | 0.6699 |
| C_motion_gate | 123 | 0.7239 | 0.2388 | 0.7129 | 0.6205 | 0.6854 | 0.2250 | 0.6356 | 0.7174 |
| C_motion_gate | 2024 | 0.6504 | 0.2402 | 0.6679 | 0.6685 | 0.6719 | 0.2210 | 0.6115 | 0.7113 |
| D_reliability_gate | 42 | 0.5599 | 0.2326 | 0.6133 | 0.3694 | 0.6422 | 0.2124 | 0.5965 | 0.6095 |
| D_reliability_gate | 123 | 0.7771 | 0.2254 | 0.7368 | 0.7219 | 0.6825 | 0.2098 | 0.6222 | 0.7690 |
| D_reliability_gate | 2024 | 0.6919 | 0.2413 | 0.6820 | 0.5467 | 0.6647 | 0.2207 | 0.6329 | 0.7119 |

## Three-seed summary

All entries are official-test mean ± sample SD (ddof=1) across seeds 42/123/2024.

| Model | roc_auc | brier | ece_15_equal_width | balanced_accuracy | f1_positive | negative_recall_specificity |
|---|---:|---:|---:|---:|---:|---:|
| A_observed_only | 0.6688 ± 0.0093 | 0.2154 ± 0.0210 | 0.3127 ± 0.0491 | 0.6358 ± 0.0099 | 0.7148 ± 0.0097 | 0.6904 ± 0.0314 |
| B_always_future | 0.6686 ± 0.0185 | 0.2069 ± 0.0058 | 0.2941 ± 0.0161 | 0.6267 ± 0.0157 | 0.6976 ± 0.0393 | 0.6928 ± 0.0269 |
| C_motion_gate | 0.6758 ± 0.0083 | 0.2223 ± 0.0023 | 0.3315 ± 0.0021 | 0.6279 ± 0.0141 | 0.6995 ± 0.0258 | 0.6934 ± 0.0546 |
| D_reliability_gate | 0.6631 ± 0.0202 | 0.2143 ± 0.0057 | 0.3114 ± 0.0174 | 0.6172 ± 0.0187 | 0.6968 ± 0.0809 | 0.6694 ± 0.0798 |

## Paired deltas

Each label is first model minus second (for example, D-A = D minus A); positive ΔAUC/ΔBAcc favors the first model, while negative ΔBrier favors the first model.

- B-A: ΔAUC -0.0002 ± 0.0094; ΔBrier -0.0086 ± 0.0249; ΔBAcc -0.0091 ± 0.0241.
- C-B: ΔAUC 0.0072 ± 0.0126; ΔBrier 0.0155 ± 0.0058; ΔBAcc 0.0012 ± 0.0223.
- D-A: ΔAUC -0.0057 ± 0.0113; ΔBrier -0.0011 ± 0.0267; ΔBAcc -0.0186 ± 0.0286.
- D-B: ΔAUC -0.0055 ± 0.0028; ΔBrier 0.0074 ± 0.0052; ΔBAcc -0.0095 ± 0.0126.
- D-C: ΔAUC -0.0127 ± 0.0134; ΔBrier -0.0080 ± 0.0074; ΔBAcc -0.0106 ± 0.0307.

## Video-cluster bootstrap

Paired 95% percentile CIs; each bootstrap draw resamples `scene_id` clusters jointly.

| Comparison | Seed | ΔAUC 95% CI | ΔBrier 95% CI |
|---|---:|---|---|
| B-A | 123 | [-0.0077, 0.0327] | [-0.0337, -0.0135] |
| B-A | 2024 | [-0.0003, 0.0020] | [0.0129, 0.0260] |
| B-A | 42 | [-0.1146, 0.0785] | [-0.0284, -0.0133] |
| D-A | 123 | [-0.0453, 0.0406] | [-0.0311, -0.0126] |
| D-A | 2024 | [-0.1004, 0.0495] | [0.0184, 0.0374] |
| D-A | 42 | [-0.1352, 0.0844] | [-0.0141, -0.0046] |
| D-B | 123 | [-0.0519, 0.0296] | [0.0002, 0.0036] |
| D-B | 2024 | [-0.1009, 0.0484] | [0.0046, 0.0126] |
| D-B | 42 | [-0.0411, 0.0341] | [0.0080, 0.0153] |
| D-C | 123 | [-0.0469, 0.0347] | [-0.0206, -0.0091] |
| D-C | 2024 | [-0.0669, 0.0529] | [-0.0023, 0.0015] |
| D-C | 42 | [-0.1263, 0.0536] | [-0.0123, -0.0042] |

## Reliability-stratified results

### high_uncertainty: N=1516
- A_observed_only: AUC 0.9654 ± 0.0285; Brier 0.1315 ± 0.0308.
- B_always_future: AUC 0.9062 ± 0.1314; Brier 0.1153 ± 0.0052.
- C_motion_gate: AUC 0.9155 ± 0.0584; Brier 0.1360 ± 0.0026.
- D_reliability_gate: AUC 0.8904 ± 0.1229; Brier 0.1431 ± 0.0063.
- B-A high/stratum AUC delta per seed: -0.1780, +0.0007, -0.0004.
- D-A high/stratum AUC delta per seed: -0.1828, -0.0051, -0.0372.
- D-B high/stratum AUC delta per seed: -0.0048, -0.0058, -0.0369.
### low_uncertainty: N=15018
- A_observed_only: AUC 0.6131 ± 0.0064; Brier 0.2299 ± 0.0188.
- B_always_future: AUC 0.6229 ± 0.0066; Brier 0.2228 ± 0.0058.
- C_motion_gate: AUC 0.6304 ± 0.0026; Brier 0.2372 ± 0.0028.
- D_reliability_gate: AUC 0.6215 ± 0.0129; Brier 0.2276 ± 0.0055.
- B-A high/stratum AUC delta per seed: +0.0150, +0.0138, +0.0006.
- D-A high/stratum AUC delta per seed: +0.0025, +0.0175, +0.0053.
- D-B high/stratum AUC delta per seed: -0.0125, +0.0037, +0.0047.
### medium_uncertainty: N=1797
- A_observed_only: AUC 0.8597 ± 0.0131; Brier 0.1649 ± 0.0316.
- B_always_future: AUC 0.8110 ± 0.0974; Brier 0.1509 ± 0.0066.
- C_motion_gate: AUC 0.8296 ± 0.0373; Brier 0.1707 ± 0.0019.
- D_reliability_gate: AUC 0.7692 ± 0.1207; Brier 0.1631 ± 0.0067.
- B-A high/stratum AUC delta per seed: -0.1466, +0.0005, +0.0000.
- D-A high/stratum AUC delta per seed: -0.2127, -0.0097, -0.0490.
- D-B high/stratum AUC delta per seed: -0.0661, -0.0102, -0.0490.

## Gate diagnostics

| Split | Gate | Mean | SD | q10 | q50 | q90 |
|---|---|---:|---:|---:|---:|---:|
| validation | motion_gate | 0.5412 | 0.2748 | 0.1732 | 0.5563 | 0.9285 |
| validation | reliability_gate | 0.9129 | 0.1631 | 0.7513 | 0.9711 | 0.9968 |
| test | motion_gate | 0.5060 | 0.2860 | 0.1278 | 0.5017 | 0.9030 |
| test | reliability_gate | 0.8311 | 0.2459 | 0.3811 | 0.9498 | 0.9931 |

D residual magnitude on test by seed (`|delta_logit|` mean; `|gate × delta_logit|` mean):
- Seed 42: 1.3096; 0.8494.
- Seed 123: 8.6435; 5.2048.
- Seed 2024: 0.6555; 0.4736.

Crossing-label-specific gate statistics and low/medium/high uncertainty residuals are recorded in `gate_diagnostics.json`.

## Gate shuffle

Within train-defined motion deciles, 200 gate permutations per seed; diagnostic only, not used for selection.
- Seed 123: true AUC=0.6825; shuffled AUC=0.6825 ± 0.0017; ΔAUC=-0.0000; true Brier=0.2098; shuffled Brier=0.2091 ± 0.0001; ΔBrier=+0.0007.
- Seed 2024: true AUC=0.6647; shuffled AUC=0.6561 ± 0.0030; ΔAUC=+0.0086; true Brier=0.2207; shuffled Brier=0.2187 ± 0.0001; ΔBrier=+0.0020.
- Seed 42: true AUC=0.6422; shuffled AUC=0.6283 ± 0.0016; ΔAUC=+0.0139; true Brier=0.2124; shuffled Brier=0.2110 ± 0.0001; ΔBrier=+0.0014.

## Limitations

- These are the results of the registered A/B/C/D comparison; they do not establish state-of-the-art performance or superiority over papers with different data/protocols.
- Ensemble disagreement is a deterministic three-checkpoint spread proxy, not a complete estimate of aleatoric uncertainty.
- D effective residual magnitudes are reported above; a near-zero contribution would not support a mechanism-effectiveness claim.
- Gate-shuffle results are diagnostic only and were not used for model selection.
- Future-shuffle diagnostic was not run.

## Tests

Full pytest suite: 78 passed, 0 failed, 0 skipped.
