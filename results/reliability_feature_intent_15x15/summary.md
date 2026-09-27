# Reliability as an intention feature: frozen JAAD experiment

- Protocol SHA-256: `a863306a8b9c8c36e0e67d485d2c077d694b08556b5f2d4291d8cc30d635eea6`
- Labeled test cache SHA-256: `d3e77f33978e657bcdf5b66bfa0d3efb21948a29342dc556b67373bdeecb69a6`
- Samples: 24,556 train OOF; 2,636 validation; 18,331 test.
- Trajectory predictors were frozen; no future ground truth, ADE/FDE, scene, or social inputs were used.
- Normalization was fit only on official train OOF; validation-fitted temperatures and thresholds were frozen before test labels were decoded.
- Test labels were used only for this single frozen evaluation, not training, selection, or normalization.

## Test performance (mean ± sample SD across 3 seeds)

| Model | ROC-AUC | Brier ↓ | ECE ↓ | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| A: Observed-only | 0.6759 ± 0.0054 | 0.2242 ± 0.0398 | 0.3221 ± 0.0851 | 0.6376 ± 0.0054 | 0.7354 ± 0.0110 |
| B: Observed + future trajectory | 0.6749 ± 0.0092 | 0.2447 ± 0.0091 | 0.3670 ± 0.0127 | 0.6172 ± 0.0292 | 0.7787 ± 0.0913 |
| C: Observed + future + raw reliability | 0.6459 ± 0.0070 | 0.2504 ± 0.0008 | 0.3750 ± 0.0011 | 0.5947 ± 0.0011 | 0.6164 ± 0.0210 |
| D: Observed + future + motion-adjusted reliability | 0.6012 ± 0.0679 | 0.2344 ± 0.0181 | 0.3354 ± 0.0380 | 0.5998 ± 0.0088 | 0.6151 ± 0.2552 |
| E: Observed + future + motion + adjusted reliability | 0.6152 ± 0.0048 | 0.2485 ± 0.0016 | 0.3727 ± 0.0020 | 0.5734 ± 0.0132 | 0.5316 ± 0.0551 |
| D_no_future: Observed + adjusted reliability (no future) | 0.5934 ± 0.0259 | 0.2275 ± 0.0115 | 0.3105 ± 0.0363 | 0.5733 ± 0.0179 | 0.6705 ± 0.0777 |

## Main finding

The primary adjusted-reliability model D did not improve discrimination over future-only B (ΔAUC -0.0737); its Brier score was -0.0103 lower, so discrimination and calibration move in different directions. Motion adjustment did not improve AUC over raw reliability C (ΔAUC -0.0447), although D's Brier was -0.0160 lower. Adding motion in E changed AUC by +0.0139 and Brier by +0.0141; this is not a consistent gain. Overall, this run does not support the hypothesis that these reliability features improve intention recognition under the frozen setup.


## Paired primary comparisons

Deltas are candidate minus baseline; positive ΔAUC and negative ΔBrier favor the candidate. Bootstrap resamples `scene_id` clusters jointly (2,000 draws per seed).

| Comparison | Mean ΔAUC | Mean ΔBrier | Seed 42 paired 95% CI | Seed 123 paired 95% CI | Seed 2024 paired 95% CI |
|---|---:|---:|---|---|---|
| D-B | -0.0737 ± 0.0639 | -0.0103 ± 0.0116 | AUC [-0.2039, 0.0807]; Brier [-0.0319, -0.0067] | AUC [-0.2438, 0.0042]; Brier [-0.0369, 0.0091] | AUC [-0.0865, 0.0503]; Brier [0.0002, 0.0050] |
| D-A | -0.0746 ± 0.0667 | 0.0102 ± 0.0234 | AUC [-0.1866, 0.0436]; Brier [0.0198, 0.0502] | AUC [-0.2357, -0.0145]; Brier [-0.0280, 0.0117] | AUC [-0.0859, 0.0623]; Brier [0.0003, 0.0043] |
| D-C | -0.0447 ± 0.0630 | -0.0160 ± 0.0187 | AUC [-0.1729, 0.0884]; Brier [-0.0618, -0.0093] | AUC [-0.2212, 0.0747]; Brier [-0.0353, 0.0105] | AUC [-0.0486, 0.1114]; Brier [-0.0008, 0.0027] |
| E-D | 0.0139 ± 0.0648 | 0.0141 ± 0.0184 | AUC [-0.0944, 0.1293]; Brier [0.0070, 0.0585] | AUC [-0.0703, 0.1711]; Brier [-0.0107, 0.0342] | AUC [-0.1362, 0.0211]; Brier [-0.0055, -0.0009] |

## Reliability strata

Cutpoints were fixed at train-OOF adjusted-u q33/q67. Lower adjusted-u means comparatively higher reliability; the highest adjusted-u tertile is the low-reliability/high-uncertainty group.

| Stratum | n | B AUC | D AUC | ΔAUC D-B (per seed) | B Brier | D Brier |
|---|---:|---:|---:|---|---:|---:|
| high_reliability | 15,018 | 0.6213 ± 0.0036 | 0.5626 ± 0.0588 | -0.0680, -0.1087, +0.0006 | 0.2490 ± 0.0040 | 0.2461 ± 0.0115 |
| medium_reliability | 1,797 | 0.8689 ± 0.0071 | 0.6655 ± 0.1460 | -0.2500, -0.3163, -0.0440 | 0.2291 ± 0.0299 | 0.2048 ± 0.0454 |
| low_reliability | 1,516 | 0.9777 ± 0.0055 | 0.8243 ± 0.1000 | -0.1776, -0.2382, -0.0444 | 0.2207 ± 0.0357 | 0.1535 ± 0.0792 |

Applying the fixed train-OOF cutpoints yields 15,018/18,331 test samples in high_reliability; this imbalance is reported as observed and was not corrected with test-derived quantiles.

## Feature distributions

| Split | Feature | Mean | SD | Median | q05–q95 |
|---|---|---:|---:|---:|---:|
| Train OOF | raw_u | 2.4639 | 1.0560 | 2.1737 | 1.6025–4.4390 |
| Train OOF | adjusted_u | 0.0000 | 0.1620 | -0.0218 | -0.2170–0.2872 |
| Train OOF | motion | 55.1932 | 53.7933 | 36.7338 | 6.0828–175.6221 |
| Validation | raw_u | 1.5841 | 0.4694 | 1.4732 | 1.2522–2.3843 |
| Validation | adjusted_u | -0.2417 | 0.1154 | -0.2495 | -0.4162–-0.0502 |
| Validation | motion | 43.2541 | 34.4696 | 31.6938 | 4.5000–108.3840 |
| Test (features only) | raw_u | 1.8467 | 0.7509 | 1.5554 | 1.2570–3.4113 |
| Test (features only) | adjusted_u | -0.1839 | 0.1515 | -0.2167 | -0.3744–0.0960 |
| Test (features only) | motion | 52.2635 | 48.1749 | 36.5548 | 6.0208–160.9445 |

## Reliability-branch diagnostic

The following are first-layer weight norms, not causal feature attributions and not computed from test labels.

| Model | Seed | Input-column L2 norms |
|---|---:|---|
| C | 123 | u_mean_pixel=3.4136 |
| C | 2024 | u_mean_pixel=3.3161 |
| C | 42 | u_mean_pixel=3.3467 |
| D | 123 | adjusted_u=3.3924 |
| D | 2024 | adjusted_u=3.2614 |
| D | 42 | adjusted_u=3.3159 |
| D_no_future | 123 | adjusted_u=3.3516 |
| D_no_future | 2024 | adjusted_u=3.3394 |
| D_no_future | 42 | adjusted_u=3.9528 |
| E | 123 | adjusted_u=2.4655, observed_motion_pixel=2.2496 |
| E | 2024 | adjusted_u=2.1125, observed_motion_pixel=2.5040 |
| E | 42 | adjusted_u=2.3008, observed_motion_pixel=2.5278 |

## Protocol and leakage audit

- Optimizer/config: AdamW, lr=0.001, batch=256, epochs=20, balanced BCE unchanged from the previous frozen run.
- Training seeds: 42, 123, 2024; normalization SHA: `4fff020ed79014da126abe200e23028acbb824389614dff36e1f9cbd9be0084c`.
- Matched B/C/D/E observed/future initializations were checked: 3 seeds passed.
- Test access record: labels read after protocol freeze = `True`; used for training/selection/normalization = `False`.
- D_raw is the same trained model as C; D_adjusted is the same trained model as D. D_no_future is the separate observed + adjusted-reliability ablation.
