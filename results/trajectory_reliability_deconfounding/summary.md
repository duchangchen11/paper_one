# Motion-deconfounded trajectory reliability audit

## Protocol and scope

This is a diagnostic of the existing three frozen zero-scene checkpoints (seeds 42/123/2024), not a trained error predictor. Primary score is fixed as `u_mean`; the three-model pixel-space disagreement formula was not reselected. Observed motion is first-to-last observed target-center displacement over the 15 observation frames. No future ground truth enters motion, score construction, adjustment, thresholds, or ranking.

The validation polynomial fits `log1p(u_mean)` from `log1p(observed_motion)` only; coefficients are frozen for test. High-ADE/FDE labels use the previous validation 80th-percentile pixel thresholds: `13.2150` and `24.4552` px.
Shared ordered inference reproduces the prior audit within `1e-5`: `True`. All checkpoint hashes unchanged: `True`.

## Raw U versus motion-only and adjusted U (test)

Motion-only ranking treats lower observed motion as lower risk and retains low-motion samples first. AUPRC depends on the fixed validation-derived positive label prevalence.

| Score | Spearman ADE | Spearman FDE | High-ADE AUROC | High-ADE AUPRC | High-FDE AUROC | High-FDE AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Motion-only (low motion first) | 0.450, p=<1e-300 | 0.399, p=<1e-300 | 0.755 | 0.536 | 0.731 | 0.497 |
| Raw u_mean | 0.428, p=<1e-300 | 0.390, p=<1e-300 | 0.740 | 0.484 | 0.731 | 0.450 |
| Adjusted u_mean | 0.229, p=1.1e-216 | 0.199, p=1.7e-162 | 0.625 | 0.367 | 0.619 | 0.353 |

## Partial Spearman: U conditional on observed motion

Computed by ranking U/error/motion, separately regressing ranked U and ranked error on intercept + ranked motion, then correlating residuals.

| Split | Partial rho: U↔ADE\|Motion | Partial rho: U↔FDE\|Motion |
|---|---:|---:|
| validation | 0.048 | 0.062 |
| test | 0.204 | 0.191 |

## Validation-fitted motion adjustment

Coefficients: `b0=1.17428343`, `b1=-0.27236336`, `b2=0.05523738`. Formula: `adjusted_u = log1p(u_mean) - (b0 + b1*x + b2*x²)`, `x=log1p(observed_motion)`. Fit split: `validation`; ADE/FDE and future ground truth not used.

| Split | Raw U vs motion Spearman | Adjusted U vs motion Spearman |
|---|---:|---:|
| validation | 0.584, p=3.1e-241 | -0.228, p=2.5e-32 |
| test | 0.639, p=<1e-300 | 0.124, p=4.3e-64 |

Adjusted U test reliability bins use q33/q67 cutpoints fit on validation only:

| Test bin | N | Mean adjusted U | Mean ADE | Median ADE | Mean FDE | Median FDE |
|---|---:|---:|---:|---:|---:|---:|
| low | 3492 | -0.089 | 9.405 | 7.388 | 17.237 | 13.079 |
| medium | 5132 | -0.016 | 8.556 | 7.140 | 15.377 | 12.445 |
| high | 9707 | 0.160 | 11.870 | 9.629 | 21.760 | 16.730 |
Low→medium→high adjusted-score bin errors monotonic: {'ade': False, 'fde': False}.

## Global risk–coverage comparison

Lower score is retained first. Random is mean±SD from 100 rankings; oracle sorts by true ADE and is only an unattainable upper bound.

| Coverage | Raw U ADE/FDE | Adjusted U ADE/FDE | Motion-only ADE/FDE | Random ADE/FDE | Oracle ADE/FDE |
|---:|---:|---:|---:|---:|---:|
| 100% | 10.472/19.111 | 10.472/19.111 | 10.472/19.111 | 10.472±0.000/19.111±0.000 | 10.472/19.111 |
| 90% | 9.537/17.376 | 9.915/18.057 | 9.353/17.014 | 10.476±0.017/19.119±0.037 | 8.470/15.230 |
| 80% | 8.881/16.213 | 9.490/17.231 | 8.663/15.783 | 10.474±0.028/19.118±0.057 | 7.408/13.269 |
| 70% | 8.405/15.241 | 9.120/16.485 | 8.283/15.082 | 10.478±0.034/19.127±0.075 | 6.578/11.806 |
| 60% | 8.022/14.388 | 8.972/16.234 | 8.021/14.558 | 10.476±0.045/19.121±0.102 | 5.857/10.510 |
| 50% | 7.704/13.619 | 8.912/16.141 | 7.809/14.095 | 10.477±0.064/19.129±0.139 | 5.191/9.324 |
| 40% | 7.433/12.950 | 8.948/16.288 | 7.602/13.638 | 10.474±0.077/19.120±0.158 | 4.567/8.238 |
| 30% | 7.321/12.698 | 9.128/16.721 | 7.334/13.074 | 10.481±0.089/19.132±0.195 | 3.958/7.201 |
| 20% | 7.247/12.581 | 9.379/17.155 | 7.163/12.786 | 10.476±0.129/19.116±0.271 | 3.326/6.059 |

## Motion-stratified risk–coverage

Test is split into validation-defined motion deciles; raw or adjusted U ranks samples separately within each decile before pooling. The within-motion random column is the 100-permutation null reference.

| Nominal coverage | Raw U ADE/FDE | Adjusted U ADE/FDE | Within-motion random ADE/FDE | Actual coverage raw/adjusted |
|---:|---:|---:|---:|---:|
| 100% | 10.472/19.111 | 10.472/19.111 | 10.472±0.000/19.111±0.000 | 100.0%/100.0% |
| 90% | 10.182/18.459 | 10.186/18.510 | 10.472±0.017/19.110±0.037 | 90.0%/90.0% |
| 80% | 9.992/18.023 | 10.035/18.166 | 10.473±0.028/19.111±0.058 | 80.0%/80.0% |
| 70% | 9.833/17.695 | 9.916/17.920 | 10.475±0.031/19.116±0.071 | 70.0%/70.0% |
| 60% | 9.745/17.511 | 9.808/17.708 | 10.475±0.039/19.117±0.091 | 60.0%/60.0% |
| 50% | 9.637/17.319 | 9.736/17.576 | 10.475±0.049/19.117±0.112 | 50.0%/50.0% |
| 40% | 9.567/17.210 | 9.718/17.569 | 10.483±0.059/19.136±0.137 | 40.0%/40.0% |
| 30% | 9.514/17.061 | 9.703/17.559 | 10.491±0.078/19.145±0.172 | 30.0%/30.0% |
| 20% | 9.273/16.642 | 9.472/17.137 | 10.494±0.109/19.148±0.227 | 20.0%/20.0% |

Within-motion permutation sanity check (small lower-tail p means the observed ranking beats randomized U within the same motion deciles):

| Coverage | Raw/adjusted observed ADE | Random ADE mean±SD | Empirical p raw/adjusted |
|---:|---:|---:|---:|
| 80% | 9.992/10.035 | 10.473±0.028 | 0.010/0.010 |
| 50% | 9.637/9.736 | 10.475±0.049 | 0.010/0.010 |
| 20% | 9.273/9.472 | 10.494±0.109 | 0.010/0.010 |
At 80% coverage, adjusted-U empirical lower-tail p for ADE = `0.010`; raw-U = `0.010`.

## Motion-stratified correlations

Slow/medium/fast boundaries are validation q33=`22.338px`, q67=`52.043px`.

| Test stratum | N | Raw U↔ADE | Adjusted U↔ADE | Motion↔ADE | Raw U↔FDE | Adjusted U↔FDE |
|---|---:|---:|---:|---:|---:|---:|
| slow | 5813 | 0.160, p=1.8e-34 | 0.173, p=3.4e-40 | 0.134, p=1.4e-24 | 0.158, p=1e-33 | 0.165, p=6.4e-37 |
| medium | 5908 | 0.121, p=7.5e-21 | 0.103, p=2.8e-15 | 0.054, p=3.2e-05 | 0.144, p=7e-29 | 0.110, p=2.1e-17 |
| fast | 6610 | 0.401, p=1.4e-253 | 0.291, p=4.5e-129 | 0.426, p=5.7e-289 | 0.319, p=1.7e-156 | 0.222, p=1.7e-74 |

Validation-defined motion deciles (test samples):

| Decile | N | Raw U↔ADE | Adjusted U↔ADE | Raw U↔FDE | Adjusted U↔FDE |
|---|---:|---:|---:|---:|---:|
| D1 | 1355 | 0.052, p=0.058 | 0.092, p=0.00066 | 0.051, p=0.058 | 0.068, p=0.013 |
| D2 | 1819 | 0.145, p=4.8e-10 | 0.147, p=2.9e-10 | 0.135, p=7.7e-09 | 0.136, p=5.7e-09 |
| D3 | 2007 | 0.211, p=1.2e-21 | 0.212, p=8.7e-22 | 0.210, p=2e-21 | 0.213, p=6.3e-22 |
| D4 | 1581 | 0.091, p=0.00029 | 0.086, p=0.00062 | 0.150, p=1.8e-09 | 0.144, p=9.4e-09 |
| D5 | 1431 | 0.011, p=0.67 | 0.011, p=0.66 | 0.054, p=0.039 | 0.054, p=0.04 |
| D6 | 2275 | 0.149, p=8e-13 | 0.157, p=5.5e-14 | 0.142, p=9e-12 | 0.148, p=1.5e-12 |
| D7 | 1617 | 0.183, p=1.3e-13 | 0.185, p=5.9e-14 | 0.204, p=1e-16 | 0.209, p=2.1e-17 |
| D8 | 1797 | 0.269, p=3.1e-31 | 0.272, p=7e-32 | 0.250, p=5e-27 | 0.256, p=2.9e-28 |
| D9 | 1558 | 0.161, p=1.6e-10 | 0.149, p=3.2e-09 | 0.111, p=1.2e-05 | 0.099, p=9e-05 |
| D10 | 2891 | 0.123, p=3.8e-11 | 0.053, p=0.0045 | 0.093, p=5.7e-07 | 0.011, p=0.56 |

## Cluster bootstrap

Primary video-cluster bootstrap resamples `97` `scene_id` clusters with replacement and keeps every sample in each selected video together (1000 replicates, seed 9124). Track-cluster bootstrap is secondary.

| Cluster unit | Score | 95% CI Spearman ADE | 95% CI High-ADE AUROC | Valid rho replicates |
|---|---|---:|---:|---:|
| Video: scene_id | raw_u_mean | 0.321–0.501 | 0.698–0.769 | 1000/1000 |
| Video: scene_id | adjusted_u_mean | 0.117–0.312 | 0.569–0.662 | 1000/1000 |
| Video: scene_id | motion_only | 0.353–0.535 | 0.711–0.794 | 1000/1000 |
| Track: (scene_id,target_id) | raw_u_mean | 0.361–0.483 | 0.707–0.772 | 1000/1000 |
| Track: (scene_id,target_id) | adjusted_u_mean | 0.153–0.296 | 0.584–0.663 | 1000/1000 |

## Normalized-coordinate and image-scale checks

Normalized analysis is secondary; it does not replace pixel-space primary results. `image_size` correlations are undefined when a dimension is constant.

| Split | Normalized U↔ADE rho | Normalized U↔FDE rho | Mean normalized U | Mean normalized ADE |
|---|---:|---:|---:|---:|
| validation | 0.221, p=1.9e-30 | 0.240, p=7.5e-36 | 0.000995 | 0.006592 |
| test | 0.394, p=<1e-300 | 0.386, p=<1e-300 | 0.001144 | 0.006943 |
Test image resolution pairs: `[[1920, 1080]]`; unique widths/heights = `1/1`. Width/height correlations with raw U and ADE are `NA`, `NA`, `NA`, `NA` respectively.
Future endpoint displacement correlations, if present in `adjusted_score.json`, are post-hoc diagnostics only.

## Final decision: GO

GO requires a majority of the six evidence checks (at least four); STOP requires all four null-pattern checks; otherwise WEAK. This operationalizes the task's instruction to satisfy most GO criteria without tuning a model on test.
Evidence check count: `5/6`. Not met: `adjusted_high_ade_auroc_at_least_0_65_or_plus_0_03_vs_motion`.
Raw `u_mean` does not outperform motion-only globally on ADE correlation or High-ADE AUROC; the GO result is specifically based on positive conditional/within-motion evidence and cluster-bootstrap support, not global superiority.

This decision concerns whether to consider a later reliability-gated intention study. No intention model or gate was started in this task.

## Integrity and artifacts

- No trajectory retraining, intention/social/scene model, ADE predictor, or test-fitted adjustment was run.
- Checkpoint hashes were verified unchanged after inference; original predictions/arrays were not modified in place.
- No processed data or per-sample prediction dump is produced.
- Detailed machine-readable outputs are in this directory: partial correlation, motion baselines, adjustment, risk coverage/permutation, motion strata, cluster bootstrap, scale diagnostics, and checkpoint audit JSON.
