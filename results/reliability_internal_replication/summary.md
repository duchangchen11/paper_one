# Independent internal-video replication of motion-controlled trajectory reliability

## Protocol and blind holdout

Only `data/processed/jaad_sequences_scene_15x15/train.npz` was used to create the internal split, train models, select checkpoints, fit motion adjustment, set high-error thresholds, and make the decision. Official `val.npz` and `test.npz` were not opened by the experiment. The initial frozen PHASE D computation was interrupted; one explicitly authorized recovery re-executed the same frozen evaluation without changing models or protocol.
Manifest seed `314159`; train.npz SHA256 `6da2d1d40a918a218a7d054407d95f2791e2338a760b4836f2ca6d06c59ea19a`; canonical manifest SHA256 `41270ddb5ec35f777f37956377e4e530381935d1ae44bccc0c74b9c058aaa0ea`.
Frozen protocol SHA256 `777b9131e163176bc0eaa7eca4997784ddb32a1ea8195a1c1eae842d29b7675b`; recovery status `completed`; authorized recovery attempts `1`.
Frozen checkpoint SHA256 values: seed 42 `0f9efd94e0412c6710299d7f0d323deb7a37c0ea4588014e6ff18422f2512f9e`; seed 123 `cf9c84f5e806c3ca99e75a8a6fc34d3649794389dc1859780a99a7a405cb91d3`; seed 2024 `0f74da9131a5f2b4f97d71411149e3b5677e118584c305f24266d80b53fe6190`.

| Internal split | Videos | Samples | Unique target IDs |
|---|---:|---:|---:|
| internal_train | 101 | 17251 | 181 |
| internal_val | 22 | 4028 | 40 |
| internal_holdout | 21 | 3277 | 43 |
Scene overlap counts: `{'train_val': 0, 'train_holdout': 0, 'val_holdout': 0}`. All pairwise overlaps are zero: `True`.
`holdout_evaluated_after_protocol_frozen`: `True`; one-time access record status: `completed`.
Shared ordered inference sample metadata SHA256: `766d3969ababdf3d6749a177084e3e1b2fcc4790736419afce973c734ebdaba4` for `3277` rows (ordered scene_id, target_id, obs_end_frame recorded in holdout JSON).

## Frozen trajectory ensemble

All three models were trained from fresh random initialization with zero scene input. Checkpoint selection used only the lowest internal-val pixel ADE; no holdout was used for epoch selection.

| Model | Best epoch | Internal-val ADE/FDE px | Holdout ADE/FDE px |
|---|---:|---:|---:|
| seed 42 | 20 | 10.980/20.342 | 10.320/19.217 |
| seed 123 | 18 | 11.242/19.917 | 10.611/19.021 |
| seed 2024 | 18 | 10.922/19.510 | 10.369/18.095 |
| Ensemble mean | — | 10.609/19.460 | 10.010/18.418 |

## Internal-validation reliability (descriptive; protocol fitting split)

Primary score remained fixed as pixel `u_mean`. Motion is first-to-last observed target-center displacement from the 15 input frames. High-error thresholds and the quadratic motion adjustment were fit only on internal_val.

Internal-val fixed high-error thresholds: ADE `14.9602px`; FDE `26.8109px`.
Adjustment: `adjusted_u=log1p(u_mean)-(b0+b1*x+b2*x²)`, `x=log1p(motion)`; coefficients b0/b1/b2 = `1.42515952`, `-0.26737580`, `0.05741443`.

| Score | Val rho ADE | Val rho FDE | Holdout rho ADE | Holdout rho FDE | Holdout High-ADE AUROC/AUPRC | Holdout High-FDE AUROC/AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Raw u_mean | 0.560, p=<1e-300 | 0.466, p=1.1e-216 | 0.456, p=2.3e-168 | 0.419, p=3.4e-139 | 0.779/0.502 | 0.799/0.521 |
| Motion-only | 0.532, p=1e-293 | 0.460, p=7.7e-210 | 0.439, p=4e-154 | 0.434, p=2.9e-150 | 0.782/0.567 | 0.800/0.573 |
| Adjusted u_mean | 0.188, p=3e-33 | 0.141, p=2e-19 | 0.173, p=2.3e-23 | 0.148, p=1.7e-17 | 0.595/0.309 | 0.612/0.330 |
Partial Spearman raw U↔ADE|Motion: internal-val `0.340`, holdout `0.292`; U↔FDE|Motion: internal-val `0.255`, holdout `0.245`.

## Independent internal holdout: motion-controlled analyses

### Slow / medium / fast

| Stratum | N | Raw U↔ADE | Adjusted U↔ADE | Motion↔ADE | Raw U↔FDE | Adjusted U↔FDE |
|---|---:|---:|---:|---:|---:|---:|
| slow | 733 | 0.120, p=0.0011 | 0.111, p=0.0026 | 0.312, p=5.3e-18 | 0.107, p=0.0036 | 0.094, p=0.011 |
| medium | 1842 | 0.284, p=1.9e-35 | 0.232, p=6.9e-24 | 0.193, p=7.2e-17 | 0.265, p=4.9e-31 | 0.202, p=1.8e-18 |
| fast | 702 | 0.481, p=5.2e-42 | 0.331, p=2e-19 | 0.529, p=5.9e-52 | 0.425, p=4.4e-32 | 0.294, p=1.7e-15 |

### Internal-val motion deciles

| Decile | N | Raw U↔ADE rho | Adjusted U↔ADE rho |
|---|---:|---:|---:|
| D1 | 134 | 0.141, p=0.1 | 0.076, p=0.38 |
| D2 | 287 | 0.162, p=0.0059 | 0.167, p=0.0045 |
| D3 | 210 | 0.066, p=0.34 | 0.061, p=0.38 |
| D4 | 277 | 0.028, p=0.64 | 0.034, p=0.58 |
| D5 | 797 | 0.167, p=2e-06 | 0.158, p=7.8e-06 |
| D6 | 643 | 0.308, p=1.2e-15 | 0.318, p=1.5e-16 |
| D7 | 276 | 0.336, p=1.1e-08 | 0.331, p=1.8e-08 |
| D8 | 264 | 0.566, p=9.3e-24 | 0.525, p=4e-20 |
| D9 | 214 | 0.045, p=0.52 | -0.041, p=0.55 |
| D10 | 175 | 0.125, p=0.1 | 0.171, p=0.023 |

### Global risk–coverage (ADE/FDE px)

Random is 100 random rankings (mean±SD). Oracle ranks by holdout ADE and is an upper bound only.

| Coverage | Raw U | Adjusted U | Motion-only | Random | Oracle |
|---:|---:|---:|---:|---:|---:|
| 100% | 10.010/18.418 | 10.010/18.418 | 10.010/18.418 | 10.010±0.000/18.418±0.000 | 10.010/18.418 |
| 90% | 9.032/16.432 | 9.479/17.479 | 8.642/15.454 | 10.010±0.042/18.419±0.097 | 7.997/14.141 |
| 80% | 8.463/15.176 | 8.991/16.428 | 8.091/14.280 | 10.007±0.059/18.419±0.136 | 6.939/12.210 |
| 70% | 7.739/13.620 | 8.832/16.169 | 7.807/13.547 | 10.007±0.080/18.413±0.175 | 6.078/10.738 |
| 60% | 7.056/12.217 | 8.894/16.362 | 7.587/12.901 | 10.007±0.089/18.408±0.200 | 5.341/9.467 |
| 50% | 6.847/11.854 | 9.140/16.778 | 7.280/12.377 | 9.997±0.114/18.376±0.262 | 4.691/8.298 |
| 40% | 6.642/11.464 | 9.363/17.336 | 7.169/12.193 | 9.993±0.145/18.371±0.311 | 4.105/7.311 |
| 30% | 6.810/11.603 | 9.657/18.055 | 6.958/11.628 | 9.987±0.178/18.348±0.386 | 3.545/6.345 |
| 20% | 7.237/12.085 | 10.307/19.300 | 6.891/11.209 | 9.980±0.269/18.331±0.581 | 3.017/5.377 |

### Motion-stratified risk–coverage (ADE/FDE px)

Within each internal-val motion decile, retain the lowest-score fraction and then pool. Random is the 500-permutation within-motion reference.

| Coverage | Raw U | Adjusted U | Within-motion random mean±SD |
|---:|---:|---:|---:|
| 100% | 10.010/18.418 | 10.010/18.418 | 10.010±0.000/18.418±0.000 |
| 90% | 9.605/17.716 | 9.603/17.717 | 10.006±0.037/18.411±0.081 |
| 80% | 9.367/17.261 | 9.393/17.337 | 10.011±0.055/18.419±0.121 |
| 70% | 9.196/16.972 | 9.214/16.969 | 10.006±0.072/18.410±0.156 |
| 60% | 8.960/16.491 | 9.034/16.629 | 10.008±0.089/18.414±0.193 |
| 50% | 8.807/16.113 | 8.938/16.414 | 10.010±0.109/18.417±0.232 |
| 40% | 8.696/15.794 | 8.725/15.988 | 10.011±0.136/18.426±0.290 |
| 30% | 8.770/15.962 | 8.652/15.801 | 10.000±0.165/18.404±0.364 |
| 20% | 9.029/16.096 | 9.301/16.834 | 9.998±0.225/18.398±0.484 |

### Within-motion permutation (lower-tail empirical p)

| Coverage | ADE p raw/adjusted | FDE p raw/adjusted |
|---:|---:|---:|
| 80% | 0.002/0.002 | 0.002/0.002 |
| 50% | 0.002/0.002 | 0.002/0.002 |
| 20% | 0.002/0.002 | 0.002/0.002 |
The permutation minimum is `1/501`; interpret it as a finite-resolution reference, not extreme certainty.

## Cluster bootstrap

Primary video bootstrap uses `21` holdout videos and 2000 resamples; secondary track bootstrap uses `43` tracks and 1000 resamples. With only about 21 videos, video-cluster intervals are necessarily sensitive to the small number of clusters.

| Cluster unit | Score | 95% CI Spearman ADE | 95% CI High-ADE AUROC | Valid rho replicates |
|---|---|---:|---:|---:|
| Video scene_id | raw_u_mean | 0.272–0.571 | 0.677–0.837 | 2000/2000 |
| Video scene_id | adjusted_u_mean | 0.029–0.278 | 0.523–0.675 | 2000/2000 |
| Video scene_id | motion_only | 0.216–0.610 | 0.668–0.857 | 2000/2000 |
| Track scene_id,target_id | raw_u_mean | 0.330–0.550 | 0.704–0.832 | 1000/1000 |
| Track scene_id,target_id | adjusted_u_mean | 0.039–0.285 | 0.523–0.661 | 1000/1000 |

## Normalized-space sanity check

Holdout normalized-coordinate Spearman: u_mean↔ADE `0.375, p=4.7e-110`, u_mean↔FDE `0.363, p=2.3e-102`. Pixel remains the primary unit.

## Post-hoc comparison with prior official-test diagnostic

This section was generated only after the internal holdout metrics and decision were frozen. It did not affect any training, checkpoint, adjustment, threshold, or decision step.

| Diagnostic | Previous official test | New independent internal holdout |
|---|---:|---:|
| Partial Spearman U↔ADE\|Motion | 0.204 | 0.292 |
| Partial Spearman U↔FDE\|Motion | 0.191 | 0.245 |
| Adjusted U↔ADE rho | 0.229 | 0.173, p=2.3e-23 |
| Raw U↔ADE rho | 0.428 | 0.456, p=2.3e-168 |
| Motion-only↔ADE rho | 0.450 | 0.439, p=4e-154 |

## Final decision: REPLICATED

REPLICATED requires at least 4/5 frozen criteria and adjusted video-cluster rho CI lower bound > 0; PARTIAL for 2-3 criteria or insufficient cluster certainty; NOT_REPLICATED for <=1 criterion.
Frozen criteria passed: `5/5`. Decision source: `internal_holdout only`. Official validation/test were not used for the decision: `True`.

| Frozen replication condition | Passed |
|---|---:|
| partial_spearman_u_ade_given_motion_ge_0_15 | True |
| adjusted_u_ade_spearman_ge_0_15 | True |
| video_cluster_adjusted_rho_ci_lower_gt_0 | True |
| adjusted_stratified_20pct_ade_at_least_5pct_better_than_permutation_mean | True |
| at_least_two_motion_tertiles_raw_or_adjusted_rho_gt_0_10 | True |

No intention classifier or reliability gate was trained. The experiment stops here as instructed.
