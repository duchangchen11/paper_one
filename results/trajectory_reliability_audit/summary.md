# Zero-scene trajectory ensemble reliability audit

Feasibility audit only. No trajectory was retrained; no uncertainty predictor, intention classifier, social input, or scene input was used. All errors and disagreement scores are in pixels.

## Ensemble performance

| Model | Validation ADE/FDE px | Test ADE/FDE px |
|---|---:|---:|
| seed 42 | 9.484/17.491 | 10.512/19.023 |
| seed 123 | 9.826/18.051 | 10.908/19.392 |
| seed 2024 | 9.603/17.285 | 10.631/19.365 |
| Ensemble mean | 9.443/17.476 | 10.472/19.111 |

All three individual pixel ADE/FDE values reproduced their saved experiment metrics within 1e-4 px: validation `True`, test `True`.

## Validation score selection

High-error thresholds are the validation 80th percentiles and are frozen in pixel units before test evaluation.
ADE threshold = `13.2150 px`; FDE threshold = `24.4552 px`.

| Score | Spearman vs ADE (rho, p) | Spearman vs FDE (rho, p) | High-ADE AUROC | High-FDE AUROC |
|---|---:|---:|---:|---:|
| u_mean | 0.274 (p=1.2e-46) | 0.277 (p=1.6e-47) | 0.717 | 0.729 |
| u_endpoint | 0.110 (p=1.4e-08) | 0.178 (p=3.8e-20) | 0.663 | 0.702 |
| u_pairwise | 0.283 (p=1.4e-49) | 0.282 (p=1.5e-49) | 0.719 | 0.731 |
| u_endpoint_pairwise | 0.116 (p=2.6e-09) | 0.185 (p=1.2e-21) | 0.662 | 0.699 |

Primary score selected using validation only: **u_mean — Mean trajectory spread**. maximize Spearman(U,ADE); scores within 0.01 rho tie; among them prefer higher validation High-ADE AUROC (within 0.01 tie), then prefer simpler U_mean.

## Test reliability of selected score

- Spearman vs ADE: 0.428 (p=<1e-300); bootstrap 95% CI `[0.416, 0.439]` (1000/1000 valid replicates).
- Spearman vs FDE: 0.390 (p=<1e-300).
- High-ADE (fixed validation threshold) AUROC/AUPRC: 0.740/0.484; AUROC bootstrap 95% CI `[0.732, 0.749]`.
- High-FDE (fixed validation threshold) AUROC/AUPRC: 0.731/0.450.
- Secondary test-top20% diagnostic AUROC (not used for selection): ADE `0.753`, FDE `0.741`.

## Reliability bins

Low/medium/high cutpoints were validation q33=`1.3748` and q67=`1.5638`; applied unchanged to test.

| Test bin | N | Mean U | Mean ADE | Median ADE | Mean FDE | Median FDE |
|---|---:|---:|---:|---:|---:|---:|
| low | 4545 | 1.302 | 7.243 | 6.243 | 12.557 | 10.704 |
| medium | 4765 | 1.460 | 8.196 | 6.988 | 14.760 | 12.227 |
| high | 9021 | 2.326 | 13.302 | 10.962 | 24.712 | 19.717 |
- Low→medium→high error monotonicity: ADE `True`, FDE `True`.
- Validation-derived uncertainty decile boundaries and validation/test bin statistics are in `reliability_bins.csv`.

## Risk-coverage

Keep the lowest-uncertainty fraction. Random reference is mean±sample-SD over 100 rankings (seed 9124); oracle ranks by true ADE and is not deployable.

| Coverage | Selected ADE/FDE | Random ADE / FDE | Oracle-ADE-ranked ADE/FDE |
|---:|---:|---:|---:|
| 100% | 10.472/19.111 | 10.472±0.000 / 19.111±0.000 | 10.472/19.111 |
| 90% | 9.537/17.376 | 10.476±0.017 / 19.119±0.037 | 8.470/15.230 |
| 80% | 8.881/16.213 | 10.474±0.028 / 19.118±0.057 | 7.408/13.269 |
| 70% | 8.405/15.241 | 10.478±0.034 / 19.127±0.075 | 6.578/11.806 |
| 60% | 8.022/14.388 | 10.476±0.045 / 19.121±0.102 | 5.857/10.510 |
| 50% | 7.704/13.619 | 10.477±0.064 / 19.129±0.139 | 5.191/9.324 |
| 40% | 7.433/12.950 | 10.474±0.077 / 19.120±0.158 | 4.567/8.238 |
| 30% | 7.321/12.698 | 10.481±0.089 / 19.132±0.195 | 3.958/7.201 |
| 20% | 7.247/12.581 | 10.476±0.129 / 19.116±0.271 | 3.326/6.059 |
- Selected ADE curve non-increasing as coverage drops: `True`; FDE: `True`.
- Selected 100%→20% reduction: ADE `3.225px`, FDE `6.530px`.

## Motion-magnitude confound

Observed displacement is the pixel distance between first/last observed target centers. Selected-score test Spearman vs observed displacement: 0.639 (p=<1e-300); vs GT future endpoint displacement (post-hoc diagnostic only): 0.599 (p=<1e-300).
Slow/medium/fast strata use validation motion q33=`22.338px`, q67=`52.043px`.

| Test motion stratum | N | Mean motion px | Selected U vs ADE Spearman |
|---|---:|---:|---:|
| slow | 5813 | 12.678 | 0.160 (p=1.8e-34) |
| medium | 5908 | 35.494 | 0.121 (p=7.5e-21) |
| fast | 6610 | 102.065 | 0.401 (p=1.4e-253) |

## Horizon analysis

| t | Mean point error px | Mean disagreement px | Spearman(disagreement, point error) |
|---:|---:|---:|---:|
| 1 | 1.824 | 0.781 | 0.224 (p=7.8e-207) |
| 2 | 3.235 | 1.724 | 0.102 (p=1.2e-43) |
| 3 | 4.566 | 2.234 | 0.087 (p=2.5e-32) |
| 4 | 5.687 | 1.456 | 0.197 (p=5.1e-159) |
| 5 | 7.144 | 1.763 | -0.135 (p=1.1e-75) |
| 6 | 8.154 | 1.484 | 0.251 (p=5.1e-261) |
| 7 | 9.490 | 1.613 | 0.271 (p=3.8e-307) |
| 8 | 10.799 | 1.399 | 0.265 (p=1.2e-292) |
| 9 | 11.685 | 1.421 | 0.229 (p=7.2e-217) |
| 10 | 12.610 | 1.616 | 0.312 (p=<1e-300) |
| 11 | 13.832 | 2.173 | 0.249 (p=1.1e-256) |
| 12 | 14.895 | 2.377 | 0.335 (p=<1e-300) |
| 13 | 16.297 | 1.862 | 0.317 (p=<1e-300) |
| 14 | 17.755 | 3.417 | 0.275 (p=<1e-300) |
| 15 | 19.111 | 2.379 | 0.323 (p=<1e-300) |

## Final decision: Promising

Promising requires rho>=0.30, fixed-validation High-ADE AUROC>=0.65, high-bin mean ADE >=1.10x low-bin ADE, and non-increasing ADE risk as coverage declines. Stop requires rho<0.15, AUROC within 0.05 of 0.5, and no ADE risk reduction; otherwise weak/inconclusive.

This is a feasibility audit only; it does not establish that trajectory reliability should gate crossing-intention evidence.
