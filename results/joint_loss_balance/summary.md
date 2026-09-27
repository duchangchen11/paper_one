# Joint trajectory/intention loss-balance sweep

## Protocol

Executed from baseline revision `6bbace5`, with fixed seeds `42/123/2024` and trajectory weights `λ ∈ {1, 5, 10, 50, 100}` (15 full runs total). The only optimized objective change was the multiplier on normalized-coordinate trajectory SmoothL1. Dataset and JAAD splits, architecture, batch size 512, 15 epochs, AdamW (`lr=1e-3`, `weight_decay=1e-4`), scheduler, prior weight 0.5, ambiguity weight 0.2, gate mode, checkpoint-selection rule, and each seed were held constant. No adapter, PCGrad, GradNorm, uncertainty or reliability input was added.

For each epoch, `gradient_history.json` records component losses and shared-parameter gradient norms at one fixed balanced 512-sample training subset plus one fixed ambiguous subset. The gradient diagnostic uses `autograd.grad`, does not update parameters, and restores Python/NumPy/CPU/CUDA RNG states before ordinary training continues. Gradient ratio is `||g_intent|| / (λ ||g_traj||)`; its table value below is the average of the 15 per-epoch ratios, first averaged within each run, then summarized across seeds. Thus it is not the same estimand as the earlier single saved-checkpoint estimate of ≈454×.

Each run writes `metrics.json` and `gradient_history.json` under its condition/seed directory. Best checkpoints use the original validation composite score; test values below are descriptive. The cross-λ selection below uses mean validation ADE, not test ADE.

## Aggregate results

Values are test-set mean ± sample standard deviation across three seeds. Lower Brier/ADE/FDE is better; higher AUC/F1/balanced accuracy is better.

| λtraj | ROC-AUC | Brier | F1 | Balanced accuracy | ADE (px) | FDE (px) | Mean epoch gradient ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (baseline) | 0.7727 ± 0.0116 | 0.1032 ± 0.0029 | 0.9377 ± 0.0010 | 0.6017 ± 0.0245 | 28.69 ± 5.04 | 47.01 ± 7.93 | 1,175.4 ± 188.0 |
| 5 | 0.8034 ± 0.0089 | 0.1000 ± 0.0059 | 0.9367 ± 0.0084 | 0.5914 ± 0.0599 | 25.43 ± 2.32 | 42.92 ± 5.77 | 312.6 ± 114.4 |
| 10 | 0.7748 ± 0.0048 | 0.1065 ± 0.0031 | 0.9325 ± 0.0053 | 0.5956 ± 0.0425 | 23.81 ± 2.49 | 39.85 ± 3.75 | 121.3 ± 31.3 |
| 50 | 0.7688 ± 0.0159 | 0.1018 ± 0.0012 | 0.9385 ± 0.0029 | 0.6070 ± 0.0104 | 16.45 ± 1.77 | 28.54 ± 2.75 | 42.2 ± 19.1 |
| 100 | 0.7914 ± 0.0135 | 0.0994 ± 0.0004 | 0.9368 ± 0.0072 | 0.6239 ± 0.0249 | **15.89 ± 1.47** | **26.92 ± 2.74** | **17.3 ± 2.8** |

### Per-seed held-out results

| λtraj | Seed | Test AUC | Brier | F1 | Balanced accuracy | ADE (px) | FDE (px) | Best epoch | Validation ADE (px) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 42 | 0.7701 | 0.1029 | 0.9386 | 0.6116 | 34.44 | 56.17 | 7 | 30.13 |
| 1 | 123 | 0.7625 | 0.1062 | 0.9367 | 0.5738 | 25.03 | 42.66 | 14 | 21.73 |
| 1 | 2024 | 0.7853 | 0.1004 | 0.9380 | 0.6196 | 26.61 | 42.21 | 12 | 25.81 |
| 5 | 42 | 0.7937 | 0.1068 | 0.9270 | 0.5227 | 28.03 | 48.59 | 8 | 25.24 |
| 5 | 123 | 0.8055 | 0.0959 | 0.9403 | 0.6330 | 24.71 | 43.10 | 15 | 20.10 |
| 5 | 2024 | 0.8111 | 0.0974 | 0.9427 | 0.6184 | 23.56 | 37.05 | 12 | 23.26 |
| 10 | 42 | 0.7715 | 0.1048 | 0.9343 | 0.6096 | 25.87 | 44.02 | 14 | 22.40 |
| 10 | 123 | 0.7727 | 0.1101 | 0.9265 | 0.5478 | 24.52 | 38.77 | 14 | 20.40 |
| 10 | 2024 | 0.7803 | 0.1046 | 0.9366 | 0.6294 | 21.04 | 36.76 | 11 | 21.95 |
| 50 | 42 | 0.7869 | 0.1025 | 0.9417 | 0.6130 | 18.30 | 31.64 | 10 | 14.06 |
| 50 | 123 | 0.7569 | 0.1004 | 0.9360 | 0.5950 | 14.76 | 26.37 | 15 | 13.61 |
| 50 | 2024 | 0.7626 | 0.1025 | 0.9378 | 0.6130 | 16.29 | 27.62 | 11 | 15.18 |
| 100 | 42 | 0.7795 | 0.0998 | 0.9412 | 0.6136 | 14.76 | 25.96 | 14 | 14.29 |
| 100 | 123 | 0.8061 | 0.0993 | 0.9285 | 0.6523 | 17.55 | 30.01 | 12 | 12.97 |
| 100 | 2024 | 0.7886 | 0.0990 | 0.9408 | 0.6059 | 15.36 | 24.79 | 11 | 13.66 |

## Baseline reproduction

The λ=1, seed-123 test result reproduced the existing baseline exactly to reported precision: AUC **0.7625**, Brier **0.1062**, F1 **0.9367**, balanced accuracy **0.5738**, ADE/FDE **25.0303/42.6616 px**. The three-seed baseline mean is worse (ADE/FDE **28.69/47.01 px**) because seed42 reached **34.44/56.17 px**; this is substantial seed sensitivity and argues against treating the single-seed ≈25 px baseline as the whole distribution.

## Gradient-ratio response

The mean of epoch-wise ratios declines overall as λ increases:

| λtraj | Mean of per-epoch ratios, averaged across seeds |
|---:|---:|
| 1 | 1,175.4 ± 188.0 |
| 5 | 312.6 ± 114.4 |
| 10 | 121.3 ± 31.3 |
| 50 | 42.2 ± 19.1 |
| 100 | 17.3 ± 2.8 |

The λ=1, seed123 ratio at its selected epoch 14 is **403×**, reasonably close to the earlier one-checkpoint estimate ≈454×; the earlier value used a different fixed audit subset. Ratios are not strictly monotonic for every individual epoch or seed because model weights evolve, but the across-epoch/seed trend is clear. Full per-epoch records are in each condition's `gradient_history.json`.

## Best λ and interpretation

Using mean validation ADE at each run's existing selected checkpoint, λ=100 is the best tested setting (**13.64 ± 0.66 px**; λ=50 is next at **14.28 ± 0.81 px**). Its descriptive test mean is **15.89/26.92 px ADE/FDE**, versus baseline **28.69/47.01 px**: reductions of approximately **44.6% ADE** and **42.7% FDE**. Test AUC rises from **0.7727** to **0.7914** on average; F1 is essentially unchanged (**0.9377 → 0.9368**), and mean Brier improves (**0.1032 → 0.0994**). Thus the tested λ increase did not show an aggregate intention-performance penalty.

This controlled intervention **strongly supports trajectory loss underweighting as a major cause** of the joint failure: λ=50/100 lower the measured gradient ratio and substantially improve ADE/FDE without degrading mean AUC/F1. It does not prove loss imbalance is the sole cause. λ=100 does **not fully restore** trajectory-only accuracy: its ADE/FDE remain about 4.83/7.33 px above the independent trajectory Transformer’s 11.06/19.59 px. There are three seeds per setting, and the λ sweep is exploratory; the chosen λ was based on validation ADE, not test.

## Recommended next step

Treat λ=100 as the best tested candidate, not yet as a universal optimum. Repeat or extend a narrow predeclared λ sweep (e.g. 75/100/150) with additional seeds, keep the same selection rule, and inspect validation AUC/Brier alongside ADE/FDE. Do not add reliability weighting or gradient surgery until this fixed-weight effect is replicated; preserve the full per-epoch gradient/loss logging.

## Verification

- 15/15 runs completed: five λ values × three seeds; each run contains 15 epoch loss/gradient records.
- `python -m pytest tests -q`: **87 passed, 0 failed**. In this environment, pytest was run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` because the system ROS pytest plugin otherwise fails to import missing `lark`; 27 existing Transformer warnings remain.
- No model architecture, data, split, optimizer, batch size, learning rate, epoch count, or seed was changed; no uncertainty/reliability features or gradient surgery were used.
