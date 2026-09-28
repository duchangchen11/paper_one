# M1 Pretrained-Clone Trainable Intention Experiment

Frozen protocol SHA256: `24840578d8f6afcdc97a2bd7f2bff9e20df504bdd1213249ccb716d2bf85d3af`.

M1 isolates encoder initialization: its target-only intention Transformer is trainable from trajectory-pretrained cloned weights; a separate, frozen trajectory branch is retained. All official metrics below use the held-out test set once after freeze. Seed values are not pooled as statistical replicates; cluster bootstrap is performed independently within each seed.

## Test metrics across seeds (mean ± sample SD)

| Method | AUC | Brier | F1 | BAcc | ADE (px) | FDE (px) |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.6732 ± 0.0079 | 0.2016 ± 0.0815 | 0.8191 ± 0.0858 | 0.6071 ± 0.0293 | 11.0107 ± 0.1785 | 19.4592 ± 0.4057 |
| P1 | 0.6878 ± 0.0262 | 0.2131 ± 0.0024 | 0.7633 ± 0.0211 | 0.6383 ± 0.0339 | 11.0107 ± 0.1785 | 19.4592 ± 0.4057 |
| M1 | 0.6604 ± 0.0154 | 0.2023 ± 0.0813 | 0.8142 ± 0.0747 | 0.6032 ± 0.0202 | 11.0107 ± 0.1785 | 19.4592 ± 0.4057 |

## Matched-seed intention comparison

| Seed | M0 AUC | P1 AUC | M1 AUC | M1−M0 ΔAUC | M1−P1 ΔAUC |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.6662 | 0.6576 | 0.6672 | +0.0010 | +0.0096 |
| 123 | 0.6718 | 0.7036 | 0.6713 | -0.0005 | -0.0324 |
| 2024 | 0.6817 | 0.7023 | 0.6428 | -0.0389 | -0.0595 |

## M1−M0 paired cluster bootstrap (per seed)

| Seed | ΔAUC | 95% CI | ΔBrier | 95% CI |
|---:|---:|---:|---:|---:|
| 42 | +0.0010 | [-0.0129, +0.0181] | +0.1420 | [+0.0907, +0.1846] |
| 123 | -0.0005 | [-0.0115, +0.0132] | -0.0018 | [-0.0031, -0.0005] |
| 2024 | -0.0389 | [-0.0944, +0.0077] | -0.1381 | [-0.1769, -0.0932] |

## Trajectory preservation

| Seed | T0 ADE | M1 ADE | ΔADE | T0 FDE | M1 FDE | ΔFDE | max |Δprediction| |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 11.1605 | 11.1605 | +0.000000 | 19.7809 | 19.7809 | +0.000000 | 0 |
| 123 | 11.0585 | 11.0585 | +0.000000 | 19.5931 | 19.5931 | +0.000000 | 0 |
| 2024 | 10.8133 | 10.8133 | +0.000000 | 19.0035 | 19.0035 | +0.000000 | 0 |

## Initialization and isolation checks

- Exact pretrained clone: pass for all 3 seeds; expected/copy counts and zero maximum absolute tensor difference are in `pretrained_clone_report.json`.
- Independent parameters: pass for all 3 seeds; optimizer step changed intention parameters only, with no shared object/storage and no trajectory parameter update (`parameter_independence_test.json`).
- Trajectory equivalence before training: pass for all seeds (`trajectory_equivalence.json`).
- Training: three seeds completed 20 epochs; frozen trajectory SHA256 was unchanged every epoch; intention optimizer excluded trajectory parameters.

## Pre-registered transfer rule

- Mean ΔAUC(M1−M0) > 0: **False** (-0.0128).
- At least 2/3 positive seed ΔAUC: **False** (1/3).
- At least 2/3 seed-specific 95% bootstrap CI lower bounds > 0: **False** (0/3).
- M1/M0 AUC seed-SD ratio ≤ 1.5: **False** (1.951; M0 SD=0.0079, M1 SD=0.0154).
- Trajectory predictions preserved exactly: **True**.
- Overall: **does not meet the pre-registered stable-transfer rule; bootstrap intervals crossing zero are weak/inconclusive, not significant transfer**.

## Encoder drift and representation shift

Selected-epoch, per-layer parameter drift is recorded in `encoder_drift.json`. The validation-only 1,000-sample feature comparison (means, standard deviations, norms, per-dimension variance, per-sample cosine, and linear CKA) is in `representation_shift.json`; it did not affect model selection.

## Research interpretation and next step

- M1 vs M0 isolates trainable pretrained initialization. The result **does not establish** reliable transfer under the registered rule.
- M1 vs P1 is the secondary task-adaptation comparison; consult the seedwise paired results and bootstrap in `cluster_bootstrap.json`.
- Adapter and partial-unfreeze work is **not automatically authorized by these results**. If transfer is inconclusive or negative, do not present trajectory-pretrained initialization as established value; diagnose task mismatch and preserve M0 as the reference before expanding architecture.
- A positive M1−M0 mean with confidence intervals crossing zero is described as weak/inconclusive, not statistically significant transfer.

## Reproducibility artifacts

The frozen protocol records source/config/checkpoint/data hashes and the already-existing M0/P1 result artifact hashes. M1 official test predictions are stored under each seed directory. No adapter, partial unfreeze, scene/social/reliability input, or trajectory loss was used.

## Execution and reporting notes

The held-out NPZ contains `scene_id`, `target_id`, and `obs_end_frame` but no standalone `video_id`. After the frozen evaluator stopped at field validation (before inference or test-metric computation), video IDs were attached by an exact one-to-one join to the hash-frozen P1 outputs; the original protocol remains unchanged and this metadata-only correction is documented in `protocol_frozen_addendum.json`. A report-shape error occurred after all bootstrap results had been written; this summary was rebuilt from those existing bootstrap and metrics files without reopening the test archive or recomputing the bootstrap.

## Direct answers to the registered questions

1. **Exact pretrained initialization:** yes. All expected target-encoder tensors were copied, with maximum absolute difference 0 for every seed.
2. **Parameter decoupling:** yes. The trajectory and intention branches have distinct Parameter objects and storage; the smoke optimizer changed only intention parameters.
3. **Trajectory preservation:** yes. M1's official predictions match P1 exactly for all seeds (`max_abs_future_prediction_difference = 0`); ADE/FDE are unchanged.
4. **M1 intention test results:** seed 42 AUC/Brier/F1/BAcc = 0.6672/0.2495/0.8272/0.5803; seed 123 = 0.6713/0.2489/0.7338/0.6182; seed 2024 = 0.6428/0.1084/0.8815/0.6112.
5. **Stable improvement over M0:** no. Mean ΔAUC is −0.0128, only one seed is positive, none of the three 95% cluster-bootstrap intervals has a lower bound above zero, and M1's AUC seed SD is 1.95× M0's.
6. **Stable improvement over P1:** no. Mean matched ΔAUC is −0.0274; the seed deltas are +0.0096, −0.0324, and −0.0595, and all three bootstrap intervals cross zero.
7. **Bootstrap evidence:** it does not support positive transfer. M1−M0 intervals cross zero in all seeds; these data are inconclusive and do not establish a statistically reliable positive effect.
8. **Practical value of pretrained initialization:** not demonstrated under this matched trainable comparison. In particular, seed 2024 has a substantial AUC decrease; M1 is not a reliable replacement for M0.
9. **Encoder drift:** the position embedding has the largest relative drift for each seed (1.36–4.04×; its initial norm is small, so this ratio should not be interpreted alone). Among Transformer blocks, layer 0 changes most (relative L2 0.21, 0.55, 0.21 for seeds 42/123/2024); selected-checkpoint per-layer values are in `encoder_drift.json`.
10. **Task-specific adapter:** these results do not yet justify an adapter as the next experiment. First resolve M1's seed instability and substantial representation-scale shift under the unchanged M0 training recipe.
11. **Partial unfreezing:** not recommended from this evidence; the independent frozen trajectory branch already preserves trajectory performance, while unfreezing would add a new variable without evidence of positive transfer.
12. **Research claim to stop:** do not claim that trajectory-pretrained target representations provide established crossing-intention transfer. This result does not invalidate trajectory prediction or the broader intention project; it rejects the stronger transfer claim under this protocol.

The validation representation diagnostic shows strong feature change after adaptation: mean sample cosine is 0.610–0.655 and linear CKA is 0.177–0.342 across seeds. M1 context norms/per-dimension variance also increase substantially, especially for seed 123; these are diagnostic observations, not evidence of better prediction.
