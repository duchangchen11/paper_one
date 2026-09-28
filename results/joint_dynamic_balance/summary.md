# Dynamic Gradient Balance (DGB-20) results

## Frozen protocol and validation-only review

DGB-20 used initial λ=100, target weighted shared-gradient ratio ρ=20, log-space EMA β=0.9, λ∈[10,300], one fixed warm-up epoch, and updates every 10 training batches. Training settings, architecture, splits, optimizer, scheduler, and checkpoint selection were held at the `ad05ef5` baseline. No reliability/uncertainty weighting, adapters, PCGrad, or GradNorm were used.

The three training runs completed with the test split withheld. The following validation metrics at each seed's selected checkpoint were reviewed before protocol freeze:

| Seed | Best epoch | Val AUC | Val Brier | Val F1 | Val BAcc | Val ADE (px) | Val FDE (px) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 13 | 0.8645 | 0.0856 | 0.9382 | 0.8016 | 12.98 | 21.73 |
| 123 | 12 | 0.8649 | 0.1077 | 0.9213 | 0.7904 | 13.40 | 21.97 |
| 2024 | 9 | 0.8858 | 0.0952 | 0.9213 | 0.7904 | 18.74 | 25.87 |

Frozen protocol SHA256: `f287026ee2927d6a4b14d7f54baeb95d0b80c5420c8cbc51e52a4d41b434cd9e  protocol_frozen.json`.

Protocol audit caveat: before freezing, the processed test archive was opened and each compressed array indexed to enumerate field names and shapes, which materialized/decompressed the arrays in memory. Array contents were not printed or summarized; no inference, metrics, training, selection, or tuning used test outcomes. This is a procedural deviation from strict no-access-before-freeze and is detailed in [protocol_frozen.errata.md](protocol_frozen.errata.md); the original frozen protocol/hash are retained.

## Test-set summary (mean ± sample SD across three seeds)

| Method | AUC | Brier | F1 | BAcc | ADE (px) | FDE (px) | Mean weighted gradient ratio | Mean λ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed λ=100 | 0.7914 ± 0.0135 | 0.0994 ± 0.0004 | 0.9368 ± 0.0072 | 0.6239 ± 0.0249 | 15.8904 ± 1.4719 | 26.9228 ± 2.7406 | 17.2695 ± 2.7607 | 100.0000 ± 0.0000 |
| DGB-20 | 0.7903 ± 0.0117 | 0.1039 ± 0.0028 | 0.9337 ± 0.0037 | 0.6154 ± 0.0107 | 16.9239 ± 2.6330 | 28.1264 ± 2.4562 | 27.2811 ± 37.7455 | 59.1881 ± 26.1344 |

## Per-seed test results and paired deltas (DGB − Fixed100)

| Seed | Method | AUC | Brier | F1 | BAcc | ADE (px) | FDE (px) | Best epoch |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | Fixed100 | 0.7795 | 0.0998 | 0.9412 | 0.6136 | 14.76 | 25.96 | 14 |
| 42 | DGB-20 | 0.7972 | 0.1068 | 0.9308 | 0.6088 | 14.69 | 25.60 | 13 |
| 123 | Fixed100 | 0.8061 | 0.0993 | 0.9285 | 0.6523 | 17.55 | 30.01 | 12 |
| 123 | DGB-20 | 0.7969 | 0.1013 | 0.9379 | 0.6278 | 16.25 | 28.27 | 12 |
| 2024 | Fixed100 | 0.7886 | 0.0990 | 0.9408 | 0.6059 | 15.36 | 24.79 | 11 |
| 2024 | DGB-20 | 0.7768 | 0.1036 | 0.9324 | 0.6096 | 19.83 | 30.51 | 9 |
| Δ mean | DGB−Fixed100 | -0.0011 AUC | +0.0045 Brier | — | — | +1.03 ADE | +1.20 FDE | — |

Paired differences are descriptive only; three seeds do not support a seed-level significance claim.

| Seed | ΔAUC | ΔBrier | ΔADE (px) | ΔFDE (px) |
|---:|---:|---:|---:|---:|
| 42 | +0.0176 | +0.0071 | -0.07 | -0.36 |
| 123 | -0.0092 | +0.0020 | -1.31 | -1.74 |
| 2024 | -0.0118 | +0.0046 | +4.47 | +5.71 |

## Controller and gradient diagnostics

- λ over all training batches: mean **59.188**, std **26.134**, min **23.725**, max **131.838**.
- Applied controller updates: **168**; λ_min hits **0/168 (0.00%)**; λ_max hits **0/168 (0.00%)**.
- DGB weighted gradient ratio at online measurement batches: mean **27.281**, sample SD **37.746**, median **17.710**, range **[0.179, 299.354]**.
- Fixed100 diagnostic ratio: mean **17.269**, sample SD **16.749** across its fixed post-epoch audit batches.
- Sampling caveat: DGB ratios are measured online every 10th training batch, while the pre-existing Fixed100 values are measured once per epoch on a fixed balanced diagnostic subset. Their spread is descriptive, not a strictly paired estimator.
- Training itself was numerically stable (finite losses/gradients, 56/56 valid updates per seed, and no λ-bound hits), but the realized weighted ratio was not tightly controlled around 20: mean 27.28, SD 37.75, median 17.71. Thus DGB did not demonstrate more stable ratio control in this run.

## Interpretation

- DGB-20 does not meet the predeclared practical comparison against Fixed100 under the simple descriptive checks (ADE within 0.5 px, FDE within 5%, AUC within 0.01, Brier within 0.01): ADE 16.92 vs 15.89 px; FDE 28.13 vs 26.92 px; AUC 0.7903 vs 0.7914; Brier 0.1039 vs 0.0994.
- Against the trajectory-only reference (ADE/FDE 11.06/19.59 px), DGB-20 does not restore that reference. Remaining mean gaps: ADE **+5.86 px**, FDE **+8.54 px**.
- λ changed during training; boundary saturation was not observed.
- Recommendation: do not proceed to reliability-aware weighting yet; first have the next-stage research decision assess why DGB-20 did not outperform the fixed λ=100 control.
- The test set is descriptive and was not used for DGB hyperparameter or checkpoint selection. No significance claim is made from three seeds.

## Artifacts

- Validation-only metrics and checkpoints: `results/joint_dynamic_balance/dgb20/seed{42,123,2024}/`.
- Online gradient/lambda traces: each seed's `gradient_history.json`.
- Frozen protocol: `results/joint_dynamic_balance/protocol_frozen.json` and `.sha256`.
- Per-sample official-test predictions: each seed's `test_predictions.npz` (scene/video ID, target ID/frame, intention label/probability, future prediction/ground truth, image size, per-sample ADE/FDE).
