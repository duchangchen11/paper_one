# Clean trajectory-supervision attribution

- Frozen protocol SHA256: `1b43c5063cced64ea4a2606f1d02c0ef98fc7531dd026b24bf0b1403faf3c9e3`; base commit `6e98ef1dd68fa7970dc4cd2c53feef2ab80ddd11`.
- Official test access: one-time; 18331 samples; test archive SHA256 `23658152706d460493fa6146f6b401648a509cc8708ab5eb6c0892e300de1935`.
- Clean J0/J100 share each seed's exact initial tensor state and main WeightedRandomSampler sequence. Scheduler monitors raw validation AUC; checkpoint selection uses raw validation AUC with Brier-only tie-break (≤1e-4).
- All historical metrics are secondary context and were not pooled with clean runs.

## Test metrics across methods

| Method | AUC | Brier | F1 | BAcc | ADE (px) | FDE (px) |
|---|---:|---:|---:|---:|---:|---:|
| Historical J0 | 0.7544 ± 0.0053 | 0.0975 ± 0.0042 | 0.9335 ± 0.0033 | 0.5820 ± 0.0368 | 527.1868 ± 102.3785 | 487.3561 ± 197.7073 |
| Historical J100 | 0.7914 ± 0.0135 | 0.0994 ± 0.0004 | 0.9368 ± 0.0072 | 0.6239 ± 0.0249 | 15.8904 ± 1.4719 | 26.9228 ± 2.7406 |
| Clean J0 | 0.7644 ± 0.0210 | 0.0953 ± 0.0032 | 0.9315 ± 0.0039 | 0.5896 ± 0.0211 | 550.4221 ± 53.6462 | 502.2770 ± 190.6811 |
| Clean J100 | 0.7740 ± 0.0160 | 0.1098 ± 0.0065 | 0.9240 ± 0.0181 | 0.6160 ± 0.0619 | 23.4680 ± 12.5568 | 39.9542 ± 22.3370 |

## Matched clean AUC attribution

| Seed | J0-clean AUC | J100-clean AUC | ΔAUC |
|---:|---:|---:|---:|
| 42 | 0.7698 | 0.7730 | +0.0032 |
| 123 | 0.7413 | 0.7586 | +0.0172 |
| 2024 | 0.7822 | 0.7906 | +0.0084 |

## Paired scene_id cluster bootstrap (2,000 draws per seed; no pooling)

| Seed | ΔAUC point | Bootstrap mean | 95% CI | ΔBrier point | 95% CI |
|---:|---:|---:|---|---:|---|
| 42 | +0.0032 | +0.0045 | [-0.0620, +0.0766] | +0.0180 | [+0.0003, +0.0391] |
| 123 | +0.0172 | +0.0174 | [-0.0084, +0.0482] | +0.0154 | [+0.0015, +0.0290] |
| 2024 | +0.0084 | +0.0093 | [-0.0513, +0.0749] | +0.0098 | [-0.0032, +0.0258] |

## Trajectory ADE comparison

| Seed | J0-clean ADE | J100-clean ADE | T0 standalone ADE |
|---:|---:|---:|---:|
| 42 | 514.02 | 15.73 | 11.16 |
| 123 | 612.03 | 37.96 | 11.06 |
| 2024 | 525.22 | 16.71 | 10.81 |

## Historical versus clean selection

Historical mean ΔAUC was +0.0370; clean mean ΔAUC is +0.0096; clean-minus-historical change is -0.0274.
Clean J0 mean AUC shifts from historical by +0.0101; clean J100 shifts by -0.0174.
Historical and clean experiments are reported separately; no historical samples or seeds were pooled into the primary clean inference.

## Validation AUC–ADE Pareto diagnostic

AUC–ADE trade-off across J100-clean validation epochs observed: **True**. Per-epoch points and Pareto-efficient epochs are in `pareto_history.json`; this diagnostic did not affect selection.

## Representation diagnostic (selected checkpoint; fixed 1,000 validation samples)

| Feature | Arm | Mean L2 norm | Mean per-dimension variance | Cosine to initialization | Linear CKA to initialization |
|---|---|---:|---:|---:|---:|
| target_encoder_last | J0_clean | 17.6521 | 0.670433 | 0.1704 | 0.6633 |
| target_encoder_last | J100_clean | 16.7396 | 0.777594 | 0.2373 | 0.6307 |
| fused_representation | J0_clean | 10.4039 | 0.200549 | 0.2624 | 0.4854 |
| fused_representation | J100_clean | 9.8501 | 0.281904 | 0.1741 | 0.4597 |

These endpoint representation moments and similarities are descriptive only; they do not establish a causal mechanism.

## Selected-checkpoint gradient diagnostic

| Arm | Intent shared-grad norm | Raw trajectory shared-grad norm | Weighted trajectory shared-grad norm | Cosine(intent, trajectory) |
|---|---:|---:|---:|---:|
| J0_clean | 8.088 | 0.3838 | 0 | +0.0100 |
| J100_clean | 15.56 | 0.001362 | 0.1362 | -0.3188 |

Selected-checkpoint gradients are endpoint diagnostics only and do not represent gradient behavior over training.

## Direct answers

1. Initialization: **exactly matched** per seed; max absolute initial parameter difference is 0. The evidence is recorded in `initialization_match.json`.
2. Configuration: **only `traj_weight` differs**; audit passed. The model architecture file was unchanged.
3. Scheduler and checkpoint selection: **neither uses ADE/FDE**. Scheduler uses raw validation AUC; selector uses raw validation AUC then raw Brier within 1e-4.
4. J0-clean AUC by seed: 42=0.7698, 123=0.7413, 2024=0.7822.
5. J100-clean AUC by seed: 42=0.7730, 123=0.7586, 2024=0.7906.
6. Mean clean ΔAUC: **+0.0096**; positive direction 3/3 seeds; bootstrap lower bound > 0 for 0/3 seeds.
7. Bootstrap: see the per-seed AUC and Brier intervals above; three seeds are independent and were not pooled.
8. Historical +0.0370 versus clean: change -0.0274; historical Δ=+0.0370, clean Δ=+0.0096.
9. J100-clean test trajectory ADE/FDE: seed 42: 15.73/27.55 px; seed 123: 37.96/65.74 px; seed 2024: 16.71/26.57 px.
10. AUC–ADE Pareto pattern: a validation trade-off exists; inspect `pareto_history.json` for epochs.
11. Representation: fixed-validation feature norm/variance and initialization CKA/cosine are tabulated above; see `representation_diagnostic.json` for per-seed/per-dimension values.
12. Gradient: see endpoint table and `gradient_diagnostic.json`; Clean J100 AUC sample SD=0.0160; historical J100 SD=0.0135; ratio=1.19×; preregistered stability cutoff=0.0270 AUC SD; pass=True.
13. Trajectory-supervision support classification: **weak positive trend only**. The mean clean ΔAUC is +0.0096 and all three seed deltas are positive, but all three paired 95% AUC intervals include zero; this is not reliable or practically meaningful evidence that trajectory supervision improves intention.
14. Next step: do not claim trajectory supervision improves intention; proceed to the planned joint-architecture component attribution rather than automatically adding modules.
15. No reliability, DGB, PCGrad, GradNorm, adapter, ablation, or lambda sweep was run in this phase.

## Artifacts

- `cluster_bootstrap.json`: per-seed paired scene_id bootstrap.
- `historical_vs_clean.json`: historical and clean per-seed metrics/deltas kept separate.
- `pareto_history.json`, `representation_diagnostic.json`, `gradient_diagnostic.json`: validation-only/descriptive diagnostics.
- Protocol: `protocol_frozen.json` (SHA256 `1b43c5063cced64ea4a2606f1d02c0ef98fc7531dd026b24bf0b1403faf3c9e3`).
- Post-freeze report-label correction is documented in `post_freeze_report_amendment.json`; no test archive was reopened and no numeric result changed.
