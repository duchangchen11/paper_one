# Input-Matched Scratch Intention Transformer vs P1

M0 protocol `Intention-Scratch-Matched-v1` SHA256: `af9f641db1cd35ed2d0697c248407cf4eb9c5a346694485f0ad05d3644f9fb8b`. P1 is reused from its separately frozen prior run. P1 test predictions have therefore been previously generated; the M0 test split was not opened until this protocol was frozen. No M0 test result was used for checkpoint selection or tuning.

All summary values are mean ± sample SD across the three matched seeds. Higher AUC/F1/BAcc/accuracy is better; lower Brier is better. Brier/F1/BAcc/accuracy use each method's validation-fitted temperature and threshold. AUC is invariant to temperature scaling.

## Input and architecture match

The pre-training equivalence audit passed. Both methods receive `concat(target_obs, target_abs_obs)` with shape `[B,15,8]`, use the same Linear(8→128), learned `[1,15,128]` positional embedding, 3-layer/4-head/128-d Transformer encoder, last-token context, and identical LayerNorm→Linear(128,128)→GELU→Dropout(0.1)→Linear(128,1) intention head. No scene input is used.

| Method | Input | Initialization/training | AUC | Brier | F1 | BAcc | Accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| Historical observed-only | 15×4 | No trajectory pretraining; GRU baseline | 0.6688 ± 0.0093 | 0.2154 ± 0.0210 | 0.7148 ± 0.0097 | 0.6358 ± 0.0099 | — |
| M0 Scratch Transformer | 15×8 | Random init; encoder and head trained | 0.6732 ± 0.0079 | 0.2016 ± 0.0815 | 0.8191 ± 0.0858 | 0.6071 ± 0.0293 | 0.7189 ± 0.1021 |
| P1 Pretrained Frozen | 15×8 | Trajectory-pretrained encoder frozen; head trained | 0.6878 ± 0.0262 | 0.2131 ± 0.0024 | 0.7633 ± 0.0211 | 0.6383 ± 0.0339 | 0.6480 ± 0.0210 |

Historical observed-only is context only, not the primary control: it uses 15×4 and a different GRU architecture.

## Matched-seed held-out comparison (P1 − M0)

| Seed | M0 AUC | P1 AUC | ΔAUC | M0 Brier | P1 Brier | ΔBrier | M0 F1 | P1 F1 | ΔF1 | M0 BAcc | P1 BAcc | ΔBAcc | M0 Acc | P1 Acc | ΔAcc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6662 | 0.6576 | -0.0086 | 0.1076 | 0.2109 | +0.1033 | 0.8673 | 0.7742 | -0.0931 | 0.5921 | 0.6014 | +0.0093 | 0.7760 | 0.6563 | -0.1197 |
| 123 | 0.6718 | 0.7036 | +0.0318 | 0.2507 | 0.2128 | -0.0379 | 0.8700 | 0.7767 | -0.0933 | 0.5883 | 0.6450 | +0.0567 | 0.7796 | 0.6636 | -0.1160 |
| 2024 | 0.6817 | 0.7023 | +0.0205 | 0.2465 | 0.2156 | -0.0309 | 0.7201 | 0.7390 | +0.0189 | 0.6409 | 0.6683 | +0.0274 | 0.6011 | 0.6242 | +0.0231 |
| Mean Δ | — | — | +0.0146 | — | — | +0.0115 | — | — | -0.0558 | — | — | +0.0312 | — | — | -0.0709 |

## Per-seed scene-cluster bootstrap (2,000 replicates each; seeds not pooled)

| Seed | ΔAUC point | ΔAUC 95% CI | ΔBrier point | ΔBrier 95% CI | Video/scene clusters |
|---:|---:|---:|---:|---:|---:|
| 42 | -0.0086 | [-0.0785, +0.0474] | +0.1033 | [+0.0611, +0.1449] | 97 |
| 123 | +0.0318 | [-0.0204, +0.0854] | -0.0379 | [-0.0649, -0.0122] | 97 |
| 2024 | +0.0205 | [-0.0648, +0.1007] | -0.0309 | [-0.0569, -0.0051] | 97 |

## Validation-to-test AUC gaps

AUC checkpoint selection used validation only; the held-out difference is included as a stability diagnostic.

| Seed | M0 val AUC | M0 test AUC | M0 test−val | P1 val AUC | P1 test AUC | P1 test−val |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.7404 | 0.6662 | -0.0742 | 0.7114 | 0.6576 | -0.0538 |
| 123 | 0.7554 | 0.6718 | -0.0836 | 0.6620 | 0.7036 | +0.0416 |
| 2024 | 0.7712 | 0.6817 | -0.0895 | 0.6415 | 0.7023 | +0.0608 |

## Validation representation diagnostic

Representation moments use the same first 1000 validation samples (requested 1,000); logistic regression is fit on each representation's train features and evaluated on the full validation split. This diagnostic did not affect checkpoint selection.

| Representation | Linear val AUC | Linear val Brier |
|---|---:|---:|
| P1 pretrained frozen | 0.4488 ± 0.0117 | 0.2835 ± 0.0061 |
| M0 scratch trained | 0.4343 ± 0.0181 | 0.2765 ± 0.0034 |

Feature mean/std, mean L2 norm, per-dimension variance, and each seed's linear diagnostic are in `representation_analysis.json`.

## Conclusions and requested questions

1. **Input match:** yes; both consume the exact P1 `[15,8]` history concatenation with the same stored arrays and no added normalization.
2. **Architecture match:** yes; all P1 target encoder and intention head tensor shapes/names mapped exactly in `architecture_equivalence.json`; M0 has no scene encoder or trajectory decoder.
3. **M0 scores:** AUC 0.6732 ± 0.0079, Brier 0.2016 ± 0.0815, F1 0.8191 ± 0.0858, BAcc 0.6071 ± 0.0293 (per-seed values above).
4. **P1 scores:** AUC 0.6878 ± 0.0262, Brier 0.2131 ± 0.0024, F1 0.7633 ± 0.0211, BAcc 0.6383 ± 0.0339.
5. **Matched ΔAUC (P1−M0):** 0.0146 ± 0.0208; seed-specific deltas are 42: -0.0086, 123: +0.0318, 2024: +0.0205.
6. **Direction:** P1 has higher AUC in 2/3 seeds; this is not sufficient by itself to claim transfer.
7. **Bootstrap:** see per-seed paired scene-cluster intervals above. 0/3 AUC interval lower bounds are above zero; the rule (positive mean, at least 2/3 positive seeds, and at least 2/3 positive lower bounds) is not met.
8. **Transfer judgment:** the results do not establish stable positive trajectory-pretraining transfer under the predeclared descriptive rule.
9. **Does the comparison rule out ‘just 8D + Transformer’?** This is the correct input/architecture control, so P1−M0 measures the practical difference between a frozen trajectory-pretrained target representation and a scratch-trained target Transformer. Because P1 freezes its encoder while M0 trains its encoder, it does not isolate initialization alone from the frozen-vs-trainable strategy.
10. **Linear separability:** P1 train-fitted logistic regression validation AUC/Brier is 0.4488 ± 0.0117/0.2835 ± 0.0061; M0 is 0.4343 ± 0.0181/0.2765 ± 0.0034. These values are descriptive only and do not replace the nonlinear intention-head comparison.
11. **Next phase:** do not proceed automatically. If the bootstrap rule is met, a narrowly scoped task-specific adapter study has support; otherwise first treat transfer as inconclusive and retain P1/M0 as controls. Do not begin partial-unfreezing in this task.

## Reproducibility and caveats

- Verification: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q` — 107 passed, 0 failed (37 existing PyTorch nested-tensor warnings).
- Frozen protocol SHA256: `af9f641db1cd35ed2d0697c248407cf4eb9c5a346694485f0ad05d3644f9fb8b`; M0 test archive was read once after freeze for all three seeds.
- P1's stored predictions and metrics are reused from the earlier frozen experiment and were not recomputed. Thus P1's test output was previously available; the paired bootstrap is descriptive on this existing holdout, not a fresh blinded replication.
- Cluster bootstrap resamples `scene_id` videos separately within each seed, 2,000 draws per seed; the three seeds are never pooled as if they were samples.
- M0 selected checkpoint files are saved locally under `checkpoints/intention_scratch_matched/` (each is 2.4 MB and remains excluded by the repository's existing `/checkpoints/` ignore rule); initialization reports, validation histories, per-sample test predictions, and protocol hashes are committed under `results/intention_scratch_matched/`.
- Reporting note: the frozen analysis script completed and saved all bootstrap statistics, then encountered a validation-statistics JSON key mismatch while rendering Markdown. To preserve the frozen source/protocol, the report was rendered with a read-only in-memory key alias and the already-saved bootstrap outputs; no training, test inference, or source file was changed. The script should be corrected in a separately authorized follow-up.
