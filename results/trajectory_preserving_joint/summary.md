# Trajectory-Preserving Frozen Backbone Baseline

Protocol: `J-Preserve-Frozen-v1`; SHA256 `e0494823ef0adc5ee40ba1f78dd3596d17b3d76f8fe89bdb5b6fa1f65c45c71f`. Validation selection and calibration were frozen before the official test archive was opened. P1/P2 test metrics below are descriptive only and were not used for tuning.

Three-seed values are mean ± sample SD. Intent metrics: higher AUC/F1/BAcc is better; lower Brier is better. Trajectory errors are pixels; lower is better. N/A means the model does not produce that task output.

## Table 1. Aggregate comparison

| Method | Intent AUC | Brier | F1 | BAcc | ADE (px) | FDE (px) |
|---|---:|---:|---:|---:|---:|---:|
| Observed-only intention | 0.6688 ± 0.0093 | 0.2154 ± 0.0210 | 0.7148 ± 0.0097 | 0.6358 ± 0.0099 | N/A | N/A |
| Trajectory-only (T0) | N/A | N/A | N/A | N/A | 11.01 ± 0.18 | 19.46 ± 0.41 |
| Joint fixed λ=100 | 0.7914 ± 0.0135 | 0.0994 ± 0.0004 | 0.9368 ± 0.0072 | 0.6239 ± 0.0249 | 15.89 ± 1.47 | 26.92 ± 2.74 |
| P1_target_only | 0.6878 ± 0.0262 | 0.2131 ± 0.0024 | 0.7633 ± 0.0211 | 0.6383 ± 0.0339 | 11.01 ± 0.18 | 19.46 ± 0.41 |
| P2_target_scene | 0.6908 ± 0.1092 | 0.1268 ± 0.0216 | 0.8847 ± 0.0266 | 0.6005 ± 0.1144 | 11.01 ± 0.18 | 19.46 ± 0.41 |

## Table 2. Matched-seed trajectory preservation

| Seed | T0 ADE | P1 ADE | ΔADE | P2 ADE | ΔADE |
|---:|---:|---:|---:|---:|---:|
| 42 | 11.160 | 11.160 | -0.000001 | 11.160 | -0.000001 |
| 123 | 11.058 | 11.058 | -0.000001 | 11.058 | -0.000001 |
| 2024 | 10.813 | 10.813 | +0.000000 | 10.813 | +0.000000 |

Matched-seed FDE deltas: P1 0.000001 ± 0.000001 px; P2 0.000001 ± 0.000001 px.

## Table 3. Per-seed intention metrics

| Method | Seed | AUC | Brier | F1 | BAcc |
|---|---:|---:|---:|---:|---:|
| P1_target_only | 42 | 0.6576 | 0.2109 | 0.7742 | 0.6014 |
| P1_target_only | 123 | 0.7036 | 0.2128 | 0.7767 | 0.6450 |
| P1_target_only | 2024 | 0.7023 | 0.2156 | 0.7390 | 0.6683 |
| P2_target_scene | 42 | 0.6364 | 0.1397 | 0.9151 | 0.5086 |
| P2_target_scene | 123 | 0.8165 | 0.1019 | 0.8735 | 0.7287 |
| P2_target_scene | 2024 | 0.6194 | 0.1388 | 0.8656 | 0.5642 |

## Scientific answers

1. **Numerical preservation:** initialization equivalence passed on 1024 validation samples per seed with max absolute future-coordinate difference 0. The final matched-seed test ADE change is P1 -0.000001 ± 0.000001 px and P2 -0.000001 ± 0.000001 px; FDE changes are reported above.
2. **Did intention training modify trajectory weights?** No. All six runs recorded identical backbone SHA256 before training, after every epoch, and after selected-checkpoint reload (126 hash observations checked).
3. **Can the frozen representation support crossing-intention recognition?** P1 AUC is 0.6878 ± 0.0262 and P2 is 0.6908 ± 0.1092, compared with observed-only 0.6688 ± 0.0093. This is a modest mean AUC increase (+0.0190/+0.0220), not a stable or decisive gain; P2 varies substantially across seeds.
4. **Target-only or target+scene?** P1 AUC 0.6878 ± 0.0262 vs P2 0.6908 ± 0.1092; P1 Brier 0.2131 ± 0.0024 vs P2 0.1268 ± 0.0216. P2 has better mean Brier and marginally higher mean AUC, but its test AUC ranges from 0.6194 to 0.8165 and BAcc from 0.5086 to 0.7287. In seeds 42 and 2024, P2 validation AUCs (0.9429/0.8973) did not carry over to test (0.6364/0.6194), signaling a material generalization gap. These are descriptive results, not a test-selected model choice.
5. **Observed-only comparison:** observed-only AUC 0.6688 ± 0.0093, Brier 0.2154 ± 0.0210; P1/P2 AUC and Brier are shown in Table 1. Note the input difference: the historical observed-only model receives target_obs `[15,4]`, while P1 uses the frozen trajectory encoder over concatenated target_obs + target_abs_obs `[15,8]`; the comparison is informative but not strictly input-matched.
6. **Recovery versus fixed λ=100:** fixed λ=100 mean ADE/FDE is 15.890/26.923 px; T0 is 11.011/19.459 px. P1 and P2 both return to 11.011/19.459 px. Relative to λ=100, this improves mean ADE by 4.879 px (30.7%) and FDE by 7.464 px (27.7%), closing the observed trajectory gap to T0 for all three matched seeds.
7. **Interpretation:** preserving the original trajectory feature path and freezing its parameters is sufficient to avoid the prior trajectory degradation. Earlier audits found both changed feature flow and shared task-updated parameters; because this experiment preserves the path and freezes weights together, it does not isolate which of those factors was individually causal. Intention transfer remains separate: P1/P2 only modestly improve mean AUC over the historical observed-only comparator, with high P2 seed variation.
8. **Next stage:** Yes—a small, hypothesis-driven task-specific adapter study is worthwhile, keeping this frozen model as the trajectory-preservation control; test partial unfreezing only after that. The motivation is to address intention transfer, not trajectory recovery. Do not add reliability weighting or gradient surgery in this phase.

## Protocol and artifacts

- Frozen protocol SHA256: `e0494823ef0adc5ee40ba1f78dd3596d17b3d76f8fe89bdb5b6fa1f65c45c71f`; test set was opened only after all six validation runs, checkpoint selection, equivalence check, and protocol freeze.
- Frozen trajectory-only checkpoints are seed-matched; checkpoint SHA256 mapping and 47/47 tensor loads are in `trajectory_reference.json` and `weight_loading_report.json`.
- Initialization equivalence: `equivalence_test.json`; checkpoint mapping: `architecture_mapping.md` and `weight_loading_report.json`.
- P1/P2 each contain `metrics.json`, `validation_history.json`, `parameter_hashes.json`, `test_access_record.json`, and per-sample `test_predictions.npz` under their seed folders.
- Historical T0, observed-only, and λ=100 test metrics are reused from existing frozen artifacts; those baselines were not rerun.
