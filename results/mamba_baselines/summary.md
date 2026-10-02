# JAAD Mamba baseline validation audit

All new experiments use the existing train/val arrays only. No test data were opened. Existing frozen results are unchanged.

## Protocol

Target input: concat(target_obs, target_abs_obs), [B,15,8]. TT and MT share exactly the same decoder design and per-seed decoder initialization. All baselines train from scratch, with no scene, social, reliability, or intention conditioning in trajectory models.

Trajectory: 20 epochs, batch 512, AdamW lr=1e-3/wd=1e-4, clip=5, SmoothL1, ReduceLROnPlateau(factor=0.5, patience=2), select minimum validation pixel ADE. Intention: 20 epochs, batch 256, natural shuffle, inverse-frequency weighted BCE; select highest raw AUC with the same 1e-4 Brier tie-break as M0.

## Selected validation results

Means ± sample SD across the three seeds; intention metrics in this table are raw (threshold=0.5).

| Method | Parameters | Val AUC | Val Brier | Val ADE (px) | Val FDE (px) |
|---|---:|---:|---:|---:|---:|
| M0 Transformer (existing) | 614785 | 0.7557 ± 0.0154 | 0.2557 ± 0.0495 | — | — |
| trajectory_transformer_target | 618526 | — | — | 10.0191 ± 0.0927 | 18.5650 ± 0.0991 |
| trajectory_mamba | 372254 | — | — | 9.1820 ± 0.0890 | 16.4520 ± 0.1384 |
| intention_mamba | 368513 | 0.6343 ± 0.0937 | 0.2053 ± 0.0202 | — | — |

## Per-seed trajectory comparison

Delta = MT − TT; negative error delta favors MT.

| Variant | Seed | TT ADE | TT FDE | MT ADE | MT FDE | Delta ADE | Delta FDE |
|---|---:|---:|---:|---:|---:|---:|---:|
| trajectory_mamba | 42 | 9.962 | 18.451 | 9.081 | 16.492 | -0.880 | -1.959 |
| trajectory_mamba | 123 | 10.126 | 18.631 | 9.250 | 16.566 | -0.876 | -2.065 |
| trajectory_mamba | 2024 | 9.970 | 18.612 | 9.214 | 16.298 | -0.756 | -2.314 |

trajectory_mamba: mean delta ADE=-0.837px; FDE=-2.113px. Mean relative improvement: ADE=8.35% (3/3 seeds better), FDE=11.38% (3/3 better). Gate: **GO**.

## Intention metrics

| Seed | MI raw AUC | MI raw Brier | MI raw F1 | MI raw BAcc | MI raw Accuracy | M0 raw AUC |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.6060 | 0.1890 | 0.7408 | 0.5054 | 0.6066 | 0.7404 |
| 123 | 0.5580 | 0.2279 | 0.7161 | 0.4918 | 0.5778 | 0.7554 |
| 2024 | 0.7390 | 0.1990 | 0.9053 | 0.6250 | 0.8342 | 0.7712 |

Mean MI − M0 validation AUC: -0.1213. MI gate: **STOP**; GO here means no degradation exceeding 0.02, not proof of superiority. Validation-calibrated metrics are also retained in each run and follow the same procedure as M0.

## Speed and memory

CUDA event timing after warmup, FP32, eval mode, excludes host-to-device transfers; measured 30 repetitions per seed. Batch 512 for TT/MT and 256 for MI. Sequence length is only 15; these measurements do not establish asymptotic-complexity benefits. The optional causal-conv1d package is absent; Mamba uses the officially supported PyTorch convolution fallback plus compiled selective-scan CUDA kernels. Timings describe this installation, not an optimally fused installation.

| Method | Single sample (ms) | Batch latency (ms) | Peak train allocated GPU memory (MiB) |
|---|---:|---:|---:|
| trajectory_transformer_target | 0.8495 ± 0.0080 | 1.9718 ± 0.0334 | 278.6 |
| trajectory_mamba | 1.4997 ± 0.0249 | 5.9977 ± 0.0382 | 321.7 |
| intention_mamba | 1.5176 ± 0.0080 | 3.2881 ± 0.0617 | 173.2 |

## Decision and scope

- Mamba trajectory: **GO**. Rule: ≥2% mean improvement in ADE or FDE with ≥2/3 seeds improving the same metric.
- Mamba intention: **STOP**. Rule: mean raw AUC must not decrease by more than 0.02 versus existing input-matched M0.
- Permitted two-layer trajectory check needed: False.
- All completed runs have finite losses/gradients/predictions; no NaN or Inf detected.
- Hidden dimensions, input, decoder, seeds, data order, and trajectory optimizer/schedule are matched; total parameter counts differ between temporal encoder families.
- Validation selection is exploratory. No paper contribution, intent-guidance benefit, or test generalization is established by this baseline stage.
- Intent-Guided G1/G2/G3 and protocol freeze are deferred to a later instruction.

## Post-run verification and interpretation

The following notes were added after independent verification of the generated results. All nine runs completed exactly 20 epochs (180 epochs total); selected epochs were rechecked against their histories, all nine local checkpoint hashes matched, and train/val and existing M0-reference hashes matched their recorded values. The final full test suite passed: **176 passed, 0 failed, 0 skipped**, 60 standard Transformer warnings, 9.33 seconds. All new Python files passed `py_compile`; `pip check` reported no broken requirements. Detailed checks are retained in `verification.json`.

Finite arithmetic is not the same as stable optimization or reliable generalization. MI showed substantial epoch-to-epoch validation variation and large pre-clipping gradient norms, despite a gradient clip of 5:

| MI seed | Selected epoch | Raw validation AUC range over 20 epochs | Maximum gradient norm before clipping | First → last training weighted BCE |
|---:|---:|---:|---:|---:|
| 42 | 4 | 0.3136–0.6060 | 163.94 | 0.6729 → 0.2092 |
| 123 | 8 | 0.3105–0.5580 | 223.13 | 0.6623 → 0.1345 |
| 2024 | 1 | 0.3337–0.7390 | 202.18 | 0.6872 → 0.2333 |

These observations are consistent with an optimization/generalization problem in this prescribed intention baseline, but do not establish its cause. No architecture, loss, learning-rate, sampling, or epoch changes were made in response. Validation contains 2322 crossing and 314 non-crossing samples; F1/accuracy alone cannot establish a useful intent source under this imbalance. MI's lower mean raw Brier than M0 does not override its failed prespecified AUC gate.

MT's error improvements are repeatable across the three seeds, but the installed implementation is **not faster** than TT: mean single-sample latency is 1.500 vs 0.849 ms, and batch-512 latency is 5.998 vs 1.972 ms. MT has fewer parameters, but higher measured peak training allocated memory (321.7 vs 278.6 MiB). Do not claim a demonstrated speed/memory advantage.

**Handoff:** retain the three-layer MT trajectory baseline; stop using standalone MI as the intent source. No two-layer sweep was needed, no alternative intention model was substituted, no Intent-Guided model was implemented, and no test data were accessed. The next stage requires a new instruction.
