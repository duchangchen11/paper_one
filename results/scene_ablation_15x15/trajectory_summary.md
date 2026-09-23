# 15×15 trajectory scene ablation

Pixel ADE/FDE are primary; normalized ADE/FDE and validation metrics are retained in the JSON. Paired Δ = zero-scene − real-scene, so a positive value favors real-scene input.

| Seed | Real ADE px | Zero ADE px | ΔADE | Real FDE px | Zero FDE px | ΔFDE |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 11.160 | 10.512 | -0.648 | 19.781 | 19.023 | -0.758 |
| 123 | 11.058 | 10.908 | -0.151 | 19.593 | 19.392 | -0.201 |
| 2024 | 10.813 | 10.631 | -0.182 | 19.003 | 19.365 | +0.362 |

- Real ADE: 11.011 ± 0.178 px; zero ADE: 10.684 ± 0.203 px; paired ΔADE: -0.327 ± 0.279 px.
- Real FDE: 19.459 ± 0.406 px; zero FDE: 19.260 ± 0.205 px; paired ΔFDE: -0.199 ± 0.560 px.
- Paired seed outcome: real-scene lower ADE in 0/3 and lower FDE in 1/3; zero-scene lower ADE in 3/3 and lower FDE in 2/3.
- Conclusion: No stable trajectory benefit from real scene input is supported: real-scene ADE is lower in 0/3 seeds and FDE is lower in 1/3; zero-scene has lower ADE in 3/3.

The separate same-checkpoint real-vs-zero inference diagnostic is not the retrained no-scene baseline; see `trajectory_zero_scene_inference_diagnostic.json`.
