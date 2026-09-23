# Frozen Transformer + uncertainty-guided social residual experiments

All trajectory distances are in image pixels. Three-seed summaries use mean ± sample standard deviation (ddof=1). `crossing=-1` was not used as training supervision.

## 1. Frozen Transformer baseline

Seed123 test ADE/FDE: **11.058475 / 19.593115 px**.
Difference from stored baseline: ADE -0.000000 px; FDE +0.000002 px. Within 0.01 px: **True**.

## 2. Stage A: intent-only social residual

| Seed | Gate mode | Residual | AUC | BAcc | F1 | Brier | ADE px | FDE px | Gate mean±std | Entropy mean±std |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 123 | none | off | 0.8089 | 0.5162 | 0.9304 | 0.0957 | 11.058 | 19.593 | 0.0000±0.0000 | 0.5433±0.2715 |
| 123 | always | off | 0.7891 | 0.7182 | 0.9141 | 0.1160 | 11.058 | 19.593 | 1.0000±0.0000 | 0.6894±0.2607 |
| 123 | uncertainty | off | 0.7366 | 0.6465 | 0.9355 | 0.0984 | 11.058 | 19.593 | 0.5844±0.2572 | 0.6086±0.2618 |

### Stage A multi-seed summary

Not run: the seed123 Stage A stop criterion fired (uncertainty AUC did not exceed no-social, and always-social had no AUC advantage). Multi-seed expansion was intentionally halted.

## 3. Stage B: trajectory social residual

Not run: Stage A seed123 met the predefined stop condition. No Stage B training was started.

## 4. Uncertainty diagnostic

Seed 123: ambiguous-vs-clean entropy AUROC 0.4565; gate AUROC 0.4565. This is a diagnostic proxy on the ambiguous annotation subset, not uncertainty ground truth.
Clean entropy 0.6086±0.2618; ambiguous entropy 0.5716±0.2217.
Clean gate 0.5844±0.2572; ambiguous gate 0.5563±0.2307.

## 5. Uncertainty-stratified analysis

Strata use the seed-matched uncertainty model's clean-test entropy terciles; compare all three Stage A models on the same samples.

| Stratum | N | Model | AUC | Brier | ADE px | FDE px |
|---|---:|---|---:|---:|---:|---:|
| low | 6111 | none | 0.7931 | 0.0311 | 14.692 | 26.905 |
| low | 6111 | always | 0.8432 | 0.0346 | 14.692 | 26.905 |
| low | 6111 | uncertainty | 0.8698 | 0.0348 | 14.692 | 26.905 |
| medium | 6110 | none | 0.6259 | 0.0431 | 9.870 | 17.132 |
| medium | 6110 | always | 0.3303 | 0.0959 | 9.870 | 17.132 |
| medium | 6110 | uncertainty | 0.3069 | 0.0751 | 9.870 | 17.132 |
| high | 6110 | none | 0.5833 | 0.2128 | 8.614 | 14.741 |
| high | 6110 | always | 0.6998 | 0.2175 | 8.614 | 14.741 |
| high | 6110 | uncertainty | 0.7163 | 0.1853 | 8.614 | 14.741 |

Interpretation must be based on the stratum-level comparisons and sample counts; a single seed is exploratory.

## Current decision

Stage A seed123 test AUC: none=0.8089, always=0.7891, uncertainty=0.7366. The predefined stop condition fired: uncertainty did not exceed no-social and always-social did not outperform no-social. Therefore no multi-seed Stage A expansion or Stage B run was started.

The Transformer baseline remains frozen. No backbone fine-tuning or next-stage model work was started.
