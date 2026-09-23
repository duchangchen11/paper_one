# Uncertainty-stratified Stage A metrics (seed 123)

Samples are stratified by terciles of the uncertainty model's normalized prior entropy on the clean test split. `crossing=-1` is not used as uncertainty ground truth.

Cut points: `{'low_to_medium': 0.4832247197628021, 'medium_to_high': 0.7497923374176025}`.

| Entropy stratum | N | Model | AUC | Brier | ADE (px) | FDE (px) |
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

## Clean vs ambiguous diagnostic proxy

Entropy AUROC: 0.4565; gate AUROC: 0.4565.

Clean entropy mean±std: 0.6086±0.2618; ambiguous: 0.5716±0.2217.
Clean gate mean±std: 0.5844±0.2572; ambiguous: 0.5563±0.2307.

This is a diagnostic proxy on the ambiguous annotation subset, not a claim that the annotation is ground-truth uncertainty.
