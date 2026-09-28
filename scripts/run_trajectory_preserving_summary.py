#!/usr/bin/env python3
"""Run the frozen-results report with compatibility for NumPy generator means.

The main report generator was included in the pre-test protocol hash. Keep that
file immutable; this post-freeze presentation wrapper only materializes generator
arguments as lists for NumPy versions that do not accept iterators in ``mean``.
It reads saved metrics and never opens the test archive or performs inference.
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import summarize_trajectory_preserving_joint as report


def _mean_with_iterable_support(values, *args, **kwargs):
    try:
        is_iterator = iter(values) is values
    except TypeError:
        is_iterator = False
    if is_iterator:
        values = list(values)
    return np_mean(values, *args, **kwargs)


np_mean = np.mean
np.mean = _mean_with_iterable_support


if __name__ == "__main__":
    report.main()
    summary_path = report.RESULTS / "summary.md"
    summary = summary_path.read_text(encoding="utf-8")
    replacements = {
        "3. **Can the frozen representation support crossing-intention recognition?** It yields P1 AUC 0.6878 ± 0.0262 and P2 AUC 0.6908 ± 0.1092. The observed-only reference is 0.6688 ± 0.0093. This establishes measurable transfer, but performance relative to that reference should be judged with the per-seed spread and input-definition caveat below.":
        "3. **Can the frozen representation support crossing-intention recognition?** P1 AUC is 0.6878 ± 0.0262 and P2 is 0.6908 ± 0.1092, compared with observed-only 0.6688 ± 0.0093. This is a modest mean AUC increase (+0.0190/+0.0220), not a stable or decisive gain; P2 varies substantially across seeds.",
        "4. **Target-only or target+scene?** P1 AUC 0.6878 ± 0.0262 vs P2 0.6908 ± 0.1092; P1 Brier 0.2131 ± 0.0024 vs P2 0.1268 ± 0.0216. Descriptively, P2_target_scene has the higher mean AUC; this is not a test-selected model choice.":
        "4. **Target-only or target+scene?** P1 AUC 0.6878 ± 0.0262 vs P2 0.6908 ± 0.1092; P1 Brier 0.2131 ± 0.0024 vs P2 0.1268 ± 0.0216. P2 has better mean Brier and marginally higher mean AUC, but its test AUC ranges from 0.6194 to 0.8165 and BAcc from 0.5086 to 0.7287. In seeds 42 and 2024, P2 validation AUCs (0.9429/0.8973) did not carry over to test (0.6364/0.6194), signaling a material generalization gap. These are descriptive results, not a test-selected model choice.",
        "6. **Recovery versus fixed λ=100:** fixed λ=100 mean ADE/FDE 15.890/26.923 px; T0 11.011/19.459 px. P1 is 11.011/19.459 px and P2 is 11.011/19.459 px. This puts trajectory forecasting back on the pretrained trajectory-only path rather than merely reducing the joint degradation.":
        "6. **Recovery versus fixed λ=100:** fixed λ=100 mean ADE/FDE is 15.890/26.923 px; T0 is 11.011/19.459 px. P1 and P2 both return to 11.011/19.459 px. Relative to λ=100, this improves mean ADE by 4.879 px (30.7%) and FDE by 7.464 px (27.7%), closing the observed trajectory gap to T0 for all three matched seeds.",
        "7. **Interpretation:** because the original joint model changed trajectory feature flow and shared task-updated parameters, while this controlled model preserves the original decoder path and freezes all trajectory weights, any maintained trajectory accuracy is evidence that the previous degradation was caused by joint architecture/representation interference. Intention quality is a separate representation-transfer question; this experiment does not prove frozen trajectory features are universally sufficient.":
        "7. **Interpretation:** preserving the original trajectory feature path and freezing its parameters is sufficient to avoid the prior trajectory degradation. Earlier audits found both changed feature flow and shared task-updated parameters; because this experiment preserves the path and freezes weights together, it does not isolate which of those factors was individually causal. Intention transfer remains separate: P1/P2 only modestly improve mean AUC over the historical observed-only comparator, with high P2 seed variation.",
        "8. **Next stage:** The frozen transfer is promising; retain it as a control and only proceed to a preregistered adapter/partial-unfreezing study if it has a clearly stated hypothesis. Do not add reliability weighting, PCGrad, or further modules in this phase.":
        "8. **Next stage:** Yes—a small, hypothesis-driven task-specific adapter study is worthwhile, keeping this frozen model as the trajectory-preservation control; test partial unfreezing only after that. The motivation is to address intention transfer, not trajectory recovery. Do not add reliability weighting or gradient surgery in this phase.",
    }
    for old, new in replacements.items():
        if old not in summary:
            raise RuntimeError("Expected generated summary paragraph was not found")
        summary = summary.replace(old, new)
    summary_path.write_text(summary, encoding="utf-8")
