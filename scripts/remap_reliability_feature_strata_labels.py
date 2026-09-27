#!/usr/bin/env python3
"""Correct reliability naming on frozen strata without recomputing metrics."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reliability_feature_intent_utils import load_json, write_json

RESULT = PROJECT_ROOT / "results/reliability_feature_intent_15x15/reliability_strata.json"


def main() -> None:
    result = load_json(RESULT)
    old = result["strata"]
    if set(old) != {"low_reliability", "medium_reliability", "high_reliability"}:
        raise ValueError("unexpected strata keys; refusing to remap")
    # Lower adjusted_u is more reliable; the old labels followed ascending
    # adjusted_u and therefore named the two extremes in the wrong direction.
    result["strata"] = {
        "high_reliability": old["low_reliability"],
        "medium_reliability": old["medium_reliability"],
        "low_reliability": old["high_reliability"],
    }
    result["meaning"] = {
        "high_reliability": "lower adjusted_u residual; comparatively lower uncertainty and more reliable trajectory feature",
        "medium_reliability": "middle adjusted_u tertile",
        "low_reliability": "higher adjusted_u residual; comparatively higher uncertainty",
    }
    result["high_uncertainty_group_primary"] = result["strata"]["low_reliability"]
    result["strata_label_mapping_corrected_after_metric_computation"] = True
    write_json(RESULT, result)
    print(f"Corrected labels only; metric values preserved: {RESULT}")


if __name__ == "__main__":
    main()
