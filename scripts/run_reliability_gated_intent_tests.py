#!/usr/bin/env python3
"""Run the registered regression tests and record pass/fail/skip totals."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/reliability_gated_intent_15x15/test_run_report.json"
TEST_TARGETS = (
    "tests/test_trajectory_reliability.py",
    "tests/test_trajectory_reliability_deconfounding.py",
    "tests/test_reliability_internal_replication.py",
    "tests/test_reliability_gated_intent.py",
    "tests",
)


def parse_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0, "xfailed": 0, "xpassed": 0}
    for name in counts:
        found = re.search(rf"(\d+) {name}", output)
        if found:
            counts[name] = int(found.group(1))
    return counts


def main() -> None:
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    python = sys.executable
    records: list[dict[str, Any]] = []
    overall_failure = False
    for target in TEST_TARGETS:
        command = [python, "-m", "pytest", target, "-q"]
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout
        counts = parse_counts(output)
        records.append({
            "target": target,
            "command": command,
            "exit_code": result.returncode,
            **counts,
            "output_tail": "\n".join(output.splitlines()[-30:]),
        })
        overall_failure |= result.returncode != 0
        print(json.dumps({"target": target, "exit_code": result.returncode, **counts}, ensure_ascii=False), flush=True)
        if result.returncode:
            print(output, flush=True)
    full_suite = records[-1]
    report = {
        "environment": {"python": python, "pytest_plugin_autoload_disabled": True},
        "runs": records,
        "full_suite": {key: full_suite[key] for key in ("target", "exit_code", "passed", "failed", "skipped", "errors", "xfailed", "xpassed")},
        "passed": full_suite["passed"],
        "failed": full_suite["failed"],
        "skipped": full_suite["skipped"],
        "errors": full_suite["errors"],
        "all_commands_passed": not overall_failure,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}", flush=True)
    if overall_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
