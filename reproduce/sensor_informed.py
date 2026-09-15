"""Reproduce the sensor-informed robustness evaluation: 120 runs across 6 scenarios
x 10 seeds x {position_only, full_state}, checking per-scenario identity-switch counts against the bundled canonical result."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_METRICS_CSV = REPO_ROOT / "data" / "sensor_informed" / "run_metrics.csv"
EXPECTED_JSON = REPO_ROOT / "expected" / "sensor_informed" / "summary.json"

ASSOCIATION_MODES = ["position_only", "full_state"]


def main() -> None:
    with RUN_METRICS_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = json.loads(EXPECTED_JSON.read_text(encoding="utf-8"))["identity_switch_counts"]

    scenarios = sorted({row["scenario"] for row in rows})

    print("Sensor-informed robustness evaluation")
    print("-" * 38)

    all_ok = True
    for scenario in scenarios:
        for mode in ASSOCIATION_MODES:
            subset = [row for row in rows if row["scenario"] == scenario and row["association_mode"] == mode]
            n_switch = sum(1 for row in subset if row["has_identity_switch"] == "True")
            exp = expected[f"{scenario}__{mode}"]

            ok = len(subset) == exp["n_runs"] and n_switch == exp["n_with_identity_switch"]
            all_ok = all_ok and ok

            print(
                f"{scenario:<32} {mode:<15} {n_switch}/{len(subset)} runs with an identity switch  "
                f"{'MATCH' if ok else 'DIFFERS'}"
            )

    print(f"\nOVERALL: {'MATCH' if all_ok else 'DIFFERS'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
