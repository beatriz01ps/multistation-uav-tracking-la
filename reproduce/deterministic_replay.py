"""Reproduce the deterministic-replay validation."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiment.pipeline_configs import (
    build_replay_contrast_configs,
    load_frozen_d_star,
)
from experiment.replay import run_replay_and_evaluate
from experiment.spec import compute_config_hash

FROZEN_INPUTS_DIR = (
    REPO_ROOT
    / "data"
    / "deterministic_replay"
    / "frozen_inputs"
)
EXPECTED_CSV = (
    REPO_ROOT
    / "expected"
    / "deterministic_replay"
    / "expected_cases.csv"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "data"
    / "deterministic_replay"
    / "_generated"
)

SCENARIOS = [
    "geometry_inline",
    "extreme_dense_swarm_crossing",
]

SEEDS_DEFAULT = [2001] #25
SEEDS_FULL = [2001, 2002, 2003, 2004, 2005]

REFERENCE_CONFIG_HASH = "dc2328da"


def _load_expected() -> dict[tuple[str, int, str], dict]:

    expected = {}

    with EXPECTED_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["scenario"],
                int(row["seed"]),
                row["config"],
            )
            expected[key] = row

    return expected


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all 5 seeds (70 cases) instead of just seed 2001",
    )
    args = parser.parse_args()

    seeds = SEEDS_FULL if args.full else SEEDS_DEFAULT

    d_star = load_frozen_d_star(
        REPO_ROOT / "frozen" / "frozen_experiment_config.yaml"
    )

    configs = build_replay_contrast_configs(d_star)

    assert (
        compute_config_hash(configs["reference"])[:8]
        == REFERENCE_CONFIG_HASH
    )

    expected = _load_expected()

    print("Deterministic replay")
    print("-" * 21)

    all_ok = True

    for scenario in SCENARIOS:
        for seed in seeds:
            source_sim_dir = (
                FROZEN_INPUTS_DIR
                / f"{scenario}__seed{seed}"
                / "sim"
            )

            for config_label, config in configs.items():
                metrics, _diag = run_replay_and_evaluate(
                    source_sim_dir,
                    config,
                    OUTPUT_DIR
                    / scenario
                    / str(seed)
                    / config_label,
                )

                expected_row = expected.get(
                    (scenario, seed, config_label)
                )

                if expected_row is None:
                    print(
                        f"{scenario} seed{seed} "
                        f"{config_label}: NO EXPECTED ROW"
                    )
                    all_ok = False
                    continue

                ok = (
                    str(metrics["association_accuracy"])
                    == expected_row["association_accuracy"]
                    and str(metrics["id_switches"])
                    == expected_row["id_switches"]
                )

                all_ok = all_ok and ok
                status = "MATCH" if ok else "DIFFERS"

                print(
                    f"{scenario:<28} "
                    f"seed{seed} "
                    f"{config_label:<20} "
                    f"acc={metrics['association_accuracy']:.3f} "
                    f"id_switches={metrics['id_switches']:<3} "
                    f"{status}"
                )

    print(f"\nOVERALL: {'MATCH' if all_ok else 'DIFFERS'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":

    main()