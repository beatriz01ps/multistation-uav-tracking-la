"""Prepare frozen Ma-inspired inputs for reproduction."""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiment.gap_horizon.study_config import (
    MA_COMPARISON_SCENARIOS,
    MA_EXPLORATORY_SEEDS,
)

INPUTS_DIR = REPO_ROOT / "reproducibility" / "inputs" / "ma_inspired"
BASE_INPUTS_DIR = (
    REPO_ROOT
    / "results"
    / "gap_horizon_base_inputs"
    / "unclassified"
)

_FROZEN_INPUT_LABEL = "frozen"


def prepare_case(scenario: str, seed: int) -> None:

    """Decompress the frozen inputs for one scenario and seed."""
    seed_dir = INPUTS_DIR / scenario / f"seed{seed}"
    sim_dir = (
        BASE_INPUTS_DIR
        / f"{scenario}__seed{seed}__{_FROZEN_INPUT_LABEL}"
        / "sim"
    )
    sim_dir.mkdir(parents=True, exist_ok=True)

    (sim_dir / "sent_messages.jsonl").write_bytes(
        gzip.decompress(
            (seed_dir / "sent_messages.jsonl.gz").read_bytes()
        )
    )

    (sim_dir / "tracklet_origins.jsonl").write_bytes(
        gzip.decompress(
            (seed_dir / "tracklet_origins.jsonl.gz").read_bytes()
        )
    )

    ground_truth_gz = (
        INPUTS_DIR
        / "ground_truth"
        / f"{scenario}.jsonl.gz"
    )

    (sim_dir / "ground_truth.jsonl").write_bytes(
        gzip.decompress(ground_truth_gz.read_bytes())
    )


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=None,
        help="one or more scenario names (default: all 3)",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=None,
        help="one or more seeds (default: all 10, 5001-5010)",
    )
    args = parser.parse_args()

    scenarios = args.scenario or MA_COMPARISON_SCENARIOS
    seeds = args.seed or MA_EXPLORATORY_SEEDS

    prepared = 0
    failed = []

    for scenario in scenarios:
        for seed in seeds:
            try:
                prepare_case(scenario, seed)
                prepared += 1
            except Exception as exc:
                failed.append(
                    (scenario, seed, repr(exc))
                )

    print(
        f"prepared {prepared}/"
        f"{len(scenarios) * len(seeds)} base captures"
    )

    for scenario, seed, error in failed:
        print(f"  FAILED {scenario}/seed{seed}: {error}")

    print(f"output: {BASE_INPUTS_DIR}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    
    main()