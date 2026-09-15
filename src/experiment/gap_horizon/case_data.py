"""Loads Table 4's case data """

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = REPO_ROOT / "data" / "table4" / "gap_horizon_cases.csv"


@dataclass(frozen=True)
class GapHorizonCase:
    scenario: str
    seed: int
    gap_s: float
    pre_gap_track: bool
    terminal_event_type: str  # "", "DELETED", or "MERGED"
    recovery_outcome: str
    gate_accepted: bool | None
    exact_position_error_m: float | None
    last_bucket_position_error_m: float | None
    last_bucket_delta_t_s: float | None


def _optional_bool(value: str) -> bool | None:
    
    return None if value == "" else value == "True"


def _optional_float(value: str) -> float | None:

    return None if value == "" else float(value)


_CACHE: dict[tuple[str, int, float], GapHorizonCase] | None = None


def load_cases() -> dict[tuple[str, int, float], GapHorizonCase]:

    global _CACHE
    if _CACHE is not None:
        return _CACHE
    cases: dict[tuple[str, int, float], GapHorizonCase] = {}
    with CSV_PATH.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case = GapHorizonCase(
                scenario=row["scenario"],
                seed=int(row["seed"]),
                gap_s=float(row["gap_s"]),
                pre_gap_track=row["pre_gap_track"] == "True",
                terminal_event_type=row["terminal_event_type"],
                recovery_outcome=row["recovery_outcome"],
                gate_accepted=_optional_bool(row["gate_accepted"]),
                exact_position_error_m=_optional_float(row["exact_position_error_m"]),
                last_bucket_position_error_m=_optional_float(row["last_bucket_position_error_m"]),
                last_bucket_delta_t_s=_optional_float(row["last_bucket_delta_t_s"]),
            )
            cases[(case.scenario, case.seed, case.gap_s)] = case
    _CACHE = cases
    return cases


def get_case(scenario: str, seed: int, gap: float) -> GapHorizonCase | None:

    return load_cases().get((scenario, seed, float(gap)))
