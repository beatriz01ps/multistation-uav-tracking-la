"""Recovery-failure-cause classification (Table 4): assigns each case to one
category, checking failure modes in the actual operational order (deleted/merged pre-gap take priority) rather than a post-hoc one."""

from __future__ import annotations

from collections import Counter

from experiment.gap_horizon.case_data import GapHorizonCase, load_cases
from experiment.gap_horizon.study_config import (
    DENSE_SCENARIO,
    DENSE_SCENARIO_SENTINEL_GAPS,
    G1_G2_G3_SCENARIOS,
    PRIMARY_GAP_RANGE,
    SEEDS,
)


def classify_failure_cause(case: GapHorizonCase) -> str:

    if case.terminal_event_type == "DELETED":
        return "DELETED_BEFORE_REACQUISITION"
    if case.terminal_event_type == "MERGED":
        return "MERGED_BEFORE_REACQUISITION"

    if case.recovery_outcome == "NEVER_REDETECTED":
        return "NEVER_REDETECTED"
    if case.recovery_outcome == "FALSE_RECOVERY":
        return "ASSOCIATED_TO_WRONG_TARGET"
    if case.recovery_outcome == "CORRECT_RECOVERY":
        return "CORRECT_RECOVERY"

    # outcome == NO_RECOVERY: alive, not deleted/merged, not reassigned
    # to the wrong target, but did not keep the same identity either.
    if case.gate_accepted is False:
        return "ALIVE_BUT_GATE_REJECTED"
    if case.gate_accepted is True:
        return "GATE_PASSED_BUT_ASSIGNMENT_NOT_SELECTED"
    return "OTHER_RECOVERY_FAILURE"  # no gate diagnostic available under NO_RECOVERY


def audit_scenario(scenario: str, gaps: list[int]) -> dict:

    cases = load_cases()
    counts_by_gap: dict[int, Counter] = {}
    total_counts = Counter()
    examples_other = []

    for gap in gaps:
        counts_by_gap[gap] = Counter()
        for seed in SEEDS:
            case = cases.get((scenario, seed, float(gap)))
            if case is None:
                counts_by_gap[gap]["MISSING_OR_FAILED_CASE"] += 1
                total_counts["MISSING_OR_FAILED_CASE"] += 1
                continue
            cause = classify_failure_cause(case)
            counts_by_gap[gap][cause] += 1
            total_counts[cause] += 1
            if cause == "OTHER_RECOVERY_FAILURE":
                examples_other.append({"scenario": scenario, "seed": seed, "gap": gap})

    total = sum(total_counts.values())
    return {
        "total_cases": total,
        "counts_by_gap": {g: dict(c) for g, c in counts_by_gap.items()},
        "total_counts": dict(total_counts),
        "fraction_by_cause": {cause: n / total for cause, n in total_counts.items()},
        "other_recovery_failure_examples": examples_other,
    }


def build_audit() -> dict:
    
    audit = {}
    for scenario in G1_G2_G3_SCENARIOS:
        audit[scenario] = audit_scenario(scenario, PRIMARY_GAP_RANGE)
    audit[DENSE_SCENARIO] = audit_scenario(DENSE_SCENARIO, DENSE_SCENARIO_SENTINEL_GAPS)
    return audit


if __name__ == "__main__":
    audit = build_audit()
    for scenario, data in audit.items():
        print(f"=== {scenario} (n={data['total_cases']}) ===")
        for cause, frac in sorted(data["fraction_by_cause"].items(), key=lambda kv: -kv[1]):
            print(f"  {cause}: {data['total_counts'][cause]} ({frac:.1%})")
        if data["other_recovery_failure_examples"]:
            print("  OTHER_RECOVERY_FAILURE examples:", data["other_recovery_failure_examples"][:5])
