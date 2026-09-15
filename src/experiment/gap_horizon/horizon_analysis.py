"""Aggregated horizon analysis across seeds"""

from __future__ import annotations


def recovery_rate_by_gap(outcomes_by_gap_and_seed: dict[int, dict[int, str]], success_outcome: str = "CORRECT_RECOVERY") -> dict[int, float]:

    """outcomes_by_gap_and_seed[gap][seed] = outcome -> fraction of seeds with
    success_outcome, per gap. Only gaps with >=1 executed seed appear."""
    
    return {
        gap: sum(1 for outcome in seeds.values() if outcome == success_outcome) / len(seeds)
        for gap, seeds in outcomes_by_gap_and_seed.items()
        if seeds
    }


def recovery_horizon(rate_by_gap: dict[int, float], threshold: float) -> int | None:

    """Largest gap g where every known gap <= g has rate>=threshold, walking up
    from the smallest; requires a continuous integer sequence (see gaps_needing_refinement). None if even the smallest gap fails."""

    horizon: int | None = None
    for gap in sorted(rate_by_gap):
        if rate_by_gap[gap] >= threshold:
            horizon = gap
        else:
            break
    return horizon


def gaps_needing_refinement(sampled_gaps: list[int], rate_by_gap: dict[int, float], thresholds: list[float]) -> set[int]:

    """Per threshold, finds the first consecutive sampled pair crossing that
    threshold and unions every integer strictly between them. Empty if no threshold drops in range, or the first gap already fails it."""

    sampled_sorted = sorted(sampled_gaps)
    needed: set[int] = set()
    for threshold in thresholds:
        for g_lo, g_hi in zip(sampled_sorted, sampled_sorted[1:]):
            if rate_by_gap.get(g_lo, 0.0) >= threshold and rate_by_gap.get(g_hi, 1.0) < threshold:
                needed.update(range(g_lo + 1, g_hi))
                break
    return needed


def refinement_is_complete(sampled_gaps: list[int], rate_by_gap: dict[int, float], thresholds: list[float]) -> bool:

    """True when gaps_needing_refinement has nothing left to propose. Aall 3
    H_q already have 1s resolution up to their drop point (or the last sampled gap)."""

    return len(gaps_needing_refinement(sampled_gaps, rate_by_gap, thresholds)) == 0
