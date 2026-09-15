"""Hungarian assignment over a gated cost matrix."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

REJECTED_COST = 1.0e6

#pair evaluation returns (passed_gate, cost).
PairEvaluator = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], tuple[bool, float]]

def build_cost_matrix(track_states: list[tuple[np.ndarray, np.ndarray]], tracklet_states: list[tuple[np.ndarray, np.ndarray]],
    evaluate_pair: PairEvaluator) -> tuple[np.ndarray, np.ndarray]:

    """Build the cost matrix and valid-pair mask."""

    n_tracks = len(track_states)
    n_tracklets = len(tracklet_states)
    cost = np.full((n_tracks, n_tracklets), REJECTED_COST)
    valid = np.zeros((n_tracks, n_tracklets), dtype=bool)

    for i, (track_state, track_cov) in enumerate(track_states):
        for j, (meas_state, meas_cov) in enumerate(tracklet_states):
            passed, pair_cost = evaluate_pair(track_state, track_cov, meas_state, meas_cov)
            if passed:
                cost[i, j] = pair_cost
                valid[i, j] = True

    return cost, valid


def solve_assignment(cost: np.ndarray, valid: np.ndarray) -> tuple[list[tuple[int, int]], list[int], list[int]]:

    """Return matches and unmatched indices."""

    n_tracks, n_tracklets = cost.shape
    if n_tracks == 0 or n_tracklets == 0:
        return [], list(range(n_tracks)), list(range(n_tracklets))

    row_idx, col_idx = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(row_idx, col_idx) if valid[r, c]]

    matched_tracks = {m[0] for m in matches}
    matched_tracklets = {m[1] for m in matches}
    unmatched_tracks = [i for i in range(n_tracks) if i not in matched_tracks]
    unmatched_tracklets = [j for j in range(n_tracklets) if j not in matched_tracklets]

    return matches, unmatched_tracks, unmatched_tracklets