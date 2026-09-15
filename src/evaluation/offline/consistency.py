"""Covariance consistency: NEES (against ground truth) and NIS (already computed per row, never reconstructed here)."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from evaluation.offline.loaders import GroundTruthTrajectory, TrackHistoryRow
from models.validation import regularize_for_inversion


def nees(estimated_state: np.ndarray, estimated_covariance: np.ndarray, true_state: np.ndarray) -> float:

    """(x_hat - x)^T P^-1 (x_hat - x), over whichever dims of estimated_state/
    estimated_covariance are passed (full 6D, 3D position, or 3D velocity)."""

    error = estimated_state - true_state
    precision = np.linalg.inv(regularize_for_inversion(estimated_covariance))
    return float(error @ precision @ error)


def nees_series(rows: list[TrackHistoryRow], ground_truth: dict[str, GroundTruthTrajectory], majority_labels: dict[int, str],
    dims: slice = slice(0, 6)) -> list[float]:

    """NEES per row of one run, in chronological order"""

    values = []
    for row in rows:
        target_id = majority_labels.get(row.global_track_id)
        if target_id is None:
            continue
        trajectory = ground_truth.get(target_id)
        if trajectory is None:
            continue
        true_state = trajectory.at(row.timestamp)
        if true_state is None:
            continue
        values.append(
            nees(row.state[dims], row.covariance[dims, dims], true_state[dims])
        )
    return values


def nis_series(rows: list[TrackHistoryRow]) -> list[float]:

    """NIS already computed per row (None on PREDICTION_ONLY cycles, excluded here)"""

    return [row.nis for row in rows if row.nis is not None]


def nees_by_epoch_and_target(rows: list[TrackHistoryRow], ground_truth: dict[str, GroundTruthTrajectory],
    majority_labels: dict[int, str], dims: slice = slice(0, 6)) -> dict[tuple[float, str], float]:

    """Same logic as nees_series, but keyed by (timestamp, real target) -> needed
    for ANEES aggregation across seeds, matching the same epoch and target between independent runs."""

    result: dict[tuple[float, str], float] = {}
    for row in rows:
        target_id = majority_labels.get(row.global_track_id)
        if target_id is None:
            continue
        trajectory = ground_truth.get(target_id)
        if trajectory is None:
            continue
        true_state = trajectory.at(row.timestamp)
        if true_state is None:
            continue
        result[(row.timestamp, target_id)] = nees(row.state[dims], row.covariance[dims, dims], true_state[dims])
    return result


def nis_by_epoch_and_target( rows: list[TrackHistoryRow], majority_labels: dict[int, str]) -> dict[tuple[float, str], float]:

    result: dict[tuple[float, str], float] = {}
    for row in rows:
        if row.nis is None:
            continue
        target_id = majority_labels.get(row.global_track_id)
        if target_id is None:
            continue
        result[(row.timestamp, target_id)] = row.nis
    return result


def average_nees(values: list[float]) -> float:

    return float(np.mean(values)) if values else float("nan")


def chi_square_consistency_bounds(num_independent_samples: int, degrees_of_freedom: int, confidence: float = 0.95) -> tuple[float, float]:

    """The classical (Bar-Shalom) consistency interval for ANEES/ANIS"""

    alpha = 1.0 - confidence
    total_df = num_independent_samples * degrees_of_freedom
    lower = chi2.ppf(alpha / 2.0, df=total_df) / num_independent_samples
    upper = chi2.ppf(1.0 - alpha / 2.0, df=total_df) / num_independent_samples
    return lower, upper


def is_consistent(average_value: float, num_independent_samples: int, degrees_of_freedom: int, confidence: float = 0.95) -> bool:

    lower, upper = chi_square_consistency_bounds(num_independent_samples, degrees_of_freedom, confidence)
    return bool(lower <= average_value <= upper)
