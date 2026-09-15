"""Position/velocity RMSE against ground truth, split into RMSE_update (real measurement) vs RMSE_coasting (pure prediction) vs RMSE_overall."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from evaluation.offline.loaders import GroundTruthTrajectory, TrackHistoryRow


@dataclass(frozen=True)
class RmseReport:
    position_rmse_update_m: float
    position_rmse_coasting_m: float
    position_rmse_overall_m: float
    velocity_rmse_update_mps: float
    velocity_rmse_coasting_mps: float
    velocity_rmse_overall_mps: float
    num_samples_update: int
    num_samples_coasting: int


def _rmse(errors: list[float]) -> float:
    return math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else float("nan")


def compute_rmse(rows: list[TrackHistoryRow], ground_truth: dict[str, GroundTruthTrajectory], majority_labels: dict[int, str]) -> RmseReport:

    pos_update: list[float] = []
    pos_coasting: list[float] = []
    vel_update: list[float] = []
    vel_coasting: list[float] = []

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

        position_error = float(np.linalg.norm(row.state[:3] - true_state[:3]))
        velocity_error = float(np.linalg.norm(row.state[3:6] - true_state[3:6]))

        if row.prediction_only:
            pos_coasting.append(position_error)
            vel_coasting.append(velocity_error)
        else:
            pos_update.append(position_error)
            vel_update.append(velocity_error)

    return RmseReport(
        position_rmse_update_m=_rmse(pos_update),
        position_rmse_coasting_m=_rmse(pos_coasting),
        position_rmse_overall_m=_rmse(pos_update + pos_coasting),
        velocity_rmse_update_mps=_rmse(vel_update),
        velocity_rmse_coasting_mps=_rmse(vel_coasting),
        velocity_rmse_overall_mps=_rmse(vel_update + vel_coasting),
        num_samples_update=len(pos_update),
        num_samples_coasting=len(pos_coasting),
    )
