"""Chooses UKF or EKF by config (filter.type)"""

from __future__ import annotations

import numpy as np

from filtering.base import FilterTracker
from filtering.ekf import EkfTracker
from filtering.motion_models import MotionModel
from filtering.ukf import UkfTracker
from models.enums import FilterType


def create_filter_tracker(
    filter_type: FilterType,
    *,
    initial_state: np.ndarray,
    initial_covariance: np.ndarray,
    motion_model: MotionModel,
    process_noise_acceleration_std: float,
) -> FilterTracker:
    if filter_type is FilterType.UKF:
        return UkfTracker(initial_state, initial_covariance, motion_model, process_noise_acceleration_std)
    if filter_type is FilterType.EKF:
        return EkfTracker(initial_state, initial_covariance, motion_model, process_noise_acceleration_std)
    raise ValueError(f"unknown filter type: {filter_type}")
