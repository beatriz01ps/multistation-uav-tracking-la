"""Compute squared Mahalanobis distance"""

from __future__ import annotations
import numpy as np

from models.validation import regularize_for_inversion

def mahalanobis_squared(predicted_state: np.ndarray, predicted_covariance: np.ndarray, measurement: np.ndarray,
    measurement_covariance: np.ndarray, dims: tuple[int, ...]) -> float:

    idx = list(dims)
    innovation = (measurement - predicted_state)[idx]
    innovation_covariance = (predicted_covariance + measurement_covariance)[np.ix_(idx, idx)]
    innovation_covariance = regularize_for_inversion(innovation_covariance)
    y = np.linalg.solve(innovation_covariance, innovation)
    return float(innovation @ y)