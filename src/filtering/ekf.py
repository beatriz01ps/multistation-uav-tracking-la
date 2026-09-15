"""EkfTracker"""

from __future__ import annotations

import numpy as np

from filtering.motion_models import MotionModel
from models.validation import regularize_for_inversion as regularize_covariance

MEASUREMENT_DIM = 6  # dimension of the ext contract 


class EkfTracker:
    def __init__(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        motion_model: MotionModel,
        process_noise_acceleration_std: float,
    ) -> None:
        augmented_state, augmented_covariance = motion_model.augment_initial_state(
            np.asarray(initial_state, dtype=float).copy(),
            np.asarray(initial_covariance, dtype=float).copy(),
        )
        self._x = augmented_state
        self._P = regularize_covariance(augmented_covariance)
        self._motion_model = motion_model
        self._process_noise_acceleration_std = process_noise_acceleration_std
        self._last_innovation: np.ndarray | None = None  
        self._last_innovation_covariance: np.ndarray | None = None

    def predict(self, dt: float) -> None:
        
        self._last_innovation = None
        self._last_innovation_covariance = None
        if dt <= 0:
            return
        
        transition_jacobian = self._motion_model.jacobian(self._x, dt)
        self._x = self._motion_model.step(self._x, dt)
        process_noise = self._motion_model.process_noise(dt, self._process_noise_acceleration_std)
        self._P = regularize_covariance(
            transition_jacobian @ self._P @ transition_jacobian.T + process_noise
        )

    def update(self, measurement: np.ndarray, measurement_covariance: np.ndarray) -> None:
        
        state_dim = self._x.shape[0]
        observation_matrix = np.zeros((MEASUREMENT_DIM, state_dim))
        observation_matrix[:, :MEASUREMENT_DIM] = np.eye(MEASUREMENT_DIM)

        measurement = np.asarray(measurement, dtype=float)
        measurement_covariance = np.asarray(measurement_covariance, dtype=float)

        innovation = measurement - observation_matrix @ self._x
        innovation_covariance = regularize_covariance(
            observation_matrix @ self._P @ observation_matrix.T + measurement_covariance
        )
        self._last_innovation = innovation.copy()
        self._last_innovation_covariance = innovation_covariance.copy()
       
        kalman_gain = np.linalg.solve(innovation_covariance.T, observation_matrix @ self._P.T).T

        self._x = self._x + kalman_gain @ innovation
        identity = np.eye(state_dim)
        
        gain_term = identity - kalman_gain @ observation_matrix
        self._P = regularize_covariance(
            gain_term @ self._P @ gain_term.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )

    def set_state(self, state: np.ndarray, covariance: np.ndarray) -> None:

       
        self._x[:MEASUREMENT_DIM] = np.asarray(state, dtype=float)
        self._P[:MEASUREMENT_DIM, :MEASUREMENT_DIM] = regularize_covariance(np.asarray(covariance, dtype=float))
        self._P = regularize_covariance(self._P)
        
        self._last_innovation = None
        self._last_innovation_covariance = None

    @property
    def state(self) -> np.ndarray:
        return self._x[:MEASUREMENT_DIM].copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._P[:MEASUREMENT_DIM, :MEASUREMENT_DIM].copy()

    @property
    def last_innovation(self) -> np.ndarray | None:
        return self._last_innovation.copy() if self._last_innovation is not None else None

    @property
    def last_innovation_covariance(self) -> np.ndarray | None:
        return self._last_innovation_covariance.copy() if self._last_innovation_covariance is not None else None
