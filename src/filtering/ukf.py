"""UkfTracker"""

from __future__ import annotations

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from filtering.motion_models import MotionModel
from models.validation import regularize_for_inversion as regularize_covariance

MEASUREMENT_DIM = 6  # dimension of the exr contract 


class UkfTracker:
    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray,
        motion_model: MotionModel, process_noise_acceleration_std: float) -> None:
        state_dim = motion_model.state_dim

        augmented_state, augmented_covariance = motion_model.augment_initial_state(
            np.asarray(initial_state, dtype=float).copy(),
            np.asarray(initial_covariance, dtype=float).copy(),
        )

        points = MerweScaledSigmaPoints(n=state_dim, alpha=0.1, beta=2.0, kappa=3 - state_dim)
        self._ukf = UnscentedKalmanFilter(
            dim_x=state_dim,
            dim_z=MEASUREMENT_DIM,
            dt=1.0,
            fx=lambda x, dt: motion_model.step(x, dt),
            hx=lambda x: x[:MEASUREMENT_DIM],
            points=points,
        )

        self._ukf.x = augmented_state
        self._ukf.P = regularize_covariance(augmented_covariance)
        self._motion_model = motion_model
        self._process_noise_acceleration_std = process_noise_acceleration_std
        self._has_updated = False 

    def predict(self, dt: float) -> None:
        self._has_updated = False
        if dt <= 0:
            return
        self._ukf.Q = self._motion_model.process_noise(dt, self._process_noise_acceleration_std)
        
        self._ukf.P = regularize_covariance(self._ukf.P)
        self._ukf.predict(dt=dt)
        self._ukf.P = regularize_covariance(self._ukf.P)

    def update(self, measurement: np.ndarray, measurement_covariance: np.ndarray) -> None:

        self._ukf.update(np.asarray(measurement, dtype=float), R=np.asarray(measurement_covariance, dtype=float))
        self._ukf.P = regularize_covariance(self._ukf.P)
        self._has_updated = True

    def set_state(self, state: np.ndarray, covariance: np.ndarray) -> None:

        """Overwrites the kine part of the filter's state"""

        self._ukf.x[:MEASUREMENT_DIM] = np.asarray(state, dtype=float)
        self._ukf.P[:MEASUREMENT_DIM, :MEASUREMENT_DIM] = regularize_covariance(np.asarray(covariance, dtype=float))
        self._ukf.P = regularize_covariance(self._ukf.P)

        self._has_updated = False

    @property
    def state(self) -> np.ndarray:
        return self._ukf.x[:MEASUREMENT_DIM].copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._ukf.P[:MEASUREMENT_DIM, :MEASUREMENT_DIM].copy()

    @property
    def last_innovation(self) -> np.ndarray | None:
        
        return self._ukf.y.copy() if self._has_updated else None

    @property
    def last_innovation_covariance(self) -> np.ndarray | None:
       
        return self._ukf.S.copy() if self._has_updated else None
