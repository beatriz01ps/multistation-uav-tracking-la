"""Motion models"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from filtering.process_noise import constant_velocity_process_noise
from models.enums import MotionModelName

STATE_DIM = 6  # dimension of the EXTERNAL contract - never changes, even with an augmented internal state


class MotionModel(Protocol):
    state_dim: int

    def step(self, state: np.ndarray, dt: float) -> np.ndarray: ...

    def process_noise(self, dt: float, acceleration_std: float) -> np.ndarray: ...

    def augment_initial_state(
        self, state: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def jacobian(self, state: np.ndarray, dt: float) -> np.ndarray:

        ...


def constant_velocity_transition_matrix(dt: float) -> np.ndarray:

    transition = np.eye(STATE_DIM)
    for axis in range(3):
        transition[axis, axis + 3] = dt
    return transition


class ConstantVelocityModel:
    """x(k+1) = x(k) + vx(k)*dt (same for y, z). Constant Velocity."""

    state_dim = STATE_DIM

    def step(self, state: np.ndarray, dt: float) -> np.ndarray:
        return constant_velocity_transition_matrix(dt) @ state

    def process_noise(self, dt: float, acceleration_std: float) -> np.ndarray:
        return constant_velocity_process_noise(dt, acceleration_std)

    def augment_initial_state(
        self, state: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return state, covariance  # no extra dimension

    def jacobian(self, state: np.ndarray, dt: float) -> np.ndarray:
        return constant_velocity_transition_matrix(dt)  # linear model: Jacobian = F itself


_OMEGA_EPSILON = 1e-6


def coordinated_turn_step(state: np.ndarray, dt: float) -> np.ndarray:

    """Classic Coordinated Turn (Bar-Shalom**, Li & Kirubarajan, sec. 11.7)"""

    x, y, z, vx, vy, vz, omega = state

    if abs(omega) < _OMEGA_EPSILON:
        new_x = x + vx * dt
        new_y = y + vy * dt
        new_vx = vx
        new_vy = vy
    else:
        sin_wt = np.sin(omega * dt)
        cos_wt = np.cos(omega * dt)
        new_x = x + (sin_wt / omega) * vx - ((1.0 - cos_wt) / omega) * vy
        new_y = y + ((1.0 - cos_wt) / omega) * vx + (sin_wt / omega) * vy
        new_vx = cos_wt * vx - sin_wt * vy
        new_vy = sin_wt * vx + cos_wt * vy

    return np.array([new_x, new_y, z + vz * dt, new_vx, new_vy, vz, omega])


def coordinated_turn_jacobian(state: np.ndarray, dt: float) -> np.ndarray:

    """Analytic Jacobian of coordinated_turn_step (only EKF)"""

    _, _, _, vx, vy, _, omega = state
    jacobian = np.eye(STATE_DIM + 1)
    jacobian[2, 5] = dt 

    if abs(omega) < _OMEGA_EPSILON:

        a, da_dw = dt, 0.0
        b, db_dw = 0.0, dt**2 / 2.0
        c, s = 1.0, 0.0  # cos(0), sin(0)
    else:
        phi = omega * dt
        s, c = np.sin(phi), np.cos(phi)
        a = s / omega
        b = (1.0 - c) / omega
        da_dw = (dt * c * omega - s) / omega**2
        db_dw = (dt * s * omega - (1.0 - c)) / omega**2

    # x' = x + a*vx - b*vy
    jacobian[0, 3] = a
    jacobian[0, 4] = -b
    jacobian[0, 6] = da_dw * vx - db_dw * vy

    # y' = y + b*vx + a*vy
    jacobian[1, 3] = b
    jacobian[1, 4] = a
    jacobian[1, 6] = db_dw * vx + da_dw * vy

    # vx' = c*vx - s*vy
    jacobian[3, 3] = c
    jacobian[3, 4] = -s
    jacobian[3, 6] = -dt * (s * vx + c * vy)

    # vy' = s*vx + c*vy
    jacobian[4, 3] = s
    jacobian[4, 4] = c
    jacobian[4, 6] = dt * (c * vx - s * vy)

    # vz' = vz, omega' = omega: already covered by the initial identity

    return jacobian


def coordinated_turn_process_noise(dt: float, acceleration_std: float, turn_rate_process_noise_std: float) -> np.ndarray:

    """Reuses the same CV kinematic noise block"""
    q = np.zeros((STATE_DIM + 1, STATE_DIM + 1))
    q[:STATE_DIM, :STATE_DIM] = constant_velocity_process_noise(dt, acceleration_std)
    q[STATE_DIM, STATE_DIM] = (turn_rate_process_noise_std**2) * dt
    return q


class CoordinatedTurnModel:

    """Augmented state [x,y,z,vx,vy,vz,omega"""

    state_dim = STATE_DIM + 1

    def __init__(self, turn_rate_process_noise_std: float, initial_turn_rate_std: float) -> None:

        self._turn_rate_process_noise_std = turn_rate_process_noise_std
        self._initial_turn_rate_variance = initial_turn_rate_std**2

    def step(self, state: np.ndarray, dt: float) -> np.ndarray:

        return coordinated_turn_step(state, dt)

    def process_noise(self, dt: float, acceleration_std: float) -> np.ndarray:

        return coordinated_turn_process_noise(dt, acceleration_std, self._turn_rate_process_noise_std)

    def augment_initial_state(self, state: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        
        augmented_state = np.concatenate([state, [0.0]])
        augmented_covariance = np.zeros((self.state_dim, self.state_dim))
        augmented_covariance[:STATE_DIM, :STATE_DIM] = covariance
        augmented_covariance[STATE_DIM, STATE_DIM] = self._initial_turn_rate_variance
        return augmented_state, augmented_covariance

    def jacobian(self, state: np.ndarray, dt: float) -> np.ndarray:

        return coordinated_turn_jacobian(state, dt)


def create_motion_model(name: MotionModelName, *, turn_rate_process_noise_std: float = 0.05, initial_turn_rate_std: float = 0.3) -> MotionModel:

    if name == MotionModelName.CONSTANT_VELOCITY:
        return ConstantVelocityModel()
    if name == MotionModelName.COORDINATED_TURN:
        return CoordinatedTurnModel(turn_rate_process_noise_std, initial_turn_rate_std)
    raise NotImplementedError(
        f"motion model '{name.value}' is a future extension anticipated by the architecture "
        "(see filtering/motion_models.py) but not yet implemented in this version."
    )
