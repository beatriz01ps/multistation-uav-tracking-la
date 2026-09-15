"""Ground truth trajectory for the message generato"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from filtering.motion_models import constant_velocity_transition_matrix, coordinated_turn_step


def true_state_at(initial_state, t: float) -> np.ndarray:
    
    initial = np.asarray(initial_state, dtype=float)
    return constant_velocity_transition_matrix(t) @ initial


@dataclass(frozen=True)
class TurnSegment:

    duration_s: float
    omega: float


def true_state_at_turning(initial_state, schedule: list, t: float) -> np.ndarray:

    state = np.asarray(initial_state, dtype=float).copy()
    remaining = t

    for segment in schedule:
        if remaining <= 0:
            return state
        segment_dt = min(segment.duration_s, remaining)
        state = coordinated_turn_step(np.concatenate([state, [segment.omega]]), segment_dt)[:6]
        remaining -= segment_dt

    if remaining > 0:
        # after the last declared segment, continues straight (omega=0).
        state = coordinated_turn_step(np.concatenate([state, [0.0]]), remaining)[:6]

    return state


def true_state_for_uav(uav, t: float) -> np.ndarray:

    if uav.turn_schedule:
        return true_state_at_turning(uav.initial_state, uav.turn_schedule, t)
    return true_state_at(uav.initial_state, t)
