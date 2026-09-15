"""Aligns a batch's tracklets to a common reference timestamp"""

from __future__ import annotations

from filtering.motion_models import constant_velocity_transition_matrix
from filtering.process_noise import constant_velocity_process_noise
from models.local_tracklet import LocalTracklet


def align_tracklet_to_timestamp(tracklet: LocalTracklet, reference_timestamp: float, process_noise_acceleration_std: float = 0.0,
    include_process_noise: bool = True) -> LocalTracklet:

    dt = reference_timestamp - tracklet.timestamp
    if dt == 0:
        return tracklet

    transition = constant_velocity_transition_matrix(dt)
    aligned_state = transition @ tracklet.state
    aligned_covariance = transition @ tracklet.covariance @ transition.T

    if include_process_noise and process_noise_acceleration_std > 0:
        aligned_covariance = aligned_covariance + constant_velocity_process_noise(
            abs(dt), process_noise_acceleration_std
        )

    return tracklet.model_copy(
        update={"state": aligned_state, "covariance": aligned_covariance, "timestamp": reference_timestamp}
    )


def align_batch_to_timestamp(tracklets: list[LocalTracklet], reference_timestamp: float, process_noise_acceleration_std: float = 0.0,
    include_process_noise: bool = True) -> list[LocalTracklet]:
    
    return [
        align_tracklet_to_timestamp(t, reference_timestamp, process_noise_acceleration_std, include_process_noise)
        for t in tracklets
    ]
