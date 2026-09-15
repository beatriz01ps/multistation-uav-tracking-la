"""Fusion in information space (covariance-weighted average)"""

from __future__ import annotations

import numpy as np

from models.fused_measurement import FusedMeasurement
from models.local_tracklet import LocalTracklet
from models.validation import regularize_for_inversion


class InformationFusion:
    def fuse_states(self, entries: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
        
        if not entries:
            raise ValueError("fuse_states() needs at least one entry")

        if len(entries) == 1:
            return entries[0]

        info_matrix = np.zeros((6, 6))
        info_vector = np.zeros(6)

        for state, covariance in entries:
            covariance = regularize_for_inversion(covariance)
            precision = np.linalg.inv(covariance)
            info_matrix += precision
            info_vector += precision @ state

        fused_covariance = np.linalg.inv(regularize_for_inversion(info_matrix))
        fused_state = fused_covariance @ info_vector
        return fused_state, fused_covariance

    def fuse(self, tracklets: list[LocalTracklet]) -> FusedMeasurement:

        if not tracklets:
            raise ValueError("fuse() needs at least one tracklet")

        fused_state, fused_covariance = self.fuse_states([(t.state, t.covariance) for t in tracklets])

        return FusedMeasurement(
            state=fused_state,
            covariance=fused_covariance,
            timestamp=max(t.timestamp for t in tracklets),
            contributing_tracklets=[(t.station_id, t.local_track_id) for t in tracklets],
        )
