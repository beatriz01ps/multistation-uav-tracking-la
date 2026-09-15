"""Covariance Intersection (CI)"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from models.fused_measurement import FusedMeasurement
from models.local_tracklet import LocalTracklet
from models.validation import regularize_for_inversion


def _pairwise_ci(x1: np.ndarray, p1: np.ndarray, x2: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    p1_inv = np.linalg.inv(regularize_for_inversion(p1))
    p2_inv = np.linalg.inv(regularize_for_inversion(p2))

    def fused_trace(omega: float) -> float:
        p_inv = omega * p1_inv + (1 - omega) * p2_inv
        return float(np.trace(np.linalg.inv(p_inv)))

    result = minimize_scalar(fused_trace, bounds=(1e-6, 1 - 1e-6), method="bounded")
    omega = float(result.x)

    p_fused_inv = omega * p1_inv + (1 - omega) * p2_inv
    p_fused = np.linalg.inv(p_fused_inv)
    x_fused = p_fused @ (omega * p1_inv @ x1 + (1 - omega) * p2_inv @ x2)
    return x_fused, p_fused


class CovarianceIntersectionFusion:
    """Chains CI pairwise (A,B -> AB; AB,C -> ABC)"""

    def fuse_states(self, entries: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:

        if not entries:
            raise ValueError("fuse_states() needs at least one entry")

        x, p = entries[0]
        for state, covariance in entries[1:]:
            x, p = _pairwise_ci(x, p, state, covariance)
        return x, p

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
