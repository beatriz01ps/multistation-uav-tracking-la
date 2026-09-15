"""Base interface for fusion strategies combining tracklets on the same GlobalTrack."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from models.enums import FusionStrategyName
from models.fused_measurement import FusedMeasurement
from models.local_tracklet import LocalTracklet


class TrackFusionStrategy(Protocol):
    def fuse(self, tracklets: list[LocalTracklet]) -> FusedMeasurement: ...

    def fuse_states(self, entries: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]: ...


def create_fusion_strategy(name: FusionStrategyName) -> TrackFusionStrategy:
    
    from fusion.covariance_intersection import CovarianceIntersectionFusion
    from fusion.information_fusion import InformationFusion
    from fusion.local_only import LocalOnlyFusion

    if name is FusionStrategyName.INFORMATION:
        return InformationFusion()
    if name is FusionStrategyName.COVARIANCE_INTERSECTION:
        return CovarianceIntersectionFusion()
    if name is FusionStrategyName.LOCAL_ONLY:
        return LocalOnlyFusion()
    raise ValueError(f"unknown fusion strategy: {name}")
