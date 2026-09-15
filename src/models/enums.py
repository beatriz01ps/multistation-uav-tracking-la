"""Enumerations shared across the tracking domain."""

from __future__ import annotations

from enum import Enum


class TrackStatus(str, Enum):
    TENTATIVE = "tentative"  # candidate target (not yet confirmed)
    CONFIRMED = "confirmed"  # confirmed target (receiving measurement)s
    COASTING = "coasting"    # no measurement this cycle (extrapolated by prediction)
    LOST = "lost"            # no measurement for long enough (coasting_timeout) to reduce confidence
    DELETED = "deleted"      # lost for long enough (deletion_timeout) to be removed


class UpdateKind(str, Enum):

    """Explicitly distinguishes a real-measurement update from a pure prediction"""

    MEASUREMENT_UPDATED = "measurement_updated"
    PREDICTION_ONLY = "prediction_only"


class LateMessagePolicy(str, Enum):
    DROP = "drop"
    LOG_ONLY = "log_only"


class AssociationMode(str, Enum):
    POSITION_ONLY = "position_only"
    FULL_STATE = "full_state"


class AssociationMetric(str, Enum):
    """Which cost/gate T2TA uses per (track, tracklet) pair, before Hungarian"""

    MAHALANOBIS = "mahalanobis"
    EUCLIDEAN = "euclidean"


class FusionStrategyName(str, Enum):
    INFORMATION = "information"
    COVARIANCE_INTERSECTION = "covariance_intersection"
    LOCAL_ONLY = "local_only"


class MotionModelName(str, Enum):
    CONSTANT_VELOCITY = "constant_velocity"
    CONSTANT_ACCELERATION = "constant_acceleration"
    COORDINATED_TURN = "coordinated_turn"


class FilterType(str, Enum):
    
    """UKF and EKF filter."""

    UKF = "ukf"
    EKF = "ekf"
