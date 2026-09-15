"""State transitions: TENTATIVE -> CONFIRMED -> COASTING -> LOST -> DELETED."""

from __future__ import annotations

from config.models import TrackingConfig
from models.enums import TrackStatus, UpdateKind
from models.global_track import GlobalTrack


def on_measurement(track: GlobalTrack, timestamp: float, config: TrackingConfig) -> None:

    track.hit_count += 1
    track.miss_count = 0
    track.last_measurement_timestamp = timestamp
    track.last_update_kind = UpdateKind.MEASUREMENT_UPDATED

    if track.status is TrackStatus.TENTATIVE:
        if track.hit_count >= config.tentative_confirmation_hits:
            track.status = TrackStatus.CONFIRMED
    elif track.status in (TrackStatus.COASTING, TrackStatus.LOST):
        track.status = TrackStatus.CONFIRMED  # reacquisition


"""
Called after the predict() of a cycle with no measurement associated with this track. Returns True if the track must be
removed (DELETED).
"""
def on_no_measurement(track: GlobalTrack, timestamp: float, config: TrackingConfig) -> bool:

    track.miss_count += 1
    track.last_update_kind = UpdateKind.PREDICTION_ONLY

    time_since_measurement = track.time_since_measurement(timestamp)

    if time_since_measurement >= config.deletion_timeout_seconds:
        track.status = TrackStatus.DELETED
        return True

    if track.status is TrackStatus.TENTATIVE:
        if time_since_measurement >= config.tentative_timeout_seconds:
            track.status = TrackStatus.DELETED
            return True
        return False

    if track.status is TrackStatus.CONFIRMED:
        track.status = TrackStatus.COASTING
        return False

    if track.status is TrackStatus.COASTING:
        if time_since_measurement >= config.coasting_timeout_seconds:
            track.status = TrackStatus.LOST
        return False

    if track.status is TrackStatus.LOST:
        if time_since_measurement >= config.coasting_timeout_seconds + config.lost_timeout_seconds:
            track.status = TrackStatus.DELETED
            return True
        return False

    return False  # already DELETED
