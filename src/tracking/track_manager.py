"""Storage and low-level operations over active GlobalTracks and their filters"""

from __future__ import annotations

from filtering.base import FilterTracker
from models.enums import TrackStatus
from models.global_track import GlobalTrack


class TrackManager:
    def __init__(self) -> None:
        self._tracks: dict[int, GlobalTrack] = {}
        self._filters: dict[int, FilterTracker] = {}

    @property
    def tracks(self) -> dict[int, GlobalTrack]:
        return self._tracks

    def active_tracks(self) -> list[GlobalTrack]:

        return [t for t in self._tracks.values() if t.status is not TrackStatus.DELETED]

    def filter_for(self, global_track_id: int) -> FilterTracker:

        return self._filters[global_track_id]

    def add(self, track: GlobalTrack, tracker: FilterTracker) -> None:

        self._tracks[track.global_track_id] = track
        self._filters[track.global_track_id] = tracker

    def remove(self, global_track_id: int) -> None:
        
        del self._tracks[global_track_id]
        del self._filters[global_track_id]

    def sync_track_from_filter(self, global_track_id: int) -> None:

        track = self._tracks[global_track_id]
        tracker = self._filters[global_track_id]
        track.state = tracker.state
        track.covariance = tracker.covariance

    def sync_innovation_from_filter(self, global_track_id: int) -> None:

        track = self._tracks[global_track_id]
        tracker = self._filters[global_track_id]
        track.last_innovation = tracker.last_innovation
        track.last_innovation_covariance = tracker.last_innovation_covariance
