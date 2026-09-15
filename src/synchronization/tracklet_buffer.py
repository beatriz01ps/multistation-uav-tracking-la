"""Temporal synchronization buffer"""

from __future__ import annotations

import logging
import threading
import time as time_module

from config.models import AssociationConfig
from models.enums import LateMessagePolicy
from models.local_tracklet import LocalTracklet

logger = logging.getLogger(__name__)


class _Cluster:
    __slots__ = ("opened_at", "tracklets")

    def __init__(self, opened_at: float, first_tracklet: LocalTracklet) -> None:
        self.opened_at = opened_at
        self.tracklets: list[LocalTracklet] = [first_tracklet]

    @property
    def reference_timestamp(self) -> float:
        return self.tracklets[0].timestamp

    @property
    def processing_timestamp(self) -> float:

       return max(t.timestamp for t in self.tracklets)

    def accepts(self, tracklet: LocalTracklet, window_s: float) -> bool:

        within_window = abs(tracklet.timestamp - self.reference_timestamp) <= window_s
        key = (tracklet.station_id, tracklet.local_track_id)
        key_already_present = any((t.station_id, t.local_track_id) == key for t in self.tracklets)
        return within_window and not key_already_present


class TrackletBuffer:
    def __init__(self, config: AssociationConfig, clock=time_module.monotonic) -> None:

        self._config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._clusters: list[_Cluster] = []
        self._last_released_timestamp: float = float("-inf")

    def add(self, tracklet: LocalTracklet) -> bool:

        """Returns True if accepted into the buffer, False if dropped"""

        window_s = self._config.synchronization_window_ms / 1000.0

        with self._lock:
            if tracklet.timestamp < self._last_released_timestamp:
                if self._config.late_message_policy is LateMessagePolicy.LOG_ONLY:
                    logger.warning(
                        "late tracklet dropped (policy=log_only, diagnostic): "
                        "station=%s local=%s timestamp=%.3f < last_released=%.3f",
                        tracklet.station_id,
                        tracklet.local_track_id,
                        tracklet.timestamp,
                        self._last_released_timestamp,
                    )
                else:
                    logger.warning(
                        "late tracklet dropped (policy=drop): station=%s local=%s timestamp=%.3f < last_released=%.3f",
                        tracklet.station_id,
                        tracklet.local_track_id,
                        tracklet.timestamp,
                        self._last_released_timestamp,
                    )
                return False

            now = self._clock()

            candidates = [c for c in self._clusters if c.accepts(tracklet, window_s)]
            if candidates:

                nearest = min(
                    candidates,
                    key=lambda c: (abs(tracklet.timestamp - c.reference_timestamp), c.reference_timestamp),
                )
                nearest.tracklets.append(tracklet)
                return True

            self._clusters.append(_Cluster(now, tracklet))
            return True

    def pop_ready_batches(self) -> list[list[LocalTracklet]]:

        window_s = self._config.synchronization_window_ms / 1000.0

        with self._lock:
            now = self._clock()
            ordered = sorted(self._clusters, key=lambda c: c.processing_timestamp)
            ready: list[_Cluster] = []
            for cluster in ordered:
                if now - cluster.opened_at >= window_s:
                    ready.append(cluster)
                else:
                    break  

            if not ready:
                return []

            ready_ids = {id(c) for c in ready}
            self._clusters = [c for c in self._clusters if id(c) not in ready_ids]

        return self._finalize(ready)

    def flush_all_batches(self) -> list[list[LocalTracklet]]:

        """Like pop_ready_batches"""

        with self._lock:
            ready = self._clusters
            self._clusters = []

        return self._finalize(ready)

    def _finalize(self, clusters: list[_Cluster]) -> list[list[LocalTracklet]]:

        clusters.sort(key=lambda c: c.processing_timestamp)
        batches = [sorted(c.tracklets, key=lambda t: t.timestamp) for c in clusters]

        if batches:
            newest = max(t.timestamp for batch in batches for t in batch)
            with self._lock:
                self._last_released_timestamp = max(self._last_released_timestamp, newest)

        return batches
