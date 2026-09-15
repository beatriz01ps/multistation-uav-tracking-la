"""Detects and fuses duplicate GlobalTracks (two tracks representing the same target, since gating only compares track vs tracklet) lastone"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from association.gating import gate
from config.models import AssociationConfig, DuplicateMergerConfig
from fusion.base import TrackFusionStrategy
from models.enums import AssociationMode, TrackStatus
from models.global_track import GlobalTrack
from tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)

_DUPLICATE_CHECK_MODE = AssociationMode.FULL_STATE


@dataclass(frozen=True)
class MergeEvent:
    survivor_global_track_id: int
    merged_global_track_id: int
    d2: float
    reason: str  


class DuplicateTrackMerger:
    
    def __init__(self, association_config: AssociationConfig, duplicate_merger_config: DuplicateMergerConfig,
        fusion_strategy: TrackFusionStrategy) -> None:

        self._chi_square_probability = association_config.chi_square_probability
        self._fusion_strategy = fusion_strategy
        self._enabled = duplicate_merger_config.enabled

    def merge_duplicates(self, track_manager: TrackManager) -> list[MergeEvent]:

        if not self._enabled:
            return []

        events: list[MergeEvent] = []

        while True:
            pair = self._closest_duplicate_pair(track_manager.active_tracks())
            if pair is None:
                break

            track_a, track_b, d2, reason = pair
            if track_a.global_track_id < track_b.global_track_id:
                survivor, merged = track_a, track_b
            else:
                survivor, merged = track_b, track_a

            self._merge_into(track_manager, survivor, merged)
            events.append(MergeEvent(survivor.global_track_id, merged.global_track_id, d2, reason))
            logger.info(
                "GLOBAL_TRACK_MERGED survivor=%s merged=%s d2=%.2f reason=%s",
                survivor.global_track_id,
                merged.global_track_id,
                d2,
                reason,
            )

        return events

    def _closest_duplicate_pair(self, active: list[GlobalTrack]) -> Optional[tuple[GlobalTrack, GlobalTrack, float, str]]:
        best: Optional[tuple[GlobalTrack, GlobalTrack, float, str]] = None
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                track_a, track_b = active[i], active[j]
                passed, d2 = gate(
                    track_a.state,
                    track_a.covariance,
                    track_b.state,
                    track_b.covariance,
                    _DUPLICATE_CHECK_MODE,
                    self._chi_square_probability,
                )
                shares_local_id = self._shares_local_track_id(track_a, track_b)
                if passed and not shares_local_id and self._is_stale_covariance_pair(track_a, track_b):
                    passed = False
                if not (passed or shares_local_id):
                    continue

                if passed and shares_local_id:
                    reason = "gate+shared_local_track_id"
                elif passed:
                    reason = "gate"
                else:
                    reason = "shared_local_track_id"

                if best is None or d2 < best[2]:
                    best = (track_a, track_b, d2, reason)
        return best

    @staticmethod
    def _is_stale_covariance_pair(track_a: GlobalTrack, track_b: GlobalTrack) -> bool:

        """True if exactly one of the two is CONFIRMED and the other COASTING/LOST"""

        statuses = {track_a.status, track_b.status}
        return TrackStatus.CONFIRMED in statuses and bool(statuses & {TrackStatus.COASTING, TrackStatus.LOST})

    @staticmethod
    def _shares_local_track_id(track_a: GlobalTrack, track_b: GlobalTrack) -> bool:

        """True if the two tracks ever shared a (station_id, local_track_id)"""

        return bool(track_a.local_track_history & track_b.local_track_history)

    def _merge_into(self, track_manager: TrackManager, survivor: GlobalTrack, merged: GlobalTrack) -> None:

        fused_state, fused_covariance = self._fusion_strategy.fuse_states(
            [(survivor.state, survivor.covariance), (merged.state, merged.covariance)]
        )

        track_manager.filter_for(survivor.global_track_id).set_state(fused_state, fused_covariance)
        track_manager.sync_track_from_filter(survivor.global_track_id)

        for station_id, local_track_id in merged.associated_local_tracks.items():
            merged_timestamp = merged.associated_local_track_timestamps.get(station_id, float("-inf"))
            survivor_timestamp = survivor.associated_local_track_timestamps.get(station_id, float("-inf"))

            if merged_timestamp > survivor_timestamp:
                survivor.associated_local_tracks[station_id] = local_track_id
                survivor.associated_local_track_timestamps[station_id] = merged_timestamp
            elif station_id not in survivor.associated_local_tracks:
                survivor.associated_local_tracks[station_id] = local_track_id
                survivor.associated_local_track_timestamps[station_id] = merged_timestamp
        survivor.current_contributors = list(set(survivor.current_contributors) | set(merged.current_contributors))
        survivor.local_track_history = survivor.local_track_history | merged.local_track_history

        track_manager.remove(merged.global_track_id)
