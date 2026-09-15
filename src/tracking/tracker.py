"""Main orchestrator: connects the synchronization buffer, association, fusion, UKF filter, and lifecycle."""

from __future__ import annotations

import logging
import math
import time as time_module
from dataclasses import dataclass

from association.association_manager import HungarianAssociation
from association.gating import dims_for_mode
from association.mahalanobis import mahalanobis_squared
from config.models import AppConfig
from fusion.base import create_fusion_strategy
from models.enums import TrackStatus, UpdateKind
from models.global_track import GlobalTrack
from models.local_tracklet import LocalTracklet
from synchronization.temporal_alignment import align_batch_to_timestamp
from synchronization.tracklet_buffer import TrackletBuffer
from synchronization.tracking_clock import TrackingClock
from tracking import lifecycle
from tracking.duplicate_merger import DuplicateTrackMerger
from tracking.track_initializer import TrackInitializer
from tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackUpdateEvent:
    """A local == global association from a cycle."""

    station_id: str
    local_track_id: str
    global_track_id: int
    timestamp: float
    update_kind: UpdateKind = UpdateKind.MEASUREMENT_UPDATED


@dataclass(frozen=True)
class TrackTerminalEvent:

    """The explicit reason a global_track_id stops existing -> DELETED"""

    global_track_id: int
    timestamp: float
    event_type: str  # "DELETED" | "MERGED"
    merged_into_global_track_id: int | None = None


@dataclass(frozen=True)
class AssociationCandidateEvent:
  
    global_track_id: int
    station_id: str
    local_track_id: str
    timestamp: float
    predicted_position: tuple[float, float, float]
    tracklet_position: tuple[float, float, float]
    euclidean_distance_m: float


class Tracker:
    def __init__(self, config: AppConfig, *, clock=None) -> None:
        self._config = config
        self.track_manager = TrackManager()
        self.terminal_events: list[TrackTerminalEvent] = []

        self.association_candidates: list[AssociationCandidateEvent] = []
        self._association = HungarianAssociation(config.association)
        self._fusion_strategy = create_fusion_strategy(config.fusion.strategy)
        self._initializer = TrackInitializer(config)

        duplicate_merge_fusion_strategy = create_fusion_strategy(config.duplicate_merger.fusion_strategy)
        self._duplicate_merger = DuplicateTrackMerger(
            config.association, config.duplicate_merger, duplicate_merge_fusion_strategy
        )

        monotonic = clock if clock is not None else time_module.monotonic
        self._buffer = TrackletBuffer(config.association, clock=monotonic)
        self._tracking_clock = TrackingClock(time_scale=config.time.time_scale, monotonic=monotonic)

    def ingest(self, tracklet: LocalTracklet) -> bool:
        
        return self._buffer.add(tracklet)

    def tick(self, *, flush_all: bool = False) -> list[TrackUpdateEvent]:

        batches = self._buffer.flush_all_batches() if flush_all else self._buffer.pop_ready_batches()

        events: list[TrackUpdateEvent] = []

        processed_this_tick: set[int] = set()
        for batch in batches:
            batch_timestamp = max(tracklet.timestamp for tracklet in batch)
            batch_events, processed_ids = self._process_batch_tracked(batch, batch_timestamp)
            events.extend(batch_events)
            processed_this_tick.update(processed_ids)

        if not flush_all and self._tracking_clock.is_anchored:

            self._catch_up(self._tracking_clock.now(), already_handled_this_tick=processed_this_tick)

        return events

    def _catch_up(self, timestamp: float, already_handled_this_tick: set[int]) -> None:

        active = self.track_manager.active_tracks()

        for track in active:
            dt = timestamp - track.last_prediction_timestamp
            if dt > 0:
                self.track_manager.filter_for(track.global_track_id).predict(dt)
                track.last_prediction_timestamp = timestamp
                self.track_manager.sync_track_from_filter(track.global_track_id)
            track.age += 1

        untouched = [t for t in active if t.global_track_id not in already_handled_this_tick]
        self._apply_misses(untouched, timestamp)

    def process_batch(self, tracklets: list[LocalTracklet], timestamp: float) -> list[TrackUpdateEvent]:

        events, _ = self._process_batch_tracked(tracklets, timestamp)
        return events

    def _process_batch_tracked(self, tracklets: list[LocalTracklet], timestamp: float) -> tuple[list[TrackUpdateEvent], set[int]]:

        active = self.track_manager.active_tracks()

        for track in active:
            dt = timestamp - track.last_prediction_timestamp
            if dt > 0:
                self.track_manager.filter_for(track.global_track_id).predict(dt)
                track.last_prediction_timestamp = timestamp
                self.track_manager.sync_track_from_filter(track.global_track_id)
            track.age += 1

        if not tracklets:
            self._apply_misses(active, timestamp)
            processed_ids = {t.global_track_id for t in active}
            return self._merge_duplicates_and_remap_events([], timestamp), processed_ids

        self._tracking_clock.anchor(timestamp)

        tracklets = align_batch_to_timestamp(
            tracklets, timestamp, self._config.filter.process_noise_acceleration_std
        )

        events: list[TrackUpdateEvent] = []
        result = self._association.associate(active, tracklets, timestamp)

        dims = dims_for_mode(self._config.association.mode)
        matched_track_ids: set[int] = set()
        for position, tracklet_indices in result.matched_tracklets_by_track.items():
            track = active[position]
            group = sorted((tracklets[i] for i in tracklet_indices), key=lambda t: t.station_id)

            for tracklet in group:
                d2 = mahalanobis_squared(track.state, track.covariance, tracklet.state, tracklet.covariance, dims)
                logger.info(
                    "LOCAL_TRACK_ASSOCIATED station=%s local=%s -> global=%s d2=%.2f",
                    tracklet.station_id,
                    tracklet.local_track_id,
                    track.global_track_id,
                    d2,
                )
                predicted_position = (float(track.state[0]), float(track.state[1]), float(track.state[2]))
                tracklet_position = (float(tracklet.state[0]), float(tracklet.state[1]), float(tracklet.state[2]))
                self.association_candidates.append(
                    AssociationCandidateEvent(
                        global_track_id=track.global_track_id,
                        station_id=tracklet.station_id,
                        local_track_id=tracklet.local_track_id,
                        timestamp=timestamp,
                        predicted_position=predicted_position,
                        tracklet_position=tracklet_position,
                        euclidean_distance_m=math.sqrt(sum(
                            (a - b) ** 2 for a, b in zip(predicted_position, tracklet_position)
                        )),
                    )
                )

            fused = self._fusion_strategy.fuse(group)
            self.track_manager.filter_for(track.global_track_id).update(fused.state, fused.covariance)
            self.track_manager.sync_track_from_filter(track.global_track_id)
            self.track_manager.sync_innovation_from_filter(track.global_track_id)

            previous_status = track.status
            lifecycle.on_measurement(track, timestamp, self._config.tracking)
            if track.status is TrackStatus.CONFIRMED and previous_status is not TrackStatus.CONFIRMED:
                event_name = "GLOBAL_TRACK_RECOVERED" if previous_status in (
                    TrackStatus.COASTING, TrackStatus.LOST
                ) else "GLOBAL_TRACK_CONFIRMED"
                logger.info("%s global=%s", event_name, track.global_track_id)

            track.current_contributors = [(t.station_id, t.local_track_id) for t in group]
            for tracklet in group:
                track.associated_local_tracks[tracklet.station_id] = tracklet.local_track_id
                track.associated_local_track_timestamps[tracklet.station_id] = tracklet.timestamp
                track.local_track_history.add((tracklet.station_id, tracklet.local_track_id))
                events.append(
                    TrackUpdateEvent(tracklet.station_id, tracklet.local_track_id, track.global_track_id, timestamp)
                )
            matched_track_ids.add(track.global_track_id)

        unmatched_active = [t for t in active if t.global_track_id not in matched_track_ids]
        self._apply_misses(unmatched_active, timestamp)

        processed_ids = {t.global_track_id for t in active}

        new_tracks = self._create_new_tracks(result.new_track_candidates, tracklets, timestamp)
        for track in new_tracks:
            for station_id, local_track_id in track.associated_local_tracks.items():
                events.append(TrackUpdateEvent(station_id, local_track_id, track.global_track_id, timestamp))

       
        processed_ids.update(t.global_track_id for t in new_tracks)

        return self._merge_duplicates_and_remap_events(events, timestamp), processed_ids

    def _merge_duplicates_and_remap_events(self, events: list[TrackUpdateEvent], timestamp: float) -> list[TrackUpdateEvent]:

        merge_events = self._duplicate_merger.merge_duplicates(self.track_manager)
        for merge_event in merge_events:
            self.terminal_events.append(
                TrackTerminalEvent(
                    merge_event.merged_global_track_id, timestamp, "MERGED", merge_event.survivor_global_track_id
                )
            )
        if not merge_events:
            return events

        resolution: dict[int, int] = {}
        for merge_event in merge_events:
            resolution[merge_event.merged_global_track_id] = merge_event.survivor_global_track_id

        def resolve(global_track_id: int) -> int:
            while global_track_id in resolution:
                global_track_id = resolution[global_track_id]
            return global_track_id

        return [
            event
            if resolve(event.global_track_id) == event.global_track_id
            else TrackUpdateEvent(
                event.station_id, event.local_track_id, resolve(event.global_track_id), event.timestamp, event.update_kind
            )
            for event in events
        ]

    def _create_new_tracks(self, candidate_groups: list[list[int]], tracklets: list[LocalTracklet], timestamp: float) -> list[GlobalTrack]:

        created = []
        for group_indices in candidate_groups:
            group = [tracklets[i] for i in group_indices]
            track, tracker = self._initializer.create_track(group, timestamp)
            self.track_manager.add(track, tracker)
            created.append(track)
            logger.info(
                "GLOBAL_TRACK_CREATED global=%s stations=%s",
                track.global_track_id,
                [t.station_id for t in group],
            )
        return created

    def _apply_misses(self, tracks: list[GlobalTrack], timestamp: float) -> None:
        
        to_delete = []
        for track in tracks:
            track.current_contributors = [] 
            track.last_innovation_covariance = None
            previous_status = track.status
            deleted = lifecycle.on_no_measurement(track, timestamp, self._config.tracking)
            if track.status != previous_status:
                logger.info("GLOBAL_TRACK_%s global=%s", track.status.value.upper(), track.global_track_id)
            if deleted:
                to_delete.append(track.global_track_id)
        for global_track_id in to_delete:
            self.terminal_events.append(TrackTerminalEvent(global_track_id, timestamp, "DELETED"))
            self.track_manager.remove(global_track_id)
