"""Ma segment builder."""

from __future__ import annotations

from dataclasses import dataclass

from experiment.literature_baselines._segment_builder_core import (
    SegmentBuilderOutput,
    SegmentTerminalEvent,
    SegmentTrack,
    max_expected_inter_message_gap_seconds,  # = network_period_s (max(1/frequency_hz))
)
from experiment.literature_baselines._segment_builder_batching import _process_global_window_batch
from experiment.replay import ReplayClock
from models.enums import AssociationMode
from models.local_tracklet import LocalTracklet
from network_simulator.scenario import ScenarioConfig
from synchronization.temporal_alignment import align_batch_to_timestamp
from synchronization.tracklet_buffer import TrackletBuffer
from config.models import AssociationConfig

# A numeric tolerance for "t == deadline is still eligible" - only guards
# against floating-point error, never used as a tuning margin.
_DEADLINE_EPSILON = 1e-9


def network_period_seconds(scenario: ScenarioConfig) -> float:
    
    return max_expected_inter_message_gap_seconds(scenario)


def support_horizon_seconds(scenario: ScenarioConfig, synchronization_window_ms: float) -> float:
    
    network_period_s = network_period_seconds(scenario)
    synchronization_window_s = synchronization_window_ms / 1000.0
    return max(network_period_s, synchronization_window_s)


@dataclass(frozen=True)
class MaSegmentBuilderConfig:
    support_horizon_s: float 
    support_horizon_mode: str = "rolling_max_network_period_and_sync_window"  
    synchronization_window_ms: float = 100.0
    association_mode: AssociationMode = AssociationMode.POSITION_ONLY 
    chi_square_probability: float = 0.9983  
    process_noise_acceleration_std: float = 2.0 


class MaSegmentBuilder:
    def __init__(self, config: MaSegmentBuilderConfig) -> None:
        self._config = config

    def build(self, tracklet_sequence: list[LocalTracklet]) -> SegmentBuilderOutput:
        
        association_config = AssociationConfig(synchronization_window_ms=self._config.synchronization_window_ms)
        clock = ReplayClock()
        buffer = TrackletBuffer(association_config, clock=clock)

        batches: list[tuple[float, list[LocalTracklet]]] = []

        def collect(batch: list[LocalTracklet]) -> None:
            epoch_timestamp = max(t.timestamp for t in batch)
            aligned = align_batch_to_timestamp(batch, epoch_timestamp, self._config.process_noise_acceleration_std)
            batches.append((epoch_timestamp, aligned))

        for tracklet in tracklet_sequence:
            clock.advance_to(tracklet.timestamp)
            buffer.add(tracklet)
            for batch in buffer.pop_ready_batches():
                collect(batch)

        for batch in buffer.flush_all_batches():
            collect(batch)


        horizon = self._config.support_horizon_s
        open_segments: dict[int, SegmentTrack] = {}
        deadlines: dict[int, float] = {}  # segment_id -> last_real_support_time + horizon
        next_segment_id = 0
        segment_history_rows = []
        terminal_events: list[SegmentTerminalEvent] = []

        for epoch_timestamp, aligned_batch in batches:

            expired_ids = [
                seg_id for seg_id in open_segments if epoch_timestamp > deadlines[seg_id] + _DEADLINE_EPSILON
            ]
            for seg_id in sorted(expired_ids):
                segment = open_segments[seg_id]
                terminal_events.append(
                    SegmentTerminalEvent(
                        segment_id=seg_id, birth_time=segment.birth_time,
                        last_detection_time=segment.last_detection_time,
                        close_time=deadlines[seg_id], close_reason="NO_MEASUREMENT_SUPPORT",
                    )
                )
            for seg_id in expired_ids:
                del open_segments[seg_id]
                del deadlines[seg_id]


            open_segments, new_rows, next_segment_id = _process_global_window_batch(
                aligned_batch, epoch_timestamp, open_segments, next_segment_id, self._config
            )
            segment_history_rows.extend(new_rows)

            for seg_id, segment in open_segments.items():
                if segment.last_detection_time == epoch_timestamp:
                    deadlines[seg_id] = epoch_timestamp + horizon


        for seg_id, segment in sorted(open_segments.items()):
            terminal_events.append(
                SegmentTerminalEvent(
                    segment_id=seg_id, birth_time=segment.birth_time,
                    last_detection_time=segment.last_detection_time,
                    close_time=segment.last_detection_time, close_reason="NO_MEASUREMENT_SUPPORT",
                )
            )

        return SegmentBuilderOutput(
            segment_history=tuple(segment_history_rows), terminal_events=tuple(terminal_events)
        )
