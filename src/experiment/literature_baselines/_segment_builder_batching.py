"""Per-epoch prediction/association/fusion core for the Ma-inspired segment builder
(CV prediction is gating-only, never persisted). Unmatched segments are left untouched, the caller decides closure."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from association.gating import gate
from association.hungarian import build_cost_matrix, solve_assignment
from experiment.literature_baselines._segment_builder_core import SegmentHistoryRow, SegmentTrack
from filtering.motion_models import constant_velocity_transition_matrix
from filtering.process_noise import constant_velocity_process_noise
from fusion.information_fusion import InformationFusion
from models.enums import AssociationMode
from models.local_tracklet import LocalTracklet
from models.validation import regularize_for_inversion


def _predict_segment_forward(segment: SegmentTrack, epoch_timestamp: float, process_noise_acceleration_std: float):

    dt = epoch_timestamp - segment.last_detection_time
    transition = constant_velocity_transition_matrix(dt)
    predicted_state = transition @ segment.state
    predicted_covariance = (
        transition @ segment.covariance @ transition.T
        + constant_velocity_process_noise(dt, process_noise_acceleration_std)
    )
    return predicted_state, predicted_covariance


def _weighted_average(entries: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:

    if len(entries) == 1:
        return entries[0]
    info_matrix = np.zeros((6, 6))
    info_vector = np.zeros(6)
    for state, covariance in entries:
        precision = np.linalg.inv(regularize_for_inversion(covariance))
        info_matrix += precision
        info_vector += precision @ state
    fused_covariance = np.linalg.inv(regularize_for_inversion(info_matrix))
    fused_state = fused_covariance @ info_vector
    return fused_state, fused_covariance


def _gate_evaluator(config):

    def evaluate(track_state, track_cov, meas_state, meas_cov):
        return gate(track_state, track_cov, meas_state, meas_cov, config.association_mode, config.chi_square_probability)

    return evaluate


def _process_global_window_batch(tracklets: list[LocalTracklet], epoch_timestamp: float, open_segments: dict[int, SegmentTrack],
    next_segment_id: int, config) -> tuple[dict[int, SegmentTrack], list[SegmentHistoryRow], int]:

    """One epoch's prediction/association/fusion: segments not matched here are
    left untouched (never closed), the caller (ma_segment_builder.py) decides closure."""

    evaluate_pair = _gate_evaluator(config)

    predicted = {
        seg_id: _predict_segment_forward(seg, epoch_timestamp, config.process_noise_acceleration_std)
        for seg_id, seg in sorted(open_segments.items())
    }
    pool_ids = list(predicted.keys())

    matched_group_by_segment: dict[int, list[LocalTracklet]] = {}
    candidates: list[list[LocalTracklet]] = []

    for station_id in sorted({t.station_id for t in tracklets}):
        station_tracklets = [t for t in tracklets if t.station_id == station_id]

        pool_states = [predicted[i] for i in pool_ids]
        pool_states += [_weighted_average([(t.state, t.covariance) for t in group]) for group in candidates]

        tracklet_states = [(t.state, t.covariance) for t in station_tracklets]
        cost, valid = build_cost_matrix(pool_states, tracklet_states, evaluate_pair)
        matches, _, unmatched = solve_assignment(cost, valid)

        n_pool = len(pool_ids)
        for pool_idx, tracklet_idx in matches:
            tracklet = station_tracklets[tracklet_idx]
            if pool_idx < n_pool:
                seg_id = pool_ids[pool_idx]
                matched_group_by_segment.setdefault(seg_id, []).append(tracklet)
            else:
                candidates[pool_idx - n_pool].append(tracklet)

        for tracklet_idx in unmatched:
            candidates.append([station_tracklets[tracklet_idx]])

    fusion = InformationFusion()
    new_open_segments: dict[int, SegmentTrack] = dict(open_segments)  # unmatched survive as-is (never close here)
    new_rows: list[SegmentHistoryRow] = []

    for seg_id, segment in sorted(open_segments.items()):
        group = matched_group_by_segment.get(seg_id)
        if group is None:
            continue  # no support IN THIS batch - it can still get support in another batch of the SAME window; the closure decision is left to the boundary
        fused_state, fused_covariance = fusion.fuse_states([(t.state, t.covariance) for t in group])
        extended = SegmentTrack(
            segment_id=seg_id,
            birth_time=segment.birth_time,
            last_detection_time=epoch_timestamp,
            state=fused_state,
            covariance=fused_covariance,
            state_history=segment.state_history + ((epoch_timestamp, fused_state),),
            covariance_history=segment.covariance_history + ((epoch_timestamp, fused_covariance),),
            contributing_stations=segment.contributing_stations | frozenset(t.station_id for t in group),
            contributing_local_tracks=segment.contributing_local_tracks
            | frozenset((t.station_id, t.local_track_id) for t in group),
        )
        new_open_segments[seg_id] = extended
        new_rows.append(
            SegmentHistoryRow(
                seg_id, epoch_timestamp, fused_state, fused_covariance,
                extended.contributing_stations, extended.contributing_local_tracks,
            )
        )

    for group in candidates:
        fused_state, fused_covariance = fusion.fuse_states([(t.state, t.covariance) for t in group])
        contributing_stations = frozenset(t.station_id for t in group)
        contributing_local_tracks = frozenset((t.station_id, t.local_track_id) for t in group)
        new_segment = SegmentTrack(
            segment_id=next_segment_id,
            birth_time=epoch_timestamp,
            last_detection_time=epoch_timestamp,
            state=fused_state,
            covariance=fused_covariance,
            state_history=((epoch_timestamp, fused_state),),
            covariance_history=((epoch_timestamp, fused_covariance),),
            contributing_stations=contributing_stations,
            contributing_local_tracks=contributing_local_tracks,
        )
        new_open_segments[next_segment_id] = new_segment
        new_rows.append(
            SegmentHistoryRow(
                next_segment_id, epoch_timestamp, fused_state, fused_covariance,
                contributing_stations, contributing_local_tracks,
            )
        )
        next_segment_id += 1

    return new_open_segments, new_rows, next_segment_id
