"""Hungarian T2TA association with configurable pair metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from association.gating import euclidean_gate, gate
from association.hungarian import PairEvaluator, build_cost_matrix, solve_assignment
from config.models import AssociationConfig
from models.enums import AssociationMetric, TrackStatus
from models.global_track import GlobalTrack
from models.local_tracklet import LocalTracklet
from models.validation import regularize_for_inversion


@dataclass
class AssociationResult:
    # Global track index -> local tracklet indices.
    matched_tracklets_by_track: dict[int, list[int]] = field(default_factory=dict)

    # Each group becomes one new GlobalTrack.
    new_track_candidates: list[list[int]] = field(default_factory=list)


def _weighted_average(estimates: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:

    """Build a temporary information-weighted candidate state."""

    if len(estimates) == 1:
        return estimates[0]
    info_matrix = np.zeros((6, 6))
    info_vector = np.zeros(6)
    for state, covariance in estimates:
        precision = np.linalg.inv(regularize_for_inversion(covariance))
        info_matrix += precision
        info_vector += precision @ state
    fused_covariance = np.linalg.inv(regularize_for_inversion(info_matrix))
    fused_state = fused_covariance @ info_vector
    return fused_state, fused_covariance


def _build_evaluator(config: AssociationConfig) -> PairEvaluator:

    """Build the pair evaluator for the configured metric."""

    if config.metric is AssociationMetric.EUCLIDEAN:

        def evaluate(track_state, track_cov, meas_state, meas_cov):
            return euclidean_gate(track_state, meas_state, config.euclidean_gate_distance)

        return evaluate

    def evaluate(track_state, track_cov, meas_state, meas_cov):
        return gate(track_state, track_cov, meas_state, meas_cov, config.mode, config.chi_square_probability)

    return evaluate


class HungarianAssociation:

    """Hungarian T2TA association with active/lost priority and candidate grouping."""

    def __init__(self, config: AssociationConfig) -> None:

        self._config = config
        self._evaluate_pair = _build_evaluator(config)

    def associate( self, global_tracks: list[GlobalTrack], local_tracklets: list[LocalTracklet], timestamp: float) -> AssociationResult:
        by_station: dict[str, list[int]] = {}
        for idx, tracklet in enumerate(local_tracklets):
            by_station.setdefault(tracklet.station_id, []).append(idx)

        priority_positions = [i for i, t in enumerate(global_tracks) if t.status is not TrackStatus.LOST]
        lost_positions = [i for i, t in enumerate(global_tracks) if t.status is TrackStatus.LOST]

        result = AssociationResult()
        candidates: list[list[int]] = []

        for station_id in sorted(by_station.keys()):
            tracklet_indices = by_station[station_id]

            stage1_matches, remaining = self._match_against_pool(
                global_tracks, priority_positions, candidates, local_tracklets, tracklet_indices
            )
            stage2_matches: list[tuple[str, int, int]] = []
            if remaining and lost_positions:
                stage2_matches, remaining = self._match_against_pool(
                    global_tracks, lost_positions, [], local_tracklets, remaining
                )

            for kind, ref, tracklet_idx in stage1_matches + stage2_matches:
                if kind == "track":
                    result.matched_tracklets_by_track.setdefault(ref, []).append(tracklet_idx)
                else:
                    candidates[ref].append(tracklet_idx)

            for tracklet_idx in remaining:
                candidates.append([tracklet_idx])

        result.new_track_candidates = candidates
        return result

    def _match_against_pool(self, global_tracks: list[GlobalTrack], track_positions: list[int], candidates: list[list[int]],
        local_tracklets: list[LocalTracklet], tracklet_indices: list[int]) -> tuple[list[tuple[str, int, int]], list[int]]:

        pool_states = [(global_tracks[p].state, global_tracks[p].covariance) for p in track_positions]
        pool_states += [
            _weighted_average([(local_tracklets[i].state, local_tracklets[i].covariance) for i in group])
            for group in candidates
        ]

        tracklet_states = [(local_tracklets[i].state, local_tracklets[i].covariance) for i in tracklet_indices]

        cost, valid = build_cost_matrix(pool_states, tracklet_states, self._evaluate_pair)
        raw_matches, _, unmatched_local = solve_assignment(cost, valid)

        n_tracks = len(track_positions)
        matches: list[tuple[str, int, int]] = []
        for pool_idx, tracklet_local_idx in raw_matches:
            tracklet_idx = tracklet_indices[tracklet_local_idx]
            if pool_idx < n_tracks:
                matches.append(("track", track_positions[pool_idx], tracklet_idx))
            else:
                matches.append(("candidate", pool_idx - n_tracks, tracklet_idx))

        still_unmatched = [tracklet_indices[i] for i in unmatched_local]
        return matches, still_unmatched