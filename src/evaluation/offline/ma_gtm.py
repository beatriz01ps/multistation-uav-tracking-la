"""Ma-inspired GTM: a literature-inspired baseline for fragment reconciliation,
adapted from Ma et al. (2026), IEEE Wireless Communications Letters (DOI 10.1109/LWC.2026.3700122)."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Union

import numpy as np

from association.gating import chi_square_threshold, dims_for_mode
from association.hungarian import REJECTED_COST, solve_assignment
from association.mahalanobis import mahalanobis_squared
from evaluation.offline.loaders import TrackHistoryRow, load_terminal_events, load_track_history
from models.enums import AssociationMode
from models.validation import regularize_for_inversion
from tracking.tracker import TrackTerminalEvent

PathLike = Union[str, Path]

STATE_DIM = 6  

@dataclass(frozen=True)
class MaGtmConfig:
    tau_g_seconds: float = 20.0              # PAPER_GENERALIZED (20 steps @ K0=1s -> 20 continuous seconds)
    gate_probability: float = 0.9            # PAPER_EXACT (the paper uses alpha=0.9)
    gate_position_only: bool = True          # PAPER_GENERALIZED (see the docstring: gamma_gate=4.61 indicates df=2 in the paper)
    process_noise_velocity_std: float = 1.0  # PROJECT_ADAPTATION - sigma_v (m/s^1.5), never re-tuned by looking at the result
    reconstruction_step_seconds: float = 1.0 # PROJECT_ADAPTATION - the reconstruction grid inside the gap (paper: K0=1s)

    # For census/reporting only (classify_fragment_status) - never used by
    # match_fragments (which decides eligibility pairwise). None -> uses the last timestamp observed across any fragment.
    reference_epoch: float | None = None

    def gate_mode(self) -> AssociationMode:
        return AssociationMode.POSITION_ONLY if self.gate_position_only else AssociationMode.FULL_STATE


FragmentStatus = Literal["active", "pending", "dead"]

def classify_fragment_status(fragment: TrackFragment, reference_epoch: float, tau_g_seconds: float) -> FragmentStatus:

    """The paper's eq.4, computed from last_detection_time"""

    gap = reference_epoch - fragment.last_detection_time
    if gap <= 0:
        return "active"
    if gap <= tau_g_seconds:
        return "pending"
    return "dead"

@dataclass(frozen=True)
class TrackFragment:
    original_global_track_id: int
    birth_time: float
    last_detection_time: float
    timestamps: tuple[float, ...]
    states: tuple[np.ndarray, ...]       # each 6D: x,y,z,vx,vy,vz
    covariances: tuple[np.ndarray, ...]  # each 6x6
    terminal_reason: str | None          # "DELETED" | "MERGED" | None (never terminated within the run)

    @property
    def birth_state(self) -> np.ndarray:
        return self.states[0]

    @property
    def birth_covariance(self) -> np.ndarray:
        return self.covariances[0]

    @property
    def last_detection_state(self) -> np.ndarray:
        return self.states[-1]

    @property
    def last_detection_covariance(self) -> np.ndarray:
        return self.covariances[-1]

    @property
    def is_deleted(self) -> bool:
        return self.terminal_reason == "DELETED"


def build_fragments(rows: list[TrackHistoryRow], terminal_events: dict[int, TrackTerminalEvent]) -> dict[int, TrackFragment]:

    """Groups TrackHistoryRow by global_track_id. last_detection_time is the timestamp of the last row with prediction_only=False (the paper's eq.3)"""

    by_track: dict[int, list[TrackHistoryRow]] = {}
    for row in rows:
        by_track.setdefault(row.global_track_id, []).append(row)

    fragments: dict[int, TrackFragment] = {}
    for global_track_id, track_rows in by_track.items():
        track_rows.sort(key=lambda r: r.timestamp)
        measured_rows = [r for r in track_rows if not r.prediction_only]
        last_detection_row = measured_rows[-1] if measured_rows else track_rows[-1]
        terminal = terminal_events.get(global_track_id)
        fragments[global_track_id] = TrackFragment(
            original_global_track_id=global_track_id,
            birth_time=track_rows[0].timestamp,
            last_detection_time=last_detection_row.timestamp,
            timestamps=tuple(r.timestamp for r in track_rows),
            states=tuple(r.state for r in track_rows),
            covariances=tuple(r.covariance for r in track_rows),
            terminal_reason=terminal.event_type if terminal is not None else None,
        )
    return fragments


def _cv_transition_matrix(dt: float, n: int = 3) -> np.ndarray:
   
    eye = np.eye(n)
    zero = np.zeros((n, n))
    top = np.hstack([eye, dt * eye])
    bottom = np.hstack([zero, eye])
    return np.vstack([top, bottom])


def _cv_process_noise(dt: float, sigma_v: float, n: int = 3) -> np.ndarray:

    eye = np.eye(n)
    top = np.hstack([(dt**3 / 3.0) * eye, (dt**2 / 2.0) * eye])
    bottom = np.hstack([(dt**2 / 2.0) * eye, dt * eye])
    return (sigma_v**2) * np.vstack([top, bottom])


def predict_through_gap(state: np.ndarray, covariance: np.ndarray, dt: float, process_noise_velocity_std: float) -> tuple[np.ndarray, np.ndarray]:

    if dt < 0:
        raise ValueError(f"predict_through_gap: negative dt ({dt}) - temporal order must be guaranteed by the caller")
    f = _cv_transition_matrix(dt)
    q = _cv_process_noise(dt, process_noise_velocity_std)
    predicted_state = f @ state
    predicted_covariance = f @ covariance @ f.T + q
    return predicted_state, predicted_covariance


def predictive_mahalanobis_gate(predicted_state: np.ndarray, predicted_covariance: np.ndarray, candidate_state: np.ndarray,
    candidate_covariance: np.ndarray, config: MaGtmConfig) -> tuple[bool, float]:

    dims = dims_for_mode(config.gate_mode())
    d2 = mahalanobis_squared(predicted_state, predicted_covariance, candidate_state, candidate_covariance, dims)
    threshold = chi_square_threshold(config.gate_probability, degrees_of_freedom=len(dims))
    return d2 <= threshold, d2


@dataclass(frozen=True)
class CandidatePairAudit:
    """One audit trail row per candidate pair evaluated."""

    pending_fragment_id: int
    active_fragment_id: int
    temporal_gap_seconds: float
    predicted_state: list
    candidate_state: list
    mahalanobis_distance_squared: float
    gate_threshold: float
    gate_passed: bool
    rejection_reason: str | None
    assignment_cost: float | None
    accepted: bool
    resulting_label: int | None


@dataclass(frozen=True)
class GtmAssignmentResult:
    matches: list[tuple[int, int]]  # (pending_id, active_id)
    audit_trail: list[CandidatePairAudit]


def extract_pending_and_active(fragments: dict[int, TrackFragment], config: MaGtmConfig) -> tuple[list[TrackFragment], list[TrackFragment]]:
    all_fragments = list(fragments.values())
    return all_fragments, all_fragments


def match_fragments(pending: list[TrackFragment], active: list[TrackFragment], config: MaGtmConfig) -> GtmAssignmentResult:

    audit_trail: list[CandidatePairAudit] = []

    if not pending or not active:
        return GtmAssignmentResult(matches=[], audit_trail=audit_trail)

    @dataclass(frozen=True)
    class _PairEvaluation:
        passed: bool
        d2: float
        predicted_state: np.ndarray | None
        dt: float
        reason: str | None

    def evaluate_pair(pending_frag: TrackFragment, active_frag: TrackFragment) -> _PairEvaluation:

        if active_frag.original_global_track_id == pending_frag.original_global_track_id:
            return _PairEvaluation(False, float("inf"), None, float("nan"), "self")

        dt = active_frag.birth_time - pending_frag.last_detection_time
        if dt <= 0:
            return _PairEvaluation(False, float("inf"), None, dt, "non_positive_temporal_gap")
        if dt > config.tau_g_seconds:
            return _PairEvaluation(False, float("inf"), None, dt, "exceeds_tau_g")

        predicted_state, predicted_covariance = predict_through_gap(
            pending_frag.last_detection_state,
            pending_frag.last_detection_covariance,
            dt,
            config.process_noise_velocity_std,
        )
        passed, d2 = predictive_mahalanobis_gate(
            predicted_state, predicted_covariance, active_frag.birth_state, active_frag.birth_covariance, config
        )
        reason = None if passed else "mahalanobis_gate_rejected"
        return _PairEvaluation(passed, d2, predicted_state, dt, reason)

    dims = dims_for_mode(config.gate_mode())
    threshold = chi_square_threshold(config.gate_probability, degrees_of_freedom=len(dims))

    cost = np.full((len(pending), len(active)), REJECTED_COST)
    valid = np.zeros((len(pending), len(active)), dtype=bool)

    for i, pending_frag in enumerate(pending):
        for j, active_frag in enumerate(active):
            evaluation = evaluate_pair(pending_frag, active_frag)
            audit_trail.append(
                CandidatePairAudit(
                    pending_fragment_id=pending_frag.original_global_track_id,
                    active_fragment_id=active_frag.original_global_track_id,
                    temporal_gap_seconds=evaluation.dt,
                    predicted_state=list(evaluation.predicted_state) if evaluation.predicted_state is not None else [],
                    candidate_state=list(active_frag.birth_state),
                    mahalanobis_distance_squared=evaluation.d2,
                    gate_threshold=threshold,
                    gate_passed=evaluation.passed,
                    rejection_reason=evaluation.reason,
                    assignment_cost=evaluation.d2 if evaluation.passed else None,
                    accepted=False,  # filled in after the assignment
                    resulting_label=None,
                )
            )
            if evaluation.passed:
                cost[i, j] = evaluation.d2
                valid[i, j] = True

    matches, _, _ = solve_assignment(cost, valid)
    match_set = set(matches)

    result_matches = [
        (pending[i].original_global_track_id, active[j].original_global_track_id) for i, j in matches
    ]
    reconciliation = reconcile_labels(result_matches)

    accepted_audit: list[CandidatePairAudit] = []
    audit_idx = 0
    for i, _pending_frag in enumerate(pending):
        for j, active_frag in enumerate(active):
            entry = audit_trail[audit_idx]
            audit_idx += 1
            accepted = (i, j) in match_set

            reason = entry.rejection_reason
            if entry.gate_passed and not accepted:
                reason = "not_selected_by_assignment"
            resulting_label = reconciliation.get(active_frag.original_global_track_id) if accepted else None
            accepted_audit.append(
                replace(entry, accepted=accepted, resulting_label=resulting_label, rejection_reason=None if accepted else reason)
            )

    return GtmAssignmentResult(matches=result_matches, audit_trail=accepted_audit)


def reconcile_labels(matches: list[tuple[int, int]]) -> dict[int, int]:
    """matches: (pending_id, active_id) pairs already resolved by the 1-to-1
    assignment. Returns active_id -> final_label (the oldest fragment in the chain); unmatched fragments don't appear."""
    active_to_pending = {active_id: pending_id for pending_id, active_id in matches}

    def resolve(active_id: int) -> int:
        current = active_id
        seen = {current}
        while current in active_to_pending:
            current = active_to_pending[current]
            if current in seen:
                raise ValueError(f"cycle detected in the reconciliation chain involving global_track_id={current}")
            seen.add(current)
        return current

    return {active_id: resolve(active_id) for active_id in active_to_pending}


@dataclass(frozen=True)
class ReconstructedEpoch:
    timestamp: float
    state: np.ndarray


def retrodict_gap(state_prev: np.ndarray, t_prev: float, state_next: np.ndarray, t_next: float,
    process_noise_velocity_std: float, step_seconds: float) -> list[ReconstructedEpoch]:

    if t_next <= t_prev:
        raise ValueError(f"retrodict_gap: t_next ({t_next}) must be > t_prev ({t_prev})")

    reconstructed: list[ReconstructedEpoch] = []
    t = t_prev + step_seconds
    while t < t_next:
        dt1 = t - t_prev
        dt2 = t_next - t
        f1 = _cv_transition_matrix(dt1)
        f2 = _cv_transition_matrix(dt2)
        q1 = _cv_process_noise(dt1, process_noise_velocity_std)
        q2 = _cv_process_noise(dt2, process_noise_velocity_std)

        # eq.12: Kr = Q1 F2^T (F2 Q1 F2^T + Q2)^-1
        innovation_like = f2 @ q1 @ f2.T + q2
        innovation_like = regularize_for_inversion(innovation_like)
        kr = q1 @ f2.T @ np.linalg.inv(innovation_like)

        # eq.13: xk = (F1 - Kr F2 F1) x_prev + Kr x_next
        state_k = (f1 - kr @ f2 @ f1) @ state_prev + kr @ state_next
        reconstructed.append(ReconstructedEpoch(timestamp=t, state=state_k))
        t += step_seconds

    return reconstructed

@dataclass(frozen=True)
class MaGtmRunResult:
    num_fragments_total: int
    num_pending_eligible: int
    num_active_candidates: int
    num_candidate_pairs_evaluated: int
    num_pairs_accepted: int
    num_pairs_rejected_gate: int  
    num_reconciled_labels: int
    num_reconstructed_epochs: int
    audit_trail: list[CandidatePairAudit]
    reconciliation: dict[int, int]  
    reconstructed_by_pair: dict[tuple[int, int], list[ReconstructedEpoch]]

    reference_epoch: float
    num_fragments_active_at_reference: int
    num_fragments_pending_at_reference: int
    num_fragments_dead_at_reference: int

    rejection_reason_counts: dict[str, int]


def run_ma_gtm(track_history_path: PathLike, terminal_events_path: PathLike, config: MaGtmConfig | None = None) -> MaGtmRunResult:

    """Main entry point: reads track_history_path/terminal_events_path"""

    config = config or MaGtmConfig()

    rows = load_track_history(track_history_path)
    terminal_events = load_terminal_events(terminal_events_path)
    fragments = build_fragments(rows, terminal_events)

    pending, active = extract_pending_and_active(fragments, config)
    assignment = match_fragments(pending, active, config)
    reconciliation = reconcile_labels(assignment.matches)

    fragments_by_id = fragments
    reconstructed_by_pair: dict[tuple[int, int], list[ReconstructedEpoch]] = {}
    total_reconstructed = 0
    for pending_id, active_id in assignment.matches:
        pending_frag = fragments_by_id[pending_id]
        active_frag = fragments_by_id[active_id]
        epochs = retrodict_gap(
            pending_frag.last_detection_state,
            pending_frag.last_detection_time,
            active_frag.birth_state,
            active_frag.birth_time,
            config.process_noise_velocity_std,
            config.reconstruction_step_seconds,
        )
        reconstructed_by_pair[(pending_id, active_id)] = epochs
        total_reconstructed += len(epochs)

    num_rejected_gate = sum(1 for a in assignment.audit_trail if a.rejection_reason == "mahalanobis_gate_rejected")

    rejection_reason_counts: dict[str, int] = {}
    for entry in assignment.audit_trail:
        key = "accepted" if entry.accepted else (entry.rejection_reason or "accepted")
        rejection_reason_counts[key] = rejection_reason_counts.get(key, 0) + 1

    reference_epoch = config.reference_epoch
    if reference_epoch is None:
        reference_epoch = max((f.last_detection_time for f in fragments.values()), default=0.0)
    status_counts = {"active": 0, "pending": 0, "dead": 0}
    for fragment in fragments.values():
        status_counts[classify_fragment_status(fragment, reference_epoch, config.tau_g_seconds)] += 1

    return MaGtmRunResult(
        num_fragments_total=len(fragments),
        num_pending_eligible=len(pending),
        num_active_candidates=len(active),
        num_candidate_pairs_evaluated=len(assignment.audit_trail),
        num_pairs_accepted=len(assignment.matches),
        num_pairs_rejected_gate=num_rejected_gate,
        num_reconciled_labels=len(reconciliation),
        num_reconstructed_epochs=total_reconstructed,
        audit_trail=assignment.audit_trail,
        reconciliation=reconciliation,
        reconstructed_by_pair=reconstructed_by_pair,
        reference_epoch=reference_epoch,
        num_fragments_active_at_reference=status_counts["active"],
        num_fragments_pending_at_reference=status_counts["pending"],
        num_fragments_dead_at_reference=status_counts["dead"],
        rejection_reason_counts=rejection_reason_counts,
    )


def write_ma_gtm_outputs(result: MaGtmRunResult, output_dir: PathLike) -> None:

    """Writes 3 new artifacts (never overwrites the originals): ma_gtm_associations.jsonl
    (audit trail), ma_gtm_reconciliation.jsonl (active_id -> final label), ma_gtm_reconstruction.jsonl (reconstructed states)."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "ma_gtm_associations.jsonl").open("w", encoding="utf-8") as handle:
        for entry in result.audit_trail:
            record = dict(entry.__dict__)
            record["predicted_state"] = list(record["predicted_state"])
            record["candidate_state"] = list(record["candidate_state"])
            handle.write(json.dumps(record) + "\n")

    with (out / "ma_gtm_reconciliation.jsonl").open("w", encoding="utf-8") as handle:
        for active_id, final_label in result.reconciliation.items():
            handle.write(json.dumps({"active_global_track_id": active_id, "reconciled_label": final_label}) + "\n")

    with (out / "ma_gtm_reconstruction.jsonl").open("w", encoding="utf-8") as handle:
        for (pending_id, active_id), epochs in result.reconstructed_by_pair.items():
            for epoch in epochs:
                handle.write(
                    json.dumps(
                        {
                            "pending_global_track_id": pending_id,
                            "active_global_track_id": active_id,
                            "timestamp": epoch.timestamp,
                            "state": list(epoch.state),
                        }
                    )
                    + "\n"
                )
