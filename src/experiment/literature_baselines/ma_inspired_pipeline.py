"""MA_INSPIRED: orchestrates ma_segment_builder.py + ma_gtm.py."""

from __future__ import annotations

import dataclasses
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union

import numpy as np

from evaluation.association_metrics import AssociationEvent
from evaluation.offline.evaluate_run import evaluate_run
from evaluation.offline.loaders import TrackHistoryRow
from evaluation.offline.ma_gtm import MaGtmConfig, MaGtmRunResult, build_fragments, run_ma_gtm, write_ma_gtm_outputs
from experiment.literature_baselines._segment_builder_core import (
    to_ma_gtm_input,
    write_segment_builder_outputs,
)
from experiment.literature_baselines.ma_segment_builder import (
    MaSegmentBuilder,
    MaSegmentBuilderConfig,
    support_horizon_seconds,
)
from experiment.replay import load_frozen_tracklet_sequence
from network_simulator.scenario import ScenarioConfig, load_scenario
from tracking.tracker import TrackTerminalEvent

PathLike = Union[str, Path]


# The baseline's official config, reuses each module's existing/tested
# defaults, no parameter re-tuned.


def _write_track_history(rows: list[TrackHistoryRow], path: Path) -> None:

    with path.open("w", encoding="utf-8") as handle:
        for r in rows:
            handle.write(
                json.dumps(
                    {
                        "global_track_id": r.global_track_id,
                        "timestamp": r.timestamp,
                        "state": {
                            "x": float(r.state[0]), "y": float(r.state[1]), "z": float(r.state[2]),
                            "vx": float(r.state[3]), "vy": float(r.state[4]), "vz": float(r.state[5]),
                        },
                        "covariance": np.asarray(r.covariance).tolist(),
                        "status": r.status,
                        "prediction_only": r.prediction_only,
                        "nis": r.nis,
                    }
                )
                + "\n"
            )


def _write_terminal_events(events: dict[int, TrackTerminalEvent], path: Path) -> None:

    with path.open("w", encoding="utf-8") as handle:
        for event in events.values():
            handle.write(
                json.dumps(
                    {
                        "global_track_id": event.global_track_id,
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                        "merged_into_global_track_id": event.merged_into_global_track_id,
                    }
                )
                + "\n"
            )


def _write_association_events(events: list[AssociationEvent], path: Path) -> None:

    with path.open("w", encoding="utf-8") as handle:
        for e in events:
            handle.write(
                json.dumps(
                    {
                        "station_id": e.station_id,
                        "local_track_id": e.local_track_id,
                        "global_track_id": e.global_track_id,
                        "timestamp": e.timestamp,
                    }
                )
                + "\n"
            )

def build_final_trajectory(builder_output, gtm_result: MaGtmRunResult, fragments: dict) -> tuple[list[TrackHistoryRow], dict[int, TrackTerminalEvent], list[AssociationEvent]]:

    """Resolves each segment_id to its final label via the GTM's reconciliation"""

    reconciliation = gtm_result.reconciliation  # active_id -> final label
    matches = list(gtm_result.reconstructed_by_pair.keys())  # accepted (pending_id, active_id)

    def resolve(seg_id: int) -> int:
        return reconciliation.get(seg_id, seg_id)

    rows_by_final: dict[int, list[TrackHistoryRow]] = {}
    association_events: list[AssociationEvent] = []

    for row in builder_output.segment_history:
        final_id = resolve(row.segment_id)
        rows_by_final.setdefault(final_id, []).append(
            TrackHistoryRow(
                global_track_id=final_id, timestamp=row.timestamp, state=row.state, covariance=row.covariance,
                status="CONFIRMED", prediction_only=False, nis=None,
            )
        )
        for station_id, local_track_id in row.contributing_local_tracks:
            association_events.append(
                AssociationEvent(
                    station_id=station_id, local_track_id=local_track_id,
                    global_track_id=final_id, timestamp=row.timestamp,
                )
            )

    for (pending_id, active_id), epochs in gtm_result.reconstructed_by_pair.items():
        pending_frag = fragments[pending_id]
        final_id = resolve(active_id)
        for epoch in epochs:
            rows_by_final.setdefault(final_id, []).append(
                TrackHistoryRow(
                    global_track_id=final_id, timestamp=epoch.timestamp, state=epoch.state,
                    # ma_gtm.py's retrodiction computes no covariance of its own -
                    # it uses the previous fragment's as a conservative proxy.
                    covariance=pending_frag.last_detection_covariance,
                    status="CONFIRMED", prediction_only=True, nis=None,
                )
            )

    final_rows: list[TrackHistoryRow] = []
    for rows in rows_by_final.values():
        final_rows.extend(sorted(rows, key=lambda r: r.timestamp))

    matched_pending_ids = {p for p, _a in matches}
    chain_end_ids = [seg_id for seg_id in fragments if seg_id not in matched_pending_ids]

    final_terminal_events: dict[int, TrackTerminalEvent] = {}
    _, original_terminal_events = to_ma_gtm_input(builder_output)
    for seg_id in chain_end_ids:
        final_id = resolve(seg_id)
        original = original_terminal_events.get(seg_id)
        if original is not None:
            final_terminal_events[final_id] = TrackTerminalEvent(
                global_track_id=final_id, timestamp=original.timestamp,
                event_type="DELETED", merged_into_global_track_id=None,
            )

    return final_rows, final_terminal_events, association_events



def _finish_pipeline(builder_output, gtm_config: MaGtmConfig, source_sim_dir: Path, output_dir: Path) -> dict:

    raw_segments_dir = output_dir / "raw_segments"
    write_segment_builder_outputs(builder_output, raw_segments_dir)

    rows_ma, terminal_events_ma = to_ma_gtm_input(builder_output)
    gtm_dir = output_dir / "gtm"
    gtm_dir.mkdir(parents=True, exist_ok=True)
    ma_gtm_input_history_path = gtm_dir / "ma_gtm_input_track_history.jsonl"
    ma_gtm_input_terminal_path = gtm_dir / "ma_gtm_input_track_terminal_events.jsonl"
    _write_track_history(rows_ma, ma_gtm_input_history_path)
    _write_terminal_events(terminal_events_ma, ma_gtm_input_terminal_path)

    gtm_result = run_ma_gtm(ma_gtm_input_history_path, ma_gtm_input_terminal_path, gtm_config)
    write_ma_gtm_outputs(gtm_result, gtm_dir)

    fragments = build_fragments(rows_ma, terminal_events_ma)

    final_rows, final_terminal_events, association_events = build_final_trajectory(
        builder_output, gtm_result, fragments
    )

    final_dir = output_dir / "final"
    final_sim_dir = final_dir / "sim"
    final_tracks_dir = final_dir / "tracks"
    final_sim_dir.mkdir(parents=True, exist_ok=True)
    final_tracks_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(source_sim_dir / "ground_truth.jsonl", final_sim_dir / "ground_truth.jsonl")
    shutil.copy(source_sim_dir / "tracklet_origins.jsonl", final_sim_dir / "tracklet_origins.jsonl")
    _write_association_events(association_events, final_sim_dir / "responses.jsonl")
    _write_track_history(final_rows, final_tracks_dir / "track_history.jsonl")
    _write_terminal_events(final_terminal_events, final_tracks_dir / "track_terminal_events.jsonl")

    report = evaluate_run(final_sim_dir, final_tracks_dir)

    metrics_path = final_dir / "metrics.json"
    metrics_path.write_text(json.dumps(dataclasses.asdict(report), indent=2, default=str), encoding="utf-8")

    return {
        "number_of_segments": len(fragments),
        "number_of_gtm_candidates": gtm_result.num_candidate_pairs_evaluated,
        "number_of_gtm_assignments": gtm_result.num_pairs_accepted,
        "number_of_reconciliations": gtm_result.num_reconciled_labels,
        "number_of_reconstructed_epochs": gtm_result.num_reconstructed_epochs,
    }


@dataclass(frozen=True)
class MaInspiredConfig:
    segment_builder: MaSegmentBuilderConfig
    gtm: MaGtmConfig = MaGtmConfig()

    @staticmethod
    def for_scenario(scenario_config: ScenarioConfig) -> "MaInspiredConfig":
        sync_window_ms = MaSegmentBuilderConfig.__dataclass_fields__["synchronization_window_ms"].default
        return MaInspiredConfig(
            segment_builder=MaSegmentBuilderConfig(
                support_horizon_s=support_horizon_seconds(scenario_config, sync_window_ms)
            ),
            gtm=MaGtmConfig(),
        )

    def to_dict(self) -> dict:
        return {"segment_builder": asdict(self.segment_builder), "gtm": asdict(self.gtm)}


def run_ma_inspired(source_sim_dir: PathLike, scenario_yaml_path: PathLike, output_dir: PathLike, config: MaInspiredConfig | None = None) -> dict:

    """Runs the complete pipeline"""

    source_sim_dir = Path(source_sim_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_config = load_scenario(scenario_yaml_path)
    if config is None:
        config = MaInspiredConfig.for_scenario(scenario_config)

    sent_messages_path = source_sim_dir / "sent_messages.jsonl"
    tracklet_sequence = load_frozen_tracklet_sequence(sent_messages_path)

    builder_output = MaSegmentBuilder(config.segment_builder).build(tracklet_sequence)

    return _finish_pipeline(builder_output, config.gtm, source_sim_dir, output_dir)
