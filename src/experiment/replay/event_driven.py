"""EventDrivenReplay"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Optional, Union

from config.models import AppConfig
from evaluation.offline.evaluate_run import evaluate_run
from models.local_tracklet import LocalTracklet
from tracking.tracker import Tracker

from experiment.replay.common import ReplayClock, ReplayDiagnostics, ReplaySink, load_frozen_tracklet_sequence

PathLike = Union[str, Path]

__all__ = [
    "ReplayClock",
    "ReplayDiagnostics",
    "load_frozen_tracklet_sequence",
    "run_replay",
    "run_replay_and_evaluate",
]


def run_replay(tracklet_sequence: list[LocalTracklet], config: AppConfig, tracks_dir: PathLike,
    *, responses_path: Optional[PathLike] = None) -> tuple[Tracker, ReplayDiagnostics]:

    """Drives a new, in-process Tracker over the frozen sequence in its given order, writing tracks_dir in the same layout as a real run."""

    clock = ReplayClock()
    tracker = Tracker(config, clock=clock)
    sink = ReplaySink(tracks_dir, responses_path=responses_path)

    try:
        for tracklet in tracklet_sequence:
            clock.advance_to(tracklet.timestamp)
            accepted = tracker.ingest(tracklet)
            sink.record_ingest(tracklet, accepted)
            events = tracker.tick()
            sink.flush_tick(tracker, events)

        # Final flush: closes any batch still open at sequence end, or its
        # tracklets are never evaluated - valid only offline (the online runtime, and PeriodicVirtualReplay, never do this).
        events = tracker.tick(flush_all=True)
        sink.flush_tick(tracker, events)
    finally:
        sink.close()
    return tracker, sink.diagnostics


def run_replay_and_evaluate(source_sim_dir: PathLike, config: AppConfig, output_dir: PathLike) -> tuple[dict, ReplayDiagnostics]:

    """Replays one already-executed (scenario, seed) run against one config: reuses
    ground_truth.jsonl/tracklet_origins.jsonl as-is, generates a fresh tracks/, and runs the same offline evaluator as the UDP battery."""

    source_sim_dir = Path(source_sim_dir)
    output_dir = Path(output_dir)
    sim_dir = output_dir / "sim"
    tracks_dir = output_dir / "tracks"
    sim_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(source_sim_dir / "ground_truth.jsonl", sim_dir / "ground_truth.jsonl")
    shutil.copy(source_sim_dir / "tracklet_origins.jsonl", sim_dir / "tracklet_origins.jsonl")

    sequence = load_frozen_tracklet_sequence(source_sim_dir / "sent_messages.jsonl")
    return _replay_evaluate_common(sequence, config, sim_dir, tracks_dir, output_dir)


def _replay_evaluate_common(sequence: list[LocalTracklet], config: AppConfig, sim_dir: Path, tracks_dir: Path, output_dir: Path) -> tuple[dict, ReplayDiagnostics]:

    responses_path = sim_dir / "responses.jsonl"
    _tracker, diagnostics = run_replay(sequence, config, tracks_dir, responses_path=responses_path)

    report = evaluate_run(sim_dir, tracks_dir)
    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(dataclasses.asdict(report), indent=2, default=str), encoding="utf-8")
    (output_dir / "replay_diagnostics.json").write_text(json.dumps(diagnostics.to_dict(), indent=2), encoding="utf-8")
    return dataclasses.asdict(report), diagnostics
