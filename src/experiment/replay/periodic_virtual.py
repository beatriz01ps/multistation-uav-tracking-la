"""PeriodicVirtualReplay"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from config.models import AppConfig
from io_.receiver import DEFAULT_PROCESS_INTERVAL_S
from models.local_tracklet import LocalTracklet
from tracking.tracker import Tracker

from experiment.replay.common import ReplayClock, ReplayDiagnostics, ReplaySink

PathLike = Union[str, Path]

__all__ = ["PeriodicVirtualReplayConfig", "run_periodic_virtual_replay"]


@dataclass(frozen=True)
class PeriodicVirtualReplayConfig:

    """Explicit temporal config"""

    run_until_s: float
    tick_interval_s: float = DEFAULT_PROCESS_INTERVAL_S
    tick_anchor_s: float = 0.0
    tick_interval_source: str = "online_runtime_process_interval (io_.receiver.DEFAULT_PROCESS_INTERVAL_S)"

    def __post_init__(self) -> None:
        if self.tick_interval_s <= 0:
            raise ValueError("tick_interval_s must be positive")
        if self.run_until_s < self.tick_anchor_s:
            raise ValueError("run_until_s cannot be earlier than tick_anchor_s")


def _tick_schedule(config: PeriodicVirtualReplayConfig) -> list[float]:

    """t0, t0+dt, t0+2dt, ... up to <= run_until_s - purely arithmetic, never time.sleep."""

    n_ticks = math.floor((config.run_until_s - config.tick_anchor_s) / config.tick_interval_s + 1e-9) + 1
    return [config.tick_anchor_s + k * config.tick_interval_s for k in range(n_ticks)]


def run_periodic_virtual_replay(tracklet_sequence: list[LocalTracklet], config: AppConfig, tracks_dir: PathLike,
    replay_config: PeriodicVirtualReplayConfig, *, responses_path: Optional[PathLike] = None) -> tuple[Tracker, ReplayDiagnostics]:

    """Drives a Tracker exactly like event_driven.run_replay, except tick() fires at
    a fixed cadence rather than only on message arrival. Messages are ingested in their exact given order."""

    tick_times = _tick_schedule(replay_config)

    last_message_ts = max((t.timestamp for t in tracklet_sequence), default=None)
    if last_message_ts is not None and (not tick_times or last_message_ts > tick_times[-1]):
        raise ValueError(
            f"run_until_s={replay_config.run_until_s} does not cover the last real message "
            f"(timestamp={last_message_ts}): the last scheduled tick is "
            f"{tick_times[-1] if tick_times else None}, BEFORE it - that message would "
            f"never be released from the synchronization buffer (no tick() left to call "
            f"pop_ready_batches()). Increase run_until_s (recommended: at least "
            f"synchronization_window_s + tick_interval_s past the last real timestamp)."
        )

    clock = ReplayClock()
    tracker = Tracker(config, clock=clock)
    sink = ReplaySink(tracks_dir, responses_path=responses_path)

    try:
        i = 0  # pointer into tracklet_sequence (file order preserved, never reordered)
        j = 0  # pointer into tick_times
        n_msgs = len(tracklet_sequence)
        n_ticks = len(tick_times)

        while i < n_msgs or j < n_ticks:
            next_msg_t = tracklet_sequence[i].timestamp if i < n_msgs else math.inf
            next_tick_t = tick_times[j] if j < n_ticks else math.inf

            # Tie (same virtual instant): message before the tick, by convention -
            # `<=`, not `<`, is what implements this.
            if next_msg_t <= next_tick_t:
                tracklet = tracklet_sequence[i]
                clock.advance_to(tracklet.timestamp)
                accepted = tracker.ingest(tracklet)
                sink.record_ingest(tracklet, accepted)
                i += 1
            else:
                clock.advance_to(next_tick_t)
                events = tracker.tick()  # never flush_all=True - see the module docstring
                sink.flush_tick(tracker, events)
                j += 1
    finally:
        sink.close()

    return tracker, sink.diagnostics
