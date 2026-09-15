"""Blackout generator"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from evaluation.association_metrics import LocalTrackTruth, TruthLabels, local_track_truth_map

PathLike = Union[str, Path]

POSTGAP_ID_SUFFIX = "__postgap"


@dataclass(frozen=True)
class BlackoutSpec:
    target_true_name: str
    gap_start_s: float
    gap_end_s: float


@dataclass(frozen=True)
class BlackoutResult:
    spec: BlackoutSpec
    messages: tuple[dict, ...]
    origins: tuple[dict, ...]
    postgap_id_remap: dict  # (station_id, original_local_id) -> new_local_id
    num_dropped_messages: int
    num_postgap_relabeled_messages: int


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _load_truth_labels(origins: list[dict]) -> TruthLabels:
    return {(o["station_id"], o["local_track_id"], o["timestamp"]): o["true_target_id"] for o in origins}


def build_blackout_derived_input(base_sim_dir: PathLike, spec: BlackoutSpec) -> BlackoutResult:

    base_sim_dir = Path(base_sim_dir)
    messages = _load_jsonl(base_sim_dir / "sent_messages.jsonl")
    origins = _load_jsonl(base_sim_dir / "tracklet_origins.jsonl")

    truth_labels = _load_truth_labels(origins)
    # Fail-fast if the local_track_id non-reuse contract doesn't hold for
    # this input - never resolved silently (same policy used project-wide).
    local_track_truth: LocalTrackTruth = local_track_truth_map(truth_labels)

    postgap_id_remap: dict[tuple, str] = {}
    derived_messages: list[dict] = []
    derived_origins: list[dict] = []
    num_dropped = 0
    num_relabeled = 0

    for msg in messages:
        key = (msg["station_id"], msg["local_track_id"])
        target = local_track_truth.get(key)
        if target is None:
            raise ValueError(
                f"message {key} in sent_messages.jsonl has no matching entry "
                f"in tracklet_origins.jsonl - the station_id/local_track_id contract is broken"
            )

        is_target = target == spec.target_true_name
        t = msg["timestamp"]

        if is_target and spec.gap_start_s <= t < spec.gap_end_s:
            num_dropped += 1
            continue  # blackout - row removed entirely

        if is_target and t >= spec.gap_end_s:
            if key not in postgap_id_remap:
                postgap_id_remap[key] = f"{msg['local_track_id']}{POSTGAP_ID_SUFFIX}"
            new_id = postgap_id_remap[key]
            msg = {**msg, "local_track_id": new_id}
            num_relabeled += 1

        derived_messages.append(msg)
        derived_origins.append(
            {"station_id": msg["station_id"], "local_track_id": msg["local_track_id"], "timestamp": t, "true_target_id": target}
        )

    return BlackoutResult(
        spec=spec,
        messages=tuple(derived_messages),
        origins=tuple(derived_origins),
        postgap_id_remap=postgap_id_remap,
        num_dropped_messages=num_dropped,
        num_postgap_relabeled_messages=num_relabeled,
    )


def write_blackout_derived_sim_dir(base_sim_dir: PathLike, result: BlackoutResult, output_sim_dir: PathLike) -> None:

    """Writes the derived sim/: ground_truth.jsonl copied verbatim (only
    observability changes), sent_messages.jsonl/tracklet_origins.jsonl rewritten
    per the blackout result. --Never overwrites base_sim_dir."""

    base_sim_dir = Path(base_sim_dir)
    output_sim_dir = Path(output_sim_dir)
    if output_sim_dir.resolve() == base_sim_dir.resolve():
        raise ValueError("output_sim_dir cannot equal base_sim_dir - never overwrite the base input")
    output_sim_dir.mkdir(parents=True, exist_ok=True)

    (output_sim_dir / "ground_truth.jsonl").write_text(
        (base_sim_dir / "ground_truth.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )

    with (output_sim_dir / "sent_messages.jsonl").open("w", encoding="utf-8") as handle:
        for msg in result.messages:
            handle.write(json.dumps(msg) + "\n")

    with (output_sim_dir / "tracklet_origins.jsonl").open("w", encoding="utf-8") as handle:
        for origin in result.origins:
            handle.write(json.dumps(origin) + "\n")
