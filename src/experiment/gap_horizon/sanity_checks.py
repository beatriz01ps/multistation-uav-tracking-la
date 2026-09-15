"""Verifies the blackout contract: never fails silently nor assumes truth without checking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from experiment.gap_horizon.blackout import BlackoutResult, BlackoutSpec


@dataclass(frozen=True)
class BlackoutContractCheck:
    zero_target_messages_inside_window: bool
    all_target_messages_outside_window_preserved: bool
    other_targets_message_count_unchanged: bool
    postgap_local_ids_are_new: bool
    pregap_local_ids_unchanged: bool
    details: dict

    @property
    def passed(self) -> bool:
        return (
            self.zero_target_messages_inside_window
            and self.all_target_messages_outside_window_preserved
            and self.other_targets_message_count_unchanged
            and self.postgap_local_ids_are_new
            and self.pregap_local_ids_unchanged
        )


def verify_blackout_contract(base_sim_dir: Path, result: BlackoutResult, spec: BlackoutSpec) -> BlackoutContractCheck:
    
    """Verifies, from the BlackoutResult itself."""

    base_origins = [json.loads(l) for l in (base_sim_dir / "tracklet_origins.jsonl").open(encoding="utf-8")]

    def _in_window(t: float) -> bool:
        return spec.gap_start_s <= t < spec.gap_end_s

    # no message from the target survives inside the window
    zero_inside = all(not (o["true_target_id"] == spec.target_true_name and _in_window(o["timestamp"])) for o in result.origins)

    # every message from the target outside the window, in the base, survives in the derived version (same count)
    base_target_outside = sum(1 for o in base_origins if o["true_target_id"] == spec.target_true_name and not _in_window(o["timestamp"]))
    derived_target_outside = sum(1 for o in result.origins if o["true_target_id"] == spec.target_true_name and not _in_window(o["timestamp"]))
    all_outside_preserved = base_target_outside == derived_target_outside

    # messages from any other physical target .an identical count, base vs derived (never touched).
    base_other = sum(1 for o in base_origins if o["true_target_id"] != spec.target_true_name)
    derived_other = sum(1 for o in result.origins if o["true_target_id"] != spec.target_true_name)
    other_unchanged = base_other == derived_other

    # every new id (postgap_id_remap) is literally diff from the original id it replaces
    postgap_ids_are_new = all(new_id != original_local_id for (_station, original_local_id), new_id in result.postgap_id_remap.items())

    # no pre-gap message from the target uses one of the new ids (only post-gap messages do).
    new_ids = set(result.postgap_id_remap.values())
    pregap_target_ids_used = {
        m["local_track_id"] for m, o in zip(result.messages, result.origins)
        if o["true_target_id"] == spec.target_true_name and o["timestamp"] < spec.gap_start_s
    }
    pregap_unchanged = pregap_target_ids_used.isdisjoint(new_ids)

    return BlackoutContractCheck(
        zero_target_messages_inside_window=zero_inside,
        all_target_messages_outside_window_preserved=all_outside_preserved,
        other_targets_message_count_unchanged=other_unchanged,
        postgap_local_ids_are_new=postgap_ids_are_new,
        pregap_local_ids_unchanged=pregap_unchanged,
        details={
            "num_dropped_messages": result.num_dropped_messages,
            "num_postgap_relabeled_messages": result.num_postgap_relabeled_messages,
            "postgap_id_remap": {f"{k[0]}/{k[1]}": v for k, v in result.postgap_id_remap.items()},
            "base_target_outside_count": base_target_outside,
            "derived_target_outside_count": derived_target_outside,
            "base_other_count": base_other,
            "derived_other_count": derived_other,
        },
    )
