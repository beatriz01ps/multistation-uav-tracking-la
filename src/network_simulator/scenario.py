"""Scenario configuration for the message generator: UAVs, stations, obstacles,
time-windowed dropout, and per-station local ID switching"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml

from network_simulator.trajectories import TurnSegment


@dataclass(frozen=True)
class DropoutWindow:

    """A specific station stops observing a specific UAV during [start_s, end_s]
    (scenario time, never a local clock/epoch)."""

    station_id: str
    start_s: float
    end_s: float

    def covers(self, station_id: str, t: float) -> bool:
        return station_id == self.station_id and self.start_s <= t <= self.end_s


@dataclass(frozen=True)
class IdSwitchEvent:

    """A specific station "forgets" the local ID it had for a UAV at at_s (scenario
    time) ***the central tracker must reassociate from state/covariance."""

    station_id: str
    at_s: float


@dataclass
class UavSpec:
    name: str  # GT identity -> never sent to the tracker
    initial_state: list  # [x, y, z, vx, vy, vz] at t=0
    dropout_windows: list = field(default_factory=list) 
    id_switch_events: list = field(default_factory=list)
    turn_schedule: list = field(default_factory=list)

@dataclass(frozen=True)
class StationSpec:
    station_id: str
    frequency_hz: float
    position_std: float
    velocity_std: float
    position: tuple = (0.0, 0.0)


@dataclass(frozen=True)
class Obstacle:

    """A circular obstacle in the XY plane blocking line of sight between a stationand a UAV"""

    center: tuple
    radius: float


@dataclass(frozen=True)
class StationErrorModelConfig:

    """Which noise generator stations use: "legacy" (simple isotropic Gaussian) or
    "sensor_informed_v1" PE (camera+geometry-informed). Never changes LocalTracklet's 6D+P6x6 contract, 
    only how noise/availability is generated."""

    type: str = "legacy"
    calibration_profile: Optional[str] = None  


@dataclass
class ScenarioConfig:
    duration_s: float
    seed: int
    uavs: list  
    stations: list  
    obstacles: list = field(default_factory=list)  
    scenario_id: Optional[str] = None
   
    split: Optional[str] = None
    station_error_model: StationErrorModelConfig = field(default_factory=StationErrorModelConfig)


def _parse_uav(raw: dict) -> UavSpec:

    dropout_windows = [
        DropoutWindow(station_id=w["station_id"], start_s=float(w["start_s"]), end_s=float(w["end_s"]))
        for w in raw.get("dropout_windows", [])
    ]
    id_switch_events = [
        IdSwitchEvent(station_id=e["station_id"], at_s=float(e["at_s"])) for e in raw.get("id_switch_events", [])
    ]
    turn_schedule = [
        TurnSegment(duration_s=float(s["duration_s"]), omega=float(s["omega"]))
        for s in raw.get("turn_schedule", [])
    ]
    return UavSpec(
        name=raw["name"],
        initial_state=[float(v) for v in raw["initial_state"]],
        dropout_windows=dropout_windows,
        id_switch_events=id_switch_events,
        turn_schedule=turn_schedule,
    )


def _parse_station(raw: dict) -> StationSpec:

    position = raw.get("position", [0.0, 0.0])
    return StationSpec(
        station_id=str(raw["id"]),
        frequency_hz=float(raw["frequency_hz"]),
        position_std=float(raw["position_std"]),
        velocity_std=float(raw["velocity_std"]),
        position=(float(position[0]), float(position[1])),
    )


def _parse_obstacle(raw: dict) -> Obstacle:

    center = raw["center"]
    return Obstacle(center=(float(center[0]), float(center[1])), radius=float(raw["radius"]))


def _parse_station_error_model(sim: dict) -> StationErrorModelConfig:

    block = sim.get("station_error_model", {})
    return StationErrorModelConfig(
        type=block.get("type", "legacy"), calibration_profile=block.get("calibration_profile"),
    )


def load_scenario(path: Union[str, Path]) -> ScenarioConfig:

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sim = raw.get("simulation", {})
    return ScenarioConfig(
        duration_s=float(sim.get("duration", 30.0)),
        seed=int(sim.get("seed", 0)),
        uavs=[_parse_uav(u) for u in raw["uavs"]],
        stations=[_parse_station(s) for s in raw["stations"]],
        obstacles=[_parse_obstacle(o) for o in raw.get("obstacles", [])],
        scenario_id=sim.get("scenario_id"),
        split=sim.get("split"),
        station_error_model=_parse_station_error_model(sim),
    )


def station_rngs(scenario: ScenarioConfig) -> dict:

    seed_sequence = np.random.SeedSequence(scenario.seed)
    child_seeds = seed_sequence.spawn(len(scenario.stations))
    return {
        station.station_id: np.random.default_rng(child_seed)
        for station, child_seed in zip(scenario.stations, child_seeds)
    }
