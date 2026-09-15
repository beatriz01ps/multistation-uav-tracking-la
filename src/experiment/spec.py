"""ExperimentRunSpec"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config.models import AppConfig


def compute_config_hash(config: AppConfig) -> str:

    """8 hex chars"""

    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def compute_run_id(scenario_id: str, seed: int, config: AppConfig) -> str:
    return f"{scenario_id}__seed{seed}__{compute_config_hash(config)}"


@dataclass(frozen=True)
class ExperimentRunSpec:
    scenario_path: Path
    scenario_id: str
    split: Optional[str]
    seed: int
    config: AppConfig

    @property
    def config_hash(self) -> str:
        return compute_config_hash(self.config)

    @property
    def run_id(self) -> str:
        return compute_run_id(self.scenario_id, self.seed, self.config)
