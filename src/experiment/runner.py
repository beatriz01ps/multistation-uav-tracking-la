"""Runs one ExperimentRunSpec end to end"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from evaluation.offline.evaluate_run import evaluate_run
from experiment.spec import ExperimentRunSpec
from network_simulator.scenario import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]

PYTHON = sys.executable

STARTUP_WAIT_S = 1.0
SETTLE_AFTER_SIM_S = 2.0
EXTRA_TIMEOUT_S = 30.0

EXPERIMENT_PROCESS_INTERVAL_S = 0.2


def extract_tracker_error_lines(tracker_log_text: str, *, tail: int = 5) -> list[str]:

    """Lines of tracker.log relevant for diagnosing a failure """

    matched = [line for line in tracker_log_text.splitlines() if "ERROR" in line or "Traceback" in line or "Error" in line]
    return matched[-tail:]


def run_experiment(spec: ExperimentRunSpec, output_root: Path, port: int) -> Path:

    """Runs the tracker+simulator over real UDP for spec, evaluates the result, and returns the run's sim/ directory."""

    run_dir = output_root / (spec.split or "unclassified") / spec.run_id
    tracks_dir = run_dir / "tracks"
    sim_dir = run_dir / "sim"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "resolved_config.yaml"
    config_path.write_text(yaml.safe_dump(spec.config.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    scenario = load_scenario(spec.scenario_path)

    tracker_log_path = run_dir / "tracker.log"
    tracker_log = tracker_log_path.open("w", encoding="utf-8")
    tracker_proc = subprocess.Popen(
        [
            PYTHON, "src/main.py",
            "--port", str(port),
            "--config", str(config_path),
            "--process-interval", str(EXPERIMENT_PROCESS_INTERVAL_S),
            "--track-log-dir", str(tracks_dir),
        ],
        cwd=str(REPO_ROOT), stdout=tracker_log, stderr=subprocess.STDOUT,
    )
    try:
        time.sleep(STARTUP_WAIT_S)
        sim_result = subprocess.run(
            [
                PYTHON, "src/network_simulator/main.py",
                "--scenario", str(spec.scenario_path),
                "--port", str(port),
                "--seed", str(spec.seed),  # the EFFECTIVE seed - never rewrites the original YAML
                "--out-dir", str(sim_dir),
            ],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=scenario.duration_s + EXTRA_TIMEOUT_S,
        )
        time.sleep(SETTLE_AFTER_SIM_S)
    except subprocess.TimeoutExpired as exc:
        sim_result = None
        sim_error = f"simulator did not finish within the timeout: {exc!r}"
    else:
        sim_error = None
    finally:
        tracker_proc.terminate()
        try:
            tracker_proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            tracker_proc.kill()
            tracker_proc.wait(timeout=5.0)
        tracker_log.close()

    tracker_log_text = tracker_log_path.read_text(encoding="utf-8", errors="replace")
    tracker_errors = extract_tracker_error_lines(tracker_log_text)

    if sim_error is not None:
        raise RuntimeError(sim_error)
    if sim_result.returncode != 0:
        raise RuntimeError(f"simulator exited with rc={sim_result.returncode}: {sim_result.stderr[-800:]}")
    if tracker_errors:
        raise RuntimeError(f"tracker.log contains an error: {tracker_errors}")

    report = evaluate_run(sim_dir, tracks_dir)
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(dataclasses.asdict(report), indent=2, default=str), encoding="utf-8")

    return sim_dir
