# Multi-Station UAV Tracking — Replication Package

## Overview

Centralized multi-station UAV tracking system: fuses local track
estimates from distributed ground stations into persistent global
target identities. Supports Euclidean/Mahalanobis association,
Information Fusion/Covariance Intersection, EKF/UKF over Constant
Velocity/Coordinated Turn motion models, and duplicate-track
reconciliation. 

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

`requirements-final-lock.txt` pins the exact dependency versions used
to produce the results reported in the paper.

## Reproducing the Paper Results

Reproduce Table 3 and the complete-architecture evaluation, recomputed from the
bundled per-run metrics of the main campaign:

```bash
python reproduce/table3.py
```

Reproduce Table 4 (gap-duration robustness):

```bash
python reproduce/table4.py
```

Reproduce the Ma-inspired baseline:

```bash
python reproduce/ma_inspired.py
```

Reproduce selected paired contrasts via deterministic replay of frozen
LocalTracklet sequences:

```bash
python reproduce/deterministic_replay.py
```

Reproduce the sensor-informed robustness evaluation (120 runs: 6 scenarios x
10 seeds x {position-only, full-state}):

```bash
python reproduce/sensor_informed.py
```

`table3.py` and `sensor_informed.py` recompute the reported statistics from
bundled per-run metrics rather than re-executing the UDP campaigns;
`deterministic_replay.py` and `ma_inspired.py` replay frozen LocalTracklet
sequences directly. All four are exact given the same bundled input.
`ma_inspired.py --full` reconstructs all 720 cases from the same bundled
frozen inputs (the default runs 2 representative cases).

## Expected Outputs

Each command above prints a short per-comparison summary and exits with
`OVERALL: MATCH` (exit code 0) when every recomputed value agrees with
the bundled canonical result in `expected/`, or `OVERALL: DIFFERS`
(exit code 1) otherwise.

## Running a Custom Scenario

`src/main.py` (tracker) and `src/network_simulator/main.py` (simulator) are
general-purpose CLIs, not limited to the scenarios above:

```bash
python src/main.py --port 9999
python src/network_simulator/main.py --scenario <path/to/your.yaml> --port 9999 --out-dir <output_dir>
```

Write a scenario YAML following the schema in `src/network_simulator/scenario.py`
(`uavs`, `stations`, `obstacles`, `dropout_windows`, `id_switch_events`). This is
the underlying tracking system, not part of the paper's reproduction claims —
there is no bundled CLI for evaluating a custom run; call
`evaluation.offline.evaluate_run.evaluate_run(sim_dir, tracks_dir)` directly,
using `reproduce/sensor_informed.py` as a wiring example.

## Repository Structure

```
multistation-uav-tracking-la/
├── src/                tracker, simulator, evaluation, experiment code
├── reproduce/          the five scripts above
├── data/               bundled inputs and per-run metrics
├── expected/           canonical results
├── frozen/             frozen scientific configuration (D*, thresholds)
├── calibration/        frozen calibration report behind the Euclidean gate D*
├── reproducibility/    exact frozen inputs behind the Ma-inspired baseline
└── docs/               architecture, replay semantics, Ma-GTM derivation
```

See [`docs/architecture.md`](docs/architecture.md) for more information