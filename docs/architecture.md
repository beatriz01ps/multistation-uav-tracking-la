# Architecture

## Overview

The system receives local UAV tracklets from multiple stations and maintains persistent global tracks.

Each tracklet contains:

`[x, y, z, vx, vy, vz]`

plus a 6x6 covariance matrix, `station_id`, `local_track_id`, and a measurement timestamp.

```json
{
  "station_id": "A",
  "local_track_id": "A001",
  "timestamp": 12.4,
  "state": [x, y, z, vx, vy, vz],
  "covariance": [[...], ...]
}
```

The central tracker is responsible for association, track-to-track fusion, temporal filtering, lifecycle management, and persistent global identity.

## Pipeline

```text
LocalTracklets
    ↓
Temporal synchronization
    ↓
Global-track prediction
    ↓
T2TA association
    ↓
Track-to-track fusion
    ↓
Filter update
    ↓
Lifecycle management
    ↓
Duplicate-track reconciliation
    ↓
GlobalTracks
```

Main modules:

- `synchronization/`: batching and temporal alignment
- `association/`: gating and Hungarian assignment
- `fusion/`: track-to-track fusion
- `filtering/`: EKF/UKF and motion models
- `tracking/`: lifecycle, global identities, and duplicate merging

## Synchronization and time

Tracklets from different stations are grouped into temporal batches before association.

Measurement time is kept separate from local network/waiting time. Batches are released in chronological order, and tracklets in the same batch are aligned before association and fusion.

During observation gaps, the tracker continues predicting through its internal tracking clock.

## Association

Association is performed per station.

The reference configuration uses Mahalanobis distance, chi-square gating, and Hungarian assignment. A position-only Euclidean variant is available as an experimental baseline.

Active tracks are matched before `LOST` tracks so that a stale track with large covariance does not take an observation from a nearby active track.

Unmatched tracklets from different stations may be grouped into a shared candidate before a new `GlobalTrack` is created.

## Fusion and filtering

Associated local estimates are fused before the temporal update.

Available fusion strategies:

- Information Fusion
- Covariance Intersection

Available filters:

- UKF
- EKF

Available motion models:

- Constant Velocity
- Coordinated Turn

The reference configuration uses UKF with Coordinated Turn.

## Lifecycle

```text
TENTATIVE → CONFIRMED → COASTING → LOST → DELETED
```

`COASTING` and `LOST` tracks can return to `CONFIRMED` after reassociation.

The default lifecycle uses elapsed measurement time rather than tracker-cycle counts.

## Duplicate-track reconciliation

Regular association compares tracklets with tracks, not tracks with tracks. A separate reconciliation stage therefore handles duplicate `GlobalTrack`s.

It uses full-state statistical compatibility and shared local-track history. When two tracks are merged, the older global identity is kept.

The merger has its own fusion strategy so it can be enabled, disabled, or changed independently in experiments.

## Configuration

The main experimental dimensions are configurable independently:

- association metric and mode
- fusion strategy
- filter
- motion model
- duplicate merger

This allows the same pipeline to be reused across the experimental contrasts.

## Known limitations

- no Out-of-Sequence Measurement support;
- station clocks are assumed to be comparable within the synchronization window;
- duplicate-track reconciliation does not explicitly model correlation between `GlobalTrack` errors;
- the Euclidean association threshold is a development value;
- under extreme noise and dropout, covariance can diverge numerically in some runs.
