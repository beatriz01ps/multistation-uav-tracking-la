# Replay semantics

The project has two deterministic replay modes: `EventDrivenReplay` and `PeriodicVirtualReplay`.

Both use the same tracker. The difference is only when `Tracker.tick()` is called.

## EventDrivenReplay

`EventDrivenReplay` advances when messages arrive.

For each frozen tracklet, it advances to the tracklet timestamp, ingests the message, and calls `tracker.tick()`. At the end, pending data is flushed.

There are no empty ticks between messages. A 5 s observation gap therefore produces one 5 s prediction step instead of several smaller periodic steps.

This mode is appropriate when the goal is to replay the same frozen observations across configurations and the behavior inside long gaps is not the quantity being studied.

It should not be treated as numerically equivalent to the online periodic scheduler during a long gap.

## PeriodicVirtualReplay

`PeriodicVirtualReplay` calls `tracker.tick()` at a fixed interval using a deterministic virtual clock.

It keeps ticking even when no message arrives, so it reproduces the temporal behavior needed for experiments involving:

- prediction through observation gaps;
- covariance growth;
- lifecycle timing;
- reassociation after a gap.

This is the replay mode used when gap-duration results depend on the predicted state during the blackout.

## Ordering

When a message and a tick occur at the same virtual instant, the message is ingested first.

The replay does not change the input data:

- measurement timestamps are preserved;
- message order is preserved;
- no synthetic measurements are created.

`PeriodicVirtualReplay` does not use `flush_all=True`; `run_until_s` must therefore extend far enough for the last real message to leave the synchronization buffer.

## Why the modes can differ

With the current UKF, one large prediction step is not bit-identical to several smaller steps over the same interval.

The UKF recomputes sigma points at each prediction, so process noise accumulates differently across repeated steps.

This matters when the predicted state or covariance is later used by the association gate. For that reason, gap-duration experiments use `PeriodicVirtualReplay`.

## Scope

`experiment/replay/` contains only generic replay behavior.

Study-specific logic stays outside it. The Ma-inspired baseline uses neither replay mode because it is an offline batch method.
