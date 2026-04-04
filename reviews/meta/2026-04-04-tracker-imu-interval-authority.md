# Meta Review

## Commit Goal

Make tracker-side `pending_imu` authoritative for exactly one capture interval so the VIO runtime no longer silently concatenates IMU from multiple `CaptureBundle`s that were already interval-scoped upstream.

## Current Repo Starting Point

Replay and live capture both already construct `CaptureBundle`s with interval-scoped IMU, but the tracker was still appending each batch into `pending_imu`. If a frame skipped VIO or had insufficient IMU support, that state could leak into the next frame and create a preintegration over multiple capture intervals.

## Previous Invariants

- `CaptureBundle` is the source of truth for per-frame visual and IMU interval pairing.
- `pending_imu` should only exist to feed the current frame’s VIO refinement.
- Runtime continuity state and IMU interval state are separate concerns.

## New Invariants Claimed

- `pending_imu` is overwritten at the start of every processed capture and therefore represents only the current capture’s IMU interval.
- An absent IMU batch explicitly clears `pending_imu`; the tracker must not inherit the previous frame’s IMU interval.
- Existing authoritative branches still clear `pending_imu` after consuming or superseding the current interval.

## Touched Files

- `/home/makerspace/kiko-vio/crates/kiko-slam/src/tracker.rs`

## New Or Changed Metrics

- none

## New Or Changed Solver Outputs

- none

## Tests Added

- `tracker::tests::vio_runtime_set_capture_imu_interval_replaces_previous_interval`
- `tracker::tests::vio_runtime_set_capture_imu_interval_clears_on_absent_batch`

## Tests Run

- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml --quiet`
- `bash /home/makerspace/kiko-vio/scripts/check_meta_review.sh`

## Known Risks Or Deferred Follow-Ups

- This commit does not yet resolve broader tracker continuity issues such as relocalization recovery alignment or visual-free VIO-window adoption semantics.
- Gravity, bias, and early-window initialization remain separate upstream causes of high VIO residuals.

## Findings

- none in reviewed scope

## Invariant Verdict

- strengthened: `pending_imu` now has one lawful meaning, the current `CaptureBundle` interval
- strengthened: absent capture IMU no longer inherits previous-frame measurements
- weakened or ambiguous: none in reviewed scope

## Metric Verdict

- trustworthy: no metrics changed; this commit improves the validity of downstream IMU and reprojection diagnostics by removing cross-capture interval contamination
- partial or misleading: none introduced in reviewed scope

## Test Verdict

- covered: explicit replacement and clearing behavior for `pending_imu`
- missing: end-to-end replay trace proving reduced IMU-cost drift remains a follow-up for the broader estimator stack, not a blocker for this local interval-authority fix

## Merge Decision

`accept`
