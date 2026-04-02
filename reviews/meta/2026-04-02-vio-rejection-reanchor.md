# Meta Review

## Commit Goal

Fix a stale-state authority bug in the tightly coupled VIO path. When a VIO proposal loses to the visual pose on shared-support reprojection error, the tracker must still consume the current frame's IMU interval and advance the inertial runtime to the current visual frame. Without that, rejected frames silently stretch preintegration across multiple captures and poison later VIO proposals.

## Current Repo Starting Point

The previous tranche made VIO proposal adoption transactional and honest on cost reporting. That removed silent pose mutation on rejected proposals, but it left one deeper state-machine hole: the inertial runtime still only advanced on adopted VIO proposals. If VIO lost arbitration, `pending_imu`, `last_optimized_state`, and `vio_window` remained anchored on an older frame.

## Previous Invariants

- VIO proposals were only adopted when they did not worsen shared projectable tracked reprojection RMSE.
- Rejected VIO proposals did not mutate the authoritative tracker pose.
- The inertial runtime implicitly assumed that "proposal rejected" and "current frame not consumed" were compatible states.

## New Invariants Claimed

- Every VIO proposal that reached shared-support arbitration now consumes the current IMU interval, regardless of whether the proposal wins pose adoption.
- Rejected VIO proposals reanchor the inertial runtime to a single-frame visual anchor at the current frame instead of leaving the runtime on a stale older window.
- `pending_imu` is cleared exactly when the current frame becomes the new inertial anchor, whether by accepted VIO or by rejected-VIO visual reanchor.

## Touched Files

- `crates/kiko-slam/src/tracker.rs`

## New Or Changed Metrics

- none

## New Or Changed Solver Outputs

- internal only: `VioPoseProposal` now carries a `rejected_visual_anchor` so rejection can still advance the inertial runtime lawfully

## Tests Added

- `tracker::tests::vio_runtime_visual_reanchor_replaces_stale_window_and_clears_pending_imu`

## Tests Run

- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml tracker::tests::vio_runtime_visual_reanchor_replaces_stale_window_and_clears_pending_imu`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml tracker::tests::vio_pose_adoption_rejects_changed_projectable_support_even_when_counts_match`

## Known Risks Or Deferred Follow-Ups

- This fixes stale interval accumulation, but it does not solve inertial bootstrap. Gravity, early velocity, and bias initialization remain likely dominant causes of the remaining IMU-vs-visual tension.
- The rejected-path reanchor keeps the optimized VIO proposal's velocity and bias while replacing pose authority with the visual pose. That is a lawful compromise for consuming the interval, but it still needs empirical validation on real traces.
- A reviewer agent was invoked for this diff, but no automated findings were returned before commit preparation. Findings below are from direct code review plus the targeted test pass.

## Findings

- none blocking in this tranche
- follow-up: add an end-to-end diagnostic proving that repeated rejected VIO proposals no longer cause preintegration interval growth across frames

## Invariant Verdict

- strengthened: rejected VIO proposals can no longer leave the inertial runtime on a stale frame; current-frame IMU consumption is explicit and symmetric with accepted VIO
- weakened or ambiguous: latent velocity/bias from the rejected VIO solve are still carried into the visual reanchor anchor state; this is intentional but not yet empirically calibrated

## Metric Verdict

- trustworthy: unchanged proposal-adoption metrics remain honest and now better aligned with runtime state transitions
- partial or misleading: stderr trace alone still cannot prove whether reanchoring happened correctly over long runs; a structured per-frame trace remains a follow-up

## Test Verdict

- covered: direct unit coverage for clearing `pending_imu`, replacing the stale window, and updating `last_optimized_state` / `predicted_state`
- missing: dataset-level regression showing that IMU cost no longer ramps purely because of repeated proposal rejection

## Merge Decision

`accept with follow-up`
