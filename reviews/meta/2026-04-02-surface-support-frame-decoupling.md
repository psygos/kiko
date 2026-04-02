# Meta Review

## Commit Goal

Stop starving the stable surface map behind SLAM keyframe creation. The surface estimator already has its own pose-quality gate, novelty grouping, and consistency filters; this tranche lets any support frame that passes those checks contribute to the surface map, while preserving the SLAM keyframe bit as an explicit diagnostic instead of an implicit fusion gate.

## Current Repo Starting Point

After `35e8164`, rejected VIO proposals no longer left the inertial runtime on a stale frame. The next visible bottleneck was surface integration policy: dataset SLAM only requested stable-surface fusion when `output.keyframe.is_some()`, so the low-resolution surface map could remain empty even when the tracker had many acceptable non-keyframe frames with stereo support.

## Previous Invariants

- Surface integration required both a pose-gate accept and `surface_integration_requested == true`.
- In dataset SLAM, `surface_integration_requested` was effectively synonymous with `output.keyframe.is_some()`.
- The stable surface estimator and SLAM keyframe policy were still partially coupled.

## New Invariants Claimed

- Stable surface support can be requested independently of SLAM keyframe creation.
- `diagnostics/surface/frame_gate/integration_requested` now represents “surface support requested for this frame”, not “SLAM keyframe was created”.
- `diagnostics/surface/frame_gate/slam_keyframe` preserves the old keyframe bit explicitly so the two policies remain distinguishable.
- The pose-quality gate remains authoritative for surface fusion; this change does not relax reprojection or degeneracy thresholds.

## Touched Files

- `crates/kiko-slam/src/bin/kiko_slam/slam.rs`
- `crates/kiko-slam/src/viz.rs`

## New Or Changed Metrics

- `diagnostics/surface/frame_gate/slam_keyframe`
- changed semantics: `diagnostics/surface/frame_gate/integration_requested` now means surface-support request rather than SLAM-keyframe request

## New Or Changed Solver Outputs

- none

## Tests Added

- `viz::tests::log_surface_observations_support_frame_mutates_surface_map_without_slam_keyframe`

## Tests Run

- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml viz::tests::log_surface_observations_support_frame_mutates_surface_map_without_slam_keyframe`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml viz::tests::log_surface_observations_visual_only_path_logs_candidates_without_mutating_map`
- `bash /home/makerspace/kiko-vio/scripts/check_meta_review.sh`

## Known Risks Or Deferred Follow-Ups

- This fixes starvation from keyframe policy, but it does not fix the underlying pose-quality problem. Frames still need to clear the existing surface pose gate.
- The `integration_requested` metric changed semantics. Existing dashboards or manual interpretations must now read `slam_keyframe` if they specifically want keyframe cadence.
- A reviewer agent was requested as part of the standing process, but no automated findings were available before commit preparation. Findings below are from direct review plus targeted regression tests.

## Findings

- none blocking in this tranche
- follow-up: add a dataset-level trace that reports support-frame request count vs SLAM-keyframe count so starvation diagnosis is visible without Rerun inspection

## Invariant Verdict

- strengthened: stable-surface support policy is now explicitly separate from SLAM keyframe management
- weakened or ambiguous: `integration_requested` changed meaning, so any external consumer that assumed “keyframe request” must now consult the new `slam_keyframe` scalar

## Metric Verdict

- trustworthy: the new `slam_keyframe` scalar makes the policy split explicit rather than implicit
- partial or misleading: `integration_requested` is honest but semantically changed; historical comparisons need to account for that rename-in-place

## Test Verdict

- covered: a support frame can now mutate the surface map without being a SLAM keyframe, and the visual-only non-mutating path still stays non-authoritative
- missing: end-to-end dataset verification showing surface voxels begin accumulating on non-keyframe support frames under real pose-gate acceptance

## Merge Decision

`accept with follow-up`
