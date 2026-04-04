# Meta Review

## Commit Goal

Prevent a VIO proposal from becoming authoritative on the current frame unless the accepted VIO linearization actually used at least one lawful current-frame visual residual, and represent the missing-support case explicitly in diagnostics and observability.

## Current Repo Starting Point

The tracker previously allowed a current-frame VIO proposal to compete downstream as long as the raw `ObservationSet` existed, even if the current frame ultimately contributed no lawful visual residuals to the VIO solve after observation resolution and projectability checks.

## Previous Invariants

- `NotRun` means no VIO proposal was evaluated.
- `Rejected*` means a VIO proposal ran but lost on an explicit reason.
- Current-frame visual support should be defined by actual residual participation, not by raw observation intent.

## New Invariants Claimed

- A current-frame VIO proposal is only eligible for adoption if the accepted VIO solve used at least one current-frame visual reprojection residual.
- The missing-current-support case is an explicit `RejectedInsufficientCurrentVioObservationSupport`, not an implicit metric-side artifact.
- `NotRun` vs `Rejected` semantics remain distinct in diagnostics and observability.

## Touched Files

- `/home/makerspace/kiko-vio/crates/kiko-slam/src/local_ba.rs`
- `/home/makerspace/kiko-vio/crates/kiko-slam/src/tracker.rs`
- `/home/makerspace/kiko-vio/crates/kiko-slam/src/diagnostics.rs`
- `/home/makerspace/kiko-vio/crates/kiko-slam/src/observability.rs`

## New Or Changed Metrics

- none

## New Or Changed Solver Outputs

- `VioSolveResult.last_frame_visual_residual_count`

## Tests Added

- `tracker::tests::vio_pose_adoption_rejects_missing_current_vio_observation_support`

## Tests Run

- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml --quiet`
- `bash /home/makerspace/kiko-vio/scripts/check_meta_review.sh`

## Known Risks Or Deferred Follow-Ups

- This commit does not stop the tracker from computing an under-supported VIO proposal; it prevents that proposal from becoming authoritative. Compute-side pruning remains a follow-up if desired.
- This commit does not yet add a synthetic local-BA test that directly asserts `last_frame_visual_residual_count == 0` on a fully non-projectable current-frame observation set.

## Findings

- none in reviewed scope

## Invariant Verdict

- strengthened: current-frame VIO support is now keyed to actual accepted residual participation rather than raw observation-set existence
- strengthened: missing current-frame support is explicit and observable as a rejection disposition
- weakened or ambiguous: none in reviewed scope

## Metric Verdict

- trustworthy: `NotRun` vs `Rejected` semantics remain distinct; no residual metric was relabeled
- partial or misleading: none introduced in reviewed scope

## Test Verdict

- covered: tracker-level adoption policy now rejects zero-current-residual proposals explicitly, and the full suite passed with updated diagnostics/observability
- missing: direct local-BA synthetic coverage for `last_frame_visual_residual_count` remains a follow-up, not a blocker for this authority fix

## Merge Decision

`accept`
