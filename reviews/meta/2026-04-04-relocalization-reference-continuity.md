# Meta Review

## Commit Goal

Preserve the pre-loss `cam_from_odom` reference through relocalization so map/odom realignment on recovery no longer silently no-ops after inertial continuity has already been reset.

## Current Repo Starting Point

Entering relocalization reset inertial continuity immediately, which cleared `predicted_state` and therefore `current_odom_pose()`. Recovery then attempted to align `map_from_odom` using that now-missing runtime state, so the intended realignment could quietly do nothing.

## Previous Invariants

- `map_from_odom` realignment on relocalization recovery should use the last lawful odom pose before continuity is reset.
- Relocalization session state must survive `Continue(...)` transitions without losing the authority needed for recovery.

## New Invariants Claimed

- The pre-loss `cam_from_odom` reference is captured before continuity reset when entering relocalization.
- That reference is stored in `RelocalizationSession` and preserved across relocalization confirmation/failure transitions.
- Recovery uses the stored pre-loss reference to realign `map_from_odom` before resetting runtime continuity again.

## Touched Files

- `/home/makerspace/kiko-vio/crates/kiko-slam/src/tracker.rs`

## New Or Changed Metrics

- none

## New Or Changed Solver Outputs

- none

## Tests Added

- `tracker::tests::relocalization_initial_session_preserves_reference_cam_from_odom`
- `tracker::tests::relocalization_reference_cam_from_odom_prefers_session_reference`

## Tests Run

- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml --quiet`
- `bash /home/makerspace/kiko-vio/scripts/check_meta_review.sh`

## Known Risks Or Deferred Follow-Ups

- There is still no end-to-end tracker test that enters relocalization, resets continuity, recovers, and directly asserts that `map_from_odom.align_to_pose(...)` used the saved pre-loss reference.
- This commit does not change VIO initialization, gravity, or bias bootstrap.

## Findings

- none in reviewed scope

## Invariant Verdict

- strengthened: relocalization recovery no longer depends on `current_odom_pose()` surviving an earlier continuity reset
- strengthened: the pre-loss odom reference is now explicit session state, not hidden mutable runtime state
- weakened or ambiguous: none in reviewed scope

## Metric Verdict

- trustworthy: no metrics changed; this commit improves frame-authority correctness for downstream poses without relabeling any diagnostics
- partial or misleading: none introduced in reviewed scope

## Test Verdict

- covered: relocalization session now stores and prefers the saved pre-loss odom reference
- missing: full recovery-path integration test for actual `map_from_odom` realignment remains a follow-up, not a blocker

## Merge Decision

`accept`
