# Meta Review

## Commit Goal

Make the stable-surface path visibly debuggable in Rerun even before any voxel reaches confirmed state. The change logs frame-level surface candidates plus classified voxel clouds for pending-support and rejected beliefs, and it exports matching summary scalars so operators can tell whether the stable surface is absent because nothing was integrated, because support is still sparse, or because consistency/uncertainty gates are rejecting voxels.

## Current Repo Starting Point

This tranche starts from `913ce1f` on `measurement-system`, where the stable-surface path already fused support views conservatively and gated integration on pose quality, but the viewer-facing output was still too binary: if no voxel was confirmed yet, the main 3D surface entity could remain effectively invisible and the operator had to infer rejection reasons from scattered diagnostics instead of directly seeing pending and rejected surface classes.

## Previous Invariants

- The surface belief map internally distinguished confirmed from unconfirmed/rejected beliefs, but only confirmed voxels were extracted for rendering.
- `viz.rs` emitted scalar diagnostics for integration and pose gating, but the 3D viewer did not expose pending-support and rejected voxel classes as first-class entities.
- A frame with accepted raw surface points but no confirmed voxels could look like “no stable surface exists yet” rather than “surface evidence is present but not yet mature.”

## New Invariants Claimed

- `SurfaceBeliefMap` classifies beliefs into `Confirmed`, `PendingSupport`, `RejectedConsistency`, `RejectedUncertainty`, and `RejectedConsistencyAndUncertainty`.
- `RerunSink::log_surface_observations` always logs frame candidate points in map frame under `world/stable_surface_debug/frame_candidates`, colored by whether the pose gate accepted integration.
- `RerunSink::log_surface_map_state` exports classified voxel clouds under:
  - `world/stable_surface_voxels`
  - `world/stable_surface_debug/pending_support_voxels`
  - `world/stable_surface_debug/rejected_consistency_voxels`
  - `world/stable_surface_debug/rejected_uncertainty_voxels`
  - `world/stable_surface_debug/rejected_consistency_and_uncertainty_voxels`
- Surface summary scalars now report pending and rejected voxel counts explicitly beside confirmed count.

## Touched Files

- `crates/kiko-slam/src/surface_map.rs`
- `crates/kiko-slam/src/viz.rs`

## New Or Changed Metrics

- `diagnostics/surface/pending_support_voxels`
- `diagnostics/surface/rejected_consistency_voxels`
- `diagnostics/surface/rejected_uncertainty_voxels`
- `diagnostics/surface/rejected_consistency_and_uncertainty_voxels`

## New Or Changed Solver Outputs

- none

## Tests Added

- `surface_map::tests::debug_clouds_and_summary_classify_pending_and_rejected_voxels`
- `viz::tests::log_surface_observations_logs_debug_entities_before_confirmation`

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/surface_map.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/viz.rs`
- `cargo test -p kiko-slam viz::tests::log_surface_observations_logs_debug_entities_before_confirmation --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `bash /home/makerspace/kiko-vio/scripts/check_meta_review.sh`
- direct diff review by Codex; delegated review agent was attempted but failed due platform usage cap before returning findings

## Known Risks Or Deferred Follow-Ups

- The debug clouds are truthful but still derived from the current isotropic voxel belief model; they are visibility/debug aids, not calibrated posterior classes in the full anisotropic sense.
- Each debug class is capped independently by `max_render_points`, so aggregate debug output can exceed the confirmed-surface render cap. That is acceptable for observability, but it is not a global render-budget guarantee.
- The main stable surface can still remain visually empty if there are no accepted raw surface points at all; in that case the operator should rely on `frame_candidates` plus the scalar diagnostics to distinguish “no evidence generated” from “evidence generated but not integrated.”

## Findings

- none: the change strengthens viewer honesty without loosening estimator policy, and the new tests pin both the classifier accounting and the pre-confirmation Rerun entity visibility.

## Invariant Verdict

- strengthened: pending and rejected voxel states are now explicit in both typed summaries and viewer entities; accepted-but-unconfirmed frames no longer depend on confirmed-voxel existence to become visible in Rerun.
- weakened or ambiguous: the debug classes remain policy-derived labels over the current isotropic belief model, so they should not be over-read as fully calibrated statistical categories.

## Metric Verdict

- trustworthy: the new pending/rejected counts are honestly named and align with the exact render-class partition used by the map.
- partial or misleading: none newly introduced; the debug classes are clearly labeled as debug surface states rather than posterior probabilities.

## Test Verdict

- covered: class partition accounting in `surface_map`, scalar export in `viz`, and Rerun debug-entity emission before confirmation.
- missing: no end-to-end saved-RRD assertion yet verifies these entity paths in a full dataset replay.

## Merge Decision

`accept`
