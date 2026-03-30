# Meta Review

## Commit Goal

Prevent a keyframe with an explicitly degenerate local BA result from strengthening the stable surface map, while preserving observability for that reject mode and its degenerate reason.

## Current Repo Starting Point

This tranche starts after `5ff4ca2` on `measurement-system`. At the start of the change, stable surface fusion already required tracked-count and tracked-RMSE gates and rejected predictively inconsistent novel support views. But if the tracker already knew local BA had degenerated on a keyframe, that frame could still pass the existing pose gate and strengthen the stable surface map.

## Previous Invariants

- Stable surface fusion required projectable tracked observation count and tracked reprojection RMSE to pass.
- Predictively inconsistent novel support views were rejected before entering the stable surface belief map.
- Missing BA diagnostics were tolerated by the surface pose gate.

## New Invariants Claimed

- If `FrameDiagnostics.ba_result` is explicitly `BaResult::Degenerate`, stable surface fusion rejects the frame before any surface integration.
- Missing `ba_result` remains non-blocking; only explicit degeneracy rejects.
- The degenerate reject mode is observable as a distinct pose-gate metric, with one-hot degenerate-reason metrics.
- When tracked RMSE is available on the degenerate branch, it remains observable instead of being discarded.

## Touched Files

- `crates/kiko-slam/src/viz.rs`

## New Or Changed Metrics

- `diagnostics/surface/pose_gate/rejected_degenerate_ba_result`
- `diagnostics/surface/pose_gate/ba_degenerate_too_few_poses`
- `diagnostics/surface/pose_gate/ba_degenerate_too_few_landmarks`
- `diagnostics/surface/pose_gate/ba_degenerate_no_factors`

## New Or Changed Solver Outputs

- none

## Tests Added

- `viz::tests::surface_pose_quality_gate_rejects_degenerate_ba_result`
- `viz::tests::surface_pose_quality_scalars_export_ba_degenerate_reason`
- `viz::tests::log_surface_observations_degenerate_ba_path_leaves_surface_map_unchanged`

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/viz.rs`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- reviewer agent `Euclid`: `accept`

## Known Risks Or Deferred Follow-Ups

- This is still a coarse solver-health veto over `BaResult::Degenerate`, not a covariance-calibrated confidence test.
- The pose gate remains one-hot and first-hit: low-count rejection still wins before BA-degenerate rejection if both are true.

## Findings

- `major`: the stable surface map is materially harder to pollute because a frame with explicit BA degeneracy now fails closed before surface integration.
- `medium`: the new observability is semantically honest because the reject mode and each degenerate reason are exported separately, and the branch now preserves tracked RMSE when it exists.
- `minor risk`: the new gate is still a coarse policy signal rather than a first-principles uncertainty metric.

## Invariant Verdict

- strengthened: explicit BA degeneracy now vetoes stable surface fusion; missing BA diagnostics remain non-blocking; the degenerate branch preserves realized tracked RMSE observability.
- weakened or ambiguous: the one-hot gate ordering remains a prioritization choice rather than a multi-cause diagnostic model.

## Metric Verdict

- trustworthy: `rejected_degenerate_ba_result` and the three reason metrics are honest one-hot policy signals; tracked RMSE remains visible on the degenerate branch when present.
- partial or misleading: the BA-degenerate signal is intentionally coarse and should not be read as a calibrated uncertainty estimate.

## Test Verdict

- covered: direct gate rejection on degenerate BA; scalar export for the degenerate branch; full `log_surface_observations` reject path for degenerate BA; default and `vio` test matrices.
- missing: no blocker-level gap remains for this tranche.

## Merge Decision

`accept`
