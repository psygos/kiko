# Meta Review

## Commit Goal

Strengthen the stable surface belief map so that "confirmed enough to render" means "supported, internally consistent, and confident relative to the map resolution." The change should also make capped rendering preserve the best-confirmed voxels first instead of applying a stride that can discard the most trustworthy surface beliefs.

## Current Repo Starting Point

This tranche starts immediately after `528620c` on `measurement-system`. The stable surface path already used measured sparse stereo observations, grouped same-batch same-voxel support conservatively, and exported support-honest metrics. But confirmation was still based only on support count and residual consistency, and the render cap still used a stride-based thinning step that could discard the most stable voxels while keeping noisier ones.

## Previous Invariants

- A voxel became confirmed when it had enough support views and a low enough residual consistency score.
- Posterior uncertainty affected fusion but did not participate in the confirmation predicate.
- When confirmed output exceeded `max_render_points`, extraction kept a deterministic stride sample rather than the most stable subset.
- The default surface confirmation semantics were not explicitly tied to map resolution.

## New Invariants Claimed

- A voxel is confirmed only if it has enough support views, passes the consistency gate, and has posterior standard deviation no larger than the configured confirmation sigma threshold.
- By default, `max_confirmed_std_dev_m` tracks `voxel_size`, so a rendered voxel cannot claim map resolution finer than its current posterior uncertainty.
- If the confirmed set exceeds the render cap, extraction keeps the lowest-uncertainty confirmed voxels first, breaking ties by support count and position for determinism.
- The confirmation sigma threshold remains explicitly overrideable by `KIKO_SURFACE_MAX_CONFIRMED_STD_DEV_M`, but inherits voxel-size changes automatically when not overridden.

## Touched Files

- `crates/kiko-slam/src/surface_map.rs`

## New Or Changed Metrics

- none

## New Or Changed Solver Outputs

- none

## Tests Added

- `surface_map::tests::high_uncertainty_voxel_stays_unconfirmed_even_with_support`
- `surface_map::tests::extract_confirmed_prefers_lower_std_dev_when_capped`
- `surface_map::tests::three_consistent_support_views_confirmed` updated to exercise the new confirmation sigma gate

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/surface_map.rs`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- reviewer agent `Euclid`: `accept`

## Known Risks Or Deferred Follow-Ups

- `std_dev <= voxel_size` is still a scalar proxy, not a fully anisotropic surface-confidence test.
- Stability-first truncation is intentionally a render policy, not a map-state mutation, so strongly capped renders will be biased toward easy low-noise voxels unless operators also read the rendered-vs-confirmed counts already exported in `viz.rs`.
- There is still no calibrated stereo uncertainty model in this path; upstream `StableSurfacePoint::position_variance` remains limited by the current fixed disparity-noise prior and off-axis geometry model.

## Findings

- `major`: confirmation semantics are materially stronger because posterior uncertainty is now part of the state predicate, not just an internal fusion byproduct.
- `major`: the render cap no longer discards the best voxels arbitrarily through stride thinning; it preserves the most stable confirmed beliefs first.
- `medium`: tying the default confirmation sigma to voxel size makes the surface map's claimed resolution honest by default, while preserving an explicit override for expert tuning.

## Invariant Verdict

- strengthened: confirmation now reflects support, consistency, and uncertainty; the default confirmation threshold stays coupled to voxel resolution unless explicitly overridden; capped rendering remains deterministic while preferring more stable confirmed voxels.
- weakened or ambiguous: posterior uncertainty is still summarized as a scalar sigma rather than a directional surface uncertainty model.

## Metric Verdict

- trustworthy: none added; the change improves state semantics rather than introducing a new metric stream.
- partial or misleading: visual appearance of the rendered surface remains a partial view under heavy capping and must be read together with the existing confirmed-vs-rendered counts.

## Test Verdict

- covered: confirmation rejects high-uncertainty voxels even when support count is satisfied; capped extraction prefers the lowest-uncertainty confirmed voxel; default and `vio` package test suites remain green.
- missing: a focused observability test around `diagnostics/surface/rendered_confirmed_voxels`; a future anisotropic uncertainty test once surface confidence stops being scalar.

## Merge Decision

`accept`
