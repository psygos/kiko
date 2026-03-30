# Meta Review

## Commit Goal

Make stable-surface support geometrically meaningful across time by requiring distinct viewing rays before a voxel gains support, and prevent repeated near-identical views from tightening the posterior or maturing the voxel.

## Current Repo Starting Point

This tranche starts immediately after `58c1b6e` on `measurement-system`. The stable surface map already grouped same-batch same-voxel observations, required posterior sigma for confirmation, and used row-mismatch-aware per-observation uncertainty. But support across batches was still too permissive: the same voxel could become confirmed from nearly identical repeated views, and identical-view repeats still tightened the posterior even after support counting was later made stricter.

## Previous Invariants

- `support_views` effectively counted grouped per-batch voxel hits, even when those hits came from nearly identical viewing rays.
- Repeated same-direction views could shrink posterior sigma and help a voxel confirm.
- `SurfaceBatchIntegrationSummary::support_views_integrated` counted grouped evidence items, not necessarily geometrically novel support.
- `mean_confirmed_support_views` reflected grouped support count, not distinct viewpoint support.

## New Invariants Claimed

- A voxel gains support only when the incoming viewing ray is novel relative to the voxel’s angular extent at that range.
- Redundant same-direction views no longer increment support or tighten the posterior; they only increase `raw_observations`.
- `support_views` now means distinct support viewpoints rather than grouped batch count.
- `SurfaceBatchIntegrationSummary::support_views_integrated` now counts only novel support views actually accepted into the belief state.

## Touched Files

- `crates/kiko-slam/src/surface_map.rs`

## New Or Changed Metrics

- `SurfaceMapSummary::mean_confirmed_support_views` now reflects distinct viewpoint support rather than grouped support count.
- `SurfaceBatchIntegrationSummary::support_views_integrated` now reflects accepted novel support views rather than all grouped voxel evidences.

## New Or Changed Solver Outputs

- none

## Tests Added

- `surface_map::tests::repeated_same_view_batches_do_not_accumulate_support_views`
- `surface_map::tests::three_consistent_distinct_support_views_confirmed`
- existing confirmation tests updated to use distinct viewpoints where support is intended

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/surface_map.rs`
- `cargo test -p kiko-slam surface_map:: --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- reviewer agent `Euclid`: `accept`

## Known Risks Or Deferred Follow-Ups

- The novelty threshold is still an orientation-agnostic angular proxy derived from voxel size and range; it does not model surface normal or anisotropic uncertainty.
- Redundant views are now intentionally ignored by the posterior update, which is robust but conservative; only `raw_observations` records their continued presence.
- There is still no direct regression test that redundant views increase `raw_observations` while leaving support count unchanged.

## Findings

- `major`: the earlier false-confidence path is closed because same-direction repeats no longer tighten `std_dev` or increase support.
- `major`: support semantics are materially more truthful now; a voxel must be seen from distinct rays relative to its claimed spatial resolution before it matures.
- `medium`: summary metrics around support count now changed meaning in a more honest direction, so downstream consumers should treat them as distinct-view counts rather than grouped-hit counts.

## Invariant Verdict

- strengthened: support now encodes distinct viewpoint evidence; repeated same-view exposure cannot mature a voxel; per-support consistency semantics once again match the contributing updates.
- weakened or ambiguous: novelty remains a scalar angular proxy rather than a full surface-geometry test.

## Metric Verdict

- trustworthy: `support_views`, `mean_confirmed_support_views`, and `support_views_integrated` are now support-honest with respect to distinct-view semantics.
- partial or misleading: `raw_observations` still includes redundant views, so it must not be read as independent evidence count.

## Test Verdict

- covered: repeated same-view batches no longer increase support or tighten sigma; distinct viewpoints still confirm stable voxels; full default and `vio` package suites remain green.
- missing: a direct raw-observation regression for redundant views; tests for edge cases near grazing incidence or strongly anisotropic geometry.

## Merge Decision

`accept`
