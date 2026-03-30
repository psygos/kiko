# Meta Review

## Commit Goal

Make stable-surface support accounting fully observable and semantically honest by exporting the full integration trio for each frame: raw accepted observations, novel grouped support views, and grouped redundant views ignored as same-ray evidence.

## Current Repo Starting Point

This tranche starts immediately after `58c1b6e`, `1dd01f3`, and `7a5732b`, plus the already-landed support-view diversity commit on `measurement-system`. At the start of this change, the estimator-side semantics were already improved: same-view repeats no longer tightened the posterior or increased support. But the observability layer still only logged accepted novel grouped views, and later added ignored redundant grouped views, without exporting the raw integrated observation count alongside them. That made the per-frame evidence accounting only partially visible.

## Previous Invariants

- `support_views_integrated` counted only grouped views that actually contributed novel support.
- `redundant_grouped_views_ignored` was computed in the estimator path but not yet part of the previous committed state.
- `raw_observations_integrated` existed in the batch summary but was not exported through Rerun, so raw/support/redundant accounting was not visible as one coherent telemetry set.
- Repeated same-view batches increased `raw_observations` but did not change support or posterior sigma.

## New Invariants Claimed

- Per-frame stable-surface integration now exports the full raw/support/redundant trio from one helper in `viz.rs`.
- The logged counters are explicitly non-additive by semantics: raw observations are per-sample counts, while the other two are grouped-view counts.
- Repeated same-view regressions remain pinned at the estimator layer, and the exact emitted observability trio is now pinned at the telemetry layer.

## Touched Files

- `crates/kiko-slam/src/surface_map.rs`
- `crates/kiko-slam/src/viz.rs`

## New Or Changed Metrics

- `diagnostics/surface/integrated_raw_observations`
- `diagnostics/surface/integrated_support_views`
- `diagnostics/surface/redundant_grouped_views_ignored`

## New Or Changed Solver Outputs

- none

## Tests Added

- `surface_map::tests::repeated_same_view_batches_do_not_accumulate_support_views`
- `viz::tests::surface_integration_scalars_export_honest_support_accounting`

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/viz.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/surface_map.rs`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- reviewer agent `Euclid`: `accept`

## Known Risks Or Deferred Follow-Ups

- `integrated_raw_observations` is a raw-observation count, while `integrated_support_views` and `redundant_grouped_views_ignored` are grouped-view counts. They are honest together but must not be treated as a single additive partition.
- Terminal stderr summaries still do not print the new per-frame integration trio; Rerun is the authoritative observability path for these counters.
- The viewpoint novelty test remains an angular proxy derived from voxel size and range, not a full surface-normal-aware independence test.

## Findings

- `major`: the previous observability blind spot is closed; per-frame stable-surface telemetry now exposes whether a frame had raw evidence but little or no novel support because grouped views were redundant.
- `major`: the regression surface is stronger because both estimator behavior and telemetry paths are now pinned by tests.
- `medium`: the new counters have intentionally different denominators, so dashboards and operators must keep their meanings distinct.

## Invariant Verdict

- strengthened: support accounting is now observable end-to-end; repeated same-view evidence cannot silently masquerade as new support in either the estimator or telemetry.
- weakened or ambiguous: no new weakness introduced; the remaining ambiguity is only the already-known angular novelty proxy.

## Metric Verdict

- trustworthy: `integrated_raw_observations`, `integrated_support_views`, and `redundant_grouped_views_ignored` now reflect their actual estimator semantics and are emitted together from one helper.
- partial or misleading: none new, provided consumers do not interpret raw-observation counts and grouped-view counts as additive categories.

## Test Verdict

- covered: repeated same-view estimator behavior; exact emitted scalar paths and values; full default and `vio` package test suites.
- missing: no terminal-summary regression for the new trio, because Rerun remains the authoritative path.

## Merge Decision

`accept`
