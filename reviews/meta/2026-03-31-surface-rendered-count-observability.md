# Meta Review

## Commit Goal

Keep the stable-surface render path observability-honest by exporting the number of rendered confirmed voxels separately from the number of confirmed voxels in the map, and keep the CLI help aligned with the newly added `KIKO_SURFACE_MAX_CONFIRMED_STD_DEV_M` knob.

## Current Repo Starting Point

This tranche starts immediately after `7a5732b` on `measurement-system`, where stable-surface uncertainty became row-mismatch-aware. The map already distinguished confirmed voxels from rendered output in behavior, but that rendered-vs-confirmed split was still easy to miss operationally: the Rerun log path did not export rendered confirmed voxel count explicitly, and the SLAM CLI help string still omitted the confirmation-sigma environment variable.

## Previous Invariants

- The stable-surface map exported `confirmed_voxels`, but not the separate count of confirmed voxels that survived render capping.
- The stderr summary for stable-surface output did not print rendered count explicitly.
- `KIKO_SURFACE_MAX_CONFIRMED_STD_DEV_M` existed in code but was not listed in the SLAM CLI expert tuning help text.

## New Invariants Claimed

- `diagnostics/surface/rendered_confirmed_voxels` is logged explicitly beside `confirmed_voxels`.
- The stderr surface summary now prints `rendered=` separately from `confirmed=`.
- `SLAM_ENV_HELP` now includes `KIKO_SURFACE_MAX_CONFIRMED_STD_DEV_M`, and a unit test pins that help entry in place.

## Touched Files

- `crates/kiko-slam/src/viz.rs`
- `crates/kiko-slam/src/bin/kiko_slam/slam.rs`

## New Or Changed Metrics

- `diagnostics/surface/rendered_confirmed_voxels`

## New Or Changed Solver Outputs

- none

## Tests Added

- `slam::tests::slam_env_help_mentions_confirmed_surface_sigma_env`

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/bin/kiko_slam/slam.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/viz.rs`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- reviewer agent `Euclid`: `accept`

## Known Risks Or Deferred Follow-Ups

- `rendered_confirmed_voxels` is honestly named, but dashboards must still show it alongside `confirmed_voxels` or operators may over-read the rendered subset as authoritative map size.
- The stderr summary is still free-text observability rather than a typed metric stream.
- The new help-string test only pins the presence of `KIKO_SURFACE_MAX_CONFIRMED_STD_DEV_M`; it does not verify broader help completeness or formatting.

## Findings

- `major`: the rendered-vs-confirmed distinction is now explicit in the metric stream, which prevents capped renders from silently masquerading as full map state.
- `medium`: the human-readable stderr summary now matches the Rerun scalar split, reducing operator confusion when render caps are active.
- `medium`: the new help-string test turns the confirmation-sigma env knob from tribal knowledge into a pinned CLI contract.

## Invariant Verdict

- strengthened: rendered stable-surface output is now explicitly distinguishable from confirmed map state; the confirmation-sigma env knob is now part of the tested CLI contract.
- weakened or ambiguous: none in this small follow-up.

## Metric Verdict

- trustworthy: `diagnostics/surface/rendered_confirmed_voxels` is an honest rendered-subset count and does not pretend to be map size.
- partial or misleading: stderr output remains secondary and untyped.

## Test Verdict

- covered: the CLI help string now pins the confirmation-sigma env knob; full default and `vio` package suites remain green after the observability/help changes.
- missing: a focused test around `viz.rs` Rerun export of `rendered_confirmed_voxels`.

## Merge Decision

`accept`
