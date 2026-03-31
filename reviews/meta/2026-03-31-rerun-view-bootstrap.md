# Meta Review

## Commit Goal

Make dataset `viz` and `slam` runs truthful and usable when Rerun is not already connected: default local runs should try to open or reuse a viewer instead of silently streaming to nowhere, explicit saved-recording debugging should accept the intuitive `--rerun-save` alias, dataset commands should continue headless with a concrete error when viewer bootstrap fails, and sink-level tests should prove that the base stereo view entities are actually emitted.

## Current Repo Starting Point

The branch already had stable-surface logging and typed measurement work in flight, but dataset-mode Rerun setup still defaulted to `connect_grpc()`, which only works when a viewer is already listening. `live` mode degraded to headless on recorder initialization failure, while dataset `viz` and `slam` did not. The sink had coverage for surface-map behavior, but no regression proving that `view/left` and `view/right` were present in the emitted log stream.

## Previous Invariants

- `RerunSink::log_frames` and `RerunSink::log_with_points` should emit stereo image data under stable entity paths.
- Dataset `viz` and `slam` should not change estimator behavior when visualization is unavailable.
- Rerun diagnostics should not imply that a live viewer exists when no viewer is reachable.

## New Invariants Claimed

- Default local Rerun initialization now attempts to spawn or reuse a local viewer instead of silently assuming one is already running.
- Dataset `viz` and `slam` explicitly continue headless with a concrete initialization error when Rerun bootstrap fails.
- SLAM falls back to raw stereo view logging if `VizPacket::try_new` fails, instead of silently dropping the frame visualization.
- The CLI accepts `--rerun-save` as a visible alias for `.rrd` capture debugging.
- Sink tests prove that the emitted in-memory log stream contains `/view/left` and `/view/right`.

## Touched Files

- `crates/kiko-slam/src/bin/kiko_slam/args.rs`
- `crates/kiko-slam/src/bin/kiko_slam/main.rs`
- `crates/kiko-slam/src/bin/kiko_slam/viz.rs`
- `crates/kiko-slam/src/bin/kiko_slam/slam.rs`
- `crates/kiko-slam/src/viz.rs`

## New Or Changed Metrics

- none

## New Or Changed Solver Outputs

- none

## Tests Added

- `viz::tests::log_frames_emits_left_and_right_view_entities`
- `tests::rerun_save_alias_parses_for_viz`

## Tests Run

- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `bash /home/makerspace/kiko-vio/scripts/check_meta_review.sh`
- `cargo run -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml -- viz --backend cpu --max-pairs 1 --save-rrd /tmp/kiko-rerun-check /home/makerspace/full_slam_lab`
- `strings /tmp/kiko-rerun-check | rg '/view/left|/view/right|/view/matches'`
- `cargo run -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml -- viz --backend cpu --max-pairs 1 /home/makerspace/full_slam_lab`

## Known Risks Or Deferred Follow-Ups

- If no `rerun` viewer binary is installed, default local runs still end up headless; that is now explicit and truthful, but users still need to install the viewer or use `--save-rrd`.
- The saved-recording smoke test confirmed `/view/left` and `/view/right`; it did not prove a `view/matches` entity on disk for the sampled frame.
- Viewer layout/blueprint state is still external to this change; this commit fixes bootstrap and truthfulness, not the entire viewer UX.

## Findings

- none

## Invariant Verdict

- strengthened: default local visualization no longer pretends a viewer exists, dataset paths no longer fail hard on viewer bootstrap, and base stereo view logging is sink-tested.
- weakened or ambiguous: none.

## Metric Verdict

- trustworthy: initialization status is now surfaced honestly to stderr, and headless fallback no longer implies live visualization.
- partial or misleading: none introduced by this change.

## Test Verdict

- covered: sink-level left/right entity emission, CLI alias parsing, full package tests with and without `vio`, saved-recording smoke run, and headless fallback smoke run on a machine without `rerun` in `PATH`.
- missing: no automated integration test currently asserts viewer-process spawn behavior itself.

## Merge Decision

`accept`
