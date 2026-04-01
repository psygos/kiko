# Meta Review

## Commit Goal

Make the stable-surface path visibly present in Rerun even when no retained stable-surface points or confirmed voxels exist yet. The change should increase operator observability only: raw measured stereo support should become visible every stereo-matched frame, while persistent surface-map fusion must remain keyframe-gated and pose-gated.

## Current Repo Starting Point

This tranche starts from `b3486f6` on `measurement-system`, where the stable-surface path already exposed pending/rejected voxel classes and frame candidates, but the viewer could still look empty when `generate_stable_surface_points` retained zero points for a frame or when the current frame was not a keyframe. In that state, the operator could have valid measured stereo support but still see no 3D surface-like entity.

## Previous Invariants

- Stable-surface map fusion was conservative and should only mutate from explicitly integrated frames.
- `world/stable_surface_debug/frame_candidates` depended on retained stable-surface points, so a frame with raw stereo support but zero retained points could still look visually empty.
- Dataset SLAM only extracted surface samples for keyframes, so non-keyframes could never emit any surface debug entity.

## New Invariants Claimed

- Dataset SLAM extracts measured stereo samples for every stereo-matched frame when `KIKO_DENSE_CLOUD=true`, not only for keyframes.
- Persistent surface-map fusion is still requested only on keyframes and still requires the pose-quality gate to accept integration.
- `RerunSink::log_surface_observations` now always logs `world/stable_surface_debug/frame_raw_observations` from measured stereo support in map frame, independent of stable-point retention or keyframe integration.
- `world/stable_surface_debug/frame_candidates` remains the stricter retained-point view; `frame_raw_observations` is explicitly debug-only and does not alter estimator state.

## Touched Files

- `crates/kiko-slam/src/bin/kiko_slam/slam.rs`
- `crates/kiko-slam/src/viz.rs`

## New Or Changed Metrics

- `diagnostics/surface/frame_gate/integration_requested`

## New Or Changed Solver Outputs

- none

## Tests Added

- `viz::tests::log_surface_observations_logs_raw_frame_observations_without_retained_candidates`

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/viz.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/bin/kiko_slam/slam.rs`
- `cargo test -p kiko-slam log_surface_observations_logs_debug_entities_before_confirmation --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam log_surface_observations_logs_raw_frame_observations_without_retained_candidates --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo build --release -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo build --release -p kiko-slam --features "vio,ort-cuda" --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- runtime smoke attempt with dataset replay and `--save-rrd`, blocked before inference by `BackendUnavailable { requested: Cuda, selected: Cpu }`

## Known Risks Or Deferred Follow-Ups

- `frame_raw_observations` is intentionally a debug-only measurement view. It is not filtered by stable-surface uncertainty thresholds and must not be mistaken for fused or confirmed structure.
- The Jetson runtime in this shell still fails to select the CUDA provider even after a CUDA-feature rebuild, so this review cannot claim a successful end-to-end remote-viewer validation on the exact requested launch command.
- Frames with zero extracted stereo samples will still have no surface debug points; this change only guarantees visibility once measured stereo support exists.

## Findings

- none: the change improves viewer honesty and early visibility without weakening keyframe-only fusion or pose-quality gating.

## Invariant Verdict

- strengthened: raw measured stereo support is now separately visible from retained stable-surface candidates; keyframe-only mutation semantics remain explicit via `integration_requested`.
- weakened or ambiguous: the viewer now contains one more debug-only surface-like entity, so operator guidance must keep `frame_raw_observations` clearly distinguished from the persistent surface map.

## Metric Verdict

- trustworthy: `diagnostics/surface/frame_gate/integration_requested` is honestly named and matches the actual map-mutation request boundary.
- partial or misleading: `frame_raw_observations` is visual output, not a metric; it should not be interpreted as estimator confidence.

## Test Verdict

- covered: non-keyframe visualization path, retained-candidate path, visual-only non-mutating path, and the full lib/bin/doc test suites.
- missing: successful CUDA-backed end-to-end dataset replay on this shell environment.

## Merge Decision

`accept with follow-up`
