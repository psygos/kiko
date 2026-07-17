# Host core-hardening evidence — 2026-07-17

## Scope

- Repository: Kiko canonical worktree
- Branch: `codex/core-hardening`
- Requested base: `27d63e7b91fd1b1903b37233a13e2efc083f6f3c`
- Occupancy code benchmarked at: `d7146e9faf28ed0280a98fb4341c77e8b7765ef4`
- Benchmark worktree state: clean
- Workspace topology audited: `kiko-slam`, `oak-sys`, `robot-protocol`, `robot-server`, `robot-control-client`, and `embedded`

The new 2D map is deterministic geometric occupancy derived from calibrated depth. It is not a
learned occupancy network. It keeps the existing SLAM architecture and consumes authoritative
keyframe poses and corrections from that architecture.

## Reproducible occupancy benchmark

Command:

```text
cargo bench --locked -p kiko-slam --bench occupancy_mapping
```

Environment and workload reported by the benchmark itself:

```text
git_commit=d7146e9faf28ed0280a98fb4341c77e8b7765ef4
git_worktree=clean
rustc="rustc 1.97.0 (2d8144b78 2026-07-07)"
os=macos
arch=aarch64
short=false
depth=160x120
sample_block_px=4
sampled_input_blocks_per_integrate=1200
grid=400x400
resolution_m=0.05
samples=7
warmup_rounds=2
integration_remove_cycles_per_sample=40
snapshot_calls_per_sample=200
```

Median results:

```text
integration_remove_cycle_median_ns=846815.6
integration_remove_cycles_per_s=1180.9
snapshot_call_median_ns=9828.5
snapshot_output_cells_per_s=16279113541.2
```

An integration/remove cycle integrates the fixed synthetic depth image into an empty map and then
removes that keyframe exactly. A snapshot call classifies and returns all 160,000 grid cells. The
timed regions exclude setup, repository inspection, hashing, and output formatting.

These are absolute measurements for the stated commit and machine, not a before/after performance
claim. Cross-machine comparisons require the same commit, toolchain, profile, workload, and power
conditions.

The occupancy scratch design uses one byte of visit state per grid cell plus one `usize` entry per
touched cell. At the configured four-million-cell safety limit on this 64-bit host, the worst-case
vector payload is 36,000,000 bytes, excluding allocator bookkeeping. This is a structural bound,
not an RSS measurement or timing claim.

## Verification

- `cargo test --locked -p kiko-slam`: 639 library tests and 25 default binary tests passed; the
  compile-fail doctest passed and one backend example remained intentionally ignored.
- `cargo check --locked -p kiko-slam --all-targets`: passed.
- `cargo clippy --locked -p kiko-slam --all-targets -- -D warnings`: passed.
- Stub-backed `record` feature gate: all-target check passed, 35/35 host binary tests passed, and
  all-target Clippy with warnings denied passed. The real `oak-sys` path and unchanged lockfile were
  restored afterward.
- Header-free current-source `oak-sys` boundary tests: 15/15 passed with warnings denied; build
  discovery tests: 2/2 passed; generated CXX header and source were byte-identical to the prior
  bridge artifacts.
- Typed comms contract: `robot-protocol` 10/10, `robot-server` 14/14, and
  `robot-control-client` 6/6 tests passed.
- `cargo check --locked -p embedded --target thumbv7em-none-eabihf`: passed.
- `cargo fmt --all -- --check`, `git diff --check`, and the unchanged `Cargo.lock` check passed.
- The occupancy traversal suite includes a 200,000-case randomized DDA/reference regression with
  no mismatch.

## Rerun contract

Occupancy is logged under `world/map2d` with an explicit occupancy-to-world `ParentFromChild`
transform, pixel-to-metre grid placement, class annotations, exact map identifier and revision,
and the same `capture_ns` timeline as the corresponding host data. The metric outline supports a
dedicated 2D Rerun view. The segmentation is deliberately not represented as a pinhole camera
image or a textured 3D floor.

Source and serialization tests verify the pinned Rerun 0.27.x contract. A live viewer/gRPC session
was not exercised, so actual viewer auto-spawn and rendering remain unverified.

## Explicit verification limits

- Native DepthAI/OpenCV compilation and physical OAK behavior were not verified because the local
  SDK headers, libraries, and device were unavailable.
- Linux aarch64 cross-checking was attempted but the host lacks a Linux cross-C toolchain/sysroot:
  the third-party `ring` build first lacked `aarch64-linux-gnu-gcc`, and the Clang attempt then
  lacked the target `assert.h`. Linux aarch64 compilation therefore remains unverified.
- Jetson GPU benchmarking, power/thermal work, physical STM32 validation, and deployment were not
  performed; those remain in the Nano/hardware lane.
