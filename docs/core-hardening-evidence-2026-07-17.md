# Host core-hardening evidence — 2026-07-17

## Scope

- Repository: Kiko canonical worktree
- Branch: `codex/core-hardening`
- Requested base: `27d63e7b91fd1b1903b37233a13e2efc083f6f3c`
- Occupancy code benchmarked at: `d7146e9faf28ed0280a98fb4341c77e8b7765ef4`
- Benchmark worktree state: clean
- Workspace topology audited at that revision: `kiko-slam`, `oak-sys`,
  `robot-protocol`, `robot-server`, the now-retired direct
  `robot-control-client`, and `embedded`.

The new 2D map is deterministic geometric occupancy derived from calibrated depth. It is not a
learned occupancy network. It keeps the existing SLAM architecture and consumes authoritative
keyframe poses and corrections from that architecture.

## Replay-bound navigation shadow closure

The host closure is transport-free shadow execution, not robot actuation. The live boundary takes
`--navigation-config`, `--navigation-goal X_M,Y_M`, and `--navigation-record` as an all-or-none
set. It parses the JSON and goal once, requires depth, IMU, rectified stereo, and dense occupancy,
and records the coordinator's actual admission order in a dataset-bound ingress sidecar. Thread
scheduling itself is not claimed deterministic. The journal is authoritative for admitted order
and identity rather than a reconstructed schedule or Rerun output; because it omits tracker-derived
outputs and the live deadline clock, it is not a deterministic end-to-end SLAM or MPC replay log.

Every admitted control tick reaches the fail-closed safety supervisor and either produces one
bounded shadow command record or ends the session on a fatal evidence-recording error. The shadow
session exposes `motor_packets_sent = 0` and has no command transport. STOP is recorded for stale
or missing observations, identity/provenance mismatch, blocked trajectories, infeasible solutions,
or deadline failures. Failure to append the shadow decision is fatal and does not fabricate a STOP
record. A global map admitted before the first visual localization anchor is deliberately not
rebound after the fact; planning waits for a later map revision.

The local costmap clears the raw cells conservatively occupied by the robot's current footprint,
then inflates observed obstacles. It does not mark the clearance ring or unseen exterior free.
Consequently, a forward-only depth observation can leave rear/side clearance blocked and stop the
first rollout even though the body's current cell is free. This is the intended unknown-is-blocked
contract, not evidence that the synthetic fixture can drive the physical robot.

Focused host checks at the navigation library state through `a4213ff`:

```text
cargo test --locked -p kiko-slam --lib navigation::shadow_config::tests
18 passed; 0 failed

cargo test --locked -p kiko-slam --lib navigation::ingress::tests
30 passed; 0 failed

cargo test --locked -p kiko-slam --lib navigation::coordinator::tests
12 passed; 0 failed

cargo test --locked -p kiko-slam --lib navigation::safety::tests
11 passed; 0 failed

cargo test --locked -p kiko-slam --lib navigation::local_costmap::tests
15 passed; 0 failed
```

At the recorded revision, `configs/navigation-shadow-v1.example.json` was valid JSON and was passed
through the public `ShadowNavigationConfigV1::parse_json` boundary with an explicit 640x400 runtime
depth-camera model. The parser accepted the 4,695-byte fixture at implementation commit `3968993`;
those exact bytes have SHA-256
`2c08ea565ca59935669e088d35c7c826ec74395329e2f603812a80c1e9bd98b0`.
V1 evolved after that recorded test. Its final 4,735-byte form is retained at
`crates/kiko-slam/testdata/legacy-navigation-shadow-v1.example.json` with SHA-256
`9cb77b15ae38acd6b21c56f65687f6c5683415348de4851a299b8637522c370f`.
Navigation V2 subsequently retired that configuration path. Both V1 fixtures
identify themselves as synthetic, non-actuating, non-physically-validated
schema examples; their values are not Kiko plant identification or sensor
calibration.

Rerun remains output-only. The live adapter submits best-effort entries on explicit capture and
navigation timelines for the CLI goal, poses/transforms, map and local grid, path, predicted
trajectory, solver/safety outcome, requested PWM, and zero-motor-packet counter. Its bounded
diagnostic queue can omit a newest entry under backpressure. It does not implement or claim a
viewer map-click callback, and Rerun failure cannot authorize motion.

Final macOS aarch64 host verification for implementation commit `ebae23f`:

```text
cargo test --locked -p kiko-slam --quiet
875 library tests passed; 31 default binary tests passed; compile-fail doctest passed;
1 backend example doctest remained intentionally ignored

cargo check --locked -p kiko-slam --all-targets
passed

cargo clippy --locked -p kiko-slam --all-targets -- -D warnings
passed

cargo fmt --all -- --check
passed

git diff --check
passed
```

The final `record`-feature source also passed strict Clippy for the `kiko-slam` binary and tests in
an isolated archive with a temporary build-script early return that bypassed unavailable native
OAK discovery, C++ compilation, and linking. This proves the Rust feature path type-checks with
warnings denied; it is not a native DepthAI/OpenCV link or device-runtime result.

## Reproducible navigation shadow benchmark

Command:

```text
cargo bench --locked -p kiko-slam --bench navigation_shadow
```

Representative host workload and absolute result:

```text
git_commit=44d3689bad3dac5de290e88835b8eb5580d53fff
git_worktree=dirty
rustc="rustc 1.97.0 (2d8144b78 2026-07-07)"
os=macos
arch=aarch64
cpu_model="Apple M4 Max"
short=false
samples=9
warmup_rounds=3
iterations_per_sample=10000
total_timed_iterations=90000
horizon_steps=8
shadow_retained_records=64
median_ns_per_iteration=591.0
iterations_per_second=1692190.5
behavior_digest=0xa36205e266434cac
stable_timed_digest=0x428bf62449d07aa5
motor_packets_sent=0
allocations=not_instrumented
successful_mpc_timing=false
```

The timed scope is path-reference construction plus the public-API
ready-but-unproven-depth fail-closed safety admission. Public APIs intentionally prevent an
external benchmark from forging time-aligned odometry or local-costmap provenance, so this run
does not time a successful collision-checked MPC solve or final revalidation. It also makes no
allocation or before/after performance claim. The worktree was dirty because the live binary and
closure documentation were under review; the benchmark and navigation library source were the
tracked `44d3689` state.

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

## Earlier occupancy-lane verification

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
  the since-retired direct `robot-control-client` 6/6 tests passed. It was
  removed after the integrated loopback operator/agent console became the sole
  production control ingress.
- `cargo check --locked -p embedded --target thumbv7em-none-eabihf`: passed.
- `cargo fmt --all -- --check`, `git diff --check`, and the unchanged `Cargo.lock` check passed.
- The occupancy traversal suite includes a 200,000-case randomized DDA/reference regression with
  no mismatch.

## Occupancy Rerun contract

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
- A published navigation dataset proves completed payload writers and a synchronized, revalidated
  ingress sidecar. OAK close is post-publication cleanup: its failure is returned separately and
  does not invalidate the complete artifact.
- Post-activation workers retain the existing `std::thread::spawn` panic boundary. OS
  resource-exhaustion spawn failure and a combined multi-worker fault-injection matrix were not
  converted into or verified as typed abort paths in this closure.
- The recording structurally binds payloads to admitted ingress by identity, path, count, and
  length; it is not cryptographic integrity, authentication, or origin proof. It also does not
  embed the navigation JSON or code revision, so reproduction requires the same external
  configuration and software revision.
- Linux aarch64 cross-checking was attempted but the host lacks a Linux cross-C toolchain/sysroot:
  the third-party `ring` build first lacked `aarch64-linux-gnu-gcc`, and the Clang attempt then
  lacked the target `assert.h`. Linux aarch64 compilation therefore remains unverified.
- Jetson GPU benchmarking, power/thermal work, physical STM32 validation, and deployment were not
  performed; those remain in the Nano/hardware lane.
