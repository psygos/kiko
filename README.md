<p align="center">
  <img src="assets/kiko-logo.svg" alt="kiko" width="200">
</p>

Kiko is a social robot combining custom SLAM and an expression engine. The SLAM implementation is designed for model hot-swap as vision models improve, and Kiko improves with them. Built entirely in Rust.

**Status:** Early development. The host pipeline includes stereo visual odometry, local bundle
adjustment, map/covisibility maintenance, relocalization, and loop closure. Hardware-backed
recording and TSDF/ESDF reconstruction still depend on external native components. The host also
provides deterministic geometric 2D occupancy mapping from calibrated depth; it is not a learned
occupancy network.

## Structure

```
kiko/
├── crates/
│   ├── kiko-slam/    # SLAM, visual odometry, feature detection
│   └── oak-sys/      # OAK-D camera FFI bindings
├── comms/
│   ├── robot-server/ # Communication hub (UDP, Serial, HTTP)
│   └── desktop-client/ # Tauri control UI
└── embedded/         # STM32F446 firmware
```

## Quick Start

Record a dataset (requires OAK-D):

```
cargo run -p kiko-slam --features record -- record recordings/<name>
```

The `record` feature requires DepthAI headers and libraries. Set `DEPTHAI_INCLUDE` and
`DEPTHAI_LIB` to OS path lists when they are not installed in the build script's standard
locations. The macOS build also resolves OpenCV through `OPENCV_INCLUDE` and `OPENCV_LIB`; the
Linux build does not consume those overrides. Explicit dependency paths are authoritative and fail
rather than silently compiling against a different installation.

Live match visualization (requires OAK-D + Rerun viewer):

```
cargo run -p kiko-slam --features record -- live
```

Run the encoderless host navigator in transport-free shadow mode while recording its exact sensor
payloads and coordinator admissions:

```
KIKO_LIVE_DEPTH=true \
KIKO_DENSE=true \
cargo run --locked -p kiko-slam --features record -- live \
  --imu-rate-hz 200 \
  --navigation-config configs/navigation-shadow-v1.example.json \
  --navigation-goal 2.0,1.0 \
  --navigation-record recordings/shadow-example
```

The checked-in configuration is a synthetic schema example, not physical calibration or an
identified Kiko plant. This mode computes and displays requested PWM but owns no motor transport
and sends zero motor packets. The goal is a typed map-frame point; the pinned Rerun SDK is
output-only and does not provide a map-click callback. See
[the shadow-navigation architecture](docs/navigation-shadow-architecture.md) for frames, dynamic
obstacle behavior, replay evidence, and the exact closure boundary.

Visualize stereo matches on a dataset:

```
cargo run -p kiko-slam -- viz recordings/<name>
```

Run visual odometry on a dataset (Rerun viewer):

```
cargo run -p kiko-slam -- viz --odometry recordings/<name>
```

Enable the 2D map only after declaring the physical floor assumption:

```
KIKO_DENSE=true \
KIKO_OCCUPANCY_ASSUME_LEVEL_OPTICAL_WORLD=true \
KIKO_OCCUPANCY_CAMERA_HEIGHT_M=0.42 \
cargo run -p kiko-slam -- viz --odometry recordings/<name>
```

For live mapping, also set `KIKO_LIVE_DEPTH=true`. New recordings identify the depth optical
frame. Legacy datasets without that metadata are rejected unless
`KIKO_OCCUPANCY_ASSUME_RECTIFIED_LEFT=true` is explicitly set from known calibration evidence.

Benchmark a dataset:

```
cargo run -p kiko-slam -- bench recordings/<name>
```

Run the dependency-free occupancy microbenchmark with:

```
cargo bench -p kiko-slam --bench occupancy_mapping
```

## Config (flags or env)

**Inference:**

- `ORT_DYLIB_PATH` — path to an ONNX Runtime 1.24+ shared library; otherwise the platform loader searches for `libonnxruntime.so`, `onnxruntime.dll`, or `libonnxruntime.dylib`
- `--downscale` / `KIKO_DOWNSCALE` — positive integer input downscale factor; it must divide the input width and height exactly
- `--max-keypoints` / `KIKO_MAX_KEYPOINTS` — max keypoints per frame (default 1024)
- `--backend` / `KIKO_BACKEND` — inference backend for both models
- `--superpoint-backend` / `KIKO_SUPERPOINT_BACKEND` — SuperPoint backend override
- `--lightglue-backend` / `KIKO_LIGHTGLUE_BACKEND` — LightGlue backend override
- `--superpoint-model` / `KIKO_SUPERPOINT_MODEL` — custom SuperPoint ONNX path
- `--lightglue-model` / `KIKO_LIGHTGLUE_MODEL` — custom LightGlue ONNX path
- `KIKO_ORT_INTRA_THREADS` — ONNX intra-op threads; unset or `0` selects `max(2, available_parallelism / 2)` for CPU sessions and 2 for accelerator sessions, while an explicit count must be at least 2 because Kiko uses asynchronous inference
- `KIKO_ORT_RUN_WARN_MS` — slow-inference warning threshold in milliseconds (default 200); `0` warns for every nonzero observed duration
- `KIKO_ORT_RUN_TIMEOUT_MS` — strict inference deadline in milliseconds (default 5000, must be greater than zero and at least the warning threshold); successful completion observed at or after the deadline returns an error but leaves the session usable, while a run still pending at the deadline makes that session fail-stop because ONNX Runtime cancellation is nonblocking

**Visualization:**

- `--rerun-decimation` / `KIKO_RERUN_DECIMATION` — image decimation for Rerun
- `--rerun-finish-timeout-ms` / `KIKO_RERUN_FINISH_TIMEOUT_MS` — Rerun sink-flush timeout in milliseconds (default 5000); success confirms the calling thread's prior data reached the configured sink, not that a viewer consumed it
- `--odometry` / `KIKO_VIZ_ODOMETRY` — enable visual odometry in viz mode

**2D occupancy mapping:**

- `KIKO_DENSE` — enable host occupancy mapping; offline mode requires `viz --odometry`, and live mode also requires `KIKO_LIVE_DEPTH=true`
- `KIKO_OCCUPANCY_ASSUME_LEVEL_OPTICAL_WORLD` — required explicit assertion that the initial visual world is a level optical frame (`+x` right, `+y` down, `+z` forward); occupancy uses `[x, y, height] = [world_x, world_z, camera_height - world_y]`
- `KIKO_OCCUPANCY_CAMERA_HEIGHT_M` — required camera height above the floor in metres
- `KIKO_OCCUPANCY_ASSUME_RECTIFIED_LEFT` — compatibility assertion for legacy datasets whose depth optical frame was not recorded; not needed for newly recorded or live rectified-left depth
- `KIKO_OCCUPANCY_RESOLUTION_M` — grid cell size in metres (default `0.05`)
- `KIKO_OCCUPANCY_LOWER_X_M` / `KIKO_OCCUPANCY_LOWER_Y_M` — fixed grid lower bounds in metres (defaults `-10` / `-5`)
- `KIKO_OCCUPANCY_WIDTH_CELLS` / `KIKO_OCCUPANCY_HEIGHT_CELLS` — fixed grid shape (defaults `400` / `400`)
- `KIKO_OCCUPANCY_MAX_CELLS` — allocation safety bound (default `4000000`)
- `KIKO_OCCUPANCY_MIN_HEIGHT_M` / `KIKO_OCCUPANCY_MAX_HEIGHT_M` — inclusive obstacle-height slab in metres (defaults `0.05` / `1.8`)
- `KIKO_OCCUPANCY_MIN_DEPTH_M` / `KIKO_OCCUPANCY_MAX_DEPTH_M` — inclusive accepted depth range in metres (defaults `0.2` / `10`)
- `KIKO_OCCUPANCY_SAMPLE_BLOCK_PX` — nearest-valid depth sampling block width in pixels (default `4`)
- `KIKO_OCCUPANCY_MAX_KEYFRAMES` — maximum retained keyframe contributions (default `300`)
- `KIKO_OCCUPANCY_RERUN_EVERY_KEYFRAMES` — successful integrations between regular map snapshots (default `5`)

Rerun receives the map in Kiko's spatial graph under `world/map2d`: class `0` unknown, `1` free,
and `2` occupied. Each snapshot logs the actual occupancy-to-world rigid transform at
`world/map2d`, pixel-to-occupancy metric placement and the segmentation image at
`world/map2d/grid`, and exact revision/map-identifier metadata at `world/map2d/metadata` on the
same `capture_ns` timeline. A metric bounds outline lets Rerun's auto-spawn heuristics recommend a
dedicated 2D map view rooted at `world/map2d`, so its axes are metres rather than pixels. The
segmentation image is not a textured floor overlay in Rerun's 3D world view; the rigid transform
registers the data truthfully without inventing a pinhole projection. The fixed world coordinate
convention and class annotations are static.

**Bundle adjustment:**

- `KIKO_BA_WINDOW` — sliding window size (default 10)
- `KIKO_BA_ITERS` — maximum local-BA LM iterations (default 6)
- `KIKO_BA_MIN_OBS` — minimum observations per frame
- `KIKO_BA_HUBER_PX` — Huber robust cost threshold in pixels
- `KIKO_BA_DAMPING` — initial Levenberg-Marquardt damping value
- `KIKO_LM_FACTOR` — factor used to increase or decrease LM damping
- `KIKO_LM_MIN` / `KIKO_LM_MAX` — inclusive LM damping bounds
- `KIKO_BA_MOTION_WEIGHT` is rejected. The former absolute pose-parameter penalty was removed
  because it was not a frame-invariant SE(3) objective; remove this setting from the environment.

**Keyframe policy:**

- `KIKO_KEYFRAME_MIN_POINTS` — minimum triangulated landmarks required to accept a keyframe
- `KIKO_KEYFRAME_REFRESH_INLIERS` — tracked inlier count below which a new keyframe is requested
- `KIKO_KEYFRAME_PARALLAX_PX` — median parallax (px) above which a new keyframe is requested
- `KIKO_KEYFRAME_COVISIBILITY` — covisibility ratio below which a new keyframe is requested
- `KIKO_KEYFRAME_REDUNDANT_COVISIBILITY` — covisibility ratio at or above which a keyframe is eligible for redundancy culling
- `KIKO_TRACK_MIN_INLIERS` — minimum RANSAC inliers required to accept a tracked pose

## Models

Default model paths are resolved under `crates/kiko-slam/models/`:

- `sp.onnx` (SuperPoint)
- `lg.onnx` (LightGlue)

Override with `--superpoint-model` / `--lightglue-model` or `KIKO_SUPERPOINT_MODEL` /
`KIKO_LIGHTGLUE_MODEL`. A SuperPoint override must implement Kiko's exact
[tensor and coordinate profile](crates/kiko-slam/models/README.md); the path option does not infer
or adapt alternate layouts, units, or axis orders.

Learned place recognition resolves `eigenplaces.onnx` from the same directory. That model is not
stored in this repository; provide it at that path or set `KIKO_EIGENPLACES_MODEL`. Learned
descriptors are required when loop closure is enabled.

## Roadmap

- ~~Quick dataset recording~~
- ~~Live match visualisation in Rerun~~
- ~~Unified CLI (record/live/viz/bench)~~
- ~~Pipeline benchmarks + per-stage timing~~
- ~~Stereo triangulation~~
- ~~Frame-to-keyframe tracking (PnP + RANSAC)~~
- ~~Local bundle adjustment (sliding window Gauss-Newton)~~
- ~~Parallax + covisibility keyframe policy~~
- ~~Keyframe database + map point management~~
- ~~Covisibility graph~~
- ~~Learned place recognition (EigenPlaces ONNX)~~
- ~~Loop closure (geometric verification + SE(3) pose graph correction)~~
- ~~Deterministic geometric 2D occupancy mapping + Rerun visualization~~
- ~~Replay-bound encoderless host navigation in transport-free shadow mode~~
- Global bundle adjustment
- Dense mapping via nvblox (TSDF / ESDF on Jetson)
