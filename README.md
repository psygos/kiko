<p align="center">
  <img src="assets/kiko-logo.svg" alt="kiko" width="200">
</p>

Kiko is a social robot combining custom SLAM and an expression engine. The SLAM implementation is designed for model hot-swap as vision models improve, and Kiko improves with them. Built entirely in Rust.

**Status:** Early development. The host pipeline includes stereo visual odometry, local bundle
adjustment, map/covisibility maintenance, relocalization, and loop closure. Hardware-backed
recording and TSDF/ESDF reconstruction still depend on external native components.

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

Live match visualization (requires OAK-D + Rerun viewer):

```
cargo run -p kiko-slam --features record -- live
```

Visualize stereo matches on a dataset:

```
cargo run -p kiko-slam -- viz recordings/<name>
```

Run visual odometry on a dataset (Rerun viewer):

```
cargo run -p kiko-slam -- viz --odometry recordings/<name>
```

Benchmark a dataset:

```
cargo run -p kiko-slam -- bench recordings/<name>
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
- Global bundle adjustment
- Dense mapping via nvblox (TSDF / ESDF on Jetson)
