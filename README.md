<p align="center">
  <img src="assets/kiko-logo.svg" alt="kiko" width="200">
</p>

Kiko is a social robot combining custom SLAM and an expression engine. The SLAM implementation is designed for model hot-swap as vision models improve, and Kiko improves with them. Built entirely in Rust.

**Status:** Early development. Stereo visual odometry with local bundle adjustment is working.

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

- `--downscale` / `KIKO_DOWNSCALE` — input downscale factor (1, 2, or 4)
- `--max-keypoints` / `KIKO_MAX_KEYPOINTS` — max keypoints per frame (default 1024)
- `--backend` / `KIKO_BACKEND` — inference backend for both models
- `--sp-backend` / `KIKO_SUPERPOINT_BACKEND` — SuperPoint backend override
- `--lg-backend` / `KIKO_LIGHTGLUE_BACKEND` — LightGlue backend override
- `--sp-model` / `KIKO_SUPERPOINT_MODEL` — custom SuperPoint ONNX path
- `--lg-model` / `KIKO_LIGHTGLUE_MODEL` — custom LightGlue ONNX path

**Visualization:**

- `--rerun-decimation` / `KIKO_RERUN_DECIMATION` — image decimation for Rerun
- `--odometry` / `KIKO_VIZ_ODOMETRY` — enable visual odometry in viz mode
- `--allow-unrectified` / `KIKO_ALLOW_UNRECTIFIED` — skip rectification check

**Bundle adjustment:**

- `KIKO_BA_WINDOW` — sliding window size (default 10)
- `KIKO_BA_ITERS` — Gauss-Newton iterations (default 6)
- `KIKO_BA_MIN_OBS` — minimum observations per frame
- `KIKO_BA_HUBER_PX` — Huber robust cost threshold in pixels
- `KIKO_BA_DAMPING` — LM damping factor
- `KIKO_BA_MOTION_WEIGHT` — motion prior weight (0 to disable)

**Keyframe policy:**

- `KIKO_KF_MIN_INLIERS` — inlier count below which a keyframe is forced
- `KIKO_KF_PARALLAX_PX` — median parallax (px) above which a keyframe is created
- `KIKO_KF_MIN_COVISIBILITY` — covisibility ratio below which a keyframe is created

## Models

Default model paths are resolved under `crates/kiko-slam/models/`:

- `sp.onnx` (SuperPoint)
- `lg.onnx` (LightGlue)

Override with `--sp-model` / `--lg-model` or `KIKO_SUPERPOINT_MODEL` / `KIKO_LIGHTGLUE_MODEL`.

## Roadmap

- ~~Quick dataset recording~~
- ~~Live match visualisation in Rerun~~
- ~~Unified CLI (record/live/viz/bench)~~
- ~~Pipeline benchmarks + per-stage timing~~
- ~~Stereo triangulation~~
- ~~Frame-to-keyframe tracking (PnP + RANSAC)~~
- ~~Local bundle adjustment (sliding window Gauss-Newton)~~
- ~~Parallax + covisibility keyframe policy~~
- Keyframe database + map point management
- Covisibility graph
- Place recognition (NetVLAD / CosPlace ONNX)
- Loop closure (Sim3 + pose graph correction)
- Global bundle adjustment
- Dense mapping via nvblox (TSDF / ESDF on Jetson)
