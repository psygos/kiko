<p align="center">
  <img src="assets/kiko-logo.svg" alt="kiko" width="200">
</p>

Kiko is a social robot combining custom SLAM and an expression engine. The SLAM implementation is designed for model hot-swap as vision models improve, and Kiko improves with them. Built entirely in Rust.

**Status:** Early development. Robust SLAM is the primary target.

## Structure

```
kiko/
├── crates/
│   ├── kiko-slam/    # SLAM and feature detection
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

Visualize a dataset:

```
cargo run -p kiko-slam -- viz recordings/<name>
```

Benchmark a dataset:

```
cargo run -p kiko-slam -- bench recordings/<name>
```

## Config (flags or env)

- `--downscale` / `KIKO_DOWNSCALE`
- `--max-keypoints` / `KIKO_MAX_KEYPOINTS`
- `--backend` / `KIKO_BACKEND`
- `--sp-backend` / `KIKO_SUPERPOINT_BACKEND`
- `--lg-backend` / `KIKO_LIGHTGLUE_BACKEND`
- `--sp-model` / `KIKO_SUPERPOINT_MODEL`
- `--lg-model` / `KIKO_LIGHTGLUE_MODEL`
- `--rerun-decimation` / `KIKO_RERUN_DECIMATION`

## Models

Default model paths are resolved under `crates/kiko-slam/models/`:

- `sp.onnx` (SuperPoint)
- `lg.onnx` (LightGlue)

Override with `--sp-model` / `--lg-model` or `KIKO_SUPERPOINT_MODEL` / `KIKO_LIGHTGLUE_MODEL`.

## Immediate Roadmap

- ~~Quick dataset recording~~
- ~~Live match visualisation in Rerun~~
- ~~Unified CLI (record/live/viz/bench)~~
- ~~Pipeline benchmarks + per-stage timing~~
- Stereo triangulation
- Frame-to-keyframe tracking
- 30 FPS visual odometry
