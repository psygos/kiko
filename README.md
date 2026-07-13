<p align="center">
  <img src="assets/kiko-logo.svg" alt="kiko" width="200">
</p>

Kiko is a social robot combining custom SLAM and an expression engine. The SLAM stack is built in Rust and is designed around swappable learned vision models.

**Status:** early development. Recorded-dataset stereo SLAM works with keyframes, local BA, covisibility, loop closure, relocalization, and the VIO runtime. Live OAK-D support is available behind the `record` feature.

## Structure

```text
kiko/
├── crates/
│   ├── kiko-slam/       # SLAM, VIO, inference, Rerun visualization
│   └── oak-sys/         # OAK-D camera FFI bindings
├── comms/
│   ├── robot-server/    # Robot communication hub
│   └── desktop-client/  # Tauri control UI
└── embedded/            # STM32F446 firmware
```

## Jetson Fast Path

Build dynamically against the Jetson ONNX Runtime build:

```sh
ORT_LIB_PATH=/home/makerspace/work/onnxruntime/build-jetson/Release \
ORT_PREFER_DYNAMIC_LINK=1 \
cargo build -p kiko-slam --features vio,ort-cuda,ort-tensorrt --release
```

Run the full dataset and stream Rerun to the laptop:

```sh
LD_LIBRARY_PATH=/home/makerspace/work/onnxruntime/build-jetson/Release \
./target/release/kiko-slam run \
  --profile jetson \
  --rerun-laptop \
  /home/makerspace/full_slam_lab
```

`run` is an alias for `slam`. `--profile jetson` is the normal robot dataset profile. It enables VIO unless `--visual-only` is passed, selects CUDA inference, keeps the full `640x480` image (`downscale=1`), raises the keypoint cap to `2048`, uses `sp_topk2048.onnx`, uses `superpoint_lightglue_fused_fp16.onnx` for stereo/keyframe fallback matching, applies realtime BA settings, disables learned loop descriptors for the fast path, and uses topology-aware projected tracking by default.

Current full-resolution visual-only validation on `/home/makerspace/full_slam_lab`:

- Projected tracking default: `2084/2084` frames processed, `19.38 FPS`, `0` tracker errors, `2084` poses, `96` keyframes, `downscale=1`, `max_keypoints=2048`.
- LightGlue-only baseline: `2084/2084` frames processed, `6.21 FPS`, `0` tracker errors, `2084` poses, `96` keyframes, `downscale=1`, `max_keypoints=2048`.

Use explicit flags to override any profile default:

```sh
./target/release/kiko-slam run --profile jetson --keypoints 2048 --rerun-laptop /path/to/dataset
./target/release/kiko-slam run --profile jetson --visual-only --rerun-laptop /path/to/dataset
```

## Everyday Commands

Record a dataset from OAK-D:

```sh
cargo run -p kiko-slam --features record -- record recordings/<name>
```

Run live SLAM from OAK-D:

```sh
cargo run -p kiko-slam --features record -- live --rerun-laptop
```

Run recorded SLAM with default settings:

```sh
cargo run -p kiko-slam --features vio -- run recordings/<name>
```

Visualize stereo matches:

```sh
cargo run -p kiko-slam -- viz recordings/<name>
```

Benchmark inference:

```sh
cargo run -p kiko-slam -- bench recordings/<name>
```

## CLI Design

Use profiles for complete known-good operating modes, and use flags only for deliberate overrides.

- `--profile default` preserves ordinary flags and environment defaults.
- `--profile jetson` is the measured Jetson Orin dataset-SLAM profile.
- `--vio` enables VIO when the binary is built with `--features vio`.
- `--visual-only` forces visual-only tracking even if the profile or environment enables VIO.
- `--rerun-laptop` streams to `rerun+http://192.168.50.1:9876/proxy`.
- `--rerun-url <URL>` streams to a custom Rerun endpoint.
- `--rerun-serve` serves from the robot on `0.0.0.0:<port>`.
- `--save-rrd <PATH>` writes a persistent Rerun recording instead of streaming.

Inference flags:

- `--backend {auto,cpu,coreml-gpu,cuda,tensorrt}` sets the default model backend.
- `--sp-backend` and `--lg-backend` override SuperPoint and LightGlue independently.
- `--keypoints` is an alias for `--max-keypoints`.
- `--sp-model` is an alias for `--superpoint-model`.
- `--lg-model` is an alias for `--lightglue-model`.
- `--pipeline` is an alias for `--pipeline-model`.

## Models

Default model paths are resolved from `crates/kiko-slam/models/`:

- `sp.onnx` - original SuperPoint export; note that this build has an internal 512 top-k cap.
- `sp_topk2048.onnx` - full-resolution SuperPoint export with the internal top-k cap raised for the Jetson profile.
- `lg.onnx` - standard LightGlue model.
- `superpoint_lightglue_fused_fp16.onnx` - FP16 LightGlue model used for stereo/keyframe matching and projected-tracking fallback.

## Advanced Environment

Most knobs are still available as `KIKO_*` environment variables for experiments and CI. Prefer CLI profiles for normal runs.

- `KIKO_BA_WINDOW`, `KIKO_BA_ITERS`, `KIKO_BA_MIN_OBS` tune local BA.
- `KIKO_LOOP_CLOSURE`, `KIKO_LEARNED_DESCRIPTORS`, `KIKO_RELOCALIZATION` control loop subsystems.
- `KIKO_IMU_CALIBRATION_FILE` and `KIKO_IMU_*` override only the runtime IMU calibration block.
- `KIKO_CUDA_CONV_SEARCH={heuristic,exhaustive,default}` controls cuDNN conv algorithm selection. Jetson defaults to `heuristic` to avoid multi-second full-resolution autotune.
- `KIKO_CUDA_PREFER_NHWC` and `KIKO_CUDA_FUSE_CONV_BIAS` are available for experiments but default off on Jetson because they regress or fail on the current full-resolution SuperPoint path.
- `KIKO_TRACKING_MATCHER={projected,lightglue}` selects topology-aware projected tracking or the legacy global LightGlue tracker.
- `KIKO_PROJECTED_MATCH_RADIUS_PX`, `KIKO_PROJECTED_MATCH_MIN_SIMILARITY`, `KIKO_PROJECTED_MATCH_MIN_MATCHES`, and `KIKO_PROJECTED_MATCH_MIN_INLIERS` tune the projected matcher.
- `KIKO_TRT_CACHE_DIR` controls the TensorRT engine cache path.

The Jetson binary should be dynamically linked against `/home/makerspace/work/onnxruntime/build-jetson/Release`. The binary may be built with `ort-tensorrt`, and TensorRT sessions use CUDA fallback for unsupported nodes, but the measured full-resolution SLAM path uses the CUDA backend. Do not force `--backend tensorrt` for normal SLAM unless validating TensorRT-specific model coverage or prebuilt engine caches.
