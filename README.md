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

- Fail-closed CUDA run: `2084/2084` pairs processed, `19.37` total FPS and
  `19.61` steady FPS after a 12-pair startup warm-up, `0` tracker errors,
  `2072/2072` steady current poses, `2` explicitly stale startup poses,
  `98` keyframes, `downscale=1`, and `max_keypoints=2048`.
- Evidence:
  `/home/makerspace/kiko-benchmarks/20260716T212612+0530-full-visual-settled-501943`.
- Jetson-profile VIO invocation: `2084/2084` pairs processed, `19.68` total FPS
  and `19.91` steady FPS after the same warm-up, `0` tracker errors,
  `2072/2072` steady current poses, `3` explicitly stale startup poses, and
  `98` keyframes. Evidence:
  `/home/makerspace/kiko-benchmarks/20260716T213222+0530-full-vio-settled-504627`.

Use explicit flags to override any profile default:

```sh
./target/release/kiko-slam run --profile jetson --keypoints 2048 --rerun-laptop /path/to/dataset
./target/release/kiko-slam run --profile jetson --visual-only --rerun-laptop /path/to/dataset
```

### Reproducible Jetson validation

`scripts/jetson_benchmark.py` runs one already-built stage and creates a new,
non-overwriting evidence directory. It never changes `nvpmodel`, clocks, sudo
state, or Git. Set MAXN/clocks separately, then record the exact approved
artifacts in a SHA256 manifest before invoking it:

```sh
sha256sum \
  /home/makerspace/kiko/target/release/kiko-slam \
  /home/makerspace/kiko/crates/kiko-slam/models/sp_topk2048.onnx \
  /home/makerspace/kiko/crates/kiko-slam/models/superpoint_lightglue_fused_fp16.onnx \
  /home/makerspace/work/onnxruntime/build-jetson/Release/libonnxruntime.so.1.24.2 \
  /home/makerspace/work/onnxruntime/build-jetson/Release/libonnxruntime_providers_shared.so \
  /home/makerspace/work/onnxruntime/build-jetson/Release/libonnxruntime_providers_cuda.so \
  > /home/makerspace/kiko-benchmarks/approved-artifacts.sha256

python3 /home/makerspace/kiko/scripts/jetson_benchmark.py \
  --stage full-visual \
  --output-root /home/makerspace/kiko-benchmarks \
  --binary /home/makerspace/kiko/target/release/kiko-slam \
  --dataset /home/makerspace/full_slam_lab \
  --superpoint-model /home/makerspace/kiko/crates/kiko-slam/models/sp_topk2048.onnx \
  --lightglue-model /home/makerspace/kiko/crates/kiko-slam/models/superpoint_lightglue_fused_fp16.onnx \
  --ort-library /home/makerspace/work/onnxruntime/build-jetson/Release/libonnxruntime.so.1.24.2 \
  --ort-shared-provider /home/makerspace/work/onnxruntime/build-jetson/Release/libonnxruntime_providers_shared.so \
  --ort-cuda-provider /home/makerspace/work/onnxruntime/build-jetson/Release/libonnxruntime_providers_cuda.so \
  --sha256-manifest /home/makerspace/kiko-benchmarks/approved-artifacts.sha256 \
  --provider cuda \
  --expected-pairs 2084 \
  --expected-command-items 2084 \
  --working-directory /home/makerspace/kiko \
  --ld-library-path /home/makerspace/work/onnxruntime/build-jetson/Release:/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu \
  --timeout-seconds 180 \
  --kill-grace-seconds 5 \
  --expected-nvpmodel MAXN_SUPER \
  --min-cpu-hz 1728000000 \
  --min-gpu-hz 1020000000 \
  --min-emc-hz 3199000000 \
  --expected-emc-override 1 \
  --min-memory-available-mib 4096 \
  --max-swap-used-mib 0 \
  --max-cpu-node-fraction 0.20 \
  --max-temperature-c 85 \
  --expected-triangulation-max-vertical-disparity-px 3 \
  --require-power-rail VDD_IN \
  --min-steady-fps 17 \
  --kiko-env KIKO_DOWNSCALE=1 \
  --kiko-env KIKO_MAX_KEYPOINTS=2048 \
  --kiko-env KIKO_CUDA_CONV_SEARCH=heuristic \
  --kiko-env KIKO_TRIANGULATION_MAX_VERTICAL_DISPARITY_PX=3 \
  --ld-debug \
  -- run --profile jetson --visual-only --warmup-pairs 12 /home/makerspace/full_slam_lab
```

Every SLAM benchmark must attest the exact triangulation policy. The current
Jetson profile uses the explicit finite `3 px` gate above; a historical
unbounded baseline must instead pass `--expect-unbounded-triangulation-policy`
and must not set the finite `KIKO_TRIANGULATION_MAX_VERTICAL_DISPARITY_PX`
override.

Run each gate as a separate invocation and continue only after exit status `0`:

- One-pair smoke: change the stage to `cuda-smoke`, expected command items to
  `1`, timeout to `60`, remove `--min-steady-fps`, and add
  `--max-pairs 1` while replacing `--warmup-pairs 12` with
  `--warmup-pairs 0` in the workload arguments.
- Canary: change the stage to `cuda-canary`, expected command items to `300`,
  timeout to `30`, and add `--max-pairs 300` to the workload arguments. Keep
  the 12-pair startup warm-up and the `17 FPS` steady-state gate. Startup pose
  outcomes remain reported even though they are excluded from steady metrics.
- Full visual: use the command above unchanged.
- Full VIO: only after full visual passes, change the stage to `full-vio`,
  timeout to `300`, and remove `--visual-only`. Omit the FPS threshold unless a
  VIO-specific baseline has been established; the visual-only threshold is not
  a truthful VIO threshold.

The sterile workload environment contains only `HOME`, `PATH`,
`LD_LIBRARY_PATH`, explicit `KIKO_*` values, and optional `LD_DEBUG`; it never
sets `ORT_DYLIB_PATH`. A passing run requires exact hashes and ELF/`ldd`
linkage, the complete 640x480 manifest, accessible Jetson GPU devices, expected
MAXN mode, locked CPU/GPU/EMC clocks and EMC override, strict accelerator
registration, committed ORT sessions, verbose session-construction node
placement, a bounded per-session CPU auxiliary-node fraction, warning-level
inference-run logging, exact model-session paths, absolute ORT/CUDA runtime
library evidence, nonzero
GR3D activity, exact reader/tracker/processed and warm-up/steady counts, any
configured steady FPS gate, no stale or unavailable steady-state poses, and
clean memory, thermal, boot, and kernel evidence. The runner also refuses to
start while another `kiko-slam`, benchmark runner, or `tegrastats` process is
active. On timeout it sends `TERM`, then `KILL` after the configured grace
period, only to process groups created by the
runner. Analyze a saved directory again without touching hardware:

```sh
python3 scripts/analyze_jetson_benchmark.py /home/makerspace/kiko-benchmarks/<run-directory>
```

The analyzer prints JSON and does not overwrite the runner's `analysis.json`.
Use `--output /path/to/new-analysis.json` for a create-new saved copy.

TensorRT is never initialized by the CUDA command. A TensorRT validation must
deliberately use `--provider tensorrt`, supply `--ort-tensorrt-provider`, add
that library to the SHA256 manifest, and pass the same placement and telemetry
gates.

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
- `--rerun-finish-timeout-ms <MS>` controls the final sink flush (default 5000 ms); success means the calling thread's prior data reached the configured sink, not that a viewer consumed it.

Inference flags:

- `--backend {auto,cpu,coreml-cpu-gpu,cuda,tensorrt}` sets the default model backend.
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
- `KIKO_PROJECTED_MATCH_RADIUS_PX`, `KIKO_PROJECTED_MATCH_MIN_DOT_PRODUCT`, `KIKO_PROJECTED_MATCH_MIN_MATCHES`, and `KIKO_PROJECTED_MATCH_MIN_INLIERS` tune the projected matcher. The dot-product gate does not normalize descriptors, so its scale is model-dependent; `KIKO_PROJECTED_MATCH_MIN_SIMILARITY` remains only as a warned, deprecated compatibility alias.
- `KIKO_TRACK_MIN_DOT_PRODUCT` controls the same raw descriptor dot-product gate in Rerun-only feature-track visualization; `KIKO_TRACK_MIN_SIM` is its warned, deprecated compatibility alias.
- `KIKO_TRT_CACHE_DIR` controls the TensorRT engine cache path.

The Jetson binary should be dynamically linked against `/home/makerspace/work/onnxruntime/build-jetson/Release`. The binary may be built with `ort-tensorrt`, and TensorRT sessions use CUDA fallback for unsupported nodes, but the measured full-resolution SLAM path uses the CUDA backend. Do not force `--backend tensorrt` for normal SLAM unless validating TensorRT-specific model coverage or prebuilt engine caches.
