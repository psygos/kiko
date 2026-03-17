# Kiko VIO — Build & Development Guide

## Platform

Jetson Orin Nano, Ubuntu 22.04, GCC 11, Rust 1.88, CUDA 12.6, TensorRT 10.3.
Camera: OAK-D-S2 with BMI270 IMU. Static IP: 192.168.50.2.

## Building kiko-slam

The SLAM binary links dynamically against a custom ORT build with CUDA support.

```bash
ORT_LIB_PATH=/home/makerspace/work/onnxruntime/build-jetson/Release \
ORT_PREFER_DYNAMIC_LINK=1 \
cargo build -p kiko-slam --features vio,ort-cuda,ort-tensorrt --release
```

### Why these flags matter

- **`ORT_LIB_PATH` + `ORT_PREFER_DYNAMIC_LINK=1`**: Links against the full Jetson ORT build at `build-jetson/Release/` which includes CUDA EP with all contrib ops (notably `com.microsoft.FusedConv` needed by `sp.onnx`). Without `ORT_PREFER_DYNAMIC_LINK`, ort-sys tries static linking against its cached archive which fails with GCC 14 ABI symbols.
- **Do NOT use `build-kiko-reduced`**: That ORT build was compiled with reduced ops and is missing FusedConv.
- **The `ort_compat.rs` module**: Provides stubs for `__cxa_call_terminate` (GCC 14) and `__isoc23_strto*` (glibc 2.38) symbols that the ort-sys static archive references. These exist as a fallback if static linking is ever needed.

### Running

```bash
LD_LIBRARY_PATH=/home/makerspace/work/onnxruntime/build-jetson/Release \
./target/release/kiko-slam slam \
  --backend cuda \
  --downscale 2 \
  --max-keypoints 384 \
  --lightglue-model crates/kiko-slam/models/superpoint_lightglue_fused_fp16.onnx \
  --rerun-serve \
  /path/to/dataset
```

### Remote visualization

`--rerun-serve` hosts a gRPC server on `0.0.0.0:9876`. Connect from any machine:
```bash
rerun --connect rerun+http://192.168.50.2:9876/proxy
```
The server dies when the process exits. Use `--save-rrd /path/to/file.rrd` for persistent recordings.

## Key performance knobs

| Flag | Effect |
|------|--------|
| `--max-keypoints N` | Main FPS lever. 256→24fps, 384→19fps, 512→18fps |
| `--downscale {1,2}` | 2 = half res (320×240). Don't go below 2. |
| `--lightglue-model` | Use `superpoint_lightglue_fused_fp16.onnx` for speed |
| `KIKO_BA_WINDOW=6` | Smaller BA window = less compute per keyframe |
| `KIKO_BA_ITERS=4` | Fewer BA iterations |
| `KIKO_BA_MIN_OBS=4` | Lower threshold prevents early BA degenerates |

## VIO status

VIO (`KIKO_VIO=true`) is work-in-progress. Known issues:
- IMU-camera extrinsics in `calibration.json` need proper calibration (kalibr or factory data). Current values are gravity-derived approximations.
- VIO smoother drifts linearly due to unestimated velocity/bias. The optimizer's pose prior doesn't constrain velocity, and the information floor caps IMU trust too aggressively.
- First ~50 frames have poor tracking (camera white balance settling + IMU startup delay).

Use `KIKO_VIO=false` (or omit it) for stable visual-only SLAM.

## Dataset format

Datasets live in directories with:
- `meta.json` — device, resolution, fps, IMU rate
- `calibration.json` — stereo intrinsics, baseline, IMU noise/extrinsics
- `manifest.json` — frame pairing with timestamps
- `frames/` — raw 8-bit grayscale frames (`{timestamp}_{sensor}.raw`)
- `imu.bin` — binary IMU: `[i64 timestamp_ns][f64 ax][f64 ay][f64 az][f64 gx][f64 gy][f64 gz]` × N (56 bytes/record)

## ORT builds on this machine

| Path | CUDA | FusedConv | Use for |
|------|------|-----------|---------|
| `work/onnxruntime/build-jetson/Release/` | ✅ | ✅ | **Use this** |
| `work/onnxruntime/build-kiko-reduced/Release/` | ✅ | ❌ | Don't use (missing contrib ops) |
| `~/.cache/ort.pyke.io/dfbin/.../libonnxruntime.a` | ❌* | ✅ | Static fallback (needs ort_compat.rs) |

*Static archive has CUDA symbols but `GetAvailableProviders` doesn't list CUDA.
