# Nano Gate A evidence refresh — 2026-07-27

This file records current, bounded evidence for the wheel-attachment gate. It
does not declare Gate A passed and does not authorize wheel attachment,
powered motion, deployment, or production use.

## Exact native qualification build

The clean Nano checkout at `/home/makerspace/kiko` was fast-forwarded to exact
source revision:

```text
b3df4b52fb9ffeb0e2abb59c33077eb9da0f944d
```

The checked-out branch was
`codex/nano-expression-integration-stage`, tracking the same remote branch,
and `git status --short --branch` reported no worktree changes.

Using the retained native build inputs at
`/home/makerspace/kiko-native-evidence/3f262f1-20260727T013755Z`, the following
exact feature build completed on Linux aarch64:

```text
cargo build --locked --release -p kiko-slam \
  --features nano-wheels-off-qualification --bin kiko-slam
```

The resulting executable identity was:

```text
path: target/release/kiko-slam
size_bytes: 28731544
sha256: a82c1be7c8733033f2ef9e5ae7cea6cab059a4ddf8b823c5e1f4aa757b32f0e3
elf_machine: AArch64
elf_type: PIE
build_id_sha1: a32e5fb1865db91f69dd0bd21bae72bfa57f1642
```

`readelf -d` found the expected direct DepthAI, OpenCV, C++, and system
dependencies. `ldd` under the retained `LD_LIBRARY_PATH` resolved every
entry, including the retained DepthAI, dynamic-calibration, OpenCV core,
OpenCV image-processing, OpenCV object-detection, and USB libraries. No
unresolved dependency was reported.

With that exact retained library path, the executable's
`nano-wheels-off-qualification --help` boundary loaded successfully. It
advertised only the deployment root, relative launch document, and state root;
the physical preconditions remain attended-TTY inputs with no flag or
environment bypass.

The five-minute build duration is recorded only to distinguish the completed
command from an interrupted build. It is not a benchmark or a performance
claim. This build did not open a device or exercise camera frames, serial
traffic, SLAM, occupancy, head motion, eye output, motor output, MPC timing, or
physical safety behavior.

## Live owner and USB snapshot

At `2026-07-27T22:57:45+05:30`, read-only inspection found:

- all three persistent serial-by-id endpoints present;
- the STM32 endpoint `/dev/ttyACM0` had no process owner;
- one legacy `kiko_face_follow.py` child held `/dev/ttyACM1`,
  `/dev/ttyACM2`, and the OAK USB node;
- its two-level `engine-guardian.sh` owner was still running and configured to
  respawn the child every eight seconds;
- OAK MXID `19443010F1B43A2E00` was the USBFS device below the `480M` USB2
  tree;
- the separate `10000M` USB3 root and hub were present with no OAK below
  them; and
- no canonical qualification owner was started.

The legacy process is not an architectural or repository owner. It is an
exclusive live device owner that must be handed off explicitly. The canonical
Kiko process must not overlap it on the OAK, head, or eye endpoints. The
handoff must target only the two known guardian entries and their identified
process family, wait past the respawn interval, verify every endpoint is free,
and start the canonical supervised owner without an unsupported gap.

No process was stopped, signalled, reconfigured, or replaced during this
snapshot. No camera or serial endpoint was opened by the inspection.

After that snapshot, the operator reported that Fable is no longer operating
on the Nano. That report supersedes any plan that treats Fable as an expected
runtime owner, but it is not a fresh process/endpoint measurement. Final Gate A
qualification must still re-read the current owners and prove each endpoint
free before starting exactly one canonical Kiko owner.

## Gate status after this refresh

This is exact predecessor-build evidence for `b3df4b5`. Later executable
changes supersede it for the final-source gate, so a final frozen revision
still requires a new native release build, loader check, executable identity,
and updated `--help` observation. It leaves these physical/current items open:

- fresh exclusive-endpoint proof followed by one canonical Kiko owner;
- canonical OAK SuperSpeed negotiation and one-graph RGB/stereo/depth/IMU;
- reviewed tracking-camera-to-base and native-IMU-to-base calibration;
- sentinel-free immutable V4 qualification bundle and installation;
- attended candidate STM32 flash, journal, identity, disarm, and applied zero;
- continuous natural head hold and RGB-derived eye behavior;
- live SLAM, localized occupancy, Rerun, and the unified control gateway;
- manual/deadman/reconnect/applied-receipt and fault-matrix sessions;
- independently reachable motor-power cut and bounded wheels-off shaft-sign
  checks.

Until those items have direct current evidence, the wheel-attachment request
must not be issued.
