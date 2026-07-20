# Nano validation evidence — 2026-07-20

This record separates read-only device presence and simulated host tests from
physical robot acceptance. It contains no motor or servo command and no
firmware flash.

## Host and repository

- Target: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit,
  aarch64, Ubuntu 22.04.5 LTS, Linux `5.15.148-tegra`.
- Observed boot ID: `5b2518dd46cd4977859a46dfb5448dd9`.
- Rust: `rustc 1.88.0`; Cargo: `1.88.0`.
- Storage at observation: 329 GiB available on `/`.
- User `makerspace` belongs to `dialout`, `video`, `render`, `i2c`, `gpio`,
  and `plugdev` among its groups.
- No `kiko` or `robot-server` process was running.
- The existing `/home/makerspace/kiko` checkout remained clean on
  `codex/jetson-hardware-validation`.
- A separate clean worktree was created at
  `/home/makerspace/kiko-nano-expression-integration` on commit `186ffe3` of
  `codex/nano-expression-integration`.

## Read-only USB presence

The following stable paths and USB functions were present:

| Role | Stable path / USB identity | Kernel node |
| --- | --- | --- |
| Head adapter | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00`; `1a86:55d3` | `/dev/ttyACM0` |
| STM32 ST-Link serial function | `/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066EFF313946303143221230-if02`; `0483:374b` | `/dev/ttyACM1` |
| Eye controller | `/dev/serial/by-id/usb-kiko_kiko-eyes_1-if00`; `c0de:cafe` | `/dev/ttyACM2` |
| OAK accelerator | `03e7:2485 Intel Movidius MyriadX` | USB only |

All three serial nodes were `root:dialout` with mode `0660`. USB enumeration
does not prove protocol identity, firmware build, calibration, electrical
wiring, camera capture, actuator response, or physical behavior.

## Native aarch64 simulated evidence

The following command ran in the isolated worktree without opening physical
serial devices:

```text
cargo test -p kiko-device-inventory -p kiko-supervisor-core \
  -p kiko-expression-runtime -p kiko-eye-runtime -p kiko-head-runtime
```

It completed successfully with 113 tests:

- `kiko-device-inventory`: 31;
- `kiko-expression-runtime`: 27;
- `kiko-eye-runtime`: 27;
- `kiko-head-runtime`: 20; and
- `kiko-supervisor-core`: 8.

These tests prove the covered typed boundaries and simulated fault behavior on
the deployed CPU architecture. They do not prove native DepthAI linkage,
real-time scheduling, KEP2 compatibility of the currently flashed eye image,
head telemetry, STM32 actuation, PWM-to-velocity calibration, emergency-stop
operation, thermal behavior, or safe robot motion.

## Outstanding physical gates

Before an end-to-end drive claim, the operator must provide a clear test area
and an independent emergency stop. The runtime must then establish exact OAK
MXID and provenance, controller UID/build/profile/capabilities and a newly
applied zero, KEP2 eye UID/build/capabilities after compatible firmware is
flashed, redundant head telemetry and verified present-position hold, and the
reviewed encoderless plant/calibration artifacts. No one may infer those facts
from USB presence or host tests.
