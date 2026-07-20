# Nano wheels-off discovery evidence — 2026-07-21

This record captures one wheels-removed, zero-only discovery session on the
Kiko Jetson. It is evidence for deployment work, not a general hardware
qualification certificate. No nonzero base command, firmware flash, power-mode
change, thermal test, or deployment was performed.

## Session boundary

- Host: NVIDIA Jetson Orin Nano, aarch64, Ubuntu 22.04.5.
- Boot ID: `ac13a80c-3116-4757-a837-76844238b715`.
- Source checkout: `codex/nano-expression-integration` at `d49ef6a`.
- Operator constraint: wheels remain physically removed.
- The Nano initially had no Kiko, SLAM, eye, expression, or `robot-server`
  process running and no installed Kiko systemd unit.

The operator reported that the head path was clear and a power cut was
reachable before accessory work continued. Software cannot independently
prove either physical condition.

## Stable device identities

| Device | Stable Linux path or identity | USB identity |
| --- | --- | --- |
| STM32 ST-Link VCP | `/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066EFF313946303143221230-if02` | `0483:374b`, serial `066EFF313946303143221230` |
| STS head adapter | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00` | `1a86:55d3`, serial `5B14031114` |
| Eye/Pico controller | `/dev/serial/by-id/usb-kiko_kiko-eyes_1-if00` | `c0de:cafe`, product `kiko-eyes`, serial `1` |
| OAK | MXID `19443010F1B43A2E00` | MyriadX `03e7:2485` |

All three serial endpoints were owned by `root:dialout`, mode `0660`, and had
no process owner when probed. Device discovery used exact identities; no
`ttyACM*` fallback was admitted.

## Camera findings

DepthAI Python 3.4.0 discovered exactly the configured OAK MXID. Two diagnostic
RGB captures were attempted:

1. Requested 640 by 400 BGR at 15 frames/s: 30 valid `uint8` images arrived in
   8.991 s (3.337 host-received frames/s). Sequence numbers were not strictly
   increasing, and XLink disconnected and reconnected during the run.
2. Requested 320 by 200 BGR at 5 frames/s: 20 valid images arrived in 3.905 s
   (5.122 host-received frames/s), with exact sequence numbers `0..19` and
   strictly increasing device timestamps.

The kernel log and `lsusb -t` showed repeated SuperSpeed disconnects followed
by a USB 2 high-speed fallback at 480 Mbit/s. This proves that the RGB sensor
and low-rate delivery path worked during this boot. It does **not** prove a
stable USB 3 path, simultaneous stereo/depth/IMU throughput, camera
calibration, SLAM readiness, or the production native DepthAI build.

The checkout cannot currently build the native OAK path on this Nano: no
standard `depthai/depthai.hpp` or matching `libdepthai-core.so` installation
was present. The installed Python wheel is not substituted as build
provenance. Existing older `kiko-slam` binaries also report a missing
`libonnxruntime.so.1` through `ldd` and are not accepted for deployment.

## Head findings

The STS bus was probed at 1,000,000 baud with DTR low and RTS high. Each request
was a checksum-valid non-broadcast READ. No goal, speed, torque, limit, or
EEPROM write was sent. Three checksum-valid full-telemetry samples per servo
agreed exactly in the final observation:

| Joint / ID | Position (tick) | Speed | Load | Voltage raw | Temperature raw | Moving | Current | Torque enable | Response level |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bow / 1 | 2210 | 0 | 0 | 120 | 29 | 0 | 0 | 0 | 0 |
| curl / 2 | 2583 | 0 | 0 | 119 | 29 | 0 | 0 | 0 | 0 |
| yaw / 3 | 2952 | 0 | 0 | 119 | 33 | 0 | 0 | 0 | 0 |
| roll / 4 | 2884 | 0 | 0 | 119 | 33 | 0 | 0 | 0 | 0 |

These values establish a motionless, torque-disabled observation. They are not
automatically a calibrated natural pose. Any hold must use the typed head
actor, an explicitly reviewed narrow window around a contemporaneous redundant
observation, bounded torque and speed, explicit physical consent, stopped
readback, and fail-closed torque-disable cleanup.

## Controller firmware findings

The Nano built and tested the checked-out `robot-server` and
`v2_identity_probe`. A byte-read-only KRP2 identity probe on the exact STM32
path failed with `Decode(OversizedRecord { maximum: 73 })`. A separate bounded
read that transmitted zero bytes observed recurring legacy ASCII records such
as `ODO,0,0,0,0,...` and `DBG,0,0`.

The attached STM32 therefore was not running the canonical KRP2 contract during
this session. Zero-valued legacy telemetry is not typed applied-zero evidence.
The production server must not own this controller until the exact firmware
identity and contract are provisioned. The checked-in default KRP2 firmware is
intentionally motion-disabled (`max_abs_pwm = 0`) and cannot be represented as
a motion-capable plant.

## Eye findings

The Pico replied to the legacy ASCII ping `P` with `kiko-eyes 1`. Two
LED-only command sequences were then written. The second refreshed each state
at 20 Hz and held far-left gaze for 2 s, closed lids for 2 s, far-right gaze for
2 s, and centered/open gaze for 1 s.

The operator reported no corresponding visible state change. This is a failed
visual check: a serial ping and completed host writes do not prove that the
renderer or LED panels applied a command. Further blind expression writes were
stopped. The attached firmware is also the legacy ASCII protocol rather than
the canonical challenged KEP2 session, so it cannot satisfy the production eye
admission contract even if visible output is later restored.

## Claims that remain open

- stable OAK USB 3, stereo, depth, and IMU delivery;
- native aarch64 OAK/ONNX build provenance and a complete SLAM stream;
- typed fresh STM32 applied-zero evidence and physical watchdog behavior;
- any nonzero base actuation, PWM-to-velocity model, MPC tracking, or stop
  distance;
- a continuously supervised natural head hold;
- visible KEP2 eye expressions and calibrated panel signs;
- RGB person/face detection and head/eye gaze following;
- map persistence, cold-boot relocalization, autonomous exploration, and
  supervised point navigation on the production Nano owner.

Until those items have their own evidence, deployment must report them as
unknown rather than infer them from this discovery session.
