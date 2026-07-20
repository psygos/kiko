# Nano wheels-off discovery evidence — 2026-07-21

This record captures one wheels-removed, zero-only discovery session on the
Kiko Jetson. It is evidence for deployment work, not a general hardware
qualification certificate. No nonzero base command, firmware flash, power-mode
change, thermal test, or deployment was performed during the initial discovery
phase. Later sections separately record the eye-controller flash, canonical
STM32 provisioning, and explicitly authorized wheels-off motor commissioning.
No Jetson power-mode, thermal, GPU, deployment, or wheels-on test was performed.

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

Two March 20 recordings provide a useful same-host baseline. The historical
`record_oakd_kiko_dataset.py` using the same DepthAI 3.4 installation captured
`cam_static` and `imu_dynamic` for 60 s each at 640 by 480, 30 frames/s stereo
and 400 Hz IMU. Both contain 1,797 left frames, 1,797 right frames, 1,797
pairs, zero orphans, and zero recorded drops. Re-running that exact recorder
for 10 s during this session produced only 66 pairs and 788 IMU samples before
XLink errors, reconnects, and a device crash. CAM_A also failed under the same
current transport state, so RGB node selection, queue shape, and SDK-version
drift are not supported as the primary explanation.

The operator explicitly reported that the camera and cable were known-good, so
subsequent work treated the software ownership path as the primary hypothesis.
The earlier recorder opened the device for calibration, closed it, and then let
the pipeline reopen a default device. A corrected discriminator instead pinned
MXID `19443010F1B43A2E00`, constructed exactly one `dai::Device` capped at
`HIGH`, reused that same owner for calibration and `dai::Pipeline`, and verified
the negotiated speed from it. In 15 s it captured 447 left frames, 447 right
frames, all 447 pairs, zero orphans, zero recorded drops, and 5,908 IMU samples.
The calibration and manifest SHA-256 values were respectively
`15c7d334857d6e5291e02219484b8e8d44d8da0b3525adce902f415038f6ccd9`
and `9d4e48c548a18217219e5524182c8f771541ae0ed3b01c6b08598745db14d83f`.

This clean `HIGH` diagnostic removes the earlier open/close/default-reopen flaw
and is consistent with that flaw causing the failed diagnostic; one successful
run does not independently prove causation. It also does not qualify USB
`SUPER`, the full RGB + rectified stereo + depth production graph, camera
calibration, or the repository's native C++ bridge. DepthAI Python 3.4.0 is
installed, but the Nano still has no independently identified
`depthai/depthai.hpp` and `libdepthai-core` pair for a native Rust build.

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

After the repository's canonical `kiko-head-commission` tool was built natively
on the Nano, its read-only probe performed exactly eight fixed STS READs: torque
switch and telemetry for IDs 1 through 4. It observed zero framing noise and
torque disabled on every servo. The later fresh observation used for the active
natural hold was bow 2154, curl 2545, yaw 2943, and roll 2876 ticks; all speeds,
loads, currents, and moving flags were zero. Voltage raw values were 120, 119,
119, and 119; temperatures were 31, 32, 36, and 35. These values supersede the
earlier sample for any future narrow admission window but do not constitute
servo geometry calibration.

With explicit operator torque consent, the tool admitted that exact observation
inside the reviewed narrow bounds and actively held it. The first bounded hold
reached its hard 900,000 ms maximum and then reported completed torque-disable
writes for bow, curl, yaw, and roll. The second hold was admitted only after
that cleanup and another fresh torque-off probe, reached the same hard limit,
and reported the same complete four-joint cleanup. A third hold was then
admitted from another fresh torque-off probe, later reached the same hard
limit, and completed the same cleanup. A fourth fresh hold was admitted at bow
2155, curl 2545, yaw 2943, and roll 2876 ticks; it too reached 900,000 ms and
reported four completed torque-disable writes. No timeout was bypassed.

After that cleanup, a later re-admission probe observed bow 2512, curl 2916,
yaw 2903, and roll 2903 ticks with torque disabled. Bow was outside the
configured 2140..=2172 admission window, so startup failed before any hold was
accepted and the tool again completed all four torque-disable writes. The head
was therefore left torque-disabled rather than silently treating the changed
pose as the calibrated neutral. These runs prove bounded host ownership,
window enforcement, and cleanup—not an unbounded or independently supervised
production head controller.

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

## Subsequent STM32 provisioning and bounded commissioning

Before changing flash, `stlink-tools` 1.7.0 identified ST-Link V2J33S25 and an
STM32F446 with 512 KiB flash, 128 KiB SRAM, and chip ID `0x0421`. Two complete
524,288-byte reads of the legacy main flash were byte-identical with SHA-256
`8e8f658e5ee65b2eca3ca8de7cb045ea2b08dbf3ec82d70b654fe6fa02bec7dc`.
The 16 option bytes were
`ef aa 10 55 ef aa 10 55 ff 7f 00 80 ff 7f 00 80`, SHA-256
`d292558017cf9ca0a2e40e262a5c1daa4b305ccf084ce06128133d282f905115`.
The exact backup exists on both the Nano and the development Mac under the
timestamped `stm32f446re-20260721T0349IST` hardware-backup directory.

The Nano then built the canonical motion-disabled KRP2 image and flashed its
39,216-byte raw image. The written bytes and an exact-length readback were
byte-identical, SHA-256
`11a4b23b66302d306c7530566d36f122eaba4ca3abe49fd9f5b8622f78c8b2b3`.
A fresh read-only `ControllerHello` reported controller UID
`2c0018001750314242353320`, firmware ABI 2, build ID `0x00020001`, hardware
profile `KIKO-NO-ACT-V1!!`, capability bits 63, 20 kHz PWM, maximum PWM 0,
disabled safe output, no motion authority, and unverified physical-stop
semantics. That is typed firmware-identity evidence, not an applied command or
physical-stop test.

For the operator-authorized wheels-off check, the repository added a separate
feature-gated commissioning image; the canonical firmware source and zero-PWM
profile were not weakened. The final Nano-built raw image was 19,012 bytes,
SHA-256
`8bc272e8b8e901471d025ff1b5df5425d4e498e3bf18c3e4946b68deb790f4be`,
and was compared byte-for-byte with an exact flash readback before execution.
Its fresh-nonce, CRC-protected, one-shot recipe commanded left forward at 8%
for 250 ms, held zero for 500 ms, then right forward at 8% for 250 ms. Two
explicit operator-requested executions each reported the same verified
controller deltas: 750 ms from acceptance to left completion, 750 ms from left
to right completion, and 2 ms from right completion to terminal safe state.
The priority-zero TIM5 ISR is the nominal cutoff; emergency output clearing and
a nominal 500 ms independent watchdog are backstops. This is not an
unconditional arbitrary-fault timing proof.

At the operator's later explicit request, KMC2 added a separate one-shot recipe
for both logical forward channels together at 8% for 10,000 TIM5 milliseconds.
The final Nano-built raw image at commit `6df5191` was 20,000 bytes, SHA-256
`2e8e971c5b4a5d024afd5f2def6f70798690692fa4a8a72b0f429c139a64b576`,
and matched its exact-length flash readback. The controller reported acceptance
at uptime 30,005 ms, completion of the both-forward segment at 40,505 ms (the
declared 500 ms initial zero dwell plus 10,000 ms active), and terminal safe
completion 3 ms later. The host observed and verified the complete sequence in
10,470 ms.

The long recipe configures the IWDG for a nominal 15,500 ms, waits for both
prescaler and reload update flags to clear, reloads while outputs are still
zero, and never feeds it while PWM is active. At the STM32F446 maximum specified
47 kHz LSI frequency, the 32 kHz model leaves at least the designed 500 ms after
the TIM5 cutoff; lower LSI frequency can make the watchdog interval much longer.
The timer evidence is therefore not an unconditional physical cutoff guarantee.
During this exact 10,000 ms run, the operator observed that neither motor shaft
moved. This is a failed physical-actuation check: the protocol and timer
evidence prove that the bounded recipe ran, but they do not establish that PWM
reached an energized motor driver or produced motor torque. No direction,
velocity response, start threshold, or MPC drive authority was established.
The cause remained unresolved at this point; possible causes must be
discriminated with evidence rather than attributed to wiring, power, firmware,
or an insufficient duty cycle by assumption.

After every one-shot check, the exact canonical 39,216-byte image was restored,
read back byte-for-byte with the canonical SHA-256 above, and re-probed. The
last observed state again reported maximum PWM 0, disabled safe output, and no
motion authority. Software timing does not establish visible shaft movement;
operator observation is recorded separately when supplied. For the long KMC2
run, that supplied observation was explicitly **no shaft motion on either
side**.

Read-only diagnosis found two June 2025 scripts outside the repository on the
Nano. The more complete script identifies a BTS7960 and explicitly drives its
`R_EN` and `L_EN` inputs high before applying 1 kHz `RPWM`/`LPWM`; its lowest
named demo duty is 30%. This is historical implementation evidence only. It
does not prove that either script describes the robot's present wiring, that
the named Jetson header pins still own the installed drivers, or that the
enable inputs were asserted during the STM32 KMC2 run. Every historical STM32
motor revision found in Git uses the same PA0/PA1 and PB4/PB5 timer channels as
KMC2, but the first two-bridge demo's smallest nonzero step was 50/255
(approximately 19.6%) and no preserved physical-success record establishes a
breakaway threshold. Consequently, neither increasing PWM nor asserting old
Jetson GPIO assignments can be treated as a valid diagnosis without a bounded
test and separate physical observation.

Commit `bb0f78d` added KMC3 as a separate commissioning-only discriminator;
the canonical KRP2 profile remained unchanged. KMC3 admits exactly one selected
wheel at 30% forward for 500 ms after a 500 ms zero dwell, requires a fresh
reset and fresh typed trigger for the other wheel, and has no automatic sweep.
Its 1,500 ms nominal watchdog profile has a designed minimum 500 ms margin
after the TIM5 cutoff under the datasheet's 47 kHz maximum LSI model. The final
source SHA-256 was
`1d37387becc2c7fca8632e68d6f40bd5b862557c696afcfd011f3e2e368326d1`.
The Nano-built thumb ELF SHA-256 was
`2a7628c8b15e7bd94eedc88a9755f3d510d3d52c73ff5156138fdb8ae67863a6`;
the 20,276-byte raw image SHA-256 was
`e100add1b2c6fcc5483a64f80e9c353775c8d21cefdc8d1c59b72568ddb3ecef`
and matched its exact-length flash readback. Twenty-two focused host tests,
host and thumb strict Clippy, and the thumb release build passed on the Nano.

The historical Nano enable candidates, BOARD 33 / `PH.00` and BOARD 26 /
`PZ.07`, initially read low. A two-second enable-only ownership check held both
high through one bounded `gpioset` process while canonical KRP2 remained at
zero, then released them; both read low afterward. For the first KMC3 physical
checks, one bounded process held those candidates high for at most four seconds
around each pulse and released them immediately after terminal evidence. The
controller verified a right pulse from accepted uptime 9,446 ms to completion
at 10,446 ms and a left repeat from 23,004 ms to 24,004 ms; each terminal safe
record followed 3 ms later. The operator explicitly reported visible right
shaft motion and visible left shaft motion. This establishes one finite
physical-motion observation for each selected output under those test
conditions. It does not establish shaft direction, velocity, the 30% duty as a
minimum or safe operating bound, present driver identity, or causation by the
two Nano enable candidates.

An otherwise identical right-side KMC3 pulse was subsequently executed while
both Nano enable candidates remained read back low before and after the run.
The controller accepted it at uptime 15,005 ms, reported pulse completion at
16,005 ms, and terminal safe state 3 ms later. The operator explicitly reported
visible right shaft motion again. This A/B observation shows that those two
historical Nano GPIO assignments were not required for the observed present
right-side actuation. Together with the failed 8% run, it supports—not proves—a
below-breakaway duty as the explanation for KMC2. It does not identify the
installed driver's actual enable wiring or establish a minimum duty.

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

## Subsequent eye commissioning

The operator placed the eye RP2350, and only that controller, into its UF2
bootloader. Linux identified it as USB `2e8a:000f`, model and board ID `RP2350`.
The boot volume did not expose `CURRENT.UF2`, so an exact readback backup of the
legacy flash was unavailable. A separately preserved legacy UF2 remains only a
fallback of uncertain equivalence and is not claimed as an exact backup.

The first KEP2 commissioning image reproduced a deterministic failure: Linux
could read its USB device and CDC descriptors but timed out setting
configuration 1 (`-110`), and no TTY appeared. The 16-byte OTP UID becomes a
32-character USB serial, whose UTF-16LE descriptor is 66 bytes. The firmware's
64-byte Embassy control buffer therefore hit Embassy USB 0.6's strict bounds
assertion, after which `panic-halt` stopped servicing USB. The corrected image
uses a 128-byte buffer and a compile-time proof that it exceeds the complete
serial descriptor.

The corrected commissioning image has:

- build ID `08134c20df747e68d38bea8af1eb8e62e86b085d347d8e18d5bf18301f368076`;
- ELF SHA-256 `5d0d8b962b33f5c2154fc7f7fe8c9f3d60f942a47296a78570553a9808f1ddd9`;
- UF2 SHA-256 `13dee99d699a0840e53874fee64b56da3fcc33054ea92a1e97da12124e5d4c94`;
- RP2350 ARM Secure UF2 metadata; and
- provisional right-panel X sign `+1`, which still requires a commanded
  left/right optical check.

After flashing, Linux configured CDC without another `-110` and created the
stable path
`/dev/serial/by-id/usb-kiko_kiko-eyes-kep2_98c47919804f9f1aaacfd5fa0a20bf74-if00`.
The native Nano identity-only probe opened that path exclusively, read back
115200 baud, 8 data bits, no parity, 1 stop bit, and no flow control, then sent
exactly one fresh nonce-challenged `IdentityQuery`. The exact
`IdentityReport` echoed the challenge and reported:

- device UID `98c47919804f9f1aaacfd5fa0a20bf74`, matching the USB serial;
- the expected build ID above;
- nonzero boot ID `420323034556454353`;
- protocol version 2; and
- capability bits `255`, satisfying all eight host-required capabilities.

The probe did not acquire control or transmit an expression. Independently,
the operator observed a newly centered eye with small autonomous movements
after the corrected flash. That is optical evidence that the new renderer is
driving the installed panel.

A later exact-identity KEP2 commissioning owner used a fresh session, finite
lease, bounded nine-step recipe, and confirmed release. The first lower-contrast
five-step protocol run was admitted by the firmware but the operator saw no
change, so it remains optically inconclusive. The deliberately unmistakable
second recipe used full brightness and commanded white center, red full-left
twice, blue full-right twice, three white blinks, and cyan neutral. The
controller admitted frame sequences 6 through 14 and confirmed release; the
operator then explicitly confirmed seeing that exact sequence. This qualifies
the installed system's commanded color, conjugate left/right gaze, blink, and
neutral-return semantics for that run. It does not calibrate gaze angle in
radians, prove every pixel, or replace long-duration display reliability.

The configured camera-to-head transform records the assembly statement that
the head center is 0.25 m above and 0.20 m behind the OAK, with neutral axes
parallel: OAK-frame translation `[0,-0.25,-0.20]` m. The host parses that
geometry into bounded domain types and computes right-positive yaw and
down-positive pitch, but this remains declared geometry rather than a measured
extrinsic or servo-angle calibration.

## Claims that remain open

- stable OAK USB 3 `SUPER`, production RGB/stereo/depth/IMU delivery, and native
  C++ bridge linkage (the exact-device `HIGH` stereo/IMU diagnostic passed);
- native aarch64 OAK/ONNX build provenance and a complete SLAM stream;
- fresh production-server applied-zero evidence and physical watchdog behavior;
- present motor-driver identity, enable/fault ownership, and energized-supply
  instrumentation (KMC3 established finite left/right shaft motion at 30% and
  supports, but does not prove, a below-breakaway explanation for KMC2);
- PWM-to-velocity and wheel-sign calibration, MPC tracking, and stop distance
  (bounded commissioning software sequences are not a drive-plant model);
- an unbounded continuously supervised natural head hold;
- calibrated eye/head gaze angles and long-duration display reliability;
- RGB person/face detection and head/eye gaze following;
- map persistence, cold-boot relocalization, autonomous exploration, and
  supervised point navigation on the production Nano owner.

Until those items have their own evidence, deployment must report them as
unknown rather than infer them from this discovery session.
