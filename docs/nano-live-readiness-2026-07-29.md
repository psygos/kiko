# Nano live-readiness evidence — 2026-07-29

This record captures the latest read-only and motor-inert observations made
before rendering the attended wheels-off qualification bundle. It is not a
wheel-attachment, motion, calibration-quality, SLAM, occupancy, MPC, or
production-deployment claim.

## Host software verification refresh

The verified integration source is frozen at
`12a611371c86e252e2c80afa8526a82c11849731`, comprising the intentional
post-`4238f1463cdc25b341db2e0770ec582f53686af2` commits `1c381ec`,
`35bc861`, and `12a6113`. No source changed between the final test runs and
that source commit; only this evidence record was finalized afterward. Every
host OAK build below used `OAK_SYS_CHECK_ONLY=1`; this checks the compile-time
boundary and deliberately does not open or emulate an OAK device.

- `cargo test --workspace --locked` passed with ordinary local Unix-socket and
  loopback-listener permission. The same run inside the restricted filesystem
  sandbox first produced only `Operation not permitted` bind failures; it was
  rerun unchanged outside that sandbox.
- The exact production `nano-agent` feature graph passed 1,418 library tests,
  seven deployment-qualifier binary tests, 76 main-binary tests, four
  template-boundary tests, and its compile-fail documentation test.
- The exact `nano-wheels-off-qualification` graph passed 1,488 library tests,
  seven deployment-qualifier binary tests, 94 main-binary tests, five
  template-boundary tests, and its compile-fail documentation test.
- The exact `nano-base-commissioning` graph passed 1,349 library tests, three
  commissioning-binary tests, 57 main-binary tests, four template-boundary
  tests, and its compile-fail documentation test.
- The complete
  `nano-agent,nano-base-commissioning,nano-wheels-off-qualification,nano-plant-promotion`
  feature union passed 1,528 library tests, three commissioning-binary tests,
  seven deployment-qualifier tests, two plant-promotion CLI tests, 94
  main-binary tests, five template-boundary tests, and its compile-fail
  documentation test.
- Strict workspace Clippy and that complete all-target feature union passed
  with `-D warnings`.
- The focused plant-promotion adversarial set passed 6/6, its commissioning
  set passed 16/16, and its CLI set passed 2/2.
- The operator-console Node test command passed eight tests.
- `tools/nano-cold-boot-fault-acceptance.sh` returned
  `acceptance_result=pass`.

These are software results, not installation, PID-1 execution, cold-power
boot, device ownership, camera delivery, physical watchdog/E-stop, wheel
motion, SLAM accuracy, MPC tracking, or performance evidence.

The Nano checkout was clean at
`4238f1463cdc25b341db2e0770ec582f53686af2` for every observation below.
The legacy `/home/makerspace/kiko-follow/engine-guardian.sh` process remained
stopped, its automatic-launch authority remained present, and the OAK, head,
eye, and STM32 serial endpoints had no active child owner before each probe.
That stopped guardian and its retained crontab are still a deployment conflict;
they must be replaced deliberately by the canonical owner before Gate A.

## Exact OAK full-stream transport

Evidence directory:

```text
/home/makerspace/kiko-native-evidence/4238f14-oak-SUPER-a2-20260729T003023+0530
```

The release `oak_stream_qualify` executable had SHA-256
`7ba87dacc9fdedbe5bbb6a5b065ed13a9ab388c47026bb02441f4e12e1911c0b`.
Its exact invocation requested the OAK with MXID
`19443010F1B43A2E00`, maximum `SUPER`, minimum `SUPER`, 150 frames per
image stream, and a 20-second outer bound. The connected device reported
`SUPER`.

After one accepted all-queue-empty epoch, the 10.009947476-second measurement
delivered:

| stream | delivered | native sequence defects |
| --- | ---: | ---: |
| RGB 640×400 @ 15 Hz | 150 | 0 |
| rectified left 640×400 @ 15 Hz | 150 | 0 |
| rectified right 640×400 @ 15 Hz | 150 | 0 |
| rectified-left depth 640×400 @ 15 Hz | 150 | 0 |
| IMU requested at 200 Hz | 1,980 samples | not sequence-labelled |

There were zero image-sequence gaps, duplicates, or regressions. The measured
host-delivery rates were approximately 14.985 frames/s for each image stream
and 197.803 IMU samples/s. These are bounded host-delivery observations for
this graph, not USB line-capacity, sustained-performance, thermal, or SLAM
measurements. The retained result SHA-256 is
`77466710e0d2b46dbf904a07551b1bdd03ce02cb5d04d2c45e98394804d0352e`.

## Fresh stereo/depth dataset

Evidence directory:

```text
/home/makerspace/kiko-native-evidence/4238f14-calibration-stereo-20260729T001905+0530
```

The finalized dataset contains 768 regular files and occupied 402,593,897
bytes at observation time. Its manifest has 248 paired left/right records,
zero manifest orphans, and zero manifest drops. The maximum paired timestamp
delta is 1,096,606 ns under the declared 5,000,000 ns window.

Exact source hashes:

```text
calibration.json  13cdf7c81036b9983f4f12f51b4353295468b88de896ec0d7bd3a37a1147503e
meta.json         012a87f13a56a5d0b315a863e2ba7c1fbe910d12b226da524bc3f92050825a62
manifest.json     f3aa0cfdd1399155b41768998bbf3a3d7792e18c7a27012cccb84d1091334313
```

The rectified calibration observation is:

```text
left:  fx=398.1716  fy=398.1898  cx=308.64267  cy=199.88481
right: fx=396.992   fy=397.00247 cx=326.84726  cy=194.88861
size:  640×400
baseline_m: 0.07503394
```

Opening the finalized dataset through the release reader reported 248 paired
records at an observed dataset rate of 13.81 pairs/s. A bounded three-pair CPU
pipeline check decoded all selected payloads and produced nonzero matches for
all three pairs. That check is data-integrity evidence only; its CPU timing is
not a Nano inference-performance or production-backend claim.

The same recorder could not create the requested combined IMU dataset because
the OAK EEPROM reported no IMU calibration/extrinsics. The successful dataset
therefore has `imu: null`. Runtime IMU delivery is independently present in the
full-stream result above, while canonical native-IMU-to-base calibration
remains a separate physical input.

## Independent OAK-D S2 baseline

The operator identified the installed camera as an OAK-D S2. A clean upstream
OAK-D S2 board-design declaration was copied into:

```text
/home/makerspace/kiko-native-evidence/4238f14-oak-s2-board-baseline-20260729T005342+0530
```

The source `OAK-D-S2.json` has SHA-256
`b6c50050a9d45bd28d76102cfc44ff399591f10f5518b0eedc47d15b83c28281`
and declares a 7.5 cm left-to-right mono-camera design translation. Its
independently derived `0.075 m` design baseline differs from the fresh live
`0.07503394 m` value by `0.045232864%` using the displayed decimal values, or
`0.0452294%` using the preparer's parsed `f32` inputs. Both are below its 2%
consistency gate. The exact source revisions, interpretation, and claim
boundary are in `docs/nano-oak-s2-baseline-evidence-2026-07-29.md`.

## Eye identity

Evidence directory:

```text
/home/makerspace/kiko-hardware-evidence/20260729T003738IST-4238f14-eye-identity
```

One fresh nonce-bound KEP2 identity query, with no control acquisition or
expression write, reported:

```text
UID:                 98c47919804f9f1aaacfd5fa0a20bf74
protocol:            2
firmware build:      08134c20df747e68d38bea8af1eb8e62e86b085d347d8e18d5bf18301f368076
capabilities bits:   255
required supported:  true
boot ID:             6166948355183154669
```

The retained result SHA-256 is
`dba5338527695b5a1b3e57f25379283dcfa9c76ef6e2a79d45e376715b8367b4`.
This is challenged firmware identity evidence, not optical-display evidence.

## Head read-only state

Evidence directory:

```text
/home/makerspace/kiko-hardware-evidence/20260729T003811IST-4238f14-head-readonly
```

The canonical 1,000,000-baud, DTR-false, RTS-true probe performed telemetry
reads only. It observed all four joints stopped with torque-switch raw value
one and zero framing-noise bytes:

| joint | position ticks | voltage raw | temperature raw | current raw |
| --- | ---: | ---: | ---: | ---: |
| bow | 2,153 | 120 | 45 | 10 |
| curl | 2,640 | 119 | 49 | 20 |
| yaw | 1,832 | 119 | 35 | 0 |
| roll | 3,044 | 119 | 36 | 0 |

A later read-only probe observed the same pose and temperature raw
`[45,48,35,36]`. The 3,044-tick roll reading is one tick above the earlier
five-sample stopped-pose record. Neither probe reported a device-status or
framing fault. The retained first result SHA-256 is
`311a6e432f357f57f646bc669759d47c5b5fbce197834f27f8cdfc5a7e960599`.
These raw fields are not physical-unit calibration or long-duration thermal
evidence.

## STM32 motor-inert identity

Evidence directory:

```text
/home/makerspace/kiko-hardware-evidence/20260729T003910IST-4238f14-stm32-identity
```

The read-only KRP2 probe transmitted no serial bytes and reported:

```text
controller UID:       2c0018001750314242353320
firmware ABI/build:   2 / 131074
fingerprint:          4b494b4f2d4e4f2d4143542d56312121
capabilities bits:    319
maximum PWM:          0%
output state:         disabled
motion authority:     false
watchdog period:      250 ms
boot ID:              12638770094519703627
```

The retained result SHA-256 is
`ec5a135a726ce53b852928f963ebc28948fa3a4ba2e8ed89b95ec60fe12c22ab`.
This proves the currently installed controller is motor-inert. It cannot drive
the wheels or qualify the motion-capable command path.

## Remaining gates

Before the attended qualification bundle can run truthfully:

1. retain the tracking-camera-optical-frame to wheel-axle-base transform and
   the native-IMU-to-base proper rotation;
2. retain the reviewed world-to-occupancy transform, footprint radius, and
   separately framed floor-relative and base-relative obstacle-height
   intervals;
3. build and flash the exact motion-capable commissioning image only after a
   fresh wheels-removed, motor-power-disconnected, head-supported attestation;
4. re-probe that image and require exact applied-zero and disarm receipts;
5. render and check a sentinel-free immutable qualification bundle;
6. replace the legacy guardian's launch authority with the single canonical
   owner; and
7. execute the console deadman, release, E-stop, disconnect, SLAM, occupancy,
   Rerun, head/eye, and fault-injection matrix before attaching wheels.

Wheel-on plant identification and physical MPC driving remain subsequent,
attended gates. No entry in this record should be used to bypass them.
