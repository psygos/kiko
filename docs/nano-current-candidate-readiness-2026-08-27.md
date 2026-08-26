# Current Nano candidate readiness — 2026-08-27

This ledger identifies the exact canonical source candidate and the evidence
available for the next attended wheels-off qualification. It prevents the live
Fable worktree, an old Nano binary, retained native libraries, or historical
calibration records from being mistaken for a qualified deployment.

No source was transferred to the Orin while preparing this record. No live
process was stopped or signalled, no endpoint was opened, and no firmware,
bundle, service, configuration, or hardware state was changed. This document
does not claim installation, Linux-aarch64 linkage, camera delivery, physical
head behavior, PID-1 supervision, SLAM accuracy, MPC tracking, or performance.

## Exact source candidate

| Field | Value |
| --- | --- |
| branch | `codex/nano-expression-integration-stage` |
| commit | `60983f0b60fa28f4da9dc61af96e0d883cd7c9d9` |
| Git tree | `4a6ba7f0ddaaed2e0f9b63f540391131b9e8099c` |
| source tar | `/tmp/kiko-60983f0-source.tar` |
| source tar bytes | `191631360` |
| source tar SHA-256 | `f0a099bf1ef4be91b1ce5a2298b2359269afed938569a308a844b16a1bfb4e05` |
| compressed source tar | `/tmp/kiko-60983f0-source.tar.gz` |
| compressed bytes | `164836698` |
| compressed SHA-256 | `1d8cf717840e32c8cde696c51e48990b894e87689d09de76b801dda72b6b4e06` |

The archive size includes the Git-tracked ONNX model assets. The archive is an
ephemeral local preparation artifact, not a checked-in or qualified release.
The private Orin directory `/home/makerspace/kiko-candidate-60983f0` exists and
was empty at `2026-08-27T02:37:48+05:30`; the archive has **not** been copied
there. Transferring this private repository source requires explicit approval
of that exact payload and destination.

## Live Orin state retained during preparation

The read-only refresh at `2026-08-27T02:37:48+05:30` observed an NVIDIA Jetson
Orin Nano running Linux `aarch64`. `/home/makerspace/kiko` remained at
`e53d7cb084a9b56f49df484f6d8bc7f46f0b39e6`; it is the intentionally preserved
dirty field worktree and must not be reset, overwritten, or used as a clean
candidate build source.

Fable's guardian and `kiko_face_follow.py` remained the sole live owner family.
The current stable serial paths were:

```text
head  /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00
base  /dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066EFF313946303143221230-if02
eyes  /dev/serial/by-id/usb-kiko_kiko-eyes-kep2_98c47919804f9f1aaacfd5fa0a20bf74-if00
```

These are read-only topology observations, not fresh protocol admission. The
qualification process must probe the exact controller, head, and KEP2
identities again after the explicit single-owner handoff.

## Located candidate leaf assets

The checked-in models selected by qualification input V4 are:

| Role | Candidate source | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| SuperPoint | `crates/kiko-slam/models/sp.onnx` | `5226093` | `aaefb94ad6dd3624fe4300b39f0f1a77e8739ed6d5430162729fd6a72c265431` |
| LightGlue | `crates/kiko-slam/models/lg.onnx` | `46463559` | `7fbb5814811dbc6d170de1c86bc0352a14691efa32cae33d952b6039258f74ef` |

The following retained Orin closure is available as a build input candidate:

`/home/makerspace/kiko-native-evidence/3f262f1-20260727T013755Z/lib`

| Required role | Bytes | SHA-256 |
| --- | ---: | --- |
| `libdepthai-core.so` | `42632152` | `0744500ab4f665af0641fd10881988146b73241212ac9523a86294e5737edae8` |
| `libdynamic_calibration.so` | `36820008` | `30730ae6d367dcd927be7081f6a21d3bc4af65d857421ea3d3776d4ac00c7c53` |
| `libusb-1.0.so` | `202888` | `74eac03235e61b326ecb6532bd1d840f7b8fbaf55cfaa32b7e3079fc1208ede0` |
| `libonnxruntime.so.1` | `25969728` | `5246cdc32cf54afe0a108b9326f232ed1ed2bfcb9b4431738e2ad35eb20329aa` |
| `libopencv_core.so.4.5d` | `2607080` | `3abc549967c52f594b2b597db44b0013c55edb2198e11f9110d564277eb00beb` |
| `libopencv_imgproc.so.4.5d` | `2906064` | `15b2448af215493a79f4638cad8eefcb9b43f15926724caffbdbd06a9c018261` |
| `libopencv_objdetect.so.4.5d` | `366632` | `94d3ddfb2111e72658d4bd005d22fd0ce402f8ae45ff8a79e9f7bdbd9b194b0b` |

That closure was retained for commit `3f262f1`. It may supply the seven exact
native leaves, but it is not current-candidate linkage evidence. The new
qualification executable still requires fresh `readelf`, `ldd`, byte-identity,
and `/proc/self/maps` admission against the rendered V4 bundle.

The Orin's installed OpenCV cascades are available at:

| Role | Source | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| frontal | `/home/makerspace/.local/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml` | `930127` | `0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0` |
| profile | `/home/makerspace/.local/lib/python3.10/site-packages/cv2/data/haarcascade_profileface.xml` | `828514` | `b39a4a3be45539db146a7fc1d3e761a292c196eb88421185e6a615b3055e612d` |

Their presence and hashes do not claim detector quality or throughput.

## Retained calibration inputs and their limits

The following exact retained records were present and matched their documented
digests:

| Record | SHA-256 | Permitted use |
| --- | --- | --- |
| `/home/makerspace/kiko-native-evidence/4238f14-calibration-stereo-20260729T001905+0530/calibration.json` | `13cdf7c81036b9983f4f12f51b4353295468b88de896ec0d7bd3a37a1147503e` | historical stereo candidate; must be superseded or bound by a fresh exact-candidate capture |
| `/home/makerspace/kiko-native-evidence/4238f14-oak-s2-board-baseline-20260729T005342+0530/OAK-D-S2.json` | `b6c50050a9d45bd28d76102cfc44ff399591f10f5518b0eedc47d15b83c28281` | independent `0.075 m` board-design baseline corroboration |
| `/home/makerspace/calibration/oakd-basalt/results_rectified_imu_seed/calibration_recommended_first_pass.json` | `1361d60c4cad7e5c4aa0d43945e6eeea72fd23922759be42281c094620c58c27` | Basalt candidate only; not a canonical raw-IMU calibration |

None is a current `NanoCalibrationArtifactV1`. In particular, neither the OAK
EEPROM transform nor the declared head-centre offset may be relabelled as the
tracking-camera-to-base or native-IMU-to-base transform.

## Gate-A V4 inputs: known and still missing

The checked-in qualification template already fixes the 640 by 400 at 15 Hz
RGB/stereo/depth graph, 200 Hz IMU, SuperSpeed requirement, 20 m by 20 m global
occupancy at 0.05 m resolution, CPU SuperPoint/LightGlue policy, Rerun bounds,
shadow-only synthetic plant, MPC shadow bounds, current four-servo natural
return policy, and expression head origin `[0,-0.25,-0.20] m`. Gate A omits
expressive physical head gaze while retaining the reviewed natural hold.

The following must still be produced or freshly observed:

1. a release `aarch64-unknown-linux-gnu` qualification executable built from
   the exact candidate commit with `--locked` and the
   `nano-wheels-off-qualification` feature;
2. the compiled DepthAI header SDK, commit, device-artifact, and bootloader
   identities from that exact build;
3. fresh exact OAK, STM32 protocol/build/profile/capability, head-adapter, and
   KEP2 session/build/capability admission;
4. a fresh finalized candidate-recorder capture at the V4 stream geometry;
5. a measurement record naming the drive-axle midpoint base frame and the OAK
   rectified-left optical centre, with their translation and proper rotation;
6. a sourced proper rotation from native OAK IMU axes to the same base frame;
7. the seven frame-specific navigation leaves: world-to-occupancy rotation and
   translation, inflated footprint radius, global floor-relative obstacle
   minimum/maximum, and axle-centred base-frame obstacle-z minimum/maximum;
8. a canonical calibration-artifact V1 and navigation-shadow V2 generated
   transactionally by `kiko-nano-calibration-prepare`; and
9. a completely materialized bundle-render-input V4 followed by an immutable
   wheels-off qualification launch V4.

The operator declarations retained so far are useful but insufficient:

- head centre is `0.25 m` above and `0.20 m` behind the OAK, with parallel
  neutral axes; this is expression gaze geometry only;
- footprint body radius is `0.293 m`; it lacks the reviewed navigation margin;
- axle midpoint is `0.05 m` above the floor;
- OAK housing is declared `0.1485 m` forward and `0.30 m` up from an unnamed
  “centre”; the datum is ambiguous and is a housing surface, not the optical
  centre.

No transform, rotation, obstacle slab, or clearance margin may be guessed from
those declarations.

## Exact next transaction

1. Obtain explicit approval to transfer the exact compressed candidate above
   to `/home/makerspace/kiko-candidate-60983f0`.
2. Extract into that isolated private directory and prove the Git tree/source
   identity before building; never build from `/home/makerspace/kiko`.
3. Record the Orin toolchain and run:

   ```text
   cargo build --locked --release -p kiko-slam \
     --no-default-features \
     --features nano-wheels-off-qualification \
     --bin kiko-slam
   ```

4. Retain the executable hash, byte count, `file`, `readelf`, `ldd`, model,
   cascade, native-leaf, and compiled-DepthAI evidence without installing it.
5. During the attended single-owner window, stop the Fable owner exactly once,
   acquire the OAK with the exact candidate recorder, and either finish the
   capture and calibration inputs or restore Fable before leaving the window.
6. Prepare calibration and navigation with the fail-closed assembler; render
   and inspect the immutable V4 bundle before any installation.
7. With wheels removed, head supported, motor power physically disconnected,
   and the power cut reachable, launch the qualification binary once in the
   foreground and complete the challenge-gated procedure in
   `docs/nano-wheels-off-qualification.md`.
8. Prove the sole-owner OAK/head/eye/STM32 graph, natural hold and pet response,
   live RGB/stereo/depth/IMU, sparse-SLAM completions, occupancy, Rerun, GUI
   deadman and emergency stop, MPC shadow stream, watchdog, and every terminal
   stop/shutdown path. Restore Fable on failure; never run both owners.

Only that attended wheels-off evidence can justify requesting wheel
attachment. Wheel-on work must then calibrate encoderless wheel signs,
breakaway, PWM-to-velocity response, effective wheelbase, timing, and stop
distance from synchronized visual translation and calibrated IMU yaw before
MPC, mapping, exploration, relocalization, or click-to-goal is called ready.
