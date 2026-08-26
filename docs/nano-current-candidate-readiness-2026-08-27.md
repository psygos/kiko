# Current Nano candidate readiness — 2026-08-27

This ledger identifies the exact canonical source candidate and the evidence
available for the next attended wheels-off qualification. It prevents the live
Fable worktree, an old Nano binary, retained native libraries, or historical
calibration records from being mistaken for a qualified deployment.

The approved `60983f0` archive was transferred to the Orin, extracted, and
built in an isolated private directory. That build exposed one release-binary
provenance defect: generic development model defaults embedded the build
machine's absolute source directory. Commit `9deda40` removes that fallback
and adds regression coverage. Building the offline bundle renderer then exposed
a second feature-graph defect: its production dependency graph could not import
a byte bound that was hidden behind OAK recording features. Commit `5680472`
moves that bound to the hardware-neutral policy contract and adds an exact CI
check. It is now the hardened candidate. Its exact source archive and three
executables are staged only below
`/home/makerspace/kiko-candidate-5680472`; none is installed or deployed.

No live process was stopped or signalled, no hardware endpoint was opened, and
no firmware, bundle, service, configuration, or hardware state was changed.
Fable remained live throughout. This document does not claim camera delivery,
physical head behavior, PID-1 supervision, SLAM accuracy, MPC tracking, or
performance.

## Superseded approved transfer

| Field | Value |
| --- | --- |
| branch | `codex/nano-expression-integration-stage` |
| commit | `60983f0b60fa28f4da9dc61af96e0d883cd7c9d9` |
| Git tree | `4a6ba7f0ddaaed2e0f9b63f540391131b9e8099c` |
| compressed source tar | `/tmp/kiko-60983f0-source.tar.gz` |
| compressed bytes | `164836698` |
| compressed SHA-256 | `1d8cf717840e32c8cde696c51e48990b894e87689d09de76b801dda72b6b4e06` |
| isolated Orin root | `/home/makerspace/kiko-candidate-60983f0` |

The local and transferred archives matched byte-for-byte. The archive includes
the Git-tracked ONNX model assets. The locked native AArch64 release build
completed in that isolated root without modifying `/home/makerspace/kiko`.
The first executable was invalidated after inspection found the absolute source
path described above; it is not the staged qualification candidate.

## Intermediate hardened candidate

| Field | Value |
| --- | --- |
| branch | `codex/nano-expression-integration-stage` |
| commit | `9deda40ec0638b7e84dbcd90fa09cc574e831d85` |
| Git tree | `2c41e5e1adce68c8c0d6a8788caceaf57e13d803` |
| source tar | `/tmp/kiko-9deda40-source.tar` |
| source tar bytes | `191641600` |
| source tar SHA-256 | `8c2432cd1991ac181e27f4b1c6a36e679dbe373a4ba964ae673f6a9584a3b68c` |
| compressed source tar | `/tmp/kiko-9deda40-source.tar.gz` |
| compressed bytes | `164892982` |
| compressed SHA-256 | `e6d5a5ed1990692721c020c59c7d99dd07cb041fa7ac9fd0a55e099e740979db` |
| isolated Orin root | `/home/makerspace/kiko-candidate-9deda40` |

The remote compressed archive matched that SHA-256 after transfer. The staged
executable was built from the `60983f0` archive plus a SHA-verified patch whose
resulting five source files matched `9deda40` byte-for-byte. The exact
`9deda40` archive was then transferred and extracted beside it. This is exact
source identity evidence, but not a claim that a second cold build was made
from the later directory.

## Final exact candidate

| Field | Value |
| --- | --- |
| branch | `codex/nano-expression-integration-stage` |
| commit | `568047255446343dd3c0ce8e08900d7c829b3b9d` |
| Git tree | `cafc1419022d021575741fea40a29c35f2cb75d0` |
| source tar | `/tmp/kiko-5680472-source.tar` |
| source tar bytes | `191651840` |
| source tar SHA-256 | `28b1ccb2dc3e8021eb2566f125840e6d75c7251ccaeb2ba71335ffb64e67d4c9` |
| compressed source tar | `/tmp/kiko-5680472-source.tar.gz` |
| compressed bytes | `164842868` |
| compressed SHA-256 | `1b37b67c3954c29f221c6a8c074ce3e118b1318daa1f081f3447e14dcf3696b7` |
| isolated Orin root | `/home/makerspace/kiko-candidate-5680472` |

The transferred archive matched the local compressed bytes and was extracted
into the private final-candidate root. Hashes of the four source files changed
by `5680472` matched the local commit. The three binaries below were built from
the `60983f0` archive after applying the SHA-verified `9deda40` and `5680472`
patches; the resulting source files match the final archive. No claim of a
second cold build from the final directory is made.

## Isolated Linux AArch64 build evidence

The build used Cargo's locked release graph and exactly the production
qualification feature:

```text
cargo build --locked --release -p kiko-slam \
  --no-default-features \
  --features nano-wheels-off-qualification \
  --bin kiko-slam
```

The build host and toolchain were:

| Field | Observed value |
| --- | --- |
| kernel | `Linux 5.15.148-tegra aarch64` |
| rustc | `1.88.0 (6b00bc388 2025-06-23)`, host `aarch64-unknown-linux-gnu`, LLVM `20.1.5` |
| cargo | `1.88.0 (873a06493 2025-05-10)`, host `aarch64-unknown-linux-gnu` |
| C compiler | Ubuntu GCC `11.4.0` |
| DepthAI headers | SDK `3.4.0`, commit `ba7a920a2568ea6eaaaebf3f92bbdb40924187ae`, device artifact `86cc8f6aa527b7c1f4b62129decd68e12bcf7d8a`, bootloader `0.0.28` |
| DepthAI version header SHA-256 | `91eded0aa1468a5e8ca7ee13b51f2e2f8475c616922a59c684ecb59fc61e6e80` |

The current runtime executable is retained, not installed:

| Field | Value |
| --- | --- |
| path | `/home/makerspace/kiko-candidate-5680472/bin/kiko-nano-wheels-off-qualification` |
| bytes | `31221240` |
| SHA-256 | `6b0c2ee3ce0184975d35d8d13618b3c14c2e0f275982fd5cc5e753d00d63a64d` |
| GNU build ID | `c44ed1c959faa6dae6b93f26857e2a1ef9115bc9` |
| format | ELF64 AArch64 PIE, GNU/Linux 3.7, interpreter `/lib/ld-linux-aarch64.so.1` |
| mode and owner | `0755`, `makerspace:makerspace` |

Two matching-revision offline tools are staged beside it:

| Tool | Bytes | SHA-256 | GNU build ID |
| --- | ---: | --- | --- |
| `kiko-nano-bundle-renderer` | `2971536` | `f0f26c56c57816ec1079a32aeeb2cf3e484ccd31e5e62da38d8eaebbe0bb38e8` | `71f320a6bc43370bf9e4ace1289f827787578990` |
| `kiko-nano-calibration-prepare` | `2276784` | `9fe343b61834e48c675658874916bc39439485a46f7206df1b152326e7a95784` | `695b5816198ae1985fa2f4c0f0302b91887ef108` |

Both are AArch64 PIE executables, have complete system dependency closures,
and ran their `--help` boundaries. The renderer explicitly describes itself as
offline and non-installing; the preparer explicitly describes itself as doing
no device I/O. Neither was invoked on an input or output directory.

`readelf` found direct dependencies on DepthAI, the three required OpenCV
libraries, and the expected C/C++ runtime libraries. With the retained native
directory on `LD_LIBRARY_PATH`, `ldd` resolved DepthAI, dynamic calibration,
OpenCV, and the pinned libusb leaf. A system libusb also remains in the
transitive closure, so this is not described as a wholly hermetic OS closure.
The exact build source path is absent from all three hardened executables.
`--help` ran successfully and exposed the qualification command without
opening a hardware endpoint. Runtime `/proc/self/maps` admission remains a
bundle-launch gate.

Host verification for the exact wheels-off feature graph passed:

- `cargo clippy --locked ... --all-targets -- -D warnings`;
- 1,538 library unit tests;
- 7 deployment-qualifier tests;
- 103 CLI/runtime unit tests;
- 7 qualification-template integration tests; and
- doctests, with one intentionally ignored backend example.

After the Orin found the renderer-only graph defect, its fix also passed the
exact production-feature-only renderer library check, strict renderer Clippy,
all 36 renderer integration tests, and renderer doctests. The CI workflow now
keeps that smaller graph separate because `--all-targets` feature unification
can otherwise hide the missing-export class of error.
The exact standalone `nano-calibration-prepare` feature/bin graph separately
passed strict Clippy and all 14 preparer tests, including duplicate-key,
sentinel, provenance, unit, affine-stability, baseline-discrepancy, bit-exact
binding, cleanup, no-replace race, and transactional-publication coverage.

The first in-sandbox test pass produced 27 local-socket `EPERM` failures; the
identical command was rerun with local socket permission and all tests passed.
`OAK_SYS_CHECK_ONLY=1` intentionally makes these host checks compile-only for
the native OAK bridge. The native bridge is instead evidenced by the Orin
release build above.

## Live Orin state retained during preparation

The final read-only refresh at `2026-08-27T04:55:54+05:30` observed an NVIDIA
Jetson Orin Nano running Linux `aarch64`. `/home/makerspace/kiko` remained at
`e53d7cb084a9b56f49df484f6d8bc7f46f0b39e6`; it is the intentionally preserved
dirty field worktree and must not be reset, overwritten, or used as a clean
candidate build source.

Fable's guardian and `kiko_face_follow.py` remained the sole live owner family.
The face-follow process was still PID `1073`, its heartbeat advanced at the
final observation, and no `kiko-slam` or staged qualification process was
running. The source transfer, cold build, incremental hardened rebuild, and
artifact staging therefore did not replace the live owner.
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

That closure was originally retained for commit `3f262f1`. The current
candidate was compiled and link-inspected against it, so its seven leaf hashes
are current build inputs rather than merely located candidates. Bundle
rendering must copy and re-hash those exact leaves, and the launched process
must still prove the admitted mappings through `/proc/self/maps`.

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

The source, AArch64 build, compiled DepthAI identities, executable inspection,
and host feature-graph verification are complete. The following must still be
produced or freshly observed:

1. fresh exact OAK, STM32 protocol/build/profile/capability, head-adapter, and
   KEP2 session/build/capability admission;
2. a fresh finalized candidate-recorder capture at the V4 stream geometry;
3. a measurement record naming the drive-axle midpoint base frame and the OAK
   rectified-left optical centre, with their translation and proper rotation;
4. a sourced proper rotation from native OAK IMU axes to the same base frame;
5. the seven frame-specific navigation leaves: world-to-occupancy rotation and
   translation, inflated footprint radius, global floor-relative obstacle
   minimum/maximum, and axle-centred base-frame obstacle-z minimum/maximum;
6. a canonical calibration-artifact V1 and navigation-shadow V2 generated
   transactionally by `kiko-nano-calibration-prepare`; and
7. a completely materialized bundle-render-input V4 followed by an immutable
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

Transfer, isolated build, executable inspection, hardened source staging, and
host verification are complete. The next transaction begins only after the
operator explicitly supersedes the current **do not stop Fable or deploy yet**
constraint:

1. During an attended single-owner window, stop the Fable owner exactly once,
   acquire the OAK with the exact candidate recorder, and either finish the
   capture and calibration inputs or restore Fable before leaving the window.
2. Prepare calibration and navigation with the fail-closed assembler; render
   and inspect the immutable V4 bundle before any installation.
3. With wheels removed, head supported, motor power physically disconnected,
   and the power cut reachable, launch the qualification binary once in the
   foreground and complete the challenge-gated procedure in
   `docs/nano-wheels-off-qualification.md`.
4. Prove the sole-owner OAK/head/eye/STM32 graph, natural hold and pet response,
   live RGB/stereo/depth/IMU, sparse-SLAM completions, occupancy, Rerun, GUI
   deadman and emergency stop, MPC shadow stream, watchdog, and every terminal
   stop/shutdown path. Restore Fable on failure; never run both owners.

Only that attended wheels-off evidence can justify requesting wheel
attachment. Wheel-on work must then calibrate encoderless wheel signs,
breakaway, PWM-to-velocity response, effective wheelbase, timing, and stop
distance from synchronized visual translation and calibrated IMU yaw before
MPC, mapping, exploration, relocalization, or click-to-goal is called ready.
