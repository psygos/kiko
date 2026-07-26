# Nano integration acceptance report — 2026-07-24

## Decision

The host-side integration is a tested, fail-closed release candidate. It
provides one strict startup graph, one STM32 owner, one OAK owner, supervised
head/eye behavior, live SLAM and geometric occupancy, a unified browser/agent
control plane, bounded map and dataset persistence, attended wheels-off
qualification, and attended wheel-on identification.

The attended calibration-only wheel-attach handoff remains **closed** until
Gate A in `docs/nano-wheel-attach-gate-2026-07-23.md` has fresh Nano evidence.
That gate requires the coordinated device handoff, canonical SuperSpeed OAK
graph, exact-zero candidate, accessories, live SLAM/occupancy/Rerun, unified
console, wheels-off commands/faults, and an independently tested reachable
motor-power cut. Passing it authorizes attachment and the separate attended
commissioning executable only.

Production motion is a second, stricter gate. Under the evidenced PA0/PA1 plus
PB4/PB5 four-PWM wiring, the repository has no known default-off external
driver-enable line and no driver-fault/E-stop feedback input. The production
controller contract therefore cannot truthfully advertise
`production_external_interlocks`. Host code does not replace physical wiring,
reviewed commissioning evidence, or an independent cut with a boolean, prompt,
or manifest claim.

## Delivered software

- `kiko-slam nano-agent` owns the admitted OAK, RGB/stereo/depth/IMU graph,
  online SLAM, local and global occupancy, Rerun publication, head, eyes,
  STM32 lifecycle, control API, and console in one process.
- Startup is parse-once and fail-closed. Launch, policy, device inventory,
  calibration, plant, models, native libraries, serial identity, USB
  transport, and installed bytes are cross-bound before production ownership.
- The natural head target is continuously supervised. Eye behavior consumes
  RGB borrowed from the sole OAK owner; the browser never opens the camera.
- Manual, map-only, frontier exploration, and revision-bound point goals share
  one authority arbiter and one downstream controller sequence.
- The browser and agent API expose pose, occupancy, selected goal, global path,
  MPC rollout, requested output, exact applied STM32 receipts, stop certainty,
  subsystem health, and tick timing.
- Manual input has a server-side monotonic deadman. Blur, release, page hide,
  stale connection, and explicit release enter the same stop path.
- The process-lifetime software safety stop has a priority ingress and cannot
  be remotely reset. It is always labelled separately from the independent
  physical emergency stop.
- New manual, frontier, and point-goal authority now requires the
  coordinator's at-now odometry and aligned local-depth freshness evidence.
  OAK dashboard health is no longer an ever-seen latch: recent visual, depth,
  and IMU activity plus the same typed coordinator readiness are required.
  Periodic motion retains its independent stop-on-stale gate.
- The STM32 KRP2 path has bounded UART work, record-preserving transmit queues,
  reserved stop/applied-result capacity, priority/coalesced host stop, exact
  acknowledgements, deadline/watchdog enforcement, and measured-evidence
  fields. The motor-inert diagnostic transport was measured at 20 Hz and
  50 Hz on the Nano; the motion-capable production command path and end-to-end
  MPC loop were not measured.
- Map finalization is a terminal, causal drain. The selected manifest and
  occupancy bytes are descriptor-bound, checksummed, quota-bounded, and
  replayed only as an exact warm-start candidate. A loaded map never claims
  current localization until live relocalization succeeds.
- Live navigation datasets have typed byte, file, ingress-record, free-space,
  inode, and terminal-reserve limits. Quota failure poisons the session and
  prevents a manifest from being published.
- The old split desktop PWM client, standalone production motor server, and
  duplicate wheels-off abstractions were removed.
- Deployment rendering, offline install qualification, a marker-gated systemd
  unit, and a bounded cold-boot/fault acceptance script are included. The base
  unit cannot silently fall back to an unqualified start.

## Verification evidence

The following historical frozen-tree matrix completed before the final
sensor-freshness hardening and before mandatory launch-V3 face perception.
Its counts are retained as provenance; they are not current V3-tree evidence:

| Graph | Evidence |
| --- | --- |
| robot protocol | 34 tests and strict Clippy passed |
| command client | 16 tests, 2 doctests, and strict Clippy passed |
| robot server | 93 tests and strict Clippy passed |
| embedded pure logic | 60 tests and strict Clippy passed |
| boot-journal tool | 2 tests and strict Clippy passed |
| Nano support crates | 305 tests, 3 doctests, and strict Clippy passed |
| OAK boundary | 24 tests and strict Clippy passed with `OAK_SYS_CHECK_ONLY=1` |
| KEP2 eye host logic | 20 tests and strict Clippy passed |
| Kiko minimal | 1,061 tests passed |
| Kiko production graph | 1,394 tests passed |
| Kiko wheels-off graph | 1,427 tests passed |
| Kiko wheel-on graph | 1,361 tests passed |
| Kiko complete attended union | 1,460 tests passed |
| exact Kiko Clippy graphs | minimal, agent runtime, production, wheels-off, wheel-on, and union all passed with `-D warnings` |
| STM32 builds | inert, wheels-off candidate, and attended wheel-on release builds passed; six unsafe feature unions were rejected |
| RP2350 eye firmware | provisioned compile-only release build passed |
| operator console | JavaScript syntax and view-model test passed |
| cold-boot/fault acceptance | passed |

After the final freshness hardening:

- all 28 focused live-motion-owner tests passed, including the regression that
  rejects an unready production manual start without retaining authority;
- the exact production Kiko test graph passed with local socket permission;
- the exact full attended-feature-union Clippy graph passed with
  `-D warnings`;
- the OAK freshness regression, JavaScript syntax, view-model test, formatting,
  and `git diff --check` passed;
- the complete offline cold-boot/fault acceptance script passed again.

At transport-evidence closure on exact pre-V3 revision `35adc90`:

- the exact production `nano-agent` graph passed 1,331 library tests, 6
  deployment-qualifier binary tests, 69 production binary tests, and 1
  compile-fail doctest;
- the complete production, wheels-off, and wheel-on attended feature union
  passed 1,392 library tests, 3 base-commission binary tests, 6 deployment
  qualifier tests, 71 Kiko binary tests, and 1 compile-fail doctest;
- strict all-target Clippy passed for that complete feature union with
  `-D warnings`;
- the operator-console JavaScript syntax and view-model test passed; and
- the complete offline cold-boot/fault acceptance script again returned
  `acceptance_result=pass`.

### Post-V3 host validation — 2026-07-27

The mandatory face/head/lifecycle tree was frozen for host validation at exact
source revision
`de6c8626e0e69fbcfadd2b1db7ec68635d3c0bb2`. The following are separate
results; they are not added into a synthetic aggregate:

- `OAK_SYS_CHECK_ONLY=1 cargo test --locked -p kiko-slam
  --no-default-features --features nano-agent --lib` passed 1,387/1,387;
- the same production graph with `--bin kiko-slam` passed 75/75;
- the complete
  `nano-agent,nano-base-commissioning,nano-wheels-off-qualification` union,
  filtered to `navigation::nano_base_commissioning_live::tests`, passed 5/5;
- `OAK_SYS_CHECK_ONLY=1 cargo test --locked -p oak-sys
  --features opencv-face-detector` passed 33/33;
- expression core passed 15/15, expression runtime 61/61, head protocol
  17/17, head runtime 72/72, the head commissioning binary 9/9, bundle
  renderer 16/16, and deployment gate 6/6;
- focused lifecycle reruns passed 45/45 accessory-worker tests and 24/24
  Nano-bootstrap tests; these are subsets of the production library result,
  not additional tests; and
- formatting, `git diff --check`, locked metadata, and the final independent
  lifecycle/scope/documentation audits passed.

Strict all-target Clippy with `-D warnings` passed separately for:

1. standalone `nano-face-perception`;
2. `nano-base-commissioning`;
3. production `nano-agent`; and
4. the complete attended feature union above.

Every host OAK run used `OAK_SYS_CHECK_ONLY=1`. This proves the Rust/C++ build
boundary, typed parsing, deterministic face/tracker logic, supervised
lifecycle, and launch/deployment contracts on the macOS host. It does not
prove native OpenCV cascade construction on the Nano, OAK frame delivery,
detector accuracy or latency, USB SuperSpeed, head motion, STM32 motion, SLAM
accuracy, MPC tracking, or a performance improvement.

The exact candidate and attended wheel-on STM32 feature sets also passed
strict target Clippy and release compilation for
`thumbv7em-none-eabihf`. With the source root remapped to `/kiko-source`, the
local release ELF SHA-256 values were:

| Firmware class | Feature identity | ELF SHA-256 |
| --- | --- | --- |
| wheels-off candidate | `firmware,flash-boot-journal,operator-supervised-four-pwm-candidate` | `5670fb523a1221b6d3d73856d26798f2087345b0c7be56ecff3c1b3cb57f0adb` |
| attended wheel-on commissioning | `firmware,flash-boot-journal,attended-wheel-on-commissioning` | `7404673b9cf65539f58afe6f718113a29567ab9db6b1b8726c04fa3573a8db24` |

These are local compile identities, not independently reproduced Nano builds,
flash images, installed identities, or physical qualification.

The first production test invocation ran inside a restricted filesystem
sandbox and its 23 real-socket tests failed only because local bind returned
`Operation not permitted`. The exact graph was rerun with ordinary
loopback/Unix-socket permission and passed completely. All OAK host checks used
`OAK_SYS_CHECK_ONLY=1`; this is source and host-runtime evidence, not native
camera evidence.

The offline acceptance script exercised strict launch and marker admission,
map/session identity replacement, bounded storage and exact quota boundaries,
terminal map publication order, private per-boot console capability handling,
controller-owner shutdown ordering, and session-owned terminal HTTP
completion. Its output explicitly excludes installation, PID-1 execution,
cold power boot, device presence, USB exclusivity, physical watchdog and
E-stop behavior, stopping distance, head torque, camera streaming, SLAM
accuracy, MPC tracking, and performance.

Portable pure/support crates compiled for Linux aarch64. The macOS host did not
claim a GNU-cross-sysroot Kiko link. A later native Nano check moved only the
separate clean compile-check checkout to exact revision
`e723fc722a66741b59ef1dfcdac86c99ba1abe97`, then ran:

```text
OAK_SYS_CHECK_ONLY=1 cargo test --locked -p kiko-slam \
  --features nano-agent --lib navigation::nano_bootstrap::tests
```

The fresh aarch64 Linux build completed in 5 minutes 29 seconds and all 19
focused bootstrap tests passed. This proves the compile-only OAK graph links
and the new startup logic executes on that Nano CPU/OS. It does not prove a
native DepthAI link, device ownership, camera capture, serial traffic, timing,
temperature, deployment, or actuation.

### Native face, release-link, and inert-transport refresh — 2026-07-27

The clean `/home/makerspace/kiko` checkout was advanced only by fast-forward
on `codex/nano-expression-integration-stage`. A first real native
`nano-agent` all-target check exposed that locked `cxx-build 1.0.194` silently
omitted face declarations guarded by the hyphenated Cargo feature
`opencv-face-detector`: Cargo exported
`CARGO_FEATURE_OPENCV_FACE_DETECTOR`, while the CXX cfg evaluator compared the
literal hyphenated spelling without normalizing hyphens to underscores. Host
`OAK_SYS_CHECK_ONLY=1` runs could not expose the missing generated C++ types.

Revision `98e75eed48860a45b590260acdab1fbbb346e1a4` split the detector into a
feature-selected CXX bridge generated from the already parsed build input. On
Linux aarch64, the exact native

```text
cargo check --locked -p kiko-slam --no-default-features \
  --features nano-agent --all-targets
```

then completed in 1 minute 45 seconds against the installed DepthAI source
and OpenCV 4.5.4 headers. The generated detector header contained the
configuration, source enum, detection, batch, and opaque detector types; its
generated source contained both constructor and detection call wrappers.

The follow-up native smoke test linked the test executable, which exposed a
second boundary that `cargo check` cannot test: `cc` had emitted the static
bridge after its dynamic dependencies, so GNU ld with `--as-needed` discarded
OpenCV before later bridge references appeared. Final revision
`9c9f3a9b92d7f610a2153129849e12863a2646c8` emits the static bridge first and
DepthAI, USB, and OpenCV dependencies afterward.

At that exact revision, the explicit ignored native test
`native_haar_detector_loads_cascades_and_detects_a_blank_frame` passed. It
read, bounded, and parsed these exact installed payloads into C++-owned
classifiers, then processed one blank 160 by 120 tightly packed BGR frame:

| Cascade | Bytes | SHA-256 |
| --- | ---: | --- |
| `haarcascade_frontalface_default.xml` | 930,127 | `0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0` |
| `haarcascade_profileface.xml` | 828,514 | `b39a4a3be45539db146a7fc1d3e761a292c196eb88421185e6a615b3055e612d` |

The native detector call itself returned successfully in 0.06 seconds with
zero detections on the blank image. That is a functional constructor,
linkage, and call-boundary smoke test. It is not a detector latency
measurement, an accuracy result, evidence about a real RGB frame, or a
performance claim. It opened no OAK device.

The exact release wheels-off executable was then built natively with the
lockfile and
`--features nano-wheels-off-qualification --bin kiko-slam`. The build
completed in 7 minutes 10 seconds. The resulting aarch64 PIE was 28,119,992
bytes with SHA-256
`0b58ca0f07392ff65af516c7d82fade87c770ab3e0845ea4f996da2d4ee8d2c4`.
Its direct non-system `DT_NEEDED` entries included
`libdepthai-core.so`, `libusb-1.0.so.0`,
`libopencv_objdetect.so.4.5d`, `libopencv_imgproc.so.4.5d`, and
`libopencv_core.so.4.5d`. A target `ldd` run with the audited DepthAI build
directory first in `LD_LIBRARY_PATH` reported no unresolved entry. This
qualifies the observed build-tree link, not an immutable `/opt/kiko`
installation or the larger transitive OS ABI closure.

The same refresh found the exact STM32 endpoint unowned and performed one
byte-read-only identity observation. It reported UID
`2c0018001750314242353320`, boot ID `12638770094519703627`, ABI `2`, build
`131074`, fingerprint `KIKO-NO-ACT-V1!!`, capability bits `319`, maximum PWM
zero, output disabled, and unverified physical-stop semantics. It grants no
motion authority.

Fresh schema-3 motor-inert qualification then passed at both baseline rates:

| Rate | Reports | Missing/duplicate/reordered/skipped/late | Maximum diagnostic RTT |
| ---: | ---: | --- | ---: |
| 20 Hz | 200/200 | all zero | 19.348572 ms |
| 50 Hz | 500/500 | all zero | 18.550369 ms |

Both runs admitted the exact identity through a nonce-bound freshness
challenge, retained final idle-safe heartbeat evidence, and reported maximum
host writer in-flight count one. They measure only this diagnostic traffic;
they do not prove motion-capable streaming, motor current, a physical stop,
or a performance improvement. The mode-`0700` Nano evidence root is
`/home/makerspace/kiko-hardware-evidence/20260727T034913IST-9c9f3a9-native-refresh`.
The identity, 20 Hz, and 50 Hz JSON SHA-256 values are respectively
`ec5a135a726ce53b852928f963ebc28948fa3a4ba2e8ed89b95ec60fe12c22ab`,
`600bb59010ea1a792656ada772f9707f01c85d7c4876ef394ac14ebff62a2be9`,
and
`8f4fd2eae085ece5ffdfa745b125f92fb6d64c7477e9ef9c9b5c65382c50a2ad`.

These results close the source-native-build, detector-boundary, release-link,
and refreshed motor-inert transport portions of Gate A. They do not close the
coordinated Fable handoff, OAK SuperSpeed/live-stream admission, head thermal
and hold evidence, candidate firmware/zero/sign/fault matrix, immutable
qualification bundle, live SLAM/occupancy/Rerun, or console checks.

## Live Nano evidence preserved

The July 24 read-only Nano audit found:

- the Fable guardian still running and owning the head/eye/OAK lifecycle;
- the Fable dirty worktree preserved on
  `codex/fable-preserved-20260724`;
- the separate `/home/makerspace/kiko-codex-native-check` checkout clean and
  detached at the exact pushed integration revision above;
- the STM32 serial endpoint present but emitting legacy ASCII `ODO,...`
  telemetry rather than KRP2;
- the OAK then opened by Fable in forced USB High-Speed mode, so its observed
  480 Mbit/s link was not a valid SuperSpeed failure diagnosis;
- no installed canonical Kiko service or immutable `/opt/kiko` deployment.

At `2026-07-24T07:15:10+05:30`, while the compile-only build was running, the
Fable child reported `bow overtemp 93`, began its park path, and then logged an
OAK `X_LINK_ERROR`. The guardian respawned it at `07:15:17`; the new admission
reported raw servo temperatures `32`, `31`, `35`, and `35`, re-established
natural hold, and resumed eye acts. The build and the fault are temporally
correlated only. This evidence does not establish whether the raw temperature
was physical heat, an electrical/telemetry fault, or another cause, and it
does not qualify the head thermally.

After the build had completed, the recovered child ran for about 170 seconds
and reported a second `bow overtemp 79` at `07:18:12`, followed by the same OAK
`X_LINK_ERROR`; the guardian recovered it again at `07:18:22`. This recurrence
means the first fault must not be explained away as build-resource pressure.
It still does not identify the physical or electrical cause. The head remains
unqualified and requires attended support, independent power control, and
read-only thermal/electrical diagnosis before any ownership handoff.

At `2026-07-24T21:28:07+05:30`, a later read-only audit counted 35
`bow overtemp` reports in `/tmp/follow-track.log` and 38 guardian respawns
since `19:40` in `/tmp/engine-guardian.log`. Recent raw fault values included
`84`, `150`, `98`, `150`, `142`, `68`, `93`, and `74`; each was followed by an
OAK `X_LINK_ERROR` and a guardian restart. Fresh admissions repeatedly
reported raw temperatures around `35` to `38`. The then-current child,
PID `8797`, had run for about 21 minutes and was reporting normal
`head=TRACKING eyes=SLEEPY person=False` state. That stable interval does not
clear the recurrent fault. The temperature register read checks servo ID,
length, and checksum, but this evidence still cannot distinguish physical
heating from an electrical, telemetry, or servo fault. No owner was disturbed
to collect it.

A still later read-only audit on `2026-07-26` superseded any inference that
the standalone Fable child remained healthy. Guardian PIDs `1062` and `1081`
still supervised child PID `14807`
(`python3 kiko_face_follow.py --duration-s 864000`), but the last child log
entry was from `2026-07-24T23:00:08+05:30`: admission reported bow raw
temperature `33`, the next sample reported `bow overtemp 82`, and the log then
stopped after `park_begin` and an OAK `X_LINK_ERROR`, without a completed park
or shutdown receipt. The child still held `/dev/ttyACM1` for the head adapter,
but held neither the eye endpoint nor a visible OAK descriptor. Its main
thread was waiting on a futex while an XLink thread remained present. That
evidence proves neither the exact blocking call nor the physical meaning of
the raw temperature value; it does prove that process liveness and the
guardian's `pgrep` check were not sufficient health evidence. No process or
device owner was stopped, restarted, or displaced during this audit.

A `2026-07-27` read-only refresh found the same guardian/child process family
active and the child still holding the head endpoint. It did not establish a
then-current OAK or eye process owner. OAK MXID `19443010F1B43A2E00`
enumerated at `480M` on the USB2 tree; the separate `10000M` USB3 tree had no
OAK beneath it. `/opt/kiko` and `kiko-nano-agent.service` were absent.
`/home/makerspace/kiko` was clean at
`482023e0fa69c381cb5d5946c445234a0ae88105` on
`codex/jetson-hardware-validation`. No process, device, firmware, service, or
file was changed by that refresh. These facts require a fresh
endpoint-by-endpoint ownership check and a canonical SuperSpeed attempt; they
do not requalify the stale Fable child, camera, accessories, STM32, or SLAM.

No process was killed, no live device owner was deliberately displaced, and no
firmware, installed service, or deployment file was changed. The only Nano
mutation was fetching/checking out the exact revision in the separate
compile-check tree and writing ignored Cargo build artifacts.

## Later attended motor-inert flash attempt

The preceding no-mutation statement describes the earlier read-only audit.
Later in the same attended session, the operator confirmed both wheels
removed, motor power independently disconnected and kept cut, the head
supported, the motor area clear, and an independent power cut reachable.

The first fresh evidence directory,
`/home/makerspace/kiko-hardware-evidence/20260724T154623IST-5526fc0-attended-inert`,
was abandoned after backup when the same revision built from a different
absolute checkout path produced a different firmware hash. No flash write
occurred in that run, and the directory was not reused.

The complete restart used
`/home/makerspace/kiko-hardware-evidence/20260724T155122IST-5526fc0-attended-inert-restart2`
at exact source revision
`5526fc0de2f5d56fe2dea94010b09ef06c2949ff`. Two 512 KiB backups compared
byte-for-byte at SHA-256
`4472c2a5c24ed408bae651a080a40b807ede4afb6a7cfb01f6047de1331fd9ae`;
two 16-byte option reads compared at
`d292558017cf9ca0a2e40e262a5c1daa4b305ccf084ce06128133d282f905115`;
and the original 128 KiB sector 7 was
`b5a41c3758763bbec72769fab4a2533bf2db0b6312d93d25a695f9e4b9e02260`.
The motor-inert main image and same-session readback compared byte-for-byte at
SHA-256
`6974f25ce983a056f78f02180de8c4d018b4509b84314edc1ddc3b5077c02d49`,
sector 7 remained byte-identical, and a separate exact-target OpenOCD
invocation issued `reset run`. Motor power remained physically disconnected.

The first serial-opening host operation was the read-only identity probe on
the exact ST-Link VCP after a no-owner `fuser` check. It emitted no stdout and
failed with
`Error: Decode(OversizedRecord { maximum: 73 })`. The zero-byte JSON file has
the empty SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
Because stdout alone was piped to `tee`, the actual pipeline status was not
persisted and is recorded as unknown. No 20/50 Hz qualifier, journal
generation or write, candidate flash, session, PWM, or motor command followed.
The directory remains failed evidence and must not be reused.

This failure proves only that at least 74 consecutive nonzero bytes reached
the decoder before a delimiter. It does not distinguish old TTY/USB backlog,
upstream or in-flight bytes, current controller output, or noise. The verified
main-flash bytes and preserved sector 7 therefore do not establish runtime
identity. Commits `5c0a51e` and `1c543c2` subsequently made every fresh KRP2
host owner clear only its host input queue once, exclude bytes through one
initial delimiter, and decode every later record strictly. They do not
reinterpret the failed run as a pass.

At `1c543c27185e5b41d54cc93ea40980406a573a7d`, two independent Nano builds
from different absolute source paths remapped repository locations to
`/kiko-source`. The ELFs compared byte-for-byte at SHA-256
`fe1f055c076d700fd65d0f02db2a55163f82b7789b3bb2bf8b0ee25ede130dcc`;
the two 393,216-byte padded motor-inert images compared at
`270e553f5c18a53393f0234f334d3ccc71be32ac7827240b54c939c6d6def38d`.
This is build reproducibility evidence only. A new attended run still requires
fresh backups, main write/readback, reset, exact identity, and separate passing
20/50 Hz measurements before journal provisioning.

While hardening that restart procedure, one preparatory no-write
`/usr/bin/st-info --probe` normal attach was run after the operator reconfirmed
the full motor-inert physical gate. It reported one ST-Link with serial
`066EFF313946303143221230` and an F446 target with 524,288 flash bytes and
131,072 SRAM bytes. It opened no UART and issued no application or actuator
command, but normal attach does not satisfy this procedure's
connect-under-reset rule and is not qualification evidence. The revised
procedure therefore removes that probe, enumerates the ST-Link through
USB/sysfs without target attach, and requires the next evidence run to begin
again with fresh duplicate connect-under-reset backups.

The next complete restart used the new mode-`0700` directory
`/home/makerspace/kiko-hardware-evidence/20260724T164932IST-1c543c2-attended-inert-restart3`
and exact source revision
`1c543c27185e5b41d54cc93ea40980406a573a7d`. USB/sysfs enumeration found
exactly one `0483:374b` ST-Link, tied serial
`066EFF313946303143221230` to the persistent VCP, and did not attach to the
target. Its JSON SHA-256 is
`5dac5ac52003ca5aaa34c043a259ac0349e0794105288fc80c18cb5af2bc7f87`.

Two fresh connect-under-reset 512 KiB reads were byte-identical at SHA-256
`768dfb8ce3beda16031740b2b4b6ccbb532ee3451f46179bf539c086f22b64cd`.
Their executable prefix was the previously installed motor-inert main image
`6974f25ce983a056f78f02180de8c4d018b4509b84314edc1ddc3b5077c02d49`;
sector 7 remained
`b5a41c3758763bbec72769fab4a2533bf2db0b6312d93d25a695f9e4b9e02260`.
The old full-bank hash differs from the earlier pre-flash backup because its
main prefix had been replaced; this is expected and does not imply a sector-7
change. Two 16-byte option reads again matched at
`d292558017cf9ca0a2e40e262a5c1daa4b305ccf084ce06128133d282f905115`.

The new build reproduced ELF SHA-256
`fe1f055c076d700fd65d0f02db2a55163f82b7789b3bb2bf8b0ee25ede130dcc`
and padded main SHA-256
`270e553f5c18a53393f0234f334d3ccc71be32ac7827240b54c939c6d6def38d`.
OpenOCD connected under reset to device ID `0x10006421`, reported 512 KiB
flash, programmed and verified only the 384 KiB main region, read it back at
the same `270e...` hash, and independently re-read unchanged sector 7 at
`b5a4...`. A separate invocation then issued the exact-target `reset run`.
Motor power remained physically disconnected.

The privileged no-owner check passed immediately before the first serial
open. The read-only schema-2 identity probe nevertheless again returned status
1 with empty stdout and
`Error: Decode(OversizedRecord { maximum: 73 })`. The output SHA-256 is the
empty-file hash
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
stderr is
`cd80431c799d27f2a1e5af5479a328ac5da5cd0092b44230f00b97d7a3742321`;
the captured status file is
`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`
and contains `1`. The fail-fast shell closed the SSH session. No 20/50 Hz
qualifier, journal operation, candidate flash, session, PWM, motor command, or
later serial access followed. A new SSH connection read only those preserved
files.

Under the flashed motor-inert image's specified idle behavior, the only
legitimate output records are a 70-byte delimited `ControllerHello` and
59-byte delimited `Heartbeat`; their nonzero COBS bodies are 69 and 58 bytes
respectively. The 74th consecutive post-boundary nonzero byte is therefore not
a valid idle KRP2 record or an off-by-one maximum. The current evidence still
cannot distinguish a lost delimiter, reset-fragment concatenation, wrong
producer, or line corruption. The next run must preserve strict failure while
capturing bounded raw-wire evidence through the earliest terminating delimiter
when one is observed, or otherwise record the exact bounded stop.

Commit `6cc59a1a3972c44df77dfd2cc02920ba40d896a2` makes that next observation
failure-discriminating without adding a serial transmit path. The read-only
probe now preserves the original typed decoder failure, captures a bounded
wire suffix plus delimiter/run accounting and a non-cryptographic full-trace
fingerprint, enforces exclusive deadline commits, and retains typed secondary
read and output failures. Its exact source file SHA-256 is
`6813e688e28de5d456c0c50ffbd11393f79364c69293ffaaa7472d917a60d05b`;
two independent exact-byte audits found no remaining actionable issue. All 36
`robot-server` bin tests passed both on macOS and natively on the Nano.

Two clean Nano checkouts at that revision independently rebuilt the
motor-inert firmware with their absolute roots remapped to `/kiko-source`.
Their ELFs were byte-identical at
`fe1f055c076d700fd65d0f02db2a55163f82b7789b3bb2bf8b0ee25ede130dcc`;
their 393,216-byte padded main images were byte-identical at
`270e553f5c18a53393f0234f334d3ccc71be32ac7827240b54c939c6d6def38d`.
The build roots are `/home/makerspace/kiko-build-repro/6cc59a1-path-{a,b}`.
That rehearsal opened no debugger, serial device, camera, or actuator.

The next attended restart used the fresh mode-`0700` evidence directory
`/home/makerspace/kiko-hardware-evidence/20260724T175852IST-6cc59a1-attended-inert-restart4`
and exact firmware source revision
`6cc59a1a3972c44df77dfd2cc02920ba40d896a2`. Under the operator-confirmed
safety gate, motor power was reported to remain independently disconnected.
Two fresh connect-under-reset 512 KiB backups were byte-identical at
`dfda9a32a6dede174ce55a29acfb59fc754277c421d23db886c8155d0f40dd55`;
the duplicate option reads remained
`d292558017cf9ca0a2e40e262a5c1daa4b305ccf084ce06128133d282f905115`.
Independent absolute-path-remapped builds reproduced ELF
`fe1f055c076d700fd65d0f02db2a55163f82b7789b3bb2bf8b0ee25ede130dcc`
and padded main
`270e553f5c18a53393f0234f334d3ccc71be32ac7827240b54c939c6d6def38d`.
The main write and same-session readback compared byte-for-byte at that
`270e...` hash, and sector 7 remained byte-identical at
`b5a41c3758763bbec72769fab4a2533bf2db0b6312d93d25a695f9e4b9e02260`.

After the exact-target reset and privileged no-owner check, the prebuilt
read-only identity probe passed the complete strict schema-2 check. Its JSON,
empty stderr, captured zero status, and checked-values hashes are respectively
`ec5a135a726ce53b852928f963ebc28948fa3a4ba2e8ed89b95ec60fe12c22ab`,
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`,
and
`1504d3c3b881b47724465909eefd7ab8fbea81bc17a7e26abfed01adc88292bd`.
It reported controller UID `2c0018001750314242353320` and boot ID
`12638770094519703627`.
This proves only the captured software identity under the probe's documented
boundary; it does not prove physical output behavior.

The subsequent interactive 20 Hz invocation accidentally omitted the `4e`
byte from the checked-in 32-character actuator fingerprint, producing
`4b494b4f2d4f2d4143542d56312121`. The typed CLI rejected that 30-character
value and returned status 2. Its empty JSON hash was
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
stderr hash was
`40062daae29450e27ae459bc6b3c118abf251888d7a34ab9ed44267cdf51ebb9`,
and status-file hash was
`53c234e5e8472b6ac51c1ae1cab3fe06fad053beb8ebfd8977b010655bfdd3c3`.
Source inspection confirms the first serial open is reachable only after
`Cli::parse()` returns successfully, so this invocation neither opened,
cleared, read, nor wrote the UART. The checked-in runbook had the correct
value; its setup now binds that value immutably, checks it against the
canonical literal, and supplies it to the identity-output checker and
qualifier CLI/output checker.

The fail-fast shell exited. No 50 Hz qualifier, later serial access, journal
operation, candidate flash, session, PWM, or motor command was issued in this
attended run. The directory remains failed evidence and must not be reused.
Qualification still requires a complete fresh evidence run.

The corrected restart used the fresh mode-`0700` evidence directory
`/home/makerspace/kiko-hardware-evidence/20260724T181500IST-6cc59a1-attended-inert-restart5`
at exact firmware source revision
`6cc59a1a3972c44df77dfd2cc02920ba40d896a2`. Motor power was
operator-reported, but not independently instrumented, to remain disconnected.
Two new connect-under-reset 512 KiB backups again matched at
`dfda9a32a6dede174ce55a29acfb59fc754277c421d23db886c8155d0f40dd55`,
and the duplicate option reads matched at
`d292558017cf9ca0a2e40e262a5c1daa4b305ccf084ce06128133d282f905115`.
The fresh build reproduced ELF
`fe1f055c076d700fd65d0f02db2a55163f82b7789b3bb2bf8b0ee25ede130dcc`
and padded main
`270e553f5c18a53393f0234f334d3ccc71be32ac7827240b54c939c6d6def38d`.
The main write and readback matched at that `270e...` hash, while sector 7
remained
`b5a41c3758763bbec72769fab4a2533bf2db0b6312d93d25a695f9e4b9e02260`.

After the exact-target reset and no-owner check, the read-only identity probe
again passed the complete schema-2 check. Its JSON, empty stderr, zero-status
file, and checked-values hashes were respectively
`ec5a135a726ce53b852928f963ebc28948fa3a4ba2e8ed89b95ec60fe12c22ab`,
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`,
and
`1504d3c3b881b47724465909eefd7ab8fbea81bc17a7e26abfed01adc88292bd`.
It reported controller UID `2c0018001750314242353320` and boot ID
`12638770094519703627`.

The subsequent corrected 20 Hz qualifier opened the UART but returned status
1 with empty JSON and the strict primary error
`Error: Decode(OversizedRecord { maximum: 73 })`. Its JSON, stderr, and
status-file hashes were respectively
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
`cd80431c799d27f2a1e5af5479a328ac5da5cd0092b44230f00b97d7a3742321`,
and
`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`;
the status file contains `1`. That qualifier error does not encode the
qualifier phase at failure or whether any diagnostic write had already
occurred, so both facts remain unknown. Fail-fast stopped the run: no 50 Hz
qualifier or later scripted phase followed. The directory is failed evidence
and must not be reused.

A deliberately qualifier-first diagnostic restart then used the fresh
mode-`0700` evidence directory
`/home/makerspace/kiko-hardware-evidence/20260724T183340IST-6cc59a1-qualifier-first-diagnostic-restart6`
at the same exact source revision. Motor power was again operator-reported,
not independently instrumented, to remain disconnected. Its fresh duplicate
full backups, option reads, ELF, padded main, same-session main readback, and
preserved sector-7 hashes were the same pinned values from restart 5:
`dfda9a32a6dede174ce55a29acfb59fc754277c421d23db886c8155d0f40dd55`,
`d292558017cf9ca0a2e40e262a5c1daa4b305ccf084ce06128133d282f905115`,
`fe1f055c076d700fd65d0f02db2a55163f82b7789b3bb2bf8b0ee25ede130dcc`,
`270e553f5c18a53393f0234f334d3ccc71be32ac7827240b54c939c6d6def38d`
for both the padded main and its readback, and
`b5a41c3758763bbec72769fab4a2533bf2db0b6312d93d25a695f9e4b9e02260`,
in that order.

After the exact-target reset and no-owner check, the corrected 20 Hz qualifier
was the first UART opener; no identity probe preceded it. It produced the same
strict oversized-record error and status `1`. Its exact empty JSON, stderr,
and status-file hashes were
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
`cd80431c799d27f2a1e5af5479a328ac5da5cd0092b44230f00b97d7a3742321`,
and
`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`.
This rules out the identity probe's close/reopen boundary as the sole cause of
the observed failure. It does not identify the producer or corruption
mechanism, and the qualifier still does not encode its failure phase or
whether it had already made a diagnostic write. Fail-fast stopped the run, so
no 50 Hz qualifier or later scripted phase followed. This directory is also
failed evidence and must not be reused.

The phase-tagged host diagnostic was then built natively from the clean,
detached Nano checkout at
`ba72b4404bd1bf7b0c123c61c128985b089460b0`. Its aarch64 identity and
qualifier binaries hashed respectively to
`80da2203560bf2942fae60731c06c0dea7e58c7995a03951c69205c8329f346f`
and
`a25e040b2045084028d0beebce4a71b9debed7116e594226f9ee5a5a56a84504`.
All 24 identity and 25 qualifier tests passed natively. The source checkout
remained clean.

This was deliberately a non-qualifying diagnostic under existing Fable load,
not another attended flash restart. It used the fresh evidence directory
`/home/makerspace/kiko-hardware-evidence/20260724T190859IST-ba72b44-fable-load-phase-diagnostic`.
No backup, flash, reset, quiet-load claim, physical instrumentation, control
session, or PWM command was part of this run. Motor power remained
operator-reported, not independently instrumented, as disconnected. The
existing guardian and `kiko_face_follow.py` child had the same PIDs in the
before and after process snapshots. Point-in-time `fuser` output, with the
visibility of the unprivileged collection account, reported that child owning
the head adapter at `/dev/ttyACM1` and reported no owner for the STM32 VCP at
`/dev/ttyACM0` immediately before the qualifier. This is not proof that no
other process held, or could subsequently acquire, a descriptor outside that
snapshot and visibility boundary.

Live USB topology showed that this separation is logical but not physical:
the 12 Mbit/s ST-Link VCP at `1-2.1`, 12 Mbit/s head adapter at `1-2.2`,
480 Mbit/s OAK at `1-2.3`, and 12 Mbit/s eyes at `1-2.4` all share USB-2 hub
`1-2`. Fable did not appear to own the STM32 endpoint in the captured
snapshots, but its OAK, head, USB, and CPU activity remains a load confound
for later coexistence and performance qualification. The captured evidence
does not show that Fable caused this transport failure.

The 20 Hz qualifier returned status `1`, empty JSON, and phase-tagged stderr.
Their hashes were respectively
`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`,
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
and
`a8372bbac7eb95b48533e3e5bedb22964da5448e0ccac181ed788d3975284047`.
The strict primary error remained `OversizedRecord { maximum: 73 }`, but the
new bounded evidence places it in `diagnostic_stream`, after the then-current
`ba72b44` exact read-only admission logic accepted the stale candidate. It
recorded 330 delivered bytes, failure after 328 decoder-processed bytes, an
11-byte failing read with 9 bytes processed through failure, and 2
already-delivered suffix bytes. It made no additional read, retry,
resynchronization, decoder reset, or hidden recovery.

The retained trace explains why the then-current read-only admission was not
a freshness proof. Its initial synchronization delimiter was at zero-based
offset `6`. The next records decoded, in order, as heartbeats with controller
uptimes `80,500 ms` and `80,750 ms`, a `ControllerHello`, and a heartbeat at
`81,000 ms`; their delimiters were at offsets `65`, `124`, `194`, and `253`.
The candidate diagnostic body that follows reports both request receipt and
response preparation at controller uptime `1,672,521 ms`. The gap from the
last complete heartbeat is therefore exactly `1,591,521 ms`, or 26 minutes
31.521 seconds. Those admitted heartbeat and hello bytes were old backlog
relative to the later candidate, even though they arrived freshly at the
host.

After the delimiter at offset `253`, offsets `254` through `256` are
`08 4b 52`; offsets `257` through `329` begin again with
`08 4b 52 50 32` and contain a valid, current-like diagnostic body. The bytes
alone do not select one of the four equivalent deletion windows described
below. Their chronology is consistent with treating offsets `254..256` as
the partial next stale-record prefix and offsets `257..329` as the
current-like candidate. No trailing zero delimiter for that candidate was
observed among the 330 delivered bytes, so it was not a complete record in
the captured host stream and the strict decoder correctly did not accept it.
The complete 76-byte nonzero tail was:

```text
084b52084b52503202413c022c02181917503142423533204b58ad1970ee65afca8d2e94baa066f801010104fc0409010101010101010101010101044985190449851901010101057b3be509
```

Because of the overlapping periodic prefix, deleting any one contiguous
three-byte window beginning at tail offset `0`, `1`, `2`, or `3` produces the
same unique 73-byte COBS candidate; the bytes do not identify which equivalent
window is the omitted partial prefix. The candidate hash is
`cda219604a77215699d4172d0362bd1d32b9ec27bf14a85f8084cd5460ae830f`.
Canonical COBS and KRP2 decoding yields one 72-byte raw frame with hash
`ebc540876d44c99f51a967b008a99c17fef2b0708364e2c2ab8f38d049278387`
and a valid CRC-32C. It decodes as a sequence-0
`TransportDiagnosticReport` with run ID `17899170492241055178`, host elapsed
token `591100`, result `EchoedMotorInert`, zero left/right PWM, disabled
output, clear faults, zero recorded RX/TX queue depth, and the admitted
controller UID and boot ID. However, the collection did not persist the
host-generated expected run ID independently before the write. Binding this
current-like body to this qualifier invocation is therefore an inference, not
proof that this invocation's probe was processed. The absent trailing
delimiter is an additional reason not to treat it as an accepted report.

The evidence is consistent with stale ST-Link/USB/TTY pipeline bytes remaining
upstream or in flight when the host cleared its input queue: the 257 bytes
before the current-like candidate are consistent with an initial stale tail,
three stale heartbeats, one stale hello, and a three-byte partial prefix of
another stale record. The exact layer that retained or discarded each byte
was not directly observed. What is established is a host freshness-boundary
defect: clearing the host input queue once, excluding only through the first
delimiter, and measuring freshness from host receive time allowed a
26-minute-old coherent identity sequence to satisfy admission. This trace is
not evidence of payload insertion or replay. The checked-in firmware's atomic
record queue and the qualifier's exact-once byte tracing remain relevant
source facts, but they do not turn a non-atomic multi-layer clear into a
freshness guarantee. There were no kernel-journal entries in the captured run
window; absence of a log does not establish that all buffering layers behaved
atomically.

The bridge probe reported `V2J33S25`. ST's
[current RN0093 release note](https://www.st.com/resource/en/release_note/rn0093-firmware-upgrade-for-stlink-stlinkv2-stlinkv21-and-stlinkv3-boards-stmicroelectronics.pdf)
lists `V2J48M35` for ST-LINK/V2-1 and records intervening Virtual COM Port
buffer and behavior changes at `V2J35M26`. The installed bridge is behind
that target, but an update is not a proven fix for the host
freshness-boundary defect demonstrated by that revision, and no ST-Link
firmware update was attempted. The exact offline-analysis artifact hash is
`1d401a71b552a19f750428ff756aa0e9bf32bbabd394aba5947740334a363331`.

Commits `6b02fe7`, `b0f0022`, `5f19954`, and `35adc90` implement and harden
the host-side correction. After the one-time host input clear, the qualifier
accounts for and raw-discards every byte delivered during a fixed 1,000 ms
quarantine, discards through one subsequent zero delimiter, and starts strict
decoding at that known record boundary. The exact motor-inert Hello and
idle-safe Heartbeat observed next are candidate-selection evidence only; they
do not establish freshness or permit a control session or PWM.

Before any measured diagnostic probe, the qualifier generates a fresh
entropy-derived run ID and writes at most three motor-inert challenges,
250 ms apart, using reserved descending sequences and per-attempt host-elapsed
tokens. Only the latest successfully written exact challenge tuple may match;
nonmatching reports and replies to superseded attempts are discarded and
counted. After a match, admission requires a subsequently decoded exact Hello
and an idle-safe Heartbeat whose controller uptime strictly follows the
matched report's response-preparation time. The diagnostic service delta and
report-to-Heartbeat delta are each capped by a conservative host-elapsed upper
bound. This is a run-bound round trip and controller-clock liveness witness,
not a multi-Heartbeat cadence measurement and not proof that every upstream
buffer is empty.

Strict post-boundary decode failures remain terminal. Successfully written
challenge tuples are included in success evidence and in typed strict-decode
failure evidence, but the process does not durably journal a tuple before its
serial write.

## Successful motor-inert transport qualification

The correction was built natively on the Nano from exact source revision
`35adc901e50d0ccb893c66582238bea438e86f97`. The release
`v2_transport_qualify` binary had SHA-256
`eef7f4feb7ae2ec67e4a6ad067b61b12fefef92546043e03dcc45332dd3485c5`.
All 40 focused qualifier tests passed on Linux aarch64 before the live runs.

Before the successful runs, the controller reported a latched
`SERIAL_INTEGRITY` fault. An exact-target OpenOCD invocation selected ST-Link
serial `066EFF313946303143221230`, connected under reset, issued `reset run`,
and exited zero. It did not write flash. The reset evidence is retained at
`/home/makerspace/kiko-hardware-evidence/20260724T210706IST-5f19954-reset-schema3-fable-load`.
The reset stderr and status SHA-256 values are respectively
`27e7b9beed726b30648644a34cc18ee6063c461e590e940173bfe54bcf0a4785`
and
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
The following schema-3 freshness admission, rather than a passive identity
read, is the evidence that the controller was live and fault-clear.

Two fresh, separate qualifier processes then passed:

| Rate | Evidence directory | Run ID | Reports | Maximum diagnostic RTT | Maximum host Heartbeat gap |
| --- | --- | ---: | ---: | ---: | ---: |
| 20 Hz | `/home/makerspace/kiko-hardware-evidence/20260724T211051IST-35adc90-schema3-20hz-fable-load` | `6951038622635299946` | 200/200 | 17.939749 ms | 255.004196 ms |
| 50 Hz | `/home/makerspace/kiko-hardware-evidence/20260724T211135IST-35adc90-schema3-50hz-fable-load` | `6814291295675353613` | 500/500 | 17.937019 ms | 258.822044 ms |

Both runs used one challenge attempt after discarding 563 quarantine bytes and
59 boundary-alignment bytes. Both had zero missing, duplicate, reordered,
scheduler-skipped, in-flight-skipped, writer-queue-skipped, or period-late
probes. Maximum observed in-flight work was one. Controller receive queue
depth was always zero; maximum pre-response transmit queue depth was 45 bytes.
Maximum decoded controller service time was 1 ms. The final idle-safe
Heartbeat was observed after the last write. The checked-in schema-3 verifier
returned `qualified` for both JSON files.

The 20 Hz JSON, empty stderr, and zero-status SHA-256 values are
`93ae596a2828263961b87f877adc7ae40f3b05a66dce57b89aedf6b0b2124d4e`,
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
and
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`.
The corresponding 50 Hz values are
`ad50a66153b51e19808abc78e43828e512ff940ed60d05b143f93a0584843134`,
the same empty-file hash, and the same zero-status hash.

Fable remained running throughout. Its before/after process snapshots were
byte-identical at SHA-256
`2ec644b1d0c25a85f635a549873a34b05d772596a13045ce31725d55e0bd7a3a`;
the serial-owner snapshots were byte-identical at
`63c07d03922575aea5d104ede88026c9f0ddc9b83fdc7f37b32524cde4356e82`.
Fable owned the head and eye endpoints while the qualifier exclusively opened
the otherwise unowned STM32 VCP. This proves coexistence for this
motor-inert diagnostic workload, not shared OAK ownership or production
runtime coexistence.

Motor power was operator-reported, not independently instrumented, to remain
disconnected. The successful runs created no control session and sent no PWM.
They qualify the invocation-bound diagnostic transport at the measured rates;
they do not prove motor output, candidate/production command handling, a
performance improvement, or any physical stop behavior.

## Exact remaining gates

The wheel-attachment gate and production-motion gate are deliberately
separate. Requiring an approved production plant before attaching wheels would
be circular because the encoderless plant must be measured with the wheels on
the floor.

Before the attended calibration-only wheel-attachment sentence is allowed:

1. ensure an independent, immediately reachable physical cut removes motor
   power outside Jetson and STM32 control, and keep it open while the wheels
   are attached;
2. diagnose the recurrent historical Fable `bow overtemp` reports, then
   coordinate an endpoint-by-endpoint Fable handoff with the head supported;
   never use broad process killing or start a second OAK/head/eye owner;
3. provision and read back the boot journal, flash and admit the exact
   operator-supervised wheels-off candidate, prove exact applied zero, and
   complete its attended command/fault matrix. The separate motor-inert 20/50
   Hz diagnostic transport gate is already complete;
4. capture and bind the exact live OAK calibration and the shadow-only inputs
   required for the immutable wheels-off/commissioning bundle. A historically
   prepared boot-journal image is not evidence that it was provisioned or read
   back;
5. prove canonical SuperSpeed OAK, natural head hold, RGB eye behavior, live
   SLAM/occupancy/Rerun, single console ownership, and coordinated cleanup on
   the Nano; and
6. with wheels still absent, prove the bounded shaft-sign commands and
   re-confirm disarmed applied zero immediately before the handoff.

After those items, wheel attachment authorizes only the separately invoked,
attended `kiko-nano-base-commission` schedule. It may measure visual forward
velocity, calibrated IMU yaw, unequal left/right plants, wheelbase, and
low-speed stopping, then emit a non-activatable proposal.

Production motion remains closed until a distinct promotion review:

1. reviews motor-driver wiring and real default-off enable,
   driver-fault/E-stop-feedback pins, voltage levels, active polarities,
   reset/brownout behavior, and physical stop semantics;
2. admits a uniquely identified production four-PWM firmware profile that
   samples real fault-clear state;
3. consumes accepted commissioning evidence, repeated-run consistency,
   wiring/stop qualification, approver identity, exact plant bytes, and the
   flashed production STM32 identity;
4. emits and verifies the immutable active plant/controller bundle and
   qualified-only service enablement; and
5. qualifies production deadman, fault stops, obstacle stops, and MPC only
   inside the measured support envelope.

## Residual software limits

- The live dataset manifest is bounded by the 65,536-file ceiling but is still
  constructed monolithically; future larger limits require chunked
  publication.
- Descriptor-relative map/session publication prevents path substitution and
  detects root replacement. A malicious same-UID process that can mutate an
  already-open generic dataset payload remains outside that narrower
  checkpoint-integrity claim.
- The local occupancy grid is geometric, not learned. Moving people are
  reflected by fresh depth in the expiring local costmap; there is no semantic
  person prediction.
- No performance improvement or physical behavior is claimed without a
  reproducible measurement.
