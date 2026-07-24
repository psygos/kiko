# Nano integration acceptance report — 2026-07-24

## Decision

The host-side integration is a tested, fail-closed release candidate. It
provides one strict startup graph, one STM32 owner, one OAK owner, supervised
head/eye behavior, live SLAM and geometric occupancy, a unified browser/agent
control plane, bounded map and dataset persistence, attended wheels-off
qualification, and attended wheel-on identification.

The wheel-attach handoff remains **closed**. This is not a test failure hidden
behind a warning. Under the currently evidenced PA0/PA1 plus PB4/PB5 four-PWM
wiring, the repository has no known default-off external driver-enable line
and no driver-fault/E-stop feedback input. The production controller contract
therefore cannot truthfully advertise
`production_external_interlocks`. The checked-in STM32 images intentionally
remain motor-inert, wheels-off candidate, or attended wheel-on commissioning
images.

The repository must not claim production MPC motion, a physical emergency
stop, or permission to attach the wheels until the physical interlock wiring
and independent power cut are reviewed and implemented. Host code does not
replace those facts with a boolean, prompt, or manifest claim.

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
  fields. The intended 50 Hz rate is a declared software schedule, not a
  measured Nano/STM32 performance claim.
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

The broad frozen-tree matrix completed before the final sensor-freshness
hardening:

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

## Live Nano evidence preserved

The read-only Nano audit found:

- the Fable guardian still running and owning the head/eye/OAK lifecycle;
- the Fable dirty worktree preserved on
  `codex/fable-preserved-20260724`;
- the separate `/home/makerspace/kiko-codex-native-check` checkout clean and
  detached at the exact pushed integration revision above;
- the STM32 serial endpoint present but emitting legacy ASCII `ODO,...`
  telemetry rather than KRP2;
- the OAK currently opened by Fable in forced USB High-Speed mode, so its
  observed 480 Mbit/s link is not a valid SuperSpeed failure diagnosis;
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

## Exact remaining gate

Before the wheel-attach sentence is allowed:

1. review the motor-driver wiring and select real default-off enable,
   driver-fault/E-stop-feedback pins, voltage levels, and active polarities;
2. ensure an independent physical emergency cut removes motor power outside
   Jetson and STM32 control;
3. add and review the uniquely identified production four-PWM firmware
   profile, sampling real fault-clear state rather than deriving it from a
   capability class;
4. add the explicit promotion boundary that consumes completed commissioning
   evidence, repeated-run consistency, wiring/stop qualification, approver
   identity, and the flashed production STM32 identity before emitting the
   active plant/controller bundle;
5. coordinate the exact Fable handoff with the head supported; never use broad
   process killing;
6. flash and admit KRP2, prove exact zero, run motor-inert 20/50 Hz transport
   qualification, then complete the attended wheels-off fault matrix;
7. prove canonical SuperSpeed OAK, RGB expression, natural head hold, live
   SLAM/occupancy/Rerun, console ownership, and cleanup on the Nano.

Only then may wheel-on commissioning measure visual forward velocity and
calibrated IMU yaw, fit the encoderless left/right plants, qualify stopping
behavior, bind the approved plant, and tune MPC inside measured support.

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
