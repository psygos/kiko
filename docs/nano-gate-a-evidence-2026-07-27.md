# Nano Gate A evidence refresh — 2026-07-27/28

This file records current, bounded evidence for the wheel-attachment gate. It
does not declare Gate A passed and does not authorize wheel attachment,
powered motion, deployment, or production use.

## OAK-D S2 10 Gbit/s transport diagnosis

At approximately `2026-07-28T16:19+05:30`, the current legacy Kiko child had
exited naturally after its existing pre-engage raw head-temperature gate. Its
source was inspected and found to open the exact OAK MXID with
`dai.UsbSpeed.HIGH`; that argument deliberately caps DepthAI at USB 2. The
observed 480 Mbit/s legacy link was therefore requested behavior, not evidence
that the Orin Nano port, installed cable, or OAK-D S2 lacked USB 3.

The exact guardian PID `1068` was temporarily paused while no child existed.
No serial endpoint, head, eye, STM32, or motor was opened or commanded. A
camera-only probe then opened MXID `19443010F1B43A2E00` with
`dai.UsbSpeed.SUPER_PLUS`. DepthAI read back `SUPER_PLUS`; while that same
owner retained the open device, the kernel placed the OAK below Bus 02's
four-port USB-3 hub and reported `10000M`:

```text
requested=SUPER_PLUS observed=SUPER_PLUS
Bus 02 ... 4-Port USB 3.0 Hub, 10000M
    Port 3 ... Vendor Specific Class, usbfs, 10000M
```

The probe closed normally with status zero and the exact guardian was resumed.
An earlier probe attempted without exclusive ownership failed after the
legacy process raced it; that failure is retained as ownership-conflict
evidence and is not counted as a transport result.

This establishes that the current host/port/cable/OAK path can negotiate the
10 Gbit/s DepthAI mode. It does not measure sustained throughput, frame loss,
latency, thermal behavior, or full-graph stability. The `SUPER_PLUS/SUPER`
policy selected immediately after this probe is superseded by the full-graph
evidence below; the retained probe remains negotiation evidence only.

## OAK full-graph USB-3 qualification and production cap

Camera-only qualification used exact MXID `19443010F1B43A2E00` with RGB,
rectified left, rectified right, and depth at `640x400@15`, IMU at `200 Hz`,
and nonblocking queues of four. The legacy guardian was paused only after its
only live child was an idle `sleep`; a trap restored it after every attempt.
No serial endpoint, head, eye, STM32, or motor was opened or commanded.

The first quiet-window full-graph `SUPER_PLUS` connection reached a 10 Gbit/s
same-owner readback and delivered two exact RGB frames plus 12 IMU samples. It
then exposed a host-contract defect: DepthAI StereoDepth documents rectified
mono as `RAW8`, while Kiko required `GRAY8`. Commit `cf0659e` fixed the bridge
to require `RAW8` only for rectified mono and retain `GRAY8` for direct-camera
mono. The retained pre-fix evidence is
`/home/makerspace/kiko-native-evidence/692508f-oak-20260728T213822+0530`.

After that fix, one standalone attempt and three bounded retry-window
attempts, all from `cf0659e`, failed with an explicit `SUPER_PLUS/SUPER`
policy during startup with DepthAI's
`Device already closed or disconnected: Input/output error`. None of the
three controlled retry-window attempts observed the OAK device at `10000M`.
This proves repeated startup failure in the tested DepthAI 3.4 state; it does
not identify USB transport, device firmware, power, or another cause.

The retained failed-start evidence directories are:

- `/home/makerspace/kiko-native-evidence/cf0659e-oak-20260728T214543+0530`;
- `/home/makerspace/kiko-native-evidence/cf0659e-oak-a1-20260728T214924+0530`;
- `/home/makerspace/kiko-native-evidence/cf0659e-oak-a2-20260728T214942+0530`;
- `/home/makerspace/kiko-native-evidence/cf0659e-oak-a3-20260728T215001+0530`.

The subsequently built `94d4f15` binary then ran the unchanged graph twice
with an explicit `SUPER/SUPER` policy. Both runs connected on their first
controlled attempt, read back `SUPER`, and observed the owned OAK at kernel
`5000M` in 11 samples.

| evidence directory | elapsed (s) | each image stream | native sequences | IMU delivered / required | measured IMU delivery |
|---|---:|---:|---|---:|---:|
| `94d4f15-oak-SUPER-a1-20260728T215339+0530` | 10.007847785 | 150 / 150 | 0–149; 0 gaps, duplicates, regressions | 1959 / 1602 | 195.746382 Hz |
| `94d4f15-oak-SUPER-a1-20260728T215459+0530` | 10.000549404 | 150 / 150 | 0–149; 0 gaps, duplicates, regressions | 1967 / 1601 | 196.689194 Hz |

The table basenames resolve below
`/home/makerspace/kiko-native-evidence/`; both directories retain the complete
report, raw stdout, stderr, USB topology timeline, source commit, executable
hash, promotion decision, and evidence manifest.

Each run delivered exactly 115,200,000 RGB bytes, 38,400,000 bytes from each
rectified mono stream, and 76,800,000 depth bytes. The combined measured host
payload was therefore about 26.86 and 26.88 MB/s respectively. These are
host-delivery measurements for this graph, not USB line-capacity, latency, or
thermal claims.

Freshly rendered production now requests and requires `SUPER`: explicit 5
Gbit/s USB 3, with no USB2 or automatic `SUPER_PLUS` fallback. Retained
`SUPER_PLUS/SUPER` launch documents remain parseable and retain that exact
request; fresh rendering neither selects nor relabels them. The camera
qualifier is the explicit path for new `SUPER_PLUS` diagnosis.

## Current 9f1061d native qualification build

The clean Nano checkout at `/home/makerspace/kiko` was fast-forwarded to the
exact pushed revision:

```text
9f1061dcb5c72d7abbdaea3e1983d3cdfe2265ea
```

That revision contains the fresh full-telemetry head gate, authoritative
offline deployment-graph binding, and challenged motor-power transition
workflow. The retained native inputs from the prior `bd6987a` evidence were
reused without modification. A first timing wrapper referenced unavailable
`/usr/bin/time` and exited with status 127 before Cargo started. The successful
command was:

```text
CARGO_BUILD_JOBS=2 nice -n 15 cargo build --locked --release \
  -p kiko-slam --no-default-features \
  --features nano-wheels-off-qualification --bin kiko-slam
```

The Linux aarch64 release build completed in 211.14 seconds. This is command
completion context, not a benchmark or performance claim. The resulting
executable identity is:

```text
path: /home/makerspace/kiko/target/release/kiko-slam
size_bytes: 29155440
sha256: b20dcab182823ce6ec9459118ea202d2206397c88b940bcd3cfcfd956bcb6a4e
elf_machine: AArch64
elf_type: PIE
build_id_sha1: 89f5a273da03cec2299e5dc56342950c2e82888a
```

The complete loader closure resolved with `not_found_count=0`. The
device-free `nano-wheels-off-qualification --help` boundary loaded
successfully. The owner-private evidence directory has mode `0700`; its 22
retained files have mode `0600`, and its SHA-256 manifest verifies:

```text
/home/makerspace/kiko-native-evidence/9f1061d-20260728T033436+0530
```

The evidence retains the source revision and clean status, exact build inputs
and command, completion timing, toolchains, executable identity, ELF metadata,
loader closure, CLI help, and read-only process/service/USB/serial snapshots.
Both canonical Kiko services remained inactive. The Kiko-owned
`engine-guardian.sh` process remained present and no Fable process was
present. No process was stopped, signalled, paused, reconfigured, or replaced,
and no camera, serial endpoint, STM32, head, eye, or actuator was opened.

The retained post-build USB tree showed the `10000M` root and hub but no OAK
child on either the USB3 or USB2 tree. All three persistent serial-by-id
endpoints remained present. This snapshot therefore does not establish a
connected OAK, and it does not infer a camera, cable, or hardware cause from
the absence.

## Current bd6987a native qualification build

The clean Nano checkout at `/home/makerspace/kiko` was fast-forwarded to the
exact pushed revision:

```text
bd6987a9def8d17b15b91683210739ce98387894
```

The retained 5d1b1bc native build inputs were reused without modification. A
first non-login SSH launch omitted Cargo from `PATH` and exited with status 127
before compilation began. The successful launch added the retained
`/home/makerspace/.cargo/bin` path explicitly and ran the same bounded build:

```text
CARGO_BUILD_JOBS=2 nice -n 15 cargo build --locked --release \
  -p kiko-slam --no-default-features \
  --features nano-wheels-off-qualification --bin kiko-slam
```

The Linux aarch64 release build completed in 5m09s. This duration is completion
context, not a benchmark or performance claim. The resulting executable is:

```text
path: /home/makerspace/kiko/target/release/kiko-slam
size_bytes: 28994400
sha256: b6269de500bd46d2fb312d0fd3063558627b296df2ade2cdd8665de4ed7b36be
elf_machine: AArch64
elf_type: PIE
build_id_sha1: ac3e82ec6e3639927d0dc5e752657356793fb04f
```

`readelf -d` retained the direct DepthAI, OpenCV, C++, and system dependency
set. `ldd` under the retained native library path resolved its complete
closure with `not_found_count=0`. The artifact differs from the prior native
build, but no source of the byte-level difference or reproducible-build claim
is inferred from that observation.

At `2026-07-28T02:37:11+05:30`, the owner-private evidence directory and its
17-entry manifest verified:

```text
/home/makerspace/kiko-native-evidence/bd6987a-20260728T023552+0530
```

It retains the exact source and clean status, build inputs and command,
toolchain identity, executable identity, ELF metadata, loader closure, and a
read-only process/service snapshot. Both canonical services remained inactive.
The standalone Kiko `engine-guardian.sh` and `kiko_face_follow.py` process
family remained running; no Fable process was present. No process was stopped,
signalled, paused, reconfigured, or replaced, and no camera, serial endpoint,
STM32, or actuator was opened.

## Superseding native tracker-startup refresh

The clean Nano checkout at `/home/makerspace/kiko` was fast-forwarded to the
exact code-bearing revision:

```text
5d1b1bc939dc34e0c80b861baa21afd81073f2df
```

This revision removes the canonical tracker's unbound EigenPlaces startup
dependency. Production and wheels-off startup now use deterministic aggregate
SuperPoint descriptors for loop closure and relocalization plus an explicit
worker/culling policy. Offline compatibility mode retains the learned
descriptor boundary. This software result does not prove representative-room
place-recognition quality.

The first noninteractive native build attempt omitted the retained DepthAI
include/library environment and failed closed in `oak-sys` before a native
bridge or Kiko executable was produced. The retained build-input file from the
prior evidence directory was then rechecked against its SHA-256 manifest and
used without modification. The exact successful command was:

```text
CARGO_BUILD_JOBS=2 nice -n 15 cargo build --locked --release \
  -p kiko-slam --no-default-features \
  --features nano-wheels-off-qualification --bin kiko-slam
```

The Linux aarch64 release build completed in 4m30s. That duration records
completion context only; it is not a benchmark or performance claim. The
resulting executable identity is:

```text
path: /home/makerspace/kiko/target/release/kiko-slam
size_bytes: 28994536
sha256: ded41b4c2f2a024efde0de5aef744ee6a2d518201156f3e70b12324137036caa
elf_machine: AArch64
elf_type: PIE
build_id_sha1: 12505297ee34baa04e9d88acda682b05f83a818a
```

`readelf -d` recorded the direct DepthAI, OpenCV, C++, and system dependencies.
`ldd` under the retained native library path resolved the complete dependency
closure with `not_found_count=0`. The device-free
`nano-wheels-off-qualification --help` boundary loaded successfully, and an
invalid fault declaration exited at parsing with status 2 while naming the
exact four supported one-shot fault declarations.

The new owner-private directory is mode `0700`, every retained file is mode
`0600`, and its 35-entry SHA-256 manifest verifies:

```text
/home/makerspace/kiko-native-evidence/5d1b1bc-20260728T014635+0530
```

It retains source/clean-status evidence, the verified build inputs and command,
toolchain identity, executable identity, ELF metadata, the complete loader
closure, CLI parsing evidence, and the fresh read-only owner/USB snapshot. No
camera, serial endpoint, service, actuator, STM32, or running Kiko process was
opened, stopped, signalled, reconfigured, or replaced.

## Fresh post-build owner and USB snapshot

At `2026-07-28T01:47:30+05:30`, read-only inspection found:

- all three persistent serial-by-id endpoints present;
- the STM32 endpoint `/dev/ttyACM0` free;
- PID 65144, the `kiko_face_follow.py` child of the existing
  `/home/makerspace/kiko-follow/engine-guardian.sh`, owning `/dev/ttyACM1`,
  `/dev/ttyACM2`, and the OAK USB node;
- both the reboot entry and minute-level guardian restoration entry still in
  the user crontab;
- the OAK `03e7:f63b` USBFS device under the `480M` USB2 tree while the separate
  `10000M` USB3 root/hub had no OAK below it;
- both canonical Kiko services inactive, with
  `kiko-nano-agent.service` not installed; and
- the Nano repository clean on the expected tracking branch after the build.

Fable is not a current subsystem, runtime owner, dependency, or operator. The
observed process is recorded only as a legacy Kiko runtime component and exact
endpoint owner. Its ownership still conflicts with starting the single
canonical owner, irrespective of its provenance. No process or respawn entry
was changed during this snapshot.

## Current Kiko-owner and head-fault snapshot

At `2026-07-28T02:43+05:30`, a later read-only Nano inspection reconfirmed:

- boot ID `305f4c72-249f-4225-b34e-3decc740764f`;
- the clean Nano checkout at `af4a24a`;
- `/home/makerspace/kiko-follow/engine-guardian.sh` as automatic
  reacquisition authority, repeatedly launching a short-lived
  `kiko_face_follow.py` child that opens OAK/head/eye;
- both the reboot crontab entry and the minute-level guardian restoration
  entry;
- no active canonical Kiko service;
- STM32, head, and eye at the retained persistent serial-by-id identities; and
- the OAK USB device below sysfs node `1-2.3` at `480M`, with sysfs product
  `2485` and sysfs serial `03e72485`, while the separate `10000M` root/hub had
  no OAK child.

The separately retained DepthAI/deployment identity is MXID
`19443010F1B43A2E00`. This snapshot did not obtain that MXID from sysfs and
does not equate it with the distinct USB serial string.

This is a Kiko-owned legacy runtime. Fable is not running it and is not a
current owner, dependency, service, or subsystem. The process name and
provenance do not relax the single-owner rule: the canonical Kiko runtime
cannot overlap this Kiko guardian on the OAK, head, or eye endpoints.

The guardian log then showed a repeated unhealthy restart sequence. Exact
reported raw faults included:

```text
bow overtemp 150
yaw overtemp 68
curl too hot pre-engage: 56
```

The guardian continued respawning the child at approximately eight-second
intervals through the observed `02:45:02..03:02:18` window. A fresh
`03:02:23` tail showed repeated `curl too hot pre-engage: 57` failures after
admitting Bow at raw temperature `54..55` and raw voltage `119..120`. Each
restart first reported `camera_ready usb=HIGH`, so the current evidence says
the camera opens successfully under this legacy Kiko runtime; it does not
support a bad-camera or bad-cable diagnosis. `HIGH` is consistent with the
observed USB2 link and still fails the canonical SuperSpeed gate. No inference
is made about calibrated degrees, electrical health, the physical cause, or
whether each reading came from a stable servo response. These are raw
legacy-runtime reports and a reason to keep Gate A closed.

The inspected files had SHA-256 identities
`1255a9563b1e03ef917b74f220698a1ee80804c3c474f30f1d0e3f3d703b4336`
(`engine-guardian.sh`) and
`ef0c9fb48743bd51ec8af317084273682553ac6b30bed384c74731a0eb3daf4e`
(`kiko_face_follow.py`). The legacy source admits pre-engage raw temperature
only when `<=55`, admits raw voltage in inclusive `90..=135`, and rejects
energized raw temperature at or above the effective `65` override in
`config.json`. Those deployed values justify a conservative canonical
software admission policy; they do not independently qualify the servo
register scaling, safe operating envelope, thermal response, or fault action.

No process was stopped, signalled, paused, reconfigured, or replaced. No
camera or serial endpoint was opened. The head must remain mechanically
supported, and final handoff requires an explicit, separately authorized
retirement of the exact Kiko guardian plus both respawn authorities followed
by a fresh conflict-free owner audit.

## V4 software-leaf inventory

At `2026-07-28T01:56:58+05:30`, a second read-only inspection resolved the
source leaves that do not depend on a new physical calibration or candidate
STM32 flash. Every listed source was a regular file, not a symbolic link, and
`namei` found no symbolic-link component in its absolute path:

| V4 role | Absolute source path | Bytes | SHA-256 |
|---|---|---:|---|
| qualification executable | `/home/makerspace/kiko/target/release/kiko-slam` | 28,994,536 | `ded41b4c2f2a024efde0de5aef744ee6a2d518201156f3e70b12324137036caa` |
| SuperPoint | `/home/makerspace/kiko/crates/kiko-slam/models/sp.onnx` | 5,226,093 | `aaefb94ad6dd3624fe4300b39f0f1a77e8739ed6d5430162729fd6a72c265431` |
| LightGlue | `/home/makerspace/kiko/crates/kiko-slam/models/lg.onnx` | 46,463,559 | `7fbb5814811dbc6d170de1c86bc0352a14691efa32cae33d952b6039258f74ef` |
| frontal-face cascade | `/home/makerspace/.local/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml` | 930,127 | `0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0` |
| profile-face cascade | `/home/makerspace/.local/lib/python3.10/site-packages/cv2/data/haarcascade_profileface.xml` | 828,514 | `b39a4a3be45539db146a7fc1d3e761a292c196eb88421185e6a615b3055e612d` |
| DepthAI core | `/home/makerspace/work/depthai-core-v34/build/libdepthai-core.so` | 42,632,152 | `0744500ab4f665af0641fd10881988146b73241212ac9523a86294e5737edae8` |
| dynamic calibration | `/home/makerspace/work/depthai-core-v34/build/_deps/dynamic_calibration-src/lib/libdynamic_calibration.so` | 36,820,008 | `30730ae6d367dcd927be7081f6a21d3bc4af65d857421ea3d3776d4ac00c7c53` |
| pinned DepthAI libusb | `/home/makerspace/work/depthai-core-v34/build/vcpkg_installed/arm64-linux/lib/libusb-1.0.so` | 202,888 | `74eac03235e61b326ecb6532bd1d840f7b8fbaf55cfaa32b7e3079fc1208ede0` |
| ONNX Runtime | `/home/makerspace/work/onnxruntime/build-jetson/Release/libonnxruntime.so.1.24.2` | 25,969,728 | `5246cdc32cf54afe0a108b9326f232ed1ed2bfcb9b4431738e2ad35eb20329aa` |
| OpenCV core | `/usr/lib/aarch64-linux-gnu/libopencv_core.so.4.5.4d` | 2,607,080 | `3abc549967c52f594b2b597db44b0013c55edb2198e11f9110d564277eb00beb` |
| OpenCV image processing | `/usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.5.4d` | 2,906,064 | `15b2448af215493a79f4638cad8eefcb9b43f15926724caffbdbd06a9c018261` |
| OpenCV object detection | `/usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.5.4d` | 366,632 | `94d3ddfb2111e72658d4bd005d22fd0ce402f8ae45ff8a79e9f7bdbd9b194b0b` |

The qualification-executable row in this timestamped inventory is superseded
by the bd6987a native artifact above. The other rows remain historical
01:56:58 observations and were not silently promoted to a later revalidation.

The same inspection read the OAK USB descriptor from sysfs without opening the
device. It reported serial/MXID `19443010F1B43A2E00`, corroborating the
retained identity, and link speed `480`, which remains a current SuperSpeed
gate failure rather than a camera-function diagnosis. The three persistent
serial-by-id paths were also unchanged.

This inventory makes these exact files eligible as later renderer inputs; it
does not freeze them or substitute for the renderer's own one-read hashing and
readback. Rehash every leaf immediately before rendering. It does not provide
the missing camera-to-base or native-IMU-to-base measurements, a current
candidate STM32 identity, a staged-loader proof, or any live camera, SLAM,
head, eye, motor, MPC, or performance evidence.

## Gate-A shadow-contract repair

The V4 input audit found that the prior synthetic shadow plant could not
possibly pass the candidate bootstrap. Its fixed 100 ms sample period had to
equal both the MPC step and control period, while the controller lease,
acknowledgement, and scheduling budgets impose a strict maximum runtime
service interval of 54,999,999 ns. No token choice could satisfy both
contracts.

The corrected qualification-only V2 plant uses a 50 ms period and narrows its
synthetic envelope to the candidate's ±30% PWM range and corresponding
synthetic ±0.3 m/s endpoints. Its exact checked-in identity is:

```text
path: configs/nano-wheels-off-qualification-template/qualification-shadow-only-synthetic-unvalidated-plant-v2.json
size_bytes: 962
sha256: aa96e9a3e75c540112d645a8dbefa54ba647574e90fc33e48b314b3c0094ded8
```

The paired navigation seed uses a 50 ms MPC/control period, 100 ms shadow
lease, ±30% PWM bounds, 5% slew bounds, and a 5% initial search radius. A
collision-free regression starts from STOP with a nonzero reference and
requires the solver to choose a nonzero command while respecting that slew
bound. This proves that the discrete search contains a useful candidate; it
does not tune the physical robot, prove Nano solver latency, or authorize
actuation.

The preparation template now fixes the remaining software policy and leaves
only seven unresolved values: the two calibration-preparer replacement
markers plus physically reviewed world-to-occupancy rotation/translation,
footprint radius, and obstacle-height bounds. The V4 render-input template
likewise fixes one bounded OAK/occupancy/inference/Rerun/storage/head/expression
policy and leaves only observed identity, exact source-path, build-provenance,
and generated-calibration fields. The renderer binds the supplied plant bytes
to the exact V2 ID, destination, semantic identity, and checked-in digest; a
valid JSON mutation under the same label is rejected. A fresh rendered bundle
must use the V2 plant and new derived hashes; no prior V1 bundle or digest is
reinterpreted.

## Prior 9da248a native qualification build (superseded)

The clean Nano checkout at `/home/makerspace/kiko` was fast-forwarded to exact
source revision:

```text
9da248a9944721173200cd57b9318db73b9890ec
```

The checked-out branch was
`codex/nano-expression-integration-stage`, tracking the same remote branch,
and `git status --short --branch` reported no worktree changes.

Using the retained native build inputs from
`/home/makerspace/kiko-native-evidence/83bad59-20260728T001205+0530`, the following
exact feature build completed on Linux aarch64 in 4m54s:

```text
cargo build --locked --release -p kiko-slam \
  --no-default-features \
  --features nano-wheels-off-qualification --bin kiko-slam
```

The resulting executable identity was:

```text
path: target/release/kiko-slam
size_bytes: 28975536
sha256: dc0870eb037cc46323952d677783c6bacbf254d6a85468f4a970421c79380262
elf_machine: AArch64
elf_type: PIE
build_id_sha1: 485cd2cfbdaa3bcf791560e5fd0b3088c9dce597
```

`readelf -d` found the expected direct DepthAI, OpenCV, C++, and system
dependencies. `ldd` under the retained `LD_LIBRARY_PATH` resolved every
entry, including the retained DepthAI, dynamic-calibration, OpenCV core,
OpenCV image-processing, OpenCV object-detection, and USB libraries. No
unresolved dependency was reported.

With that exact retained library path, the executable's
`nano-wheels-off-qualification --help` boundary loaded successfully. It
advertised the deployment root, relative launch document, state root, and the
qualification-only one-shot typed fault seam. The physical preconditions
remain attended-TTY inputs with no flag or environment bypass. The production
`nano-agent` feature does not compile the fault seam.

An invalid fault declaration was rejected at the command-line boundary with
the exact four allowed values: host monotonic-clock regression, partial UART
record, stale depth, and localization loss. This parsing check opened no
device. The new live sensor seams are explicitly synthetic, trigger only after
a controller-confirmed nonzero applied step, latch the corresponding
navigation state, and queue terminal stop. A selected declaration that reaches
an error-free normal teardown without being exercised is now a typed failure.

The timestamped, owner-private, SHA-256-manifested evidence directory is:

```text
/home/makerspace/kiko-native-evidence/9da248a-20260728T010119+0530
```

It retains the source revision, clean status, build inputs and command,
toolchain identity, executable identity, ELF headers and dynamic section,
complete loader output, CLI help, live owner and USB observations, and a
SHA-256 manifest. The retained build-input file itself has SHA-256
`835761b1d37c5cb6d868c10e69af4a22525f14ef2aa057c84b76028b6777906e`.

The 4m54s build duration is recorded only to distinguish the completed
command from an interrupted build. It is not a benchmark or a performance
claim. This build did not open a device or exercise camera frames, serial
traffic, SLAM, occupancy, head motion, eye output, motor output, MPC timing, or
physical safety behavior.

## Prior live owner and USB snapshot

At `2026-07-28T01:01:40+05:30`, read-only inspection found:

- all three persistent serial-by-id endpoints present;
- the STM32 endpoint `/dev/ttyACM0` had no process owner;
- one legacy Kiko `kiko_face_follow.py` child, PID 61621, held `/dev/ttyACM1`,
  `/dev/ttyACM2`, and the OAK USB node;
- its two-level `engine-guardian.sh` owner was still running and the user
  crontab still contained both reboot launch and minute-level guardian
  restoration entries;
- this child PID differed from the earlier read-only snapshot while the same
  guardian remained, which is consistent with a restart; the exact cause of
  that restart was not observed;
- the OAK USBFS device `03e7:f63b` was below the `480M` USB2 tree; its exact
  MXID was not re-queried because the existing process owned it;
- the separate `10000M` USB3 root and hub were present with no OAK below
  them;
- both canonical Kiko services were inactive; and
- no canonical qualification owner was started.

This process is not Fable and Fable is not treated as a current subsystem,
dependency, or runtime owner. It is a separate legacy Kiko process that is
currently an exclusive device owner. The canonical Kiko process must not
overlap it on the OAK, head, or eye endpoints. Under the current
canonical-owner policy, finding this conflict keeps Gate A closed: this
qualification workflow does not stop, signal, reconfigure, or otherwise
mutate the conflicting workload.

No process was stopped, signalled, reconfigured, or replaced during this
snapshot. No camera or serial endpoint was opened by the inspection.

The operator's clarification that Fable is no longer operating on the Nano is
therefore reflected directly in the current architecture. Final Gate A
qualification must still re-read the actual endpoint and respawner owners,
fail closed on any conflict, and start exactly one canonical Kiko owner only
after a conflict-free observation.

## Gate status after this refresh

This is exact native build evidence for code-bearing revision `9f1061d`. A
later evidence-only documentation commit does not change the executable
inputs; any later executable change would supersede this evidence and require
a new native build and identity. These physical/current items remain open:

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
