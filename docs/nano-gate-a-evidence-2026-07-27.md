# Nano Gate A evidence refresh — 2026-07-27/28

This file records current, bounded evidence for the wheel-attachment gate. It
does not declare Gate A passed and does not authorize wheel attachment,
powered motion, deployment, or production use.

## Exact native qualification build

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

## Live owner and USB snapshot

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

This is exact native build evidence for the final code-bearing revision
`9da248a`. A later evidence-only documentation commit does not change the
executable inputs; any later executable change would supersede this evidence
and require a new native build and identity. These physical/current items
remain open:

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
