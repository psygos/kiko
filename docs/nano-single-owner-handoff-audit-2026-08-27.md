# Nano single-owner handoff audit — 2026-08-27

Audit time: `2026-08-27T02:01:27+05:30`

Scope: read-only SSH inspection of the connected Jetson Orin Nano. No process
was stopped or signalled; no serial or USB endpoint was opened; no firmware,
bundle, service, configuration, or repository state was changed; and no motor,
servo, camera-pipeline, SLAM, or navigation command was issued.

## Host identity

- Hardware: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit.
- Architecture: Linux `aarch64`.
- Kernel: `5.15.148-tegra`.
- Working source: `/home/makerspace/kiko`.
- Checked-out commit: `e53d7cb084a9b56f49df484f6d8bc7f46f0b39e6`.
- Tracked remote branch resolved to the same commit at inspection time.

The Orin working tree intentionally contains the field Fable expression files
as modifications plus macOS `._` resource-fork files. It must not be reset,
overwritten, or used as a clean build source. The canonical Mac branch was 54
commits ahead of its configured remote after local commits `92acb38` and
`01efa81`; those commits were not pushed or deployed by this audit.

## Active physical owners

The running process chain was:

```text
PID 1049  /bin/sh -c .../kiko-accessory-commissioning-guardian.sh
PID 1058  bash .../kiko-accessory-commissioning-guardian.sh
PID 1073  python3 kiko_face_follow.py --duration-s 864000
          --heartbeat-file /tmp/kiko-expression-heartbeat
```

The heartbeat was fresh during inspection. The Python process exclusively
owned:

- OAK: `/dev/bus/usb/002/003`;
- head adapter: `/dev/ttyACM1`;
- KEP2 eyes: `/dev/ttyACM2`.

The STM32 `/dev/ttyACM0` was present and had no open owner. Its stable paths
were:

```text
head  /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00
base  /dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066EFF313946303143221230-if02
eyes  /dev/serial/by-id/usb-kiko_kiko-eyes-kep2_98c47919804f9f1aaacfd5fa0a20bf74-if00
```

The OAK enumerated at `5000M` beneath a `10000M` root/hub path. This is direct
USB topology evidence that the current connection is USB3. It does not prove
camera frame delivery, configured stream rates, SLAM throughput, or long-run
USB stability.

## Live Fable state

The current field configuration SHA-256 was exactly:

```text
46d69519425caba5ace1920d39ff8a07101bf86b79eacb1cdeb53f1dd8957a56
```

That matches the canonical source now bound by `92acb38`. The active log
reported `head=TRACKING`, `compliant=FOLLOWING_EXPRESSION`, eyes `SLEEPY`, and
a held goal close to the current natural pose. It also retained intermittent
implausible temperature bytes. Those bytes are protocol observations only;
this audit makes no thermal diagnosis or temperature-unit claim.

The live copies of `config.json`, `compliant_head.py`, and
`organic_motion.py` matched the canonical retained files by SHA-256. The live
`kiko_face_follow.py` and `head_thermal.py` hashes differed from current
canonical source, consistent with the intentionally preserved older field
working tree. Therefore a live Python demonstration is not evidence that every
later Fable replay or canonical Rust change is active on the robot.

The legacy Python pet journal was `233,937,023` bytes. Its size confirms why
the canonical Rust owner uses a validated, owner-only, 16 MiB bounded journal;
it is not a performance measurement. This audit did not truncate or rotate the
legacy file.

## Lifecycle state

The guardian is launched by both:

```text
@reboot .../kiko-accessory-commissioning-guardian.sh
* * * * * .../kiko-accessory-commissioning-guardian.sh
```

No Kiko, Fable, or robot systemd unit was installed. No canonical `/opt/kiko`,
`/etc/kiko`, or `/var/lib/kiko` deployment root existed.

The newest `target/release/kiko-slam` on the Orin was dated 2026-08-01. Its
default dynamic-link search could not resolve `libdepthai-core.so`. That binary
predates the current single-owner, attended-navigation, Fable merge-back, and
live-SLAM evidence work and is not a deployment candidate.

## Readiness classification

| Requirement | Evidence at audit | Classification |
| --- | --- | --- |
| head/eye character visibly alive | active Fable Python owner and fresh heartbeat | active legacy field path |
| OAK on USB3 | `5000M` readback and exclusive Python file descriptor | transport present; streams unqualified in this audit |
| STM32 available | stable `/dev/ttyACM0`, no owner | present and idle |
| one canonical endpoint owner | Python owns accessories; no canonical owner running | not achieved |
| canonical service at boot | cron guardian only; no systemd unit | not achieved |
| current canonical binary/bundle | old unlinked binary; no `/opt/kiko` bundle | not achieved |
| online sparse SLAM and occupancy | no canonical process | not running |
| GUI manual/MPC/navigation | no canonical process | not running |
| wheels-off fault qualification under PID 1 | no current bundle/service | not performed |

## Exact next handoff

Do not start the canonical process beside PID 1073. The next attended session
must first produce an immutable current aarch64 bundle from a clean source tree
and materialize the exact V4 policy/review pair. With motor power disconnected,
perform a coordinated ownership handoff from the cron guardian to the one
canonical foreground owner and verify:

1. exact OAK/head/eye/STM32 identities and USB3 readback;
2. current replacement-servo natural admission, compliance and expression;
3. RGB, stereo, depth, IMU, sparse-SLAM completion evidence, occupancy and
   Rerun from the one OAK graph;
4. GUI authentication, manual deadman, software emergency stop, and simulated
   MPC command streaming while the base remains physically powerless;
5. every startup, disconnect, stale-data, controller-reset, accessory,
   watchdog, and shutdown fault under PID 1; and
6. restoration of the Fable owner if the canonical candidate fails before
   promotion, without running both simultaneously.

Only that attended evidence can justify asking to attach wheels for
encoderless plant calibration and live mapping/navigation.
