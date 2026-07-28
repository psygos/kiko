# Nano live-state audit — 2026-07-23

This is a read-only observation of the Jetson at
`makerspace@192.168.50.2`, taken at `2026-07-23T21:59:24+05:30`. Process IDs,
USB device numbers, uptime, and free space are transient. Re-run the same
inventory immediately before a handoff or deployment. No process, service,
repository, firmware, device configuration, or deployed file was changed by
this audit.

## Host and deployment state

- Hostname `ubuntu`; Linux `5.15.148-tegra`; aarch64; root filesystem is ext4
  on `/dev/nvme0n1p1`.
- Uptime was 9 hours 26 minutes. The 456 GiB root filesystem had 311 GiB free.
- No system or user `kiko*` or `fable*` systemd unit was installed.
- `/opt/kiko`, `/run/kiko`, and `/var/lib/kiko-nano-agent` had no installed
  canonical deployment content.
- `/home/makerspace/kiko` was clean at `482023e` on
  `codex/jetson-hardware-validation`.
- `/home/makerspace/kiko-nano-expression-integration` remained deliberately
  dirty at `bb0f78d`; it contains the preserved Fable-era head, control,
  navigation, OAK, documentation, and bench changes listed in
  `FABLE-NANO-HANDOFF-2026-07-21.md`. It must not be overwritten or treated as
  an immutable deployable revision.
- The Nano Git remote is `https://github.com/psygos/kiko.git`. The local
  `codex/nano-expression-integration` branch had no upstream or matching
  remote-tracking branch at the time of inspection.

## Current physical endpoint owners

The following Fable process tree was live:

```text
CRON -f
└─ /bin/sh -c /home/makerspace/kiko-follow/engine-guardian.sh
   └─ /bin/bash /home/makerspace/kiko-follow/engine-guardian.sh
      └─ python3 kiko_face_follow.py --duration-s 864000
```

The guardian is started by both of these user-crontab entries:

```text
@reboot /home/makerspace/kiko-follow/engine-guardian.sh
* * * * * pgrep -f 'engine-guardian[.]sh' >/dev/null || (setsid /home/makerspace/kiko-follow/engine-guardian.sh >/dev/null 2>&1 &)
```

It also respawns the Python child every eight seconds. Stopping only the child
is therefore not a handoff. A coordinated handoff must first prevent both cron
launch paths from recreating the guardian, terminate the exact guardian
normally, allow its exact child to park, and prove that neither process
returns. Broad `pkill`/`killall` operations are not acceptable.

The live Python process held these exact descriptors:

- head adapter `/dev/ttyACM1`;
- eye controller `/dev/ttyACM2`;
- OAK USB node `/dev/bus/usb/001/036`.

It did not hold the STM32 `/dev/ttyACM0`. The preserved Fable artifact hashes
still matched the earlier audit:

| Artifact | SHA-256 |
| --- | --- |
| `engine-guardian.sh` | `1255a9563b1e03ef917b74f220698a1ee80804c3c474f30f1d0e3f3d703b4336` |
| `kiko_face_follow.py` | `ef0c9fb48743bd51ec8af317084273682553ac6b30bed384c74731a0eb3daf4e` |
| `config.json` | `6444ce331d0fe66faf7de9b2696c8d0640881678975831505dd3e7a4e1eebcbc` |

The current Fable run was emitting five-second status records with
`head=TRACKING`, `eyes=SLEEPY`, and `person=False`. Its guardian log records
many earlier respawns, and the camera log records repeated `X_LINK_ERROR`
failures before the current run. This proves neither long-term camera
stability nor a production head-health contract.

## Enumerated devices and USB topology

Persistent serial identities were:

| Role | Persistent identity | Kernel endpoint |
| --- | --- | --- |
| STM32 ST-Link VCP | `usb-STMicroelectronics_STM32_STLink_066EFF313946303143221230-if02` | `/dev/ttyACM0` |
| Four-servo head adapter | `usb-1a86_USB_Single_Serial_5B14031114-if00` | `/dev/ttyACM1` |
| KEP2 eye controller | `usb-kiko_kiko-eyes-kep2_98c47919804f9f1aaacfd5fa0a20bf74-if00` | `/dev/ttyACM2` |

The OAK was the only Movidius device and remained on Bus 001 under the
480 Mbit/s USB-2 hub. Bus 002 exposed a 10,000 Mbit/s root and hub but no OAK.
The preserved Fable owner explicitly opens the exact device with
`dai.UsbSpeed.HIGH`, so its current 480 Mbit/s enumeration is requested
behavior and cannot diagnose the port, cable, or device's SuperSpeed ability.
The canonical owner now requests and requires `SUPER`. Newly rendered bundles
therefore select the twice-qualified 5 Gbit/s USB-3 mode and fail closed on any
different same-owner readback. Retained `SUPER_PLUS/SUPER` launches remain
parseable and retain that exact request rather than being silently capped or
relabeled; the camera qualifier is the explicit new diagnostic path. The
canonical same-owner readback—not this legacy topology—is the production
admission evidence.

## STM32 state

A passive, bounded two-second read opened the otherwise unowned persistent
STM32 serial endpoint at 115200 baud and wrote no application bytes. It
received 440 bytes of legacy lines such as:

```text
ODO,0,0,0,0,34890268
```

The installed controller is therefore not the KRP2 V2 image and cannot be
admitted by the typed controller owner. This passive observation says nothing
about motor direction, velocity, stop distance, or wheels. Flashing and
nonzero testing remain separate explicitly supervised operations.

The same ST-Link enumerated as USB `0483:374b`, reported serial
`066EFF313946303143221230`, and exposed its debug interface in addition to the
VCP above. `/usr/bin/st-flash` version `1.7.0` and
`/usr/bin/arm-none-eabi-objcopy` were installed. This establishes that a
concrete programmer and conversion tool are available on the Nano. A
read-only `st-info --probe` found exactly one programmer, identified the target
as an STM32F446 (`chipid 0x0421`), and reported 524,288 flash bytes with a
131,072-byte page/sector size plus 131,072 SRAM bytes. This is not evidence
that the firmware image, reserved boot-journal bytes, or a post-flash readback
has been qualified.

## Native build inputs found on Nano

- Python DepthAI was version 3.4.0.
- A clean detached DepthAI Core source tree existed at
  `/home/makerspace/work/depthai-core-v34`, exact commit
  `ba7a920a2568ea6eaaaebf3f92bbdb40924187ae` / tag `v3.4.0`.
- Its generated header reported SDK `3.4.0`, the same full commit, embedded
  device artifact
  `86cc8f6aa527b7c1f4b62129decd68e12bcf7d8a`, and embedded bootloader artifact
  `0.0.28`. Those are compiled-header provenance values, not a claim about
  firmware currently running on the connected OAK.
- Its aarch64 `build/libdepthai-core.so` SHA-256 was
  `0744500ab4f665af0641fd10881988146b73241212ac9523a86294e5737edae8`;
  `ldd` reported no unresolved dependency in the observed shell.
- That library's embedded `RUNPATH` points into the build tree under
  `/home/makerspace`, which `ProtectHome=true` intentionally hides from the
  service. Its two directly resolved non-system dependencies were
  `libdynamic_calibration.so` with SHA-256
  `30730ae6d367dcd927be7081f6a21d3bc4af65d857421ea3d3776d4ac00c7c53`
  and `libusb-1.0.so` with SHA-256
  `74eac03235e61b326ecb6532bd1d840f7b8fbaf55cfaa32b7e3079fc1208ede0`.
  All three files therefore need an exact-byte-bound runtime directory and
  explicit service search path; relying on the observed build-tree `RUNPATH`
  would make the installed unit fail closed before camera admission.
- Python ONNX Runtime was version 1.23.2 and supplied
  `libonnxruntime.so.1.23.2` with observed SHA-256
  `ab8c4363e06ac80b3d1279ea55ebea44e906c5b131ba783ff684a067540c0e94`.
  Its direct `NEEDED` entries were system libraries and its `RUNPATH` was
  `$ORIGIN`. The launch package must still bind the exact copied bytes; a
  Python package version alone is not runtime admission.
- SuperPoint and LightGlue model files were present in existing Kiko
  worktrees. Their presence is not a selected model identity or proof that the
  canonical process loads or runs them.

The production bundle must bind every installed runtime byte, including native
DepthAI dependencies if they are copied outside an independently managed
system installation. A successful build or `ldd` result does not qualify the
OAK stream graph, SLAM, inference correctness, latency, or physical control.

## Immediate gate implications recorded at audit time

This dated list preserves the conclusion drawn from that live snapshot. It is
not the current operational owner-release procedure. The 2026-07-27 Gate A
refresh and current qualification runbook supersede item 4: freshly prove no
competing owner or respawner exists; if one is found, retain the conflict and
stop without disabling, signalling, or killing that workload.

Before the wheel-attachment question is allowed:

1. finish, review, test, and commit the canonical source;
2. render a sentinel-free, byte-bound deployment package;
3. let the exact canonical owner request SuperSpeed and require its readback,
   relocating the connection only if that admission actually fails;
4. freshly prove no competing endpoint owner or respawner exists; retain any
   conflict and stop without mutating another workload, while preserving neck
   support and tension continuity;
5. install and admit the KRP2 firmware with an exact zero receipt;
6. start one canonical owner and prove head hold, RGB/eyes, all OAK streams,
   live SLAM/occupancy/Rerun, console ownership, and fault cleanup;
7. complete motor-inert transport tests and separately supervised wheels-off
   nonzero/MPC streaming tests.

None of those remaining steps may be inferred from this read-only audit.
