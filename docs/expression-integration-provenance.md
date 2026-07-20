# Expression integration provenance and evidence boundary

This document records the source drop inspected before Kiko expression support
was implemented. It prevents experimental scripts, copied build products, and
unqualified constants from silently becoming runtime authority.

## Source drop

- Source path at inspection: `/Users/ttrb/Downloads/Kiko_expression_engine`
- Inspection date: 2026-07-20
- Files: 1,281 regular files
- Apparent size from `du -sh`: 151 MiB
- Git metadata: absent
- Deterministic inventory digest: `7b0ff876aac7fd2c88727142012990973be8098905c890d724982f269b6d3e71`

The inventory digest is SHA-256 over the byte stream produced from the source
directory by:

```text
find . -type f -print0 |
  LC_ALL=C sort -z |
  xargs -0 shasum -a 256 |
  shasum -a 256
```

It identifies the inspected local tree, including copied dependencies and
build products. It is not a claim that the drop is a reproducible release.

Important individual SHA-256 digests:

| Artifact | SHA-256 |
| --- | --- |
| `README.md` | `56ebf6931a86b223a5323c3254b900fa9ee3ff95411956134a03516803462a74` |
| `eyes/Cargo.toml` | `5ffc8a41d522a6c59c478488a06300e78fe55a40b8b49a90e8311aaf8174436f` |
| `eyes/src/main.rs` | `543adcf15281046248a5d49fae92050b886193a89ae6e7057c370c3a0a818b15` |
| `eyes/src/host.rs` | `14b7afd7b3d5972177b85a7a2f22cd2bdca8b123d170d9f5543c66c95bbc7853` |
| `eyes/kiko.uf2` | `8756b365d51922abbf879c7dcd5fa8a62277b87997bca2638a52b9cc02222e9a` |
| `neck/sysid/calib.json` | `8ec182d486dace3cc3747ba19448d06c4153a4b23af21df5dcd211d16cf97533` |
| `neck/sysid/params.json` | `32cd933800aff7505cb98b288c47d4d02d973ab9344f120b4ce81a3437379862` |
| `neck/sysid/fit_report.json` | `f3b620643f18873c9c634db52f2a717b48790c62a3fad7facec29b80f65c3ef9` |
| `neck/sysid/runner.py` | `e9a5f8d60cbba825e5eb13a76a95a870fa93b68868918a0b42fc960f905059d2` |
| `neck/sysid/servo_bus.py` | `8b0b9b4461584632f90a89e3e8d138edcfa4eae28a1c75680d809fa778b0dd39` |

## Evidence accepted from the demonstrations

The supplied demonstrations are positive evidence that the particular neck
assembly, adapter, provisional calibration, and Python playback path produced
useful motions under supervision. The real logs and the user's observed demo
quality justify preserving the calibration and model as commissioning inputs,
replay fixtures, and regression evidence.

That evidence does not establish all of the properties required for unattended
runtime authority. In particular, the encoder-space logs cannot independently
prove the physical yaw angle when both command and observation use the same
ticks-per-radian conversion. The documents describe an approximately 2:1 yaw
transmission while the provisional calibration uses the direct-servo scale for
every axis. Physical yaw sign, ratio, zero, backlash, and safe approach envelope
therefore remain explicit commissioning facts. They must be measured rather
than guessed.

The fitted neck parameters are valid only as provisional, local predictive
evidence over the demonstrated trajectories. Flat or unidentifiable fit
parameters, corrupt telemetry samples, and larger-amplitude residuals must not
be hidden by the good small-motion result.

## Assets eligible for selective import

- Pico eye geometry, renderer, easing, and reaction ideas, after host/firmware
  parity tests.
- The GP15/GP16 source wiring and documented external eye-power requirement,
  subject to physical manifest confirmation.
- Raw neck logs, calibration JSON, model JSON, and MuJoCo model as immutable
  legacy evidence with their source digests.
- Reaction timelines as strictly parsed fixtures.
- Design documents as proposals or historical context, not implemented facts.

Production implementations derived from these assets use new typed boundaries,
tests, and source-controlled builds in this repository. Their provenance must
point back to this document.

## Quarantined from runtime authority

The following source-drop artifacts must not be installed as Nano services or
used to arm the robot:

- `runner.py`, `servo_bus.py`, `animate.py`, `eyes.py`, servo-ID assignment,
  flashing, and recovery scripts;
- `bus_info.json` as device identity;
- `calib.json` or `params.json` as self-authenticating physical approval;
- the copied UF2 as a release binary without a reproducible source/build ID;
- copied `node_modules`, Rust `target`, browser `dist`, media, and unrelated CAD
  assets; and
- proposed base-control, DWA, or pure-pursuit interfaces that would bypass the
  canonical occupancy, MPC, KRP2 V2, and STM32 safety contract.

The source Python head loop is specifically disqualified as production control
because a failed position read can be converted to raw tick zero during its
neutral fade, timing uses wall clock, shutdown is not guaranteed, commands do
not produce applied evidence, and process death cannot execute its host-side
recovery behavior.

## Required qualification before head motion

The production head path is introduced in ordered gates:

1. exact adapter and servo identity, read-only register inventory, and checked
   telemetry;
2. externally observed neutral, joint signs, yaw transmission, backlash, and
   eye orientation;
3. present-pose hold before any commanded transition;
4. bounded, monotonic, rate-limited natural-pose approach inside a qualified
   envelope;
5. unplug, process-kill, low-voltage, thermal, implausible-telemetry, and
   communication-failure tests; and
6. only then, bounded expressive head motion.

The initial integrated demonstration intentionally holds the qualified natural
head pose and expresses through the eyes. Working demo calibration is retained
and respected; it is not promoted into a stronger physical claim than the
available evidence supports.

## Reproducibility status at import

- Python source syntax compilation succeeded during audit.
- A clean RP2350 firmware build was not reproduced because the required Rust
  embedded target was absent on the audit host.
- The copied browser dependency tree contained platform-specific build
  products and did not provide a reproducible clean test result.
- No Nano runtime or performance claim is inherited from the Mac demonstration.

These limits remain visible until source-controlled builds and target-specific
tests replace them.
