# MPC-to-STM32 actuation safety contract

Kiko navigation remains shadow-only unless an operator deliberately builds the
actuation feature, supplies a separate physical-authority manifest, names the
exact robot on the command line, and completes a live V2 handshake. The
checked-in synthetic plant model is not actuation authority.

This document describes software guarantees. It is not evidence that a
particular robot, motor driver, calibration, watchdog tolerance, braking mode,
or plant model has been physically validated.

## One-command continuation invariant

An armed navigation process has at most one physical command in flight. Every
navigation tick follows this order:

1. prove that the prior STM32-applied receipt is still current;
2. compute and journal one shadow safety decision;
3. send that decision once with the exact controller boot ID, control epoch,
   sequence, PWM pair, and a bounded absolute controller deadline;
4. wait within the configured apply budget for the matching STM32
   `AppliedResult` relayed by the robot server; and
5. only after exact, timely agreement may another MPC tick begin.

The live worker enforces this as one `LiveMpcControlDriver` call whose success
contains both the coordinator outcome and the matching applied receipt. Point
goals and admitted manual body-twist commands use this same owner. Manual input
is converted only to an expiring body-frame MPC reference; it has no direct
PWM API. Explicit manual stop and deadman expiry still create a journaled zero
safety decision and require an exact zero receipt.

This makes the existing shadow record a valid previous-PWM input during an
armed run: the process continues only after the same PWM was confirmed at the
controller timer boundary. A timeout, malformed or mismatched result,
controller restart, stale receipt, serial/network failure, or sequence fault
permanently latches that process out of motion. It makes bounded stop attempts
and exits; it never retries an uncertain nonzero command and never falls back
to requested shadow PWM.

Initial sequence zero is always zero PWM and must itself receive an exact
controller-applied result. Normal shutdown and every authoritative error issue
a request-correlated `HostStop` and preserve the returned `HostStopResult`, or
preserve exact uncertainty when no matching result arrives. A destructor can
make only a best-effort stop attempt and does not claim success.

## Deadline model

The host manifest must prove:

```text
solver budget + apply-ack budget + scheduling guard < control period
```

For motion, collision-valid-through is an exclusive host acknowledgement
admission deadline for that decision. The command client uses the earlier of
that instant and its apply-ack timeout. It does not treat collision freshness as
a promise that a dynamic scene remains unchanged for the controller lease, and
it does not truncate a successful receipt's conservative controller-lease
evidence at the old decision deadline. The server records the first datagram
receive time, subtracts time already spent in transit, and translates only the
remaining lease into the controller uptime domain using a fresh heartbeat, a
two-sided operator-supplied clock-error bound, and a quantization margin. It
never refreshes a deadline on a duplicate. A delayed UART frame therefore
cannot start a fresh relative lease.

The authority manifest must also prove that the controller lease is strictly
longer than the control-period, solver-budget, apply-ack-budget, and
scheduling-guard bridge, and that `apply-ack budget + controller lease +
deadline tolerance` does not exceed the operator's maximum uncommanded-motion
bound.

The STM32 compares the absolute expiry in an independent timer interrupt. The
main loop separately evaluates the same lease, and the independent watchdog
covers a stalled loop or interrupt subsystem. Server or host acknowledgement
timeouts are failures, never successful application claims.

All configured durations have explicit millisecond or nanosecond names. Bounds
are calculated with checked integer arithmetic. Equality at a deadline is
expired.

Dynamic obstacles are reconsidered only when a new sensor observation reaches
the next admitted control tick. If current collision evidence is unavailable,
the safety decision is zero and that zero still requires an exact applied ACK;
if the host stops progressing, the prior nonzero command expires at the
controller lease. These are software bounds, not a measured human-avoidance
reaction time. Camera, inference, scheduling, serial, motor, and braking
latencies still require end-to-end physical measurement.

The controller manifest also declares the maximum host command rate and one
bounded serial-transmit duration. The server enforces that rate for `ApplyPwm`
without delaying `HostStop`, checks the exact 8N1 record budget against
115200 bit/s in each full-duplex direction, and applies one deadline across
partial serial writes and flush. A transmit interruption records its phase and
written-byte count but cannot prove whether the controller received a partial
or complete record. Coordinated owner shutdown is clean only after the exact
safe `HostStopResult`; a completed write alone remains uncertain. The
calculation and still-unmeasured WCET/hardware gates are recorded in
`docs/stm32-streaming-qualification.md`.

## Identity and integrity

Arm-capable traffic uses V2 fixed binary frames with explicit little-endian
fields, exact lengths, reserved-zero checks, CRC-32C, and COBS record boundaries
on UART. Legacy eight-byte UDP packets and ASCII serial commands cannot reach
the V2 actuator.

The handshake binds all commands and results to:

- the STM32 96-bit device UID;
- an externally supplied, reset-unique controller boot ID for every
  motion-capable build;
- a per-control-session epoch;
- the expected firmware ABI and build ID;
- the actuator configuration fingerprint; and
- exact, nonwrapping command succession within that epoch.

A controller reset invalidates all server and host ownership. Duplicate
byte-identical commands can replay a cached result but do not reapply output or
extend a lease. A duplicate key with different content, an old/gapped/exhausted
sequence, wrong identity, or wrong epoch stops and revokes the session.

CRC detects accidental corruption; it does not authenticate a sender. The host
actuation parser accepts only an IP-literal loopback command endpoint. Remote
operation requires a separately authenticated tunnel and network policy.

The STM32F446 has no hardware random-number generator. The checked-in default
firmware therefore uses no motion-capable boot identity: it is a diagnostic,
motion-disabled profile. A reviewed motion profile must supply per-boot
identity/entropy externally; a device UID, timer value, or fixed token is not a
substitute for session uniqueness.

The repository now also provides a provisioned flash-journal identity source.
The firmware linker reserves STM32F446 sector 7
(`0x08060000..0x0807ffff`) outside the executable image. The host-only
`kiko-boot-journal-image` tool creates a new, mode-`0600`, 128 KiB sector image
with one CSPRNG-generated nonzero provisioning seed. Deployment must flash and
read back all 131,072 bytes and compare SHA-256 before selecting the
`flash-boot-journal` firmware feature.

Each journal-enabled boot programs and rereads one 16-byte counter record
before serial or motor admission. A valid counter equals its one-based physical
slot, so an interrupted or later-corrupted programmed slot is burned rather
than allowing its identity to be reused. Any programmed record after an erased
gap, a malformed header, a nonmatching record, a failed program/readback, or a
full journal fails startup with outputs disabled. The firmware never erases the
journal automatically. There are 8,190 bounded boot slots per provisioning;
explicit maintenance must create a new random provisioning seed after any
sector erase. This is reset-identity evidence, not authentication, motor
authority, or physical safety evidence.

## Motor output contract

Nonzero output is representable only while armed, fault-free, in-envelope, and
before the exact command deadline. The firmware rejects values outside the
compiled actuator envelope; it does not clamp them.

For a sign reversal, each wheel independently:

1. disables both direction channels and writes zero duty;
2. waits the configured neutral interval, including at least one PWM period;
3. preloads only the new direction channel while disabled;
4. waits for preload transfer; and
5. enables only that channel before reporting application.

Zero, lease expiry, receive corruption, fault, and emergency stop bypass slew
limits and disable all motor PWM channels immediately. “Channels disabled and
low” is the software claim. It must not be called coast or brake until the
specific driver truth table and wiring have been physically established.

## Required physical evidence before motion

The external actuation manifest is intentionally named an
operator-claimed approval. Parsing proves content agreement, not authenticity
or physical truth. It binds the raw navigation-config SHA-256, robot ID,
controller UID, firmware/build/config identities, plant dataset and fit claims,
calibration IDs, deadlines, and the operator's maximum uncommanded-motion
limit. The live controller handshake must match it exactly.

Navigation-actuation schema V2 has two intentionally different plant
bindings. `plant_dataset_content_id` is the canonical
`sha256:<64-lowercase-hex>` identity of the exact evidence-dataset artifact
named by the plant model. `plant_artifact_sha256_hex` is the 64-digit lowercase
SHA-256 of the exact serialized plant-model artifact selected by the device
manifest. The parser represents them as different domain types, rejects the
ambiguous V1 schema, and rejects a V2 document that reuses the dataset digest
as the artifact digest. A commissioning proposal supplies candidate values
only; separate evidence review, physical approval, manifest rebind, and normal
production admission are still required.

Motion must remain disabled until hardware work establishes at least:

- canonical left/right and forward/reverse wiring and signs;
- the motor-driver input truth table and required reversal dead time;
- the physical result of disabled/both-low inputs;
- current limiting, thermal protection, and driver-fault wiring;
- actual MCU watchdog and deadline-clock tolerances;
- safe output behavior during reset, brownout, and hard fault;
- a default-off driver-enable gate and independent external cutoff;
- a physical emergency stop independent of Jetson and STM32 software; and
- plant, IMU, camera, and extrinsic calibration provenance on this robot.

Host tests, target builds, fault injection, and benchmarks cannot replace
those checks. Until they are recorded, the defensible status is “bench-ready
software, motion authority withheld,” not “reliably drives the robot.”

## Observability boundary

The authoritative runtime status is the typed V2 `StatusQuery`/`StatusReport`
path. Rerun records the host's exact applied receipts and the reported
remaining lease at server emission for diagnosis, but it does not observe wheel
motion and is not physical evidence.
The standalone `robot-server` process and its unrelated HTTP/camera state
store have been removed. The package now supplies only the typed controller
owner used inside `kiko-slam` plus bounded identity/transport qualification
tools. The unified loopback console observes the in-process owner's exact
evidence and never opens a second controller or camera.
