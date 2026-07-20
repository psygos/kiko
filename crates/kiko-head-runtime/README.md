# Kiko head runtime

`kiko-head-runtime` is the sole host-side serial owner for Kiko's four Feetech
STS head servos. Its `kiko-head-commission` binary is read-only by default: it
reads the torque-switch register and qualified telemetry window for the four
fixed servo IDs and reports exactly what it observed. The runtime implements
one separately gated energising operation: capture the physical pose twice,
command that same observed pose with explicit nonzero speed and torque limits,
enable torque, and read back all four positions. It exposes no calibrated
motion command.

## Commissioning command

The safe default sends eight fixed STS READ requests and no register write:

```text
cargo run -p kiko-head-runtime --bin kiko-head-commission -- \
  --config /absolute/path/to/head-commission.json
```

Configuration is strict JSON: unknown and duplicate fields fail, the file is
bounded to 16 KiB, and the path must be absolute without `.` or `..`
components. A read-only file needs only:

```json
{
  "schema_version": 1,
  "probe": {
    "device_path": "/dev/serial/by-id/REPLACE_WITH_REVIEWED_IDENTITY",
    "response_timeout_ms": 100,
    "request_timeout_ms": 100,
    "noise_budget_bytes": 32
  },
  "hold_observed": null
}
```

The optional `hold_observed` object additionally requires `write_timeout_ms`,
`arming_freshness_ms`, `write_attempts`, both position tolerances,
`goal_speed_ticks_per_second`, four `torque_limit_permille` values, exact
`minimum_ticks` and `maximum_ticks` arrays in bow/curl/yaw/roll order, and a
positive `maximum_hold_ms` no greater than 900,000. Every joint window is at
most 256 ticks wide. These assembly-specific values must come from reviewed
physical evidence; the software does not guess them.

Even with a valid hold configuration, torque writes remain unavailable unless
both flags are present:

```text
--hold-observed --physical-torque-consent
```

That mode first completes the read-only probe, then uses the existing typed
actor to torque-disable, observe twice, admit the pose inside all four reviewed
windows, apply the bounded observed-position hold, and verify stopped
telemetry. It does not move to a stored or inferred “normal” pose. SIGINT,
SIGTERM, or the configured duration initiates a four-joint torque-disable; the
per-joint cleanup result is printed. SIGKILL, power loss, or process failure
cannot provide that cleanup guarantee.

## Boundary contract

Weak process configuration is admitted once by `HeadRuntimeConfig::parse`.
The runtime accepts only:

- an exact Linux `/dev/serial/by-id/<identity>` path or exact macOS
  `/dev/cu.<identity>` path (no enumeration, `/dev/ttyUSB*` guess, or fallback);
- positive response/write timeouts and an observation-to-arming freshness bound,
  each no greater than 5,000 ms;
- one through eight explicitly configured write attempts;
- at most 1,024 prefix-noise bytes and protocol-qualified position tolerances;
- a protocol-valid nonzero speed and four nonzero torque limits.

Production opening claims OS-level exclusive ownership and applies 1,000,000
baud, 8 data bits, no parity, 1 stop bit, and no flow control. Because opening a
TTY can itself alter modem outputs, DTR false and RTS true are applied *after*
open and before any protocol traffic. Line coding is read back through the
driver. The serial API has no electrical readback for output DTR/RTS, so the
corresponding evidence means their OS setters accepted the values—not that an
electrical probe measured them.

Starting the actor requires `PhysicalTorqueEnableConsent::explicitly_granted()`.
Natural hold is not guaranteed to be motionless: the head can move between the
second observation and torque engagement or settle within the configured
readback bound. No pre-recorded/calibrated target is sent.

Both generic and production spawn paths first require an active Tokio runtime.
Absence is returned as a typed error; production checks this before opening or
changing the physical serial port.

## State and error semantics

Startup is ordered and fail-closed:

1. Attempt torque-disable on every joint and stop if any write does not complete.
2. Read every joint twice and admit only same-ID positions within tolerance.
3. Write all four nonzero torque limits before any goal write.
4. Write each freshly observed position together with mandatory nonzero speed.
5. Enable torque on all four joints.
6. Before each enable write, require that the oldest admitted observation is
   still inside the typed arming-freshness bound and that the remaining window
   covers every configured bounded write attempt; otherwise fail and disable.
7. Read exact full telemetry twice for each expected ID. Both samples must be
   stopped, each must agree with its observed target, and the pair must agree
   with each other inside the configured tick bound.

Every status response is delimited into a fixed 21-byte buffer. Prefix noise is
bounded and counted. After `FF FF`, the frame is never silently skipped:
declared length, expected ID, exact parameter count, checksum, device status,
and typed telemetry are propagated from `kiko-head-protocol`. A response timeout
covers the complete frame rather than restarting for every chunk.

A write timeout covers one attempt. Only timeout/interrupted failures with zero
known bytes transferred may use another configured attempt. Partial or
otherwise non-retryable writes fail immediately. Every recovered failure is
retained in success evidence; the transport performs no internal retry.

Startup success is `VerifiedNaturalHoldEvidence`: all requested host writes
completed and all eight stopped verification samples parsed and agreed. The installed servos'
response level is zero and the qualified full-telemetry window does not include
goal, speed, torque-limit, or torque-switch registers. Therefore this evidence
does **not** claim those register writes were acknowledged or independently
read back.

On startup fault, requested shutdown, or loss of the public handle, the actor
attempts a torque-disable write for every joint even if an earlier disable
fails. `all_writes_completed` means only that all four host writes completed.
The exact outcomes are returned in `TorqueDisableReport` and remain
in `ActorExit`. Dropping the handle requests this asynchronous cleanup; callers
must await the actor task to know its outcome. Process termination, power loss,
or forcibly aborting the actor task cannot be made safe by Rust `Drop` and is
not claimed to disable hardware.

An open, exclusivity, or serial-configuration error occurs before an actor or a
qualified bus exists. That path sends no protocol byte and reports
`SerialOpenError`; it cannot truthfully claim that torque-disable was attempted.

## Evidence and validation

The deterministic transport/clock suite covers nominal ordering, bounded noise
resynchronisation, truncation, wrong ID, checksum corruption, timeout, partial
write failure, explicit zero-progress retry, stale observations, moving and
unstable readbacks, readback mismatch, cancellation by handle drop, shutdown
queued during an in-flight startup fault, absent Tokio runtime, and continued
four-joint shutdown after disable failures.

Run the host checks with:

```text
cargo test -p kiko-head-runtime
cargo clippy -p kiko-head-runtime --all-targets -- -D warnings
cargo doc -p kiko-head-runtime --no-deps
cargo check -p kiko-head-runtime --target aarch64-unknown-linux-gnu
```

No physical-head test, electrical DTR/RTS measurement, or movement benchmark is
reported here. No performance claim is made, so there is no synthetic benchmark
standing in for hardware evidence.
