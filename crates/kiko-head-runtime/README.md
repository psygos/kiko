# Kiko head runtime

`kiko-head-runtime` is the sole host-side serial owner for Kiko's four Feetech
STS head servos. Its `kiko-head-commission` binary is read-only by default: it
reads the torque-switch register and qualified telemetry window for the four
fixed servo IDs and reports exactly what it observed. The runtime implements
two separately gated energising operations:

- capture the physical pose twice, command that same observed pose with
  explicit nonzero speed and torque limits, enable torque, and read back all
  four positions; and
- from an admitted start window, execute one bounded waypoint return to one
  exact, reviewed raw-encoder target.

Neither operation is an unbounded production head controller, an angular
calibration, or permission to infer a target from a camera observation.

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
  "hold_observed": null,
  "return_to_target": null
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

## Bounded return-to-target command

`return_to_target` is a separate strict configuration object. It contains the
same runtime tuning fields as `hold_observed`, plus exact
`minimum_start_ticks`, `maximum_start_ticks`, `target_ticks`, and
`maximum_travel_ticks` arrays in bow/curl/yaw/roll order. It also separates the
startup-hold `readback_tolerance_ticks` from
`final_target_tolerance_ticks`, `path_corridor_tolerance_ticks`, and
`direction_regression_tolerance_ticks`; all are raw encoder ticks. Completion
and direction tolerances cannot exceed the corridor tolerance, and none of the
three return tolerances can exceed the corresponding travel limit. Each
nonzero travel limit is at most 512 ticks and must cover the target's distance
from both ends of the corresponding admitted start window. The object also
requires a positive commissioning-only `maximum_hold_ms` no greater than
900,000 and the literal JSON consent value:

```json
"physical_motion_consent": "return_to_reviewed_target"
```

The process boundary independently requires all three CLI flags:

```text
--return-to-target --physical-torque-consent --physical-motion-consent
```

The command first establishes and verifies the observed-position hold. It then
reads all four complete telemetry windows on each control cycle and commands a
goal no more than 50 ticks ahead of each fresh position, at the configured
nonzero speed. The control period is 100 ms, the per-joint no-progress deadline
is 2 s, and the whole motion deadline is 20 s; these timing and step values are
fixed in the typed runtime rather than accepted from JSON. The exact target is
complete only after two consecutive, stopped, status-zero telemetry sets are
inside the final tolerance and agree inside the redundant-read tolerance.

At command time the actor reads two new complete, stopped, status-zero,
identity-ordered telemetry sets. Each four-joint set must span no more than one
100 ms control period, must still be no older than that period when admitted,
and corresponding positions must agree within the parsed redundant-read
tolerance. The freshest pose must still lie inside all four reviewed start
windows, and the actor rechecks its target distance against every travel limit.
Motion is not initialized from the earlier startup-hold pose.

Only a specialized `HeadReturnActorHandle`, created by consuming one complete
`ReturnToTargetConfig` and both physical-consent tokens, can issue the private
zero-argument return command. Target, device, start bounds, speed, torque,
tolerances, and travel limits therefore cannot be cross-paired at the command
API. Every telemetry identity and device-status field is checked before any
position from that set may influence motion or recovery.

The `maximum_hold_ms` field is parsed by `kiko-head-commission` into a separate
typed timer. It is not part of `ReturnToTargetConfig` or
`ReturnToTargetConfigInput`: the reusable return transaction describes motion,
not the lifetime of a caller which retains the actor afterward.

The complete path geometry for all four joints is admitted before a recovery
pose capability exists. A corridor departure therefore cannot authorize a new
goal from rejected telemetry. For a later kinematic fault such as timeout,
direction regression, no progress, or final-sample disagreement, the actor may
try once to replace the outstanding waypoint with the complete admitted
four-joint present pose. `KinematicFaultRecoveryWritten` is emitted only if all
four host writes complete and the actor remains the serial owner. Partial
recovery or waypoint batches retain the exact completed prefix and original
kinematic fault in the typed error. This is write-completion evidence, not a
servo acknowledgement or stopped readback.

Clock regression, identity/order mismatch, device status, telemetry failure,
or an incomplete recovery write terminates the actor. The commissioning actor
then attempts four-joint torque-disable cleanup. The production actor instead
closes ownership without a torque-switch write. The commissioning CLI also
shuts down immediately after every return fault, including an owner-retaining
fault; it does not turn an unsupervised recovery write into a long hold. After
success, only the commissioning wrapper applies its finite hold timer; the
reusable actor itself has no lease.

`HeadActorHandle`, `HeadReturnActorHandle`, and
`TensionPreservingHeadReturnActorHandle` expose a repeatable, read-only
`check_health` request for a supervisor to poll. Each
successful request retains exact bow/curl/yaw/roll telemetry, the typed active
hold target, and monotonic request, receive, and whole-check timing. Before a
successful return the return actor reports `StartupObserved`; after success it
reports `ReviewedReturn` with the exact configured target. Recoverable
kinematic faults retain a distinct `RecoverableReturnCommand` target. The actor
admits success only when every response has the expected identity, telemetry
device status is zero, the moving flag reports stopped, and position remains
inside the parsed readback tolerance around that active target. Speed, load,
voltage, temperature, current, and unqualified registers remain raw; this
crate assigns them no physical units or policy thresholds. Transport, framing,
protocol, clock, status, moving, and position failures remain distinct typed
outcomes with the accepted joint prefix. A failed or caller-cancelled health
receipt does not itself change goals or torque and does not release the actor's
exclusive bus ownership.

One actor admits at most one return attempt. A second request during motion is
reported as `CommandAlreadyInProgress`; a request after the recorded attempt is
reported separately as `CommandAlreadyAttempted` rather than being mislabeled
as concurrent work.

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
readback bound. A return additionally requires
`PhysicalHeadMotionConsent::explicitly_granted()`. The entire parsed
`ReturnToTargetConfig` is consumed by a commissioning return API or the
separate `spawn_tension_preserving_head_return_actor` /
`start_serial_tension_preserving_head_return_actor` production API; the
internal motion plan is not public and cannot be supplied later by a command
caller. The production API additionally requires
`ProductionTensionPreservingTakeoverConsent` and returns a narrower handle
which has no torque-disable method. The type deliberately contains no hold
lease or duration. A continuous production owner retains that handle, polls
health, and coordinates a hold-preserving ownership release; the commissioning
CLI independently owns its finite timer and disable-first cleanup.

Both generic and production spawn paths first require an active Tokio runtime.
Absence is returned as a typed error; production checks this before opening or
changing the physical serial port.

## State and error semantics

Commissioning and production have distinct first steps. Commissioning attempts
torque-disable on every joint and stops if any write does not complete.
Production performs no torque-switch read or write and makes no claim about
the prior torque state. Its first protocol traffic is the observation below.
The remaining startup transaction is shared and ordered:

1. Read every joint twice and admit only same-ID positions within tolerance.
2. Write all four nonzero torque limits before any goal write.
3. Write each freshly observed position together with mandatory nonzero speed.
4. Enable or refresh torque on all four joints.
5. Before each enable write, require that the oldest admitted observation is
   still inside the typed arming-freshness bound and that the remaining window
   covers every configured bounded write attempt; otherwise fail.
6. Read exact full telemetry twice for each expected ID. Both samples must be
   status zero and stopped, each must agree with its observed target, and the
   pair must agree with each other inside the configured tick bound.

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
completed and all eight stopped verification samples parsed and agreed.
`HeadStartupTorqueEvidence` distinguishes the commissioning disable report
from a tension-preserving production takeover; the latter deliberately carries
no prior torque-state claim. The installed servos' response level is zero and
the qualified full-telemetry window does not include goal, speed, torque-limit,
or torque-switch registers. Therefore this evidence does **not** claim those
register writes were acknowledged or independently read back.

`VerifiedHeadHealthEvidence` is narrower: it proves one later, complete,
identity-ordered status-zero stopped telemetry pass remained inside the
startup hold target tolerance. It retains every raw telemetry field and
timing observation, but still cannot claim that goal, torque, or torque-limit
registers were read back. Repeated calls perform repeated physical reads; the
runtime does not substitute cached startup evidence.

Return success is `VerifiedHeadReturnEvidence`: it retains the startup-hold
pose, the two fresh command-time telemetry sets and their monotonic receive
times, the fresh command-time start pose, exact reviewed target, every complete
four-joint waypoint write cycle, and the two final stopped telemetry sets. The
exact reviewed target must have completed one full four-joint write before
tolerance-based stopped samples can produce success.

Commissioning startup fault, requested shutdown, or handle loss attempts a
torque-disable write for every joint even if an earlier disable fails.
`all_writes_completed` means only that all four host writes completed; exact
outcomes remain in `ActorExit`.

The production actor never torque-disables on startup failure, return fault,
health fault, handle loss, ordinary API shutdown, or process/systemd shutdown.
Its distinct handle exposes only `release_ownership_preserving_hold`, and its
distinct exit type carries `HoldPreservingOwnershipReleaseEvidence`. That
evidence proves only that actor cleanup issued no torque-switch write; it does
not prove that torque was enabled before or after serial close. Electrical
power loss, another bus owner, servo protection, or forcibly aborting the task
can still release the neck. Physical torque release requires a separately
reviewed commissioning operation with physical support, not a generic
production lifecycle event.

An open, exclusivity, or serial-configuration error occurs before an actor or a
qualified bus exists. That path sends no protocol byte and reports
`SerialOpenError`; it cannot truthfully claim that torque-disable was attempted.

## Recorded physical evidence and present state

The detailed run record is in `docs/nano-bench-evidence-2026-07-21.md`. The
first return attempt started at `[2337, 2938, 2748, 2748]` ticks with a 10-tick
waypoint lead, speed 50 ticks/s, and torque limits `[600, 400, 400, 400]`
permille. It moved, then Curl reported `NoProgress` after 2 s with a best
remaining distance of 391 ticks. The then-running actor torque-disabled during
cleanup, and the operator observed the gravity-loaded neck fall. That outcome
is a failed return, not a successful safety demonstration.

The lead was changed to 50 ticks without increasing the 50 ticks/s speed or the
torque limits, and recoverable kinematic faults gained the evidenced
present-pose-write/owner-retention path described above. A second attempt
returned from
`[2211, 2576, 2858, 2906]` to `[2155, 2545, 2943, 2876]` in 18 waypoint cycles.
Its two stopped final samples agreed at bow 2160, curl 2548, yaw 2941, and roll
2874 ticks; all four device-status values were zero. After the bounded hold
ended, all four torque-disable writes completed. At the end of that session no
head process was running and the neck was free, not actively held. These facts
qualify one bounded return transaction; they do not qualify continuous
supervision or cold-boot head ownership.

The physical run above predates the later config-bound owner, command-time
start admission, bounded-span telemetry, hard I/O-inclusive deadlines,
all-joint recovery capability, lossless partial-batch errors, exact-target
write requirement, startup device-status rejection, and signal-driven CLI
shutdown described in this README. Those later changes have software test
evidence only and must not be retroactively described as physically qualified.

## Evidence and validation

The deterministic transport/clock suite covers nominal ordering, bounded noise
resynchronisation, truncation, wrong ID, checksum corruption, timeout, partial
write failure, explicit zero-progress retry, stale observations, moving and
unstable readbacks, readback mismatch, cancellation by handle drop, shutdown
queued during an in-flight startup fault, absent Tokio runtime, and continued
four-joint shutdown after disable failures. The return controller suite also
covers 50-tick non-crossing waypoints, consecutive stopped completion, travel
admission, full-set status precedence, corridor/direction/no-progress/motion
deadlines, clock regression, final-sample disagreement, and the narrow fault
classes allowed to attempt a recovery write. The health-check cases cover raw
field retention, canonical identity order, status and moving precedence,
target drift, transport and clock faults, repeated fresh reads, and a dropped
request receiver without an implicit torque write.

The current macOS host software-only run passed 49 runtime library tests and 8
commissioning-binary tests. A prior aarch64 software-only Nano run, before the
health seam existed, passed 17 protocol tests, 41 runtime library tests, and 8
commissioning-binary tests. No later Nano result is inferred from the host run.

Run the host checks with:

```text
cargo test -p kiko-head-runtime
cargo clippy -p kiko-head-runtime --all-targets -- -D warnings
cargo doc -p kiko-head-runtime --no-deps
cargo check -p kiko-head-runtime --target aarch64-unknown-linux-gnu
```

No electrical DTR/RTS measurement or movement benchmark is reported here. The
single physical return above is behavioral evidence, not a performance claim;
there is no synthetic benchmark standing in for hardware evidence.
