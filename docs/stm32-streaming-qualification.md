# STM32/Jetson streaming contract and qualification

This document records software bounds, two measured motor-inert Nano transport
runs, and the still-unmeasured hardware qualification plan. The measured
section below reports only its exact 20 Hz and 50 Hz diagnostic
request/response runs. It does not claim motion-capable throughput or latency,
firmware WCET, motor response, or physical stopping distance.

The attended full-stack raw-PWM qualification procedure and strict candidate
asset templates are in `docs/nano-wheels-off-qualification.md` and
`configs/nano-wheels-off-qualification-template/README.md`. That procedure is
manual-only, keeps the wheels removed, and does not route shadow MPC output to
the provisional controller.

## Admitted host contract

The controller-server JSON now parses two explicit transport fields:

- `maximum_host_command_rate_hz`: `1..=100`; the actor enforces the derived
  minimum interval between serial `ApplyPwm` records. A rate-limited command is
  rejected without consuming its sequence. `HostStop` and coordinated shutdown
  bypass this limiter.
- `serial_transmit_timeout_ms`: `7..=100`, strictly below
  `serial_applied_ack_timeout_ms`. Seven milliseconds is the integer ceiling
  for one maximum 74-byte KRP2 record at 115200 bit/s, 8N1
  (`74 * 10 / 115200` seconds). One deadline covers all partial writes and the
  flush. Deadline, shutdown cancellation, write failure, and flush failure
  retain the exact phase and written-byte count and never become application
  evidence.

The intended deployed closed-loop baseline is **20 ms / 50 Hz**. The parser
retains 100 Hz as a bounded configuration-domain ceiling, and the motor-inert
qualification tool retains 75 and 100 Hz schedules, but **75/100 Hz are
transport-stress cases only** pending hardware measurement. They do not admit
motion, do not establish a closed-loop MPC rate, and do not justify weakening
the positive scheduling guard between the server interval and navigation
control period.

Admission computes both full-duplex directions from maximum COBS record sizes.
At a configured 20 ms heartbeat, the controller-to-host calculation reserves:

- 50 maximum-size Heartbeat schedules/s;
- 10 maximum-size ObservationalOdometry schedules/s;
- one maximum-size ControllerHello schedule/s; and
- one maximum-size AppliedResult for every admitted host command.

The canonical maximum record sizes are 58 bytes for AppliedResult, 59 for
Heartbeat, 62 for ObservationalOdometry, 70 for ControllerHello, and 48 for
ApplyPwm. Each encoded byte costs ten 8N1 wire bits. The resulting necessary
average-bandwidth bounds are:

| Schedule | Purpose | Controller to host | Fraction of 115200 | Host to controller | Fraction of 115200 |
| ---: | :--- | ---: | ---: | ---: | ---: |
| 20 Hz | lower-rate baseline | 48,000 bit/s | 41.67% | 9,600 bit/s | 8.33% |
| 50 Hz | intended closed-loop baseline | 65,400 bit/s | 56.77% | 24,000 bit/s | 20.83% |
| 75 Hz | motor-inert transport stress | 79,900 bit/s | 69.36% | 36,000 bit/s | 31.25% |
| 100 Hz | motor-inert transport stress | 94,400 bit/s | 81.94% | 48,000 bit/s | 41.67% |

This proves that the declared periodic bytes do not exceed the nominal
full-duplex wire rate. It does not prove USB CDC buffering, OS scheduling,
deadline compliance, or controller service time; those require measurements.

## Fairness and transmit priority

The host serial actor:

- polls coordinated shutdown first, a generation-latched host `HostStop`
  second, and an expired ACK/heartbeat timer third;
- coalesces concurrent host stops onto one ForceStop transaction without an
  unbounded request queue or actor-held waiter set; a host stop interrupts an
  ordinary partial UART record, explicitly re-delimits it, and then sends the
  ForceStop;
- alternates preference between one bounded serial read turn and one host
  request turn, so a continuously readable UART cannot monopolize the actor;
- reads at most one maximum UART record per turn; and
- requires an exact safe `HostStopResult` before coordinated shutdown is
  reported clean. A completed `ForceStop` write without that receipt remains
  explicitly uncertain.

The STM32 main loop consumes at most 74 RX bytes and one decoded record per
iteration before progressing the controller, lease checks, telemetry schedules,
and watchdog decision. Its TX scheduler owns three record-preserving queues:

1. HostStopResult;
2. AppliedResult and ControllerReady; and
3. best-effort Hello, Heartbeat, odometry, and transport diagnostics.

A selected record finishes through its zero delimiter before priority changes,
so records cannot interleave. Saturating best-effort capacity cannot consume
stop or applied-result capacity. Critical-capacity exhaustion latches the
serial-integrity fault and stops motion; best-effort saturation drops that
observation, allowing the host heartbeat-age gate to fail closed without
discarding a reserved safety response.

## Reproducible motor-inert measurement

`v2_transport_qualify` admits only the exact motor-inert diagnostic capability,
an exact controller identity, and run-bound live freshness. It never creates a
control session and never sends PWM. Run a separate process for each exact
rate and retain each schema-3 JSON output.

The read-only `v2_identity_probe` retains its legacy receive boundary and
schema-2 output: a fresh host owner clears only the host input queue once and
excludes subsequently delivered bytes through the first zero delimiter. That
establishes one explicit record boundary for the passive identity observation,
but neither the clear nor that boundary proves that upstream ST-Link, USB, or
in-flight buffers were empty.

The transport qualifier uses a stronger schema-3 freshness protocol. After the
same one-time host input clear, it raw-discards every byte delivered during a
fixed 1,000 ms quarantine, then discards through one explicit following zero
delimiter. Strict canonical decoding begins only at that boundary: an empty,
malformed, or oversized subsequent record fails the run, and retries never
resynchronize the decoder.

An exact motor-inert `ControllerHello` and idle-safe `Heartbeat` decoded after
the boundary are candidate evidence only. They authorize at most three
pre-measurement, motor-inert diagnostic challenges under the qualifier's fresh
nonzero run ID. The attempts use reserved descending sequences
`u32::MAX`, `u32::MAX - 1`, and `u32::MAX - 2`, and each carries its recorded
host-elapsed-nanosecond token. Only an exact run/sequence/token echo for the
latest outstanding attempt can match. Final admission additionally requires a
subsequently decoded exact `ControllerHello` and an idle-safe `Heartbeat`
strictly forward from the matched report preparation time. Both the
request-to-prepare service delta and the report-to-heartbeat delta must be
wrapping-forward by less than half the `u32` clock range and no greater than
their conservative host-elapsed bounds.

This is evidence of a live, invocation-bound request/response path followed by
new controller liveness. It is not proof that any upstream buffer was empty.
The quarantine, alignment, candidate observations, one-to-three challenge
writes, matched report, and post-match liveness are excluded from benchmark
metrics.

```text
cargo run --locked -p robot-server --bin v2_transport_qualify -- \
  --serial-device /dev/serial/by-id/REPLACE_WITH_EXACT_STM32_ID \
  --controller-uid-hex REPLACE_WITH_24_HEX \
  --boot-id REPLACE_WITH_CURRENT_NONZERO_BOOT_ID \
  --firmware-abi 2 \
  --firmware-build-id REPLACE_WITH_MOTOR_INERT_BUILD \
  --actuator-config-fingerprint-hex 4b494b4f2d4e4f2d4143542d56312121 \
  --capabilities-bits REPLACE_WITH_EXACT_BITS \
  --rate-hz 20 \
  --duration-ms 10000 \
  --serial-write-timeout-ms 10
```

Repeat with `--rate-hz 50` for the intended baseline, then with `75` and `100`
only as explicitly labelled motor-inert transport stress. The 75 Hz schedule
is generated from the rational `index * 1_000_000_000 / 75` nanosecond offset
rather than a rounded millisecond period. Output reports counts, loss,
reordering, queue depths, dispatch/write lateness, signed inter-write jitter,
RTT, controller service time, liveness gaps, and directional 8N1 loads as
distributions from that run only. A run with missing, duplicate, reordered,
skipped, or period-late probes fails. Even a passing 75/100 Hz stress run does
not promote that rate to a motion or MPC claim.

Runtime liveness compares host decode times, not controller emission times. Its
Heartbeat bound is therefore the advertised nominal period plus a checked
10 percent clock tolerance and 100 ms scheduling/transport margin; the
motor-inert 250 ms profile yields a 375 ms host-observation bound. This admits
bounded UART/USB/host jitter without relabelling the 250 ms on-wire claim.
Each ControllerHello gap must remain within twice the protocol's canonical
one-second Hello period. Admission separately rejects an otherwise valid
post-match Heartbeat once its host receive age exceeds the advertised watchdog
period.

The parsed serial timeout is restricted to `7..=100 ms` and one deadline covers
both the complete diagnostic-record write and its flush. After scheduling ends,
the tool waits at most `(writer queue capacity + one active write) * timeout +
100 ms` for writer completions, then allows at most another 100 ms for task
join; timeout or cancellation is a typed failed run, never timing evidence. The
JSON records all three bounds. These are host-software limits, not proof of when
USB CDC bytes reached the controller.

## Measured Nano motor-inert baseline

Exact revision `35adc901e50d0ccb893c66582238bea438e86f97` was
built natively on the Linux aarch64 Nano. The release qualifier binary had
SHA-256
`eef7f4feb7ae2ec67e4a6ad067b61b12fefef92546043e03dcc45332dd3485c5`.
The checked-in schema-3 verifier accepted two separate runs under the
unchanged Fable OAK/head/eye workload:

| Rate | Planned/completed/unique | Integrity defects | Maximum RTT | Maximum controller service | Maximum RX/TX queue | Maximum host Heartbeat gap |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 20 Hz | 200/200/200 | none | 17.939749 ms | 1 ms | 0/45 bytes | 255.004196 ms |
| 50 Hz | 500/500/500 | none | 17.937019 ms | 1 ms | 0/45 bytes | 258.822044 ms |

Here “none” means zero missing, duplicate, reordered, scheduler-skipped,
in-flight-skipped, writer-queue-skipped, and period-late probes in those exact
runs. It does not mean such failures are impossible. The evidence directories
are:

```text
/home/makerspace/kiko-hardware-evidence/20260724T211051IST-35adc90-schema3-20hz-fable-load
/home/makerspace/kiko-hardware-evidence/20260724T211135IST-35adc90-schema3-50hz-fable-load
```

Both runs admitted build `131074`, fingerprint `KIKO-NO-ACT-V1!!`,
capability bits `319`, zero maximum PWM, disabled output, and unverified
physical-stop semantics. Each run used one fresh challenge and received an
idle-safe Heartbeat after its final write. Motor power was operator-reported,
not independently instrumented, to remain disconnected. No control session or
PWM was sent.

These are reproducible measurements of the diagnostic request/report path at
20 and 50 Hz, not evidence of a performance improvement. They do not measure
the motion-capable `BeginSession`/`ApplyPwm`/`AppliedResult` path, physical
output, stopping, or end-to-end MPC behavior. Exact hashes and the full
freshness chronology are recorded in
`docs/nano-integration-acceptance-2026-07-24.md`.

## Evidence still missing

No checked-in result measures the following, so none is a current claim:

- worst-case execution time of decode, controller progression, motor MMIO, TX
  admission, or the USART ISR on the STM32F446;
- maximum interrupt masking, USART IRQ latency, or independent-watchdog margin;
- ST-Link USB CDC packetization semantics or exactly where the observed
  buffering occurred;
- Linux serial-driver write/flush completion semantics;
- 75/100 Hz motor-inert stress distributions or a cold-boot repeat of the
  measured 20/50 Hz baseline;
- cable disconnect, partial-record, controller reset, host death, TX saturation,
  stale heartbeat, and delayed-ACK fault-injection results on hardware; or
- PWM-to-wheel velocity, motor current, traction, stopping time, or stopping
  distance.

Hardware qualification must use an immutable firmware/config identity, start
motor-inert or with wheels physically removed, record the required 20/50 Hz
baseline and any explicitly selected stress rates, then exercise each listed
fault while verifying the outputs remain electrically safe. Motion/MPC
qualification remains a later wheel-attached calibration lane.
