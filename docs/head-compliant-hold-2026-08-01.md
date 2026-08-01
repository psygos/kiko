# Four-axis compliant hold

## Outcome

Kiko now has a source-complete, disabled-until-reviewed compliant-hold path in
the canonical head owner. It is designed for the interaction “a person gently
displaces or pets the held head; Kiko yields with the contact and then returns
gracefully to the gaze/expression pose that was active when contact began.”

This is not a force controller. The deployed STS telemetry contract exposes
`load_raw` and `current_raw`, but it does not establish their sign, scale,
linearity, or physical units. The implementation retains those fields for
diagnostics and later characterization but never calls them torque or force and
never uses them to admit contact.

## Ownership and arbitration

There remains one serial owner and one physical-command path:

1. The actor reads one fresh, identity-ordered four-servo telemetry set.
2. The compliant controller prepares a generation-bound candidate.
3. If a target changes, the same actor writes all four goals in canonical
   bow/curl/yaw/roll order and reads every goal register back exactly.
4. Only a complete verified application commits the candidate. A partial or
   uncertain application latches an absorbing compliance fault.
5. From the first contact-candidate sample through completed recovery, gaze and
   character head proposals are lower priority and cannot overwrite the
   compliant target.

Passive compliance synchronizes to an already verified gaze target without
duplicating its four writes or readbacks. Compliance and gaze are required to
use the same control period. An active compliant state retains arbitration
between ticks without performing another telemetry pass, so camera frame rate
cannot multiply head-bus traffic.

The four telemetry requests share one exclusive transaction deadline. Every
write attempt and response read is capped by the budget remaining from the
start of the set; four slow devices therefore cannot each consume a complete
per-request timeout. Completion exactly at the deadline is rejected. The
aggregate timeout must fit inside the gaze controller's admitted tick
lateness, while the sample-span bound remains a separate constraint on the
coherence of successfully received values. Deadline, clock-regression, I/O,
and completed-request evidence remain distinguishable.

## State machine

The explicit states are:

- `FollowingExpression`: the latest verified gaze/character target remains in
  control. Contact admission is disabled while that target has commanded
  velocity or has just changed. It arms only after every axis is stationary
  and inside its release band for the complete configured arming dwell; normal
  servo settling therefore cannot be relabelled as touch.
- `ConfirmingContact`: at least one encoder error crossed its joint's entry
  threshold with a stable direction. A configurable number of consecutive
  samples is required.
- `Yielding`: the target follows a configured fraction of observed
  displacement from the captured expression pose. Per-joint hard envelopes,
  maximum yield, and maximum command step all apply independently and every
  limiting event is reported.
- `ReleaseDwell`: all four actual-to-command errors are inside their smaller
  release bands and all moving flags are false for the complete dwell.
- `Recovering`: the target follows a quintic minimum-jerk curve back to the
  captured expression pose. Every emitted target is also command-step bounded.
- `FaultHeld`: observation clock regression, an out-of-envelope observation,
  an implausible sample discontinuity, generation exhaustion, timestamp
  overflow, or uncertain hardware application is absorbing.

If a hand remains during recovery, return motion recreates an encoder error.
After the same consecutive-sample admission, Kiko yields again instead of
continuing to push against the hand. This slow return probe provides release
semantics without inventing a force measurement.

## Numerical behavior

For an observed displacement `d` from the captured return pose, the compliant
desired offset is:

```text
round_nearest(d * follow_permille / 1000)
```

It is then bounded by the joint's maximum yield, its calibrated inclusive
encoder envelope, and its maximum per-control-tick command step. Signed
division is symmetric around zero.

Recovery uses the standard minimum-jerk progress polynomial
`6u^5 - 15u^4 + 10u^3`. It is evaluated in integer millionths with exact zero
and one endpoints; no floating-point state or incremental integration can
accumulate drift. If lateness makes a nominal curve sample exceed a command
step, the step bound remains authoritative and recovery truthfully continues
past the nominal curve duration until the exact endpoint is reached.

Every observed encoder value must remain inside its calibrated envelope.
Consecutive sample displacement must remain under a separately typed maximum,
which is required to be at least the controller's own maximum command step.
Four response timestamps must be monotonic, their set span must be bounded, and
the oldest sample must remain strictly younger than the configured TTL.

## Torque semantics

The compliant declaration contains exact four-joint torque limits and physical
admission requires them to equal the limits in the inseparable reviewed-return
runtime. This prevents a policy reviewed for one holding stiffness from being
silently paired with another.

The attended Nano values are `[650,550,400,400]` permille in
bow/curl/yaw/roll order. They are existing runtime values, not evidence that
they are the lowest stable values. “Just enough torque” is assembly-, pose-,
temperature-, and supply-dependent and cannot be derived from source code.
Bow and curl carry gravity load, so unattended automatic torque reduction is
not implemented. Commissioning must establish each final value with the head
physically supported and preserve the resulting evidence.

## Policy and observability

`head-gaze-policy-v1` accepts an optional strict
`compliant_hold_declaration`. Unknown and duplicate fields fail. It carries:

- exact holding torque limits;
- a control period equal to gaze;
- an aggregate four-request transaction deadline plus four-sample span and
  freshness bounds;
- stationary in-band arming dwell, contact sample count, release dwell,
  recovery curve duration, and follow gain; and
- per-joint entry/release hysteresis, maximum yield, maximum command step, and
  maximum observed step.

The checked-in qualification template retains the already established
four-axis pose calibration and torque values, but leaves every unmeasured
touch dynamic as an `UNVALIDATED` sentinel. It remains `proposal_only` and
cannot activate the head.

Each committed receipt exposes state, disposition, exact command target,
observation time, positions, moving flags, raw load/current diagnostics, and
whether envelope or command-step limiting occurred. Those values are
allocation-free controller evidence suitable for a later Rerun/log adapter;
this change does not claim that a Rerun consumer has already been deployed.

## Physical verification gate

Before enabling this mode on the robot:

1. Support the head mechanically; keep the base stopped and wheels off.
2. Measure the complete four-servo transaction duration and observation-set
   span under the production serial owner. Select an aggregate transaction
   deadline, span bound, and TTL with evidenced margin; confirm the transaction
   deadline remains within gaze tick lateness.
3. For each axis and both directions, measure idle encoder error, mechanical
   settling duration, and ordinary gaze tracking error before selecting the
   arming dwell and entry/release hysteresis.
4. Starting from current torque limits, reduce only one supported axis at a
   time. Verify stable holding across the complete admitted envelope, supply
   range, and temperature range. Gravity axes take precedence over softness.
5. Exercise slow and faster hand displacements, release while stationary, a
   hand held during recovery, and removal near every hard envelope.
6. Verify every emitted command step, no envelope crossing, exact return pose,
   contact reacquisition, raw thermal/electrical safety admission, emergency
   stop, owner loss, and partial-write fault behavior.
7. Retain the exact policy bytes, evidence digest, operator identity, raw
   telemetry, and video. Only then change the lifecycle from `proposal_only`
   through the existing physical-review admission path.

No physical touch, minimum-torque, latency, or thermal result is claimed by the
software tests in this change.

## Attended activation surface

`kiko-head-commission --compliant-hold` is the bounded head-only lane used to
collect that missing evidence without weakening production admission. It:

- parses one strict configuration and derives all domain types before opening
  the serial endpoint;
- performs a tension-preserving takeover, returns to the reviewed natural
  pose, then services compliance continuously inside the same actor;
- has no camera, eye, STM32, base-motion, gaze, deployment, or production
  authority;
- retains the exact typed compliant failure in actor-exit evidence; and
- on SIGINT, SIGTERM, or lease expiry, releases serial ownership without a
  torque-switch write, preserving the last verified hold.

The checked-in
`configs/nano-head-compliant-commissioning-v1.json` is a conservative first
commissioning candidate. Its natural pose, four-axis envelopes, adapter
identity, and existing `[650,550,400,400]` torque values come from the retained
Kiko configuration. Its 100 ms service period, touch hysteresis, bounded yield,
35% follow gain, dwell values, and 2.4 s minimum-jerk recovery are hypotheses
selected for an attended wheels-off trial. They are not a promoted physical
review, minimum-torque evidence, or a claim that a person will perceive the
motion as soft. Physical observations must either support those exact bytes or
produce a new candidate; they must never be rewritten into evidence after the
fact.

The first Nano activation observed the already-energized bow joint settling
from 2153 to 2123 ticks after the legacy owner released its serial endpoint.
The original narrow startup window correctly refused every motion write, but
inspection then proved it was inconsistent with the retained four-axis
calibration. The legacy engine admits offsets of `[110,180,480,160]` ticks
around the same `[2174,2570,1637,3047]` natural pose. The typed takeover domain
also limits every per-axis startup window to 256 ticks, so this attended
configuration retains the complete 220-tick calibrated bow span and a
natural-centered 256-tick subset for curl, yaw, and roll. Maximum return travel
is cross-bound to 128 ticks on every axis. During the handoff the bow settled
at 2062 ticks around the legacy minimum command of 2064, a 2-tick tracking
offset inside the existing 20-tick readback tolerance. The natural-centered
bow window is therefore also 256 ticks (`2046..=2302`): it remains inside the
structural span and inside the calibrated command range plus readback
tolerance. The final observed handoff pose `[2062,2673,1632,3041]` lies inside
the exact windows. This is takeover compatibility evidence, not evidence for
touch quality or perceived softness.

The same live handoff also showed a repeatable raw-moving transient after the
actor wrote the observed goal and repeated the already-enabled torque state:
the preceding probe was stationary, but the immediate first stopped readback
reported `moving`. Startup now waits exactly one existing 100 ms head-return
control period before collecting its two required stopped readbacks. The wait
does not convert motion into success; both samples still fail closed on raw
moving, mismatch, unstable position, device status, or telemetry limits.

The retained live expression configuration was also the authority actually
holding the assembled mechanism: `[650,550,400,400]` permille, not the earlier
`[600,400,400,400]` commissioning hypothesis. At its natural goal, repeated
stationary telemetry showed gravity-axis tracking offsets of 22 and 23 ticks.
The attended config therefore uses the proven torque values and a 24-tick
startup readback tolerance. The distinct final-target, path-corridor, and
direction-regression tolerances are likewise 24 ticks, so the return controller
uses the same measured tracking-error envelope without conflating their typed
roles. This does not claim minimum sufficient torque; it only prevents a
takeover from weakening the known working hold or rejecting its measured
stationary tracking error.

Most importantly, attended takeover no longer writes the observed shaft pose
back as a new goal. Under gravity that erased the prior goal error which was
producing holding torque, and repeated failed starts could ratchet the head
downward. The actor now proves all four retained torque switches are enabled,
writes only the reviewed torque limits, verifies two stopped readbacks around
the pre-write observation, and leaves the prior goals untouched until the
bounded return controller issues its first waypoint.

For the attended Nano handoff, `deploy/kiko-accessory-commissioning-guardian.sh`
keeps two deliberately separate device owners alive. The existing expression
process runs with `--no-head` and therefore retains only OAK/eye behavior; the
typed Rust commissioning process is the sole head-serial owner. Independent
lifecycle handling keeps a camera failure from interrupting head tension and
keeps a head-control fault from taking down the eyes. The script never opens or
starts
the STM32, base, navigation, or SLAM and must not be represented as the final
production owner.

Head startup or runtime failure is latched for the lifetime of the guardian;
only a deliberate guardian restart permits another physical takeover. This is
asymmetric by design: the camera/eye process may restart independently because
it does not rewrite the head's retained goal, while blindly restarting a
tension-preserving head takeover can re-adopt a sagged pose after a partial
startup. A typed head fault therefore leaves the last torque/goal state intact
and the expression child alive instead of creating an actuation retry loop.

## Energized-temperature confirmation and live handoff

The first typed compliant run ended safely after 18.24 seconds when one
checksum-valid Bow response carried raw temperature `72` against the exclusive
energized limit `65`. Long-running evidence from the retained STS owner had
independently shown isolated raw temperature bytes `65`, `97`, and `122`
immediately followed by ordinary values around `39..41`. That is evidence for a
recurrent response-level transient; it is not evidence that every future high
sample is corrupt or that the physical thermal limit may be relaxed.

The compliant boundary now distinguishes the two cases without admitting the
suspect sample into controller state:

1. The complete four-joint response set is parsed once. Voltage, device status,
   identity, framing, ordering, span, and freshness faults still fail
   immediately.
2. An initially high energized temperature freezes all physical output for the
   transaction and does not prepare or commit a planner generation.
3. The actor waits 100 ms before each of two fresh responses for the same
   joint. Each read has its own configured transaction deadline, so the
   complete confirmation is bounded while neither read borrows time from the
   fixed inter-sample interval.
4. Any admitted temperature among the three produces typed transient evidence
   retaining all raw responses. Gaze actuation is also suppressed for that
   service transaction.
5. Three consecutive high responses produce a distinct confirmed-temperature
   fail-stop retaining all three responses. A failed confirmation read,
   nonzero device status, or voltage violation remains its own typed error and
   is never relabelled as a transient.

On 2026-08-01 the Orin Nano release binary built from local commit `f82b4d1`
(Nano-applied commit `3b912ca`) completed an attended tension-preserving
takeover from `[2163,2572,1633,3041]` ticks. It retained enabled torque, used
one bounded return waypoint toward the reviewed
`[2174,2570,1637,3047]` target, and reported raw startup temperatures
`[39,36,38,39]`. The manual typed owner then remained live for 92 seconds,
beyond the previous 18.24-second exit, and handled SIGTERM by recording
`RequestedHoldPreservingRelease` without a torque-disable write.

The updated singleton guardian subsequently started at
`2026-08-01T11:55:01+05:30`. After 77 seconds of observation its typed head
child still exclusively owned `/dev/ttyACM1`; its expression child ran with
`--no-head` and exclusively owned `/dev/ttyACM2`; `/dev/ttyACM0` remained
unowned. The OAK process reported `camera_ready usb=SUPER`, and `lsusb -t`
placed the OAK vendor interface at `5000M` beneath the Orin `10000M` root/hub
path. Eye-state logs continued through `IDLE`, `SLEEPY`, blink, perk-up, and
sweep acts while the head owner remained separate. The installed crontab has
both an `@reboot` guardian entry and a minute-level singleton recovery entry.

This proves process/serial ownership, tension-preserving startup, bounded
natural return, USB3 camera transport, expression activity, and sustained
software liveness for that interval. It does not prove physical petting feel,
minimum stable torque, thermal calibration, face-detection quality, complete
head-tracking geometry, locomotion, MPC, or SLAM.

The first guardian run then supplied stronger evidence than the manual run.
At 171.79 seconds it received three high Bow bytes `150`, `88`, and `97` over
25 ms and correctly latched off. With the head retry still latched and torque
and goal retained, 20 read-only probes at approximately 50 ms request spacing
observed the ordinary raw value `38`, one isolated `150`, ordinary `38` again
85 ms later, one admitted `50`, and then only `38`. This proves that 10 ms
confirmations can remain inside one response-corruption burst on this bus. It
does not prove a universal maximum burst length. The deployed confirmation
interval is therefore 100 ms, beyond the observed 85 ms recovery, and retains
two confirmation samples over 200 ms before a genuine three-high fail-stop.
After rebuilding, the deliberately restarted guardian and both of its exact
children remained unchanged for a 274-second observation ending at
`2026-08-01T12:06:32+05:30`. The typed head owner therefore passed the earlier
171.79-second failure point under the extended policy. Process survival does
not reveal how many transient classifications occurred because the autonomous
commissioning loop does not emit each non-fault receipt.
