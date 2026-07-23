# Attended Nano base commissioning

## Scope

`kiko-nano-base-commission` is a one-shot, wheel-on system-identification
process. It is the only executable in the commissioning feature and is
separate from production operation and wheels-off qualification.

The process may apply only the fixed calibration schedule admitted by the
policy. It cannot accept joystick, manual, MPC, map-click, planner, agent, or
network motion commands. A successful run publishes a
`proposed_unapproved` plant and evidence; it does not activate that plant or
grant any subsequent motion authority.

No Nano, OAK, STM32, wheel, emergency-cut, or physical-motion result is implied
by host compilation or tests. Physical acceptance exists only after an
attended run produces the exact artifacts and the operator independently
records the required hardware evidence.

## Boundary and ownership model

Static admission parses each weak boundary exactly once into typed values and
binds every input by deployment-relative path, size ceiling, and SHA-256:

- the commissioning policy and controller profile;
- a controller-server V3 contract;
- a device-inventory V3 manifest;
- the exact calibration artifact;
- canonical Nano-agent launch V2 and its policy, ORT models, SuperPoint model,
  live graph, and calibration bindings.

The V3 server contract and V3 inventory must independently identify build
`139265`, fingerprint `KIKO-WHEELON-CM1`, a 20% cap,
`attended_wheel_on_commissioning`, and `unverified` physical-stop semantics.
Production V1, wheels-off V2, cross-class paths, aliased assets, mismatched
hashes, and identity disagreement reject before controller access.

The live adapter then:

1. retains the exact model bytes and pins the ORT runtime allocations;
2. starts the manifest-bound head/eye expression worker and verifies natural
   head hold plus eye ownership;
3. opens the one exact OAK MXID and canonical graph in the same process;
4. validates USB/stereo/calibration evidence;
5. feeds borrowed OAK RGB frames to the expression engine;
6. derives body-frame FLU forward/lateral velocity from the visual transform
   and calibrated IMU yaw rate from the same clock epoch;
7. proves at least one successful RGB expression frame before readiness.

There is no external OAK, velocity, IMU, or RGB injection seam. The process
starts the sole in-process STM32 V3 owner only after four fresh attended claims
are parsed from its controlling terminal. The owner begins at verified exact
zero and every accepted sample refreshes the bounded command lease.

The head center configured for gaze geometry is 0.25 m above and 0.20 m behind
the OAK, with neutral head gaze parallel to the OAK optical axis. That geometry
belongs in the exact canonical agent policy bound by the nested launch; this
commissioning lane does not silently substitute geometry.

## Physical gate

All four physical claims are required together:

- both wheels are attached;
- the complete motion envelope is clear;
- an operator is attending continuously;
- an independent power cut has already been tested and is immediately
  reachable.

There are no physical-claim flags or environment aliases. After OAK,
natural-head, eye, and RGB-expression readiness, the process opens its fixed
controlling terminal (`/dev/tty`), proves it is a TTY, discards pending input
before every prompt, and generates a new 128-bit challenge for each exact
response. Each response is limited to 96 bytes and 15 seconds. The private,
non-cloneable confirmation is bound to the exact session, launch/config/model
digests, admitted stream identities, and clock epoch. Its issue timestamp is
captured inside the ceremony; it must be consumed by that same prepared
session within five seconds and never crosses the public API. This makes
redirected stdin, prebuffered text, copied old commands, reusable confirmation
tokens, and unattended flag-based invocation invalid; it cannot turn an
operator claim into a sensor measurement. If any claim becomes false, use the
independent cut and interrupt the process. Do not use the command unattended,
on a bench with free-spinning wheels, or as a production startup service.
This rejects passive and reusable replay; it cannot distinguish a human from
an active same-user program controlling the process's PTY. Account/session
control, the attended procedure, direct observation, and the independent
physical cut remain explicit trust boundaries.

Before launch, coordinate release of any Fable or other OAK/head/eye owner.
Do not kill unrelated processes. The previous head owner must release serial
ownership while preserving torque at the natural pose; the canonical
commissioning worker verifies and retains that hold before opening OAK. There
must also be no competing STM32 serial or controller-UDP owner.

The exact provisioning and command are in
`configs/nano-base-commissioning-template/README.md`.

## Motion and timing contract

The checked template uses a 15% identification schedule under an absolute 20%
cap. One cycle contains symmetric forward, symmetric reverse, positive-yaw
spin, and negative-yaw spin. Four cycles are run, with a one-second exact-zero
dwell before, between, and after two-second excitations.

This is a bounded calibration schedule, not general drive control. It rejects:

- PWM outside the profile or any applied/requested mismatch;
- sequence regression, a missing lease refresh, or an excessive sequence gap;
- sample periods outside the fit interval;
- cross-sensor skew or stale controller/visual/IMU evidence;
- motion above the stationary gate while exact zero is required;
- non-finite, out-of-frame, out-of-bound, or out-of-order observations;
- interruption, observer/source failure, duration/sample limits, and journal
  or publication failure.

The template's effective sample interval is 40–100 ms. A 100 ms maximum sample
gap plus a 30 ms applied-ack timeout is strictly below the 250 ms STM32 command
lease. Command sequence is refreshed on every sample even when requested PWM
is unchanged; fit segmentation uses actual applied PWM changes, so lease
refreshes do not fabricate plant-input transitions. At the slowest admitted
100 ms cadence, the deterministic schedule completes in 523 samples, within
the configured 500–3000 sample budget.

Wheel velocity reconstruction uses:

```text
v_left  = v_forward - 0.5 * wheelbase * yaw_rate
v_right = v_forward + 0.5 * wheelbase * yaw_rate
```

with forward velocity in metres per second, yaw rate in radians per second,
wheelbase in metres, and FLU body coordinates. Signed wheel gain is learned;
negative polarity is valid evidence when the policy sets
`require_positive_velocity_gain` to `false`.

## NDJSON event contract

Standard output contains one JSON object per line. Diagnostics are written to
standard error, while attended prompts use `/dev/tty` directly. This NDJSON is
an evidence/tooling interface; the production web console does not currently
launch or consume this commissioning process. Expected high-level events are:

1. `starting`
2. `static_admission_ready`
3. `oak_live_ready`
4. `accessory_live_ready`
5. `attestation_consumed`
6. `controller_owned_at_exact_zero`
7. repeated `progress` and `sample_ready`
8. exactly one terminal `completed` or `failed`

`progress` exposes the state, step index, sample count, requested PWM, last
applied sequence, exact-zero state, and applied receipt. A `completed` event
lists the immutable dataset, proposed plant, proposal evidence, journal digest
and counts, and lateral-validity evidence. Its manual, MPC, mapping, and
activation fields remain false.

Absence of `completed`, an output encoding/write failure, a cleanup failure,
or a nonzero process exit invalidates the attempt even if artifact files are
present.

## Stop and cleanup semantics

Every handled error, NDJSON output failure after live/controller ownership,
terminal source event, SIGINT, SIGTERM, or SIGHUP path after controller
ownership, session fault, and proposal-publication failure closes live
resources and, once the controller is owned, issues an explicit emergency-zero
request and records whether exact zero was verified before the bounded owner
shutdown. Successful completion also ends at verified exact zero before owner
shutdown. SIGKILL and host/power loss remain outside process cleanup; the
firmware lease and independent physical cut remain required backstops.

The object destructor contains only a best-effort fallback. It cannot prove
completion and must not be used as stop evidence. OAK closes before accessory
release. Eye ownership must release cleanly; the head serial handoff must
preserve the verified natural-position torque rather than disabling it.

If controller communication is unavailable or exact zero cannot be verified,
use the independent physical power cut. Software stop semantics are
deliberately declared `unverified` until physical testing proves otherwise.

## Outputs and review

Each session is create-new and append-only under the private state root:

- attended attestation;
- NDJSON evidence journal and its terminal digest/counts;
- identification dataset;
- proposed plant;
- proposal evidence tying policy, launch, inventory, calibration, controller,
  OAK, fit, lateral bound, and journal together.

The state root and each session directory must be owned by the current user
with mode `0700`. Admission retains their open directory descriptors and
records device/inode identity. Journal, attestation, temporary artifact,
rename, duplicate verification, cleanup, and directory-sync operations are
descriptor-relative with no symlink following. If another same-user process
renames or replaces an admitted root or session path, publication fails with a
typed binding-change error and the controller is explicitly stopped; it never
redirects output into the replacement.

The fitter uses an exact zero-order-hold first-order model, stable
`expm1` evaluation, compensated sums, deterministic time-constant search,
conditioning/sensitivity gates, command-segment holdout, residual limits, and
independent lateral holdout. These checks can reject a run; they cannot prove
that an accepted proposal is safe for MPC.

Before activation, review the complete journal and artifacts, repeat runs for
consistency, verify polarity and units, compare holdout residuals, validate
support/envelopes on the actual surface and payload, and separately qualify
the production controller/firmware path. Activation belongs to a later,
explicit admission process.
