# Kiko base commissioning core

This crate is the deterministic, transport-free core for commissioning Kiko's
encoderless differential-drive base. It does not open a serial port, command a
device, acquire a camera, authorize motion, or decide that the physical area is
safe. A Nano-side adapter may perform those jobs only after its normal
supervisor and collision-safety gates admit commissioning authority.

## Evidence boundary

The identification input is an already time-aligned sequence containing:

- the exact canonical left/right PWM reported as **applied** by one controller
  session;
- visual forward velocity in metres per second; and
- calibrated IMU yaw rate in radians per second, with a calibration identity.

The DTO also carries robot, controller-session, visual-source, IMU-calibration,
wheelbase-calibration, and dataset-content identities. Weak strings, PWM
integers, timestamps, sample counts, floating-point values, and configuration
bounds are parsed once before fitting. Timestamps must increase, controller
sequences may not regress, and a repeated applied sequence may not change PWM.
Only intervals whose two endpoints retain the same applied sequence are used;
an unobserved command transition is never guessed to have occupied a complete
sample interval.

Boundary-sized sample and transition storage is reserved fallibly. Capacity or
allocator failure is returned as typed source evidence rather than relying on a
panicking `Vec::with_capacity` path. The input DTO's allocation occurs before
this crate receives it and therefore remains the caller/deserializer's boundary.

IMU data alone cannot identify translation. For wheelbase `b`, forward velocity
`v`, and yaw rate `omega`, the observations determine

```text
left_velocity  = v - b * omega / 2
right_velocity = v + b * omega / 2
```

Yaw rate supplies only the difference between wheel velocities. Without the
visual `v`, adding the same unknown velocity to both wheels leaves IMU yaw
unchanged. The API therefore has no IMU-only fit path.

## Model and gates

Each wheel is fit to the same exact zero-order-hold first-order convention used
by Kiko's MPC:

```text
v1 = exp(-dt / tau) * v0
   + gain * (1 - exp(-dt / tau)) * applied_pwm
```

The response term uses `-exp_m1(-dt/tau)` to avoid cancellation. For every
candidate log time constant, gain is solved with compensated normal-equation
sums. A fixed log grid and fixed-iteration golden-section refinement make the
result deterministic. The fitter rejects, with typed evidence, datasets that
miss any configured requirement for:

- sample count, sample-period bounds, or maximum period ratio;
- training and held-out transition counts (complete applied-command holds are
  partitioned together so adjacent endpoints do not leak into training);
- symmetric, spin, zero, positive, negative, and command-change coverage;
- gain range/sign or time-constant search-bound margin;
- log-time-constant sensitivity or scaled normal-matrix conditioning; or
- held-out wheel, forward, yaw-rate, or maximum absolute residuals.

The returned support envelope is limited to applied PWM observed in usable
intervals and wheel velocities observed or predicted at those supported PWM
extrema. Zero velocity is not inserted into the envelope unless observations
or a supported model equilibrium actually include it. The dataset does not
independently observe lateral velocity, so the result explicitly reports
`LateralVelocityEvidence::Unidentified`. It must not be converted into an
active MPC plant by inventing a lateral-slip bound; that requires separate
physical evidence.

## Safe excitation state machine

`CommissioningController` emits only four bounded canonical patterns per cycle:
symmetric forward, symmetric reverse, positive-yaw spin, and negative-yaw spin.
It requires a fresh applied-zero result, visual/IMU motion inside configured
stationarity bounds, and a configured zero dwell before each step. It waits for
a newer controller sequence proving the step was applied, times excitation from
that observation, then requires a newer applied-zero result and another zero
dwell.

Emitted `CanonicalPwmCommand` values are deliberately a different type from
input `AppliedPwm` evidence: a requested command never masquerades as proof that
the controller applied it.

Cancellation, monotonic-clock regression, future or stale controller/visual/IMU
evidence, controller-sequence regression, unexpected applied PWM, application
timeouts, total-duration exhaustion, and step-limit exhaustion latch an aborted
state whose only output is required zero. Completion also continues to require
zero. The state machine cannot prove that the robot is unobstructed or that a
zero PWM result means physically stationary wheels; those remain external
safety obligations.

## What tests prove

The tests recover multiple synthetic unequal-wheel plants, assert deterministic
bit-for-bit output, exercise sample-period/excitation/conditioning/holdout
gates, demonstrate IMU translation non-identifiability, and cover fail-closed
state transitions and bounded emitted patterns. They are software evidence
only. No physical STM32, Nano, motor, wheel-slip, timing, or performance claim
is made by this crate or its tests.
