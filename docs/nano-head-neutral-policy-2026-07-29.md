# Nano head neutral-policy evidence — 2026-07-29

> Historical assembly record. The bow servo was subsequently replaced.
> Current canonical constants are source-bound in
> `docs/kiko-head-assembly-policy-2026-08-27.md`; do not use the values below
> to render a new bundle for the installed neck.

This document records the evidence used to supersede Kiko's canonical
bow/curl/yaw/roll neutral target and startup policy. It does not alter the
historical records of earlier head returns, and it does not authorize wheel
attachment, unattended head motion, deployment, or autonomous driving.

## Operator declaration and read-only observation

The operator physically placed the head in its intended neutral orientation
and confirmed the resulting target:

```text
operator-confirmed neutral target: [2174, 2570, 1637, 3047] ticks
```

After the flawed Python owner was quiesced, five consecutive probes used the
canonical Kiko read-only head path. Every probe reported the exact same
stopped bow/curl/yaw/roll pose:

```text
observed stopped pose: [2153, 2640, 1832, 3043] ticks
samples: 5 / 5 exact
moving: false for all four joints
torque raw: 1 for all four joints
voltage raw: 118..120
temperature raw: [39, 41, 35, 35]
device status: 0 for all four joints
framing noise: 0 bytes
```

The raw torque, voltage, and temperature fields are retained exactly as
protocol observations. This record does not assign unproven physical units to
them or infer delivered torque, neck support, long-term thermal margin, or
power quality.

## Derived admission policy

The startup envelope is the component-wise union of:

- the operator-confirmed target expanded by the existing 20-tick readback
  tolerance; and
- the five-sample observed stopped pose expanded by the same 20 ticks.

The arithmetic is exact:

| joint | target ±20 | observed ±20 | union |
|---|---:|---:|---:|
| bow | `2154..2194` | `2133..2173` | `2133..2194` |
| curl | `2550..2590` | `2620..2660` | `2550..2660` |
| yaw | `1617..1657` | `1812..1852` | `1617..1852` |
| roll | `3027..3067` | `3023..3063` | `3023..3067` |

Therefore the exact bow/curl/yaw/roll startup bounds are:

```text
minimum_start_ticks: [2133, 2550, 1617, 3023]
maximum_start_ticks: [2194, 2660, 1852, 3067]
```

The maximum target distance anywhere in that admitted window is
`[41,90,215,24]` ticks. The selected exact software travel caps and their
resulting headroom are:

```text
maximum_travel_ticks: [48, 96, 224, 32]
headroom_ticks:       [ 7,  6,   9,  8]
```

These are software authorization caps for this startup policy. They are not
measurements of mechanical joint range, stops, backlash, clearance, or safe
continuous travel.

The canonical torque limits remain `[600,400,400,400]` permille. The existing
motion and observation policy remains unchanged:

| field | retained value |
|---|---:|
| response timeout | 100 ms |
| write timeout | 100 ms |
| arming freshness | 250 ms |
| write attempts | 2 |
| framing-noise budget | 32 bytes |
| redundant-read tolerance | 10 ticks |
| readback tolerance | 20 ticks |
| final-target tolerance | 20 ticks |
| path-corridor tolerance | 20 ticks |
| direction-regression tolerance | 20 ticks |
| goal speed | 50 ticks/s |

The host parser and offline renderer require the exact target, startup bounds,
travel caps, and torque limits. A policy cannot silently widen any of those
four arrays.

## Claim boundary and remaining physical gate

This evidence establishes:

- an operator-confirmed neutral target;
- five repeatable canonical read-only observations at one stopped pose;
- zero observed protocol status or framing faults in those five probes; and
- the arithmetic provenance of the exact startup and travel policy.

It does not establish:

- that Kiko's canonical actor has moved from every admitted start to the new
  target;
- that the new target was reached, held, or released by the canonical actor;
- mechanical clearance, collision freedom, backlash, stall behavior, torque
  sufficiency, thermal endurance, or power-loss behavior;
- camera-to-head gaze calibration or expression-tracking accuracy; or
- safe operation with wheels attached.

Before wheel attachment, the attended wheels-off gate must still exercise the
canonical return-and-hold actor from an admitted start, retain exact telemetry
and terminal ownership evidence, and verify the operator can cut physical
power. Any result outside this document's exact bounds is new evidence, not a
reason to widen the policy implicitly.

Older dated evidence remains truthful for the transactions it recorded. Where
an older document calls `[2155,2545,2943,2876]` the current target, this
2026-07-29 record supersedes only that present-policy statement; it does not
rewrite the earlier observation.
