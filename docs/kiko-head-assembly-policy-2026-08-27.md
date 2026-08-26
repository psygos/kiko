# Current Kiko head-assembly policy — 2026-08-27

This document binds the canonical host policy to the field configuration for
the bow-servo replacement recorded on 2026-08-06. It supersedes the physical
constants in `docs/nano-head-neutral-policy-2026-07-29.md`; that older document
remains historical evidence for the previous assembly.

This is a source and provenance correction. It does not claim that the current
Rust binary, rendered bundle, cold-boot sequence, compliance feel, or shutdown
path has been requalified physically.

## Authoritative retained source

- file: `deploy/expression/config.json`;
- source commit: `3a3051d8ca60717cfcffa2c65a693069b8ee9230`;
- source date: `2026-08-06T17:35:48+05:30`;
- SHA-256:
  `46d69519425caba5ace1920d39ff8a07101bf86b79eacb1cdeb53f1dd8957a56`;
- provenance identifier:
  `fable-config-sha256-46d69519425caba5ace1920d39ff8a07101bf86b79eacb1cdeb53f1dd8957a56`.

The retained field record says the old bow servo was replaced, the operator
hand-placed the standing balance, and all four joints engaged with zero jump
and reported zero load in that attended observation. Curl is close to its
encoder wrap, so its expressive range is intentionally 150 ticks until the
horn is mechanically reseated and requalified.

## Exact current policy

All arrays are bow, curl, yaw, and roll order.

| Quantity | Current value | Meaning |
| --- | --- | --- |
| natural target | `[1505,3937,1551,3018]` ticks | field standing balance |
| return start minimum | `[1377,3809,1423,2890]` ticks | software admission window |
| return start maximum | `[1633,4065,1679,3146]` ticks | software admission window |
| maximum return travel | `[128,128,128,128]` ticks | exact start-to-target cap |
| expressive offset limits | `[110,150,480,160]` ticks | command-envelope half widths |
| expressive minimum | `[1395,3787,1071,2858]` ticks | inclusive command bounds |
| expressive maximum | `[1615,4087,2031,3178]` ticks | inclusive command bounds |
| holding torque | `[650,550,400,400]` permille | field holding profile |
| yield torque | `[450,350,220,250]` permille | field touch-yield profile |
| position tolerances | `24` ticks | readback, final, corridor, regression |
| goal speed | `50` ticks/s | current return policy |

The return start window and expressive command envelope are different domain
objects. The former determines whether a bounded return may begin; the latter
limits every expression/gaze target. Neither is a mechanical joint limit.

## Coordinate and pitch derivation

The Fable configuration describes its vertical bearing as pitch-up. Its
current source values are:

```text
pitch scale = 620 ticks/radian
bow share = 0.35
curl share = 0.65
curl sign = -1
```

For Fable pitch-up demand `P`, the source computes:

```text
bow  = -curl_sign * bow_share  * 620 * P = +217 * P
curl =  curl_sign * curl_share * 620 * P = -403 * P
```

The canonical gaze coordinate is pitch-down, `D = -P`. Therefore the typed
mapping is exactly:

```text
pitch-down -> bow  -217 ticks/radian
pitch-down -> curl +403 ticks/radian
yaw-right  -> yaw -1050 ticks/radian
```

The baseline bow share derived from the canonical coefficient magnitudes is
`217 / (217 + 403) = 0.35`. Dynamic recruitment then grows that share toward
600 permille and reaches it at 140 combined absolute pitch-demand ticks. No
independent or stale share is stored in the runtime.

## Drift prevention

- `kiko-slam::navigation` exports the one current natural-return constant set.
- `kiko-nano-bundle-renderer` imports those constants instead of duplicating
  their numeric values.
- both production and wheels-off head-gaze templates retain the full source
  digest in their provenance identifier;
- `qualification_head_gaze_policy_template` derives natural pose, envelopes,
  pitch/yaw coefficients, character ranges, and torque from the checked-in
  Fable JSON and rejects drift;
- the current compliant commissioning document independently retains the same
  natural pose, windows, tolerances, and holding torque.

## Remaining physical gate

Before installing a canonical single-owner bundle, a fresh attended wheels-off
session must materialize its exact policy and review pair and demonstrate:

1. present-pose admission and bounded return to the current target;
2. stable four-joint hold and complete torque/readback health;
3. gaze signs, expressive envelopes, dynamic bow recruitment, and turn dip;
4. pet yield, rest, recontact, recovery, and torque restoration;
5. eye/head coordination, startup recovery, goodnight, watchdog, and every
   stop/fault path under PID 1.

Host tests prove the source relationship and parser behavior only.
