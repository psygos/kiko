# kiko-expression-core

Pure expression-domain logic for Kiko. This crate deliberately performs no
camera, serial, network, clock, filesystem, logging, or actuator I/O. It is
`no_std`, allocation-free, deterministic, dependency-free, and forbids unsafe
code.

The host runtime should parse each external boundary once:

1. Timestamp and validate an RGB frame as `RgbObservation`/`RgbFrameView`.
2. Convert perception results into checked `PersonObservation` and
   `SceneObservation` values referring to that exact frame ID.
3. Convert agent commands into expiring semantic `ExpressionIntent` values.
4. Call `ReactionMixer::mix` with a monotonic timestamp.
5. Give the resulting normalized `EyeIntention` and `HeadIntention` to the
   single owners of those devices.

The mixer ignores future, stale, or frame-mismatched visual observations. It
selects one person deterministically (confidence, then known nearest distance,
then track ID), admits scene motion only when no valid person exists, and mixes
all fresh intents at the highest active priority using exact fixed-point
weights. Output validity is the earliest deadline of every contribution used.
With no usable contribution it returns an explicit neutral fallback.

`ReactionMixer::default()` locks the head at `NaturalHold`. An explicit
`HeadMotionPolicy::FollowGaze` can request bounded normalized offsets, but this
crate never contains servo IDs, ticks, angles, calibration, or hardware limits.
The head actor must map those offsets through a versioned, physically qualified
envelope and remains free to reject or replace them with safe hold.

Pixel data is borrowed by `RgbFrameView`; it is never copied. Reaction mixing
uses only checked frame metadata and compact perception observations. No serde
implementation is provided intentionally: wire/storage formats belong to the
boundary crates, where versioning and parse errors can remain explicit.
