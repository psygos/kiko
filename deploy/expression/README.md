# Frozen field-expression behavior laboratory

This directory preserves Fable's attended OAK/KEP2/STS expression owner and
its incident tests as an executable specification. It is neither canonical
production ownership nor a booted companion service. The single Rust Nano
agent owns production camera, eye, and head lifecycles; starting this Python
owner beside it would violate the one-owner device contract.

When run historically, `kiko_face_follow.py` owned exactly one OAK RGB session,
one KEP2 eye session, and one STS head serial session. `config.json` retains the
observed natural pose, the operator-confirmed camera-to-head translation
(`0.25 m` above and `0.20 m` behind), mounting signs, and calibrated travel as
field evidence. These values are not self-authenticating production approval.

The source was copied byte-for-byte from the live
`/home/makerspace/kiko-follow/kiko_face_follow.py` on 2026-08-01, then changed
only to use its adjacent canonical config, state the retained-torque shutdown
semantics accurately, and separate high temperature confirmations by 100 ms
using the bus evidence recorded in `docs/head-compliant-hold-2026-08-01.md`.

The retained field owner implemented the same conservative encoder-domain
compliant-hold policy qualified by the typed Rust controller: one second of
settled motion
captures a bounded four-axis gravity/tracking bias and arms touch detection,
three direction-consistent residual samples admit contact, the
head follows 50% of bounded displacement, waits 600 ms after release, and
returns with a 2.4 s minimum-jerk trajectory. The pure planner is isolated in
`compliant_head.py`, strictly parses its complete policy once, binds the policy
to the installed torque limits, ignores unqualified load/current units, and is
covered by deterministic transition and fault tests. Serial I/O remains in the
single full-expression process.

This preserved owner never owned MPC, SLAM, base motion, or STM32. It must not
be launched for production or beside the single Nano agent. Its tests remain
in CI until the Rust behavior-trace qualification has attended physical parity
evidence and this laboratory can be retired deliberately.
