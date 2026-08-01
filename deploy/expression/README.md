# Attended full-expression runtime

This directory is the canonical, boot-launched copy of the full OAK/KEP2/STS
expression owner proven on the Orin Nano. It replaces the misleading split in
which the same runtime was launched with `--no-head` beside a natural-only head
owner: that split could animate the eyes but could never perform face-follow or
four-axis character motion.

`kiko_face_follow.py` owns exactly one OAK RGB session, one KEP2 eye session,
and one STS head serial session. `config.json` binds the observed natural pose,
the operator-confirmed camera-to-head translation (`0.25 m` above and `0.20 m`
behind), mounting signs, and calibrated travel. The startup guardian latches
any process exit; it never blindly re-engages a possibly displaced neck.

The source was copied byte-for-byte from the live
`/home/makerspace/kiko-follow/kiko_face_follow.py` on 2026-08-01, then changed
only to use its adjacent canonical config, state the retained-torque shutdown
semantics accurately, and separate high temperature confirmations by 100 ms
using the bus evidence recorded in `docs/head-compliant-hold-2026-08-01.md`.

The live owner also runs the same conservative encoder-domain compliant-hold
policy qualified by the typed Rust controller: one second of settled motion
captures a bounded four-axis gravity/tracking bias and arms touch detection,
three direction-consistent residual samples admit contact, the
head follows 35% of bounded displacement, waits 600 ms after release, and
returns with a 2.4 s minimum-jerk trajectory. The pure planner is isolated in
`compliant_head.py`, strictly parses its complete policy once, binds the policy
to the installed torque limits, ignores unqualified load/current units, and is
covered by deterministic transition and fault tests. Serial I/O remains in the
single full-expression process.

This attended runtime does not own MPC, SLAM, base motion, or STM32. Those
capabilities must enter through the single production Nano agent; they must
not be started beside this owner.
