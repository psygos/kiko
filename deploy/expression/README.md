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

This attended runtime provides the visible expression behavior now. It does
not implement the typed Rust compliant-petting controller, MPC, SLAM, base
motion, or STM32 ownership. Those capabilities must enter through the single
production Nano agent; they must not be started beside this owner.
