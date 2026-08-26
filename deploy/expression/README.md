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

The initial source was copied byte-for-byte from the live
`/home/makerspace/kiko-follow/kiko_face_follow.py` on 2026-08-01. It has since
been intentionally hardened as the executable behavior specification: adjacent
configuration, truthful retained-torque shutdown, incident-derived thermal and
compliance handling, organic motion, pet choreography, and named timing facts
are covered by the tests and the dated unification ledger. The current file is
therefore a traceable derivative, not a claim of byte identity with that field
snapshot.

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

`fable_behavior_trace.py` is the deterministic cross-language qualification
boundary. It hashes the exact Python behavior sources, derives the complete
24-act vocabulary directly from the scheduled and touch-reaction libraries,
selects both RNG boundaries for every act duration, proves neutral act
endpoints, and records the shared mode and pet-reaction sequences in
`fixtures/fable-behavior-trace-v1.json`. `--check` rejects a stale fixture.
The Rust character-owner tests strictly parse and replay that artifact, reject
unknown or duplicate fields, compare all shared timing and eligibility facts,
and prove that every Fable act channel remains present in the richer four-axis
Rust act. This is semantic software parity: it deliberately does not equate
Python floats with Rust fixed-point samples, different RNG streams, encoder
ticks with normalized character space, or host output with physical motion.

This preserved owner never owned MPC, SLAM, base motion, or STM32. It must not
be launched for production or beside the single Nano agent. Its 85 tests
remain in CI until the Rust behavior-trace qualification has attended physical
parity evidence and this laboratory can be retired deliberately.
