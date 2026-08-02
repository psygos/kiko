# Unification plan — merging the field-proven expression engine into `kiko-slam nano-agent`

Date: 2026-08-03. Author: Fable (attended session with operator).
Status: proposal for the single explicit handoff demanded by
`docs/nano-integrated-readiness-2026-07-31.md:182-191` ("running both is not
integration").

## 1. The situation, stated honestly

The recorded target (`docs/nano-agent-architecture.md`,
`docs/fable-integration-audit-2026-07-23.md:136-155`) is one production
process owning STM32 + OAK + head + eyes + SLAM + occupancy + navigation,
with typed intentions, one owner per physical stream, and the base disarmed
at start. That architecture exists and runs. But its head is natural-hold
only: the gaze/compliant path is complete, reviewed, and **dead code** —
`with_evidence_bound_physical_head_gaze` has zero call sites, and
`nano_bootstrap.rs:414-451` hardcodes `physical_head_gaze: None`.

Meanwhile the quarantined-then-vendored Python engine (`deploy/expression/`)
is the live accessory owner and, as of 2026-08-02/03, embodies a generation
of behavior the Rust side lacks, all proven in live incidents:

- streak-based thermal supervision (plausibility ceiling, consecutive-hot
  abort, unreadable-channel abort, zero sleeps in the control path);
- the YIELDING stillness rule (static-residual release as the *primary* pet
  exit) and stillness-gated recontact — both fixes for live freeze
  incidents (the 20-minute trance; the sag>entry knead-forever cycle);
- the full pet choreography (RESTING with rest-pose arrival–measured pause,
  per-episode latched comfort roll tilt, yield dwell backstop);
- backdrivable yield torque with bench-measured per-joint floors
  (2026-08-02 staircase: bow holds statically at 300 permille / 7 ticks
  drift, curl at 200 / 4 ticks, at natural — this is the "separate
  evidence" the natural-hold doctrine was waiting for);
- admission recovery (pose beyond window but inside reviewed envelope →
  zero-jump engage + bounded slew home; kills the interrupted-park →
  refused-admission brick loop);
- the rest envelope (true natural rest when alone, instant wake);
- jerk-bounded organic motion with clamped-dt gap handling;
- character dynamics (dynamic bow/curl pitch share, turn-dip weight shift,
  living motion in Idle/Tracking, sigh/bow_bob/bow-greet);
- loop-liveness heartbeat + guardian v2 (the 21-hour futex-hang class —
  currently UNGUARDED on the Rust side: `Restart=no`, no `WatchdogSec`);
- the pet-interaction field log (`~/kiko-pet-sessions.jsonl`) — the
  regression corpus for all future compliance tuning.

The port is therefore a **merge-back**: Python's field-proven deltas into
the architecturally stricter Rust controllers, followed by one switchover.

## 2. Phase 0 — freeze the specification (immediately)

1. Commit the entire 2026-08-02/03 working tree (deploy/expression, guardian,
   this doc). The behavior spec currently exists only as uncommitted files.
2. Declare `deploy/expression/` + its 71 tests the **executable
   specification** for phases 1-4. The Python tests encode the incidents;
   each Rust milestone is done when the equivalent scenario passes there.
3. Add the Python suite to CI (it is entirely outside CI today) until the
   day the lane is deleted.

## 3. Phase 1 — correctness prerequisites in Rust (no behavior change)

1. **Observation-envelope fix**, `compliant_hold.rs:1058-1070`: observations
   admit `[minimum − maximum_yield, maximum + maximum_yield]` (saturating);
   commands keep the strict envelope. Already specified by Python commit
   `7f56127`, which fixed the identical live fault.
2. **Replace the 200 ms in-step temperature confirmation**
   (`actor.rs:3956-4051`, sleep at `:3970`) with per-joint streaks carried
   across control slots: plausibility ceiling (~95 raw ≙ corruption, holds
   all counters), N-consecutive plausible-hot abort, unreadable-streak
   abort. One rework closes three gaps and removes the only blocking sleep
   in the compliant service step.
3. **Fake servo plant** — a new `AsyncByteTransport` impl that parses STS
   frames, integrates commanded goals through a gravity/torque model, and
   synthesizes telemetry. Every existing Rust fake is a byte replayer; the
   plant is the single highest-leverage test addition, prerequisite for
   testing park honesty, admission recovery, compliance under load, torque
   switching, and the rest envelope. Port the Python incident tests onto it
   first (trance, sag-23 knead, park-from-full-yaw, torque staircase).
4. **Saccade retune** (`autonomic.rs:43-44` + hardcoded amplitudes at
   `:719-720`): fixation dwell, decay to centre, gating during Tracking.
   Eyes-only; can ship independently at any time.

## 4. Phase 2 — compliance parity (the pet)

5. Add `Resting` to the compliant FSM: rest-pose offsets, comfy pause
   measured from rest-target arrival, per-episode latched comfort roll
   tilt, recontact behind the stillness gate.
6. Static-residual release as the primary yield exit + `maximum_yield_dwell`
   backstop (mirror `compliant_head.py` semantics and constants).
7. **Backdrivable torque**: redefine `admit_runtime_torque_limits`
   (`compliant_hold.rs:336-352`) from an equality invariant to a declared
   {holding profile, yield profile, per-joint floors} invariant, floors
   from the 2026-08-02 bench (300/200/150/150). Torque-limit writes on
   state transitions only; unconditional holding restore on
   park/admission/starvation paths. The doc objection at
   `docs/head-compliant-hold-2026-08-01.md:110-113` ("not implemented"
   because gravity was unmeasured) is now answered by measurement.

## 5. Phase 3 — character and motion parity

8. **Widen the character boundary** — `render_character(now, bool, …)`
   (`autonomic.rs:600`) must take bearings/residual/proximity/derate, and a
   feedback edge must carry the commanded head pose back from
   `HeadGazeController` (residual-driven eyes reverse today's dataflow at
   `mixer.rs:245-251`). This is the structural gate for everything below.
9. Living motion on the head in Idle/Tracking (today: literally none —
   `autonomic.rs:1025`), rest envelope + instant wake, dynamic bow/curl
   pitch share, turn-dip weight shift.
10. **Motion feel**: port the jerk-bounded spring axes (response_hz,
    damping, jerk/accel/velocity limits, clamped-dt gap) as a target-shaping
    prefilter ahead of `plan_joint`. Rust's bang-coast-brake planner is
    exactly the constant-speed robotic feel the operator rejected; the
    spring model is the product feel. Keep Rust's fixed-tick lateness
    semantics underneath.
11. Acts: scheduler inputs (derate/proximity/burst), sigh, bow_bob, greet
    style 3, breath/blink unification. The whimsy tier (arousal scalar,
    familiarity greeting, REM rest) ports last, after parity.

## 6. Phase 4 — lifecycle and supervision

12. Heartbeat from the accessory loop + `sd_notify`/`WatchdogSec` +
    `Restart=on-failure` with backoff in `kiko-nano-agent.service` — this
    retires the shell guardian's reason to exist and closes the frozen-loop
    class on the Rust side.
13. Admission recovery second tier (reviewed-envelope slew-home, tighter
    RETURNING divergence bound) on top of Rust's already-stronger return
    verdict machinery.
14. Behavior telemetry sink: serialize `PreparedCharacterFrame` +
    `CompliantHoldDisposition` episodes as NDJSON (reuse
    `DurableCommissioningJournal`), format-compatible with
    `kiko-pet-sessions.jsonl` so the field corpus carries over.

## 7. Phase 5 — the single explicit handoff

15. Activation evidence bundle: head-gaze policy JSON + physical review
    asset assembled from the attended record — axis-dance sign
    verification, torque staircase, calibrated geometry, live incident
    fixes. Wire `.with_evidence_bound_physical_head_gaze(...)` at
    `nano_bootstrap.rs:1016-1028`; bind the base-zero lease issuer (already
    plumbed, currently `PhysicalGazeNotConfigured`).
16. Replay qualification: drive the Rust stack with recorded frames +
    pet-session episodes; diff behavior traces against Python logs. (True
    shadow mode is impossible — one owner per device — so replay is the
    parity instrument.)
17. Switchover night (attended): remove guardian crontab entries; stop
    guardian + Python (hard-kill, torque held); start
    `kiko-nano-agent.service` with the qualified-boot drop-in; run the
    same live verification protocol as 2026-08-02 (tracking, rest,
    petting, axis dance). Rollback is the exact reverse; the Python lane
    stays in-tree as the behavior lab (never boot-launched) until two
    quiet weeks pass.

## 8. Phase 6 — what unification buys

With head/eyes/OAK inside the agent, the base-motion interlock
(`base_motion_interlock.rs`) finally binds against a real gaze lease — head
expression and base motion coordinate by construction. Then, in order:
wheel gates and plant calibration, MPC out of shadow
(`navigation/mpc.rs` is transport-free by design today), and expression
that leans into base motion instead of being suspended by it.

## 9. Ordering constraints

- Phases 1-4 run while the Python owner keeps the robot alive; zero
  downtime until step 17.
- Phase 1.3 (the plant) precedes all of phases 2-3 testing.
- Phase 3.8 precedes 3.9-3.11. Phase 2.7 requires 1.2 (streaks) only for
  its starvation-path restore semantics.
- The operator sign-offs already pending (stiffness thresholds, soft-torque
  feel, tilt/sink amplitudes) become the recorded values in the Phase 5
  policy bundle — tune them in Python first, where iteration is minutes.
