# Unification plan — merging the field-proven expression engine into `kiko-slam nano-agent`

Date: 2026-08-03. Author: Fable (attended session with operator).
Status: active merge-back; the dated ledger below is authoritative and the
remaining sections retain Fable's original plan for provenance. The target is
the single explicit handoff demanded by
`docs/nano-integrated-readiness-2026-07-31.md:182-191` ("running both is not
integration").

## 0. Implementation ledger (updated 2026-08-26)

This is the durable handoff between Fable's field behavior work and the
canonical Rust owner. It distinguishes source behavior, ported behavior, and
remaining work; a passing host test is never presented as physical evidence.

### Retained from Fable as an executable specification

- `deploy/expression/` remains the behavior laboratory and incident corpus.
  Its 82 Python tests are a required CI job (`3e11c8f`); it is not a second
  production owner.
- The touch-observation envelope fix was merged into the Rust compliant hold
  in `a4c8213`. Its intent is to let a yielding head be *observed* outside the
  strict command envelope without ever permitting a command outside that
  envelope.
- The incident semantics retained for the Rust port are: streak-based thermal
  supervision; static-residual pet release; stillness-gated recontact;
  rest-pose arrival–measured comfort; bounded yield duration; per-joint
  backdrivable torque floors; true unattended rest; organic target shaping;
  explicit liveness supervision; and durable pet-session evidence.

### Implemented in the canonical Rust path

- `bd1c7bd`: autonomic saccades now hold, decay smoothly to centre, and are
  suppressed while authoritative face tracking owns gaze. Intent: make the
  two-position eye ping-pong impossible during tracking.
- `727ea1a`: energized temperature decisions are parsed into per-joint streak
  state across control slots; corrupt/unreadable/hot evidence is distinct and
  the control step has no confirmation sleep. Intent: retain the live thermal
  incident fixes without blocking the owner loop or swallowing uncertainty.
- `e29f365`: a deterministic protocol-level STS servo plant synthesizes
  register traffic and motion evidence. Intent: exercise transactional actor
  behavior without pretending a byte replay is a physical plant.
- `a7a0db4`: the compliant controller owns typed `Resting`, static release,
  stillness-gated recontact, contextual multi-axis rest, tap classification,
  bounded yield, and a complete episode summary. Intent: represent a pet as
  one evidence-bearing state transition system rather than timing scripts.
- `533db00`: holding and yield torque profiles are parsed once and switched
  transactionally; gravity axes soften selectively, lateral axes remain
  backdrivable, and holding torque restoration is mandatory on exit paths.
  Intent: allow touch while preserving a known load-bearing floor.
- `365cdbf`: completed pet episodes cross one typed feedback edge into the
  character engine and select `StartleBoop`, `PlayBow`, or `AffectionMelt`;
  their eye and four-joint responses share one clock and minimum-jerk
  keyframes. Intent: let sensed interaction cause a coherent social answer,
  not an unrelated canned eye change.
- `4e6b756`: periodic energized health reads every servo's torque-enable
  register before accepting pose telemetry and reports exact prefixes on
  failure. Intent: never call a free or partially observed neck "healthy" and
  never hide the condition behind an automatic write.
- `f2fc5d8`: Idle/Tracking now have deterministic, phase-varied living motion
  on bow, curl, yaw, and roll; an integer smooth-wave texture avoids float/libm
  and triangle-wave velocity corners; after 20 unattended seconds a six-second
  quintic envelope reaches exact natural and remains there through `Sleepy`;
  face return wakes immediately. Intent: make the bow/base a living joint while
  guaranteeing a real, stable rest state.
- `1883a76`: the physical gaze controller now owns a deterministic fixed-point
  spring prefilter with per-axis response, damping, velocity, acceleration,
  jerk, and transient attack dynamics. Its state participates in the same
  prepare/commit transaction as the safety planner; gaps service one declared
  tick, bounds are exact, and expiry settles the hidden spring and commanded
  pose at natural. Intent: replace constant-speed target feel without weakening
  actor readback, lateness, or motion-envelope guarantees.
- `29ebc94`: the strict head-policy boundary parses Fable's SI-rate dynamics,
  binds them once to the exact control period, and rejects a policy whose
  attacked acceleration or velocity can outrun the downstream planner. Intent:
  keep tuning legible in physical units while the runtime remains fixed-point
  and deterministic.
- `94e529d`: the character boundary no longer collapses an established face to
  a boolean. One parsed value now retains track identity, normalized image
  bearing, non-metric apparent width, freshness, and
  observed/switched/coasting provenance. Every committed compliant phase also
  crosses a typed feedback edge: candidate, yielding, release dwell, resting,
  and recovery progress drive a minimum-jerk soften/shrink/look-down eye
  envelope while scripted head overlays are suppressed. Completed episodes
  remain a separate edge that selects the later social act. Intent: let Kiko
  visibly receive touch while it is happening and preserve the face facts
  required by later character policy without inventing range or servo facts.
- `27df4f6`: Fable's fourth greeting style is retained as a distinct, longer,
  blink-free warm-gold bow/curl gesture using the same minimum-jerk character
  overlay as the other four-axis acts. The overlay remains semantic; reviewed
  hardware mapping still owns servo signs and tick conversion. Intent: recover
  the formal whole-neck greeting without smuggling mechanical assumptions into
  the character engine.
- `23173cc`: the sole accessory owner publishes a monotonic liveness sequence
  only after complete four-joint health transactions. The OAK capture loop is
  the only watchdog sender and kicks systemd only after observing that sequence
  advance; a PID or detached timer cannot mask a stuck expression, eye,
  gaze/compliance, or head-bus operation. The process boundary parses the exact
  60-second systemd contract once, refuses a fault-latched owner, sends
  `STOPPING=1` before coordinated teardown, and the byte-qualified unit applies
  15-second failure backoff plus a five-in-ten-minute start limiter. Intent:
  retire the shell guardian's reason to exist without ever starting a second
  hardware owner alongside a retained first one.
- `0a174e4`: production render-input V2 and launch V4 make the physically
  reviewed expression policy and its separate attended-review record mandatory
  launch assets. The renderer rejects missing, unresolved, aliased, oversized,
  or non-JSON inputs; offline qualification and runtime bootstrap exact-load
  both before hardware, parse the policy once, cross-bind its claimed evidence
  digest, and reject production policy without Fable's four-axis character
  mapping or compliant hold. The same V4 filename is now fixed across the
  renderer, offline marker, base-commissioning live graph, CLI, deployment
  gate, and systemd service. Intent: close the discovered boot-path gap where
  the typed physical controller existed but production always left it
  unconfigured, without treating a JSON operator claim as authenticated or
  physically proven evidence.
- `5d4dbd6`: Fable's historical pet-session NDJSON and Nano V1 records cross
  one strict parser into the same typed episode evidence. V1 retains the
  legacy keys but adds exact integer wall/monotonic times; replay compares the
  recorded Fable reaction with the Rust reaction. The `5.995 -> 6.00` play
  boundary uses exact integer round-half-even arithmetic, while unknown,
  hybrid, oversized, inconsistent, or impossible records fail closed. Intent:
  carry the field corpus forward without making floating-point compatibility
  values authoritative or silently repairing damaged evidence.
- `6f8ab84`: production bootstrap now starts a state-root pet journal before
  the head/eye actors and requires its readiness. The fixed
  `pet-episodes-v1.ndjson` file is opened relative to a component-wise
  no-follow root, must remain a single-link owner-mode-`0600` regular file,
  validates every retained record, and is capped at 16 MiB. Completed episodes
  are encoded and admitted to an eight-record FIFO before the social reaction;
  a separate named thread performs append and `sync_data`. Queue saturation,
  clock/encoding failure, file replacement, I/O failure, or writer exit is a
  terminal accessory/base-stop fault, and shutdown retains bounded join
  evidence. Intent: preserve Fable's durable pet evidence without putting disk
  I/O in the servo/eye loop or swallowing logging failures. No throughput or
  latency improvement is claimed without a benchmark.
- `1bc3cc7`: Fable's soft pitch-workload policy is now a typed hysteretic
  controller in the sole Rust head owner: bow/curl engage after three admitted
  samples at raw `60`, clear after ten trend-preserving cool samples at or
  below `56`, and remain subordinate to the separately bound exclusive hard
  boundary at `65` and plausibility ceiling `95`. The constraint is applied at
  every planner tick, so an already-active proposal is derated immediately;
  yaw/roll and the unmodified proposal remain available, making cool recovery
  deterministic and reversible. The same typed state produces a gradual tired
  eye reflex rather than a false nominal fallback. The existing base/head
  interlock phase also enters every character sample: unbound, transacting,
  moving, and faulted states suppress all four semantic head axes while eyes
  continue; only confirmed stationary or a non-physical renderer permits head
  choreography. Intent: make interoception a fact-driven core reflex and make
  expression/base exclusion structural rather than an ordering convention.

Host evidence at this checkpoint is 97/97 `kiko-expression-runtime` unit
tests plus its compile-fail doctest, warning-free expression, head, and Nano
compile-only Clippy, 177 `kiko-head-runtime` library tests plus 11 binary
tests, 1,453/1,453 `kiko-slam` Nano-agent library tests, 80/80 Nano-agent
binary tests, 7/7 offline-qualifier tests, 36/36 immutable-renderer tests,
all 6 deployment-gate tests, the 13 focused base-commissioning tests, and
82/82 Python behavior tests. The complete Nano-agent library suite was run
outside the filesystem sandbox because 23 otherwise-green local socket and
loopback-listener tests are denied binding by that sandbox; the unrestricted
run passed all 1,453 tests.
The Linux-aarch64 standard-library abstract notify-socket API was compiled
directly. A complete Linux-aarch64 dependency cross-check remains unclaimed
because this Mac does not have `aarch64-linux-gnu-gcc`; native OAK linking,
PID-1/systemd execution, restart behavior, and physical feel are also not
claimed by these host results.

### Remaining before the single-owner handoff is complete

1. Complete the now-typed character boundary with commanded-head feedback and
   metric proximity when depth has valid association evidence. Thermal/body
   state and base-motion exclusion facts are complete. Then port dynamic
   bow/curl recruitment, turn-dip weight shift, and the remaining interoceptive
   acts. Image-plane face bearing, apparent-width proximity cue, continuous
   pet-state eye choreography, and the formal bow greeting are complete;
   apparent width is not relabeled as metric range.
2. Pet NDJSON compatibility, durable production recording, and reaction replay
   are complete. Add the remaining general behavior-trace replay comparison;
   retain the Python lane as a non-booted behavior lab until the Rust owner has
   attended physical parity evidence.
3. Re-audit the current navigation graph, make live SLAM rate/backend/fallback
   observable, verify the manual/emergency-stop GUI and autonomous MPC share
   one authority path, and close every wheels-off gate before requesting wheel
   attachment. Only an attended wheels-on session can calibrate the
   encoderless plant and prove mapping/navigation on the robot.
4. Materialize launch V4's policy/review pair from a fresh attended Nano
   session, run the exact offline gate, and then prove detector, head, eye,
   compliance, watchdog, and shutdown behavior under PID 1. The software path
   is now connected; no native OAK, systemd, or physical-motion claim follows
   from the host evidence above.

## 1. Original situation recorded 2026-08-03

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
12. **Interoception (operator principle, 2026-08-03: "not prompted — a
    core routine Kiko just does").** Subsystems publish body-state FACTS
    (booting, admission recovery, thermal derate, firmware flash in
    progress, park/shutdown, fault recovery, base moving); the character
    engine holds standing reflexes that render them — wake ritual,
    groggy recovery, tired derate, matrix-green flash dreaming,
    anesthesia/wake around eye reflashes, goodnight on park. Nothing
    outside the character engine may request an expression; the rule is
    facts in, behavior owned by Kiko. This is additive to the
    render_character boundary widening in item 8 (body-state joins
    person/proximity/derate as an input struct) and costs no new sensing
    for most events: the engine itself performs admission, release,
    derate, and shutdown, and an STM32 reflash is visible as ST-Link
    re-enumeration without owning any device.

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

## 8a. SLAM readiness track (parallel to phases 1-4; audit 2026-08-03)

Findings: the SLAM core is frozen since 2026-07-20 (0.86% of the 12-day
diff), correct and fail-closed, but (a) live SLAM has never run on this
branch and NO instrument in the tree can emit a live SLAM rate — only
replay paths print FPS; (b) nano-agent enables neither `ort-cuda` nor
`ort-tensorrt`, and backend "auto" silently falls back to CPU, so
SuperPoint x2 + LightGlue run on the same CPU cores the expression engine
will share; (c) the fast stack — projected tracker (19.38 FPS replay vs
6.21 LightGlue-only), VIO, 2048-keypoint models, the benchmark harness —
is stranded on the unmerged February-divergent
`codex/ort-session-placement-evidence` branch, whose own README revokes
the historical CUDA-placement claim; (d) tracker has zero IMU fusion
(gyro-only yaw extrapolation in navigation), loop closure runs on
downgraded deterministic descriptors, the OAK EEPROM has no IMU
calibration, and zero thermal/power data exists.

Track items, independent of the head merge-back:
S1. Port the benchmark/telemetry harness (or a minimal live-rate
    instrument) into this branch — without it the first live SLAM run
    cannot even be measured.
S2. Decide GPU inference: enable `ort-cuda`/`ort-tensorrt` in the
    nano-agent feature graph and make backend fallback LOUD, or accept
    CPU and measure the joint expression+SLAM CPU budget explicitly.
S3. Cherry-pick the projected tracker + 2048-keypoint models from the
    ORT-evidence branch with fresh placement analysis (do not inherit
    the revoked claims).
S4. First live SLAM run remains gated behind the Phase 5 handoff (one
    owner per device — no coexistence evidence is possible before it);
    schedule it as the first post-switchover session, with tegrastats
    thermal/CPU capture alongside.

## 9. Ordering constraints

- Phases 1-4 run while the Python owner keeps the robot alive; zero
  downtime until step 17.
- Phase 1.3 (the plant) precedes all of phases 2-3 testing.
- Phase 3.8 precedes 3.9-3.11. Phase 2.7 requires 1.2 (streaks) only for
  its starvation-path restore semantics.
- The operator sign-offs already pending (stiffness thresholds, soft-torque
  feel, tilt/sink amplitudes) become the recorded values in the Phase 5
  policy bundle — tune them in Python first, where iteration is minutes.
