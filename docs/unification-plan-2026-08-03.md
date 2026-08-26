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
- `e561c97`: the current face proposal now retains its desired neutral-head
  angles, while the sole servo actor returns the exact target it actually
  committed. The reviewed sparse mapping reconstructs commanded optical gaze
  through a scale-stable signed-total-demand inverse; desired minus
  commanded gaze replaces only the original face contribution before KEP2
  application. Character saccades, act offsets, lids, colour, blink, and
  freshness remain intact. Bounded tracker coasting retains the last typed
  angular target without reissuing a stale head command, so detector gaps
  cannot alternate raw image gaze with residual gaze. Invalid mapping,
  nonfinite residual, missing committed state, and impossible composition are
  typed terminal faults. Intent: close Fable's eye/head feedback loop in the
  canonical owner without reconstructing radians from normalized image
  coordinates or weakening head authority.
- `07d64e8`: Fable's dynamic serial-pitch recruitment is now a typed optional
  mapping policy and a production requirement. Bow begins at the
  coefficient-derived baseline share, recruits monotonically toward `600`
  permille, and reaches it at `140` combined absolute pitch-demand ticks;
  curl receives the exact complementary share. The inverse sums signed joint
  demand rather than treating the two shares as independent measurements, so
  redistribution preserves gaze within the two-joint one-tick quantization
  bound. Zero, closed-unit, non-recruiting, and numerically collapsed policies
  fail during parsing, while recruited envelope overflow rejects rather than
  clamps. Legacy documents that omit the block remain linear but cannot cross
  production physical admission. Intent: make large pitch movements recruit
  the base joint without changing total gaze demand or corrupting residual eye
  coordination.
- `6883a06`: Fable's rapid-turn weight shift is now a typed, production-required
  posture in the evidence-bound Rust owner. Physical yaw demand below
  `120 ticks/s` is deadbanded; excess rate uses an `80 ms` dimensioned gain,
  saturates at `26` equal negative bow/curl ticks, retains `850` permille per
  `50 ms`, and derives no fresh rate from intervals at or above `500 ms`.
  Opposing pitch signs and natural-to-minimum travel are cross-checked during
  parsing. Runtime rate and decay use deterministic checked integer/fixed-point
  arithmetic anchored to monotonic time, so recovery is independent of frame
  segmentation. The posture composes after face and character demand, remains
  optically gaze-neutral under the admitted signed-total inverse, can finish
  without a face instead of snapping upright, and reports envelope withholding
  instead of clamping. Intent: preserve Fable's whole-neck lean into a turn
  without feeding posture back into gaze, hiding scheduler gaps, or weakening
  physical bounds.
- `9043c1f`: lifecycle interoception is now one closed character-owned input:
  booting, admission recovery, operational, controller- and phase-specific
  firmware maintenance, fault recovery, and parking. Exclusive facts suppress
  face gaze, saccades, pet/thermal decoration, acts, and all four semantic head
  axes. Every entry captures the last rendered eye state, every ambient return
  uses a 1.2-second minimum-jerk release, and the coherent frame retains its
  exact body evidence. A camera-independent bridge sample does not fabricate
  or advance RGB continuity. Eye BOOTSEL is stated truthfully as unrenderable;
  literal Matrix remains the post-upload firmware boot renderer, while another
  controller may receive a deliberately distinct green KEP2 dream. Intent:
  make body facts standing reflexes owned by Kiko instead of animation calls.
- `4ff1d1e`: the sole Nano accessory loop now publishes `Booting`, waits for
  the eye session and four-joint head admission, streams the 2.6-second
  character-owned admission recovery, and publishes `Operational` only after
  it succeeds. Normal shutdown streams the 1.8-second goodnight at 20 Hz before
  KEP2 release. Generation or eye-apply failures retain their typed cause,
  latch the owner fault, close both actors, and cannot report false readiness.
  A terminal fault publishes `FaultRecovery` immediately without delaying the
  existing stop latch; the terminal branch does not yet stream fresh fault-eye
  samples, because secondary presentation failure still needs durable evidence
  without replacing the primary fault. Intent: make startup and park observable
  consequences of lifecycle facts without weakening fault ordering.
- `8704e50`: the retained Python lab now generates a strict, source-hashed
  schema-V1 semantic behavior trace. Rust rejects unknown and duplicate fields,
  checks all 24 distinct acts against Fable's exact duration, cooldown, and
  mode-eligibility contract, proves every recorded Fable eye/head channel is
  retained by the richer four-axis act, replays the seven-step attention-mode
  sequence, replays all three pet classifications, and requires exact natural
  at unattended rest. The comparison exposed and corrected three genuine
  drifts: re-greeting `10 s -> 6 s`, nod cooldown `11 s -> 10 s`, and soft-nod
  cooldown `12 s -> 11 s`. Python float paths, independent RNG samples,
  normalized character offsets, encoder ticks, and physical motion remain
  explicitly non-equivalent domains. Intent: make general software parity a
  reproducible gate without manufacturing a physical-parity claim.
- `92acb38`: the canonical natural-return and head-gaze policy now uses
  Fable's 2026-08-06 replacement-bow-servo configuration rather than the
  superseded July neck values. The one exported constant set carries target
  `[1505,3937,1551,3018]`, symmetric 128-tick start/travel bounds, and
  `[650,550,400,400]` holding torque into both the agent and renderer. The
  canonical pitch-down mapping is derived from Fable's opposite pitch-up
  convention as bow/curl `[-217,+403]` ticks/radian; expressive envelopes,
  character scale, and full source digest follow the same current JSON. A
  regression test derives these facts from the retained Fable file and checks
  both policy templates, while the renderer no longer owns a duplicate
  constant copy. Intent: make the single-owner binary capable of representing
  the neck that is actually installed, without relabelling source parity as a
  fresh physical qualification.
- `29ec63d`: console snapshot schema V5 now separates OAK stream health from
  sparse-SLAM completion health. One fixed-size telemetry owner retains exact
  started, successful, recoverable, and fatal outcomes; successful source and
  completion clocks; requested and actually selected SuperPoint/LightGlue
  providers; and a 64-completion rate window. Rerun receives the same
  diagnostic evidence, while the browser rejects impossible counters, clocks,
  providers, and windows and requires both OAK and SLAM readiness for manual
  control. The provider is runtime session-selection evidence and the rate is
  successful-completion evidence, not an accelerator-placement, utilization,
  camera-FPS, or performance claim. Intent: make a stalled or silently
  CPU-fallback tracker visible instead of inheriting a green state from fresh
  OAK traffic.

Host evidence at this checkpoint is 112/112 `kiko-expression-runtime` unit
tests plus its compile-fail doctest, warning-free expression, head, and Nano
compile-only Clippy, 177 `kiko-head-runtime` library tests plus 11 binary
tests, 1,464/1,464 `kiko-slam` Nano-agent library tests, 80/80 Nano-agent
binary tests, 7/7 offline-qualifier tests, 36/36 immutable-renderer tests,
all 6 deployment-gate tests, the 13 focused base-commissioning tests, and
85/85 Python behavior tests. The standalone KEP2 firmware passed all 8
renderer tests, including literal Matrix dynamics and maximum-time totality.
The complete Nano-agent library suite was run
outside the filesystem sandbox because 23 otherwise-green local socket and
loopback-listener tests are denied binding by that sandbox; the unrestricted
run passed all 1,464 tests.
The Linux-aarch64 standard-library abstract notify-socket API was compiled
directly. A complete Linux-aarch64 dependency cross-check remains unclaimed
because this Mac does not have `aarch64-linux-gnu-gcc`; native OAK linking,
PID-1/systemd execution, restart behavior, and physical feel are also not
claimed by these host results.

### Remaining before the single-owner handoff is complete

1. Commanded-head feedback, residual-driven eyes, turn dip, thermal/body state,
   base-motion exclusion, image-plane face bearing, apparent-width proximity,
   continuous pet-state eyes, the formal bow, startup recovery, firmware
   anesthesia/wake semantics, and normal goodnight are complete in the typed
   owner. Add metric proximity only when depth has valid face-association
   evidence. Wire firmware-maintenance facts when a canonical firmware
   coordinator exists, and add durable secondary-fault evidence before the
   terminal branch streams fault-recovery eyes; do not replace the primary
   stop fault or call apparent width metric range.
2. Pet NDJSON compatibility, durable production recording, reaction replay,
   and general source-hashed semantic behavior-trace replay are complete.
   Retain the Python lane as a non-booted behavior lab until the Rust owner has
   attended physical parity evidence; do not reinterpret the host semantic
   trace as actuator, optical, timing-load, or physical-feel evidence.
3. The navigation graph re-audit and host observability portion are complete:
   live SLAM rate, requested/selected backend, fallback, outcomes, and
   freshness are exposed by `29ec63d`; the manual GUI, agent ingress, software
   stop, mapping, frontier, point-goal, and MPC paths retain the one typed
   arbitrator. The attended wheels-off gates remain open. Only an attended
   wheels-on session can calibrate the encoderless plant and prove
   mapping/navigation on the robot.
4. Materialize launch V4's policy/review pair from a fresh attended Nano
   session, run the exact offline gate, and then prove detector, head, eye,
   compliance, watchdog, and shutdown behavior under PID 1. The software path
   is now connected; no native OAK, systemd, or physical-motion claim follows
   from the host evidence above.

The read-only 2026-08-27 Orin checkpoint is retained in
`docs/nano-single-owner-handoff-audit-2026-08-27.md`. It confirms that Fable's
Python owner remains active on OAK/head/eyes at USB3 while STM32 is idle, but
no canonical service, current bundle, online SLAM, occupancy, GUI, MPC, or
navigation owner is running. This is the safe pre-handoff state, not completion.

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
    schedule it as the first post-switchover session. Any Jetson GPU,
    thermal, power-mode, or utilization work belongs to the separate Nano
    hardware lane and is not authorized by this host-side plan.

Current status at `29ec63d` (2026-08-27): S1's minimum truthful live instrument
is implemented. S2's fallback is now loud because the request and actual
session selection are separate typed values in logs, console V5, and Rerun;
the production feature graph has not been changed to claim CUDA or TensorRT.
S3 remains intentionally unported because its historical placement claim was
revoked and no fresh evidence justifies changing Kiko's SLAM architecture.
S4 remains a physical Nano gate. No Jetson GPU benchmark, power-mode change,
thermal diagnosis, or runtime performance claim was made in this host-side
work.

## 9. Ordering constraints

- Phases 1-4 run while the Python owner keeps the robot alive; zero
  downtime until step 17.
- Phase 1.3 (the plant) precedes all of phases 2-3 testing.
- Phase 3.8 precedes 3.9-3.11. Phase 2.7 requires 1.2 (streaks) only for
  its starvation-path restore semantics.
- The operator sign-offs already pending (stiffness thresholds, soft-torque
  feel, tilt/sink amplitudes) become the recorded values in the Phase 5
  policy bundle — tune them in Python first, where iteration is minutes.
