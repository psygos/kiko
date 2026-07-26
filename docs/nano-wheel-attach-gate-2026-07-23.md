# Kiko Nano wheel-attach qualification gate

This is the controlling checklist for the operator handoff:

> Please attach the wheels; I will calibrate and drive Kiko around and make a
> map.

That sentence is permitted only after every pre-attach item below has direct,
current evidence. Historical demonstrations, host-only tests, device
enumeration, and a process that merely remains alive do not satisfy a physical
gate.

## Historical read-only Nano snapshots

Observed on `makerspace@192.168.50.2` at
`2026-07-23T14:18:06+05:30`, without stopping or opening any owned device:

- boot ID: `9c8e1a8e-e000-42bd-81bb-5db22674d954`;
- host architecture: Linux aarch64;
- the Fable guardian was running
  `/home/makerspace/kiko-follow/engine-guardian.sh`;
- its child `python3 kiko_face_follow.py --duration-s 864000` exclusively held
  the OAK USB node, the head adapter, and the KEP2 eye serial port;
- the current face-follow log was continuing to report RGB-derived person
  tracking, head tracking, eye tracking, and expression acts;
- the guardian log contained repeated respawns during this boot. The latest
  child was running, but that is not cold-boot stability evidence;
- no running Kiko or robot-server systemd service was observed;
- all three persistent serial links were present:
  - head:
    `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031114-if00`;
  - STM32:
    `/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066EFF313946303143221230-if02`;
  - eyes:
    `/dev/serial/by-id/usb-kiko_kiko-eyes-kep2_98c47919804f9f1aaacfd5fa0a20bf74-if00`;
- OAK MXID `19443010F1B43A2E00` was present at Linux USB path
  `platform-3610000.usb-usb-0:2.3`, but negotiated only `480M` on the USB2
  tree. The separate SuperSpeed root/hub was present at `10000M` with no OAK
  below it. The running Fable prototype explicitly requests
  `dai.UsbSpeed.HIGH`, so that observation does **not** by itself prove a bad
  cable, port, or SuperSpeed negotiation failure.

A subsequent read-only check during the same boot, at
`2026-07-23T18:01:40+05:30`, found:

- the same guardian and face-follow child still owned the OAK, head adapter,
  and eye controller; no canonical Kiko or robot-server owner was running;
- the Fable source at `kiko_face_follow.py:408` explicitly opened the exact
  OAK with `dai.UsbSpeed.HIGH`;
- the STM32 serial path had no process owner, but the canonical zero-write
  `v2_identity_probe` failed with
  `Decode(OversizedRecord { maximum: 73 })`. A prior bounded capture that
  transmitted no bytes observed recurring legacy ASCII `ODO,...` records.
  The current image is therefore not an admitted KRP2 controller. Zero-valued
  legacy telemetry is not a typed applied-zero receipt;
- the pre-flash STM32 backup was present on both development Mac and Nano.
  Each location contained two byte-identical 524,288-byte main-flash reads
  with SHA-256
  `8e8f658e5ee65b2eca3ca8de7cb045ea2b08dbf3ec82d70b654fe6fa02bec7dc`
  and a 16-byte option-byte read with SHA-256
  `d292558017cf9ca0a2e40e262a5c1daa4b305ccf084ce06128133d282f905115`.

These timestamped observations prove presence and observed ownership only at
those instants. They do not prove later servo temperature, head torque state,
STM32 control identity, camera-frame correctness, SLAM, occupancy, motion
safety, or service restart behavior.

### 2026-07-24 motor-inert transport update

At the time of this update, the STM32 ran the exact motion-disabled KRP2 image
at build `131074`, fingerprint `KIKO-NO-ACT-V1!!`, capability bits `319`, and
maximum PWM zero. Fresh schema-3 runs from source
`35adc901e50d0ccb893c66582238bea438e86f97` passed at 20 Hz
(200/200 reports) and 50 Hz (500/500 reports). Both had zero missing,
duplicate, reordered, skipped, queue-skipped, or period-late probes. Their
maximum diagnostic RTTs were 17.939749 ms and 17.937019 ms respectively.

The exact evidence directories and hashes are recorded in
`docs/nano-integration-acceptance-2026-07-24.md`. Fable's process and
head/eye serial ownership snapshots remained byte-identical across both runs;
the STM32 VCP was separately available to the qualifier. Motor power was
operator-reported, not independently instrumented, to remain disconnected.
The runs created no control session and sent no PWM. This closes only the
motor-inert 20/50 Hz diagnostic transport item below.

During those measurements Fable remained the OAK/head/eye owner. That did not
prevent the separate STM32 transport measurement, but it prevented canonical
`kiko-slam nano-agent` startup. DepthAI and both accessory transports are
exclusive owners, not shareable process resources.

### 2026-07-27 read-only status refresh

A later read-only check, without stopping a process or opening a device, found:

- the Fable guardian and `kiko_face_follow.py` child still running;
- the Fable child holding the head serial endpoint. That check did not
  establish the then-current OAK or eye process owner, so the attended handoff
  must re-check every endpoint rather than inheriting the July 23 ownership
  result;
- OAK MXID `19443010F1B43A2E00` enumerated at `480M` on the USB2 tree, while a
  separate `10000M` USB3 tree existed without the OAK beneath it;
- no `/opt/kiko` installation and no `kiko-nano-agent.service`;
- `/home/makerspace/kiko` clean at
  `482023e0fa69c381cb5d5946c445234a0ae88105` on
  `codex/jetson-hardware-validation`.

This refresh did not re-probe STM32 firmware identity, camera frames, eye
ownership, head health, SLAM, or motion. The canonical handoff therefore still
requires a fresh endpoint-by-endpoint owner check, a canonical SuperSpeed
attempt, and exact live admission.

### 2026-07-27 native and motor-inert refresh

Exact source revision `9c9f3a9b92d7f610a2153129849e12863a2646c8`
passed a real native Linux-aarch64 `nano-agent` all-target check, a
native OpenCV 4.5.4 cascade construction/detection smoke test that opened no
camera, and a release
`nano-wheels-off-qualification` executable build. The 28,119,992-byte release
ELF has SHA-256
`0b58ca0f07392ff65af516c7d82fade87c770ab3e0845ea4f996da2d4ee8d2c4`;
its target loader trace had no unresolved dependency under the audited build
environment. The exact details and limitations are recorded in
`docs/nano-integration-acceptance-2026-07-24.md`.

A fresh byte-read-only STM32 identity probe found the motion-disabled KRP2
image at ABI `2`, build `131074`, fingerprint `KIKO-NO-ACT-V1!!`,
capabilities `319`, maximum PWM zero, and output disabled. Fresh schema-3
transport runs passed 200/200 probes at 20 Hz and 500/500 at 50 Hz, with zero
missing, duplicate, reordered, scheduler-skipped, queue-skipped, or
period-late probes. This does not qualify the four-PWM candidate or any
physical behavior.

At the same refresh, the stale Fable guardian/child process family remained
present and the child still held the head endpoint. The OAK remained on the
480M USB2 tree while the 10000M USB3 tree had no OAK below it. No process was
stopped and no camera was opened. Gate A therefore remains closed.

## Gate A: required before asking to attach wheels

Passing this gate authorizes physical wheel attachment for an attended
calibration session only. It does not authorize production, autonomous,
point-goal, frontier, or MPC-driven motion. At the moment the sentence is
issued, motor power must remain independently cut and the controller must
remain disarmed.

- [ ] Freeze an immutable, reviewable source revision and
      wheels-off/commissioning bundle.
- [x] Build that exact source natively on the Nano with its pinned native
      dependencies; record source and binary identities. Revision `9c9f3a9`
      and the release ELF identity are recorded in the acceptance report.
- [ ] Perform a coordinated handoff from every existing Fable device owner.
      Never start a second OAK, head, or eye owner and never use a broad
      process kill.
- [ ] Have the canonical owner request SuperSpeed for the exact OAK, read back
      the negotiated transport, and admit RGB, stereo, rectified-left depth,
      and IMU from one graph. Request a physical port/cable move only if that
      canonical attempt fails; Fable's forced High-Speed mode is not such an
      attempt.
- [ ] Provision and read back the boot journal, flash and exactly identify the
      operator-supervised four-PWM candidate, and keep it distinct from both
      the motion-disabled diagnostic image and any later production image.
- [ ] Start one least-privilege qualification lifecycle with the sole typed
      STM32 owner. Admit the exact manifest, artifacts, OAK, STM32 session,
      KEP2 eye identity, head adapter, and servos 1 through 4.
- [ ] Establish a confirmed applied base zero, remain explicitly disarmed, and
      independently verify that the reachable physical cut removes motor
      power outside Jetson and STM32 control.
- [ ] Establish and continuously supervise the natural head hold. A reported
      fault must stop base authority without silently abandoning a still-live
      hold; coordinated cleanup must explicitly report every attempted torque
      release.
- [ ] Produce current RGB-derived eye behavior through the same OAK owner.
- [ ] Produce live stereo/IMU SLAM, localized rectified-left depth occupancy,
      and diagnostic Rerun/status output.
- [ ] Serve one loopback-only operator/agent control gateway. It must own the
      sole downstream request sequence, keep production autonomous authority
      disabled, display only exact applied-controller evidence, and expose
      live pose, occupancy, goal, path, shadow MPC rollout, and timing without
      opening a second camera or motor connection.
- [ ] Prove typed arm/disarm, manual deadman streaming, priority-latched stop,
      client-reconnect handling, and exact applied receipts first against fake
      or loopback transports and then in the bounded wheels-off candidate run
      described by `docs/nano-wheels-off-qualification.md`.
- [x] With the motion-disabled KRP2 profile, run the real STM32 diagnostic
      transport qualifier at 20 Hz and 50 Hz. The schema-3 results record exact
      sequence, scheduling, round-trip, controller-service, queue, wire-rate,
      identity, and idle-safe evidence. They do not qualify the
      motion-capable command path or claim a performance improvement.
- [ ] With the wheels absent and the robot/head supported, run the candidate
      fault matrix: camera loss, stale depth, localization loss, controller
      reset/serial loss, client disconnect, command expiry, clock fault,
      SIGTERM, and cold restart. Each motion-relevant path must confirm zero or
      report exact uncertainty.
- [ ] With the wheels still absent and separate operator approval, establish
      left/right shaft/body sign conventions through bounded low-output
      candidate commands. This is sign evidence, not a velocity model.
- [ ] Prepare a clear commissioning area, support/restrain the robot until
      attachment is complete, station an operator at the independent power
      cut, and re-confirm disarmed applied zero immediately before asking for
      wheels.

## After Gate A: attended calibration-only work

Wheel attachment does not itself authorize power or motion. After the operator
attaches the wheels, separately authorize the bounded
`kiko-nano-base-commission` schedule in a clear area with the independent
power cut continuously reachable:

1. record exact applied PWM, visual forward velocity, calibrated IMU yaw rate,
   controller/session identity, and common monotonic timing;
2. fit unequal left/right first-order wheel plants and reject the result unless
   coverage, conditioning, parameter-domain, repeated-run, and holdout
   residual gates pass;
3. measure effective wheelbase evidence and low-speed stopping behavior; and
4. emit a non-activatable plant proposal and complete evidence set for review.

## Gate B: required before production motion

Calibration output cannot authorize itself. Production manual, MPC,
autonomous, point-goal, and frontier motion remain disabled until all of the
following are independently reviewed:

1. installed left/right wiring, maximum output, voltage levels, active
   polarities, default-off external driver enable, driver-fault input,
   E-stop feedback, reset/brownout behavior, and physical stop semantics;
2. a uniquely identified production four-PWM firmware profile that samples
   real fault-clear inputs rather than inferring them from a capability class;
3. an explicit promotion boundary binding accepted commissioning evidence,
   repeated-run consistency, wiring/stop qualification, approver identity,
   exact plant bytes, and flashed production STM32 identity;
4. an immutable production bundle and qualified-only service enablement whose
   offline marker, installed bytes, least-privilege unit, and cold-boot
   admission all verify exactly;
5. the production fault matrix, applied-zero behavior, deadman, obstacle stop,
   and independent physical emergency cut; and
6. MPC tuning only inside the measured plant envelope, followed by bounded
   online-SLAM/map-save/reload/relocalization, frontier, and revision-bound
   point-goal qualification.

IMU yaw rate cannot identify translation or PWM-to-linear-velocity gain by
itself. Visual forward velocity is required. Persisted occupancy alone cannot
establish the robot pose.
