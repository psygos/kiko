# Kiko Nano wheel-attach qualification gate

This is the controlling checklist for the operator handoff:

> Please attach the wheels; I will calibrate and drive Kiko around and make a
> map.

That sentence is permitted only after every pre-attach item below has direct,
current evidence. Historical demonstrations, host-only tests, device
enumeration, and a process that merely remains alive do not satisfy a physical
gate.

## Current read-only Nano snapshot

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
  below it.

These observations prove current presence and observed ownership only. They do
not prove current servo temperature, head torque state, STM32 control identity,
camera-frame correctness, SLAM, occupancy, motion safety, or service restart
behavior.

## Required before asking for wheels

- [ ] Freeze an immutable, reviewable source revision and deployment bundle.
- [ ] Build the exact source natively on the Nano with the pinned DepthAI and
      ONNX Runtime libraries; record source and binary identities.
- [ ] Move the OAK to a SuperSpeed path and admit the exact MXID, transport
      speed, RGB, stereo, rectified-left depth, and IMU graph.
- [ ] Perform a coordinated handoff from the Fable guardian. Never start a
      second OAK, head, or eye owner and never use a broad process kill.
- [ ] Start one least-privilege production lifecycle, including the sole typed
      STM32 owner.
- [ ] Admit the exact manifest, artifacts, OAK, STM32 session, KEP2 eye
      identity, head adapter, and servos 1 through 4.
- [ ] Establish a confirmed applied base zero and remain explicitly disarmed.
- [ ] Establish and continuously supervise the natural head hold. A reported
      fault must stop base authority but must not silently drop the still-live
      head hold; torque release is an explicit coordinated shutdown action.
- [ ] Produce current RGB-derived eye behavior through the same OAK owner.
- [ ] Produce live stereo/IMU SLAM, localized rectified-left depth occupancy,
      and diagnostic Rerun/status output.
- [ ] Prove typed arm/disarm and manual deadman command streaming against fake
      and loopback transports, including exact applied receipts.
- [ ] With the wheels physically absent and the area/head supported, run the
      bounded production fault matrix: camera loss, stale depth, localization
      loss, controller reset/serial loss, client disconnect, command expiry,
      clock fault, SIGTERM, and cold restart. Each motion-relevant path must
      require or confirm zero.
- [ ] With the wheels still absent and separate operator approval, establish
      left/right body-sign conventions through bounded low-output production
      commands. This is sign evidence, not a velocity model.

## Work after wheels are attached

Wheel attachment does not authorize unbounded manual driving. In a clear area
with an immediately reachable independent power cut:

1. run the bounded encoderless commissioning schedule;
2. record exact applied PWM, visual forward velocity, calibrated IMU yaw rate,
   controller/session identity, and common monotonic timing;
3. fit unequal left/right first-order wheel plants and reject the result unless
   coverage, conditioning, parameter-domain, and holdout residual gates pass;
4. measure effective wheelbase evidence and low-speed stopping behavior;
5. bind the accepted plant artifact into the exact manifest and re-admit the
   production motion owner;
6. tune MPC only inside the measured support envelope;
7. qualify deadman, obstacle stop, continued online SLAM and mapping;
8. drive a bounded map run, atomically save it, reload it with exact replay
   binding, establish fresh live relocalization, then qualify frontier and
   revision-bound point-goal navigation.

IMU yaw rate cannot identify translation or PWM-to-linear-velocity gain by
itself. Visual forward velocity is required. Persisted occupancy alone cannot
establish the robot pose.
