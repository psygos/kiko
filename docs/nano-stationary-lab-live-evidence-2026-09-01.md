# Nano stationary-lab live evidence — 2026-09-01

This record covers the prompt-free, zero-motion-authority stationary entry
point only. It is integration evidence, not wheels-off motion qualification,
wheel-on plant calibration, occupancy-map acceptance, or autonomous-navigation
evidence.

## Candidate identity

- Source commit: `2355be9e4777b99ec0b5a380651293e29ea8d81e`
- Source archive SHA-256:
  `052ea00004a892f1921bc213acd2721e5adf02cf3ba3bc0e508ac7ea53110d7c`
- Native Linux aarch64 executable size: `31,283,704` bytes
- Native executable SHA-256:
  `8c2f5f02e78412cc3b2ede2f8df5b9b10e404438011149559f405e6a51a40ad7`
- Rendered launch-record SHA-256:
  `a22054aaef38cba582d3d48da804666a2f7778ee3558dfdd12d1992c605d179b`
- Render evidence SHA-256:
  `6d89ad813dc0a73af1d8b032d08f839a88f9bf1c4707ecbf66fd739c6dfd0e7f`
- Installed immutable root: `/opt/kiko/qualification`

The source archive was transferred to the Jetson, verified before extraction,
built natively with the locked dependency graph, and rendered through the
canonical bundle renderer. The installed tree matched the staging tree with
`diff -qr`. `ldd`, with the launch-bound library path, reported no missing
libraries. A preliminary binary-only replacement was rejected before device
ownership because the old launch record's executable-size bound did not match;
the installed candidate is the subsequently rerendered bundle, not that
rejected tree.

## Invocation and observation

The candidate was left running in the foreground with:

```bash
sudo /usr/bin/env LD_LIBRARY_PATH=/opt/kiko/qualification/lib \
  /opt/kiko/qualification/bin/kiko-nano-wheels-off-qualification \
  nano-stationary-lab \
  --deployment-root /opt/kiko/qualification \
  --launch-config nano-wheels-off-qualification-launch-v4.json \
  --state-root /var/lib/kiko-nano-qualification
```

At `2026-09-01T00:00:53+05:30`, PID `95537` had remained alive for more than
five minutes, beyond the earlier approximately 174-second raw-dataset quota
failure. The state root measured `8,135,808,633` bytes before and after the
observation: delta `0` bytes. No file under that root had been modified during
the observation window. The pre-existing size includes retained files from
older aborted persistent sessions; they were not deleted or rewritten.

The operator snapshot after 5 minutes 4 seconds reported:

- runtime `ready_stopped`, requested owner absent, and actual authority absent;
- process-lifetime software safety stop latched;
- controller output disabled, requested/applied left and right PWM all exactly
  zero, and controller fault bits zero;
- STM32, head, eyes, and OAK `ready`;
- SLAM `degraded`, with 161 started pairs, 159 successful pairs, one
  recoverable failure, and zero fatal failures; and
- map, pose, path, goal, and MPC prediction absent.

The SLAM degradation is truthful: CPU SuperPoint and LightGlue inference was
repeatedly slower than the configured 200 ms warning threshold. This run is
therefore evidence that live perception stayed operational, not evidence of a
real-time SLAM rate or a usable occupancy map.

The OAK enumerated as a 5,000 Mbit/s USB device beneath a 10,000 Mbit/s hub and
root hub. Rerun and the operator console listened only on loopback ports 9876
and 9877. The current console schema still exposes legacy
`wheels_off_qualification` labels and raw-PWM pattern metadata even though the
stationary entry point cannot acquire motion authority; that presentation debt
must not be mistaken for an enabled command path.

## Result and remaining gates

The stationary entry point now starts without a physical-attestation dialog,
retains constant-memory navigation-ingress ordering checks, does not allocate
or publish a replay dataset, and keeps nonzero base authority structurally
unavailable. Motion-enabled attended and production modes continue to use the
persistent journal and their existing safety gates.

This run does not establish occupancy output, physical head-gaze behavior,
grounded wheel signs, PWM-to-velocity calibration, the drive plant, MPC
tracking, click-to-goal navigation, or recovery behavior. Those remain explicit
acceptance work; no claim is inferred from the stationary observation.
