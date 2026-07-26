# Nano attended base-commissioning template

This directory is a provisioning template for the distinct, attended,
wheel-on commissioning lane. It is deliberately not a deployable example.
Every `${...}` placeholder must be replaced with evidence from the exact Kiko
assembly and every rendered file must be hashed into its parent launch
document. An unexpanded template must fail parsing.

The lane is isolated from both production and wheels-off qualification:

- controller protocol/server/inventory class:
  `attended_wheel_on_commissioning`;
- firmware build ID: `139265` (`0x00022001`);
- actuator fingerprint:
  `4b494b4f2d574845454c4f4e2d434d31`
  (`KIKO-WHEELON-CM1`);
- physical-stop semantics: `unverified`;
- absolute PWM limit: 20%, with a 20 percentage-point command-step limit;
- output: immutable dataset, evidence, and a
  `proposed_unapproved` plant artifact only.

The result grants no manual, MPC, mapping, production, or autonomous motion
authority.

## Rendered bundle

The outer `nano-base-commissioning-launch-v1.json` must bind these exact
rendered assets by lower-case SHA-256 and byte limit:

1. `commissioning/base-commissioning-policy-v1.json`
2. `commissioning/controller-profile-v1.json`
3. `commissioning/controller-server-v3.json`
4. `commissioning/device-inventory-v3.json`
5. the exact calibration artifact
6. canonical `nano-agent-launch-v3.json`

The nested canonical launch binds the exact agent-policy V3, OAK ORT models,
SuperPoint model, live graph, and calibration. Commissioning admits its
accessory policy only after binding it to the commissioning inventory; it
opens the exact OAK MXID and graph in the same process. External camera,
velocity, or IMU injection is not supported.

Render leaf assets first, compute their exact sizes and SHA-256 digests, render
the canonical `nano-agent-launch-v3.json`, then render the outer commissioning
launch last. Use a new commissioning session ID for each attempt. The state
writer uses create-new semantics and will not overwrite a prior session.

## Assembly-specific evidence

Do not invent the numeric placeholders in
`base-commissioning-policy-v1.json.template`. They require review against the
assembled robot:

- measured wheelbase and its calibration identity;
- stationary visual-forward and IMU-yaw noise bounds;
- plausible observed forward, lateral, and yaw bounds;
- time-constant and signed gain-magnitude search bounds;
- conditioning, sensitivity, and holdout-residual gates;
- lateral holdout margin and the exact floor/surface validity scope.

`require_positive_velocity_gain` is intentionally `false`. Wheel direction is
learned as a signed proposal from OAK visual velocity plus calibrated IMU yaw;
the configuration does not assume that motor wiring has canonical polarity.
The proposal still requires separate review before any activation.

The policy schedules four cycles of forward, reverse, positive-yaw, and
negative-yaw excitation at 15%, separated by exact-zero dwell. Every live
sample refreshes the STM32 command lease. The configured 100 ms effective
sample-gap bound plus 30 ms applied-ack timeout is below the 250 ms command
lease. A deterministic state-machine regression at the slowest admitted
100 ms cadence completes in 523 samples, inside the policy's 500–3000 sample
budget.

## Firmware identity

Build the dedicated STM32 image from the same source revision:

```sh
cargo build --locked --release -p embedded \
  --target thumbv7em-none-eabihf \
  --bin embedded \
  --features firmware,attended-wheel-on-commissioning,flash-boot-journal
```

Flash and read back that exact image through the approved hardware procedure.
The durable flash boot journal is part of this identity. A V1 production or V2
wheels-off image must be rejected; changing JSON to resemble V3 does not
convert the firmware. No hardware success is claimed by this template.

## Ownership handoff

Before the attended command:

1. Coordinate a normal stop of all current OAK/head/eye owners, including
   Fable where present. Do not kill unrelated processes.
2. The prior head owner must issue no torque-disable write. Support the head
   throughout the handoff; serial cleanup does not prove physical torque
   continuity.
3. Freshly check each exact OAK, head-serial, eye-serial, STM32-serial, and
   controller UDP endpoint and confirm there is no competing owner.
4. Confirm the head and eye by-ID paths, exact OAK MXID, STM32 by-ID path, and
   loopback endpoint match the rendered inventory.
5. Place Kiko on a flat, high-friction test surface with both wheels attached,
   the full motion envelope clear, the operator continuously attending, and a
   separately powered emergency cut already tested and immediately reachable.

The one-shot process starts and verifies the natural head hold, eye owner, and
an RGB-driven expression frame before opening `/dev/tty` for four fresh,
nonce-bound exact confirmations. The private linear result is bound to this
session's exact launch/config/model digests, admitted stream identities, and
clock epoch; its host timestamp is captured by the ceremony and the same
prepared session must consume it within five seconds. It then starts the
in-process V3 STM32 owner at exact zero.

Commissioning parses and content-binds the nested launch-V3 face cascades but
does not load or execute them. Its `scene_motion` expression mode proves only
the base RGB-expression lane. Production launch V3 adds the separate mandatory
face-perception lane and must qualify detector readiness and latency itself.

Do not install this command as an unattended systemd service or boot task.

## One attended command

From the matching checkout on the Nano:

```sh
cargo run --locked --release -p kiko-slam \
  --features nano-base-commissioning \
  --bin kiko-nano-base-commission -- \
  --deployment-root /opt/kiko/deployment \
  --launch commissioning/nano-base-commissioning-launch-v1.json \
  --state-root /var/lib/kiko/base-commissioning
```

The state root must already exist, be absolute, be owned by the invoking user,
and have mode `0700`. Run this command from a real controlling terminal. After
live readiness it discards pending terminal input before each prompt, presents
a fresh 128-bit challenge, and requires the displayed exact response within 15
seconds and 96 bytes. Redirected stdin, old confirmation flags, environment
claim aliases, and reusable cross-session confirmation tokens are not accepted.
This is a passive-replay barrier, not proof that a human rather than an active
same-user PTY controller supplied the fresh text; the physical attended
procedure and independent cut remain mandatory.

Standard output is NDJSON for evidence capture and external tooling;
diagnostics are on standard error and prompts go directly to `/dev/tty`. The
production web console does not currently launch or consume this commissioning
process. Treat any `failed` event, nonzero exit, terminal signal, journal
failure, output failure, live-resource cleanup failure, or missing final
`completed` event as a failed attempt.

On every handled SIGINT, SIGTERM, SIGHUP, terminal-source, or output-failure
path after ownership, the runtime closes live resources and, if the controller
exists, requests and verifies explicit zero before its bounded shutdown.
SIGKILL and host/power loss cannot run process cleanup, so the firmware lease
and independent cut remain mandatory backstops. `Drop` remains a best-effort
fallback only and is never completion evidence. The natural head hold remains
torque-preserving during the verified accessory handoff; eye ownership is
explicitly released.

See `docs/nano-base-commissioning.md` for the boundary, event contract,
failure semantics, and post-run review requirements.
