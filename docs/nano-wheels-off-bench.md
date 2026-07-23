# Retired Nano wheels-off bench

The former `kiko-nano-wheels-off-bench` executable, `nano-bench` Cargo
feature, standalone `kiko-robot-server.service`, paired bench systemd unit,
bench launch JSON, and one-shot `kiko-wheels-off-attest` helper have been
removed. Do not reconstruct or deploy that topology.

That path split OAK/accessory ownership from the production agent, launched a
second motor service, used a separate zero-only policy stack, and could
torque-disable the neck during cleanup. Those behaviors conflict with the
canonical single-owner, tension-preserving, launch-admitted runtime.

Use the attended qualifier in
[`nano-wheels-off-qualification.md`](nano-wheels-off-qualification.md).
It uses the same OAK, head, eye, SLAM, occupancy, console, and controller-owner
architecture as production while keeping autonomous actuation disabled and
raw candidate PWM explicitly separate from SI/MPC control.

Historical evidence that names the removed bench remains evidence about that
past run only. It is not a current launch instruction, production
qualification, or permission to attach the wheels.
