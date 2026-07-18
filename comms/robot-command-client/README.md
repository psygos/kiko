# Robot command client

This crate is the disarmed-by-default host boundary for physical Kiko motor commands. It does not
make the current V1 acceptance acknowledgement safe for actuation. A wire adapter must provide the
V2 contract: the exact controller UID, reset-unique boot ID, control epoch, controller-enforced
command lease, and a result emitted only after the STM32 reports the exact timer PWM it applied.

`DisarmedCommandClient` has no command method. It can become `ArmedCommandClient` only after a V2
`StatusQuery` proving stopped state, an exact `AcquireControl` identity handshake, and one timely
sequence-zero applied-zero result. An armed
command consumes both the client and a non-cloneable `PendingPhysicalCommand`, so at most one command
can be in flight. A nonzero command is never retried. Timeout, malformed or mismatched evidence,
controller reset, sequence exhaustion, clock regression, or expired validity returns only a
`LatchedCommandClient`. The latch can record a later controller-confirmed `HostStop` but cannot arm
again in that process.

Dropping an armed client makes bounded best-effort, UID-targeted `HostStop` attempts, but `Drop`
returns no receipt and therefore never claims the robot stopped. Call `disarm` and retain its
`DisarmReceipt` whenever the caller needs controller-confirmed safe output.

The UDP implementation's default `RobotProtocolV2WireAdapter` encodes and decodes only canonical
`robot_protocol::v2::Message` values. It cannot reinterpret the legacy V1 "accepted by server"
packet as an applied result.

```compile_fail
use robot_command_client::{DisarmedCommandClient, PendingPhysicalCommand};

fn cannot_apply_while_disarmed<T, C>(
    client: DisarmedCommandClient<T, C>,
    command: PendingPhysicalCommand,
) {
    client.apply(command);
}
```

```compile_fail
use robot_command_client::LatchedCommandClient;

fn a_latched_process_cannot_rearm<T, C>(client: LatchedCommandClient<T, C>) {
    client.acquire_zero();
}
```
