# Nano cold-boot and fault component acceptance

`tools/nano-cold-boot-fault-acceptance.sh` is the bounded, offline software
acceptance runner for the production Nano boot graph. It opens no USB or serial
device, starts no service, sends no motor/head/eye command, and performs no
deployment. Run it from a clean checkout of the exact revision and lockfile:

```bash
tools/nano-cold-boot-fault-acceptance.sh
```

The runner sets `OAK_SYS_CHECK_ONLY=1` unless the caller already supplied a
value. This keeps host-side acceptance independent of a locally installed
DepthAI runtime; it is not OAK runtime evidence.

Each case invokes one fully qualified Rust test with `--exact`, verifies that
the named test actually ran, and stops at the first failure. A successful run
ends with `acceptance_result=pass` followed by the explicit
`claims_not_established` set. Preserve the complete stdout/stderr log with the
reviewed revision and lockfile when using it as acceptance evidence.

## Evidence matrix

| Boundary | Component evidence |
| --- | --- |
| Bundle construction | Exact retained and rendered byte counts/digests match the plan; the launch document is the final write in the offline staging simulation. Mandatory dataset byte/file/record ceilings, descriptor-relative free-space floor, and checked terminal reserve are retained by the typed launch. |
| Boot enablement | The base unit is not enableable. The enablement-only drop-in cannot replace the invariant `ExecStartPre` gate, and missing or drifted marker-bound bytes fail admission. |
| One production owner | The effective unit has one integrated production `ExecStart` and no standalone robot-server dependency; controller promotion and the control-socket ready barrier are linear and zero-gated. |
| Zero before ready | Inventory, disarmed, zero-requested, ready-stopped, and active states remain distinct. Initial zero must match the admitted controller session. |
| Watchdog, leases, and UART scheduling | The firmware feed gate requires one complete safe loop; controller and supervisor lease deadlines are exclusive and lead to a zero requirement. Reserved record-preserving TX capacity protects stop/applied receipts from telemetry saturation; the rendered candidate's intended 20 ms / 50 Hz runtime policy is cross-bound to a 100 Hz server/parser ceiling, retaining its explicit scheduling margin, and its baseline load is checked against the exact 115200 8N1 byte budget. The 100 Hz ceiling is not the deployed control cadence. Bounded UART turns and partial-write shutdown recovery are exercised at their software boundaries. |
| Unified console and browser loss | One typed surface projects map, pose, path, MPC timing, requested/actual authority, exact controller receipt, health, and stop state. Stale/lost connections visibly freeze the map and inhibit motion/persistence; stop-reducing actions remain available, the server deadman owns exact-zero fallback, and response IDs never cross sessions. |
| Fault stop | Clock regression and simulated transport failure latch typed faults; uncertain stop is never reported as a safe cancellation. |
| Map recovery | Atomic save/reload preserves exact map bytes; warm start requires exact dataset replay and never invents localization. |
| Shutdown and restart | Lifecycle zero precedes controller disarm, uncertain terminal stop remains faulted, and the control-socket owner joins. The OAK capture loop can kick the 60-second watchdog only after a new sole-accessory-owner four-joint health transaction; `Restart=on-failure` waits 15 seconds and for the preceding stop job, exact device admission rejects retained old owners, five failures in ten minutes trip the start limiter, and the console capability is atomically replaced rather than reused. |

The systemd assertions are source/effective-text component checks. The bundle
case uses a temporary directory as a simulated installation destination.
Neither executes PID 1 or performs a privileged filesystem installation.

## Claims deliberately not established

A passing component run does **not** establish:

- root-owned installation or effective Nano systemd behavior;
- a real cold power boot, scheduler/IO timing, or cleanup deadline;
- installed filesystem capacity, allocation-fragment behavior, or resistance
  to an external writer racing the retained free-space floor;
- device presence, exclusive OAK/serial ownership, or USB SuperSpeed;
- the STM32 independent watchdog oscillator/timing on the physical board;
- a physical emergency stop, motor-driver enable/fault wiring, stop distance,
  wheel direction, or PWM-to-velocity calibration;
- neck torque, head geometry, eye output, or an RGB camera stream;
- SLAM accuracy, occupancy correctness for the room, relocalization success,
  MPC tracking, collision avoidance, or performance.

Those remain separate attended Nano and wheels-off acceptance items in
`docs/nano-wheel-attach-gate-2026-07-23.md`. No wheel-attach milestone may be
issued from this software result alone.
