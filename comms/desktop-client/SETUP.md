# Robot Control Client Setup

## Robot Server

From the workspace root:

```bash
RUST_LOG=info cargo run -p robot-server
```

The server requires an accessible controller serial port at 115200 baud. It probes the explicit device list in `comms/robot-server/src/protocol.rs` and exits if none can be opened or an opened port reaches EOF. `ffmpeg` and a supported `/dev/video*` device are optional; without them, the command and odometry services still run and `/status` reports video as unavailable. Video is not reported as streaming until the first complete JPEG arrives. An MJPEG frame is limited to 4 MiB, and a viewer that falls behind the four-frame broadcast queue is closed instead of having frame loss hidden.

Services:

- UDP 8080: exact leased command packets and sequence acknowledgements
- HTTP 3030: status, diagnostics, raw odometry, and optional MJPEG video
- Serial 115200 baud: leased PWM commands and controller reports

## Desktop Client

Install Node dependencies and run Tauri:

```bash
cd comms/desktop-client
npm install
npm run dev
```

The host must have Rust, Node.js, and the GTK 3/WebKitGTK development packages required by Tauri 1. Package names vary by distribution. A missing `gdk-3.0.pc` means the GTK development package is not installed or is outside `PKG_CONFIG_PATH`.

Enter the robot host plus UDP and HTTP ports in the connection controls. A connection is reported only after an exact UDP acknowledgement. Video and odometry use separate HTTP endpoints; odometry request failures are surfaced separately from command-channel failures.

The desktop binds each control invocation, acknowledgement event, and failure event to a local stream generation, coalesces rapid key changes in order, and rejects work or UI events from superseded connections. While PWM is nonzero, the renderer refreshes its desired state every 50 ms. The desired state becomes expired after 150 ms without a refresh, but the Rust command thread checks that deadline only before its next send attempt; prior UDP round-trip time and the 40 ms minimum inter-attempt delay add detection latency. The generation is process-local and is not transmitted in the UDP packet, so it provides no authentication or network replay protection.

## Safety Boundary

- Renderer-to-command-thread desired-PWM lease: 150 ms, refreshed every 50 ms only while nonzero and observed on the next command attempt rather than by a hard-deadline timer
- Dashboard-to-server UDP command lease: 150 ms
- Server-to-controller active lease: 50 ms, refreshed every 20 ms
- Server-to-controller zero-PWM lease: 1 ms
- PWM domain: -100% through 100%, rejected rather than clamped
- Spacebar: acknowledged zero-PWM network command, not a hardware E-stop
- Disconnect: attempts an acknowledged zero-PWM command, then discards the local stream even when acknowledgement fails
- Controller lease expiry: software-polled stop after main-loop work and a minimum 5 ms loop delay, not a hard deadline
- UDP source transition: zero PWM may preempt another source; nonzero preemption is rejected, and a different source must acquire with zero before a later increasing sequence can request motion
- Sequence ordering: modulo-u32 duplicate, older, and exactly half-range-ambiguous values are rejected; only an unambiguously newer value advances an existing source session

The current protocol has no authentication, network session nonce, CRC, independent hardware watchdog, driver-fault input, brake input, or physical E-stop path. UDP source addresses can be spoofed, and captured command traffic can be replayed. Zero-only source preemption is fail-safe for motor output but allows unauthenticated stop denial of service; it is not an authorization boundary, and a sender can acquire with zero and then send motion. Do not represent the network stop as protection against Jetson, network, firmware, power-stage, or wiring failure.

## Odometry Boundary

The controller reports raw 16-bit quadrature counters extended into estimated wrapping i64 totals, signed modulo-i16 deltas per sample, and wrapping u32 uptime milliseconds. The HTTP response also carries the elapsed milliseconds since the server received that serial sample; the client displays this age without imposing an undocumented freshness threshold. A pending encoder overflow is extended using the timer's current direction, which can be wrong if the input reversed before the snapshot, and multiple wraps missed while interrupts are unavailable cannot be reconstructed. The modulo delta is ambiguous at or beyond half the 16-bit counter range between samples. No calibrated wheel radius, track width, encoder ticks-per-revolution, or ticks-per-meter is present in this repository, so the client intentionally does not derive metric motion.
