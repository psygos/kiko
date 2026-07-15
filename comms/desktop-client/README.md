# Robot Control Desktop Client

This Tauri application sends leased wheel-PWM commands to `robot-server`, receives exact sequence acknowledgements, polls raw encoder odometry, and displays the server's optional MJPEG stream.

## Behavior

- Command attempts are separated by at least 40 ms; network round-trip time is additional.
- Each acknowledgement proves that the server accepted one sequence. It does not prove that the controller applied the PWM.
- Each desktop connection receives a local stream generation. UI commands, acknowledgements, and failure events carry that local provenance, and the renderer ignores events from superseded streams. This is not a network session nonce or replay defense.
- Rapid key updates are coalesced into one ordered desired-PWM stream. Nonzero state is refreshed by the renderer every 50 ms and becomes expired after 150 ms without a refresh; the Rust command thread observes that deadline immediately before its next send attempt, so prior UDP round-trip time and the 40 ms minimum inter-attempt delay add detection latency. An acknowledged stop holds movement input at zero until the stop attempt finishes.
- PWM values are validated in the shared `robot-protocol` crate and are never silently clamped. The Shift boost is explicitly capped at the protocol maximum of 100%.
- The local renderer-to-command-thread lease and the dashboard-to-server UDP lease are each 150 ms. The server forwards a separate 50 ms controller lease every 20 ms while the UDP lease remains active.
- Spacebar sends and waits for an acknowledged zero-PWM command. This is not an independent hardware E-stop.
- Disconnect attempts an acknowledged zero-PWM command before discarding the local stream. If that attempt fails, the UI reports the failure and only the software command leases remain as a fallback.
- A zero-PWM command may preempt an active lease from another UDP source; nonzero preemption is rejected. After a source transition, that source must advance its sequence before requesting motion. This fail-safe ordering is not authentication and permits unauthenticated stop denial of service.
- Odometry is displayed only as estimated wrapping extended encoder ticks, modulo-i16 sample deltas, wrapping controller uptime, and the sample's age since the server received it. No freshness threshold is inferred. A pending encoder overflow is extended using the timer's current direction, which can be wrong if the input reversed before the snapshot; multiple missed wraps are unrecoverable. The repository does not provide calibrated wheel geometry or ticks-per-meter, so the client does not claim metric pose, distance, or speed.

## Controls

- Movement: Arrow keys or WASD
- PWM boost: Hold Shift while moving
- Zero-PWM stop: Spacebar
- PWM magnitude: Slider in the control panel

## Development

```bash
npm install
npm run dev
```

Build the desktop package with:

```bash
npm run build
```

The desktop host must provide Tauri's GTK/WebKit development dependencies. See [SETUP.md](SETUP.md) for the server, serial, and safety boundaries.
