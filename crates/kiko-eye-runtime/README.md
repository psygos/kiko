# Kiko eye runtime

`kiko-eye-runtime` is the sole host-side serial owner for a KEP2 eye
controller. It wraps `kiko-expression-runtime::EyeSession`; it does not replace
or duplicate the protocol state machine.

## Boundary and ownership

Callers first parse `StaticEyeRuntimeConfigInput` into
`StaticEyeRuntimeConfig`. This restart-safe deployment policy deliberately has
no identity nonce, acquisition nonce, or control epoch. Static parsing rejects
generic Linux tty names, discovery paths, zero identities, unknown or
incomplete capabilities, unbounded timeouts/retries/noise, and a host
round-trip budget that cannot fit strictly inside the configured firmware
lease. The only supported port forms are:

- Linux: `/dev/serial/by-id/<one component>`
- macOS: `/dev/cu.<one component>`

The production constructor opens exactly that path, claims OS exclusivity,
sets the configured baud rate with 8 data bits, no parity, one stop bit, and no
flow control, then reads those settings back before any KEP2 traffic. It does
not scan ports or fall back to another device. The KEP2 UID, firmware build,
capabilities, boot ID, nonces, epoch, and non-wrapping host intent sequence are
then checked by `EyeSession`.

Immediately before each actor start, the caller must pass an
`EyeSessionMaterialGenerator` to `StaticEyeRuntimeConfig::new_session`. The
production `OsEyeSessionMaterialGenerator` draws all 160 bits from the
operating system CSPRNG and rejects zero identifiers or equal phase nonces. It
returns one non-cloneable `EyeRuntimeConfig`, which the actor start consumes.
The static policy cannot start an actor directly, and constructing a second
actor configuration requires another generator call.

Fresh random material makes accidental reuse cryptographically improbable; it
does not provide a deterministic proof of uniqueness across process starts.
Such a proof would require a durable monotonic allocator with its own crash and
storage-failure semantics. Custom generators therefore remain responsible for
not reusing material across calls or starts.

Both spawn paths first require an active Tokio runtime. Production proves this
before opening the serial path, so a missing executor is a typed error and
cannot change physical TTY state.

`start_serial_eye_actor` requires the caller to inject the monotonic clock. The
same clock epoch must generate every `PreparedEyeIntent` timestamp; the
constructor never creates a private epoch that could be confused with RGB
freshness time. A `TokioClock` is provided and cloneable when the host wants one
shared Tokio-backed origin.

The handle is deliberately non-cloneable. `apply_intent` needs `&mut self`, the
mailbox has capacity one, and the actor processes one intent through its result
before accepting another. Canceling an apply future drops its response channel;
if firmware admission nevertheless completes, the actor records that admission,
enters fallback, and attempts release rather than hiding the uncertainty.

## Framing and timing

KEP2 COBS records are decoded incrementally into fixed-size protocol buffers.
Bytes read beyond one delimiter are retained for the next exchange. A bounded
number of empty delimiters may be treated as line noise; every malformed or
oversized non-empty record is a fault. Responses received before their request
write completed are rejected. Read, write, and flush deadlines are exclusive,
and a transport that reports success after a deadline is still rejected.

Partial writes continue within the same attempt and deadline. A whole-frame
retry is permitted only after a timeout or interruption with exactly zero known
progress. Positive progress, an invalid transport report, or a failed flush is
transmission uncertainty and is never retransmitted. Retry history is bounded
by the parsed attempt limit and retained in the result.

Prepared expression freshness is checked when an intent is submitted and
again when its firmware result is handled. Device lease time and host monotonic
time remain separate domains.

## Identity-only commissioning probe

`kep2_identity_probe` bootstraps the expected eye UID after a firmware flash
without acquiring renderer control. It opens one exact stable path exclusively,
reads back the canonical 115200/8-N-1/no-flow-control configuration, sends one
OS-random nonzero `IdentityQuery`, and accepts exactly one canonical
`IdentityReport` that echoes that challenge. It never sends `AcquireControl`,
`ApplyIntent`, or `ReleaseControl`.

On Linux:

```bash
cargo run --locked -p kiko-eye-runtime --bin kep2_identity_probe -- \
  --serial-device /dev/serial/by-id/<exact-eye-identity> \
  --timeout-ms 5000
```

On macOS, pass the exact `/dev/cu.<identity>` callout instead. The JSON result
contains the manifest-ready UID/build byte arrays and hex strings, current boot
ID, device uptime, capability bits, challenge, and serial-setting readback. It
is a nonce-bound firmware claim only; it does not prove optical output, panel
wiring, or fallback visibility. Stop every eye owner before running it.

## Eye-only expression commissioning

After copying the exact UID, build ID, and capability bits from the identity
probe into the deployment manifest, `kep2_eye_commission` exercises the normal
typed actor rather than reimplementing KEP2. It opens only the exact eye serial
path, verifies that complete pinned identity, acquires renderer control with
fresh OS-generated session material, and runs one fixed, deliberately obvious
6.55-second optical recipe: bright white center, two full-left red holds, two
full-right blue holds, three separately triggered white blinks, and neutral.
The repeated holds distinguish a genuine panel update from the firmware's
subtle autonomous gaze. The command has no base, head, or camera interface.

```bash
cargo run --locked --release -p kiko-eye-runtime --bin kep2_eye_commission -- \
  --serial-device /dev/serial/by-id/<exact-eye-identity> \
  --expected-device-uid-hex <32-hex-digits> \
  --expected-firmware-build-id-hex <64-hex-digits> \
  --expected-capabilities-bits <u32-decimal> \
  --execute-eye-sequence
```

Every apply has a 1.8-second firmware lease. The configured maximum response
wait, complete write-attempt budget, and longest requested hold total 1.6
seconds, leaving 200 milliseconds of protocol-budget margin before that lease;
scheduler stalls remain bounded by firmware fallback. Normal completion, and Ctrl-C when
observed during a visual hold, request a graceful release and require the
actor's final admission count, last admission, release report, and termination
reason to agree. Transport operations are bounded; after any ungraceful host
loss the final lease bounds the time until firmware autonomous fallback. The
resulting JSON records serial readback, identity/acquisition binding, every
exact requested intent and firmware admission, and confirmed release. Those
are protocol facts, not proof that either physical panel emitted the intended
pixels; the operator must still confirm the visible sequence.

## Failure and evidence semantics

Timeout, stale/future expression input, malformed or unexpected messages,
identity/build/nonce/boot/epoch/sequence mismatch, reboot, transport failure,
explicit cancellation, caller cancellation, and handle drop all fail closed.
`EyeRuntimeFault` retains:

- the exact runtime cause;
- the exact `EyeSessionFault` and any session-provided release;
- the best-effort cleanup write outcome; and
- when graceful release was already written before its response failed, that
  prior release request and write evidence.

`FirmwareAdmissionEvidence` means only that a correctly bound KEP2 result said
the intent was admitted with the reported device-clock lease and renderer
sequence. It does **not** prove that either display showed the requested pixels.
Likewise, a cleanup `WriteCompleted` proves only host write/flush completion,
not firmware receipt or autonomous fallback visibility.

This runtime requires firmware that implements canonical KEP2. The quarantined
legacy ASCII eye demo and its prebuilt image are not compatible runtime
authorities and are never selected as a fallback.
