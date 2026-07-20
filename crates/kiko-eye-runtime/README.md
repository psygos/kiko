# Kiko eye runtime

`kiko-eye-runtime` is the sole host-side serial owner for a KEP2 eye
controller. It wraps `kiko-expression-runtime::EyeSession`; it does not replace
or duplicate the protocol state machine.

## Boundary and ownership

Callers first parse `EyeRuntimeConfigInput` into `EyeRuntimeConfig`. Parsing
rejects generic Linux tty names, discovery paths, zero identities, zero or
reused nonces, zero control epochs, unknown or incomplete capabilities,
unbounded timeouts/retries/noise, and a host round-trip budget that cannot fit
strictly inside the configured firmware lease. The only supported port forms
are:

- Linux: `/dev/serial/by-id/<one component>`
- macOS: `/dev/cu.<one component>`

The production constructor opens exactly that path, claims OS exclusivity,
sets the configured baud rate with 8 data bits, no parity, one stop bit, and no
flow control, then reads those settings back before any KEP2 traffic. It does
not scan ports or fall back to another device. The KEP2 UID, firmware build,
capabilities, boot ID, nonces, epoch, and non-wrapping host intent sequence are
then checked by `EyeSession`.

The caller must allocate identity/acquire nonces and the control epoch for this
startup and must not reuse persisted values. Parsing proves that they are
non-zero and that the two nonces differ; it cannot infer cross-process
freshness from integers alone.

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
