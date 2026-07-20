# Kiko eye protocol

`kiko-eye-protocol` is the allocation-free wire contract shared by the Nano
host and RP2350 eye firmware. It deliberately contains no serial, USB, clock,
renderer, or operating-system code.

Version 2 replaces the experimental ASCII commands with:

- COBS records terminated by one zero byte;
- a fixed header and exact message lengths;
- CRC-32C over the header and payload;
- nonzero device, build, boot, and control-session identities;
- distinct nonzero-nonce-bound identity discovery and explicit control
  acquisition;
- monotonically sequenced intents with bounded device-relative leases;
- explicit release and applied-result messages; and
- strict rejection of reserved fields, unknown values, trailing bytes, and
  oversized records.

The lease begins when the firmware admits the command on its monotonic clock.
Device-clock instants and renderer-frame sequences have distinct wire-domain
types, so they cannot be mixed with host time or intent sequences. The applied
report returns its device-clock instant and checked expiry. Its result fields
are private and checked as one value: admitted outcomes carry a bounded lease,
while rejected, released, and fallback outcomes have a zero interval.
It proves only that the firmware accepted the requested eye state; it does not
claim that a particular LED was optically observed.

The streaming decoder enters a discard state after an oversized record and
does not parse any suffix before the next delimiter. Link loss or lease expiry
must return the renderer to its autonomous behavior.
