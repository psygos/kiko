# Canonical KEP2 eye firmware

This directory replaces the quarantined ASCII eye demo with the device side of
canonical KEP2. It is intentionally a nested Cargo workspace: the host
workspace does not build embedded firmware by default, while both sides import
the same `crates/kiko-eye-protocol` codec and domain types.

The firmware has two layers:

- `src/lib.rs` is `no_std`, allocation-free, safe Rust that can be tested on a
  host. It owns stream framing, exclusive sessions, sequence ordering, finite
  leases, fallback decisions, measured panel geometry, and rendering.
- `src/bin/rp2350.rs` is the RP2350A adapter. It owns USB CDC, OTP/TRNG startup
  inputs, the 60 Hz clock, two PIO state machines, two DMA channels, and GP15 /
  GP16 WS2812 output.

There is no legacy ASCII parser or compatibility mode. A malformed KEP2
record, unexpected device-direction message, disconnect, expired ownership,
clock regression, or internal renderer failure removes host ownership and
selects autonomous output. Graceful release requires the exact boot, epoch,
and next sequence.

## Protocol and timing semantics

The endpoint uses the canonical fixed-capacity `StreamDecoder` and `encode`
functions, so COBS framing, the zero delimiter, CRC-32C, reserved bytes, exact
payload lengths, and domain parsing are not reimplemented here.

Acquisition grants a 2,000 ms device-clock window for the first intent because
KEP2 `AcquireControl` has no lease field. Each admitted intent then replaces
that deadline with its exact 20–2,000 ms lease. Deadlines are exclusive:
`now == expires_at` is expired. Expiry relinquishes the session as well as
selecting autonomous rendering, so a silent connection cannot stay busy
forever.

Exact duplicate `ApplyIntent` frames return `DuplicateCached` with the original
admission time, expiry, and renderer sequence. They do not rerender or extend
the lease. Reusing a sequence with different content is rejected. A command
queued for at least its requested lease is rejected before admission. The
reported renderer sequence proves only that the bounded renderer intent slot
accepted the command; it is not optical evidence that LEDs displayed it.

The physical output can change only on a render/DMA frame. Consequently the
software decision changes at the exact lease deadline, while the panel can
retain the preceding pixels until the next 60 Hz sample and WS2812 transfer.
That latency has not been measured on hardware.

## Renderer contract

The measured 56-position circular layout is retained in chain order. Gaze is
KEP2 normalized `[-1000, 1000]`, with positive X to the image right and
positive Y upward, mapped with nearest rounding to a ±18 mm physical range.
The build-provisioned right-panel sign is applied only at the panel mounting
boundary. Gaze and lid movement use elapsed device milliseconds and explicit
rate limits, not a frame-count-dependent filter.

Pupil `0..1000` maps linearly from a 7 mm to 15 mm radius (500 = the legacy
11 mm neutral radius). Lid is closure (`0` open, `1000` closed); a blink is a
200 ms smooth close/open pulse edge-triggered by a new intent sequence. All
five KEP2 expression values have an exhaustive lid treatment. Host RGB,
brightness, and pupil values are all consumed.

Gamma is applied before host brightness and the hard per-channel ceiling of
56/255. Host tests exhaust every input channel and brightness value to prove
the ceiling. This software limit is not a substitute for a correctly sized
external 5 V supply, shared ground, level shifter, fusing, or physical current
measurement.

## Identity and release provisioning

Three identities have deliberately different sources:

1. `DeviceUid` is the RP2350 factory 128-bit OTP random number returned by
   `embassy_rp::otp::get_private_random_number()`, encoded big-endian. The same
   bytes become the 32-character USB serial descriptor. This OTP value is a
   stable identifier, not a secret. Commissioning must query KEP2 once under
   controlled conditions and pin the exact 16 bytes in Kiko's device manifest.
   If OTP read fails or returns zero, the firmware does not start USB control.
2. `FirmwareBuildId` is supplied as exactly 64 hexadecimal characters in
   `KIKO_EYE_FIRMWARE_BUILD_ID_HEX`. It must identify an immutable reviewed
   release input set and be recorded alongside that release. The build script
   rejects missing, malformed, or all-zero values. Do not use the synthetic
   value from verification commands for a deployable image.
3. `DeviceBootId` is a fresh RP2350 TRNG `u64` with its low bit set, making zero
   unrepresentable. It is probabilistically fresh per boot; it is not a
   persistent monotonic boot counter and this code does not claim otherwise.

`KIKO_EYE_RIGHT_X_SIGN` is also mandatory and must be exactly `1` or `-1`.
The legacy source used `1` but explicitly marked the physical polarity
unverified. A release must use a value established on the installed panels;
changing it changes rendered behavior and therefore requires a new build ID.

## Build and test

Host checks do not require embedded tooling:

```sh
cargo test --manifest-path firmware/eyes-kep2/Cargo.toml
cargo clippy --manifest-path firmware/eyes-kep2/Cargo.toml \
  --all-targets -- -D warnings
cargo fmt --manifest-path firmware/eyes-kep2/Cargo.toml -- --check
```

Install the RP2350 Arm target, select reviewed provisioning, then build:

```sh
rustup target add thumbv8m.main-none-eabihf
export KIKO_EYE_FIRMWARE_BUILD_ID_HEX='<64 reviewed hex characters>'
export KIKO_EYE_RIGHT_X_SIGN='1' # or -1, only after physical calibration
cargo build --manifest-path firmware/eyes-kep2/Cargo.toml \
  --release --target thumbv8m.main-none-eabihf --features rp2350 \
  --bin kiko-eyes-kep2-rp2350
```

The ELF is
`firmware/eyes-kep2/target/thumbv8m.main-none-eabihf/release/kiko-eyes-kep2-rp2350`.
Conversion to UF2 may use `picotool uf2 convert ... --family rp2350-arm-s`, but
the build command itself neither converts nor flashes an image. The separate
[`FABLE-NANO-HANDOFF-2026-07-21.md`](../../docs/FABLE-NANO-HANDOFF-2026-07-21.md)
records the exact earlier UF2 identity and the operator-observed physical KEP2
sequence. Rebuilding this source does not inherit that physical evidence or
permit reuse of the earlier firmware build ID.

Every successful application boot now renders a 2.4-second green Matrix cue
before the USB KEP2 endpoint enumerates. Delaying enumeration is intentional:
the firmware cannot truthfully acknowledge an applied eye intent while a boot
animation is overriding the panels. This also provides a deterministic
post-UF2-copy indication that the new application started.

The RP2350 application is not executing while ROM BOOTSEL owns the board, so
this firmware cannot animate the physical panels during the actual UF2 file
copy. A host updater may show a separate progress animation and may request a
pre-reset cue from an already running application, but it must label the
BOOTSEL interval as bootloader-owned and must wait for the post-boot Matrix cue
and exact KEP2 identity before reporting success. Continuous physical Matrix
animation across that interval would require a reviewed custom bootloader or
an independent display controller; this repository does not pretend otherwise.

## Evidence boundary before deployment

Host tests and an RP2350 release link prove compilation and software state
transitions. They do not establish:

- that this robot's left/right data harness is GP15/GP16 or that the chosen
  right-panel polarity is correct;
- physical LED chain ordering, color order, current draw, supply integrity,
  brightness, thermal behavior, or optical appearance;
- the actual board's OTP UID, TRNG behavior, USB serial path/permissions on the
  Nano, or disconnect behavior;
- end-to-end KEP2 identity/acquire/apply/release timing over that USB cable; or
- that the semantic expression tuning looks natural on the installed face.

Those are commissioning tests, not facts inferred from a successful build.
No performance benchmark was run, so this directory makes no throughput or
CPU-improvement claim. `embassy-rp 0.10` currently brings an upstream
future-incompatibility warning through `proc-macro-error2 2.0.1`; the pinned
build succeeds today, but the dependency must be revisited before a toolchain
that turns that warning into an error.
