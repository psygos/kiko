# Kiko device inventory

`kiko-device-inventory` is the transport-independent boundary between a weak
inventory document, an external probe report, and Kiko's startup policy. It
parses each report once into bounded domain types and then performs an exact,
allocation-free comparison of those parsed values.

## Contract

An expected V1 manifest always identifies one robot, one OAK device, one STM32
controller, at least one calibration artifact, and at least one plant artifact.
The head and eye are optional because those accessories can legitimately be
absent from a physical build. If either is declared, its complete contract is
required. An observed report may omit any device or artifact so that absence is
represented as a mismatch rather than replaced with a fallback.

The parser rejects:

- unknown schema versions, empty, zero, malformed, or oversized identities;
- generic or relative serial names instead of one
  `/dev/serial/by-id/<identity>` path;
- remote or malformed STM32 control endpoints;
- duplicate physical paths, head servo IDs, artifact IDs, or artifact digests;
- zero protocol IDs, boot IDs, firmware builds, hardware fingerprints, or
  SHA-256 values;
- unknown protocol capability bits and oversized device/artifact collections;
- expected STM32, head, or KEP2 declarations that contradict the protocol
  contract compiled into this workspace.

The resulting domain objects expose getters only. OAK MXIDs are normalized to
uppercase hexadecimal, and serial/socket identities reject alias-producing dot
segments and non-canonical TCP ports. Comparison borrows both parsed snapshots,
is infallible for those domain types, and accumulates every mismatch in
deterministic field and artifact order without heap allocation or copying the
bounded identity storage. It does not accept a different device, build,
capability set, path, or digest as a substitute. STM32 and eye boot IDs are
retained in observed identities but are not compared: they are intentionally
per-boot values and have no stable counterpart in the manifest.

## Evidence boundary

This crate does **not** access the filesystem, udev, USB, serial ports, sockets,
OAK hardware, STM32 hardware, head servos, eye firmware, or artifact contents.
Both DTOs contain caller-supplied claims. Successful parsing proves only that a
claim is structurally valid; an exact comparison proves only that two claims
agree. Neither result proves physical identity, connectivity, firmware
authenticity, artifact authenticity, readiness, calibration quality, or safe
motion. A host integration must obtain observed values from authoritative
protocol handshakes and hash the actual artifacts before constructing the
observed DTO.

## Verification

From the workspace root:

```sh
cargo test -p kiko-device-inventory
cargo clippy -p kiko-device-inventory --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p kiko-device-inventory --no-deps
cargo check -p kiko-device-inventory --target aarch64-unknown-linux-gnu
cargo bench -p kiko-device-inventory --bench inventory -- 100000
```

The benchmark reports parse-plus-compare and already-parsed comparison timing
for the machine that ran it. It is a reproducible measurement harness, not a
cross-machine latency guarantee and not evidence of a performance improvement.
