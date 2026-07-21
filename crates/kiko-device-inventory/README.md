# Kiko device inventory

`kiko-device-inventory` owns the typed boundary between a weak inventory
document, an external probe report, and Kiko's startup policy. Its domain model
is transport-independent. On Unix hosts it also provides bounded file loading
and artifact hashing for Linux and macOS.

Weak values are parsed into bounded domain types once and compared exactly.
Already-parsed inventory comparison is allocation-free and borrows both
snapshots. An exact comparison can be consumed to mint an owned
`ExactInventoryAdmission`; a mismatch instead produces an owned, bounded
`InventoryMismatchReport` that retains both complete parsed snapshots.

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
`InventoryComparison` equality is defined solely by the ordered mismatch
prefix, so these intentionally ignored boot IDs cannot change comparison
equality through the internal references retained for admission conversion.

`ExactInventoryAdmission` has private fields and is constructed only through
an exact comparison. Production callers that own both parsed snapshots should
use `admit_exact_inventory`, which performs that comparison and moves both
inputs into the result without cloning them. The borrowed
`InventoryComparison::into_exact_admission` path remains available when a
caller cannot transfer ownership. Either result owns immutable snapshots, so
its evidence remains available after the original inputs are dropped and
cannot silently refer to subsequently mutated DTOs. It proves exact snapshot
agreement only; it does not prove ongoing device liveness. On failure, the
mismatch report owns the same full snapshots and can reconstruct the complete
typed comparison without truncated strings, fixed-message buffers, or a
dependency on the original inputs' lifetimes. Only the failure path boxes this
fixed-capacity payload, so the `Err` handle is one pointer. The `Ok` admission
remains inline and owns both complete snapshots; it is multi-KiB and therefore
still determines the overall `Result` size.

## OAK evidence semantics

The expected and observed OAK DTOs are deliberately different Rust types.
`OakManifestV1Dto` is an expectation; `ObservedOakV1Dto` is a caller-supplied
probe result. The latter must be assembled from the exact connected-device
MXID and `oak_sys::depthai_build_metadata()`:

- `compiled_depthai_header_sdk_version` and
  `compiled_depthai_header_sdk_commit` preserve `dai::build::VERSION` and
  `dai::build::COMMIT` from the `depthai/build/version.hpp` used to compile
  `oak_device.cpp`;
- `compiled_depthai_header_embedded_device_artifact_version` and
  `compiled_depthai_header_embedded_bootloader_artifact_version` preserve the
  corresponding `dai::build::DEVICE_VERSION` and
  `dai::build::BOOTLOADER_VERSION` header constants.

These four values prove only compiled-header metadata. Kiko's build can resolve
DepthAI include and library roots independently, and this query does not
inspect the linked or dynamically loaded library. It therefore does **not**
prove that the runtime library matches the compiled header. The embedded
artifact fields are also **not** readbacks of firmware or bootloader code
currently executing on the physical OAK. The inventory makes neither claim.
The former `runtime_provenance`, `sdk_build_provenance`, and
`adapter_build_provenance` strings were removed because their sources and
semantics could not be established by the runtime boundary. Host executable,
runtime-library, or adapter provenance belongs in a separately hashed
deployment artifact until Kiko has a typed, reproducible identity source for
it.

### Draft V1 migration status

This field correction applies to an unreleased draft V1, not a deployed wire
contract. At the time of the correction, `crates/kiko-device-inventory` was
absent from `main`, no repository tag contained its introducing commits, the
integration branch had not been pushed, and no production inventory caller
constructed an observed OAK report. The earlier ambiguous keys therefore had
neither a released decoder nor a valid admission meaning.

Local branch-only draft manifests must replace those three keys with the four
exact compiled-DepthAI-header fields above. The loader uses
`deny_unknown_fields`, so an old draft fails closed instead of being silently
reinterpreted. After V1 is published or deployed, every incompatible wire
change must introduce a new schema version and correspondingly named
DTO/domain entrypoint.

## Unix host boundary

`load_expected_manifest_v1_from_slice` accepts at most 64 KiB of JSON. It
deserializes directly into `DeviceInventoryManifestV1Dto`, rejects duplicate
fields, unknown fields at every DTO level, malformed values, and trailing JSON,
then calls `DeviceInventoryManifestV1::parse`. That existing parser remains the
only admission path into the manifest domain.

`load_expected_manifest_v1_file` applies the same parser after opening one
absolute, canonical path without following a symlink in any component. It
accepts only a regular file, checks the metadata length before allocation,
streams no more than 64 KiB, and reports a length change observed during the
read. Relative paths, dot components, repeated or trailing separators, paths
over 1,024 bytes, non-regular files, allocation failures, and I/O failures are
distinct typed errors. No alternate manifest path is tried.

The loaded value also retains SHA-256 of the exact admitted JSON byte slice.
That identity is computed during the one bounded read; callers do not reopen
the path. Whitespace and key order intentionally affect it, so it identifies
the reviewed file representation rather than claiming semantic JSON
canonicalization.

`ArtifactFileBindingSet::parse` consumes weak caller-declared paths once. It
requires calibration and plant entries, enforces the per-kind limits and a
globally unique artifact-ID/path set, and returns only bounded parsed bindings.
Each relative path is at most 512 bytes and 64 components, uses `/`, and
contains no empty, dot, parent, NUL, or backslash component.

`hash_manifest_artifacts` consumes that parsed set and only then binds it to
the artifact kind and ID already present in a parsed manifest. The count and
membership must match exactly; it does not reparse IDs or paths. The artifact
root must be an absolute canonical path of at most 1,024 bytes.

The root and every artifact path component are opened relative to anchored
directory descriptors with no-follow semantics. Only regular files of at most
128 MiB are hashed. SHA-256 is computed in a bounded 64 KiB streaming buffer;
the result retains the manifest digest, observed digest, exact relative path,
and bytes read. A digest difference is successful identity evidence with
`content_matches_manifest() == false`, not a hashing failure.

After all artifact contents match, `exact_calibration_bundle_sha256` derives a
domain-separated V1 digest from the sorted calibration artifact IDs and their
observed SHA-256 values. It allocates no intermediate serialization and refuses
to produce readiness evidence for any content mismatch. Plant artifacts stay
covered by exact inventory comparison and the enclosing manifest-content
identity; they are not mislabeled as calibration inputs.

## Evidence boundary

The Unix functions access only the explicitly supplied manifest or artifact
paths. This crate does **not** access udev, USB, serial ports, sockets, OAK
hardware, STM32 hardware, head servos, or eye firmware. Observed device DTOs
remain caller-supplied claims.

Successful parsing proves structural validity. Exact inventory comparison
proves only agreement between two parsed snapshots. A SHA-256 match means the
bytes read produced the digest declared by the manifest; an untrusted manifest
can truthfully describe malicious content, so the result does not prove origin,
signature, authorization, provenance, or authenticity. It also does not
establish physical identity, connectivity, readiness, calibration quality, or
safe motion.

In particular, an observed OAK DTO remains a caller claim even though its
field names correspond exactly to APIs available from `oak-sys`. This crate
cannot prove that the caller sourced those values from the same open device;
the production owner must construct the DTO directly from its one OAK handle
and linked-build query, then retain the resulting admission capability in that
same startup epoch.

Hashing is not an atomic filesystem snapshot. The code detects a file-length
change between metadata and end of read, but a concurrent writer can replace
bytes without changing the length and can make a streamed digest represent
more than one write epoch. Production artifact trees must therefore be made
immutable before startup admission. The returned digest is content identity
for the bytes the reader observed, with no stronger claim.

## Verification

From the workspace root:

```sh
cargo test -p kiko-device-inventory
cargo clippy -p kiko-device-inventory --all-targets --no-deps -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc -p kiko-device-inventory --no-deps
cargo check -p kiko-device-inventory --target aarch64-unknown-linux-gnu
cargo bench -p kiko-device-inventory --bench inventory -- 100000
```

The benchmark reports parse-plus-compare and already-parsed comparison timing
for the machine that ran it. It is a reproducible measurement harness, not a
cross-machine latency guarantee and not evidence of a performance improvement.
