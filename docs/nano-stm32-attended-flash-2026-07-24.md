# Nano STM32 attended flash procedure — 2026-07-24

Revision status: 2026-07-27.

The current software-only evidence is bound to Kiko revision
`3f262f1a9b377448c903ed61305b28498b2b2c7f`. No STM32 candidate image was
flashed, no Fable process was handed off, and no robot device was opened while
collecting that evidence. A fresh operator attestation is still absent.
Consequently Gate A remains closed: motor power must be independently
disconnected and the wheels removed before the attended transport, OAK
SuperSpeed/live-stream, head/accessory, and SLAM/occupancy/Rerun/GUI checks.
The evidence below proves only that the candidate can be reproduced and fits
the repository flash layout; it makes no live-hardware or motion claim.

This procedure is specific to the installed STM32F446, ST-Link/V2-1, and the
repository memory contract at the reviewed Kiko revision. It is not an
unattended deployment script and does not authorize motion.

The hard physical precondition is stronger than “wheels off”: both wheels are
removed, motor power is independently disconnected and remains disconnected,
the head is supported, the motor area is clear, and an operator remains
present. Every debugger connection can reset or resume the target. Stop if
NRST/connect-under-reset is unavailable; do not silently fall back to a normal
attach.

## Memory contract

The F446 has 512 KiB of main flash:

```text
0x08000000..0x08060000  executable image, 384 KiB, sectors 0..=6
0x08060000..0x08080000  Kiko boot journal, 128 KiB, sector 7
```

`embedded/memory.x` ends executable flash at `0x08060000`. A main image that
crosses that address destroys the journal sector. `st-flash erase` is a mass
erase and is forbidden.

The audited Nano tools are:

```text
st-flash 1.7.0
OpenOCD 0.11.0
GNU Arm objcopy/readelf 2.38
```

Recheck those versions and enumerate exactly one expected ST-Link through
USB/sysfs at the attended session. Do not use a normal-attach debugger probe
before the first connect-under-reset backup.

## Fresh backup before any write

Run every command block in one fail-fast shell. Use a new mode-`0700` evidence
directory and never retry in that directory: a failed phase requires a new
directory and a complete restart from a fresh backup of the target's
then-current state. Do not restore or overwrite the prior backup. This prevents
a skipped or failed read from leaving an older output eligible for comparison.
`st-flash read` and OpenOCD `dump_image` otherwise truncate an existing output
file.

```bash
set -euo pipefail
umask 077

REPO=/ABSOLUTE/CLEAN/KIKO/CHECKOUT
EVIDENCE_DIR=/ABSOLUTE/NEW/NONEXISTENT/EVIDENCE-DIRECTORY
EXPECTED_FIRMWARE_REVISION=6cc59a1a3972c44df77dfd2cc02920ba40d896a2
EXPECTED_MOTOR_INERT_ELF_SHA256=fe1f055c076d700fd65d0f02db2a55163f82b7789b3bb2bf8b0ee25ede130dcc
EXPECTED_MOTOR_INERT_MAIN_SHA256=270e553f5c18a53393f0234f334d3ccc71be32ac7827240b54c939c6d6def38d
STLINK_SERIAL=066EFF313946303143221230
STM32_SERIAL_BY_ID=/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066EFF313946303143221230-if02
EXPECTED_CONTROLLER_UID_HEX=2c0018001750314242353320
readonly EXPECTED_ACTUATOR_CONFIG_FINGERPRINT_HEX=4b494b4f2d4e4f2d4143542d56312121
CARGO=/home/makerspace/.cargo/bin/cargo
RUSTC=/home/makerspace/.cargo/bin/rustc
PYTHON=/usr/bin/python3
test -f "$REPO/Cargo.lock"
test -x "$CARGO"
test -x "$RUSTC"
test -x "$PYTHON"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_FIRMWARE_REVISION"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test "$EXPECTED_ACTUATOR_CONFIG_FINGERPRINT_HEX" = \
  "4b494b4f2d4e4f2d4143542d56312121"
test -L "$STM32_SERIAL_BY_ID"
test -c "$(readlink -f "$STM32_SERIAL_BY_ID")"
test ! -e "$EVIDENCE_DIR"
install -d -m 0700 "$EVIDENCE_DIR"
cd "$EVIDENCE_DIR"

test ! -e cargo-version.txt
test ! -e rustc-version.txt
test ! -e python-version.txt
test ! -e st-flash-version.txt
test ! -e openocd-version.txt
test ! -e objcopy-version.txt
test ! -e readelf-version.txt
"$CARGO" --version >cargo-version.txt
"$RUSTC" --version --verbose >rustc-version.txt
"$PYTHON" --version >python-version.txt 2>&1
/usr/bin/st-flash --version >st-flash-version.txt
/usr/bin/openocd --version >openocd-version.txt 2>&1
/usr/bin/arm-none-eabi-objcopy --version >objcopy-version.txt
/usr/bin/arm-none-eabi-readelf --version >readelf-version.txt

test "$(cat cargo-version.txt)" = \
  "cargo 1.88.0 (873a06493 2025-05-10)"
test "$(sed -n '1p' rustc-version.txt)" = \
  "rustc 1.88.0 (6b00bc388 2025-06-23)"
test "$(sed -n '2p' rustc-version.txt)" = "binary: rustc"
test "$(sed -n '3p' rustc-version.txt)" = \
  "commit-hash: 6b00bc3880198600130e1cf62b8f8a93494488cc"
test "$(sed -n '5p' rustc-version.txt)" = \
  "host: aarch64-unknown-linux-gnu"
test "$(cat python-version.txt)" = "Python 3.10.12"
test "$(cat st-flash-version.txt)" = "v1.7.0"
test "$(sed -n '1p' openocd-version.txt)" = \
  "Open On-Chip Debugger 0.11.0"
test "$(sed -n '1p' objcopy-version.txt)" = \
  "GNU objcopy (2.38-3ubuntu1+15build1) 2.38"
test "$(sed -n '1p' readelf-version.txt)" = \
  "GNU readelf (2.38-3ubuntu1+15build1) 2.38"
sha256sum \
  cargo-version.txt \
  rustc-version.txt \
  python-version.txt \
  st-flash-version.txt \
  openocd-version.txt \
  objcopy-version.txt \
  readelf-version.txt

OPENOCD_SCRIPTS=/usr/share/openocd/scripts
OPENOCD_INTERFACE=/usr/share/openocd/scripts/interface/stlink.cfg
OPENOCD_TARGET=/usr/share/openocd/scripts/target/stm32f4x.cfg
OPENOCD_SWJ_DP=/usr/share/openocd/scripts/target/swj-dp.tcl
OPENOCD_MEM_HELPER=/usr/share/openocd/scripts/mem_helper.tcl
test -f "$OPENOCD_INTERFACE"
test -f "$OPENOCD_TARGET"
test -f "$OPENOCD_SWJ_DP"
test -f "$OPENOCD_MEM_HELPER"
sha256sum \
  "$OPENOCD_INTERFACE" \
  "$OPENOCD_TARGET" \
  "$OPENOCD_SWJ_DP" \
  "$OPENOCD_MEM_HELPER"

test ! -e stlink-usb-inventory.json
"$PYTHON" -c '
import json
import os
import pathlib
import sys

serial_path = pathlib.Path(sys.argv[1])
expected_serial = sys.argv[2]
matches = []
for candidate in pathlib.Path("/sys/bus/usb/devices").iterdir():
    try:
        vendor = (candidate / "idVendor").read_text(encoding="ascii").strip()
        product = (candidate / "idProduct").read_text(encoding="ascii").strip()
    except (FileNotFoundError, NotADirectoryError, OSError):
        continue
    if (vendor, product) != ("0483", "374b"):
        continue
    try:
        serial = (candidate / "serial").read_text(encoding="ascii").strip()
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"ST-Link USB identity has no readable serial: {candidate}") from error
    matches.append((candidate.resolve(), serial))

if len(matches) != 1:
    raise ValueError(f"expected one 0483:374b ST-Link, found {len(matches)}")
usb_path, observed_serial = matches[0]
if observed_serial != expected_serial:
    raise ValueError(
        f"ST-Link serial differs: expected {expected_serial!r}, "
        f"got {observed_serial!r}"
    )

tty_name = pathlib.Path(os.path.realpath(serial_path)).name
tty_sysfs = pathlib.Path("/sys/class/tty", tty_name, "device").resolve()
if usb_path != tty_sysfs and usb_path not in tty_sysfs.parents:
    raise ValueError(
        f"persistent VCP {serial_path} is not under the exact ST-Link USB device"
    )

json.dump(
    {
        "schema_version": 1,
        "enumeration_kind": "usb_sysfs_no_target_attach",
        "usb_vendor_id_hex": "0483",
        "usb_product_id_hex": "374b",
        "stlink_serial": observed_serial,
        "persistent_vcp_path": str(serial_path),
        "resolved_tty_name": tty_name,
    },
    sys.stdout,
    sort_keys=True,
)
sys.stdout.write("\n")
' \
  "$STM32_SERIAL_BY_ID" \
  "$STLINK_SERIAL" \
  >stlink-usb-inventory.json
test -s stlink-usb-inventory.json
sha256sum stlink-usb-inventory.json
```

For each command, st-flash connects under reset, then releases NRST and reads
while the core is halted. Read all 512 KiB twice and require byte equality:

```bash
test ! -e main-a.bin
test ! -e main-b.bin
install -m 0600 /dev/null main-a.bin
install -m 0600 /dev/null main-b.bin
sudo /usr/bin/st-flash --connect-under-reset \
  --serial "$STLINK_SERIAL" \
  read main-a.bin 0x08000000 0x80000

sudo /usr/bin/st-flash --connect-under-reset \
  --serial "$STLINK_SERIAL" \
  read main-b.bin 0x08000000 0x80000

test "$(stat -c %s main-a.bin)" = 524288
test "$(stat -c %s main-b.bin)" = 524288
cmp main-a.bin main-b.bin
sha256sum main-a.bin main-b.bin
```

The v1.7.0 F446 table defaults to only four option bytes even though the
official option region is 16 bytes. Supply the size explicitly and compare two
reads:

```bash
test ! -e option-a.bin
test ! -e option-b.bin
install -m 0600 /dev/null option-a.bin
install -m 0600 /dev/null option-b.bin
sudo /usr/bin/st-flash --connect-under-reset \
  --serial "$STLINK_SERIAL" \
  --area=option read option-a.bin 0x10

sudo /usr/bin/st-flash --connect-under-reset \
  --serial "$STLINK_SERIAL" \
  --area=option read option-b.bin 0x10

test "$(stat -c %s option-a.bin)" = 16
test "$(stat -c %s option-b.bin)" = 16
cmp option-a.bin option-b.bin
sha256sum option-a.bin option-b.bin
```

This is backup evidence, not a restore claim. The installed CLI's option write
accepts one 32-bit value and cannot restore this 16-byte image. Never write
option bytes, change RDP, or invoke unprotect in this procedure.

st-flash 1.7.0 exits debug mode after every read command and thereby releases
the core; it does not promise to keep the pre-existing firmware halted after
the process exits. Independent motor-power disconnection must therefore be in
place before even the first backup command and remain in place throughout.

Preserve the prior sector 7 separately and require exactly 131,072 bytes:

```bash
tail -c 131072 main-a.bin > sector7-before.bin
test "$(stat -c %s sector7-before.bin)" = 131072
sha256sum sector7-before.bin
```

## Build one deterministic motor-inert image

Build from an exact clean revision and locked dependency graph:

```bash
test ! -e cargo-target
RUSTFLAGS="--remap-path-prefix=$REPO=/kiko-source" "$CARGO" build \
  --manifest-path "$REPO/Cargo.toml" \
  --target-dir "$EVIDENCE_DIR/cargo-target" \
  --locked \
  --release \
  -p embedded \
  --features firmware \
  --bin embedded \
  --target thumbv7em-none-eabihf
```

The feature set above is the motor-inert KRP2 image. It must identify as ABI
2, build `0x00020002`, fingerprint `KIKO-NO-ACT-V1!!`, capability bits 319,
and maximum PWM zero.

Review every ELF `LOAD` physical address. All flash-backed bytes must lie
strictly below `0x08060000`:

```bash
/usr/bin/arm-none-eabi-readelf -lW \
  "$EVIDENCE_DIR/cargo-target/thumbv7em-none-eabihf/release/embedded"

test "$(sha256sum "$EVIDENCE_DIR/cargo-target/thumbv7em-none-eabihf/release/embedded" | cut -d ' ' -f 1)" = \
  "$EXPECTED_MOTOR_INERT_ELF_SHA256"
```

Generate a complete executable-region binary. Padding with erased bytes makes
sectors 0 through 6 deterministic and prevents stale legacy code from
surviving beyond the short natural image:

```bash
test ! -e motor-inert-main-384k.bin
/usr/bin/arm-none-eabi-objcopy \
  -O binary \
  --gap-fill 0xff \
  --pad-to 0x08060000 \
  "$EVIDENCE_DIR/cargo-target/thumbv7em-none-eabihf/release/embedded" \
  motor-inert-main-384k.bin

chmod 0600 motor-inert-main-384k.bin
test "$(stat -c %s motor-inert-main-384k.bin)" = 393216
test "$(sha256sum motor-inert-main-384k.bin | cut -d ' ' -f 1)" = \
  "$EXPECTED_MOTOR_INERT_MAIN_SHA256"
sha256sum motor-inert-main-384k.bin
```

Do not use `st-flash --opt`: it trims erased tails and can retain stale bytes.

The remap makes repository-owned panic locations independent of the absolute
checkout path. The pinned revision, lockfile, compiler identity, target, and
flags record build provenance; the final hashes reject any resulting output
byte difference, but do not prove that no other inputs could produce the same
bytes. A Nano rehearsal at revision
`6cc59a1a3972c44df77dfd2cc02920ba40d896a2` built independently from
`/home/makerspace/kiko-codex-native-check` and
`/home/makerspace/kiko-stm32-flash-5526fc0`, each remapped to
`/kiko-source`. The two ELFs compared byte-for-byte at SHA-256
`fe1f055c076d700fd65d0f02db2a55163f82b7789b3bb2bf8b0ee25ede130dcc`;
the two 393,216-byte padded images compared byte-for-byte at SHA-256
`270e553f5c18a53393f0234f334d3ccc71be32ac7827240b54c939c6d6def38d`.
The evidence roots are
`/home/makerspace/kiko-build-repro/6cc59a1-path-{a,b}`. The same revision
passed all 36 native `robot-server` bin tests on the Nano. That rehearsal
performed no debugger, serial, firmware, or actuator operation.

## Prebuild the host evidence tools

Build both host processes before the debugger write/reset sequence. This keeps
compilation time and its queued UART backlog outside the first post-reset
identity observation. Building these binaries does not open the serial device:

```bash
test ! -e host-cargo-target
RUSTFLAGS="--remap-path-prefix=$REPO=/kiko-source" "$CARGO" build \
  --manifest-path "$REPO/Cargo.toml" \
  --target-dir "$EVIDENCE_DIR/host-cargo-target" \
  --locked \
  --release \
  -p robot-server \
  --bin v2_identity_probe \
  --bin v2_transport_qualify

IDENTITY_PROBE="$EVIDENCE_DIR/host-cargo-target/release/v2_identity_probe"
TRANSPORT_QUALIFIER="$EVIDENCE_DIR/host-cargo-target/release/v2_transport_qualify"
test -x "$IDENTITY_PROBE"
test -x "$TRANSPORT_QUALIFIER"
```

## First write: motor-inert main only

The first firmware change preserves the freshly backed-up sector 7. Do not use
st-flash for this write. Its 1.7.0 write path calls its run-core finalizer even
after a failed or partial erase/write/verify result, so it can try to execute a
partial image before reporting the failure.

Use OpenOCD to connect under reset, then operate with the target debug-halted
except while the STM32 flash helpers execute debugger-owned algorithms from
RAM. The exact 384 KiB image intersects sectors 0 through 6 only. Omitting the
`reset` option from `program` prevents an explicit `reset run`. Dump both the
main region and the preserved journal sector before ending the same debugger
session:

```bash
test ! -e motor-inert-main-readback.bin
test ! -e sector7-after-inert.bin
install -m 0600 /dev/null motor-inert-main-readback.bin
install -m 0600 /dev/null sector7-after-inert.bin
sudo /usr/bin/openocd \
  -s "$OPENOCD_SCRIPTS" \
  -f "$OPENOCD_INTERFACE" \
  -c "hla_serial $STLINK_SERIAL" \
  -c 'transport select hla_swd' \
  -f "$OPENOCD_TARGET" \
  -c 'reset_config srst_only srst_nogate connect_assert_srst' \
  -c "program {$EVIDENCE_DIR/motor-inert-main-384k.bin} 0x08000000 verify" \
  -c "dump_image {$EVIDENCE_DIR/motor-inert-main-readback.bin} 0x08000000 0x60000" \
  -c "dump_image {$EVIDENCE_DIR/sector7-after-inert.bin} 0x08060000 0x20000" \
  -c shutdown

test "$(stat -c %s motor-inert-main-readback.bin)" = 393216
cmp motor-inert-main-384k.bin motor-inert-main-readback.bin
sha256sum motor-inert-main-384k.bin motor-inert-main-readback.bin

test "$(stat -c %s sector7-after-inert.bin)" = 131072
cmp sector7-before.bin sector7-after-inert.bin
sha256sum sector7-before.bin sector7-after-inert.bin
```

OpenOCD 0.11 does not document that a debug halt survives ST-Link disconnect,
so target execution state after shutdown is unknown. Motor power therefore
remains physically disconnected even on the apparent success path.

Only after every comparison succeeds, issue one explicit exact-target reset:

```bash
sudo /usr/bin/openocd \
  -s "$OPENOCD_SCRIPTS" \
  -f "$OPENOCD_INTERFACE" \
  -c "hla_serial $STLINK_SERIAL" \
  -c 'transport select hla_swd' \
  -f "$OPENOCD_TARGET" \
  -c 'reset_config srst_only srst_nogate connect_assert_srst' \
  -c init \
  -c 'reset run' \
  -c shutdown
```

Require no serial owner; do not kill or displace one. Run the privileged check
immediately before every process that opens the TTY. Exit status 1 with empty
output is the installed `fuser`'s exact no-owner result. Every owner, sudo
failure, missing path, diagnostic, or other result stops the run:

```bash
require_no_serial_owner() {
  local owner_output
  local owner_status=0

  owner_output="$(
    sudo -n /usr/bin/fuser -v "$STM32_SERIAL_BY_ID" 2>&1
  )" || owner_status=$?

  case "$owner_status" in
    0)
      printf '%s\n' "serial owner exists; refusing to displace it" >&2
      printf '%s\n' "$owner_output" >&2
      return 1
      ;;
    1)
      if test -n "$owner_output"; then
        printf '%s\n' "serial ownership check failed" >&2
        printf '%s\n' "$owner_output" >&2
        return 1
      fi
      ;;
    *)
      printf '%s\n' "serial ownership check returned $owner_status" >&2
      printf '%s\n' "$owner_output" >&2
      return 1
      ;;
  esac
}
```

The first operation that opens the TTY is the prebuilt read-only identity
probe. The helper captures stdout, stderr, and the actual child status
separately. It parses the JSON once, rejects duplicate keys and non-standard
constants, requires the complete schema-2 motor-inert identity, and emits only
the already-checked UID and boot ID. A failed command cannot leave an empty
file that looks like valid JSON:

```bash
capture_and_check_motor_inert_identity() {
  local stem="$1"
  local identity_status=0

  sudo -v
  require_no_serial_owner
  test ! -e "$stem.json"
  test ! -e "$stem.stderr"
  test ! -e "$stem.exit-status"
  test ! -e "$stem.checked-values"
  "$IDENTITY_PROBE" \
    --serial-device "$STM32_SERIAL_BY_ID" \
    --timeout-ms 5000 \
    >"$stem.json" \
    2>"$stem.stderr" || identity_status=$?
  printf '%s\n' "$identity_status" >"$stem.exit-status"
  sha256sum \
    "$stem.json" \
    "$stem.stderr" \
    "$stem.exit-status"
  test "$identity_status" -eq 0
  test -s "$stem.json"
  test ! -s "$stem.stderr"

  "$PYTHON" -c '
import json
import sys

def reject_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

with open(sys.argv[1], "r", encoding="utf-8") as source:
    observed = json.load(
        source,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )

expected = {
    "schema_version": 2,
    "observation_kind": "read_only_krp2_controller_hello",
    "serial_by_id_path": sys.argv[2],
    "host_input_queue_cleared_before_observation": True,
    "initial_unknown_record_prefix_excluded": True,
    "controller_uid_hex": sys.argv[3],
    "firmware_abi": 2,
    "firmware_build_id": 131074,
    "actuator_config_fingerprint_hex": sys.argv[4],
    "capabilities_bits": 319,
    "supports_required_safety_capabilities": False,
    "maximum_absolute_pwm_percent": 0,
    "grants_motion_authority": False,
    "maximum_command_lease_ms": 250,
    "reported_output_state": "disabled",
    "reported_output_state_is_safe": True,
    "pwm_frequency_hz": 20000,
    "watchdog_nominal_period_ms": 250,
    "neutral_output": "both_low",
    "physical_stop_semantics": "unverified",
    "evidence_boundary": (
        "the host input queue was cleared once; subsequently delivered bytes "
        "through the first zero delimiter were excluded, including any upstream "
        "or in-flight bytes delivered after that clear; the result is one decoded "
        "software claim, no serial bytes were transmitted, and no physical "
        "behavior was observed"
    ),
}
if type(observed) is not dict:
    raise ValueError("identity evidence must be a JSON object")
expected_keys = set(expected) | {"observed_boot_id"}
if set(observed) != expected_keys:
    raise ValueError(
        f"identity evidence keys differ: "
        f"missing={sorted(expected_keys - set(observed))}, "
        f"extra={sorted(set(observed) - expected_keys)}"
    )
for key, value in expected.items():
    actual = observed[key]
    if type(actual) is not type(value) or actual != value:
        raise ValueError(
            f"identity field {key!r} differs: expected {value!r}, got {actual!r}"
        )
boot_id = observed["observed_boot_id"]
if type(boot_id) is not int or not 0 < boot_id < (1 << 64):
    raise ValueError(f"invalid nonzero u64 boot ID: {boot_id!r}")
print(observed["controller_uid_hex"], boot_id)
' \
    "$stem.json" \
    "$STM32_SERIAL_BY_ID" \
    "$EXPECTED_CONTROLLER_UID_HEX" \
    "$EXPECTED_ACTUATOR_CONFIG_FINGERPRINT_HEX" \
    >"$stem.checked-values"
  test -s "$stem.checked-values"
  sha256sum "$stem.checked-values"
}

capture_and_check_motor_inert_identity v2-identity
read -r CONTROLLER_UID_HEX BOOT_ID trailing_value \
  <v2-identity.checked-values
test -z "${trailing_value:-}"
test "$CONTROLLER_UID_HEX" = "$EXPECTED_CONTROLLER_UID_HEX"
```

A strict decode failure remains a failed identity run. The probe adds one
machine-readable `failure_wire_evidence_json=` object to standard error while
preserving the typed decoder error as primary. After the first oversized
record it remains read-only and traces only through the earliest bounded stop:
the next zero delimiter, 4,096 additional bytes, the exclusive 250 ms
completion deadline, the original probe deadline, the 65,536-byte global
observation budget, EOF, a typed read failure, or a checked counter failure.
The capture helper hashes that standard error before stopping, so do not retry
or continue in the same evidence directory.

The single-clear/first-delimiter boundary in the preceding paragraph applies
only to the legacy read-only identity probe and its schema-2 output. A
successful identity observation does not prove that upstream or in-flight
bytes were absent. Every decoded identity record after that selected boundary
remains strict.

Run separate 10-second 20 Hz and 50 Hz qualifier processes. Each schema-3 run
clears the host input queue once, raw-discards every byte delivered during a
fixed 1,000 ms quarantine, then discards through one explicit following zero
delimiter. The canonical decoder is strict from that boundary and never
resynchronizes.

An exact motor-inert `ControllerHello` and idle-safe `Heartbeat` are only
pre-challenge candidate evidence. Before measurement, the qualifier may write
one to three motor-inert diagnostic challenges under its fresh nonzero run ID.
Attempts use reserved descending sequences and a recorded
host-elapsed-nanosecond token. Only an exact run/sequence/token echo for the
latest outstanding attempt may match. A subsequently decoded exact
`ControllerHello` and strictly forward, host-time-bounded idle-safe
`Heartbeat` are then required. This establishes a live round trip bound to the
current qualifier invocation; it does not prove that ST-Link, USB, TTY, or
other upstream buffers were empty. Candidate traffic, challenge writes, the
matched report, and post-match liveness are excluded from the measured probe
stream.

The qualifier never begins a control session and never sends PWM. Each run
repeats the privileged no-owner check, captures the real status, then parses
and checks its schema-3 freshness evidence, exact identity, requested
rate/window, zero-loss counts, and final idle-safe heartbeat:

```bash
run_and_check_motor_inert_transport() {
  local rate_hz="$1"
  local boot_id="$2"
  local stem="v2-transport-${rate_hz}hz"
  local qualifier_status=0

  sudo -v
  require_no_serial_owner
  test ! -e "$stem.json"
  test ! -e "$stem.stderr"
  test ! -e "$stem.exit-status"
  test ! -e "$stem.checked"
  "$TRANSPORT_QUALIFIER" \
    --serial-device "$STM32_SERIAL_BY_ID" \
    --controller-uid-hex "$EXPECTED_CONTROLLER_UID_HEX" \
    --boot-id "$boot_id" \
    --firmware-abi 2 \
    --firmware-build-id 0x00020002 \
    --actuator-config-fingerprint-hex "$EXPECTED_ACTUATOR_CONFIG_FINGERPRINT_HEX" \
    --capabilities-bits 319 \
    --rate-hz "$rate_hz" \
    --duration-ms 10000 \
    --serial-write-timeout-ms 10 \
    >"$stem.json" \
    2>"$stem.stderr" || qualifier_status=$?
  printf '%s\n' "$qualifier_status" >"$stem.exit-status"
  sha256sum \
    "$stem.json" \
    "$stem.stderr" \
    "$stem.exit-status"
  test "$qualifier_status" -eq 0
  test -s "$stem.json"
  test ! -s "$stem.stderr"

  "$PYTHON" -c '
import json
import sys

def reject_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

with open(sys.argv[1], "r", encoding="utf-8") as source:
    evidence = json.load(
        source,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )

serial_path = sys.argv[2]
controller_uid = sys.argv[3]
actuator_config_fingerprint_hex = sys.argv[4]
boot_id = int(sys.argv[5], 10)
rate_hz = int(sys.argv[6], 10)
planned_periods = rate_hz * 10

def value_at(path):
    value = evidence
    for key in path:
        if type(value) is not dict or key not in value:
            raise ValueError(f"missing qualifier field: {path!r}")
        value = value[key]
    return value

def exact(path, expected):
    value = value_at(path)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(
            f"qualifier field {path!r} differs: expected {expected!r}, got {value!r}"
        )

def bounded_int(path, minimum, maximum):
    value = value_at(path)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            f"qualifier field {path!r} is outside "
            f"{minimum}..={maximum}: {value!r}"
        )
    return value

def exact_object_keys(value, label, expected_keys):
    if type(value) is not dict:
        raise ValueError(f"{label} is not a JSON object: {value!r}")
    actual_keys = set(value)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{label} keys differ: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

exact(("schema_version",), 3)
exact(("evidence_kind",), "motor_inert_krp2_uart_transport_qualification")
exact(("passed",), True)
exact(("serial_by_id_path",), serial_path)
exact(("receive_startup", "host_input_queue_cleared_before_observation"), True)
exact(("receive_startup", "initial_unknown_record_prefix_excluded"), True)
exact(
    ("receive_startup", "post_boundary_decode_policy"),
    "startup bytes are raw-discarded for the declared bounded quarantine and "
    "through one selected delimiter; every subsequent complete record is "
    "decoded strictly; bounded motor-inert challenge retries do not "
    "resynchronize the decoder; a framing error fails the run",
)
exact(("identity", "controller_uid_hex"), controller_uid)
exact(("identity", "boot_id"), boot_id)
exact(("identity", "firmware_abi"), 2)
exact(("identity", "firmware_build_id"), 131074)
exact(
    ("identity", "actuator_config_fingerprint_hex"),
    actuator_config_fingerprint_hex,
)
exact(("identity", "capabilities_bits"), 319)
exact(("identity", "max_abs_pwm_percent"), 0)
exact(("identity", "output_state"), "disabled")
exact(("identity", "max_command_lease_ms"), 250)
exact(("identity", "watchdog_nominal_period_ms"), 250)
exact(("identity", "pwm_frequency_hz"), 20000)
exact(("identity", "neutral_output"), "both_low")
exact(("identity", "physical_stop_semantics"), "unverified")
exact(("liveness", "controller_hello_maximum_allowed_gap_ms"), 2000)
exact(("liveness", "idle_safe_heartbeat_maximum_allowed_gap_ms"), 375)
exact(
    ("liveness", "bound_policy"),
    "Heartbeat host-receive gap <= advertised watchdog_nominal_period plus "
    "ceil(10 percent clock tolerance) plus 100 ms scheduling/transport "
    "margin; ControllerHello host-receive gap <= 2x the canonical protocol "
    "Hello period. These are host qualification bounds; only the watchdog "
    "period is an on-wire field.",
)
hello_gap_ns = bounded_int(
    (
        "liveness",
        "controller_hello_maximum_observed_gap_ns_including_trailing_boundary",
    ),
    0,
    2_000_000_000,
)
heartbeat_gap_ns = bounded_int(
    (
        "liveness",
        "idle_safe_heartbeat_maximum_observed_gap_ns_including_trailing_boundary",
    ),
    0,
    375_000_000,
)
bounded_int(
    ("liveness", "controller_hello_messages_validated_including_admission"),
    1,
    (1 << 64) - 1,
)
bounded_int(
    ("liveness", "idle_safe_heartbeat_messages_validated_including_admission"),
    1,
    (1 << 64) - 1,
)
exact(("plan", "rate_hz"), rate_hz)
exact(("plan", "nominal_period_ns"), 1_000_000_000 // rate_hz)
exact(("plan", "duration_ms"), 10000)
exact(("plan", "serial_write_and_flush_timeout_ms"), 10)
exact(("plan", "planned_periods"), planned_periods)
exact(("counts", "planned_periods"), planned_periods)
exact(("counts", "probes_dispatched_to_writer"), planned_periods)
exact(("counts", "completed_writes"), planned_periods)
exact(("counts", "unique_reports"), planned_periods)
for field in (
    "missing_reports",
    "duplicate_reports",
    "reordered_reports",
    "scheduler_skipped_periods",
    "in_flight_limit_skips",
    "writer_queue_skips",
    "writes_late_by_at_least_one_period",
):
    exact(("counts", field), 0)
exact(("missing_sequences",), [])
exact(("final_idle_safe_heartbeat_received_after_last_write",), True)
run_id = evidence.get("run_id")
if type(run_id) is not int or not 0 < run_id < (1 << 64):
    raise ValueError(f"invalid nonzero u64 run ID: {run_id!r}")

freshness_path = ("receive_startup", "freshness_admission")
freshness = value_at(freshness_path)
expected_freshness_keys = {
    "boundary",
    "challenge",
    "pre_challenge_reports_discarded",
    "nonmatching_reports_discarded_before_match",
    "earlier_attempt_reports_discarded_after_later_challenge",
    "nonforward_heartbeats_discarded_after_match",
    "matched_report_request_received_uptime_ms_wrapping",
    "matched_report_response_prepared_uptime_ms_wrapping",
    "matched_report_controller_service_ms",
    "matched_report_host_elapsed_controller_clock_upper_bound_ms",
    "admitted_heartbeat_delta_after_report_ms",
    "admitted_heartbeat_host_elapsed_controller_clock_upper_bound_ms",
}
exact_object_keys(freshness, "freshness admission", expected_freshness_keys)

boundary_path = freshness_path + ("boundary",)
boundary = value_at(boundary_path)
expected_boundary_keys = {
    "input_quarantine_target_ms",
    "input_quarantine_elapsed_ns",
    "input_quarantine_bytes_discarded",
    "input_quarantine_delimiters_discarded",
    "boundary_alignment_bytes_discarded_including_delimiter",
    "strict_record_boundary_established",
}
exact_object_keys(boundary, "freshness boundary", expected_boundary_keys)
exact(boundary_path + ("input_quarantine_target_ms",), 1000)
bounded_int(
    boundary_path + ("input_quarantine_elapsed_ns",),
    1_000_000_000,
    (1 << 64) - 1,
)
for field in (
    "input_quarantine_bytes_discarded",
    "input_quarantine_delimiters_discarded",
):
    bounded_int(boundary_path + (field,), 0, (1 << 64) - 1)
bounded_int(
    boundary_path + ("boundary_alignment_bytes_discarded_including_delimiter",),
    1,
    (1 << 64) - 1,
)
exact(boundary_path + ("strict_record_boundary_established",), True)

challenge_path = freshness_path + ("challenge",)
challenge = value_at(challenge_path)
expected_challenge_keys = {
    "attempts_written",
    "attempts",
    "matched_attempt_index_zero_based",
}
exact_object_keys(challenge, "freshness challenge", expected_challenge_keys)
attempts_written = bounded_int(
    challenge_path + ("attempts_written",),
    1,
    3,
)
exact(
    challenge_path + ("matched_attempt_index_zero_based",),
    attempts_written - 1,
)
attempts = challenge["attempts"]
if type(attempts) is not list or len(attempts) != 3:
    raise ValueError(f"expected exactly three freshness-attempt slots: {attempts!r}")
diagnostic_probe_encoded_bytes = bounded_int(
    ("wire_load", "diagnostic_probe_encoded_bytes"),
    1,
    (1 << 64) - 1,
)
expected_attempt_keys = {
    "run_id",
    "reserved_sequence",
    "host_elapsed_ns_token",
    "encoded_bytes_written",
}
for index, attempt in enumerate(attempts):
    if index >= attempts_written:
        if attempt is not None:
            raise ValueError(
                f"unwritten freshness-attempt slot {index} is not null: {attempt!r}"
            )
        continue
    if type(attempt) is not dict or set(attempt) != expected_attempt_keys:
        raise ValueError(
            f"freshness-attempt slot {index} has unexpected shape: {attempt!r}"
        )
    attempt_run_id = attempt["run_id"]
    if type(attempt_run_id) is not int or attempt_run_id != run_id:
        raise ValueError(
            f"freshness-attempt slot {index} run ID differs: {attempt_run_id!r}"
        )
    expected_sequence = ((1 << 32) - 1) - index
    attempt_sequence = attempt["reserved_sequence"]
    if (
        type(attempt_sequence) is not int
        or attempt_sequence != expected_sequence
    ):
        raise ValueError(
            f"freshness-attempt slot {index} sequence differs: "
            f"{attempt_sequence!r}"
        )
    token = attempt["host_elapsed_ns_token"]
    if type(token) is not int or not 0 <= token < (1 << 64):
        raise ValueError(
            f"freshness-attempt slot {index} has invalid u64 token: {token!r}"
        )
    attempt_encoded_bytes = attempt["encoded_bytes_written"]
    if (
        type(attempt_encoded_bytes) is not int
        or attempt_encoded_bytes != diagnostic_probe_encoded_bytes
    ):
        raise ValueError(
            f"freshness-attempt slot {index} encoded length differs: "
            f"{attempt_encoded_bytes!r}"
        )

for field in (
    "pre_challenge_reports_discarded",
    "nonmatching_reports_discarded_before_match",
    "earlier_attempt_reports_discarded_after_later_challenge",
    "nonforward_heartbeats_discarded_after_match",
):
    bounded_int(freshness_path + (field,), 0, (1 << 64) - 1)

request_uptime = bounded_int(
    freshness_path + ("matched_report_request_received_uptime_ms_wrapping",),
    0,
    (1 << 32) - 1,
)
response_uptime = bounded_int(
    freshness_path + ("matched_report_response_prepared_uptime_ms_wrapping",),
    0,
    (1 << 32) - 1,
)
service_delta = (response_uptime - request_uptime) & ((1 << 32) - 1)
exact(
    freshness_path + ("matched_report_controller_service_ms",),
    service_delta,
)
if service_delta >= (1 << 31):
    raise ValueError(
        f"freshness service delta is not wrapping-forward: {service_delta!r}"
    )
service_host_bound = bounded_int(
    freshness_path
    + ("matched_report_host_elapsed_controller_clock_upper_bound_ms",),
    0,
    (1 << 64) - 1,
)
if service_delta > service_host_bound:
    raise ValueError(
        f"freshness service delta exceeds host bound: "
        f"{service_delta} > {service_host_bound}"
    )

admitted_heartbeat_uptime = bounded_int(
    ("identity", "admitted_idle_heartbeat_uptime_ms_wrapping"),
    0,
    (1 << 32) - 1,
)
heartbeat_delta = (
    admitted_heartbeat_uptime - response_uptime
) & ((1 << 32) - 1)
exact(
    freshness_path + ("admitted_heartbeat_delta_after_report_ms",),
    heartbeat_delta,
)
if not 0 < heartbeat_delta < (1 << 31):
    raise ValueError(
        f"post-report heartbeat delta is not strictly wrapping-forward: "
        f"{heartbeat_delta!r}"
    )
heartbeat_host_bound = bounded_int(
    freshness_path
    + ("admitted_heartbeat_host_elapsed_controller_clock_upper_bound_ms",),
    0,
    (1 << 64) - 1,
)
if heartbeat_delta > heartbeat_host_bound:
    raise ValueError(
        f"post-report heartbeat delta exceeds host bound: "
        f"{heartbeat_delta} > {heartbeat_host_bound}"
    )
print("qualified")
' \
    "$stem.json" \
    "$STM32_SERIAL_BY_ID" \
    "$EXPECTED_CONTROLLER_UID_HEX" \
    "$EXPECTED_ACTUATOR_CONFIG_FINGERPRINT_HEX" \
    "$boot_id" \
    "$rate_hz" \
    >"$stem.checked"
  test "$(cat "$stem.checked")" = "qualified"
  sha256sum "$stem.checked"
}

run_and_check_motor_inert_transport 20 "$BOOT_ID"
run_and_check_motor_inert_transport 50 "$BOOT_ID"
```

Any loss, duplicate, reorder, timeout, stale heartbeat, skipped deadline probe,
nonzero output observation, post-boundary framing error, nonzero process
status, or evidence-parser mismatch fails the run.

Stop here if motor-inert transport does not pass. Do not provision a new
journal or install candidate firmware.

## Later journal provisioning without executing journal data

The st-flash `--flash` guard can bound an address range, but its v1.7.0 write
path is not used anywhere in this procedure because its finalizer can run the
write start even after an earlier failure. A journal-only write at
`0x08060000` is additionally invalid because that finalizer would load SP/PC
from journal bytes.

Only after motor-inert transport passes, generate the exact 131,072-byte
journal with the repository tool. Then create one contiguous 512 KiB
programming binary whose prefix is the already qualified motor-inert main
image and whose suffix is the journal. Use binary-aware tooling and prove both
component comparisons before hardware access.

```bash
test ! -e boot-journal-sector7.bin
test ! -e boot-journal-evidence.json
"$CARGO" run \
  --manifest-path "$REPO/Cargo.toml" \
  --target-dir "$EVIDENCE_DIR/cargo-target" \
  --locked \
  -p embedded \
  --features boot-journal-tool \
  --bin kiko-boot-journal-image -- \
  --output boot-journal-sector7.bin > boot-journal-evidence.json

test "$(stat -c %s boot-journal-sector7.bin)" = 131072
sha256sum boot-journal-sector7.bin

test ! -e motor-inert-with-journal-512k.bin
test ! -e combined-sector7-check.bin
cp motor-inert-main-384k.bin motor-inert-with-journal-512k.bin
dd if=boot-journal-sector7.bin \
  of=motor-inert-with-journal-512k.bin \
  bs=131072 seek=3 conv=notrunc status=none

test "$(stat -c %s motor-inert-with-journal-512k.bin)" = 524288
cmp -n 393216 \
  motor-inert-main-384k.bin \
  motor-inert-with-journal-512k.bin
tail -c 131072 \
  motor-inert-with-journal-512k.bin > combined-sector7-check.bin
cmp boot-journal-sector7.bin combined-sector7-check.bin
sha256sum motor-inert-with-journal-512k.bin
```

Program the single full image with OpenOCD. It connects under reset and then
operates with the target halted except for debugger-owned RAM algorithms.
Omitting the `reset` option from `program` is intentional: OpenOCD performs
`init`, `reset init`, sector-scoped erase/write, and verification, but does not
call `reset run`. Dump the entire programmed bank in the same debugger session:

```bash
test ! -e motor-inert-with-journal-readback.bin
install -m 0600 /dev/null motor-inert-with-journal-readback.bin
sudo /usr/bin/openocd \
  -s "$OPENOCD_SCRIPTS" \
  -f "$OPENOCD_INTERFACE" \
  -c "hla_serial $STLINK_SERIAL" \
  -c 'transport select hla_swd' \
  -f "$OPENOCD_TARGET" \
  -c 'reset_config srst_only srst_nogate connect_assert_srst' \
  -c "program {$EVIDENCE_DIR/motor-inert-with-journal-512k.bin} 0x08000000 verify" \
  -c "dump_image {$EVIDENCE_DIR/motor-inert-with-journal-readback.bin} 0x08000000 0x80000" \
  -c shutdown
```

Any connect, erase, write, verify, or dump error stops the procedure. With
motor power still cut, require an exact 524,288-byte readback and byte
comparison:

```bash
test "$(stat -c %s motor-inert-with-journal-readback.bin)" = 524288
cmp motor-inert-with-journal-512k.bin motor-inert-with-journal-readback.bin
sha256sum motor-inert-with-journal-512k.bin \
  motor-inert-with-journal-readback.bin
```

As above, target execution state after the ST-Link disconnect is unknown; the
absence of `reset run` is not a claim that the CPU cannot have resumed.
Only after the comparison succeeds may a separate exact-target OpenOCD
invocation issue `init` plus `reset run`. That reset is the explicit
synchronization/start point, not proof that no instruction ran earlier. The
reset vector belongs to the motor-inert main image, not the journal suffix:

```bash
sudo /usr/bin/openocd \
  -s "$OPENOCD_SCRIPTS" \
  -f "$OPENOCD_INTERFACE" \
  -c "hla_serial $STLINK_SERIAL" \
  -c 'transport select hla_swd' \
  -f "$OPENOCD_TARGET" \
  -c 'reset_config srst_only srst_nogate connect_assert_srst' \
  -c init \
  -c 'reset run' \
  -c shutdown
```

Motor power remains disconnected. Repeat the captured passive identity
operation under a newly privileged no-owner check. The distinct filenames
make it impossible to confuse the pre-journal observation with the
post-journal one:

```bash
capture_and_check_motor_inert_identity v2-identity-after-journal
read -r AFTER_JOURNAL_CONTROLLER_UID_HEX AFTER_JOURNAL_BOOT_ID trailing_value \
  <v2-identity-after-journal.checked-values
test -z "${trailing_value:-}"
test "$AFTER_JOURNAL_CONTROLLER_UID_HEX" = "$EXPECTED_CONTROLLER_UID_HEX"
test -n "$AFTER_JOURNAL_BOOT_ID"
```

## Build and reproduce the operator-supervised candidate

Candidate firmware may be installed only after the preceding inert/journal
readback and identity evidence. The candidate source revision reviewed for this
phase is
`3f262f1a9b377448c903ed61305b28498b2b2c7f`. Two independently located,
clean, path-remapped native Nano builds produced byte-identical artifacts:

```text
ELF bytes         1375976
ELF SHA-256       fcb0624f6422309f8cd4a383e31e6fd3596465a4efbb98317310d7febefa63e0
natural BIN bytes 48820
natural BIN SHA-256
                  a751e9f177a3a42840ef84a9901391b750092ede96262f77ee516effd48e92c3
final flash end   0x0800beb4
padded BIN bytes  393216
padded BIN SHA-256
                  8b8fa6f2b7498ac9c7e5ad86d5e44d98a6fe8fb4d50035fb9ec9f6828b564424
```

The retained Nano evidence directory is
`/home/makerspace/kiko-build-repro/3f262f1-20260727T014620Z`; its 1,572-byte
`SHA256SUMS` manifest has SHA-256
`51d1aa9da78258c3715ab3d9a34d66d84594bd5b9d23a11e55d6841976f07563`.
The two retained `readelf` reports are identical and each has SHA-256
`d6debe347e5f567bda68efc5b1993aa32fce04619d8e84ab1c4c31c67c259940`.
The journal inspector test suite passed all 6 tests in each checkout; both
release inspector binaries are identical with SHA-256
`07f3c10a7cf6ed9cd5c2c400f64f389e2841b47066f733626f91b529426f5215`.
Strict Clippy with the exact candidate feature set and `-D warnings` also
passed at this revision.
Those values identify the reproduced build; they do not by themselves replace
local reproduction and comparison before hardware access. Reproduce them from
two independently located clean checkouts. Both checkouts must name the same
exact commit, use the same lockfile and audited toolchain, remap their
different absolute roots to `/kiko-source`, and produce byte-identical ELFs,
natural binaries, and padded main images:

```bash
CANDIDATE_REPO_A=/ABSOLUTE/FIRST/CLEAN/KIKO-CHECKOUT
CANDIDATE_REPO_B=/ABSOLUTE/SECOND/CLEAN/KIKO-CHECKOUT
EXPECTED_CANDIDATE_REVISION=3f262f1a9b377448c903ed61305b28498b2b2c7f
EXPECTED_CANDIDATE_ELF_BYTES=1375976
EXPECTED_CANDIDATE_ELF_SHA256=fcb0624f6422309f8cd4a383e31e6fd3596465a4efbb98317310d7febefa63e0
EXPECTED_CANDIDATE_RAW_BYTES=48820
EXPECTED_CANDIDATE_RAW_SHA256=a751e9f177a3a42840ef84a9901391b750092ede96262f77ee516effd48e92c3
EXPECTED_CANDIDATE_FLASH_END_HEX=0x0800beb4
EXPECTED_CANDIDATE_MAIN_BYTES=393216
EXPECTED_CANDIDATE_MAIN_SHA256=8b8fa6f2b7498ac9c7e5ad86d5e44d98a6fe8fb4d50035fb9ec9f6828b564424
readonly CANDIDATE_REPO_A CANDIDATE_REPO_B
readonly EXPECTED_CANDIDATE_REVISION
readonly EXPECTED_CANDIDATE_ELF_BYTES
readonly EXPECTED_CANDIDATE_ELF_SHA256
readonly EXPECTED_CANDIDATE_RAW_BYTES
readonly EXPECTED_CANDIDATE_RAW_SHA256
readonly EXPECTED_CANDIDATE_FLASH_END_HEX
readonly EXPECTED_CANDIDATE_MAIN_BYTES
readonly EXPECTED_CANDIDATE_MAIN_SHA256

test "$(realpath "$CANDIDATE_REPO_A")" != \
  "$(realpath "$CANDIDATE_REPO_B")"
for candidate_repo in "$CANDIDATE_REPO_A" "$CANDIDATE_REPO_B"; do
  test -f "$candidate_repo/Cargo.lock"
  test "$(git -C "$candidate_repo" rev-parse HEAD)" = \
    "$EXPECTED_CANDIDATE_REVISION"
  test -z "$(
    git -C "$candidate_repo" \
      status --porcelain=v1 --untracked-files=all
  )"
done
cmp "$CANDIDATE_REPO_A/Cargo.lock" "$CANDIDATE_REPO_B/Cargo.lock"

test ! -e candidate-target-a
test ! -e candidate-target-b
RUSTFLAGS="--remap-path-prefix=$CANDIDATE_REPO_A=/kiko-source" \
  "$CARGO" clippy \
    --manifest-path "$CANDIDATE_REPO_A/Cargo.toml" \
    --target-dir "$EVIDENCE_DIR/candidate-target-a" \
    --locked \
    -p embedded \
    --features \
      firmware,flash-boot-journal,operator-supervised-four-pwm-candidate \
    --bin embedded \
    --target thumbv7em-none-eabihf \
    -- -D warnings
RUSTFLAGS="--remap-path-prefix=$CANDIDATE_REPO_A=/kiko-source" \
  "$CARGO" build \
    --manifest-path "$CANDIDATE_REPO_A/Cargo.toml" \
    --target-dir "$EVIDENCE_DIR/candidate-target-a" \
    --locked \
    --release \
    -p embedded \
    --features \
      firmware,flash-boot-journal,operator-supervised-four-pwm-candidate \
    --bin embedded \
    --target thumbv7em-none-eabihf
RUSTFLAGS="--remap-path-prefix=$CANDIDATE_REPO_B=/kiko-source" \
  "$CARGO" build \
    --manifest-path "$CANDIDATE_REPO_B/Cargo.toml" \
    --target-dir "$EVIDENCE_DIR/candidate-target-b" \
    --locked \
    --release \
    -p embedded \
    --features \
      firmware,flash-boot-journal,operator-supervised-four-pwm-candidate \
    --bin embedded \
    --target thumbv7em-none-eabihf

CANDIDATE_ELF_A="$EVIDENCE_DIR/candidate-target-a/thumbv7em-none-eabihf/release/embedded"
CANDIDATE_ELF_B="$EVIDENCE_DIR/candidate-target-b/thumbv7em-none-eabihf/release/embedded"
test -f "$CANDIDATE_ELF_A"
test -f "$CANDIDATE_ELF_B"
cmp "$CANDIDATE_ELF_A" "$CANDIDATE_ELF_B"
test "$(stat -c %s "$CANDIDATE_ELF_A")" = \
  "$EXPECTED_CANDIDATE_ELF_BYTES"
test "$(stat -c %s "$CANDIDATE_ELF_B")" = \
  "$EXPECTED_CANDIDATE_ELF_BYTES"
test "$(sha256sum "$CANDIDATE_ELF_A" | cut -d ' ' -f 1)" = \
  "$EXPECTED_CANDIDATE_ELF_SHA256"
test "$(sha256sum "$CANDIDATE_ELF_B" | cut -d ' ' -f 1)" = \
  "$EXPECTED_CANDIDATE_ELF_SHA256"

test ! -e candidate-a.readelf.txt
test ! -e candidate-b.readelf.txt
/usr/bin/arm-none-eabi-readelf -lW "$CANDIDATE_ELF_A" \
  >candidate-a.readelf.txt
/usr/bin/arm-none-eabi-readelf -lW "$CANDIDATE_ELF_B" \
  >candidate-b.readelf.txt
cmp candidate-a.readelf.txt candidate-b.readelf.txt

test ! -e candidate-a-raw.bin
test ! -e candidate-b-raw.bin
/usr/bin/arm-none-eabi-objcopy -O binary \
  "$CANDIDATE_ELF_A" candidate-a-raw.bin
/usr/bin/arm-none-eabi-objcopy -O binary \
  "$CANDIDATE_ELF_B" candidate-b-raw.bin
chmod 0600 candidate-a-raw.bin candidate-b-raw.bin
cmp candidate-a-raw.bin candidate-b-raw.bin
test "$(stat -c %s candidate-a-raw.bin)" = \
  "$EXPECTED_CANDIDATE_RAW_BYTES"
test "$(stat -c %s candidate-b-raw.bin)" = \
  "$EXPECTED_CANDIDATE_RAW_BYTES"
test "$(sha256sum candidate-a-raw.bin | cut -d ' ' -f 1)" = \
  "$EXPECTED_CANDIDATE_RAW_SHA256"
test "$(sha256sum candidate-b-raw.bin | cut -d ' ' -f 1)" = \
  "$EXPECTED_CANDIDATE_RAW_SHA256"
test "$(
  printf '0x%08x' "$((0x08000000 + EXPECTED_CANDIDATE_RAW_BYTES))"
)" = "$EXPECTED_CANDIDATE_FLASH_END_HEX"
```

Review every `LOAD` physical address in both retained `readelf` outputs.
Size/hash equality is not a substitute for that review. Every flash-backed byte
must be at or above `0x08000000` and strictly below `0x08060000`. A `LOAD`
with a RAM virtual address can still have a flash physical address: the
candidate `.data` segment has virtual address `0x20000000`, physical address
`0x0800bba8`, and file size `0x30c`, so it is included. The zero-file-size
`.bss` load at `0x20000310` is RAM-only and is not part of the raw image.

The four file-backed loads begin at physical addresses `0x08000000`,
`0x080001a8`, `0x0800a9b0`, and `0x0800bba8`. Their maximum exclusive end is
`0x0800beb4`. The resulting span from `0x08000000` is exactly 48,820 bytes,
matching the raw binary, and leaves 344,396 bytes before `0x08060000`. This is
the reviewed flash-layout result, not an inference from ELF file size.

Pad both independently reproduced ELFs through the complete executable region.
This prevents stale bytes beyond the natural image from surviving:

```bash
test ! -e candidate-a-main-384k.bin
test ! -e candidate-b-main-384k.bin
/usr/bin/arm-none-eabi-objcopy \
  -O binary \
  --gap-fill 0xff \
  --pad-to 0x08060000 \
  "$CANDIDATE_ELF_A" \
  candidate-a-main-384k.bin
/usr/bin/arm-none-eabi-objcopy \
  -O binary \
  --gap-fill 0xff \
  --pad-to 0x08060000 \
  "$CANDIDATE_ELF_B" \
  candidate-b-main-384k.bin
chmod 0600 candidate-a-main-384k.bin candidate-b-main-384k.bin
test "$(stat -c %s candidate-a-main-384k.bin)" = \
  "$EXPECTED_CANDIDATE_MAIN_BYTES"
test "$(stat -c %s candidate-b-main-384k.bin)" = \
  "$EXPECTED_CANDIDATE_MAIN_BYTES"
cmp candidate-a-main-384k.bin candidate-b-main-384k.bin
test "$(sha256sum candidate-a-main-384k.bin | cut -d ' ' -f 1)" = \
  "$EXPECTED_CANDIDATE_MAIN_SHA256"
test "$(sha256sum candidate-b-main-384k.bin | cut -d ' ' -f 1)" = \
  "$EXPECTED_CANDIDATE_MAIN_SHA256"

test ! -e candidate-main-384k.bin
test ! -e candidate-main-384k.sha256
install -m 0600 candidate-a-main-384k.bin candidate-main-384k.bin
CANDIDATE_MAIN_SHA256="$(
  sha256sum candidate-main-384k.bin | cut -d ' ' -f 1
)"
test "$CANDIDATE_MAIN_SHA256" = "$EXPECTED_CANDIDATE_MAIN_SHA256"
printf '%s  %s\n' \
  "$CANDIDATE_MAIN_SHA256" \
  candidate-main-384k.bin \
  >candidate-main-384k.sha256
sha256sum \
  "$CANDIDATE_ELF_A" \
  "$CANDIDATE_ELF_B" \
  candidate-a-raw.bin \
  candidate-b-raw.bin \
  candidate-main-384k.bin \
  candidate-a.readelf.txt \
  candidate-b.readelf.txt
```

Do not continue if either independently located build differs, even if one
matches the previously observed hash.

Prebuild and test the host-only journal inspector from the exact clean
candidate source before any candidate hardware access. It exact-length reads
one or two complete sector images and calls the same parser, next-record
planner, and commit verifier used by the firmware. Transition mode
additionally requires that no byte other than the one planned 16-byte record
changed:

```bash
test ! -e candidate-journal-tool-target
RUSTFLAGS="--remap-path-prefix=$CANDIDATE_REPO_A=/kiko-source" \
  "$CARGO" test \
    --manifest-path "$CANDIDATE_REPO_A/Cargo.toml" \
    --target-dir "$EVIDENCE_DIR/candidate-journal-tool-target" \
    --locked \
    -p embedded \
    --features boot-journal-tool \
    --bin kiko-boot-journal-inspect
RUSTFLAGS="--remap-path-prefix=$CANDIDATE_REPO_A=/kiko-source" \
  "$CARGO" build \
    --manifest-path "$CANDIDATE_REPO_A/Cargo.toml" \
    --target-dir "$EVIDENCE_DIR/candidate-journal-tool-target" \
    --locked \
    --release \
    -p embedded \
    --features boot-journal-tool \
    --bin kiko-boot-journal-inspect
JOURNAL_INSPECTOR="$EVIDENCE_DIR/candidate-journal-tool-target/release/kiko-boot-journal-inspect"
readonly JOURNAL_INSPECTOR
test -x "$JOURNAL_INSPECTOR"
sha256sum "$JOURNAL_INSPECTOR"
```

## Fresh candidate-boundary backup

The earlier backup predates journal provisioning. Take a new full-bank backup
immediately before the candidate write. Motor power remains independently
disconnected. Take both dumps in one connect-under-reset OpenOCD session while
the core remains halted. Two separate debugger sessions are not equivalent:
each release/reset can legitimately append another journal record.

```bash
sudo -v
require_no_serial_owner
test ! -e pre-candidate-full-a.bin
test ! -e pre-candidate-full-b.bin
test ! -e pre-candidate-main-prefix.bin
test ! -e pre-candidate-sector7.bin
install -m 0600 /dev/null pre-candidate-full-a.bin
install -m 0600 /dev/null pre-candidate-full-b.bin
sudo /usr/bin/openocd \
  -s "$OPENOCD_SCRIPTS" \
  -f "$OPENOCD_INTERFACE" \
  -c "hla_serial $STLINK_SERIAL" \
  -c 'transport select hla_swd' \
  -f "$OPENOCD_TARGET" \
  -c 'reset_config srst_only srst_nogate connect_assert_srst' \
  -c init \
  -c 'reset init' \
  -c "dump_image {$EVIDENCE_DIR/pre-candidate-full-a.bin} 0x08000000 0x80000" \
  -c "dump_image {$EVIDENCE_DIR/pre-candidate-full-b.bin} 0x08000000 0x80000" \
  -c shutdown
test "$(stat -c %s pre-candidate-full-a.bin)" = 524288
test "$(stat -c %s pre-candidate-full-b.bin)" = 524288
cmp pre-candidate-full-a.bin pre-candidate-full-b.bin
head -c 393216 pre-candidate-full-a.bin >pre-candidate-main-prefix.bin
test "$(stat -c %s pre-candidate-main-prefix.bin)" = 393216
cmp motor-inert-main-384k.bin pre-candidate-main-prefix.bin
tail -c 131072 pre-candidate-full-a.bin >pre-candidate-sector7.bin
test "$(stat -c %s pre-candidate-sector7.bin)" = 131072
sha256sum \
  pre-candidate-full-a.bin \
  pre-candidate-full-b.bin \
  pre-candidate-main-prefix.bin \
  pre-candidate-sector7.bin
```

This backup is evidence and emergency source material, not permission to
restore its full 512 KiB later. Once candidate firmware has booted, restoring
this sector-7 snapshot could rewind journal counters and permit reuse of a
previous boot identity. The prefix comparison also makes the candidate's
starting main image self-contained: stop if it is not the exact qualified
motor-inert image.

## Main-only candidate write and readback

Do not use `st-flash write`, `st-flash erase`, `st-flash --opt`, unprotect, or
an option-byte write. Use one OpenOCD connect-under-reset session and one
`reset init`. Take the sector-7 baseline after that halt, use direct
sector-scoped flash commands so no nested `program` reset can occur, write only
the 384 KiB candidate main region, and dump both regions before ending that
same debugger session:

```bash
sudo -v
require_no_serial_owner
test ! -e candidate-main-readback.bin
test ! -e candidate-sector7-write-boundary.bin
test ! -e candidate-sector7-after-write.bin
install -m 0600 /dev/null candidate-main-readback.bin
install -m 0600 /dev/null candidate-sector7-write-boundary.bin
install -m 0600 /dev/null candidate-sector7-after-write.bin
sudo /usr/bin/openocd \
  -s "$OPENOCD_SCRIPTS" \
  -f "$OPENOCD_INTERFACE" \
  -c "hla_serial $STLINK_SERIAL" \
  -c 'transport select hla_swd' \
  -f "$OPENOCD_TARGET" \
  -c 'reset_config srst_only srst_nogate connect_assert_srst' \
  -c init \
  -c 'reset init' \
  -c "dump_image {$EVIDENCE_DIR/candidate-sector7-write-boundary.bin} 0x08060000 0x20000" \
  -c "flash write_image erase {$EVIDENCE_DIR/candidate-main-384k.bin} 0x08000000 bin" \
  -c "verify_image {$EVIDENCE_DIR/candidate-main-384k.bin} 0x08000000 bin" \
  -c "dump_image {$EVIDENCE_DIR/candidate-main-readback.bin} 0x08000000 0x60000" \
  -c "dump_image {$EVIDENCE_DIR/candidate-sector7-after-write.bin} 0x08060000 0x20000" \
  -c shutdown

test "$(stat -c %s candidate-main-readback.bin)" = 393216
cmp candidate-main-384k.bin candidate-main-readback.bin
test "$(stat -c %s candidate-sector7-write-boundary.bin)" = 131072
test "$(stat -c %s candidate-sector7-after-write.bin)" = 131072
cmp candidate-sector7-write-boundary.bin candidate-sector7-after-write.bin
sha256sum \
  candidate-main-384k.bin \
  candidate-main-readback.bin \
  candidate-sector7-write-boundary.bin \
  candidate-sector7-after-write.bin
```

Any difference is terminal. OpenOCD does not guarantee that the core remains
halted after disconnect, so motor power remains physically disconnected even
while the comparisons run.

## Causal journal progression and passive candidate identity

Do not use two passive UART observations to infer reset progression. The
probe's documented input boundary deliberately admits upstream or in-flight
records, so mere inequality would not establish which reset produced either
record.

Only after the candidate main and sector comparisons pass, keep the complete
progression window inside one new exact-target OpenOCD connection. `J0`
absorbs any uncertain execution state after the write-session disconnect.
Each evidenced boot uses `reset run`, a bounded 100 ms allowance, then
`reset init`. That second command resets the independent watchdog and halts at
the reset boundary before candidate firmware executes again, so the complete
sector can be dumped without racing the 250 ms watchdog. The final
`reset run` starts the boot predicted from `J2` before the debugger
disconnects. Motor power remains physically disconnected and the UART remains
unowned:

```bash
sudo -v
require_no_serial_owner
test ! -e candidate-journal-j0.bin
test ! -e candidate-journal-j1.bin
test ! -e candidate-journal-j2.bin
install -m 0600 /dev/null candidate-journal-j0.bin
install -m 0600 /dev/null candidate-journal-j1.bin
install -m 0600 /dev/null candidate-journal-j2.bin
sudo /usr/bin/openocd \
  -s "$OPENOCD_SCRIPTS" \
  -f "$OPENOCD_INTERFACE" \
  -c "hla_serial $STLINK_SERIAL" \
  -c 'transport select hla_swd' \
  -f "$OPENOCD_TARGET" \
  -c 'reset_config srst_only srst_nogate connect_assert_srst' \
  -c init \
  -c 'reset init' \
  -c "dump_image {$EVIDENCE_DIR/candidate-journal-j0.bin} 0x08060000 0x20000" \
  -c 'reset run' \
  -c 'sleep 100' \
  -c 'reset init' \
  -c "dump_image {$EVIDENCE_DIR/candidate-journal-j1.bin} 0x08060000 0x20000" \
  -c 'reset run' \
  -c 'sleep 100' \
  -c 'reset init' \
  -c "dump_image {$EVIDENCE_DIR/candidate-journal-j2.bin} 0x08060000 0x20000" \
  -c 'reset run' \
  -c shutdown

for journal_snapshot in \
  candidate-journal-j0.bin \
  candidate-journal-j1.bin \
  candidate-journal-j2.bin
do
  test "$(stat -c %s "$journal_snapshot")" = 131072
done

test ! -e candidate-journal-j0-to-j1.json
test ! -e candidate-journal-j1-to-j2.json
"$JOURNAL_INSPECTOR" transition \
  --previous candidate-journal-j0.bin \
  --current candidate-journal-j1.bin \
  >candidate-journal-j0-to-j1.json
"$JOURNAL_INSPECTOR" transition \
  --previous candidate-journal-j1.bin \
  --current candidate-journal-j2.bin \
  >candidate-journal-j1-to-j2.json
test -s candidate-journal-j0-to-j1.json
test -s candidate-journal-j1-to-j2.json
sha256sum \
  candidate-journal-j0.bin \
  candidate-journal-j1.bin \
  candidate-journal-j2.bin \
  candidate-journal-j0-to-j1.json \
  candidate-journal-j1-to-j2.json
```

The second typed transition already contains the boot identity planned from
`J2`. Parse that JSON boundary once with duplicate-key and non-standard
constant rejection, require its closed schema, and retain only the predicted
next boot ID:

```bash
CANDIDATE_BOOT_ID_PREDICTED="$(
  "$PYTHON" -c '
import json
import sys

def reject_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

with open(sys.argv[1], "r", encoding="utf-8") as source:
    observed = json.load(
        source,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
if type(observed) is not dict:
    raise ValueError("journal transition evidence must be an object")
expected_keys = {
    "schema_version",
    "observation_kind",
    "previous_path",
    "previous_sha256_hex",
    "current_path",
    "current_sha256_hex",
    "image_bytes",
    "target_flash_address_hex",
    "journal_schema_version",
    "previous_state",
    "current_state",
    "committed_boot",
    "current_planned_next_boot",
    "exact_planned_record_only",
    "evidence_boundary",
}
if set(observed) != expected_keys:
    raise ValueError("journal transition evidence keys differ")
if observed["schema_version"] != 1:
    raise ValueError("journal transition schema differs")
if observed["observation_kind"] != (
    "verified_stm32f446_kiko_boot_journal_transition"
):
    raise ValueError("journal transition kind differs")
if observed["image_bytes"] != 131072:
    raise ValueError("journal transition image length differs")
if observed["target_flash_address_hex"] != "0x08060000":
    raise ValueError("journal target address differs")
if observed["journal_schema_version"] != 1:
    raise ValueError("journal schema differs")
if observed["exact_planned_record_only"] is not True:
    raise ValueError("journal transition was not exact")
planned = observed["current_planned_next_boot"]
if type(planned) is not dict or set(planned) != {
    "counter", "boot_id", "record_offset_bytes"
}:
    raise ValueError("planned boot schema differs")
boot_id = planned["boot_id"]
if type(boot_id) is not int or not 0 < boot_id < (1 << 64):
    raise ValueError("planned boot ID is not a nonzero u64")
print(boot_id)
' \
    candidate-journal-j1-to-j2.json
)"
readonly CANDIDATE_BOOT_ID_PREDICTED
test -n "$CANDIDATE_BOOT_ID_PREDICTED"
```

The inspector proves the exact byte transitions, while the single OpenOCD
connection proves which controlled run windows separated them and starts the
predicted next boot. Running that final boot before offline inspection is
intentional: no UART writer or motion session exists, motor power is cut, and
any failed inspection is terminal. The subsequent exact UART identity check
also detects an unexpected debugger-disconnect reset. None of this proves
physical behavior.

The generic identity probe is read-only, but the earlier checker is
intentionally motor-inert-specific. Capture candidate observations under a
separate exact checker:

```bash
capture_and_check_candidate_identity() {
  local stem="$1"
  local identity_status=0

  sudo -v
  require_no_serial_owner
  test ! -e "$stem.json"
  test ! -e "$stem.stderr"
  test ! -e "$stem.exit-status"
  test ! -e "$stem.checked-values"
  "$IDENTITY_PROBE" \
    --serial-device "$STM32_SERIAL_BY_ID" \
    --timeout-ms 5000 \
    >"$stem.json" \
    2>"$stem.stderr" || identity_status=$?
  printf '%s\n' "$identity_status" >"$stem.exit-status"
  sha256sum \
    "$stem.json" \
    "$stem.stderr" \
    "$stem.exit-status"
  test "$identity_status" -eq 0
  test -s "$stem.json"
  test ! -s "$stem.stderr"

  "$PYTHON" -c '
import json
import sys

def reject_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

with open(sys.argv[1], "r", encoding="utf-8") as source:
    observed = json.load(
        source,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )

expected = {
    "schema_version": 2,
    "observation_kind": "read_only_krp2_controller_hello",
    "serial_by_id_path": sys.argv[2],
    "host_input_queue_cleared_before_observation": True,
    "initial_unknown_record_prefix_excluded": True,
    "controller_uid_hex": sys.argv[3],
    "firmware_abi": 2,
    "firmware_build_id": 135169,
    "actuator_config_fingerprint_hex": (
        "4b494b4f2d3450574d2d43414e443121"
    ),
    "capabilities_bits": 575,
    "supports_required_safety_capabilities": False,
    "maximum_absolute_pwm_percent": 30,
    "grants_motion_authority": True,
    "maximum_command_lease_ms": 250,
    "reported_output_state": "disabled",
    "reported_output_state_is_safe": True,
    "pwm_frequency_hz": 20000,
    "watchdog_nominal_period_ms": 250,
    "neutral_output": "both_low",
    "physical_stop_semantics": "unverified",
    "evidence_boundary": (
        "the host input queue was cleared once; subsequently delivered bytes "
        "through the first zero delimiter were excluded, including any upstream "
        "or in-flight bytes delivered after that clear; the result is one decoded "
        "software claim, no serial bytes were transmitted, and no physical "
        "behavior was observed"
    ),
}
if type(observed) is not dict:
    raise ValueError("candidate identity evidence must be a JSON object")
expected_keys = set(expected) | {"observed_boot_id"}
if set(observed) != expected_keys:
    raise ValueError(
        f"candidate identity keys differ: "
        f"missing={sorted(expected_keys - set(observed))}, "
        f"extra={sorted(set(observed) - expected_keys)}"
    )
for key, value in expected.items():
    actual = observed[key]
    if type(actual) is not type(value) or actual != value:
        raise ValueError(
            f"candidate identity field {key!r} differs: "
            f"expected {value!r}, got {actual!r}"
        )
boot_id = observed["observed_boot_id"]
if type(boot_id) is not int or not 0 < boot_id < (1 << 64):
    raise ValueError(f"invalid nonzero u64 candidate boot ID: {boot_id!r}")
print(observed["controller_uid_hex"], boot_id)
' \
    "$stem.json" \
    "$STM32_SERIAL_BY_ID" \
    "$EXPECTED_CONTROLLER_UID_HEX" \
    >"$stem.checked-values"
  test -s "$stem.checked-values"
  sha256sum "$stem.checked-values"
}

capture_and_check_candidate_identity candidate-identity-predicted
read -r CANDIDATE_CONTROLLER_UID_HEX CANDIDATE_BOOT_ID_OBSERVED trailing_value \
  <candidate-identity-predicted.checked-values
test -z "${trailing_value:-}"
test "$CANDIDATE_CONTROLLER_UID_HEX" = "$EXPECTED_CONTROLLER_UID_HEX"
test "$CANDIDATE_BOOT_ID_OBSERVED" = "$CANDIDATE_BOOT_ID_PREDICTED"
```

An older in-flight `J1`/`J2` Hello, a missing append, an extra reset that
produced `J4`, or any different identity fails the exact predicted-ID check.
The matched Hello remains only one parsed software claim. It does not prove
fault-clear session admission, an applied zero, motor behavior, or physical
stop semantics.

Keep motor power disconnected. Do not reuse the 20/50 Hz motor-inert
diagnostic qualifier: this candidate intentionally does not advertise that
diagnostic capability. The next serial writer must be the sole canonical
wheels-off qualifier in
[`nano-wheels-off-qualification.md`](nano-wheels-off-qualification.md). It
must admit the exact candidate class, acquire with sequence zero, obtain an
exact applied-zero receipt, disarm, and remain stopped through all fallible
preparation before any separately attended nonzero test.

## Journal-preserving rollback to motor-inert firmware

Prepare rollback before candidate start by retaining the already qualified
`motor-inert-main-384k.bin` and its pinned hash. A rollback never restores an
old full-bank image, old sector-7 snapshot, or option bytes. It replaces only
sectors 0 through 6 and preserves sector 7 at its latest write-boundary state.

On any candidate identity, startup, exact-zero, or cleanup failure, physically
disconnect motor power and keep it disconnected. Rollback is a new attended
operation, not a retry in the candidate evidence directory. Start a new
fail-fast shell, create a new mode-`0700` rollback directory, and bind the
already-qualified motor-inert main image and read-only identity probe from the
preserved successful evidence directory:

```bash
set -euo pipefail
umask 077

QUALIFIED_EVIDENCE_DIR=/ABSOLUTE/PRESERVED/SUCCESSFUL/EVIDENCE-DIRECTORY
ROLLBACK_EVIDENCE_DIR=/ABSOLUTE/NEW/NONEXISTENT/ROLLBACK-EVIDENCE-DIRECTORY
EXPECTED_MOTOR_INERT_MAIN_SHA256=270e553f5c18a53393f0234f334d3ccc71be32ac7827240b54c939c6d6def38d
EXPECTED_IDENTITY_PROBE_SHA256=f29f8d37b576605c2801d53803a003f96caa94bc9ef2a95d64e39b59334abeaf
EXPECTED_OPTION_BYTES_SHA256=d292558017cf9ca0a2e40e262a5c1daa4b305ccf084ce06128133d282f905115
EXPECTED_ST_FLASH_SHA256=9b7fa140274dd6ff1a45feeebea45e060316fbeba73f47b1f3e9d2a3ed0d6aeb
EXPECTED_OPENOCD_SHA256=6386b3a27752a4808c8cf1c580640fbbc4a87526fea23cc1a1b60bfb70ae90ef
EXPECTED_PYTHON_SHA256=c4408788fddefb8db9a0ba0ab56941d10fa52be0bf6bb423f07c3a2ca5fd9665
EXPECTED_OPENOCD_INTERFACE_SHA256=ad96ec170c21d923e98a386c7653ec230613baaae73e5f955a36307fb520840e
EXPECTED_OPENOCD_TARGET_SHA256=1037825f6e5c96b75256a2b7afa3d6faf19018611cdb8d27825f43c14c1ed314
EXPECTED_OPENOCD_SWJ_DP_SHA256=dfb3e88754c1ad6ce7562913dc4b37e76e6254abe4dc8faa05275c59f3cf85c2
EXPECTED_OPENOCD_MEM_HELPER_SHA256=ecc5a0cc16fd38199b39a668e8597dd2e86bc2b9a613a2bf03fec8bbf5652f96
STLINK_SERIAL=066EFF313946303143221230
STM32_SERIAL_BY_ID=/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066EFF313946303143221230-if02
EXPECTED_CONTROLLER_UID_HEX=2c0018001750314242353320
EXPECTED_ACTUATOR_CONFIG_FINGERPRINT_HEX=4b494b4f2d4e4f2d4143542d56312121
PYTHON=/usr/bin/python3
OPENOCD_SCRIPTS=/usr/share/openocd/scripts
OPENOCD_INTERFACE=/usr/share/openocd/scripts/interface/stlink.cfg
OPENOCD_TARGET=/usr/share/openocd/scripts/target/stm32f4x.cfg
OPENOCD_SWJ_DP=/usr/share/openocd/scripts/target/swj-dp.tcl
OPENOCD_MEM_HELPER=/usr/share/openocd/scripts/mem_helper.tcl
MOTOR_INERT_MAIN="$QUALIFIED_EVIDENCE_DIR/motor-inert-main-384k.bin"
IDENTITY_PROBE="$QUALIFIED_EVIDENCE_DIR/host-cargo-target/release/v2_identity_probe"
QUALIFIED_OPTION_BYTES="$QUALIFIED_EVIDENCE_DIR/option-a.bin"
readonly QUALIFIED_EVIDENCE_DIR ROLLBACK_EVIDENCE_DIR
readonly EXPECTED_MOTOR_INERT_MAIN_SHA256
readonly EXPECTED_IDENTITY_PROBE_SHA256 EXPECTED_OPTION_BYTES_SHA256
readonly EXPECTED_ST_FLASH_SHA256 EXPECTED_OPENOCD_SHA256 EXPECTED_PYTHON_SHA256
readonly EXPECTED_OPENOCD_INTERFACE_SHA256 EXPECTED_OPENOCD_TARGET_SHA256
readonly EXPECTED_OPENOCD_SWJ_DP_SHA256 EXPECTED_OPENOCD_MEM_HELPER_SHA256
readonly STLINK_SERIAL STM32_SERIAL_BY_ID
readonly EXPECTED_CONTROLLER_UID_HEX
readonly EXPECTED_ACTUATOR_CONFIG_FINGERPRINT_HEX
readonly PYTHON OPENOCD_SCRIPTS OPENOCD_INTERFACE OPENOCD_TARGET
readonly OPENOCD_SWJ_DP OPENOCD_MEM_HELPER
readonly MOTOR_INERT_MAIN IDENTITY_PROBE QUALIFIED_OPTION_BYTES

test -d "$QUALIFIED_EVIDENCE_DIR"
test -f "$MOTOR_INERT_MAIN"
test -x "$IDENTITY_PROBE"
test -f "$QUALIFIED_OPTION_BYTES"
test -L "$STM32_SERIAL_BY_ID"
test -c "$(readlink -f "$STM32_SERIAL_BY_ID")"
test -f "$OPENOCD_INTERFACE"
test -f "$OPENOCD_TARGET"
test -f "$OPENOCD_SWJ_DP"
test -f "$OPENOCD_MEM_HELPER"
test ! -e "$ROLLBACK_EVIDENCE_DIR"
install -d -m 0700 "$ROLLBACK_EVIDENCE_DIR"
cd "$ROLLBACK_EVIDENCE_DIR"
EVIDENCE_DIR="$ROLLBACK_EVIDENCE_DIR"
readonly EVIDENCE_DIR
```

Before continuing in that new shell, define `require_no_serial_owner` and
`capture_and_check_motor_inert_identity` by executing their exact definitions
from the earlier sections of this procedure. Do not weaken their status,
stderr, schema, duplicate-key, identity, or safe-output checks.

Re-run the complete target and tooling preflight in this new attended
operation. The hashes below are the pinned bytes from the qualified Nano
environment; a matching version string alone is insufficient:

```bash
test "$(stat -c %s "$MOTOR_INERT_MAIN")" = 393216
test "$(sha256sum "$MOTOR_INERT_MAIN" | cut -d ' ' -f 1)" = \
  "$EXPECTED_MOTOR_INERT_MAIN_SHA256"
test "$(sha256sum "$IDENTITY_PROBE" | cut -d ' ' -f 1)" = \
  "$EXPECTED_IDENTITY_PROBE_SHA256"
test "$(stat -c %s "$QUALIFIED_OPTION_BYTES")" = 16
test "$(sha256sum "$QUALIFIED_OPTION_BYTES" | cut -d ' ' -f 1)" = \
  "$EXPECTED_OPTION_BYTES_SHA256"

test "$(sha256sum /usr/bin/st-flash | cut -d ' ' -f 1)" = \
  "$EXPECTED_ST_FLASH_SHA256"
test "$(sha256sum /usr/bin/openocd | cut -d ' ' -f 1)" = \
  "$EXPECTED_OPENOCD_SHA256"
test "$(sha256sum "$PYTHON" | cut -d ' ' -f 1)" = \
  "$EXPECTED_PYTHON_SHA256"
test "$(sha256sum "$OPENOCD_INTERFACE" | cut -d ' ' -f 1)" = \
  "$EXPECTED_OPENOCD_INTERFACE_SHA256"
test "$(sha256sum "$OPENOCD_TARGET" | cut -d ' ' -f 1)" = \
  "$EXPECTED_OPENOCD_TARGET_SHA256"
test "$(sha256sum "$OPENOCD_SWJ_DP" | cut -d ' ' -f 1)" = \
  "$EXPECTED_OPENOCD_SWJ_DP_SHA256"
test "$(sha256sum "$OPENOCD_MEM_HELPER" | cut -d ' ' -f 1)" = \
  "$EXPECTED_OPENOCD_MEM_HELPER_SHA256"

test ! -e rollback-st-flash-version.txt
test ! -e rollback-openocd-version.txt
test ! -e rollback-python-version.txt
/usr/bin/st-flash --version >rollback-st-flash-version.txt
/usr/bin/openocd --version >rollback-openocd-version.txt 2>&1
"$PYTHON" --version >rollback-python-version.txt 2>&1
test "$(cat rollback-st-flash-version.txt)" = "v1.7.0"
test "$(sed -n '1p' rollback-openocd-version.txt)" = \
  "Open On-Chip Debugger 0.11.0"
test "$(cat rollback-python-version.txt)" = "Python 3.10.12"
sha256sum \
  /usr/bin/st-flash \
  /usr/bin/openocd \
  "$PYTHON" \
  "$OPENOCD_INTERFACE" \
  "$OPENOCD_TARGET" \
  "$OPENOCD_SWJ_DP" \
  "$OPENOCD_MEM_HELPER" \
  "$MOTOR_INERT_MAIN" \
  "$IDENTITY_PROBE" \
  "$QUALIFIED_OPTION_BYTES" \
  rollback-st-flash-version.txt \
  rollback-openocd-version.txt \
  rollback-python-version.txt
```

Enumerate USB/sysfs again and require exactly the one pinned ST-Link. This
does not attach to or reset the STM32:

```bash
test ! -e rollback-stlink-usb-inventory.json
"$PYTHON" -c '
import json
import os
import pathlib
import sys

serial_path = pathlib.Path(sys.argv[1])
expected_serial = sys.argv[2]
matches = []
for candidate in pathlib.Path("/sys/bus/usb/devices").iterdir():
    try:
        vendor = (candidate / "idVendor").read_text(encoding="ascii").strip()
        product = (candidate / "idProduct").read_text(encoding="ascii").strip()
    except (FileNotFoundError, NotADirectoryError, OSError):
        continue
    if (vendor, product) != ("0483", "374b"):
        continue
    try:
        serial = (candidate / "serial").read_text(encoding="ascii").strip()
    except (FileNotFoundError, OSError) as error:
        raise ValueError(
            f"ST-Link USB identity has no readable serial: {candidate}"
        ) from error
    matches.append((candidate.resolve(), serial))

if len(matches) != 1:
    raise ValueError(f"expected one 0483:374b ST-Link, found {len(matches)}")
usb_path, observed_serial = matches[0]
if observed_serial != expected_serial:
    raise ValueError(
        f"ST-Link serial differs: expected {expected_serial!r}, "
        f"got {observed_serial!r}"
    )

tty_name = pathlib.Path(os.path.realpath(serial_path)).name
tty_sysfs = pathlib.Path("/sys/class/tty", tty_name, "device").resolve()
if usb_path != tty_sysfs and usb_path not in tty_sysfs.parents:
    raise ValueError(
        f"persistent VCP {serial_path} is not under the exact ST-Link USB device"
    )

json.dump(
    {
        "schema_version": 1,
        "enumeration_kind": "usb_sysfs_no_target_attach",
        "usb_vendor_id_hex": "0483",
        "usb_product_id_hex": "374b",
        "stlink_serial": observed_serial,
        "persistent_vcp_path": str(serial_path),
        "resolved_tty_name": tty_name,
    },
    sys.stdout,
    sort_keys=True,
)
sys.stdout.write("\n")
' \
  "$STM32_SERIAL_BY_ID" \
  "$STLINK_SERIAL" \
  >rollback-stlink-usb-inventory.json
test -s rollback-stlink-usb-inventory.json
sha256sum rollback-stlink-usb-inventory.json
```

Read the complete 16-byte F446 option region twice using connect-under-reset.
Require both reads to match each other and the pinned qualified value. This is
the fresh read/write-protection and option-byte preflight; no option write,
unprotect, or mass erase is authorized:

```bash
sudo -v
require_no_serial_owner
test ! -e rollback-option-a.bin
test ! -e rollback-option-b.bin
install -m 0600 /dev/null rollback-option-a.bin
install -m 0600 /dev/null rollback-option-b.bin
sudo /usr/bin/st-flash --connect-under-reset \
  --serial "$STLINK_SERIAL" \
  --area=option read rollback-option-a.bin 0x10
sudo /usr/bin/st-flash --connect-under-reset \
  --serial "$STLINK_SERIAL" \
  --area=option read rollback-option-b.bin 0x10
test "$(stat -c %s rollback-option-a.bin)" = 16
test "$(stat -c %s rollback-option-b.bin)" = 16
cmp rollback-option-a.bin rollback-option-b.bin
cmp "$QUALIFIED_OPTION_BYTES" rollback-option-a.bin
test "$(sha256sum rollback-option-a.bin | cut -d ' ' -f 1)" = \
  "$EXPECTED_OPTION_BYTES_SHA256"
sha256sum rollback-option-a.bin rollback-option-b.bin
```

First obtain two fresh
connect-under-reset full-bank dumps in one halted OpenOCD session. These are
evidence; the later write session takes its own sector-7 boundary snapshot
because candidate execution between debugger sessions may legitimately append
another boot record:

```bash
sudo -v
require_no_serial_owner
test ! -e rollback-full-before-a.bin
test ! -e rollback-full-before-b.bin
install -m 0600 /dev/null rollback-full-before-a.bin
install -m 0600 /dev/null rollback-full-before-b.bin
sudo /usr/bin/openocd \
  -s "$OPENOCD_SCRIPTS" \
  -f "$OPENOCD_INTERFACE" \
  -c "hla_serial $STLINK_SERIAL" \
  -c 'transport select hla_swd' \
  -f "$OPENOCD_TARGET" \
  -c 'reset_config srst_only srst_nogate connect_assert_srst' \
  -c init \
  -c 'reset init' \
  -c "dump_image {$EVIDENCE_DIR/rollback-full-before-a.bin} 0x08000000 0x80000" \
  -c "dump_image {$EVIDENCE_DIR/rollback-full-before-b.bin} 0x08000000 0x80000" \
  -c shutdown
test "$(stat -c %s rollback-full-before-a.bin)" = 524288
test "$(stat -c %s rollback-full-before-b.bin)" = 524288
cmp rollback-full-before-a.bin rollback-full-before-b.bin
sha256sum rollback-full-before-a.bin rollback-full-before-b.bin
```

After that comparison passes, use a new connect-under-reset session. Dump the
latest sector 7, write only the known motor-inert main image, and read both
regions back before disconnect:

```bash
sudo -v
require_no_serial_owner
test ! -e rollback-sector7-write-boundary.bin
test ! -e rollback-inert-main-readback.bin
test ! -e rollback-sector7-after.bin
install -m 0600 /dev/null rollback-sector7-write-boundary.bin
install -m 0600 /dev/null rollback-inert-main-readback.bin
install -m 0600 /dev/null rollback-sector7-after.bin
sudo /usr/bin/openocd \
  -s "$OPENOCD_SCRIPTS" \
  -f "$OPENOCD_INTERFACE" \
  -c "hla_serial $STLINK_SERIAL" \
  -c 'transport select hla_swd' \
  -f "$OPENOCD_TARGET" \
  -c 'reset_config srst_only srst_nogate connect_assert_srst' \
  -c init \
  -c 'reset init' \
  -c "dump_image {$EVIDENCE_DIR/rollback-sector7-write-boundary.bin} 0x08060000 0x20000" \
  -c "flash write_image erase {$MOTOR_INERT_MAIN} 0x08000000 bin" \
  -c "verify_image {$MOTOR_INERT_MAIN} 0x08000000 bin" \
  -c "dump_image {$EVIDENCE_DIR/rollback-inert-main-readback.bin} 0x08000000 0x60000" \
  -c "dump_image {$EVIDENCE_DIR/rollback-sector7-after.bin} 0x08060000 0x20000" \
  -c shutdown

test "$(stat -c %s rollback-sector7-write-boundary.bin)" = 131072
test "$(stat -c %s rollback-inert-main-readback.bin)" = 393216
test "$(stat -c %s rollback-sector7-after.bin)" = 131072
cmp "$MOTOR_INERT_MAIN" rollback-inert-main-readback.bin
cmp rollback-sector7-write-boundary.bin rollback-sector7-after.bin
sha256sum \
  "$MOTOR_INERT_MAIN" \
  rollback-inert-main-readback.bin \
  rollback-sector7-write-boundary.bin \
  rollback-sector7-after.bin
```

Only after both comparisons pass may a separate exact-target command issue
`reset run`:

```bash
sudo -v
require_no_serial_owner
sudo /usr/bin/openocd \
  -s "$OPENOCD_SCRIPTS" \
  -f "$OPENOCD_INTERFACE" \
  -c "hla_serial $STLINK_SERIAL" \
  -c 'transport select hla_swd' \
  -f "$OPENOCD_TARGET" \
  -c 'reset_config srst_only srst_nogate connect_assert_srst' \
  -c init \
  -c 'reset run' \
  -c shutdown

capture_and_check_motor_inert_identity rollback-motor-inert-identity
```

Repeat the motor-inert transport qualification before any later journal
maintenance or candidate retry. If sector 7 is corrupt or cannot be proven
unchanged, remain on motor-inert firmware, keep motor power disconnected, and
provision a fresh CSPRNG journal through the complete inert-plus-journal
procedure. Never restore an older journal snapshot: doing so can reuse a
previously issued boot identity.

Candidate qualification remains attended and wheels-off. Production motion
remains closed until real default-off motor enable, driver-fault/E-stop
feedback, and an independent physical motor-power cut are implemented and
reviewed.

## Stop conditions

Stop without improvising on any of these:

- more or fewer than one exact ST-Link target;
- absent NRST or failed connect-under-reset;
- read-protection, write-protection, or option-byte disagreement;
- backup mismatch or unexpected file length;
- any file-backed ELF `LOAD` byte outside
  `0x08000000..0x08060000`;
- either independently located candidate build differing at the ELF, natural
  binary, or padded-main boundary;
- a generated image with the wrong length or component comparison;
- sector 7 changing during a main-only candidate or rollback write;
- any candidate journal transition other than exactly one canonically planned
  record, or a passive candidate boot ID other than the exact next prediction;
- any debugger warning suggesting fallback attach, unprotect, or mass erase;
- any rollback plan that would restore an old full-bank image, journal sector,
  or option-byte image;
- uncertain readback, reset state, identity, exact zero, or cleanup;
- motor power not physically and independently disconnected.
