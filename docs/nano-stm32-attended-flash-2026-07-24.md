# Nano STM32 attended flash procedure — 2026-07-24

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
EXPECTED_FIRMWARE_REVISION=1c543c27185e5b41d54cc93ea40980406a573a7d
EXPECTED_MOTOR_INERT_ELF_SHA256=fe1f055c076d700fd65d0f02db2a55163f82b7789b3bb2bf8b0ee25ede130dcc
EXPECTED_MOTOR_INERT_MAIN_SHA256=270e553f5c18a53393f0234f334d3ccc71be32ac7827240b54c939c6d6def38d
STLINK_SERIAL=066EFF313946303143221230
STM32_SERIAL_BY_ID=/dev/serial/by-id/usb-STMicroelectronics_STM32_STLink_066EFF313946303143221230-if02
EXPECTED_CONTROLLER_UID_HEX=2c0018001750314242353320
CARGO=/home/makerspace/.cargo/bin/cargo
RUSTC=/home/makerspace/.cargo/bin/rustc
PYTHON=/usr/bin/python3
test -f "$REPO/Cargo.lock"
test -x "$CARGO"
test -x "$RUSTC"
test -x "$PYTHON"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_FIRMWARE_REVISION"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
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
`1c543c27185e5b41d54cc93ea40980406a573a7d` built independently from
`/home/makerspace/kiko-codex-native-check` and
`/home/makerspace/kiko-stm32-flash-5526fc0`, each remapped to
`/kiko-source`. The two ELFs compared byte-for-byte at SHA-256
`fe1f055c076d700fd65d0f02db2a55163f82b7789b3bb2bf8b0ee25ede130dcc`;
the two 393,216-byte padded images compared byte-for-byte at SHA-256
`270e553f5c18a53393f0234f334d3ccc71be32ac7827240b54c939c6d6def38d`.
The evidence roots are
`/home/makerspace/kiko-build-repro/1c543c2-path-{a,b}`. That rehearsal
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
    "actuator_config_fingerprint_hex": "4b494b4f2d4e4f2d4143542d56312121",
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

A successful fresh owner cleared only its host input queue once and excluded
subsequently delivered bytes through the first zero delimiter. It did not
prove that upstream or in-flight bytes were absent. Every record after that one
boundary remains strict.

Run separate 10-second 20 Hz and 50 Hz processes. The motor-inert token is
deliberately not session-unique. The qualifier admits a fresh idle-safe
heartbeat, never begins a session, and never sends PWM. Each run repeats the
privileged no-owner check, captures the real status, then parses and checks its
schema, exact identity, requested rate/window, zero-loss counts, and final
idle-safe heartbeat:

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
    --actuator-config-fingerprint-hex 4b494b4f2d4e4f2d4143542d56312121 \
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
boot_id = int(sys.argv[4], 10)
rate_hz = int(sys.argv[5], 10)
planned_periods = rate_hz * 10

def exact(path, expected):
    value = evidence
    for key in path:
        if type(value) is not dict or key not in value:
            raise ValueError(f"missing qualifier field: {path!r}")
        value = value[key]
    if type(value) is not type(expected) or value != expected:
        raise ValueError(
            f"qualifier field {path!r} differs: expected {expected!r}, got {value!r}"
        )

exact(("schema_version",), 2)
exact(("evidence_kind",), "motor_inert_krp2_uart_transport_qualification")
exact(("passed",), True)
exact(("serial_by_id_path",), serial_path)
exact(("receive_startup", "host_input_queue_cleared_before_observation"), True)
exact(("receive_startup", "initial_unknown_record_prefix_excluded"), True)
exact(
    ("receive_startup", "post_boundary_decode_policy"),
    "every complete record after the one initial synchronization delimiter is "
    "decoded strictly; a framing error fails the run",
)
exact(("identity", "controller_uid_hex"), controller_uid)
exact(("identity", "boot_id"), boot_id)
exact(("identity", "firmware_abi"), 2)
exact(("identity", "firmware_build_id"), 131074)
exact(
    ("identity", "actuator_config_fingerprint_hex"),
    "4b494b4f2d4e4f2d4143542d56312121",
)
exact(("identity", "capabilities_bits"), 319)
exact(("identity", "max_abs_pwm_percent"), 0)
exact(("identity", "output_state"), "disabled")
exact(("identity", "max_command_lease_ms"), 250)
exact(("identity", "watchdog_nominal_period_ms"), 250)
exact(("identity", "pwm_frequency_hz"), 20000)
exact(("identity", "neutral_output"), "both_low")
exact(("identity", "physical_stop_semantics"), "unverified")
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
print("qualified")
' \
    "$stem.json" \
    "$STM32_SERIAL_BY_ID" \
    "$EXPECTED_CONTROLLER_UID_HEX" \
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

Candidate firmware is installed only after this inert/journal readback and
identity evidence. Its main image must still end at `0x08060000`, and its
main-only write must preserve and independently re-read the provisioned sector
7. Its build must use the same `/kiko-source` remap and an independently
reproduced exact hash; the motor-inert hash does not authorize candidate
bytes. Candidate qualification remains attended and wheels-off.
Production motion remains closed until real default-off motor enable,
driver-fault/E-stop feedback, and an independent physical motor-power cut are
implemented and reviewed.

## Stop conditions

Stop without improvising on any of these:

- more or fewer than one exact ST-Link target;
- absent NRST or failed connect-under-reset;
- read-protection, write-protection, or option-byte disagreement;
- backup mismatch or unexpected file length;
- an ELF load address outside the executable region;
- a generated image with the wrong length or component comparison;
- any debugger warning suggesting fallback attach, unprotect, or mass erase;
- uncertain readback, reset state, identity, exact zero, or cleanup;
- motor power not physically and independently disconnected.
