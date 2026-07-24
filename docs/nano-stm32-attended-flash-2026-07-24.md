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

Recheck those versions and the exact one-target probe at the attended session.

## Fresh backup before any write

Run every command block in one fail-fast shell. Use a new mode-`0700` evidence
directory and never retry in that directory: a failed phase requires a new
directory and a complete restart from backup. This prevents a skipped or
failed read from leaving an older output eligible for comparison. `st-flash
read` and OpenOCD `dump_image` otherwise truncate an existing output file.

```bash
set -euo pipefail
umask 077

REPO=/ABSOLUTE/CLEAN/KIKO/CHECKOUT
EVIDENCE_DIR=/ABSOLUTE/NEW/NONEXISTENT/EVIDENCE-DIRECTORY
EXPECTED_FIRMWARE_REVISION=5526fc0de2f5d56fe2dea94010b09ef06c2949ff
EXPECTED_MOTOR_INERT_MAIN_SHA256=6974f25ce983a056f78f02180de8c4d018b4509b84314edc1ddc3b5077c02d49
STLINK_SERIAL=066EFF313946303143221230
CARGO=/home/makerspace/.cargo/bin/cargo
RUSTC=/home/makerspace/.cargo/bin/rustc
test -f "$REPO/Cargo.lock"
test -x "$CARGO"
test -x "$RUSTC"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_FIRMWARE_REVISION"
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)"
test ! -e "$EVIDENCE_DIR"
install -d -m 0700 "$EVIDENCE_DIR"
cd "$EVIDENCE_DIR"

"$CARGO" --version
"$RUSTC" --version --verbose

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
"$CARGO" build \
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

The build-only rehearsal on the Nano used a new explicit target directory at
revision `5526fc0de2f5d56fe2dea94010b09ef06c2949ff`. It reproduced ELF SHA-256
`2dee0a538c5475b3cf2d85b14ed739306820736a9c0c0bb9f8460bd287e33315`
and the exact 384 KiB SHA-256 bound above. Its evidence directory is
`/home/makerspace/kiko-hardware-evidence/20260724T0624IST-5526fc0-runbook-repro`.
That rehearsal performed no debugger, serial, firmware, or actuator operation.

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

The first host operation is the read-only `v2_identity_probe`. Require the
exact motor-inert identity above, disabled safe output, and no advertised
motion authority. Then run separate 10-second 20 Hz and 50 Hz
`v2_transport_qualify` processes using the newly observed current boot token.
The motor-inert token is deliberately not session-unique. The
qualifier admits a fresh idle-safe heartbeat, never begins a session, and
never sends PWM. Any loss, duplicate, reorder, timeout, stale heartbeat,
skipped deadline probe, or nonzero output observation fails the run.

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
invocation issue the exact `init` plus `reset run` sequence shown in the
main-only section. That reset is the explicit synchronization/start point, not
proof that no instruction ran earlier. Motor power remains disconnected while
the passive identity is rechecked. The reset vector belongs to the motor-inert
main image, not the journal suffix.

Candidate firmware is installed only after this inert/journal readback and
identity evidence. Its main image must still end at `0x08060000`, and its
main-only write must preserve and independently re-read the provisioned sector
7. Candidate qualification remains attended and wheels-off.
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
