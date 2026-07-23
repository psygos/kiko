# Nano offline install qualification and boot enablement

This procedure separates three different claims:

1. **Offline install qualification** proves that one root-owned installation
   passed the repository's bounded parsers and that the marker names the exact
   bytes inspected.
2. **Wheels-off qualification** proves selected live hardware behavior under
   the separately documented fault matrix.
3. **Wheel-on commissioning** measures the assembled encoderless plant.

The offline marker proves only the first claim. It never proves that a device
is connected, that calibration is physically correct, that motion is safe, or
that a process which remains active is ready.

## Immutable layout

The production layout is intentionally fixed:

```text
/opt/kiko/bin/kiko-slam
/opt/kiko/bin/kiko-nano-deployment-gate
/opt/kiko/deployment/nano-agent-launch-v2.json
/opt/kiko/deployment/native-runtime-v1.json
/opt/kiko/deployment/lib/...
/var/lib/kiko-nano-agent/
/etc/systemd/system/kiko-nano-agent.service
```

The launch document, policy, manifest, controller contract, navigation and
actuation documents, plant, models, and native manifest must name exact
deployment-relative files. Files qualified by the marker must be root-owned
regular files with one link and no group/world write bit. Source templates
containing replacement sentinels cannot mint a marker. Every directory from
the filesystem root to each qualified file must also be a real (not
symlinked), root-owned directory with no group/world write bit. Normalize
`/opt/kiko`, `/opt/kiko/bin`, `/opt/kiko/deployment`, and
`/opt/kiko/deployment/lib` explicitly before qualification; a safe final file
does not compensate for an unsafe ancestor.

The base unit has no `[Install]` section, always runs the root-privileged
offline-install verifier, uses `Restart=no`, and has no process-kill or
competing-owner cleanup. A missing marker, binary, launch file, or changed
bound byte fails every start; none is a `ConditionPath` skip. Keeping the
verifier in the base unit matters because an enablement symlink survives
removal of a drop-in. The production service therefore has no ungated
fallback start.

The qualifier compares the installed base unit byte-for-byte with the unit
compiled from this checkout. It also admits exactly one drop-in,
`10-qualified-boot.conf`, in the service-specific `/etc` directory and rejects
entries in the standard runtime, generator, control, vendor, service-prefix,
and type-wide drop-in directories that can affect this unit. This closes
later `ExecStartPre=` resets and duplicate overrides of `User`, `Restart`, or
`ExecStart`. `systemd-analyze verify` remains an additional Nano-side syntax
and dependency check; it does not replace the exact-byte and closed-directory
checks. If PID 1 was configured with a nonstandard system unit search path,
stop and review that path explicitly before treating the qualifier result as
complete.

Install the exact enablement-only file
`deploy/systemd/kiko-nano-agent-qualified-boot.conf` as:

```text
/etc/systemd/system/kiko-nano-agent.service.d/10-qualified-boot.conf
```

That drop-in adds only the `[Install]` section; it never owns the verifier.
The service process still runs as `makerspace`. `Restart=no` remains
unchanged, so a failed gate or runtime does not create a restart loop. Do not
call `systemctl enable` until the wheels-off gate has passed.

## Native runtime closure

The 2026-07-23 read-only Nano audit found these source bytes:

| Role | Audited source SHA-256 |
| --- | --- |
| DepthAI Core | `0744500ab4f665af0641fd10881988146b73241212ac9523a86294e5737edae8` |
| dynamic calibration | `30730ae6d367dcd927be7081f6a21d3bc4af65d857421ea3d3776d4ac00c7c53` |
| libusb 1.0 dependency | `74eac03235e61b326ecb6532bd1d840f7b8fbaf55cfaa32b7e3079fc1208ede0` |
| ONNX Runtime 1.23.2 | `ab8c4363e06ac80b3d1279ea55ebea44e906c5b131ba783ff684a067540c0e94` |

These are source observations, not installed identities. Copy the reviewed
regular bytes into `/opt/kiko/deployment/lib` under the exact filenames needed
by the ELF `NEEDED` entries, then hash the installed regular files. Render
those installed names, byte ceilings, and lowercase installed hashes into
`native-runtime-v1.json`. The qualifier requires exactly one
`depthai_core`, `dynamic_calibration`, `libusb_1_0`, and `onnxruntime` role.
Add a `runtime_dependency` entry for every other copied non-system library
found by the reviewed `readelf -d`/`ldd` closure. The qualifier enumerates the
entire installed `lib` directory and rejects missing entries, symlinks,
subdirectories, and every regular file not bound by that manifest. A listed
subset beside an unbound loadable library is not qualified.

Do not put Nano home paths in the deployment document. The audited
`libdepthai-core.so` RUNPATH points into `/home`, which is hidden by
`ProtectHome=true`; it is not a deployable runtime path. The base unit invokes
only the production process through:

```text
/usr/bin/env LD_LIBRARY_PATH=/opt/kiko/deployment/lib ...
```

The service does not set `LD_LIBRARY_PATH` globally. Consequently the
root-privileged pre-start verifier uses its normal system runtime and verifies
the copied libraries before the production executable can load them. Do not
copy libc, the ELF loader, `/usr/bin/env`, or other system-managed runtime
files into the deployment library directory.

The `onnxruntime` native-manifest path and digest must exactly equal the
launch document's ONNX Runtime asset binding. A successful load still proves
neither provider selection nor inference accuracy, latency, or speed.

## Exact Fable handoff

At the time of the audit, these two `makerspace` crontab entries owned restart
authority:

```text
@reboot /home/makerspace/kiko-follow/engine-guardian.sh
* * * * * pgrep -f 'engine-guardian[.]sh' >/dev/null || (setsid /home/makerspace/kiko-follow/engine-guardian.sh >/dev/null 2>&1 &)
```

The guardian respawned `kiko_face_follow.py` every eight seconds. The child
owned the exact OAK, head adapter, and eye controller. Stopping only that child
is not a handoff.

Perform this as an attended, wheels-off operation with the head supported:

1. Save the current user crontab and identify the exact guardian and child
   PIDs plus their parent/child relationship.
2. Remove only the two exact launch entries above from the `makerspace`
   crontab. Do not use a substring rewrite that could remove unrelated jobs.
3. Send normal termination to the exact guardian PID and wait for its cleanup.
   If its already-identified child remains only after the guardian is proven
   gone, terminate that exact child normally and retain the cleanup result.
4. Wait longer than the guardian's eight-second respawn interval and the
   one-minute cron interval. Re-read the crontab and process tree and prove
   neither process returned.
5. Prove no process owns the exact OAK USB node, head adapter, eye controller,
   STM32 by-id endpoint, production loopback endpoint, or Kiko runtime socket.
6. Start exactly one canonical owner. Its typed identity and exclusive-open
   failures remain authoritative; never bypass them because the process list
   looked empty.

Never use `pkill`, `killall`, a broad `pgrep`-driven kill, or automatic
service-side cleanup. Never start the standalone `kiko-robot-server` beside
the production agent. If an exact owner cannot be handed off normally, stop
and preserve the uncertainty.

## Build and manual wheels-off phase

Build the reviewed revision and lockfile natively on Linux aarch64:

```bash
cargo build --locked --release -p kiko-nano-deployment-gate --bin kiko-nano-deployment-gate
cargo build --locked --release -p kiko-slam --features nano-agent --bin kiko-slam --bin kiko-nano-deployment-qualify
```

Before installing those bytes, run the offline component acceptance suite:

```bash
tools/nano-cold-boot-fault-acceptance.sh
```

Its scope and explicit physical exclusions are defined in
`docs/nano-cold-boot-fault-acceptance.md`. Passing it does not authorize
deployment, service enablement, or wheel attachment.

Install the base unit, agent, gate, rendered deployment assets, complete
native runtime closure, and the enablement-only drop-in as root-owned files.
Verify the effective unit, but do not enable or start it yet:

```bash
sudo systemd-analyze verify /etc/systemd/system/kiko-nano-agent.service
sudo systemctl daemon-reload
```

Review the complete effective unit, then mint and immediately re-verify the
offline marker using the commands below. Only after that offline step succeeds
may the service be started explicitly for the live wheels-off gate in
`docs/nano-wheel-attach-gate-2026-07-23.md`. An `active` unit is not
acceptance evidence.

## Explicit post-qualification boot phase

Before the manual wheels-off run, install the marker parent and exact
enablement-only drop-in:

```bash
sudo install -d -o root -g root -m 0755 /etc/kiko
sudo install -d -o root -g root -m 0755 /etc/systemd/system/kiko-nano-agent.service.d
sudo install -o root -g root -m 0644 deploy/systemd/kiko-nano-agent-qualified-boot.conf /etc/systemd/system/kiko-nano-agent.service.d/10-qualified-boot.conf
sudo systemctl daemon-reload
sudo systemd-analyze verify /etc/systemd/system/kiko-nano-agent.service
```

Review the complete effective unit. There must be no other override which
clears `ExecStartPre`, changes `ExecStart`, changes `User`, injects a
service-wide `LD_LIBRARY_PATH`, enables restart, or adds a competing owner.
Then mint the marker with the exact acknowledgement:

```bash
sudo target/release/kiko-nano-deployment-qualify \
  --acknowledge "I reviewed this exact offline install; hardware and wheels-off gates remain separate"
sudo /opt/kiko/bin/kiko-nano-deployment-gate verify
```

The qualifier parses and binds the launch graph, referenced assets, policy,
manifest, manifest artifacts, canonical calibration artifact and its manifest
OAK MXID, controller contract, selected plant, native manifest, copied native
libraries, both runtime binaries, base unit, and qualified drop-in. Runtime
bootstrap additionally requires the calibration artifact to match the exact
opened OAK stereo geometry, navigation IMU/extrinsic values, and all three
production actuation approval IDs. The qualifier then atomically publishes a
mode-`0400` marker and immediately verifies it. The qualifier itself is not in
the boot path and is not self-hashed.

Start the service explicitly and complete the entire wheels-off evidence
matrix. Only after that evidence passes may persistent boot enablement be
added:

```bash
sudo systemctl start kiko-nano-agent.service
# Complete and preserve the wheels-off evidence, then:
sudo systemctl stop kiko-nano-agent.service
sudo systemctl enable kiko-nano-agent.service
sudo systemctl start kiko-nano-agent.service
sudo systemctl --no-pager --full status kiko-nano-agent.service
```

Any change to a bound file makes the pre-start gate fail. Requalification
requires stopping the service, reviewing the new complete install, rerunning
the qualifier, and preserving new evidence. Do not repair a failed boot by
removing the gate, weakening ownership/mode checks, adding restart behavior,
or killing a competing process.
