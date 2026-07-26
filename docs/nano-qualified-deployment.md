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
/opt/kiko/deployment/nano-agent-launch-v3.json
/opt/kiko/deployment/native-runtime-v1.json
/opt/kiko/deployment/lib/...
/opt/kiko/deployment/models/opencv/haarcascade_frontalface_default.xml
/opt/kiko/deployment/models/opencv/haarcascade_profileface.xml
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
call `systemctl enable` until Gate B and the attended production qualification
have passed.

The compiled fixed 30-second device-presence window covers only late cold-boot
enumeration. Before any serial probe or torque operation, the process observes
the exact three serial-by-id character devices and exact OAK MXID in
`Available` or `InUse` state immediately, then pauses for at most 100 ms after
each unsuccessful observation until the fixed poll/sleep budget expires.
`Bootloader` and `Unknown` OAK states keep the wait active. This does not weaken
`Restart=no`: once one sequential composite polling pass reports all targets
present, each downstream serial probe or owner open and the OAK connect remains
a one-shot acquisition. In-use, permission, identity, USB-speed, protocol, and
all other acquisition failures end the start without another presence wait or
process restart. A signal stops the polling between observations and bounded
sleeps and between serial metadata calls before native OAK discovery. Native
discovery and filesystem calls already in progress remain operating-system
boundaries rather than cancellable tasks; no later component call starts after
the preceding boundary observes shutdown or deadline. Timeout evidence retains
the last complete composite snapshot, or explicitly has none when the first
pass stops partway through.

## Native runtime closure

The 2026-07-23 and 2026-07-26 read-only Nano audits found these source bytes:

| Role | Audited source SHA-256 |
| --- | --- |
| DepthAI Core | `0744500ab4f665af0641fd10881988146b73241212ac9523a86294e5737edae8` |
| dynamic calibration | `30730ae6d367dcd927be7081f6a21d3bc4af65d857421ea3d3776d4ac00c7c53` |
| libusb 1.0 dependency | `74eac03235e61b326ecb6532bd1d840f7b8fbaf55cfaa32b7e3079fc1208ede0` |
| ONNX Runtime 1.23.2 | `ab8c4363e06ac80b3d1279ea55ebea44e906c5b131ba783ff684a067540c0e94` |
| OpenCV Core, SONAME `libopencv_core.so.4.5d` | `3abc549967c52f594b2b597db44b0013c55edb2198e11f9110d564277eb00beb` |
| OpenCV Imgproc, SONAME `libopencv_imgproc.so.4.5d` | `15b2448af215493a79f4638cad8eefcb9b43f15926724caffbdbd06a9c018261` |
| OpenCV Objdetect, SONAME `libopencv_objdetect.so.4.5d` | `94d3ddfb2111e72658d4bd005d22fd0ce402f8ae45ff8a79e9f7bdbd9b194b0b` |

With `nano-face-perception`, `oak-sys/build.rs` requires dynamic OpenCV
libraries and emits direct link directives for `opencv_objdetect`,
`opencv_imgproc`, and `opencv_core` on Linux; archive-only discovery is
rejected because its transitive closure is not known. The detector directly
uses objdetect classifiers, imgproc color/equalization operations, and core
matrix/storage types. This source-level link contract still requires the
target ELF evidence below because the linker and target library SONAMEs are
authoritative.

The three audited regular OpenCV sources were the resolved
`/usr/lib/aarch64-linux-gnu/libopencv_{core,imgproc,objdetect}.so.4.5.4d`
files. The `.so.4.5d` objects in the system directory are symlinks; do not
submit those symlink paths as bundle sources. The renderer retains the regular
source bytes under the ELF SONAME filename in the closed deployment `lib`
directory.

These are source observations, not installed identities. Copy the reviewed
regular bytes into `/opt/kiko/deployment/lib` under the exact filenames needed
by the ELF `NEEDED` entries, then hash the installed regular files. Render
those installed names, byte ceilings, and lowercase installed hashes into
`native-runtime-v1.json`. The production qualifier requires exactly one
`depthai_core`, `dynamic_calibration`, `libusb_1_0`, `onnxruntime`,
`opencv_core`, `opencv_imgproc`, and `opencv_objdetect` role. It rejects any
other declared SONAME for the three OpenCV roles. That declaration check does
not parse the supplied ELF's internal `DT_SONAME`. The wheels-off renderer
requires the same seven native roles because its attended binary includes the
production dispatch, while continuing to reject unused face-cascade inputs.
The qualifier can parse repeated `runtime_dependency` entries, but renderer
input schema V1 deliberately accepts only the seven closed roles above. If a
final `readelf -d`/`ldd` review determines that another non-system library
must be copied, stop and extend the typed renderer input, evidence generation,
and tests before staging it; never hand-edit a rendered manifest. The
qualifier enumerates the entire installed `lib` directory and rejects missing
entries, symlinks, subdirectories, and every regular file not bound by that
manifest. A listed subset beside an unbound loadable library is not qualified.

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

This is intentionally not a claim that the entire OpenCV ABI is hermetic. A
read-only 2026-07-26 Nano inspection of OpenCV 4.5.4 found that
`libopencv_objdetect.so.4.5d` also needs system `opencv_dnn`,
`opencv_calib3d`, `opencv_imgproc`, and `opencv_core`; that observed graph
continued through system `opencv_features2d`, `opencv_flann`, protobuf, TBB,
and zlib. Core and imgproc are directly staged above. The remaining system
libraries are not content-bound by this bundle. An unrelated minimal
retained-byte parser proved that observed graph, not the final `kiko-slam`
ELF, so it cannot qualify the production executable.

The same read-only inspection reported arm64 package versions
`4.5.4+dfsg-9ubuntu4` for all seven observed OpenCV packages
(`core`, `imgproc`, `objdetect`, `dnn`, `calib3d`, `features2d`, and `flann`),
`3.12.4-1ubuntu7.22.04.4` for `libprotobuf23`, `2020.3-1ubuntu3` for
`libtbb2`, and `1:1.2.11.dfsg-2ubuntu9.2` for `zlib1g`. These are diagnostic
observations of the current image, not marker-bound identities or permission
to accept the same package names at different versions.

Before minting an offline-install marker for a newly built production binary,
retain and review all of these outputs on the target:

```bash
readelf -d /opt/kiko/bin/kiko-slam
readelf -d /opt/kiko/deployment/lib/libopencv_core.so.4.5d
readelf -d /opt/kiko/deployment/lib/libopencv_imgproc.so.4.5d
readelf -d /opt/kiko/deployment/lib/libopencv_objdetect.so.4.5d
/usr/bin/env -i PATH=/usr/bin:/bin LD_LIBRARY_PATH=/opt/kiko/deployment/lib /usr/bin/ldd /opt/kiko/bin/kiko-slam
dpkg-query -W libopencv-core4.5d libopencv-imgproc4.5d libopencv-objdetect4.5d libopencv-dnn4.5d libopencv-calib3d4.5d libopencv-features2d4.5d libopencv-flann4.5d libprotobuf23 libtbb2 zlib1g
```

The final ELF must name the three pinned OpenCV SONAMEs, each staged library's
`DT_SONAME` must equal its pinned filename, the loader trace must resolve each
of those three to `/opt/kiko/deployment/lib/<SONAME>`, and no entry may be
unresolved. Review every other non-system dependency against the target OS
image; either stage it as a content-bound `runtime_dependency` or record the
reviewed OS ABI prerequisite. Staging an additional dependency first requires
the typed renderer extension described above. Until that final-ELF evidence
exists, native runtime readiness is withheld. `LD_LIBRARY_PATH` gives the
production process first choice of the staged direct libraries; it does not
make the unstaged transitive OS libraries immutable.

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

Perform this attended, with motor power independently cut, the robot
restrained, and the head supported:

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

## Build and offline production-install phase

This production phase occurs only after attended wheel-on commissioning has
produced a separately reviewed plant and the production-motion promotion gate
in `docs/nano-wheel-attach-gate-2026-07-23.md` has passed. It is not the
pre-attachment wheels-off lane. Use the qualification bundle and one-shot
commissioning executable for those earlier gates; neither may be relabelled as
production.

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
offline marker using the commands below. Even after that offline step
succeeds, the service may be started only for the attended production
qualification after Gate B has passed. An `active` unit is not acceptance
evidence.

## Explicit post-production-qualification boot phase

Before the first attended production-qualification run, install the marker
parent and exact enablement-only drop-in:

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

Start the service explicitly and complete the entire production fault,
stop, and bounded-motion evidence matrix. Only after that evidence passes may
persistent boot enablement be added:

```bash
sudo systemctl start kiko-nano-agent.service
# Complete and preserve the attended production evidence, then:
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
