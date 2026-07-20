# Nano wheels-off deployment template

This directory mirrors the intended `/opt/kiko/deployment` layout. It is a
template, not a deployment and not physical calibration evidence.

The identities containing `REPLACE`, the `DEAD...` OAK MXID, and the
`DE`/`AD` identity bytes are deliberate non-production sentinels. The head
pose bounds are deliberately descending, so the typed bench parser rejects
this template before device I/O. The two artifact files are placeholders whose
hashes only demonstrate the binding layout; they are not camera calibration or
drive-plant artifacts.

After read-only discovery, replace every sentinel and every hardware claim,
install qualified artifacts, recompute their SHA-256 byte arrays in
`device-inventory-v1.json`, and set reviewed bow/curl/yaw/roll pose windows.
The all-stream OAK launch explicitly requests and requires DepthAI `SUPER` USB
speed; startup reads the negotiated speed from the same exact-MXID device and
fails closed below that minimum. Those required transport fields are why the
bench launch document is `nano-wheels-off-bench-v2.json`; v1 is not redefined.
The RGB gaze geometry records the current assembly claim that the neutral head
origin is `[0,-0.25,-0.20]` metres in the OAK camera frame with parallel axes;
its `0.32 m` magnitude is inside the parser's conservative `1 m` assembly
limit. It enables geometry only and is not physical extrinsic or servo
calibration evidence.
Follow `docs/nano-wheels-off-bench.md`; never make the template launchable by
broadening all pose windows to the full encoder range.
