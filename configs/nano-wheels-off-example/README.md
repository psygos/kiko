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
Follow `docs/nano-wheels-off-bench.md`; never make the template launchable by
broadening all pose windows to the full encoder range.
