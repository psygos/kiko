# Production Nano launch template

This directory is a source template, not a qualified deployment and not
evidence that any device or motion contract is satisfied. The
`nano-agent-launch-v1.json.template` file deliberately contains `${...}`
tokens, so it is not JSON and cannot be admitted until a deployment tool has
replaced every token with measured or independently reviewed values.

A qualified deployment must:

1. install the binary and every input asset under a root-owned, non-writable
   deployment root such as `/opt/kiko/deployment`;
2. hash the exact installed bytes for every asset and place the lowercase
   SHA-256 in the rendered launch document;
3. use the byte count (or a deliberately smaller reviewed ceiling) for each
   `maximum_bytes` field;
4. bind the plant ID and bytes to the exact admitted device manifest;
5. cross-check the controller endpoint and contract against both the
   robot-server configuration and physical-actuation configuration;
6. use the exact OAK stream dimensions and rates selected during wheels-off
   qualification;
7. set every `occupancy` field from the reviewed global-map resource envelope.
   This section owns only grid extent, maximum retained evidence, and snapshot
   cadence. The exact `navigation-shadow-v1.json` owns the level
   optical-world/camera-height transform, runtime rectified-left depth camera
   and intrinsics, height/depth ranges, and sampling block; do not duplicate
   or override them with environment variables;
8. select inference backends only as requested providers. A selection is not
   a claim of availability, compatibility, latency, throughput, or speedup;
9. size state quotas from available storage and enforce them before writes;
10. install the rendered document as
   `/opt/kiko/deployment/nano-agent-launch-v1.json`; and
11. start `kiko-nano-agent.service` manually for qualification. The supplied
    service has no `[Install]` section and therefore is not automatically
    enabled at boot.

Parsing the rendered document proves only structural validity and equality
with its content bindings. Runtime admission must still verify exact
inventory, OAK identity and SuperSpeed readback, model loading, controller
session/receipts, accessory health, calibration, plant evidence, and physical
stop behavior.
