# OAK-D S2 independent baseline evidence — 2026-07-29

This record corroborates the fresh live stereo baseline without relabelling a
second value obtained from the device EEPROM or live DepthAI calibration API.
It is a baseline-consistency record, not a complete camera-calibration,
camera-to-base, IMU-to-base, SLAM, or deployment claim.

The operator identified Kiko's installed camera as an OAK-D S2. The retained
upstream OAK-D S2 board declaration is:

```text
source:
  /home/makerspace/work/depthai-core-v34/shared/depthai_boards/boards/OAK-D-S2.json
retained copy:
  /home/makerspace/kiko-native-evidence/4238f14-oak-s2-board-baseline-20260729T005342+0530/OAK-D-S2.json
depthai-core commit:
  ba7a920a2568ea6eaaaebf3f92bbdb40924187ae
depthai-boards commit:
  1dda9d332864c9139616282d63b043cab0ee65fa
source SHA-256:
  b6c50050a9d45bd28d76102cfc44ff399591f10f5518b0eedc47d15b83c28281
```

The checked source file was clean at the recorded submodule commit. Its static
`CAM_B` (left mono) to `CAM_C` (right mono) design declaration gives
`specTranslation.x = -7.5` centimetres. The independent design baseline
magnitude is therefore exactly:

```text
0.075 m
```

The fresh live rectified-stereo observation retained at:

```text
/home/makerspace/kiko-native-evidence/4238f14-calibration-stereo-20260729T001905+0530/calibration.json
```

has SHA-256
`13cdf7c81036b9983f4f12f51b4353295468b88de896ec0d7bd3a37a1147503e`
and reports:

```text
0.07503394 m
```

The exact relative discrepancy against the live value is:

```text
abs(0.07503394 - 0.075) / 0.07503394 * 100
= 0.045232864 percent
```

The preparer parses those inputs as `f32`
(`0.07503394037485123` and `0.07500000298023224`), so its calculation is
`0.0452294%`. Both calculations are below the preparer's 2% consistency gate.
The preparer must retain the relationship as `independently_derived`, use the
live baseline in the generated artifact, and bind both exact source
identifiers and digests.

The earlier seeded Basalt value `0.077406368413474924 m` is neither selected
nor used as independent corroboration: it differs from the fresh live value
by approximately 3.0649% and originates in the earlier calibration chain.

This record does not establish that the physical camera assembly is
undamaged, the camera optical axis is level, the camera-to-base transform is
known, the native IMU axes are known, or any physical-motion gate has passed.
