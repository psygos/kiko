# Kiko assembly geometry declaration — 2026-08-27

This file retains operator-supplied assembly facts without silently promoting
them into calibrated navigation transforms. Values are SI-normalized here;
their original wording remains part of the record.

## Declared facts

| Declaration | SI value | Current interpretation |
| --- | ---: | --- |
| head centre relative to OAK optical frame | 0.25 m above, 0.20 m behind | `[x_right,y_down,z_forward] = [0,-0.25,-0.20] m` |
| neutral head/OAK orientation | parallel | identity rotation |
| OAK housing from “centre” | 0.1485 m forward, 0.30 m up, 0 m left | reference named “centre” is not yet resolved |
| drive-axle midpoint height from ground | 0.05 m | base origin height declaration |
| footprint radius | 0.293 m | physical body-radius declaration before safety margin |
| highest body height | 0.585 m; 0.59 m conservative when looking up | vertical body-envelope declaration |
| camera mounting | parallel | reference axes for “parallel” still need to be named |

## What is already usable

The head-to-camera gaze translation and parallel neutral orientation are
already represented by the typed expression extrinsic. They govern gaze
geometry only. They do not define the tracking-camera-to-base transform used
by SLAM, IMU fusion, occupancy, or MPC.

The 0.293 m footprint can be retained as an operator declaration and candidate
input. A collision footprint still needs a separately chosen, documented
localization/control/geometry margin; the declaration must not be relabelled
as the final inflated navigation radius.

## Ambiguities that prevent calibration use

The phrase “OAK housing from centre” does not establish whether “centre” is
the drive-axle midpoint, chassis centre, head centre, or another fixture datum.
It also names the housing rather than the OAK optical centre. Consequently the
values `forward=0.1485 m`, `up=0.30 m`, `left=0` are not yet installed as
`tracking_camera_to_base`.

Before that transform can be produced, retain one measured record that names:

1. the exact base-frame origin and axis convention;
2. the exact OAK optical-centre datum rather than a housing surface;
3. the translation from base origin to optical centre;
4. the camera-to-base rotation, including the meaning and tolerance of
   “parallel”; and
5. the native OAK IMU-to-base rotation used by raw IMU calibration.

The drive-axle midpoint height does not by itself choose the global or local
obstacle slabs. Their floor rejection margins must include depth noise,
extrinsic uncertainty and floor unevenness. The 0.59 m body height is useful
for clearance reasoning but does not turn a 2-D occupancy grid into a 3-D
collision model.

No missing translation, rotation, clearance margin, or uncertainty is guessed
from these declarations.
