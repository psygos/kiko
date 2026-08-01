"""Time-based organic motion primitives for Kiko's head and eyes.

Targets may change abruptly; emitted trajectories cannot. Each scalar channel
is a softly overdamped spring whose acceleration is jerk-, acceleration-, and
velocity-bounded. A long scheduler gap freezes the channel instead of
integrating a catch-up jump.
"""

from dataclasses import dataclass
import math


class MotionConfigError(ValueError):
    pass


class MotionInputError(RuntimeError):
    pass


def _finite_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MotionConfigError(f"{field} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise MotionConfigError(f"{field} must be a finite number")
    return parsed


def _plain_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int):
        raise MotionConfigError(f"{field} must be an integer")
    return value


def _float4(raw, field, minimum):
    value = raw.get(field)
    if not isinstance(value, list) or len(value) != 4:
        raise MotionConfigError(f"{field} must contain exactly four numbers")
    result = tuple(_finite_number(item, f"{field}[{index}]")
                   for index, item in enumerate(value))
    if any(item <= minimum for item in result):
        raise MotionConfigError(f"{field} values must be above {minimum}")
    return result


@dataclass(frozen=True)
class MotionAxisPolicy:
    minimum: float
    maximum: float
    response_hz: float
    maximum_velocity: float
    maximum_acceleration: float
    maximum_jerk: float
    maximum_interval_s: float

    def __post_init__(self):
        values = (
            self.minimum, self.maximum, self.response_hz,
            self.maximum_velocity, self.maximum_acceleration,
            self.maximum_jerk, self.maximum_interval_s,
        )
        if any(not math.isfinite(value) for value in values):
            raise MotionConfigError("motion policy values must be finite")
        if self.minimum >= self.maximum:
            raise MotionConfigError("motion policy has an empty range")
        if (self.response_hz <= 0 or self.maximum_velocity <= 0 or
                self.maximum_acceleration <= 0 or self.maximum_jerk <= 0 or
                self.maximum_interval_s <= 0):
            raise MotionConfigError("motion dynamics must be positive")


def parse_head_motion_policies(raw, limits):
    if not isinstance(raw, dict):
        raise MotionConfigError("head motion configuration must be an object")
    if (not isinstance(limits, (list, tuple)) or len(limits) != 4 or
            any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in limits)):
        raise MotionConfigError("head limits must contain four positive integers")
    response = _float4(raw, "head_motion_response_hz", 0.0)
    velocity = _float4(raw, "head_motion_max_velocity_ticks_s", 0.0)
    acceleration = _float4(
        raw, "head_motion_max_acceleration_ticks_s2", 0.0)
    jerk = _float4(raw, "head_motion_max_jerk_ticks_s3", 0.0)
    maximum_interval_ms = _plain_int(
        raw.get("head_motion_max_interval_ms"),
        "head_motion_max_interval_ms")
    if not 1 <= maximum_interval_ms <= 1_000:
        raise MotionConfigError(
            "head_motion_max_interval_ms must be in [1, 1000]")
    interval = maximum_interval_ms / 1000.0
    return tuple(
        MotionAxisPolicy(
            -float(limit), float(limit), response[index], velocity[index],
            acceleration[index], jerk[index], interval)
        for index, limit in enumerate(limits)
    )


@dataclass(frozen=True)
class MotionAxisStep:
    position: float
    velocity: float
    acceleration: float
    gap_reset: bool = False


class OrganicMotionAxis:
    DAMPING_RATIO = 1.4

    def __init__(self, policy, initial_position, started_at):
        self.policy = policy
        self.position = self._admit_position(initial_position, "initial_position")
        self.velocity = 0.0
        self.acceleration = 0.0
        self.last_update_at = self._admit_time(started_at, "started_at")

    @staticmethod
    def _admit_time(value, field):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MotionInputError(f"{field} must be a finite time")
        parsed = float(value)
        if not math.isfinite(parsed):
            raise MotionInputError(f"{field} must be a finite time")
        return parsed

    def _admit_position(self, value, field):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MotionInputError(f"{field} must be a finite position")
        parsed = float(value)
        if (not math.isfinite(parsed) or parsed < self.policy.minimum or
                parsed > self.policy.maximum):
            raise MotionInputError(
                f"{field} outside [{self.policy.minimum}, {self.policy.maximum}]")
        return parsed

    @staticmethod
    def _move_toward(current, target, maximum_delta):
        delta = max(-maximum_delta, min(maximum_delta, target - current))
        return current + delta

    def hold(self, now):
        admitted_now = self._admit_time(now, "now")
        if admitted_now < self.last_update_at:
            raise MotionInputError("motion clock regressed")
        self.last_update_at = admitted_now
        self.velocity = 0.0
        self.acceleration = 0.0
        return MotionAxisStep(self.position, 0.0, 0.0)

    def step(self, target, now):
        admitted_target = self._admit_position(target, "target")
        admitted_now = self._admit_time(now, "now")
        if admitted_now < self.last_update_at:
            raise MotionInputError("motion clock regressed")
        elapsed = admitted_now - self.last_update_at
        self.last_update_at = admitted_now
        if elapsed == 0:
            return MotionAxisStep(
                self.position, self.velocity, self.acceleration)
        if elapsed > self.policy.maximum_interval_s:
            self.velocity = 0.0
            self.acceleration = 0.0
            return MotionAxisStep(self.position, 0.0, 0.0, True)

        omega = 2.0 * math.pi * self.policy.response_hz
        desired_acceleration = (
            omega * omega * (admitted_target - self.position)
            - 2.0 * self.DAMPING_RATIO * omega * self.velocity)
        desired_acceleration = max(
            -self.policy.maximum_acceleration,
            min(self.policy.maximum_acceleration, desired_acceleration))
        next_acceleration = self._move_toward(
            self.acceleration, desired_acceleration,
            self.policy.maximum_jerk * elapsed)
        next_velocity = self.velocity + (
            self.acceleration + next_acceleration) * 0.5 * elapsed
        next_velocity = max(
            -self.policy.maximum_velocity,
            min(self.policy.maximum_velocity, next_velocity))
        next_position = self.position + (
            self.velocity + next_velocity) * 0.5 * elapsed

        new_error = admitted_target - next_position
        if (abs(new_error) < 0.01 and abs(next_velocity) < 0.05 and
                abs(next_acceleration) < 0.2):
            next_position = admitted_target
            next_velocity = 0.0
            next_acceleration = 0.0
        if next_position <= self.policy.minimum:
            next_position = self.policy.minimum
            next_velocity = max(0.0, next_velocity)
            next_acceleration = max(0.0, next_acceleration)
        elif next_position >= self.policy.maximum:
            next_position = self.policy.maximum
            next_velocity = min(0.0, next_velocity)
            next_acceleration = min(0.0, next_acceleration)

        self.position = next_position
        self.velocity = next_velocity
        self.acceleration = next_acceleration
        return MotionAxisStep(
            self.position, self.velocity, self.acceleration)


EYE_EXPRESSIONS = frozenset(
    ("neutral", "curious", "greet", "concerned", "sleepy"))


@dataclass(frozen=True)
class EyeIntent:
    gaze_x: int
    gaze_y: int
    lid: int
    pupil: int
    brightness: int
    expression: str
    blink: bool
    color: tuple

    @classmethod
    def bounded(cls, gaze_x=0, gaze_y=0, lid=80, pupil=550,
                brightness=500, expression="neutral", blink=False,
                color=(80, 180, 200)):
        values = (gaze_x, gaze_y, lid, pupil, brightness)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(float(value)) for value in values):
            raise MotionInputError("eye scalar intents must be finite numbers")
        if expression not in EYE_EXPRESSIONS:
            raise MotionInputError(f"unknown eye expression {expression!r}")
        if not isinstance(blink, bool):
            raise MotionInputError("blink must be boolean")
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            raise MotionInputError("eye color must contain three channels")
        admitted_color = []
        for index, value in enumerate(color):
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(float(value))):
                raise MotionInputError(
                    f"eye color channel {index} must be finite")
            admitted_color.append(int(round(max(0.0, min(255.0, float(value))))))
        return cls(
            int(round(max(-1000.0, min(1000.0, float(gaze_x))))),
            int(round(max(-1000.0, min(1000.0, float(gaze_y))))),
            int(round(max(0.0, min(1000.0, float(lid))))),
            int(round(max(0.0, min(1000.0, float(pupil))))),
            int(round(max(0.0, min(1000.0, float(brightness))))),
            expression, blink, tuple(admitted_color))

    def with_overrides(self, **overrides):
        values = {
            "gaze_x": self.gaze_x, "gaze_y": self.gaze_y,
            "lid": self.lid, "pupil": self.pupil,
            "brightness": self.brightness, "expression": self.expression,
            "blink": self.blink, "color": self.color,
        }
        values.update(overrides)
        return EyeIntent.bounded(**values)

    def wire_values(self):
        return {
            "gaze_x": self.gaze_x, "gaze_y": self.gaze_y,
            "lid": self.lid, "pupil": self.pupil,
            "brightness": self.brightness, "expression": self.expression,
            "blink": self.blink, "color": self.color,
        }


class OrganicEyeDynamics:
    EXPRESSION_DWELL_S = 0.25

    def __init__(self):
        self.axes = None
        self.expression = None
        self.pending_expression = None
        self.pending_since = None
        self.previous_blink_request = False
        self.gap_reset = False

    @staticmethod
    def _policies():
        return (
            MotionAxisPolicy(-1000, 1000, 1.0, 1500, 7000, 35_000, 0.15),
            MotionAxisPolicy(-1000, 1000, 1.0, 1500, 7000, 35_000, 0.15),
            MotionAxisPolicy(0, 1000, 0.9, 1200, 5000, 25_000, 0.15),
            MotionAxisPolicy(0, 1000, 0.7, 800, 3000, 12_000, 0.15),
            MotionAxisPolicy(0, 1000, 0.55, 700, 2200, 8_000, 0.15),
            MotionAxisPolicy(0, 255, 0.38, 180, 500, 1_800, 0.15),
            MotionAxisPolicy(0, 255, 0.38, 180, 500, 1_800, 0.15),
            MotionAxisPolicy(0, 255, 0.38, 180, 500, 1_800, 0.15),
        )

    @staticmethod
    def _values(intent):
        return (
            intent.gaze_x, intent.gaze_y, intent.lid, intent.pupil,
            intent.brightness, *intent.color,
        )

    def step(self, now, target):
        if not isinstance(target, EyeIntent):
            raise MotionInputError("eye dynamics require an EyeIntent")
        values = self._values(target)
        if self.axes is None:
            self.axes = tuple(
                OrganicMotionAxis(policy, value, now)
                for policy, value in zip(self._policies(), values))
            self.expression = target.expression
            output_values = values
        else:
            steps = tuple(axis.step(value, now)
                          for axis, value in zip(self.axes, values))
            self.gap_reset = any(step.gap_reset for step in steps)
            output_values = tuple(step.position for step in steps)

        if target.expression == self.expression:
            self.pending_expression = None
            self.pending_since = None
        elif target.expression != self.pending_expression:
            self.pending_expression = target.expression
            self.pending_since = float(now)
        elif float(now) - self.pending_since >= self.EXPRESSION_DWELL_S:
            self.expression = self.pending_expression
            self.pending_expression = None
            self.pending_since = None

        blink = target.blink and not self.previous_blink_request
        self.previous_blink_request = target.blink
        rounded = tuple(int(round(value)) for value in output_values)
        return EyeIntent.bounded(
            gaze_x=rounded[0], gaze_y=rounded[1], lid=rounded[2],
            pupil=rounded[3], brightness=rounded[4],
            expression=self.expression, blink=blink,
            color=rounded[5:8])
