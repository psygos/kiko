"""Pure four-axis encoder-domain compliant hold for Kiko's live head owner.

The controller deliberately ignores the STS load/current registers: their sign
and physical units have not been qualified.  A touch is a sustained encoder
error against the sole owner's command.  Weak JSON values are parsed once into
an immutable policy before this state machine can be constructed.
"""

from dataclasses import dataclass
import math


JOINT_COUNT = 4
YAW = 2
ROLL = 3
FOLLOWING = "FOLLOWING_EXPRESSION"
CONFIRMING = "CONFIRMING_CONTACT"
YIELDING = "YIELDING"
RELEASE_DWELL = "RELEASE_DWELL"
RESTING = "RESTING"
RECOVERING = "RECOVERING"
FAULT_HELD = "FAULT_HELD"


class CompliantConfigError(ValueError):
    pass


class CompliantObservationError(RuntimeError):
    pass


def _plain_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompliantConfigError(f"{field} must be an integer")
    return value


def _int4(value, field, minimum, maximum):
    if not isinstance(value, list) or len(value) != JOINT_COUNT:
        raise CompliantConfigError(f"{field} must contain exactly four integers")
    parsed = tuple(_plain_int(item, f"{field}[{index}]")
                   for index, item in enumerate(value))
    if any(item < minimum or item > maximum for item in parsed):
        raise CompliantConfigError(
            f"{field} values must be in [{minimum}, {maximum}]")
    return parsed


def _signed_int4(value, field, magnitude):
    if not isinstance(value, list) or len(value) != JOINT_COUNT:
        raise CompliantConfigError(f"{field} must contain exactly four integers")
    parsed = tuple(_plain_int(item, f"{field}[{index}]")
                   for index, item in enumerate(value))
    if any(abs(item) > magnitude for item in parsed):
        raise CompliantConfigError(
            f"{field} magnitudes must not exceed {magnitude}")
    return parsed


@dataclass(frozen=True)
class CompliantHeadPolicy:
    minimum_ticks: tuple
    maximum_ticks: tuple
    maximum_baseline_error_ticks: tuple
    contact_entry_error_ticks: tuple
    contact_release_error_ticks: tuple
    maximum_yield_ticks: tuple
    contact_rest_pose_offset_ticks: tuple
    contact_directional_rest_offset_ticks: tuple
    maximum_command_step_ticks: tuple
    maximum_observed_step_ticks: tuple
    quiet_command_step_ticks: tuple
    holding_torque_limit_permille: tuple
    control_period_s: float
    maximum_observation_span_s: float
    contact_arm_dwell_s: float
    contact_acquisition_samples: int
    release_dwell_s: float
    rest_dwell_s: float
    rest_per_additional_joint_s: float
    maximum_rest_dwell_s: float
    recovery_duration_s: float
    recovery_per_additional_joint_fraction: float
    follow_fraction: float
    yield_static_release_s: float
    maximum_yield_dwell_s: float
    comfort_roll_tilt_ticks: int
    yield_torque_limit_permille: tuple
    tap_max_contact_s: float
    tap_recovery_s: float

    @classmethod
    def parse(cls, raw, installed_torque_limits):
        if not isinstance(raw, dict):
            raise CompliantConfigError("compliant_hold must be an object")
        expected = {
            "minimum_ticks", "maximum_ticks", "maximum_baseline_error_ticks",
            "contact_entry_error_ticks",
            "contact_release_error_ticks", "maximum_yield_ticks",
            "contact_rest_pose_offset_ticks",
            "contact_directional_rest_offset_ticks",
            "maximum_command_step_ticks", "maximum_observed_step_ticks",
            "quiet_command_step_ticks", "holding_torque_limit_permille",
            "control_period_ms", "maximum_observation_span_ms",
            "contact_arm_dwell_ms", "contact_acquisition_samples",
            "release_dwell_ms", "rest_dwell_ms",
            "rest_per_additional_joint_ms", "maximum_rest_dwell_ms",
            "recovery_duration_ms",
            "recovery_per_additional_joint_permille", "follow_permille",
            "yield_static_release_ms", "maximum_yield_dwell_ms",
            "comfort_roll_tilt_ticks", "yield_torque_limit_permille",
            "tap_max_contact_ms", "tap_recovery_ms",
        }
        unknown = sorted(set(raw) - expected)
        missing = sorted(expected - set(raw))
        if unknown or missing:
            raise CompliantConfigError(
                f"compliant_hold fields mismatch: missing={missing} unknown={unknown}")

        minimum = _int4(raw["minimum_ticks"], "minimum_ticks", 0, 4095)
        maximum = _int4(raw["maximum_ticks"], "maximum_ticks", 0, 4095)
        maximum_baseline = _int4(
            raw["maximum_baseline_error_ticks"],
            "maximum_baseline_error_ticks", 1, 4095)
        entry = _int4(raw["contact_entry_error_ticks"],
                      "contact_entry_error_ticks", 1, 4095)
        release = _int4(raw["contact_release_error_ticks"],
                        "contact_release_error_ticks", 0, 4095)
        maximum_yield = _int4(raw["maximum_yield_ticks"],
                              "maximum_yield_ticks", 1, 4095)
        rest_pose = _signed_int4(
            raw["contact_rest_pose_offset_ticks"],
            "contact_rest_pose_offset_ticks", 4095)
        directional_rest = _int4(
            raw["contact_directional_rest_offset_ticks"],
            "contact_directional_rest_offset_ticks", 0, 4095)
        command_step = _int4(raw["maximum_command_step_ticks"],
                             "maximum_command_step_ticks", 1, 4095)
        observed_step = _int4(raw["maximum_observed_step_ticks"],
                              "maximum_observed_step_ticks", 1, 4095)
        quiet_step = _int4(raw["quiet_command_step_ticks"],
                           "quiet_command_step_ticks", 0, 4095)
        holding_torque = _int4(raw["holding_torque_limit_permille"],
                               "holding_torque_limit_permille", 1, 1000)
        installed_torque = tuple(installed_torque_limits)
        if holding_torque != installed_torque:
            raise CompliantConfigError(
                "compliant holding torque must exactly match installed torque limits")

        for joint in range(JOINT_COUNT):
            if minimum[joint] >= maximum[joint]:
                raise CompliantConfigError(f"joint {joint} has an empty envelope")
            if release[joint] >= entry[joint]:
                raise CompliantConfigError(
                    f"joint {joint} release error must be below entry error")
            if maximum_yield[joint] < entry[joint]:
                raise CompliantConfigError(
                    f"joint {joint} maximum yield is below entry error")
            if maximum_yield[joint] > maximum[joint] - minimum[joint]:
                raise CompliantConfigError(
                    f"joint {joint} maximum yield exceeds its envelope")
            if abs(rest_pose[joint]) > maximum_yield[joint]:
                raise CompliantConfigError(
                    f"joint {joint} rest pose exceeds maximum yield")
            if abs(rest_pose[joint]) + directional_rest[joint] > maximum_yield[joint]:
                raise CompliantConfigError(
                    f"joint {joint} directional rest pose exceeds maximum yield")
            if observed_step[joint] < max(command_step[joint], entry[joint]):
                raise CompliantConfigError(
                    f"joint {joint} observed-step limit is internally inconsistent")

        control_ms = _plain_int(raw["control_period_ms"], "control_period_ms")
        observation_ms = _plain_int(
            raw["maximum_observation_span_ms"], "maximum_observation_span_ms")
        arm_ms = _plain_int(raw["contact_arm_dwell_ms"], "contact_arm_dwell_ms")
        acquisition = _plain_int(
            raw["contact_acquisition_samples"], "contact_acquisition_samples")
        release_ms = _plain_int(raw["release_dwell_ms"], "release_dwell_ms")
        rest_ms = _plain_int(raw["rest_dwell_ms"], "rest_dwell_ms")
        rest_additional_ms = _plain_int(
            raw["rest_per_additional_joint_ms"],
            "rest_per_additional_joint_ms")
        maximum_rest_ms = _plain_int(
            raw["maximum_rest_dwell_ms"], "maximum_rest_dwell_ms")
        recovery_ms = _plain_int(raw["recovery_duration_ms"], "recovery_duration_ms")
        recovery_additional_permille = _plain_int(
            raw["recovery_per_additional_joint_permille"],
            "recovery_per_additional_joint_permille")
        follow_permille = _plain_int(raw["follow_permille"], "follow_permille")
        static_release_ms = _plain_int(
            raw["yield_static_release_ms"], "yield_static_release_ms")
        maximum_yield_dwell_ms = _plain_int(
            raw["maximum_yield_dwell_ms"], "maximum_yield_dwell_ms")
        comfort_tilt = _plain_int(
            raw["comfort_roll_tilt_ticks"], "comfort_roll_tilt_ticks")
        if comfort_tilt < 0:
            raise CompliantConfigError(
                "comfort_roll_tilt_ticks must be non-negative")
        # Backdrivable yield: the serial owner drops torque limits to this
        # profile while YIELDING so a hand can physically move the head.
        # Floors are the OPERATOR HARD RULE in parser form, per joint from
        # the 2026-08-02 bench staircase: bow held statically at 300
        # permille (7 ticks drift) and curl at 200 (4 ticks) with the head
        # at natural; gravity-neutral yaw/roll get the generic 150 floor.
        yield_torque = _int4(raw["yield_torque_limit_permille"],
                             "yield_torque_limit_permille", 150, 1000)
        yield_floor = (300, 200, 150, 150)
        for joint in range(JOINT_COUNT):
            if yield_torque[joint] < yield_floor[joint]:
                raise CompliantConfigError(
                    f"joint {joint} yield torque below measured static-hold "
                    f"floor {yield_floor[joint]}")
            if yield_torque[joint] > holding_torque[joint]:
                raise CompliantConfigError(
                    f"joint {joint} yield torque exceeds holding torque")
        # Tap fast path: a brief single contact skips the rest liturgy and
        # recovers quickly so the character layer can answer with play.
        tap_contact_ms = _plain_int(
            raw["tap_max_contact_ms"], "tap_max_contact_ms")
        tap_recovery_ms = _plain_int(raw["tap_recovery_ms"], "tap_recovery_ms")
        if tap_contact_ms <= 0 or tap_recovery_ms <= 0:
            raise CompliantConfigError("tap timings must be positive")
        if tap_recovery_ms > recovery_ms:
            raise CompliantConfigError(
                "tap recovery must not exceed the full recovery duration")
        if (abs(rest_pose[ROLL]) + directional_rest[ROLL] + comfort_tilt
                > maximum_yield[ROLL]):
            raise CompliantConfigError(
                "comfort roll tilt plus rest offsets exceeds roll yield")
        if control_ms <= 0 or observation_ms <= 0 or observation_ms > control_ms:
            raise CompliantConfigError("observation span must fit inside control period")
        if static_release_ms < 3 * control_ms:
            raise CompliantConfigError(
                "yield_static_release_ms needs at least three control periods "
                "of evidence")
        if maximum_yield_dwell_ms <= static_release_ms:
            raise CompliantConfigError(
                "maximum_yield_dwell_ms must exceed yield_static_release_ms")
        if (arm_ms <= 0 or release_ms <= 0 or rest_ms <= 0 or
                recovery_ms <= 0):
            raise CompliantConfigError("compliant dwell/recovery durations must be positive")
        if rest_additional_ms < 0 or maximum_rest_ms < rest_ms:
            raise CompliantConfigError("compliant rest timing is inconsistent")
        if not 0 <= recovery_additional_permille <= 1000:
            raise CompliantConfigError(
                "recovery_per_additional_joint_permille must be in [0, 1000]")
        if acquisition <= 0 or acquisition > 255:
            raise CompliantConfigError("contact_acquisition_samples must be in [1, 255]")
        if follow_permille <= 0 or follow_permille > 1000:
            raise CompliantConfigError("follow_permille must be in [1, 1000]")
        for joint in range(JOINT_COUNT):
            yielded_at_entry = (
                entry[joint] * follow_permille + 500) // 1000
            retained_error = entry[joint] - yielded_at_entry
            if retained_error <= release[joint]:
                raise CompliantConfigError(
                    f"joint {joint} follow gain collapses contact/release "
                    "hysteresis at the entry boundary")

        return cls(
            minimum, maximum, maximum_baseline, entry, release, maximum_yield,
            rest_pose, directional_rest,
            command_step, observed_step, quiet_step, holding_torque,
            control_ms / 1000.0,
            observation_ms / 1000.0, arm_ms / 1000.0, acquisition,
            release_ms / 1000.0, rest_ms / 1000.0,
            rest_additional_ms / 1000.0, maximum_rest_ms / 1000.0,
            recovery_ms / 1000.0,
            recovery_additional_permille / 1000.0,
            follow_permille / 1000.0,
            static_release_ms / 1000.0,
            maximum_yield_dwell_ms / 1000.0,
            comfort_tilt,
            yield_torque,
            tap_contact_ms / 1000.0,
            tap_recovery_ms / 1000.0,
        )


@dataclass(frozen=True)
class CompliantStep:
    state: str
    target_ticks: tuple
    residual_error_ticks: tuple
    event: str = None


class CompliantHeadController:
    """Stateful planner; the serial owner remains responsible for all I/O."""

    def __init__(self, policy, initial_target_ticks, started_at):
        self.policy = policy
        self._admit_command_pose(initial_target_ticks, "initial target")
        self.state = FOLLOWING
        self.target = tuple(initial_target_ticks)
        self.previous_expression_target = tuple(initial_target_ticks)
        self.previous_observation = None
        self.quiescent_since = None
        self.contact_armed = False
        self.baseline_error = (0, 0, 0, 0)
        self.candidate_directions = None
        self.candidate_samples = 0
        self.return_target = None
        self.quiet_since = None
        self.recovery_start = None
        self.recovery_started_at = None
        self.rest_started_at = None
        self.reacquire_directions = None
        self.reacquire_samples = 0
        self.contact_directions = (0, 0, 0, 0)
        self.contact_rest_pose = self.policy.contact_rest_pose_offset_ticks
        self.active_rest_duration_s = self.policy.rest_dwell_s
        self.active_recovery_duration_s = self.policy.recovery_duration_s
        self.next_service_due = started_at
        self.fault = None
        self.yield_entered_at = None
        self.yield_stable_since = None
        self.yield_stable_reference = None
        self.rest_settled_at = None
        self.rest_prev_residual = None
        self.comfort_tilt_side = None
        # Touch-style field record: one dict per completed contact episode,
        # consumed by the character layer to choose a social response
        # (boop / play / affection). Facts only — no behavior in here.
        self.episode = None
        self.episode_prev_residual = None
        self.last_episode = None

    @property
    def active(self):
        return self.state in (
            CONFIRMING, YIELDING, RELEASE_DWELL, RESTING, RECOVERING)

    def note_observation_gap(self):
        """A control slot was skipped (slow bus cycle): the next observation
        is not adjacent to the previous one, so the per-slot discontinuity
        check must not treat ordinary motion across the gap as a yank."""
        self.previous_observation = None

    def _admit_command_pose(self, pose, field):
        if not isinstance(pose, (list, tuple)) or len(pose) != JOINT_COUNT:
            raise CompliantObservationError(f"{field} must contain four ticks")
        for joint, value in enumerate(pose):
            if isinstance(value, bool) or not isinstance(value, int):
                raise CompliantObservationError(f"{field}[{joint}] is not an integer")
            if not self.policy.minimum_ticks[joint] <= value <= self.policy.maximum_ticks[joint]:
                raise CompliantObservationError(
                    f"{field}[{joint}]={value} outside reviewed envelope "+
                    f"[{self.policy.minimum_ticks[joint]}, "+
                    f"{self.policy.maximum_ticks[joint]}]")

    def _admit_observation_pose(self, pose):
        if not isinstance(pose, (list, tuple)) or len(pose) != JOINT_COUNT:
            raise CompliantObservationError(
                "observation must contain four ticks")
        for joint, value in enumerate(pose):
            if isinstance(value, bool) or not isinstance(value, int):
                raise CompliantObservationError(
                    f"observation[{joint}] is not an integer")
            # minimum/maximum_ticks bound commands. At that boundary a person
            # must still be able to move the joint by the already-reviewed
            # maximum yield before the physical observation becomes a fault.
            excursion = self.policy.maximum_yield_ticks[joint]
            minimum = max(0, self.policy.minimum_ticks[joint] - excursion)
            maximum = min(4095, self.policy.maximum_ticks[joint] + excursion)
            if not minimum <= value <= maximum:
                raise CompliantObservationError(
                    f"observation[{joint}]={value} outside reviewed physical "+
                    f"envelope [{minimum}, {maximum}]")

    def _admit_observation(self, positions, moving, observation_span_s):
        self._admit_observation_pose(positions)
        if not isinstance(moving, (list, tuple)) or len(moving) != JOINT_COUNT:
            raise CompliantObservationError("moving must contain four booleans")
        if any(not isinstance(value, bool) for value in moving):
            raise CompliantObservationError("moving values must be booleans")
        if observation_span_s < 0 or observation_span_s > self.policy.maximum_observation_span_s:
            raise CompliantObservationError(
                f"observation span {observation_span_s:.6f}s exceeds reviewed maximum")
        if self.previous_observation is not None:
            for joint, (previous, actual) in enumerate(
                    zip(self.previous_observation, positions)):
                difference = abs(actual - previous)
                if difference > self.policy.maximum_observed_step_ticks[joint]:
                    raise CompliantObservationError(
                        f"joint {joint} observation discontinuity {difference} ticks")

    def _inside_release(self, positions):
        return all(abs(error) <= release for error, release in zip(
            self._residual_errors(positions),
            self.policy.contact_release_error_ticks))

    def _residual_errors(self, positions, target=None):
        command_ticks = self.target if target is None else target
        return tuple(actual - command - baseline
                     for actual, command, baseline in zip(
                         positions, command_ticks, self.baseline_error))

    def _directions(self, positions):
        result = []
        for error, threshold in zip(
                self._residual_errors(positions),
                self.policy.contact_entry_error_ticks):
            result.append(0 if abs(error) < threshold else (1 if error > 0 else -1))
        return tuple(result)

    @staticmethod
    def _directions_continue(previous, actual):
        return (any(old != 0 and old == new for old, new in zip(previous, actual))
                and all(old == 0 or new == 0 or old == new
                        for old, new in zip(previous, actual)))

    def _command_step(self, desired):
        target = []
        for current, want, maximum in zip(
                self.target, desired, self.policy.maximum_command_step_ticks):
            delta = max(-maximum, min(maximum, want - current))
            target.append(current + delta)
        return tuple(target)

    @staticmethod
    def _round_nearest(value):
        return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)

    def _yield_target(self, positions):
        desired = []
        for joint, actual in enumerate(positions):
            origin = self.return_target[joint]
            rest = self.contact_rest_pose[joint]
            displacement_from_rest = (
                actual - origin - self.baseline_error[joint] - rest)
            offset = rest + self._round_nearest(
                displacement_from_rest * self.policy.follow_fraction)
            maximum = self.policy.maximum_yield_ticks[joint]
            offset = max(-maximum, min(maximum, offset))
            desired.append(max(self.policy.minimum_ticks[joint],
                               min(self.policy.maximum_ticks[joint], origin + offset)))
        return self._command_step(tuple(desired))

    @staticmethod
    def _minimum_jerk(elapsed, duration):
        if elapsed >= duration:
            return 1.0
        u = max(0.0, elapsed / duration)
        return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))

    def recovery_progress(self, now):
        """Return the admitted [0, 1] progress of the active recovery."""
        if self.state != RECOVERING or self.recovery_started_at is None:
            return 0.0
        if (isinstance(now, bool) or not isinstance(now, (int, float)) or
                not math.isfinite(float(now)) or now < self.recovery_started_at):
            raise CompliantObservationError("invalid recovery progress time")
        return self._minimum_jerk(
            float(now) - self.recovery_started_at,
            self.active_recovery_duration_s)

    def _enter_yield(self, now, positions, directions, event="pet_contact"):
        active_joints = max(1, sum(direction != 0 for direction in directions))
        additional_joints = active_joints - 1
        self.contact_directions = tuple(directions)
        residual = self._residual_errors(positions)
        rest_pose = [
            base + direction * directional
            for base, direction, directional in zip(
                self.policy.contact_rest_pose_offset_ticks,
                directions,
                self.policy.contact_directional_rest_offset_ticks)]
        if directions[ROLL] == 0 and self.policy.comfort_roll_tilt_ticks > 0:
            # A pet on bow/curl alone still tilts the head into the hand,
            # cat-style. The side is chosen ONCE per contact episode, on the
            # fresh pet_contact whose residual is measured against an
            # untilted target — re-entries reuse it, because their residual
            # embeds the previous tilt and re-deriving would deterministically
            # flip the head side to side. Roll hint only when it clears
            # noise; else the yaw lean decides; favorite side when centered.
            if event == "pet_contact" or self.comfort_tilt_side is None:
                hint = (residual[ROLL] if abs(residual[ROLL]) >= 3
                        else residual[YAW])
                self.comfort_tilt_side = -1 if hint < 0 else 1
            rest_pose[ROLL] += (self.comfort_tilt_side
                                * self.policy.comfort_roll_tilt_ticks)
        self.contact_rest_pose = tuple(rest_pose)
        self.active_rest_duration_s = min(
            self.policy.maximum_rest_dwell_s,
            self.policy.rest_dwell_s
            + additional_joints * self.policy.rest_per_additional_joint_s)
        self.active_recovery_duration_s = self.policy.recovery_duration_s * (
            1.0 + additional_joints
            * self.policy.recovery_per_additional_joint_fraction)
        self.state = YIELDING
        self.quiet_since = None
        self.rest_started_at = None
        self.reacquire_directions = None
        self.reacquire_samples = 0
        self.yield_entered_at = float(now)
        self.yield_stable_since = float(now)
        self.yield_stable_reference = None
        self.rest_settled_at = None
        self.rest_prev_residual = None
        if event == "pet_contact" or self.episode is None:
            self.episode = {
                "started_at": float(now),
                "yield_entries": 0,
                "samples": 0,
                "peak_residual": [0, 0, 0, 0],
                "delta_accum": 0.0,
                "delta_samples": 0,
                "reached_rest": False,
                "reached_comfy": False,
                "tap": False,
            }
            self.episode_prev_residual = None
        else:
            # A recontact converts any tap into a sustained episode: the
            # boop flag must not survive into a long pet's classification.
            self.episode["tap"] = False
        self.episode["yield_entries"] += 1
        self.target = self._yield_target(positions)
        return CompliantStep(self.state, self.target, residual, event)

    def _recontact_confirmed(self, positions):
        directions = self._directions(positions)
        # Stillness rule, same as YIELDING's static release: only a residual
        # that MOVES is a returning pet. A static over-threshold residual is
        # pose sag or a motionless hand; counting it re-entered yield forever
        # (sag > entry knead-cycled indefinitely and face-follow never
        # resumed) and raced the rest-pose travel so bow never went comfy.
        residual = self._residual_errors(positions)
        varying = (self.rest_prev_residual is not None and any(
            abs(actual - previous) > 3
            for actual, previous in zip(residual, self.rest_prev_residual)))
        self.rest_prev_residual = residual
        if not varying and directions != (0, 0, 0, 0):
            return False
        if directions == (0, 0, 0, 0):
            self.reacquire_directions = None
            self.reacquire_samples = 0
        elif (self.reacquire_directions is not None and
              self._directions_continue(self.reacquire_directions, directions)):
            self.reacquire_directions = tuple(
                old if old != 0 else new for old, new in zip(
                    self.reacquire_directions, directions))
            self.reacquire_samples += 1
        else:
            self.reacquire_directions = directions
            self.reacquire_samples = 1
        return self.reacquire_samples >= self.policy.contact_acquisition_samples

    def _rest_target(self):
        return tuple(max(self.policy.minimum_ticks[joint],
                         min(self.policy.maximum_ticks[joint],
                             self.return_target[joint]
                             + self.contact_rest_pose[joint]))
                     for joint in range(JOINT_COUNT))

    def service(self, now, expression_target_ticks, positions, moving,
                observation_span_s):
        if self.fault is not None:
            raise CompliantObservationError(f"compliant controller fault held: {self.fault}")
        if now + 1e-9 < self.next_service_due:
            raise CompliantObservationError("compliant service called before scheduled tick")
        try:
            self._admit_command_pose(
                expression_target_ticks, "expression target")
            self._admit_observation(positions, moving, observation_span_s)
        except CompliantObservationError as exc:
            self.fault = str(exc)
            self.state = FAULT_HELD
            raise

        expression_target = tuple(expression_target_ticks)
        # The serial owner has already issued expression_target as the
        # physical command. Compute diagnostics against that exact generation,
        # not the planner's preceding target.
        observed_residual = self._residual_errors(positions, expression_target)
        expression_quiet = all(
            abs(actual - previous) <= maximum
            for actual, previous, maximum in zip(
                expression_target, self.previous_expression_target,
                self.policy.quiet_command_step_ticks))
        event = None

        if self.state == FOLLOWING:
            self.target = expression_target
            if not expression_quiet:
                self.quiescent_since = None
                self.contact_armed = False
                self.baseline_error = (0, 0, 0, 0)
            elif not self.contact_armed:
                candidate_baseline = tuple(
                    actual - command
                    for actual, command in zip(positions, self.target))
                # Natural tracking bias and touch sensitivity are distinct
                # physical quantities. A bounded stopped bias may be larger
                # than the residual contact threshold, but an observation
                # outside the separately reviewed bias envelope cannot be
                # quietly reclassified as gravity.
                settled = (not any(moving) and all(
                    abs(error) <= maximum for error, maximum in zip(
                        candidate_baseline,
                        self.policy.maximum_baseline_error_ticks)))
                self.quiescent_since = (self.quiescent_since if settled
                                        else None)
                if settled and self.quiescent_since is None:
                    self.quiescent_since = now
                if settled:
                    self.baseline_error = candidate_baseline
                else:
                    self.baseline_error = (0, 0, 0, 0)
                if (self.quiescent_since is not None and
                        now - self.quiescent_since >= self.policy.contact_arm_dwell_s):
                    self.contact_armed = True
                    event = "pet_ready"
            else:
                directions = self._directions(positions)
                if directions != (0, 0, 0, 0):
                    self.return_target = self.target
                    self.candidate_directions = directions
                    self.candidate_samples = 1
                    if self.policy.contact_acquisition_samples == 1:
                        result = self._enter_yield(now, positions, directions)
                        event = result.event
                    else:
                        self.state = CONFIRMING
                        event = "pet_candidate"

        elif self.state == CONFIRMING:
            if not expression_quiet or expression_target != self.return_target:
                self.state = FOLLOWING
                self.target = expression_target
                self.quiescent_since = None
                self.contact_armed = False
                self.baseline_error = (0, 0, 0, 0)
                self.candidate_directions = None
                self.candidate_samples = 0
            else:
                directions = self._directions(positions)
                if not self._directions_continue(self.candidate_directions, directions):
                    self.state = FOLLOWING
                    self.target = expression_target
                    self.quiescent_since = None
                    self.contact_armed = False
                    self.baseline_error = (0, 0, 0, 0)
                else:
                    self.candidate_directions = tuple(
                        old if old != 0 else new for old, new in zip(
                            self.candidate_directions, directions))
                    self.candidate_samples += 1
                    if self.candidate_samples >= self.policy.contact_acquisition_samples:
                        result = self._enter_yield(
                            now, positions, self.candidate_directions)
                        event = result.event

        elif self.state in (YIELDING, RELEASE_DWELL):
            if self.episode is not None:
                current = self._residual_errors(positions)
                self.episode["samples"] += 1
                for joint in range(JOINT_COUNT):
                    self.episode["peak_residual"][joint] = max(
                        self.episode["peak_residual"][joint],
                        abs(current[joint]))
                if self.episode_prev_residual is not None:
                    self.episode["delta_accum"] += max(
                        abs(actual - previous)
                        for actual, previous in zip(
                            current, self.episode_prev_residual))
                    self.episode["delta_samples"] += 1
                self.episode_prev_residual = current
            released = self._inside_release(positions) and not any(moving)
            if released:
                if self.quiet_since is None:
                    self.quiet_since = now
                # The stability clock must not keep accruing across a
                # YIELDING<->RELEASE_DWELL bounce, or static release fires
                # early with stale evidence.
                self.yield_stable_reference = None
                self.yield_stable_since = now
                self.state = RELEASE_DWELL
                tap = (self.episode is not None and
                       self.episode["yield_entries"] == 1 and
                       not self.episode["reached_rest"] and
                       self.yield_entered_at is not None and
                       now - self.yield_entered_at
                       <= self.policy.tap_max_contact_s + 1e-9)
                if tap:
                    # A brief single touch is a boop, not a request to
                    # settle: skip the rest liturgy, recover fast, and let
                    # the character layer answer with play.
                    self.episode["tap"] = True
                    self.state = RECOVERING
                    self.recovery_start = self.target
                    self.recovery_started_at = now
                    self.active_recovery_duration_s = (
                        self.policy.tap_recovery_s)
                    self.reacquire_directions = None
                    self.reacquire_samples = 0
                    event = "pet_tap"
                elif now - self.quiet_since >= self.policy.release_dwell_s:
                    self.state = RESTING
                    self.rest_started_at = now
                    self.rest_settled_at = None
                    self.rest_prev_residual = None
                    self.reacquire_directions = None
                    self.reacquire_samples = 0
                    event = "pet_resting"
                    if self.episode is not None:
                        self.episode["reached_rest"] = True
            else:
                self.quiet_since = None
                self.state = YIELDING
                # YIELDING means ACTIVE contact: a live pet varies the
                # residual sample to sample. A statue-still residual is a
                # resting hand or pose-dependent gravity sag (2026-08-02
                # trance: bow sag differed 11 ticks between poses while the
                # release band is 5, so release was unreachable and the head
                # froze mid-pet for 20 minutes). Both belong in RESTING —
                # recontact detection catches a real hand, and recovery
                # re-learns the baseline, absorbing the sag.
                #
                # DESIGN NOTE (2026-08-02, operator-directed): at the
                # deployed follow fraction (0.65) the post-yield residual
                # sits just outside the release band by construction, so
                # this static exit is the PRIMARY end of a pet, not a
                # backstop — stillness is what a settling cat responds to.
                # The +-3 epsilon is deliberately at or below every joint's
                # maximum_command_step, so a residual still converging with
                # the target resets the reference and can never fake
                # stillness; only genuinely sub-epsilon contact sustains it.
                residual = self._residual_errors(positions)
                stable = (self.yield_stable_reference is not None and all(
                    abs(actual - reference) <= 3
                    for actual, reference in zip(
                        residual, self.yield_stable_reference)))
                if not stable:
                    self.yield_stable_reference = residual
                    self.yield_stable_since = now
                if (now - self.yield_stable_since
                        >= self.policy.yield_static_release_s):
                    event = "pet_release_static"
                elif (self.yield_entered_at is not None and
                      now - self.yield_entered_at
                      >= self.policy.maximum_yield_dwell_s):
                    # Bounds a continuously-varying residual with no
                    # recontact opportunity. A persistent active hand will
                    # re-enter yield from RESTING — that cycle is by design,
                    # not a hostage cap.
                    event = "pet_yield_timeout"
                if event is not None:
                    self.state = RESTING
                    self.rest_started_at = now
                    self.rest_settled_at = None
                    self.rest_prev_residual = None
                    self.reacquire_directions = None
                    self.reacquire_samples = 0
                    if self.episode is not None:
                        self.episode["reached_rest"] = True
                else:
                    self.target = self._yield_target(positions)

        elif self.state == RESTING:
            if self._recontact_confirmed(positions):
                result = self._enter_yield(
                    now, positions, self.reacquire_directions, "pet_recontact")
                event = result.event
            else:
                rest_target = self._rest_target()
                self.target = self._command_step(rest_target)
                if self.rest_settled_at is None and self.target == rest_target:
                    # Comfy pose reached: the pause is measured from HERE.
                    # Measuring from rest entry let travel eat the whole
                    # dwell, so the head stood up the instant it settled —
                    # which read as no pause at all.
                    self.rest_settled_at = now
                    event = "pet_comfy"
                    if self.episode is not None:
                        self.episode["reached_comfy"] = True
                settled = (self.rest_settled_at is not None and
                           now - self.rest_settled_at
                           >= self.active_rest_duration_s)
                # From-entry backstop covers a rest pose that stays out of
                # reach (a hand blocking travel) without a recontact firing:
                # worst travel is under 3 s at the per-joint command step.
                overdue = (now - self.rest_started_at
                           >= self.policy.maximum_rest_dwell_s + 3.0)
                if settled or overdue:
                    self.state = RECOVERING
                    self.recovery_start = self.target
                    self.recovery_started_at = now
                    self.reacquire_directions = None
                    self.reacquire_samples = 0
                    event = "pet_recovering"

        elif self.state == RECOVERING:
            if self._recontact_confirmed(positions):
                result = self._enter_yield(
                    now, positions, self.reacquire_directions, "pet_recontact")
                event = result.event
            else:
                elapsed = now - self.recovery_started_at
                progress = self._minimum_jerk(
                    elapsed, self.active_recovery_duration_s)
                desired = tuple(self._round_nearest(start + (finish - start) * progress)
                                for start, finish in zip(
                                    self.recovery_start, self.return_target))
                self.target = self._command_step(desired)
                if (elapsed >= self.active_recovery_duration_s and
                        self.target == self.return_target):
                    self.state = FOLLOWING
                    self.quiescent_since = None
                    self.contact_armed = False
                    self.baseline_error = (0, 0, 0, 0)
                    self.comfort_tilt_side = None
                    if self.episode is not None:
                        episode = self.episode
                        episode["duration_s"] = round(
                            float(now) - episode["started_at"], 2)
                        episode["mean_delta"] = round(
                            episode["delta_accum"]
                            / max(1, episode["delta_samples"]), 2)
                        self.last_episode = episode
                        self.episode = None
                        self.episode_prev_residual = None
                    event = "pet_returned"

        self.previous_expression_target = expression_target
        self.previous_observation = tuple(positions)
        self.next_service_due = now + self.policy.control_period_s
        return CompliantStep(self.state, self.target, observed_residual, event)
