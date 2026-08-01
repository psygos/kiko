"""Pure four-axis encoder-domain compliant hold for Kiko's live head owner.

The controller deliberately ignores the STS load/current registers: their sign
and physical units have not been qualified.  A touch is a sustained encoder
error against the sole owner's command.  Weak JSON values are parsed once into
an immutable policy before this state machine can be constructed.
"""

from dataclasses import dataclass
import math


JOINT_COUNT = 4
FOLLOWING = "FOLLOWING_EXPRESSION"
CONFIRMING = "CONFIRMING_CONTACT"
YIELDING = "YIELDING"
RELEASE_DWELL = "RELEASE_DWELL"
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


@dataclass(frozen=True)
class CompliantHeadPolicy:
    minimum_ticks: tuple
    maximum_ticks: tuple
    contact_entry_error_ticks: tuple
    contact_release_error_ticks: tuple
    maximum_yield_ticks: tuple
    maximum_command_step_ticks: tuple
    maximum_observed_step_ticks: tuple
    quiet_command_step_ticks: tuple
    holding_torque_limit_permille: tuple
    control_period_s: float
    maximum_observation_span_s: float
    contact_arm_dwell_s: float
    contact_acquisition_samples: int
    release_dwell_s: float
    recovery_duration_s: float
    follow_fraction: float

    @classmethod
    def parse(cls, raw, installed_torque_limits):
        if not isinstance(raw, dict):
            raise CompliantConfigError("compliant_hold must be an object")
        expected = {
            "minimum_ticks", "maximum_ticks", "contact_entry_error_ticks",
            "contact_release_error_ticks", "maximum_yield_ticks",
            "maximum_command_step_ticks", "maximum_observed_step_ticks",
            "quiet_command_step_ticks", "holding_torque_limit_permille",
            "control_period_ms", "maximum_observation_span_ms",
            "contact_arm_dwell_ms", "contact_acquisition_samples",
            "release_dwell_ms", "recovery_duration_ms", "follow_permille",
        }
        unknown = sorted(set(raw) - expected)
        missing = sorted(expected - set(raw))
        if unknown or missing:
            raise CompliantConfigError(
                f"compliant_hold fields mismatch: missing={missing} unknown={unknown}")

        minimum = _int4(raw["minimum_ticks"], "minimum_ticks", 0, 4095)
        maximum = _int4(raw["maximum_ticks"], "maximum_ticks", 0, 4095)
        entry = _int4(raw["contact_entry_error_ticks"],
                      "contact_entry_error_ticks", 1, 4095)
        release = _int4(raw["contact_release_error_ticks"],
                        "contact_release_error_ticks", 0, 4095)
        maximum_yield = _int4(raw["maximum_yield_ticks"],
                              "maximum_yield_ticks", 1, 4095)
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
        recovery_ms = _plain_int(raw["recovery_duration_ms"], "recovery_duration_ms")
        follow_permille = _plain_int(raw["follow_permille"], "follow_permille")
        if control_ms <= 0 or observation_ms <= 0 or observation_ms > control_ms:
            raise CompliantConfigError("observation span must fit inside control period")
        if arm_ms <= 0 or release_ms <= 0 or recovery_ms <= 0:
            raise CompliantConfigError("compliant dwell/recovery durations must be positive")
        if acquisition <= 0 or acquisition > 255:
            raise CompliantConfigError("contact_acquisition_samples must be in [1, 255]")
        if follow_permille <= 0 or follow_permille > 1000:
            raise CompliantConfigError("follow_permille must be in [1, 1000]")

        return cls(
            minimum, maximum, entry, release, maximum_yield, command_step,
            observed_step, quiet_step, holding_torque, control_ms / 1000.0,
            observation_ms / 1000.0, arm_ms / 1000.0, acquisition,
            release_ms / 1000.0, recovery_ms / 1000.0,
            follow_permille / 1000.0,
        )


@dataclass(frozen=True)
class CompliantStep:
    state: str
    target_ticks: tuple
    event: str = None


class CompliantHeadController:
    """Stateful planner; the serial owner remains responsible for all I/O."""

    def __init__(self, policy, initial_target_ticks, started_at):
        self.policy = policy
        self._admit_pose(initial_target_ticks, "initial target")
        self.state = FOLLOWING
        self.target = tuple(initial_target_ticks)
        self.previous_expression_target = tuple(initial_target_ticks)
        self.previous_observation = None
        self.quiescent_since = None
        self.contact_armed = False
        self.candidate_directions = None
        self.candidate_samples = 0
        self.return_target = None
        self.quiet_since = None
        self.recovery_start = None
        self.recovery_started_at = None
        self.reacquire_directions = None
        self.reacquire_samples = 0
        self.next_service_due = started_at
        self.fault = None

    @property
    def active(self):
        return self.state in (CONFIRMING, YIELDING, RELEASE_DWELL, RECOVERING)

    def _admit_pose(self, pose, field):
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

    def _admit_observation(self, positions, moving, observation_span_s):
        self._admit_pose(positions, "observation")
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
        return all(abs(actual - command) <= release
                   for actual, command, release in zip(
                       positions, self.target,
                       self.policy.contact_release_error_ticks))

    def _directions(self, positions):
        result = []
        for actual, command, threshold in zip(
                positions, self.target, self.policy.contact_entry_error_ticks):
            error = actual - command
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
            displacement = actual - origin
            offset = self._round_nearest(displacement * self.policy.follow_fraction)
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

    def _enter_yield(self, positions):
        self.state = YIELDING
        self.quiet_since = None
        self.target = self._yield_target(positions)
        return CompliantStep(self.state, self.target, "pet_contact")

    def service(self, now, expression_target_ticks, positions, moving,
                observation_span_s):
        if self.fault is not None:
            raise CompliantObservationError(f"compliant controller fault held: {self.fault}")
        if now + 1e-9 < self.next_service_due:
            raise CompliantObservationError("compliant service called before scheduled tick")
        try:
            self._admit_pose(expression_target_ticks, "expression target")
            self._admit_observation(positions, moving, observation_span_s)
        except CompliantObservationError as exc:
            self.fault = str(exc)
            self.state = FAULT_HELD
            raise

        expression_target = tuple(expression_target_ticks)
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
            elif not self.contact_armed:
                settled = self._inside_release(positions) and not any(moving)
                self.quiescent_since = (self.quiescent_since if settled
                                        else None)
                if settled and self.quiescent_since is None:
                    self.quiescent_since = now
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
                        result = self._enter_yield(positions)
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
                self.candidate_directions = None
                self.candidate_samples = 0
            else:
                directions = self._directions(positions)
                if not self._directions_continue(self.candidate_directions, directions):
                    self.state = FOLLOWING
                    self.target = expression_target
                    self.quiescent_since = None
                    self.contact_armed = False
                else:
                    self.candidate_samples += 1
                    if self.candidate_samples >= self.policy.contact_acquisition_samples:
                        result = self._enter_yield(positions)
                        event = result.event

        elif self.state in (YIELDING, RELEASE_DWELL):
            released = self._inside_release(positions) and not any(moving)
            if released:
                if self.quiet_since is None:
                    self.quiet_since = now
                self.state = RELEASE_DWELL
                if now - self.quiet_since >= self.policy.release_dwell_s:
                    self.state = RECOVERING
                    self.recovery_start = self.target
                    self.recovery_started_at = now
                    self.reacquire_directions = None
                    self.reacquire_samples = 0
                    event = "pet_recovering"
            else:
                self.quiet_since = None
                self.state = YIELDING
                self.target = self._yield_target(positions)

        elif self.state == RECOVERING:
            directions = self._directions(positions)
            if directions == (0, 0, 0, 0):
                self.reacquire_directions = None
                self.reacquire_samples = 0
            elif (self.reacquire_directions is not None and
                  self._directions_continue(self.reacquire_directions, directions)):
                self.reacquire_samples += 1
            else:
                self.reacquire_directions = directions
                self.reacquire_samples = 1
            if self.reacquire_samples >= self.policy.contact_acquisition_samples:
                result = self._enter_yield(positions)
                event = "pet_recontact"
            else:
                elapsed = now - self.recovery_started_at
                progress = self._minimum_jerk(elapsed, self.policy.recovery_duration_s)
                desired = tuple(self._round_nearest(start + (finish - start) * progress)
                                for start, finish in zip(
                                    self.recovery_start, self.return_target))
                self.target = self._command_step(desired)
                if elapsed >= self.policy.recovery_duration_s and self.target == self.return_target:
                    self.state = FOLLOWING
                    self.quiescent_since = None
                    self.contact_armed = False
                    event = "pet_returned"

        self.previous_expression_target = expression_target
        self.previous_observation = tuple(positions)
        self.next_service_due = now + self.policy.control_period_s
        return CompliantStep(self.state, self.target, event)
