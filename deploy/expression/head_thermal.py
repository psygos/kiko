"""Pure thermal-derate policy for Kiko's four-axis head.

Only bow and curl carry the pitch workload which derating suppresses. Raw
servo temperature bytes are admitted once here, and isolated bus transients
cannot latch pitch motion off indefinitely.
"""

from dataclasses import dataclass


JOINT_COUNT = 4
BOW = 0
CURL = 1


class ThermalConfigError(ValueError):
    pass


class ThermalObservationError(RuntimeError):
    pass


def _plain_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ThermalConfigError(f"{field} must be an integer")
    return value


@dataclass(frozen=True)
class ThermalDeratePolicy:
    engage_temperature_raw: int
    clear_temperature_raw: int
    abort_temperature_raw: int
    engage_samples: int
    clear_samples: int
    plausible_temperature_raw: int = 95

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict):
            raise ThermalConfigError("thermal configuration must be an object")
        engage = _plain_int(raw.get("derate_temp_raw"), "derate_temp_raw")
        clear = _plain_int(
            raw.get("derate_clear_temp_raw"), "derate_clear_temp_raw")
        abort = _plain_int(raw.get("temp_abort_raw"), "temp_abort_raw")
        engage_samples = _plain_int(
            raw.get("derate_confirm_samples"), "derate_confirm_samples")
        clear_samples = _plain_int(
            raw.get("derate_clear_samples"), "derate_clear_samples")
        plausible = _plain_int(
            raw.get("temp_plausible_max_raw", 95), "temp_plausible_max_raw")
        if not 1 <= engage < abort <= 255:
            raise ThermalConfigError(
                "temperature thresholds must satisfy "
                "1 <= derate_temp_raw < temp_abort_raw <= 255")
        if not 0 <= clear < engage:
            raise ThermalConfigError(
                "derate_clear_temp_raw must be below derate_temp_raw")
        if not 1 <= engage_samples <= 255:
            raise ThermalConfigError(
                "derate_confirm_samples must be in [1, 255]")
        if not 1 <= clear_samples <= 255:
            raise ThermalConfigError(
                "derate_clear_samples must be in [1, 255]")
        if not abort <= plausible <= 255:
            raise ThermalConfigError(
                "temp_plausible_max_raw must satisfy "
                "temp_abort_raw <= temp_plausible_max_raw <= 255")
        return cls(engage, clear, abort, engage_samples, clear_samples,
                   plausible)


@dataclass(frozen=True)
class ThermalDerateStep:
    active: bool
    pitch_hottest_raw: int
    event: str = None


class ThermalDerateController:
    def __init__(self, policy):
        self.policy = policy
        self.active = False
        self._engage_count = 0
        self._clear_count = 0

    @staticmethod
    def _admit_temperatures(temperatures):
        if (not isinstance(temperatures, (list, tuple)) or
                len(temperatures) != JOINT_COUNT):
            raise ThermalObservationError(
                "temperatures must contain four raw integers")
        result = []
        for joint, value in enumerate(temperatures):
            if (isinstance(value, bool) or not isinstance(value, int) or
                    not 0 <= value <= 255):
                raise ThermalObservationError(
                    f"temperatures[{joint}] must be a raw byte")
            result.append(value)
        return tuple(result)

    def update(self, temperatures):
        admitted = self._admit_temperatures(temperatures)
        # Bytes above the plausibility ceiling are bus corruption (observed
        # checksum-valid 78->150->140 on 2026-08-02), not heat. A pair with
        # any corrupt pitch byte carries no usable thermal evidence — the
        # corrupted joint's true temperature is unknown — so it holds every
        # counter rather than advancing engagement or a cool streak.
        pitch_pair = (admitted[BOW], admitted[CURL])
        if any(value > self.policy.plausible_temperature_raw
               for value in pitch_pair):
            return ThermalDerateStep(self.active, max(pitch_pair), None)
        pitch_hottest = max(pitch_pair)
        event = None
        if not self.active:
            self._clear_count = 0
            self._engage_count = (
                self._engage_count + 1
                if pitch_hottest >= self.policy.engage_temperature_raw else 0)
            if self._engage_count >= self.policy.engage_samples:
                self.active = True
                self._engage_count = 0
                event = "thermal_derate_on"
        else:
            self._engage_count = 0
            if pitch_hottest <= self.policy.clear_temperature_raw:
                self._clear_count += 1
            else:
                # Tolerate isolated warm readings inside a cooling trend:
                # decrement instead of reset, so one flaky byte per second
                # cannot latch the derate on forever.
                self._clear_count = max(0, self._clear_count - 1)
            if self._clear_count >= self.policy.clear_samples:
                self.active = False
                self._clear_count = 0
                event = "thermal_derate_off"
        return ThermalDerateStep(self.active, pitch_hottest, event)
