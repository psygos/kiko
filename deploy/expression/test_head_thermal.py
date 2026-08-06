import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from head_thermal import (
    ThermalConfigError, ThermalDerateController, ThermalDeratePolicy,
    ThermalObservationError,
)


def policy_json():
    return {
        "derate_temp_raw": 48,
        "derate_clear_temp_raw": 45,
        "temp_abort_raw": 65,
        "derate_confirm_samples": 3,
        "derate_clear_samples": 10,
    }


class ThermalDeratePolicyTests(unittest.TestCase):
    def test_deployed_policy_parses_as_reviewed(self):
        with open(os.path.join(os.path.dirname(__file__), "config.json")) as source:
            raw = json.load(source)
        policy = ThermalDeratePolicy.parse(raw)
        # Derate band sits ABOVE idle-holding warmth (bow idles 54-58 in a
        # warm room): it must catch activity heat, not ambient. Abort 65
        # with the 3-consecutive streak rule remains the hard ceiling.
        self.assertEqual(policy, ThermalDeratePolicy(60, 56, 65, 3, 10))
        # Bow-forward era (operator, 2026-08-04): rest is free at the
        # balance natural, so bow carries a bigger share of every pitch.
        self.assertEqual(raw["bow_pitch_share"], 0.35)

    def test_invalid_hysteresis_and_boolean_samples_are_rejected(self):
        raw = policy_json()
        raw["derate_clear_temp_raw"] = 48
        with self.assertRaises(ThermalConfigError):
            ThermalDeratePolicy.parse(raw)
        raw = policy_json()
        raw["derate_confirm_samples"] = True
        with self.assertRaises(ThermalConfigError):
            ThermalDeratePolicy.parse(raw)
        raw = policy_json()
        raw["temp_abort_raw"] = 48
        with self.assertRaises(ThermalConfigError):
            ThermalDeratePolicy.parse(raw)


class ThermalDerateControllerTests(unittest.TestCase):
    def make_controller(self):
        return ThermalDerateController(ThermalDeratePolicy.parse(policy_json()))

    def test_isolated_pitch_spike_does_not_latch_derate(self):
        controller = self.make_controller()
        self.assertFalse(controller.update((62, 45, 39, 39)).active)
        self.assertFalse(controller.update((43, 45, 39, 39)).active)

    def test_three_consecutive_pitch_samples_engage(self):
        controller = self.make_controller()
        for _ in range(2):
            self.assertFalse(controller.update((48, 46, 39, 39)).active)
        step = controller.update((49, 46, 39, 39))
        self.assertTrue(step.active)
        self.assertEqual(step.event, "thermal_derate_on")

    def test_non_pitch_heat_does_not_suppress_pitch_motion(self):
        controller = self.make_controller()
        for _ in range(5):
            step = controller.update((43, 45, 80, 90))
        self.assertFalse(step.active)

    def test_clear_requires_ten_consecutive_cool_samples(self):
        controller = self.make_controller()
        for _ in range(3):
            controller.update((49, 48, 39, 39))
        for _ in range(9):
            self.assertTrue(controller.update((43, 45, 39, 39)).active)
        step = controller.update((43, 45, 39, 39))
        self.assertFalse(step.active)
        self.assertEqual(step.event, "thermal_derate_off")

    def test_invalid_temperature_sample_is_rejected(self):
        controller = self.make_controller()
        with self.assertRaises(ThermalObservationError):
            controller.update((43, True, 39, 39))

    def engage(self, controller):
        for _ in range(3):
            controller.update((49, 48, 39, 39))
        self.assertTrue(controller.active)

    def test_implausible_bytes_never_engage_derate(self):
        controller = self.make_controller()
        for _ in range(5):
            step = controller.update((150, 140, 39, 39))
        self.assertFalse(step.active)

    def test_implausible_byte_holds_clear_counter(self):
        controller = self.make_controller()
        self.engage(controller)
        for _ in range(9):
            controller.update((43, 44, 39, 39))
        # Corruption-band byte: no evidence either way, counter holds at 9.
        self.assertTrue(controller.update((150, 44, 39, 39)).active)
        step = controller.update((43, 44, 39, 39))
        self.assertFalse(step.active)
        self.assertEqual(step.event, "thermal_derate_off")

    def test_isolated_warm_sample_decrements_instead_of_resetting(self):
        controller = self.make_controller()
        self.engage(controller)
        for _ in range(9):
            controller.update((43, 44, 39, 39))
        # Plausible warm byte inside a cooling trend costs one count (9->8),
        # not the whole streak; two more cool samples clear the derate.
        self.assertTrue(controller.update((47, 44, 39, 39)).active)
        self.assertTrue(controller.update((43, 44, 39, 39)).active)
        step = controller.update((43, 44, 39, 39))
        self.assertFalse(step.active)
        self.assertEqual(step.event, "thermal_derate_off")


if __name__ == "__main__":
    unittest.main()
