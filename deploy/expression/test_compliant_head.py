import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compliant_head import (
    CompliantConfigError, CompliantHeadController, CompliantHeadPolicy,
    CompliantObservationError, CONFIRMING, FAULT_HELD, FOLLOWING,
    RECOVERING, RELEASE_DWELL, YIELDING,
)


def policy_json():
    return {
        "minimum_ticks": [50, 50, 50, 50],
        "maximum_ticks": [150, 150, 150, 150],
        "maximum_baseline_error_ticks": [30, 30, 30, 30],
        "contact_entry_error_ticks": [18, 18, 18, 18],
        "contact_release_error_ticks": [8, 8, 8, 8],
        "maximum_yield_ticks": [30, 30, 30, 30],
        "maximum_command_step_ticks": [3, 3, 3, 3],
        "maximum_observed_step_ticks": [64, 64, 64, 64],
        "quiet_command_step_ticks": [1, 1, 1, 1],
        "holding_torque_limit_permille": [650, 550, 400, 400],
        "control_period_ms": 100,
        "maximum_observation_span_ms": 60,
        "contact_arm_dwell_ms": 1000,
        "contact_acquisition_samples": 3,
        "release_dwell_ms": 600,
        "recovery_duration_ms": 2400,
        "follow_permille": 350,
    }


def make_controller():
    policy = CompliantHeadPolicy.parse(
        policy_json(), [650, 550, 400, 400])
    return CompliantHeadController(policy, (100, 100, 100, 100), 0.0)


def service(controller, at, position=(100, 100, 100, 100),
            command=(100, 100, 100, 100), moving=False):
    return controller.service(
        at, command, position, (moving,) * 4, 0.020)


class CompliantHeadPolicyTests(unittest.TestCase):
    def test_deployed_pet_profile_is_parsed_as_reviewed(self):
        with open(os.path.join(os.path.dirname(__file__), "config.json")) as source:
            raw = json.load(source)
        policy = CompliantHeadPolicy.parse(
            raw["compliant_hold"], raw["torque_limit_permille"])
        self.assertEqual(
            policy.holding_torque_limit_permille, (600, 500, 250, 300))
        self.assertEqual(policy.contact_entry_error_ticks, (14, 18, 10, 10))
        self.assertEqual(policy.contact_release_error_ticks, (5, 7, 4, 4))
        self.assertEqual(policy.contact_acquisition_samples, 2)
        self.assertEqual(policy.follow_fraction, 0.5)

    def test_boundary_rejects_unknown_and_missing_fields(self):
        raw = policy_json()
        raw["mystery"] = 1
        with self.assertRaises(CompliantConfigError):
            CompliantHeadPolicy.parse(raw, [650, 550, 400, 400])

    def test_policy_is_bound_to_installed_torque(self):
        with self.assertRaises(CompliantConfigError):
            CompliantHeadPolicy.parse(policy_json(), [600, 400, 400, 400])

    def test_release_band_must_be_strictly_inside_entry_band(self):
        raw = policy_json()
        raw["contact_release_error_ticks"][2] = 18
        with self.assertRaises(CompliantConfigError):
            CompliantHeadPolicy.parse(raw, [650, 550, 400, 400])

    def test_follow_gain_must_preserve_release_hysteresis(self):
        raw = policy_json()
        raw["follow_permille"] = 600
        with self.assertRaises(CompliantConfigError):
            CompliantHeadPolicy.parse(raw, [650, 550, 400, 400])


class CompliantHeadControllerTests(unittest.TestCase):
    def arm(self, controller):
        result = None
        for step in range(11):
            result = service(controller, step / 10.0)
        self.assertEqual(result.event, "pet_ready")
        self.assertTrue(controller.contact_armed)

    def test_contact_requires_arm_dwell_and_three_consistent_samples(self):
        controller = make_controller()
        self.arm(controller)
        first = service(controller, 1.1, (120, 100, 100, 100), moving=True)
        self.assertEqual(first.state, CONFIRMING)
        self.assertEqual(first.event, "pet_candidate")
        second = service(controller, 1.2, (122, 100, 100, 100), moving=True)
        self.assertEqual(second.state, CONFIRMING)
        third = service(controller, 1.3, (124, 100, 100, 100), moving=True)
        self.assertEqual(third.state, YIELDING)
        self.assertEqual(third.event, "pet_contact")
        self.assertEqual(third.residual_error_ticks, (24, 0, 0, 0))
        # 35% of 24 ticks rounds to 8, but one physical command may move only 3.
        self.assertEqual(third.target_ticks, (103, 100, 100, 100))

    def test_settled_gravity_bias_is_not_misclassified_as_touch(self):
        controller = make_controller()
        biased = (117, 115, 100, 100)
        for step in range(11):
            result = service(controller, step / 10.0, biased)
        self.assertEqual(result.event, "pet_ready")
        self.assertEqual(controller.baseline_error, (17, 15, 0, 0))
        # The same equilibrium remains inside the release band after arming.
        result = service(controller, 1.1, biased)
        self.assertEqual(result.state, FOLLOWING)
        # Contact is displacement beyond the learned bias, not raw goal error.
        result = service(controller, 1.2, (136, 115, 100, 100), moving=True)
        self.assertEqual(result.state, CONFIRMING)

    def test_baseline_outside_its_own_envelope_never_arms(self):
        controller = make_controller()
        outside = (131, 100, 100, 100)
        for step in range(20):
            result = service(controller, step / 10.0, outside)
        self.assertEqual(result.state, FOLLOWING)
        self.assertFalse(controller.contact_armed)
        self.assertEqual(controller.baseline_error, (0, 0, 0, 0))

    def test_direction_reversal_rejects_false_contact(self):
        controller = make_controller()
        self.arm(controller)
        service(controller, 1.1, (120, 100, 100, 100), moving=True)
        result = service(controller, 1.2, (80, 100, 100, 100), moving=True)
        self.assertEqual(result.state, FOLLOWING)
        self.assertFalse(controller.contact_armed)

    def test_release_dwells_then_returns_with_bounded_steps(self):
        controller = make_controller()
        self.arm(controller)
        service(controller, 1.1, (120, 100, 100, 100), moving=True)
        service(controller, 1.2, (122, 100, 100, 100), moving=True)
        contact = service(controller, 1.3, (124, 100, 100, 100), moving=True)
        self.assertEqual(contact.state, YIELDING)
        previous = contact.target_ticks[0]
        for index in range(7):
            result = service(controller, 1.4 + index * 0.1,
                             (previous, 100, 100, 100), moving=False)
            self.assertLessEqual(abs(result.target_ticks[0] - previous), 3)
            previous = result.target_ticks[0]
        self.assertEqual(result.state, RECOVERING)
        self.assertEqual(result.event, "pet_recovering")
        at = 2.1
        while at <= 4.7 and controller.state != FOLLOWING:
            result = service(controller, at, result.target_ticks, moving=False)
            self.assertLessEqual(abs(result.target_ticks[0] - previous), 3)
            previous = result.target_ticks[0]
            at += 0.1
        self.assertEqual(controller.state, FOLLOWING)
        self.assertEqual(controller.target, (100, 100, 100, 100))

    def test_large_observation_jump_faults_closed(self):
        controller = make_controller()
        service(controller, 0.0, (50, 100, 100, 100))
        with self.assertRaises(CompliantObservationError):
            service(controller, 0.1, (150, 100, 100, 100))
        self.assertEqual(controller.state, FAULT_HELD)

    def test_expression_motion_prevents_contact_arming(self):
        controller = make_controller()
        for step in range(20):
            command = (100 + (step % 2) * 4, 100, 100, 100)
            result = service(controller, step / 10.0,
                             position=command, command=command)
        self.assertEqual(result.state, FOLLOWING)
        self.assertFalse(controller.contact_armed)


if __name__ == "__main__":
    unittest.main()
