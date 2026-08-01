import json
import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compliant_head import (
    CompliantConfigError, CompliantHeadController, CompliantHeadPolicy,
    CompliantObservationError, CONFIRMING, FAULT_HELD, FOLLOWING,
    RECOVERING, RELEASE_DWELL, RESTING, YIELDING,
)


def policy_json():
    return {
        "minimum_ticks": [50, 50, 50, 50],
        "maximum_ticks": [150, 150, 150, 150],
        "maximum_baseline_error_ticks": [30, 30, 30, 30],
        "contact_entry_error_ticks": [18, 18, 18, 18],
        "contact_release_error_ticks": [8, 8, 8, 8],
        "maximum_yield_ticks": [30, 30, 30, 30],
        "contact_rest_pose_offset_ticks": [-6, 12, 0, 0],
        "contact_directional_rest_offset_ticks": [0, 0, 8, 8],
        "maximum_command_step_ticks": [3, 3, 3, 3],
        "maximum_observed_step_ticks": [64, 64, 64, 64],
        "quiet_command_step_ticks": [1, 1, 1, 1],
        "holding_torque_limit_permille": [650, 550, 400, 400],
        "control_period_ms": 100,
        "maximum_observation_span_ms": 60,
        "contact_arm_dwell_ms": 1000,
        "contact_acquisition_samples": 3,
        "release_dwell_ms": 600,
        "rest_dwell_ms": 400,
        "rest_per_additional_joint_ms": 200,
        "maximum_rest_dwell_ms": 1000,
        "recovery_duration_ms": 2400,
        "recovery_per_additional_joint_permille": 150,
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
        self.assertEqual(policy.contact_rest_pose_offset_ticks, (-8, 18, 0, 0))
        self.assertEqual(
            policy.contact_directional_rest_offset_ticks, (0, 0, 12, 10))

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

    def test_contextual_rest_pose_must_fit_yield_envelope(self):
        raw = policy_json()
        raw["contact_directional_rest_offset_ticks"][0] = 25
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
        # The touch is followed while a small supervised bow/curl tuck begins.
        self.assertEqual(third.target_ticks, (103, 103, 100, 100))

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
        previous = contact.target_ticks
        for index in range(7):
            result = service(controller, 1.4 + index * 0.1,
                             previous, moving=False)
            self.assertTrue(all(
                abs(actual - old) <= 3
                for actual, old in zip(result.target_ticks, previous)))
            previous = result.target_ticks
        self.assertEqual(result.state, RESTING)
        self.assertEqual(result.event, "pet_resting")
        at = 2.1
        saw_recovery = False
        while at <= 5.5 and controller.state != FOLLOWING:
            result = service(controller, at, result.target_ticks, moving=False)
            self.assertTrue(all(
                abs(actual - old) <= 3
                for actual, old in zip(result.target_ticks, previous)))
            previous = result.target_ticks
            saw_recovery = saw_recovery or result.event == "pet_recovering"
            at += 0.1
        self.assertTrue(saw_recovery)
        self.assertEqual(controller.state, FOLLOWING)
        self.assertEqual(controller.target, (100, 100, 100, 100))

    def test_multi_axis_pet_selects_contextual_pose_and_longer_grace(self):
        controller = make_controller()
        self.arm(controller)
        touched = (124, 100, 124, 76)
        service(controller, 1.1, touched, moving=True)
        service(controller, 1.2, touched, moving=True)
        result = service(controller, 1.3, touched, moving=True)
        self.assertEqual(result.state, YIELDING)
        self.assertEqual(controller.contact_directions, (1, 0, 1, -1))
        self.assertEqual(controller.contact_rest_pose, (-6, 12, 8, -8))
        self.assertAlmostEqual(controller.active_rest_duration_s, 0.8)
        self.assertAlmostEqual(controller.active_recovery_duration_s, 3.12)

    def test_every_nonempty_four_axis_touch_pattern_stays_bounded(self):
        for directions in itertools.product((-1, 0, 1), repeat=4):
            if directions == (0, 0, 0, 0):
                continue
            controller = make_controller()
            self.arm(controller)
            touched = tuple(100 + 24 * direction
                            for direction in directions)
            service(controller, 1.1, touched, moving=True)
            service(controller, 1.2, touched, moving=True)
            result = service(controller, 1.3, touched, moving=True)
            self.assertEqual(result.state, YIELDING)
            self.assertEqual(controller.contact_directions, directions)
            active = sum(direction != 0 for direction in directions)
            self.assertAlmostEqual(
                controller.active_rest_duration_s,
                min(1.0, 0.4 + 0.2 * (active - 1)))
            self.assertAlmostEqual(
                controller.active_recovery_duration_s,
                2.4 * (1.0 + 0.15 * (active - 1)))
            for joint, (target, rest) in enumerate(zip(
                    result.target_ticks, controller.contact_rest_pose)):
                self.assertLessEqual(abs(target - 100), 3)
                self.assertLessEqual(
                    abs(rest), controller.policy.maximum_yield_ticks[joint])

    def test_touch_during_rest_cancels_return_and_yields_again(self):
        controller = make_controller()
        self.arm(controller)
        for at, position in ((1.1, (124, 100, 100, 100)),
                             (1.2, (124, 100, 100, 100)),
                             (1.3, (124, 100, 100, 100))):
            result = service(controller, at, position, moving=True)
        for index in range(7):
            result = service(controller, 1.4 + index * 0.1,
                             result.target_ticks, moving=False)
        self.assertEqual(result.state, RESTING)
        for at in (2.1, 2.2, 2.3):
            position = list(result.target_ticks)
            position[2] += 20
            result = service(controller, at, tuple(position), moving=True)
        self.assertEqual(result.state, YIELDING)
        self.assertEqual(result.event, "pet_recontact")
        self.assertEqual(controller.contact_directions, (0, 0, 1, 0))

    def test_large_observation_jump_faults_closed(self):
        controller = make_controller()
        service(controller, 0.0, (50, 100, 100, 100))
        with self.assertRaises(CompliantObservationError):
            service(controller, 0.1, (150, 100, 100, 100))
        self.assertEqual(controller.state, FAULT_HELD)

    def test_touch_observation_may_cross_command_edge_only_by_yield(self):
        controller = make_controller()
        # Commands stop at 150, but the reviewed 30-tick touch yield makes an
        # actual encoder observation through 180 representable.
        result = service(controller, 0.0, (180, 100, 100, 100))
        self.assertEqual(result.state, FOLLOWING)
        outside = make_controller()
        with self.assertRaises(CompliantObservationError):
            service(outside, 0.0, (181, 100, 100, 100))
        self.assertEqual(outside.state, FAULT_HELD)

    def test_expression_command_cannot_use_observation_excursion(self):
        controller = make_controller()
        with self.assertRaises(CompliantObservationError):
            service(
                controller, 0.0, (151, 100, 100, 100),
                command=(151, 100, 100, 100))
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
