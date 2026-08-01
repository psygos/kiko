import math
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from organic_motion import (
    EyeIntent, MotionAxisPolicy, MotionConfigError, MotionInputError,
    OrganicEyeDynamics, OrganicMotionAxis, parse_head_motion_policies,
)


def axis_policy():
    return MotionAxisPolicy(-100.0, 100.0, 0.8, 30.0, 60.0, 240.0, 0.15)


class OrganicMotionAxisTests(unittest.TestCase):
    def test_deployed_config_has_no_duplicate_json_keys(self):
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate key {key}")
                result[key] = value
            return result

        with open(os.path.join(os.path.dirname(__file__), "config.json")) as source:
            json.load(source, object_pairs_hook=unique_object)

    def test_step_response_is_bounded_and_converges_with_small_overshoot(self):
        axis = OrganicMotionAxis(axis_policy(), 0.0, 0.0)
        previous_acceleration = 0.0
        maximum_position = 0.0
        for tick in range(1, 401):
            step = axis.step(80.0, tick * 0.05)
            self.assertGreaterEqual(step.position, -100.0)
            self.assertLessEqual(step.position, 100.0)
            self.assertLessEqual(abs(step.velocity), 30.0 + 1e-9)
            self.assertLessEqual(abs(step.acceleration), 60.0 + 1e-9)
            self.assertLessEqual(
                abs(step.acceleration - previous_acceleration), 12.0 + 1e-8)
            maximum_position = max(maximum_position, step.position)
            previous_acceleration = step.acceleration
        self.assertLessEqual(maximum_position, 81.0)
        self.assertAlmostEqual(axis.position, 80.0, places=3)

    def test_long_gap_freezes_instead_of_catching_up(self):
        axis = OrganicMotionAxis(axis_policy(), 0.0, 0.0)
        moving = axis.step(80.0, 0.05)
        gap = axis.step(80.0, 0.50)
        self.assertTrue(gap.gap_reset)
        self.assertEqual(gap.position, moving.position)
        self.assertEqual(gap.velocity, 0.0)
        self.assertEqual(gap.acceleration, 0.0)

    def test_invalid_target_and_regressed_clock_are_rejected(self):
        axis = OrganicMotionAxis(axis_policy(), 0.0, 1.0)
        with self.assertRaises(MotionInputError):
            axis.step(math.nan, 1.05)
        with self.assertRaises(MotionInputError):
            axis.step(10.0, 0.9)

    def test_head_policy_parser_rejects_wrong_shapes(self):
        raw = {
            "head_motion_response_hz": [0.6, 0.7, 0.8],
            "head_motion_max_velocity_ticks_s": [1, 1, 1, 1],
            "head_motion_max_acceleration_ticks_s2": [1, 1, 1, 1],
            "head_motion_max_jerk_ticks_s3": [1, 1, 1, 1],
            "head_motion_max_interval_ms": 150,
        }
        with self.assertRaises(MotionConfigError):
            parse_head_motion_policies(raw, [10, 10, 10, 10])

    def test_deployed_head_motion_is_bounded_by_servo_speed(self):
        with open(os.path.join(os.path.dirname(__file__), "config.json")) as source:
            raw = json.load(source)
        limits = [raw["bow_limit_ticks"], raw["curl_limit_ticks"],
                  raw["yaw_limit_ticks"], raw["roll_limit_ticks"]]
        policies = parse_head_motion_policies(raw, limits)
        self.assertEqual(len(policies), 4)
        self.assertTrue(all(
            policy.maximum_velocity <= raw["track_speed_max"]
            for policy in policies))


class OrganicEyeDynamicsTests(unittest.TestCase):
    def test_color_and_gaze_transition_over_multiple_frames(self):
        dynamics = OrganicEyeDynamics()
        start = EyeIntent.bounded(gaze_x=-800, color=(0, 0, 255))
        finish = EyeIntent.bounded(gaze_x=800, color=(255, 0, 0))
        self.assertEqual(dynamics.step(0.0, start), start)
        first = dynamics.step(0.05, finish)
        self.assertGreater(first.gaze_x, -800)
        self.assertLess(first.gaze_x, 800)
        third = dynamics.step(0.15, finish)
        self.assertGreater(third.color[0], 0)
        self.assertLess(third.color[0], 255)
        self.assertGreater(third.color[2], 0)

    def test_expression_requires_stable_request_and_blink_is_edge_triggered(self):
        dynamics = OrganicEyeDynamics()
        neutral = EyeIntent.bounded(expression="neutral", blink=False)
        greet = EyeIntent.bounded(expression="greet", blink=True)
        dynamics.step(0.0, neutral)
        first = dynamics.step(0.05, greet)
        self.assertEqual(first.expression, "neutral")
        self.assertTrue(first.blink)
        second = dynamics.step(0.10, greet)
        self.assertFalse(second.blink)
        switched = dynamics.step(0.31, greet)
        self.assertEqual(switched.expression, "greet")

    def test_eye_intent_rejects_nonfinite_and_unknown_expression(self):
        with self.assertRaises(MotionInputError):
            EyeIntent.bounded(gaze_x=math.inf)
        with self.assertRaises(MotionInputError):
            EyeIntent.bounded(expression="surprised")

    def test_alternating_targets_cannot_create_two_position_jitter(self):
        dynamics = OrganicEyeDynamics()
        first = EyeIntent.bounded(gaze_x=-900)
        previous = dynamics.step(0.0, first)
        maximum_output_step = 0
        outputs = []
        for tick in range(1, 101):
            target = EyeIntent.bounded(gaze_x=900 if tick % 2 else -900)
            actual = dynamics.step(tick * 0.05, target)
            maximum_output_step = max(
                maximum_output_step, abs(actual.gaze_x - previous.gaze_x))
            outputs.append(actual.gaze_x)
            previous = actual
        self.assertLessEqual(maximum_output_step, 75)
        self.assertGreater(len(set(outputs)), 2)


if __name__ == "__main__":
    unittest.main()
