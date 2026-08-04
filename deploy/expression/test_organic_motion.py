import math
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from organic_motion import (
    EyeIntent, MotionAxisPolicy, MotionConfigError, MotionInputError,
    OrganicEyeDynamics, OrganicMotionAxis, PetEyeChoreographer,
    parse_head_motion_policies,
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
        # Bounds are asserted at the ATTACK ceilings: a step this large
        # triggers the cat-toy envelope (accel x2, jerk x2.5). Velocity is
        # never raised by attack.
        for tick in range(1, 401):
            step = axis.step(80.0, tick * 0.05)
            self.assertGreaterEqual(step.position, -100.0)
            self.assertLessEqual(step.position, 100.0)
            self.assertLessEqual(abs(step.velocity), 30.0 + 1e-9)
            self.assertLessEqual(abs(step.acceleration), 120.0 + 1e-9)
            self.assertLessEqual(
                abs(step.acceleration - previous_acceleration), 30.0 + 1e-8)
            maximum_position = max(maximum_position, step.position)
            previous_acceleration = step.acceleration
        self.assertLessEqual(maximum_position, 84.0)
        self.assertAlmostEqual(axis.position, 80.0, places=3)

    def test_target_jump_snaps_while_steady_follow_stays_gentle(self):
        # Cat-toy reflex: one big jump arrives far sooner than the same
        # distance fed as a creeping target that never triggers attack.
        pounce = OrganicMotionAxis(axis_policy(), 0.0, 0.0)
        creep = OrganicMotionAxis(axis_policy(), 0.0, 0.0)
        now = 0.0
        for tick in range(1, 25):  # 1.2 s
            now = tick * 0.05
            pounce.step(80.0, now)
            creep.step(min(80.0, tick * 4.0), now)  # 4/tick: below trigger
            if tick == 12:
                # The pounce advantage lives in the launch, before both
                # axes saturate at the helper policy's tiny velocity cap.
                self.assertGreater(pounce.position, creep.position * 1.3)
        self.assertGreater(pounce.position, 20.0)

    def test_attack_with_underdamping_arrives_fast_and_rings(self):
        # A policy with real dynamic budget (roll-like damping): the pounce
        # must arrive quickly AND ring visibly past the target, then settle.
        springy = MotionAxisPolicy(-100.0, 100.0, 0.8, 200.0, 800.0, 8000.0,
                                   0.15, 0.85)
        axis = OrganicMotionAxis(springy, 0.0, 0.0)
        peak = 0.0
        arrived_at = None
        for tick in range(1, 201):
            step = axis.step(80.0, tick * 0.05)
            peak = max(peak, step.position)
            if arrived_at is None and step.position >= 72.0:
                arrived_at = tick * 0.05
        self.assertLess(arrived_at, 0.6)
        self.assertGreater(peak, 80.3)   # rings past the target
        self.assertLess(peak, 92.0)      # bounded, not a fling
        self.assertAlmostEqual(axis.position, 80.0, places=2)

    def test_long_gap_cannot_lunge_but_still_advances(self):
        axis = OrganicMotionAxis(axis_policy(), 0.0, 0.0)
        moving = axis.step(80.0, 0.05)
        gap = axis.step(80.0, 5.00)
        self.assertTrue(gap.gap_reset)
        # The interval is clamped to maximum_interval_s, so a 5 s stall can
        # never buy a catch-up jump larger than one bounded 0.15 s step —
        # and the axis keeps moving instead of freezing forever.
        self.assertGreater(gap.position, moving.position)
        self.assertLessEqual(gap.position - moving.position,
                             30.0 * 0.15 + 1e-9)
        self.assertLessEqual(abs(gap.velocity), 30.0 + 1e-9)

    def test_chronically_slow_loop_converges_at_bounded_rate(self):
        axis = OrganicMotionAxis(axis_policy(), 0.0, 0.0)
        now = 0.0
        previous = 0.0
        step = None
        for _ in range(60):
            now += 0.4  # every tick far beyond maximum_interval_s
            step = axis.step(80.0, now)
            # Per-step advance stays inside one clamped interval's travel.
            self.assertLessEqual(step.position - previous, 30.0 * 0.15 + 1e-9)
            previous = step.position
        # Effective tracking rate degrades proportionally (0.15/0.4), not to
        # a crawl: 60 late ticks are ample for an 80-tick move. The coarse
        # clamped steps cost a slightly larger (still small) overshoot.
        self.assertGreater(step.position, 70.0)
        self.assertLessEqual(step.position, 84.0)

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


class PetEyeChoreographerTests(unittest.TestCase):
    def test_touch_direction_selects_distinct_continuous_visual_fields(self):
        base = EyeIntent.bounded(gaze_x=10, gaze_y=20, color=(20, 30, 40))
        yaw = PetEyeChoreographer().apply(
            1.0, base, "YIELDING", (0, 0, 1, 0))
        roll = PetEyeChoreographer().apply(
            1.0, base, "YIELDING", (0, 0, 0, -1))
        multi = PetEyeChoreographer().apply(
            1.0, base, "YIELDING", (1, 1, 1, 0))
        self.assertGreater(yaw.gaze_x, 0)
        self.assertLess(roll.gaze_x, 0)
        self.assertNotEqual(yaw.color, roll.color)
        self.assertGreater(multi.lid, yaw.lid)
        self.assertLess(multi.gaze_y, yaw.gaze_y)

    def test_rest_looks_down_blinks_once_and_softens(self):
        choreography = PetEyeChoreographer()
        base = EyeIntent.bounded(gaze_y=100, lid=60, brightness=900)
        first = choreography.apply(
            1.0, base, "RESTING", (1, 0, 0, 0))
        second = choreography.apply(
            1.1, base, "RESTING", (1, 0, 0, 0))
        self.assertLess(first.gaze_y, -350)
        self.assertGreater(first.lid, 500)
        self.assertLess(first.brightness, base.brightness)
        self.assertEqual(first.expression, "sleepy")
        self.assertTrue(first.blink)
        self.assertFalse(second.blink)

    def test_recovery_arc_blends_exactly_back_to_living_intent(self):
        choreography = PetEyeChoreographer()
        base = EyeIntent.bounded(
            gaze_x=-120, gaze_y=80, lid=90, pupil=540,
            brightness=760, expression="curious", color=(30, 170, 210))
        start = choreography.apply(
            1.0, base, "RECOVERING", (0, 0, 1, 1), 0.0)
        before_mid = choreography.apply(
            1.3, base, "RECOVERING", (0, 0, 1, 1), 0.4)
        middle = choreography.apply(
            1.4, base, "RECOVERING", (0, 0, 1, 1), 0.55)
        after_mid = choreography.apply(
            1.5, base, "RECOVERING", (0, 0, 1, 1), 0.6)
        finish = choreography.apply(
            2.0, base, "RECOVERING", (0, 0, 1, 1), 1.0)
        self.assertTrue(start.blink)
        self.assertFalse(before_mid.blink)
        self.assertTrue(middle.blink)
        self.assertFalse(after_mid.blink)
        self.assertEqual(finish, base)

    def test_invalid_direction_is_rejected_once_at_boundary(self):
        with self.assertRaises(MotionInputError):
            PetEyeChoreographer().apply(
                0.0, EyeIntent.bounded(), "RESTING", (0, 0, 2, 0))
        with self.assertRaises(MotionInputError):
            PetEyeChoreographer().apply(
                0.0, EyeIntent.bounded(), "RECOVERING", (0, 0, 1, 0),
                math.nan)


if __name__ == "__main__":
    unittest.main()
