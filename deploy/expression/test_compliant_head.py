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
        "yield_static_release_ms": 4000,
        "maximum_yield_dwell_ms": 30000,
        "comfort_roll_tilt_ticks": 6,
        "yield_torque_limit_permille": [400, 300, 200, 200],
        "tap_max_contact_ms": 1200,
        "tap_recovery_ms": 800,
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
            policy.holding_torque_limit_permille, (650, 550, 400, 400))
        self.assertEqual(policy.contact_entry_error_ticks, (18, 24, 32, 18))
        self.assertEqual(policy.contact_release_error_ticks, (5, 7, 4, 4))
        self.assertEqual(policy.contact_acquisition_samples, 2)
        # Exaggerated pet profile (operator-directed 2026-08-02): deeper
        # yield led by bow, theatrical rest lean, slower min-jerk rise.
        self.assertEqual(policy.follow_fraction, 0.65)
        self.assertEqual(policy.maximum_yield_ticks, (40, 48, 80, 36))
        self.assertEqual(policy.contact_rest_pose_offset_ticks, (-24, 30, 0, 0))
        self.assertEqual(
            policy.contact_directional_rest_offset_ticks, (0, 0, 20, 16))
        self.assertEqual(policy.rest_dwell_s, 1.2)
        self.assertEqual(policy.recovery_duration_s, 3.0)
        # Cat choreography: stillness settles the head in 1.8 s — at 0.65
        # follow the post-yield residual sits outside the release band by
        # construction, so static release is the PRIMARY pet exit.
        self.assertEqual(policy.yield_static_release_s, 1.8)
        self.assertEqual(policy.maximum_yield_dwell_s, 30.0)
        self.assertEqual(policy.comfort_roll_tilt_ticks, 14)
        # Backdrivable yield: soft but never near-zero (parser floors 150).
        self.assertEqual(policy.yield_torque_limit_permille,
                         (450, 350, 220, 250))
        # Tap fast path: boops skip the rest liturgy.
        self.assertEqual(policy.tap_max_contact_s, 1.2)
        self.assertEqual(policy.tap_recovery_s, 0.8)

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
        # The touch is followed while a small supervised tuck begins — bow,
        # curl, and (via the comfort tilt) an anticipatory roll lean.
        self.assertEqual(third.target_ticks, (103, 103, 100, 103))

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
        # Hold the touch past the tap window so this exercises the FULL
        # arc (a shorter contact now takes the tap fast path by design).
        for index in range(13):
            contact = service(controller, 1.4 + index * 0.1,
                              (124 + (index % 2), 100, 100, 100),
                              moving=True)
        previous = contact.target_ticks
        for index in range(7):
            result = service(controller, 2.7 + index * 0.1,
                             previous, moving=False)
            self.assertTrue(all(
                abs(actual - old) <= 3
                for actual, old in zip(result.target_ticks, previous)))
            previous = result.target_ticks
        self.assertEqual(result.state, RESTING)
        self.assertEqual(result.event, "pet_resting")
        at = 3.5
        saw_recovery = False
        while at <= 9.0 and controller.state != FOLLOWING:
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
        # Hold past the tap window so release takes the full rest path.
        for index in range(13):
            result = service(controller, 1.4 + index * 0.1,
                             (124 + (index % 2), 100, 100, 100), moving=True)
        for index in range(7):
            result = service(controller, 2.7 + index * 0.1,
                             result.target_ticks, moving=False)
        self.assertEqual(result.state, RESTING)
        # A returning hand APPROACHES — the residual sweeps as it presses.
        # (A perfectly constant offset is the sag signature and must not
        # recontact; see test_static_pin_during_rest_does_not_recontact.)
        for at, poke in ((3.5, 20), (3.6, 26), (3.7, 32), (3.8, 38)):
            position = list(result.target_ticks)
            position[2] += poke
            result = service(controller, at, tuple(position), moving=True)
        self.assertEqual(result.state, YIELDING)
        self.assertEqual(result.event, "pet_recontact")
        self.assertEqual(controller.contact_directions, (0, 0, 1, 0))

    def test_static_pin_during_rest_does_not_recontact(self):
        # A sag (or motionless hand) above entry used to knead-cycle forever:
        # RESTING -> recontact -> YIELDING -> static release -> RESTING ...
        # and face-follow never resumed. Statue-still residual must let the
        # rest complete and recovery re-learn the baseline instead.
        controller = make_controller()
        self.enter_yield(controller)
        at = 1.3
        events = []
        result = None
        for _ in range(200):
            at += 0.1
            sagged = tuple(target + (23 if joint == 0 else 0)
                           for joint, target in enumerate(controller.target))
            result = service(controller, at, sagged)
            if result.event:
                events.append(result.event)
            if result.state == FOLLOWING:
                break
        self.assertEqual(result.state, FOLLOWING)
        self.assertIn("pet_comfy", events)
        self.assertNotIn("pet_recontact", events)

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

    def enter_yield(self, controller):
        self.arm(controller)
        service(controller, 1.1, (120, 100, 100, 100), moving=True)
        service(controller, 1.2, (122, 100, 100, 100), moving=True)
        result = service(controller, 1.3, (124, 100, 100, 100), moving=True)
        self.assertEqual(result.state, YIELDING)
        return result

    def test_static_sag_residual_releases_and_self_heals(self):
        # The 2026-08-02 live trance: the hand leaves, but pose-dependent
        # gravity sag keeps bow a constant ~11 ticks off target — outside
        # the 8-tick release band — so the machine yielded for 20 minutes
        # (frozen head, white pet-eyes). From here the servo just tracks
        # whatever is commanded, always with that 11-tick sag; the machine
        # must classify the statue-still residual as static contact, rest,
        # recover, and re-learn the sag into a fresh baseline.
        controller = make_controller()
        self.enter_yield(controller)
        at = 1.3
        events = []
        result = None
        for _ in range(120):
            at += 0.1
            sagged = tuple(target + (11 if joint == 0 else 0)
                           for joint, target in enumerate(controller.target))
            result = service(controller, at, sagged)
            if result.event:
                events.append(result.event)
            if result.state == FOLLOWING:
                break
        self.assertIn("pet_release_static", events)
        self.assertIn("pet_recovering", events)
        self.assertIn("pet_returned", events)
        self.assertEqual(result.state, FOLLOWING)
        self.assertLess(at, 1.3 + 12.0)  # seconds of trance, never minutes

    def test_live_pet_with_varying_residual_keeps_yielding(self):
        controller = make_controller()
        self.enter_yield(controller)
        at = 1.3
        for step in range(70):  # 7 s of active stroking
            at += 0.1
            wobble = 124 + (10 if step % 2 == 0 else -10)
            result = service(controller, at, (wobble, 100, 100, 100),
                             moving=True)
            self.assertEqual(result.state, YIELDING)

    def test_relentless_contact_hits_hard_timeout_into_rest(self):
        controller = make_controller()
        self.enter_yield(controller)
        at = 1.3
        result = None
        for step in range(400):  # far past maximum_yield_dwell_ms
            at += 0.1
            wobble = 124 + (10 if step % 2 == 0 else -10)
            result = service(controller, at, (wobble, 100, 100, 100),
                             moving=True)
            if result.state != YIELDING:
                break
        self.assertEqual(result.event, "pet_yield_timeout")
        self.assertEqual(result.state, RESTING)
        self.assertLess(at, 1.3 + 31.0)

    def test_quick_tap_skips_rest_and_recovers_fast(self):
        # A boop is not a request to settle: single brief contact must go
        # straight to a fast recovery, never through RESTING, and hand the
        # character layer a tap-flagged episode to answer with play.
        controller = make_controller()
        self.enter_yield(controller)
        at = 1.3
        events = []
        result = None
        for _ in range(40):
            at += 0.1
            # Hand gone instantly; servo tracks the command exactly.
            result = service(controller, at, tuple(controller.target))
            if result.event:
                events.append(result.event)
            if result.state == FOLLOWING:
                break
        self.assertIn("pet_tap", events)
        self.assertNotIn("pet_resting", events)
        self.assertNotIn("pet_comfy", events)
        self.assertIn("pet_returned", events)
        self.assertLess(at, 1.3 + 2.5)  # fast bounce, not a 6 s liturgy
        episode = controller.last_episode
        self.assertIsNotNone(episode)
        self.assertTrue(episode["tap"])
        self.assertEqual(episode["yield_entries"], 1)

    def test_recontact_clears_the_tap_flag(self):
        # A boop followed by a real pet must classify as the pet, not the
        # boop: the tap flag dies on recontact.
        controller = make_controller()
        self.enter_yield(controller)
        at = 1.3
        # Instant release -> tap fast path into RECOVERING.
        result = None
        while controller.state != RECOVERING:
            at += 0.1
            result = service(controller, at, tuple(controller.target))
        self.assertTrue(controller.episode["tap"])
        # The hand comes back with approach dynamics during recovery.
        for poke in (20, 26, 32, 38):
            at += 0.1
            position = list(controller.target)
            position[0] += poke
            result = service(controller, at, tuple(position), moving=True)
        self.assertEqual(result.state, YIELDING)
        self.assertEqual(result.event, "pet_recontact")
        self.assertFalse(controller.episode["tap"])
        self.assertEqual(controller.episode["yield_entries"], 2)
        # Ride out to completion with a sagged-tracking servo.
        for _ in range(200):
            at += 0.1
            sagged = tuple(target + (11 if joint == 0 else 0)
                           for joint, target in enumerate(controller.target))
            result = service(controller, at, sagged)
            if result.state == FOLLOWING:
                break
        episode = controller.last_episode
        self.assertIsNotNone(episode)
        self.assertFalse(episode["tap"])

    def test_full_pet_episode_summary_is_recorded(self):
        controller = make_controller()
        self.enter_yield(controller)
        at = 1.3
        result = None
        for _ in range(200):
            at += 0.1
            sagged = tuple(target + (11 if joint == 0 else 0)
                           for joint, target in enumerate(controller.target))
            result = service(controller, at, sagged)
            if result.state == FOLLOWING:
                break
        episode = controller.last_episode
        self.assertIsNotNone(episode)
        self.assertFalse(episode["tap"])
        self.assertTrue(episode["reached_rest"])
        self.assertTrue(episode["reached_comfy"])
        self.assertGreater(episode["duration_s"], 3.0)
        self.assertGreaterEqual(episode["peak_residual"][0], 11)

    def test_yield_dwell_bounds_are_validated(self):
        raw = policy_json()
        raw["maximum_yield_dwell_ms"] = 4000  # not above static release
        with self.assertRaises(CompliantConfigError):
            CompliantHeadPolicy.parse(raw, [650, 550, 400, 400])
        raw = policy_json()
        raw["yield_static_release_ms"] = 200  # under three control periods
        with self.assertRaises(CompliantConfigError):
            CompliantHeadPolicy.parse(raw, [650, 550, 400, 400])
        raw = policy_json()
        raw["comfort_roll_tilt_ticks"] = 23  # 0 + 8 + 23 > roll yield 30
        with self.assertRaises(CompliantConfigError):
            CompliantHeadPolicy.parse(raw, [650, 550, 400, 400])
        raw = policy_json()
        raw["yield_torque_limit_permille"] = [700, 300, 200, 200]  # > holding
        with self.assertRaises(CompliantConfigError):
            CompliantHeadPolicy.parse(raw, [650, 550, 400, 400])
        raw = policy_json()
        raw["yield_torque_limit_permille"] = [100, 300, 200, 200]  # below floor
        with self.assertRaises(CompliantConfigError):
            CompliantHeadPolicy.parse(raw, [650, 550, 400, 400])
        raw = policy_json()
        # Below the measured bow static-hold floor (300, bench 2026-08-02).
        raw["yield_torque_limit_permille"] = [250, 300, 200, 200]
        with self.assertRaises(CompliantConfigError):
            CompliantHeadPolicy.parse(raw, [650, 550, 400, 400])

    def run_to_event(self, controller, at, wanted, position_of, limit=200):
        events = []
        for _ in range(limit):
            at += 0.1
            result = service(controller, at, position_of(controller))
            if result.event:
                events.append((round(at, 1), result.event))
                if result.event == wanted:
                    return at, events, result
        self.fail(f"{wanted} never fired; saw {events}")

    def test_cat_sequence_tilts_sinks_pauses_then_rises(self):
        # Operator-specified feel: press -> yield -> tilt -> get comfy ->
        # real pause -> slow rise. Bow-only contact must still tilt roll.
        controller = make_controller()
        self.enter_yield(controller)

        def sagged(c):
            return tuple(target + (11 if joint == 0 else 0)
                         for joint, target in enumerate(c.target))

        at, _, _ = self.run_to_event(
            controller, 1.3, "pet_release_static", sagged)
        comfy_at, _, _ = self.run_to_event(
            controller, at, "pet_comfy", sagged)
        # Tilt: roll rest target carries the comfort tilt even though the
        # contact was pure bow (helper tilt 6, favorite side +1).
        self.assertEqual(controller.target[3], 106)
        # Sink: bow rest target sits below the return pose (rest offset -6).
        self.assertEqual(controller.target[0], 94)
        rise_at, _, _ = self.run_to_event(
            controller, comfy_at, "pet_recovering", sagged)
        # Pause: dwell is measured from settling, not from rest entry.
        self.assertGreaterEqual(rise_at - comfy_at,
                                controller.active_rest_duration_s - 1e-6)
        self.run_to_event(controller, rise_at, "pet_returned", sagged)


if __name__ == "__main__":
    unittest.main()
