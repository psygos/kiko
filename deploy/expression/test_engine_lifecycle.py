"""Reality-modeling lifecycle tests for kiko_face_follow.

These pin the failure modes observed live on 2026-08-02:
- a park that cannot physically complete must say so (no false park_complete);
- checksum-valid garbage temperature bytes must not abort the run;
- a thread that dies holding the serial bus lock must surface as a typed
  fault within one second, never as an unbounded futex wait.

The engine module imports cv2/numpy/serial at module scope; none are needed
by the units under test, so absent ones are stubbed before import.
"""

import os
import sys
import threading
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for _name in ("cv2", "numpy", "serial"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except ImportError:
            sys.modules[_name] = types.ModuleType(_name)

import kiko_face_follow as kff


class FakeClock:
    """Deterministic stand-in for the time module inside kiko_face_follow."""

    def __init__(self, start=1000.0):
        self.now = start

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeBus:
    """Ideal servos follow the last written goal; stalled ones never move."""

    def __init__(self, positions, stalled=()):
        self.positions = list(positions)
        self.stalled = set(stalled)
        self.goal_writes = []
        self.torque_writes = []

    def read_position_redundant(self, servo_id):
        joint = kff.SERVO_IDS.index(servo_id)
        return {"position": self.positions[joint]}

    def write_goal(self, servo_id, position, speed):
        joint = kff.SERVO_IDS.index(servo_id)
        self.goal_writes.append((joint, position, speed))
        if joint not in self.stalled:
            self.positions[joint] = position

    def write_torque_limit(self, servo_id, permille):
        self.torque_writes.append((kff.SERVO_IDS.index(servo_id), permille))

    def write_torque_switch(self, servo_id, enabled):
        pass


class TelemetryBus(FakeBus):
    def __init__(self, temperature_sequence):
        super().__init__([0, 0, 0, 0])
        self.temperature_sequence = list(temperature_sequence)

    def read_telemetry(self, servo_id):
        return {
            "position": 0,
            "temperature_raw": self.temperature_sequence.pop(0),
            "voltage_raw": 120,
            "moving": False,
        }


class StreakBus(FakeBus):
    """Per-joint temperatures: an int serves forever, a list serves in order
    (last value repeats)."""

    def __init__(self, positions, temps_by_joint):
        super().__init__(positions)
        self.temps_by_joint = [list(t) if isinstance(t, list) else [t]
                               for t in temps_by_joint]

    def read_telemetry(self, servo_id):
        joint = kff.SERVO_IDS.index(servo_id)
        temps = self.temps_by_joint[joint]
        temp = temps.pop(0) if len(temps) > 1 else temps[0]
        return {
            "position": self.positions[joint],
            "temperature_raw": temp,
            "voltage_raw": 120,
            "moving": False,
        }


def make_head(bus):
    cfg = kff.load_config(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "config.json"))
    return kff.HeadController(bus, cfg)


class ParkHonestyTests(unittest.TestCase):
    def setUp(self):
        self.real_time = kff.time
        kff.time = FakeClock()

    def tearDown(self):
        kff.time = self.real_time

    def test_full_yaw_park_completes_and_reports_true(self):
        head = make_head(FakeBus([0, 0, 0, 0]))
        head.bus.positions = list(head.natural)
        head.goal = list(head.natural)
        head.goal[kff.YAW] += 480  # worst admissible distance
        # Physical pose matches the far goal before parking begins.
        head.bus.positions[kff.YAW] = head.goal[kff.YAW]
        self.assertTrue(head.park_and_release())
        self.assertEqual(head.bus.positions, head.natural)

    def test_stalled_yaw_park_reports_incomplete_not_complete(self):
        head = make_head(FakeBus([0, 0, 0, 0], stalled={kff.YAW}))
        head.bus.positions = list(head.natural)
        head.bus.positions[kff.YAW] += 400
        head.goal = list(head.bus.positions)
        # The old park printed park_complete unconditionally; the honest one
        # must read the mechanism back and admit the residual.
        self.assertFalse(head.park_and_release())

    def test_unengaged_head_park_does_not_command_the_bus(self):
        head = make_head(FakeBus([0, 0, 0, 0]))
        head.bus.positions = list(head.natural)
        self.assertIsNone(head.goal)  # admission never engaged
        head.park_and_release()
        self.assertEqual(head.bus.goal_writes, [])

    def test_park_always_restores_full_torque_first(self):
        # A run that dies mid-yield leaves the soft profile installed; park
        # must put the holding profile back before anything else.
        head = make_head(FakeBus([0, 0, 0, 0]))
        head.bus.positions = list(head.natural)
        head.goal = list(head.natural)
        head.torque_softened = True
        head.park_and_release()
        full = head.cfg["torque_limit_permille"]
        self.assertEqual(head.bus.torque_writes[:4],
                         [(joint, full[joint]) for joint in range(4)])
        self.assertFalse(head.torque_softened)


class TemperaturePlausibilityTests(unittest.TestCase):
    def setUp(self):
        self.real_time = kff.time
        kff.time = FakeClock()

    def tearDown(self):
        kff.time = self.real_time

    def confirm(self, head, first_raw, limit):
        return head._confirm_temperature(
            kff.BOW, kff.SERVO_IDS[kff.BOW],
            {"position": 0, "temperature_raw": first_raw,
             "voltage_raw": 120, "moving": False},
            limit, "energized")

    def test_sustained_garbage_bytes_do_not_abort(self):
        # The 2026-08-02 incident: 78 -> 150 -> 140, checksum-valid, fatal.
        head = make_head(TelemetryBus([150, 140]))
        result = self.confirm(head, 178, limit=64)
        self.assertEqual(result["temperature_raw"], 140)

    def test_sustained_plausible_heat_still_aborts(self):
        head = make_head(TelemetryBus([66, 67]))
        with self.assertRaises(kff.StsError):
            self.confirm(head, 66, limit=64)

    def test_transient_clears_on_recovery(self):
        head = make_head(TelemetryBus([150, 40]))
        result = self.confirm(head, 150, limit=64)
        self.assertEqual(result["temperature_raw"], 40)


class RuntimeTemperatureStreakTests(unittest.TestCase):
    """The 2026-08-02 incident class: what the 10 Hz observation path does
    with abort-band and corruption-band temperature bytes."""

    def observe(self, head, times):
        result = None
        for _ in range(times):
            result = head._read_safe_observation()
        return result

    def make_streak_head(self, temps_by_joint):
        head = make_head(StreakBus([0, 0, 0, 0], temps_by_joint))
        head.bus.positions = list(head.natural)
        head.goal = list(head.natural)
        return head

    def test_garbage_bytes_never_starve_the_other_joints(self):
        head = self.make_streak_head([150, 40, 40, 40])
        observed = head._read_safe_observation()
        self.assertIsNotNone(observed)
        telemetry, _span = observed
        # All four joints observed despite bow's corruption-band byte: the
        # old inline confirmation returned early and left curl/yaw/roll
        # unsupervised forever.
        self.assertEqual(len(telemetry), 4)

    def test_chronic_garbage_eventually_parks_the_run(self):
        head = self.make_streak_head([150, 40, 40, 40])
        limit = head.cfg["temp_unreadable_abort_samples"]
        self.observe(head, limit - 1)
        with self.assertRaises(kff.StsError):
            head._read_safe_observation()

    def test_three_consecutive_plausible_hot_samples_abort(self):
        head = self.make_streak_head([[66, 66, 66], 40, 40, 40])
        self.observe(head, 2)
        with self.assertRaises(kff.StsError):
            head._read_safe_observation()

    def test_hot_streak_broken_by_cool_sample_does_not_abort(self):
        head = self.make_streak_head([[66, 66, 40, 66, 66, 40], 40, 40, 40])
        self.assertIsNotNone(self.observe(head, 6))

    def test_garbage_byte_does_not_break_a_real_hot_streak(self):
        # Corruption is no evidence of cooling either: 66, 150, 66, 66 must
        # still count two-then-three plausible-hot samples.
        head = self.make_streak_head([[66, 150, 66, 66], 40, 40, 40])
        self.observe(head, 3)
        with self.assertRaises(kff.StsError):
            head._read_safe_observation()


class LagBus(FakeBus):
    """Servo model that follows the last written goal at a fixed rate."""

    def __init__(self, positions, follow_rate_ticks_s):
        super().__init__(positions)
        self.rate = follow_rate_ticks_s
        self.last_goal = list(positions)
        self.last_t = kff.time.monotonic()

    def _advance(self):
        dt = kff.time.monotonic() - self.last_t
        self.last_t = kff.time.monotonic()
        for joint in range(4):
            delta = self.last_goal[joint] - self.positions[joint]
            step = min(abs(delta), self.rate * dt)
            self.positions[joint] += step if delta > 0 else -step

    def write_goal(self, servo_id, position, speed):
        self._advance()
        joint = kff.SERVO_IDS.index(servo_id)
        self.goal_writes.append((joint, position, speed))
        self.last_goal[joint] = position

    def read_telemetry(self, servo_id):
        self._advance()
        joint = kff.SERVO_IDS.index(servo_id)
        return {
            "position": int(round(self.positions[joint])),
            "temperature_raw": 40,
            "voltage_raw": 120,
            "moving": False,
        }

    def read_position_redundant(self, servo_id):
        self._advance()
        joint = kff.SERVO_IDS.index(servo_id)
        return {"position": int(round(self.positions[joint]))}


class ReturningRecoveryTests(unittest.TestCase):
    """Admission recovery slews at 40 t/s under 10 Hz jam supervision: a
    legitimate full-envelope slew must complete, a hard jam must abort."""

    def setUp(self):
        self.real_time = kff.time
        kff.time = FakeClock()

    def tearDown(self):
        kff.time = self.real_time

    def run_returning(self, follow_rate, start_offset_yaw):
        bus = LagBus([0, 0, 0, 0], follow_rate)
        head = make_head(bus)
        start = list(head.natural)
        start[kff.YAW] += start_offset_yaw
        bus.positions = [float(v) for v in start]
        bus.last_goal = list(start)
        head.goal = list(start)
        head.state = "RETURNING"
        for tick in range(4000):
            head.step([0, 0, 0, 0], kff.time.monotonic())
            if tick % 2 == 0 and head.compliance is None:
                head.telemetry_check()
            if head.state == "TRACKING":
                return head
            kff.time.sleep(0.05)
        self.fail("RETURNING never completed")

    def test_full_envelope_recovery_slew_completes(self):
        head = self.run_returning(follow_rate=40.0, start_offset_yaw=-480)
        self.assertEqual(head.state, "TRACKING")

    def test_marginally_slow_joint_still_completes(self):
        # The reviewer's worst plausible healthy case: following well below
        # the ramp rate accumulates lag but never reaches the abort bound.
        head = self.run_returning(follow_rate=30.0, start_offset_yaw=-480)
        self.assertEqual(head.state, "TRACKING")

    def test_hard_jam_aborts_within_bound(self):
        with self.assertRaises(kff.StsError):
            self.run_returning(follow_rate=0.0, start_offset_yaw=-300)


class SpanStarvationTests(unittest.TestCase):
    def setUp(self):
        self.real_time = kff.time
        kff.time = FakeClock()

    def tearDown(self):
        kff.time = self.real_time

    def test_chronic_over_span_escalates_after_time_bound(self):
        clock = kff.time

        class SlowSpanBus(LagBus):
            def read_telemetry(self, servo_id):
                clock.sleep(0.03)  # 4 joints => ~120 ms span, budget is 60
                return super().read_telemetry(servo_id)

        bus = SlowSpanBus([0, 0, 0, 0], 1000.0)
        head = make_head(bus)
        bus.positions = [float(v) for v in head.natural]
        bus.last_goal = list(head.natural)
        head.goal = list(head.natural)
        head.state = "TRACKING"
        head.compliance = kff.CompliantHeadController(
            head.compliant_policy, tuple(head.natural),
            kff.time.monotonic())
        started = kff.time.monotonic()
        with self.assertRaises(kff.StsError):
            for _ in range(2000):
                head._service_compliance(kff.time.monotonic())
                kff.time.sleep(0.1)
        starved = kff.time.monotonic() - started
        self.assertGreaterEqual(starved, head.cfg["span_skip_abort_s"])
        self.assertLess(starved, head.cfg["span_skip_abort_s"] + 5.0)


class PetTorqueTests(unittest.TestCase):
    """Backdrivability: yielding softens the servos, rest restores them."""

    def setUp(self):
        self.real_time = kff.time
        kff.time = FakeClock()

    def tearDown(self):
        kff.time = self.real_time

    def test_yield_softens_servos_and_rest_restores(self):
        clock = kff.time

        class HandBus(LagBus):
            push = 0.0  # a hand displacing bow, on top of servo tracking

            def read_telemetry(self, servo_id):
                t = super().read_telemetry(servo_id)
                if kff.SERVO_IDS.index(servo_id) == kff.BOW:
                    t["position"] = int(round(t["position"] + self.push))
                return t

        bus = HandBus([0, 0, 0, 0], 1000.0)
        head = make_head(bus)
        bus.positions = [float(v) for v in head.natural]
        bus.last_goal = list(head.natural)
        head.goal = list(head.natural)
        head.state = "TRACKING"
        head.compliance = kff.CompliantHeadController(
            head.compliant_policy, tuple(head.natural), clock.monotonic())
        soft = head.compliant_policy.yield_torque_limit_permille
        full = head.compliant_policy.holding_torque_limit_permille

        def slots(count):
            for _ in range(count):
                head._service_compliance(clock.monotonic())
                clock.sleep(0.1)

        slots(12)  # quiet arm dwell
        self.assertTrue(head.compliance.contact_armed)
        bus.push = 30.0  # firm press on bow
        slots(3)
        self.assertEqual(head.compliance.state, kff.YIELDING)
        # Gravity axes soften only where touched: bow (the contact) goes
        # soft, curl holds, and the gravity-neutral yaw/roll always soften.
        self.assertEqual(bus.torque_writes[-4:],
                         [(0, soft[0]), (1, full[1]),
                          (2, soft[2]), (3, soft[3])])
        # The press goes statue-still: static release into RESTING must
        # hand full authority back for the nuzzle and the rise.
        slots(25)
        self.assertNotEqual(head.compliance.state, kff.YIELDING)
        self.assertEqual(bus.torque_writes[-4:],
                         [(joint, full[joint]) for joint in range(4)])


class CharacterEngineRestTests(unittest.TestCase):
    def make_engine(self):
        cfg = kff.load_config(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "config.json"))
        return kff.CharacterEngine(cfg), cfg

    def test_head_truly_rests_at_natural_when_alone(self):
        engine, _cfg = self.make_engine()
        now = engine.created_at + 100.0  # far past rest_after + ease
        _intent, desired4 = engine.compute(
            now, False, (0.0, 0.0), (0.0, 0.0), 0.0, False)
        self.assertTrue(engine.resting)
        self.assertEqual([round(abs(v), 9) for v in desired4], [0.0] * 4)

    def test_fast_turn_dips_the_neck(self):
        engine, _cfg = self.make_engine()
        t = engine.created_at + 1.0
        engine.compute(t, True, (0.0, 0.0), (0.0, 0.0), 0.3, False)
        engine.compute(t + 0.05, True, (0.5, 0.0), (0.0, 0.0), 0.3, False)
        # A 0.5 rad snap turn must produce a visible weight-shift dip.
        self.assertLess(engine.turn_dip, -10.0)

    def test_large_pitch_recruits_bow(self):
        engine, _cfg = self.make_engine()
        t = engine.created_at + 1.0
        _intent, desired4 = engine.compute(
            t, True, (0.0, 0.45), (0.0, 0.0), 0.0, False)
        # At ~280 ticks of pitch demand the bow share approaches one half:
        # bow rivals curl instead of trailing at a third.
        self.assertGreater(abs(desired4[kff.BOW]),
                           0.6 * abs(desired4[kff.CURL]))

    def test_person_arrival_wakes_the_head_immediately(self):
        engine, _cfg = self.make_engine()
        now = engine.created_at + 100.0
        engine.compute(now, False, (0.0, 0.0), (0.0, 0.0), 0.0, False)
        self.assertTrue(engine.resting)
        _intent, desired4 = engine.compute(
            now + 0.05, True, (0.3, 0.1), (0.0, 0.0), 0.5, False)
        # The rest state itself must clear (yaw_aim alone would move the
        # head even with a stuck envelope, so assert the transition too).
        self.assertFalse(engine.resting)
        self.assertGreater(abs(desired4[kff.YAW]), 100.0)


class BusLockTimeoutTests(unittest.TestCase):
    def test_orphaned_lock_surfaces_as_typed_fault_within_bound(self):
        bus = object.__new__(kff.StsBus)
        bus.lock = threading.Lock()
        holder = threading.Thread(target=bus.lock.acquire)
        holder.start()
        holder.join()
        # The lock's owner is dead and will never release — exactly the
        # 21-hour futex hang. The transact path must raise within ~1 s.
        import time as real_time
        started = real_time.monotonic()
        with self.assertRaises(kff.StsError):
            bus._transact(1, 0x02, [56, 15], 15)
        self.assertLess(real_time.monotonic() - started, 3.0)


if __name__ == "__main__":
    unittest.main()
