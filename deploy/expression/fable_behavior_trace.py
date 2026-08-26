#!/usr/bin/env python3
"""Generate the strict Fable-to-Rust semantic behavior trace.

This is a qualification artifact, never a hardware owner.  It records facts
that both implementations are expected to share while deliberately excluding
float trajectory equality, RNG sample equality, encoder calibration, and
physical-motion claims.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import random
import sys
import types
from pathlib import Path

# Trace generation touches no camera, array, or serial path. Keep the same
# dependency-free import boundary as the retained Fable unit suite.
for _optional_module in ("cv2", "numpy", "serial"):
    if _optional_module not in sys.modules:
        try:
            __import__(_optional_module)
        except ImportError:
            sys.modules[_optional_module] = types.ModuleType(_optional_module)

import kiko_face_follow as kff


SCHEMA_VERSION = 1
COMPARISON_CONTRACT = "semantic_contract_v1_not_numeric_or_physical_parity"
HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURE = HERE / "fixtures" / "fable-behavior-trace-v1.json"


class BoundaryRng:
    """Deterministically select every lower or upper random boundary."""

    def __init__(self, upper: bool):
        self.upper = upper

    def uniform(self, minimum, maximum):
        return maximum if self.upper else minimum

    def choice(self, values):
        return values[-1] if self.upper else values[0]

    def random(self):
        return 1.0 if self.upper else 0.0

    def randint(self, minimum, maximum):
        return maximum if self.upper else minimum

    def randrange(self, start, stop=None, step=1):
        if stop is None:
            start, stop = 0, start
        values = range(start, stop, step)
        return values[-1] if self.upper else values[0]


def _milliseconds(seconds):
    value = float(seconds) * 1000.0
    rounded = int(round(value))
    if not math.isclose(value, rounded, abs_tol=1e-9):
        raise ValueError(f"timing {seconds!r} is not an exact millisecond")
    return rounded


def _config():
    cfg = dict(kff.DEFAULT_CONFIG)
    with (HERE / "config.json").open(encoding="utf-8") as source:
        cfg.update(json.load(source, object_pairs_hook=_unique_object))
    return cfg


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _source_digest(name):
    return hashlib.sha256((HERE / name).read_bytes()).hexdigest()


def _act_records():
    scheduled = {
        name: (builder, tuple(mode.lower() for mode in modes), cooldown)
        for name, builder, modes, _weight, cooldown in kff.ACT_LIBRARY
    }
    reactions_by_name = {}
    builders = {name: item[0] for name, item in scheduled.items()}
    for reaction, builder in kff.PET_REACTIONS.items():
        low = builder(BoundaryRng(False), {"energy": 0.0})
        high = builder(BoundaryRng(True), {"energy": 0.0})
        if low.name != high.name:
            raise ValueError("pet builder changes act identity across RNG boundaries")
        existing = builders.get(low.name)
        if existing is not None and existing is not builder:
            raise ValueError(f"multiple builders claim act {low.name!r}")
        builders[low.name] = builder
        reactions_by_name[low.name] = reaction

    records = []
    for name in sorted(builders):
        builder = builders[name]
        low = builder(BoundaryRng(False), {"energy": 0.0})
        high = builder(BoundaryRng(True), {"energy": 0.0})
        if low.name != name or high.name != name:
            raise ValueError(f"builder identity mismatch for {name!r}")
        lower_ms = _milliseconds(low.duration)
        upper_ms = _milliseconds(high.duration)
        if lower_ms > upper_ms:
            raise ValueError(f"duration bounds reversed for {name!r}")
        low_start = low.sample(0.0)
        low_end = low.sample(1.0)
        high_start = high.sample(0.0)
        high_end = high.sample(1.0)
        channels = sorted(set(low.keys) | set(high.keys))
        # Hue is an absolute palette coordinate, not a zero-centred motion
        # offset. Every other sampled channel must enter and leave at zero.
        nonzero_endpoints = [
            channel for channel in channels
            if channel != "hue"
            if abs(float(low_start.get(channel, 0.0))) > 1e-9
            or abs(float(low_end.get(channel, 0.0))) > 1e-9
            or abs(float(high_start.get(channel, 0.0))) > 1e-9
            or abs(float(high_end.get(channel, 0.0))) > 1e-9
        ]
        if nonzero_endpoints:
            raise ValueError(
                f"act {name!r} does not return to neutral on {nonzero_endpoints!r}")
        scheduled_item = scheduled.get(name)
        records.append({
            "name": name,
            "scheduled": scheduled_item is not None,
            "eligible_modes": list(scheduled_item[1]) if scheduled_item else [],
            "cooldown_ms": _milliseconds(scheduled_item[2]) if scheduled_item else 0,
            "duration_min_ms": lower_ms,
            "duration_max_ms": upper_ms,
            "fable_channels": channels,
            "pet_reaction": reactions_by_name.get(name),
        })
    return records


def _mode_trace(cfg):
    engine = kff.CharacterEngine(cfg)
    engine.rng = random.Random(0xFABA1E)
    engine.created_at = 0.0
    engine.mode = "IDLE"
    engine.mode_since = 0.0
    engine.last_person_at = 0.0
    engine.greet_until = 0.0
    engine.search_until = 0.0
    engine.next_blink = math.inf
    engine.next_act_at = math.inf
    engine.act = None
    engine.saccade = (0.0, 0.0)
    engine.saccade_until = math.inf
    engine.hue_phase = 0.0
    engine.life_phases = (0.0,) * 6
    engine.still_ref = 0.0
    engine.still_since = 0.0
    engine.resting = False

    inputs = (
        (0, False),
        (7_000, True),
        (9_201, True),
        (9_250, False),
        (9_951, False),
        (13_752, False),
        (70_000, False),
    )
    trace = []
    with contextlib.redirect_stdout(io.StringIO()):
        for at_ms, face_present in inputs:
            engine.next_act_at = math.inf
            eye, head = engine.compute(
                at_ms / 1000.0,
                face_present,
                (0.0, 0.0),
                (0.0, 0.0),
                0.0,
                False,
            )
            trace.append({
                "at_ms": at_ms,
                "face_present": face_present,
                "expected_mode": engine.mode.lower(),
                "fable_eye_expression": eye.expression,
                "fable_head_natural": all(abs(value) <= 1e-9 for value in head),
                "require_rust_head_natural": at_ms == 70_000,
            })
    return trace


def _pet_trace(cfg):
    engine = kff.CharacterEngine(cfg)
    engine.rng = random.Random(0xB00B1E)
    engine.created_at = 0.0
    cases = (
        (5_000, 600, 200, 100, False, True, "boop", "startle_boop"),
        (8_000, 2_000, 950, 100, False, False, "play", "play_bow"),
        (13_000, 6_000, 150, 100, True, False, "affection", "affection_melt"),
    )
    trace = []
    with contextlib.redirect_stdout(io.StringIO()):
        for (at_ms, duration_ms, accumulated, samples, comfy, tap,
             reaction, expected_act) in cases:
            mean_hundredths = accumulated * 100 // samples
            engine.note_pet_episode({
                "tap": tap,
                "duration_s": duration_ms / 1000.0,
                "mean_delta": mean_hundredths / 100.0,
                "reached_comfy": comfy,
            }, at_ms / 1000.0)
            if engine.act.name != expected_act:
                raise ValueError("Fable pet reaction trace changed unexpectedly")
            trace.append({
                "at_ms": at_ms,
                "duration_ms": duration_ms,
                "accumulated_max_delta_ticks": accumulated,
                "delta_samples": samples,
                "reached_comfy": comfy,
                "tap": tap,
                "expected_reaction": reaction,
                "expected_act": expected_act,
            })
    return trace


def build_trace():
    cfg = _config()
    timing = kff.BEHAVIOR_TIMING_S
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "fable_python_expression_lab",
        "comparison_contract": COMPARISON_CONTRACT,
        "source_sha256": {
            "kiko_face_follow_py": _source_digest("kiko_face_follow.py"),
            "organic_motion_py": _source_digest("organic_motion.py"),
        },
        "timing": {
            "greeting_cooldown_ms": _milliseconds(cfg["greet_cooldown_s"]),
            "greeting_min_ms": _milliseconds(timing["greeting"][0]),
            "greeting_max_ms": _milliseconds(timing["greeting"][1]),
            "formal_greeting_min_ms": _milliseconds(timing["formal_greeting"][0]),
            "formal_greeting_max_ms": _milliseconds(timing["formal_greeting"][1]),
            "lost_ms": _milliseconds(timing["lost"]),
            "search_min_ms": _milliseconds(timing["search"][0]),
            "search_max_ms": _milliseconds(timing["search"][1]),
            "sleepy_after_idle_ms": _milliseconds(cfg["sleepy_after_idle_s"]),
            "rest_after_idle_ms": _milliseconds(cfg["rest_after_idle_s"]),
            "rest_ease_ms": _milliseconds(cfg["rest_ease_s"]),
            "first_act_min_ms": _milliseconds(timing["first_act"][0]),
            "first_act_max_ms": _milliseconds(timing["first_act"][1]),
            "tracking_act_gap_min_ms": _milliseconds(timing["tracking_act_gap"][0]),
            "tracking_act_gap_max_ms": _milliseconds(timing["tracking_act_gap"][1]),
            "idle_act_gap_min_ms": _milliseconds(timing["idle_act_gap"][0]),
            "idle_act_gap_max_ms": _milliseconds(timing["idle_act_gap"][1]),
            "saccade_min_ms": _milliseconds(timing["recurrent_saccade"][0]),
            "saccade_max_ms": _milliseconds(timing["recurrent_saccade"][1]),
        },
        "acts": _act_records(),
        "mode_trace": _mode_trace(cfg),
        "pet_trace": _pet_trace(cfg),
    }


def canonical_bytes():
    return (json.dumps(
        build_trace(), sort_keys=True, indent=2, separators=(",", ": "))
        + "\n").encode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true")
    action.add_argument("--write", action="store_true")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    args = parser.parse_args(argv)
    expected = canonical_bytes()
    if args.check:
        actual = args.fixture.read_bytes()
        if actual != expected:
            raise SystemExit(
                f"stale Fable behavior trace: regenerate {args.fixture}")
        return 0
    if args.write:
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        args.fixture.write_bytes(expected)
        return 0
    os.write(1, expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
