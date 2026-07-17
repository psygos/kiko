#!/usr/bin/env python3
"""Analyze logs produced by jetson_benchmark.py without touching hardware."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def _series(values: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(values),
        "min": min(values) if values else None,
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
    }


def parse_tegrastats(text: str) -> dict[str, Any]:
    gr3d: list[float] = []
    gr3d_mhz: list[float] = []
    ram_used_mb: list[float] = []
    ram_total_mb: list[float] = []
    swap_used_mb: list[float] = []
    temperatures: dict[str, list[float]] = {}
    rails: dict[str, dict[str, list[float]]] = {}

    for line in text.splitlines():
        match = re.search(r"\bGR3D_FREQ\s+(\d+(?:\.\d+)?)%", line)
        if match:
            gr3d.append(float(match.group(1)))
        match = re.search(r"\bGR3D_FREQ\s+\d+(?:\.\d+)?%@\[?(\d+(?:\.\d+)?)", line)
        if match:
            gr3d_mhz.append(float(match.group(1)))
        match = re.search(r"\bRAM\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)MB", line)
        if match:
            ram_used_mb.append(float(match.group(1)))
            ram_total_mb.append(float(match.group(2)))
        match = re.search(r"\bSWAP\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)MB", line)
        if match:
            swap_used_mb.append(float(match.group(1)))
        for sensor, raw in re.findall(r"([A-Za-z0-9_-]+)@(-?\d+(?:\.\d+)?)C", line):
            temperatures.setdefault(sensor, []).append(float(raw))
        for rail, current, average in re.findall(
            r"\b([A-Z][A-Z0-9_]+)\s+(\d+(?:\.\d+)?)mW/(\d+(?:\.\d+)?)mW",
            line,
        ):
            data = rails.setdefault(rail, {"current_mw": [], "average_mw": []})
            data["current_mw"].append(float(current))
            data["average_mw"].append(float(average))

    all_temperatures = [value for values in temperatures.values() for value in values]
    return {
        "line_count": len([line for line in text.splitlines() if line.strip()]),
        "gr3d_utilization_pct": _series(gr3d),
        "gr3d_nonzero_samples": sum(value > 0 for value in gr3d),
        "gr3d_frequency_mhz": _series(gr3d_mhz),
        "ram_used_mb": _series(ram_used_mb),
        "ram_total_mb": max(ram_total_mb) if ram_total_mb else None,
        "swap_used_mb": _series(swap_used_mb),
        "temperature_c": _series(all_temperatures),
        "temperature_by_sensor_c": {
            sensor: _series(values) for sensor, values in sorted(temperatures.items())
        },
        "power_rails": {
            rail: {
                "current_mw": _series(values["current_mw"]),
                "reported_average_mw": _series(values["average_mw"]),
            }
            for rail, values in sorted(rails.items())
        },
    }


KERNEL_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "xid": (re.compile(r"\bXid\b", re.IGNORECASE), re.compile(r"\bNVRM\b.*\bXid\b", re.IGNORECASE)),
    "oom": (
        re.compile(r"out of memory", re.IGNORECASE),
        re.compile(r"oom-kill", re.IGNORECASE),
        re.compile(r"killed process", re.IGNORECASE),
    ),
    "thermal": (
        re.compile(r"thermal.*(?:trip|critical|shutdown|thrott)", re.IGNORECASE),
        re.compile(r"overheat", re.IGNORECASE),
        re.compile(r"soctherm.*(?:alarm|shutdown|thrott)", re.IGNORECASE),
    ),
    "power": (
        re.compile(r"brownout", re.IGNORECASE),
        re.compile(r"under[- ]?voltage", re.IGNORECASE),
        re.compile(r"over[- ]?current|oc alarm", re.IGNORECASE),
        re.compile(r"power.*(?:fault|shutdown)", re.IGNORECASE),
    ),
    "storage": (
        re.compile(
            r"\bnvme\b.*(?:\bI/O\b.*\btimeout\b|\btimeout\b|\breset\b|\babort\b|\bI/O error\b)",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:buffer|blk_update_request).*\bI/O error\b", re.IGNORECASE),
        re.compile(r"\b(?:ext4|xfs|btrfs)[- ]fs.*\berror\b", re.IGNORECASE),
    ),
}


def new_kernel_lines(before: str, after: str) -> list[str]:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    if not before_lines:
        return after_lines
    if after_lines[: len(before_lines)] == before_lines:
        return after_lines[len(before_lines) :]
    anchor = before_lines[-1]
    matches = [index for index, line in enumerate(after_lines) if line == anchor]
    return after_lines[matches[-1] + 1 :] if matches else after_lines


def classify_kernel_lines(lines: Iterable[str]) -> dict[str, list[str]]:
    findings = {category: [] for category in KERNEL_PATTERNS}
    for line in lines:
        for category, patterns in KERNEL_PATTERNS.items():
            if any(pattern.search(line) for pattern in patterns):
                findings[category].append(line)
    return findings


SESSION_POLICY = re.compile(
    r"ort session policy:\s+model=(?P<model>.+?)\s+"
    r"requested_backend=(?P<requested>\w+)\s+"
    r"configured_primary_backend=(?P<primary>\w+)\s+"
    r"configured_providers=\[(?P<providers>[^]]*)]\s+"
    r"strict_backend_registration=(?P<registration_strict>true|false)\s+"
    r"ort_cpu_ep_fallback_disabled=(?P<cpu_disabled>true|false)\s+"
    r"session_committed=(?P<committed>true|false)",
    re.IGNORECASE,
)


def parse_session_policies(text: str) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for match in SESSION_POLICY.finditer(text):
        providers = [
            value.strip()
            for value in match.group("providers").split(",")
            if value.strip()
        ]
        policies.append(
            {
                "model": match.group("model"),
                "requested_backend": match.group("requested"),
                "configured_primary_backend": match.group("primary"),
                "providers": providers,
                "strict_backend_registration": (
                    match.group("registration_strict").lower() == "true"
                ),
                "ort_cpu_ep_fallback_disabled": (
                    match.group("cpu_disabled").lower() == "true"
                ),
                "session_committed": match.group("committed").lower() == "true",
            }
        )
    return policies


def session_evidence_failures(
    policies: list[dict[str, Any]],
    expected_provider: str,
    expected_model_paths: set[str],
) -> list[str]:
    failures: list[str] = []
    expected_stack = ["CUDA"] if expected_provider == "cuda" else ["TensorRT", "CUDA"]
    if not policies:
        failures.append("missing_ort_session_policy")
    for policy in policies:
        if policy["requested_backend"].lower() != expected_provider:
            failures.append("wrong_requested_backend")
        if policy["configured_primary_backend"].lower() != expected_provider:
            failures.append("wrong_configured_primary_backend")
        if policy["providers"] != expected_stack:
            failures.append("wrong_provider_stack")
        if not policy["strict_backend_registration"]:
            failures.append("backend_registration_not_strict")
        if not policy["session_committed"]:
            failures.append("session_not_committed")
    observed_models = {policy["model"] for policy in policies}
    for path in expected_model_paths - observed_models:
        failures.append(f"missing_model_session_{Path(path).name}")
    for path in observed_models - expected_model_paths:
        failures.append(f"unexpected_model_session_{Path(path).name}")
    return failures


def _provider_name(raw: str) -> str | None:
    lowered = raw.lower()
    if "tensorrt" in lowered:
        return "TensorRT"
    if "cuda" in lowered:
        return "CUDA"
    if "cpu" in lowered:
        return "CPU"
    return None


def parse_node_placements(text: str) -> dict[str, Any]:
    evidence = [
        line
        for line in text.splitlines()
        if "VerifyEachNodeIsAssignedToAnEp" in line
        or re.search(r"nodes? placed on .*ExecutionProvider", line, re.IGNORECASE)
    ]
    counts: dict[str, int] = {}
    sessions: list[dict[str, Any]] = []
    current: dict[str, int] = {}

    def finish_session() -> None:
        nonlocal current
        if not current:
            return
        total = sum(current.values())
        sessions.append(
            {
                "provider_node_counts": current,
                "total_nodes": total,
                "cpu_node_fraction": current.get("CPU", 0) / total,
            }
        )
        current = {}

    for line in evidence:
        if re.search(r"VerifyEachNodeIsAssignedToAnEp.*\bNode placements\s*$", line):
            finish_session()
        matches = re.findall(
            r"(?:All nodes|Node\(s\)) placed on "
            r"\[?([^\].:]+ExecutionProvider)\]?[.:].*?Number of nodes:\s*(\d+)"
            r"|Number of nodes placed on "
            r"\[?([^\].:]+ExecutionProvider)\]?[.:]\s*(\d+)",
            line,
            re.IGNORECASE,
        )
        for first_provider, first_count, legacy_provider, legacy_count in matches:
            provider_raw = first_provider or legacy_provider
            raw_count = first_count or legacy_count
            if provider := _provider_name(provider_raw):
                count = int(raw_count)
                if provider in current:
                    finish_session()
                current[provider] = count
                counts[provider] = counts.get(provider, 0) + count
    finish_session()
    return {
        "evidence": evidence,
        "provider_node_counts": counts,
        "sessions": sessions,
        "maximum_cpu_node_fraction": max(
            (session["cpu_node_fraction"] for session in sessions), default=None
        ),
    }


def placement_evidence_failures(
    placements: dict[str, Any],
    expected_provider: str,
    max_cpu_node_fraction: float = 0.0,
) -> list[str]:
    failures: list[str] = []
    counts = placements["provider_node_counts"]
    if not placements["evidence"]:
        failures.append("missing_ort_node_placement")
    sessions = placements.get("sessions", [])
    if not sessions:
        failures.append("missing_ort_node_placement_counts")
    for session in sessions:
        if session["cpu_node_fraction"] > max_cpu_node_fraction:
            failures.append("cpu_node_fraction_exceeded")
        required_provider = "TensorRT" if expected_provider == "tensorrt" else "CUDA"
        if session["provider_node_counts"].get(required_provider, 0) <= 0:
            failures.append(f"session_missing_{required_provider.lower()}_node_placement")
    if counts.get("CUDA", 0) <= 0:
        failures.append("missing_cuda_node_placement")
    if expected_provider == "tensorrt":
        if counts.get("TensorRT", 0) <= 0:
            failures.append("missing_tensorrt_node_placement")
    elif counts.get("TensorRT", 0) > 0:
        failures.append("unexpected_tensorrt_node_placement")
    return failures


def parse_command_counts(text: str) -> dict[str, Any] | None:
    done = re.search(r"^done:\s*(?P<body>.+)$", text, re.MULTILINE)
    if done:
        fields: dict[str, Any] = {}
        for key, raw in re.findall(r"([A-Za-z_]+)=([^,\s]+)", done.group("body")):
            try:
                fields[key] = int(raw)
            except ValueError:
                try:
                    fields[key] = float(raw.removesuffix("s"))
                except ValueError:
                    fields[key] = raw
        return {"kind": "slam", **fields}

    completion = re.search(
        r"^completion:\s+expected_pairs=(\d+)\s+entries_attempted=(\d+)\s+"
        r"read_samples=(\d+)\s+processed=(\d+)\s+warmup_processed=(\d+)\s+"
        r"steady_processed=(\d+)$",
        text,
        re.MULTILINE,
    )
    if completion:
        errors = re.search(
            r"^errors:\s+read=(\d+)\s+pairing=(\d+)\s+inference=(\d+)$",
            text,
            re.MULTILINE,
        )
        return {
            "kind": "bench",
            "expected": int(completion.group(1)),
            "attempted": int(completion.group(2)),
            "read_samples": int(completion.group(3)),
            "processed": int(completion.group(4)),
            "warmup_processed": int(completion.group(5)),
            "steady_processed": int(completion.group(6)),
            "read_errors": int(errors.group(1)) if errors else None,
            "pairing_errors": int(errors.group(2)) if errors else None,
            "inference_errors": int(errors.group(3)) if errors else None,
        }

    processed = re.search(
        r"^pipeline wall fps:.*?\bprocessed=(\d+)", text, re.MULTILINE
    )
    if not processed:
        return None
    read_samples = re.search(
        r"^reader stage fps:.*?\bread_samples=(\d+)", text, re.MULTILINE
    )
    errors = re.search(
        r"^errors:\s+read=(\d+)\s+pairing=(\d+)\s+inference=(\d+)$",
        text,
        re.MULTILINE,
    )
    return {
        "kind": "bench",
        "processed": int(processed.group(1)),
        "attempted": int(read_samples.group(1)) if read_samples else None,
        "read_errors": int(errors.group(1)) if errors else None,
        "pairing_errors": int(errors.group(2)) if errors else None,
        "inference_errors": int(errors.group(3)) if errors else None,
    }


def count_evidence_failures(
    counts: dict[str, Any] | None, expected_items: int, expected_warmup: int
) -> list[str]:
    if counts is None:
        return ["missing_command_counts"]
    failures: list[str] = []
    if counts.get("processed") != expected_items:
        failures.append("processed_count_mismatch")
    if counts.get("expected") != expected_items:
        failures.append("reported_expected_count_mismatch")
    if counts.get("kind") == "bench":
        if counts.get("attempted") != expected_items:
            failures.append("attempted_count_mismatch")
        if counts.get("read_samples") != expected_items:
            failures.append("read_sample_count_mismatch")
        required_errors = ("read_errors", "pairing_errors", "inference_errors")
    else:
        if counts.get("entries_consumed") != expected_items:
            failures.append("entries_consumed_count_mismatch")
        if counts.get("tracker_attempts") != expected_items:
            failures.append("tracker_attempt_count_mismatch")
        required_errors = ("read_errors", "tracker_errors")
    if counts.get("warmup_processed") != expected_warmup:
        failures.append("warmup_count_mismatch")
    if counts.get("steady_processed") != expected_items - expected_warmup:
        failures.append("steady_count_mismatch")
    for key in required_errors:
        if counts.get(key) != 0:
            failures.append(f"nonzero_{key}")
    return failures


POSE_OUTCOMES = re.compile(
    r"^pose outcomes:\s+total_current=(\d+)\s+total_predicted=(\d+)\s+"
    r"total_stale=(\d+)\s+total_unavailable=(\d+)\s+steady_current=(\d+)\s+"
    r"steady_predicted=(\d+)\s+steady_stale=(\d+)\s+steady_unavailable=(\d+)$",
    re.MULTILINE,
)


def parse_pose_outcomes(text: str) -> dict[str, int] | None:
    match = POSE_OUTCOMES.search(text)
    if match is None:
        return None
    keys = (
        "total_current",
        "total_predicted",
        "total_stale",
        "total_unavailable",
        "steady_current",
        "steady_predicted",
        "steady_stale",
        "steady_unavailable",
    )
    return dict(zip(keys, (int(value) for value in match.groups()), strict=True))


TRIANGULATION_POLICY = re.compile(
    r"^triangulation: .*\bmax_vertical_disparity_px="
    r"(?P<maximum>unbounded|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$",
    re.MULTILINE,
)


def parse_triangulation_policy(text: str) -> dict[str, Any] | None:
    matches = list(TRIANGULATION_POLICY.finditer(text))
    if len(matches) != 1:
        return None
    raw_maximum = matches[0].group("maximum")
    if raw_maximum == "unbounded":
        return {"kind": "unbounded"}
    maximum = float(raw_maximum)
    if not math.isfinite(maximum) or maximum < 0.0:
        return None
    return {"kind": "finite", "max_vertical_disparity_px": maximum}


DIAGNOSTIC_TOTAL_KEYS = frozenset(
    {
        "frames",
        "steady_frames",
        "final_map_keyframes",
        "final_map_points",
        "peak_map_points",
        "features_detected_samples",
        "features_detected_total",
        "steady_features_detected_samples",
        "steady_features_detected_total",
        "features_matched_samples",
        "features_matched_total",
        "steady_features_matched_samples",
        "steady_features_matched_total",
        "pnp_tracked_observations_samples",
        "pnp_tracked_observations_total",
        "steady_pnp_tracked_observations_samples",
        "steady_pnp_tracked_observations_total",
        "pnp_projectable_tracked_observations_samples",
        "pnp_projectable_tracked_observations_total",
        "steady_pnp_projectable_tracked_observations_samples",
        "steady_pnp_projectable_tracked_observations_total",
        "pnp_accepted_inliers_samples",
        "pnp_accepted_inliers_total",
        "steady_pnp_accepted_inliers_samples",
        "steady_pnp_accepted_inliers_total",
        "triangulation_samples",
        "triangulation_candidate_matches_total",
        "triangulation_kept_total",
        "triangulation_dropped_disparity_total",
        "triangulation_dropped_epipolar_total",
        "triangulation_dropped_depth_total",
        "triangulation_dropped_numerical_total",
        "triangulation_dropped_unrepresentable_total",
        "triangulation_dropped_duplicate_total",
    }
)


def parse_diagnostic_totals(text: str) -> dict[str, int] | None:
    match = re.search(r"^diagnostic totals:\s*(.+)$", text, re.MULTILINE)
    if match is None:
        return None
    pairs = re.findall(r"([a-z_]+)=(\d+)", match.group(1))
    if len(pairs) != len({key for key, _ in pairs}):
        return None
    values = {key: int(value) for key, value in pairs}
    if not DIAGNOSTIC_TOTAL_KEYS.issubset(values):
        return None
    return values


def diagnostic_total_failures(
    totals: dict[str, int] | None, counts: dict[str, Any]
) -> list[str]:
    if totals is None:
        return ["missing_diagnostic_totals"]
    failures: list[str] = []
    if totals["frames"] != counts.get("processed"):
        failures.append("diagnostic_frame_count_mismatch")
    if totals["steady_frames"] != counts.get("steady_processed"):
        failures.append("steady_diagnostic_frame_count_mismatch")
    if totals["final_map_keyframes"] != counts.get("keyframes"):
        failures.append("diagnostic_keyframe_count_mismatch")
    if totals["final_map_points"] > totals["peak_map_points"]:
        failures.append("diagnostic_map_point_range_invalid")
    for metric in ("features_detected", "features_matched"):
        if totals[f"{metric}_samples"] != totals["frames"]:
            failures.append(f"{metric}_sample_count_mismatch")
        if totals[f"steady_{metric}_samples"] != totals["steady_frames"]:
            failures.append(f"steady_{metric}_sample_count_mismatch")
    triangulation_samples = totals["triangulation_samples"]
    triangulation_candidates = totals["triangulation_candidate_matches_total"]
    if triangulation_samples == 0 or triangulation_samples > totals["frames"]:
        failures.append("triangulation_sample_count_invalid")
    triangulation_accounted = sum(
        totals[key]
        for key in (
            "triangulation_kept_total",
            "triangulation_dropped_disparity_total",
            "triangulation_dropped_epipolar_total",
            "triangulation_dropped_depth_total",
            "triangulation_dropped_numerical_total",
            "triangulation_dropped_unrepresentable_total",
            "triangulation_dropped_duplicate_total",
        )
    )
    if triangulation_accounted != triangulation_candidates:
        failures.append("triangulation_candidate_accounting_invalid")
    return failures


def pose_outcome_failures(
    outcomes: dict[str, int] | None, steady_processed: int | None
) -> list[str]:
    if outcomes is None:
        return ["missing_pose_outcomes"]
    failures: list[str] = []
    steady_valid = outcomes["steady_current"] + outcomes["steady_predicted"]
    if steady_valid != steady_processed:
        failures.append("steady_pose_count_mismatch")
    if outcomes["steady_stale"] != 0:
        failures.append("steady_stale_poses")
    if outcomes["steady_unavailable"] != 0:
        failures.append("steady_unavailable_poses")
    return failures


def parse_reported_metrics(
    text: str, counts: dict[str, Any] | None = None
) -> dict[str, Any]:
    status = re.search(r"^(\w+) metrics status:\s*(\w+)$", text, re.MULTILINE)
    steady_fps = re.search(
        r"^steady pipeline wall fps:\s*([0-9.]+).*?processed=(\d+).*?elapsed=([0-9.]+)s",
        text,
        re.MULTILINE,
    )
    if not steady_fps:
        steady_fps = re.search(
            r"^done:.*?steady_processed=(\d+).*?steady_elapsed=([0-9.]+)s,\s*steady_fps=([0-9.]+)",
            text,
            re.MULTILINE,
        )
        steady = (
            {
                "processed": int(steady_fps.group(1)),
                "elapsed_seconds": float(steady_fps.group(2)),
                "fps": float(steady_fps.group(3)),
            }
            if steady_fps
            else None
        )
    else:
        steady = {
            "fps": float(steady_fps.group(1)),
            "processed": int(steady_fps.group(2)),
            "elapsed_seconds": float(steady_fps.group(3)),
        }
    latency = re.search(
        r"^steady latency ms \(median/p95, samples=(\d+)\):\s*(.+)$",
        text,
        re.MULTILINE,
    )
    stage_latency: dict[str, dict[str, float]] = {}
    if latency:
        for stage, median, p95 in re.findall(
            r"([a-z_]+)=([0-9.]+)/([0-9.]+)", latency.group(2)
        ):
            stage_latency[stage] = {"median_ms": float(median), "p95_ms": float(p95)}
    service = re.search(
        r"^steady pipeline service interval ms \(median/p95, samples=(\d+)\):\s*"
        r"([0-9.]+)/([0-9.]+)$",
        text,
        re.MULTILINE,
    )
    initialization = re.search(
        r"^inference session initialization:\s*([0-9.]+)ms$", text, re.MULTILINE
    )
    selected = re.search(
        r"^selected-run pipeline wall fps \(session initialization excluded\):\s*"
        r"([0-9.]+).*?processed=(\d+).*?elapsed=([0-9.]+)s",
        text,
        re.MULTILINE,
    )
    total = (
        {
            "fps": float(selected.group(1)),
            "processed": int(selected.group(2)),
            "elapsed_seconds": float(selected.group(3)),
        }
        if selected
        else None
    )
    if total is None and counts and counts.get("kind") == "slam":
        if all(key in counts for key in ("fps", "processed", "elapsed")):
            total = {
                "fps": float(counts["fps"]),
                "processed": int(counts["processed"]),
                "elapsed_seconds": float(counts["elapsed"]),
            }
    stage_fps: dict[str, float] = {}
    for key, pattern in {
        "inference_attempt": r"^steady inference-attempt wall fps .*?:\s*([0-9.]+)",
        "model_pipeline": r"^steady model-pipeline timing fps:\s*([0-9.]+)",
    }.items():
        if match := re.search(pattern, text, re.MULTILINE):
            stage_fps[key] = float(match.group(1))
    return {
        "status_scope": status.group(1) if status else None,
        "status": status.group(2) if status else None,
        "steady": steady,
        "total": total,
        "inference_initialization_ms": (
            float(initialization.group(1)) if initialization else None
        ),
        "steady_stage_fps": stage_fps,
        "stage_latency_samples": int(latency.group(1)) if latency else None,
        "stage_latency": stage_latency,
        "service_interval": (
            {
                "samples": int(service.group(1)),
                "median_ms": float(service.group(2)),
                "p95_ms": float(service.group(3)),
            }
            if service
            else None
        ),
    }


def parse_loaded_realpaths(maps_text: str, loader_text: str) -> list[str]:
    paths: set[str] = set()
    for raw in re.findall(r"\brealpath=(\"(?:\\.|[^\"])*\")", maps_text):
        try:
            paths.add(json.loads(raw))
        except json.JSONDecodeError:
            pass
    for raw in re.findall(r"calling init:\s+(/\S+)", loader_text):
        candidate = Path(raw)
        try:
            paths.add(str(candidate.resolve(strict=True)))
        except OSError:
            paths.add(raw)
    return sorted(paths)


GPU_LIBRARY_PREFIXES = (
    "libcuda.so",
    "libcudart.so",
    "libcublas.so",
    "libcublasLt.so",
    "libcudnn",
    "libnvrtc",
    "libnvinfer",
    "libnvonnxparser",
    "libnvcaffe_parser",
)
TENSORRT_LIBRARY_PREFIXES = ("libnvinfer", "libnvonnxparser", "libnvcaffe_parser")


def runtime_library_evidence_failures(
    loaded_paths: set[str], expected_paths: set[str], expected_provider: str
) -> list[str]:
    failures: list[str] = []
    for path in expected_paths - loaded_paths:
        failures.append(f"runtime_library_not_observed_{Path(path).name}")
    expected_ort_paths = {
        path for path in expected_paths if Path(path).name.startswith("libonnxruntime")
    }
    for path in loaded_paths:
        name = Path(path).name
        if name.startswith("libonnxruntime") and path not in expected_ort_paths:
            failures.append(f"unintended_ort_library_{name}")
        if expected_provider == "cuda":
            if "providers_tensorrt" in name:
                failures.append("unexpected_tensorrt_provider_loaded")
            if name.startswith(TENSORRT_LIBRARY_PREFIXES):
                failures.append("unexpected_tensorrt_library_loaded")
    return failures


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def artifact_paths(artifacts: dict[str, Any], keys: set[str]) -> set[str]:
    paths: set[str] = set()
    for key in keys:
        value = artifacts.get(key)
        if isinstance(value, dict) and isinstance(value.get("path"), str):
            paths.add(value["path"])
    return paths


def analyze_run(
    run_dir: Path, *, accepted_result_states: frozenset[str] = frozenset({"completed"})
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    result = _read_json(run_dir / "result.json")
    manifest = _read_json(run_dir / "command.json")
    before = _read_json(run_dir / "system-pre.json")
    after = _read_json(run_dir / "system-post.json")
    stderr = _read(run_dir / "stderr.log")
    stdout = _read(run_dir / "stdout.log")
    combined_output = stderr + "\n" + stdout
    tegra = parse_tegrastats(_read(run_dir / "tegrastats.log"))
    kernel_pre_text = _read(run_dir / "kernel-pre.log")
    kernel_post_text = _read(run_dir / "kernel-post.log")
    kernel_delta = new_kernel_lines(kernel_pre_text, kernel_post_text)
    kernel = classify_kernel_lines(kernel_delta)
    policies = parse_session_policies(combined_output)
    placements = parse_node_placements(combined_output)
    counts = parse_command_counts(combined_output)
    pose_outcomes = parse_pose_outcomes(combined_output)
    triangulation_policy = parse_triangulation_policy(combined_output)
    diagnostic_totals = parse_diagnostic_totals(combined_output)
    reported_metrics = parse_reported_metrics(combined_output, counts)
    loaded_realpaths = parse_loaded_realpaths(
        _read(run_dir / "process-maps.log"), stderr
    )

    expected_provider = str(manifest.get("provider", "")).lower()
    failures: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        failures.append("command_schema_version_mismatch")
    if result.get("schema_version") != SCHEMA_VERSION:
        failures.append("result_schema_version_mismatch")
    if expected_provider not in {"cuda", "tensorrt"}:
        failures.append("invalid_expected_provider")
    if result.get("returncode") != 0:
        failures.append("workload_not_completed_successfully")
    elif result.get("state") not in accepted_result_states:
        failures.append("run_not_completed_successfully")
    if before.get("boot_id") and after.get("boot_id") and before["boot_id"] != after["boot_id"]:
        failures.append("boot_id_changed")
    if before.get("reset") != after.get("reset"):
        failures.append("reset_state_changed")
    if not kernel_pre_text or kernel_pre_text.startswith("unavailable:"):
        failures.append("missing_kernel_pre_evidence")
    if not kernel_post_text or kernel_post_text.startswith("unavailable:"):
        failures.append("missing_kernel_post_evidence")
    if tegra["gr3d_utilization_pct"]["samples"] == 0:
        failures.append("missing_gr3d_samples")
    elif tegra["gr3d_nonzero_samples"] == 0:
        failures.append("no_gr3d_activity")
    artifacts = result.get("preflight", {}).get("artifacts", {})
    expected_models = artifact_paths(
        artifacts, {"superpoint_model", "lightglue_model"}
    )
    failures.extend(session_evidence_failures(policies, expected_provider, expected_models))
    thresholds = manifest.get("thresholds", {})
    max_cpu_node_fraction = thresholds.get("max_cpu_node_fraction", 0.0)
    if (
        not isinstance(max_cpu_node_fraction, (int, float))
        or isinstance(max_cpu_node_fraction, bool)
        or not math.isfinite(max_cpu_node_fraction)
        or not 0.0 <= max_cpu_node_fraction <= 1.0
    ):
        failures.append("invalid_max_cpu_node_fraction")
        max_cpu_node_fraction = 0.0
    failures.extend(
        placement_evidence_failures(
            placements, expected_provider, float(max_cpu_node_fraction)
        )
    )
    for category, lines in kernel.items():
        if lines:
            failures.append(f"kernel_{category}_evidence")

    expected_items = manifest.get("expected_command_items")
    selection = manifest.get("selection", {})
    expected_warmup = selection.get("warmup_pairs")
    if not isinstance(expected_items, int) or isinstance(expected_items, bool):
        failures.append("invalid_expected_command_items")
    elif not isinstance(expected_warmup, int) or isinstance(expected_warmup, bool):
        failures.append("invalid_expected_warmup")
    else:
        failures.extend(count_evidence_failures(counts, expected_items, expected_warmup))

    if counts and reported_metrics["status"] != "valid":
        failures.append(f"{counts.get('kind')}_metrics_not_valid")
    if counts and counts.get("kind") == "slam":
        failures.extend(pose_outcome_failures(pose_outcomes, counts.get("steady_processed")))
        failures.extend(diagnostic_total_failures(diagnostic_totals, counts))

    expected_nvpmodel = thresholds.get("expected_nvpmodel")
    minimum_cpu_hz = thresholds.get("min_cpu_hz")
    minimum_gpu_hz = thresholds.get("min_gpu_hz")
    minimum_emc_hz = thresholds.get("min_emc_hz")
    expected_emc_override = thresholds.get("expected_emc_override")
    minimum_memory_mib = thresholds.get("min_memory_available_mib")
    maximum_swap = thresholds.get("max_swap_used_mib")
    maximum_temperature = thresholds.get("max_temperature_c")
    for label, state in (("pre", before), ("post", after)):
        if expected_nvpmodel is not None:
            nvpmodel = state.get("nvpmodel", {})
            nvp_text = str(nvpmodel.get("stdout", "")) + str(nvpmodel.get("stderr", ""))
            mode = re.search(r"NV Power Mode:\s*([^\r\n]+)", nvp_text)
            if not mode or mode.group(1).strip() != expected_nvpmodel:
                failures.append(f"{label}_nvpmodel_gate_failed")
        gpu_hz = state.get("gpu_frequency", {}).get("hz")
        if minimum_gpu_hz is not None and (gpu_hz is None or gpu_hz < minimum_gpu_hz):
            failures.append(f"{label}_gpu_clock_gate_failed")
        emc_hz = state.get("emc_frequency", {}).get("hz")
        if minimum_emc_hz is not None and (emc_hz is None or emc_hz < minimum_emc_hz):
            failures.append(f"{label}_emc_clock_gate_failed")
        cpu_frequencies = state.get("cpu_frequencies_hz", [])
        if minimum_cpu_hz is not None and (
            not cpu_frequencies or min(cpu_frequencies) < minimum_cpu_hz
        ):
            failures.append(f"{label}_cpu_clock_gate_failed")
        emc_override = state.get("emc_override") or {}
        override_value = emc_override.get("value", emc_override.get("hz"))
        if expected_emc_override is not None and override_value != expected_emc_override:
            failures.append(f"{label}_emc_override_gate_failed")
        meminfo = state.get("meminfo_kib", {})
        if minimum_memory_mib is not None and (
            meminfo.get("MemAvailable") is None
            or meminfo["MemAvailable"] < minimum_memory_mib * 1024
        ):
            failures.append(f"{label}_memory_gate_failed")
        if maximum_swap is not None:
            swap_total = meminfo.get("SwapTotal")
            swap_free = meminfo.get("SwapFree")
            if (
                swap_total is None
                or swap_free is None
                or swap_total - swap_free > maximum_swap * 1024
            ):
                failures.append(f"{label}_swap_gate_failed")
        temperatures = [
            zone.get("temperature_c")
            for zone in state.get("thermal_zones", [])
            if isinstance(zone, dict)
            and isinstance(zone.get("temperature_c"), (int, float))
        ]
        if maximum_temperature is not None and (
            not temperatures or max(temperatures) >= maximum_temperature
        ):
            failures.append(f"{label}_temperature_gate_failed")
    observed_swap = tegra["swap_used_mb"]["max"]
    zero_swap_proven = all(
        state.get("meminfo_kib", {}).get("SwapTotal") == 0
        and state.get("meminfo_kib", {}).get("SwapFree") == 0
        for state in (before, after)
    )
    if maximum_swap is not None:
        if observed_swap is None and not zero_swap_proven:
            failures.append("swap_gate_failed")
        elif observed_swap is not None and observed_swap > maximum_swap:
            failures.append("swap_gate_failed")
    observed_temperature = tegra["temperature_c"]["max"]
    if maximum_temperature is not None and (
        observed_temperature is None or observed_temperature >= maximum_temperature
    ):
        failures.append("temperature_gate_failed")
    for rail in thresholds.get("required_power_rails", []):
        if rail not in tegra["power_rails"]:
            failures.append(f"missing_power_rail_{rail}")
    minimum_steady_fps = thresholds.get("min_steady_fps")
    steady = reported_metrics.get("steady")
    if minimum_steady_fps is not None:
        if steady is None:
            failures.append("missing_steady_fps")
        elif steady["fps"] < minimum_steady_fps:
            failures.append("steady_fps_below_gate")
    expected_triangulation_policy = thresholds.get("expected_triangulation_policy")
    if counts and counts.get("kind") == "slam":
        expected_kind = (
            expected_triangulation_policy.get("kind")
            if isinstance(expected_triangulation_policy, dict)
            else None
        )
        expected_policy_valid = expected_kind == "unbounded"
        if expected_kind == "finite":
            expected_maximum = expected_triangulation_policy.get(
                "max_vertical_disparity_px"
            )
            expected_policy_valid = (
                isinstance(expected_maximum, (int, float))
                and not isinstance(expected_maximum, bool)
                and math.isfinite(expected_maximum)
                and expected_maximum >= 0.0
            )
        if expected_triangulation_policy is None:
            failures.append("missing_expected_triangulation_policy")
        elif not expected_policy_valid:
            failures.append("invalid_expected_triangulation_policy")
        elif triangulation_policy is None:
            failures.append("missing_triangulation_policy")
        elif expected_kind != triangulation_policy.get("kind"):
            failures.append("triangulation_policy_mismatch")
        elif expected_kind == "finite" and not math.isclose(
            triangulation_policy["max_vertical_disparity_px"],
            float(expected_triangulation_policy["max_vertical_disparity_px"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            failures.append("triangulation_policy_mismatch")

    expected_library_paths = artifact_paths(
        artifacts,
        {
            "onnxruntime",
            "ort_shared_provider",
            "ort_cuda_provider",
            "ort_tensorrt_provider",
        },
    )
    runtime_dependencies = (
        result.get("preflight", {}).get("elf", {}).get("runtime_dependencies", {})
    )
    if isinstance(runtime_dependencies, dict):
        expected_library_paths.update(
            path for path in runtime_dependencies.values() if isinstance(path, str)
        )
    loaded = set(loaded_realpaths)
    failures.extend(
        runtime_library_evidence_failures(loaded, expected_library_paths, expected_provider)
    )
    loaded_gpu_libraries = sorted(
        path for path in loaded if Path(path).name.startswith(GPU_LIBRARY_PREFIXES)
    )

    comparison_metrics_valid = not failures

    return {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "pass": not failures,
        "failures": sorted(set(failures)),
        "result_state": result.get("state"),
        "boot_id_before": before.get("boot_id"),
        "boot_id_after": after.get("boot_id"),
        "boot_id_changed": bool(
            before.get("boot_id")
            and after.get("boot_id")
            and before["boot_id"] != after["boot_id"]
        ),
        "tegrastats": tegra,
        "kernel_new_line_count": len(kernel_delta),
        "kernel_evidence": kernel,
        "session_policies": policies,
        "node_placement": placements,
        "loaded_realpaths": loaded_realpaths,
        "loaded_gpu_libraries": loaded_gpu_libraries,
        "command_counts": counts,
        "pose_outcomes": pose_outcomes,
        "triangulation_policy": triangulation_policy,
        "diagnostic_totals": diagnostic_totals,
        "reported_metrics": reported_metrics,
        "comparison_metrics_valid": comparison_metrics_valid,
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analysis = analyze_run(args.run_dir)
    if args.output is not None:
        write_json(args.output, analysis)
    print(json.dumps(analysis, sort_keys=True))
    return 0 if analysis["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
