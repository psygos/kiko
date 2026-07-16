#!/usr/bin/env python3
"""Run one fail-closed, evidence-preserving Jetson SLAM validation stage."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from analyze_jetson_benchmark import (
    GPU_LIBRARY_PREFIXES,
    SCHEMA_VERSION,
    analyze_run,
    classify_kernel_lines,
    count_evidence_failures,
    new_kernel_lines,
    parse_command_counts,
    parse_node_placements,
    parse_pose_outcomes,
    parse_reported_metrics,
    parse_session_policies,
    parse_tegrastats,
    placement_evidence_failures,
    pose_outcome_failures,
    runtime_library_evidence_failures,
    session_evidence_failures,
)


DEFAULT_PATH = "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class GateError(RuntimeError):
    pass


class RunInterrupted(RuntimeError):
    def __init__(self, signum: int):
        super().__init__(f"received signal {signal.Signals(signum).name}")
        self.signum = signum


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_text_fsynced(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


class ResultJournal:
    def __init__(self, path: Path, output_dir: Path):
        self.path = path
        self.value: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "state": "initializing",
            "pass": False,
            "output_dir": str(output_dir),
            "runner_pid": os.getpid(),
            "started_at": now(),
            "failures": [],
        }
        atomic_json(self.path, self.value)

    def update(self, **fields: Any) -> None:
        self.value.update(fields)
        atomic_json(self.path, self.value)


class LineLog:
    def __init__(self, path: Path):
        self.handle = path.open("x", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()

    def write(self, line: str) -> None:
        with self.lock:
            self.handle.write(line.rstrip("\n") + "\n")
            self.handle.flush()

    def close(self) -> None:
        with self.lock:
            if not self.handle.closed:
                self.handle.flush()
                os.fsync(self.handle.fileno())
                self.handle.close()


def create_output_dir(root: Path, stage: str, explicit: Path | None) -> Path:
    safe_stage = re.sub(r"[^a-zA-Z0-9_.-]+", "-", stage).strip("-.")
    if not safe_stage:
        raise GateError("stage name has no filesystem-safe characters")
    if explicit is not None:
        output = explicit.expanduser()
        if not output.is_absolute():
            raise GateError("--output-dir must be absolute")
        output.mkdir(mode=0o750, parents=False, exist_ok=False)
        return output
    root = root.expanduser()
    if not root.is_absolute():
        raise GateError("--output-root must be absolute")
    root.mkdir(mode=0o750, parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    output = root / f"{stamp}-{safe_stage}-{os.getpid()}"
    output.mkdir(mode=0o750, exist_ok=False)
    return output


def regular_file(path: Path, label: str, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise GateError(f"{label} path must be absolute: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise GateError(f"{label} is unavailable: {path}: {error}") from error
    if not resolved.is_file():
        raise GateError(f"{label} is not a regular file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise GateError(f"{label} is not executable: {resolved}")
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hash_manifest(path: Path) -> dict[Path, str]:
    entries: dict[Path, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+[ *](.+)", line)
        if not match:
            raise GateError(f"invalid SHA256 manifest line {line_number}: {line!r}")
        candidate = Path(match.group(2))
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        resolved = candidate.resolve(strict=True)
        expected = match.group(1).lower()
        if resolved in entries and entries[resolved] != expected:
            raise GateError(f"conflicting hashes for {resolved}")
        entries[resolved] = expected
    return entries


def verify_artifacts(paths: dict[str, Path], hash_manifest: Path) -> dict[str, Any]:
    expected = load_hash_manifest(hash_manifest)
    artifacts: dict[str, Any] = {}
    for label, path in paths.items():
        digest = sha256(path)
        wanted = expected.get(path)
        if wanted is None:
            raise GateError(f"SHA256 manifest has no entry for {label}: {path}")
        if digest != wanted:
            raise GateError(f"SHA256 mismatch for {label}: expected {wanted}, got {digest}")
        artifacts[label] = {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}
    return artifacts


def validate_dataset(root: Path, expected_pairs: int) -> dict[str, Any]:
    if not root.is_absolute():
        raise GateError("dataset path must be absolute")
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise GateError(f"dataset is not a directory: {root}")
    manifest_path = regular_file(root / "manifest.json", "dataset manifest")
    meta_path = regular_file(root / "meta.json", "dataset metadata")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise GateError(f"dataset JSON is invalid: {error}") from error
    if not isinstance(manifest, dict) or not isinstance(metadata, dict):
        raise GateError("dataset manifest and metadata must be JSON objects")

    header = manifest.get("header", {})
    stats = manifest.get("stats", {})
    entries = manifest.get("entries")
    mono = metadata.get("mono", {})
    if not all(isinstance(value, dict) for value in (header, stats, mono)):
        raise GateError("dataset header, stats, and mono metadata must be objects")
    if not isinstance(entries, list):
        raise GateError("dataset manifest entries must be a list")
    if (header.get("width"), header.get("height")) != (640, 480):
        raise GateError("dataset manifest is not exact 640x480")
    if (mono.get("width"), mono.get("height")) != (640, 480):
        raise GateError("dataset metadata is not exact 640x480")
    if header.get("format") != "raw":
        raise GateError(f"dataset format is not raw: {header.get('format')!r}")
    count_fields = ("total_left", "total_right", "paired_count")
    if any(stats.get(field) != expected_pairs for field in count_fields):
        raise GateError(f"dataset counts do not all equal {expected_pairs}: {stats}")
    if stats.get("left_orphans") != 0 or stats.get("right_orphans") != 0:
        raise GateError("dataset contains orphaned stereo frames")
    drops = stats.get("drops_by_reason", {})
    if not isinstance(drops, dict) or any(value != 0 for value in drops.values()):
        raise GateError(f"dataset declares dropped frames: {drops}")
    if len(entries) != expected_pairs:
        raise GateError(f"manifest has {len(entries)} entries, expected {expected_pairs}")

    expected_size = 640 * 480
    referenced: set[Path] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise GateError(f"manifest entry {index} must be an object")
        if entry.get("status") != "paired":
            raise GateError(f"manifest entry {index} is not paired")
        for side in ("left", "right"):
            reference = entry.get(side)
            if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
                raise GateError(f"manifest entry {index} has no {side} path")
            relative = Path(reference["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise GateError(f"manifest entry {index} has unsafe {side} path")
            frame = (root / relative).resolve(strict=True)
            if root not in frame.parents or frame in referenced:
                raise GateError(f"manifest entry {index} has escaped or duplicate {side} path")
            if not frame.is_file() or frame.stat().st_size != expected_size:
                raise GateError(f"manifest entry {index} {side} frame is not {expected_size} bytes")
            referenced.add(frame)

    scanned = {
        path.resolve()
        for path in (root / "frames").glob("*_mono_*.raw")
        if path.name.endswith(("_mono_left.raw", "_mono_right.raw"))
    }
    if scanned != referenced:
        raise GateError(
            "dataset mono files differ from manifest: "
            f"referenced={len(referenced)} scanned={len(scanned)}"
        )
    return {
        "path": str(root),
        "width": 640,
        "height": 480,
        "paired_count": expected_pairs,
        "frame_count": len(referenced),
        "frame_size_bytes": expected_size,
        "manifest_sha256": sha256(manifest_path),
        "metadata_sha256": sha256(meta_path),
    }


def group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_group(
    process: subprocess.Popen[Any], grace_seconds: float, cleanup: LineLog | None, reason: str
) -> dict[str, bool]:
    outcome = {"term_sent": False, "kill_sent": False}
    process_group = process.pid
    if group_exists(process_group):
        if cleanup:
            cleanup.write(f"{now()} pid={process.pid} action=TERM reason={reason}")
        try:
            os.killpg(process_group, signal.SIGTERM)
            outcome["term_sent"] = True
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace_seconds
        while group_exists(process_group) and time.monotonic() < deadline:
            process.poll()
            time.sleep(0.02)
    if group_exists(process_group):
        if cleanup:
            cleanup.write(f"{now()} pid={process.pid} action=KILL reason={reason}")
        try:
            os.killpg(process_group, signal.SIGKILL)
            outcome["kill_sent"] = True
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=max(0.1, grace_seconds))
    except subprocess.TimeoutExpired:
        pass
    return outcome


def capture_command(
    command: list[str], env: dict[str, str], timeout: float = 10.0
) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    timed_out = False
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_group(process, 1.0, None, "capture_timeout")
            stdout, stderr = process.communicate()
    except BaseException:
        terminate_group(process, 1.0, None, "capture_interrupted")
        raise
    return {
        "command": command,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
    }


def write_capture(path: Path, capture: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(f"command={shlex.join(capture['command'])}\n")
        handle.write(f"returncode={capture['returncode']} timed_out={capture['timed_out']}\n")
        handle.write("[stdout]\n")
        handle.write(capture["stdout"])
        handle.write("\n[stderr]\n")
        handle.write(capture["stderr"])
        handle.flush()
        os.fsync(handle.fileno())


def parse_ldd_resolutions(text: str) -> dict[str, str]:
    resolutions: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"\s*(\S+)\s+=>\s+(/\S+)\s+\(", line)
        if not match:
            match = re.match(r"\s*(/\S+)\s+\(", line)
            if not match:
                continue
            soname = Path(match.group(1)).name
            raw_path = match.group(1)
        else:
            soname = match.group(1)
            raw_path = match.group(2)
        try:
            resolved = str(Path(raw_path).resolve(strict=True))
        except OSError:
            continue
        previous = resolutions.get(soname)
        if previous is not None and previous != resolved:
            raise GateError(f"ldd resolved {soname} to conflicting paths: {previous}, {resolved}")
        resolutions[soname] = resolved
    return resolutions


def preflight_elf(
    binary: Path,
    libraries: dict[str, Path],
    env: dict[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    readelf = shutil.which("readelf", path=env["PATH"])
    ldd = shutil.which("ldd", path=env["PATH"])
    if not readelf or not ldd:
        raise GateError("readelf and ldd must be available in the sterile PATH")
    notes = capture_command([readelf, "-n", str(binary)], env)
    write_capture(output_dir / "binary-readelf-notes.log", notes)
    if notes["returncode"] != 0 or notes["timed_out"]:
        raise GateError("readelf -n failed")
    match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", notes["stdout"])
    if not match:
        raise GateError("binary has no ELF build ID")
    dynamic = capture_command([readelf, "-d", str(binary)], env)
    write_capture(output_dir / "binary-readelf-dynamic.log", dynamic)
    if dynamic["returncode"] != 0 or dynamic["timed_out"]:
        raise GateError("readelf -d failed")

    captures: dict[str, dict[str, Any]] = {}
    for label, target in {"binary": binary, **libraries}.items():
        capture = capture_command([ldd, str(target)], env)
        write_capture(output_dir / f"ldd-{label}.log", capture)
        if capture["returncode"] != 0 or capture["timed_out"]:
            raise GateError(f"ldd failed for {label}")
        combined = capture["stdout"] + capture["stderr"]
        if "not found" in combined:
            raise GateError(f"ldd reports a missing dependency for {label}")
        captures[label] = {
            "path": str(target),
            "output": combined,
            "resolved_dependencies": parse_ldd_resolutions(combined),
        }

    ort_expected = libraries["onnxruntime"].resolve()
    ort_resolutions = re.findall(
        r"libonnxruntime\.so[^ ]*\s+=>\s+(\S+)", captures["binary"]["output"]
    )
    if not ort_resolutions:
        raise GateError("binary ldd does not resolve libonnxruntime")
    if all(Path(path).resolve() != ort_expected for path in ort_resolutions):
        raise GateError(f"binary resolves an unintended ONNX Runtime: {ort_resolutions}")
    version_match = re.search(r"libonnxruntime\.so\.(\d+(?:\.\d+)+)$", ort_expected.name)
    cuda_dependencies = captures["ort_cuda_provider"]["resolved_dependencies"]
    for prefix in ("libcudart.so", "libcublas.so", "libcudnn.so"):
        if not any(soname.startswith(prefix) for soname in cuda_dependencies):
            raise GateError(f"ORT CUDA provider ldd does not resolve {prefix}")
    if "ort_tensorrt_provider" in captures:
        tensorrt_dependencies = captures["ort_tensorrt_provider"]["resolved_dependencies"]
        if not any(soname.startswith("libnvinfer.so") for soname in tensorrt_dependencies):
            raise GateError("ORT TensorRT provider ldd does not resolve libnvinfer.so")

    runtime_dependencies: dict[str, str] = {}
    for capture in captures.values():
        for soname, path in capture["resolved_dependencies"].items():
            if not soname.startswith(GPU_LIBRARY_PREFIXES):
                continue
            previous = runtime_dependencies.get(soname)
            if previous is not None and previous != path:
                raise GateError(
                    f"GPU dependency {soname} resolves inconsistently: {previous}, {path}"
                )
            runtime_dependencies[soname] = path
    return {
        "build_id": match.group(1).lower(),
        "onnxruntime_version": version_match.group(1) if version_match else None,
        "ldd": captures,
        "runtime_dependencies": runtime_dependencies,
    }


def parse_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        match = re.match(r"([^:]+):\s+(\d+)\s+kB", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    required = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
    if any(key not in values for key in required):
        raise GateError("/proc/meminfo is missing required fields")
    return values


def read_thermal_zone(directory: Path) -> dict[str, Any] | None:
    try:
        name = (directory / "type").read_text(encoding="utf-8").strip()
        raw = int((directory / "temp").read_text(encoding="utf-8").strip())
    except (OSError, TypeError, ValueError):
        return None
    return {"zone": directory.name, "name": name, "temperature_c": raw / 1000.0}


def thermal_state() -> list[dict[str, Any]]:
    zones = [
        zone
        for directory in sorted(Path("/sys/class/thermal").glob("thermal_zone*"))
        if (zone := read_thermal_zone(directory)) is not None
    ]
    if not zones:
        raise GateError("no readable Jetson thermal zones")
    return zones


def first_integer(patterns: tuple[str, ...], value_key: str) -> dict[str, Any] | None:
    for pattern in patterns:
        for raw_path in sorted(glob.glob(pattern)):
            path = Path(raw_path)
            try:
                value = int(path.read_text(encoding="utf-8").strip())
            except (OSError, ValueError):
                continue
            return {"path": str(path), value_key: value}
    return None


def first_frequency(patterns: tuple[str, ...]) -> dict[str, Any] | None:
    return first_integer(patterns, "hz")


GPU_FREQUENCY_PATHS = (
    "/sys/class/devfreq/*gpu*/cur_freq",
    "/sys/devices/platform/*gpu*/devfreq/*/cur_freq",
)
EMC_FREQUENCY_PATHS = (
    "/sys/kernel/debug/bpmp/debug/clk/emc/rate",
    "/sys/class/devfreq/*emc*/cur_freq",
)
EMC_OVERRIDE_PATHS = (
    "/sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked",
)


def sysfs_cpu_frequencies() -> list[int]:
    frequencies: list[int] = []
    for path in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq")):
        try:
            frequencies.append(int(path.read_text(encoding="utf-8").strip()) * 1000)
        except (OSError, ValueError):
            continue
    return frequencies


def cpu_frequency_hz(raw: str) -> int:
    value = int(raw)
    return value * 1000 if value < 100_000_000 else value


def capture_system_state(
    env: dict[str, str], nvpmodel: Path, jetson_clocks: Path | None, output_dir: Path, label: str
) -> dict[str, Any]:
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    uptime_seconds = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    nvp = capture_command([str(nvpmodel), "-q"], env)
    write_capture(output_dir / f"nvpmodel-{label}.log", nvp)
    clocks = (
        capture_command([str(jetson_clocks), "--show"], env)
        if jetson_clocks is not None
        else {"command": [], "returncode": None, "timed_out": False, "stdout": "", "stderr": ""}
    )
    write_capture(output_dir / f"jetson-clocks-{label}.log", clocks)
    clock_text = clocks["stdout"] + clocks["stderr"]
    gpu_match = re.search(r"GPU .*?CurrentFreq=(\d+)", clock_text)
    emc_match = re.search(r"EMC .*?CurrentFreq=(\d+)", clock_text)
    emc_override_match = re.search(r"EMC .*?FreqOverride=(\d+)", clock_text)
    cpu_frequencies = [
        cpu_frequency_hz(value)
        for value in re.findall(r"^cpu\d+:.*?CurrentFreq=(\d+)", clock_text, re.MULTILINE)
    ] or sysfs_cpu_frequencies()
    gpu = (
        {"source": "jetson_clocks", "hz": int(gpu_match.group(1))}
        if gpu_match
        else first_frequency(GPU_FREQUENCY_PATHS)
    )
    emc = (
        {"source": "jetson_clocks", "hz": int(emc_match.group(1))}
        if emc_match
        else first_frequency(EMC_FREQUENCY_PATHS)
    )
    emc_override = (
        {"source": "jetson_clocks", "value": int(emc_override_match.group(1))}
        if emc_override_match
        else first_integer(EMC_OVERRIDE_PATHS, "value")
    )
    reset_paths = (
        "/sys/devices/platform/bus@0/c360000.pmc/reset_reason",
        "/sys/devices/platform/bus@0/c360000.pmc/reset_level",
    )
    reset: dict[str, str | None] = {}
    for raw in reset_paths:
        try:
            reset[Path(raw).name] = Path(raw).read_text(encoding="utf-8").strip()
        except OSError:
            reset[Path(raw).name] = None
    return {
        "captured_at": now(),
        "boot_id": boot_id,
        "uptime_seconds": uptime_seconds,
        "meminfo_kib": parse_meminfo(),
        "thermal_zones": thermal_state(),
        "nvpmodel": nvp,
        "jetson_clocks": clocks,
        "gpu_frequency": gpu,
        "emc_frequency": emc,
        "cpu_frequencies_hz": cpu_frequencies,
        "emc_override": emc_override,
        "reset": reset,
    }


def validate_system_gate(state: dict[str, Any], args: argparse.Namespace) -> None:
    nvp = state["nvpmodel"]
    if nvp["returncode"] != 0 or nvp["timed_out"]:
        raise GateError("nvpmodel -q failed")
    mode = re.search(r"NV Power Mode:\s*([^\r\n]+)", nvp["stdout"] + nvp["stderr"])
    if not mode or mode.group(1).strip() != args.expected_nvpmodel:
        raise GateError(f"nvpmodel is not {args.expected_nvpmodel}")
    gpu = state.get("gpu_frequency")
    emc = state.get("emc_frequency")
    if not gpu or gpu["hz"] < args.min_gpu_hz:
        raise GateError(f"GPU clock is below {args.min_gpu_hz}Hz or unreadable: {gpu}")
    if not emc or emc["hz"] < args.min_emc_hz:
        raise GateError(f"EMC clock is below {args.min_emc_hz}Hz or unreadable: {emc}")
    if not state["cpu_frequencies_hz"] or min(state["cpu_frequencies_hz"]) < args.min_cpu_hz:
        raise GateError(
            f"CPU clock is below {args.min_cpu_hz}Hz or unreadable: "
            f"{state['cpu_frequencies_hz']}"
        )
    override = state.get("emc_override")
    override_value = None if override is None else override.get("value", override.get("hz"))
    if override_value != args.expected_emc_override:
        raise GateError(
            f"EMC override is not {args.expected_emc_override}: {override}"
        )
    meminfo = state["meminfo_kib"]
    if meminfo["MemAvailable"] < args.min_memory_available_mib * 1024:
        raise GateError("available memory is below the configured gate")
    swap_used = meminfo["SwapTotal"] - meminfo["SwapFree"]
    if swap_used > args.max_swap_used_mib * 1024:
        raise GateError(f"swap usage exceeds the configured gate: {swap_used}KiB")
    hottest = max(zone["temperature_c"] for zone in state["thermal_zones"])
    if hottest >= args.max_temperature_c:
        raise GateError(f"preflight temperature {hottest:.1f}C exceeds the gate")


def capture_kernel(env: dict[str, str], output: Path) -> dict[str, Any]:
    candidates: list[list[str]] = []
    if command := shutil.which("dmesg", path=env["PATH"]):
        candidates.append([command, "--color=never"])
    if command := shutil.which("journalctl", path=env["PATH"]):
        candidates.append([command, "-k", "--no-pager", "-o", "short-monotonic"])
    last: dict[str, Any] | None = None
    for command in candidates:
        capture = capture_command(command, env)
        last = capture
        if capture["returncode"] == 0 and not capture["timed_out"]:
            write_text_fsynced(output, capture["stdout"])
            return {"available": True, "command": command}
    reason = last["stderr"] if last else "no dmesg or journalctl in PATH"
    write_text_fsynced(output, f"unavailable: {reason.strip()}\n")
    return {"available": False, "reason": reason.strip()}


class Heartbeat(threading.Thread):
    def __init__(self, path: Path, process: subprocess.Popen[Any], stop: threading.Event):
        super().__init__(name="benchmark-heartbeat", daemon=True)
        self.log = LineLog(path)
        self.process = process
        self.stop = stop

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
                    encoding="utf-8"
                ).strip()
                uptime = Path("/proc/uptime").read_text(encoding="utf-8").split()[0]
                self.log.write(
                    f"ts={now()} monotonic_ns={time.monotonic_ns()} boot_id={boot_id} "
                    f"uptime_seconds={uptime} workload_pid={self.process.pid} "
                    f"returncode={self.process.poll()}"
                )
            except OSError as error:
                self.log.write(f"ts={now()} heartbeat_error={error}")
            self.stop.wait(1.0)
        self.log.close()


class MapsSampler(threading.Thread):
    def __init__(self, path: Path, process: subprocess.Popen[Any], stop: threading.Event):
        super().__init__(name="benchmark-maps", daemon=True)
        self.log = LineLog(path)
        self.process = process
        self.stop = stop
        self.realpaths: set[str] = set()

    def run(self) -> None:
        maps = Path(f"/proc/{self.process.pid}/maps")
        seen: set[str] = set()
        while not self.stop.is_set():
            try:
                lines = maps.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                if self.process.poll() is not None:
                    break
                self.stop.wait(0.05)
                continue
            for line in lines:
                fields = line.split(maxsplit=5)
                if len(fields) < 6 or not fields[5].startswith("/"):
                    continue
                mapped = fields[5].replace("\\040", " ")
                try:
                    realpath = str(Path(mapped).resolve(strict=True))
                except OSError:
                    realpath = mapped
                if realpath in seen:
                    continue
                seen.add(realpath)
                self.realpaths.add(realpath)
                self.log.write(
                    f"monotonic_ns={time.monotonic_ns()} path={json.dumps(mapped)} "
                    f"realpath={json.dumps(realpath)}"
                )
            self.stop.wait(0.05)
        self.log.close()


@dataclass
class WorkloadOutcome:
    returncode: int | None
    timed_out: bool
    elapsed_seconds: float
    term_sent: bool
    kill_sent: bool
    mapped_realpaths: list[str]


def run_workload(
    command: list[str],
    env: dict[str, str],
    args: argparse.Namespace,
    output_dir: Path,
    cleanup: LineLog,
) -> WorkloadOutcome:
    stdout = (output_dir / "stdout.log").open("xb", buffering=0)
    stderr = (output_dir / "stderr.log").open("xb", buffering=0)
    tegra_log = (output_dir / "tegrastats.log").open("xb", buffering=0)
    tegra_command = [str(args.stdbuf), "-oL", "-eL", str(args.tegrastats), "--interval", "200"]
    tegra: subprocess.Popen[Any] | None = None
    workload: subprocess.Popen[Any] | None = None
    heartbeat: Heartbeat | None = None
    maps: MapsSampler | None = None
    stop = threading.Event()
    started = time.monotonic()
    timed_out = False
    termination = {"term_sent": False, "kill_sent": False}
    try:
        tegra = subprocess.Popen(
            tegra_command,
            stdin=subprocess.DEVNULL,
            stdout=tegra_log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        cleanup.write(f"{now()} child=tegrastats pid={tegra.pid} action=spawn")
        workload = subprocess.Popen(
            command,
            cwd=args.working_directory,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=env,
            start_new_session=True,
        )
        cleanup.write(f"{now()} child=workload pid={workload.pid} action=spawn")
        heartbeat = Heartbeat(output_dir / "heartbeat.log", workload, stop)
        maps = MapsSampler(output_dir / "process-maps.log", workload, stop)
        heartbeat.start()
        maps.start()
        try:
            workload.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            termination = terminate_group(
                workload, args.kill_grace_seconds, cleanup, "workload_timeout"
            )
    finally:
        stop.set()
        if heartbeat is not None:
            heartbeat.join(timeout=2.0)
        if maps is not None:
            maps.join(timeout=2.0)
        if workload is not None and group_exists(workload.pid):
            extra = terminate_group(workload, args.kill_grace_seconds, cleanup, "workload_cleanup")
            termination = {key: termination[key] or extra[key] for key in termination}
        if tegra is not None:
            terminate_group(tegra, 1.0, cleanup, "telemetry_cleanup")
        for handle in (stdout, stderr, tegra_log):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
    if workload is None:
        raise GateError("workload did not start")
    return WorkloadOutcome(
        returncode=workload.returncode,
        timed_out=timed_out,
        elapsed_seconds=time.monotonic() - started,
        term_sent=termination["term_sent"],
        kill_sent=termination["kill_sent"],
        mapped_realpaths=sorted(maps.realpaths) if maps is not None else [],
    )


def build_environment(args: argparse.Namespace, models: dict[str, Path]) -> dict[str, str]:
    home = args.home.resolve(strict=True)
    if not home.is_dir():
        raise GateError("--home must be a directory")
    for key, value in (("PATH", args.path), ("LD_LIBRARY_PATH", args.ld_library_path)):
        for entry in value.split(":"):
            if not entry or not Path(entry).is_absolute() or not Path(entry).is_dir():
                raise GateError(f"{key} contains a missing or non-absolute directory: {entry!r}")
    kiko: dict[str, str] = {}
    for assignment in args.kiko_env:
        if "=" not in assignment:
            raise GateError(f"invalid --kiko-env assignment: {assignment!r}")
        key, value = assignment.split("=", 1)
        if not re.fullmatch(r"KIKO_[A-Z0-9_]+", key) or "\x00" in value:
            raise GateError(f"invalid KIKO environment entry: {assignment!r}")
        if key in SELECTION_ENV_KEYS:
            raise GateError(f"{key} is forbidden; express dataset selection on the command line")
        if key in kiko:
            raise GateError(f"duplicate KIKO environment entry: {key}")
        kiko[key] = value
    backend = "cuda" if args.provider == "cuda" else "tensorrt"
    required = {
        "KIKO_BACKEND": backend,
        "KIKO_SUPERPOINT_BACKEND": backend,
        "KIKO_LIGHTGLUE_BACKEND": backend,
        "KIKO_ALLOW_BACKEND_FALLBACK": "false",
        "KIKO_DOWNSCALE": "1",
        "KIKO_MAX_KEYPOINTS": "2048",
        "KIKO_ORT_INTRA_THREADS": "1",
        "KIKO_ORT_INTER_THREADS": "1",
        "KIKO_ORT_DISABLE_CPU_EP_FALLBACK": "false",
        "KIKO_ORT_OPT_LEVEL": "level3",
        "KIKO_ORT_MEM_PATTERN": "true",
        "KIKO_ORT_PARALLEL_EXEC": "false",
        "KIKO_ORT_LOG_LEVEL": "verbose",
        "KIKO_ORT_LOG_VERBOSITY": "1",
        "KIKO_ORT_RUN_LOG_LEVEL": "warning",
        "KIKO_ORT_RUN_LOG_VERBOSITY": "0",
        "KIKO_TRACKING_MATCHER": "projected",
        "KIKO_PROJECTED_MATCH_RADIUS_PX": "32",
        "KIKO_PROJECTED_MATCH_MIN_DOT_PRODUCT": "0.45",
        "KIKO_PROJECTED_MATCH_MIN_MATCHES": "32",
        "KIKO_PROJECTED_MATCH_MIN_INLIERS": "24",
        "KIKO_SUPERPOINT_MODEL": str(models["superpoint_model"]),
        "KIKO_LIGHTGLUE_MODEL": str(models["lightglue_model"]),
    }
    if args.provider == "cuda":
        required.update(
            {
                "KIKO_CUDA_CONV_SEARCH": "heuristic",
                "KIKO_CUDA_PREFER_NHWC": "false",
                "KIKO_CUDA_FUSE_CONV_BIAS": "false",
                "KIKO_CUDA_GRAPH": "false",
            }
        )
    for key, value in required.items():
        if key in kiko and kiko[key].lower() != value.lower():
            raise GateError(f"--kiko-env conflicts with required {key}={value}")
        kiko[key] = value
    environment = {
        "HOME": str(home),
        "PATH": args.path,
        "LD_LIBRARY_PATH": args.ld_library_path,
        **kiko,
    }
    if args.ld_debug:
        environment["LD_DEBUG"] = "libs"
    if "ORT_DYLIB_PATH" in environment:
        raise GateError("ORT_DYLIB_PATH is forbidden")
    return environment


FORBIDDEN_COMMAND_FLAGS = (
    "--backend",
    "--sp-backend",
    "--superpoint-backend",
    "--lg-backend",
    "--lightglue-backend",
    "--sp-model",
    "--superpoint-model",
    "--lg-model",
    "--lightglue-model",
    "--pipeline",
    "--pipeline-model",
)

SELECTION_ENV_KEYS = {
    "KIKO_MAX_PAIRS",
    "KIKO_SKIP_FRAMES",
    "KIKO_BENCH_WARMUP_PAIRS",
    "KIKO_SLAM_WARMUP_PAIRS",
}


def _integer_option(values: list[str], name: str) -> int | None:
    found: list[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value == name:
            if index + 1 >= len(values):
                raise GateError(f"{name} requires a value")
            found.append(values[index + 1])
            index += 2
            continue
        if value.startswith(name + "="):
            found.append(value.split("=", 1)[1])
        index += 1
    if len(found) > 1:
        raise GateError(f"{name} must be specified at most once")
    if not found:
        return None
    if not found[0].isdigit():
        raise GateError(f"{name} must be a non-negative integer")
    return int(found[0])


def validate_command_selection(
    values: list[str], dataset_pairs: int, expected_items: int
) -> dict[str, int | None]:
    skip_frames = _integer_option(values, "--skip-frames") or 0
    max_pairs = _integer_option(values, "--max-pairs")
    warmup_pairs = _integer_option(values, "--warmup-pairs")
    if warmup_pairs is None:
        raise GateError("workload must specify --warmup-pairs explicitly")
    if skip_frames > dataset_pairs:
        raise GateError("--skip-frames exceeds the validated dataset pair count")
    remaining = dataset_pairs - skip_frames
    selected = min(remaining, max_pairs) if max_pairs is not None else remaining
    if selected != expected_items:
        raise GateError(
            f"workload selection is {selected} pairs but "
            f"--expected-command-items is {expected_items}"
        )
    if warmup_pairs >= selected:
        raise GateError("--warmup-pairs must leave at least one steady-state pair")
    return {
        "skip_frames": skip_frames,
        "max_pairs": max_pairs,
        "selected_pairs": selected,
        "warmup_pairs": warmup_pairs,
        "steady_pairs": selected - warmup_pairs,
    }


def workload_command(binary: Path, dataset: Path, raw_args: list[str]) -> list[str]:
    values = raw_args[1:] if raw_args[:1] == ["--"] else raw_args
    if not values:
        raise GateError("workload arguments are required after --")
    if values[0] not in {"bench", "run", "slam"}:
        raise GateError("workload must use the bench, run, or slam subcommand")
    for value in values:
        if any(value == flag or value.startswith(flag + "=") for flag in FORBIDDEN_COMMAND_FLAGS):
            raise GateError(f"workload argument may override fail-closed configuration: {value}")
    dataset_matches = 0
    for value in values:
        try:
            if Path(value).is_absolute() and Path(value).resolve(strict=True) == dataset:
                dataset_matches += 1
        except OSError:
            continue
    if dataset_matches != 1:
        raise GateError("workload arguments must contain the exact dataset path once")
    return [str(binary), *values]


def validate_device_nodes(paths: list[Path]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for path in paths:
        try:
            metadata = path.stat()
        except OSError as error:
            raise GateError(f"GPU device node is unavailable: {path}: {error}") from error
        if not stat.S_ISCHR(metadata.st_mode):
            raise GateError(f"GPU device path is not a character device: {path}")
        if not os.access(path, os.R_OK | os.W_OK):
            raise GateError(f"GPU device node is not readable and writable: {path}")
        result.append(
            {
                "path": str(path),
                "mode": oct(stat.S_IMODE(metadata.st_mode)),
                "uid": metadata.st_uid,
                "gid": metadata.st_gid,
            }
        )
    return result


def benchmark_process_kind(argv: list[str]) -> str | None:
    if not argv:
        return None
    executable = Path(argv[0]).name
    if executable in {"kiko-slam", "kiko_slam", "tegrastats"}:
        return executable
    if executable.startswith("python") and len(argv) > 1:
        if Path(argv[1]).name == "jetson_benchmark.py":
            return "jetson_benchmark.py"
    return None


def active_benchmark_processes() -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    own_pid = os.getpid()
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit() or int(process_dir.name) == own_pid:
            continue
        try:
            raw = (process_dir / "cmdline").read_bytes()
        except OSError:
            continue
        argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
        if kind := benchmark_process_kind(argv):
            active.append({"pid": int(process_dir.name), "kind": kind, "argv": argv})
    return sorted(active, key=lambda value: value["pid"])


def verify_run_evidence(
    args: argparse.Namespace,
    output_dir: Path,
    expected_models: dict[str, Path],
    expected_libraries: dict[str, Path],
    elf: dict[str, Any],
    selection: dict[str, int | None],
    outcome: WorkloadOutcome,
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if outcome.timed_out:
        failures.append("workload_timeout")
    if outcome.returncode != 0:
        failures.append(f"workload_returncode_{outcome.returncode}")
    if before["boot_id"] != after["boot_id"]:
        failures.append("boot_id_changed")
    if before.get("reset") != after.get("reset"):
        failures.append("reset_state_changed")
    try:
        validate_system_gate(after, args)
    except GateError:
        failures.append("post_system_gate_failed")

    stderr = (output_dir / "stderr.log").read_text(encoding="utf-8", errors="replace")
    stdout = (output_dir / "stdout.log").read_text(encoding="utf-8", errors="replace")
    combined = stderr + "\n" + stdout
    policies = parse_session_policies(combined)
    placements = parse_node_placements(combined)
    wanted_backend = args.provider.lower()
    expected_model_paths = {str(path) for path in expected_models.values()}
    failures.extend(
        session_evidence_failures(policies, wanted_backend, expected_model_paths)
    )
    failures.extend(
        placement_evidence_failures(
            placements, wanted_backend, args.max_cpu_node_fraction
        )
    )

    loaded = set(outcome.mapped_realpaths)
    if args.ld_debug:
        loaded.update(
            str(Path(path).resolve())
            for path in re.findall(r"calling init:\s+(/\S+)", stderr)
            if Path(path).exists()
        )
    expected_runtime_paths = {str(path) for path in expected_libraries.values()}
    expected_runtime_paths.update(elf.get("runtime_dependencies", {}).values())
    failures.extend(
        runtime_library_evidence_failures(
            loaded, expected_runtime_paths, args.provider.lower()
        )
    )

    tegra = parse_tegrastats(
        (output_dir / "tegrastats.log").read_text(encoding="utf-8", errors="replace")
    )
    if tegra["gr3d_utilization_pct"]["samples"] == 0:
        failures.append("missing_gr3d_samples")
    elif tegra["gr3d_nonzero_samples"] == 0:
        failures.append("no_gr3d_activity")
    if (
        tegra["swap_used_mb"]["max"] is not None
        and tegra["swap_used_mb"]["max"] > args.max_swap_used_mib
    ):
        failures.append("swap_gate_exceeded")
    if (
        tegra["temperature_c"]["max"] is None
        or tegra["temperature_c"]["max"] >= args.max_temperature_c
    ):
        failures.append("temperature_gate_failed")
    for rail in args.require_power_rail:
        if rail not in tegra["power_rails"]:
            failures.append(f"missing_power_rail_{rail}")

    counts = parse_command_counts(combined)
    failures.extend(
        count_evidence_failures(
            counts, args.expected_command_items, int(selection["warmup_pairs"])
        )
    )
    reported_metrics = parse_reported_metrics(combined, counts)
    if counts and reported_metrics["status"] != "valid":
        failures.append(f"{counts.get('kind')}_metrics_not_valid")
    if counts and counts.get("kind") == "slam":
        pose_outcomes = parse_pose_outcomes(combined)
        failures.extend(pose_outcome_failures(pose_outcomes, counts.get("steady_processed")))
    if args.min_steady_fps is not None:
        steady = reported_metrics.get("steady")
        if steady is None:
            failures.append("missing_steady_fps")
        elif steady["fps"] < args.min_steady_fps:
            failures.append("steady_fps_below_gate")

    kernel_pre = (output_dir / "kernel-pre.log").read_text(
        encoding="utf-8", errors="replace"
    )
    kernel_post = (output_dir / "kernel-post.log").read_text(
        encoding="utf-8", errors="replace"
    )
    kernel_lines = new_kernel_lines(kernel_pre, kernel_post)
    if not kernel_pre or kernel_pre.startswith("unavailable:"):
        failures.append("missing_kernel_pre_evidence")
    if not kernel_post or kernel_post.startswith("unavailable:"):
        failures.append("missing_kernel_post_evidence")
    for category, lines in classify_kernel_lines(kernel_lines).items():
        if lines:
            failures.append(f"kernel_{category}_evidence")
    if not (output_dir / "heartbeat.log").read_text(
        encoding="utf-8", errors="replace"
    ).strip():
        failures.append("missing_heartbeat_evidence")
    return sorted(set(failures))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", required=True)
    result.add_argument(
        "--output-root", type=Path, default=Path("/home/makerspace/kiko-benchmarks")
    )
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--binary", type=Path, required=True)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--superpoint-model", type=Path, required=True)
    result.add_argument("--lightglue-model", type=Path, required=True)
    result.add_argument("--ort-library", type=Path, required=True)
    result.add_argument("--ort-shared-provider", type=Path, required=True)
    result.add_argument("--ort-cuda-provider", type=Path, required=True)
    result.add_argument("--ort-tensorrt-provider", type=Path)
    result.add_argument("--sha256-manifest", type=Path, required=True)
    result.add_argument("--provider", choices=("cuda", "tensorrt"), required=True)
    result.add_argument("--expected-pairs", type=int, required=True)
    result.add_argument("--expected-command-items", type=int, required=True)
    result.add_argument("--working-directory", type=Path, required=True)
    result.add_argument("--home", type=Path, default=Path("/home/makerspace"))
    result.add_argument("--path", default=DEFAULT_PATH)
    result.add_argument("--ld-library-path", required=True)
    result.add_argument("--kiko-env", action="append", default=[])
    result.add_argument("--ld-debug", action="store_true")
    result.add_argument("--timeout-seconds", type=float, required=True)
    result.add_argument("--kill-grace-seconds", type=float, default=5.0)
    result.add_argument("--expected-nvpmodel", default="MAXN_SUPER")
    result.add_argument("--min-cpu-hz", type=int, default=1_728_000_000)
    result.add_argument("--min-gpu-hz", type=int, default=1_020_000_000)
    result.add_argument("--min-emc-hz", type=int, default=3_199_000_000)
    result.add_argument("--expected-emc-override", type=int, choices=(0, 1), default=1)
    result.add_argument("--min-memory-available-mib", type=int, default=4096)
    result.add_argument("--max-swap-used-mib", type=int, default=0)
    result.add_argument("--max-cpu-node-fraction", type=float, default=0.0)
    result.add_argument("--max-temperature-c", type=float, default=85.0)
    result.add_argument("--min-steady-fps", type=float)
    result.add_argument("--require-power-rail", action="append", default=[])
    result.add_argument(
        "--device-node",
        action="append",
        type=Path,
        default=[Path("/dev/nvmap"), Path("/dev/nvhost-gpu")],
    )
    result.add_argument("--nvpmodel", type=Path)
    result.add_argument("--jetson-clocks", type=Path)
    result.add_argument("--tegrastats", type=Path)
    result.add_argument("--stdbuf", type=Path)
    result.add_argument("command_args", nargs=argparse.REMAINDER)
    return result


def main() -> int:
    args = parser().parse_args()
    if not args.require_power_rail:
        args.require_power_rail = ["VDD_IN"]
    output_dir = create_output_dir(args.output_root, args.stage, args.output_dir)
    journal = ResultJournal(output_dir / "result.json", output_dir)
    cleanup = LineLog(output_dir / "cleanup.log")
    prior_handlers: dict[int, Any] = {}

    def on_signal(signum: int, _frame: Any) -> None:
        raise RunInterrupted(signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        prior_handlers[signum] = signal.signal(signum, on_signal)

    try:
        if args.expected_pairs <= 0 or args.expected_command_items <= 0:
            raise GateError("expected counts must be positive")
        if args.expected_command_items > args.expected_pairs:
            raise GateError("expected command items cannot exceed dataset pairs")
        if args.timeout_seconds <= 0 or args.kill_grace_seconds <= 0:
            raise GateError("timeout and kill grace must be positive")
        if args.min_steady_fps is not None and args.min_steady_fps <= 0:
            raise GateError("--min-steady-fps must be positive")
        if any(
            value <= 0
            for value in (
                args.min_cpu_hz,
                args.min_gpu_hz,
                args.min_emc_hz,
                args.min_memory_available_mib,
            )
        ):
            raise GateError("clock and available-memory gates must be positive")
        if args.max_swap_used_mib < 0:
            raise GateError("--max-swap-used-mib must be non-negative")
        if (
            not math.isfinite(args.max_cpu_node_fraction)
            or not 0.0 <= args.max_cpu_node_fraction <= 1.0
        ):
            raise GateError("--max-cpu-node-fraction must be finite and between 0 and 1")
        if not math.isfinite(args.max_temperature_c) or args.max_temperature_c <= 0:
            raise GateError("--max-temperature-c must be positive and finite")
        if args.min_steady_fps is not None and not math.isfinite(args.min_steady_fps):
            raise GateError("--min-steady-fps must be finite")
        active_processes = active_benchmark_processes()
        if active_processes:
            raise GateError(f"another benchmark workload is active: {active_processes}")

        binary = regular_file(args.binary, "binary", executable=True)
        dataset = args.dataset.resolve(strict=True)
        models = {
            "superpoint_model": regular_file(args.superpoint_model, "SuperPoint model"),
            "lightglue_model": regular_file(args.lightglue_model, "LightGlue model"),
        }
        libraries = {
            "onnxruntime": regular_file(args.ort_library, "ONNX Runtime library"),
            "ort_shared_provider": regular_file(args.ort_shared_provider, "ORT shared provider"),
            "ort_cuda_provider": regular_file(args.ort_cuda_provider, "ORT CUDA provider"),
        }
        if args.provider == "tensorrt":
            if args.ort_tensorrt_provider is None:
                raise GateError("TensorRT provider path is required for --provider=tensorrt")
            libraries["ort_tensorrt_provider"] = regular_file(
                args.ort_tensorrt_provider, "ORT TensorRT provider"
            )
        elif args.ort_tensorrt_provider is not None:
            raise GateError("CUDA validation must not provide or initialize TensorRT")
        args.working_directory = args.working_directory.resolve(strict=True)
        if not args.working_directory.is_dir():
            raise GateError("working directory is not a directory")

        args.nvpmodel = regular_file(
            args.nvpmodel or Path(shutil.which("nvpmodel", path=args.path) or "/usr/sbin/nvpmodel"),
            "nvpmodel",
            executable=True,
        )
        jetson_candidate = args.jetson_clocks or Path(
            shutil.which("jetson_clocks", path=args.path) or "/usr/bin/jetson_clocks"
        )
        args.jetson_clocks = regular_file(jetson_candidate, "jetson_clocks", executable=True)
        args.tegrastats = regular_file(
            args.tegrastats
            or Path(shutil.which("tegrastats", path=args.path) or "/usr/bin/tegrastats"),
            "tegrastats",
            executable=True,
        )
        args.stdbuf = regular_file(
            args.stdbuf or Path(shutil.which("stdbuf", path=args.path) or "/usr/bin/stdbuf"),
            "stdbuf",
            executable=True,
        )

        environment = build_environment(args, models)
        command = workload_command(binary, dataset, args.command_args)
        selection = validate_command_selection(
            command[1:], args.expected_pairs, args.expected_command_items
        )
        command_manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": now(),
            "stage": args.stage,
            "provider": args.provider,
            "command": command,
            "working_directory": str(args.working_directory),
            "environment": environment,
            "expected_dataset_pairs": args.expected_pairs,
            "expected_command_items": args.expected_command_items,
            "selection": selection,
            "timeout_seconds": args.timeout_seconds,
            "kill_grace_seconds": args.kill_grace_seconds,
            "thresholds": {
                "expected_nvpmodel": args.expected_nvpmodel,
                "min_cpu_hz": args.min_cpu_hz,
                "min_gpu_hz": args.min_gpu_hz,
                "min_emc_hz": args.min_emc_hz,
                "expected_emc_override": args.expected_emc_override,
                "min_memory_available_mib": args.min_memory_available_mib,
                "max_swap_used_mib": args.max_swap_used_mib,
                "max_cpu_node_fraction": args.max_cpu_node_fraction,
                "max_temperature_c": args.max_temperature_c,
                "min_steady_fps": args.min_steady_fps,
                "required_power_rails": args.require_power_rail,
            },
        }
        atomic_json(output_dir / "command.json", command_manifest)
        journal.update(state="preflight")

        hash_manifest = regular_file(args.sha256_manifest, "SHA256 manifest")
        artifacts = verify_artifacts(
            {"binary": binary, **models, **libraries},
            hash_manifest,
        )
        dataset_info = validate_dataset(dataset, args.expected_pairs)
        devices = validate_device_nodes(args.device_node)
        elf = preflight_elf(binary, libraries, environment, output_dir)
        kernel_pre = capture_kernel(environment, output_dir / "kernel-pre.log")
        system_pre = capture_system_state(
            environment, args.nvpmodel, args.jetson_clocks, output_dir, "pre"
        )
        atomic_json(output_dir / "system-pre.json", system_pre)
        validate_system_gate(system_pre, args)
        preflight = {
            "artifacts": artifacts,
            "dataset": dataset_info,
            "device_nodes": devices,
            "active_benchmark_processes": active_processes,
            "elf": elf,
            "kernel": kernel_pre,
        }
        command_manifest["identity"] = {
            "artifacts": artifacts,
            "dataset": dataset_info,
            "elf_build_id": elf["build_id"],
            "sha256_manifest": {
                "path": str(hash_manifest),
                "sha256": sha256(hash_manifest),
            },
            "runner_sha256": sha256(Path(__file__).resolve()),
            "analyzer_sha256": sha256(
                Path(__file__).with_name("analyze_jetson_benchmark.py").resolve()
            ),
        }
        atomic_json(output_dir / "command.json", command_manifest)
        journal.update(state="running", preflight=preflight)

        outcome = run_workload(command, environment, args, output_dir, cleanup)
        journal.update(
            state="workload_exited",
            returncode=outcome.returncode,
            timed_out=outcome.timed_out,
            elapsed_seconds=outcome.elapsed_seconds,
            term_sent=outcome.term_sent,
            kill_sent=outcome.kill_sent,
            mapped_realpaths=outcome.mapped_realpaths,
        )
        kernel_post = capture_kernel(environment, output_dir / "kernel-post.log")
        system_post = capture_system_state(
            environment, args.nvpmodel, args.jetson_clocks, output_dir, "post"
        )
        atomic_json(output_dir / "system-post.json", system_post)
        failures = verify_run_evidence(
            args,
            output_dir,
            models,
            libraries,
            elf,
            selection,
            outcome,
            system_pre,
            system_post,
        )
        journal.update(
            state="verifying",
            **{"pass": False},
            failures=failures,
            kernel_post=kernel_post,
            finished_at=now(),
        )
        provisional_analysis = analyze_run(
            output_dir, accepted_result_states=frozenset({"verifying"})
        )
        failures = sorted(set(failures + provisional_analysis["failures"]))
        journal.update(
            state="completed" if not failures else "failed",
            **{"pass": not failures},
            failures=failures,
            finished_at=now(),
        )
        analysis = analyze_run(output_dir)
        atomic_json(output_dir / "analysis.json", analysis)
        return 0 if not failures and analysis["pass"] else 1
    except RunInterrupted as error:
        cleanup.write(f"{now()} runner_signal={signal.Signals(error.signum).name}")
        journal.update(
            state="interrupted",
            failures=[str(error)],
            signal=signal.Signals(error.signum).name,
            finished_at=now(),
        )
        return 128 + error.signum
    except BaseException as error:
        cleanup.write(f"{now()} runner_error={type(error).__name__}:{error}")
        journal.update(
            state="failed",
            failures=[f"{type(error).__name__}: {error}"],
            finished_at=now(),
        )
        return 1
    finally:
        cleanup.write(f"{now()} runner_cleanup=complete")
        cleanup.close()
        for signum, handler in prior_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
