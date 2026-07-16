from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from jetson_benchmark import (  # noqa: E402
    GateError,
    benchmark_process_kind,
    build_environment,
    capture_command,
    cpu_frequency_hz,
    create_output_dir,
    load_hash_manifest,
    parse_ldd_resolutions,
    read_thermal_zone,
    validate_command_selection,
    validate_dataset,
    validate_system_gate,
    verify_artifacts,
    workload_command,
)


class OutputAndProcessTests(unittest.TestCase):
    def test_explicit_output_directory_uses_create_new_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "run"
            self.assertEqual(create_output_dir(Path(raw), "smoke", output), output)
            with self.assertRaises(FileExistsError):
                create_output_dir(Path(raw), "smoke", output)

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    def test_capture_timeout_terminates_owned_process_group(self) -> None:
        capture = capture_command(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            dict(os.environ),
            timeout=0.05,
        )
        self.assertTrue(capture["timed_out"])
        self.assertIsNotNone(capture["returncode"])
        self.assertLess(capture["elapsed_seconds"], 3.0)

    def test_process_classification_ignores_shell_command_text(self) -> None:
        self.assertEqual(benchmark_process_kind(["/usr/bin/tegrastats"]), "tegrastats")
        self.assertEqual(
            benchmark_process_kind(["python3", "/repo/scripts/jetson_benchmark.py"]),
            "jetson_benchmark.py",
        )
        self.assertIsNone(
            benchmark_process_kind(["bash", "-lc", "python3 scripts/jetson_benchmark.py"])
        )


class DatasetTests(unittest.TestCase):
    def make_dataset(self, root: Path, pairs: int = 2) -> None:
        frames = root / "frames"
        frames.mkdir(parents=True)
        entries = []
        for index in range(pairs):
            left = f"frames/{index * 2 + 1}_mono_left.raw"
            right = f"frames/{index * 2 + 2}_mono_right.raw"
            (root / left).write_bytes(bytes(640 * 480))
            (root / right).write_bytes(bytes(640 * 480))
            entries.append(
                {
                    "left": {"timestamp_ns": index * 2 + 1, "path": left},
                    "status": "paired",
                    "right": {"timestamp_ns": index * 2 + 2, "path": right},
                    "delta_ns": 1,
                }
            )
        manifest = {
            "header": {"format": "raw", "width": 640, "height": 480},
            "stats": {
                "total_left": pairs,
                "total_right": pairs,
                "paired_count": pairs,
                "left_orphans": 0,
                "right_orphans": 0,
                "drops_by_reason": {"write_fail": 0},
            },
            "entries": entries,
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "meta.json").write_text(
            json.dumps({"mono": {"width": 640, "height": 480}}), encoding="utf-8"
        )

    def test_full_manifest_and_raw_sizes_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            self.make_dataset(root)
            summary = validate_dataset(root, 2)
            self.assertEqual(summary["paired_count"], 2)
            manifest_path = root / "manifest.json"
            original_manifest = manifest_path.read_text(encoding="utf-8")
            manifest = json.loads(original_manifest)
            manifest["entries"][0] = None
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(GateError, "entry 0 must be an object"):
                validate_dataset(root, 2)
            manifest_path.write_text(original_manifest, encoding="utf-8")
            (root / "frames/1_mono_left.raw").write_bytes(b"short")
            with self.assertRaisesRegex(GateError, "307200 bytes"):
                validate_dataset(root, 2)


class ArtifactAndEnvironmentTests(unittest.TestCase):
    def test_transient_non_byte_thermal_read_is_skipped(self) -> None:
        with patch.object(
            Path,
            "read_text",
            side_effect=["cpu-thermal", TypeError("can't concat NoneType to bytes")],
        ):
            self.assertIsNone(read_thermal_zone(Path("/sys/class/thermal/thermal_zone0")))

    def test_jetson_power_gate_requires_cpu_gpu_emc_and_override(self) -> None:
        state = {
            "nvpmodel": {
                "returncode": 0,
                "timed_out": False,
                "stdout": "NV Power Mode: MAXN_SUPER\n",
                "stderr": "",
            },
            "gpu_frequency": {"hz": 1_020_000_000},
            "emc_frequency": {"hz": 3_199_000_000},
            "cpu_frequencies_hz": [1_728_000_000] * 6,
            "emc_override": {"value": 1},
            "meminfo_kib": {
                "MemAvailable": 5 * 1024 * 1024,
                "SwapTotal": 0,
                "SwapFree": 0,
            },
            "thermal_zones": [{"temperature_c": 50.0}],
        }
        args = Namespace(
            expected_nvpmodel="MAXN_SUPER",
            min_cpu_hz=1_728_000_000,
            min_gpu_hz=1_020_000_000,
            min_emc_hz=3_199_000_000,
            expected_emc_override=1,
            min_memory_available_mib=4096,
            max_swap_used_mib=0,
            max_temperature_c=85.0,
        )
        validate_system_gate(state, args)
        self.assertEqual(cpu_frequency_hz("1728000"), 1_728_000_000)
        self.assertEqual(cpu_frequency_hz("1728000000"), 1_728_000_000)
        state["emc_override"] = {"value": 0}
        with self.assertRaisesRegex(GateError, "EMC override"):
            validate_system_gate(state, args)

    def test_hash_manifest_must_cover_every_exact_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifact = root / "model.onnx"
            artifact.write_bytes(b"model")
            digest = hashlib.sha256(b"model").hexdigest()
            manifest = root / "SHA256SUMS"
            manifest.write_text(f"{digest}  model.onnx\n", encoding="utf-8")
            self.assertEqual(load_hash_manifest(manifest)[artifact], digest)
            verified = verify_artifacts({"model": artifact}, manifest)
            self.assertEqual(verified["model"]["sha256"], digest)

    def test_environment_is_sterile_and_configuration_cannot_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model_a = root / "sp.onnx"
            model_b = root / "lg.onnx"
            model_a.write_bytes(b"sp")
            model_b.write_bytes(b"lg")
            args = Namespace(
                home=root,
                path=str(root),
                ld_library_path=str(root),
                kiko_env=[],
                provider="cuda",
                ld_debug=False,
            )
            environment = build_environment(
                args, {"superpoint_model": model_a, "lightglue_model": model_b}
            )
            self.assertEqual(
                set(environment),
                {
                    "HOME",
                    "PATH",
                    "LD_LIBRARY_PATH",
                    "KIKO_BACKEND",
                    "KIKO_SUPERPOINT_BACKEND",
                    "KIKO_LIGHTGLUE_BACKEND",
                    "KIKO_ALLOW_BACKEND_FALLBACK",
                    "KIKO_DOWNSCALE",
                    "KIKO_MAX_KEYPOINTS",
                    "KIKO_ORT_INTRA_THREADS",
                    "KIKO_ORT_INTER_THREADS",
                    "KIKO_ORT_DISABLE_CPU_EP_FALLBACK",
                    "KIKO_ORT_OPT_LEVEL",
                    "KIKO_ORT_MEM_PATTERN",
                    "KIKO_ORT_PARALLEL_EXEC",
                    "KIKO_ORT_LOG_LEVEL",
                    "KIKO_ORT_LOG_VERBOSITY",
                    "KIKO_ORT_RUN_LOG_LEVEL",
                    "KIKO_ORT_RUN_LOG_VERBOSITY",
                    "KIKO_TRACKING_MATCHER",
                    "KIKO_PROJECTED_MATCH_RADIUS_PX",
                    "KIKO_PROJECTED_MATCH_MIN_DOT_PRODUCT",
                    "KIKO_PROJECTED_MATCH_MIN_MATCHES",
                    "KIKO_PROJECTED_MATCH_MIN_INLIERS",
                    "KIKO_CUDA_CONV_SEARCH",
                    "KIKO_CUDA_PREFER_NHWC",
                    "KIKO_CUDA_FUSE_CONV_BIAS",
                    "KIKO_CUDA_GRAPH",
                    "KIKO_SUPERPOINT_MODEL",
                    "KIKO_LIGHTGLUE_MODEL",
                },
            )
            expected_settings = {
                "KIKO_ALLOW_BACKEND_FALLBACK": "false",
                "KIKO_DOWNSCALE": "1",
                "KIKO_MAX_KEYPOINTS": "2048",
                "KIKO_ORT_INTRA_THREADS": "1",
                "KIKO_ORT_INTER_THREADS": "1",
                "KIKO_ORT_DISABLE_CPU_EP_FALLBACK": "false",
                "KIKO_ORT_OPT_LEVEL": "level3",
                "KIKO_ORT_MEM_PATTERN": "true",
                "KIKO_ORT_PARALLEL_EXEC": "false",
                "KIKO_ORT_RUN_LOG_LEVEL": "warning",
                "KIKO_ORT_RUN_LOG_VERBOSITY": "0",
                "KIKO_TRACKING_MATCHER": "projected",
                "KIKO_PROJECTED_MATCH_RADIUS_PX": "32",
                "KIKO_PROJECTED_MATCH_MIN_DOT_PRODUCT": "0.45",
                "KIKO_PROJECTED_MATCH_MIN_MATCHES": "32",
                "KIKO_PROJECTED_MATCH_MIN_INLIERS": "24",
                "KIKO_CUDA_CONV_SEARCH": "heuristic",
                "KIKO_CUDA_PREFER_NHWC": "false",
                "KIKO_CUDA_FUSE_CONV_BIAS": "false",
                "KIKO_CUDA_GRAPH": "false",
            }
            self.assertEqual(
                {key: environment[key] for key in expected_settings}, expected_settings
            )
            self.assertNotIn("ORT_DYLIB_PATH", environment)
            args.kiko_env = ["KIKO_ALLOW_BACKEND_FALLBACK=true"]
            with self.assertRaises(GateError):
                build_environment(
                    args, {"superpoint_model": model_a, "lightglue_model": model_b}
                )
            args.kiko_env = ["KIKO_MAX_PAIRS=1"]
            with self.assertRaisesRegex(GateError, "selection on the command line"):
                build_environment(
                    args, {"superpoint_model": model_a, "lightglue_model": model_b}
                )

    def test_ldd_resolutions_are_canonical_and_command_selection_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            library = root / "libcudart.so.12.6"
            library.write_bytes(b"cuda")
            link = root / "libcudart.so.12"
            link.symlink_to(library.name)
            resolved = parse_ldd_resolutions(
                f"libcudart.so.12 => {link} (0x000000000000)\n"
            )
            self.assertEqual(resolved, {"libcudart.so.12": str(library)})

        selection = validate_command_selection(
            ["run", "--max-pairs", "300", "--warmup-pairs=4", "/data"],
            2084,
            300,
        )
        self.assertEqual(selection["steady_pairs"], 296)
        with self.assertRaisesRegex(GateError, "selection is 300 pairs"):
            validate_command_selection(
                ["run", "--max-pairs=300", "--warmup-pairs", "4", "/data"],
                2084,
                299,
            )
        with self.assertRaisesRegex(GateError, "specify --warmup-pairs"):
            validate_command_selection(["run", "/data"], 2084, 2084)

    def test_workload_must_use_exact_dataset_and_no_backend_override(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            binary = root / "kiko-slam"
            dataset = root / "dataset"
            binary.write_text("", encoding="utf-8")
            dataset.mkdir()
            command = workload_command(binary, dataset, ["--", "bench", str(dataset)])
            self.assertEqual(command, [str(binary), "bench", str(dataset)])
            with self.assertRaises(GateError):
                workload_command(
                    binary, dataset, ["--", "bench", "--backend=cpu", str(dataset)]
                )
            with self.assertRaisesRegex(GateError, "bench, run, or slam"):
                workload_command(binary, dataset, ["--", "viz", str(dataset)])


if __name__ == "__main__":
    unittest.main()
