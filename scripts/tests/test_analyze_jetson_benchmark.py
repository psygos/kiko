from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from analyze_jetson_benchmark import (  # noqa: E402
    analyze_run,
    classify_kernel_lines,
    diagnostic_total_failures,
    new_kernel_lines,
    parse_command_counts,
    parse_diagnostic_totals,
    parse_node_placements,
    parse_reported_metrics,
    parse_session_policies,
    parse_tegrastats,
    parse_triangulation_policy,
    placement_evidence_failures,
)


FIXTURES = Path(__file__).parent / "fixtures"


class TegrastatsTests(unittest.TestCase):
    def test_triangulation_policy_parser_requires_one_finite_policy(self) -> None:
        line = (
            "triangulation: min_disparity_px=1 max_depth_m=None "
            "max_vertical_disparity_px=1.5\n"
        )
        self.assertEqual(
            parse_triangulation_policy(line),
            {"kind": "finite", "max_vertical_disparity_px": 1.5},
        )
        self.assertEqual(
            parse_triangulation_policy(
                "triangulation: max_vertical_disparity_px=unbounded\n"
            ),
            {"kind": "unbounded"},
        )
        self.assertIsNone(parse_triangulation_policy(""))
        self.assertIsNone(parse_triangulation_policy(line + line))

    def test_diagnostic_totals_parser_requires_complete_unique_metrics(self) -> None:
        line = (
            "diagnostic totals: frames=2 steady_frames=1 final_map_keyframes=3 "
            "final_map_points=45 peak_map_points=50 features_detected_samples=2 "
            "features_detected_total=220 steady_features_detected_samples=1 "
            "steady_features_detected_total=120 features_matched_samples=2 "
            "features_matched_total=90 steady_features_matched_samples=1 "
            "steady_features_matched_total=50 pnp_tracked_observations_samples=2 "
            "pnp_tracked_observations_total=79 steady_pnp_tracked_observations_samples=1 "
            "steady_pnp_tracked_observations_total=44 "
            "pnp_projectable_tracked_observations_samples=2 "
            "pnp_projectable_tracked_observations_total=73 "
            "steady_pnp_projectable_tracked_observations_samples=1 "
            "steady_pnp_projectable_tracked_observations_total=41 "
            "pnp_accepted_inliers_samples=2 pnp_accepted_inliers_total=64 "
            "steady_pnp_accepted_inliers_samples=1 steady_pnp_accepted_inliers_total=36 "
            "triangulation_samples=2 triangulation_candidate_matches_total=120 "
            "triangulation_kept_total=80 triangulation_dropped_disparity_total=18 "
            "triangulation_dropped_epipolar_total=22 "
            "triangulation_dropped_depth_total=0 "
            "triangulation_dropped_numerical_total=0 "
            "triangulation_dropped_unrepresentable_total=0 "
            "triangulation_dropped_duplicate_total=0\n"
        )
        totals = parse_diagnostic_totals(line)
        self.assertIsNotNone(totals)
        self.assertEqual(totals["final_map_points"], 45)
        self.assertEqual(totals["steady_pnp_accepted_inliers_total"], 36)
        self.assertEqual(totals["triangulation_dropped_epipolar_total"], 22)
        invalid = dict(totals)
        invalid["triangulation_kept_total"] = 100
        self.assertIn(
            "triangulation_candidate_accounting_invalid",
            diagnostic_total_failures(
                invalid, {"processed": 2, "steady_processed": 1, "keyframes": 3}
            ),
        )
        invalid["triangulation_kept_total"] = 70
        self.assertIn(
            "triangulation_candidate_accounting_invalid",
            diagnostic_total_failures(
                invalid, {"processed": 2, "steady_processed": 1, "keyframes": 3}
            ),
        )
        zero_candidate = dict(totals)
        zero_candidate.update(
            {
                "frames": 1,
                "steady_frames": 0,
                "final_map_keyframes": 0,
                "features_detected_samples": 1,
                "steady_features_detected_samples": 0,
                "features_matched_samples": 1,
                "steady_features_matched_samples": 0,
                "triangulation_samples": 1,
                "triangulation_candidate_matches_total": 0,
                "triangulation_kept_total": 0,
                "triangulation_dropped_disparity_total": 0,
                "triangulation_dropped_epipolar_total": 0,
                "triangulation_dropped_depth_total": 0,
                "triangulation_dropped_numerical_total": 0,
                "triangulation_dropped_unrepresentable_total": 0,
                "triangulation_dropped_duplicate_total": 0,
            }
        )
        self.assertNotIn(
            "triangulation_candidate_accounting_invalid",
            diagnostic_total_failures(
                zero_candidate,
                {"processed": 1, "steady_processed": 0, "keyframes": 0},
            ),
        )
        self.assertIsNone(parse_diagnostic_totals("diagnostic totals: frames=2\n"))

    def test_parser_reports_gpu_memory_temperature_and_power(self) -> None:
        metrics = parse_tegrastats((FIXTURES / "tegrastats-normal.txt").read_text())
        self.assertEqual(metrics["gr3d_nonzero_samples"], 1)
        self.assertEqual(metrics["gr3d_utilization_pct"]["max"], 76.0)
        self.assertEqual(metrics["gr3d_frequency_mhz"]["max"], 1020.0)
        self.assertEqual(metrics["ram_used_mb"]["max"], 1077.0)
        self.assertEqual(metrics["swap_used_mb"]["max"], 0.0)
        self.assertEqual(metrics["temperature_c"]["max"], 49.2)
        self.assertEqual(metrics["power_rails"]["VDD_IN"]["current_mw"]["max"], 6591.0)

    def test_kernel_delta_classifies_each_fault_family(self) -> None:
        before = (FIXTURES / "kernel-pre.txt").read_text()
        after = (FIXTURES / "kernel-post-fault.txt").read_text()
        findings = classify_kernel_lines(new_kernel_lines(before, after))
        self.assertTrue(findings["xid"])
        self.assertTrue(findings["oom"])
        self.assertTrue(findings["thermal"])
        self.assertTrue(findings["power"])

    def test_session_policy_requires_structured_marker(self) -> None:
        policies = parse_session_policies(
            "ort session policy: model=/models/sp.onnx requested_backend=Cuda "
            "configured_primary_backend=Cuda configured_providers=[CUDA] "
            "strict_backend_registration=true ort_cpu_ep_fallback_disabled=false "
            "session_committed=true\n"
        )
        self.assertEqual(len(policies), 1)
        self.assertEqual(policies[0]["providers"], ["CUDA"])
        self.assertTrue(policies[0]["strict_backend_registration"])
        self.assertFalse(policies[0]["ort_cpu_ep_fallback_disabled"])
        self.assertTrue(policies[0]["session_committed"])

        coreml = parse_session_policies(
            "ort session policy: model=/models/sp.onnx "
            "requested_backend=CoreMLCpuAndGpu "
            "configured_primary_backend=CoreMLCpuAndGpu "
            "configured_providers=[CoreML(CPUAndGPU)] "
            "strict_backend_registration=true "
            "ort_cpu_ep_fallback_disabled=false session_committed=true\n"
        )
        self.assertEqual(coreml[0]["requested_backend"], "CoreMLCpuAndGpu")
        self.assertEqual(coreml[0]["providers"], ["CoreML(CPUAndGPU)"])

    def test_current_completion_and_invalid_metrics_status_are_distinct(self) -> None:
        text = (
            "inference session initialization: 812.50ms\n"
            "completion: expected_pairs=32 entries_attempted=32 read_samples=32 processed=32 "
            "warmup_processed=4 steady_processed=28\n"
            "errors: read=0 pairing=0 inference=0\n"
            "benchmark metrics status: invalid_partial\n"
            "selected-run pipeline wall fps (session initialization excluded): 20.00 "
            "(processed=32, elapsed=1.60s)\n"
            "steady inference-attempt wall fps (successful pairs only): 25.00 "
            "(samples=28, attempt_time=1.12s)\n"
            "steady model-pipeline timing fps: 27.00 "
            "(samples=28, model_pipeline_time=1.04s)\n"
            "steady latency ms (median/p95, samples=28): "
            "sp_left_call=3.00/4.00 sp_right_call=3.10/4.10 "
            "lightglue_call=8.00/9.00 fused_ort_invocation=0.00/0.00 "
            "total_success=14.00/16.00\n"
        )
        counts = parse_command_counts(text)
        self.assertEqual(counts["expected"], 32)
        self.assertEqual(counts["attempted"], 32)
        self.assertEqual(counts["steady_processed"], 28)
        metrics = parse_reported_metrics(text, counts)
        self.assertEqual(metrics["status"], "invalid_partial")
        self.assertEqual(metrics["inference_initialization_ms"], 812.5)
        self.assertEqual(metrics["total"]["fps"], 20.0)
        self.assertEqual(
            metrics["steady_stage_fps"],
            {"inference_attempt": 25.0, "model_pipeline": 27.0},
        )
        self.assertEqual(metrics["stage_latency"]["sp_left_call"]["median_ms"], 3.0)
        self.assertEqual(
            metrics["stage_latency"]["fused_ort_invocation"]["p95_ms"], 0.0
        )

    def test_node_placement_counts_cpu_separately_from_cuda(self) -> None:
        placement = parse_node_placements(
            "VerifyEachNodeIsAssignedToAnEp Number of nodes placed on "
            "CUDAExecutionProvider. 40\n"
            "VerifyEachNodeIsAssignedToAnEp Number of nodes placed on "
            "CPUExecutionProvider: 2\n"
        )
        self.assertEqual(placement["provider_node_counts"], {"CUDA": 40, "CPU": 2})

    def test_node_placement_parses_ort_124_sessions_and_bounds_cpu_fraction(self) -> None:
        placement = parse_node_placements(
            "VerifyEachNodeIsAssignedToAnEp Node placements\n"
            "VerifyEachNodeIsAssignedToAnEp Node(s) placed on "
            "[CPUExecutionProvider]. Number of nodes: 36\n"
            "VerifyEachNodeIsAssignedToAnEp Node(s) placed on "
            "[CUDAExecutionProvider]. Number of nodes: 144\n"
            "VerifyEachNodeIsAssignedToAnEp Node placements\n"
            "VerifyEachNodeIsAssignedToAnEp Node(s) placed on "
            "[CPUExecutionProvider]. Number of nodes: 1\n"
            "VerifyEachNodeIsAssignedToAnEp Node(s) placed on "
            "[CUDAExecutionProvider]. Number of nodes: 1208\n"
        )
        self.assertEqual(placement["provider_node_counts"], {"CPU": 37, "CUDA": 1352})
        self.assertEqual(len(placement["sessions"]), 2)
        self.assertAlmostEqual(placement["maximum_cpu_node_fraction"], 0.2)
        self.assertEqual(placement_evidence_failures(placement, "cuda", 0.2), [])
        self.assertIn(
            "cpu_node_fraction_exceeded",
            placement_evidence_failures(placement, "cuda", 0.19),
        )


class AnalyzeRunTests(unittest.TestCase):
    def write_json(self, root: Path, name: str, value: object) -> None:
        (root / name).write_text(json.dumps(value), encoding="utf-8")

    def test_clean_cuda_run_passes_and_boot_change_fails(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            artifacts: dict[str, dict[str, str]] = {}
            for name, filename in {
                "superpoint_model": "sp.onnx",
                "lightglue_model": "lg.onnx",
                "onnxruntime": "libonnxruntime.so.1",
                "ort_shared_provider": "libonnxruntime_providers_shared.so",
                "ort_cuda_provider": "libonnxruntime_providers_cuda.so",
            }.items():
                path = root / filename
                path.write_bytes(name.encode())
                artifacts[name] = {"path": str(path)}
            dependencies: dict[str, str] = {}
            for soname in ("libcudart.so.12", "libcublas.so.12", "libcudnn.so.9"):
                path = root / soname
                path.write_bytes(soname.encode())
                dependencies[soname] = str(path)
            self.write_json(
                root,
                "result.json",
                {
                    "schema_version": 2,
                    "state": "completed",
                    "returncode": 0,
                    "preflight": {
                        "artifacts": artifacts,
                        "elf": {"runtime_dependencies": dependencies},
                    },
                },
            )
            self.write_json(
                root,
                "command.json",
                {
                    "schema_version": 2,
                    "provider": "cuda",
                    "expected_command_items": 2,
                    "selection": {"warmup_pairs": 1},
                    "thresholds": {
                        "expected_nvpmodel": "MAXN_SUPER",
                        "min_cpu_hz": 1_728_000_000,
                        "min_gpu_hz": 1_020_000_000,
                        "min_emc_hz": 3_199_000_000,
                        "expected_emc_override": 1,
                        "min_memory_available_mib": 4096,
                        "max_swap_used_mib": 0,
                        "max_cpu_node_fraction": 0.0,
                        "max_temperature_c": 85.0,
                        "required_power_rails": ["VDD_IN"],
                        "min_steady_fps": 1.5,
                        "expected_triangulation_policy": {
                            "kind": "finite",
                            "max_vertical_disparity_px": 1.5,
                        },
                    },
                },
            )
            system = {
                "boot_id": "same",
                "reset": {"reset_reason": "0"},
                "nvpmodel": {
                    "stdout": "NV Power Mode: MAXN_SUPER\n",
                    "stderr": "",
                },
                "cpu_frequencies_hz": [1_728_000_000] * 6,
                "gpu_frequency": {"hz": 1_020_000_000},
                "emc_frequency": {"hz": 3_199_000_000},
                "emc_override": {"value": 1},
                "meminfo_kib": {
                    "MemAvailable": 5 * 1024 * 1024,
                    "SwapTotal": 0,
                    "SwapFree": 0,
                },
                "thermal_zones": [{"temperature_c": 50.0}],
            }
            self.write_json(root, "system-pre.json", system)
            self.write_json(root, "system-post.json", system)
            (root / "stdout.log").write_text("", encoding="utf-8")
            (root / "stderr.log").write_text(
                "triangulation: min_disparity_px=1 max_depth_m=None "
                "max_vertical_disparity_px=1.5\n"
                "ort session policy: "
                f"model={artifacts['superpoint_model']['path']} requested_backend=Cuda "
                "configured_primary_backend=Cuda configured_providers=[CUDA] "
                "strict_backend_registration=true "
                "ort_cpu_ep_fallback_disabled=false "
                "session_committed=true\n"
                "ort session policy: "
                f"model={artifacts['lightglue_model']['path']} requested_backend=Cuda "
                "configured_primary_backend=Cuda configured_providers=[CUDA] "
                "strict_backend_registration=true "
                "ort_cpu_ep_fallback_disabled=false "
                "session_committed=true\n"
                "[V:onnxruntime:, session_state.cc:1268 VerifyEachNodeIsAssignedToAnEp] "
                "All nodes placed on [CUDAExecutionProvider]. Number of nodes: 42\n"
                "slam metrics status: valid\n"
                "done: expected=2, entries_consumed=2, tracker_attempts=2, processed=2, "
                "elapsed=1.0s, fps=2.0, "
                "warmup_processed=1, steady_processed=1, steady_elapsed=0.5s, steady_fps=2.0, "
                "read_errors=0, tracker_errors=0, keyframes=1\n"
                "pose outcomes: total_current=2 total_predicted=0 total_stale=0 "
                "total_unavailable=0 steady_current=1 steady_predicted=0 "
                "steady_stale=0 steady_unavailable=0\n"
                "diagnostic totals: frames=2 steady_frames=1 final_map_keyframes=1 "
                "final_map_points=30 peak_map_points=30 features_detected_samples=2 "
                "features_detected_total=220 steady_features_detected_samples=1 "
                "steady_features_detected_total=120 features_matched_samples=2 "
                "features_matched_total=90 steady_features_matched_samples=1 "
                "steady_features_matched_total=50 pnp_tracked_observations_samples=1 "
                "pnp_tracked_observations_total=40 steady_pnp_tracked_observations_samples=1 "
                "steady_pnp_tracked_observations_total=40 "
                "pnp_projectable_tracked_observations_samples=1 "
                "pnp_projectable_tracked_observations_total=38 "
                "steady_pnp_projectable_tracked_observations_samples=1 "
                "steady_pnp_projectable_tracked_observations_total=38 "
                "pnp_accepted_inliers_samples=1 pnp_accepted_inliers_total=32 "
                "steady_pnp_accepted_inliers_samples=1 "
                "steady_pnp_accepted_inliers_total=32 "
                "triangulation_samples=1 triangulation_candidate_matches_total=50 "
                "triangulation_kept_total=30 triangulation_dropped_disparity_total=10 "
                "triangulation_dropped_epipolar_total=10 "
                "triangulation_dropped_depth_total=0 "
                "triangulation_dropped_numerical_total=0 "
                "triangulation_dropped_unrepresentable_total=0 "
                "triangulation_dropped_duplicate_total=0\n",
                encoding="utf-8",
            )
            expected_loaded = [
                value["path"]
                for value in artifacts.values()
                if "onnxruntime" in Path(value["path"]).name
            ] + list(dependencies.values())
            maps = "".join(
                f"monotonic_ns=1 path={json.dumps(path)} realpath={json.dumps(path)}\n"
                for path in expected_loaded
            )
            (root / "process-maps.log").write_text(maps, encoding="utf-8")
            (root / "tegrastats.log").write_text(
                (FIXTURES / "tegrastats-normal.txt").read_text(), encoding="utf-8"
            )
            kernel = (FIXTURES / "kernel-pre.txt").read_text()
            (root / "kernel-pre.log").write_text(kernel, encoding="utf-8")
            (root / "kernel-post.log").write_text(kernel, encoding="utf-8")

            clean = analyze_run(root)
            self.assertTrue(clean["pass"], clean["failures"])
            self.assertEqual(len(clean["loaded_gpu_libraries"]), 3)

            command = json.loads((root / "command.json").read_text(encoding="utf-8"))
            command["schema_version"] = 1
            self.write_json(root, "command.json", command)
            stale_command = analyze_run(root)
            self.assertFalse(stale_command["pass"])
            self.assertIn("command_schema_version_mismatch", stale_command["failures"])
            command["schema_version"] = 2
            self.write_json(root, "command.json", command)

            result = json.loads((root / "result.json").read_text(encoding="utf-8"))
            result["schema_version"] = 1
            self.write_json(root, "result.json", result)
            stale_result = analyze_run(root)
            self.assertFalse(stale_result["pass"])
            self.assertIn("result_schema_version_mismatch", stale_result["failures"])
            result["schema_version"] = 2
            self.write_json(root, "result.json", result)

            command["thresholds"]["min_steady_fps"] = 2.1
            self.write_json(root, "command.json", command)
            slow = analyze_run(root)
            self.assertFalse(slow["pass"])
            self.assertIn("steady_fps_below_gate", slow["failures"])
            command["thresholds"]["min_steady_fps"] = 1.5
            self.write_json(root, "command.json", command)

            expected_policy = command["thresholds"].pop(
                "expected_triangulation_policy"
            )
            self.write_json(root, "command.json", command)
            missing_policy = analyze_run(root)
            self.assertFalse(missing_policy["pass"])
            self.assertIn(
                "missing_expected_triangulation_policy", missing_policy["failures"]
            )
            command["thresholds"]["expected_triangulation_policy"] = expected_policy
            self.write_json(root, "command.json", command)

            stderr = (root / "stderr.log").read_text(encoding="utf-8")
            (root / "stderr.log").write_text(
                stderr.replace("max_vertical_disparity_px=1.5", "max_vertical_disparity_px=2"),
                encoding="utf-8",
            )
            policy_mismatch = analyze_run(root)
            self.assertFalse(policy_mismatch["pass"])
            self.assertIn("triangulation_policy_mismatch", policy_mismatch["failures"])
            (root / "stderr.log").write_text(stderr, encoding="utf-8")

            legacy_counts = stderr.replace(
                "entries_consumed=2, tracker_attempts=2, ", "attempted=2, "
            )
            (root / "stderr.log").write_text(legacy_counts, encoding="utf-8")
            ambiguous = analyze_run(root)
            self.assertFalse(ambiguous["pass"])
            self.assertIn("entries_consumed_count_mismatch", ambiguous["failures"])
            self.assertIn("tracker_attempt_count_mismatch", ambiguous["failures"])
            (root / "stderr.log").write_text(stderr, encoding="utf-8")

            (root / "stderr.log").write_text(
                stderr.replace("entries_consumed=2", "entries_consumed=3"),
                encoding="utf-8",
            )
            overrun = analyze_run(root)
            self.assertFalse(overrun["pass"])
            self.assertIn("entries_consumed_count_mismatch", overrun["failures"])
            (root / "stderr.log").write_text(stderr, encoding="utf-8")

            (root / "stderr.log").write_text(
                stderr.replace("steady_stale=0", "steady_stale=1"),
                encoding="utf-8",
            )
            stale = analyze_run(root)
            self.assertFalse(stale["pass"])
            self.assertIn("steady_stale_poses", stale["failures"])
            (root / "stderr.log").write_text(stderr, encoding="utf-8")

            (root / "stderr.log").write_text(
                stderr
                + "VerifyEachNodeIsAssignedToAnEp Number of nodes placed on "
                "CPUExecutionProvider: 1\n",
                encoding="utf-8",
            )
            cpu_fallback = analyze_run(root)
            self.assertFalse(cpu_fallback["pass"])
            self.assertIn("cpu_node_fraction_exceeded", cpu_fallback["failures"])
            (root / "stderr.log").write_text(stderr, encoding="utf-8")

            unexpected = root / "system" / "libonnxruntime.so.999"
            unexpected.parent.mkdir()
            unexpected.write_bytes(b"wrong")
            (root / "process-maps.log").write_text(
                maps
                + f"monotonic_ns=2 path={json.dumps(str(unexpected))} "
                f"realpath={json.dumps(str(unexpected))}\n",
                encoding="utf-8",
            )
            wrong_library = analyze_run(root)
            self.assertFalse(wrong_library["pass"])
            self.assertIn("unintended_ort_library_libonnxruntime.so.999", wrong_library["failures"])

            (root / "process-maps.log").write_text(maps, encoding="utf-8")
            tensorrt = root / "libnvinfer.so.10"
            tensorrt.write_bytes(b"unexpected")
            (root / "process-maps.log").write_text(
                maps
                + f"monotonic_ns=3 path={json.dumps(str(tensorrt))} "
                f"realpath={json.dumps(str(tensorrt))}\n",
                encoding="utf-8",
            )
            wrong_stack = analyze_run(root)
            self.assertFalse(wrong_stack["pass"])
            self.assertIn("unexpected_tensorrt_library_loaded", wrong_stack["failures"])

            (root / "process-maps.log").write_text(maps, encoding="utf-8")

            self.write_json(root, "system-post.json", {**system, "boot_id": "rebooted"})
            changed = analyze_run(root)
            self.assertFalse(changed["pass"])
            self.assertIn("boot_id_changed", changed["failures"])


if __name__ == "__main__":
    unittest.main()
