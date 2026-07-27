#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use kiko_device_inventory::{
    load_expected_manifest_v1_from_slice, load_expected_manifest_v2_from_slice,
};
use kiko_nano_bundle_renderer::{RenderMode, render_bundle};
use kiko_slam::navigation::{
    NanoAgentLaunchV3, NanoAgentPolicyConfigV3, NanoCalibrationArtifactV1,
    NanoWheelsOffNativeRuntimeV1, NanoWheelsOffQualificationLaunchV2,
    WheelsOffCandidateControllerBinding,
};
use robot_server::config::{ControllerServerConfig, ControllerServerConfigV1};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn write_source(root: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = root.join(name);
    fs::write(&path, bytes).expect("write fixture source");
    path
}

fn canonical_root(temporary: &TempDir) -> PathBuf {
    fs::canonicalize(temporary.path()).expect("canonical temporary root")
}

fn add_production_face_assets(root: &Path, input: &mut Value) {
    input["schema_version"] = json!(1);
    let frontal = write_source(root, "frontal.xml", b"frontal-cascade-exact");
    let profile = write_source(root, "profile.xml", b"profile-cascade-exact");
    input["assets"]["face_perception"] = json!({
        "frontal_face_cascade": {
            "source_path": frontal,
            "destination_relative_path":
                "models/opencv/haarcascade_frontalface_default.xml"
        },
        "profile_face_cascade": {
            "source_path": profile,
            "destination_relative_path":
                "models/opencv/haarcascade_profileface.xml"
        }
    });
}

fn source_fixture() -> (TempDir, Value) {
    let temporary = tempfile::tempdir().expect("temporary fixture root");
    let root = canonical_root(&temporary);
    let calibration_bytes = serde_json::to_vec(&json!({
        "schema_version": 1,
        "oak_mxid": "19443010F1B43A2E00",
        "imu_calibration_id": "imu-v1",
        "stereo_calibration_id": "stereo-v1",
        "tracking_camera_to_base_calibration_id": "camera-base-v1",
        "rectified_stereo": {
            "rectified": true,
            "left": {
                "fx_px": 400.0,
                "fy_px": 400.0,
                "cx_px": 320.0,
                "cy_px": 200.0,
                "width_px": 640,
                "height_px": 400
            },
            "right": {
                "fx_px": 400.0,
                "fy_px": 400.0,
                "cx_px": 320.0,
                "cy_px": 200.0,
                "width_px": 640,
                "height_px": 400
            },
            "baseline_m": 0.075
        },
        "raw_imu_calibration": {
            "format_version": 1,
            "source_id": "fixture",
            "content_id": "imu-v1",
            "gyro_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "gyro_bias_native_rad_per_sec": [0.0, 0.0, 0.0],
            "accel_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "accel_bias_native_m_per_sec2": [0.0, 0.0, 0.0],
            "native_imu_to_base_rotation": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]
        },
        "tracking_camera_to_base": {
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "translation_m": [0.20, 0.0, -0.25]
        }
    }))
    .expect("serialize calibration fixture");
    let calibration = write_source(&root, "calibration.json", &calibration_bytes);
    let plant = write_source(&root, "plant.json", br#"{"plant":"exact"}"#);
    let navigation = write_source(&root, "navigation.json", br#"{"navigation":"exact"}"#);
    let superpoint = write_source(&root, "superpoint.onnx", b"superpoint-exact");
    let lightglue = write_source(&root, "lightglue.onnx", b"lightglue-exact");
    let depthai = write_source(&root, "libdepthai-core.so", b"depthai-exact");
    let calibration_lib = write_source(&root, "libdynamic_calibration.so", b"calibration-exact");
    let libusb = write_source(&root, "libusb-1.0.so.0", b"libusb-exact");
    let onnx = write_source(&root, "libonnxruntime.so.1", b"onnxruntime-exact");
    let opencv_core = write_source(&root, "libopencv_core.so.4.5d", b"opencv-core-exact");
    let opencv_imgproc = write_source(&root, "libopencv_imgproc.so.4.5d", b"opencv-imgproc-exact");
    let opencv_objdetect = write_source(
        &root,
        "libopencv_objdetect.so.4.5d",
        b"opencv-objdetect-exact",
    );
    let qualification_executable = write_source(
        &root,
        "kiko-nano-wheels-off-qualification",
        b"qualification-executable-exact",
    );
    let input = json!({
        "schema_version": 2,
        "bundle": {
            "kind": "wheels_off_qualification",
            "qualification_executable_path": qualification_executable
        },
        "robot_id": "kiko-test",
        "discovery": {
            "oak": {
                "mxid": "19443010F1B43A2E00",
                "compiled_depthai_header_sdk_version": "2.30.0",
                "compiled_depthai_header_sdk_commit": "exact-commit",
                "compiled_depthai_header_embedded_device_artifact_version": "exact-device",
                "compiled_depthai_header_embedded_bootloader_artifact_version": "exact-bootloader"
            },
            "stm32": {
                "serial_by_id_path": "/dev/serial/by-id/usb-STMicroelectronics_STM32_066EFF-if00",
                "controller_uid_hex": "0102030405060708090a0b0c",
                "firmware_abi": 2,
                "firmware_build_id": 135169,
                "hardware_profile_fingerprint_hex": "4b494b4f2d3450574d2d43414e443121",
                "capabilities_bits": 575
            },
            "head": {
                "adapter_serial_by_id_path": "/dev/serial/by-id/usb-head-exact",
                "bow_servo_id": 1,
                "curl_servo_id": 2,
                "yaw_servo_id": 3,
                "roll_servo_id": 4,
                "baud_rate_bps": 1000000,
                "dtr_asserted": false,
                "rts_asserted": true
            },
            "eye": {
                "serial_by_id_path": "/dev/serial/by-id/usb-eye-exact",
                "kep_protocol_version": 2,
                "device_uid_hex": "0102030405060708090a0b0c0d0e0f10",
                "firmware_build_id_hex": "0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f20",
                "capabilities_bits": 255
            }
        },
        "assets": {
            "calibration": {
                "artifact_id": "assembly-calibration-v1",
                "source_path": calibration,
                "destination_relative_path": "artifacts/calibration/assembly-v1.json"
            },
            "plant": {
                "artifact_id": "plant-exact-v1",
                "source_path": plant,
                "destination_relative_path": "artifacts/plant/plant.json"
            },
            "navigation_shadow_source_path": navigation,
            "superpoint_model": {
                "source_path": superpoint,
                "destination_relative_path": "models/superpoint.onnx"
            },
            "lightglue_model": {
                "source_path": lightglue,
                "destination_relative_path": "models/lightglue.onnx"
            }
        },
        "native_libraries": [
            {
                "role": "depthai_core",
                "soname": "libdepthai-core.so",
                "source_path": depthai
            },
            {
                "role": "dynamic_calibration",
                "soname": "libdynamic_calibration.so",
                "source_path": calibration_lib
            },
            {
                "role": "libusb1_0",
                "soname": "libusb-1.0.so.0",
                "source_path": libusb
            },
            {
                "role": "onnxruntime",
                "soname": "libonnxruntime.so.1",
                "source_path": onnx
            },
            {
                "role": "opencv_core",
                "soname": "libopencv_core.so.4.5d",
                "source_path": opencv_core
            },
            {
                "role": "opencv_imgproc",
                "soname": "libopencv_imgproc.so.4.5d",
                "source_path": opencv_imgproc
            },
            {
                "role": "opencv_objdetect",
                "soname": "libopencv_objdetect.so.4.5d",
                "source_path": opencv_objdetect
            }
        ],
        "runtime": {
            "oak": {
                "rgb_width_px": 640,
                "rgb_height_px": 400,
                "rgb_fps": 15,
                "stereo_width_px": 640,
                "stereo_height_px": 400,
                "stereo_fps": 15,
                "imu_rate_hz": 200,
                "queue_size": 4
            },
            "occupancy": {
                "resolution_m": 0.05,
                "lower_x_m": -10.0,
                "lower_y_m": -10.0,
                "width_cells": 400,
                "height_cells": 400,
                "maximum_cells": 160000,
                "maximum_keyframes": 20000,
                "snapshot_every_keyframes": 20
            },
            "inference": {
                "superpoint_backend": "cpu",
                "lightglue_backend": "cpu",
                "downscale_factor": 2,
                "maximum_keypoints": 1024
            },
            "rerun": {
                "decimation": 2,
                "memory_limit_bytes": 268435456,
                "flush_timeout_ms": 2000
            },
            "storage": {
                "maximum_map_snapshot_bytes": 67108864,
                "minimum_free_bytes_after_map_save": 268435456,
                "maximum_navigation_dataset_bytes": 8589934592_u64,
                "maximum_navigation_dataset_files": 65536,
                "maximum_navigation_ingress_records": 100000,
                "minimum_free_bytes_after_navigation_dataset_write": 1073741824,
                "navigation_dataset_terminal_reserve_bytes": 268435456,
                "warm_start": { "kind": "none" }
            }
        },
        "head_policy": {
            "response_timeout_ms": 100,
            "write_timeout_ms": 100,
            "arming_freshness_ms": 250,
            "write_attempts": 2,
            "noise_budget_bytes": 32,
            "redundant_read_tolerance_ticks": 10,
            "readback_tolerance_ticks": 20,
            "final_target_tolerance_ticks": 20,
            "path_corridor_tolerance_ticks": 20,
            "direction_regression_tolerance_ticks": 20,
            "goal_speed_ticks_per_second": 50,
            "torque_limit_permille": [600, 400, 400, 400],
            "minimum_start_ticks": [2135, 2525, 2842, 2856],
            "maximum_start_ticks": [2227, 2592, 2963, 2922],
            "reviewed_natural_target_ticks": [2155, 2545, 2943, 2876],
            "maximum_travel_ticks": [80, 64, 128, 64]
        },
        "rgb_expression_policy": {
            "sampling_columns": 16,
            "sampling_rows": 12,
            "minimum_residual_luma": 24,
            "minimum_active_fraction_basis_points": 500,
            "frame_freshness_ms": 80,
            "brightness_basis_points": 7000,
            "color_rgb": [32, 128, 255],
            "blink": false,
            "head_origin_in_camera_m": [0.0, -0.25, -0.20],
            "neutral_head_from_camera_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
        }
    });
    (temporary, input)
}

fn write_input(root: &Path, input: &Value) -> PathBuf {
    let path = root.join("render-input.json");
    fs::write(
        &path,
        serde_json::to_vec_pretty(input).expect("serialize input fixture"),
    )
    .expect("write render input");
    path
}

fn valid_production_profile() -> Value {
    json!({
        "schema_version": 1,
        "admission_scope": "production_motion_profile_after_physical_wheels_off_review_v1",
        "admission_id": "reviewed-profile-v1",
        "reviewer_id": "operator-test",
        "controller": {
            "controller_uid_hex": "0102030405060708090a0b0c",
            "firmware_abi": 2,
            "firmware_build_id": 196609,
            "actuator_config_fingerprint_hex": "4b494b4f2d3450574d2d50524f443121",
            "hardware_profile_claim_id": "physically-reviewed-profile-v1",
            "controller_ready_timeout_ms": 3000,
            "heartbeat_period_ms": 20,
            "maximum_heartbeat_age_ms": 60,
            "maximum_host_command_rate_hz": 100,
            "serial_transmit_timeout_ms": 10,
            "serial_applied_ack_timeout_ms": 30,
            "controller_clock_abs_error_ppm_bound": 50000,
            "deadline_quantization_margin_ms": 2,
            "expected_max_abs_pwm_percent": 30,
            "expected_pwm_frequency_hz": 20000,
            "expected_watchdog_nominal_timeout_ms": 250,
            "expected_neutral_output": "both_low",
            "expected_physical_stop_semantics": "coast_verified",
            "command_udp_port": 8081
        },
        "actuation": {
            "plant_model_id": "reviewed-plant-v1",
            "plant_model_version": 1,
            "operator_claimed_physical_approval": {
                "approval_id": "physical-approval-v1",
                "approver_id": "operator-test",
                "plant_dataset_content_id": format!("sha256:{}", "aa".repeat(32)),
                "plant_identification_method_id": "method-v1",
                "plant_sample_count": 1000,
                "plant_fit_residuals": {
                    "left_velocity_rmse_mps": 0.01,
                    "right_velocity_rmse_mps": 0.01,
                    "yaw_rate_rmse_rad_s": 0.02,
                    "max_abs_velocity_error_mps": 0.04
                },
                "imu_calibration_id": "imu-v1",
                "stereo_calibration_id": "stereo-v1",
                "tracking_camera_to_base_calibration_id": "camera-base-v1"
            },
            "apply_ack_budget_ns": 30000000,
            "stop_ack_budget_ns": 40000000,
            "scheduling_guard_ns": 5000000,
            "controller_motion_lease_ms": 200,
            "controller_deadline_tolerance_ns": 5000000,
            "maximum_uncommanded_motion_ns": 500000000
        },
        "live_mode_policy": {
            "startup": "disarmed_map_only",
            "manual": { "permission": "disabled" },
            "point_goal": { "permission": "disabled" },
            "frontier_explore": { "permission": "disabled" }
        }
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}

fn restore_writable(path: &Path) {
    if !path.exists() {
        return;
    }
    let mut paths = fs::read_dir(path)
        .expect("read staged paths")
        .map(|entry| entry.expect("entry").path())
        .collect::<Vec<_>>();
    for child in &paths {
        if child.is_dir() {
            restore_writable(child);
        } else {
            let mut permissions = fs::metadata(child).expect("file metadata").permissions();
            permissions.set_mode(0o644);
            fs::set_permissions(child, permissions).expect("restore file");
        }
    }
    paths.push(path.to_path_buf());
    for directory in paths.into_iter().rev().filter(|entry| entry.is_dir()) {
        let mut permissions = fs::metadata(&directory)
            .expect("directory metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(directory, permissions).expect("restore directory");
    }
}

#[test]
fn launch_is_written_last_and_every_file_matches_plan_digest() {
    let (temporary, input) = source_fixture();
    let root = canonical_root(&temporary);
    let input_path = write_input(&root, &input);
    let destination = root.join("bundle");
    let plan = render_bundle(
        &input_path,
        RenderMode::Stage {
            destination: &destination,
        },
    )
    .expect("qualification bundle");

    assert_eq!(plan.bundle_kind, "wheels_off_qualification");
    assert_eq!(
        plan.files.last().expect("last file").relative_path,
        "nano-wheels-off-qualification-launch-v2.json"
    );
    assert_eq!(
        plan.files
            .get(plan.files.len() - 2)
            .expect("evidence before launch")
            .relative_path,
        "evidence/render-evidence-v1.json"
    );
    for expected in &plan.files {
        let staged_path = destination.join(&expected.relative_path);
        let bytes = fs::read(&staged_path).expect("read staged exact file");
        assert_eq!(u64::try_from(bytes.len()).unwrap(), expected.byte_len);
        assert_eq!(sha256_hex(&bytes), expected.sha256_hex);
        assert!(
            !bytes.windows(2).any(|window| window == b"${"),
            "{} retained unresolved token",
            expected.relative_path
        );
        let permissions = fs::metadata(&staged_path)
            .expect("staged file metadata")
            .permissions();
        if expected.relative_path == "bin/kiko-nano-wheels-off-qualification" {
            assert_eq!(permissions.mode() & 0o777, 0o555);
        } else {
            assert!(permissions.readonly());
            assert_eq!(
                permissions.mode() & 0o111,
                0,
                "only the qualification executable may be executable: {}",
                expected.relative_path
            );
        }
    }
    let render_evidence: Value = serde_json::from_slice(
        &fs::read(destination.join("evidence/render-evidence-v1.json")).expect("render evidence"),
    )
    .expect("render evidence JSON");
    let recorded_order = render_evidence["deterministic_write_order"]
        .as_array()
        .expect("write order")
        .iter()
        .map(|path| path.as_str().expect("path").to_owned())
        .collect::<Vec<_>>();
    let planned_order = plan
        .files
        .iter()
        .map(|file| file.relative_path.clone())
        .collect::<Vec<_>>();
    assert_eq!(recorded_order, planned_order);
    assert_eq!(
        recorded_order.last().expect("last recorded write"),
        "nano-wheels-off-qualification-launch-v2.json"
    );
    let inventory_bytes = fs::read(destination.join("device-inventory-candidate-v2.json"))
        .expect("candidate inventory");
    let typed_inventory = load_expected_manifest_v2_from_slice(&inventory_bytes)
        .expect("rendered candidate inventory must be admitted");
    let inventory: Value = serde_json::from_slice(&inventory_bytes).expect("inventory JSON");
    let controller_bytes = fs::read(destination.join("controller-server-candidate-v2.json"))
        .expect("candidate controller contract");
    let controller: Value =
        serde_json::from_slice(&controller_bytes).expect("candidate controller JSON");
    assert_eq!(
        controller["maximum_host_command_rate_hz"],
        Value::from(100),
        "100 Hz is the server/parser ceiling, not the runtime baseline"
    );
    let typed_controller = ControllerServerConfig::parse_json(&controller_bytes)
        .expect("rendered candidate controller contract must be admitted");
    let ControllerServerConfig::OperatorSupervisedFourPwmCandidateV2(typed_controller) =
        typed_controller
    else {
        panic!("rendered qualification controller must retain candidate schema V2")
    };
    assert_eq!(
        typed_controller.minimum_host_command_interval(),
        std::time::Duration::from_millis(10)
    );
    let candidate_policy_bytes = fs::read(destination.join("candidate-controller-policy-v1.json"))
        .expect("candidate controller policy");
    let admitted_controller =
        WheelsOffCandidateControllerBinding::parse_json(&candidate_policy_bytes)
            .expect("rendered candidate policy must parse")
            .admit(
                &typed_inventory,
                &typed_controller,
                "127.0.0.1:8080"
                    .parse()
                    .expect("fixed candidate loopback endpoint"),
            )
            .expect("rendered candidate contracts must cross-bind");
    assert_eq!(
        admitted_controller.command_interval(),
        std::time::Duration::from_millis(20)
    );
    let calibration =
        fs::read(destination.join("artifacts/calibration/assembly-v1.json")).expect("calibration");
    NanoCalibrationArtifactV1::parse_json(&calibration).expect("canonical calibration artifact");
    let expected_digest = Sha256::digest(&calibration)
        .iter()
        .copied()
        .map(Value::from)
        .collect::<Vec<_>>();
    assert_eq!(
        inventory["calibration_artifacts"][0]["sha256"],
        Value::Array(expected_digest)
    );
    let launch_bytes = fs::read(destination.join("nano-wheels-off-qualification-launch-v2.json"))
        .expect("qualification launch");
    NanoWheelsOffQualificationLaunchV2::parse_json(&launch_bytes)
        .expect("typed qualification launch");
    let launch: Value = serde_json::from_slice(&launch_bytes).expect("qualification launch JSON");
    assert_eq!(
        launch["calibration_artifact"]["artifact_id"],
        "assembly-calibration-v1"
    );
    assert_eq!(
        launch["calibration_artifact"]["asset"]["relative_path"],
        "artifacts/calibration/assembly-v1.json"
    );
    assert_eq!(
        launch["calibration_artifact"]["asset"]["sha256_hex"],
        sha256_hex(&calibration)
    );
    let executable = fs::read(destination.join("bin/kiko-nano-wheels-off-qualification"))
        .expect("qualification executable");
    assert_eq!(
        launch["qualification_executable_asset"]["relative_path"],
        "bin/kiko-nano-wheels-off-qualification"
    );
    assert_eq!(
        launch["qualification_executable_asset"]["sha256_hex"],
        sha256_hex(&executable)
    );
    assert_eq!(
        launch["native_runtime_manifest_asset"]["relative_path"],
        "native-runtime-v1.json"
    );
    let native_runtime_manifest =
        fs::read(destination.join("native-runtime-v1.json")).expect("native runtime manifest");
    NanoWheelsOffNativeRuntimeV1::parse_json(&native_runtime_manifest)
        .expect("renderer and qualification bootstrap share one native-runtime contract");
    assert_eq!(
        launch["native_runtime_manifest_asset"]["sha256_hex"],
        sha256_hex(&native_runtime_manifest)
    );
    let input_evidence = destination.join("evidence/render-input-v2.json");
    assert!(
        input_evidence.exists(),
        "qualification retains its schema-V2 render input under a versioned evidence name"
    );
    let executable_source = render_evidence["sources"]
        .as_array()
        .expect("source evidence")
        .iter()
        .find(|source| source["role"] == "qualification_executable")
        .expect("qualification executable source evidence");
    assert_eq!(
        executable_source["source_path"],
        input["bundle"]["qualification_executable_path"]
    );
    assert_eq!(
        executable_source["destination_relative_path"],
        "bin/kiko-nano-wheels-off-qualification"
    );
    assert_eq!(
        executable_source["byte_len"],
        u64::try_from(executable.len()).expect("executable length")
    );
    assert_eq!(executable_source["sha256_hex"], sha256_hex(&executable));
    assert!(fs::metadata(&destination).unwrap().permissions().readonly());
    restore_writable(&destination);
}

#[test]
fn bundle_variants_own_disjoint_executable_inputs() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    let qualification_executable_path = input["bundle"]["qualification_executable_path"].clone();

    input["bundle"] = json!({ "kind": "wheels_off_qualification" });
    assert!(
        render_bundle(&write_input(&root, &input), RenderMode::DryRun).is_err(),
        "qualification must bind its exact executable"
    );

    input["bundle"] = json!({
        "kind": "production",
        "production_controller_profile_path": null,
        "qualification_executable_path": qualification_executable_path
    });
    input["schema_version"] = json!(1);
    assert!(
        render_bundle(&write_input(&root, &input), RenderMode::DryRun).is_err(),
        "production must reject qualification-only executable input"
    );
}

#[test]
fn render_input_versions_are_bundle_specific_and_fail_closed() {
    let (temporary, input) = source_fixture();
    let root = canonical_root(&temporary);

    let mut legacy_qualification = input.clone();
    legacy_qualification["schema_version"] = json!(1);
    legacy_qualification["bundle"]
        .as_object_mut()
        .expect("qualification bundle")
        .remove("qualification_executable_path");
    let error = render_bundle(
        &write_input(&root, &legacy_qualification),
        RenderMode::DryRun,
    )
    .expect_err("published qualification V1 cannot be reinterpreted as V2");
    assert!(
        error
            .to_string()
            .contains("unsupported wheels_off_qualification render-input schema 1; expected 2")
    );

    let mut production = input;
    add_production_face_assets(&root, &mut production);
    production["bundle"] = json!({
        "kind": "production",
        "production_controller_profile_path": null
    });
    let error = render_bundle(&write_input(&root, &production), RenderMode::DryRun)
        .expect_err("production V1 reaches the independent profile gate");
    assert!(
        error
            .to_string()
            .contains("fail-closed without a separate admitted"),
        "production render-input schema V1 remains admitted"
    );

    production["schema_version"] = json!(2);
    let error = render_bundle(&write_input(&root, &production), RenderMode::DryRun)
        .expect_err("production must not silently adopt qualification schema V2");
    assert!(
        error
            .to_string()
            .contains("unsupported production render-input schema 2; expected 1")
    );
}

#[test]
fn qualification_requires_every_reviewed_exact_nano_soname() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    let onnx = input["native_libraries"]
        .as_array_mut()
        .expect("native libraries")
        .iter_mut()
        .find(|library| library["role"] == "onnxruntime")
        .expect("ONNX Runtime role");
    onnx["soname"] = json!("libonnxruntime.so");

    let error = render_bundle(&write_input(&root, &input), RenderMode::DryRun)
        .expect_err("loader-compatible but unreviewed qualification SONAME must fail closed");
    assert!(
        error
            .to_string()
            .contains("requires Nano SONAME \"libonnxruntime.so.1\"")
    );
}

#[test]
fn production_without_separate_profile_is_fail_closed() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    add_production_face_assets(&root, &mut input);
    input["bundle"] = json!({
        "kind": "production",
        "production_controller_profile_path": null
    });
    let input_path = write_input(&root, &input);
    let error = render_bundle(&input_path, RenderMode::DryRun).expect_err("profile is mandatory");
    assert!(
        error
            .to_string()
            .contains("fail-closed without a separate admitted")
    );
}

#[test]
fn production_without_exact_face_assets_is_fail_closed() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    input["bundle"] = json!({
        "kind": "production",
        "production_controller_profile_path": null
    });
    input["schema_version"] = json!(1);
    let input_path = write_input(&root, &input);
    let error =
        render_bundle(&input_path, RenderMode::DryRun).expect_err("face assets are mandatory");
    assert!(
        error
            .to_string()
            .contains("requires exact frontal and profile face-cascade")
    );
}

#[test]
fn production_open_cv_roles_require_the_observed_nano_sonames() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    add_production_face_assets(&root, &mut input);
    input["bundle"] = json!({
        "kind": "production",
        "production_controller_profile_path": null
    });
    let objdetect = input["native_libraries"]
        .as_array_mut()
        .expect("fixture native libraries")
        .iter_mut()
        .find(|library| library["role"] == "opencv_objdetect")
        .expect("objdetect fixture");
    objdetect["soname"] = json!("libopencv_objdetect.so");
    let error = render_bundle(&write_input(&root, &input), RenderMode::DryRun)
        .expect_err("unversioned OpenCV SONAME must fail closed");
    assert!(
        error
            .to_string()
            .contains("requires Nano SONAME \"libopencv_objdetect.so.4.5d\"")
    );
}

#[test]
fn production_derives_navigation_digest_and_loopback_port() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    add_production_face_assets(&root, &mut input);
    input["discovery"]["stm32"]["firmware_build_id"] = json!(196609);
    input["discovery"]["stm32"]["hardware_profile_fingerprint_hex"] =
        json!("4b494b4f2d3450574d2d50524f443121");
    input["discovery"]["stm32"]["capabilities_bits"] = json!(255);
    let profile_path = root.join("production-profile.json");
    fs::write(
        &profile_path,
        serde_json::to_vec_pretty(&valid_production_profile()).unwrap(),
    )
    .unwrap();
    input["bundle"] = json!({
        "kind": "production",
        "production_controller_profile_path": profile_path
    });
    let input_path = write_input(&root, &input);
    let destination = root.join("production-bundle");
    let plan = render_bundle(
        &input_path,
        RenderMode::Stage {
            destination: &destination,
        },
    )
    .expect("production render");
    assert_eq!(plan.bundle_kind, "production");
    assert!(destination.join("evidence/render-input-v1.json").exists());
    assert!(!destination.join("evidence/render-input-v2.json").exists());
    let production_input_permissions =
        fs::metadata(destination.join("evidence/render-input-v1.json"))
            .expect("production render-input evidence metadata")
            .permissions();
    assert!(production_input_permissions.readonly());
    assert_eq!(
        production_input_permissions.mode() & 0o111,
        0,
        "production and other non-executable leaves retain readonly, non-executable permissions"
    );
    assert_eq!(
        plan.files.last().unwrap().relative_path,
        "nano-agent-launch-v3.json"
    );
    let navigation = fs::read(destination.join("navigation-shadow-v1.json")).unwrap();
    let actuation: Value = serde_json::from_slice(
        &fs::read(destination.join("navigation-actuation-v2.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        actuation["navigation_config_sha256_hex"],
        sha256_hex(&navigation)
    );
    assert_eq!(actuation["schema_version"], 2);
    assert_eq!(
        actuation["plant_artifact_sha256_hex"],
        sha256_hex(&fs::read(destination.join("artifacts/plant/plant.json")).unwrap())
    );
    assert_eq!(
        actuation["operator_claimed_physical_approval"]["plant_dataset_content_id"],
        format!("sha256:{}", "aa".repeat(32))
    );
    assert_ne!(
        actuation["operator_claimed_physical_approval"]["plant_dataset_content_id"]
            .as_str()
            .unwrap()
            .strip_prefix("sha256:")
            .unwrap(),
        actuation["plant_artifact_sha256_hex"].as_str().unwrap()
    );
    assert_eq!(actuation["command_endpoint"], "127.0.0.1:8081");
    let inventory_bytes = fs::read(destination.join("device-inventory-v1.json")).unwrap();
    let inventory: Value = serde_json::from_slice(&inventory_bytes).unwrap();
    load_expected_manifest_v1_from_slice(&inventory_bytes).expect("production inventory");
    assert_eq!(
        inventory["stm32"]["control_endpoint_identity"],
        "udp://127.0.0.1:8081"
    );
    let launch_bytes =
        fs::read(destination.join("nano-agent-launch-v3.json")).expect("production launch");
    let typed_launch =
        NanoAgentLaunchV3::parse_json(&launch_bytes).expect("typed production launch");
    let limits = typed_launch.storage().navigation_dataset_limits();
    assert_eq!(limits.maximum_bytes(), 8_589_934_592);
    assert_eq!(limits.maximum_files(), 65_536);
    assert_eq!(limits.maximum_ingress_records(), 100_000);
    assert_eq!(limits.minimum_free_bytes_after_write(), 1_073_741_824);
    assert_eq!(limits.terminal_reserve_bytes(), 268_435_456);
    let launch: Value = serde_json::from_slice(&launch_bytes).expect("production launch JSON");
    assert_eq!(
        launch["calibration_artifact"]["asset"]["sha256_hex"],
        sha256_hex(
            &fs::read(destination.join("artifacts/calibration/assembly-v1.json"))
                .expect("production calibration")
        )
    );
    assert_eq!(
        launch["face_perception"]["frontal_face_cascade_asset"]["sha256_hex"],
        sha256_hex(
            &fs::read(destination.join("models/opencv/haarcascade_frontalface_default.xml"))
                .expect("production frontal cascade")
        )
    );
    assert_eq!(
        launch["face_perception"]["profile_face_cascade_asset"]["sha256_hex"],
        sha256_hex(
            &fs::read(destination.join("models/opencv/haarcascade_profileface.xml"))
                .expect("production profile cascade")
        )
    );
    let native: Value = serde_json::from_slice(
        &fs::read(destination.join("native-runtime-v1.json")).expect("native manifest"),
    )
    .expect("native manifest JSON");
    for (role, soname) in [
        ("opencv_core", "libopencv_core.so.4.5d"),
        ("opencv_imgproc", "libopencv_imgproc.so.4.5d"),
        ("opencv_objdetect", "libopencv_objdetect.so.4.5d"),
    ] {
        let library = native["libraries"]
            .as_array()
            .expect("native library list")
            .iter()
            .find(|library| library["role"] == role)
            .expect("required direct OpenCV role");
        let relative_path = format!("lib/{soname}");
        assert_eq!(library["soname"], soname);
        assert_eq!(library["relative_path"], relative_path);
        assert_eq!(
            library["sha256_hex"],
            sha256_hex(
                &fs::read(destination.join(&relative_path)).expect("staged OpenCV direct library")
            )
        );
    }
    let controller_bytes = fs::read(destination.join("controller-server-v1.json")).unwrap();
    let controller =
        ControllerServerConfigV1::parse_json(&controller_bytes).expect("production controller");
    assert_eq!(
        controller.serial_device().to_string_lossy(),
        input["discovery"]["stm32"]["serial_by_id_path"]
            .as_str()
            .unwrap()
    );
    let agent_policy_bytes = fs::read(destination.join("agent-policy-v3.json")).unwrap();
    NanoAgentPolicyConfigV3::parse_json(&agent_policy_bytes).expect("production agent policy");
    let agent_policy: Value =
        serde_json::from_slice(&agent_policy_bytes).expect("production agent policy JSON");
    assert_eq!(
        agent_policy["control"]["runtime_response_timeout_ms"],
        json!(30_000),
        "ordinary production command response timeout remains bounded separately"
    );
    assert_eq!(
        agent_policy["control"]["terminal_response_timeout_ms"],
        json!(300_000),
        "terminal drain/finalize/hash/fsync has a distinct bounded response budget"
    );
    restore_writable(&destination);
}

#[test]
fn candidate_controller_identity_cannot_be_relabelled_as_production() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    add_production_face_assets(&root, &mut input);
    let profile_path = root.join("candidate-relabelled-production.json");
    let mut profile = valid_production_profile();
    profile["controller"]["firmware_build_id"] = json!(135169);
    profile["controller"]["actuator_config_fingerprint_hex"] =
        json!("4b494b4f2d3450574d2d43414e443121");
    fs::write(&profile_path, serde_json::to_vec_pretty(&profile).unwrap()).unwrap();
    input["bundle"] = json!({
        "kind": "production",
        "production_controller_profile_path": profile_path
    });
    let input_path = write_input(&root, &input);
    let error =
        render_bundle(&input_path, RenderMode::DryRun).expect_err("candidate is not production");
    assert!(
        error
            .to_string()
            .contains("candidate STM32 firmware identity is forbidden")
    );
}

#[test]
fn production_profile_must_pass_the_authoritative_controller_parser() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    add_production_face_assets(&root, &mut input);
    input["discovery"]["stm32"]["firmware_build_id"] = json!(196609);
    input["discovery"]["stm32"]["hardware_profile_fingerprint_hex"] =
        json!("4b494b4f2d3450574d2d50524f443121");
    input["discovery"]["stm32"]["capabilities_bits"] = json!(255);
    let mut profile = valid_production_profile();
    profile["controller"]["maximum_heartbeat_age_ms"] = json!(30);
    let profile_path = root.join("invalid-controller-timing.json");
    fs::write(&profile_path, serde_json::to_vec_pretty(&profile).unwrap()).unwrap();
    input["bundle"] = json!({
        "kind": "production",
        "production_controller_profile_path": profile_path
    });
    let input_path = write_input(&root, &input);
    let error = render_bundle(&input_path, RenderMode::DryRun)
        .expect_err("authoritative controller timing rejection");
    assert!(error.to_string().contains("authoritative parser"));
}

#[test]
fn production_warm_start_renders_the_exact_saved_map_and_dataset_pair() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    add_production_face_assets(&root, &mut input);
    input["discovery"]["stm32"]["firmware_build_id"] = json!(196609);
    input["discovery"]["stm32"]["hardware_profile_fingerprint_hex"] =
        json!("4b494b4f2d3450574d2d50524f443121");
    input["discovery"]["stm32"]["capabilities_bits"] = json!(255);
    input["runtime"]["storage"]["warm_start"] = json!({ "kind": "dataset_replay" });
    let profile_path = root.join("production-profile.json");
    fs::write(
        &profile_path,
        serde_json::to_vec_pretty(&valid_production_profile()).unwrap(),
    )
    .unwrap();
    input["bundle"] = json!({
        "kind": "production",
        "production_controller_profile_path": profile_path
    });
    let input_path = write_input(&root, &input);
    let destination = root.join("warm-production-bundle");
    render_bundle(
        &input_path,
        RenderMode::Stage {
            destination: &destination,
        },
    )
    .expect("warm production render");
    let policy_bytes = fs::read(destination.join("agent-policy-v3.json")).unwrap();
    let policy: Value = serde_json::from_slice(&policy_bytes).unwrap();
    assert_eq!(
        policy["map_persistence"]["warm_start"],
        json!({
            "kind": "dataset_replay",
            "occupancy_snapshot_path": "/var/lib/kiko-nano-agent/maps/current.kmap",
            "slam_dataset_directory_path": "/var/lib/kiko-nano-agent/navigation"
        })
    );
    assert_eq!(
        policy["control"]["runtime_response_timeout_ms"],
        json!(30_000),
        "ordinary command responses retain their independent short bound"
    );
    assert_eq!(
        policy["control"]["terminal_response_timeout_ms"],
        json!(300_000)
    );
    NanoAgentPolicyConfigV3::parse_json(&policy_bytes).expect("warm-start policy");
    restore_writable(&destination);
}

#[test]
fn qualification_rejects_production_warm_start_state() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    input["runtime"]["storage"]["warm_start"] = json!({ "kind": "dataset_replay" });
    let input_path = write_input(&root, &input);
    let error =
        render_bundle(&input_path, RenderMode::DryRun).expect_err("qualification warm start");
    assert!(
        error
            .to_string()
            .contains("cannot replay persisted production map state")
    );
}

#[test]
fn wheels_off_bundle_still_requires_direct_open_cv_elf_closure() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    input["native_libraries"]
        .as_array_mut()
        .expect("fixture native libraries")
        .retain(|library| library["role"] != "opencv_core");
    let error = render_bundle(&write_input(&root, &input), RenderMode::DryRun)
        .expect_err("qualification binary also carries production dispatch");
    assert!(
        error
            .to_string()
            .contains("exact four legacy roles and three direct OpenCV roles")
    );
}

#[test]
fn renderer_requires_and_checks_navigation_dataset_storage_limits() {
    let (temporary, input) = source_fixture();
    let root = canonical_root(&temporary);

    let mut missing = input.clone();
    missing["runtime"]["storage"]
        .as_object_mut()
        .expect("storage object")
        .remove("maximum_navigation_dataset_bytes");
    let error = render_bundle(&write_input(&root, &missing), RenderMode::DryRun)
        .expect_err("dataset byte bound is mandatory");
    assert!(
        error
            .to_string()
            .contains("maximum_navigation_dataset_bytes")
    );

    let mut too_many_files = input.clone();
    too_many_files["runtime"]["storage"]["maximum_navigation_dataset_files"] = json!(65_537);
    let error = render_bundle(&write_input(&root, &too_many_files), RenderMode::DryRun)
        .expect_err("dataset file bound");
    assert!(
        error
            .to_string()
            .contains("runtime.storage.maximum_navigation_dataset_files")
    );

    let mut too_many_records = input.clone();
    too_many_records["runtime"]["storage"]["maximum_navigation_ingress_records"] = json!(1_048_577);
    let error = render_bundle(&write_input(&root, &too_many_records), RenderMode::DryRun)
        .expect_err("journal record bound");
    assert!(
        error
            .to_string()
            .contains("runtime.storage.maximum_navigation_ingress_records")
    );

    let mut reserve_reaches_maximum = input.clone();
    reserve_reaches_maximum["runtime"]["storage"]["navigation_dataset_terminal_reserve_bytes"] =
        reserve_reaches_maximum["runtime"]["storage"]["maximum_navigation_dataset_bytes"].clone();
    let error = render_bundle(
        &write_input(&root, &reserve_reaches_maximum),
        RenderMode::DryRun,
    )
    .expect_err("terminal reserve must leave capture capacity");
    assert!(
        error
            .to_string()
            .contains("must be below runtime.storage.maximum_navigation_dataset_bytes")
    );

    let mut one_byte_short = input.clone();
    one_byte_short["runtime"]["storage"]["navigation_dataset_terminal_reserve_bytes"] =
        json!(134_221_823);
    let error = render_bundle(&write_input(&root, &one_byte_short), RenderMode::DryRun)
        .expect_err("terminal reserve fragment rounding");
    assert!(
        error
            .to_string()
            .contains("below the checked fragment-rounded")
    );

    let mut exact_minimum = input;
    exact_minimum["runtime"]["storage"]["navigation_dataset_terminal_reserve_bytes"] =
        json!(134_221_824);
    render_bundle(&write_input(&root, &exact_minimum), RenderMode::DryRun)
        .expect("exact checked terminal reserve");
}

#[test]
fn renderer_rejects_policy_values_the_agent_would_reject() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    input["head_policy"]["reviewed_natural_target_ticks"][0] = json!(2156);
    let input_path = write_input(&root, &input);
    let error = render_bundle(&input_path, RenderMode::DryRun).expect_err("unreviewed head target");
    assert!(
        error
            .to_string()
            .contains("physically reviewed natural target")
    );

    input["head_policy"]["reviewed_natural_target_ticks"][0] = json!(2155);
    input["rgb_expression_policy"]["sampling_columns"] = json!(65);
    input["rgb_expression_policy"]["sampling_rows"] = json!(65);
    let input_path = write_input(&root, &input);
    let error =
        render_bundle(&input_path, RenderMode::DryRun).expect_err("oversized RGB sampling grid");
    assert!(error.to_string().contains("outside the runtime sampling"));

    input["rgb_expression_policy"]["sampling_columns"] = json!(16);
    input["rgb_expression_policy"]["sampling_rows"] = json!(12);
    input["rgb_expression_policy"]["frame_freshness_ms"] = json!(30);
    let input_path = write_input(&root, &input);
    let error =
        render_bundle(&input_path, RenderMode::DryRun).expect_err("stale before eye round trip");
    assert!(error.to_string().contains("outside the runtime sampling"));
}

#[test]
fn unknown_fields_and_unresolved_tokens_are_rejected() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    input["unexpected"] = json!(true);
    let input_path = write_input(&root, &input);
    let error = render_bundle(&input_path, RenderMode::DryRun).expect_err("unknown field");
    assert!(error.to_string().contains("unknown field"));

    input.as_object_mut().unwrap().remove("unexpected");
    let navigation = PathBuf::from(
        input["assets"]["navigation_shadow_source_path"]
            .as_str()
            .unwrap(),
    );
    fs::write(&navigation, br#"{"bad":"${UNRESOLVED}"}"#).unwrap();
    let input_path = write_input(&root, &input);
    let error = render_bundle(&input_path, RenderMode::DryRun).expect_err("token");
    assert!(error.to_string().contains("unresolved ${...} token"));
}

#[test]
fn nonempty_destination_is_never_mutated() {
    let (temporary, input) = source_fixture();
    let root = canonical_root(&temporary);
    let input_path = write_input(&root, &input);
    let destination = root.join("occupied");
    fs::create_dir(&destination).unwrap();
    fs::write(destination.join("owner.txt"), b"preserve").unwrap();
    let error = render_bundle(
        &input_path,
        RenderMode::Stage {
            destination: &destination,
        },
    )
    .expect_err("nonempty destination");
    assert!(error.to_string().contains("not empty"));
    assert_eq!(
        fs::read(destination.join("owner.txt")).unwrap(),
        b"preserve"
    );
}

#[test]
fn dry_run_creates_no_destination() {
    let (temporary, input) = source_fixture();
    let root = canonical_root(&temporary);
    let input_path = write_input(&root, &input);
    let destination = root.join("never-created");
    let plan = render_bundle(&input_path, RenderMode::DryRun).expect("dry run");
    assert!(!plan.files.is_empty());
    assert!(!destination.exists());
}

#[test]
fn symlinked_source_is_rejected() {
    use std::os::unix::fs::symlink;

    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    let real = PathBuf::from(
        input["assets"]["superpoint_model"]["source_path"]
            .as_str()
            .unwrap(),
    );
    let linked = root.join("linked-superpoint.onnx");
    symlink(real, &linked).unwrap();
    input["assets"]["superpoint_model"]["source_path"] = json!(linked);
    let input_path = write_input(&root, &input);
    let error = render_bundle(&input_path, RenderMode::DryRun).expect_err("symlink");
    assert!(error.to_string().contains("symlink component rejected"));
}
