#![cfg(unix)]

use std::collections::BTreeSet;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use kiko_device_inventory::{
    load_expected_manifest_v1_from_slice, load_expected_manifest_v2_from_slice,
};
use kiko_nano_bundle_renderer::{
    QualificationFaceCascadeRole, RenderError, RenderMode, render_bundle,
};
use kiko_slam::navigation::{
    NanoAgentLaunchV3, NanoAgentPolicyConfigV3, NanoCalibrationArtifactV1,
    NanoCalibrationBindingError, NanoWheelsOffNativeRuntimeV1, NanoWheelsOffQualificationLaunchV4,
    OfflineNavigationGraphParseError, ProductionNavigationControllerBindingError,
    WheelsOffCandidateControllerBinding,
};
use robot_server::config::{ControllerServerConfig, ControllerServerConfigV1};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

fn unresolved_tokens(input: &str) -> BTreeSet<&str> {
    let mut tokens = BTreeSet::new();
    let mut remaining = input;
    while let Some(start) = remaining.find("${") {
        remaining = &remaining[start..];
        let end = remaining
            .find('}')
            .expect("every render-input token must close");
        tokens.insert(&remaining[..=end]);
        remaining = &remaining[end + 1..];
    }
    tokens
}

fn template_shape_with_unquoted_tokens_replaced_by_null(input: &str) -> Value {
    let input = input.as_bytes();
    let mut output = Vec::with_capacity(input.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    while index < input.len() {
        let byte = input[index];
        if !in_string && byte == b'$' && input.get(index + 1) == Some(&b'{') {
            let relative_end = input[index + 2..]
                .iter()
                .position(|candidate| *candidate == b'}')
                .expect("render-input token must close");
            index += relative_end + 3;
            output.extend_from_slice(b"null");
            continue;
        }

        output.push(byte);
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
        } else if byte == b'"' {
            in_string = true;
        }
        index += 1;
    }
    serde_json::from_slice(&output).expect("sentinelized V4 render-input shape must be JSON")
}

fn template_replacement(value: &Value) -> String {
    let serialized = serde_json::to_string(value).expect("template replacement is JSON");
    if value.is_string() {
        serialized[1..serialized.len() - 1].to_owned()
    } else {
        serialized
    }
}

fn materialize_v4_template(template: &str, fixture: &Value) -> Value {
    let native_source = |role: &str| {
        fixture["native_libraries"]
            .as_array()
            .expect("fixture native libraries")
            .iter()
            .find(|library| library["role"] == role)
            .expect("fixture has every native role")["source_path"]
            .clone()
    };
    let replacements = [
        (
            "${QUALIFICATION_EXECUTABLE_SOURCE_ABSOLUTE_PATH}",
            fixture["bundle"]["qualification_executable_path"].clone(),
        ),
        ("${ROBOT_ID}", fixture["robot_id"].clone()),
        ("${OAK_MXID}", fixture["discovery"]["oak"]["mxid"].clone()),
        (
            "${DEPTHAI_HEADER_SDK_VERSION}",
            fixture["discovery"]["oak"]["compiled_depthai_header_sdk_version"].clone(),
        ),
        (
            "${DEPTHAI_HEADER_SDK_COMMIT}",
            fixture["discovery"]["oak"]["compiled_depthai_header_sdk_commit"].clone(),
        ),
        (
            "${DEPTHAI_HEADER_DEVICE_ARTIFACT_VERSION}",
            fixture["discovery"]["oak"]
                ["compiled_depthai_header_embedded_device_artifact_version"]
                .clone(),
        ),
        (
            "${DEPTHAI_HEADER_BOOTLOADER_ARTIFACT_VERSION}",
            fixture["discovery"]["oak"]
                ["compiled_depthai_header_embedded_bootloader_artifact_version"]
                .clone(),
        ),
        (
            "${STM32_SERIAL_BY_ID_PATH}",
            fixture["discovery"]["stm32"]["serial_by_id_path"].clone(),
        ),
        (
            "${CONTROLLER_UID_HEX}",
            fixture["discovery"]["stm32"]["controller_uid_hex"].clone(),
        ),
        (
            "${FIRMWARE_ABI}",
            fixture["discovery"]["stm32"]["firmware_abi"].clone(),
        ),
        (
            "${FIRMWARE_BUILD_ID}",
            fixture["discovery"]["stm32"]["firmware_build_id"].clone(),
        ),
        (
            "${ACTUATOR_CONFIG_FINGERPRINT_HEX}",
            fixture["discovery"]["stm32"]["hardware_profile_fingerprint_hex"].clone(),
        ),
        (
            "${CONTROLLER_CAPABILITIES_BITS}",
            fixture["discovery"]["stm32"]["capabilities_bits"].clone(),
        ),
        (
            "${HEAD_SERIAL_BY_ID_PATH}",
            fixture["discovery"]["head"]["adapter_serial_by_id_path"].clone(),
        ),
        (
            "${HEAD_BOW_SERVO_ID}",
            fixture["discovery"]["head"]["bow_servo_id"].clone(),
        ),
        (
            "${HEAD_CURL_SERVO_ID}",
            fixture["discovery"]["head"]["curl_servo_id"].clone(),
        ),
        (
            "${HEAD_YAW_SERVO_ID}",
            fixture["discovery"]["head"]["yaw_servo_id"].clone(),
        ),
        (
            "${HEAD_ROLL_SERVO_ID}",
            fixture["discovery"]["head"]["roll_servo_id"].clone(),
        ),
        (
            "${HEAD_BAUD_RATE_BPS}",
            fixture["discovery"]["head"]["baud_rate_bps"].clone(),
        ),
        (
            "${HEAD_DTR_ASSERTED}",
            fixture["discovery"]["head"]["dtr_asserted"].clone(),
        ),
        (
            "${HEAD_RTS_ASSERTED}",
            fixture["discovery"]["head"]["rts_asserted"].clone(),
        ),
        (
            "${EYE_SERIAL_BY_ID_PATH}",
            fixture["discovery"]["eye"]["serial_by_id_path"].clone(),
        ),
        (
            "${EYE_KEP_PROTOCOL_VERSION}",
            fixture["discovery"]["eye"]["kep_protocol_version"].clone(),
        ),
        (
            "${EYE_DEVICE_UID_HEX}",
            fixture["discovery"]["eye"]["device_uid_hex"].clone(),
        ),
        (
            "${EYE_FIRMWARE_BUILD_ID_HEX}",
            fixture["discovery"]["eye"]["firmware_build_id_hex"].clone(),
        ),
        (
            "${EYE_CAPABILITIES_BITS}",
            fixture["discovery"]["eye"]["capabilities_bits"].clone(),
        ),
        (
            "${CALIBRATION_ARTIFACT_ID}",
            fixture["assets"]["calibration"]["artifact_id"].clone(),
        ),
        (
            "${CALIBRATION_SOURCE_ABSOLUTE_PATH}",
            fixture["assets"]["calibration"]["source_path"].clone(),
        ),
        (
            "${CALIBRATION_ARTIFACT_RELATIVE_PATH}",
            json!("calibration/assembly-v1.json"),
        ),
        (
            "${PLANT_SOURCE_ABSOLUTE_PATH}",
            fixture["assets"]["plant"]["source_path"].clone(),
        ),
        (
            "${NAVIGATION_SHADOW_SOURCE_ABSOLUTE_PATH}",
            fixture["assets"]["navigation_shadow_source_path"].clone(),
        ),
        (
            "${SUPERPOINT_MODEL_SOURCE_ABSOLUTE_PATH}",
            fixture["assets"]["superpoint_model"]["source_path"].clone(),
        ),
        (
            "${LIGHTGLUE_MODEL_SOURCE_ABSOLUTE_PATH}",
            fixture["assets"]["lightglue_model"]["source_path"].clone(),
        ),
        (
            "${FRONTAL_FACE_CASCADE_SOURCE_ABSOLUTE_PATH}",
            fixture["assets"]["face_perception"]["frontal_face_cascade"]["source_path"].clone(),
        ),
        (
            "${PROFILE_FACE_CASCADE_SOURCE_ABSOLUTE_PATH}",
            fixture["assets"]["face_perception"]["profile_face_cascade"]["source_path"].clone(),
        ),
        ("${DEPTHAI_CORE_SOURCE_ABSOLUTE_PATH}", native_source("depthai_core")),
        (
            "${DYNAMIC_CALIBRATION_SOURCE_ABSOLUTE_PATH}",
            native_source("dynamic_calibration"),
        ),
        ("${LIBUSB_1_0_SOURCE_ABSOLUTE_PATH}", native_source("libusb1_0")),
        ("${ONNXRUNTIME_SOURCE_ABSOLUTE_PATH}", native_source("onnxruntime")),
        ("${OPENCV_CORE_SOURCE_ABSOLUTE_PATH}", native_source("opencv_core")),
        (
            "${OPENCV_IMGPROC_SOURCE_ABSOLUTE_PATH}",
            native_source("opencv_imgproc"),
        ),
        (
            "${OPENCV_OBJDETECT_SOURCE_ABSOLUTE_PATH}",
            native_source("opencv_objdetect"),
        ),
    ];

    let mut rendered = template.to_owned();
    for (token, value) in replacements {
        assert_eq!(
            rendered.matches(token).count(),
            1,
            "checked-in V4 template must contain exactly one {token}"
        );
        rendered = rendered.replace(token, &template_replacement(&value));
    }
    assert!(
        unresolved_tokens(&rendered).is_empty(),
        "fixture must replace every V4 evidence-boundary token"
    );
    serde_json::from_str(&rendered).expect("materialized checked-in V4 template is JSON")
}

fn write_source(root: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = root.join(name);
    fs::write(&path, bytes).expect("write fixture source");
    path
}

fn canonical_root(temporary: &TempDir) -> PathBuf {
    fs::canonicalize(temporary.path()).expect("canonical temporary root")
}

fn add_face_assets(root: &Path, input: &mut Value) {
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

fn navigation_for_plant(plant: &Value) -> Value {
    let mut navigation: Value = serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/navigation-shadow-v2.example.json"
    )))
    .expect("checked-in navigation example is JSON");
    navigation["coordinate_frames"]["tracking_camera_to_base"] = json!({
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "translation_m": [0.20, 0.0, -0.25]
    });
    navigation["odometry"]["raw_imu_calibration"] = json!({
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
    });
    navigation["plant_model"] = plant.clone();
    navigation
}

fn reviewed_production_plant() -> Value {
    json!({
        "schema_version": 1,
        "model_id": "reviewed-plant-v1",
        "model_version": 1,
        "sample_period_s": 0.1,
        "wheelbase_m": 0.4,
        "left": {
            "velocity_gain_mps_per_pwm_percent": 0.01,
            "time_constant_s": 0.5
        },
        "right": {
            "velocity_gain_mps_per_pwm_percent": 0.01,
            "time_constant_s": 0.6
        },
        "validity": {
            "left_pwm_min_percent": -30,
            "left_pwm_max_percent": 30,
            "right_pwm_min_percent": -30,
            "right_pwm_max_percent": 30,
            "left_velocity_min_mps": -0.3,
            "left_velocity_max_mps": 0.3,
            "right_velocity_min_mps": -0.3,
            "right_velocity_max_mps": 0.3,
            "maximum_absolute_yaw_rate_rad_per_sec": 2.0,
            "maximum_absolute_lateral_velocity_m_per_sec": 0.1
        },
        "evidence": {
            "kind": "claimed_physical_identification",
            "dataset_content_id": format!("sha256:{}", "aa".repeat(32)),
            "identification_method_id": "method-v1",
            "sample_count": 1000,
            "residuals": {
                "left_velocity_rmse_mps": 0.01,
                "right_velocity_rmse_mps": 0.01,
                "yaw_rate_rmse_rad_s": 0.02,
                "maximum_absolute_velocity_error_mps": 0.04
            }
        }
    })
}

fn prepare_production_assets(root: &Path, input: &mut Value) {
    input["schema_version"] = json!(1);
    add_face_assets(root, input);
    input["assets"]
        .as_object_mut()
        .expect("assets object")
        .remove("head_gaze_policy_source_path");

    let plant = reviewed_production_plant();
    let plant_path = write_source(
        root,
        "reviewed-production-plant.json",
        &serde_json::to_vec_pretty(&plant).expect("serialize reviewed production plant"),
    );
    input["assets"]["plant"] = json!({
        "artifact_id": "reviewed-plant-v1",
        "source_path": plant_path,
        "destination_relative_path": "artifacts/plant/reviewed-plant-v1.json"
    });

    let mut navigation = navigation_for_plant(&plant);
    navigation["mpc"]["step_period_s"] = json!(0.1);
    navigation["control_loop"]["control_period_ns"] = json!(100_000_000_u64);
    navigation["control_loop"]["solver_budget_ns"] = json!(50_000_000_u64);
    navigation["shadow_command"]["lease_ms"] = json!(200);
    let navigation_path = write_source(
        root,
        "reviewed-production-navigation.json",
        &serde_json::to_vec_pretty(&navigation).expect("serialize production navigation"),
    );
    input["assets"]["navigation_shadow_source_path"] = json!(navigation_path);
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
    let plant_bytes = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/nano-wheels-off-qualification-template/qualification-shadow-only-synthetic-unvalidated-plant-v2.json"
    ));
    let plant = write_source(&root, "plant.json", plant_bytes);
    let plant_value: Value =
        serde_json::from_slice(plant_bytes).expect("Gate-A plant fixture is JSON");
    let navigation = write_source(
        &root,
        "navigation.json",
        &serde_json::to_vec_pretty(&navigation_for_plant(&plant_value))
            .expect("serialize valid qualification navigation"),
    );
    let superpoint = write_source(&root, "superpoint.onnx", b"superpoint-exact");
    let lightglue = write_source(&root, "lightglue.onnx", b"lightglue-exact");
    let depthai = write_source(&root, "libdepthai-core.so", b"depthai-exact");
    let calibration_lib = write_source(&root, "libdynamic_calibration.so", b"calibration-exact");
    let libusb = write_source(&root, "libusb-1.0.so", b"libusb-exact");
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
    let head_gaze_policy = write_source(
        &root,
        "head-gaze-policy.json",
        br#"{"schema_version":1,"policy":"exact"}"#,
    );
    let mut input = json!({
        "schema_version": 4,
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
                "artifact_id": "qualification-shadow-only-synthetic-unvalidated-v2",
                "source_path": plant,
                "destination_relative_path": "artifacts/plant/qualification-shadow-only-synthetic-unvalidated-plant-v2.json"
            },
            "navigation_shadow_source_path": navigation,
            "superpoint_model": {
                "source_path": superpoint,
                "destination_relative_path": "models/superpoint.onnx"
            },
            "lightglue_model": {
                "source_path": lightglue,
                "destination_relative_path": "models/lightglue.onnx"
            },
            "head_gaze_policy_source_path": head_gaze_policy
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
                "soname": "libusb-1.0.so",
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
            "minimum_start_ticks": [2133, 2550, 1617, 3023],
            "maximum_start_ticks": [2194, 2660, 1852, 3067],
            "reviewed_natural_target_ticks": [2174, 2570, 1637, 3047],
            "maximum_travel_ticks": [48, 96, 224, 32]
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
    add_face_assets(&root, &mut input);
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
        "nano-wheels-off-qualification-launch-v4.json"
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
        "nano-wheels-off-qualification-launch-v4.json"
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
    let launch_bytes = fs::read(destination.join("nano-wheels-off-qualification-launch-v4.json"))
        .expect("qualification launch");
    NanoWheelsOffQualificationLaunchV4::parse_json(&launch_bytes)
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
    let head_gaze_policy =
        fs::read(destination.join("head-gaze-policy-v1.json")).expect("head-gaze policy");
    assert_eq!(
        launch["head_gaze_policy_asset"]["relative_path"],
        "head-gaze-policy-v1.json"
    );
    assert_eq!(
        launch["head_gaze_policy_asset"]["sha256_hex"],
        sha256_hex(&head_gaze_policy)
    );
    assert_eq!(
        launch["head_gaze_policy_asset"]["maximum_bytes"],
        u64::try_from(head_gaze_policy.len()).expect("head-gaze policy length")
    );
    for (field, relative_path) in [
        (
            "frontal_face_cascade_asset",
            "models/opencv/haarcascade_frontalface_default.xml",
        ),
        (
            "profile_face_cascade_asset",
            "models/opencv/haarcascade_profileface.xml",
        ),
    ] {
        let cascade = fs::read(destination.join(relative_path)).expect("face cascade");
        assert_eq!(
            launch["face_perception"][field]["relative_path"],
            relative_path
        );
        assert_eq!(
            launch["face_perception"][field]["sha256_hex"],
            sha256_hex(&cascade)
        );
        assert_eq!(
            launch["face_perception"][field]["maximum_bytes"],
            u64::try_from(cascade.len()).expect("cascade length")
        );
    }
    let input_evidence = destination.join("evidence/render-input-v4.json");
    assert!(
        input_evidence.exists(),
        "qualification retains its schema-V4 render input under a versioned evidence name"
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
    .expect_err("published qualification V1 cannot be reinterpreted as V4");
    assert!(
        error
            .to_string()
            .contains("unsupported wheels_off_qualification render-input schema 1; expected 4")
    );

    let mut retired_qualification_v2 = input.clone();
    retired_qualification_v2["schema_version"] = json!(2);
    let error = render_bundle(
        &write_input(&root, &retired_qualification_v2),
        RenderMode::DryRun,
    )
    .expect_err("published qualification V2 cannot be reinterpreted as V4");
    assert!(
        error
            .to_string()
            .contains("unsupported wheels_off_qualification render-input schema 2; expected 4")
    );

    let mut retired_qualification_v3 = input.clone();
    retired_qualification_v3["schema_version"] = json!(3);
    let error = render_bundle(
        &write_input(&root, &retired_qualification_v3),
        RenderMode::DryRun,
    )
    .expect_err("published qualification V3 cannot be reinterpreted as V4");
    assert!(
        error
            .to_string()
            .contains("unsupported wheels_off_qualification render-input schema 3; expected 4")
    );

    let mut production = input;
    prepare_production_assets(&root, &mut production);
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
fn qualification_v4_requires_exact_face_assets_and_allows_head_gaze_to_be_absent() {
    let (temporary, input) = source_fixture();
    let root = canonical_root(&temporary);

    let mut missing_face = input.clone();
    missing_face["assets"]
        .as_object_mut()
        .expect("fixture assets")
        .remove("face_perception");
    let error = render_bundle(&write_input(&root, &missing_face), RenderMode::DryRun)
        .expect_err("V4 face assets are mandatory");
    assert!(
        error
            .to_string()
            .contains("requires exact frontal and profile face-cascade")
    );

    let mut missing_head_gaze = input;
    missing_head_gaze["assets"]
        .as_object_mut()
        .expect("fixture assets")
        .remove("head_gaze_policy_source_path");
    let destination = root.join("qualification-without-head-gaze");
    let plan = render_bundle(
        &write_input(&root, &missing_head_gaze),
        RenderMode::Stage {
            destination: &destination,
        },
    )
    .expect("Gate A qualification does not require a head-gaze policy");
    assert!(
        plan.files
            .iter()
            .all(|file| file.relative_path != "head-gaze-policy-v1.json")
    );
    let launch: Value = serde_json::from_slice(
        &fs::read(destination.join("nano-wheels-off-qualification-launch-v4.json"))
            .expect("read launch without head gaze"),
    )
    .expect("parse launch without head gaze");
    assert!(launch.get("head_gaze_policy_asset").is_none());
    let evidence: Value = serde_json::from_slice(
        &fs::read(destination.join("evidence/render-evidence-v1.json"))
            .expect("read evidence without head gaze"),
    )
    .expect("parse evidence without head gaze");
    assert!(
        evidence["sources"]
            .as_array()
            .expect("evidence sources")
            .iter()
            .all(|source| source["role"] != "head_gaze_policy")
    );
}

#[test]
fn qualification_v4_rejects_explicit_null_head_gaze_policy() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    input["assets"]["head_gaze_policy_source_path"] = Value::Null;

    let error = render_bundle(&write_input(&root, &input), RenderMode::DryRun)
        .expect_err("explicit null must not alias absent/disabled head gaze");
    assert!(
        error
            .to_string()
            .contains("must be omitted to disable it; explicit null is forbidden")
    );
}

#[test]
fn qualification_v4_rejects_valid_plant_bytes_under_the_fixed_gate_a_v2_label() {
    let (temporary, input) = source_fixture();
    let root = canonical_root(&temporary);
    let plant_path = input["assets"]["plant"]["source_path"]
        .as_str()
        .expect("plant source path");
    let mut mutated: Value = serde_json::from_slice(
        &fs::read(plant_path).expect("read canonical Gate-A V2 plant fixture"),
    )
    .expect("parse canonical Gate-A V2 plant fixture");
    mutated["wheelbase_m"] = json!(0.41);
    fs::write(
        plant_path,
        serde_json::to_vec_pretty(&mutated).expect("serialize valid mutated plant"),
    )
    .expect("write valid mutated plant");

    let error = render_bundle(&write_input(&root, &input), RenderMode::DryRun)
        .expect_err("fixed Gate-A V2 identity cannot label mutated valid plant bytes");
    assert!(matches!(
        error,
        RenderError::QualificationPlantContentMismatch { .. }
    ));
}

#[test]
fn qualification_rejects_valid_json_navigation_with_a_different_plant_before_staging() {
    let (temporary, input) = source_fixture();
    let root = canonical_root(&temporary);
    let navigation_path = input["assets"]["navigation_shadow_source_path"]
        .as_str()
        .expect("navigation source path");
    let mut navigation: Value =
        serde_json::from_slice(&fs::read(navigation_path).expect("read valid navigation fixture"))
            .expect("navigation fixture is JSON");
    navigation["plant_model"]["wheelbase_m"] = json!(0.41);
    fs::write(
        navigation_path,
        serde_json::to_vec_pretty(&navigation).expect("serialize mismatched navigation"),
    )
    .expect("write mismatched navigation");

    let destination = root.join("must-not-stage-mismatched-navigation");
    let error = render_bundle(
        &write_input(&root, &input),
        RenderMode::Stage {
            destination: &destination,
        },
    )
    .expect_err("navigation must bind to the exact retained Gate-A plant");
    assert!(matches!(
        error,
        RenderError::OfflineNavigationGraph(OfflineNavigationGraphParseError::Navigation(_))
    ));
    assert!(
        !destination.exists(),
        "semantic rejection must precede the first staging write"
    );
}

#[test]
fn qualification_rejects_calibration_for_a_different_oak_before_staging() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    input["discovery"]["oak"]["mxid"] = json!("19443010F1B43A2E01");

    let destination = root.join("must-not-stage-calibration-for-another-oak");
    let error = render_bundle(
        &write_input(&root, &input),
        RenderMode::Stage {
            destination: &destination,
        },
    )
    .expect_err("calibration must bind to the deployment OAK identity");
    assert!(matches!(
        error,
        RenderError::CalibrationDeploymentBinding(
            NanoCalibrationBindingError::ManifestOakMxidMismatch
        )
    ));
    assert!(
        !destination.exists(),
        "calibration identity binding must precede the first staging write"
    );
}

#[test]
fn qualification_rejects_calibration_with_different_launch_dimensions_before_staging() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    input["runtime"]["oak"]["stereo_width_px"] = json!(800);

    let destination = root.join("must-not-stage-calibration-with-different-dimensions");
    let error = render_bundle(
        &write_input(&root, &input),
        RenderMode::Stage {
            destination: &destination,
        },
    )
    .expect_err("calibration dimensions must bind to the launch stereo dimensions");
    assert!(matches!(
        error,
        RenderError::CalibrationDeploymentBinding(
            NanoCalibrationBindingError::LaunchStereoDimensionsMismatch { .. }
        )
    ));
    assert!(
        !destination.exists(),
        "calibration dimension binding must precede the first staging write"
    );
}

#[test]
fn qualification_v4_rejects_face_and_head_gaze_aliases() {
    let (temporary, input) = source_fixture();
    let root = canonical_root(&temporary);

    let mut same_face_content = input.clone();
    same_face_content["assets"]["face_perception"]["profile_face_cascade"]["source_path"] =
        same_face_content["assets"]["face_perception"]["frontal_face_cascade"]["source_path"]
            .clone();
    let error = render_bundle(&write_input(&root, &same_face_content), RenderMode::DryRun)
        .expect_err("frontal and profile content must be distinct");
    assert!(matches!(
        error,
        RenderError::QualificationFaceAssetContentAlias {
            face: QualificationFaceCascadeRole::Frontal,
            ref aliased_role,
        } if aliased_role == "profile_face_cascade"
    ));

    let mut face_aliases_common = input.clone();
    face_aliases_common["assets"]["face_perception"]["frontal_face_cascade"]["source_path"] =
        face_aliases_common["assets"]["navigation_shadow_source_path"].clone();
    let error = render_bundle(
        &write_input(&root, &face_aliases_common),
        RenderMode::DryRun,
    );
    assert!(matches!(
        error,
        Err(RenderError::QualificationFaceAssetContentAlias {
            face: QualificationFaceCascadeRole::Frontal,
            ref aliased_role,
        }) if aliased_role == "navigation_shadow"
    ));

    let mut face_aliases_head = input.clone();
    face_aliases_head["assets"]["face_perception"]["profile_face_cascade"]["source_path"] =
        face_aliases_head["assets"]["head_gaze_policy_source_path"].clone();
    let error = render_bundle(&write_input(&root, &face_aliases_head), RenderMode::DryRun);
    assert!(matches!(
        error,
        Err(RenderError::QualificationFaceAssetContentAlias {
            face: QualificationFaceCascadeRole::Profile,
            ref aliased_role,
        }) if aliased_role == "head_gaze_policy"
    ));

    let mut aliased_head_content = input.clone();
    aliased_head_content["assets"]["head_gaze_policy_source_path"] =
        aliased_head_content["assets"]["navigation_shadow_source_path"].clone();
    let error = render_bundle(
        &write_input(&root, &aliased_head_content),
        RenderMode::DryRun,
    )
    .expect_err("head-gaze policy content cannot alias another role");
    assert!(
        error
            .to_string()
            .contains("head-gaze policy content aliases staged role navigation_shadow")
    );

    let mut aliased_face_destination = input;
    aliased_face_destination["assets"]["face_perception"]["frontal_face_cascade"]["destination_relative_path"] =
        aliased_face_destination["assets"]["superpoint_model"]["destination_relative_path"].clone();
    let error = render_bundle(
        &write_input(&root, &aliased_face_destination),
        RenderMode::DryRun,
    )
    .expect_err("face destination cannot alias another asset");
    assert!(
        error
            .to_string()
            .contains("multiple assets target models/superpoint.onnx")
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
    prepare_production_assets(&root, &mut input);
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
    let assets = input["assets"].as_object_mut().expect("fixture assets");
    assets.remove("face_perception");
    assets.remove("head_gaze_policy_source_path");
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
fn production_v1_rejects_the_qualification_only_head_gaze_input() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    let head_gaze_policy = input["assets"]["head_gaze_policy_source_path"].clone();
    prepare_production_assets(&root, &mut input);
    input["assets"]["head_gaze_policy_source_path"] = head_gaze_policy;
    input["bundle"] = json!({
        "kind": "production",
        "production_controller_profile_path": null
    });
    let error = render_bundle(&write_input(&root, &input), RenderMode::DryRun)
        .expect_err("production must reject the qualification-only policy input");
    assert!(
        error
            .to_string()
            .contains("qualification-only head-gaze policy input is forbidden")
    );
}

#[test]
fn production_v1_rejects_explicit_null_head_gaze_input() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    prepare_production_assets(&root, &mut input);
    input["assets"]["head_gaze_policy_source_path"] = Value::Null;
    input["bundle"] = json!({
        "kind": "production",
        "production_controller_profile_path": null
    });

    let error = render_bundle(&write_input(&root, &input), RenderMode::DryRun)
        .expect_err("production must reject presence even when the field is null");
    assert!(
        error
            .to_string()
            .contains("qualification-only head-gaze policy input is forbidden")
    );
}

#[test]
fn production_open_cv_roles_require_the_observed_nano_sonames() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    prepare_production_assets(&root, &mut input);
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
    prepare_production_assets(&root, &mut input);
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
    assert!(!destination.join("evidence/render-input-v4.json").exists());
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
    let navigation = fs::read(destination.join("navigation-shadow-v2.json")).unwrap();
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
        sha256_hex(&fs::read(destination.join("artifacts/plant/reviewed-plant-v1.json")).unwrap())
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
    assert_eq!(launch["oak"]["maximum_usb_speed"], "SUPER");
    assert_eq!(launch["oak"]["minimum_usb_speed"], "SUPER");
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
fn production_rejects_semantically_invalid_plant_json_before_staging() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    prepare_production_assets(&root, &mut input);
    input["discovery"]["stm32"]["firmware_build_id"] = json!(196609);
    input["discovery"]["stm32"]["hardware_profile_fingerprint_hex"] =
        json!("4b494b4f2d3450574d2d50524f443121");
    input["discovery"]["stm32"]["capabilities_bits"] = json!(255);
    let plant_path = input["assets"]["plant"]["source_path"]
        .as_str()
        .expect("production plant source path");
    let mut plant: Value =
        serde_json::from_slice(&fs::read(plant_path).expect("read production plant"))
            .expect("production plant is JSON");
    plant["left"]["time_constant_s"] = json!(0.0);
    fs::write(
        plant_path,
        serde_json::to_vec_pretty(&plant).expect("serialize invalid semantic plant"),
    )
    .expect("write invalid semantic plant");
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

    let destination = root.join("must-not-stage-invalid-plant");
    let error = render_bundle(
        &write_input(&root, &input),
        RenderMode::Stage {
            destination: &destination,
        },
    )
    .expect_err("valid JSON with an invalid plant domain must fail");
    assert!(matches!(
        error,
        RenderError::OfflineNavigationGraph(OfflineNavigationGraphParseError::Plant(_))
    ));
    assert!(
        !destination.exists(),
        "plant admission must precede the first staging write"
    );
}

#[test]
fn production_rejects_generated_actuation_without_control_margin_before_staging() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    prepare_production_assets(&root, &mut input);
    input["discovery"]["stm32"]["firmware_build_id"] = json!(196609);
    input["discovery"]["stm32"]["hardware_profile_fingerprint_hex"] =
        json!("4b494b4f2d3450574d2d50524f443121");
    input["discovery"]["stm32"]["capabilities_bits"] = json!(255);
    let mut profile = valid_production_profile();
    profile["actuation"]["apply_ack_budget_ns"] = json!(45_000_000_u64);
    let profile_path = root.join("production-profile-without-control-margin.json");
    fs::write(
        &profile_path,
        serde_json::to_vec_pretty(&profile).expect("serialize production profile"),
    )
    .expect("write production profile");
    input["bundle"] = json!({
        "kind": "production",
        "production_controller_profile_path": profile_path
    });

    let destination = root.join("must-not-stage-invalid-actuation");
    let error = render_bundle(
        &write_input(&root, &input),
        RenderMode::Stage {
            destination: &destination,
        },
    )
    .expect_err("generated actuation must leave strict control-period margin");
    assert!(matches!(
        error,
        RenderError::OfflineNavigationGraph(OfflineNavigationGraphParseError::Actuation(_))
    ));
    assert!(
        !destination.exists(),
        "actuation admission must precede the first staging write"
    );
}

#[test]
fn production_rejects_mpc_pwm_outside_controller_envelope_before_staging() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    prepare_production_assets(&root, &mut input);
    input["discovery"]["stm32"]["firmware_build_id"] = json!(196609);
    input["discovery"]["stm32"]["hardware_profile_fingerprint_hex"] =
        json!("4b494b4f2d3450574d2d50524f443121");
    input["discovery"]["stm32"]["capabilities_bits"] = json!(255);

    let plant_path = input["assets"]["plant"]["source_path"]
        .as_str()
        .expect("production plant source path");
    let mut plant: Value =
        serde_json::from_slice(&fs::read(plant_path).expect("read production plant"))
            .expect("production plant is JSON");
    plant["validity"]["left_pwm_min_percent"] = json!(-31);
    plant["validity"]["left_pwm_max_percent"] = json!(31);
    plant["validity"]["left_velocity_min_mps"] = json!(-0.32);
    plant["validity"]["left_velocity_max_mps"] = json!(0.32);
    fs::write(
        plant_path,
        serde_json::to_vec_pretty(&plant).expect("serialize widened production plant"),
    )
    .expect("write widened production plant");

    let navigation_path = input["assets"]["navigation_shadow_source_path"]
        .as_str()
        .expect("production navigation source path");
    let mut navigation: Value =
        serde_json::from_slice(&fs::read(navigation_path).expect("read production navigation"))
            .expect("production navigation is JSON");
    navigation["plant_model"] = plant;
    navigation["mpc"]["left_pwm_min_percent"] = json!(-31);
    navigation["mpc"]["left_pwm_max_percent"] = json!(31);
    fs::write(
        navigation_path,
        serde_json::to_vec_pretty(&navigation).expect("serialize widened production navigation"),
    )
    .expect("write widened production navigation");

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

    let destination = root.join("must-not-stage-mpc-outside-controller-envelope");
    let error = render_bundle(
        &write_input(&root, &input),
        RenderMode::Stage {
            destination: &destination,
        },
    )
    .expect_err("MPC PWM range must fit the exact controller contract");
    assert!(matches!(
        error,
        RenderError::OfflineNavigationGraph(OfflineNavigationGraphParseError::ControllerBinding(
            ProductionNavigationControllerBindingError::MpcPwmOutsideControllerEnvelope {
                wheel: kiko_slam::navigation::mpc::WheelSide::Left,
                configured_min_percent: -31,
                configured_max_percent: 31,
                controller_max_abs_percent: 30,
            }
        ))
    ));
    assert!(
        !destination.exists(),
        "controller PWM binding must precede the first staging write"
    );
}

#[test]
fn production_rejects_control_period_without_controller_rate_margin_before_staging() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    prepare_production_assets(&root, &mut input);
    input["discovery"]["stm32"]["firmware_build_id"] = json!(196609);
    input["discovery"]["stm32"]["hardware_profile_fingerprint_hex"] =
        json!("4b494b4f2d3450574d2d50524f443121");
    input["discovery"]["stm32"]["capabilities_bits"] = json!(255);
    let mut profile = valid_production_profile();
    profile["controller"]["maximum_host_command_rate_hz"] = json!(10);
    let profile_path = root.join("production-profile-with-slow-controller-rate.json");
    fs::write(
        &profile_path,
        serde_json::to_vec_pretty(&profile).expect("serialize production profile"),
    )
    .expect("write production profile");
    input["bundle"] = json!({
        "kind": "production",
        "production_controller_profile_path": profile_path
    });

    let destination = root.join("must-not-stage-controller-rate-without-margin");
    let error = render_bundle(
        &write_input(&root, &input),
        RenderMode::Stage {
            destination: &destination,
        },
    )
    .expect_err("control cadence must strictly clear the controller interval and guard");
    assert!(matches!(
        error,
        RenderError::OfflineNavigationGraph(OfflineNavigationGraphParseError::ControllerBinding(
            ProductionNavigationControllerBindingError::ControlPeriodHasNoControllerRateMargin { .. }
        ))
    ));
    assert!(
        !destination.exists(),
        "controller cadence binding must precede the first staging write"
    );
}

#[test]
fn candidate_controller_identity_cannot_be_relabelled_as_production() {
    let (temporary, mut input) = source_fixture();
    let root = canonical_root(&temporary);
    prepare_production_assets(&root, &mut input);
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
    prepare_production_assets(&root, &mut input);
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
    prepare_production_assets(&root, &mut input);
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
    input["head_policy"]["reviewed_natural_target_ticks"][0] = json!(2175);
    let input_path = write_input(&root, &input);
    let error = render_bundle(&input_path, RenderMode::DryRun).expect_err("unreviewed head target");
    assert!(
        error
            .to_string()
            .contains("operator-confirmed natural target")
    );

    input["head_policy"]["reviewed_natural_target_ticks"][0] = json!(2174);
    input["head_policy"]["maximum_travel_ticks"][0] = json!(49);
    let input_path = write_input(&root, &input);
    let error = render_bundle(&input_path, RenderMode::DryRun)
        .expect_err("widened head travel must not be admitted");
    assert!(error.to_string().contains("reviewed startup, travel"));

    input["head_policy"]["maximum_travel_ticks"][0] = json!(48);
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

#[test]
fn qualification_v4_template_renders_exact_policy_and_leaves_only_evidence_boundaries() {
    let template = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../configs/nano-wheels-off-qualification-template/bundle-render-input-v4.json.template"
    ));
    assert_eq!(
        unresolved_tokens(template),
        BTreeSet::from([
            "${ACTUATOR_CONFIG_FINGERPRINT_HEX}",
            "${CALIBRATION_ARTIFACT_ID}",
            "${CALIBRATION_ARTIFACT_RELATIVE_PATH}",
            "${CALIBRATION_SOURCE_ABSOLUTE_PATH}",
            "${CONTROLLER_CAPABILITIES_BITS}",
            "${CONTROLLER_UID_HEX}",
            "${DEPTHAI_CORE_SOURCE_ABSOLUTE_PATH}",
            "${DEPTHAI_HEADER_BOOTLOADER_ARTIFACT_VERSION}",
            "${DEPTHAI_HEADER_DEVICE_ARTIFACT_VERSION}",
            "${DEPTHAI_HEADER_SDK_COMMIT}",
            "${DEPTHAI_HEADER_SDK_VERSION}",
            "${DYNAMIC_CALIBRATION_SOURCE_ABSOLUTE_PATH}",
            "${EYE_CAPABILITIES_BITS}",
            "${EYE_DEVICE_UID_HEX}",
            "${EYE_FIRMWARE_BUILD_ID_HEX}",
            "${EYE_KEP_PROTOCOL_VERSION}",
            "${EYE_SERIAL_BY_ID_PATH}",
            "${FIRMWARE_ABI}",
            "${FIRMWARE_BUILD_ID}",
            "${FRONTAL_FACE_CASCADE_SOURCE_ABSOLUTE_PATH}",
            "${HEAD_BAUD_RATE_BPS}",
            "${HEAD_BOW_SERVO_ID}",
            "${HEAD_CURL_SERVO_ID}",
            "${HEAD_DTR_ASSERTED}",
            "${HEAD_ROLL_SERVO_ID}",
            "${HEAD_RTS_ASSERTED}",
            "${HEAD_SERIAL_BY_ID_PATH}",
            "${HEAD_YAW_SERVO_ID}",
            "${LIBUSB_1_0_SOURCE_ABSOLUTE_PATH}",
            "${LIGHTGLUE_MODEL_SOURCE_ABSOLUTE_PATH}",
            "${NAVIGATION_SHADOW_SOURCE_ABSOLUTE_PATH}",
            "${OAK_MXID}",
            "${ONNXRUNTIME_SOURCE_ABSOLUTE_PATH}",
            "${OPENCV_CORE_SOURCE_ABSOLUTE_PATH}",
            "${OPENCV_IMGPROC_SOURCE_ABSOLUTE_PATH}",
            "${OPENCV_OBJDETECT_SOURCE_ABSOLUTE_PATH}",
            "${PLANT_SOURCE_ABSOLUTE_PATH}",
            "${PROFILE_FACE_CASCADE_SOURCE_ABSOLUTE_PATH}",
            "${QUALIFICATION_EXECUTABLE_SOURCE_ABSOLUTE_PATH}",
            "${ROBOT_ID}",
            "${STM32_SERIAL_BY_ID_PATH}",
            "${SUPERPOINT_MODEL_SOURCE_ABSOLUTE_PATH}",
        ])
    );

    let shape = template_shape_with_unquoted_tokens_replaced_by_null(template);
    assert_eq!(
        shape["runtime"],
        json!({
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
                "maximum_keyframes": 4096,
                "snapshot_every_keyframes": 20
            },
            "inference": {
                "superpoint_backend": "cpu",
                "lightglue_backend": "cpu",
                "downscale_factor": 2,
                "maximum_keypoints": 512
            },
            "rerun": {
                "decimation": 2,
                "memory_limit_bytes": 134217728,
                "flush_timeout_ms": 2000
            },
            "storage": {
                "maximum_map_snapshot_bytes": 67108864,
                "minimum_free_bytes_after_map_save": 536870912,
                "maximum_navigation_dataset_bytes": 4294967296_u64,
                "maximum_navigation_dataset_files": 8192,
                "maximum_navigation_ingress_records": 100000,
                "minimum_free_bytes_after_navigation_dataset_write": 1073741824,
                "navigation_dataset_terminal_reserve_bytes": 268435456,
                "warm_start": {"kind": "none"}
            }
        })
    );
    assert_eq!(
        shape["head_policy"]["reviewed_natural_target_ticks"],
        json!([2174, 2570, 1637, 3047])
    );
    assert_eq!(
        shape["rgb_expression_policy"]["head_origin_in_camera_m"],
        json!([0.0, -0.25, -0.20])
    );
    assert_eq!(
        shape["assets"]["plant"]["artifact_id"],
        "qualification-shadow-only-synthetic-unvalidated-v2"
    );
    assert_eq!(
        shape["assets"]["plant"]["destination_relative_path"],
        "artifacts/plant/qualification-shadow-only-synthetic-unvalidated-plant-v2.json"
    );

    let (temporary, fixture) = source_fixture();
    let root = canonical_root(&temporary);
    let materialized = materialize_v4_template(template, &fixture);
    assert!(
        materialized["assets"]
            .as_object()
            .expect("materialized assets")
            .get("head_gaze_policy_source_path")
            .is_none(),
        "Gate A disables proposal-only head gaze by field absence"
    );
    let input_path = write_input(&root, &materialized);
    let destination = root.join("checked-in-v4-template-bundle");
    let plan = render_bundle(
        &input_path,
        RenderMode::Stage {
            destination: &destination,
        },
    )
    .expect("the materialized checked-in V4 template must render");
    assert!(
        plan.files
            .iter()
            .all(|file| file.relative_path != "head-gaze-policy-v1.json")
    );

    let retained_input: Value = serde_json::from_slice(
        &fs::read(destination.join("evidence/render-input-v4.json"))
            .expect("retained V4 render input"),
    )
    .expect("retained V4 render input JSON");
    assert_eq!(retained_input, materialized);

    let policy_bytes =
        fs::read(destination.join("agent-policy-v3.json")).expect("rendered agent policy");
    NanoAgentPolicyConfigV3::parse_json(&policy_bytes).expect("typed rendered agent policy");
    let policy: Value = serde_json::from_slice(&policy_bytes).expect("rendered agent policy JSON");
    assert_eq!(
        policy["head"],
        json!({
            "mode": "return_to_natural_and_hold_continuously",
            "device_path": fixture["discovery"]["head"]["adapter_serial_by_id_path"],
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
            "minimum_start_ticks": [2133, 2550, 1617, 3023],
            "maximum_start_ticks": [2194, 2660, 1852, 3067],
            "reviewed_natural_target_ticks": [2174, 2570, 1637, 3047],
            "maximum_travel_ticks": [48, 96, 224, 32],
            "physical_torque_consent": "enable_for_reviewed_natural_return_and_hold",
            "physical_motion_consent": "return_to_reviewed_natural_target"
        })
    );
    assert_eq!(
        policy["rgb_expression"],
        json!({
            "mode": "scene_motion",
            "sampling_columns": 16,
            "sampling_rows": 12,
            "minimum_residual_luma": 24,
            "minimum_active_fraction_basis_points": 500,
            "frame_freshness_ms": 80,
            "brightness_basis_points": 7000,
            "color_rgb": [32, 128, 255],
            "blink": false,
            "gaze_geometry": {
                "schema_version": 1,
                "head_origin_in_camera_m": [0.0, -0.25, -0.20],
                "neutral_head_from_camera_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
            }
        })
    );
    assert_eq!(
        policy["live_mode_policy"],
        json!({
            "startup": "disarmed_map_only",
            "manual": {"permission": "disabled"},
            "point_goal": {"permission": "disabled"},
            "frontier_explore": {"permission": "disabled"}
        })
    );

    let launch_bytes = fs::read(destination.join("nano-wheels-off-qualification-launch-v4.json"))
        .expect("rendered qualification launch");
    NanoWheelsOffQualificationLaunchV4::parse_json(&launch_bytes)
        .expect("typed rendered qualification launch");
    let launch: Value =
        serde_json::from_slice(&launch_bytes).expect("rendered qualification launch JSON");
    assert!(launch.get("head_gaze_policy_asset").is_none());
    assert_eq!(
        launch["oak"],
        json!({
            "selector_source": "exact_inventory_oak_mxid",
            "maximum_usb_speed": "SUPER",
            "minimum_usb_speed": "SUPER",
            "rgb": {"width_px": 640, "height_px": 400, "fps": 15},
            "rectified_stereo": {
                "width_px": 640,
                "height_px": 400,
                "fps": 15,
                "rectified": true
            },
            "depth": {
                "width_px": 640,
                "height_px": 400,
                "fps": 15,
                "alignment": "rectified_left"
            },
            "imu": {"rate_hz": 200},
            "queue": {"size": 4, "blocking": false}
        })
    );
    assert_eq!(launch["occupancy"], shape["runtime"]["occupancy"]);
    assert_eq!(
        launch["rerun"],
        json!({
            "kind": "serve_loopback",
            "bind": "127.0.0.1:9876",
            "decimation": 2,
            "memory_limit_bytes": 134217728,
            "flush_timeout_ms": 2000
        })
    );
    assert_eq!(
        launch["storage"],
        json!({
            "map_snapshot_relative_path": "maps/current.kmap",
            "navigation_dataset_directory_relative_path": "navigation",
            "maximum_map_snapshot_bytes": 67108864,
            "minimum_free_bytes_after_map_save": 536870912,
            "maximum_navigation_dataset_bytes": 4294967296_u64,
            "maximum_navigation_dataset_files": 8192,
            "maximum_navigation_ingress_records": 100000,
            "minimum_free_bytes_after_navigation_dataset_write": 1073741824,
            "navigation_dataset_terminal_reserve_bytes": 268435456
        })
    );
    assert_eq!(launch["inference"]["superpoint_backend"], "cpu");
    assert_eq!(launch["inference"]["lightglue_backend"], "cpu");
    assert_eq!(launch["inference"]["downscale_factor"], 2);
    assert_eq!(launch["inference"]["maximum_keypoints"], 512);
    assert_eq!(
        launch["inference"]["superpoint_model_asset"]["relative_path"],
        "models/sp.onnx"
    );
    assert_eq!(
        launch["inference"]["lightglue_model_asset"]["relative_path"],
        "models/lg.onnx"
    );
    assert_eq!(
        launch["plant_artifact"]["artifact_id"],
        "qualification-shadow-only-synthetic-unvalidated-v2"
    );
    assert_eq!(
        launch["plant_artifact"]["asset"]["relative_path"],
        "artifacts/plant/qualification-shadow-only-synthetic-unvalidated-plant-v2.json"
    );
    assert_eq!(
        launch["face_perception"]["frontal_face_cascade_asset"]["relative_path"],
        "models/opencv/haarcascade_frontalface_default.xml"
    );
    assert_eq!(
        launch["face_perception"]["profile_face_cascade_asset"]["relative_path"],
        "models/opencv/haarcascade_profileface.xml"
    );

    let native_bytes =
        fs::read(destination.join("native-runtime-v1.json")).expect("rendered native manifest");
    NanoWheelsOffNativeRuntimeV1::parse_json(&native_bytes)
        .expect("typed rendered native manifest");
    let native: Value =
        serde_json::from_slice(&native_bytes).expect("rendered native manifest JSON");
    for (role, soname) in [
        ("depthai_core", "libdepthai-core.so"),
        ("dynamic_calibration", "libdynamic_calibration.so"),
        ("libusb_1_0", "libusb-1.0.so"),
        ("onnxruntime", "libonnxruntime.so.1"),
        ("opencv_core", "libopencv_core.so.4.5d"),
        ("opencv_imgproc", "libopencv_imgproc.so.4.5d"),
        ("opencv_objdetect", "libopencv_objdetect.so.4.5d"),
    ] {
        let library = native["libraries"]
            .as_array()
            .expect("native libraries")
            .iter()
            .find(|library| library["role"] == role)
            .expect("fixed native role");
        assert_eq!(library["soname"], soname);
        assert_eq!(library["relative_path"], format!("lib/{soname}"));
    }
    restore_writable(&destination);
}
