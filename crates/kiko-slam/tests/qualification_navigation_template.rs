#![cfg(feature = "agent-runtime")]

use std::collections::BTreeSet;

const NAVIGATION_TEMPLATE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../configs/nano-wheels-off-qualification-template/navigation-shadow-preparation-v1.json.template"
));
const SHADOW_PLANT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../configs/nano-wheels-off-qualification-template/qualification-shadow-only-synthetic-unvalidated-plant-v2.json"
));

fn embedded_plant_json() -> &'static str {
    let start_marker = "\"plant_model\": ";
    let start = NAVIGATION_TEMPLATE
        .find(start_marker)
        .expect("navigation preparation template has a plant_model")
        + start_marker.len();
    let remaining = &NAVIGATION_TEMPLATE[start..];
    let end = remaining
        .find(",\n  \"mpc\":")
        .expect("plant_model is followed by the mpc declaration");
    &remaining[..end]
}

fn unresolved_tokens(input: &str) -> BTreeSet<&str> {
    let mut tokens = BTreeSet::new();
    let mut remaining = input;
    while let Some(start) = remaining.find("${") {
        remaining = &remaining[start..];
        let end = remaining
            .find('}')
            .expect("every preparation token must close");
        tokens.insert(&remaining[..=end]);
        remaining = &remaining[end + 1..];
    }
    tokens
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn materialize_test_only_navigation() -> String {
    // These replacements exist only to exercise the authoritative parser.
    // Identity transforms and nominal geometry are synthetic fixtures, not
    // measurements, calibration evidence, or deployable robot parameters.
    let raw_imu = r#"{
      "format_version": 1,
      "source_id": "synthetic-test-only-not-calibration",
      "content_id": "synthetic-test-only-not-calibration-v1",
      "gyro_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
      "gyro_bias_native_rad_per_sec": [0.0, 0.0, 0.0],
      "accel_affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
      "accel_bias_native_m_per_sec2": [0.0, 0.0, 0.0],
      "native_imu_to_base_rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }"#;
    let tracking_camera_to_base = r#"{
      "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
      "translation_m": [0.0, 0.0, 0.0]
    }"#;
    let replacements = [
        (
            "\"${CALIBRATION_PREPARER_REPLACES_TRACKING_CAMERA_TO_BASE}\"",
            tracking_camera_to_base,
        ),
        (
            "\"${CALIBRATION_PREPARER_REPLACES_RAW_IMU_CALIBRATION}\"",
            raw_imu,
        ),
        (
            "${NAV_SHADOW_UNVALIDATED_WORLD_TO_OCCUPANCY_ROTATION_F64_MATRIX}",
            "[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]",
        ),
        (
            "${NAV_SHADOW_UNVALIDATED_WORLD_TO_OCCUPANCY_TRANSLATION_M_F64_ARRAY}",
            "[0.0, 0.0, 0.0]",
        ),
        (
            "${NAV_SHADOW_UNVALIDATED_OBSTACLE_HEIGHT_MINIMUM_M}",
            "-0.5",
        ),
        ("${NAV_SHADOW_UNVALIDATED_OBSTACLE_HEIGHT_MAXIMUM_M}", "1.5"),
        ("${NAV_SHADOW_UNVALIDATED_FOOTPRINT_RADIUS_M}", "0.2"),
    ];

    let mut rendered = NAVIGATION_TEMPLATE.to_owned();
    for (token, replacement) in replacements {
        assert_eq!(
            rendered.matches(token).count(),
            1,
            "test materialization requires exactly one {token}"
        );
        rendered = rendered.replace(token, replacement);
    }
    assert!(
        !rendered.contains("${"),
        "test materialization must replace every preparation token"
    );
    rendered
}

#[test]
fn gate_a_navigation_embeds_the_exact_separate_synthetic_shadow_plant() {
    let embedded: serde_json::Value =
        serde_json::from_str(embedded_plant_json()).expect("embedded plant is exact JSON");
    let separate: serde_json::Value =
        serde_json::from_str(SHADOW_PLANT).expect("separate shadow plant is exact JSON");

    assert_eq!(embedded, separate);
    assert_eq!(embedded["evidence"]["kind"], "synthetic_fixture");
    assert_eq!(embedded["model_version"], 2);
    assert_eq!(embedded["sample_period_s"], 0.05);
}

#[test]
fn gate_a_navigation_does_not_request_pre_wheel_physical_plant_claims() {
    assert!(!NAVIGATION_TEMPLATE.contains("claimed_physical_identification"));
    assert!(!NAVIGATION_TEMPLATE.contains("NAV_SHADOW_UNVALIDATED_PLANT_"));
    assert_eq!(
        unresolved_tokens(NAVIGATION_TEMPLATE),
        BTreeSet::from([
            "${CALIBRATION_PREPARER_REPLACES_RAW_IMU_CALIBRATION}",
            "${CALIBRATION_PREPARER_REPLACES_TRACKING_CAMERA_TO_BASE}",
            "${NAV_SHADOW_UNVALIDATED_FOOTPRINT_RADIUS_M}",
            "${NAV_SHADOW_UNVALIDATED_OBSTACLE_HEIGHT_MAXIMUM_M}",
            "${NAV_SHADOW_UNVALIDATED_OBSTACLE_HEIGHT_MINIMUM_M}",
            "${NAV_SHADOW_UNVALIDATED_WORLD_TO_OCCUPANCY_ROTATION_F64_MATRIX}",
            "${NAV_SHADOW_UNVALIDATED_WORLD_TO_OCCUPANCY_TRANSLATION_M_F64_ARRAY}",
        ])
    );
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[test]
fn gate_a_shadow_plant_has_a_realizable_candidate_service_period() {
    use std::time::Duration;

    use kiko_slam::dense::occupancy::{DepthCameraModel, DepthToTrackingCamera};
    use kiko_slam::navigation::mpc::PlantEvidenceV1;
    use kiko_slam::navigation::{
        MAX_WHEELS_OFF_CANDIDATE_RUNTIME_SERVICE_INTERVAL, ShadowNavigationConfigV1,
    };
    use kiko_slam::{FrameDimensions, PinholeIntrinsics};
    use robot_protocol::v2::{
        MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT,
        OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT,
    };

    let camera = DepthCameraModel::new(
        PinholeIntrinsics::try_new(411.0, 412.0, 319.5, 199.5)
            .expect("synthetic test-only intrinsics are valid"),
        FrameDimensions::try_new(640, 400).expect("synthetic test-only dimensions are valid"),
        DepthToTrackingCamera::identity(),
    );
    let plant = kiko_slam::navigation::mpc::PlantModelV1::parse_json(SHADOW_PLANT.as_bytes())
        .expect("checked-in Gate-A plant must parse");
    let parsed = ShadowNavigationConfigV1::parse_json_bound_to_plant_artifact(
        materialize_test_only_navigation().as_bytes(),
        camera,
        plant,
    )
    .expect("materialized Gate-A navigation template must parse and bind its exact plant");
    let parsed_plant = parsed.mpc_solver().model();
    let mpc = parsed.mpc_solver().config();

    assert!(matches!(
        parsed_plant.evidence(),
        PlantEvidenceV1::SyntheticFixture { .. }
    ));
    let plant_period = Duration::from_secs_f64(parsed_plant.sample_period_s());
    let mpc_period = Duration::from_secs_f64(mpc.step_period_s());
    let control_period = parsed.control_period().as_duration();
    let shadow_lease = Duration::from_millis(u64::from(parsed.shadow_command().lease().get()));
    assert_eq!(plant_period, Duration::from_millis(50));
    assert_eq!(mpc_period, plant_period);
    assert_eq!(control_period, plant_period);
    assert_eq!(shadow_lease, Duration::from_millis(100));
    assert!(
        control_period <= MAX_WHEELS_OFF_CANDIDATE_RUNTIME_SERVICE_INTERVAL,
        "plant, MPC, and control periods must fit the candidate runtime service envelope"
    );

    for (minimum, maximum) in [
        mpc.left_pwm_bounds_percent(),
        mpc.right_pwm_bounds_percent(),
    ] {
        assert!(
            i16::from(minimum) >= -i16::from(MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT)
                && i16::from(maximum) <= i16::from(MAX_OPERATOR_SUPERVISED_FOUR_PWM_PWM_PERCENT),
            "parsed MPC PWM bounds must fit the candidate controller cap"
        );
    }
    let (left_slew, right_slew) = mpc.maximum_slew_percent_per_step();
    for slew in [left_slew, right_slew] {
        assert!(
            slew <= u16::from(OPERATOR_SUPERVISED_FOUR_PWM_MAX_COMMAND_STEP_PERCENT),
            "parsed MPC slew must fit the candidate controller transition invariant"
        );
    }
}
