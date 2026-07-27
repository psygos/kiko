#![cfg(feature = "agent-runtime")]

const NAVIGATION_TEMPLATE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../configs/nano-wheels-off-qualification-template/navigation-shadow-preparation-v1.json.template"
));
const SHADOW_PLANT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../configs/nano-wheels-off-qualification-template/qualification-shadow-only-synthetic-unvalidated-plant-v1.json"
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

#[test]
fn gate_a_navigation_embeds_the_exact_separate_synthetic_shadow_plant() {
    let embedded: serde_json::Value =
        serde_json::from_str(embedded_plant_json()).expect("embedded plant is exact JSON");
    let separate: serde_json::Value =
        serde_json::from_str(SHADOW_PLANT).expect("separate shadow plant is exact JSON");

    assert_eq!(embedded, separate);
    assert_eq!(embedded["evidence"]["kind"], "synthetic_fixture");
}

#[test]
fn gate_a_navigation_does_not_request_pre_wheel_physical_plant_claims() {
    assert!(!NAVIGATION_TEMPLATE.contains("claimed_physical_identification"));
    assert!(!NAVIGATION_TEMPLATE.contains("NAV_SHADOW_UNVALIDATED_PLANT_"));
    assert!(
        NAVIGATION_TEMPLATE.contains("${CALIBRATION_PREPARER_REPLACES_TRACKING_CAMERA_TO_BASE}")
    );
    assert!(NAVIGATION_TEMPLATE.contains("${CALIBRATION_PREPARER_REPLACES_RAW_IMU_CALIBRATION}"));
}
