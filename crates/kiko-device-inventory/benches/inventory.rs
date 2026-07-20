use std::hint::black_box;
use std::time::{Duration, Instant};

use kiko_device_inventory::{
    ArtifactDigestDto, DEVICE_INVENTORY_MANIFEST_V1, DeviceInventoryManifestV1,
    DeviceInventoryManifestV1Dto, EyeManifestV1Dto, HeadManifestV1Dto, InventoryComparison,
    OBSERVED_DEVICE_INVENTORY_V1, OakManifestV1Dto, ObservedDeviceInventoryV1,
    ObservedDeviceInventoryV1Dto, ObservedEyeV1Dto, ObservedHeadV1Dto, ObservedStm32V1Dto,
    Stm32ManifestV1Dto,
};
use robot_protocol::v2::{ControllerCapabilities, VERSION as ROBOT_PROTOCOL_VERSION};

fn artifact(id: &str, byte: u8) -> ArtifactDigestDto {
    ArtifactDigestDto {
        artifact_id: id.into(),
        sha256: [byte; 32],
    }
}

fn oak() -> OakManifestV1Dto {
    OakManifestV1Dto {
        mxid: "A1B2C3D4E5F60708".into(),
        runtime_provenance: "depthai-runtime@2.29.0".into(),
        sdk_build_provenance: "depthai-core@2.29.0+abc123".into(),
        adapter_build_provenance: "kiko-oak-adapter@abc123".into(),
    }
}

fn expected_dto() -> DeviceInventoryManifestV1Dto {
    DeviceInventoryManifestV1Dto {
        schema_version: DEVICE_INVENTORY_MANIFEST_V1,
        robot_id: "kiko-production-01".into(),
        oak: oak(),
        stm32: Stm32ManifestV1Dto {
            serial_by_id_path: "/dev/serial/by-id/usb-Kiko_STM32_A1-if00".into(),
            control_endpoint_identity: "unix:/run/kiko/robot-v2.sock".into(),
            controller_uid: [0x11; 12],
            firmware_abi: u16::from(ROBOT_PROTOCOL_VERSION),
            firmware_build_id: 0x1020_3040,
            hardware_profile_fingerprint: [0x12; 16],
            capabilities_bits: ControllerCapabilities::REQUIRED_BITS,
        },
        head: Some(HeadManifestV1Dto {
            adapter_serial_by_id_path: "/dev/serial/by-id/usb-FTDI_Kiko_Head-if00-port0".into(),
            bow_servo_id: 1,
            curl_servo_id: 2,
            yaw_servo_id: 3,
            roll_servo_id: 4,
            baud_rate_bps: kiko_head_protocol::BUS_BAUD_RATE_BPS,
            dtr_asserted: kiko_head_protocol::ADAPTER_DTR_ASSERTED,
            rts_asserted: kiko_head_protocol::ADAPTER_RTS_ASSERTED,
        }),
        eye: Some(EyeManifestV1Dto {
            serial_by_id_path: "/dev/serial/by-id/usb-Kiko_Eye_E1-if00".into(),
            kep_protocol_version: kiko_eye_protocol::PROTOCOL_VERSION,
            device_uid: [0x21; 16],
            firmware_build_id: [0x22; 32],
            capabilities_bits: kiko_eye_protocol::Capabilities::KNOWN_BITS,
        }),
        calibration_artifacts: vec![
            artifact("oak-camera-calibration-v3", 0x31),
            artifact("head-servo-calibration-v2", 0x32),
        ],
        plant_artifacts: vec![artifact("differential-drive-plant-v4", 0x41)],
    }
}

fn observed_dto() -> ObservedDeviceInventoryV1Dto {
    ObservedDeviceInventoryV1Dto {
        schema_version: OBSERVED_DEVICE_INVENTORY_V1,
        robot_id: "kiko-production-01".into(),
        oak: Some(oak()),
        stm32: Some(ObservedStm32V1Dto {
            serial_by_id_path: "/dev/serial/by-id/usb-Kiko_STM32_A1-if00".into(),
            control_endpoint_identity: "unix:/run/kiko/robot-v2.sock".into(),
            controller_uid: [0x11; 12],
            controller_boot_id: 71,
            firmware_abi: u16::from(ROBOT_PROTOCOL_VERSION),
            firmware_build_id: 0x1020_3040,
            hardware_profile_fingerprint: [0x12; 16],
            capabilities_bits: ControllerCapabilities::REQUIRED_BITS,
        }),
        head: Some(ObservedHeadV1Dto {
            adapter_serial_by_id_path: "/dev/serial/by-id/usb-FTDI_Kiko_Head-if00-port0".into(),
            baud_rate_bps: kiko_head_protocol::BUS_BAUD_RATE_BPS,
            dtr_asserted: kiko_head_protocol::ADAPTER_DTR_ASSERTED,
            rts_asserted: kiko_head_protocol::ADAPTER_RTS_ASSERTED,
            responding_servo_ids: vec![4, 3, 2, 1],
        }),
        eye: Some(ObservedEyeV1Dto {
            serial_by_id_path: "/dev/serial/by-id/usb-Kiko_Eye_E1-if00".into(),
            kep_protocol_version: kiko_eye_protocol::PROTOCOL_VERSION,
            device_uid: [0x21; 16],
            firmware_build_id: [0x22; 32],
            device_boot_id: 81,
            capabilities_bits: kiko_eye_protocol::Capabilities::KNOWN_BITS,
        }),
        calibration_artifacts: vec![
            artifact("head-servo-calibration-v2", 0x32),
            artifact("oak-camera-calibration-v3", 0x31),
        ],
        plant_artifacts: vec![artifact("differential-drive-plant-v4", 0x41)],
    }
}

fn elapsed_per_iteration(elapsed: Duration, iterations: u32) -> Duration {
    elapsed / iterations
}

fn main() {
    let iterations = std::env::args()
        .nth(1)
        .and_then(|argument| argument.parse::<u32>().ok())
        .filter(|value| *value != 0)
        .unwrap_or(100_000);

    let expected = DeviceInventoryManifestV1::parse(expected_dto()).expect("benchmark manifest");
    let observed = ObservedDeviceInventoryV1::parse(observed_dto()).expect("benchmark observation");

    let started = Instant::now();
    for _ in 0..iterations {
        let comparison = InventoryComparison::compare(black_box(&expected), black_box(&observed));
        black_box(comparison);
    }
    let compare_elapsed = started.elapsed();

    let started = Instant::now();
    for _ in 0..iterations {
        let parsed_expected = DeviceInventoryManifestV1::parse(black_box(expected_dto()))
            .expect("benchmark manifest");
        let parsed_observed = ObservedDeviceInventoryV1::parse(black_box(observed_dto()))
            .expect("benchmark observation");
        let comparison = InventoryComparison::compare(&parsed_expected, &parsed_observed);
        black_box(comparison);
    }
    let parse_compare_elapsed = started.elapsed();

    println!("iterations: {iterations}");
    println!(
        "already-parsed comparison: {compare_elapsed:?} total, {:?}/iteration",
        elapsed_per_iteration(compare_elapsed, iterations)
    );
    println!(
        "DTO construction + parse + comparison: {parse_compare_elapsed:?} total, {:?}/iteration",
        elapsed_per_iteration(parse_compare_elapsed, iterations)
    );
}
