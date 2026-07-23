use kiko_device_inventory::{
    ArtifactDigestDto, ArtifactKind, BoundedTextError, ControllerSessionClassV2Dto,
    DEVICE_INVENTORY_MANIFEST_V1, DEVICE_INVENTORY_MANIFEST_V2, DeviceInventoryManifestV1,
    DeviceInventoryManifestV1Dto, DeviceInventoryManifestV2, DeviceInventoryManifestV2Dto,
    DeviceRole, EyeManifestV1Dto, HeadManifestV1Dto, InventoryComparison, InventoryMismatch,
    InventoryParseError, MAX_ARTIFACT_ID_BYTES, MAX_BUILD_PROVENANCE_BYTES,
    MAX_CALIBRATION_ARTIFACTS, MAX_CONTROL_ENDPOINT_ID_BYTES, MAX_INVENTORY_MISMATCHES,
    MAX_OAK_MXID_BYTES, MAX_OBSERVED_HEAD_SERVOS, MAX_PLANT_ARTIFACTS, MAX_ROBOT_ID_BYTES,
    MAX_SERIAL_BY_ID_PATH_BYTES, OBSERVED_DEVICE_INVENTORY_V1, OakManifestV1Dto,
    ObservedDeviceInventoryV1, ObservedDeviceInventoryV1Dto, ObservedEyeV1Dto, ObservedHeadV1Dto,
    ObservedOakV1Dto, ObservedStm32V1Dto, PhysicalStopSemanticsV2Dto, Stm32ManifestV1Dto,
    Stm32ManifestV2Dto, TextField, admit_exact_inventory,
};
use kiko_head_protocol::HeadJoint;
use robot_protocol::v2::{
    ControllerCapabilities, ControllerSafetyClass, VERSION as ROBOT_PROTOCOL_VERSION,
};

fn digest(artifact_id: impl Into<String>, byte: u8) -> ArtifactDigestDto {
    assert_ne!(byte, 0);
    ArtifactDigestDto {
        artifact_id: artifact_id.into(),
        sha256: [byte; 32],
    }
}

fn oak() -> OakManifestV1Dto {
    OakManifestV1Dto {
        mxid: "A1B2C3D4E5F60708".into(),
        compiled_depthai_header_sdk_version: "3.6.1".into(),
        compiled_depthai_header_sdk_commit: "abc123".into(),
        compiled_depthai_header_embedded_device_artifact_version: "device-1".into(),
        compiled_depthai_header_embedded_bootloader_artifact_version: "bootloader-1".into(),
    }
}

fn observed_oak() -> ObservedOakV1Dto {
    let expected = oak();
    ObservedOakV1Dto {
        mxid: expected.mxid,
        compiled_depthai_header_sdk_version: expected.compiled_depthai_header_sdk_version,
        compiled_depthai_header_sdk_commit: expected.compiled_depthai_header_sdk_commit,
        compiled_depthai_header_embedded_device_artifact_version: expected
            .compiled_depthai_header_embedded_device_artifact_version,
        compiled_depthai_header_embedded_bootloader_artifact_version: expected
            .compiled_depthai_header_embedded_bootloader_artifact_version,
    }
}

fn stm32_manifest() -> Stm32ManifestV1Dto {
    Stm32ManifestV1Dto {
        serial_by_id_path: "/dev/serial/by-id/usb-Kiko_STM32_A1-if00".into(),
        control_endpoint_identity: "unix:/run/kiko/robot-v2.sock".into(),
        controller_uid: [0x11; 12],
        firmware_abi: u16::from(ROBOT_PROTOCOL_VERSION),
        firmware_build_id: 0x1020_3040,
        hardware_profile_fingerprint: [0x12; 16],
        capabilities_bits: ControllerCapabilities::REQUIRED_BITS,
    }
}

fn observed_stm32() -> ObservedStm32V1Dto {
    let expected = stm32_manifest();
    ObservedStm32V1Dto {
        serial_by_id_path: expected.serial_by_id_path,
        control_endpoint_identity: expected.control_endpoint_identity,
        controller_uid: expected.controller_uid,
        controller_boot_id: 71,
        firmware_abi: expected.firmware_abi,
        firmware_build_id: expected.firmware_build_id,
        hardware_profile_fingerprint: expected.hardware_profile_fingerprint,
        capabilities_bits: expected.capabilities_bits,
    }
}

fn head_manifest() -> HeadManifestV1Dto {
    HeadManifestV1Dto {
        adapter_serial_by_id_path: "/dev/serial/by-id/usb-FTDI_Kiko_Head-if00-port0".into(),
        bow_servo_id: HeadJoint::Bow.servo_id().get(),
        curl_servo_id: HeadJoint::Curl.servo_id().get(),
        yaw_servo_id: HeadJoint::Yaw.servo_id().get(),
        roll_servo_id: HeadJoint::Roll.servo_id().get(),
        baud_rate_bps: kiko_head_protocol::BUS_BAUD_RATE_BPS,
        dtr_asserted: kiko_head_protocol::ADAPTER_DTR_ASSERTED,
        rts_asserted: kiko_head_protocol::ADAPTER_RTS_ASSERTED,
    }
}

fn observed_head() -> ObservedHeadV1Dto {
    let expected = head_manifest();
    ObservedHeadV1Dto {
        adapter_serial_by_id_path: expected.adapter_serial_by_id_path,
        baud_rate_bps: expected.baud_rate_bps,
        dtr_asserted: expected.dtr_asserted,
        rts_asserted: expected.rts_asserted,
        responding_servo_ids: vec![4, 3, 2, 1],
    }
}

fn eye_manifest() -> EyeManifestV1Dto {
    EyeManifestV1Dto {
        serial_by_id_path: "/dev/serial/by-id/usb-Kiko_Eye_E1-if00".into(),
        kep_protocol_version: kiko_eye_protocol::PROTOCOL_VERSION,
        device_uid: [0x21; 16],
        firmware_build_id: [0x22; 32],
        capabilities_bits: kiko_eye_protocol::Capabilities::KNOWN_BITS,
    }
}

fn observed_eye() -> ObservedEyeV1Dto {
    let expected = eye_manifest();
    ObservedEyeV1Dto {
        serial_by_id_path: expected.serial_by_id_path,
        kep_protocol_version: expected.kep_protocol_version,
        device_uid: expected.device_uid,
        firmware_build_id: expected.firmware_build_id,
        device_boot_id: 81,
        capabilities_bits: expected.capabilities_bits,
    }
}

fn manifest_dto() -> DeviceInventoryManifestV1Dto {
    DeviceInventoryManifestV1Dto {
        schema_version: DEVICE_INVENTORY_MANIFEST_V1,
        robot_id: "kiko-production-01".into(),
        oak: oak(),
        stm32: stm32_manifest(),
        head: Some(head_manifest()),
        eye: Some(eye_manifest()),
        calibration_artifacts: vec![
            digest("oak-camera-calibration-v3", 0x31),
            digest("head-servo-calibration-v2", 0x32),
        ],
        plant_artifacts: vec![digest("differential-drive-plant-v4", 0x41)],
    }
}

fn candidate_manifest_dto() -> DeviceInventoryManifestV2Dto {
    let production = manifest_dto();
    let stm32 = production.stm32;
    DeviceInventoryManifestV2Dto {
        schema_version: DEVICE_INVENTORY_MANIFEST_V2,
        robot_id: production.robot_id,
        oak: production.oak,
        stm32: Stm32ManifestV2Dto {
            serial_by_id_path: stm32.serial_by_id_path,
            control_endpoint_identity: stm32.control_endpoint_identity,
            controller_uid: stm32.controller_uid,
            firmware_abi: u16::from(ROBOT_PROTOCOL_VERSION),
            firmware_build_id: 0x0002_1001,
            hardware_profile_fingerprint: *b"KIKO-4PWM-CAND1!",
            capabilities_bits: ControllerCapabilities::SOFTWARE_GUARD_BITS
                | ControllerCapabilities::OPERATOR_SUPERVISED_FOUR_PWM_CANDIDATE,
            controller_session_class:
                ControllerSessionClassV2Dto::OperatorSupervisedFourPwmCandidate,
            expected_max_abs_pwm_percent: 30,
            expected_physical_stop_semantics: PhysicalStopSemanticsV2Dto::Unverified,
        },
        head: production.head,
        eye: production.eye,
        calibration_artifacts: production.calibration_artifacts,
        plant_artifacts: production.plant_artifacts,
    }
}

fn candidate_observed_dto() -> ObservedDeviceInventoryV1Dto {
    let mut observed = observed_dto();
    let stm32 = observed.stm32.as_mut().expect("STM32");
    stm32.firmware_abi = u16::from(ROBOT_PROTOCOL_VERSION);
    stm32.firmware_build_id = 0x0002_1001;
    stm32.hardware_profile_fingerprint = *b"KIKO-4PWM-CAND1!";
    stm32.capabilities_bits = ControllerCapabilities::SOFTWARE_GUARD_BITS
        | ControllerCapabilities::OPERATOR_SUPERVISED_FOUR_PWM_CANDIDATE;
    observed
}

fn observed_dto() -> ObservedDeviceInventoryV1Dto {
    ObservedDeviceInventoryV1Dto {
        schema_version: OBSERVED_DEVICE_INVENTORY_V1,
        robot_id: "kiko-production-01".into(),
        oak: Some(observed_oak()),
        stm32: Some(observed_stm32()),
        head: Some(observed_head()),
        eye: Some(observed_eye()),
        calibration_artifacts: vec![
            digest("head-servo-calibration-v2", 0x32),
            digest("oak-camera-calibration-v3", 0x31),
        ],
        plant_artifacts: vec![digest("differential-drive-plant-v4", 0x41)],
    }
}

fn parse_pair() -> (DeviceInventoryManifestV1, ObservedDeviceInventoryV1) {
    (
        DeviceInventoryManifestV1::parse(manifest_dto()).expect("valid manifest fixture"),
        ObservedDeviceInventoryV1::parse(observed_dto()).expect("valid observation fixture"),
    )
}

#[test]
fn exact_inventory_matches_without_substitution() {
    let (expected, observed) = parse_pair();
    let comparison = InventoryComparison::compare(&expected, &observed);

    assert!(comparison.is_exact_match());
    assert!(comparison.is_empty());
    assert_eq!(comparison.len(), 0);
    assert_eq!(comparison.iter().count(), 0);
    assert_eq!(observed.stm32().expect("STM32").boot_id().get(), 71);
    assert_eq!(observed.eye().expect("eye").boot_id().get(), 81);
}

#[test]
fn schema_v2_candidate_manifest_carries_and_compares_the_explicit_safety_class() {
    let expected =
        DeviceInventoryManifestV2::parse(candidate_manifest_dto()).expect("candidate manifest");
    let observed =
        ObservedDeviceInventoryV1::parse(candidate_observed_dto()).expect("candidate observation");
    assert_eq!(
        expected.as_inventory().stm32().safety_class(),
        ControllerSafetyClass::OperatorSupervisedFourPwmCandidate
    );
    assert!(
        InventoryComparison::compare(expected.as_inventory(), &observed).is_exact_match(),
        "candidate class and exact capability evidence match"
    );

    let production_observed =
        ObservedDeviceInventoryV1::parse(observed_dto()).expect("production observation");
    let comparison = InventoryComparison::compare(expected.as_inventory(), &production_observed);
    assert!(comparison.iter().any(|mismatch| matches!(
        mismatch,
        InventoryMismatch::Stm32SafetyClass {
            expected: ControllerSafetyClass::OperatorSupervisedFourPwmCandidate,
            observed: ControllerSafetyClass::ProductionExternalInterlocks,
        }
    )));
}

#[test]
fn candidate_manifest_rejects_every_cross_class_or_identity_substitution() {
    let mut v1_candidate = manifest_dto();
    v1_candidate.stm32.firmware_build_id = 0x0002_1001;
    v1_candidate.stm32.hardware_profile_fingerprint = *b"KIKO-4PWM-CAND1!";
    v1_candidate.stm32.capabilities_bits = ControllerCapabilities::SOFTWARE_GUARD_BITS
        | ControllerCapabilities::OPERATOR_SUPERVISED_FOUR_PWM_CANDIDATE;
    assert!(matches!(
        DeviceInventoryManifestV1::parse(v1_candidate),
        Err(InventoryParseError::MissingControllerSafetyCapabilities { .. })
    ));

    for mutation in 0..7 {
        let mut dto = candidate_manifest_dto();
        match mutation {
            0 => {
                dto.stm32.controller_session_class =
                    ControllerSessionClassV2Dto::ProductionExternalInterlocks;
            }
            1 => dto.stm32.firmware_abi += 1,
            2 => dto.stm32.firmware_build_id += 1,
            3 => dto.stm32.hardware_profile_fingerprint = [0x12; 16],
            4 => dto.stm32.capabilities_bits = ControllerCapabilities::REQUIRED_BITS,
            5 => {
                dto.stm32.expected_physical_stop_semantics =
                    PhysicalStopSemanticsV2Dto::CoastVerified;
            }
            6 => dto.stm32.expected_max_abs_pwm_percent = 31,
            _ => unreachable!(),
        }
        assert!(
            DeviceInventoryManifestV2::parse(dto).is_err(),
            "candidate mutation {mutation} must reject"
        );
    }
}

#[test]
fn comparison_equality_ignores_intentionally_uncompared_boot_ids() {
    let expected = DeviceInventoryManifestV1::parse(manifest_dto()).expect("manifest");
    let first = ObservedDeviceInventoryV1::parse(observed_dto()).expect("first observation");
    let mut second_dto = observed_dto();
    second_dto.stm32.as_mut().expect("STM32").controller_boot_id = 72;
    second_dto.eye.as_mut().expect("eye").device_boot_id = 82;
    let second = ObservedDeviceInventoryV1::parse(second_dto).expect("second observation");

    assert_ne!(
        first.stm32().expect("first STM32").boot_id(),
        second.stm32().expect("second STM32").boot_id()
    );
    assert_ne!(
        first.eye().expect("first eye").boot_id(),
        second.eye().expect("second eye").boot_id()
    );
    let first_comparison = InventoryComparison::compare(&expected, &first);
    let second_comparison = InventoryComparison::compare(&expected, &second);
    assert!(first_comparison.is_exact_match());
    assert!(second_comparison.is_exact_match());
    assert_eq!(first_comparison, second_comparison);
}

#[test]
fn borrowed_exact_comparison_mints_owned_admission() {
    let admission = {
        let (expected, observed) = parse_pair();
        InventoryComparison::compare(&expected, &observed)
            .into_exact_admission()
            .expect("exact inventory admission")
    };

    assert_eq!(
        admission.expected().robot_id().as_str(),
        "kiko-production-01"
    );
    assert_eq!(admission.observed_stm32().boot_id().get(), 71);
    assert_eq!(
        admission.observed_oak().mxid().as_str(),
        admission.expected().oak().mxid().as_str()
    );
    assert!(
        InventoryComparison::compare(admission.expected(), admission.observed()).is_exact_match()
    );
}

#[test]
fn consuming_admission_moves_owned_snapshots_through_the_same_exact_gate() {
    let (expected, observed) = parse_pair();
    let admission = admit_exact_inventory(expected, observed).expect("exact owned admission");

    assert_eq!(admission.observed().eye().expect("eye").boot_id().get(), 81);
    assert!(
        InventoryComparison::compare(admission.expected(), admission.observed()).is_exact_match()
    );

    let expected = DeviceInventoryManifestV1::parse(manifest_dto()).expect("manifest");
    let mut dto = observed_dto();
    dto.robot_id = "other-robot".into();
    let observed = ObservedDeviceInventoryV1::parse(dto).expect("contrary observation");
    let report = admit_exact_inventory(expected, observed).expect_err("must reject mismatch");
    assert_eq!(report.len(), 1);
    assert!(matches!(
        report.comparison().iter().next(),
        Some(InventoryMismatch::RobotId { .. })
    ));
}

#[test]
fn failed_admission_owns_every_bounded_mismatch_after_inputs_are_dropped() {
    let report = {
        let expected = DeviceInventoryManifestV1::parse(manifest_dto()).expect("manifest");
        let mut dto = observed_dto();
        dto.robot_id = "other-robot".into();
        let oak = dto.oak.as_mut().expect("OAK");
        oak.mxid = "BBBBBBBBBBBBBBBB".into();
        oak.compiled_depthai_header_sdk_commit = "other-commit".into();
        dto.calibration_artifacts[0].sha256 = [0x91; 32];
        let observed = ObservedDeviceInventoryV1::parse(dto).expect("contrary observation");
        InventoryComparison::compare(&expected, &observed)
            .into_exact_admission()
            .expect_err("mismatch must not mint admission")
    };

    assert!(!report.is_empty());
    assert_eq!(report.len(), 4);
    assert_eq!(report.expected().robot_id().as_str(), "kiko-production-01");
    assert_eq!(report.observed().robot_id().as_str(), "other-robot");
    let comparison = report.comparison();
    let mismatches = comparison.iter().collect::<Vec<_>>();
    assert!(matches!(mismatches[0], InventoryMismatch::RobotId { .. }));
    assert!(matches!(mismatches[1], InventoryMismatch::OakMxid { .. }));
    assert!(matches!(
        mismatches[2],
        InventoryMismatch::OakCompiledDepthAiHeaderSdkCommit { .. }
    ));
    assert!(matches!(
        mismatches[3],
        InventoryMismatch::ArtifactDigest { .. }
    ));
    assert!(report.to_string().contains("4 mismatch(es)"));
}

#[test]
fn unknown_manifest_and_observed_schema_versions_are_rejected() {
    let mut expected = manifest_dto();
    expected.schema_version = 2;
    assert_eq!(
        DeviceInventoryManifestV1::parse(expected),
        Err(InventoryParseError::UnsupportedManifestSchema {
            actual: 2,
            supported: DEVICE_INVENTORY_MANIFEST_V1,
        })
    );

    let mut observed = observed_dto();
    observed.schema_version = 9;
    assert_eq!(
        ObservedDeviceInventoryV1::parse(observed),
        Err(InventoryParseError::UnsupportedObservedSchema {
            actual: 9,
            supported: OBSERVED_DEVICE_INVENTORY_V1,
        })
    );
}

#[test]
fn wrapped_boundary_errors_remain_available_in_the_standard_error_chain() {
    let mut dto = manifest_dto();
    dto.stm32.controller_uid = [0; 12];
    let protocol_error = DeviceInventoryManifestV1::parse(dto).expect_err("zero UID");
    assert!(std::error::Error::source(&protocol_error).is_some());

    let mut dto = manifest_dto();
    dto.robot_id.clear();
    let text_error = DeviceInventoryManifestV1::parse(dto).expect_err("empty robot ID");
    assert!(std::error::Error::source(&text_error).is_some());
}

#[test]
fn text_identities_are_bounded_nonzero_and_oak_mxids_are_canonical() {
    let mut dto = manifest_dto();
    dto.robot_id = "0".into();
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::InvalidText {
            field: TextField::RobotId,
            source: BoundedTextError::ZeroIdentity,
        })
    ));

    let mut dto = manifest_dto();
    dto.robot_id = "r".repeat(MAX_ROBOT_ID_BYTES + 1);
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::InvalidText {
            field: TextField::RobotId,
            source: BoundedTextError::TooLong { .. },
        })
    ));

    let mut dto = manifest_dto();
    dto.oak.mxid = "00000000".into();
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::InvalidText {
            field: TextField::OakMxid,
            source: BoundedTextError::ZeroIdentity,
        })
    ));

    let upper = DeviceInventoryManifestV1::parse(manifest_dto()).expect("uppercase MXID");
    let mut lower_dto = manifest_dto();
    lower_dto.oak.mxid.make_ascii_lowercase();
    let lower = DeviceInventoryManifestV1::parse(lower_dto).expect("lowercase MXID");
    assert_eq!(upper.oak().mxid(), lower.oak().mxid());
    assert_eq!(lower.oak().mxid().as_str(), "A1B2C3D4E5F60708");

    for (field, mutate) in [
        (
            TextField::OakMxid,
            (|dto: &mut DeviceInventoryManifestV1Dto| {
                dto.oak.mxid = "A".repeat(MAX_OAK_MXID_BYTES + 1);
            }) as fn(&mut DeviceInventoryManifestV1Dto),
        ),
        (
            TextField::OakCompiledDepthAiHeaderSdkVersion,
            |dto: &mut DeviceInventoryManifestV1Dto| {
                dto.oak.compiled_depthai_header_sdk_version =
                    "a".repeat(MAX_BUILD_PROVENANCE_BYTES + 1);
            },
        ),
        (
            TextField::SerialPath(DeviceRole::Stm32),
            |dto: &mut DeviceInventoryManifestV1Dto| {
                dto.stm32.serial_by_id_path = format!(
                    "/dev/serial/by-id/{}",
                    "a".repeat(MAX_SERIAL_BY_ID_PATH_BYTES)
                );
            },
        ),
        (
            TextField::Stm32ControlEndpoint,
            |dto: &mut DeviceInventoryManifestV1Dto| {
                dto.stm32.control_endpoint_identity =
                    format!("unix:/{}", "a".repeat(MAX_CONTROL_ENDPOINT_ID_BYTES));
            },
        ),
    ] {
        let mut dto = manifest_dto();
        mutate(&mut dto);
        assert!(matches!(
            DeviceInventoryManifestV1::parse(dto),
            Err(InventoryParseError::InvalidText {
                field: actual_field,
                source: BoundedTextError::TooLong { .. },
            }) if actual_field == field
        ));
    }
}

#[test]
fn serial_paths_and_control_endpoints_must_be_persistent_and_local() {
    for invalid in ["ttyACM0", "/dev/ttyACM0", "serial/by-id/stm32"] {
        let mut dto = manifest_dto();
        dto.stm32.serial_by_id_path = invalid.into();
        assert!(matches!(
            DeviceInventoryManifestV1::parse(dto),
            Err(InventoryParseError::InvalidText {
                field: TextField::SerialPath(DeviceRole::Stm32),
                source: BoundedTextError::NotPersistentSerialById,
            })
        ));
    }

    let mut dto = manifest_dto();
    dto.head.as_mut().expect("head").adapter_serial_by_id_path =
        "/dev/serial/by-id/usb/head".into();
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::InvalidText {
            field: TextField::SerialPath(DeviceRole::Head),
            source: BoundedTextError::SerialByIdHasNestedComponent,
        })
    ));

    for invalid in [
        "/run/kiko/robot.sock",
        "tcp://0.0.0.0:5000",
        "tcp://192.168.50.2:5000",
        "tcp://127.0.0.1:0",
        "tcp://127.0.0.1:05000",
        "udp://0.0.0.0:5000",
        "udp://192.168.50.2:5000",
        "udp://127.0.0.1:0",
        "udp://127.0.0.1:05000",
    ] {
        let mut dto = manifest_dto();
        dto.stm32.control_endpoint_identity = invalid.into();
        assert!(matches!(
            DeviceInventoryManifestV1::parse(dto),
            Err(InventoryParseError::InvalidText {
                field: TextField::Stm32ControlEndpoint,
                source: BoundedTextError::InvalidControlEndpoint,
            })
        ));
    }

    let mut dto = manifest_dto();
    dto.stm32.control_endpoint_identity = "udp://127.0.0.1:8080".into();
    let parsed = DeviceInventoryManifestV1::parse(dto).expect("local UDP endpoint");
    let endpoint = *parsed.stm32().control_endpoint();
    assert_eq!(
        endpoint.transport(),
        kiko_device_inventory::ControlEndpointTransport::Udp
    );
    assert_eq!(
        endpoint.socket_addr(),
        Some("127.0.0.1:8080".parse().expect("socket"))
    );
}

#[test]
fn zero_binary_ids_builds_boots_profiles_and_digests_are_rejected() {
    let mut dto = manifest_dto();
    dto.stm32.controller_uid = [0; 12];
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::InvalidControllerUid(_))
    ));

    let mut dto = manifest_dto();
    dto.stm32.firmware_abi = 0;
    assert_eq!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::ZeroStm32FirmwareAbi)
    );

    let mut dto = manifest_dto();
    dto.stm32.firmware_build_id = 0;
    assert_eq!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::ZeroStm32FirmwareBuildId)
    );

    let mut dto = manifest_dto();
    dto.stm32.hardware_profile_fingerprint = [0; 16];
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::InvalidHardwareProfile(_))
    ));

    let mut dto = manifest_dto();
    dto.eye.as_mut().expect("eye").device_uid = [0; 16];
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::InvalidEyeDeviceUid(_))
    ));

    let mut dto = manifest_dto();
    dto.eye.as_mut().expect("eye").firmware_build_id = [0; 32];
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::InvalidEyeFirmwareBuildId(_))
    ));

    let mut dto = observed_dto();
    dto.stm32.as_mut().expect("STM32").controller_boot_id = 0;
    assert!(matches!(
        ObservedDeviceInventoryV1::parse(dto),
        Err(InventoryParseError::InvalidControllerBootId(_))
    ));

    let mut dto = observed_dto();
    dto.eye.as_mut().expect("eye").device_boot_id = 0;
    assert!(matches!(
        ObservedDeviceInventoryV1::parse(dto),
        Err(InventoryParseError::InvalidEyeBootId(_))
    ));

    let mut dto = manifest_dto();
    dto.calibration_artifacts[0].sha256 = [0; 32];
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::ZeroArtifactDigest {
            kind: ArtifactKind::Calibration,
            index: 0,
            ..
        })
    ));
}

#[test]
fn expected_devices_must_satisfy_the_compiled_protocol_contracts() {
    let mut dto = manifest_dto();
    dto.stm32.firmware_abi = u16::from(ROBOT_PROTOCOL_VERSION) + 1;
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::Stm32FirmwareAbiContractMismatch { .. })
    ));

    let mut dto = manifest_dto();
    dto.stm32.capabilities_bits =
        ControllerCapabilities::REQUIRED_BITS & !ControllerCapabilities::INDEPENDENT_WATCHDOG;
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::MissingControllerSafetyCapabilities { .. })
    ));

    let mut dto = manifest_dto();
    dto.head.as_mut().expect("head").bow_servo_id = 4;
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::DuplicateHeadServoId { .. })
    ));

    let mut dto = manifest_dto();
    dto.head.as_mut().expect("head").bow_servo_id = 5;
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::HeadServoContractMismatch {
            joint: HeadJoint::Bow,
            ..
        })
    ));

    let mut dto = manifest_dto();
    dto.head.as_mut().expect("head").dtr_asserted = !kiko_head_protocol::ADAPTER_DTR_ASSERTED;
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::HeadElectricalContractMismatch { .. })
    ));

    let mut dto = manifest_dto();
    dto.eye.as_mut().expect("eye").kep_protocol_version = kiko_eye_protocol::PROTOCOL_VERSION + 1;
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::EyeProtocolContractMismatch { .. })
    ));

    let mut dto = manifest_dto();
    dto.eye.as_mut().expect("eye").capabilities_bits =
        kiko_eye_protocol::Capabilities::KNOWN_BITS & !kiko_eye_protocol::Capabilities::BLINK;
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::MissingEyeCapabilities { .. })
    ));
}

#[test]
fn unknown_capability_bits_are_rejected_at_the_boundary() {
    let mut dto = manifest_dto();
    dto.stm32.capabilities_bits = ControllerCapabilities::KNOWN_BITS | (1 << 31);
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::InvalidControllerCapabilities(_))
    ));

    let mut dto = observed_dto();
    dto.eye.as_mut().expect("eye").capabilities_bits =
        kiko_eye_protocol::Capabilities::KNOWN_BITS | (1 << 31);
    assert!(matches!(
        ObservedDeviceInventoryV1::parse(dto),
        Err(InventoryParseError::InvalidEyeCapabilities(_))
    ));
}

#[test]
fn duplicate_physical_and_logical_identities_are_rejected() {
    let mut dto = manifest_dto();
    dto.head.as_mut().expect("head").adapter_serial_by_id_path =
        dto.stm32.serial_by_id_path.clone();
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::DuplicatePhysicalPath {
            first: DeviceRole::Stm32,
            second: DeviceRole::Head,
            ..
        })
    ));

    let mut dto = observed_dto();
    dto.stm32 = None;
    let eye_path = dto.eye.as_ref().expect("eye").serial_by_id_path.clone();
    dto.head.as_mut().expect("head").adapter_serial_by_id_path = eye_path;
    assert!(matches!(
        ObservedDeviceInventoryV1::parse(dto),
        Err(InventoryParseError::DuplicatePhysicalPath {
            first: DeviceRole::Head,
            second: DeviceRole::Eye,
            ..
        })
    ));

    let mut dto = manifest_dto();
    dto.calibration_artifacts[1].artifact_id = dto.calibration_artifacts[0].artifact_id.clone();
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::DuplicateArtifactId { .. })
    ));

    let mut dto = manifest_dto();
    dto.plant_artifacts[0].sha256 = dto.calibration_artifacts[0].sha256;
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::DuplicateArtifactDigest { .. })
    ));
}

#[test]
fn artifact_requirements_counts_and_ids_are_bounded() {
    let mut dto = manifest_dto();
    dto.calibration_artifacts.clear();
    assert_eq!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::MissingRequiredArtifactKind {
            kind: ArtifactKind::Calibration,
        })
    );

    let mut dto = manifest_dto();
    dto.plant_artifacts.clear();
    assert_eq!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::MissingRequiredArtifactKind {
            kind: ArtifactKind::Plant,
        })
    );

    let mut dto = manifest_dto();
    dto.calibration_artifacts = (0..=MAX_CALIBRATION_ARTIFACTS)
        .map(|index| {
            digest(
                format!("cal-{index}"),
                u8::try_from(index + 1).expect("small"),
            )
        })
        .collect();
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::TooManyArtifacts {
            kind: ArtifactKind::Calibration,
            ..
        })
    ));

    let mut dto = observed_dto();
    dto.plant_artifacts = (0..=MAX_PLANT_ARTIFACTS)
        .map(|index| {
            digest(
                format!("plant-{index}"),
                u8::try_from(index + 1).expect("small"),
            )
        })
        .collect();
    assert!(matches!(
        ObservedDeviceInventoryV1::parse(dto),
        Err(InventoryParseError::TooManyArtifacts {
            kind: ArtifactKind::Plant,
            ..
        })
    ));

    let mut dto = manifest_dto();
    dto.calibration_artifacts[0].artifact_id = "a".repeat(MAX_ARTIFACT_ID_BYTES + 1);
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::InvalidText {
            field: TextField::ArtifactId {
                kind: ArtifactKind::Calibration,
                index: 0,
            },
            source: BoundedTextError::TooLong { .. },
        })
    ));

    let mut dto = manifest_dto();
    dto.calibration_artifacts[0].artifact_id = "0".into();
    assert!(matches!(
        DeviceInventoryManifestV1::parse(dto),
        Err(InventoryParseError::InvalidText {
            field: TextField::ArtifactId {
                kind: ArtifactKind::Calibration,
                index: 0,
            },
            source: BoundedTextError::ZeroIdentity,
        })
    ));
}

#[test]
fn changed_artifact_content_is_one_exact_digest_mismatch() {
    let expected = DeviceInventoryManifestV1::parse(manifest_dto()).expect("manifest");
    let mut dto = observed_dto();
    dto.calibration_artifacts[0].sha256 = [0x91; 32];
    let observed = ObservedDeviceInventoryV1::parse(dto).expect("different nonzero digest");
    let comparison = InventoryComparison::compare(&expected, &observed);
    assert_eq!(comparison.len(), 1);
    assert!(matches!(
        comparison.iter().next(),
        Some(InventoryMismatch::ArtifactDigest {
            kind: ArtifactKind::Calibration,
            ..
        })
    ));
}

#[test]
fn observed_head_faults_are_not_silently_coerced() {
    let mut dto = observed_dto();
    dto.head.as_mut().expect("head").baud_rate_bps = 0;
    assert_eq!(
        ObservedDeviceInventoryV1::parse(dto),
        Err(InventoryParseError::ZeroObservedHeadBaudRate)
    );

    let mut dto = observed_dto();
    dto.head.as_mut().expect("head").responding_servo_ids = vec![1, 1];
    assert!(matches!(
        ObservedDeviceInventoryV1::parse(dto),
        Err(InventoryParseError::DuplicateHeadServoId { .. })
    ));

    let mut dto = observed_dto();
    dto.head.as_mut().expect("head").responding_servo_ids = vec![0];
    assert!(matches!(
        ObservedDeviceInventoryV1::parse(dto),
        Err(InventoryParseError::InvalidHeadServoId { index: 0, .. })
    ));

    let mut dto = observed_dto();
    dto.head.as_mut().expect("head").responding_servo_ids =
        (1..=u8::try_from(MAX_OBSERVED_HEAD_SERVOS + 1).expect("small")).collect();
    assert!(matches!(
        ObservedDeviceInventoryV1::parse(dto),
        Err(InventoryParseError::TooManyObservedHeadServos { .. })
    ));
}

#[test]
fn only_physically_optional_expected_accessories_may_be_absent() {
    let mut expected_dto = manifest_dto();
    expected_dto.head = None;
    expected_dto.eye = None;
    let expected = DeviceInventoryManifestV1::parse(expected_dto).expect("headless manifest");

    let mut headless_observed_dto = observed_dto();
    headless_observed_dto.head = None;
    headless_observed_dto.eye = None;
    let observed =
        ObservedDeviceInventoryV1::parse(headless_observed_dto).expect("headless report");
    assert!(InventoryComparison::compare(&expected, &observed).is_exact_match());

    let expected = DeviceInventoryManifestV1::parse(manifest_dto()).expect("manifest");
    let mut missing_dto = observed_dto();
    missing_dto.oak = None;
    missing_dto.stm32 = None;
    missing_dto.head = None;
    missing_dto.eye = None;
    let missing = ObservedDeviceInventoryV1::parse(missing_dto).expect("absence is reportable");
    let comparison = InventoryComparison::compare(&expected, &missing);
    let mismatches = comparison.iter().cloned().collect::<Vec<_>>();
    assert_eq!(
        mismatches,
        vec![
            InventoryMismatch::MissingOak,
            InventoryMismatch::MissingStm32,
            InventoryMismatch::MissingHead,
            InventoryMismatch::MissingEye,
        ]
    );
}

#[test]
fn undeclared_accessories_are_reported_as_unexpected() {
    let mut expected_dto = manifest_dto();
    expected_dto.head = None;
    expected_dto.eye = None;
    let expected = DeviceInventoryManifestV1::parse(expected_dto).expect("base-only manifest");
    let observed = ObservedDeviceInventoryV1::parse(observed_dto()).expect("full report");
    let comparison = InventoryComparison::compare(&expected, &observed);
    let mismatches = comparison.iter().cloned().collect::<Vec<_>>();
    assert_eq!(
        mismatches,
        vec![
            InventoryMismatch::UnexpectedHead,
            InventoryMismatch::UnexpectedEye,
        ]
    );
}

#[test]
fn every_scalar_fault_is_accumulated_in_stable_order() {
    let expected = DeviceInventoryManifestV1::parse(manifest_dto()).expect("manifest");
    let mut dto = observed_dto();
    dto.robot_id = "other-robot".into();
    let oak = dto.oak.as_mut().expect("OAK");
    oak.mxid = "BBBBBBBBBBBBBBBB".into();
    oak.compiled_depthai_header_sdk_version = "other-sdk-version".into();
    oak.compiled_depthai_header_sdk_commit = "other-sdk-commit".into();
    oak.compiled_depthai_header_embedded_device_artifact_version = "other-device-artifact".into();
    oak.compiled_depthai_header_embedded_bootloader_artifact_version =
        "other-bootloader-artifact".into();
    let stm32 = dto.stm32.as_mut().expect("STM32");
    stm32.serial_by_id_path = "/dev/serial/by-id/usb-other-stm32".into();
    stm32.control_endpoint_identity = "tcp://127.0.0.1:5001".into();
    stm32.controller_uid = [0x51; 12];
    stm32.firmware_abi = u16::from(ROBOT_PROTOCOL_VERSION) + 1;
    stm32.firmware_build_id = 0x5060_7080;
    stm32.hardware_profile_fingerprint = [0x52; 16];
    stm32.capabilities_bits = 0;
    let head = dto.head.as_mut().expect("head");
    head.adapter_serial_by_id_path = "/dev/serial/by-id/usb-other-head".into();
    head.baud_rate_bps = 115_200;
    head.dtr_asserted = !kiko_head_protocol::ADAPTER_DTR_ASSERTED;
    head.rts_asserted = !kiko_head_protocol::ADAPTER_RTS_ASSERTED;
    head.responding_servo_ids = vec![5];
    let eye = dto.eye.as_mut().expect("eye");
    eye.serial_by_id_path = "/dev/serial/by-id/usb-other-eye".into();
    eye.kep_protocol_version = kiko_eye_protocol::PROTOCOL_VERSION + 1;
    eye.device_uid = [0x61; 16];
    eye.firmware_build_id = [0x62; 32];
    eye.capabilities_bits = 0;
    let observed = ObservedDeviceInventoryV1::parse(dto).expect("valid contrary claims");

    let comparison = InventoryComparison::compare(&expected, &observed);
    let mismatches = comparison.iter().collect::<Vec<_>>();
    assert_eq!(mismatches.len(), 24);
    assert!(matches!(mismatches[0], InventoryMismatch::RobotId { .. }));
    assert!(matches!(mismatches[1], InventoryMismatch::OakMxid { .. }));
    assert!(matches!(
        mismatches[2],
        InventoryMismatch::OakCompiledDepthAiHeaderSdkVersion { .. }
    ));
    assert!(matches!(
        mismatches[3],
        InventoryMismatch::OakCompiledDepthAiHeaderSdkCommit { .. }
    ));
    assert!(matches!(
        mismatches[4],
        InventoryMismatch::OakCompiledDepthAiHeaderEmbeddedDeviceArtifactVersion { .. }
    ));
    assert!(matches!(
        mismatches[5],
        InventoryMismatch::OakCompiledDepthAiHeaderEmbeddedBootloaderArtifactVersion { .. }
    ));
    assert!(matches!(
        mismatches[6],
        InventoryMismatch::Stm32SerialPath { .. }
    ));
    assert!(matches!(
        mismatches[7],
        InventoryMismatch::Stm32ControlEndpoint { .. }
    ));
    assert!(matches!(
        mismatches[8],
        InventoryMismatch::Stm32ControllerUid { .. }
    ));
    assert!(matches!(
        mismatches[9],
        InventoryMismatch::Stm32FirmwareAbi { .. }
    ));
    assert!(matches!(
        mismatches[10],
        InventoryMismatch::Stm32FirmwareBuildId { .. }
    ));
    assert!(matches!(
        mismatches[11],
        InventoryMismatch::Stm32HardwareProfile { .. }
    ));
    assert!(matches!(
        mismatches[12],
        InventoryMismatch::Stm32Capabilities { .. }
    ));
    assert!(matches!(
        mismatches[13],
        InventoryMismatch::Stm32SafetyClass { .. }
    ));
    assert!(matches!(
        mismatches[14],
        InventoryMismatch::HeadSerialPath { .. }
    ));
    assert!(matches!(
        mismatches[15],
        InventoryMismatch::HeadBaudRate { .. }
    ));
    assert!(matches!(
        mismatches[16],
        InventoryMismatch::HeadDtrState { .. }
    ));
    assert!(matches!(
        mismatches[17],
        InventoryMismatch::HeadRtsState { .. }
    ));
    assert!(matches!(
        mismatches[18],
        InventoryMismatch::HeadServoIds { .. }
    ));
    assert!(matches!(
        mismatches[19],
        InventoryMismatch::EyeSerialPath { .. }
    ));
    assert!(matches!(
        mismatches[20],
        InventoryMismatch::EyeProtocolVersion { .. }
    ));
    assert!(matches!(
        mismatches[21],
        InventoryMismatch::EyeDeviceUid { .. }
    ));
    assert!(matches!(
        mismatches[22],
        InventoryMismatch::EyeFirmwareBuildId { .. }
    ));
    assert!(matches!(
        mismatches[23],
        InventoryMismatch::EyeCapabilities { .. }
    ));
}

#[test]
fn comparison_capacity_is_the_tight_reachable_bound() {
    let mut expected_dto = manifest_dto();
    expected_dto.calibration_artifacts = (0..MAX_CALIBRATION_ARTIFACTS)
        .map(|index| {
            digest(
                format!("expected-cal-{index}"),
                u8::try_from(index + 1).expect("small"),
            )
        })
        .collect();
    expected_dto.plant_artifacts = (0..MAX_PLANT_ARTIFACTS)
        .map(|index| {
            digest(
                format!("expected-plant-{index}"),
                u8::try_from(index + 21).expect("small"),
            )
        })
        .collect();
    let expected = DeviceInventoryManifestV1::parse(expected_dto).expect("maximal manifest");

    let mut observed_dto = observed_dto();
    observed_dto.robot_id = "other-robot".into();
    let oak = observed_dto.oak.as_mut().expect("OAK");
    oak.mxid = "BBBBBBBBBBBBBBBB".into();
    oak.compiled_depthai_header_sdk_version = "other-sdk-version".into();
    oak.compiled_depthai_header_sdk_commit = "other-sdk-commit".into();
    oak.compiled_depthai_header_embedded_device_artifact_version = "other-device-artifact".into();
    oak.compiled_depthai_header_embedded_bootloader_artifact_version =
        "other-bootloader-artifact".into();
    let stm32 = observed_dto.stm32.as_mut().expect("STM32");
    stm32.serial_by_id_path = "/dev/serial/by-id/usb-other-stm32".into();
    stm32.control_endpoint_identity = "tcp://[::1]:5002".into();
    stm32.controller_uid = [0x71; 12];
    stm32.firmware_abi = u16::from(ROBOT_PROTOCOL_VERSION) + 1;
    stm32.firmware_build_id = 0x1122_3344;
    stm32.hardware_profile_fingerprint = [0x72; 16];
    stm32.capabilities_bits = 0;
    let head = observed_dto.head.as_mut().expect("head");
    head.adapter_serial_by_id_path = "/dev/serial/by-id/usb-other-head".into();
    head.baud_rate_bps = 115_200;
    head.dtr_asserted = !kiko_head_protocol::ADAPTER_DTR_ASSERTED;
    head.rts_asserted = !kiko_head_protocol::ADAPTER_RTS_ASSERTED;
    head.responding_servo_ids = vec![5];
    let eye = observed_dto.eye.as_mut().expect("eye");
    eye.serial_by_id_path = "/dev/serial/by-id/usb-other-eye".into();
    eye.kep_protocol_version = kiko_eye_protocol::PROTOCOL_VERSION + 1;
    eye.device_uid = [0x81; 16];
    eye.firmware_build_id = [0x82; 32];
    eye.capabilities_bits = 0;
    observed_dto.calibration_artifacts = (0..MAX_CALIBRATION_ARTIFACTS)
        .map(|index| {
            digest(
                format!("observed-cal-{index}"),
                u8::try_from(index + 101).expect("small"),
            )
        })
        .collect();
    observed_dto.plant_artifacts = (0..MAX_PLANT_ARTIFACTS)
        .map(|index| {
            digest(
                format!("observed-plant-{index}"),
                u8::try_from(index + 121).expect("small"),
            )
        })
        .collect();
    let observed = ObservedDeviceInventoryV1::parse(observed_dto).expect("maximal contrary report");

    let comparison = InventoryComparison::compare(&expected, &observed);
    assert_eq!(comparison.len(), MAX_INVENTORY_MISMATCHES);
    assert_eq!(comparison.iter().count(), MAX_INVENTORY_MISMATCHES);
    assert!(comparison.iter().take(24).all(|mismatch| !matches!(
        mismatch,
        InventoryMismatch::MissingArtifact { .. }
            | InventoryMismatch::ArtifactDigest { .. }
            | InventoryMismatch::UnexpectedArtifact { .. }
    )));
    assert!(
        comparison
            .iter()
            .skip(24)
            .take(12)
            .all(|mismatch| matches!(mismatch, InventoryMismatch::MissingArtifact { .. }))
    );
    assert!(
        comparison
            .iter()
            .skip(36)
            .all(|mismatch| matches!(mismatch, InventoryMismatch::UnexpectedArtifact { .. }))
    );
}

#[test]
fn comparison_storage_retains_borrowed_evidence_instead_of_inline_identity_copies() {
    assert!(core::mem::size_of::<InventoryMismatch<'static>>() <= 64);
    assert!(core::mem::size_of::<InventoryComparison<'static>>() <= 4_096);
    assert!(core::mem::size_of::<kiko_device_inventory::InventoryMismatchReport>() <= 64);
}

#[test]
fn artifact_input_permutations_parse_to_one_canonical_set() {
    let entries = [digest("camera", 1), digest("head", 2), digest("imu", 3)];
    let permutations = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut canonical = None;
    for permutation in permutations {
        let mut dto = manifest_dto();
        dto.calibration_artifacts = permutation
            .map(|index| entries[index].clone())
            .into_iter()
            .collect();
        let parsed = DeviceInventoryManifestV1::parse(dto).expect("permutation");
        if let Some(expected) = canonical.as_ref() {
            assert_eq!(parsed.artifacts(), expected);
        } else {
            canonical = Some(parsed.artifacts().clone());
        }
    }
}

#[test]
fn every_head_response_order_has_identical_set_semantics() {
    let permutations = [
        [1, 2, 3, 4],
        [1, 2, 4, 3],
        [1, 3, 2, 4],
        [1, 3, 4, 2],
        [1, 4, 2, 3],
        [1, 4, 3, 2],
        [2, 1, 3, 4],
        [2, 1, 4, 3],
        [2, 3, 1, 4],
        [2, 3, 4, 1],
        [2, 4, 1, 3],
        [2, 4, 3, 1],
        [3, 1, 2, 4],
        [3, 1, 4, 2],
        [3, 2, 1, 4],
        [3, 2, 4, 1],
        [3, 4, 1, 2],
        [3, 4, 2, 1],
        [4, 1, 2, 3],
        [4, 1, 3, 2],
        [4, 2, 1, 3],
        [4, 2, 3, 1],
        [4, 3, 1, 2],
        [4, 3, 2, 1],
    ];
    let expected = DeviceInventoryManifestV1::parse(manifest_dto()).expect("manifest");
    for permutation in permutations {
        let mut dto = observed_dto();
        dto.head.as_mut().expect("head").responding_servo_ids = permutation.to_vec();
        let observed = ObservedDeviceInventoryV1::parse(dto).expect("valid servo permutation");
        assert!(
            InventoryComparison::compare(&expected, &observed).is_exact_match(),
            "{permutation:?}"
        );
    }
}
