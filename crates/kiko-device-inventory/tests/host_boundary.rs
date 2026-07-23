#![cfg(unix)]

use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use kiko_device_inventory::{
    ArtifactFileBindingInput, ArtifactFileBindingParseError, ArtifactFileBindingSet,
    ArtifactHashError, ArtifactKind, ArtifactRelativePath, CalibrationBundleHashError,
    DeploymentAssetByteLimit, InventoryParseError, MAX_ARTIFACT_FILE_BYTES,
    MAX_MANIFEST_JSON_BYTES, ManifestJsonError, ManifestLoadError, hash_manifest_artifacts,
    hash_manifest_artifacts_reusing_loaded_asset, load_deployment_asset,
    load_expected_manifest_v1_file, load_expected_manifest_v1_from_slice,
    load_expected_manifest_v2_from_slice,
};
use robot_protocol::v2::{ControllerCapabilities, VERSION as ROBOT_PROTOCOL_VERSION};
use serde_json::json;
use sha2::{Digest, Sha256};

static NEXT_TEMP_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TempDirectory(PathBuf);

impl TempDirectory {
    fn new() -> Self {
        let sequence = NEXT_TEMP_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let temp_root = fs::canonicalize(std::env::temp_dir()).expect("canonical temp root");
        let path = temp_root.join(format!(
            "kiko-device-inventory-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).expect("create unique test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDirectory {
    fn drop(&mut self) {
        let _cleanup_result = fs::remove_dir_all(&self.0);
    }
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn manifest_json(calibration: &[u8], plant: &[u8]) -> Vec<u8> {
    serde_json::to_vec(&json!({
        "schema_version": 1,
        "robot_id": "kiko-production-01",
        "oak": {
            "mxid": "A1B2C3D4E5F60708",
            "compiled_depthai_header_sdk_version": "3.6.1",
            "compiled_depthai_header_sdk_commit": "abc123",
            "compiled_depthai_header_embedded_device_artifact_version": "device-1",
            "compiled_depthai_header_embedded_bootloader_artifact_version": "bootloader-1"
        },
        "stm32": {
            "serial_by_id_path": "/dev/serial/by-id/usb-Kiko_STM32_A1-if00",
            "control_endpoint_identity": "unix:/run/kiko/robot-v2.sock",
            "controller_uid": [17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17],
            "firmware_abi": u16::from(ROBOT_PROTOCOL_VERSION),
            "firmware_build_id": 270544960,
            "hardware_profile_fingerprint": [18, 18, 18, 18, 18, 18, 18, 18,
                                                 18, 18, 18, 18, 18, 18, 18, 18],
            "capabilities_bits": ControllerCapabilities::REQUIRED_BITS
        },
        "head": null,
        "eye": null,
        "calibration_artifacts": [{
            "artifact_id": "calibration-main",
            "sha256": sha256(calibration)
        }],
        "plant_artifacts": [{
            "artifact_id": "plant-main",
            "sha256": sha256(plant)
        }]
    }))
    .expect("serialize manifest fixture")
}

fn bindings() -> Vec<ArtifactFileBindingInput> {
    vec![
        ArtifactFileBindingInput {
            kind: ArtifactKind::Calibration,
            artifact_id: "calibration-main".to_owned(),
            relative_path: "calibration/main.bin".to_owned(),
        },
        ArtifactFileBindingInput {
            kind: ArtifactKind::Plant,
            artifact_id: "plant-main".to_owned(),
            relative_path: "plant/main.bin".to_owned(),
        },
    ]
}

fn candidate_manifest_json(calibration: &[u8], plant: &[u8]) -> Vec<u8> {
    let mut value: serde_json::Value =
        serde_json::from_slice(&manifest_json(calibration, plant)).expect("V1 fixture");
    value["schema_version"] = json!(2);
    value["stm32"]["firmware_build_id"] = json!(0x0002_1001_u32);
    value["stm32"]["hardware_profile_fingerprint"] = json!(b"KIKO-4PWM-CAND1!".as_slice());
    value["stm32"]["capabilities_bits"] = json!(
        ControllerCapabilities::SOFTWARE_GUARD_BITS
            | ControllerCapabilities::OPERATOR_SUPERVISED_FOUR_PWM_CANDIDATE
    );
    value["stm32"]["controller_session_class"] = json!("operator_supervised_four_pwm_candidate");
    value["stm32"]["expected_max_abs_pwm_percent"] = json!(30);
    value["stm32"]["expected_physical_stop_semantics"] = json!("unverified");
    serde_json::to_vec(&value).expect("candidate manifest fixture")
}

fn parsed_bindings() -> ArtifactFileBindingSet {
    ArtifactFileBindingSet::parse(bindings()).expect("valid artifact binding fixture")
}

fn write_artifacts(root: &Path, calibration: &[u8], plant: &[u8]) {
    fs::create_dir_all(root.join("calibration")).expect("calibration directory");
    fs::create_dir_all(root.join("plant")).expect("plant directory");
    fs::write(root.join("calibration/main.bin"), calibration).expect("calibration artifact");
    fs::write(root.join("plant/main.bin"), plant).expect("plant artifact");
}

#[test]
fn slice_and_file_load_once_through_the_existing_domain_dto() {
    let calibration = b"camera calibration v3";
    let plant = b"drive plant v4";
    let json = manifest_json(calibration, plant);
    let loaded = load_expected_manifest_v1_from_slice(&json).expect("bounded slice manifest");
    assert_eq!(loaded.json_bytes(), json.len());
    assert_eq!(loaded.content_sha256().as_bytes(), &sha256(&json));
    assert_eq!(loaded.source_path(), None);
    assert_eq!(loaded.manifest().robot_id().as_str(), "kiko-production-01");
    assert_eq!(loaded.manifest().artifacts().len(), 2);

    let temp = TempDirectory::new();
    let path = temp.path().join("expected-v1.json");
    fs::write(&path, &json).expect("manifest file");
    let from_file = load_expected_manifest_v1_file(&path).expect("exact manifest file");
    assert_eq!(from_file.json_bytes(), json.len());
    assert_eq!(from_file.manifest(), loaded.manifest());
    assert_eq!(from_file.content_sha256(), loaded.content_sha256());
    assert_eq!(from_file.source_path(), Some(path.as_path()));

    let value: serde_json::Value = serde_json::from_slice(&json).expect("fixture JSON");
    let pretty = serde_json::to_vec_pretty(&value).expect("pretty fixture JSON");
    let reformatted =
        load_expected_manifest_v1_from_slice(&pretty).expect("reformatted valid manifest");
    assert_eq!(reformatted.manifest(), loaded.manifest());
    assert_ne!(reformatted.content_sha256(), loaded.content_sha256());
}

#[test]
fn candidate_manifest_json_requires_the_explicit_schema_v2_class() {
    let json = candidate_manifest_json(b"calibration", b"plant");
    let loaded = load_expected_manifest_v2_from_slice(&json).expect("candidate manifest");
    assert_eq!(loaded.json_bytes(), json.len());
    assert_eq!(loaded.content_sha256().as_bytes(), &sha256(&json));
    assert_eq!(
        loaded.manifest().as_inventory().stm32().firmware_build_id(),
        0x0002_1001
    );

    let mut missing: serde_json::Value = serde_json::from_slice(&json).expect("fixture");
    missing["stm32"]
        .as_object_mut()
        .expect("STM32")
        .remove("controller_session_class");
    assert!(
        load_expected_manifest_v2_from_slice(
            &serde_json::to_vec(&missing).expect("missing fixture")
        )
        .is_err()
    );

    let mut trailing = json;
    trailing.extend_from_slice(b" null");
    assert!(matches!(
        load_expected_manifest_v2_from_slice(&trailing),
        Err(ManifestLoadError::Json(
            ManifestJsonError::TrailingData { .. }
        ))
    ));
}

#[test]
fn duplicate_unknown_and_trailing_json_are_rejected() {
    let json = manifest_json(b"calibration", b"plant");
    let text = String::from_utf8(json.clone()).expect("UTF-8 fixture");
    let marker = "\"schema_version\":1";
    assert!(text.contains(marker));
    let duplicate = text.replacen(marker, &format!("{marker},{marker}"), 1);
    assert!(matches!(
        load_expected_manifest_v1_from_slice(duplicate.as_bytes()),
        Err(ManifestLoadError::Json(ManifestJsonError::Decode { .. }))
    ));

    let mut unknown: serde_json::Value = serde_json::from_slice(&json).expect("fixture JSON");
    unknown
        .as_object_mut()
        .expect("object")
        .insert("unknown_field".to_owned(), json!(true));
    assert!(matches!(
        load_expected_manifest_v1_from_slice(
            &serde_json::to_vec(&unknown).expect("unknown-field fixture")
        ),
        Err(ManifestLoadError::Json(ManifestJsonError::Decode { .. }))
    ));

    let mut nested_unknown: serde_json::Value =
        serde_json::from_slice(&json).expect("fixture JSON");
    nested_unknown["oak"]["unknown_field"] = json!(true);
    assert!(matches!(
        load_expected_manifest_v1_from_slice(
            &serde_json::to_vec(&nested_unknown).expect("nested unknown fixture")
        ),
        Err(ManifestLoadError::Json(ManifestJsonError::Decode { .. }))
    ));

    let mut trailing = json;
    trailing.extend_from_slice(b" {} ");
    assert!(matches!(
        load_expected_manifest_v1_from_slice(&trailing),
        Err(ManifestLoadError::Json(
            ManifestJsonError::TrailingData { .. }
        ))
    ));
}

#[test]
fn ambiguous_legacy_oak_provenance_fields_cannot_enter_the_typed_manifest() {
    let mut value: serde_json::Value =
        serde_json::from_slice(&manifest_json(b"calibration", b"plant")).expect("fixture");
    value["oak"] = json!({
        "mxid": "A1B2C3D4E5F60708",
        "runtime_provenance": "unverified-device-runtime",
        "sdk_build_provenance": "unspecified-sdk-build-semantics",
        "adapter_build_provenance": "caller-asserted-adapter"
    });

    assert!(matches!(
        load_expected_manifest_v1_from_slice(
            &serde_json::to_vec(&value).expect("legacy OAK fixture")
        ),
        Err(ManifestLoadError::Json(ManifestJsonError::Decode { .. }))
    ));
}

#[test]
fn oversized_and_truncated_json_files_fail_with_exact_categories() {
    assert!(matches!(
        load_expected_manifest_v1_from_slice(&vec![b' '; MAX_MANIFEST_JSON_BYTES + 1]),
        Err(ManifestLoadError::JsonTooLarge { .. })
    ));

    let temp = TempDirectory::new();
    let oversized = temp.path().join("oversized.json");
    File::create(&oversized)
        .expect("oversized file")
        .set_len(MAX_MANIFEST_JSON_BYTES as u64 + 1)
        .expect("sparse oversized file");
    assert!(matches!(
        load_expected_manifest_v1_file(&oversized),
        Err(ManifestLoadError::JsonTooLarge { .. })
    ));

    let mut truncated = manifest_json(b"calibration", b"plant");
    truncated.pop();
    let truncated_path = temp.path().join("truncated.json");
    fs::write(&truncated_path, truncated).expect("truncated JSON file");
    let error = load_expected_manifest_v1_file(&truncated_path).expect_err("truncated JSON");
    assert!(matches!(
        error,
        ManifestLoadError::Json(ManifestJsonError::Decode { ref source }) if source.is_eof()
    ));
}

#[test]
fn domain_file_count_limit_remains_the_only_manifest_admission_path() {
    let mut value: serde_json::Value =
        serde_json::from_slice(&manifest_json(b"calibration", b"plant")).expect("fixture");
    let first = value["calibration_artifacts"][0].clone();
    value["calibration_artifacts"] = serde_json::Value::Array(
        (0..9)
            .map(|index| {
                let mut artifact = first.clone();
                artifact["artifact_id"] = json!(format!("calibration-{index}"));
                artifact["sha256"] = json!(vec![index + 1; 32]);
                artifact
            })
            .collect(),
    );
    assert!(matches!(
        load_expected_manifest_v1_from_slice(
            &serde_json::to_vec(&value).expect("oversized collection fixture")
        ),
        Err(ManifestLoadError::Domain {
            source: InventoryParseError::TooManyArtifacts {
                kind: ArtifactKind::Calibration,
                actual: 9,
                maximum: 8,
            }
        })
    ));
}

#[test]
fn hashes_exact_manifest_bindings_and_exposes_changed_content() {
    let calibration = b"camera calibration v3";
    let plant = b"drive plant v4";
    let manifest = load_expected_manifest_v1_from_slice(&manifest_json(calibration, plant))
        .expect("manifest")
        .into_manifest();
    let temp = TempDirectory::new();
    write_artifacts(temp.path(), calibration, plant);

    let exact = hash_manifest_artifacts(&manifest, temp.path(), parsed_bindings()).expect("hashes");
    assert_eq!(exact.artifact_root_path(), temp.path());
    assert_eq!(exact.len(), 2);
    assert!(exact.all_content_matches_manifest());
    assert!(exact.iter().all(|entry| entry.bytes_hashed() != 0));
    assert_eq!(
        exact
            .exact_calibration_bundle_sha256()
            .expect("exact calibration bundle")
            .as_bytes(),
        &[
            0x7f, 0x85, 0xe8, 0x11, 0x0d, 0x10, 0xb3, 0x65, 0xe5, 0xbe, 0xac, 0x20, 0x96, 0xa7,
            0xc4, 0xdd, 0x37, 0x94, 0x93, 0x89, 0x4c, 0xce, 0x20, 0x20, 0x8c, 0x2f, 0x3e, 0xb7,
            0x08, 0xec, 0x2c, 0x89,
        ]
    );

    fs::write(temp.path().join("plant/main.bin"), b"drive plant X4")
        .expect("changed same-length artifact");
    let changed =
        hash_manifest_artifacts(&manifest, temp.path(), parsed_bindings()).expect("changed hash");
    assert!(!changed.all_content_matches_manifest());
    assert!(matches!(
        changed.exact_calibration_bundle_sha256(),
        Err(CalibrationBundleHashError::ContentMismatch { .. })
    ));
    let plant_identity = changed
        .iter()
        .find(|entry| entry.kind() == ArtifactKind::Plant)
        .expect("plant identity");
    assert!(!plant_identity.content_matches_manifest());
    assert_ne!(
        plant_identity.expected_sha256(),
        plant_identity.observed_sha256()
    );
    assert_eq!(
        plant_identity.to_observed_digest_dto().sha256,
        *plant_identity.observed_sha256()
    );
}

#[test]
fn retained_launch_asset_is_not_reopened_during_manifest_hashing() {
    let calibration = b"camera calibration v3";
    let plant = b"drive plant v4";
    let manifest = load_expected_manifest_v1_from_slice(&manifest_json(calibration, plant))
        .expect("manifest")
        .into_manifest();
    let temp = TempDirectory::new();
    let artifact_root = temp.path().join("artifacts");
    write_artifacts(&artifact_root, calibration, plant);
    let loaded_plant = load_deployment_asset(
        temp.path(),
        ArtifactRelativePath::parse("artifacts/plant/main.bin".to_owned()).expect("asset path"),
        DeploymentAssetByteLimit::try_new(1_024).expect("asset bound"),
    )
    .expect("retained plant");

    fs::remove_file(artifact_root.join("plant/main.bin"))
        .expect("remove launch pathname after retaining bytes");
    let hashes = hash_manifest_artifacts_reusing_loaded_asset(
        &manifest,
        temp.path(),
        &artifact_root,
        parsed_bindings(),
        &loaded_plant,
    )
    .expect("retained plant plus live calibration hash");
    assert!(hashes.all_content_matches_manifest());
    assert_eq!(
        hashes
            .iter()
            .find(|entry| entry.kind() == ArtifactKind::Plant)
            .expect("plant identity")
            .bytes_hashed(),
        u64::try_from(plant.len()).expect("test length fits u64")
    );
}

#[test]
fn binding_set_and_paths_are_exact_without_fallback() {
    let manifest = load_expected_manifest_v1_from_slice(&manifest_json(b"calibration", b"plant"))
        .expect("manifest")
        .into_manifest();
    let temp = TempDirectory::new();
    write_artifacts(temp.path(), b"calibration", b"plant");

    let mut extra = bindings();
    extra.push(ArtifactFileBindingInput {
        kind: ArtifactKind::Calibration,
        artifact_id: "calibration-extra".to_owned(),
        relative_path: "calibration/extra.bin".to_owned(),
    });
    assert!(matches!(
        hash_manifest_artifacts(
            &manifest,
            temp.path(),
            ArtifactFileBindingSet::parse(extra).expect("bounded extra binding")
        ),
        Err(ArtifactHashError::BindingCountMismatch { .. })
    ));

    let mut traversal = bindings();
    traversal[0].relative_path = "../calibration/main.bin".to_owned();
    assert!(matches!(
        ArtifactFileBindingSet::parse(traversal),
        Err(ArtifactFileBindingParseError::InvalidRelativePath { .. })
    ));

    let mut wrong_id = bindings();
    wrong_id[0].artifact_id = "calibration-fallback".to_owned();
    assert!(matches!(
        hash_manifest_artifacts(
            &manifest,
            temp.path(),
            ArtifactFileBindingSet::parse(wrong_id).expect("valid unexpected binding")
        ),
        Err(ArtifactHashError::UnexpectedBinding { .. })
    ));
}

#[test]
fn binding_parser_owns_collection_and_lexical_invariants_once() {
    let mut missing_plant = bindings();
    missing_plant.pop();
    assert!(matches!(
        ArtifactFileBindingSet::parse(missing_plant),
        Err(ArtifactFileBindingParseError::MissingRequiredKind {
            kind: ArtifactKind::Plant
        })
    ));

    let mut duplicate_id = bindings();
    duplicate_id[1].artifact_id = duplicate_id[0].artifact_id.clone();
    assert!(matches!(
        ArtifactFileBindingSet::parse(duplicate_id),
        Err(ArtifactFileBindingParseError::DuplicateArtifactId { .. })
    ));

    let mut duplicate_path = bindings();
    duplicate_path[1].relative_path = duplicate_path[0].relative_path.clone();
    assert!(matches!(
        ArtifactFileBindingSet::parse(duplicate_path),
        Err(ArtifactFileBindingParseError::DuplicateRelativePath { .. })
    ));

    let mut too_many_calibrations = bindings();
    for index in 1..=8 {
        too_many_calibrations.push(ArtifactFileBindingInput {
            kind: ArtifactKind::Calibration,
            artifact_id: format!("calibration-{index}"),
            relative_path: format!("calibration/{index}.bin"),
        });
    }
    assert!(matches!(
        ArtifactFileBindingSet::parse(too_many_calibrations),
        Err(ArtifactFileBindingParseError::TooManyBindings {
            kind: ArtifactKind::Calibration,
            actual: 9,
            maximum: 8,
        })
    ));
}

#[test]
fn absolute_paths_reject_aliases_before_filesystem_access() {
    let manifest = load_expected_manifest_v1_from_slice(&manifest_json(b"calibration", b"plant"))
        .expect("manifest")
        .into_manifest();
    let temp = TempDirectory::new();
    write_artifacts(temp.path(), b"calibration", b"plant");

    assert!(matches!(
        load_expected_manifest_v1_file(Path::new("expected-v1.json")),
        Err(ManifestLoadError::PathNotAbsolute { .. })
    ));
    for aliased in [
        format!("{}/./expected-v1.json", temp.path().display()),
        format!("{}//expected-v1.json", temp.path().display()),
        format!("{}/", temp.path().display()),
    ] {
        assert!(matches!(
            load_expected_manifest_v1_file(Path::new(&aliased)),
            Err(ManifestLoadError::NonCanonicalPath { .. })
        ));
    }

    assert!(matches!(
        hash_manifest_artifacts(&manifest, Path::new("artifacts"), parsed_bindings()),
        Err(ArtifactHashError::RootNotAbsolute { .. })
    ));
    for aliased in [
        format!("{}/./", temp.path().display()),
        format!("{}//artifacts", temp.path().display()),
        format!("{}/../artifacts", temp.path().display()),
    ] {
        assert!(matches!(
            hash_manifest_artifacts(&manifest, Path::new(&aliased), parsed_bindings()),
            Err(ArtifactHashError::NonCanonicalRootPath { .. })
        ));
    }
}

#[test]
fn artifact_size_non_regular_and_symlink_paths_fail_closed() {
    use std::os::unix::fs::symlink;

    let manifest = load_expected_manifest_v1_from_slice(&manifest_json(b"calibration", b"plant"))
        .expect("manifest")
        .into_manifest();

    let oversized_root = TempDirectory::new();
    write_artifacts(oversized_root.path(), b"calibration", b"plant");
    File::create(oversized_root.path().join("calibration/main.bin"))
        .expect("oversized artifact")
        .set_len(MAX_ARTIFACT_FILE_BYTES + 1)
        .expect("sparse artifact");
    assert!(matches!(
        hash_manifest_artifacts(&manifest, oversized_root.path(), parsed_bindings()),
        Err(ArtifactHashError::ArtifactTooLarge { .. })
    ));

    let directory_root = TempDirectory::new();
    write_artifacts(directory_root.path(), b"calibration", b"plant");
    fs::remove_file(directory_root.path().join("calibration/main.bin")).expect("remove file");
    fs::create_dir(directory_root.path().join("calibration/main.bin")).expect("directory artifact");
    assert!(matches!(
        hash_manifest_artifacts(&manifest, directory_root.path(), parsed_bindings()),
        Err(ArtifactHashError::NotRegularFile { .. })
    ));

    let symlink_root = TempDirectory::new();
    write_artifacts(symlink_root.path(), b"calibration", b"plant");
    let target = symlink_root.path().join("calibration/target.bin");
    fs::write(&target, b"calibration").expect("symlink target");
    fs::remove_file(symlink_root.path().join("calibration/main.bin")).expect("remove file");
    symlink(&target, symlink_root.path().join("calibration/main.bin")).expect("artifact symlink");
    assert!(matches!(
        hash_manifest_artifacts(&manifest, symlink_root.path(), parsed_bindings()),
        Err(ArtifactHashError::OpenArtifact { .. })
    ));

    let real_root = TempDirectory::new();
    write_artifacts(real_root.path(), b"calibration", b"plant");
    let parent = TempDirectory::new();
    let linked_root = parent.path().join("linked-root");
    symlink(real_root.path(), &linked_root).expect("root symlink");
    assert!(matches!(
        hash_manifest_artifacts(&manifest, &linked_root, parsed_bindings()),
        Err(ArtifactHashError::OpenRoot { .. })
    ));
}

#[test]
fn manifest_directory_and_symlink_are_never_followed_as_files() {
    use std::os::unix::fs::symlink;

    let temp = TempDirectory::new();
    assert!(matches!(
        load_expected_manifest_v1_file(temp.path()),
        Err(ManifestLoadError::NotRegularFile { .. })
    ));

    let target = temp.path().join("target.json");
    fs::write(&target, manifest_json(b"calibration", b"plant")).expect("target manifest");
    let link = temp.path().join("manifest.json");
    symlink(&target, &link).expect("manifest symlink");
    assert!(matches!(
        load_expected_manifest_v1_file(&link),
        Err(ManifestLoadError::Open { .. })
    ));
}
