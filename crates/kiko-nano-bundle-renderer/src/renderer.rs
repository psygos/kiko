use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};

use robot_server::config::{ControllerServerConfigV1, ControllerServerConfigV2, ServerConfigError};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::input::{
    BundleSelection, InputError, NativeLibraryRole, ProductionControllerProfile,
    ProductionControllerProfileDto, RenderInput, RenderInputDto, WarmStartSelection, encode_hex,
};

const MAX_RENDER_INPUT_BYTES: u64 = 1_048_576;
const MAX_PRODUCTION_PROFILE_BYTES: u64 = 262_144;
const MAX_SOURCE_ASSET_BYTES: u64 = 512 * 1_024 * 1_024;
const MAX_QUALIFICATION_EXECUTABLE_BYTES: u64 = 128 * 1_024 * 1_024;
const MAX_OPENCV_HAAR_CASCADE_BYTES: u64 = 4 * 1_024 * 1_024;
// Production V3 currently emits 23 deterministic leaves including source
// evidence, the render manifest, and launch. Wheels-off emits fewer.
const STAGED_BUNDLE_FILE_CAPACITY: usize = 23;
const EVIDENCE_SCHEMA_VERSION: u32 = 1;
const PLAN_SCHEMA_VERSION: u32 = 1;
const EVIDENCE_SCOPE: &str = "offline_staging_only_not_installation_or_hardware_qualification_v1";
const PRODUCTION_PROFILE_EVIDENCE_PATH: &str = "evidence/production-controller-profile-v1.json";
const RENDER_EVIDENCE_PATH: &str = "evidence/render-evidence-v1.json";
const QUALIFICATION_EXECUTABLE_RELATIVE_PATH: &str = "bin/kiko-nano-wheels-off-qualification";

#[derive(Clone, Copy, Debug)]
pub enum RenderMode<'a> {
    /// Parse, inspect, hash, and render entirely in memory.
    DryRun,
    /// Write one new read-only staging tree. This is never an install action.
    Stage { destination: &'a Path },
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BundleFileEvidence {
    pub role: String,
    pub relative_path: String,
    pub byte_len: u64,
    pub sha256_hex: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BundlePlanEvidence {
    pub schema_version: u32,
    pub evidence_scope: String,
    pub bundle_kind: String,
    pub launch_relative_path: String,
    pub files: Vec<BundleFileEvidence>,
}

struct LoadedSource {
    role: String,
    path: PathBuf,
    bytes: Vec<u8>,
    byte_len: u64,
    sha256: [u8; 32],
}

impl LoadedSource {
    fn read(role: &str, path: &Path, maximum_bytes: u64) -> Result<Self, RenderError> {
        validate_absolute_path(path)?;
        reject_symlink_components(path)?;
        let metadata = fs::symlink_metadata(path).map_err(|source| RenderError::ReadMetadata {
            path: path.to_path_buf(),
            source,
        })?;
        if !metadata.is_file() {
            return Err(RenderError::SourceNotRegular {
                path: path.to_path_buf(),
            });
        }
        let byte_len = metadata.len();
        if byte_len == 0 || byte_len > maximum_bytes {
            return Err(RenderError::SourceSizeOutsideBound {
                path: path.to_path_buf(),
                byte_len,
                maximum_bytes,
            });
        }
        let capacity = usize::try_from(byte_len).map_err(|_| RenderError::SourceSizeNotUsize {
            path: path.to_path_buf(),
            byte_len,
        })?;
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(path)
            .map_err(|source| RenderError::ReadSource {
                path: path.to_path_buf(),
                source,
            })?;
        let opened_metadata = file
            .metadata()
            .map_err(|source| RenderError::ReadMetadata {
                path: path.to_path_buf(),
                source,
            })?;
        if !opened_metadata.is_file()
            || opened_metadata.dev() != metadata.dev()
            || opened_metadata.ino() != metadata.ino()
            || opened_metadata.len() != metadata.len()
        {
            return Err(RenderError::SourceIdentityChangedBeforeRead {
                path: path.to_path_buf(),
            });
        }
        let mut bytes = Vec::with_capacity(capacity);
        Read::by_ref(&mut file)
            .take(maximum_bytes.saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|source| RenderError::ReadSource {
                path: path.to_path_buf(),
                source,
            })?;
        let observed_len =
            u64::try_from(bytes.len()).map_err(|_| RenderError::SourceSizeNotUsize {
                path: path.to_path_buf(),
                byte_len,
            })?;
        if observed_len != byte_len {
            return Err(RenderError::SourceChangedDuringRead {
                path: path.to_path_buf(),
                metadata_bytes: byte_len,
                retained_bytes: observed_len,
            });
        }
        let final_metadata = file
            .metadata()
            .map_err(|source| RenderError::ReadMetadata {
                path: path.to_path_buf(),
                source,
            })?;
        if final_metadata.dev() != metadata.dev()
            || final_metadata.ino() != metadata.ino()
            || final_metadata.len() != metadata.len()
        {
            return Err(RenderError::SourceIdentityChangedBeforeRead {
                path: path.to_path_buf(),
            });
        }
        reject_symlink_components(path)?;
        let sha256: [u8; 32] = Sha256::digest(&bytes).into();
        Ok(Self {
            role: role.to_owned(),
            path: path.to_path_buf(),
            bytes,
            byte_len,
            sha256,
        })
    }

    fn evidence(&self, destination_relative_path: Option<&str>) -> SourceEvidence {
        SourceEvidence {
            role: self.role.clone(),
            source_path: self.path.clone(),
            destination_relative_path: destination_relative_path.map(ToOwned::to_owned),
            byte_len: self.byte_len,
            sha256_hex: encode_hex(&self.sha256),
        }
    }
}

struct StagedFile {
    role: String,
    relative_path: String,
    bytes: Vec<u8>,
    byte_len: u64,
    sha256: [u8; 32],
    executable: bool,
}

impl StagedFile {
    fn retained(
        role: impl Into<String>,
        relative_path: impl Into<String>,
        source: LoadedSource,
    ) -> Self {
        Self {
            role: role.into(),
            relative_path: relative_path.into(),
            bytes: source.bytes,
            byte_len: source.byte_len,
            sha256: source.sha256,
            executable: false,
        }
    }

    fn retained_executable(
        role: impl Into<String>,
        relative_path: impl Into<String>,
        source: LoadedSource,
    ) -> Self {
        Self {
            role: role.into(),
            relative_path: relative_path.into(),
            bytes: source.bytes,
            byte_len: source.byte_len,
            sha256: source.sha256,
            executable: true,
        }
    }

    fn retained_clone(
        role: impl Into<String>,
        relative_path: impl Into<String>,
        source: &LoadedSource,
    ) -> Self {
        Self {
            role: role.into(),
            relative_path: relative_path.into(),
            bytes: source.bytes.clone(),
            byte_len: source.byte_len,
            sha256: source.sha256,
            executable: false,
        }
    }

    fn rendered_json(
        role: impl Into<String>,
        relative_path: impl Into<String>,
        value: &impl Serialize,
    ) -> Result<Self, RenderError> {
        let mut bytes =
            serde_json::to_vec_pretty(value).map_err(RenderError::SerializeRenderedJson)?;
        bytes.push(b'\n');
        Self::rendered_bytes(role, relative_path, bytes)
    }

    fn rendered_bytes(
        role: impl Into<String>,
        relative_path: impl Into<String>,
        bytes: Vec<u8>,
    ) -> Result<Self, RenderError> {
        let role = role.into();
        let relative_path = relative_path.into();
        validate_relative_path(&relative_path)?;
        let byte_len = u64::try_from(bytes.len()).map_err(|_| RenderError::RenderedSizeNotU64 {
            relative_path: relative_path.clone(),
        })?;
        let sha256 = Sha256::digest(&bytes).into();
        Ok(Self {
            role,
            relative_path,
            bytes,
            byte_len,
            sha256,
            executable: false,
        })
    }

    fn evidence(&self) -> BundleFileEvidence {
        BundleFileEvidence {
            role: self.role.clone(),
            relative_path: self.relative_path.clone(),
            byte_len: self.byte_len,
            sha256_hex: encode_hex(&self.sha256),
        }
    }
}

struct LoadedAssets {
    qualification_executable: Option<LoadedSource>,
    calibration: LoadedSource,
    plant: LoadedSource,
    navigation_shadow: LoadedSource,
    superpoint_model: LoadedSource,
    lightglue_model: LoadedSource,
    face_perception: Option<LoadedFacePerceptionAssets>,
    native_libraries: BTreeMap<NativeLibraryRole, LoadedSource>,
}

struct LoadedFacePerceptionAssets {
    frontal_face_cascade: LoadedSource,
    profile_face_cascade: LoadedSource,
}

struct ProductionAdmission {
    source: LoadedSource,
    profile: ProductionControllerProfile,
}

/// Parse one exact render-input file and produce either a dry-run plan or a
/// staging tree. No code path installs files or opens a hardware endpoint.
pub fn render_bundle(
    input_path: &Path,
    mode: RenderMode<'_>,
) -> Result<BundlePlanEvidence, RenderError> {
    let input_source = LoadedSource::read("render_input", input_path, MAX_RENDER_INPUT_BYTES)?;
    reject_unresolved_tokens("render input", &input_source.bytes)?;
    let input_dto: RenderInputDto = parse_exact_json(&input_source.bytes, input_path)?;
    let input = RenderInput::parse(input_dto).map_err(RenderError::Input)?;

    let production_admission = load_production_admission(&input)?;
    let loaded_assets = load_assets(&input)?;
    let mut staged = render_non_launch_files(
        &input,
        loaded_assets,
        production_admission
            .as_ref()
            .map(|admission| &admission.profile),
    )?;

    let input_evidence = StagedFile::retained_clone(
        "render_input_evidence",
        input.bundle.render_input_evidence_path(),
        &input_source,
    );
    staged.push(input_evidence);
    if let Some(admission) = &production_admission {
        staged.push(StagedFile::retained_clone(
            "production_controller_profile_evidence",
            PRODUCTION_PROFILE_EVIDENCE_PATH,
            &admission.source,
        ));
    }

    ensure_unique_relative_paths(&staged)?;
    let launch = render_launch(
        &input,
        &staged,
        production_admission
            .as_ref()
            .map(|admission| &admission.profile),
    )?;
    let render_evidence = render_evidence_manifest(
        &input,
        &input_source,
        &production_admission,
        &staged,
        &launch,
    )?;
    // Evidence is written before launch. The launch contract is always the
    // final file in the deterministic staging sequence.
    staged.push(render_evidence);
    staged.push(launch);
    ensure_unique_relative_paths(&staged)?;
    validate_rendered_tree(&staged)?;

    let plan = BundlePlanEvidence {
        schema_version: PLAN_SCHEMA_VERSION,
        evidence_scope: EVIDENCE_SCOPE.to_owned(),
        bundle_kind: input.bundle.kind_name().to_owned(),
        launch_relative_path: input.bundle.launch_relative_path().to_owned(),
        files: staged.iter().map(StagedFile::evidence).collect(),
    };
    if let RenderMode::Stage { destination } = mode {
        write_staging_tree(destination, &staged)?;
    }
    Ok(plan)
}

fn load_production_admission(
    input: &RenderInput,
) -> Result<Option<ProductionAdmission>, RenderError> {
    let BundleSelection::Production {
        production_controller_profile_path,
        ..
    } = &input.bundle
    else {
        return Ok(None);
    };
    let profile_path = production_controller_profile_path
        .as_ref()
        .ok_or(RenderError::MissingProductionControllerProfile)?;
    let source = LoadedSource::read(
        "production_controller_profile",
        profile_path.as_path(),
        MAX_PRODUCTION_PROFILE_BYTES,
    )?;
    reject_unresolved_tokens("production controller profile", &source.bytes)?;
    let dto: ProductionControllerProfileDto =
        parse_exact_json(&source.bytes, source.path.as_path())?;
    let profile = ProductionControllerProfile::parse(dto, &input.discovery.stm32)
        .map_err(RenderError::Input)?;
    Ok(Some(ProductionAdmission { source, profile }))
}

fn load_assets(input: &RenderInput) -> Result<LoadedAssets, RenderError> {
    let qualification_executable = match &input.bundle {
        BundleSelection::WheelsOffQualification {
            qualification_executable_path,
        } => Some(LoadedSource::read(
            "qualification_executable",
            qualification_executable_path.as_path(),
            MAX_QUALIFICATION_EXECUTABLE_BYTES,
        )?),
        BundleSelection::Production { .. } => None,
    };
    let calibration = LoadedSource::read(
        "calibration",
        input.assets.calibration.source_path.as_path(),
        MAX_SOURCE_ASSET_BYTES,
    )?;
    reject_tokens_if_json(&calibration)?;
    let plant = LoadedSource::read(
        "plant",
        input.assets.plant.source_path.as_path(),
        MAX_SOURCE_ASSET_BYTES,
    )?;
    reject_tokens_if_json(&plant)?;
    let navigation_shadow = LoadedSource::read(
        "navigation_shadow",
        input.assets.navigation_shadow_source_path.as_path(),
        MAX_SOURCE_ASSET_BYTES,
    )?;
    reject_unresolved_tokens("navigation shadow", &navigation_shadow.bytes)?;
    ensure_json_value(&navigation_shadow)?;
    let superpoint_model = LoadedSource::read(
        "superpoint_model",
        input.assets.superpoint_model.source_path.as_path(),
        MAX_SOURCE_ASSET_BYTES,
    )?;
    let lightglue_model = LoadedSource::read(
        "lightglue_model",
        input.assets.lightglue_model.source_path.as_path(),
        MAX_SOURCE_ASSET_BYTES,
    )?;
    let face_perception = input
        .bundle
        .face_perception()
        .map(|face| {
            Ok(LoadedFacePerceptionAssets {
                frontal_face_cascade: LoadedSource::read(
                    "frontal_face_cascade",
                    face.frontal_face_cascade.source_path.as_path(),
                    MAX_OPENCV_HAAR_CASCADE_BYTES,
                )?,
                profile_face_cascade: LoadedSource::read(
                    "profile_face_cascade",
                    face.profile_face_cascade.source_path.as_path(),
                    MAX_OPENCV_HAAR_CASCADE_BYTES,
                )?,
            })
        })
        .transpose()?;
    let mut native_libraries = BTreeMap::new();
    for library in &input.native_libraries {
        let loaded = LoadedSource::read(
            library.role.as_str(),
            library.source_path.as_path(),
            MAX_SOURCE_ASSET_BYTES,
        )?;
        if native_libraries.insert(library.role, loaded).is_some() {
            return Err(RenderError::DuplicateRoleAfterParsing {
                role: library.role.as_str(),
            });
        }
    }
    Ok(LoadedAssets {
        qualification_executable,
        calibration,
        plant,
        navigation_shadow,
        superpoint_model,
        lightglue_model,
        face_perception,
        native_libraries,
    })
}

fn render_non_launch_files(
    input: &RenderInput,
    mut loaded: LoadedAssets,
    production: Option<&ProductionControllerProfile>,
) -> Result<Vec<StagedFile>, RenderError> {
    let mut staged = Vec::with_capacity(STAGED_BUNDLE_FILE_CAPACITY);

    if let Some(executable) = loaded.qualification_executable.take() {
        staged.push(StagedFile::retained_executable(
            "qualification_executable",
            QUALIFICATION_EXECUTABLE_RELATIVE_PATH,
            executable,
        ));
    }
    staged.push(StagedFile::retained(
        "calibration",
        input.assets.calibration.destination_relative_path.as_str(),
        loaded.calibration,
    ));
    staged.push(StagedFile::retained(
        "plant",
        input.assets.plant.destination_relative_path.as_str(),
        loaded.plant,
    ));
    staged.push(StagedFile::retained(
        "navigation_shadow",
        "navigation-shadow-v1.json",
        loaded.navigation_shadow,
    ));
    staged.push(StagedFile::retained(
        "superpoint_model",
        input
            .assets
            .superpoint_model
            .destination_relative_path
            .as_str(),
        loaded.superpoint_model,
    ));
    staged.push(StagedFile::retained(
        "lightglue_model",
        input
            .assets
            .lightglue_model
            .destination_relative_path
            .as_str(),
        loaded.lightglue_model,
    ));
    if let Some(face) = input.bundle.face_perception() {
        let loaded_face = loaded
            .face_perception
            .take()
            .expect("parsed production face assets were loaded as one typed set");
        staged.push(StagedFile::retained(
            "frontal_face_cascade",
            face.frontal_face_cascade.destination_relative_path.as_str(),
            loaded_face.frontal_face_cascade,
        ));
        staged.push(StagedFile::retained(
            "profile_face_cascade",
            face.profile_face_cascade.destination_relative_path.as_str(),
            loaded_face.profile_face_cascade,
        ));
    }
    debug_assert!(loaded.face_perception.is_none());
    for library in &input.native_libraries {
        let source = loaded.native_libraries.remove(&library.role).ok_or(
            RenderError::DuplicateRoleAfterParsing {
                role: library.role.as_str(),
            },
        )?;
        staged.push(StagedFile::retained(
            format!("native_library_{}", library.role.as_str()),
            library.destination_relative_path.as_str(),
            source,
        ));
    }
    if !loaded.native_libraries.is_empty() {
        return Err(RenderError::UnexpectedNativeLibraryRemainder);
    }

    let native_runtime = render_native_runtime(input, &staged)?;
    staged.push(native_runtime);
    let inventory = render_inventory(input, &staged, production)?;
    staged.push(inventory);
    let controller = render_controller(input, production)?;
    staged.push(controller);
    if matches!(input.bundle, BundleSelection::WheelsOffQualification { .. }) {
        staged.push(render_candidate_policy()?);
    } else {
        let profile = production.ok_or(RenderError::MissingProductionControllerProfile)?;
        staged.push(render_navigation_actuation(input, &staged, profile)?);
    }
    staged.push(render_agent_policy(input, production)?);
    Ok(staged)
}

fn render_native_runtime(
    input: &RenderInput,
    staged: &[StagedFile],
) -> Result<StagedFile, RenderError> {
    let mut libraries = Vec::with_capacity(input.native_libraries.len());
    for library in &input.native_libraries {
        let file = find_staged(staged, library.destination_relative_path.as_str())?;
        libraries.push(json!({
            "role": library.role.as_str(),
            "soname": library.soname,
            "relative_path": library.destination_relative_path.as_str(),
            "maximum_bytes": file.byte_len,
            "sha256_hex": encode_hex(&file.sha256),
        }));
    }
    StagedFile::rendered_json(
        "native_runtime_manifest",
        "native-runtime-v1.json",
        &json!({
            "schema_version": 1,
            "library_search_relative_path": "lib",
            "libraries": libraries,
        }),
    )
}

fn render_inventory(
    input: &RenderInput,
    staged: &[StagedFile],
    production: Option<&ProductionControllerProfile>,
) -> Result<StagedFile, RenderError> {
    let calibration = find_staged(
        staged,
        input.assets.calibration.destination_relative_path.as_str(),
    )?;
    let plant = find_staged(
        staged,
        input.assets.plant.destination_relative_path.as_str(),
    )?;
    let discovery = &input.discovery;
    let mut value = json!({
        "schema_version": match input.bundle {
            BundleSelection::WheelsOffQualification { .. } => 2,
            BundleSelection::Production { .. } => 1,
        },
        "robot_id": input.robot_id,
        "oak": {
            "mxid": discovery.oak.mxid,
            "compiled_depthai_header_sdk_version": discovery.oak.sdk_version,
            "compiled_depthai_header_sdk_commit": discovery.oak.sdk_commit,
            "compiled_depthai_header_embedded_device_artifact_version":
                discovery.oak.device_artifact_version,
            "compiled_depthai_header_embedded_bootloader_artifact_version":
                discovery.oak.bootloader_artifact_version,
        },
        "stm32": {
            "serial_by_id_path": discovery.stm32.serial_by_id_path,
            "control_endpoint_identity": match production {
                Some(profile) => {
                    format!("udp://127.0.0.1:{}", profile.controller.command_udp_port)
                }
                None => "udp://127.0.0.1:8080".to_owned(),
            },
            "controller_uid": discovery.stm32.controller_uid,
            "firmware_abi": discovery.stm32.firmware_abi,
            "firmware_build_id": discovery.stm32.firmware_build_id,
            "hardware_profile_fingerprint": discovery.stm32.hardware_profile_fingerprint,
            "capabilities_bits": discovery.stm32.capabilities_bits,
        },
        "head": {
            "adapter_serial_by_id_path": discovery.head.serial_by_id_path,
            "bow_servo_id": discovery.head.servo_ids[0],
            "curl_servo_id": discovery.head.servo_ids[1],
            "yaw_servo_id": discovery.head.servo_ids[2],
            "roll_servo_id": discovery.head.servo_ids[3],
            "baud_rate_bps": discovery.head.baud_rate_bps,
            "dtr_asserted": discovery.head.dtr_asserted,
            "rts_asserted": discovery.head.rts_asserted,
        },
        "eye": {
            "serial_by_id_path": discovery.eye.serial_by_id_path,
            "kep_protocol_version": discovery.eye.kep_protocol_version,
            "device_uid": discovery.eye.device_uid,
            "firmware_build_id": discovery.eye.firmware_build_id,
            "capabilities_bits": discovery.eye.capabilities_bits,
        },
        "calibration_artifacts": [{
            "artifact_id": input.assets.calibration.artifact_id,
            "sha256": calibration.sha256,
        }],
        "plant_artifacts": [{
            "artifact_id": input.assets.plant.artifact_id,
            "sha256": plant.sha256,
        }],
    });
    let (role, relative_path) = match input.bundle {
        BundleSelection::WheelsOffQualification { .. } => {
            let stm32 = value
                .get_mut("stm32")
                .and_then(Value::as_object_mut)
                .ok_or(RenderError::InternalJsonShape)?;
            stm32.insert(
                "controller_session_class".to_owned(),
                json!("operator_supervised_four_pwm_candidate"),
            );
            stm32.insert("expected_max_abs_pwm_percent".to_owned(), json!(30));
            stm32.insert(
                "expected_physical_stop_semantics".to_owned(),
                json!("unverified"),
            );
            ("candidate_inventory", "device-inventory-candidate-v2.json")
        }
        BundleSelection::Production { .. } => ("device_inventory", "device-inventory-v1.json"),
    };
    StagedFile::rendered_json(role, relative_path, &value)
}

fn render_controller(
    input: &RenderInput,
    production: Option<&ProductionControllerProfile>,
) -> Result<StagedFile, RenderError> {
    let stm32 = &input.discovery.stm32;
    let rendered = match &input.bundle {
        BundleSelection::WheelsOffQualification { .. } => StagedFile::rendered_json(
            "candidate_controller_contract",
            "controller-server-candidate-v2.json",
            &json!({
                "schema_version": 2,
                "serial_device": stm32.serial_by_id_path,
                "controller_uid_hex": encode_hex(&stm32.controller_uid),
                "firmware_abi": 2,
                "firmware_build_id": 135169,
                "actuator_config_fingerprint_hex":
                    encode_hex(&stm32.hardware_profile_fingerprint),
                "hardware_profile_claim_id": "kiko-four-pwm-candidate-wheels-off-v1",
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
                "expected_physical_stop_semantics": "unverified",
                "controller_session_class": "operator_supervised_four_pwm_candidate",
            }),
        ),
        BundleSelection::Production { .. } => {
            let profile = production.ok_or(RenderError::MissingProductionControllerProfile)?;
            let controller = &profile.controller;
            StagedFile::rendered_json(
                "production_controller_contract",
                "controller-server-v1.json",
                &json!({
                    "schema_version": 1,
                    "serial_device": stm32.serial_by_id_path,
                    "controller_uid_hex": encode_hex(&controller.controller_uid),
                    "firmware_abi": controller.firmware_abi,
                    "firmware_build_id": controller.firmware_build_id,
                    "actuator_config_fingerprint_hex":
                        encode_hex(&controller.actuator_config_fingerprint),
                    "hardware_profile_claim_id": controller.hardware_profile_claim_id,
                    "controller_ready_timeout_ms": controller.controller_ready_timeout_ms,
                    "heartbeat_period_ms": controller.heartbeat_period_ms,
                    "maximum_heartbeat_age_ms": controller.maximum_heartbeat_age_ms,
                    "maximum_host_command_rate_hz":
                        controller.maximum_host_command_rate_hz,
                    "serial_transmit_timeout_ms": controller.serial_transmit_timeout_ms,
                    "serial_applied_ack_timeout_ms":
                        controller.serial_applied_ack_timeout_ms,
                    "controller_clock_abs_error_ppm_bound":
                        controller.controller_clock_abs_error_ppm_bound,
                    "deadline_quantization_margin_ms":
                        controller.deadline_quantization_margin_ms,
                    "expected_max_abs_pwm_percent":
                        controller.expected_max_abs_pwm_percent,
                    "expected_pwm_frequency_hz": controller.expected_pwm_frequency_hz,
                    "expected_watchdog_nominal_timeout_ms":
                        controller.expected_watchdog_nominal_timeout_ms,
                    "expected_neutral_output": controller.expected_neutral_output.as_str(),
                    "expected_physical_stop_semantics":
                        controller.expected_physical_stop_semantics.as_str(),
                }),
            )
        }
    }?;
    match &input.bundle {
        BundleSelection::WheelsOffQualification { .. } => {
            ControllerServerConfigV2::parse_json(&rendered.bytes)
                .map_err(RenderError::GeneratedControllerContract)?;
        }
        BundleSelection::Production { .. } => {
            ControllerServerConfigV1::parse_json(&rendered.bytes)
                .map_err(RenderError::GeneratedControllerContract)?;
        }
    }
    Ok(rendered)
}

fn render_candidate_policy() -> Result<StagedFile, RenderError> {
    StagedFile::rendered_json(
        "candidate_controller_policy",
        "candidate-controller-policy-v1.json",
        &json!({
            "schema_version": 1,
            "command_endpoint": "127.0.0.1:8080",
            "status_timeout_ns": 40000000,
            "acquire_timeout_ns": 40000000,
            "applied_ack_timeout_ns": 40000000,
            "stop_attempt_timeout_ns": 40000000,
            "maximum_stop_recovery_attempts": 3,
            "command_lease_ms": 100,
            "command_interval_ns": 20000000,
            "scheduling_margin_ns": 5000000,
            "local_max_abs_pwm_percent": 30,
            "manual_test_magnitude_timer_pwm_percent": 10,
            "manual_deadman_ms": 150,
            "maximum_attestation_age_ns": 30000000000_u64,
        }),
    )
}

fn render_navigation_actuation(
    input: &RenderInput,
    staged: &[StagedFile],
    production: &ProductionControllerProfile,
) -> Result<StagedFile, RenderError> {
    let navigation = find_staged(staged, "navigation-shadow-v1.json")?;
    let plant = find_staged(
        staged,
        input.assets.plant.destination_relative_path.as_str(),
    )?;
    let controller = &production.controller;
    let actuation = &production.actuation;
    let approval = &actuation.approval;
    StagedFile::rendered_json(
        "navigation_actuation",
        "navigation-actuation-v2.json",
        &json!({
            "schema_version": 2,
            "robot_id": input.robot_id,
            "command_endpoint": format!("127.0.0.1:{}", controller.command_udp_port),
            "navigation_config_sha256_hex": encode_hex(&navigation.sha256),
            "controller_uid_hex": encode_hex(&controller.controller_uid),
            "firmware_abi": controller.firmware_abi,
            "firmware_build_id": controller.firmware_build_id,
            "actuator_config_fingerprint_hex":
                encode_hex(&controller.actuator_config_fingerprint),
            "plant_model_id": actuation.plant_model_id,
            "plant_model_version": actuation.plant_model_version,
            "plant_artifact_sha256_hex": encode_hex(&plant.sha256),
            "operator_claimed_physical_approval": {
                "approval_id": approval.approval_id,
                "approver_id": approval.approver_id,
                "plant_dataset_content_id": approval.plant_dataset_content_id,
                "plant_identification_method_id": approval.plant_identification_method_id,
                "plant_sample_count": approval.plant_sample_count,
                "plant_fit_residuals": {
                    "left_velocity_rmse_mps": approval.residuals.left_velocity_rmse_mps,
                    "right_velocity_rmse_mps": approval.residuals.right_velocity_rmse_mps,
                    "yaw_rate_rmse_rad_s": approval.residuals.yaw_rate_rmse_rad_s,
                    "max_abs_velocity_error_mps":
                        approval.residuals.max_abs_velocity_error_mps,
                },
                "imu_calibration_id": approval.imu_calibration_id,
                "stereo_calibration_id": approval.stereo_calibration_id,
                "tracking_camera_to_base_calibration_id":
                    approval.tracking_camera_to_base_calibration_id,
            },
            "apply_ack_budget_ns": actuation.apply_ack_budget_ns,
            "stop_ack_budget_ns": actuation.stop_ack_budget_ns,
            "scheduling_guard_ns": actuation.scheduling_guard_ns,
            "controller_motion_lease_ms": actuation.controller_motion_lease_ms,
            "controller_deadline_tolerance_ns":
                actuation.controller_deadline_tolerance_ns,
            "maximum_uncommanded_motion_ns": actuation.maximum_uncommanded_motion_ns,
        }),
    )
}

fn render_agent_policy(
    input: &RenderInput,
    production: Option<&ProductionControllerProfile>,
) -> Result<StagedFile, RenderError> {
    let (deployment_root, state_root, live_mode_policy) = match &input.bundle {
        BundleSelection::WheelsOffQualification { .. } => (
            "/opt/kiko/qualification",
            "/var/lib/kiko-nano-qualification",
            json!({
                "startup": "disarmed_map_only",
                "manual": { "permission": "disabled" },
                "point_goal": { "permission": "disabled" },
                "frontier_explore": { "permission": "disabled" },
            }),
        ),
        BundleSelection::Production { .. } => {
            let profile = production.ok_or(RenderError::MissingProductionControllerProfile)?;
            (
                "/opt/kiko/deployment",
                "/var/lib/kiko-nano-agent",
                serde_json::to_value(&profile.live_mode_policy)
                    .map_err(RenderError::SerializeRenderedJson)?,
            )
        }
    };
    let head = &input.head_policy;
    let rgb = &input.rgb_expression_policy;
    let warm_start = match input.runtime.storage.warm_start {
        WarmStartSelection::None => json!({ "kind": "none" }),
        WarmStartSelection::DatasetReplay => json!({
            "kind": "dataset_replay",
            "occupancy_snapshot_path": "/var/lib/kiko-nano-agent/maps/current.kmap",
            "slam_dataset_directory_path": "/var/lib/kiko-nano-agent/navigation",
        }),
    };
    let runtime_response_timeout_ms = match &input.bundle {
        BundleSelection::WheelsOffQualification { .. } => 500,
        // Production SaveMap always creates a restart-ready terminal
        // checkpoint, including the first cold mapping session.
        BundleSelection::Production { .. } => 30_000,
    };
    let discovery = &input.discovery;
    StagedFile::rendered_json(
        "agent_policy",
        "agent-policy-v3.json",
        &json!({
            "schema_version": 3,
            "control": {
                "socket_path": "/run/kiko/agent-control.sock",
                "read_timeout_ms": 100,
                "write_timeout_ms": 100,
                "runtime_response_timeout_ms": runtime_response_timeout_ms,
                "terminal_response_timeout_ms": 300_000,
                "runtime_queue_capacity": 8,
                "operator_console": {
                    "bind_address": "127.0.0.1:9877",
                    "capability_path": "/run/kiko/operator-console.capability",
                    "deadman_tick_ms": 20,
                    "manual_command_forward_mm_per_s": 100,
                    "manual_command_yaw_millirad_per_s": 500,
                },
            },
            "inventory": {
                "manifest_path": format!(
                    "{deployment_root}/{}",
                    match input.bundle {
                        BundleSelection::WheelsOffQualification { .. } =>
                            "device-inventory-candidate-v2.json",
                        BundleSelection::Production { .. } =>
                            "device-inventory-v1.json",
                    }
                ),
                "artifact_root_path": format!("{deployment_root}/artifacts"),
                "artifact_bindings": [
                    {
                        "kind": "calibration",
                        "artifact_id": input.assets.calibration.artifact_id,
                        "relative_path":
                            input.assets.calibration.artifact_relative_path,
                    },
                    {
                        "kind": "plant",
                        "artifact_id": input.assets.plant.artifact_id,
                        "relative_path": input.assets.plant.artifact_relative_path,
                    },
                ],
            },
            "map_persistence": {
                "save_snapshot_path": format!("{state_root}/maps/current.kmap"),
                "warm_start": warm_start,
            },
            "eye": {
                "mode": "kep2",
                "device_path": discovery.eye.serial_by_id_path,
                "baud_rate_bps": 115200,
                "response_timeout_ms": 20,
                "write_timeout_ms": 5,
                "write_attempts": 2,
                "empty_delimiter_budget": 2,
                "expected_device_uid": discovery.eye.device_uid,
                "expected_firmware_build_id": discovery.eye.firmware_build_id,
                "expected_capabilities_bits": discovery.eye.capabilities_bits,
                "intent_lease_ms": 100,
            },
            "head": {
                "mode": "return_to_natural_and_hold_continuously",
                "device_path": discovery.head.serial_by_id_path,
                "response_timeout_ms": head.response_timeout_ms,
                "write_timeout_ms": head.write_timeout_ms,
                "arming_freshness_ms": head.arming_freshness_ms,
                "write_attempts": head.write_attempts,
                "noise_budget_bytes": head.noise_budget_bytes,
                "redundant_read_tolerance_ticks":
                    head.redundant_read_tolerance_ticks,
                "readback_tolerance_ticks": head.readback_tolerance_ticks,
                "final_target_tolerance_ticks": head.final_target_tolerance_ticks,
                "path_corridor_tolerance_ticks":
                    head.path_corridor_tolerance_ticks,
                "direction_regression_tolerance_ticks":
                    head.direction_regression_tolerance_ticks,
                "goal_speed_ticks_per_second": head.goal_speed_ticks_per_second,
                "torque_limit_permille": head.torque_limit_permille,
                "minimum_start_ticks": head.minimum_start_ticks,
                "maximum_start_ticks": head.maximum_start_ticks,
                "reviewed_natural_target_ticks": head.reviewed_natural_target_ticks,
                "maximum_travel_ticks": head.maximum_travel_ticks,
                "physical_torque_consent":
                    "enable_for_reviewed_natural_return_and_hold",
                "physical_motion_consent":
                    "return_to_reviewed_natural_target",
            },
            "rgb_expression": {
                "mode": "scene_motion",
                "sampling_columns": rgb.sampling_columns,
                "sampling_rows": rgb.sampling_rows,
                "minimum_residual_luma": rgb.minimum_residual_luma,
                "minimum_active_fraction_basis_points":
                    rgb.minimum_active_fraction_basis_points,
                "frame_freshness_ms": rgb.frame_freshness_ms,
                "brightness_basis_points": rgb.brightness_basis_points,
                "color_rgb": rgb.color_rgb,
                "blink": rgb.blink,
                "gaze_geometry": {
                    "schema_version": 1,
                    "head_origin_in_camera_m": rgb.head_origin_in_camera_m,
                    "neutral_head_from_camera_quaternion_xyzw":
                        rgb.neutral_head_from_camera_quaternion_xyzw,
                },
            },
            "supervisor": {
                "maximum_authority_lease_ms": 1000,
                "maximum_zero_age_ms": 250,
            },
            "live_mode_policy": live_mode_policy,
        }),
    )
}

fn render_launch(
    input: &RenderInput,
    staged: &[StagedFile],
    production: Option<&ProductionControllerProfile>,
) -> Result<StagedFile, RenderError> {
    let agent_policy = find_staged(staged, "agent-policy-v3.json")?;
    let navigation = find_staged(staged, "navigation-shadow-v1.json")?;
    let calibration = find_staged(
        staged,
        input.assets.calibration.destination_relative_path.as_str(),
    )?;
    let plant = find_staged(
        staged,
        input.assets.plant.destination_relative_path.as_str(),
    )?;
    let onnx_library_spec = input
        .native_libraries
        .iter()
        .find(|library| library.role == NativeLibraryRole::Onnxruntime)
        .ok_or(RenderError::MissingOnnxRuntime)?;
    let onnx = find_staged(staged, onnx_library_spec.destination_relative_path.as_str())?;
    let superpoint = find_staged(
        staged,
        input
            .assets
            .superpoint_model
            .destination_relative_path
            .as_str(),
    )?;
    let lightglue = find_staged(
        staged,
        input
            .assets
            .lightglue_model
            .destination_relative_path
            .as_str(),
    )?;
    let controller_path = match input.bundle {
        BundleSelection::WheelsOffQualification { .. } => "controller-server-candidate-v2.json",
        BundleSelection::Production { .. } => "controller-server-v1.json",
    };
    let controller = find_staged(staged, controller_path)?;
    let runtime = &input.runtime;
    let common = json!({
        "agent_policy_asset": binding(agent_policy),
        "navigation_shadow_config_asset": binding(navigation),
        "controller_server": {
            "contract_asset": binding(controller),
            "command_udp_endpoint": match production {
                Some(profile) => format!("127.0.0.1:{}", profile.controller.command_udp_port),
                None => "127.0.0.1:8080".to_owned(),
            },
        },
        "calibration_artifact": {
            "artifact_id": input.assets.calibration.artifact_id,
            "asset": binding(calibration),
        },
        "plant_artifact": {
            "artifact_id": input.assets.plant.artifact_id,
            "asset": binding(plant),
        },
        "oak": {
            "selector_source": "exact_inventory_oak_mxid",
            "maximum_usb_speed": "SUPER",
            "minimum_usb_speed": "SUPER",
            "rgb": {
                "width_px": runtime.oak.rgb_width_px,
                "height_px": runtime.oak.rgb_height_px,
                "fps": runtime.oak.rgb_fps,
            },
            "rectified_stereo": {
                "width_px": runtime.oak.stereo_width_px,
                "height_px": runtime.oak.stereo_height_px,
                "fps": runtime.oak.stereo_fps,
                "rectified": true,
            },
            "depth": {
                "width_px": runtime.oak.stereo_width_px,
                "height_px": runtime.oak.stereo_height_px,
                "fps": runtime.oak.stereo_fps,
                "alignment": "rectified_left",
            },
            "imu": { "rate_hz": runtime.oak.imu_rate_hz },
            "queue": {
                "size": runtime.oak.queue_size,
                "blocking": false,
            },
        },
        "occupancy": {
            "resolution_m": runtime.occupancy.resolution_m,
            "lower_x_m": runtime.occupancy.lower_x_m,
            "lower_y_m": runtime.occupancy.lower_y_m,
            "width_cells": runtime.occupancy.width_cells,
            "height_cells": runtime.occupancy.height_cells,
            "maximum_cells": runtime.occupancy.maximum_cells,
            "maximum_keyframes": runtime.occupancy.maximum_keyframes,
            "snapshot_every_keyframes":
                runtime.occupancy.snapshot_every_keyframes,
        },
        "inference": {
            "onnx_runtime_library_asset": binding(onnx),
            "superpoint_model_asset": binding(superpoint),
            "lightglue_model_asset": binding(lightglue),
            "superpoint_backend": runtime.inference.superpoint_backend.as_str(),
            "lightglue_backend": runtime.inference.lightglue_backend.as_str(),
            "downscale_factor": runtime.inference.downscale_factor,
            "maximum_keypoints": runtime.inference.maximum_keypoints,
        },
        "rerun": {
            "kind": "serve_loopback",
            "bind": "127.0.0.1:9876",
            "decimation": runtime.rerun.decimation,
            "memory_limit_bytes": runtime.rerun.memory_limit_bytes,
            "flush_timeout_ms": runtime.rerun.flush_timeout_ms,
        },
        "storage": {
            "map_snapshot_relative_path": "maps/current.kmap",
            "navigation_dataset_directory_relative_path": "navigation",
            "maximum_map_snapshot_bytes":
                runtime.storage.maximum_map_snapshot_bytes,
            "minimum_free_bytes_after_map_save":
                runtime.storage.minimum_free_bytes_after_map_save,
            "maximum_navigation_dataset_bytes":
                runtime.storage.navigation_dataset.maximum_bytes,
            "maximum_navigation_dataset_files":
                runtime.storage.navigation_dataset.maximum_files,
            "maximum_navigation_ingress_records":
                runtime.storage.navigation_dataset.maximum_ingress_records,
            "minimum_free_bytes_after_navigation_dataset_write":
                runtime.storage.navigation_dataset.minimum_free_bytes_after_write,
            "navigation_dataset_terminal_reserve_bytes":
                runtime.storage.navigation_dataset.terminal_reserve_bytes,
        },
    });
    let mut object = common
        .as_object()
        .cloned()
        .ok_or(RenderError::InternalJsonShape)?;
    match input.bundle {
        BundleSelection::WheelsOffQualification { .. } => {
            object.insert("schema_version".to_owned(), json!(2));
            object.insert("robot_id".to_owned(), json!(input.robot_id));
            let executable = find_staged(staged, QUALIFICATION_EXECUTABLE_RELATIVE_PATH)?;
            object.insert(
                "qualification_executable_asset".to_owned(),
                binding(executable),
            );
            let native_runtime = find_staged(staged, "native-runtime-v1.json")?;
            object.insert(
                "native_runtime_manifest_asset".to_owned(),
                binding(native_runtime),
            );
            let inventory = find_staged(staged, "device-inventory-candidate-v2.json")?;
            object.insert(
                "candidate_inventory_manifest_asset".to_owned(),
                binding(inventory),
            );
            let policy = find_staged(staged, "candidate-controller-policy-v1.json")?;
            object.insert(
                "candidate_controller_policy_asset".to_owned(),
                binding(policy),
            );
        }
        BundleSelection::Production { .. } => {
            object.insert("schema_version".to_owned(), json!(3));
            let actuation = find_staged(staged, "navigation-actuation-v2.json")?;
            object.insert(
                "physical_actuation_config_asset".to_owned(),
                binding(actuation),
            );
            let face = input
                .bundle
                .face_perception()
                .expect("production bundle carries parsed face assets");
            let frontal = find_staged(
                staged,
                face.frontal_face_cascade.destination_relative_path.as_str(),
            )?;
            let profile = find_staged(
                staged,
                face.profile_face_cascade.destination_relative_path.as_str(),
            )?;
            object.insert(
                "face_perception".to_owned(),
                json!({
                    "frontal_face_cascade_asset": binding(frontal),
                    "profile_face_cascade_asset": binding(profile),
                }),
            );
        }
    }
    StagedFile::rendered_json(
        "launch",
        input.bundle.launch_relative_path(),
        &Value::Object(object),
    )
}

fn binding(file: &StagedFile) -> Value {
    json!({
        "relative_path": file.relative_path,
        "maximum_bytes": file.byte_len,
        "sha256_hex": encode_hex(&file.sha256),
    })
}

fn render_evidence_manifest(
    input: &RenderInput,
    render_input_source: &LoadedSource,
    production: &Option<ProductionAdmission>,
    staged: &[StagedFile],
    launch: &StagedFile,
) -> Result<StagedFile, RenderError> {
    let render_input_evidence_path = input.bundle.render_input_evidence_path();
    let mut sources = vec![
        render_input_source.evidence(Some(render_input_evidence_path)),
        SourceEvidence::from_staged_source(
            "calibration",
            input.assets.calibration.source_path.as_path(),
            input.assets.calibration.destination_relative_path.as_str(),
            staged,
        )?,
        SourceEvidence::from_staged_source(
            "plant",
            input.assets.plant.source_path.as_path(),
            input.assets.plant.destination_relative_path.as_str(),
            staged,
        )?,
        SourceEvidence::from_staged_source(
            "navigation_shadow",
            input.assets.navigation_shadow_source_path.as_path(),
            "navigation-shadow-v1.json",
            staged,
        )?,
        SourceEvidence::from_staged_source(
            "superpoint_model",
            input.assets.superpoint_model.source_path.as_path(),
            input
                .assets
                .superpoint_model
                .destination_relative_path
                .as_str(),
            staged,
        )?,
        SourceEvidence::from_staged_source(
            "lightglue_model",
            input.assets.lightglue_model.source_path.as_path(),
            input
                .assets
                .lightglue_model
                .destination_relative_path
                .as_str(),
            staged,
        )?,
    ];
    if let BundleSelection::WheelsOffQualification {
        qualification_executable_path,
    } = &input.bundle
    {
        sources.push(SourceEvidence::from_staged_source(
            "qualification_executable",
            qualification_executable_path.as_path(),
            QUALIFICATION_EXECUTABLE_RELATIVE_PATH,
            staged,
        )?);
    }
    if let Some(face) = input.bundle.face_perception() {
        sources.push(SourceEvidence::from_staged_source(
            "frontal_face_cascade",
            face.frontal_face_cascade.source_path.as_path(),
            face.frontal_face_cascade.destination_relative_path.as_str(),
            staged,
        )?);
        sources.push(SourceEvidence::from_staged_source(
            "profile_face_cascade",
            face.profile_face_cascade.source_path.as_path(),
            face.profile_face_cascade.destination_relative_path.as_str(),
            staged,
        )?);
    }
    for library in &input.native_libraries {
        sources.push(SourceEvidence::from_staged_source(
            &format!("native_library_{}", library.role.as_str()),
            library.source_path.as_path(),
            library.destination_relative_path.as_str(),
            staged,
        )?);
    }
    if let Some(admission) = production {
        sources.push(
            admission
                .source
                .evidence(Some(PRODUCTION_PROFILE_EVIDENCE_PATH)),
        );
    }
    let mut files = staged.iter().map(StagedFile::evidence).collect::<Vec<_>>();
    files.push(launch.evidence());
    let write_order = staged
        .iter()
        .map(|file| file.relative_path.clone())
        .chain([
            RENDER_EVIDENCE_PATH.to_owned(),
            launch.relative_path.clone(),
        ])
        .collect::<Vec<_>>();
    let evidence = RenderEvidenceManifest {
        schema_version: EVIDENCE_SCHEMA_VERSION,
        evidence_scope: EVIDENCE_SCOPE,
        bundle_kind: input.bundle.kind_name(),
        robot_id: &input.robot_id,
        production_admission: production
            .as_ref()
            .map(|admission| ProductionAdmissionEvidence {
                admission_id: &admission.profile.admission_id,
                reviewer_id: &admission.profile.reviewer_id,
                profile_sha256_hex: encode_hex(&admission.source.sha256),
            }),
        sources,
        files,
        launch_relative_path: input.bundle.launch_relative_path(),
        launch_written_last: true,
        claims_not_established: [
            "installation",
            "filesystem ownership",
            "device presence",
            "hardware qualification",
            "physical stop behavior",
            "performance improvement",
        ],
        write_order_note: "leaf bytes and derived configs are written first; evidence is written next; launch is written last",
        deterministic_write_order: write_order,
    };
    StagedFile::rendered_json("render_evidence", RENDER_EVIDENCE_PATH, &evidence)
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct RenderEvidenceManifest<'a> {
    schema_version: u32,
    evidence_scope: &'static str,
    bundle_kind: &'static str,
    robot_id: &'a str,
    production_admission: Option<ProductionAdmissionEvidence<'a>>,
    sources: Vec<SourceEvidence>,
    files: Vec<BundleFileEvidence>,
    launch_relative_path: &'static str,
    launch_written_last: bool,
    claims_not_established: [&'static str; 6],
    write_order_note: &'static str,
    deterministic_write_order: Vec<String>,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct ProductionAdmissionEvidence<'a> {
    admission_id: &'a str,
    reviewer_id: &'a str,
    profile_sha256_hex: String,
}

#[derive(Serialize)]
#[serde(deny_unknown_fields)]
struct SourceEvidence {
    role: String,
    source_path: PathBuf,
    destination_relative_path: Option<String>,
    byte_len: u64,
    sha256_hex: String,
}

impl SourceEvidence {
    fn from_staged_source(
        role: &str,
        source_path: &Path,
        destination_relative_path: &str,
        staged: &[StagedFile],
    ) -> Result<Self, RenderError> {
        let file = find_staged(staged, destination_relative_path)?;
        Ok(Self {
            role: role.to_owned(),
            source_path: source_path.to_path_buf(),
            destination_relative_path: Some(destination_relative_path.to_owned()),
            byte_len: file.byte_len,
            sha256_hex: encode_hex(&file.sha256),
        })
    }
}

fn find_staged<'a>(
    staged: &'a [StagedFile],
    relative_path: &str,
) -> Result<&'a StagedFile, RenderError> {
    staged
        .iter()
        .find(|file| file.relative_path == relative_path)
        .ok_or_else(|| RenderError::MissingRenderedDependency {
            relative_path: relative_path.to_owned(),
        })
}

fn ensure_unique_relative_paths(staged: &[StagedFile]) -> Result<(), RenderError> {
    let mut seen = BTreeSet::new();
    for file in staged {
        validate_relative_path(&file.relative_path)?;
        if !seen.insert(file.relative_path.as_str()) {
            return Err(RenderError::DuplicateDestination {
                relative_path: file.relative_path.clone(),
            });
        }
    }
    Ok(())
}

fn validate_rendered_tree(staged: &[StagedFile]) -> Result<(), RenderError> {
    for file in staged {
        if file.relative_path.ends_with(".json") {
            reject_unresolved_tokens(&file.relative_path, &file.bytes)?;
        }
    }
    let launch = staged
        .last()
        .ok_or(RenderError::MissingRenderedDependency {
            relative_path: "launch".to_owned(),
        })?;
    if launch.role != "launch" {
        return Err(RenderError::LaunchNotLast);
    }
    Ok(())
}

fn reject_tokens_if_json(source: &LoadedSource) -> Result<(), RenderError> {
    if source
        .path
        .extension()
        .is_some_and(|extension| extension == "json")
    {
        reject_unresolved_tokens(&source.role, &source.bytes)?;
        ensure_json_value(source)?;
    }
    Ok(())
}

fn ensure_json_value(source: &LoadedSource) -> Result<(), RenderError> {
    let _: Value = parse_exact_json(&source.bytes, source.path.as_path())?;
    Ok(())
}

fn parse_exact_json<T>(bytes: &[u8], path: &Path) -> Result<T, RenderError>
where
    T: serde::de::DeserializeOwned,
{
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = T::deserialize(&mut deserializer).map_err(|source| RenderError::ParseJson {
        path: path.to_path_buf(),
        source,
    })?;
    deserializer
        .end()
        .map_err(|source| RenderError::ParseJson {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(value)
}

fn reject_unresolved_tokens(label: &str, bytes: &[u8]) -> Result<(), RenderError> {
    if bytes.windows(2).any(|window| window == b"${") {
        return Err(RenderError::UnresolvedTemplateToken {
            label: label.to_owned(),
        });
    }
    Ok(())
}

fn validate_absolute_path(path: &Path) -> Result<(), RenderError> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        return Err(RenderError::NonCanonicalAbsolutePath {
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn validate_relative_path(path: &str) -> Result<(), RenderError> {
    if path.is_empty()
        || path.contains("${")
        || Path::new(path)
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(RenderError::NonCanonicalRelativePath {
            path: path.to_owned(),
        });
    }
    Ok(())
}

fn reject_symlink_components(path: &Path) -> Result<(), RenderError> {
    let mut current = PathBuf::from("/");
    for component in path.components() {
        match component {
            Component::RootDir => {}
            Component::Normal(part) => {
                current.push(part);
                let metadata =
                    fs::symlink_metadata(&current).map_err(|source| RenderError::ReadMetadata {
                        path: current.clone(),
                        source,
                    })?;
                if metadata.file_type().is_symlink() {
                    return Err(RenderError::SymlinkRejected {
                        path: current.clone(),
                    });
                }
            }
            Component::CurDir | Component::ParentDir | Component::Prefix(_) => {
                return Err(RenderError::NonCanonicalAbsolutePath {
                    path: path.to_path_buf(),
                });
            }
        }
    }
    Ok(())
}

fn write_staging_tree(destination: &Path, staged: &[StagedFile]) -> Result<(), RenderError> {
    validate_absolute_path(destination)?;
    let existed = match fs::symlink_metadata(destination) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                return Err(RenderError::SymlinkRejected {
                    path: destination.to_path_buf(),
                });
            }
            if !metadata.is_dir() {
                return Err(RenderError::DestinationNotDirectory {
                    path: destination.to_path_buf(),
                });
            }
            if fs::read_dir(destination)
                .map_err(|source| RenderError::ReadDestination {
                    path: destination.to_path_buf(),
                    source,
                })?
                .next()
                .transpose()
                .map_err(|source| RenderError::ReadDestination {
                    path: destination.to_path_buf(),
                    source,
                })?
                .is_some()
            {
                return Err(RenderError::DestinationNotEmpty {
                    path: destination.to_path_buf(),
                });
            }
            true
        }
        Err(source) if source.kind() == io::ErrorKind::NotFound => false,
        Err(source) => {
            return Err(RenderError::ReadDestination {
                path: destination.to_path_buf(),
                source,
            });
        }
    };
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| RenderError::DestinationHasNoParent {
            path: destination.to_path_buf(),
        })?;
    reject_symlink_components(parent)?;
    if !existed {
        fs::create_dir(destination).map_err(|source| RenderError::CreateDirectory {
            path: destination.to_path_buf(),
            source,
        })?;
    }
    reject_symlink_components(destination)?;

    let mut directories = BTreeSet::new();
    directories.insert(destination.to_path_buf());
    for file in staged {
        let full_path = destination.join(&file.relative_path);
        let parent = full_path
            .parent()
            .ok_or_else(|| RenderError::DestinationHasNoParent {
                path: full_path.clone(),
            })?;
        create_confined_directories(destination, parent, &mut directories)?;
        reject_symlink_components(parent)?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&full_path)
            .map_err(|source| RenderError::WriteFile {
                path: full_path.clone(),
                source,
            })?;
        output
            .write_all(&file.bytes)
            .and_then(|()| output.sync_all())
            .map_err(|source| RenderError::WriteFile {
                path: full_path.clone(),
                source,
            })?;
        let retained = fs::read(&full_path).map_err(|source| RenderError::ReadSource {
            path: full_path.clone(),
            source,
        })?;
        if retained != file.bytes {
            return Err(RenderError::StagedReadbackMismatch {
                path: full_path.clone(),
            });
        }
        let mut permissions = fs::metadata(&full_path)
            .map_err(|source| RenderError::ReadMetadata {
                path: full_path.clone(),
                source,
            })?
            .permissions();
        if file.executable {
            permissions.set_mode(0o555);
        } else {
            permissions.set_readonly(true);
        }
        fs::set_permissions(&full_path, permissions).map_err(|source| {
            RenderError::SetReadOnly {
                path: full_path,
                source,
            }
        })?;
    }
    verify_staged_file_set(destination, staged)?;
    let mut directories = directories.into_iter().collect::<Vec<_>>();
    directories.sort_by_key(|path| std::cmp::Reverse(path.components().count()));
    for directory in directories {
        let mut permissions = fs::metadata(&directory)
            .map_err(|source| RenderError::ReadMetadata {
                path: directory.clone(),
                source,
            })?
            .permissions();
        permissions.set_readonly(true);
        fs::set_permissions(&directory, permissions).map_err(|source| {
            RenderError::SetReadOnly {
                path: directory,
                source,
            }
        })?;
    }
    Ok(())
}

fn verify_staged_file_set(destination: &Path, staged: &[StagedFile]) -> Result<(), RenderError> {
    let expected = staged
        .iter()
        .map(|file| file.relative_path.as_str())
        .collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    let mut pending = vec![destination.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).map_err(|source| RenderError::ReadDestination {
            path: directory.clone(),
            source,
        })? {
            let entry = entry.map_err(|source| RenderError::ReadDestination {
                path: directory.clone(),
                source,
            })?;
            let path = entry.path();
            let metadata =
                fs::symlink_metadata(&path).map_err(|source| RenderError::ReadMetadata {
                    path: path.clone(),
                    source,
                })?;
            if metadata.file_type().is_symlink() {
                return Err(RenderError::SymlinkRejected { path });
            }
            if metadata.is_dir() {
                pending.push(path);
            } else if metadata.is_file() {
                let relative = path.strip_prefix(destination).map_err(|_| {
                    RenderError::DestinationPathEscape {
                        root: destination.to_path_buf(),
                        path: path.clone(),
                    }
                })?;
                let relative = relative
                    .to_str()
                    .ok_or_else(|| RenderError::DestinationPathNotUtf8 { path: path.clone() })?;
                actual.insert(relative.to_owned());
            } else {
                return Err(RenderError::DestinationUnsupportedFileType { path });
            }
        }
    }
    let actual_refs = actual.iter().map(String::as_str).collect::<BTreeSet<_>>();
    if actual_refs != expected {
        return Err(RenderError::DestinationFileSetMismatch {
            expected: expected.into_iter().map(ToOwned::to_owned).collect(),
            actual,
        });
    }
    Ok(())
}

fn create_confined_directories(
    root: &Path,
    parent: &Path,
    directories: &mut BTreeSet<PathBuf>,
) -> Result<(), RenderError> {
    let relative = parent
        .strip_prefix(root)
        .map_err(|_| RenderError::DestinationPathEscape {
            root: root.to_path_buf(),
            path: parent.to_path_buf(),
        })?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(part) = component else {
            return Err(RenderError::DestinationPathEscape {
                root: root.to_path_buf(),
                path: parent.to_path_buf(),
            });
        };
        current.push(part);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(RenderError::SymlinkRejected { path: current });
            }
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => return Err(RenderError::DestinationNotDirectory { path: current }),
            Err(source) if source.kind() == io::ErrorKind::NotFound => {
                fs::create_dir(&current).map_err(|source| RenderError::CreateDirectory {
                    path: current.clone(),
                    source,
                })?;
            }
            Err(source) => {
                return Err(RenderError::ReadMetadata {
                    path: current,
                    source,
                });
            }
        }
        directories.insert(current.clone());
    }
    Ok(())
}

#[derive(Debug)]
pub enum RenderError {
    Input(InputError),
    MissingProductionControllerProfile,
    ParseJson {
        path: PathBuf,
        source: serde_json::Error,
    },
    SerializeRenderedJson(serde_json::Error),
    GeneratedControllerContract(ServerConfigError),
    NonCanonicalAbsolutePath {
        path: PathBuf,
    },
    NonCanonicalRelativePath {
        path: String,
    },
    SymlinkRejected {
        path: PathBuf,
    },
    ReadMetadata {
        path: PathBuf,
        source: io::Error,
    },
    SourceNotRegular {
        path: PathBuf,
    },
    SourceSizeOutsideBound {
        path: PathBuf,
        byte_len: u64,
        maximum_bytes: u64,
    },
    SourceSizeNotUsize {
        path: PathBuf,
        byte_len: u64,
    },
    ReadSource {
        path: PathBuf,
        source: io::Error,
    },
    SourceChangedDuringRead {
        path: PathBuf,
        metadata_bytes: u64,
        retained_bytes: u64,
    },
    SourceIdentityChangedBeforeRead {
        path: PathBuf,
    },
    UnresolvedTemplateToken {
        label: String,
    },
    RenderedSizeNotU64 {
        relative_path: String,
    },
    DuplicateDestination {
        relative_path: String,
    },
    MissingRenderedDependency {
        relative_path: String,
    },
    DuplicateRoleAfterParsing {
        role: &'static str,
    },
    UnexpectedNativeLibraryRemainder,
    MissingOnnxRuntime,
    InternalJsonShape,
    LaunchNotLast,
    DestinationNotDirectory {
        path: PathBuf,
    },
    DestinationNotEmpty {
        path: PathBuf,
    },
    DestinationHasNoParent {
        path: PathBuf,
    },
    ReadDestination {
        path: PathBuf,
        source: io::Error,
    },
    CreateDirectory {
        path: PathBuf,
        source: io::Error,
    },
    WriteFile {
        path: PathBuf,
        source: io::Error,
    },
    StagedReadbackMismatch {
        path: PathBuf,
    },
    SetReadOnly {
        path: PathBuf,
        source: io::Error,
    },
    DestinationPathEscape {
        root: PathBuf,
        path: PathBuf,
    },
    DestinationPathNotUtf8 {
        path: PathBuf,
    },
    DestinationUnsupportedFileType {
        path: PathBuf,
    },
    DestinationFileSetMismatch {
        expected: BTreeSet<String>,
        actual: BTreeSet<String>,
    },
}

impl fmt::Display for RenderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(source) => write!(formatter, "render input rejected: {source}"),
            Self::MissingProductionControllerProfile => formatter.write_str(
                "production rendering is fail-closed without a separate admitted production-controller profile",
            ),
            Self::ParseJson { path, source } => {
                write!(formatter, "failed to parse exact JSON {}: {source}", path.display())
            }
            Self::SerializeRenderedJson(source) => {
                write!(formatter, "failed to serialize deterministic rendered JSON: {source}")
            }
            Self::GeneratedControllerContract(source) => write!(
                formatter,
                "generated controller contract was rejected by its authoritative parser: {source}"
            ),
            Self::NonCanonicalAbsolutePath { path } => write!(
                formatter,
                "path must be absolute with no dot or parent components: {}",
                path.display()
            ),
            Self::NonCanonicalRelativePath { path } => {
                write!(formatter, "bundle-relative path escapes or is unresolved: {path}")
            }
            Self::SymlinkRejected { path } => {
                write!(formatter, "symlink component rejected: {}", path.display())
            }
            Self::ReadMetadata { path, source } => {
                write!(formatter, "failed to inspect {}: {source}", path.display())
            }
            Self::SourceNotRegular { path } => {
                write!(formatter, "source is not a regular file: {}", path.display())
            }
            Self::SourceSizeOutsideBound {
                path,
                byte_len,
                maximum_bytes,
            } => write!(
                formatter,
                "source {} has {byte_len} bytes; required range is 1..={maximum_bytes}",
                path.display()
            ),
            Self::SourceSizeNotUsize { path, byte_len } => write!(
                formatter,
                "source {} byte count {byte_len} is not addressable on this host",
                path.display()
            ),
            Self::ReadSource { path, source } => {
                write!(formatter, "failed to retain {}: {source}", path.display())
            }
            Self::SourceChangedDuringRead {
                path,
                metadata_bytes,
                retained_bytes,
            } => write!(
                formatter,
                "source {} changed while read: metadata {metadata_bytes}, retained {retained_bytes}",
                path.display()
            ),
            Self::SourceIdentityChangedBeforeRead { path } => write!(
                formatter,
                "source identity changed across its retained read: {}",
                path.display()
            ),
            Self::UnresolvedTemplateToken { label } => {
                write!(formatter, "unresolved ${{...}} token remains in {label}")
            }
            Self::RenderedSizeNotU64 { relative_path } => write!(
                formatter,
                "rendered byte count is not representable for {relative_path}"
            ),
            Self::DuplicateDestination { relative_path } => {
                write!(formatter, "multiple assets target {relative_path}")
            }
            Self::MissingRenderedDependency { relative_path } => {
                write!(formatter, "rendered dependency is missing: {relative_path}")
            }
            Self::DuplicateRoleAfterParsing { role } => {
                write!(formatter, "native role {role} was duplicated after parsing")
            }
            Self::UnexpectedNativeLibraryRemainder => {
                formatter.write_str("unconsumed native libraries remain after deterministic render")
            }
            Self::MissingOnnxRuntime => formatter.write_str("onnxruntime native role is missing"),
            Self::InternalJsonShape => {
                formatter.write_str("internal rendered JSON did not retain its required object shape")
            }
            Self::LaunchNotLast => {
                formatter.write_str("launch document is not the final staging write")
            }
            Self::DestinationNotDirectory { path } => {
                write!(formatter, "destination component is not a directory: {}", path.display())
            }
            Self::DestinationNotEmpty { path } => {
                write!(formatter, "staging destination is not empty: {}", path.display())
            }
            Self::DestinationHasNoParent { path } => {
                write!(formatter, "staging destination has no parent: {}", path.display())
            }
            Self::ReadDestination { path, source } => write!(
                formatter,
                "failed to inspect staging destination {}: {source}",
                path.display()
            ),
            Self::CreateDirectory { path, source } => write!(
                formatter,
                "failed to create staging directory {}: {source}",
                path.display()
            ),
            Self::WriteFile { path, source } => {
                write!(formatter, "failed to write staging file {}: {source}", path.display())
            }
            Self::StagedReadbackMismatch { path } => write!(
                formatter,
                "staged readback changed exact bytes for {}",
                path.display()
            ),
            Self::SetReadOnly { path, source } => write!(
                formatter,
                "failed to make staging path read-only {}: {source}",
                path.display()
            ),
            Self::DestinationPathEscape { root, path } => write!(
                formatter,
                "destination path {} escaped staging root {}",
                path.display(),
                root.display()
            ),
            Self::DestinationPathNotUtf8 { path } => write!(
                formatter,
                "destination path is not UTF-8 and cannot match a bundle path: {}",
                path.display()
            ),
            Self::DestinationUnsupportedFileType { path } => write!(
                formatter,
                "destination contains an unsupported file type: {}",
                path.display()
            ),
            Self::DestinationFileSetMismatch { expected, actual } => write!(
                formatter,
                "staged file set differs from plan; expected {expected:?}, actual {actual:?}"
            ),
        }
    }
}

impl std::error::Error for RenderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Input(source) => Some(source),
            Self::ParseJson { source, .. } | Self::SerializeRenderedJson(source) => Some(source),
            Self::GeneratedControllerContract(source) => Some(source),
            Self::ReadMetadata { source, .. }
            | Self::ReadSource { source, .. }
            | Self::ReadDestination { source, .. }
            | Self::CreateDirectory { source, .. }
            | Self::WriteFile { source, .. }
            | Self::SetReadOnly { source, .. } => Some(source),
            _ => None,
        }
    }
}
