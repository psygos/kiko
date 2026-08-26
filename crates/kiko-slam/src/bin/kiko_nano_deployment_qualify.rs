//! Mint one exact-byte, offline-only Nano install marker.
//!
//! This command performs no device I/O and cannot qualify hardware, wheels,
//! motion, calibration quality, runtime readiness, or physical safety. It is
//! intentionally separate from the small boot verifier so the verifier does
//! not link the camera, inference, controller, or SLAM dependency graph.

#![cfg(unix)]
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::path::{Path, PathBuf};

use clap::Parser;
use kiko_device_inventory::{
    ArtifactKind, ArtifactRelativePath, DeploymentAssetByteLimit, LoadedDeploymentAsset,
    MAX_MANIFEST_JSON_BYTES, hash_manifest_artifacts, load_deployment_asset,
    load_expected_manifest_v1_file,
};
use kiko_expression_core::StreamEpochId;
use kiko_nano_deployment_gate::{
    DEFAULT_QUALIFICATION_MARKER, OfflineInstallQualificationV1, QUALIFICATION_ACKNOWLEDGEMENT,
    QualifiedFileBindingV1, QualifiedFileV1, ROOT_GID, ROOT_UID, verify_qualification_marker,
    write_qualification_marker,
};
use kiko_slam::navigation::{
    HeadGazePolicyV1, NanoAccessoryHealthPeriod, NanoAccessoryWorkerConfig, NanoAgentLaunchV4,
    NanoAgentPolicyConfigV3, NanoCalibrationArtifactV1, NanoFaceCascadeAssetRole,
    NanoHeadGazeAssetRole, NanoLaunchAssetRole, OfflineProductionNavigationGraphV1,
    ProductionNavigationControllerContractV1, load_nano_agent_launch_v4,
};
use robot_server::config::ControllerServerConfigV1;
use serde::Deserialize;

const DEPLOYMENT_ROOT: &str = "/opt/kiko/deployment";
const STATE_ROOT: &str = "/var/lib/kiko-nano-agent";
const LAUNCH_RELATIVE_PATH: &str = "nano-agent-launch-v4.json";
const NATIVE_MANIFEST_RELATIVE_PATH: &str = "native-runtime-v1.json";
const NATIVE_LIBRARY_SEARCH_RELATIVE_PATH: &str = "lib";
const AGENT_BINARY: &str = "/opt/kiko/bin/kiko-slam";
const GATE_BINARY: &str = "/opt/kiko/bin/kiko-nano-deployment-gate";
const BASE_SYSTEMD_UNIT: &str = "/etc/systemd/system/kiko-nano-agent.service";
const QUALIFIED_SYSTEMD_DROP_IN: &str =
    "/etc/systemd/system/kiko-nano-agent.service.d/10-qualified-boot.conf";
const MAX_NATIVE_MANIFEST_BYTES: u64 = 64 * 1_024;
const MAX_NATIVE_LIBRARIES: usize = 16;
const SHA256_HEX_BYTES: usize = 64;

const EXPECTED_BASE_SYSTEMD_UNIT: &[u8] =
    include_bytes!("../../../../deploy/systemd/kiko-nano-agent.service");
const EXPECTED_QUALIFIED_DROP_IN: &[u8] =
    include_bytes!("../../../../deploy/systemd/kiko-nano-agent-qualified-boot.conf");
const EXPECTED_QUALIFIED_DROP_IN_NAME: &str = "10-qualified-boot.conf";
const SYSTEMD_RELEVANT_DROP_IN_NAMES: [&str; 4] = [
    "kiko-nano-agent.service.d",
    "kiko-nano-.service.d",
    "kiko-.service.d",
    "service.d",
];
const SYSTEMD_SYSTEM_UNIT_SEARCH_ROOTS: [&str; 13] = [
    "/etc/systemd/system.control",
    "/run/systemd/system.control",
    "/run/systemd/transient",
    "/run/systemd/generator.early",
    "/etc/systemd/system",
    "/etc/systemd/system.attached",
    "/run/systemd/system",
    "/run/systemd/system.attached",
    "/run/systemd/generator",
    "/usr/local/lib/systemd/system",
    "/usr/lib/systemd/system",
    "/lib/systemd/system",
    "/run/systemd/generator.late",
];

const ASSET_ROLES: [(NanoLaunchAssetRole, &str, bool); 9] = [
    (
        NanoLaunchAssetRole::AgentPolicy,
        "launch_asset:agent_policy",
        true,
    ),
    (
        NanoLaunchAssetRole::NavigationShadowConfig,
        "launch_asset:navigation_shadow_config",
        true,
    ),
    (
        NanoLaunchAssetRole::PhysicalActuationConfig,
        "launch_asset:physical_actuation_config",
        true,
    ),
    (
        NanoLaunchAssetRole::ControllerServerContract,
        "launch_asset:controller_server_contract",
        true,
    ),
    (
        NanoLaunchAssetRole::CalibrationArtifact,
        "launch_asset:calibration_artifact",
        true,
    ),
    (
        NanoLaunchAssetRole::PlantArtifact,
        "launch_asset:plant_artifact",
        true,
    ),
    (
        NanoLaunchAssetRole::OnnxRuntimeLibrary,
        "launch_asset:onnx_runtime_library",
        false,
    ),
    (
        NanoLaunchAssetRole::SuperpointModel,
        "launch_asset:superpoint_model",
        false,
    ),
    (
        NanoLaunchAssetRole::LightglueModel,
        "launch_asset:lightglue_model",
        false,
    ),
];

#[derive(Debug, Parser)]
#[command(
    name = "kiko-nano-deployment-qualify",
    about = "Create an exact-byte marker for one offline install; never qualifies hardware"
)]
struct Cli {
    /// Exact human acknowledgement; prevents accidental use as a hardware gate.
    #[arg(long)]
    acknowledge: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativeLibraryRole {
    DepthAiCore,
    DynamicCalibration,
    LibUsb1,
    OnnxRuntime,
    OpenCvCore,
    OpenCvImgproc,
    OpenCvObjdetect,
    RuntimeDependency,
}

impl NativeLibraryRole {
    fn parse(value: &str) -> Result<Self, QualifyError> {
        match value {
            "depthai_core" => Ok(Self::DepthAiCore),
            "dynamic_calibration" => Ok(Self::DynamicCalibration),
            "libusb_1_0" => Ok(Self::LibUsb1),
            "onnxruntime" => Ok(Self::OnnxRuntime),
            "opencv_core" => Ok(Self::OpenCvCore),
            "opencv_imgproc" => Ok(Self::OpenCvImgproc),
            "opencv_objdetect" => Ok(Self::OpenCvObjdetect),
            "runtime_dependency" => Ok(Self::RuntimeDependency),
            _ => Err(QualifyError::InvalidNativeRole {
                actual: value.to_owned(),
            }),
        }
    }

    const fn marker_name(self) -> &'static str {
        match self {
            Self::DepthAiCore => "depthai_core",
            Self::DynamicCalibration => "dynamic_calibration",
            Self::LibUsb1 => "libusb_1_0",
            Self::OnnxRuntime => "onnxruntime",
            Self::OpenCvCore => "opencv_core",
            Self::OpenCvImgproc => "opencv_imgproc",
            Self::OpenCvObjdetect => "opencv_objdetect",
            Self::RuntimeDependency => "runtime_dependency",
        }
    }

    const fn exact_nano_soname(self) -> Option<&'static str> {
        match self {
            Self::OpenCvCore => Some("libopencv_core.so.4.5d"),
            Self::OpenCvImgproc => Some("libopencv_imgproc.so.4.5d"),
            Self::OpenCvObjdetect => Some("libopencv_objdetect.so.4.5d"),
            Self::DepthAiCore
            | Self::DynamicCalibration
            | Self::LibUsb1
            | Self::OnnxRuntime
            | Self::RuntimeDependency => None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeRuntimeManifestV1Dto {
    schema_version: u32,
    library_search_relative_path: String,
    libraries: Vec<NativeLibraryDto>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeLibraryDto {
    role: String,
    soname: String,
    relative_path: String,
    maximum_bytes: u64,
    sha256_hex: String,
}

struct ParsedNativeLibrary {
    role: NativeLibraryRole,
    soname: String,
    relative_path: ArtifactRelativePath,
    byte_limit: DeploymentAssetByteLimit,
    sha256: [u8; 32],
}

struct ParsedNativeRuntimeManifest {
    library_search_relative_path: ArtifactRelativePath,
    libraries: Vec<ParsedNativeLibrary>,
}

fn main() {
    if let Err(source) = run(Cli::parse()) {
        eprintln!("{source}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), QualifyError> {
    if rustix::process::geteuid().as_raw() != ROOT_UID {
        return Err(QualifyError::RootRequired);
    }
    if cli.acknowledge != QUALIFICATION_ACKNOWLEDGEMENT {
        return Err(QualifyError::AcknowledgementMismatch);
    }

    let deployment_root = PathBuf::from(DEPLOYMENT_ROOT);
    let state_root = PathBuf::from(STATE_ROOT);
    let launch_relative_path = ArtifactRelativePath::parse(LAUNCH_RELATIVE_PATH.to_owned())
        .map_err(|source| QualifyError::context("parse fixed launch relative path", source))?;
    let loaded_launch = load_nano_agent_launch_v4(&deployment_root, launch_relative_path)
        .map_err(|source| QualifyError::context("load typed Nano launch graph", source))?;
    reject_template_sentinel("launch document", loaded_launch.source().bytes())?;

    let mut loaded_assets = Vec::with_capacity(ASSET_ROLES.len());
    for (role, marker_role, textual) in ASSET_ROLES {
        let loaded = loaded_launch
            .launch()
            .asset(role)
            .load_exact(&deployment_root)
            .map_err(|source| QualifyError::context(format!("load exact {marker_role}"), source))?;
        if textual {
            reject_template_sentinel(marker_role, loaded.bytes())?;
        }
        loaded_assets.push((role, marker_role, loaded));
    }
    let mut loaded_face_assets = Vec::with_capacity(NanoFaceCascadeAssetRole::ALL.len());
    for (role, marker_role) in [
        (
            NanoFaceCascadeAssetRole::FrontalFace,
            "launch_asset:frontal_face_cascade",
        ),
        (
            NanoFaceCascadeAssetRole::ProfileFace,
            "launch_asset:profile_face_cascade",
        ),
    ] {
        let loaded = loaded_launch
            .launch()
            .face_perception()
            .asset(role)
            .load_exact(&deployment_root)
            .map_err(|source| QualifyError::context(format!("load exact {marker_role}"), source))?;
        loaded_face_assets.push((marker_role, loaded));
    }
    let load_head_gaze_asset = |role, marker_role: &'static str| {
        loaded_launch
            .launch()
            .physical_head_gaze()
            .asset(role)
            .load_exact(&deployment_root)
            .map_err(|source| QualifyError::context(format!("load exact {marker_role}"), source))
    };
    let head_gaze_policy_asset = load_head_gaze_asset(
        NanoHeadGazeAssetRole::Policy,
        "launch_asset:head_gaze_policy",
    )?;
    reject_template_sentinel("head-gaze policy", head_gaze_policy_asset.bytes())?;
    let head_gaze_review_evidence = load_head_gaze_asset(
        NanoHeadGazeAssetRole::PhysicalReviewEvidence,
        "launch_asset:head_gaze_physical_review_evidence",
    )?;
    reject_template_sentinel(
        "head-gaze physical review evidence",
        head_gaze_review_evidence.bytes(),
    )?;

    let agent_policy_asset = loaded_asset(&loaded_assets, NanoLaunchAssetRole::AgentPolicy);
    let policy = NanoAgentPolicyConfigV3::parse_json(agent_policy_asset.bytes())
        .map_err(|source| QualifyError::context("parse typed Nano agent policy", source))?;
    if !policy.eye_enabled() {
        return Err(QualifyError::Kep2EyeRequired);
    }
    if !policy.head_enabled() {
        return Err(QualifyError::ContinuousNaturalHeadHoldRequired);
    }
    require_policy_path_beneath(
        "inventory manifest",
        &deployment_root,
        policy.inventory().manifest_path().as_path(),
        false,
    )?;
    require_policy_path_beneath(
        "artifact root",
        &deployment_root,
        policy.inventory().artifact_root_path().as_path(),
        true,
    )?;

    let loaded_manifest =
        load_expected_manifest_v1_file(policy.inventory().manifest_path().as_path())
            .map_err(|source| QualifyError::context("load typed inventory manifest", source))?;
    let retained_manifest = load_absolute_deployment_asset(
        &deployment_root,
        policy.inventory().manifest_path().as_path(),
        u64::try_from(MAX_MANIFEST_JSON_BYTES).expect("manifest limit fits u64"),
    )?;
    reject_template_sentinel("inventory manifest", retained_manifest.bytes())?;
    if retained_manifest.byte_len() != loaded_manifest.json_bytes()
        || retained_manifest.content_sha256().as_bytes()
            != loaded_manifest.content_sha256().as_bytes()
    {
        return Err(QualifyError::ManifestReadMismatch);
    }
    let bound_policy = policy
        .clone()
        .bind_accessories_to_manifest(loaded_manifest.manifest())
        .map_err(|source| {
            QualifyError::context("bind required head and eye policy to manifest", source)
        })?;
    let head_gaze_policy = HeadGazePolicyV1::parse_json(head_gaze_policy_asset.bytes())
        .map_err(|source| QualifyError::context("parse typed physical head-gaze policy", source))?;
    let accessory_health_period =
        NanoAccessoryHealthPeriod::try_from_duration(std::time::Duration::from_secs(1))
            .expect("one second is inside the fixed Nano health bound");
    NanoAccessoryWorkerConfig::from_manifest_bound_policy(
        &bound_policy,
        StreamEpochId::try_new(1).expect("fixed nonzero qualification epoch"),
        accessory_health_period,
    )
    .map_err(|source| QualifyError::context("construct manifest-bound accessory policy", source))?
    .with_evidence_bound_physical_head_gaze(head_gaze_policy, &head_gaze_review_evidence)
    .map_err(|source| {
        QualifyError::context("bind physical head gaze to exact review evidence", source)
    })?;

    let artifact_hashes = hash_manifest_artifacts(
        loaded_manifest.manifest(),
        policy.inventory().artifact_root_path().as_path(),
        policy.inventory().artifact_bindings().clone(),
    )
    .map_err(|source| QualifyError::context("hash every manifest artifact", source))?;
    if !artifact_hashes.all_content_matches_manifest() {
        return Err(QualifyError::ManifestArtifactMismatch);
    }

    let calibration_asset = loaded_asset(&loaded_assets, NanoLaunchAssetRole::CalibrationArtifact);
    let calibration = NanoCalibrationArtifactV1::parse_json(calibration_asset.bytes())
        .map_err(|source| QualifyError::context("parse typed calibration artifact", source))?;
    calibration
        .require_manifest_oak_mxid(loaded_manifest.manifest().oak().mxid().as_str())
        .map_err(|source| {
            QualifyError::context("bind calibration artifact to manifest OAK MXID", source)
        })?;
    let launch_stereo = loaded_launch.launch().oak().rectified_stereo();
    calibration
        .require_launch_stereo_dimensions(launch_stereo.width_px(), launch_stereo.height_px())
        .map_err(|source| {
            QualifyError::context(
                "bind calibration artifact to launch rectified-stereo dimensions",
                source,
            )
        })?;
    bind_calibration_to_manifest(
        &deployment_root,
        loaded_launch.launch(),
        calibration_asset,
        &policy,
        loaded_manifest.manifest(),
        &artifact_hashes,
    )?;

    let controller_asset = loaded_asset(
        &loaded_assets,
        NanoLaunchAssetRole::ControllerServerContract,
    );
    let controller = ControllerServerConfigV1::parse_json(controller_asset.bytes())
        .map_err(|source| QualifyError::context("parse typed controller contract", source))?;
    bind_controller_to_manifest(
        loaded_launch.launch(),
        &controller,
        loaded_manifest.manifest(),
    )?;
    let plant_asset = loaded_asset(&loaded_assets, NanoLaunchAssetRole::PlantArtifact);
    bind_plant_to_manifest(
        &deployment_root,
        loaded_launch.launch(),
        plant_asset,
        &policy,
        loaded_manifest.manifest(),
        &artifact_hashes,
    )?;
    OfflineProductionNavigationGraphV1::parse(
        &calibration,
        plant_asset.bytes(),
        loaded_asset(&loaded_assets, NanoLaunchAssetRole::NavigationShadowConfig).bytes(),
        loaded_asset(&loaded_assets, NanoLaunchAssetRole::PhysicalActuationConfig).bytes(),
        loaded_manifest.manifest().robot_id().as_str(),
        ProductionNavigationControllerContractV1::new(
            loaded_launch
                .launch()
                .controller_server()
                .command_udp_endpoint(),
            &controller,
        ),
    )
    .map_err(|source| {
        QualifyError::context(
            "parse and cross-bind the offline production navigation graph",
            source,
        )
    })?;

    let native_manifest_path =
        ArtifactRelativePath::parse(NATIVE_MANIFEST_RELATIVE_PATH.to_owned())
            .map_err(|source| QualifyError::context("parse fixed native manifest path", source))?;
    let native_manifest_asset = load_deployment_asset(
        &deployment_root,
        native_manifest_path,
        DeploymentAssetByteLimit::try_new(MAX_NATIVE_MANIFEST_BYTES)
            .expect("native manifest limit is valid"),
    )
    .map_err(|source| QualifyError::context("load native runtime manifest", source))?;
    reject_template_sentinel("native runtime manifest", native_manifest_asset.bytes())?;
    let native_manifest = parse_native_manifest(native_manifest_asset.bytes())?;
    bind_onnx_runtime_to_launch(loaded_launch.launch(), &native_manifest)?;
    require_exact_native_library_entries(&deployment_root, &native_manifest)?;

    let mut files = BTreeMap::<PathBuf, QualifiedFileV1>::new();
    let mut bindings = Vec::<QualifiedFileBindingV1>::new();
    let mut roles = BTreeSet::<String>::new();

    add_inspected_file(
        &mut files,
        &mut bindings,
        &mut roles,
        "agent_binary",
        PathBuf::from(AGENT_BINARY),
        Some(0o755),
    )?;
    add_inspected_file(
        &mut files,
        &mut bindings,
        &mut roles,
        "deployment_gate_binary",
        PathBuf::from(GATE_BINARY),
        Some(0o755),
    )?;

    let base_unit = add_inspected_file(
        &mut files,
        &mut bindings,
        &mut roles,
        "systemd_base_unit",
        PathBuf::from(BASE_SYSTEMD_UNIT),
        Some(0o644),
    )?;
    require_exact_file_bytes(
        "base systemd unit",
        Path::new(BASE_SYSTEMD_UNIT),
        EXPECTED_BASE_SYSTEMD_UNIT,
        &base_unit,
    )?;
    require_exact_systemd_drop_in_topology()?;
    let drop_in = add_inspected_file(
        &mut files,
        &mut bindings,
        &mut roles,
        "systemd_qualified_boot_drop_in",
        PathBuf::from(QUALIFIED_SYSTEMD_DROP_IN),
        Some(0o644),
    )?;
    require_exact_file_bytes(
        "qualified systemd drop-in",
        Path::new(QUALIFIED_SYSTEMD_DROP_IN),
        EXPECTED_QUALIFIED_DROP_IN,
        &drop_in,
    )?;
    add_retained_file(
        &mut files,
        &mut bindings,
        &mut roles,
        "launch_document",
        &deployment_root,
        loaded_launch.source(),
    )?;
    for (_, marker_role, asset) in &loaded_assets {
        add_retained_file(
            &mut files,
            &mut bindings,
            &mut roles,
            marker_role,
            &deployment_root,
            asset,
        )?;
    }
    for (marker_role, asset) in &loaded_face_assets {
        add_retained_file(
            &mut files,
            &mut bindings,
            &mut roles,
            marker_role,
            &deployment_root,
            asset,
        )?;
    }
    for (marker_role, asset) in [
        ("launch_asset:head_gaze_policy", &head_gaze_policy_asset),
        (
            "launch_asset:head_gaze_physical_review_evidence",
            &head_gaze_review_evidence,
        ),
    ] {
        add_retained_file(
            &mut files,
            &mut bindings,
            &mut roles,
            marker_role,
            &deployment_root,
            asset,
        )?;
    }
    add_retained_file(
        &mut files,
        &mut bindings,
        &mut roles,
        "inventory_manifest",
        &deployment_root,
        &retained_manifest,
    )?;

    for (index, artifact) in artifact_hashes.iter().enumerate() {
        let path = artifact_hashes
            .artifact_root_path()
            .join(artifact.relative_path().as_path());
        let qualified =
            QualifiedFileV1::inspect(path.clone(), ROOT_UID, ROOT_GID).map_err(|source| {
                QualifyError::context(
                    format!(
                        "inspect manifest artifact {}",
                        artifact.artifact_id().as_str()
                    ),
                    source,
                )
            })?;
        if qualified.byte_len() != artifact.bytes_hashed()
            || qualified.sha256() != artifact.observed_sha256()
        {
            return Err(QualifyError::ManifestArtifactRereadMismatch {
                artifact_id: artifact.artifact_id().as_str().to_owned(),
            });
        }
        add_qualified_file(
            &mut files,
            &mut bindings,
            &mut roles,
            &format!("manifest_artifact:{index:02}"),
            qualified,
        )?;
    }

    add_retained_file(
        &mut files,
        &mut bindings,
        &mut roles,
        "native_runtime_manifest",
        &deployment_root,
        &native_manifest_asset,
    )?;
    for (index, native) in native_manifest.libraries.iter().enumerate() {
        let loaded = load_deployment_asset(
            &deployment_root,
            native.relative_path.clone(),
            native.byte_limit,
        )
        .map_err(|source| {
            QualifyError::context(
                format!("load native runtime library {}", native.soname),
                source,
            )
        })?;
        if loaded.content_sha256().as_bytes() != &native.sha256 {
            return Err(QualifyError::NativeLibraryDigestMismatch {
                soname: native.soname.clone(),
                expected: native.sha256,
                observed: *loaded.content_sha256().as_bytes(),
            });
        }
        add_retained_file(
            &mut files,
            &mut bindings,
            &mut roles,
            &format!(
                "native_library:{index:02}:{}:{}",
                native.role.marker_name(),
                native.soname
            ),
            &deployment_root,
            &loaded,
        )?;
    }

    let marker = OfflineInstallQualificationV1::try_new(
        deployment_root,
        state_root,
        LAUNCH_RELATIVE_PATH.to_owned(),
        NATIVE_MANIFEST_RELATIVE_PATH.to_owned(),
        native_manifest
            .library_search_relative_path
            .as_str()
            .to_owned(),
        files.into_values().collect(),
        bindings,
    )
    .map_err(|source| QualifyError::context("construct offline qualification marker", source))?;
    let marker_path = Path::new(DEFAULT_QUALIFICATION_MARKER);
    write_qualification_marker(marker_path, &marker, ROOT_UID, ROOT_GID)
        .map_err(|source| QualifyError::context("publish offline qualification marker", source))?;
    let verified = verify_qualification_marker(marker_path, ROOT_UID, ROOT_GID)
        .map_err(|source| QualifyError::context("verify newly published marker", source))?;
    if verified != marker {
        return Err(QualifyError::PublishedMarkerMismatch);
    }

    eprintln!(
        "qualified exact offline install bytes; no hardware, wheels-off, motion, or runtime qualification is implied"
    );
    Ok(())
}

fn loaded_asset<'assets>(
    assets: &'assets [(NanoLaunchAssetRole, &str, LoadedDeploymentAsset)],
    requested: NanoLaunchAssetRole,
) -> &'assets LoadedDeploymentAsset {
    &assets
        .iter()
        .find(|(role, _, _)| *role == requested)
        .expect("complete fixed launch role set")
        .2
}

fn load_absolute_deployment_asset(
    deployment_root: &Path,
    absolute_path: &Path,
    maximum_bytes: u64,
) -> Result<LoadedDeploymentAsset, QualifyError> {
    let relative = absolute_path.strip_prefix(deployment_root).map_err(|_| {
        QualifyError::PolicyPathOutsideDeployment {
            field: "absolute deployment asset",
            path: absolute_path.to_path_buf(),
        }
    })?;
    let relative = relative.to_str().ok_or_else(|| QualifyError::NonUtf8Path {
        path: relative.to_path_buf(),
    })?;
    let relative = ArtifactRelativePath::parse(relative.to_owned())
        .map_err(|source| QualifyError::context("parse deployment-relative asset path", source))?;
    let byte_limit = DeploymentAssetByteLimit::try_new(maximum_bytes)
        .map_err(|source| QualifyError::context("parse deployment asset byte bound", source))?;
    load_deployment_asset(deployment_root, relative, byte_limit)
        .map_err(|source| QualifyError::context("retain exact deployment asset", source))
}

fn require_policy_path_beneath(
    field: &'static str,
    deployment_root: &Path,
    path: &Path,
    may_equal_root: bool,
) -> Result<(), QualifyError> {
    let relative = path.strip_prefix(deployment_root).map_err(|_| {
        QualifyError::PolicyPathOutsideDeployment {
            field,
            path: path.to_path_buf(),
        }
    })?;
    if relative.as_os_str().is_empty() && !may_equal_root {
        return Err(QualifyError::PolicyPathAliasesDeploymentRoot { field });
    }
    Ok(())
}

fn bind_controller_to_manifest(
    launch: &NanoAgentLaunchV4,
    controller: &ControllerServerConfigV1,
    manifest: &kiko_device_inventory::DeviceInventoryManifestV1,
) -> Result<(), QualifyError> {
    let expected = manifest.stm32();
    if controller.serial_device() != Path::new(expected.serial_path().as_str()) {
        return Err(QualifyError::ControllerManifestMismatch {
            field: "serial_device",
        });
    }
    if controller.controller_uid() != *expected.controller_uid() {
        return Err(QualifyError::ControllerManifestMismatch {
            field: "controller_uid",
        });
    }
    if controller.firmware_abi().get() != expected.firmware_abi() {
        return Err(QualifyError::ControllerManifestMismatch {
            field: "firmware_abi",
        });
    }
    if controller.firmware_build_id().get() != expected.firmware_build_id() {
        return Err(QualifyError::ControllerManifestMismatch {
            field: "firmware_build_id",
        });
    }
    if controller.actuator_config_fingerprint() != *expected.hardware_profile() {
        return Err(QualifyError::ControllerManifestMismatch {
            field: "actuator_config_fingerprint",
        });
    }
    let launch_endpoint = format!(
        "udp://{}",
        launch.controller_server().command_udp_endpoint()
    );
    if expected.control_endpoint().as_str() != launch_endpoint {
        return Err(QualifyError::ControllerManifestMismatch {
            field: "command_udp_endpoint",
        });
    }
    Ok(())
}

fn bind_plant_to_manifest(
    deployment_root: &Path,
    launch: &NanoAgentLaunchV4,
    launch_plant: &LoadedDeploymentAsset,
    policy: &NanoAgentPolicyConfigV3,
    manifest: &kiko_device_inventory::DeviceInventoryManifestV1,
    hashes: &kiko_device_inventory::ManifestArtifactHashes,
) -> Result<(), QualifyError> {
    let requested = launch.plant_artifact().artifact_id().as_str();
    let expected = manifest
        .artifacts()
        .iter()
        .find(|artifact| {
            artifact.kind() == ArtifactKind::Plant && artifact.artifact_id().as_str() == requested
        })
        .ok_or_else(|| QualifyError::PlantBindingMismatch {
            field: "manifest membership",
            artifact_id: requested.to_owned(),
        })?;
    if expected.sha256().as_bytes() != launch_plant.content_sha256().as_bytes() {
        return Err(QualifyError::PlantBindingMismatch {
            field: "launch/manifest digest",
            artifact_id: requested.to_owned(),
        });
    }
    let binding = policy
        .inventory()
        .artifact_bindings()
        .iter()
        .find(|binding| {
            binding.kind() == ArtifactKind::Plant && binding.artifact_id().as_str() == requested
        })
        .ok_or_else(|| QualifyError::PlantBindingMismatch {
            field: "policy binding",
            artifact_id: requested.to_owned(),
        })?;
    let hashed = hashes
        .iter()
        .find(|artifact| {
            artifact.kind() == ArtifactKind::Plant && artifact.artifact_id().as_str() == requested
        })
        .ok_or_else(|| QualifyError::PlantBindingMismatch {
            field: "observed hash",
            artifact_id: requested.to_owned(),
        })?;
    if hashed.observed_sha256() != launch_plant.content_sha256().as_bytes() {
        return Err(QualifyError::PlantBindingMismatch {
            field: "launch/observed digest",
            artifact_id: requested.to_owned(),
        });
    }
    let artifact_root_relative = policy
        .inventory()
        .artifact_root_path()
        .as_path()
        .strip_prefix(deployment_root)
        .map_err(|_| QualifyError::PolicyPathOutsideDeployment {
            field: "artifact root",
            path: policy
                .inventory()
                .artifact_root_path()
                .as_path()
                .to_path_buf(),
        })?;
    let deployed = artifact_root_relative.join(binding.relative_path().as_path());
    if deployed != launch.plant_artifact().asset().relative_path().as_path() {
        return Err(QualifyError::PlantBindingMismatch {
            field: "deployment-relative path",
            artifact_id: requested.to_owned(),
        });
    }
    Ok(())
}

fn bind_calibration_to_manifest(
    deployment_root: &Path,
    launch: &NanoAgentLaunchV4,
    launch_calibration: &LoadedDeploymentAsset,
    policy: &NanoAgentPolicyConfigV3,
    manifest: &kiko_device_inventory::DeviceInventoryManifestV1,
    hashes: &kiko_device_inventory::ManifestArtifactHashes,
) -> Result<(), QualifyError> {
    let requested = launch.calibration_artifact().artifact_id().as_str();
    let expected = manifest
        .artifacts()
        .iter()
        .find(|artifact| {
            artifact.kind() == ArtifactKind::Calibration
                && artifact.artifact_id().as_str() == requested
        })
        .ok_or_else(|| QualifyError::CalibrationBindingMismatch {
            field: "manifest membership",
            artifact_id: requested.to_owned(),
        })?;
    if expected.sha256().as_bytes() != launch_calibration.content_sha256().as_bytes() {
        return Err(QualifyError::CalibrationBindingMismatch {
            field: "launch/manifest digest",
            artifact_id: requested.to_owned(),
        });
    }
    let binding = policy
        .inventory()
        .artifact_bindings()
        .iter()
        .find(|binding| {
            binding.kind() == ArtifactKind::Calibration
                && binding.artifact_id().as_str() == requested
        })
        .ok_or_else(|| QualifyError::CalibrationBindingMismatch {
            field: "policy binding",
            artifact_id: requested.to_owned(),
        })?;
    let hashed = hashes
        .iter()
        .find(|artifact| {
            artifact.kind() == ArtifactKind::Calibration
                && artifact.artifact_id().as_str() == requested
        })
        .ok_or_else(|| QualifyError::CalibrationBindingMismatch {
            field: "observed hash",
            artifact_id: requested.to_owned(),
        })?;
    if hashed.observed_sha256() != launch_calibration.content_sha256().as_bytes() {
        return Err(QualifyError::CalibrationBindingMismatch {
            field: "launch/observed digest",
            artifact_id: requested.to_owned(),
        });
    }
    let artifact_root_relative = policy
        .inventory()
        .artifact_root_path()
        .as_path()
        .strip_prefix(deployment_root)
        .map_err(|_| QualifyError::PolicyPathOutsideDeployment {
            field: "artifact root",
            path: policy
                .inventory()
                .artifact_root_path()
                .as_path()
                .to_path_buf(),
        })?;
    let deployed = artifact_root_relative.join(binding.relative_path().as_path());
    if deployed
        != launch
            .calibration_artifact()
            .asset()
            .relative_path()
            .as_path()
    {
        return Err(QualifyError::CalibrationBindingMismatch {
            field: "deployment-relative path",
            artifact_id: requested.to_owned(),
        });
    }
    Ok(())
}

fn parse_native_manifest(bytes: &[u8]) -> Result<ParsedNativeRuntimeManifest, QualifyError> {
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > MAX_NATIVE_MANIFEST_BYTES {
        return Err(QualifyError::NativeManifestTooLarge {
            actual_bytes: bytes.len(),
        });
    }
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let dto = NativeRuntimeManifestV1Dto::deserialize(&mut deserializer)
        .map_err(|source| QualifyError::context("decode native runtime manifest", source))?;
    deserializer
        .end()
        .map_err(|source| QualifyError::context("reject native manifest trailing data", source))?;
    if dto.schema_version != 1 {
        return Err(QualifyError::UnsupportedNativeSchema {
            actual: dto.schema_version,
        });
    }
    let search = ArtifactRelativePath::parse(dto.library_search_relative_path)
        .map_err(|source| QualifyError::context("parse native library search path", source))?;
    if search.as_str() != NATIVE_LIBRARY_SEARCH_RELATIVE_PATH {
        return Err(QualifyError::WrongNativeLibrarySearchPath {
            actual: search.as_str().to_owned(),
        });
    }
    if dto.libraries.is_empty() || dto.libraries.len() > MAX_NATIVE_LIBRARIES {
        return Err(QualifyError::InvalidNativeLibraryCount {
            actual: dto.libraries.len(),
        });
    }

    let mut parsed = Vec::with_capacity(dto.libraries.len());
    let mut identities = BTreeSet::new();
    let mut required_roles = BTreeSet::new();
    for library in dto.libraries {
        let role = NativeLibraryRole::parse(&library.role)?;
        if role != NativeLibraryRole::RuntimeDependency && !required_roles.insert(library.role) {
            return Err(QualifyError::DuplicateNativeRole {
                role: role.marker_name(),
            });
        }
        validate_soname(&library.soname)?;
        if let Some(expected) = role.exact_nano_soname()
            && library.soname != expected
        {
            return Err(QualifyError::WrongNativeSonameForRole {
                role: role.marker_name(),
                expected,
                actual: library.soname,
            });
        }
        let relative_path = ArtifactRelativePath::parse(library.relative_path)
            .map_err(|source| QualifyError::context("parse native library path", source))?;
        let expected_parent = Path::new(NATIVE_LIBRARY_SEARCH_RELATIVE_PATH);
        if relative_path.as_path().parent() != Some(expected_parent)
            || relative_path.as_path().file_name() != Some(library.soname.as_ref())
        {
            return Err(QualifyError::NativeLibraryOutsideSearchDirectory {
                soname: library.soname,
                path: relative_path.as_str().to_owned(),
            });
        }
        let byte_limit = DeploymentAssetByteLimit::try_new(library.maximum_bytes)
            .map_err(|source| QualifyError::context("parse native library byte bound", source))?;
        let sha256 = parse_sha256(&library.sha256_hex)?;
        if !identities.insert((relative_path.clone(), sha256)) {
            return Err(QualifyError::DuplicateNativeLibrary {
                path: relative_path.as_str().to_owned(),
            });
        }
        parsed.push(ParsedNativeLibrary {
            role,
            soname: library.soname,
            relative_path,
            byte_limit,
            sha256,
        });
    }
    for role in [
        NativeLibraryRole::DepthAiCore,
        NativeLibraryRole::DynamicCalibration,
        NativeLibraryRole::LibUsb1,
        NativeLibraryRole::OnnxRuntime,
        NativeLibraryRole::OpenCvCore,
        NativeLibraryRole::OpenCvImgproc,
        NativeLibraryRole::OpenCvObjdetect,
    ] {
        if !parsed.iter().any(|library| library.role == role) {
            return Err(QualifyError::MissingNativeRole {
                role: role.marker_name(),
            });
        }
    }
    Ok(ParsedNativeRuntimeManifest {
        library_search_relative_path: search,
        libraries: parsed,
    })
}

fn validate_soname(soname: &str) -> Result<(), QualifyError> {
    if soname.is_empty()
        || soname.len() > 128
        || !soname
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'-' | b'_' | b'+'))
        || !soname.starts_with("lib")
        || !soname.contains(".so")
    {
        return Err(QualifyError::InvalidSoname {
            actual: soname.to_owned(),
        });
    }
    Ok(())
}

fn parse_sha256(value: &str) -> Result<[u8; 32], QualifyError> {
    if value.len() != SHA256_HEX_BYTES
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
    {
        return Err(QualifyError::InvalidSha256 {
            actual: value.to_owned(),
        });
    }
    let mut output = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        output[index] = (hex_nibble(pair[0]) << 4) | hex_nibble(pair[1]);
    }
    Ok(output)
}

fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => 10 + byte - b'a',
        _ => unreachable!("parse_sha256 proved lowercase hexadecimal"),
    }
}

fn bind_onnx_runtime_to_launch(
    launch: &NanoAgentLaunchV4,
    native: &ParsedNativeRuntimeManifest,
) -> Result<(), QualifyError> {
    let runtime = native
        .libraries
        .iter()
        .find(|library| library.role == NativeLibraryRole::OnnxRuntime)
        .expect("native parser requires ONNX Runtime");
    let launch_runtime = launch.inference().onnx_runtime_library();
    if runtime.relative_path != *launch_runtime.relative_path()
        || runtime.sha256 != *launch_runtime.expected_sha256()
    {
        return Err(QualifyError::OnnxRuntimeLaunchMismatch);
    }
    Ok(())
}

fn reject_template_sentinel(label: &str, bytes: &[u8]) -> Result<(), QualifyError> {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return Err(QualifyError::TextAssetNotUtf8 {
            label: label.to_owned(),
        });
    };
    let folded = text.to_ascii_lowercase();
    const SENTINELS: [&str; 8] = [
        "${",
        "replace",
        "placeholder",
        "template_only",
        "deaddead",
        "3735928559",
        "dededededededededededede",
        "adadadadadadadadadadadad",
    ];
    if let Some(sentinel) = SENTINELS
        .iter()
        .find(|sentinel| folded.contains(**sentinel))
    {
        return Err(QualifyError::TemplateSentinel {
            label: label.to_owned(),
            sentinel,
        });
    }
    Ok(())
}

fn add_inspected_file(
    files: &mut BTreeMap<PathBuf, QualifiedFileV1>,
    bindings: &mut Vec<QualifiedFileBindingV1>,
    roles: &mut BTreeSet<String>,
    role: &str,
    path: PathBuf,
    required_mode: Option<u32>,
) -> Result<QualifiedFileV1, QualifyError> {
    let qualified =
        QualifiedFileV1::inspect_with_required_mode(path, ROOT_UID, ROOT_GID, required_mode)
            .map_err(|source| QualifyError::context(format!("inspect {role}"), source))?;
    add_qualified_file(files, bindings, roles, role, qualified.clone())?;
    Ok(qualified)
}

fn add_retained_file(
    files: &mut BTreeMap<PathBuf, QualifiedFileV1>,
    bindings: &mut Vec<QualifiedFileBindingV1>,
    roles: &mut BTreeSet<String>,
    role: &str,
    deployment_root: &Path,
    asset: &LoadedDeploymentAsset,
) -> Result<(), QualifyError> {
    let path = deployment_root.join(asset.relative_path().as_path());
    let qualified = QualifiedFileV1::from_retained_bytes(
        path,
        asset.bytes(),
        *asset.content_sha256().as_bytes(),
        ROOT_UID,
        ROOT_GID,
    )
    .map_err(|source| QualifyError::context(format!("bind exact retained {role}"), source))?;
    add_qualified_file(files, bindings, roles, role, qualified)
}

fn add_qualified_file(
    files: &mut BTreeMap<PathBuf, QualifiedFileV1>,
    bindings: &mut Vec<QualifiedFileBindingV1>,
    roles: &mut BTreeSet<String>,
    role: &str,
    qualified: QualifiedFileV1,
) -> Result<(), QualifyError> {
    if !roles.insert(role.to_owned()) {
        return Err(QualifyError::DuplicateMarkerRole {
            role: role.to_owned(),
        });
    }
    if let Some(existing) = files.get(qualified.path()) {
        if existing != &qualified {
            return Err(QualifyError::ConflictingFileEvidence {
                path: qualified.path().to_path_buf(),
            });
        }
    } else {
        files.insert(qualified.path().to_path_buf(), qualified.clone());
    }
    bindings.push(
        QualifiedFileBindingV1::new(role.to_owned(), qualified.path().to_path_buf())
            .map_err(|source| QualifyError::context(format!("bind marker role {role}"), source))?,
    );
    Ok(())
}

fn require_exact_file_bytes(
    artifact: &'static str,
    path: &Path,
    expected: &[u8],
    evidence: &QualifiedFileV1,
) -> Result<(), QualifyError> {
    let observed = std::fs::read(path)
        .map_err(|source| QualifyError::context(format!("read installed {artifact}"), source))?;
    require_exact_bytes(artifact, path, &observed, expected)?;
    let expected_len = u64::try_from(expected.len()).expect("embedded unit length fits u64");
    if evidence.byte_len() != expected_len {
        return Err(QualifyError::ExactFileEvidenceMismatch {
            artifact,
            path: path.to_path_buf(),
            expected_bytes: expected_len,
            observed_bytes: evidence.byte_len(),
        });
    }
    Ok(())
}

fn require_exact_bytes(
    artifact: &'static str,
    path: &Path,
    observed: &[u8],
    expected: &[u8],
) -> Result<(), QualifyError> {
    if observed != expected {
        return Err(QualifyError::ExactFileBytesMismatch {
            artifact,
            path: path.to_path_buf(),
        });
    }
    Ok(())
}

fn require_exact_systemd_drop_in_topology() -> Result<(), QualifyError> {
    for root in SYSTEMD_SYSTEM_UNIT_SEARCH_ROOTS {
        for drop_in_name in SYSTEMD_RELEVANT_DROP_IN_NAMES {
            let directory = Path::new(root).join(drop_in_name);
            let expected =
                if root == "/etc/systemd/system" && drop_in_name == "kiko-nano-agent.service.d" {
                    BTreeSet::from([OsString::from(EXPECTED_QUALIFIED_DROP_IN_NAME)])
                } else {
                    BTreeSet::new()
                };
            require_exact_regular_directory_entries(
                &directory,
                &expected,
                "effective systemd drop-in topology",
            )?;
        }
    }
    Ok(())
}

fn require_exact_native_library_entries(
    deployment_root: &Path,
    native: &ParsedNativeRuntimeManifest,
) -> Result<(), QualifyError> {
    let expected = native
        .libraries
        .iter()
        .map(|library| OsString::from(&library.soname))
        .collect();
    require_exact_regular_directory_entries(
        &deployment_root.join(native.library_search_relative_path.as_path()),
        &expected,
        "native runtime library directory",
    )
}

fn require_exact_regular_directory_entries(
    directory: &Path,
    expected: &BTreeSet<OsString>,
    contract: &'static str,
) -> Result<(), QualifyError> {
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound && expected.is_empty() => {
            return Ok(());
        }
        Err(source) => {
            return Err(QualifyError::context(
                format!("enumerate {contract} {}", directory.display()),
                source,
            ));
        }
    };
    let mut observed = BTreeSet::new();
    for entry in entries {
        let entry = entry.map_err(|source| {
            QualifyError::context(
                format!("read {contract} entry in {}", directory.display()),
                source,
            )
        })?;
        let path = entry.path();
        let file_type = entry.file_type().map_err(|source| {
            QualifyError::context(
                format!("inspect {contract} entry {}", path.display()),
                source,
            )
        })?;
        if !file_type.is_file() {
            return Err(QualifyError::DirectoryEntryNotRegular { contract, path });
        }
        observed.insert(entry.file_name());
    }
    if observed != *expected {
        return Err(QualifyError::DirectoryEntrySetMismatch {
            contract,
            directory: directory.to_path_buf(),
            missing: expected
                .difference(&observed)
                .map(|name| directory.join(name))
                .collect(),
            unexpected: observed
                .difference(expected)
                .map(|name| directory.join(name))
                .collect(),
        });
    }
    Ok(())
}

#[derive(Debug)]
enum QualifyError {
    Context {
        operation: String,
        source: Box<dyn Error>,
    },
    RootRequired,
    AcknowledgementMismatch,
    PolicyPathOutsideDeployment {
        field: &'static str,
        path: PathBuf,
    },
    PolicyPathAliasesDeploymentRoot {
        field: &'static str,
    },
    NonUtf8Path {
        path: PathBuf,
    },
    Kep2EyeRequired,
    ContinuousNaturalHeadHoldRequired,
    ManifestReadMismatch,
    ManifestArtifactMismatch,
    ManifestArtifactRereadMismatch {
        artifact_id: String,
    },
    ControllerManifestMismatch {
        field: &'static str,
    },
    PlantBindingMismatch {
        field: &'static str,
        artifact_id: String,
    },
    CalibrationBindingMismatch {
        field: &'static str,
        artifact_id: String,
    },
    NativeManifestTooLarge {
        actual_bytes: usize,
    },
    UnsupportedNativeSchema {
        actual: u32,
    },
    WrongNativeLibrarySearchPath {
        actual: String,
    },
    InvalidNativeLibraryCount {
        actual: usize,
    },
    InvalidNativeRole {
        actual: String,
    },
    DuplicateNativeRole {
        role: &'static str,
    },
    MissingNativeRole {
        role: &'static str,
    },
    InvalidSoname {
        actual: String,
    },
    WrongNativeSonameForRole {
        role: &'static str,
        expected: &'static str,
        actual: String,
    },
    NativeLibraryOutsideSearchDirectory {
        soname: String,
        path: String,
    },
    DuplicateNativeLibrary {
        path: String,
    },
    InvalidSha256 {
        actual: String,
    },
    NativeLibraryDigestMismatch {
        soname: String,
        expected: [u8; 32],
        observed: [u8; 32],
    },
    OnnxRuntimeLaunchMismatch,
    TextAssetNotUtf8 {
        label: String,
    },
    TemplateSentinel {
        label: String,
        sentinel: &'static str,
    },
    DuplicateMarkerRole {
        role: String,
    },
    ConflictingFileEvidence {
        path: PathBuf,
    },
    ExactFileBytesMismatch {
        artifact: &'static str,
        path: PathBuf,
    },
    ExactFileEvidenceMismatch {
        artifact: &'static str,
        path: PathBuf,
        expected_bytes: u64,
        observed_bytes: u64,
    },
    DirectoryEntryNotRegular {
        contract: &'static str,
        path: PathBuf,
    },
    DirectoryEntrySetMismatch {
        contract: &'static str,
        directory: PathBuf,
        missing: Vec<PathBuf>,
        unexpected: Vec<PathBuf>,
    },
    PublishedMarkerMismatch,
}

impl QualifyError {
    fn context(operation: impl Into<String>, source: impl Error + 'static) -> QualifyError {
        QualifyError::Context {
            operation: operation.into(),
            source: Box::new(source),
        }
    }
}

impl fmt::Display for QualifyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Context { operation, source } => {
                write!(
                    formatter,
                    "offline install qualification failed while {operation}: {source}"
                )
            }
            Self::RootRequired => write!(
                formatter,
                "offline install qualification requires effective UID 0"
            ),
            Self::AcknowledgementMismatch => write!(
                formatter,
                "offline install qualification acknowledgement did not exactly match the required offline-only scope"
            ),
            Self::PolicyPathOutsideDeployment { field, path } => write!(
                formatter,
                "{field} path {} is outside the fixed deployment root",
                path.display()
            ),
            Self::PolicyPathAliasesDeploymentRoot { field } => {
                write!(formatter, "{field} path may not alias the deployment root")
            }
            Self::NonUtf8Path { path } => {
                write!(formatter, "deployment path {} is not UTF-8", path.display())
            }
            Self::Kep2EyeRequired => {
                write!(
                    formatter,
                    "production offline install requires the KEP2 eye policy"
                )
            }
            Self::ContinuousNaturalHeadHoldRequired => write!(
                formatter,
                "production offline install requires continuous natural head hold"
            ),
            Self::ManifestReadMismatch => write!(
                formatter,
                "retained inventory manifest bytes differ between the typed and deployment readers"
            ),
            Self::ManifestArtifactMismatch => write!(
                formatter,
                "one or more manifest artifact bytes do not match the manifest"
            ),
            Self::ManifestArtifactRereadMismatch { artifact_id } => write!(
                formatter,
                "manifest artifact {artifact_id} changed while qualification evidence was collected"
            ),
            Self::ControllerManifestMismatch { field } => write!(
                formatter,
                "controller contract does not match the inventory manifest field {field}"
            ),
            Self::PlantBindingMismatch { field, artifact_id } => write!(
                formatter,
                "plant artifact {artifact_id} has inconsistent {field} binding"
            ),
            Self::CalibrationBindingMismatch { field, artifact_id } => write!(
                formatter,
                "calibration artifact {artifact_id} has inconsistent {field} binding"
            ),
            Self::NativeManifestTooLarge { actual_bytes } => write!(
                formatter,
                "native runtime manifest is {actual_bytes} bytes, above the fixed bound"
            ),
            Self::UnsupportedNativeSchema { actual } => write!(
                formatter,
                "native runtime manifest schema {actual} is unsupported"
            ),
            Self::WrongNativeLibrarySearchPath { actual } => write!(
                formatter,
                "native library search path {actual:?} is not the fixed lib directory"
            ),
            Self::InvalidNativeLibraryCount { actual } => write!(
                formatter,
                "native runtime manifest has invalid library count {actual}"
            ),
            Self::InvalidNativeRole { actual } => {
                write!(formatter, "native runtime role {actual:?} is unsupported")
            }
            Self::DuplicateNativeRole { role } => {
                write!(formatter, "native runtime role {role} is duplicated")
            }
            Self::MissingNativeRole { role } => {
                write!(formatter, "native runtime role {role} is missing")
            }
            Self::InvalidSoname { actual } => {
                write!(formatter, "native library soname {actual:?} is invalid")
            }
            Self::WrongNativeSonameForRole {
                role,
                expected,
                actual,
            } => write!(
                formatter,
                "native runtime role {role} requires Nano SONAME {expected:?}, got {actual:?}"
            ),
            Self::NativeLibraryOutsideSearchDirectory { soname, path } => write!(
                formatter,
                "native library {soname} path {path:?} is not its direct lib-directory path"
            ),
            Self::DuplicateNativeLibrary { path } => {
                write!(formatter, "native library path {path:?} is duplicated")
            }
            Self::InvalidSha256 { actual } => write!(
                formatter,
                "native library SHA-256 is not 64 lowercase hexadecimal bytes (actual length {})",
                actual.len()
            ),
            Self::NativeLibraryDigestMismatch {
                soname,
                expected,
                observed,
            } => write!(
                formatter,
                "installed native library {soname} digest mismatch: expected {expected:02x?}, observed {observed:02x?}"
            ),
            Self::OnnxRuntimeLaunchMismatch => write!(
                formatter,
                "native ONNX Runtime path or digest differs from the launch binding"
            ),
            Self::TextAssetNotUtf8 { label } => {
                write!(formatter, "textual deployment asset {label} is not UTF-8")
            }
            Self::TemplateSentinel { label, sentinel } => write!(
                formatter,
                "deployment asset {label} retains forbidden template sentinel {sentinel:?}"
            ),
            Self::DuplicateMarkerRole { role } => {
                write!(formatter, "offline marker role {role} is duplicated")
            }
            Self::ConflictingFileEvidence { path } => write!(
                formatter,
                "offline marker collected conflicting evidence for {}",
                path.display()
            ),
            Self::ExactFileBytesMismatch { artifact, path } => write!(
                formatter,
                "installed {artifact} bytes at {} differ from the shipped source",
                path.display()
            ),
            Self::ExactFileEvidenceMismatch {
                artifact,
                path,
                expected_bytes,
                observed_bytes,
            } => write!(
                formatter,
                "{artifact} evidence for {} reports {observed_bytes} bytes, expected {expected_bytes}",
                path.display()
            ),
            Self::DirectoryEntryNotRegular { contract, path } => write!(
                formatter,
                "{contract} contains non-regular entry {}",
                path.display()
            ),
            Self::DirectoryEntrySetMismatch {
                contract,
                directory,
                missing,
                unexpected,
            } => write!(
                formatter,
                "{contract} at {} differs from the closed set: missing {missing:?}, unexpected {unexpected:?}",
                directory.display()
            ),
            Self::PublishedMarkerMismatch => write!(
                formatter,
                "published marker did not read back as the exact constructed marker"
            ),
        }
    }
}

impl Error for QualifyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Context { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(name: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("test clock after epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "kiko-nano-deployment-qualify-{name}-{}-{nonce}",
                std::process::id()
            ));
            std::fs::create_dir(&path).expect("create test directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn native_manifest(libraries: &str) -> Vec<u8> {
        format!(
            r#"{{
              "schema_version": 1,
              "library_search_relative_path": "lib",
              "libraries": [{libraries}]
            }}"#
        )
        .into_bytes()
    }

    fn library(role: &str, soname: &str, byte: char) -> String {
        format!(
            r#"{{
              "role": "{role}",
              "soname": "{soname}",
              "relative_path": "lib/{soname}",
              "maximum_bytes": 4096,
              "sha256_hex": "{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}{byte}"
            }}"#
        )
    }

    fn required_libraries() -> String {
        [
            library("depthai_core", "libdepthai-core.so", 'a'),
            library("dynamic_calibration", "libdynamic_calibration.so", 'b'),
            library("libusb_1_0", "libusb-1.0.so.0", 'c'),
            library("onnxruntime", "libonnxruntime.so.1.23.2", 'd'),
            library("opencv_core", "libopencv_core.so.4.5d", 'e'),
            library("opencv_imgproc", "libopencv_imgproc.so.4.5d", 'f'),
            library("opencv_objdetect", "libopencv_objdetect.so.4.5d", '1'),
        ]
        .join(",")
    }

    #[test]
    fn native_manifest_requires_the_complete_observed_non_system_runtime_set() {
        let parsed =
            parse_native_manifest(&native_manifest(&required_libraries())).expect("complete set");
        assert_eq!(parsed.libraries.len(), 7);

        let missing = [
            library("depthai_core", "libdepthai-core.so", 'a'),
            library("dynamic_calibration", "libdynamic_calibration.so", 'b'),
            library("onnxruntime", "libonnxruntime.so.1.23.2", 'd'),
            library("opencv_core", "libopencv_core.so.4.5d", 'e'),
            library("opencv_imgproc", "libopencv_imgproc.so.4.5d", 'f'),
            library("opencv_objdetect", "libopencv_objdetect.so.4.5d", '1'),
        ]
        .join(",");
        assert!(matches!(
            parse_native_manifest(&native_manifest(&missing)),
            Err(QualifyError::MissingNativeRole { role: "libusb_1_0" })
        ));
    }

    #[test]
    fn native_manifest_rejects_unknown_fields_uppercase_hashes_and_outside_paths() {
        let unknown = native_manifest(
            &library("depthai_core", "libdepthai-core.so", 'a')
                .replace("\"role\":", "\"extra\": true, \"role\":"),
        );
        assert!(parse_native_manifest(&unknown).is_err());

        let uppercase = required_libraries().replacen(
            &"a".repeat(SHA256_HEX_BYTES),
            &"A".repeat(SHA256_HEX_BYTES),
            1,
        );
        assert!(matches!(
            parse_native_manifest(&native_manifest(&uppercase)),
            Err(QualifyError::InvalidSha256 { .. })
        ));

        let outside = required_libraries().replacen(
            "\"relative_path\": \"lib/libdepthai-core.so\"",
            "\"relative_path\": \"other/libdepthai-core.so\"",
            1,
        );
        assert!(matches!(
            parse_native_manifest(&native_manifest(&outside)),
            Err(QualifyError::NativeLibraryOutsideSearchDirectory { .. })
        ));
    }

    #[test]
    fn native_manifest_rejects_relabelled_open_cv_soname() {
        let relabelled =
            required_libraries().replace("libopencv_objdetect.so.4.5d", "libopencv_objdetect.so");
        assert!(matches!(
            parse_native_manifest(&native_manifest(&relabelled)),
            Err(QualifyError::WrongNativeSonameForRole {
                role: "opencv_objdetect",
                expected: "libopencv_objdetect.so.4.5d",
                ..
            })
        ));
    }

    #[test]
    fn template_sentinel_rejection_is_case_insensitive_and_specific() {
        assert!(reject_template_sentinel("config", br#"{"x":"${VALUE}"}"#).is_err());
        assert!(reject_template_sentinel("config", br#"{"x":"Replace_Me"}"#).is_err());
        assert!(reject_template_sentinel("config", br#"{"x":"production"}"#).is_ok());
    }

    #[test]
    fn exact_base_unit_rejects_resets_and_overrides() {
        for appended in [
            b"\n[Service]\nExecStartPre=\n".as_slice(),
            b"\n[Service]\nUser=root\n".as_slice(),
            b"\n[Service]\nRestart=always\n".as_slice(),
        ] {
            let mut observed = EXPECTED_BASE_SYSTEMD_UNIT.to_vec();
            observed.extend_from_slice(appended);
            assert!(matches!(
                require_exact_bytes(
                    "base systemd unit",
                    Path::new(BASE_SYSTEMD_UNIT),
                    &observed,
                    EXPECTED_BASE_SYSTEMD_UNIT,
                ),
                Err(QualifyError::ExactFileBytesMismatch { .. })
            ));
        }
    }

    #[test]
    fn closed_directory_set_rejects_missing_extra_and_non_regular_entries() {
        let root = TestDirectory::new("closed-directory");
        let expected_name = OsString::from("10-qualified-boot.conf");
        let expected = BTreeSet::from([expected_name.clone()]);
        let expected_path = root.path().join(&expected_name);
        std::fs::write(&expected_path, EXPECTED_QUALIFIED_DROP_IN).expect("write expected entry");
        require_exact_regular_directory_entries(root.path(), &expected, "test systemd drop-ins")
            .expect("exact singleton accepted");

        let unexpected = root.path().join("90-bypass.conf");
        std::fs::write(&unexpected, b"[Service]\nExecStartPre=\n").expect("write unexpected entry");
        assert!(matches!(
            require_exact_regular_directory_entries(root.path(), &expected, "test systemd drop-ins"),
            Err(QualifyError::DirectoryEntrySetMismatch {
                unexpected: entries,
                ..
            }) if entries == vec![unexpected.clone()]
        ));
        std::fs::remove_file(&unexpected).expect("remove unexpected entry");
        std::fs::remove_file(&expected_path).expect("remove expected entry");
        assert!(matches!(
            require_exact_regular_directory_entries(root.path(), &expected, "test systemd drop-ins"),
            Err(QualifyError::DirectoryEntrySetMismatch {
                missing: entries,
                ..
            }) if entries == vec![expected_path.clone()]
        ));

        std::os::unix::fs::symlink("missing-target", &expected_path)
            .expect("create non-regular expected entry");
        assert!(matches!(
            require_exact_regular_directory_entries(root.path(), &expected, "test systemd drop-ins"),
            Err(QualifyError::DirectoryEntryNotRegular { path, .. }) if path == expected_path
        ));
    }

    #[test]
    fn native_library_directory_rejects_every_unmanifested_entry() {
        let root = TestDirectory::new("native-library-directory");
        let deployment_root = root.path();
        let lib = deployment_root.join(NATIVE_LIBRARY_SEARCH_RELATIVE_PATH);
        std::fs::create_dir(&lib).expect("create native library directory");
        let parsed =
            parse_native_manifest(&native_manifest(&required_libraries())).expect("manifest");
        for library in &parsed.libraries {
            std::fs::write(lib.join(&library.soname), b"fixture").expect("write native entry");
        }
        require_exact_native_library_entries(deployment_root, &parsed)
            .expect("manifest closes native directory");

        let injected = lib.join("libinjected.so");
        std::fs::write(&injected, b"unbound").expect("write injected library");
        assert!(matches!(
            require_exact_native_library_entries(deployment_root, &parsed),
            Err(QualifyError::DirectoryEntrySetMismatch {
                unexpected: entries,
                ..
            }) if entries == vec![injected]
        ));
    }
}
