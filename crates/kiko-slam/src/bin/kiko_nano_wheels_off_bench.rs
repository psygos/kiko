#[cfg(not(all(feature = "nano-bench", unix)))]
fn main() {
    eprintln!("kiko-nano-wheels-off-bench requires a Unix target and --features nano-bench");
    std::process::exit(2);
}

#[cfg(all(feature = "nano-bench", unix))]
mod enabled {
    use std::fmt;
    use std::net::SocketAddr;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    use clap::Parser;
    use kiko_device_inventory::{
        ArtifactRelativePath, DeploymentAssetByteLimit, ExactCalibrationBundleSha256,
        LoadedDeploymentAsset, LoadedExpectedManifestV1, ManifestArtifactHashes,
        hash_manifest_artifacts, load_deployment_asset, load_expected_manifest_v1_file,
    };
    use kiko_expression_core::StreamEpochId;
    use kiko_eye_runtime::{
        ActorExit as EyeActorExit, EyeActorHandle, EyeActorStartError, EyeActorTask,
        EyeRuntimeConfig, EyeRuntimeFault, FirmwareAdmissionEvidence,
        HandleRequestError as EyeHandleRequestError, MonotonicClock as EyeMonotonicClock,
        ReleaseReport, SerialConfigurationEvidence as EyeSerialEvidence,
        StartupEvidence as EyeActorStartupEvidence, StartupReceiptError as EyeStartupReceiptError,
        start_serial_eye_actor,
    };
    use kiko_head_protocol::PositionTicks;
    use kiko_head_runtime::{
        ActorExit as HeadActorExit, HeadActorHandle, HeadActorStartError, HeadActorTask,
        HeadRuntimeConfig, HeadRuntimeError, PhysicalTorqueEnableConsent,
        SerialConfigurationEvidence as HeadSerialEvidence, ShutdownError as HeadShutdownError,
        StartupReceiptError as HeadStartupReceiptError, TorqueDisableReport,
        VerifiedNaturalHoldEvidence, start_serial_head_actor,
    };
    use kiko_slam::HostMonotonicTimestamp;
    use kiko_slam::navigation::NavigationClockEpoch;
    use kiko_slam::navigation::{
        AgentAuthoritySupervisor, FreshZeroEvidence, NanoAgentPolicyConfigV1,
        NativeWheelsOffOakPort, RefreshedBaseZero, RerunWheelsOffTelemetry,
        WheelsOffBaseCleanupPort, WheelsOffBenchCancellation, WheelsOffBenchCancellationPort,
        WheelsOffBenchCapturePlan, WheelsOffBenchConfiguration, WheelsOffBenchOakConfig,
        WheelsOffBenchPlan, WheelsOffBenchRerunPlan, WheelsOffBenchRuntime,
        WheelsOffConfiguredPoseBounds, WheelsOffEyePort, WheelsOffHeadPort, ZeroHoldKeeper,
        ZeroHoldRequestError, ZeroHoldStatus, ZeroHoldTerminalError, ZeroOnlyActuationPolicyV1,
        wheels_off_rgb_expression_bridge,
    };
    use kiko_supervisor_core::{
        ReadinessBinding, ReadinessEpoch, Sha256Digest, StopReason, SupervisorAction,
        ZeroEvidenceError,
    };
    use oak_sys::{
        DepthAlignment, DepthConfig, DeviceConfig, ImuConfig, MonoConfig, QueueConfig, RgbConfig,
    };
    use robot_command_client::DisarmReceipt;
    use robot_server::config::{ControllerServerConfigV1, MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES};
    use serde::Deserialize;
    use tokio::task::JoinError;

    const BENCH_CONFIG_V1: u32 = 1;
    const MAX_BENCH_CONFIG_JSON_BYTES: u64 = 16 * 1_024;
    const MAX_AGENT_POLICY_JSON_BYTES: u64 = 64 * 1_024;
    const MAX_ZERO_ONLY_JSON_BYTES: u64 = 4 * 1_024;
    const MAX_OAK_DIMENSION_PX: u32 = 4_096;
    const MAX_OAK_FPS: u32 = 120;
    const MAX_OAK_IMU_RATE_HZ: u32 = 1_000;
    const MAX_OAK_QUEUE_SIZE: u32 = 16;
    const MAX_BENCH_RUN_MS: u64 = 15 * 60 * 1_000;
    const MIN_RERUN_SERVER_MEMORY_BYTES: u64 = 1024 * 1024;
    const MAX_RERUN_SERVER_MEMORY_BYTES: u64 = 512 * 1024 * 1024;

    #[derive(Parser, Debug)]
    #[command(
        name = "kiko-nano-wheels-off-bench",
        about = "Fail-closed physical wheels-off Nano camera/head/eye bench"
    )]
    struct Cli {
        /// Absolute deployment root opened without following symlinks.
        #[arg(long)]
        deployment_root: PathBuf,
        /// Canonical path to the bench launch document, relative to deployment root.
        #[arg(long)]
        config: String,
        /// Operator attests that every drive wheel is physically removed.
        #[arg(long, action = clap::ArgAction::SetTrue, required = true)]
        wheels_removed: bool,
        /// Operator attests that the complete head travel path is clear.
        #[arg(long, action = clap::ArgAction::SetTrue, required = true)]
        head_path_clear: bool,
        /// Operator attests that physical power can be cut immediately.
        #[arg(long, action = clap::ArgAction::SetTrue, required = true)]
        power_cut_reachable: bool,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct BenchConfigDto {
        schema_version: u32,
        agent_policy_asset: String,
        zero_only_policy_asset: String,
        controller_server_asset: String,
        command_bind: SocketAddr,
        oak: OakConfigDto,
        capture: CaptureConfigDto,
        ready_pose: ReadyPoseDto,
        rerun: RerunConfigDto,
        maximum_run_ms: u64,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct OakConfigDto {
        rgb_width_px: u32,
        rgb_height_px: u32,
        rgb_fps: u32,
        stereo_width_px: u32,
        stereo_height_px: u32,
        stereo_fps: u32,
        imu_rate_hz: u32,
        queue_size: u32,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct CaptureConfigDto {
        timeout_ms: u32,
        attempts: u16,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct ReadyPoseDto {
        minimum_ticks: [u16; 4],
        maximum_ticks: [u16; 4],
    }

    #[derive(Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
    enum RerunConfigDto {
        SaveRrd {
            absolute_path: String,
            flush_timeout_ms: u64,
        },
        ServeLoopback {
            port: u16,
            memory_limit_bytes: u64,
            flush_timeout_ms: u64,
        },
    }

    struct BenchConfig {
        agent_policy_asset: ArtifactRelativePath,
        zero_only_policy_asset: ArtifactRelativePath,
        controller_server_asset: ArtifactRelativePath,
        command_bind: SocketAddr,
        oak: DeviceConfig,
        capture: WheelsOffBenchCapturePlan,
        ready_pose: WheelsOffConfiguredPoseBounds,
        rerun: RerunOutput,
        rerun_plan: WheelsOffBenchRerunPlan,
        maximum_run: Duration,
    }

    enum RerunOutput {
        Save(PathBuf),
        ServeLoopback { port: u16, memory_limit_bytes: u64 },
    }

    #[derive(Debug)]
    enum BenchConfigError {
        JsonDecode(serde_json::Error),
        JsonTrailingData(serde_json::Error),
        UnsupportedSchemaVersion {
            actual: u32,
            supported: u32,
        },
        AssetPath {
            field: &'static str,
            source: kiko_device_inventory::ArtifactRelativePathError,
        },
        OakFieldOutOfRange {
            field: &'static str,
            actual: u32,
            minimum: u32,
            maximum: u32,
        },
        OakPixelCountOverflow {
            stream: &'static str,
        },
        OakConfig(oak_sys::DeviceConfigError),
        Capture(kiko_slam::navigation::BenchCapturePlanError),
        ReadyPose(kiko_slam::navigation::WheelsOffConfiguredPoseBoundsError),
        Rerun(kiko_slam::navigation::WheelsOffBenchRerunPlanError),
        RerunPortZero,
        RerunMemoryLimitOutOfRange {
            actual_bytes: u64,
            minimum_bytes: u64,
            maximum_bytes: u64,
        },
        RerunPathNotCanonicalAbsolute {
            path: PathBuf,
        },
        MaximumRunOutOfRange {
            actual_ms: u64,
            maximum_ms: u64,
        },
    }

    impl fmt::Display for BenchConfigError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("invalid wheels-off entrypoint configuration: ")?;
            match self {
                Self::JsonDecode(source) => write!(formatter, "JSON decode failed: {source}"),
                Self::JsonTrailingData(source) => {
                    write!(formatter, "trailing JSON data: {source}")
                }
                Self::UnsupportedSchemaVersion { actual, supported } => write!(
                    formatter,
                    "schema version {actual} is unsupported; expected {supported}"
                ),
                Self::AssetPath { field, source } => {
                    write!(
                        formatter,
                        "{field} is not a canonical deployment path: {source}"
                    )
                }
                Self::OakFieldOutOfRange {
                    field,
                    actual,
                    minimum,
                    maximum,
                } => write!(
                    formatter,
                    "{field}={actual} is outside {minimum}..={maximum}"
                ),
                Self::OakPixelCountOverflow { stream } => {
                    write!(
                        formatter,
                        "{stream} dimensions overflow their byte-count bound"
                    )
                }
                Self::OakConfig(source) => write!(formatter, "OAK pipeline is invalid: {source}"),
                Self::Capture(source) => write!(formatter, "capture plan is invalid: {source}"),
                Self::ReadyPose(source) => {
                    write!(formatter, "configured head pose is invalid: {source}")
                }
                Self::Rerun(source) => write!(formatter, "Rerun plan is invalid: {source}"),
                Self::RerunPortZero => formatter.write_str("Rerun loopback port is zero"),
                Self::RerunMemoryLimitOutOfRange {
                    actual_bytes,
                    minimum_bytes,
                    maximum_bytes,
                } => write!(
                    formatter,
                    "Rerun proxy memory_limit_bytes={actual_bytes} is outside {minimum_bytes}..={maximum_bytes}"
                ),
                Self::RerunPathNotCanonicalAbsolute { path } => write!(
                    formatter,
                    "Rerun output path is not canonical and absolute: {}",
                    path.display()
                ),
                Self::MaximumRunOutOfRange {
                    actual_ms,
                    maximum_ms,
                } => write!(
                    formatter,
                    "maximum_run_ms={actual_ms} is outside 1..={maximum_ms}"
                ),
            }
        }
    }

    impl std::error::Error for BenchConfigError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::JsonDecode(source) | Self::JsonTrailingData(source) => Some(source),
                Self::AssetPath { source, .. } => Some(source),
                Self::OakConfig(source) => Some(source),
                Self::Capture(source) => Some(source),
                Self::ReadyPose(source) => Some(source),
                Self::Rerun(source) => Some(source),
                Self::UnsupportedSchemaVersion { .. }
                | Self::OakFieldOutOfRange { .. }
                | Self::OakPixelCountOverflow { .. }
                | Self::RerunPortZero
                | Self::RerunMemoryLimitOutOfRange { .. }
                | Self::RerunPathNotCanonicalAbsolute { .. }
                | Self::MaximumRunOutOfRange { .. } => None,
            }
        }
    }

    impl BenchConfig {
        fn parse(bytes: &[u8]) -> Result<Self, BenchConfigError> {
            let mut deserializer = serde_json::Deserializer::from_slice(bytes);
            let dto = BenchConfigDto::deserialize(&mut deserializer)
                .map_err(BenchConfigError::JsonDecode)?;
            deserializer
                .end()
                .map_err(BenchConfigError::JsonTrailingData)?;
            if dto.schema_version != BENCH_CONFIG_V1 {
                return Err(BenchConfigError::UnsupportedSchemaVersion {
                    actual: dto.schema_version,
                    supported: BENCH_CONFIG_V1,
                });
            }
            let agent_policy_asset =
                parse_asset_path("agent_policy_asset", dto.agent_policy_asset)?;
            let zero_only_policy_asset =
                parse_asset_path("zero_only_policy_asset", dto.zero_only_policy_asset)?;
            let controller_server_asset =
                parse_asset_path("controller_server_asset", dto.controller_server_asset)?;
            let oak = parse_oak_config(dto.oak)?;
            let capture =
                WheelsOffBenchCapturePlan::try_new(dto.capture.timeout_ms, dto.capture.attempts)
                    .map_err(BenchConfigError::Capture)?;
            let ready_pose = WheelsOffConfiguredPoseBounds::try_new(
                dto.ready_pose.minimum_ticks,
                dto.ready_pose.maximum_ticks,
            )
            .map_err(BenchConfigError::ReadyPose)?;
            let (rerun, flush_timeout_ms) = match dto.rerun {
                RerunConfigDto::SaveRrd {
                    absolute_path,
                    flush_timeout_ms,
                } => {
                    if !is_canonical_absolute_path(&absolute_path) {
                        let path = PathBuf::from(absolute_path);
                        return Err(BenchConfigError::RerunPathNotCanonicalAbsolute { path });
                    }
                    let path = PathBuf::from(absolute_path);
                    (RerunOutput::Save(path), flush_timeout_ms)
                }
                RerunConfigDto::ServeLoopback {
                    port,
                    memory_limit_bytes,
                    flush_timeout_ms,
                } => {
                    if port == 0 {
                        return Err(BenchConfigError::RerunPortZero);
                    }
                    if !(MIN_RERUN_SERVER_MEMORY_BYTES..=MAX_RERUN_SERVER_MEMORY_BYTES)
                        .contains(&memory_limit_bytes)
                    {
                        return Err(BenchConfigError::RerunMemoryLimitOutOfRange {
                            actual_bytes: memory_limit_bytes,
                            minimum_bytes: MIN_RERUN_SERVER_MEMORY_BYTES,
                            maximum_bytes: MAX_RERUN_SERVER_MEMORY_BYTES,
                        });
                    }
                    (
                        RerunOutput::ServeLoopback {
                            port,
                            memory_limit_bytes,
                        },
                        flush_timeout_ms,
                    )
                }
            };
            let rerun_plan = WheelsOffBenchRerunPlan::try_from_milliseconds(flush_timeout_ms)
                .map_err(BenchConfigError::Rerun)?;
            if dto.maximum_run_ms == 0 || dto.maximum_run_ms > MAX_BENCH_RUN_MS {
                return Err(BenchConfigError::MaximumRunOutOfRange {
                    actual_ms: dto.maximum_run_ms,
                    maximum_ms: MAX_BENCH_RUN_MS,
                });
            }
            Ok(Self {
                agent_policy_asset,
                zero_only_policy_asset,
                controller_server_asset,
                command_bind: dto.command_bind,
                oak,
                capture,
                ready_pose,
                rerun,
                rerun_plan,
                maximum_run: Duration::from_millis(dto.maximum_run_ms),
            })
        }
    }

    fn parse_asset_path(
        field: &'static str,
        value: String,
    ) -> Result<ArtifactRelativePath, BenchConfigError> {
        ArtifactRelativePath::parse(value)
            .map_err(|source| BenchConfigError::AssetPath { field, source })
    }

    fn bounded_oak_field(
        field: &'static str,
        actual: u32,
        maximum: u32,
    ) -> Result<u32, BenchConfigError> {
        if actual == 0 || actual > maximum {
            Err(BenchConfigError::OakFieldOutOfRange {
                field,
                actual,
                minimum: 1,
                maximum,
            })
        } else {
            Ok(actual)
        }
    }

    fn parse_oak_config(dto: OakConfigDto) -> Result<DeviceConfig, BenchConfigError> {
        let rgb_width =
            bounded_oak_field("oak.rgb_width_px", dto.rgb_width_px, MAX_OAK_DIMENSION_PX)?;
        let rgb_height =
            bounded_oak_field("oak.rgb_height_px", dto.rgb_height_px, MAX_OAK_DIMENSION_PX)?;
        let rgb_fps = bounded_oak_field("oak.rgb_fps", dto.rgb_fps, MAX_OAK_FPS)?;
        let stereo_width = bounded_oak_field(
            "oak.stereo_width_px",
            dto.stereo_width_px,
            MAX_OAK_DIMENSION_PX,
        )?;
        let stereo_height = bounded_oak_field(
            "oak.stereo_height_px",
            dto.stereo_height_px,
            MAX_OAK_DIMENSION_PX,
        )?;
        let stereo_fps = bounded_oak_field("oak.stereo_fps", dto.stereo_fps, MAX_OAK_FPS)?;
        let imu_rate_hz =
            bounded_oak_field("oak.imu_rate_hz", dto.imu_rate_hz, MAX_OAK_IMU_RATE_HZ)?;
        let queue_size = bounded_oak_field("oak.queue_size", dto.queue_size, MAX_OAK_QUEUE_SIZE)?;
        rgb_width
            .checked_mul(rgb_height)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or(BenchConfigError::OakPixelCountOverflow { stream: "RGB" })?;
        stereo_width
            .checked_mul(stereo_height)
            .ok_or(BenchConfigError::OakPixelCountOverflow { stream: "stereo" })?;

        // The depth contract is deliberately navigation-compatible: rectified
        // mono, rectified-left depth, and equal stereo shape/timing. RGB-aligned
        // depth would silently change the projection frame used by SLAM.
        let config = DeviceConfig {
            rgb: Some(RgbConfig {
                width: rgb_width,
                height: rgb_height,
                fps: rgb_fps,
            }),
            mono: Some(MonoConfig {
                width: stereo_width,
                height: stereo_height,
                fps: stereo_fps,
                rectified: true,
            }),
            depth: Some(DepthConfig {
                width: stereo_width,
                height: stereo_height,
                fps: stereo_fps,
                alignment: DepthAlignment::RectifiedLeft,
            }),
            imu: Some(ImuConfig {
                rate_hz: imu_rate_hz,
            }),
            queue: QueueConfig {
                size: queue_size,
                blocking: false,
            },
        };
        config.validate().map_err(BenchConfigError::OakConfig)?;
        Ok(config)
    }

    fn is_canonical_absolute_path(value: &str) -> bool {
        let bytes = value.as_bytes();
        bytes.len() > 1
            && bytes.first() == Some(&b'/')
            && bytes.last() != Some(&b'/')
            && bytes[1..]
                .split(|byte| *byte == b'/')
                .all(|component| !component.is_empty() && component != b"." && component != b"..")
            && !bytes.contains(&0)
    }

    #[cfg(test)]
    mod config_tests {
        use super::*;

        fn bench_document(memory_limit_bytes: u64) -> Vec<u8> {
            serde_json::to_vec(&serde_json::json!({
                "schema_version": 1,
                "agent_policy_asset": "agent-policy-v1.json",
                "zero_only_policy_asset": "nano-zero-only-v1.json",
                "controller_server_asset": "controller-server-v1.json",
                "command_bind": "127.0.0.1:8080",
                "oak": {
                    "rgb_width_px": 640,
                    "rgb_height_px": 480,
                    "rgb_fps": 30,
                    "stereo_width_px": 640,
                    "stereo_height_px": 400,
                    "stereo_fps": 30,
                    "imu_rate_hz": 400,
                    "queue_size": 4
                },
                "capture": { "timeout_ms": 500, "attempts": 20 },
                "ready_pose": {
                    "minimum_ticks": [2000, 2000, 2000, 2000],
                    "maximum_ticks": [2100, 2100, 2100, 2100]
                },
                "rerun": {
                    "kind": "serve_loopback",
                    "port": 9876,
                    "memory_limit_bytes": memory_limit_bytes,
                    "flush_timeout_ms": 10000
                },
                "maximum_run_ms": 900000
            }))
            .expect("serialize bench fixture")
        }

        #[test]
        fn rerun_proxy_memory_is_explicitly_bounded() {
            BenchConfig::parse(&bench_document(MIN_RERUN_SERVER_MEMORY_BYTES))
                .expect("minimum bounded proxy cache");
            BenchConfig::parse(&bench_document(MAX_RERUN_SERVER_MEMORY_BYTES))
                .expect("maximum bounded proxy cache");

            for actual_bytes in [
                MIN_RERUN_SERVER_MEMORY_BYTES - 1,
                MAX_RERUN_SERVER_MEMORY_BYTES + 1,
            ] {
                assert!(matches!(
                    BenchConfig::parse(&bench_document(actual_bytes)),
                    Err(BenchConfigError::RerunMemoryLimitOutOfRange {
                        actual_bytes: actual,
                        minimum_bytes: MIN_RERUN_SERVER_MEMORY_BYTES,
                        maximum_bytes: MAX_RERUN_SERVER_MEMORY_BYTES,
                    }) if actual == actual_bytes
                ));
            }
        }

        #[test]
        fn running_phase_cannot_exceed_the_supported_unit_contract() {
            let mut document: serde_json::Value =
                serde_json::from_slice(&bench_document(MIN_RERUN_SERVER_MEMORY_BYTES))
                    .expect("decode bench fixture");
            document["maximum_run_ms"] = serde_json::json!(MAX_BENCH_RUN_MS + 1);
            let bytes = serde_json::to_vec(&document).expect("serialize oversized run fixture");
            assert!(matches!(
                BenchConfig::parse(&bytes),
                Err(BenchConfigError::MaximumRunOutOfRange {
                    actual_ms,
                    maximum_ms: MAX_BENCH_RUN_MS,
                }) if actual_ms == MAX_BENCH_RUN_MS + 1
            ));
        }
    }

    #[derive(Debug)]
    enum ZeroReceiptError {
        TimestampOutOfRange { nanoseconds: u128 },
    }

    impl fmt::Display for ZeroReceiptError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::TimestampOutOfRange { nanoseconds } => write!(
                    formatter,
                    "applied-zero host timestamp {nanoseconds} ns does not fit the supervisor clock"
                ),
            }
        }
    }

    impl std::error::Error for ZeroReceiptError {}

    fn host_now(origin: Instant) -> Result<HostMonotonicTimestamp, ZeroReceiptError> {
        let nanoseconds = origin.elapsed().as_nanos();
        let nanoseconds = u64::try_from(nanoseconds)
            .map_err(|_| ZeroReceiptError::TimestampOutOfRange { nanoseconds })?;
        Ok(HostMonotonicTimestamp::from_nanos(nanoseconds))
    }

    type AnyError = Box<dyn std::error::Error + Send + Sync>;

    #[derive(Debug)]
    struct StageError {
        stage: &'static str,
        source: AnyError,
    }

    impl fmt::Display for StageError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "{}: {}", self.stage, self.source)
        }
    }

    impl std::error::Error for StageError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(self.source.as_ref())
        }
    }

    fn at_stage<E>(stage: &'static str, source: E) -> AnyError
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Box::new(StageError {
            stage,
            source: Box::new(source),
        })
    }

    #[derive(Debug)]
    struct OwnedEvidenceError {
        detail: String,
    }

    impl fmt::Display for OwnedEvidenceError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(&self.detail)
        }
    }

    impl std::error::Error for OwnedEvidenceError {}

    struct PreparedBench {
        inventory: LoadedExpectedManifestV1,
        policy: NanoAgentPolicyConfigV1,
        zero_client: robot_command_client::ClientConfig,
        bench: WheelsOffBenchConfiguration,
        recording: rerun::RecordingStream,
        maximum_run: Duration,
        artifact_hashes: ManifestArtifactHashes,
        calibration_bundle: ExactCalibrationBundleSha256,
        bench_config_sha256: kiko_device_inventory::DeploymentAssetContentSha256,
        agent_policy_sha256: kiko_device_inventory::DeploymentAssetContentSha256,
        zero_policy_sha256: kiko_device_inventory::DeploymentAssetContentSha256,
        controller_server_sha256: kiko_device_inventory::DeploymentAssetContentSha256,
    }

    fn asset_limit(bytes: u64) -> DeploymentAssetByteLimit {
        DeploymentAssetByteLimit::try_new(bytes)
            .expect("entrypoint asset byte limits are compile-time valid")
    }

    fn load_asset(
        root: &Path,
        path: ArtifactRelativePath,
        maximum_bytes: u64,
        stage: &'static str,
    ) -> Result<LoadedDeploymentAsset, AnyError> {
        load_deployment_asset(root, path, asset_limit(maximum_bytes))
            .map_err(|source| at_stage(stage, source))
    }

    fn build_recording(output: RerunOutput) -> Result<rerun::RecordingStream, AnyError> {
        let builder = rerun::RecordingStreamBuilder::new("kiko-nano-wheels-off-bench");
        match output {
            RerunOutput::Save(path) => {
                eprintln!("rerun_output=save_rrd path={}", path.display());
                builder
                    .save(path)
                    .map_err(|source| at_stage("open Rerun RRD sink", source))
            }
            RerunOutput::ServeLoopback {
                port,
                memory_limit_bytes,
            } => {
                eprintln!(
                    "rerun_output=serve_loopback address=127.0.0.1:{port} memory_limit_bytes={memory_limit_bytes}"
                );
                eprintln!("rerun_forward_hint=ssh -L {port}:127.0.0.1:{port} <nano-host>");
                builder
                    .serve_grpc_opts(
                        "127.0.0.1",
                        port,
                        rerun::ServerOptions {
                            memory_limit: rerun::MemoryLimit::from_bytes(memory_limit_bytes),
                            ..Default::default()
                        },
                    )
                    .map_err(|source| at_stage("start loopback Rerun gRPC sink", source))
            }
        }
    }

    fn prepare(cli: &Cli) -> Result<PreparedBench, AnyError> {
        let bench_path = ArtifactRelativePath::parse(cli.config.clone())
            .map_err(|source| at_stage("parse --config relative path", source))?;
        let bench_asset = load_asset(
            &cli.deployment_root,
            bench_path,
            MAX_BENCH_CONFIG_JSON_BYTES,
            "load bench launch document",
        )?;
        let config = BenchConfig::parse(bench_asset.bytes())
            .map_err(|source| at_stage("parse bench launch document", source))?;

        let agent_policy_asset = load_asset(
            &cli.deployment_root,
            config.agent_policy_asset.clone(),
            MAX_AGENT_POLICY_JSON_BYTES,
            "load Nano agent policy",
        )?;
        let zero_policy_asset = load_asset(
            &cli.deployment_root,
            config.zero_only_policy_asset.clone(),
            MAX_ZERO_ONLY_JSON_BYTES,
            "load zero-only base policy",
        )?;
        let controller_server_asset = load_asset(
            &cli.deployment_root,
            config.controller_server_asset.clone(),
            u64::try_from(MAX_CONTROLLER_SERVER_CONFIG_JSON_BYTES)
                .expect("controller config byte bound fits u64"),
            "load controller-server contract",
        )?;

        let policy = NanoAgentPolicyConfigV1::parse_json(agent_policy_asset.bytes())
            .map_err(|source| at_stage("parse Nano agent policy", source))?;
        let inventory =
            load_expected_manifest_v1_file(policy.inventory().manifest_path().as_path())
                .map_err(|source| at_stage("load exact device inventory", source))?;
        let artifact_hashes = hash_manifest_artifacts(
            inventory.manifest(),
            policy.inventory().artifact_root_path().as_path(),
            policy.inventory().artifact_bindings().clone(),
        )
        .map_err(|source| at_stage("hash manifest-bound artifacts", source))?;
        let calibration_bundle =
            artifact_hashes
                .exact_calibration_bundle_sha256()
                .map_err(|source| {
                    at_stage(
                        "bind exact calibration bundle",
                        OwnedEvidenceError {
                            detail: source.to_string(),
                        },
                    )
                })?;

        let zero_policy = ZeroOnlyActuationPolicyV1::parse_json(zero_policy_asset.bytes())
            .map_err(|source| at_stage("parse zero-only base policy", source))?;
        let controller_server =
            ControllerServerConfigV1::parse_json(controller_server_asset.bytes())
                .map_err(|source| at_stage("parse controller-server contract", source))?;
        let zero_client = zero_policy
            .bind(&inventory, &controller_server, config.command_bind)
            .map_err(|source| at_stage("cross-bind zero-only base policy", source))?
            .into_client();

        let oak = WheelsOffBenchOakConfig::try_new(
            inventory.manifest().oak().mxid().as_str().to_owned(),
            config.oak,
        )
        .map_err(|source| at_stage("bind exact OAK bench pipeline", source))?;
        let bench = WheelsOffBenchConfiguration::new(
            oak,
            config.capture,
            config.rerun_plan,
            config.ready_pose,
        );
        let recording = build_recording(config.rerun)?;

        Ok(PreparedBench {
            inventory,
            policy,
            zero_client,
            bench,
            recording,
            maximum_run: config.maximum_run,
            artifact_hashes,
            calibration_bundle,
            bench_config_sha256: bench_asset.content_sha256(),
            agent_policy_sha256: agent_policy_asset.content_sha256(),
            zero_policy_sha256: zero_policy_asset.content_sha256(),
            controller_server_sha256: controller_server_asset.content_sha256(),
        })
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum BenchStopCause {
        Signal(WheelsOffBenchCancellation),
        TimeBudgetElapsed,
    }

    struct SignalMonitor {
        receiver: tokio::sync::oneshot::Receiver<WheelsOffBenchCancellation>,
        task: tokio::task::JoinHandle<()>,
        observed: Option<WheelsOffBenchCancellation>,
    }

    impl SignalMonitor {
        fn start() -> Result<Self, std::io::Error> {
            let mut interrupt =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
            let mut terminate =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
            let (sender, receiver) = tokio::sync::oneshot::channel();
            let task = tokio::spawn(async move {
                let cause = tokio::select! {
                    _ = interrupt.recv() => WheelsOffBenchCancellation::Interrupt,
                    _ = terminate.recv() => WheelsOffBenchCancellation::Terminate,
                };
                let _receiver_still_present = sender.send(cause).is_ok();
            });
            Ok(Self {
                receiver,
                task,
                observed: None,
            })
        }

        fn try_signal(&mut self) -> Option<WheelsOffBenchCancellation> {
            if let Some(cause) = self.observed {
                return Some(cause);
            }
            let observed = match self.receiver.try_recv() {
                Ok(cause) => Some(cause),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty) => None,
                Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                    Some(WheelsOffBenchCancellation::Terminate)
                }
            };
            if let Some(cause) = observed {
                self.observed = Some(cause);
            }
            observed
        }

        fn stop(self) {
            self.task.abort();
        }
    }

    impl Drop for SignalMonitor {
        fn drop(&mut self) {
            self.task.abort();
        }
    }

    impl WheelsOffBenchCancellationPort for SignalMonitor {
        fn poll_cancellation(&mut self) -> Option<WheelsOffBenchCancellation> {
            self.try_signal()
        }
    }

    fn fresh_stream_epoch() -> Result<StreamEpochId, AnyError> {
        let elapsed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|source| at_stage("read wall clock for RGB stream epoch", source))?;
        let nanoseconds = u64::try_from(elapsed.as_nanos()).map_err(|_| {
            at_stage(
                "build RGB stream epoch",
                OwnedEvidenceError {
                    detail: format!(
                        "Unix wall-clock nanoseconds do not fit u64: {}",
                        elapsed.as_nanos()
                    ),
                },
            )
        })?;
        StreamEpochId::try_new(nanoseconds)
            .map_err(|source| at_stage("build RGB stream epoch", source))
    }

    #[derive(Debug)]
    struct UnexpectedSupervisorAction {
        stage: &'static str,
        expected: SupervisorAction,
        actual: SupervisorAction,
    }

    impl fmt::Display for UnexpectedSupervisorAction {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "supervisor action at {} was {:?}, expected {:?}",
                self.stage, self.actual, self.expected
            )
        }
    }

    impl std::error::Error for UnexpectedSupervisorAction {}

    fn require_supervisor_action(
        stage: &'static str,
        actual: SupervisorAction,
        expected: SupervisorAction,
    ) -> Result<(), AnyError> {
        if actual == expected {
            Ok(())
        } else {
            Err(at_stage(
                stage,
                UnexpectedSupervisorAction {
                    stage,
                    expected,
                    actual,
                },
            ))
        }
    }

    fn supervisor_sha256(field: &'static str, bytes: [u8; 32]) -> Result<Sha256Digest, AnyError> {
        Sha256Digest::try_new(bytes).map_err(|source| at_stage(field, source))
    }

    struct KeeperBasePort {
        keeper: Option<ZeroHoldKeeper>,
    }

    impl KeeperBasePort {
        fn new(keeper: ZeroHoldKeeper) -> Self {
            Self {
                keeper: Some(keeper),
            }
        }
    }

    #[derive(Debug)]
    enum KeeperBasePortError {
        MissingKeeper,
        Request(ZeroHoldRequestError),
        ReceiptTimestamp(ZeroReceiptError),
        ZeroEvidence(ZeroEvidenceError),
        Terminal(ZeroHoldTerminalError),
    }

    impl fmt::Display for KeeperBasePortError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "zero-hold keeper adapter failed: {self:?}")
        }
    }

    impl std::error::Error for KeeperBasePortError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::Request(source) => Some(source),
                Self::ReceiptTimestamp(source) => Some(source),
                Self::ZeroEvidence(source) => Some(source),
                Self::Terminal(source) => Some(source),
                Self::MissingKeeper => None,
            }
        }
    }

    fn fresh_zero_timestamp(
        evidence: FreshZeroEvidence,
    ) -> Result<HostMonotonicTimestamp, ZeroReceiptError> {
        let nanoseconds = evidence.acknowledged_at().nanos_since_clock_start();
        let nanoseconds = u64::try_from(nanoseconds)
            .map_err(|_| ZeroReceiptError::TimestampOutOfRange { nanoseconds })?;
        Ok(HostMonotonicTimestamp::from_nanos(nanoseconds))
    }

    fn refreshed_zero(
        evidence: FreshZeroEvidence,
    ) -> Result<RefreshedBaseZero<FreshZeroEvidence>, KeeperBasePortError> {
        let observed_at =
            fresh_zero_timestamp(evidence).map_err(KeeperBasePortError::ReceiptTimestamp)?;
        RefreshedBaseZero::try_from_host_result(evidence, evidence.host_result(), observed_at)
            .map_err(KeeperBasePortError::ZeroEvidence)
    }

    impl WheelsOffBaseCleanupPort for KeeperBasePort {
        type Evidence = FreshZeroEvidence;
        type HealthEvidence = ZeroHoldStatus;
        type DisarmEvidence = DisarmReceipt;
        type Error = KeeperBasePortError;

        async fn check_health(&mut self) -> Result<Self::HealthEvidence, Self::Error> {
            self.keeper
                .as_mut()
                .ok_or(KeeperBasePortError::MissingKeeper)?
                .status()
                .map_err(KeeperBasePortError::Request)
        }

        async fn refresh_zero(&mut self) -> Result<RefreshedBaseZero<Self::Evidence>, Self::Error> {
            let keeper = self
                .keeper
                .as_mut()
                .ok_or(KeeperBasePortError::MissingKeeper)?;
            let evidence = keeper
                .force_fresh_zero()
                .map_err(KeeperBasePortError::Request)?;
            refreshed_zero(evidence)
        }

        async fn disarm(&mut self) -> Result<Self::DisarmEvidence, Self::Error> {
            let keeper = self
                .keeper
                .take()
                .ok_or(KeeperBasePortError::MissingKeeper)?;
            keeper
                .disarm()
                .into_disarm_result()
                .map_err(KeeperBasePortError::Terminal)
        }
    }

    #[derive(Debug)]
    struct SetupFailure {
        source: AnyError,
        terminal_stop: Result<DisarmReceipt, ZeroHoldTerminalError>,
    }

    impl fmt::Display for SetupFailure {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "bench setup failed before runtime ownership: {}; terminal_stop={:?}",
                self.source, self.terminal_stop
            )
        }
    }

    impl std::error::Error for SetupFailure {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            Some(self.source.as_ref())
        }
    }

    fn fail_setup(keeper: ZeroHoldKeeper, source: AnyError) -> AnyError {
        let terminal_stop = keeper.disarm().into_disarm_result();
        Box::new(SetupFailure {
            source,
            terminal_stop,
        })
    }

    fn build_ready_stopped_authority(
        inventory: &LoadedExpectedManifestV1,
        policy: &NanoAgentPolicyConfigV1,
        calibration_bundle: ExactCalibrationBundleSha256,
        initial_zero: FreshZeroEvidence,
        keeper: &mut ZeroHoldKeeper,
        clock_origin: Instant,
    ) -> Result<AgentAuthoritySupervisor, AnyError> {
        let hardware_manifest = supervisor_sha256(
            "parse inventory content identity for supervisor",
            *inventory.content_sha256().as_bytes(),
        )?;
        let calibration_bundle = supervisor_sha256(
            "parse calibration bundle identity for supervisor",
            *calibration_bundle.as_bytes(),
        )?;
        let first_result = initial_zero.host_result();
        let readiness = ReadinessBinding::new(
            ReadinessEpoch::try_new(1)
                .map_err(|source| at_stage("construct readiness epoch", source))?,
            first_result.controller_uid,
            first_result.boot_id,
            first_result.control_epoch,
            hardware_manifest,
            calibration_bundle,
        );
        let mut authority = AgentAuthoritySupervisor::new(
            policy.supervisor(),
            NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(0)),
        );
        let action = authority
            .begin_inventory(host_now(clock_origin)?)
            .map_err(|source| at_stage("enter supervisor inventory gate", source))?;
        require_supervisor_action(
            "enter supervisor inventory gate",
            action,
            SupervisorAction::InventoryRequired,
        )?;
        let action = authority
            .admit_readiness(readiness, host_now(clock_origin)?)
            .map_err(|source| at_stage("admit exact hardware readiness", source))?;
        require_supervisor_action(
            "admit exact hardware readiness",
            action,
            SupervisorAction::Disarmed,
        )?;
        let action = authority
            .arm(host_now(clock_origin)?)
            .map_err(|source| at_stage("arm supervisor zero gate", source))?;
        require_supervisor_action(
            "arm supervisor zero gate",
            action,
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::Arming,
            },
        )?;

        // This checkpoint is deliberately requested after the arm transition.
        // The keeper may have renewed automatically in the meantime; only this
        // exact newly sequenced response crosses the supervisor zero boundary.
        let armed_zero = keeper
            .force_fresh_zero()
            .map_err(|source| at_stage("apply post-arm base zero", source))?;
        let observed_at = fresh_zero_timestamp(armed_zero)
            .map_err(|source| at_stage("timestamp post-arm base zero", source))?;
        let action = authority
            .admit_applied_zero(
                armed_zero.host_result(),
                observed_at,
                host_now(clock_origin)?,
            )
            .map_err(|source| at_stage("admit post-arm base zero", source))?;
        require_supervisor_action(
            "admit post-arm base zero",
            action,
            SupervisorAction::ReadyStopped,
        )?;
        Ok(authority)
    }

    #[derive(Debug)]
    struct SerialHeadStartupEvidence {
        serial: HeadSerialEvidence,
        actor: VerifiedNaturalHoldEvidence,
    }

    #[derive(Debug)]
    struct SerialHeadShutdownEvidence {
        report: TorqueDisableReport,
        actor: HeadActorExit,
    }

    struct ActiveHeadActor {
        handle: HeadActorHandle,
        task: HeadActorTask,
        startup: VerifiedNaturalHoldEvidence,
    }

    #[derive(Default)]
    struct SerialHeadPort {
        active: Option<ActiveHeadActor>,
    }

    #[derive(Debug)]
    struct HeadStartupCleanup {
        actor: Result<HeadActorExit, JoinError>,
    }

    impl fmt::Display for HeadStartupCleanup {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "actor_join={:?}", self.actor)
        }
    }

    #[derive(Debug)]
    struct HeadShutdownAttempt {
        report: Result<TorqueDisableReport, HeadShutdownError>,
        actor: Result<HeadActorExit, JoinError>,
    }

    impl fmt::Display for HeadShutdownAttempt {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "torque_disable={:?}; actor_join={:?}",
                self.report, self.actor
            )
        }
    }

    #[derive(Debug)]
    enum SerialHeadPortError {
        AlreadyStarted,
        NotStarted,
        Start(HeadActorStartError),
        StartupReceipt {
            source: HeadStartupReceiptError,
            cleanup: HeadStartupCleanup,
        },
        StartupFault {
            source: HeadRuntimeError,
            cleanup: HeadStartupCleanup,
        },
        ShutdownFailed(HeadShutdownAttempt),
        ShutdownInvariant {
            reason: &'static str,
            evidence: SerialHeadShutdownEvidence,
        },
    }

    impl fmt::Display for SerialHeadPortError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("physical head port failed: ")?;
            match self {
                Self::AlreadyStarted => formatter.write_str("actor was already started"),
                Self::NotStarted => formatter.write_str("actor was not started"),
                Self::Start(source) => write!(formatter, "serial actor start: {source}"),
                Self::StartupReceipt { source, cleanup } => {
                    write!(formatter, "startup receipt: {source}; cleanup={cleanup}")
                }
                Self::StartupFault { source, cleanup } => {
                    write!(formatter, "startup fault: {source}; cleanup={cleanup}")
                }
                Self::ShutdownFailed(attempt) => {
                    write!(formatter, "shutdown attempt: {attempt}")
                }
                Self::ShutdownInvariant { reason, evidence } => {
                    write!(
                        formatter,
                        "shutdown invariant ({reason}); evidence={evidence:?}"
                    )
                }
            }
        }
    }

    impl std::error::Error for SerialHeadPortError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::Start(source) => Some(source),
                Self::StartupReceipt { source, .. } => Some(source),
                Self::StartupFault { source, .. } => Some(source),
                Self::AlreadyStarted
                | Self::NotStarted
                | Self::ShutdownFailed(_)
                | Self::ShutdownInvariant { .. } => None,
            }
        }
    }

    impl WheelsOffHeadPort for SerialHeadPort {
        type StartupEvidence = SerialHeadStartupEvidence;
        type ShutdownEvidence = SerialHeadShutdownEvidence;
        type Error = SerialHeadPortError;

        fn configured_pose(
            evidence: &Self::StartupEvidence,
        ) -> kiko_slam::navigation::ObservedPoseWithinConfiguredBounds {
            evidence.actor.configured_pose()
        }

        async fn start(
            &mut self,
            config: HeadRuntimeConfig,
            configured_pose_bounds: WheelsOffConfiguredPoseBounds,
            consent: PhysicalTorqueEnableConsent,
        ) -> Result<Self::StartupEvidence, Self::Error> {
            if self.active.is_some() {
                return Err(SerialHeadPortError::AlreadyStarted);
            }
            let (serial, handle, receipt, task) =
                start_serial_head_actor(config, configured_pose_bounds, consent)
                    .map_err(SerialHeadPortError::Start)?;
            let startup = match receipt.wait().await {
                Ok(Ok(evidence)) => evidence,
                Ok(Err(source)) => {
                    drop(handle);
                    let cleanup = HeadStartupCleanup {
                        actor: task.join().await,
                    };
                    return Err(SerialHeadPortError::StartupFault { source, cleanup });
                }
                Err(source) => {
                    drop(handle);
                    let cleanup = HeadStartupCleanup {
                        actor: task.join().await,
                    };
                    return Err(SerialHeadPortError::StartupReceipt { source, cleanup });
                }
            };
            self.active = Some(ActiveHeadActor {
                handle,
                task,
                startup: startup.clone(),
            });
            Ok(SerialHeadStartupEvidence {
                serial,
                actor: startup,
            })
        }

        async fn shutdown(&mut self) -> Result<Self::ShutdownEvidence, Self::Error> {
            let active = self.active.take().ok_or(SerialHeadPortError::NotStarted)?;
            let report = active.handle.shutdown().await;
            let actor = active.task.join().await;
            let (report, actor) = match (report, actor) {
                (Ok(report), Ok(actor)) => (report, actor),
                (report, actor) => {
                    return Err(SerialHeadPortError::ShutdownFailed(HeadShutdownAttempt {
                        report,
                        actor,
                    }));
                }
            };
            let evidence = SerialHeadShutdownEvidence { report, actor };
            if !evidence.report.all_writes_completed() {
                return Err(SerialHeadPortError::ShutdownInvariant {
                    reason: "one or more torque-disable writes did not complete",
                    evidence,
                });
            }
            if evidence.actor.torque_disable() != &evidence.report {
                return Err(SerialHeadPortError::ShutdownInvariant {
                    reason: "head handle and actor reported different torque-disable evidence",
                    evidence,
                });
            }
            if evidence.actor.startup() != &Ok(active.startup) {
                return Err(SerialHeadPortError::ShutdownInvariant {
                    reason: "head actor exit did not retain the admitted startup evidence",
                    evidence,
                });
            }
            if !matches!(
                evidence.actor.termination(),
                kiko_head_runtime::ActorTermination::RequestedShutdown
            ) {
                return Err(SerialHeadPortError::ShutdownInvariant {
                    reason: "head actor did not terminate by requested shutdown",
                    evidence,
                });
            }
            Ok(evidence)
        }
    }

    #[derive(Debug)]
    struct SerialEyeStartupEvidence {
        serial: EyeSerialEvidence,
        actor: EyeActorStartupEvidence,
    }

    #[derive(Debug)]
    struct SerialEyeShutdownEvidence {
        report: ReleaseReport,
        actor: EyeActorExit,
    }

    struct ActiveEyeActor {
        handle: EyeActorHandle,
        task: EyeActorTask,
        startup: EyeActorStartupEvidence,
    }

    #[derive(Default)]
    struct SerialEyePort {
        active: Option<ActiveEyeActor>,
    }

    #[derive(Debug)]
    struct EyeStartupCleanup {
        actor: Result<EyeActorExit, JoinError>,
    }

    impl fmt::Display for EyeStartupCleanup {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(formatter, "actor_join={:?}", self.actor)
        }
    }

    #[derive(Debug)]
    struct EyeShutdownAttempt {
        report: Result<ReleaseReport, EyeHandleRequestError>,
        actor: Result<EyeActorExit, JoinError>,
    }

    impl fmt::Display for EyeShutdownAttempt {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                formatter,
                "release={:?}; actor_join={:?}",
                self.report, self.actor
            )
        }
    }

    #[derive(Debug)]
    enum SerialEyePortError {
        AlreadyStarted,
        NotStarted,
        Start(EyeActorStartError),
        StartupReceipt {
            source: EyeStartupReceiptError,
            cleanup: Box<EyeStartupCleanup>,
        },
        StartupFault {
            source: Box<EyeRuntimeFault>,
            cleanup: Box<EyeStartupCleanup>,
        },
        Apply(EyeHandleRequestError),
        ShutdownFailed(Box<EyeShutdownAttempt>),
        ShutdownInvariant {
            reason: &'static str,
            evidence: Box<SerialEyeShutdownEvidence>,
        },
        ShutdownFallback {
            evidence: Box<SerialEyeShutdownEvidence>,
        },
    }

    impl fmt::Display for SerialEyePortError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("physical KEP2 eye port failed: ")?;
            match self {
                Self::AlreadyStarted => formatter.write_str("actor was already started"),
                Self::NotStarted => formatter.write_str("actor was not started"),
                Self::Start(source) => write!(formatter, "serial actor start: {source}"),
                Self::StartupReceipt { source, cleanup } => {
                    write!(formatter, "startup receipt: {source}; cleanup={cleanup}")
                }
                Self::StartupFault { source, cleanup } => {
                    write!(formatter, "startup fault: {source}; cleanup={cleanup}")
                }
                Self::Apply(source) => write!(formatter, "intent admission: {source}"),
                Self::ShutdownFailed(attempt) => {
                    write!(formatter, "shutdown attempt: {attempt}")
                }
                Self::ShutdownInvariant { reason, evidence } => {
                    write!(
                        formatter,
                        "shutdown invariant ({reason}); evidence={evidence:?}"
                    )
                }
                Self::ShutdownFallback { evidence } => write!(
                    formatter,
                    "firmware entered fallback during release; evidence={evidence:?}"
                ),
            }
        }
    }

    impl std::error::Error for SerialEyePortError {
        fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
            match self {
                Self::Start(source) => Some(source),
                Self::StartupReceipt { source, .. } => Some(source),
                Self::StartupFault { source, .. } => Some(source.as_ref()),
                Self::Apply(source) => Some(source),
                Self::AlreadyStarted
                | Self::NotStarted
                | Self::ShutdownFailed(_)
                | Self::ShutdownInvariant { .. }
                | Self::ShutdownFallback { .. } => None,
            }
        }
    }

    impl<C> WheelsOffEyePort<C> for SerialEyePort
    where
        C: EyeMonotonicClock,
    {
        type StartupEvidence = SerialEyeStartupEvidence;
        type ApplyEvidence = FirmwareAdmissionEvidence;
        type ShutdownEvidence = SerialEyeShutdownEvidence;
        type Error = SerialEyePortError;

        async fn start(
            &mut self,
            config: EyeRuntimeConfig,
            clock: C,
        ) -> Result<Self::StartupEvidence, Self::Error> {
            if self.active.is_some() {
                return Err(SerialEyePortError::AlreadyStarted);
            }
            let (serial, handle, receipt, task) =
                start_serial_eye_actor(config, clock).map_err(SerialEyePortError::Start)?;
            let startup = match receipt.wait().await {
                Ok(Ok(evidence)) => evidence,
                Ok(Err(source)) => {
                    drop(handle);
                    let cleanup = Box::new(EyeStartupCleanup {
                        actor: task.join().await,
                    });
                    return Err(SerialEyePortError::StartupFault {
                        source: Box::new(source),
                        cleanup,
                    });
                }
                Err(source) => {
                    drop(handle);
                    let cleanup = Box::new(EyeStartupCleanup {
                        actor: task.join().await,
                    });
                    return Err(SerialEyePortError::StartupReceipt { source, cleanup });
                }
            };
            self.active = Some(ActiveEyeActor {
                handle,
                task,
                startup: startup.clone(),
            });
            Ok(SerialEyeStartupEvidence {
                serial,
                actor: startup,
            })
        }

        async fn apply(
            &mut self,
            intent: kiko_expression_runtime::PreparedEyeIntent,
        ) -> Result<Self::ApplyEvidence, Self::Error> {
            self.active
                .as_mut()
                .ok_or(SerialEyePortError::NotStarted)?
                .handle
                .apply_intent(intent)
                .await
                .map_err(SerialEyePortError::Apply)
        }

        async fn shutdown(&mut self) -> Result<Self::ShutdownEvidence, Self::Error> {
            let active = self.active.take().ok_or(SerialEyePortError::NotStarted)?;
            let report = active.handle.shutdown().await;
            let actor = active.task.join().await;
            let (report, actor) = match (report, actor) {
                (Ok(report), Ok(actor)) => (report, actor),
                (report, actor) => {
                    return Err(SerialEyePortError::ShutdownFailed(Box::new(
                        EyeShutdownAttempt { report, actor },
                    )));
                }
            };
            let evidence = SerialEyeShutdownEvidence { report, actor };
            if evidence.actor.startup() != &Ok(active.startup) {
                return Err(SerialEyePortError::ShutdownInvariant {
                    reason: "eye actor exit did not retain the admitted startup evidence",
                    evidence: Box::new(evidence),
                });
            }
            if evidence.actor.release() != Some(&evidence.report) {
                return Err(SerialEyePortError::ShutdownInvariant {
                    reason: "eye handle and actor reported different release evidence",
                    evidence: Box::new(evidence),
                });
            }
            if !matches!(
                evidence.actor.termination(),
                kiko_eye_runtime::ActorTermination::RequestedShutdown
            ) {
                return Err(SerialEyePortError::ShutdownInvariant {
                    reason: "eye actor did not terminate by requested shutdown",
                    evidence: Box::new(evidence),
                });
            }
            if matches!(evidence.report, ReleaseReport::Fallback(_)) {
                return Err(SerialEyePortError::ShutdownFallback {
                    evidence: Box::new(evidence),
                });
            }
            Ok(evidence)
        }
    }

    fn report_error(stage: &'static str, value: &impl fmt::Debug) -> AnyError {
        at_stage(
            stage,
            OwnedEvidenceError {
                detail: format!("{value:#?}"),
            },
        )
    }

    fn digest_hex(bytes: &[u8; 32]) -> String {
        use std::fmt::Write as _;

        let mut output = String::with_capacity(64);
        for byte in bytes {
            write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
        }
        output
    }

    async fn run(cli: Cli) -> Result<(), AnyError> {
        if !(cli.wheels_removed && cli.head_path_clear && cli.power_cut_reachable) {
            return Err(at_stage(
                "operator safety gate",
                OwnedEvidenceError {
                    detail: "all three physical attestations are required".to_owned(),
                },
            ));
        }
        eprintln!("operator_attested_wheels_removed=true observed_by_software=false");
        eprintln!("operator_attested_head_path_clear=true observed_by_software=false");
        eprintln!("operator_attested_power_cut_reachable=true observed_by_software=false");

        let mut signals = SignalMonitor::start()
            .map_err(|source| at_stage("install SIGINT/SIGTERM monitor", source))?;
        let prepared = prepare(&cli)?;
        eprintln!(
            "bench_config_sha256={}",
            digest_hex(prepared.bench_config_sha256.as_bytes())
        );
        eprintln!(
            "agent_policy_sha256={}",
            digest_hex(prepared.agent_policy_sha256.as_bytes())
        );
        eprintln!(
            "zero_policy_sha256={}",
            digest_hex(prepared.zero_policy_sha256.as_bytes())
        );
        eprintln!(
            "controller_server_sha256={}",
            digest_hex(prepared.controller_server_sha256.as_bytes())
        );
        eprintln!(
            "inventory_content_sha256={}",
            digest_hex(prepared.inventory.content_sha256().as_bytes())
        );
        eprintln!(
            "calibration_bundle_sha256={} artifact_count={} all_artifacts_match={}",
            digest_hex(prepared.calibration_bundle.as_bytes()),
            prepared.artifact_hashes.len(),
            prepared.artifact_hashes.all_content_matches_manifest()
        );
        match oak_sys::depthai_build_metadata() {
            Ok(metadata) => eprintln!(
                "oak_linked_sdk_version={:?} oak_linked_sdk_commit={:?} oak_embedded_device_artifact={:?} oak_embedded_bootloader_artifact={:?}",
                metadata.sdk_version(),
                metadata.sdk_commit(),
                metadata.embedded_device_artifact_version(),
                metadata.embedded_bootloader_artifact_version(),
            ),
            Err(source) => eprintln!("oak_linked_build_metadata_unavailable={source}"),
        }
        eprintln!(
            "oak_provenance_binding=unverified reason=linked_build_fields_are_not_the_same_typed_semantics_as_manifest_provenance_strings"
        );

        let stream_epoch = fresh_stream_epoch()?;
        let clock_origin = Instant::now();
        let PreparedBench {
            inventory,
            policy,
            zero_client,
            bench,
            recording,
            maximum_run,
            calibration_bundle,
            ..
        } = prepared;
        if let Some(signal) = signals.try_signal() {
            eprintln!(
                "bench_cancelled_before_hardware=true checkpoint=before_zero_keeper signal={signal:?}"
            );
            signals.stop();
            return Ok(());
        }
        let (mut keeper, initial_zero) = ZeroHoldKeeper::start(zero_client, clock_origin)
            .map_err(|source| at_stage("start proactive zero-hold keeper", source))?;
        if let Some(signal) = signals.try_signal() {
            let terminal_stop = keeper.disarm().into_disarm_result();
            eprintln!(
                "bench_cancelled_before_accessories=true checkpoint=after_zero_keeper signal={signal:?} terminal_stop={terminal_stop:?}"
            );
            signals.stop();
            return terminal_stop.map(|_| ()).map_err(|source| {
                at_stage("disarm zero keeper after startup cancellation", source)
            });
        }
        let authority = match build_ready_stopped_authority(
            &inventory,
            &policy,
            calibration_bundle,
            initial_zero,
            &mut keeper,
            clock_origin,
        ) {
            Ok(authority) => authority,
            Err(source) => return Err(fail_setup(keeper, source)),
        };
        let plan_now = match host_now(clock_origin) {
            Ok(now) => now,
            Err(source) => {
                return Err(fail_setup(
                    keeper,
                    at_stage("timestamp wheels-off plan admission", source),
                ));
            }
        };
        let plan = match WheelsOffBenchPlan::admit(&inventory, policy, authority, plan_now, bench) {
            Ok(plan) => plan,
            Err(source) => {
                return Err(fail_setup(
                    keeper,
                    at_stage("admit manifest-bound wheels-off plan", source),
                ));
            }
        };
        let eye_clock = kiko_eye_runtime::TokioClock::new();
        let expression = wheels_off_rgb_expression_bridge(&plan, stream_epoch, eye_clock);
        let mut runtime = WheelsOffBenchRuntime::new(
            plan,
            KeeperBasePort::new(keeper),
            SerialHeadPort::default(),
            SerialEyePort::default(),
            NativeWheelsOffOakPort::default(),
            RerunWheelsOffTelemetry::new(recording),
            expression,
        );

        let startup = match runtime.start(&mut signals).await {
            Ok(evidence) => evidence,
            Err(failure) => {
                signals.stop();
                return Err(report_error("wheels-off startup and cleanup", &failure));
            }
        };
        let observed_ticks = startup
            .configured_head_pose
            .observed_pose()
            .positions()
            .map(PositionTicks::get);
        eprintln!(
            "bench_started=true opened_mxid={} head_observed_ticks={observed_ticks:?} base_zero_sequences={}/{}/{} expression_rgb_sequence={}",
            startup.connection.opened_mxid(),
            startup.before_oak.confirmed().sequence().get(),
            startup.before_head.confirmed().sequence().get(),
            startup.before_eye.confirmed().sequence().get(),
            startup.expression_rgb.capture_sequence(),
        );
        eprintln!(
            "head_serial_evidence={:?} head_hold_evidence={:?}",
            startup.head.serial, startup.head.actor
        );
        eprintln!(
            "eye_serial_evidence={:?} eye_startup_evidence={:?}",
            startup.eye.serial, startup.eye.actor
        );
        eprintln!(
            "head_claim=startup_only_encoder_ticks_within_operator_configured_windows; no_continuous_head_pose_or_torque_monitoring; torque_disable_writes_are_host_write_completion_evidence_not_servo_register_ack"
        );
        eprintln!(
            "eye_claim=firmware_admitted_rgb_derived_intent_not_photon_or_human_detection_evidence"
        );

        let run_started = Instant::now();
        let mut stop_cause = None;
        let mut cycle_failure = None;
        let mut rgb_cycles = 0_u64;
        loop {
            if let Some(signal) = signals.try_signal() {
                stop_cause = Some(BenchStopCause::Signal(signal));
                break;
            }
            if run_started.elapsed() >= maximum_run {
                stop_cause = Some(BenchStopCause::TimeBudgetElapsed);
                break;
            }
            match runtime.process_next_rgb().await {
                Ok(Some(_)) => {
                    rgb_cycles = rgb_cycles.saturating_add(1);
                }
                Ok(None) => {}
                Err(source) => {
                    cycle_failure = Some(format!("{source:#?}"));
                    break;
                }
            }
        }

        let cleanup = runtime.shutdown().await;
        signals.stop();
        let cleanup_ok = cleanup.base.is_ok()
            && cleanup.eye.as_ref().is_some_and(Result::is_ok)
            && cleanup.head.as_ref().is_some_and(Result::is_ok)
            && cleanup.oak.as_ref().is_some_and(Result::is_ok)
            && cleanup.telemetry.is_ok()
            && cleanup.base_disarm.is_ok();
        eprintln!(
            "bench_stopped=true cause={stop_cause:?} rgb_expression_cycles={rgb_cycles} cleanup_ok={cleanup_ok}"
        );
        if let Some(detail) = cycle_failure {
            eprintln!("cycle_failure={detail}");
            eprintln!("cleanup={cleanup:#?}");
            if !cleanup_ok {
                return Err(at_stage(
                    "RGB bench cycle and cleanup",
                    OwnedEvidenceError {
                        detail: format!("cycle_failure={detail}; cleanup={cleanup:#?}"),
                    },
                ));
            }
            return Err(at_stage("RGB bench cycle", OwnedEvidenceError { detail }));
        }
        if !cleanup_ok {
            return Err(report_error("wheels-off cleanup", &cleanup));
        }
        Ok(())
    }

    #[tokio::main(flavor = "multi_thread", worker_threads = 4)]
    pub async fn main() {
        let cli = Cli::parse();
        if let Err(source) = run(cli).await {
            eprintln!("kiko_nano_wheels_off_bench_failed: {source}");
            let mut cause = source.source();
            while let Some(next) = cause {
                eprintln!("caused_by: {next}");
                cause = next.source();
            }
            std::process::exit(1);
        }
    }
}

#[cfg(all(feature = "nano-bench", unix))]
fn main() {
    enabled::main();
}
