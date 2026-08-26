//! Single-owner Nano head/eye worker fed by the live OAK owner's RGB frames.
//!
//! The worker never opens or owns an OAK device. Its only camera boundary is a
//! capacity-one, replace-latest queue of already-owned [`oak_sys::ImageFrame`]
//! values. The production Nano graph adds a second capacity-one handoff and a
//! named OS thread which constructs and retains the intentionally `!Send`
//! OpenCV face detector. A separate thread owns one current-thread Tokio
//! runtime, the manifest-bound return-to-natural head actor, the manifest-bound
//! KEP2 eye actor, and one [`RgbExpressionBridge`], so detector work cannot
//! block actor servicing.
//!
//! A terminal accessory fault is a base-stop signal, not permission to alter
//! the head. After publishing the first terminal fault, the worker stops
//! accepting RGB work, latches every base-facing stop boundary, then attempts
//! one bounded character-owned fault-eye transition. Its exact completion or
//! secondary failure is retained without replacing the first fault. The
//! worker retains the natural-hold actor while continuing bounded health
//! checks. Only [`NanoAccessoryWorker::shutdown`] asks the eye actor to release
//! control and the head actor to release serial ownership.
//! Production head release performs no torque-switch write, preserving the
//! last admitted hold without claiming the physical torque state. Dropping the
//! worker is deliberately not an implicit physical shutdown.

use std::fmt;
#[cfg(feature = "nano-agent")]
use std::num::NonZeroU8;
use std::num::NonZeroU64;
#[cfg(feature = "nano-agent")]
use std::panic::{AssertUnwindSafe, catch_unwind};
#[cfg(feature = "nano-agent")]
use std::sync::Condvar;
#[cfg(feature = "nano-agent")]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
#[cfg(any(
    feature = "nano-agent",
    feature = "nano-wheels-off-qualification",
    feature = "nano-base-commissioning",
    test
))]
use std::thread;
use std::thread::JoinHandle;
#[cfg(feature = "nano-agent")]
use std::time::SystemTime;
use std::time::{Duration, Instant};

#[cfg(feature = "nano-agent")]
use kiko_device_inventory::{
    ArtifactRelativePath, DeploymentAssetContentSha256, LoadedDeploymentAsset,
};
#[cfg(feature = "nano-agent")]
use kiko_expression_core::UnitAmount as CoreUnitAmount;
#[cfg(feature = "nano-agent")]
use kiko_expression_core::{Deadline, NonZeroDuration};
use kiko_expression_core::{MonotonicTimestamp, StreamEpochId};
use kiko_expression_runtime::{
    CharacterBaseMotionState, CharacterHeadOverlay, CharacterLifecycleState, CharacterPetEpisode,
    CharacterPetState, CharacterThermalState, PreparedCharacterFrame, PreparedEyeIntent,
    TrackingEyeGaze, TrackingEyeGazeReplacementError,
};
#[cfg(feature = "nano-agent")]
use kiko_expression_runtime::{
    CharacterPetEpisodeError, CommandedHeadGazeEstimateError, FaceTrackingConfig, HeadGazeResidual,
    HeadGazeResidualError, HeadRelativeGaze, MAX_FACE_DETECTIONS,
};
use kiko_eye_runtime::{
    ActorExit as EyeActorExit, ActorTermination as EyeActorTermination, ClockError, EyeActorHandle,
    EyeActorStartError, EyeActorTask, EyeRuntimeConfig, EyeRuntimeFault,
    HandleRequestError as EyeHandleRequestError, MonotonicClock, OsEyeSessionMaterialError,
    OsEyeSessionMaterialGenerator, ReleaseReport, SerialConfigurationEvidence as EyeSerialEvidence,
    StartupEvidence as EyeStartupEvidence, StartupReceiptError as EyeStartupReceiptError,
    StaticEyeRuntimeConfig,
};
#[cfg(feature = "nano-agent")]
use kiko_head_protocol::ExactHeadTargetPose;
#[cfg(feature = "nano-agent")]
use kiko_head_runtime::compliant_hold::{CompliantHoldDisposition, CompliantPetEpisodeSummary};
#[cfg(feature = "nano-agent")]
use kiko_head_runtime::gaze_control::{
    HeadGazeProposal, HeadGazeProposalId, HeadGazeProposalIdError,
};
use kiko_head_runtime::{
    ActorTermination as HeadActorTermination, HeadActorStartError, HeadCommandError,
    HeadHealthRequestError, HeadHoldTarget, HeadReturnError, HeadRuntimeError,
    HoldPreservingOwnershipReleaseEvidence, PhysicalHeadMotionConsent, PhysicalTorqueEnableConsent,
    ProductionTensionPreservingTakeoverConsent, ReturnToTargetConfig,
    SerialConfigurationEvidence as HeadSerialEvidence, ShutdownError as HeadShutdownError,
    StartupReceiptError as HeadStartupReceiptError,
    TensionPreservingHeadActorExit as HeadActorExit,
    TensionPreservingHeadActorTask as HeadActorTask,
    TensionPreservingHeadReturnActorHandle as HeadReturnActorHandle, VerifiedHeadHealthEvidence,
    VerifiedHeadReturnEvidence, VerifiedNaturalHoldEvidence,
};
#[cfg(feature = "nano-agent")]
use kiko_head_runtime::{
    HeadGazeBaseInterlockError, HeadGazeBaseMotionPhase, HeadGazeBaseZeroExclusiveLeaseIssuer,
    HeadGazeProposalCommandError, HeadGazeServiceError, HeadGazeServiceOutcome,
    HeadThermalDerateState, TensionPreservingHeadGazeActorHandle as HeadGazeActorHandle,
};
use oak_sys::ImageFrame;
#[cfg(feature = "nano-agent")]
use oak_sys::{OpenCvHaarFaceDetectorConfig, OpenCvHaarFaceDetectorConfigError};
use tokio::task::JoinError;

use super::expression_bridge::IngressObservedRgbFrame;
#[cfg(feature = "nano-agent")]
use super::expression_bridge::{ParsedIngressRgbFrame, parse_ingress_observed_oak_frame};
#[cfg(feature = "nano-agent")]
use super::nano_face_perception::{
    NanoFacePerception, NanoFacePerceptionError, NanoFacePerceptionLoadError,
    NanoFacePerceptionOutput, OakFaceFrameProvenance,
};
#[cfg(feature = "nano-agent")]
use super::{
    EvidenceBoundPhysicalHeadGazePolicy, EvidenceBoundPhysicalHeadGazePolicyError,
    HeadGazeFaceProposalAdapter, HeadGazeFaceProposalAdapterError, HeadGazeFaceProposalError,
    HeadGazeFaceProposalOutcome, HeadGazeFaceProposalWithheld, HeadGazePolicyLifecycleClaim,
    HeadGazePolicyV1, PhysicalCharacterHeadEvaluationError, PhysicalCharacterHeadOutcome,
    RgbFacePinholeProjection,
};
use super::{
    ManifestBoundNanoAgentPolicyConfigV3, NanoRgbExpressionConfig, RgbExpressionBridge,
    RgbExpressionBridgeError,
};
#[cfg(all(feature = "nano-agent", test))]
use super::{
    NanoPetEpisodeEvidence, NanoPetEvidenceJournalExit, NanoPetEvidenceJournalJoinEvidence,
};
#[cfg(feature = "nano-agent")]
use super::{
    NanoPetEvidenceJournal, NanoPetEvidenceJournalConfig, NanoPetEvidenceJournalReadyEvidence,
    NanoPetEvidenceJournalRuntimeError, NanoPetEvidenceJournalShutdownEvidence,
    NanoPetEvidenceJournalStartError,
};
#[cfg(feature = "nano-agent")]
use crate::{
    ChannelCapacity, ChannelStats, ChannelStatsHandle, DropPolicy, DropReceiver, DropSender,
    IntrinsicsError, PinholeIntrinsics, SendOutcome, bounded_channel,
};

/// A health cadence long enough to avoid a zero-duration busy loop and short
/// enough for the base owner to receive a bounded-latency health result.
pub const MAX_NANO_ACCESSORY_HEALTH_PERIOD: Duration = Duration::from_secs(5);

/// Bounded causal handoff from a locally queued first terminal value to the
/// public terminal-fault monitor. Expiry is itself reported as a typed monitor
/// timeout; it is not evidence that the original fault disappeared.
pub const NANO_ACCESSORY_TERMINAL_PUBLICATION_TIMEOUT: Duration = Duration::from_secs(2);
/// Bounded coordinated shutdown/join budget for the non-actuating evidence
/// writer. Expiry is retained as detachment evidence, never claimed as a join.
#[cfg(feature = "nano-agent")]
pub const NANO_PET_EVIDENCE_JOURNAL_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(2);

/// Maximum wait for the hardware-free native detector constructor to report
/// readiness. Expiry requests shutdown and returns typed cleanup evidence; it
/// does not claim that a stuck native call was cancelled.
#[cfg(feature = "nano-agent")]
pub const NANO_FACE_PERCEPTION_STARTUP_TIMEOUT: Duration = Duration::from_secs(10);
/// Maximum wait to join the hardware-free detector thread after shutdown.
/// Expiry detaches that thread and reports uncertainty instead of blocking the
/// robot lifecycle indefinitely.
#[cfg(feature = "nano-agent")]
pub const NANO_FACE_PERCEPTION_JOIN_TIMEOUT: Duration = Duration::from_secs(2);
#[cfg(feature = "nano-agent")]
const NANO_FACE_PERCEPTION_JOIN_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// One monotonic epoch shared by RGB observation, face tracking, expression,
/// eye transport, and physical head-gaze control.
///
/// Keeping this as a local type lets it implement both runtime clock traits
/// without translating timestamps between independently sampled origins.
#[derive(Clone, Debug)]
struct NanoAccessoryClock {
    origin: tokio::time::Instant,
}

impl NanoAccessoryClock {
    fn new() -> Self {
        Self {
            origin: tokio::time::Instant::now(),
        }
    }
}

impl MonotonicClock for NanoAccessoryClock {
    fn now(&self) -> Result<MonotonicTimestamp, ClockError> {
        let elapsed_nanoseconds = self.origin.elapsed().as_nanos();
        let elapsed_nanoseconds = u64::try_from(elapsed_nanoseconds).map_err(|_| {
            ClockError::ElapsedNanosecondsOutOfRange {
                elapsed_nanoseconds,
            }
        })?;
        Ok(MonotonicTimestamp::from_nanos_since_epoch(
            elapsed_nanoseconds,
        ))
    }
}

impl kiko_head_runtime::MonotonicClock for NanoAccessoryClock {
    fn now(&self) -> kiko_head_runtime::MonotonicTime {
        kiko_head_runtime::MonotonicTime::from_duration_since_origin(self.origin.elapsed())
    }
}

#[cfg(feature = "nano-agent")]
const NANO_FACE_HAAR_SCALE_FACTOR: f64 = 1.15;
#[cfg(feature = "nano-agent")]
const NANO_FACE_FRONTAL_MINIMUM_NEIGHBORS: u32 = 6;
#[cfg(feature = "nano-agent")]
const NANO_FACE_PROFILE_MINIMUM_NEIGHBORS: u32 = 4;
#[cfg(feature = "nano-agent")]
const NANO_FACE_MINIMUM_WIDTH_PX: u32 = 30;
#[cfg(feature = "nano-agent")]
const NANO_FACE_MINIMUM_HEIGHT_PX: u32 = 30;

#[cfg(feature = "nano-agent")]
fn canonical_nano_face_detector_config()
-> Result<OpenCvHaarFaceDetectorConfig, NanoFacePerceptionConfigError> {
    let maximum_retained_detections = u32::try_from(MAX_FACE_DETECTIONS).map_err(|_| {
        NanoFacePerceptionConfigError::TrackerCapacityExceedsU32 {
            capacity: MAX_FACE_DETECTIONS,
        }
    })?;
    OpenCvHaarFaceDetectorConfig::try_new(
        NANO_FACE_HAAR_SCALE_FACTOR,
        NANO_FACE_FRONTAL_MINIMUM_NEIGHBORS,
        NANO_FACE_PROFILE_MINIMUM_NEIGHBORS,
        NANO_FACE_MINIMUM_WIDTH_PX,
        NANO_FACE_MINIMUM_HEIGHT_PX,
        maximum_retained_detections,
    )
    .map_err(NanoFacePerceptionConfigError::Detector)
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Debug, PartialEq)]
pub enum NanoFacePerceptionConfigError {
    TrackerCapacityExceedsU32 { capacity: usize },
    Detector(OpenCvHaarFaceDetectorConfigError),
}

#[cfg(feature = "nano-agent")]
impl fmt::Display for NanoFacePerceptionConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "canonical Nano face-perception configuration is invalid: {self:?}"
        )
    }
}

#[cfg(feature = "nano-agent")]
impl std::error::Error for NanoFacePerceptionConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Detector(source) => Some(source),
            Self::TrackerCapacityExceedsU32 { .. } => None,
        }
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Debug)]
pub struct NanoFaceCascadeAssetEvidence {
    relative_path: ArtifactRelativePath,
    content_sha256: DeploymentAssetContentSha256,
    byte_len: usize,
}

#[cfg(feature = "nano-agent")]
impl NanoFaceCascadeAssetEvidence {
    pub const fn relative_path(&self) -> &ArtifactRelativePath {
        &self.relative_path
    }

    pub const fn content_sha256(&self) -> DeploymentAssetContentSha256 {
        self.content_sha256
    }

    pub const fn byte_len(&self) -> usize {
        self.byte_len
    }
}

/// Exact retained V3 cascade assets moved into the dedicated detector thread.
///
/// The assets remain `LoadedDeploymentAsset` values until the detector thread
/// has captured their identity evidence and consumed their byte vectors.
/// Rust neither duplicates a vector nor reopens a pathname. The native OpenCV
/// boundary then makes one required owned `std::string` copy of each slice so
/// `FileStorage` can parse it during startup.
#[cfg(feature = "nano-agent")]
pub(super) struct NanoFacePerceptionAssets {
    frontal_face_cascade: LoadedDeploymentAsset,
    profile_face_cascade: LoadedDeploymentAsset,
}

#[cfg(feature = "nano-agent")]
impl NanoFacePerceptionAssets {
    fn from_loaded_assets(
        frontal_face_cascade: LoadedDeploymentAsset,
        profile_face_cascade: LoadedDeploymentAsset,
    ) -> Self {
        Self {
            frontal_face_cascade,
            profile_face_cascade,
        }
    }

    pub(super) fn from_v3_loaded_assets(
        frontal_face_cascade: LoadedDeploymentAsset,
        profile_face_cascade: LoadedDeploymentAsset,
    ) -> Self {
        Self::from_loaded_assets(frontal_face_cascade, profile_face_cascade)
    }

    fn evidence(&self) -> NanoFacePerceptionAssetEvidence {
        NanoFacePerceptionAssetEvidence {
            frontal_face_cascade: face_asset_evidence(&self.frontal_face_cascade),
            profile_face_cascade: face_asset_evidence(&self.profile_face_cascade),
        }
    }

    fn into_bytes(self) -> (Vec<u8>, Vec<u8>) {
        (
            self.frontal_face_cascade.into_bytes(),
            self.profile_face_cascade.into_bytes(),
        )
    }
}

#[cfg(feature = "nano-agent")]
fn face_asset_evidence(asset: &LoadedDeploymentAsset) -> NanoFaceCascadeAssetEvidence {
    NanoFaceCascadeAssetEvidence {
        relative_path: asset.relative_path().clone(),
        content_sha256: asset.content_sha256(),
        byte_len: asset.byte_len(),
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Debug)]
pub struct NanoFacePerceptionAssetEvidence {
    frontal_face_cascade: NanoFaceCascadeAssetEvidence,
    profile_face_cascade: NanoFaceCascadeAssetEvidence,
}

#[cfg(feature = "nano-agent")]
impl NanoFacePerceptionAssetEvidence {
    pub const fn frontal_face_cascade(&self) -> &NanoFaceCascadeAssetEvidence {
        &self.frontal_face_cascade
    }

    pub const fn profile_face_cascade(&self) -> &NanoFaceCascadeAssetEvidence {
        &self.profile_face_cascade
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Debug)]
pub struct NanoFacePerceptionReadyEvidence {
    assets: NanoFacePerceptionAssetEvidence,
    detector_config: OpenCvHaarFaceDetectorConfig,
    tracking_config: FaceTrackingConfig,
}

#[cfg(feature = "nano-agent")]
impl NanoFacePerceptionReadyEvidence {
    pub const fn assets(&self) -> &NanoFacePerceptionAssetEvidence {
        &self.assets
    }

    pub const fn detector_config(&self) -> &OpenCvHaarFaceDetectorConfig {
        &self.detector_config
    }

    pub const fn tracking_config(&self) -> FaceTrackingConfig {
        self.tracking_config
    }
}

/// Parsed, non-zero periodic head-health interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoAccessoryHealthPeriod(NonZeroU64);

impl NanoAccessoryHealthPeriod {
    pub fn try_from_duration(value: Duration) -> Result<Self, NanoAccessoryHealthPeriodError> {
        let nanoseconds = value.as_nanos();
        if nanoseconds == 0 {
            return Err(NanoAccessoryHealthPeriodError::Zero);
        }
        let nanoseconds = u64::try_from(nanoseconds)
            .map_err(|_| NanoAccessoryHealthPeriodError::NanosecondsOutOfRange { nanoseconds })?;
        if value > MAX_NANO_ACCESSORY_HEALTH_PERIOD {
            return Err(NanoAccessoryHealthPeriodError::AboveMaximum {
                actual: value,
                maximum: MAX_NANO_ACCESSORY_HEALTH_PERIOD,
            });
        }
        Ok(Self(
            NonZeroU64::new(nanoseconds).expect("non-zero duration was checked"),
        ))
    }

    pub const fn get(self) -> Duration {
        Duration::from_nanos(self.0.get())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAccessoryHealthPeriodError {
    Zero,
    NanosecondsOutOfRange { nanoseconds: u128 },
    AboveMaximum { actual: Duration, maximum: Duration },
}

impl fmt::Display for NanoAccessoryHealthPeriodError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Nano accessory head-health period: {self:?}"
        )
    }
}

impl std::error::Error for NanoAccessoryHealthPeriodError {}

/// Non-forgeable worker input derived only from a manifest-bound agent policy.
pub struct NanoAccessoryWorkerConfig {
    head_return: ReturnToTargetConfig,
    head_torque_consent: PhysicalTorqueEnableConsent,
    head_motion_consent: PhysicalHeadMotionConsent,
    head_takeover_consent: ProductionTensionPreservingTakeoverConsent,
    required_hold_target: HeadHoldTarget,
    eye: StaticEyeRuntimeConfig,
    rgb_expression: NanoRgbExpressionConfig,
    stream_epoch: StreamEpochId,
    health_period: NanoAccessoryHealthPeriod,
    #[cfg(feature = "nano-agent")]
    head_gaze_diagnostics: Option<HeadGazeFaceProposalAdapter>,
    #[cfg(feature = "nano-agent")]
    physical_head_gaze: Option<NanoPhysicalHeadGazeConfig>,
    #[cfg(feature = "nano-agent")]
    pet_evidence_journal: Option<NanoPetEvidenceJournalConfig>,
}

impl NanoAccessoryWorkerConfig {
    /// Consume no weak configuration. Both actor configurations are cloned
    /// only from a policy whose accessory identities already matched the exact
    /// parsed expected-device manifest.
    pub fn from_manifest_bound_policy(
        policy: &ManifestBoundNanoAgentPolicyConfigV3,
        stream_epoch: StreamEpochId,
        health_period: NanoAccessoryHealthPeriod,
    ) -> Result<Self, NanoAccessoryWorkerConfigError> {
        let head = policy
            .head()
            .return_to_natural_and_hold_continuously()
            .ok_or(NanoAccessoryWorkerConfigError::HeadDisabled)?
            .clone();
        let required_hold_target = head.required_hold_target();
        let (head_return, head_torque_consent, head_motion_consent) = head.into_parts();
        let eye = policy
            .eye()
            .static_runtime()
            .ok_or(NanoAccessoryWorkerConfigError::EyeDisabled)?
            .clone();
        let rgb_expression = policy
            .rgb_expression()
            .scene_motion()
            .ok_or(NanoAccessoryWorkerConfigError::RgbExpressionDisabled)?;
        Ok(Self {
            head_return,
            head_torque_consent,
            head_motion_consent,
            head_takeover_consent:
                ProductionTensionPreservingTakeoverConsent::explicitly_granted_for_manifest_bound_owner(),
            required_hold_target,
            eye,
            rgb_expression,
            stream_epoch,
            health_period,
            #[cfg(feature = "nano-agent")]
            head_gaze_diagnostics: None,
            #[cfg(feature = "nano-agent")]
            physical_head_gaze: None,
            #[cfg(feature = "nano-agent")]
            pet_evidence_journal: None,
        })
    }

    /// Attach the production state-root journal. Commissioning constructors
    /// intentionally remain journal-disabled unless their caller does this
    /// explicitly.
    #[cfg(feature = "nano-agent")]
    pub fn with_pet_evidence_journal(mut self, journal: NanoPetEvidenceJournalConfig) -> Self {
        self.pet_evidence_journal = Some(journal);
        self
    }

    /// Attach a parsed proposal-only policy to the non-actuating face
    /// diagnostic lane.
    ///
    /// This does not configure the head actor for gaze, issue a base-zero
    /// lease, or expose a serial head command. A physical-review lifecycle is
    /// rejected here because this lane has no retained physical-review
    /// evidence to cross-bind.
    #[cfg(feature = "nano-agent")]
    pub fn with_proposal_only_head_gaze_diagnostics(
        mut self,
        policy: HeadGazePolicyV1,
    ) -> Result<Self, NanoHeadGazeDiagnosticConfigError> {
        if self.head_gaze_diagnostics.is_some() || self.physical_head_gaze.is_some() {
            return Err(NanoHeadGazeDiagnosticConfigError::AlreadyConfigured);
        }
        if !matches!(
            policy.lifecycle(),
            HeadGazePolicyLifecycleClaim::ProposalOnly(_)
        ) {
            return Err(NanoHeadGazeDiagnosticConfigError::NotProposalOnly);
        }
        self.head_gaze_diagnostics = Some(
            HeadGazeFaceProposalAdapter::try_new(policy)
                .map_err(NanoHeadGazeDiagnosticConfigError::Adapter)?,
        );
        Ok(self)
    }

    /// Attach one evidence-bound physical gaze policy to the canonical
    /// single-owner head runtime.
    ///
    /// Admission reuses this config's exact manifest-bound return target and
    /// transport timeout. The resulting runtime still cannot move until the
    /// live STM32 owner binds its confirmed-zero lease issuer after startup.
    #[cfg(feature = "nano-agent")]
    pub fn with_evidence_bound_physical_head_gaze(
        mut self,
        policy: HeadGazePolicyV1,
        review_evidence: &LoadedDeploymentAsset,
    ) -> Result<Self, NanoPhysicalHeadGazeConfigError> {
        if self.head_gaze_diagnostics.is_some() || self.physical_head_gaze.is_some() {
            return Err(NanoPhysicalHeadGazeConfigError::AlreadyConfigured);
        }
        if policy.character_mapping().is_none() {
            return Err(NanoPhysicalHeadGazeConfigError::CharacterMappingRequired);
        }
        if policy.mapping().dynamic_pitch_recruitment().is_none() {
            return Err(NanoPhysicalHeadGazeConfigError::DynamicPitchRecruitmentRequired);
        }
        if policy.turn_dip_posture().is_none() {
            return Err(NanoPhysicalHeadGazeConfigError::TurnDipPostureRequired);
        }
        if policy.compliant_hold().is_none() {
            return Err(NanoPhysicalHeadGazeConfigError::CompliantHoldRequired);
        }
        if policy.thermal_derate().is_none() {
            return Err(NanoPhysicalHeadGazeConfigError::ThermalDerateRequired);
        }
        let policy =
            EvidenceBoundPhysicalHeadGazePolicy::admit(policy, review_evidence, &self.head_return)
                .map_err(NanoPhysicalHeadGazeConfigError::Admission)?;
        self.physical_head_gaze = Some(NanoPhysicalHeadGazeConfig {
            policy,
            lease_issuer: Arc::new(OnceLock::new()),
        });
        Ok(self)
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Debug)]
struct NanoPhysicalHeadGazeConfig {
    policy: EvidenceBoundPhysicalHeadGazePolicy,
    lease_issuer: Arc<OnceLock<HeadGazeBaseZeroExclusiveLeaseIssuer>>,
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NanoPhysicalHeadGazeConfigError {
    AlreadyConfigured,
    CharacterMappingRequired,
    DynamicPitchRecruitmentRequired,
    TurnDipPostureRequired,
    CompliantHoldRequired,
    ThermalDerateRequired,
    Admission(EvidenceBoundPhysicalHeadGazePolicyError),
}

#[cfg(feature = "nano-agent")]
impl fmt::Display for NanoPhysicalHeadGazeConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "physical Nano head gaze is not configured: {self:?}"
        )
    }
}

#[cfg(feature = "nano-agent")]
impl std::error::Error for NanoPhysicalHeadGazeConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Admission(source) => Some(source),
            Self::AlreadyConfigured
            | Self::CharacterMappingRequired
            | Self::DynamicPitchRecruitmentRequired
            | Self::TurnDipPostureRequired
            | Self::CompliantHoldRequired
            | Self::ThermalDerateRequired => None,
        }
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoHeadGazeDiagnosticConfigError {
    AlreadyConfigured,
    NotProposalOnly,
    Adapter(HeadGazeFaceProposalAdapterError),
}

#[cfg(feature = "nano-agent")]
impl fmt::Display for NanoHeadGazeDiagnosticConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "head-gaze proposal diagnostics are not configured: {self:?}"
        )
    }
}

#[cfg(feature = "nano-agent")]
impl std::error::Error for NanoHeadGazeDiagnosticConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Adapter(source) => Some(source),
            Self::AlreadyConfigured | Self::NotProposalOnly => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAccessoryWorkerConfigError {
    HeadDisabled,
    EyeDisabled,
    RgbExpressionDisabled,
}

impl fmt::Display for NanoAccessoryWorkerConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "manifest-bound Nano accessory policy is incomplete: {self:?}"
        )
    }
}

impl std::error::Error for NanoAccessoryWorkerConfigError {}

/// Exact successful eye actor startup evidence.
#[derive(Clone, Debug)]
pub struct NanoEyeReadyEvidence {
    serial: EyeSerialEvidence,
    actor: EyeStartupEvidence,
}

impl NanoEyeReadyEvidence {
    pub const fn serial(&self) -> &EyeSerialEvidence {
        &self.serial
    }

    pub const fn actor(&self) -> &EyeStartupEvidence {
        &self.actor
    }
}

/// Exact successful startup observation followed by a verified return to the
/// reviewed natural target.
#[derive(Clone, Debug)]
pub struct NanoHeadReadyEvidence {
    serial: HeadSerialEvidence,
    startup: VerifiedNaturalHoldEvidence,
    head_return: VerifiedHeadReturnEvidence,
    initial_health: VerifiedHeadHealthEvidence,
}

impl NanoHeadReadyEvidence {
    pub const fn serial(&self) -> &HeadSerialEvidence {
        &self.serial
    }

    pub const fn startup(&self) -> &VerifiedNaturalHoldEvidence {
        &self.startup
    }

    pub const fn head_return(&self) -> &VerifiedHeadReturnEvidence {
        &self.head_return
    }

    pub const fn initial_health(&self) -> &VerifiedHeadHealthEvidence {
        &self.initial_health
    }
}

#[derive(Clone, Debug)]
pub enum NanoAccessoryPerceptionReadyEvidence {
    /// Non-production compatibility mode used by feature graphs which do not
    /// include the canonical V3 face assets.
    SceneMotionOnly,
    #[cfg(feature = "nano-agent")]
    Face(NanoFacePerceptionReadyEvidence),
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NanoPetEvidenceLifecycleReadyEvidence {
    Disabled,
    Enabled(NanoPetEvidenceJournalReadyEvidence),
}

/// Readiness is emitted only after eye startup and the head's startup,
/// reviewed return, and immediate exact-target health check all succeeded.
/// In production, face-detector load evidence is also present because detector
/// construction completes before either actor is started.
#[derive(Clone, Debug)]
pub struct NanoAccessoryReadyEvidence {
    eye: NanoEyeReadyEvidence,
    head: NanoHeadReadyEvidence,
    perception: NanoAccessoryPerceptionReadyEvidence,
    #[cfg(feature = "nano-agent")]
    pet_evidence: NanoPetEvidenceLifecycleReadyEvidence,
    stream_epoch: StreamEpochId,
    health_period: NanoAccessoryHealthPeriod,
    rgb_frame_freshness: Duration,
}

impl NanoAccessoryReadyEvidence {
    pub const fn eye(&self) -> &NanoEyeReadyEvidence {
        &self.eye
    }

    pub const fn head(&self) -> &NanoHeadReadyEvidence {
        &self.head
    }

    pub const fn perception(&self) -> &NanoAccessoryPerceptionReadyEvidence {
        &self.perception
    }

    #[cfg(feature = "nano-agent")]
    pub const fn pet_evidence(&self) -> &NanoPetEvidenceLifecycleReadyEvidence {
        &self.pet_evidence
    }

    pub const fn stream_epoch(&self) -> StreamEpochId {
        self.stream_epoch
    }

    pub const fn health_period(&self) -> NanoAccessoryHealthPeriod {
        self.health_period
    }

    /// Maximum age of a receipt-backed RGB reaction before its health becomes
    /// degraded. This is the exact parsed source freshness policy, not the
    /// much slower periodic head-health cadence.
    pub const fn rgb_frame_freshness(&self) -> Duration {
        self.rgb_frame_freshness
    }
}

/// Result of one non-blocking RGB ingress queue-ownership attempt.
///
/// Success means either an ingress-owned frame entered the replace-latest data
/// slot or the first typed ingress failure entered the nonreplaceable terminal
/// slot. [`NanoAccessoryFrameStats`] distinguishes those cases. Semantic frame
/// acceptance happens only when the worker consumes the data slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAccessoryFrameSubmitOutcome {
    Enqueued,
    ReplacedOlderFrame,
    TerminalFaultPendingPublication,
    TerminalFaultLatched,
    IngressDisconnected,
    ChannelPoisoned,
}

/// Saturating capacity-one ingress counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NanoAccessoryFrameStats {
    pub enqueued: u64,
    pub replaced_older: u64,
    pub first_terminal_enqueued: u64,
    pub frames_discarded_for_terminal: u64,
    pub rejected_behind_pending_terminal: u64,
    pub processed_successfully: u64,
    pub rejected_after_fault: u64,
    pub rejected_disconnected: u64,
    pub channel_poisoned: u64,
}

#[derive(Debug)]
struct LatestFrameCounters {
    enqueued: AtomicU64,
    replaced_older: AtomicU64,
    first_terminal_enqueued: AtomicU64,
    frames_discarded_for_terminal: AtomicU64,
    rejected_behind_pending_terminal: AtomicU64,
    rejected_after_fault: AtomicU64,
    rejected_disconnected: AtomicU64,
    channel_poisoned: AtomicU64,
    receipts: Arc<LatestFrameReceiptCounters>,
}

/// The only accounting shared by raw ingress and the face-output lane.
///
/// A receipt is written after expression processing and eye acknowledgement,
/// so it describes successful handling of the originating raw RGB frame.
/// Queue replacement, terminal discard, rejection, and poison counters remain
/// lane-local and can never contaminate public ingress statistics.
#[derive(Debug)]
struct LatestFrameReceiptCounters {
    processed_successfully: AtomicU64,
    last_processed_successfully_at: Mutex<Option<Instant>>,
}

impl LatestFrameCounters {
    fn new() -> Self {
        Self::with_shared_receipts(Arc::new(LatestFrameReceiptCounters::new()))
    }

    fn with_shared_receipts(receipts: Arc<LatestFrameReceiptCounters>) -> Self {
        Self {
            enqueued: AtomicU64::new(0),
            replaced_older: AtomicU64::new(0),
            first_terminal_enqueued: AtomicU64::new(0),
            frames_discarded_for_terminal: AtomicU64::new(0),
            rejected_behind_pending_terminal: AtomicU64::new(0),
            rejected_after_fault: AtomicU64::new(0),
            rejected_disconnected: AtomicU64::new(0),
            channel_poisoned: AtomicU64::new(0),
            receipts,
        }
    }

    fn snapshot(&self) -> NanoAccessoryFrameStats {
        NanoAccessoryFrameStats {
            enqueued: self.enqueued.load(Ordering::Relaxed),
            replaced_older: self.replaced_older.load(Ordering::Relaxed),
            first_terminal_enqueued: self.first_terminal_enqueued.load(Ordering::Relaxed),
            frames_discarded_for_terminal: self
                .frames_discarded_for_terminal
                .load(Ordering::Relaxed),
            rejected_behind_pending_terminal: self
                .rejected_behind_pending_terminal
                .load(Ordering::Relaxed),
            processed_successfully: self.receipts.processed_successfully.load(Ordering::Relaxed),
            rejected_after_fault: self.rejected_after_fault.load(Ordering::Relaxed),
            rejected_disconnected: self.rejected_disconnected.load(Ordering::Relaxed),
            channel_poisoned: self.channel_poisoned.load(Ordering::Relaxed),
        }
    }

    fn record_processed_successfully(&self) -> Result<(), NanoAccessoryHealthStatusError> {
        self.record_processed_successfully_at(Instant::now())
    }

    fn record_processed_successfully_at(
        &self,
        observed_at: Instant,
    ) -> Result<(), NanoAccessoryHealthStatusError> {
        let mut latest = self
            .receipts
            .last_processed_successfully_at
            .lock()
            .map_err(|_| NanoAccessoryHealthStatusError::Poisoned)?;
        *latest = Some(observed_at);
        saturating_increment(&self.receipts.processed_successfully);
        Ok(())
    }

    fn last_processed_successfully_at(
        &self,
    ) -> Result<Option<Instant>, NanoAccessoryHealthStatusError> {
        self.receipts
            .last_processed_successfully_at
            .lock()
            .map(|observed_at| *observed_at)
            .map_err(|_| NanoAccessoryHealthStatusError::Poisoned)
    }
}

impl LatestFrameReceiptCounters {
    fn new() -> Self {
        Self {
            processed_successfully: AtomicU64::new(0),
            last_processed_successfully_at: Mutex::new(None),
        }
    }
}

fn saturating_increment(counter: &AtomicU64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        value.checked_add(1)
    });
}

struct LatestFrameSlot<F> {
    latest: Option<F>,
    first_terminal: Option<F>,
}

impl<F> LatestFrameSlot<F> {
    const fn empty() -> Self {
        Self {
            latest: None,
            first_terminal: None,
        }
    }

    fn take_terminal(&mut self) -> Option<F> {
        self.first_terminal.take()
    }

    fn take_latest(&mut self) -> Option<F> {
        self.latest.take()
    }
}

#[derive(Clone, Copy)]
enum LatestFrameSubmissionKind {
    ReplaceLatest,
    RetainFirstTerminal,
}

struct LatestFrameChannel<F> {
    slot: Mutex<LatestFrameSlot<F>>,
    #[cfg(feature = "nano-agent")]
    blocking_notify: Condvar,
    notify: tokio::sync::Notify,
    ingress_alive: AtomicBool,
    accepting_frames: Arc<AtomicBool>,
    shutdown_requested: AtomicBool,
    poisoned: AtomicBool,
    counters: Arc<LatestFrameCounters>,
}

impl<F> LatestFrameChannel<F> {
    fn new() -> Self {
        Self::with_counters(Arc::new(LatestFrameCounters::new()))
    }

    #[cfg(feature = "nano-agent")]
    fn with_shared_receipts(receipts: Arc<LatestFrameReceiptCounters>) -> Self {
        Self::with_counters(Arc::new(LatestFrameCounters::with_shared_receipts(
            receipts,
        )))
    }

    fn with_counters(counters: Arc<LatestFrameCounters>) -> Self {
        Self {
            slot: Mutex::new(LatestFrameSlot::empty()),
            #[cfg(feature = "nano-agent")]
            blocking_notify: Condvar::new(),
            notify: tokio::sync::Notify::new(),
            ingress_alive: AtomicBool::new(true),
            accepting_frames: Arc::new(AtomicBool::new(true)),
            shutdown_requested: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            counters,
        }
    }

    fn submit(&self, frame: F) -> NanoAccessoryFrameSubmitOutcome {
        self.submit_inner(frame, true, LatestFrameSubmissionKind::ReplaceLatest)
    }

    #[cfg(feature = "nano-agent")]
    fn submit_unmetered(&self, frame: F) -> NanoAccessoryFrameSubmitOutcome {
        self.submit_inner(frame, false, LatestFrameSubmissionKind::ReplaceLatest)
    }

    fn submit_first_terminal(
        &self,
        frame: F,
        record_submission: bool,
    ) -> NanoAccessoryFrameSubmitOutcome {
        self.submit_inner(
            frame,
            record_submission,
            LatestFrameSubmissionKind::RetainFirstTerminal,
        )
    }

    fn submit_inner(
        &self,
        frame: F,
        record_submission: bool,
        kind: LatestFrameSubmissionKind,
    ) -> NanoAccessoryFrameSubmitOutcome {
        self.submit_inner_with_pre_lock_hook(frame, record_submission, kind, || {})
    }

    fn submit_inner_with_pre_lock_hook(
        &self,
        frame: F,
        record_submission: bool,
        kind: LatestFrameSubmissionKind,
        pre_lock_hook: impl FnOnce(),
    ) -> NanoAccessoryFrameSubmitOutcome {
        if self.poisoned.load(Ordering::Acquire) {
            if record_submission {
                saturating_increment(&self.counters.channel_poisoned);
            }
            return NanoAccessoryFrameSubmitOutcome::ChannelPoisoned;
        }
        if !self.ingress_alive.load(Ordering::Acquire)
            || self.shutdown_requested.load(Ordering::Acquire)
        {
            if record_submission {
                saturating_increment(&self.counters.rejected_disconnected);
            }
            return NanoAccessoryFrameSubmitOutcome::IngressDisconnected;
        }
        if !self.accepting_frames.load(Ordering::Acquire) {
            if record_submission {
                saturating_increment(&self.counters.rejected_after_fault);
            }
            return NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched;
        }

        pre_lock_hook();
        let mut slot = match self.slot.lock() {
            Ok(slot) => slot,
            Err(_) => {
                self.poisoned.store(true, Ordering::Release);
                self.accepting_frames.store(false, Ordering::Release);
                if record_submission {
                    saturating_increment(&self.counters.channel_poisoned);
                }
                self.notify.notify_one();
                #[cfg(feature = "nano-agent")]
                self.blocking_notify.notify_one();
                return NanoAccessoryFrameSubmitOutcome::ChannelPoisoned;
            }
        };
        if !self.ingress_alive.load(Ordering::Acquire)
            || self.shutdown_requested.load(Ordering::Acquire)
        {
            if record_submission {
                saturating_increment(&self.counters.rejected_disconnected);
            }
            return NanoAccessoryFrameSubmitOutcome::IngressDisconnected;
        }
        if !self.accepting_frames.load(Ordering::Acquire) {
            if record_submission {
                saturating_increment(&self.counters.rejected_after_fault);
            }
            return NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched;
        }
        if matches!(kind, LatestFrameSubmissionKind::ReplaceLatest) && slot.first_terminal.is_some()
        {
            if record_submission {
                saturating_increment(&self.counters.rejected_behind_pending_terminal);
            }
            return NanoAccessoryFrameSubmitOutcome::TerminalFaultPendingPublication;
        }
        let replaced = match kind {
            LatestFrameSubmissionKind::ReplaceLatest => slot.latest.replace(frame).is_some(),
            LatestFrameSubmissionKind::RetainFirstTerminal => {
                if slot.first_terminal.is_some() {
                    if record_submission {
                        saturating_increment(&self.counters.rejected_behind_pending_terminal);
                    }
                    return NanoAccessoryFrameSubmitOutcome::TerminalFaultPendingPublication;
                }
                let discarded_latest = slot.latest.take().is_some();
                slot.first_terminal = Some(frame);
                if record_submission {
                    saturating_increment(&self.counters.first_terminal_enqueued);
                    if discarded_latest {
                        saturating_increment(&self.counters.frames_discarded_for_terminal);
                    }
                }
                false
            }
        };
        if record_submission && matches!(kind, LatestFrameSubmissionKind::ReplaceLatest) {
            saturating_increment(&self.counters.enqueued);
            if replaced {
                saturating_increment(&self.counters.replaced_older);
            }
        }
        drop(slot);
        self.notify.notify_one();
        #[cfg(feature = "nano-agent")]
        self.blocking_notify.notify_one();
        if replaced {
            NanoAccessoryFrameSubmitOutcome::ReplacedOlderFrame
        } else {
            NanoAccessoryFrameSubmitOutcome::Enqueued
        }
    }

    fn request_shutdown(&self) {
        let slot = self.lock_slot_recovering_poison();
        // Publish the reason before closing admission. A producer which
        // synchronizes on the later `accepting_frames=false` store can then
        // distinguish coordinated shutdown from a terminal-fault latch.
        self.shutdown_requested.store(true, Ordering::Release);
        self.accepting_frames.store(false, Ordering::Release);
        self.notify.notify_waiters();
        #[cfg(feature = "nano-agent")]
        self.blocking_notify.notify_all();
        drop(slot);
    }

    fn latch_terminal_fault(&self) {
        let mut slot = self.lock_slot_recovering_poison();
        self.latch_terminal_fault_while_locked(&mut slot);
        drop(slot);
    }

    fn latch_terminal_fault_while_locked(&self, slot: &mut LatestFrameSlot<F>) {
        if slot.latest.take().is_some() {
            saturating_increment(&self.counters.frames_discarded_for_terminal);
        }
        // A first terminal value remains authoritative even if this method is
        // the first observer of a poisoned mutex. Event consumers drain it
        // before reporting the separately latched poison.
        self.accepting_frames.store(false, Ordering::Release);
        #[cfg(feature = "nano-agent")]
        self.blocking_notify.notify_all();
    }

    fn disconnect_ingress(&self) {
        let slot = self.lock_slot_recovering_poison();
        self.ingress_alive.store(false, Ordering::Release);
        self.notify.notify_waiters();
        #[cfg(feature = "nano-agent")]
        self.blocking_notify.notify_all();
        drop(slot);
    }

    fn lock_slot_recovering_poison(&self) -> std::sync::MutexGuard<'_, LatestFrameSlot<F>> {
        match self.slot.lock() {
            Ok(slot) => slot,
            Err(poisoned) => {
                self.poisoned.store(true, Ordering::Release);
                self.accepting_frames.store(false, Ordering::Release);
                poisoned.into_inner()
            }
        }
    }

    async fn next_event(&self) -> LatestFrameEvent<F> {
        loop {
            let notified = self.notify.notified();
            let event = {
                let mut slot = self.lock_slot_recovering_poison();
                // A terminal value committed before shutdown is authoritative
                // and cannot be erased by the later lifecycle request.
                // Ordinary data remains subordinate to shutdown.
                if let Some(frame) = slot.take_terminal() {
                    Some(LatestFrameEvent::Frame(frame))
                } else if self.poisoned.load(Ordering::Acquire) {
                    saturating_increment(&self.counters.channel_poisoned);
                    Some(LatestFrameEvent::ChannelPoisoned)
                } else if self.shutdown_requested.load(Ordering::Acquire) {
                    Some(LatestFrameEvent::ShutdownRequested)
                } else if let Some(frame) = slot.take_latest() {
                    Some(LatestFrameEvent::Frame(frame))
                } else if !self.ingress_alive.load(Ordering::Acquire) {
                    Some(LatestFrameEvent::IngressDisconnected)
                } else {
                    None
                }
            };
            if let Some(event) = event {
                return event;
            }
            notified.await;
        }
    }

    async fn wait_for_shutdown(&self) {
        loop {
            let notified = self.notify.notified();
            if self.shutdown_requested.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    #[cfg(feature = "nano-agent")]
    fn next_event_blocking(&self) -> LatestFrameEvent<F> {
        let mut slot = self.lock_slot_recovering_poison();
        loop {
            if let Some(frame) = slot.take_terminal() {
                return LatestFrameEvent::Frame(frame);
            }
            if self.poisoned.load(Ordering::Acquire) {
                return LatestFrameEvent::ChannelPoisoned;
            }
            if self.shutdown_requested.load(Ordering::Acquire) {
                return LatestFrameEvent::ShutdownRequested;
            }
            if let Some(frame) = slot.take_latest() {
                return LatestFrameEvent::Frame(frame);
            }
            if !self.ingress_alive.load(Ordering::Acquire) {
                return LatestFrameEvent::IngressDisconnected;
            }
            slot = match self.blocking_notify.wait(slot) {
                Ok(slot) => slot,
                Err(poisoned) => {
                    self.poisoned.store(true, Ordering::Release);
                    self.accepting_frames.store(false, Ordering::Release);
                    poisoned.into_inner()
                }
            };
        }
    }
}

enum LatestFrameEvent<F> {
    Frame(F),
    ShutdownRequested,
    IngressDisconnected,
    ChannelPoisoned,
}

type NanoAccessoryRgbWork = Result<IngressObservedRgbFrame<ImageFrame>, ClockError>;

#[cfg(feature = "nano-agent")]
struct NanoFaceTrackedRgbFrame {
    frame: ParsedIngressRgbFrame,
    output: NanoFacePerceptionOutput,
}

trait NanoHeadGazeFrame {
    #[cfg(feature = "nano-agent")]
    fn tracked_face_frame(&self) -> Option<&NanoFaceTrackedRgbFrame> {
        None
    }
}

impl NanoHeadGazeFrame for NanoAccessoryRgbWork {}

#[cfg(feature = "nano-agent")]
impl NanoHeadGazeFrame for NanoFacePerceptionWork {
    fn tracked_face_frame(&self) -> Option<&NanoFaceTrackedRgbFrame> {
        self.as_ref().ok()
    }
}

#[cfg(test)]
impl NanoHeadGazeFrame for u64 {}
#[cfg(test)]
impl NanoHeadGazeFrame for Result<u64, ClockError> {}

/// Exact same-frame RGB projection metadata used by one diagnostic proposal.
#[cfg(feature = "nano-agent")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NanoHeadGazeRgbProjectionEvidence {
    fx_px: f32,
    fy_px: f32,
    cx_px: f32,
    cy_px: f32,
    width_px: u32,
    height_px: u32,
}

#[cfg(feature = "nano-agent")]
impl NanoHeadGazeRgbProjectionEvidence {
    fn from_frame(frame: &ImageFrame) -> Self {
        let intrinsics = frame.intrinsics();
        Self {
            fx_px: intrinsics.fx(),
            fy_px: intrinsics.fy(),
            cx_px: intrinsics.cx(),
            cy_px: intrinsics.cy(),
            width_px: intrinsics.width(),
            height_px: intrinsics.height(),
        }
    }

    pub const fn coefficients_px(self) -> [f32; 4] {
        [self.fx_px, self.fy_px, self.cx_px, self.cy_px]
    }

    pub const fn dimensions_px(self) -> [u32; 2] {
        [self.width_px, self.height_px]
    }
}

/// Per-frame proposal failure. Every value remains diagnostic-only.
#[cfg(feature = "nano-agent")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NanoHeadGazeDiagnosticError {
    ProjectionGridMismatch {
        frame_width_px: u32,
        frame_height_px: u32,
        intrinsic_width_px: u32,
        intrinsic_height_px: u32,
    },
    Intrinsics(IntrinsicsError),
    Proposal(HeadGazeFaceProposalError),
}

/// Truthful disposition of the physical head-gaze path.
///
/// The only current value is intentionally negative. No code in this worker
/// can construct the actor's opaque base-zero lease, so a proposal can never
/// become a serial transaction here.
#[cfg(feature = "nano-agent")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoHeadGazeActuationAvailability {
    UnavailableWithoutBaseZeroExclusiveLease,
}

/// One same-frame head-gaze diagnostic.
///
/// `ProposalOnly` records the exact face-tracker clock sample and RGB
/// intrinsics used by the typed adapter. It is historical observability, not a
/// live command or evidence that the target remained fresh after publication.
#[cfg(feature = "nano-agent")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NanoHeadGazeDiagnostic {
    DisabledNoPolicy {
        actuation: NanoHeadGazeActuationAvailability,
    },
    ProposalOnly {
        evaluated_at: MonotonicTimestamp,
        projection: NanoHeadGazeRgbProjectionEvidence,
        outcome: Result<HeadGazeFaceProposalOutcome, NanoHeadGazeDiagnosticError>,
        actuation: NanoHeadGazeActuationAvailability,
    },
}

#[cfg(feature = "nano-agent")]
impl NanoHeadGazeDiagnostic {
    pub const fn actuation(self) -> NanoHeadGazeActuationAvailability {
        match self {
            Self::DisabledNoPolicy { actuation } | Self::ProposalOnly { actuation, .. } => {
                actuation
            }
        }
    }

    pub const fn outcome(
        self,
    ) -> Option<Result<HeadGazeFaceProposalOutcome, NanoHeadGazeDiagnosticError>> {
        match self {
            Self::DisabledNoPolicy { .. } => None,
            Self::ProposalOnly { outcome, .. } => Some(outcome),
        }
    }
}

/// Copy-only face metadata derived from the exact authoritative RGB frame.
///
/// This contains no pixels and grants no actuation authority. A diagnostic
/// consumer must join it to an RGB visualization only through the exact OAK
/// provenance key; a merely "latest" image may describe another capture.
#[cfg(feature = "nano-agent")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NanoFaceDiagnosticFrame {
    provenance: OakFaceFrameProvenance,
    output: NanoFacePerceptionOutput,
    accessory_observed_at: MonotonicTimestamp,
    accessory_source_deadline: Deadline,
    head_gaze: NanoHeadGazeDiagnostic,
}

#[cfg(feature = "nano-agent")]
impl NanoFaceDiagnosticFrame {
    fn from_parsed(
        frame: &ParsedIngressRgbFrame,
        output: NanoFacePerceptionOutput,
        head_gaze: NanoHeadGazeDiagnostic,
    ) -> Self {
        let freshness = frame.observation().freshness();
        Self {
            provenance: OakFaceFrameProvenance::from_frame(frame.frame()),
            output,
            accessory_observed_at: freshness.observed_at(),
            accessory_source_deadline: freshness.valid_until_exclusive(),
            head_gaze,
        }
    }

    pub const fn provenance(self) -> OakFaceFrameProvenance {
        self.provenance
    }

    pub const fn output(self) -> NanoFacePerceptionOutput {
        self.output
    }

    /// Accessory-local monotonic observation time.
    ///
    /// This is not the live capture/Rerun clock domain. Use
    /// [`Self::provenance`] as the exact RGB join key.
    pub const fn accessory_observed_at(self) -> MonotonicTimestamp {
        self.accessory_observed_at
    }

    /// Exclusive freshness deadline in the same accessory-local clock domain.
    pub const fn accessory_source_deadline(self) -> Deadline {
        self.accessory_source_deadline
    }

    /// Proposal-only head-gaze result for this exact frame.
    pub const fn head_gaze(self) -> NanoHeadGazeDiagnostic {
        self.head_gaze
    }
}

#[cfg(feature = "nano-agent")]
pub type NanoFaceDiagnosticReceiver = DropReceiver<NanoFaceDiagnosticFrame>;

#[cfg(feature = "nano-agent")]
#[derive(Clone, Debug)]
pub struct NanoFaceDiagnosticStatsHandle(ChannelStatsHandle);

#[cfg(feature = "nano-agent")]
impl NanoFaceDiagnosticStatsHandle {
    pub fn snapshot(&self) -> ChannelStats {
        self.0.snapshot()
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NanoFacePerceptionStageStats {
    pub results_produced: u64,
    pub head_gaze_disabled_no_policy: u64,
    pub head_gaze_proposed: u64,
    pub head_gaze_withheld: u64,
    pub head_gaze_rejected: u64,
    pub handoff_enqueued: u64,
    pub handoff_replaced_older: u64,
    pub handoff_terminal_pending: u64,
    pub handoff_terminal_fault_latched: u64,
    pub handoff_disconnected: u64,
    pub handoff_channel_poisoned: u64,
}

/// Cloneable face-stage telemetry which carries no detector or actuator owner.
///
/// Concurrent snapshots load each saturating counter independently and may
/// transiently observe a produced result before its handoff outcome. Treat a
/// snapshot as final only after shutdown reports a joined face thread;
/// `DetachedAfterTimeout` means the producer may still advance it.
#[cfg(feature = "nano-agent")]
#[derive(Clone, Debug)]
pub struct NanoFacePerceptionStageStatsHandle(Arc<NanoFacePerceptionStageCounters>);

#[cfg(feature = "nano-agent")]
impl NanoFacePerceptionStageStatsHandle {
    pub fn snapshot(&self) -> NanoFacePerceptionStageStats {
        self.0.snapshot()
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Debug)]
struct NanoFacePerceptionStageCounters {
    results_produced: AtomicU64,
    head_gaze_disabled_no_policy: AtomicU64,
    head_gaze_proposed: AtomicU64,
    head_gaze_withheld: AtomicU64,
    head_gaze_rejected: AtomicU64,
    handoff_enqueued: AtomicU64,
    handoff_replaced_older: AtomicU64,
    handoff_terminal_pending: AtomicU64,
    handoff_terminal_fault_latched: AtomicU64,
    handoff_disconnected: AtomicU64,
    handoff_channel_poisoned: AtomicU64,
}

#[cfg(feature = "nano-agent")]
impl NanoFacePerceptionStageCounters {
    fn new() -> Self {
        Self {
            results_produced: AtomicU64::new(0),
            head_gaze_disabled_no_policy: AtomicU64::new(0),
            head_gaze_proposed: AtomicU64::new(0),
            head_gaze_withheld: AtomicU64::new(0),
            head_gaze_rejected: AtomicU64::new(0),
            handoff_enqueued: AtomicU64::new(0),
            handoff_replaced_older: AtomicU64::new(0),
            handoff_terminal_pending: AtomicU64::new(0),
            handoff_terminal_fault_latched: AtomicU64::new(0),
            handoff_disconnected: AtomicU64::new(0),
            handoff_channel_poisoned: AtomicU64::new(0),
        }
    }

    fn record_result(&self) {
        saturating_increment(&self.results_produced);
    }

    fn record_head_gaze(&self, diagnostic: NanoHeadGazeDiagnostic) {
        let counter = match diagnostic {
            NanoHeadGazeDiagnostic::DisabledNoPolicy { .. } => &self.head_gaze_disabled_no_policy,
            NanoHeadGazeDiagnostic::ProposalOnly {
                outcome: Ok(HeadGazeFaceProposalOutcome::Proposed(_)),
                ..
            } => &self.head_gaze_proposed,
            NanoHeadGazeDiagnostic::ProposalOnly {
                outcome: Ok(HeadGazeFaceProposalOutcome::Withheld(_)),
                ..
            } => &self.head_gaze_withheld,
            NanoHeadGazeDiagnostic::ProposalOnly {
                outcome: Err(_), ..
            } => &self.head_gaze_rejected,
        };
        saturating_increment(counter);
    }

    fn record_handoff(&self, outcome: NanoAccessoryFrameSubmitOutcome) {
        let counter = match outcome {
            NanoAccessoryFrameSubmitOutcome::Enqueued => &self.handoff_enqueued,
            NanoAccessoryFrameSubmitOutcome::ReplacedOlderFrame => &self.handoff_replaced_older,
            NanoAccessoryFrameSubmitOutcome::TerminalFaultPendingPublication => {
                &self.handoff_terminal_pending
            }
            NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched => {
                &self.handoff_terminal_fault_latched
            }
            NanoAccessoryFrameSubmitOutcome::IngressDisconnected => &self.handoff_disconnected,
            NanoAccessoryFrameSubmitOutcome::ChannelPoisoned => &self.handoff_channel_poisoned,
        };
        saturating_increment(counter);
    }

    fn snapshot(&self) -> NanoFacePerceptionStageStats {
        NanoFacePerceptionStageStats {
            results_produced: self.results_produced.load(Ordering::Relaxed),
            head_gaze_disabled_no_policy: self.head_gaze_disabled_no_policy.load(Ordering::Relaxed),
            head_gaze_proposed: self.head_gaze_proposed.load(Ordering::Relaxed),
            head_gaze_withheld: self.head_gaze_withheld.load(Ordering::Relaxed),
            head_gaze_rejected: self.head_gaze_rejected.load(Ordering::Relaxed),
            handoff_enqueued: self.handoff_enqueued.load(Ordering::Relaxed),
            handoff_replaced_older: self.handoff_replaced_older.load(Ordering::Relaxed),
            handoff_terminal_pending: self.handoff_terminal_pending.load(Ordering::Relaxed),
            handoff_terminal_fault_latched: self
                .handoff_terminal_fault_latched
                .load(Ordering::Relaxed),
            handoff_disconnected: self.handoff_disconnected.load(Ordering::Relaxed),
            handoff_channel_poisoned: self.handoff_channel_poisoned.load(Ordering::Relaxed),
        }
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Debug, PartialEq)]
pub enum NanoFacePerceptionRuntimeError {
    IngressClock(ClockError),
    Parse(Box<RgbExpressionBridgeError>),
    Perception(Box<NanoFacePerceptionError>),
    RgbIngressDisconnected,
    RgbChannelPoisoned,
    PerceptionOutputDisconnected,
    PerceptionOutputChannelPoisoned,
    ExpressionHandoffUnavailable {
        outcome: NanoAccessoryFrameSubmitOutcome,
    },
    PerceptionThreadPanicked,
}

#[cfg(feature = "nano-agent")]
impl fmt::Display for NanoFacePerceptionRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano face-perception lane failed: {self:?}")
    }
}

#[cfg(feature = "nano-agent")]
impl std::error::Error for NanoFacePerceptionRuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IngressClock(source) => Some(source),
            Self::Parse(source) => Some(source.as_ref()),
            Self::Perception(source) => Some(source.as_ref()),
            Self::RgbIngressDisconnected
            | Self::RgbChannelPoisoned
            | Self::PerceptionOutputDisconnected
            | Self::PerceptionOutputChannelPoisoned
            | Self::ExpressionHandoffUnavailable { .. }
            | Self::PerceptionThreadPanicked => None,
        }
    }
}

#[cfg(feature = "nano-agent")]
type NanoFacePerceptionWork = Result<NanoFaceTrackedRgbFrame, NanoFacePerceptionRuntimeError>;

#[cfg(feature = "nano-agent")]
#[derive(Debug)]
pub enum NanoFacePerceptionThreadExit {
    Shutdown,
    LoadFailed(NanoFacePerceptionLoadError),
    StartupObserverDropped,
    RuntimeFault {
        source: NanoFacePerceptionRuntimeError,
        published_to_accessory: bool,
    },
    AccessoryFaultPendingPublication,
    AccessoryFaultLatched,
}

#[cfg(feature = "nano-agent")]
impl fmt::Display for NanoFacePerceptionThreadExit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano face-perception thread exited: {self:?}")
    }
}

#[cfg(feature = "nano-agent")]
impl std::error::Error for NanoFacePerceptionThreadExit {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LoadFailed(source) => Some(source),
            Self::RuntimeFault { source, .. } => Some(source),
            Self::Shutdown
            | Self::StartupObserverDropped
            | Self::AccessoryFaultPendingPublication
            | Self::AccessoryFaultLatched => None,
        }
    }
}

/// Bounded evidence from attempting to join the hardware-free detector owner.
///
/// `DetachedAfterTimeout` proves only that the thread had not exited by the
/// deadline; it does not identify whether it was in OpenCV, Rust conversion,
/// scheduling delay, or teardown. The thread owns no OAK, serial bus, or
/// actuator, but may continue consuming CPU. This is not cancellation
/// evidence.
#[cfg(feature = "nano-agent")]
#[derive(Debug)]
pub enum NanoFacePerceptionJoinEvidence {
    Joined(NanoFacePerceptionThreadExit),
    DetachedAfterTimeout {
        configured_timeout: Duration,
        active_join_budget: Duration,
    },
}

#[cfg(feature = "nano-agent")]
impl NanoFacePerceptionJoinEvidence {
    pub const fn joined_exit(&self) -> Option<&NanoFacePerceptionThreadExit> {
        match self {
            Self::Joined(exit) => Some(exit),
            Self::DetachedAfterTimeout { .. } => None,
        }
    }

    pub const fn detached_timeout(&self) -> Option<Duration> {
        match self {
            Self::Joined(_) => None,
            Self::DetachedAfterTimeout {
                configured_timeout, ..
            } => Some(*configured_timeout),
        }
    }

    pub const fn detached_active_join_budget(&self) -> Option<Duration> {
        match self {
            Self::Joined(_) => None,
            Self::DetachedAfterTimeout {
                active_join_budget, ..
            } => Some(*active_join_budget),
        }
    }
}

#[cfg(feature = "nano-agent")]
impl fmt::Display for NanoFacePerceptionJoinEvidence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano face-perception join evidence: {self:?}")
    }
}

#[cfg(feature = "nano-agent")]
impl std::error::Error for NanoFacePerceptionJoinEvidence {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Joined(exit) => Some(exit),
            Self::DetachedAfterTimeout { .. } => None,
        }
    }
}

/// Whether the production face lane participated in accessory shutdown.
///
/// `Disabled` is explicit evidence that this worker was constructed without
/// the face lane. `Join` retains the exact bounded join result. This removes
/// the former ambiguous `None`, which could mean either deliberately disabled
/// or accidentally lost evidence.
#[cfg(feature = "nano-agent")]
#[derive(Debug)]
pub enum NanoFacePerceptionShutdownEvidence {
    Disabled,
    Join(NanoFacePerceptionJoinEvidence),
}

/// Pure interpretation of face shutdown evidence paired with the accessory's
/// first terminal fault.
///
/// This classifier never infers cancellation from a detached thread. A
/// runtime fault counts as published only when the face thread says it
/// published and the retained accessory terminal fault carries the exact same
/// typed source. Accessory-fault follower exits are coordinated only when an
/// accessory terminal fault was actually retained.
#[cfg(feature = "nano-agent")]
#[derive(Clone, Copy, Debug)]
pub enum NanoFacePerceptionShutdownClass<'a> {
    Disabled,
    CoordinatedShutdown,
    PublishedRuntimeFault {
        thread_source: &'a NanoFacePerceptionRuntimeError,
        terminal_source: &'a NanoFacePerceptionRuntimeError,
    },
    AccessoryFaultFollower {
        exit: &'a NanoFacePerceptionThreadExit,
        terminal_fault: &'a NanoAccessoryTerminalFault,
    },
    UnexpectedDisabledFaceFault {
        terminal_source: &'a NanoFacePerceptionRuntimeError,
    },
    UnexpectedJoined {
        exit: &'a NanoFacePerceptionThreadExit,
        terminal_fault: Option<&'a NanoAccessoryTerminalFault>,
    },
    DetachedAfterTimeout {
        configured_timeout: Duration,
        active_join_budget: Duration,
        terminal_fault: Option<&'a NanoAccessoryTerminalFault>,
    },
}

#[cfg(feature = "nano-agent")]
impl NanoFacePerceptionShutdownEvidence {
    pub const fn join_evidence(&self) -> Option<&NanoFacePerceptionJoinEvidence> {
        match self {
            Self::Disabled => None,
            Self::Join(evidence) => Some(evidence),
        }
    }

    pub fn classify<'a>(
        &'a self,
        terminal_fault: Option<&'a NanoAccessoryTerminalFault>,
    ) -> NanoFacePerceptionShutdownClass<'a> {
        match (self, terminal_fault) {
            (Self::Disabled, Some(NanoAccessoryTerminalFault::FacePerception(terminal_source))) => {
                NanoFacePerceptionShutdownClass::UnexpectedDisabledFaceFault { terminal_source }
            }
            (Self::Disabled, _) => NanoFacePerceptionShutdownClass::Disabled,
            (
                Self::Join(NanoFacePerceptionJoinEvidence::Joined(
                    NanoFacePerceptionThreadExit::Shutdown,
                )),
                None,
            ) => NanoFacePerceptionShutdownClass::CoordinatedShutdown,
            (
                Self::Join(NanoFacePerceptionJoinEvidence::Joined(
                    NanoFacePerceptionThreadExit::RuntimeFault {
                        source: thread_source,
                        published_to_accessory: true,
                    },
                )),
                Some(NanoAccessoryTerminalFault::FacePerception(terminal_source)),
            ) if thread_source == terminal_source => {
                NanoFacePerceptionShutdownClass::PublishedRuntimeFault {
                    thread_source,
                    terminal_source,
                }
            }
            (
                Self::Join(NanoFacePerceptionJoinEvidence::Joined(
                    exit @ (NanoFacePerceptionThreadExit::AccessoryFaultPendingPublication
                    | NanoFacePerceptionThreadExit::AccessoryFaultLatched),
                )),
                Some(terminal_fault),
            ) => NanoFacePerceptionShutdownClass::AccessoryFaultFollower {
                exit,
                terminal_fault,
            },
            (Self::Join(NanoFacePerceptionJoinEvidence::Joined(exit)), terminal_fault) => {
                NanoFacePerceptionShutdownClass::UnexpectedJoined {
                    exit,
                    terminal_fault,
                }
            }
            (
                Self::Join(NanoFacePerceptionJoinEvidence::DetachedAfterTimeout {
                    configured_timeout,
                    active_join_budget,
                }),
                terminal_fault,
            ) => NanoFacePerceptionShutdownClass::DetachedAfterTimeout {
                configured_timeout: *configured_timeout,
                active_join_budget: *active_join_budget,
                terminal_fault,
            },
        }
    }
}

#[cfg(feature = "nano-agent")]
impl NanoFacePerceptionShutdownClass<'_> {
    /// No face shutdown fault or uncertainty was observed.
    pub const fn is_healthy(&self) -> bool {
        matches!(self, Self::Disabled | Self::CoordinatedShutdown)
    }

    /// Shutdown is internally consistent, including accurately propagated
    /// terminal-fault exits. Coordinated does not imply healthy.
    pub const fn is_coordinated(&self) -> bool {
        matches!(
            self,
            Self::Disabled
                | Self::CoordinatedShutdown
                | Self::PublishedRuntimeFault { .. }
                | Self::AccessoryFaultFollower { .. }
        )
    }

    pub const fn is_uncertain_or_unexpected(&self) -> bool {
        !self.is_coordinated()
    }
}

#[cfg(feature = "nano-agent")]
impl fmt::Display for NanoFacePerceptionShutdownClass<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano face-perception shutdown classification: {self:?}"
        )
    }
}

#[derive(Clone, Debug)]
enum NanoAccessoryRgbProcessingError {
    Bridge(RgbExpressionBridgeError),
    #[cfg(feature = "nano-agent")]
    FacePerception(NanoFacePerceptionRuntimeError),
}

/// Sole synchronous producer for the capacity-one RGB handoff.
#[must_use = "dropping the sole RGB ingress publishes a terminal disconnect fault"]
struct NanoAccessoryRgbIngress {
    channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>>,
    clock: NanoAccessoryClock,
    connected: bool,
}

impl NanoAccessoryRgbIngress {
    /// Observe and move the already-owned frame into the capacity-one slot.
    ///
    /// The clock shares the bridge's exact origin. A sampling failure is moved
    /// into the same slot as a typed error and becomes terminal when consumed;
    /// it is not mislabeled as a successfully observed frame. Neither path
    /// clones or converts the frame or its pixel storage.
    pub fn submit(&mut self, frame: ImageFrame) -> NanoAccessoryFrameSubmitOutcome {
        self.submit_after_observation(frame, |_| {})
    }

    /// Sample ingress time before running one borrowed diagnostic projection.
    ///
    /// The callback cannot retain the borrowed frame. Its execution time is
    /// nevertheless part of queue residence, so an expensive diagnostic copy
    /// cannot make a stale authoritative frame appear fresh. The original
    /// frame and pixel allocation are moved into the replace-latest slot only
    /// after the callback returns.
    fn submit_after_observation(
        &mut self,
        frame: ImageFrame,
        diagnostic: impl FnOnce(&ImageFrame),
    ) -> NanoAccessoryFrameSubmitOutcome {
        match observe_rgb_at_ingress(&self.clock, frame, diagnostic) {
            Ok(frame) => self.channel.submit(Ok(frame)),
            Err(source) => self.channel.submit_first_terminal(Err(source), true),
        }
    }

    fn disconnect(&mut self) {
        if self.connected {
            self.connected = false;
            self.channel.disconnect_ingress();
        }
    }
}

fn observe_rgb_at_ingress<F, C: MonotonicClock>(
    clock: &C,
    frame: F,
    diagnostic: impl FnOnce(&F),
) -> Result<IngressObservedRgbFrame<F>, ClockError> {
    let observed_at = clock.now()?;
    diagnostic(&frame);
    Ok(IngressObservedRgbFrame::new(frame, observed_at))
}

impl Drop for NanoAccessoryRgbIngress {
    fn drop(&mut self) {
        self.disconnect();
    }
}

/// First terminal worker fault. It remains a stop/disarm signal even if later
/// health checks recover.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NanoHeadHealthError {
    Request(HeadHealthRequestError),
    UnexpectedHoldTarget {
        required: HeadHoldTarget,
        observed: HeadHoldTarget,
    },
    #[cfg(feature = "nano-agent")]
    GazeTargetOutsideAdmittedEnvelope {
        observed: HeadHoldTarget,
    },
}

impl fmt::Display for NanoHeadHealthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano head health check failed: {self:?}")
    }
}

impl std::error::Error for NanoHeadHealthError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Request(source) => Some(source),
            Self::UnexpectedHoldTarget { .. } => None,
            #[cfg(feature = "nano-agent")]
            Self::GazeTargetOutsideAdmittedEnvelope { .. } => None,
        }
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Debug, PartialEq)]
pub enum NanoPhysicalHeadGazeRuntimeError {
    ProjectionGridMismatch {
        frame_width_px: u32,
        frame_height_px: u32,
        intrinsic_width_px: u32,
        intrinsic_height_px: u32,
    },
    Intrinsics(IntrinsicsError),
    Proposal(PhysicalCharacterHeadEvaluationError),
    ProposalSequenceExhausted {
        detector_result_sequence: u64,
    },
    ProposalId(HeadGazeProposalIdError),
    ProposalCommand(HeadCommandError),
    ProposalRejected(HeadGazeProposalCommandError),
    BaseInterlock(HeadGazeBaseInterlockError),
    ServiceCommand(HeadCommandError),
    Service(Box<HeadGazeServiceError>),
    PetEpisode(CharacterPetEpisodeError),
    PetEpisodeTimestampOutOfRange {
        completed_at_ns: u128,
    },
    PetContactCandidateHasZeroSamples,
    PetRecoveryProgressOutOfRange {
        progress_millionths: u32,
    },
    CommandedGaze(CommandedHeadGazeEstimateError),
    GazeResidual(HeadGazeResidualError),
    CommandedTargetUnavailable,
}

#[cfg(feature = "nano-agent")]
impl fmt::Display for NanoPhysicalHeadGazeRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "physical head-gaze runtime failed: {self:?}")
    }
}

#[cfg(feature = "nano-agent")]
impl std::error::Error for NanoPhysicalHeadGazeRuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Intrinsics(source) => Some(source),
            Self::Proposal(source) => Some(source),
            Self::ProposalId(source) => Some(source),
            Self::ProposalCommand(source) => Some(source),
            Self::ProposalRejected(source) => Some(source),
            Self::BaseInterlock(source) => Some(source),
            Self::ServiceCommand(source) => Some(source),
            Self::Service(source) => Some(source.as_ref()),
            Self::PetEpisode(source) => Some(source),
            Self::CommandedGaze(source) => Some(source),
            Self::GazeResidual(source) => Some(source),
            Self::ProjectionGridMismatch { .. }
            | Self::ProposalSequenceExhausted { .. }
            | Self::PetEpisodeTimestampOutOfRange { .. }
            | Self::PetContactCandidateHasZeroSamples
            | Self::PetRecoveryProgressOutOfRange { .. }
            | Self::CommandedTargetUnavailable => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum NanoHeadGazePortError {
    #[cfg(feature = "nano-agent")]
    Physical(NanoPhysicalHeadGazeRuntimeError),
}

#[derive(Clone, Debug)]
pub enum NanoAccessoryTerminalFault {
    HeadHealth(NanoHeadHealthError),
    #[cfg(feature = "nano-agent")]
    PhysicalHeadGaze(NanoPhysicalHeadGazeRuntimeError),
    HeadHealthStatusPoisoned,
    RgbHealthStatusPoisoned,
    ExpressionBridge(RgbExpressionBridgeError),
    #[cfg(feature = "nano-agent")]
    FacePerception(NanoFacePerceptionRuntimeError),
    EyeApply(EyeHandleRequestError),
    #[cfg(feature = "nano-agent")]
    PetEvidence(NanoPetEvidenceJournalRuntimeError),
    CharacterCoordination(TrackingEyeGazeReplacementError),
    LifecyclePresentationFrameCountExhausted,
    RgbIngressDisconnected,
    RgbChannelPoisoned,
    ReadinessObserverDropped,
}

/// A failure encountered while presenting the character-owned fault reflex
/// after the first accessory fault and all stop latches were already
/// published.
///
/// This is deliberately not a [`NanoAccessoryTerminalFault`]: it is secondary
/// shutdown evidence and must never replace the first fault that stopped the
/// base.
#[derive(Debug)]
pub enum NanoFaultRecoveryPresentationFault {
    MissingPresentationDuration,
    FrameCountExhausted,
    ExpressionBridge(RgbExpressionBridgeError),
    #[cfg(feature = "nano-agent")]
    FacePerception(NanoFacePerceptionRuntimeError),
    EyeApply(EyeHandleRequestError),
}

impl fmt::Display for NanoFaultRecoveryPresentationFault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano fault-recovery presentation failed: {self:?}"
        )
    }
}

impl std::error::Error for NanoFaultRecoveryPresentationFault {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ExpressionBridge(source) => Some(source),
            #[cfg(feature = "nano-agent")]
            Self::FacePerception(source) => Some(source),
            Self::EyeApply(source) => Some(source),
            Self::MissingPresentationDuration | Self::FrameCountExhausted => None,
        }
    }
}

/// Exact terminal fault-reflex outcome retained with coordinated accessory
/// shutdown evidence.
#[derive(Debug)]
pub enum NanoFaultRecoveryPresentationEvidence {
    /// No terminal fault occurred after both accessory actors were admitted.
    NotRequired,
    /// At least one fresh eye frame was acknowledged after the stop latch and
    /// the complete bounded transition reached its endpoint.
    Presented { frames_applied: NonZeroU64 },
    /// The primary stop remains authoritative. `frames_applied` records any
    /// fresh frames acknowledged before this secondary failure.
    Failed {
        frames_applied: u64,
        source: NanoFaultRecoveryPresentationFault,
    },
}

impl NanoFaultRecoveryPresentationEvidence {
    pub const fn frames_applied(&self) -> u64 {
        match self {
            Self::NotRequired => 0,
            Self::Presented { frames_applied } => frames_applied.get(),
            Self::Failed { frames_applied, .. } => *frames_applied,
        }
    }

    pub const fn completed(&self) -> bool {
        matches!(self, Self::Presented { .. })
    }
}

impl fmt::Display for NanoAccessoryTerminalFault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano accessory worker terminal fault: {self:?}")
    }
}

impl std::error::Error for NanoAccessoryTerminalFault {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::HeadHealth(source) => Some(source),
            #[cfg(feature = "nano-agent")]
            Self::PhysicalHeadGaze(source) => Some(source),
            Self::ExpressionBridge(source) => Some(source),
            #[cfg(feature = "nano-agent")]
            Self::FacePerception(source) => Some(source),
            Self::EyeApply(source) => Some(source),
            #[cfg(feature = "nano-agent")]
            Self::PetEvidence(source) => Some(source),
            Self::CharacterCoordination(source) => Some(source),
            Self::HeadHealthStatusPoisoned
            | Self::RgbHealthStatusPoisoned
            | Self::LifecyclePresentationFrameCountExhausted
            | Self::RgbIngressDisconnected
            | Self::RgbChannelPoisoned
            | Self::ReadinessObserverDropped => None,
        }
    }
}

#[derive(Debug)]
pub enum NanoEyeActorStartupError {
    Start(EyeActorStartError),
    Receipt {
        source: EyeStartupReceiptError,
        actor: Result<EyeActorExit, JoinError>,
    },
    Runtime {
        source: Box<EyeRuntimeFault>,
        actor: Result<EyeActorExit, JoinError>,
    },
}

impl fmt::Display for NanoEyeActorStartupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "KEP2 eye actor startup failed: {self:?}")
    }
}

impl std::error::Error for NanoEyeActorStartupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Start(source) => Some(source),
            Self::Receipt { source, .. } => Some(source),
            Self::Runtime { source, .. } => Some(source.as_ref()),
        }
    }
}

#[derive(Debug)]
pub enum NanoHeadActorStartupError {
    Start(HeadActorStartError),
    Receipt {
        source: HeadStartupReceiptError,
        actor: Result<HeadActorExit, JoinError>,
    },
    Runtime {
        source: Box<HeadRuntimeError>,
        actor: Result<HeadActorExit, JoinError>,
    },
    ReturnCommand {
        source: HeadCommandError,
        startup: Box<VerifiedNaturalHoldEvidence>,
        hold_preserving_release: Result<HoldPreservingOwnershipReleaseEvidence, HeadShutdownError>,
        actor: Result<HeadActorExit, JoinError>,
    },
    Return {
        source: Box<HeadReturnError>,
        startup: Box<VerifiedNaturalHoldEvidence>,
        hold_preserving_release: Result<HoldPreservingOwnershipReleaseEvidence, HeadShutdownError>,
        actor: Result<HeadActorExit, JoinError>,
    },
    PostReturnHealth {
        source: NanoHeadHealthError,
        startup: Box<VerifiedNaturalHoldEvidence>,
        head_return: Box<VerifiedHeadReturnEvidence>,
        hold_preserving_release: Result<HoldPreservingOwnershipReleaseEvidence, HeadShutdownError>,
        actor: Result<HeadActorExit, JoinError>,
    },
}

impl fmt::Display for NanoHeadActorStartupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "return-to-natural head actor startup failed: {self:?}"
        )
    }
}

impl std::error::Error for NanoHeadActorStartupError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Start(source) => Some(source),
            Self::Receipt { source, .. } => Some(source),
            Self::Runtime { source, .. } => Some(source.as_ref()),
            Self::ReturnCommand { source, .. } => Some(source),
            Self::Return { source, .. } => Some(source.as_ref()),
            Self::PostReturnHealth { source, .. } => Some(source),
        }
    }
}

/// Raw eye release request and actor-exit evidence. `release_verified` is
/// deliberately a derived predicate rather than a replacement for either.
#[derive(Debug)]
pub struct NanoEyeShutdownEvidence {
    startup: EyeStartupEvidence,
    release: Result<ReleaseReport, EyeHandleRequestError>,
    actor: Result<EyeActorExit, JoinError>,
}

impl NanoEyeShutdownEvidence {
    pub const fn startup(&self) -> &EyeStartupEvidence {
        &self.startup
    }

    pub const fn release(&self) -> &Result<ReleaseReport, EyeHandleRequestError> {
        &self.release
    }

    pub const fn actor(&self) -> &Result<EyeActorExit, JoinError> {
        &self.actor
    }

    pub fn release_verified(&self) -> bool {
        let (Ok(release), Ok(actor)) = (&self.release, &self.actor) else {
            return false;
        };
        matches!(release, ReleaseReport::Released(_))
            && actor.startup().as_ref() == Ok(&self.startup)
            && actor.release() == Some(release)
            && matches!(actor.termination(), EyeActorTermination::RequestedShutdown)
    }
}

/// Raw production head ownership-release evidence.
///
/// Ordinary agent shutdown never writes the torque switch. This preserves the
/// last admitted natural hold across serial close but does not prove the
/// physical torque state; electrical power loss or another owner can still
/// release the neck.
#[derive(Debug)]
pub struct NanoHeadShutdownEvidence {
    startup: VerifiedNaturalHoldEvidence,
    head_return: VerifiedHeadReturnEvidence,
    hold_preserving_release: Result<HoldPreservingOwnershipReleaseEvidence, HeadShutdownError>,
    actor: Result<HeadActorExit, JoinError>,
}

impl NanoHeadShutdownEvidence {
    pub const fn startup(&self) -> &VerifiedNaturalHoldEvidence {
        &self.startup
    }

    pub const fn head_return(&self) -> &VerifiedHeadReturnEvidence {
        &self.head_return
    }

    pub const fn hold_preserving_release(
        &self,
    ) -> &Result<HoldPreservingOwnershipReleaseEvidence, HeadShutdownError> {
        &self.hold_preserving_release
    }

    pub const fn actor(&self) -> &Result<HeadActorExit, JoinError> {
        &self.actor
    }

    pub fn hold_preserving_release_completed(&self) -> bool {
        let (Ok(release), Ok(actor)) = (&self.hold_preserving_release, &self.actor) else {
            return false;
        };
        actor.startup().as_ref() == Ok(&self.startup)
            && matches!(
                actor.head_return(),
                Some(Ok(head_return)) if head_return == &self.head_return
            )
            && actor.hold_preserving_release() == release
            && matches!(
                actor.termination(),
                HeadActorTermination::RequestedHoldPreservingRelease
            )
    }
}

#[derive(Debug)]
pub struct NanoAccessoryShutdownEvidence {
    eye: NanoEyeShutdownEvidence,
    head: NanoHeadShutdownEvidence,
    fault_recovery_presentation: NanoFaultRecoveryPresentationEvidence,
    #[cfg(feature = "nano-agent")]
    pet_evidence: NanoPetEvidenceLifecycleShutdownEvidence,
    #[cfg(feature = "nano-agent")]
    face_perception: NanoFacePerceptionShutdownEvidence,
}

impl NanoAccessoryShutdownEvidence {
    pub const fn eye(&self) -> &NanoEyeShutdownEvidence {
        &self.eye
    }

    pub const fn head(&self) -> &NanoHeadShutdownEvidence {
        &self.head
    }

    pub const fn fault_recovery_presentation(&self) -> &NanoFaultRecoveryPresentationEvidence {
        &self.fault_recovery_presentation
    }

    #[cfg(feature = "nano-agent")]
    pub const fn pet_evidence(&self) -> &NanoPetEvidenceLifecycleShutdownEvidence {
        &self.pet_evidence
    }

    #[cfg(feature = "nano-agent")]
    pub const fn face_perception(&self) -> &NanoFacePerceptionShutdownEvidence {
        &self.face_perception
    }

    #[cfg(feature = "nano-agent")]
    pub fn into_parts(
        self,
    ) -> (
        NanoEyeShutdownEvidence,
        NanoHeadShutdownEvidence,
        NanoFaultRecoveryPresentationEvidence,
        NanoPetEvidenceLifecycleShutdownEvidence,
        NanoFacePerceptionShutdownEvidence,
    ) {
        (
            self.eye,
            self.head,
            self.fault_recovery_presentation,
            self.pet_evidence,
            self.face_perception,
        )
    }
}

/// Exact lifecycle result for the optional pet-evidence writer. Production
/// enables it; qualification paths retain an explicit `Disabled` value.
#[cfg(feature = "nano-agent")]
#[derive(Debug)]
pub enum NanoPetEvidenceLifecycleShutdownEvidence {
    Disabled,
    Enabled(NanoPetEvidenceJournalShutdownEvidence),
}

#[cfg(feature = "nano-agent")]
impl NanoPetEvidenceLifecycleShutdownEvidence {
    pub fn clean(&self) -> bool {
        match self {
            Self::Disabled => true,
            Self::Enabled(evidence) => evidence.clean(),
        }
    }
}

#[derive(Debug)]
pub enum NanoAccessoryWorkerExit {
    RuntimeBuildFailed {
        message: Box<str>,
    },
    EyeSessionMaterialFailed(OsEyeSessionMaterialError),
    #[cfg(feature = "nano-agent")]
    PetEvidenceJournalStartupFailed(NanoPetEvidenceJournalStartError),
    EyeStartupFailed {
        source: Box<NanoEyeActorStartupError>,
        #[cfg(feature = "nano-agent")]
        pet_evidence: NanoPetEvidenceLifecycleShutdownEvidence,
    },
    HeadStartupFailed {
        source: Box<NanoHeadActorStartupError>,
        eye_shutdown: Box<NanoEyeShutdownEvidence>,
        #[cfg(feature = "nano-agent")]
        pet_evidence: NanoPetEvidenceLifecycleShutdownEvidence,
    },
    Shutdown {
        terminal_fault: Option<NanoAccessoryTerminalFault>,
        evidence: Box<NanoAccessoryShutdownEvidence>,
    },
    #[cfg(feature = "nano-agent")]
    UnexpectedAccessoryExitWithFaceEvidence {
        accessory: Box<NanoAccessoryWorkerExit>,
        face_perception: NanoFacePerceptionJoinEvidence,
    },
}

impl fmt::Display for NanoAccessoryWorkerExit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano accessory worker exited: {self:?}")
    }
}

impl std::error::Error for NanoAccessoryWorkerExit {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RuntimeBuildFailed { .. } => None,
            Self::EyeSessionMaterialFailed(source) => Some(source),
            #[cfg(feature = "nano-agent")]
            Self::PetEvidenceJournalStartupFailed(source) => Some(source),
            Self::EyeStartupFailed { source, .. } => Some(source.as_ref()),
            Self::HeadStartupFailed { source, .. } => Some(source.as_ref()),
            Self::Shutdown {
                terminal_fault: Some(source),
                ..
            } => Some(source),
            Self::Shutdown {
                terminal_fault: None,
                ..
            } => None,
            #[cfg(feature = "nano-agent")]
            Self::UnexpectedAccessoryExitWithFaceEvidence { accessory, .. } => {
                Some(accessory.as_ref())
            }
        }
    }
}

#[derive(Debug)]
pub enum NanoAccessoryWorkerStartError {
    #[cfg(feature = "nano-agent")]
    PhysicalHeadGazeRequiresFacePerception,
    ThreadSpawn(std::io::Error),
    StartupFailed(Box<NanoAccessoryWorkerExit>),
    ThreadPanickedBeforeReadiness,
    #[cfg(feature = "nano-agent")]
    FacePerceptionConfig(NanoFacePerceptionConfigError),
    #[cfg(feature = "nano-agent")]
    FacePerceptionThreadSpawn(std::io::Error),
    #[cfg(feature = "nano-agent")]
    AccessoryThreadSpawnWithFace {
        source: std::io::Error,
        face_perception: NanoFacePerceptionJoinEvidence,
    },
    #[cfg(feature = "nano-agent")]
    FacePerceptionStartupFailed(NanoFacePerceptionJoinEvidence),
    #[cfg(feature = "nano-agent")]
    FacePerceptionStartupTimedOut {
        timeout: Duration,
        cleanup: NanoFacePerceptionJoinEvidence,
    },
    #[cfg(feature = "nano-agent")]
    AccessoryStartupFailedWithFace {
        accessory: Box<NanoAccessoryWorkerExit>,
        face_perception: NanoFacePerceptionJoinEvidence,
    },
    #[cfg(feature = "nano-agent")]
    AccessoryThreadPanickedBeforeReadinessWithFace {
        face_perception: NanoFacePerceptionJoinEvidence,
    },
}

impl fmt::Display for NanoAccessoryWorkerStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano accessory worker did not become ready: {self:?}"
        )
    }
}

impl std::error::Error for NanoAccessoryWorkerStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ThreadSpawn(source) => Some(source),
            #[cfg(feature = "nano-agent")]
            Self::FacePerceptionConfig(source) => Some(source),
            #[cfg(feature = "nano-agent")]
            Self::FacePerceptionThreadSpawn(source) => Some(source),
            #[cfg(feature = "nano-agent")]
            Self::AccessoryThreadSpawnWithFace { source, .. } => Some(source),
            Self::StartupFailed(source) => Some(source.as_ref()),
            Self::ThreadPanickedBeforeReadiness => None,
            #[cfg(feature = "nano-agent")]
            Self::PhysicalHeadGazeRequiresFacePerception => None,
            #[cfg(feature = "nano-agent")]
            Self::FacePerceptionStartupFailed(source) => Some(source),
            #[cfg(feature = "nano-agent")]
            Self::FacePerceptionStartupTimedOut { .. } => None,
            #[cfg(feature = "nano-agent")]
            Self::AccessoryStartupFailedWithFace { accessory, .. } => Some(accessory.as_ref()),
            #[cfg(feature = "nano-agent")]
            Self::AccessoryThreadPanickedBeforeReadinessWithFace { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum NanoAccessoryWorkerJoinError {
    ThreadPanicked,
    #[cfg(feature = "nano-agent")]
    ThreadPanickedWithFaceEvidence {
        face_perception: NanoFacePerceptionJoinEvidence,
    },
}

impl fmt::Display for NanoAccessoryWorkerJoinError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano accessory worker join failed: {self:?}")
    }
}

impl std::error::Error for NanoAccessoryWorkerJoinError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAccessoryFaultWaitError {
    Timeout,
    PublisherDisconnected,
}

impl fmt::Display for NanoAccessoryFaultWaitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano accessory fault monitor failed: {self:?}")
    }
}

impl std::error::Error for NanoAccessoryFaultWaitError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAccessoryOwnerState {
    Starting,
    Running,
    FaultLatched,
    ShuttingDown,
    Stopped,
    Detached,
    OwnerExitedUnexpectedly,
}

impl NanoAccessoryOwnerState {
    const fn encoded(self) -> u8 {
        match self {
            Self::Starting => 0,
            Self::Running => 1,
            Self::FaultLatched => 2,
            Self::ShuttingDown => 3,
            Self::Stopped => 4,
            Self::Detached => 5,
            Self::OwnerExitedUnexpectedly => 6,
        }
    }

    fn decode(value: u8) -> Self {
        match value {
            0 => Self::Starting,
            1 => Self::Running,
            2 => Self::FaultLatched,
            3 => Self::ShuttingDown,
            4 => Self::Stopped,
            5 => Self::Detached,
            6 => Self::OwnerExitedUnexpectedly,
            _ => Self::Detached,
        }
    }
}

#[derive(Debug)]
struct NanoAccessoryOwnerLifecycle {
    state: AtomicU8,
    completed_health_transactions: AtomicU64,
}

impl NanoAccessoryOwnerLifecycle {
    fn starting() -> Self {
        Self {
            state: AtomicU8::new(NanoAccessoryOwnerState::Starting.encoded()),
            completed_health_transactions: AtomicU64::new(0),
        }
    }

    fn state(&self) -> NanoAccessoryOwnerState {
        NanoAccessoryOwnerState::decode(self.state.load(Ordering::Acquire))
    }

    fn mark_running(&self) {
        if self
            .state
            .compare_exchange(
                NanoAccessoryOwnerState::Starting.encoded(),
                NanoAccessoryOwnerState::Running.encoded(),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            // Readiness is published only after the first complete four-joint
            // health transaction. Count that transaction as the first proof
            // that the sole owner loop reached a bounded completion boundary.
            self.completed_health_transactions
                .store(1, Ordering::Release);
        }
    }

    fn record_completed_health_transaction(&self) -> bool {
        self.completed_health_transactions
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .is_ok()
    }

    fn liveness_snapshot(&self) -> NanoAccessoryLoopLivenessSnapshot {
        NanoAccessoryLoopLivenessSnapshot {
            owner_state: self.state(),
            completed_health_transactions: self
                .completed_health_transactions
                .load(Ordering::Acquire),
        }
    }

    fn mark_fault_latched(&self) {
        let mut current = self.state.load(Ordering::Acquire);
        loop {
            if !matches!(
                NanoAccessoryOwnerState::decode(current),
                NanoAccessoryOwnerState::Starting | NanoAccessoryOwnerState::Running
            ) {
                return;
            }
            match self.state.compare_exchange_weak(
                current,
                NanoAccessoryOwnerState::FaultLatched.encoded(),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    fn mark_shutting_down(&self) {
        self.state.store(
            NanoAccessoryOwnerState::ShuttingDown.encoded(),
            Ordering::Release,
        );
    }

    fn mark_stopped(&self) {
        self.state.store(
            NanoAccessoryOwnerState::Stopped.encoded(),
            Ordering::Release,
        );
    }

    fn mark_detached_if_live(&self) {
        let mut current = self.state.load(Ordering::Acquire);
        loop {
            if matches!(
                NanoAccessoryOwnerState::decode(current),
                NanoAccessoryOwnerState::Stopped | NanoAccessoryOwnerState::OwnerExitedUnexpectedly
            ) {
                return;
            }
            match self.state.compare_exchange_weak(
                current,
                NanoAccessoryOwnerState::Detached.encoded(),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    fn mark_owner_exited_unexpectedly_if_live(&self) {
        let mut current = self.state.load(Ordering::Acquire);
        loop {
            if matches!(
                NanoAccessoryOwnerState::decode(current),
                NanoAccessoryOwnerState::ShuttingDown
                    | NanoAccessoryOwnerState::Stopped
                    | NanoAccessoryOwnerState::Detached
                    | NanoAccessoryOwnerState::OwnerExitedUnexpectedly
            ) {
                return;
            }
            match self.state.compare_exchange_weak(
                current,
                NanoAccessoryOwnerState::OwnerExitedUnexpectedly.encoded(),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }
}

/// Monotonic proof that the sole accessory owner completed bounded work.
///
/// The counter advances only after a complete four-joint servo-health
/// transaction. Because the same single-threaded actor also owns expression
/// bridging, gaze/compliance, and KEP2 writes, a stuck operation on any of
/// those paths prevents the next health transaction and therefore prevents a
/// false heartbeat. This is observation only; it carries no device handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoAccessoryLoopLivenessSnapshot {
    pub owner_state: NanoAccessoryOwnerState,
    pub completed_health_transactions: u64,
}

/// Cloneable observer for process supervision; never a second device owner.
#[derive(Clone, Debug)]
pub struct NanoAccessoryLoopLivenessObserver {
    lifecycle: Arc<NanoAccessoryOwnerLifecycle>,
}

impl NanoAccessoryLoopLivenessObserver {
    pub fn snapshot(&self) -> NanoAccessoryLoopLivenessSnapshot {
        self.lifecycle.liveness_snapshot()
    }
}

struct NanoAccessoryThreadLifecycleGuard {
    lifecycle: Arc<NanoAccessoryOwnerLifecycle>,
}

impl NanoAccessoryThreadLifecycleGuard {
    fn new(lifecycle: Arc<NanoAccessoryOwnerLifecycle>) -> Self {
        Self { lifecycle }
    }
}

impl Drop for NanoAccessoryThreadLifecycleGuard {
    fn drop(&mut self) {
        self.lifecycle.mark_owner_exited_unexpectedly_if_live();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAccessoryHealthStatusError {
    Poisoned,
    OwnerNotRunning { state: NanoAccessoryOwnerState },
    IngressDisconnected,
    ChannelPoisoned,
}

impl fmt::Display for NanoAccessoryHealthStatusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano accessory runtime health status unavailable: {self:?}"
        )
    }
}

impl std::error::Error for NanoAccessoryHealthStatusError {}

#[derive(Debug)]
struct NanoAccessoryHeadHealthState {
    evidence: Option<VerifiedHeadHealthEvidence>,
    last_success_observed_at: Option<Instant>,
}

impl NanoAccessoryHeadHealthState {
    const fn empty() -> Self {
        Self {
            evidence: None,
            last_success_observed_at: None,
        }
    }

    fn record(&mut self, evidence: VerifiedHeadHealthEvidence) {
        self.evidence = Some(evidence);
        self.last_success_observed_at = Some(Instant::now());
    }
}

/// Live status of one accessory subsystem derived from the running sole owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAccessoryComponentHealth {
    Ready,
    Degraded,
    Faulted,
}

/// One point-in-time view of the running accessory owner.
///
/// `Ready` for the head requires a successful complete four-joint health
/// transaction no older than three configured health periods. Eye readiness
/// means the KEP2 actor remains owned and no terminal worker fault is latched.
/// The expression frame count is receipt-backed: it advances only after a
/// complete RGB bridge operation and successful eye acknowledgement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoAccessoryRuntimeHealth {
    pub head: NanoAccessoryComponentHealth,
    pub eyes: NanoAccessoryComponentHealth,
    pub rgb_expression: NanoAccessoryComponentHealth,
    pub successful_rgb_expression_frames: u64,
}

/// Cloneable, read-only health view that does not duplicate serial ownership.
#[derive(Clone)]
pub struct NanoAccessoryHealthObserver {
    channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>>,
    head: Arc<Mutex<NanoAccessoryHeadHealthState>>,
    lifecycle: Arc<NanoAccessoryOwnerLifecycle>,
    health_period: NanoAccessoryHealthPeriod,
    rgb_frame_freshness: Duration,
}

impl NanoAccessoryHealthObserver {
    pub fn snapshot(&self) -> Result<NanoAccessoryRuntimeHealth, NanoAccessoryHealthStatusError> {
        let frames = self.channel.counters.snapshot();
        let owner_state = self.lifecycle.state();
        if matches!(
            owner_state,
            NanoAccessoryOwnerState::Starting
                | NanoAccessoryOwnerState::ShuttingDown
                | NanoAccessoryOwnerState::Stopped
                | NanoAccessoryOwnerState::Detached
                | NanoAccessoryOwnerState::OwnerExitedUnexpectedly
        ) {
            return Err(NanoAccessoryHealthStatusError::OwnerNotRunning { state: owner_state });
        }
        if owner_state == NanoAccessoryOwnerState::FaultLatched {
            return Ok(NanoAccessoryRuntimeHealth {
                head: NanoAccessoryComponentHealth::Faulted,
                eyes: NanoAccessoryComponentHealth::Faulted,
                rgb_expression: NanoAccessoryComponentHealth::Faulted,
                successful_rgb_expression_frames: frames.processed_successfully,
            });
        }
        if self.channel.poisoned.load(Ordering::Acquire) {
            return Err(NanoAccessoryHealthStatusError::ChannelPoisoned);
        }
        if self.channel.shutdown_requested.load(Ordering::Acquire) {
            return Err(NanoAccessoryHealthStatusError::OwnerNotRunning {
                state: NanoAccessoryOwnerState::ShuttingDown,
            });
        }
        // Face fault publication closes raw admission under the ingress-slot
        // mutex before the accessory actor can consume that terminal value
        // and update its lifecycle latch. Admission closure is therefore
        // itself authoritative fault evidence once non-running and poison
        // states have been given precedence.
        if !self.channel.accepting_frames.load(Ordering::Acquire) {
            return Ok(NanoAccessoryRuntimeHealth {
                head: NanoAccessoryComponentHealth::Faulted,
                eyes: NanoAccessoryComponentHealth::Faulted,
                rgb_expression: NanoAccessoryComponentHealth::Faulted,
                successful_rgb_expression_frames: frames.processed_successfully,
            });
        }
        if !self.channel.ingress_alive.load(Ordering::Acquire) {
            return Err(NanoAccessoryHealthStatusError::IngressDisconnected);
        }
        let head = self
            .head
            .lock()
            .map_err(|_| NanoAccessoryHealthStatusError::Poisoned)?;
        let maximum_age = self
            .health_period
            .get()
            .checked_mul(3)
            .unwrap_or(Duration::MAX);
        let head = match head.last_success_observed_at {
            Some(observed_at)
                if Instant::now()
                    .checked_duration_since(observed_at)
                    .is_some_and(|age| age <= maximum_age) =>
            {
                NanoAccessoryComponentHealth::Ready
            }
            Some(_) | None => NanoAccessoryComponentHealth::Degraded,
        };
        let rgb_expression = match self.channel.counters.last_processed_successfully_at()? {
            Some(observed_at)
                if Instant::now()
                    .checked_duration_since(observed_at)
                    .is_some_and(|age| age < self.rgb_frame_freshness) =>
            {
                NanoAccessoryComponentHealth::Ready
            }
            Some(_) | None => NanoAccessoryComponentHealth::Degraded,
        };
        Ok(NanoAccessoryRuntimeHealth {
            head,
            eyes: NanoAccessoryComponentHealth::Ready,
            rgb_expression,
            successful_rgb_expression_frames: frames.processed_successfully,
        })
    }
}

enum StartupSignal {
    Ready(Box<NanoAccessoryReadyEvidence>),
    Failed,
}

#[cfg(feature = "nano-agent")]
enum FacePerceptionStartupSignal {
    Ready(Box<NanoFacePerceptionReadyEvidence>),
    Failed,
}

/// Running worker plus its sole RGB ingress.
///
/// # Drop behavior
///
/// Accidental `Drop` disconnects RGB and detaches the actor thread plus, in the
/// production graph, the perception thread. Perception can observe the
/// disconnect only if any in-flight native call returns. If it reaches that
/// boundary, it attempts one typed terminal publication and exits; the
/// detached actor can then latch the first terminal result, keep owning the
/// head bus, and continue bounded health checks. `Drop` proves neither
/// publication nor thread exit. It does **not** request torque disable.
/// The in-object fault receiver is dropped too, so no caller can observe that
/// publication or later coordinate shutdown through this worker. Process
/// termination still does not prove the resulting physical torque state. A
/// service owner must retain this value and call [`Self::shutdown`].
#[must_use = "the accessory owner must be retained and explicitly shut down"]
pub struct NanoAccessoryWorker {
    ready: NanoAccessoryReadyEvidence,
    ingress: Option<NanoAccessoryRgbIngress>,
    channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>>,
    lifecycle: Arc<NanoAccessoryOwnerLifecycle>,
    latest_head_health: Arc<Mutex<NanoAccessoryHeadHealthState>>,
    fault_rx: crossbeam_channel::Receiver<NanoAccessoryTerminalFault>,
    thread: Option<JoinHandle<NanoAccessoryWorkerExit>>,
    #[cfg(feature = "nano-agent")]
    face_output: Option<Arc<LatestFrameChannel<NanoFacePerceptionWork>>>,
    #[cfg(feature = "nano-agent")]
    face_thread: Option<JoinHandle<NanoFacePerceptionThreadExit>>,
    #[cfg(feature = "nano-agent")]
    face_diagnostics: Option<(NanoFaceDiagnosticReceiver, NanoFaceDiagnosticStatsHandle)>,
    #[cfg(feature = "nano-agent")]
    face_diagnostics_active: Option<Arc<AtomicBool>>,
    #[cfg(feature = "nano-agent")]
    face_stage_counters: Option<Arc<NanoFacePerceptionStageCounters>>,
    #[cfg(feature = "nano-agent")]
    head_gaze_lease_issuer: Option<Arc<OnceLock<HeadGazeBaseZeroExclusiveLeaseIssuer>>>,
}

impl NanoAccessoryWorker {
    /// Start the scene-motion-only compatibility runtime.
    ///
    /// Canonical production bootstrap does not call this path; it uses the
    /// navigation-private V3 face-enabled constructor below. This constructor
    /// remains public for commissioning/qualification feature graphs which do
    /// not include the production detector lane.
    #[cfg(any(
        feature = "nano-wheels-off-qualification",
        feature = "nano-base-commissioning",
        test
    ))]
    pub fn start_scene_motion_only(
        config: NanoAccessoryWorkerConfig,
    ) -> Result<Self, NanoAccessoryWorkerStartError> {
        #[cfg(feature = "nano-agent")]
        if config.physical_head_gaze.is_some() {
            return Err(NanoAccessoryWorkerStartError::PhysicalHeadGazeRequiresFacePerception);
        }
        let expression_clock = NanoAccessoryClock::new();
        let channel = Arc::new(LatestFrameChannel::new());
        let ingress = NanoAccessoryRgbIngress {
            channel: Arc::clone(&channel),
            clock: expression_clock.clone(),
            connected: true,
        };
        let (startup_tx, startup_rx) = std::sync::mpsc::sync_channel(1);
        let (fault_tx, fault_rx) = crossbeam_channel::bounded(1);
        let lifecycle = Arc::new(NanoAccessoryOwnerLifecycle::starting());
        let worker_lifecycle = Arc::clone(&lifecycle);
        let worker_channel = Arc::clone(&channel);
        let latest_head_health = Arc::new(Mutex::new(NanoAccessoryHeadHealthState::empty()));
        let worker_head_health = Arc::clone(&latest_head_health);
        let thread = thread::Builder::new()
            .name("kiko-nano-accessories".into())
            .spawn(move || {
                let _lifecycle_guard =
                    NanoAccessoryThreadLifecycleGuard::new(Arc::clone(&worker_lifecycle));
                run_production_worker(
                    config,
                    worker_channel,
                    worker_head_health,
                    startup_tx,
                    fault_tx,
                    expression_clock,
                    NanoAccessoryPerceptionReadyEvidence::SceneMotionOnly,
                    worker_lifecycle,
                )
            })
            .map_err(NanoAccessoryWorkerStartError::ThreadSpawn)?;

        match startup_rx.recv() {
            Ok(StartupSignal::Ready(ready)) => Ok(Self {
                ready: *ready,
                ingress: Some(ingress),
                channel,
                lifecycle,
                latest_head_health,
                fault_rx,
                thread: Some(thread),
                #[cfg(feature = "nano-agent")]
                face_output: None,
                #[cfg(feature = "nano-agent")]
                face_thread: None,
                #[cfg(feature = "nano-agent")]
                face_diagnostics: None,
                #[cfg(feature = "nano-agent")]
                face_diagnostics_active: None,
                #[cfg(feature = "nano-agent")]
                face_stage_counters: None,
                #[cfg(feature = "nano-agent")]
                head_gaze_lease_issuer: None,
            }),
            Ok(StartupSignal::Failed) | Err(_) => {
                drop(ingress);
                match thread.join() {
                    Ok(exit) => Err(NanoAccessoryWorkerStartError::StartupFailed(Box::new(exit))),
                    Err(_) => Err(NanoAccessoryWorkerStartError::ThreadPanickedBeforeReadiness),
                }
            }
        }
    }

    /// Start face perception from already loaded cascade assets and only then
    /// start the head/eye owner.
    ///
    /// This boundary does not infer cascade roles or trust file paths. The
    /// caller must pass assets which its launch/bootstrap boundary has already
    /// bound to the frontal and profile roles respectively. The retained bytes
    /// move once into the private face-perception asset wrapper; neither path
    /// is reopened.
    #[cfg(feature = "nano-agent")]
    pub fn start_with_loaded_face_perception(
        config: NanoAccessoryWorkerConfig,
        frontal_face_cascade: LoadedDeploymentAsset,
        profile_face_cascade: LoadedDeploymentAsset,
    ) -> Result<Self, NanoAccessoryWorkerStartError> {
        Self::start_with_face_perception(
            config,
            NanoFacePerceptionAssets::from_loaded_assets(
                frontal_face_cascade,
                profile_face_cascade,
            ),
        )
    }

    /// Start the face-perception lane and only then start the head/eye owner.
    ///
    /// The exact retained cascade byte vectors move into the named perception
    /// thread. `NanoFacePerception` is constructed and retained there, so its
    /// CXX-backed `!Send` owner never crosses a thread boundary. Detector load
    /// must produce typed readiness before the actor thread is spawned. If it
    /// does not report within [`NANO_FACE_PERCEPTION_STARTUP_TIMEOUT`], startup
    /// requests shutdown, performs one bounded join, and returns explicit
    /// timeout/detachment evidence; a native call is never claimed cancelled.
    #[cfg(feature = "nano-agent")]
    pub(super) fn start_with_face_perception(
        mut config: NanoAccessoryWorkerConfig,
        assets: NanoFacePerceptionAssets,
    ) -> Result<Self, NanoAccessoryWorkerStartError> {
        let detector_config = canonical_nano_face_detector_config()
            .map_err(NanoAccessoryWorkerStartError::FacePerceptionConfig)?;
        let tracking_config = FaceTrackingConfig::default();
        let expression_clock = NanoAccessoryClock::new();
        let channel = Arc::new(LatestFrameChannel::new());
        let face_output = Arc::new(LatestFrameChannel::with_shared_receipts(Arc::clone(
            &channel.counters.receipts,
        )));
        let ingress = NanoAccessoryRgbIngress {
            channel: Arc::clone(&channel),
            clock: expression_clock.clone(),
            connected: true,
        };
        let (face_startup_tx, face_startup_rx) = std::sync::mpsc::sync_channel(1);
        let (face_diagnostic_tx, face_diagnostic_rx, face_diagnostic_stats) = bounded_channel(
            ChannelCapacity::try_from(1_usize).expect("one is a nonzero channel capacity"),
            DropPolicy::DropOldest,
        );
        let face_diagnostics_active = Arc::new(AtomicBool::new(false));
        let face_thread_diagnostics_active = Arc::clone(&face_diagnostics_active);
        let face_stage_counters = Arc::new(NanoFacePerceptionStageCounters::new());
        let face_thread_stage_counters = Arc::clone(&face_stage_counters);
        let face_input = Arc::clone(&channel);
        let face_thread_output = Arc::clone(&face_output);
        let face_clock = expression_clock.clone();
        let stream_epoch = config.stream_epoch;
        let freshness = config.rgb_expression.frame_freshness();
        let head_gaze_diagnostics = config.head_gaze_diagnostics.take();
        let head_gaze_lease_issuer = config
            .physical_head_gaze
            .as_ref()
            .map(|physical| Arc::clone(&physical.lease_issuer));
        let face_thread = thread::Builder::new()
            .name("kiko-nano-face-perception".into())
            .spawn(move || {
                run_face_perception_thread_catching_panics(
                    assets,
                    detector_config,
                    tracking_config,
                    stream_epoch,
                    freshness,
                    face_clock,
                    face_input,
                    face_thread_output,
                    face_diagnostic_tx,
                    face_thread_diagnostics_active,
                    face_thread_stage_counters,
                    face_startup_tx,
                    head_gaze_diagnostics,
                )
            })
            .map_err(NanoAccessoryWorkerStartError::FacePerceptionThreadSpawn)?;

        let face_ready = match face_startup_rx.recv_timeout(NANO_FACE_PERCEPTION_STARTUP_TIMEOUT) {
            Ok(FacePerceptionStartupSignal::Ready(ready)) => *ready,
            Ok(FacePerceptionStartupSignal::Failed)
            | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                channel.request_shutdown();
                face_output.request_shutdown();
                drop(ingress);
                drop(face_startup_rx);
                let exit = join_face_perception_thread_bounded(
                    face_thread,
                    NANO_FACE_PERCEPTION_JOIN_TIMEOUT,
                    NANO_FACE_PERCEPTION_JOIN_TIMEOUT,
                );
                return Err(NanoAccessoryWorkerStartError::FacePerceptionStartupFailed(
                    exit,
                ));
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                channel.request_shutdown();
                face_output.request_shutdown();
                drop(ingress);
                drop(face_startup_rx);
                let cleanup = join_face_perception_thread_bounded(
                    face_thread,
                    NANO_FACE_PERCEPTION_JOIN_TIMEOUT,
                    NANO_FACE_PERCEPTION_JOIN_TIMEOUT,
                );
                return Err(
                    NanoAccessoryWorkerStartError::FacePerceptionStartupTimedOut {
                        timeout: NANO_FACE_PERCEPTION_STARTUP_TIMEOUT,
                        cleanup,
                    },
                );
            }
        };

        let (startup_tx, startup_rx) = std::sync::mpsc::sync_channel(1);
        let (fault_tx, fault_rx) = crossbeam_channel::bounded(1);
        let lifecycle = Arc::new(NanoAccessoryOwnerLifecycle::starting());
        let worker_lifecycle = Arc::clone(&lifecycle);
        let worker_face_output = Arc::clone(&face_output);
        let worker_raw_channel = Arc::clone(&channel);
        let latest_head_health = Arc::new(Mutex::new(NanoAccessoryHeadHealthState::empty()));
        let worker_head_health = Arc::clone(&latest_head_health);
        let thread = match thread::Builder::new()
            .name("kiko-nano-accessories".into())
            .spawn(move || {
                let _lifecycle_guard =
                    NanoAccessoryThreadLifecycleGuard::new(Arc::clone(&worker_lifecycle));
                run_production_worker_with_face(
                    config,
                    worker_face_output,
                    worker_raw_channel,
                    worker_head_health,
                    startup_tx,
                    fault_tx,
                    expression_clock,
                    NanoAccessoryPerceptionReadyEvidence::Face(face_ready),
                    worker_lifecycle,
                )
            }) {
            Ok(thread) => thread,
            Err(source) => {
                channel.request_shutdown();
                face_output.request_shutdown();
                drop(ingress);
                let face_perception = join_face_perception_thread_bounded(
                    face_thread,
                    NANO_FACE_PERCEPTION_JOIN_TIMEOUT,
                    NANO_FACE_PERCEPTION_JOIN_TIMEOUT,
                );
                return Err(
                    NanoAccessoryWorkerStartError::AccessoryThreadSpawnWithFace {
                        source,
                        face_perception,
                    },
                );
            }
        };

        match startup_rx.recv() {
            Ok(StartupSignal::Ready(ready)) => Ok(Self {
                ready: *ready,
                ingress: Some(ingress),
                channel,
                lifecycle,
                latest_head_health,
                fault_rx,
                thread: Some(thread),
                face_output: Some(face_output),
                face_thread: Some(face_thread),
                face_diagnostics: Some((
                    face_diagnostic_rx,
                    NanoFaceDiagnosticStatsHandle(face_diagnostic_stats),
                )),
                face_diagnostics_active: Some(face_diagnostics_active),
                face_stage_counters: Some(face_stage_counters),
                head_gaze_lease_issuer,
            }),
            Ok(StartupSignal::Failed) | Err(_) => {
                channel.request_shutdown();
                face_output.request_shutdown();
                drop(ingress);
                let accessory = thread.join();
                let face_perception = join_face_perception_thread_bounded(
                    face_thread,
                    NANO_FACE_PERCEPTION_JOIN_TIMEOUT,
                    NANO_FACE_PERCEPTION_JOIN_TIMEOUT,
                );
                match accessory {
                    Ok(accessory) => {
                        Err(NanoAccessoryWorkerStartError::AccessoryStartupFailedWithFace {
                            accessory: Box::new(accessory),
                            face_perception,
                        })
                    }
                    Err(_) => Err(
                        NanoAccessoryWorkerStartError::AccessoryThreadPanickedBeforeReadinessWithFace {
                            face_perception,
                        },
                    ),
                }
            }
        }
    }

    pub const fn readiness(&self) -> &NanoAccessoryReadyEvidence {
        &self.ready
    }

    /// Bind the sole live base owner's confirmed-zero lease issuer once.
    ///
    /// The issuer is cloned into the already-running head owner; cloning does
    /// not mint a lease. A second bind is rejected, and a natural-hold or
    /// proposal-only worker cannot accept an issuer.
    #[cfg(feature = "nano-agent")]
    pub fn bind_head_gaze_base_zero_lease_issuer(
        &self,
        issuer: &HeadGazeBaseZeroExclusiveLeaseIssuer,
    ) -> Result<(), NanoHeadGazeLeaseBindError> {
        let slot = self
            .head_gaze_lease_issuer
            .as_ref()
            .ok_or(NanoHeadGazeLeaseBindError::PhysicalGazeNotConfigured)?;
        slot.set(issuer.clone())
            .map_err(|_| NanoHeadGazeLeaseBindError::AlreadyBound)
    }

    pub fn submit_rgb(&mut self, frame: ImageFrame) -> NanoAccessoryFrameSubmitOutcome {
        match &mut self.ingress {
            Some(ingress) => ingress.submit(frame),
            None => NanoAccessoryFrameSubmitOutcome::IngressDisconnected,
        }
    }

    /// Observe a frame at the authoritative ingress boundary, then allow one
    /// borrowed non-authoritative diagnostic projection before queueing it.
    ///
    /// Use this instead of copying diagnostics before [`Self::submit_rgb`].
    /// The clock is sampled first, so callback time truthfully contributes to
    /// the bridge's exclusive source-freshness deadline.
    pub fn submit_rgb_after_observation(
        &mut self,
        frame: ImageFrame,
        diagnostic: impl FnOnce(&ImageFrame),
    ) -> NanoAccessoryFrameSubmitOutcome {
        match &mut self.ingress {
            Some(ingress) => ingress.submit_after_observation(frame, diagnostic),
            None => NanoAccessoryFrameSubmitOutcome::IngressDisconnected,
        }
    }

    pub fn frame_stats(&self) -> NanoAccessoryFrameStats {
        self.channel.counters.snapshot()
    }

    /// Take the sole best-effort face-metadata diagnostic consumer.
    ///
    /// The capacity-one queue drops the oldest metadata under backpressure.
    /// Its stats are independent from authoritative RGB/perception/expression
    /// counters, and consumer disconnect never faults the robot.
    #[cfg(feature = "nano-agent")]
    pub fn take_face_diagnostics(
        &mut self,
    ) -> Option<(NanoFaceDiagnosticReceiver, NanoFaceDiagnosticStatsHandle)> {
        let diagnostics = self.face_diagnostics.take()?;
        self.face_diagnostics_active
            .as_ref()
            .expect("face diagnostics have one activation flag")
            .store(true, Ordering::Release);
        Some(diagnostics)
    }

    /// Snapshot authoritative detector-to-expression handoff outcomes.
    ///
    /// This is intentionally separate from raw ingress replacement counters
    /// and best-effort diagnostic-channel drop counters.
    #[cfg(feature = "nano-agent")]
    pub fn face_perception_stage_stats(&self) -> Option<NanoFacePerceptionStageStats> {
        self.face_stage_counters
            .as_ref()
            .map(|counters| counters.snapshot())
    }

    /// Clone a telemetry-only handle which can be sampled after worker
    /// shutdown. See [`NanoFacePerceptionStageStatsHandle`] for finality rules.
    #[cfg(feature = "nano-agent")]
    pub fn face_perception_stage_stats_handle(&self) -> Option<NanoFacePerceptionStageStatsHandle> {
        self.face_stage_counters
            .as_ref()
            .map(|counters| NanoFacePerceptionStageStatsHandle(Arc::clone(counters)))
    }

    /// Borrow the running worker's status without duplicating any device owner.
    pub fn health_observer(&self) -> NanoAccessoryHealthObserver {
        NanoAccessoryHealthObserver {
            channel: Arc::clone(&self.channel),
            head: Arc::clone(&self.latest_head_health),
            lifecycle: Arc::clone(&self.lifecycle),
            health_period: self.ready.health_period(),
            rgb_frame_freshness: self.ready.rgb_frame_freshness(),
        }
    }

    /// Observe completed sole-owner loop transactions without cloning any
    /// serial, camera, or actuator authority.
    pub fn loop_liveness_observer(&self) -> NanoAccessoryLoopLivenessObserver {
        NanoAccessoryLoopLivenessObserver {
            lifecycle: Arc::clone(&self.lifecycle),
        }
    }

    /// Clone the most recent complete successful four-joint health evidence.
    ///
    /// The worker updates this slot only at the configured health cadence, not
    /// on RGB ingress. `None` means no periodic check has succeeded yet.
    pub fn latest_successful_head_health(
        &self,
    ) -> Result<Option<VerifiedHeadHealthEvidence>, NanoAccessoryHealthStatusError> {
        self.latest_head_health
            .lock()
            .map(|state| state.evidence.clone())
            .map_err(|_| NanoAccessoryHealthStatusError::Poisoned)
    }

    /// Mark the sole OAK-to-worker producer unavailable without shutting down
    /// the natural-hold actor. The resulting terminal fault tells the main
    /// owner to stop the base.
    pub fn disconnect_rgb_ingress(&mut self) {
        self.ingress.take();
    }

    pub fn try_terminal_fault(
        &self,
    ) -> Result<Option<NanoAccessoryTerminalFault>, NanoAccessoryFaultWaitError> {
        match self.fault_rx.try_recv() {
            Ok(fault) => Ok(Some(fault)),
            Err(crossbeam_channel::TryRecvError::Empty) => Ok(None),
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                Err(NanoAccessoryFaultWaitError::PublisherDisconnected)
            }
        }
    }

    pub fn wait_for_terminal_fault(
        &self,
        timeout: Duration,
    ) -> Result<NanoAccessoryTerminalFault, NanoAccessoryFaultWaitError> {
        self.fault_rx
            .recv_timeout(timeout)
            .map_err(|source| match source {
                crossbeam_channel::RecvTimeoutError::Timeout => {
                    NanoAccessoryFaultWaitError::Timeout
                }
                crossbeam_channel::RecvTimeoutError::Disconnected => {
                    NanoAccessoryFaultWaitError::PublisherDisconnected
                }
            })
    }

    /// The only operation which requests eye release and hold-preserving head
    /// ownership release.
    ///
    /// Eye and head cleanup is joined first. The hardware-free face thread is
    /// then joined only for the time remaining in the original
    /// [`NANO_FACE_PERCEPTION_JOIN_TIMEOUT`] deadline. A timeout is retained in
    /// shutdown evidence and the thread is detached; it owns no camera or
    /// actuator.
    pub fn shutdown(mut self) -> Result<NanoAccessoryWorkerExit, NanoAccessoryWorkerJoinError> {
        self.lifecycle.mark_shutting_down();
        self.channel.request_shutdown();
        #[cfg(feature = "nano-agent")]
        if let Some(face_output) = &self.face_output {
            face_output.request_shutdown();
        }
        self.ingress.take();
        #[cfg(feature = "nano-agent")]
        let face_join_deadline_started_at = Instant::now();
        let accessory = self
            .thread
            .take()
            .expect("running worker owns one thread")
            .join();
        #[cfg(feature = "nano-agent")]
        let face_perception = match self.face_thread.take() {
            Some(thread) => {
                let remaining = NANO_FACE_PERCEPTION_JOIN_TIMEOUT
                    .checked_sub(face_join_deadline_started_at.elapsed())
                    .unwrap_or(Duration::ZERO);
                NanoFacePerceptionShutdownEvidence::Join(join_face_perception_thread_bounded(
                    thread,
                    remaining,
                    NANO_FACE_PERCEPTION_JOIN_TIMEOUT,
                ))
            }
            None => NanoFacePerceptionShutdownEvidence::Disabled,
        };
        self.lifecycle.mark_stopped();
        match accessory {
            Ok(exit) => {
                #[cfg(feature = "nano-agent")]
                {
                    let mut exit = exit;
                    if let NanoAccessoryWorkerExit::Shutdown { evidence, .. } = &mut exit {
                        evidence.face_perception = face_perception;
                    } else if let NanoFacePerceptionShutdownEvidence::Join(face_perception) =
                        face_perception
                    {
                        exit = NanoAccessoryWorkerExit::UnexpectedAccessoryExitWithFaceEvidence {
                            accessory: Box::new(exit),
                            face_perception,
                        };
                    }
                    Ok(exit)
                }
                #[cfg(not(feature = "nano-agent"))]
                Ok(exit)
            }
            Err(_) => {
                #[cfg(feature = "nano-agent")]
                if let NanoFacePerceptionShutdownEvidence::Join(face_perception) = face_perception {
                    return Err(
                        NanoAccessoryWorkerJoinError::ThreadPanickedWithFaceEvidence {
                            face_perception,
                        },
                    );
                }
                Err(NanoAccessoryWorkerJoinError::ThreadPanicked)
            }
        }
    }
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoHeadGazeLeaseBindError {
    PhysicalGazeNotConfigured,
    AlreadyBound,
}

#[cfg(feature = "nano-agent")]
impl fmt::Display for NanoHeadGazeLeaseBindError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "head-gaze base-zero lease issuer was not bound: {self:?}"
        )
    }
}

#[cfg(feature = "nano-agent")]
impl std::error::Error for NanoHeadGazeLeaseBindError {}

impl Drop for NanoAccessoryWorker {
    fn drop(&mut self) {
        // This changes only the truthfulness of detached observers. It does
        // not request eye/head release or claim a physical actuator state.
        self.lifecycle.mark_detached_if_live();
    }
}

#[cfg(feature = "nano-agent")]
fn face_perception_thread_panicked_exit() -> NanoFacePerceptionThreadExit {
    NanoFacePerceptionThreadExit::RuntimeFault {
        source: NanoFacePerceptionRuntimeError::PerceptionThreadPanicked,
        published_to_accessory: false,
    }
}

#[cfg(feature = "nano-agent")]
fn join_face_perception_thread_bounded(
    thread_handle: JoinHandle<NanoFacePerceptionThreadExit>,
    active_join_budget: Duration,
    configured_timeout: Duration,
) -> NanoFacePerceptionJoinEvidence {
    let started_at = Instant::now();
    while !thread_handle.is_finished() {
        let elapsed = started_at.elapsed();
        let Some(remaining) = active_join_budget.checked_sub(elapsed) else {
            drop(thread_handle);
            return NanoFacePerceptionJoinEvidence::DetachedAfterTimeout {
                configured_timeout,
                active_join_budget,
            };
        };
        thread::sleep(remaining.min(NANO_FACE_PERCEPTION_JOIN_POLL_INTERVAL));
    }
    NanoFacePerceptionJoinEvidence::Joined(
        thread_handle
            .join()
            .unwrap_or(face_perception_thread_panicked_exit()),
    )
}

#[cfg(feature = "nano-agent")]
fn evaluate_head_gaze_diagnostic(
    adapter: Option<&HeadGazeFaceProposalAdapter>,
    frame: &ParsedIngressRgbFrame,
    output: NanoFacePerceptionOutput,
) -> NanoHeadGazeDiagnostic {
    let actuation = NanoHeadGazeActuationAvailability::UnavailableWithoutBaseZeroExclusiveLease;
    let Some(adapter) = adapter else {
        return NanoHeadGazeDiagnostic::DisabledNoPolicy { actuation };
    };

    let observation = frame.observation();
    let layout = observation.layout();
    let oak_intrinsics = frame.frame().intrinsics();
    let projection_evidence = NanoHeadGazeRgbProjectionEvidence::from_frame(frame.frame());
    let evaluated_at = output.tracked_at();
    if oak_intrinsics.width() != layout.width_px() || oak_intrinsics.height() != layout.height_px()
    {
        return NanoHeadGazeDiagnostic::ProposalOnly {
            evaluated_at,
            projection: projection_evidence,
            outcome: Err(NanoHeadGazeDiagnosticError::ProjectionGridMismatch {
                frame_width_px: layout.width_px(),
                frame_height_px: layout.height_px(),
                intrinsic_width_px: oak_intrinsics.width(),
                intrinsic_height_px: oak_intrinsics.height(),
            }),
            actuation,
        };
    }
    let intrinsics = match PinholeIntrinsics::try_new(
        oak_intrinsics.fx(),
        oak_intrinsics.fy(),
        oak_intrinsics.cx(),
        oak_intrinsics.cy(),
    ) {
        Ok(intrinsics) => intrinsics,
        Err(source) => {
            return NanoHeadGazeDiagnostic::ProposalOnly {
                evaluated_at,
                projection: projection_evidence,
                outcome: Err(NanoHeadGazeDiagnosticError::Intrinsics(source)),
                actuation,
            };
        }
    };
    let outcome = adapter
        .evaluate(
            output.tracking(),
            evaluated_at,
            RgbFacePinholeProjection::new(intrinsics, layout),
        )
        .map_err(NanoHeadGazeDiagnosticError::Proposal);
    NanoHeadGazeDiagnostic::ProposalOnly {
        evaluated_at,
        projection: projection_evidence,
        outcome,
        actuation,
    }
}

/// Last-resort ownership closure around the entire face OS-thread body.
///
/// The inner `catch_unwind` translates detector/body panics into a typed
/// terminal fault. This outer RAII boundary covers a second panic in that
/// handler or in post-body cleanup: raw ingress is closed and the face-output
/// producer is disconnected even when no typed thread exit can be returned.
#[cfg(feature = "nano-agent")]
struct NanoFacePerceptionThreadLifecycleGuard {
    input: Arc<LatestFrameChannel<NanoAccessoryRgbWork>>,
    output: Arc<LatestFrameChannel<NanoFacePerceptionWork>>,
    armed: bool,
}

#[cfg(feature = "nano-agent")]
impl NanoFacePerceptionThreadLifecycleGuard {
    fn new(
        input: Arc<LatestFrameChannel<NanoAccessoryRgbWork>>,
        output: Arc<LatestFrameChannel<NanoFacePerceptionWork>>,
    ) -> Self {
        Self {
            input,
            output,
            armed: true,
        }
    }

    fn finish(self, exit: NanoFacePerceptionThreadExit) -> NanoFacePerceptionThreadExit {
        self.finish_with_post_latch_hook(exit, || {})
    }

    fn finish_with_post_latch_hook(
        mut self,
        exit: NanoFacePerceptionThreadExit,
        post_latch_hook: impl FnOnce(),
    ) -> NanoFacePerceptionThreadExit {
        if !matches!(exit, NanoFacePerceptionThreadExit::Shutdown) {
            // Runtime faults already latch through
            // `publish_face_perception_fault`; this idempotent close also
            // covers startup failures and a lane already terminal.
            self.input.latch_terminal_fault();
        }
        post_latch_hook();
        self.output.disconnect_ingress();
        self.armed = false;
        exit
    }
}

#[cfg(feature = "nano-agent")]
impl Drop for NanoFacePerceptionThreadLifecycleGuard {
    fn drop(&mut self) {
        if self.armed {
            self.input.latch_terminal_fault();
            self.output.disconnect_ingress();
        }
    }
}

#[cfg(feature = "nano-agent")]
#[allow(clippy::too_many_arguments)]
fn run_face_perception_thread_catching_panics(
    assets: NanoFacePerceptionAssets,
    detector_config: OpenCvHaarFaceDetectorConfig,
    tracking_config: FaceTrackingConfig,
    stream_epoch: StreamEpochId,
    freshness: NonZeroDuration,
    clock: NanoAccessoryClock,
    input: Arc<LatestFrameChannel<NanoAccessoryRgbWork>>,
    output: Arc<LatestFrameChannel<NanoFacePerceptionWork>>,
    diagnostic_tx: DropSender<NanoFaceDiagnosticFrame>,
    diagnostics_active: Arc<AtomicBool>,
    stage_counters: Arc<NanoFacePerceptionStageCounters>,
    startup_tx: std::sync::mpsc::SyncSender<FacePerceptionStartupSignal>,
    head_gaze_diagnostics: Option<HeadGazeFaceProposalAdapter>,
) -> NanoFacePerceptionThreadExit {
    let lifecycle_guard =
        NanoFacePerceptionThreadLifecycleGuard::new(Arc::clone(&input), Arc::clone(&output));
    let panic_startup_tx = startup_tx.clone();
    let panic_input = Arc::clone(&input);
    let panic_output = Arc::clone(&output);
    let exit = catch_unwind(AssertUnwindSafe(|| {
        run_face_perception_thread(
            assets,
            detector_config,
            tracking_config,
            stream_epoch,
            freshness,
            clock,
            Arc::clone(&input),
            Arc::clone(&output),
            diagnostic_tx,
            diagnostics_active,
            stage_counters,
            startup_tx,
            head_gaze_diagnostics,
        )
    }))
    .unwrap_or_else(|_| {
        let _ = panic_startup_tx.send(FacePerceptionStartupSignal::Failed);
        let source = NanoFacePerceptionRuntimeError::PerceptionThreadPanicked;
        let published_to_accessory =
            publish_face_perception_fault(&panic_input, &panic_output, source.clone());
        NanoFacePerceptionThreadExit::RuntimeFault {
            source,
            published_to_accessory,
        }
    });
    lifecycle_guard.finish(exit)
}

#[cfg(feature = "nano-agent")]
#[allow(clippy::too_many_arguments)]
fn run_face_perception_thread(
    assets: NanoFacePerceptionAssets,
    detector_config: OpenCvHaarFaceDetectorConfig,
    tracking_config: FaceTrackingConfig,
    stream_epoch: StreamEpochId,
    freshness: NonZeroDuration,
    clock: NanoAccessoryClock,
    input: Arc<LatestFrameChannel<NanoAccessoryRgbWork>>,
    output: Arc<LatestFrameChannel<NanoFacePerceptionWork>>,
    diagnostic_tx: DropSender<NanoFaceDiagnosticFrame>,
    diagnostics_active: Arc<AtomicBool>,
    stage_counters: Arc<NanoFacePerceptionStageCounters>,
    startup_tx: std::sync::mpsc::SyncSender<FacePerceptionStartupSignal>,
    head_gaze_diagnostics: Option<HeadGazeFaceProposalAdapter>,
) -> NanoFacePerceptionThreadExit {
    let mut diagnostic_tx = Some(diagnostic_tx);
    let asset_evidence = assets.evidence();
    let (frontal_face_cascade, profile_face_cascade) = assets.into_bytes();
    let mut perception = match NanoFacePerception::load(
        &frontal_face_cascade,
        &profile_face_cascade,
        detector_config,
        tracking_config,
    ) {
        Ok(perception) => perception,
        Err(source) => {
            let _ = startup_tx.send(FacePerceptionStartupSignal::Failed);
            return NanoFacePerceptionThreadExit::LoadFailed(source);
        }
    };
    // The native detector has consumed the exact retained bytes in memory.
    // Drop the source vectors before entering the steady-state frame loop.
    drop(frontal_face_cascade);
    drop(profile_face_cascade);

    let ready = NanoFacePerceptionReadyEvidence {
        assets: asset_evidence,
        detector_config: perception.detector_config().clone(),
        tracking_config: perception.tracking_config(),
    };
    if startup_tx
        .send(FacePerceptionStartupSignal::Ready(Box::new(ready)))
        .is_err()
    {
        return NanoFacePerceptionThreadExit::StartupObserverDropped;
    }

    loop {
        let frame = match input.next_event_blocking() {
            LatestFrameEvent::Frame(Ok(frame)) => frame,
            LatestFrameEvent::Frame(Err(source)) => {
                return finish_face_perception_fault(
                    &input,
                    &output,
                    NanoFacePerceptionRuntimeError::IngressClock(source),
                );
            }
            LatestFrameEvent::ShutdownRequested => return NanoFacePerceptionThreadExit::Shutdown,
            LatestFrameEvent::IngressDisconnected => {
                return finish_face_perception_fault(
                    &input,
                    &output,
                    NanoFacePerceptionRuntimeError::RgbIngressDisconnected,
                );
            }
            LatestFrameEvent::ChannelPoisoned => {
                return finish_face_perception_fault(
                    &input,
                    &output,
                    NanoFacePerceptionRuntimeError::RgbChannelPoisoned,
                );
            }
        };
        let parsed = match parse_ingress_observed_oak_frame(frame, stream_epoch, freshness) {
            Ok(parsed) => parsed,
            Err(source) => {
                return finish_face_perception_fault(
                    &input,
                    &output,
                    NanoFacePerceptionRuntimeError::Parse(Box::new(source)),
                );
            }
        };
        let perception_output = match perception.process_parsed(&parsed, &clock) {
            Ok(output) => output,
            Err(source) => {
                return finish_face_perception_fault(
                    &input,
                    &output,
                    NanoFacePerceptionRuntimeError::Perception(Box::new(source)),
                );
            }
        };
        stage_counters.record_result();
        let head_gaze = evaluate_head_gaze_diagnostic(
            head_gaze_diagnostics.as_ref(),
            &parsed,
            perception_output,
        );
        stage_counters.record_head_gaze(head_gaze);
        let diagnostic =
            NanoFaceDiagnosticFrame::from_parsed(&parsed, perception_output, head_gaze);
        let outcome = output.submit_unmetered(Ok(NanoFaceTrackedRgbFrame {
            frame: parsed,
            output: perception_output,
        }));
        stage_counters.record_handoff(outcome);
        match outcome {
            NanoAccessoryFrameSubmitOutcome::Enqueued
            | NanoAccessoryFrameSubmitOutcome::ReplacedOlderFrame => {
                // Best-effort observability is strictly downstream of the
                // authoritative handoff. Its disconnect/drop outcome cannot
                // alter robot state.
                publish_face_diagnostic_if_active(
                    &diagnostics_active,
                    &mut diagnostic_tx,
                    diagnostic,
                );
            }
            outcome if face_handoff_is_coordinated_shutdown(&input, &output, outcome) => {
                return NanoFacePerceptionThreadExit::Shutdown;
            }
            NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched => {
                return NanoFacePerceptionThreadExit::AccessoryFaultLatched;
            }
            NanoAccessoryFrameSubmitOutcome::TerminalFaultPendingPublication => {
                return NanoFacePerceptionThreadExit::AccessoryFaultPendingPublication;
            }
            outcome => {
                return finish_face_perception_fault(
                    &input,
                    &output,
                    NanoFacePerceptionRuntimeError::ExpressionHandoffUnavailable { outcome },
                );
            }
        }
    }
}

#[cfg(feature = "nano-agent")]
fn face_handoff_is_coordinated_shutdown<I, O>(
    input: &LatestFrameChannel<I>,
    output: &LatestFrameChannel<O>,
    outcome: NanoAccessoryFrameSubmitOutcome,
) -> bool {
    matches!(
        outcome,
        NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched
            | NanoAccessoryFrameSubmitOutcome::IngressDisconnected
    ) && (input.shutdown_requested.load(Ordering::Acquire)
        || output.shutdown_requested.load(Ordering::Acquire))
}

#[cfg(feature = "nano-agent")]
fn finish_face_perception_fault(
    input: &LatestFrameChannel<NanoAccessoryRgbWork>,
    output: &LatestFrameChannel<NanoFacePerceptionWork>,
    source: NanoFacePerceptionRuntimeError,
) -> NanoFacePerceptionThreadExit {
    let published_to_accessory = publish_face_perception_fault(input, output, source.clone());
    NanoFacePerceptionThreadExit::RuntimeFault {
        source,
        published_to_accessory,
    }
}

#[cfg(feature = "nano-agent")]
fn publish_face_perception_fault(
    input: &LatestFrameChannel<NanoAccessoryRgbWork>,
    output: &LatestFrameChannel<NanoFacePerceptionWork>,
    source: NanoFacePerceptionRuntimeError,
) -> bool {
    publish_face_perception_fault_with_raw_lock_hook(input, output, source, || {})
}

#[cfg(feature = "nano-agent")]
fn publish_face_perception_fault_with_raw_lock_hook(
    input: &LatestFrameChannel<NanoAccessoryRgbWork>,
    output: &LatestFrameChannel<NanoFacePerceptionWork>,
    source: NanoFacePerceptionRuntimeError,
    raw_lock_hook: impl FnOnce(),
) -> bool {
    // Hold the raw slot across output publication and raw admission closure.
    // A producer that committed before this lock is discarded by the latch;
    // one that raced after the output terminal commit must recheck admission
    // under this same mutex and is rejected. This preserves the required
    // publication-before-latch order without an ownerless-frame window.
    let mut raw_slot = input.lock_slot_recovering_poison();
    raw_lock_hook();
    let published = matches!(
        output.submit_first_terminal(Err(source), false),
        NanoAccessoryFrameSubmitOutcome::Enqueued
    );
    input.latch_terminal_fault_while_locked(&mut raw_slot);
    drop(raw_slot);
    published
}

#[cfg(feature = "nano-agent")]
fn publish_face_diagnostic_if_active(
    active: &AtomicBool,
    sender: &mut Option<DropSender<NanoFaceDiagnosticFrame>>,
    diagnostic: NanoFaceDiagnosticFrame,
) {
    if !active.load(Ordering::Acquire) {
        return;
    }
    let disconnected = sender
        .as_ref()
        .is_some_and(|sender| matches!(sender.try_send(diagnostic), SendOutcome::Disconnected));
    if disconnected {
        *sender = None;
    }
}

// Intentionally no shutdown in Drop. A service owner must coordinate physical
// support and call `shutdown`; an accidental handle drop may not silently
// remove neck torque.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AccessoryPetEpisode {
    completed_at: MonotonicTimestamp,
    episode: CharacterPetEpisode,
    #[cfg(feature = "nano-agent")]
    controller_evidence: AccessoryPetControllerEvidence,
}

#[cfg(feature = "nano-agent")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AccessoryPetControllerEvidence {
    Completed(CompliantPetEpisodeSummary),
    #[cfg(test)]
    GenericCoreFixture(NanoPetEpisodeEvidence),
}

#[cfg(feature = "nano-agent")]
impl AccessoryPetControllerEvidence {
    fn try_append(
        self,
        journal: &mut NanoPetEvidenceJournal,
    ) -> Result<(), NanoPetEvidenceJournalRuntimeError> {
        match self {
            Self::Completed(summary) => journal.try_append_completed(summary, SystemTime::now()),
            #[cfg(test)]
            Self::GenericCoreFixture(evidence) => journal.try_append(evidence),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AccessoryHeadFeedback {
    state: Option<CharacterPetState>,
    thermal_state: Option<CharacterThermalState>,
    completed_episode: Option<AccessoryPetEpisode>,
    tracking_eye_gaze: Option<TrackingEyeGaze>,
}

#[cfg(feature = "nano-agent")]
const fn adapt_thermal_state(state: HeadThermalDerateState) -> CharacterThermalState {
    match state {
        HeadThermalDerateState::Nominal => CharacterThermalState::Nominal,
        HeadThermalDerateState::PitchDerated => CharacterThermalState::PitchDerated,
    }
}

#[cfg(feature = "nano-agent")]
fn adapt_pet_episode(
    summary: CompliantPetEpisodeSummary,
) -> Result<AccessoryPetEpisode, NanoPhysicalHeadGazeRuntimeError> {
    let completed_at_ns = summary.completed_at().duration_since_origin().as_nanos();
    let completed_at_ns = u64::try_from(completed_at_ns).map_err(|_| {
        NanoPhysicalHeadGazeRuntimeError::PetEpisodeTimestampOutOfRange { completed_at_ns }
    })?;
    let episode = CharacterPetEpisode::try_new(
        summary.duration(),
        summary.accumulated_max_delta_ticks(),
        summary.delta_samples(),
        summary.reached_comfy(),
        summary.was_tap(),
    )
    .map_err(NanoPhysicalHeadGazeRuntimeError::PetEpisode)?;
    Ok(AccessoryPetEpisode {
        completed_at: MonotonicTimestamp::from_nanos_since_epoch(completed_at_ns),
        episode,
        controller_evidence: AccessoryPetControllerEvidence::Completed(summary),
    })
}

#[cfg(feature = "nano-agent")]
fn adapt_pet_state(
    disposition: CompliantHoldDisposition,
) -> Result<CharacterPetState, NanoPhysicalHeadGazeRuntimeError> {
    match disposition {
        CompliantHoldDisposition::FollowingExpression
        | CompliantHoldDisposition::ReturnedToExpression => Ok(CharacterPetState::Ready),
        CompliantHoldDisposition::ContactCandidate {
            consecutive_samples,
        } => Ok(CharacterPetState::ContactCandidate {
            consecutive_samples: NonZeroU8::new(consecutive_samples)
                .ok_or(NanoPhysicalHeadGazeRuntimeError::PetContactCandidateHasZeroSamples)?,
        }),
        CompliantHoldDisposition::Yielding { .. } => Ok(CharacterPetState::Yielding),
        CompliantHoldDisposition::ReleaseDwell => Ok(CharacterPetState::ReleaseDwell),
        CompliantHoldDisposition::Resting { at_rest_target, .. } => {
            Ok(CharacterPetState::Resting { at_rest_target })
        }
        CompliantHoldDisposition::Recovering {
            progress_millionths,
            ..
        } => {
            if progress_millionths > 1_000_000 {
                return Err(
                    NanoPhysicalHeadGazeRuntimeError::PetRecoveryProgressOutOfRange {
                        progress_millionths,
                    },
                );
            }
            // Floor rather than round: the character layer must not claim an
            // exact completed recovery before the compliant controller does.
            let progress_basis_points = u64::from(progress_millionths) / 100;
            let progress_basis_points = u16::try_from(progress_basis_points).map_err(|_| {
                NanoPhysicalHeadGazeRuntimeError::PetRecoveryProgressOutOfRange {
                    progress_millionths,
                }
            })?;
            let progress =
                CoreUnitAmount::try_from_basis_points(progress_basis_points).map_err(|_| {
                    NanoPhysicalHeadGazeRuntimeError::PetRecoveryProgressOutOfRange {
                        progress_millionths,
                    }
                })?;
            Ok(CharacterPetState::Recovering { progress })
        }
    }
}

#[cfg(feature = "nano-agent")]
fn adapt_pet_feedback(
    evidence: &kiko_head_runtime::VerifiedHeadCompliantHoldStep,
) -> Result<AccessoryHeadFeedback, NanoPhysicalHeadGazeRuntimeError> {
    let controller = evidence.controller();
    Ok(AccessoryHeadFeedback {
        state: Some(adapt_pet_state(controller.disposition())?),
        thermal_state: evidence
            .thermal_derate()
            .map(|step| adapt_thermal_state(step.state())),
        completed_episode: controller
            .completed_episode()
            .map(adapt_pet_episode)
            .transpose()?,
        tracking_eye_gaze: None,
    })
}

#[cfg(feature = "nano-agent")]
const fn adapt_base_motion_state(phase: HeadGazeBaseMotionPhase) -> CharacterBaseMotionState {
    match phase {
        HeadGazeBaseMotionPhase::ConfirmedZero | HeadGazeBaseMotionPhase::HeadGazeTransaction => {
            CharacterBaseMotionState::ConfirmedStationary
        }
        HeadGazeBaseMotionPhase::BaseTransaction => CharacterBaseMotionState::CommandTransaction,
        HeadGazeBaseMotionPhase::BaseMayMove => CharacterBaseMotionState::MayMove,
        HeadGazeBaseMotionPhase::Faulted => CharacterBaseMotionState::Faulted,
    }
}

trait ReviewedNaturalHeadPort<F> {
    type Ready;
    type StartError;
    type Health;
    type HealthError: Clone;
    type GazeError: Clone;
    type Shutdown;

    async fn start(&mut self) -> Result<Self::Ready, Self::StartError>;
    async fn check_health(&mut self) -> Result<Self::Health, Self::HealthError>;
    async fn process_gaze_frame(
        &mut self,
        _frame: &F,
        _character_head: CharacterHeadOverlay,
    ) -> Result<Option<AccessoryHeadFeedback>, Self::GazeError> {
        Ok(None)
    }
    fn character_base_motion_state(&self) -> CharacterBaseMotionState {
        CharacterBaseMotionState::NotApplicable
    }
    async fn shutdown(&mut self) -> Self::Shutdown;
}

trait Kep2EyePort<I> {
    type Ready;
    type StartError;
    type ApplyError: Clone;
    type Shutdown;

    async fn start(&mut self) -> Result<Self::Ready, Self::StartError>;
    async fn apply(&mut self, intent: I) -> Result<(), Self::ApplyError>;
    async fn shutdown(&mut self) -> Self::Shutdown;
}

trait RgbBridgePort<F> {
    type Intent;
    type Error: Clone;

    fn process(&mut self, frame: &F) -> Result<Self::Intent, Self::Error>;

    fn note_pet_state(&mut self, _state: CharacterPetState) {}

    fn note_thermal_state(&mut self, _state: CharacterThermalState) {}

    fn note_base_motion_state(&mut self, _state: CharacterBaseMotionState) {}

    fn note_lifecycle_state(&mut self, _state: CharacterLifecycleState) {}

    fn prepare_lifecycle_reflex(
        &mut self,
        state: CharacterLifecycleState,
    ) -> Result<Self::Intent, Self::Error>;

    fn lifecycle_reflex_duration(&self, _state: CharacterLifecycleState) -> Duration {
        Duration::ZERO
    }

    fn note_pet_episode(&mut self, _episode: AccessoryPetEpisode) {}
}

trait AccessoryCharacterIntent: Copy {
    type EyeIntent;

    fn head_overlay(self) -> CharacterHeadOverlay;
    fn replace_tracking_gaze(
        self,
        gaze: TrackingEyeGaze,
    ) -> Result<Self, TrackingEyeGazeReplacementError>;
    fn into_eye(self) -> Self::EyeIntent;
}

impl AccessoryCharacterIntent for PreparedCharacterFrame {
    type EyeIntent = PreparedEyeIntent;

    fn head_overlay(self) -> CharacterHeadOverlay {
        self.head()
    }

    fn replace_tracking_gaze(
        self,
        gaze: TrackingEyeGaze,
    ) -> Result<Self, TrackingEyeGazeReplacementError> {
        PreparedCharacterFrame::replace_tracking_gaze(self, gaze)
    }

    fn into_eye(self) -> Self::EyeIntent {
        self.eye()
    }
}

#[cfg(test)]
impl AccessoryCharacterIntent for u64 {
    type EyeIntent = u64;

    fn head_overlay(self) -> CharacterHeadOverlay {
        CharacterHeadOverlay::NATURAL
    }

    fn replace_tracking_gaze(
        self,
        gaze: TrackingEyeGaze,
    ) -> Result<Self, TrackingEyeGazeReplacementError> {
        let encoded = i32::from(gaze.x_right().get()) + 1_000;
        Ok(u64::try_from(encoded).expect("test tracking gaze encodes into 0..=2000"))
    }

    fn into_eye(self) -> Self::EyeIntent {
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CoreTerminalFault<H, G, B, E> {
    HeadHealth(H),
    HeadGaze(G),
    HeadHealthStatusPoisoned,
    RgbHealthStatusPoisoned,
    Bridge(B),
    EyeApply(E),
    CharacterCoordination(TrackingEyeGazeReplacementError),
    LifecyclePresentationFrameCountExhausted,
    #[cfg(feature = "nano-agent")]
    PetEvidence(NanoPetEvidenceJournalRuntimeError),
    IngressDisconnected,
    ChannelPoisoned,
    ReadinessObserverDropped,
}

enum CoreExit<ES, HS, H, G, B, E, EyeShutdown, HeadShutdown> {
    EyeStartupFailed(ES),
    HeadStartupFailed {
        source: HS,
        eye_shutdown: EyeShutdown,
    },
    Shutdown {
        terminal_fault: Option<CoreTerminalFault<H, G, B, E>>,
        fault_recovery_presentation: CoreFaultRecoveryPresentationEvidence<B, E>,
        eye_shutdown: EyeShutdown,
        head_shutdown: HeadShutdown,
    },
}

struct CoreObservers<Ready, RecordHealth, PublishFault, LatchFault> {
    ready: Ready,
    record_health: RecordHealth,
    publish_fault: PublishFault,
    latch_fault: LatchFault,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum LifecyclePresentationFault<B, E> {
    Bridge { frames_applied: u64, source: B },
    EyeApply { frames_applied: u64, source: E },
    FrameCountExhausted { frames_applied: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CoreFaultRecoveryPresentationEvidence<B, E> {
    NotRequired,
    Presented { frames_applied: NonZeroU64 },
    MissingPresentationDuration,
    Failed(LifecyclePresentationFault<B, E>),
}

async fn present_lifecycle_reflex<F, E, B>(
    bridge: &mut B,
    eye: &mut E,
    state: CharacterLifecycleState,
) -> Result<u64, LifecyclePresentationFault<B::Error, E::ApplyError>>
where
    E: Kep2EyePort<<B::Intent as AccessoryCharacterIntent>::EyeIntent>,
    B: RgbBridgePort<F>,
    B::Intent: AccessoryCharacterIntent,
{
    let duration = bridge.lifecycle_reflex_duration(state);
    bridge.note_lifecycle_state(state);
    if duration.is_zero() {
        return Ok(0);
    }

    let started = Instant::now();
    let mut frames_applied = 0_u64;
    loop {
        if frames_applied == u64::MAX {
            return Err(LifecyclePresentationFault::FrameCountExhausted { frames_applied });
        }
        let intent = bridge.prepare_lifecycle_reflex(state).map_err(|source| {
            LifecyclePresentationFault::Bridge {
                frames_applied,
                source,
            }
        })?;
        eye.apply(intent.into_eye()).await.map_err(|source| {
            LifecyclePresentationFault::EyeApply {
                frames_applied,
                source,
            }
        })?;
        frames_applied += 1;

        let elapsed = started.elapsed();
        if elapsed >= duration {
            return Ok(frames_applied);
        }
        let remaining = duration.saturating_sub(elapsed);
        tokio::time::sleep(remaining.min(Duration::from_millis(50))).await;
    }
}

async fn present_fault_recovery_after_stop_latch<F, E, B>(
    bridge: &mut B,
    eye: &mut E,
) -> CoreFaultRecoveryPresentationEvidence<B::Error, E::ApplyError>
where
    E: Kep2EyePort<<B::Intent as AccessoryCharacterIntent>::EyeIntent>,
    B: RgbBridgePort<F>,
    B::Intent: AccessoryCharacterIntent,
{
    match present_lifecycle_reflex::<F, _, _>(bridge, eye, CharacterLifecycleState::FaultRecovery)
        .await
    {
        Ok(frames_applied) => match NonZeroU64::new(frames_applied) {
            Some(frames_applied) => {
                CoreFaultRecoveryPresentationEvidence::Presented { frames_applied }
            }
            None => CoreFaultRecoveryPresentationEvidence::MissingPresentationDuration,
        },
        Err(source) => CoreFaultRecoveryPresentationEvidence::Failed(source),
    }
}

async fn run_accessory_core<F, H, E, B, Ready, RecordHealth, PublishFault, LatchFault>(
    mut head: H,
    mut eye: E,
    mut bridge: B,
    channel: Arc<LatestFrameChannel<F>>,
    health_period: NanoAccessoryHealthPeriod,
    #[cfg(feature = "nano-agent")] mut pet_evidence_journal: Option<&mut NanoPetEvidenceJournal>,
    observers: CoreObservers<Ready, RecordHealth, PublishFault, LatchFault>,
) -> CoreExit<
    E::StartError,
    H::StartError,
    H::HealthError,
    H::GazeError,
    B::Error,
    E::ApplyError,
    E::Shutdown,
    H::Shutdown,
>
where
    F: NanoHeadGazeFrame + Send + 'static,
    H: ReviewedNaturalHeadPort<F>,
    E: Kep2EyePort<<B::Intent as AccessoryCharacterIntent>::EyeIntent>,
    B: RgbBridgePort<F>,
    B::Intent: AccessoryCharacterIntent,
    Ready: FnOnce(H::Ready, E::Ready) -> bool,
    RecordHealth: FnMut(H::Health) -> bool,
    PublishFault: FnMut(CoreTerminalFault<H::HealthError, H::GazeError, B::Error, E::ApplyError>),
    LatchFault: FnMut(),
{
    let CoreObservers {
        ready,
        mut record_health,
        mut publish_fault,
        mut latch_fault,
    } = observers;
    bridge.note_lifecycle_state(CharacterLifecycleState::Booting);
    // Start eyes first: a failed eye admission never leaves a successfully
    // energized natural-hold head with no returned owner.
    let eye_ready = match eye.start().await {
        Ok(ready) => ready,
        Err(source) => return CoreExit::EyeStartupFailed(source),
    };
    let head_ready = match head.start().await {
        Ok(ready) => ready,
        Err(source) => {
            let eye_shutdown = eye.shutdown().await;
            return CoreExit::HeadStartupFailed {
                source,
                eye_shutdown,
            };
        }
    };

    if let Err(source) = present_lifecycle_reflex::<F, _, _>(
        &mut bridge,
        &mut eye,
        CharacterLifecycleState::AdmissionRecovery,
    )
    .await
    {
        let fault = match source {
            LifecyclePresentationFault::Bridge { source, .. } => CoreTerminalFault::Bridge(source),
            LifecyclePresentationFault::EyeApply { source, .. } => {
                CoreTerminalFault::EyeApply(source)
            }
            LifecyclePresentationFault::FrameCountExhausted { .. } => {
                CoreTerminalFault::LifecyclePresentationFrameCountExhausted
            }
        };
        publish_fault(fault.clone());
        latch_fault();
        channel.latch_terminal_fault();
        let fault_recovery_presentation =
            present_fault_recovery_after_stop_latch::<F, _, _>(&mut bridge, &mut eye).await;
        let eye_shutdown = eye.shutdown().await;
        let head_shutdown = head.shutdown().await;
        return CoreExit::Shutdown {
            terminal_fault: Some(fault),
            fault_recovery_presentation,
            eye_shutdown,
            head_shutdown,
        };
    }
    bridge.note_lifecycle_state(CharacterLifecycleState::Operational);

    let mut terminal_fault = if ready(head_ready, eye_ready) {
        None
    } else {
        let fault = CoreTerminalFault::ReadinessObserverDropped;
        publish_fault(fault.clone());
        latch_fault();
        channel.latch_terminal_fault();
        Some(fault)
    };
    let mut fault_recovery_presentation = if terminal_fault.is_some() {
        present_fault_recovery_after_stop_latch::<F, _, _>(&mut bridge, &mut eye).await
    } else {
        CoreFaultRecoveryPresentationEvidence::NotRequired
    };
    let period = health_period.get();
    let first_health = tokio::time::Instant::now() + period;
    let mut health = tokio::time::interval_at(first_health, period);
    health.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        if terminal_fault.is_some() {
            tokio::select! {
                _ = health.tick() => {
                    // Retain bus ownership and keep making bounded observations.
                    // The first fault stays latched even if this later succeeds.
                    if let Ok(evidence) = head.check_health().await {
                        let _status_retained = record_health(evidence);
                    }
                }
                () = channel.wait_for_shutdown() => break,
            }
            continue;
        }

        let fault = tokio::select! {
            health_result = async {
                health.tick().await;
                head.check_health().await
            } => {
                let health_fault = match health_result {
                    Ok(evidence) => {
                        if record_health(evidence) {
                            None
                        } else {
                            Some(CoreTerminalFault::HeadHealthStatusPoisoned)
                        }
                    }
                    Err(source) => Some(CoreTerminalFault::HeadHealth(source)),
                };
                #[cfg(feature = "nano-agent")]
                let health_fault = if let Some(source) = pet_evidence_journal
                    .as_deref_mut()
                    .and_then(NanoPetEvidenceJournal::poll_fault)
                {
                    Some(CoreTerminalFault::PetEvidence(source))
                } else {
                    health_fault
                };
                health_fault
            }
            event = channel.next_event() => {
                match event {
                    LatestFrameEvent::Frame(frame) => {
                        bridge.note_base_motion_state(head.character_base_motion_state());
                        match bridge.process(&frame) {
                            Err(source) => Some(CoreTerminalFault::Bridge(source)),
                            Ok(intent) => match head
                                .process_gaze_frame(&frame, intent.head_overlay())
                                .await
                            {
                                Err(source) => Some(CoreTerminalFault::HeadGaze(source)),
                                Ok(head_feedback) => {
                                    let mut completed_episode = None;
                                    let mut tracking_eye_gaze = None;
                                    if let Some(head_feedback) = head_feedback {
                                        if let Some(state) = head_feedback.state {
                                            bridge.note_pet_state(state);
                                        }
                                        if let Some(state) = head_feedback.thermal_state {
                                            bridge.note_thermal_state(state);
                                        }
                                        completed_episode = head_feedback.completed_episode;
                                        tracking_eye_gaze = head_feedback.tracking_eye_gaze;
                                    }
                                    let intent = match tracking_eye_gaze {
                                        Some(gaze) => intent.replace_tracking_gaze(gaze),
                                        None => Ok(intent),
                                    };
                                    match intent {
                                        Err(source) => Some(CoreTerminalFault::CharacterCoordination(source)),
                                        Ok(intent) => {
                                            #[cfg(feature = "nano-agent")]
                                            let pet_evidence_fault = completed_episode
                                                .and_then(|episode| {
                                                    pet_evidence_journal.as_deref_mut().map(|journal| {
                                                        episode.controller_evidence.try_append(journal)
                                                    })
                                                })
                                                .and_then(Result::err)
                                                .map(CoreTerminalFault::PetEvidence);
                                            #[cfg(not(feature = "nano-agent"))]
                                            let pet_evidence_fault = None;
                                            if let Some(fault) = pet_evidence_fault {
                                                Some(fault)
                                            } else {
                                                if let Some(pet_episode) = completed_episode {
                                                    bridge.note_pet_episode(pet_episode);
                                                }
                                                match eye.apply(intent.into_eye()).await {
                                                    Ok(()) => match channel.counters.record_processed_successfully() {
                                                        Ok(()) => None,
                                                        Err(
                                                            NanoAccessoryHealthStatusError::Poisoned
                                                            | NanoAccessoryHealthStatusError::OwnerNotRunning { .. }
                                                            | NanoAccessoryHealthStatusError::IngressDisconnected
                                                            | NanoAccessoryHealthStatusError::ChannelPoisoned,
                                                        ) => Some(CoreTerminalFault::RgbHealthStatusPoisoned),
                                                    },
                                                    Err(source) => Some(CoreTerminalFault::EyeApply(source)),
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                        }
                    }
                    LatestFrameEvent::ShutdownRequested => break,
                    LatestFrameEvent::IngressDisconnected => {
                        Some(CoreTerminalFault::IngressDisconnected)
                    }
                    LatestFrameEvent::ChannelPoisoned => {
                        Some(CoreTerminalFault::ChannelPoisoned)
                    }
                }
            }
        };
        if let Some(fault) = fault {
            publish_fault(fault.clone());
            latch_fault();
            channel.latch_terminal_fault();
            terminal_fault = Some(fault);
            fault_recovery_presentation =
                present_fault_recovery_after_stop_latch::<F, _, _>(&mut bridge, &mut eye).await;
        }
    }

    if terminal_fault.is_none()
        && let Err(source) = present_lifecycle_reflex::<F, _, _>(
            &mut bridge,
            &mut eye,
            CharacterLifecycleState::Parking,
        )
        .await
    {
        let fault = match source {
            LifecyclePresentationFault::Bridge { source, .. } => CoreTerminalFault::Bridge(source),
            LifecyclePresentationFault::EyeApply { source, .. } => {
                CoreTerminalFault::EyeApply(source)
            }
            LifecyclePresentationFault::FrameCountExhausted { .. } => {
                CoreTerminalFault::LifecyclePresentationFrameCountExhausted
            }
        };
        publish_fault(fault.clone());
        latch_fault();
        channel.latch_terminal_fault();
        terminal_fault = Some(fault);
        fault_recovery_presentation =
            present_fault_recovery_after_stop_latch::<F, _, _>(&mut bridge, &mut eye).await;
    }

    // Explicit coordinated shutdown releases eyes first, then preserves the
    // complete result of the head actor's existing shutdown transaction.
    let eye_shutdown = eye.shutdown().await;
    let head_shutdown = head.shutdown().await;
    CoreExit::Shutdown {
        terminal_fault,
        fault_recovery_presentation,
        eye_shutdown,
        head_shutdown,
    }
}

struct ActiveEyeActor {
    handle: EyeActorHandle,
    task: EyeActorTask,
    startup: EyeStartupEvidence,
}

struct SerialKep2EyePort {
    config: Option<EyeRuntimeConfig>,
    clock: Option<NanoAccessoryClock>,
    active: Option<ActiveEyeActor>,
}

impl SerialKep2EyePort {
    fn new(config: EyeRuntimeConfig, clock: NanoAccessoryClock) -> Self {
        Self {
            config: Some(config),
            clock: Some(clock),
            active: None,
        }
    }
}

impl Kep2EyePort<PreparedEyeIntent> for SerialKep2EyePort {
    type Ready = NanoEyeReadyEvidence;
    type StartError = NanoEyeActorStartupError;
    type ApplyError = EyeHandleRequestError;
    type Shutdown = NanoEyeShutdownEvidence;

    async fn start(&mut self) -> Result<Self::Ready, Self::StartError> {
        let config = self.config.take().expect("eye starts exactly once");
        let clock = self.clock.take().expect("eye clock is consumed once");
        let (serial, handle, receipt, task) =
            kiko_eye_runtime::start_serial_eye_actor(config, clock)
                .map_err(NanoEyeActorStartupError::Start)?;
        let startup = match receipt.wait().await {
            Ok(Ok(startup)) => startup,
            Ok(Err(source)) => {
                drop(handle);
                let actor = task.join().await;
                return Err(NanoEyeActorStartupError::Runtime {
                    source: Box::new(source),
                    actor,
                });
            }
            Err(source) => {
                drop(handle);
                let actor = task.join().await;
                return Err(NanoEyeActorStartupError::Receipt { source, actor });
            }
        };
        self.active = Some(ActiveEyeActor {
            handle,
            task,
            startup: startup.clone(),
        });
        Ok(NanoEyeReadyEvidence {
            serial,
            actor: startup,
        })
    }

    async fn apply(&mut self, intent: PreparedEyeIntent) -> Result<(), Self::ApplyError> {
        self.active
            .as_mut()
            .expect("apply occurs only after eye readiness")
            .handle
            .apply_intent(intent)
            .await
            .map(|_| ())
    }

    async fn shutdown(&mut self) -> Self::Shutdown {
        let active = self
            .active
            .take()
            .expect("shutdown occurs only after eye readiness");
        let release = active.handle.shutdown().await;
        let actor = active.task.join().await;
        NanoEyeShutdownEvidence {
            startup: active.startup,
            release,
            actor,
        }
    }
}

enum ActiveHeadHandle {
    Natural(HeadReturnActorHandle),
    #[cfg(feature = "nano-agent")]
    Gaze(HeadGazeActorHandle),
}

impl ActiveHeadHandle {
    async fn return_to_target(
        &self,
    ) -> Result<Result<VerifiedHeadReturnEvidence, HeadReturnError>, HeadCommandError> {
        match self {
            Self::Natural(handle) => handle.return_to_target().await,
            #[cfg(feature = "nano-agent")]
            Self::Gaze(handle) => handle.return_to_target().await,
        }
    }

    async fn check_health(&self) -> Result<VerifiedHeadHealthEvidence, HeadHealthRequestError> {
        match self {
            Self::Natural(handle) => handle.check_health().await,
            #[cfg(feature = "nano-agent")]
            Self::Gaze(handle) => handle.check_health().await,
        }
    }

    async fn release_ownership_preserving_hold(
        self,
    ) -> Result<HoldPreservingOwnershipReleaseEvidence, HeadShutdownError> {
        match self {
            Self::Natural(handle) => handle.release_ownership_preserving_hold().await,
            #[cfg(feature = "nano-agent")]
            Self::Gaze(handle) => handle.release_ownership_preserving_hold().await,
        }
    }

    #[cfg(feature = "nano-agent")]
    fn gaze(&self) -> Option<&HeadGazeActorHandle> {
        match self {
            Self::Natural(_) => None,
            Self::Gaze(handle) => Some(handle),
        }
    }
}

struct ActiveHeadActor {
    handle: ActiveHeadHandle,
    task: HeadActorTask,
    startup: VerifiedNaturalHoldEvidence,
    head_return: VerifiedHeadReturnEvidence,
    #[cfg(feature = "nano-agent")]
    physical_gaze: Option<NanoPhysicalHeadGazeConfig>,
    #[cfg(feature = "nano-agent")]
    committed_target: Option<ExactHeadTargetPose>,
    #[cfg(feature = "nano-agent")]
    last_face_gaze: Option<HeadRelativeGaze>,
}

struct SerialReviewedNaturalHeadPort {
    config: Option<ReturnToTargetConfig>,
    torque_consent: PhysicalTorqueEnableConsent,
    motion_consent: PhysicalHeadMotionConsent,
    takeover_consent: ProductionTensionPreservingTakeoverConsent,
    required_hold_target: HeadHoldTarget,
    #[cfg(feature = "nano-agent")]
    clock: NanoAccessoryClock,
    #[cfg(feature = "nano-agent")]
    physical_gaze: Option<NanoPhysicalHeadGazeConfig>,
    active: Option<ActiveHeadActor>,
}

impl SerialReviewedNaturalHeadPort {
    fn new(
        config: ReturnToTargetConfig,
        torque_consent: PhysicalTorqueEnableConsent,
        motion_consent: PhysicalHeadMotionConsent,
        takeover_consent: ProductionTensionPreservingTakeoverConsent,
        required_hold_target: HeadHoldTarget,
        #[cfg(feature = "nano-agent")] clock: NanoAccessoryClock,
    ) -> Self {
        Self {
            config: Some(config),
            torque_consent,
            motion_consent,
            takeover_consent,
            required_hold_target,
            #[cfg(feature = "nano-agent")]
            clock,
            #[cfg(feature = "nano-agent")]
            physical_gaze: None,
            active: None,
        }
    }

    #[cfg(feature = "nano-agent")]
    fn with_physical_gaze(mut self, physical_gaze: Option<NanoPhysicalHeadGazeConfig>) -> Self {
        self.physical_gaze = physical_gaze;
        self
    }

    fn require_hold_target(
        &self,
        evidence: VerifiedHeadHealthEvidence,
    ) -> Result<VerifiedHeadHealthEvidence, NanoHeadHealthError> {
        let observed = evidence.hold_target();
        if observed == self.required_hold_target {
            return Ok(evidence);
        }
        #[cfg(feature = "nano-agent")]
        if let HeadHoldTarget::ReviewedGaze(target) | HeadHoldTarget::ReviewedCompliant(target) =
            observed
            && let Some(physical) = self
                .active
                .as_ref()
                .and_then(|active| active.physical_gaze.as_ref())
        {
            let limits = physical
                .policy
                .actuation_config()
                .controller()
                .motion_limits();
            if kiko_head_protocol::HeadJoint::ALL.into_iter().all(|joint| {
                let joint_limits = limits.joint(joint);
                let position = target.position(joint);
                position.get() >= joint_limits.minimum().get()
                    && position.get() <= joint_limits.maximum().get()
            }) {
                return Ok(evidence);
            }
            return Err(NanoHeadHealthError::GazeTargetOutsideAdmittedEnvelope { observed });
        }
        Err(NanoHeadHealthError::UnexpectedHoldTarget {
            required: self.required_hold_target,
            observed,
        })
    }

    #[cfg(feature = "nano-agent")]
    async fn process_physical_gaze_frame(
        &mut self,
        frame: &NanoFaceTrackedRgbFrame,
        character_head: CharacterHeadOverlay,
    ) -> Result<Option<AccessoryHeadFeedback>, NanoPhysicalHeadGazeRuntimeError> {
        let active = self
            .active
            .as_mut()
            .expect("gaze frames are processed only after head readiness");
        let Some(physical) = active.physical_gaze.as_mut() else {
            return Ok(None);
        };
        let Some(issuer) = physical.lease_issuer.get() else {
            return Ok(None);
        };
        let lease = match issuer.try_acquire() {
            Ok(lease) => lease,
            Err(
                HeadGazeBaseInterlockError::BaseTransactionAlreadyActive
                | HeadGazeBaseInterlockError::BaseMayMove
                | HeadGazeBaseInterlockError::HeadGazeLeaseActive,
            ) => return Ok(None),
            Err(source) => {
                return Err(NanoPhysicalHeadGazeRuntimeError::BaseInterlock(source));
            }
        };

        let projection = physical_head_gaze_projection(frame)?;
        let outcome = physical
            .policy
            .evaluate_character(
                frame.output.tracking(),
                frame.output.tracked_at(),
                projection,
                character_head,
            )
            .map_err(NanoPhysicalHeadGazeRuntimeError::Proposal)?;
        let (current_face_gaze, face_withheld) = match outcome {
            PhysicalCharacterHeadOutcome::Proposed(proposal) => (
                proposal.face().map(|face| face.gaze()),
                proposal.face_withheld(),
            ),
            PhysicalCharacterHeadOutcome::Withheld { face, .. } => (None, Some(face)),
        };
        let desired_gaze = if let Some(gaze) = current_face_gaze {
            active.last_face_gaze = Some(gaze);
            Some(gaze)
        } else if matches!(face_withheld, Some(HeadGazeFaceProposalWithheld::Coasting)) {
            active.last_face_gaze
        } else {
            active.last_face_gaze = None;
            None
        };
        let handle = active
            .handle
            .gaze()
            .expect("physical gaze policy and gaze actor are constructed together");
        if let PhysicalCharacterHeadOutcome::Proposed(proposal) = outcome {
            let detector_result_sequence = frame.output.tracking().detector_result_sequence().get();
            let proposal_id = detector_result_sequence.checked_add(1).ok_or(
                NanoPhysicalHeadGazeRuntimeError::ProposalSequenceExhausted {
                    detector_result_sequence,
                },
            )?;
            let proposal_id = HeadGazeProposalId::try_new(proposal_id)
                .map_err(NanoPhysicalHeadGazeRuntimeError::ProposalId)?;
            let observed_at = kiko_head_runtime::MonotonicTime::from_duration_since_origin(
                Duration::from_nanos(frame.output.tracked_at().nanos_since_epoch()),
            );
            handle
                .admit_gaze_proposal(HeadGazeProposal::new(
                    proposal_id,
                    observed_at,
                    proposal.command_target(),
                ))
                .await
                .map_err(NanoPhysicalHeadGazeRuntimeError::ProposalCommand)?
                .map_err(NanoPhysicalHeadGazeRuntimeError::ProposalRejected)?;
        }
        let outcome = handle
            .service_gaze(lease)
            .await
            .map_err(NanoPhysicalHeadGazeRuntimeError::ServiceCommand)?
            .map_err(|source| NanoPhysicalHeadGazeRuntimeError::Service(Box::new(source)))?;
        let mut feedback = AccessoryHeadFeedback {
            state: None,
            thermal_state: None,
            completed_episode: None,
            tracking_eye_gaze: None,
        };
        match outcome {
            HeadGazeServiceOutcome::Compliant(evidence) => {
                active.committed_target = Some(evidence.controller().committed_target());
                feedback = adapt_pet_feedback(&evidence)?;
            }
            HeadGazeServiceOutcome::Applied(evidence) => {
                active.committed_target = Some(evidence.controller().committed_target());
                feedback.thermal_state = evidence
                    .thermal_derate()
                    .map(|step| adapt_thermal_state(step.state()));
            }
            HeadGazeServiceOutcome::BeforeScheduledTick { .. }
            | HeadGazeServiceOutcome::CompliantTemperatureDeferred(_) => {}
        }
        if let Some(desired_gaze) = desired_gaze {
            let committed_target = active
                .committed_target
                .ok_or(NanoPhysicalHeadGazeRuntimeError::CommandedTargetUnavailable)?;
            let commanded_gaze = physical
                .policy
                .estimate_commanded_gaze(committed_target)
                .map_err(NanoPhysicalHeadGazeRuntimeError::CommandedGaze)?;
            feedback.tracking_eye_gaze = Some(
                HeadGazeResidual::try_between(desired_gaze, commanded_gaze)
                    .map_err(NanoPhysicalHeadGazeRuntimeError::GazeResidual)?
                    .tracking_eye_gaze(),
            );
        }
        let has_feedback = feedback.state.is_some()
            || feedback.thermal_state.is_some()
            || feedback.completed_episode.is_some()
            || feedback.tracking_eye_gaze.is_some();
        Ok(has_feedback.then_some(feedback))
    }
}

#[cfg(feature = "nano-agent")]
fn physical_head_gaze_projection(
    frame: &NanoFaceTrackedRgbFrame,
) -> Result<RgbFacePinholeProjection, NanoPhysicalHeadGazeRuntimeError> {
    let observation = frame.frame.observation();
    let layout = observation.layout();
    let oak_intrinsics = frame.frame.frame().intrinsics();
    if oak_intrinsics.width() != layout.width_px() || oak_intrinsics.height() != layout.height_px()
    {
        return Err(NanoPhysicalHeadGazeRuntimeError::ProjectionGridMismatch {
            frame_width_px: layout.width_px(),
            frame_height_px: layout.height_px(),
            intrinsic_width_px: oak_intrinsics.width(),
            intrinsic_height_px: oak_intrinsics.height(),
        });
    }
    let intrinsics = PinholeIntrinsics::try_new(
        oak_intrinsics.fx(),
        oak_intrinsics.fy(),
        oak_intrinsics.cx(),
        oak_intrinsics.cy(),
    )
    .map_err(NanoPhysicalHeadGazeRuntimeError::Intrinsics)?;
    Ok(RgbFacePinholeProjection::new(intrinsics, layout))
}

impl<F: NanoHeadGazeFrame> ReviewedNaturalHeadPort<F> for SerialReviewedNaturalHeadPort {
    type Ready = NanoHeadReadyEvidence;
    type StartError = NanoHeadActorStartupError;
    type Health = VerifiedHeadHealthEvidence;
    type HealthError = NanoHeadHealthError;
    type GazeError = NanoHeadGazePortError;
    type Shutdown = NanoHeadShutdownEvidence;

    async fn start(&mut self) -> Result<Self::Ready, Self::StartError> {
        let config = self.config.take().expect("head starts exactly once");
        #[cfg(feature = "nano-agent")]
        let physical_gaze = self.physical_gaze.take();
        #[cfg(feature = "nano-agent")]
        let physical_actuation = physical_gaze
            .as_ref()
            .map(|physical| physical.policy.actuation_config());
        #[cfg(feature = "nano-agent")]
        let initial_committed_target = physical_actuation
            .as_ref()
            .map(|actuation| actuation.controller().natural_pose());
        #[cfg(feature = "nano-agent")]
        let (serial, handle, receipt, task) = match physical_actuation {
            Some(actuation) => {
                let (serial, handle, receipt, task) =
                    kiko_head_runtime::start_serial_tension_preserving_head_gaze_actor(
                        config,
                        actuation,
                        self.clock.clone(),
                        self.torque_consent,
                        self.motion_consent,
                        self.takeover_consent,
                    )
                    .map_err(NanoHeadActorStartupError::Start)?;
                (serial, ActiveHeadHandle::Gaze(handle), receipt, task)
            }
            None => {
                let (serial, handle, receipt, task) =
                    kiko_head_runtime::start_serial_tension_preserving_head_return_actor(
                        config,
                        self.torque_consent,
                        self.motion_consent,
                        self.takeover_consent,
                    )
                    .map_err(NanoHeadActorStartupError::Start)?;
                (serial, ActiveHeadHandle::Natural(handle), receipt, task)
            }
        };
        #[cfg(not(feature = "nano-agent"))]
        let (serial, handle, receipt, task) = {
            let (serial, handle, receipt, task) =
                kiko_head_runtime::start_serial_tension_preserving_head_return_actor(
                    config,
                    self.torque_consent,
                    self.motion_consent,
                    self.takeover_consent,
                )
                .map_err(NanoHeadActorStartupError::Start)?;
            (serial, ActiveHeadHandle::Natural(handle), receipt, task)
        };
        let startup = match receipt.wait().await {
            Ok(Ok(startup)) => startup,
            Ok(Err(source)) => {
                drop(handle);
                let actor = task.join().await;
                return Err(NanoHeadActorStartupError::Runtime {
                    source: Box::new(source),
                    actor,
                });
            }
            Err(source) => {
                drop(handle);
                let actor = task.join().await;
                return Err(NanoHeadActorStartupError::Receipt { source, actor });
            }
        };
        let head_return = match handle.return_to_target().await {
            Ok(Ok(evidence)) => evidence,
            Ok(Err(source)) => {
                let hold_preserving_release = handle.release_ownership_preserving_hold().await;
                let actor = task.join().await;
                return Err(NanoHeadActorStartupError::Return {
                    source: Box::new(source),
                    startup: Box::new(startup),
                    hold_preserving_release,
                    actor,
                });
            }
            Err(source) => {
                let hold_preserving_release = handle.release_ownership_preserving_hold().await;
                let actor = task.join().await;
                return Err(NanoHeadActorStartupError::ReturnCommand {
                    source,
                    startup: Box::new(startup),
                    hold_preserving_release,
                    actor,
                });
            }
        };
        let initial_health = match handle
            .check_health()
            .await
            .map_err(NanoHeadHealthError::Request)
            .and_then(|evidence| self.require_hold_target(evidence))
        {
            Ok(evidence) => evidence,
            Err(source) => {
                let hold_preserving_release = handle.release_ownership_preserving_hold().await;
                let actor = task.join().await;
                return Err(NanoHeadActorStartupError::PostReturnHealth {
                    source,
                    startup: Box::new(startup),
                    head_return: Box::new(head_return),
                    hold_preserving_release,
                    actor,
                });
            }
        };
        self.active = Some(ActiveHeadActor {
            handle,
            task,
            startup: startup.clone(),
            head_return: head_return.clone(),
            #[cfg(feature = "nano-agent")]
            physical_gaze,
            #[cfg(feature = "nano-agent")]
            committed_target: initial_committed_target,
            #[cfg(feature = "nano-agent")]
            last_face_gaze: None,
        });
        Ok(NanoHeadReadyEvidence {
            serial,
            startup,
            head_return,
            initial_health,
        })
    }

    async fn check_health(&mut self) -> Result<Self::Health, Self::HealthError> {
        self.active
            .as_ref()
            .expect("health checks occur only after head readiness")
            .handle
            .check_health()
            .await
            .map_err(NanoHeadHealthError::Request)
            .and_then(|evidence| self.require_hold_target(evidence))
    }

    async fn process_gaze_frame(
        &mut self,
        frame: &F,
        character_head: CharacterHeadOverlay,
    ) -> Result<Option<AccessoryHeadFeedback>, Self::GazeError> {
        #[cfg(feature = "nano-agent")]
        {
            let Some(frame) = frame.tracked_face_frame() else {
                return Ok(None);
            };
            self.process_physical_gaze_frame(frame, character_head)
                .await
                .map_err(NanoHeadGazePortError::Physical)
        }
        #[cfg(not(feature = "nano-agent"))]
        {
            let _ = (frame, character_head);
            Ok(None)
        }
    }

    fn character_base_motion_state(&self) -> CharacterBaseMotionState {
        #[cfg(feature = "nano-agent")]
        {
            let Some(physical) = self
                .active
                .as_ref()
                .and_then(|active| active.physical_gaze.as_ref())
            else {
                return CharacterBaseMotionState::NotApplicable;
            };
            physical
                .lease_issuer
                .get()
                .map(|issuer| adapt_base_motion_state(issuer.phase()))
                .unwrap_or(CharacterBaseMotionState::Unavailable)
        }
        #[cfg(not(feature = "nano-agent"))]
        {
            CharacterBaseMotionState::NotApplicable
        }
    }

    async fn shutdown(&mut self) -> Self::Shutdown {
        let active = self
            .active
            .take()
            .expect("shutdown occurs only after head readiness");
        let hold_preserving_release = active.handle.release_ownership_preserving_hold().await;
        let actor = active.task.join().await;
        NanoHeadShutdownEvidence {
            startup: active.startup,
            head_return: active.head_return,
            hold_preserving_release,
            actor,
        }
    }
}

impl RgbBridgePort<IngressObservedRgbFrame<ImageFrame>>
    for RgbExpressionBridge<NanoAccessoryClock>
{
    type Intent = PreparedCharacterFrame;
    type Error = RgbExpressionBridgeError;

    fn process(
        &mut self,
        frame: &IngressObservedRgbFrame<ImageFrame>,
    ) -> Result<Self::Intent, Self::Error> {
        self.process_queued_oak_frame_borrowed(frame)
            .map(|outcome| outcome.into_character())
    }

    fn note_pet_state(&mut self, state: CharacterPetState) {
        RgbExpressionBridge::note_pet_state(self, state);
    }

    fn note_thermal_state(&mut self, state: CharacterThermalState) {
        RgbExpressionBridge::note_thermal_state(self, state);
    }

    fn note_base_motion_state(&mut self, state: CharacterBaseMotionState) {
        RgbExpressionBridge::note_base_motion_state(self, state);
    }

    fn note_lifecycle_state(&mut self, state: CharacterLifecycleState) {
        RgbExpressionBridge::note_lifecycle_state(self, state);
    }

    fn prepare_lifecycle_reflex(
        &mut self,
        state: CharacterLifecycleState,
    ) -> Result<Self::Intent, Self::Error> {
        RgbExpressionBridge::prepare_lifecycle_reflex(self, state)
    }

    fn lifecycle_reflex_duration(&self, state: CharacterLifecycleState) -> Duration {
        state.presentation_transition_duration()
    }

    fn note_pet_episode(&mut self, feedback: AccessoryPetEpisode) {
        let _reaction =
            RgbExpressionBridge::note_pet_episode(self, feedback.completed_at, feedback.episode);
    }
}

fn process_rgb_work<F, B>(
    bridge: &mut B,
    work: &Result<F, ClockError>,
) -> Result<B::Intent, RgbExpressionBridgeError>
where
    B: RgbBridgePort<F, Error = RgbExpressionBridgeError>,
{
    match work {
        Ok(frame) => bridge.process(frame),
        Err(source) => Err(RgbExpressionBridgeError::Clock(source.clone())),
    }
}

impl RgbBridgePort<NanoAccessoryRgbWork> for RgbExpressionBridge<NanoAccessoryClock> {
    type Intent = PreparedCharacterFrame;
    type Error = RgbExpressionBridgeError;

    fn process(&mut self, work: &NanoAccessoryRgbWork) -> Result<Self::Intent, Self::Error> {
        process_rgb_work(self, work)
    }

    fn note_pet_state(&mut self, state: CharacterPetState) {
        RgbExpressionBridge::note_pet_state(self, state);
    }

    fn note_thermal_state(&mut self, state: CharacterThermalState) {
        RgbExpressionBridge::note_thermal_state(self, state);
    }

    fn note_base_motion_state(&mut self, state: CharacterBaseMotionState) {
        RgbExpressionBridge::note_base_motion_state(self, state);
    }

    fn note_lifecycle_state(&mut self, state: CharacterLifecycleState) {
        RgbExpressionBridge::note_lifecycle_state(self, state);
    }

    fn prepare_lifecycle_reflex(
        &mut self,
        state: CharacterLifecycleState,
    ) -> Result<Self::Intent, Self::Error> {
        RgbExpressionBridge::prepare_lifecycle_reflex(self, state)
    }

    fn lifecycle_reflex_duration(&self, state: CharacterLifecycleState) -> Duration {
        state.presentation_transition_duration()
    }

    fn note_pet_episode(&mut self, feedback: AccessoryPetEpisode) {
        let _reaction =
            RgbExpressionBridge::note_pet_episode(self, feedback.completed_at, feedback.episode);
    }
}

#[cfg(any(
    feature = "nano-wheels-off-qualification",
    feature = "nano-base-commissioning",
    test
))]
struct ProductionSceneRgbBridge(RgbExpressionBridge<NanoAccessoryClock>);

#[cfg(any(
    feature = "nano-wheels-off-qualification",
    feature = "nano-base-commissioning",
    test
))]
impl RgbBridgePort<NanoAccessoryRgbWork> for ProductionSceneRgbBridge {
    type Intent = PreparedCharacterFrame;
    type Error = NanoAccessoryRgbProcessingError;

    fn process(&mut self, work: &NanoAccessoryRgbWork) -> Result<Self::Intent, Self::Error> {
        self.0
            .process(work)
            .map_err(NanoAccessoryRgbProcessingError::Bridge)
    }

    fn note_pet_state(&mut self, state: CharacterPetState) {
        self.0.note_pet_state(state);
    }

    fn note_thermal_state(&mut self, state: CharacterThermalState) {
        self.0.note_thermal_state(state);
    }

    fn note_base_motion_state(&mut self, state: CharacterBaseMotionState) {
        self.0.note_base_motion_state(state);
    }

    fn note_lifecycle_state(&mut self, state: CharacterLifecycleState) {
        self.0.note_lifecycle_state(state);
    }

    fn prepare_lifecycle_reflex(
        &mut self,
        state: CharacterLifecycleState,
    ) -> Result<Self::Intent, Self::Error> {
        self.0
            .prepare_lifecycle_reflex(state)
            .map_err(NanoAccessoryRgbProcessingError::Bridge)
    }

    fn lifecycle_reflex_duration(&self, state: CharacterLifecycleState) -> Duration {
        state.presentation_transition_duration()
    }

    fn note_pet_episode(&mut self, feedback: AccessoryPetEpisode) {
        let _reaction = self
            .0
            .note_pet_episode(feedback.completed_at, feedback.episode);
    }
}

#[cfg(feature = "nano-agent")]
struct ProductionFaceRgbBridge(RgbExpressionBridge<NanoAccessoryClock>);

#[cfg(feature = "nano-agent")]
impl RgbBridgePort<NanoFacePerceptionWork> for ProductionFaceRgbBridge {
    type Intent = PreparedCharacterFrame;
    type Error = NanoAccessoryRgbProcessingError;

    fn process(&mut self, work: &NanoFacePerceptionWork) -> Result<Self::Intent, Self::Error> {
        let frame = work
            .as_ref()
            .map_err(|source| NanoAccessoryRgbProcessingError::FacePerception(source.clone()))?;
        self.0
            .process_queued_oak_frame_with_face(&frame.frame, frame.output.tracking())
            .map(|outcome| outcome.into_character())
            .map_err(NanoAccessoryRgbProcessingError::Bridge)
    }

    fn note_pet_state(&mut self, state: CharacterPetState) {
        self.0.note_pet_state(state);
    }

    fn note_thermal_state(&mut self, state: CharacterThermalState) {
        self.0.note_thermal_state(state);
    }

    fn note_base_motion_state(&mut self, state: CharacterBaseMotionState) {
        self.0.note_base_motion_state(state);
    }

    fn note_lifecycle_state(&mut self, state: CharacterLifecycleState) {
        self.0.note_lifecycle_state(state);
    }

    fn prepare_lifecycle_reflex(
        &mut self,
        state: CharacterLifecycleState,
    ) -> Result<Self::Intent, Self::Error> {
        self.0
            .prepare_lifecycle_reflex(state)
            .map_err(NanoAccessoryRgbProcessingError::Bridge)
    }

    fn lifecycle_reflex_duration(&self, state: CharacterLifecycleState) -> Duration {
        state.presentation_transition_duration()
    }

    fn note_pet_episode(&mut self, feedback: AccessoryPetEpisode) {
        let _reaction = self
            .0
            .note_pet_episode(feedback.completed_at, feedback.episode);
    }
}

#[cfg(any(
    feature = "nano-wheels-off-qualification",
    feature = "nano-base-commissioning",
    test
))]
#[allow(clippy::too_many_arguments)]
fn run_production_worker(
    config: NanoAccessoryWorkerConfig,
    channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>>,
    latest_head_health: Arc<Mutex<NanoAccessoryHeadHealthState>>,
    startup_tx: std::sync::mpsc::SyncSender<StartupSignal>,
    fault_tx: crossbeam_channel::Sender<NanoAccessoryTerminalFault>,
    expression_clock: NanoAccessoryClock,
    perception_ready: NanoAccessoryPerceptionReadyEvidence,
    lifecycle: Arc<NanoAccessoryOwnerLifecycle>,
) -> NanoAccessoryWorkerExit {
    let bridge = ProductionSceneRgbBridge(RgbExpressionBridge::new(
        config.stream_epoch,
        config.rgb_expression,
        expression_clock.clone(),
    ));
    run_production_worker_core(
        config,
        channel,
        latest_head_health,
        startup_tx,
        fault_tx,
        expression_clock,
        bridge,
        perception_ready,
        false,
        || {},
        lifecycle,
    )
}

#[cfg(feature = "nano-agent")]
#[allow(clippy::too_many_arguments)]
fn run_production_worker_with_face(
    config: NanoAccessoryWorkerConfig,
    channel: Arc<LatestFrameChannel<NanoFacePerceptionWork>>,
    raw_channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>>,
    latest_head_health: Arc<Mutex<NanoAccessoryHeadHealthState>>,
    startup_tx: std::sync::mpsc::SyncSender<StartupSignal>,
    fault_tx: crossbeam_channel::Sender<NanoAccessoryTerminalFault>,
    expression_clock: NanoAccessoryClock,
    perception_ready: NanoAccessoryPerceptionReadyEvidence,
    lifecycle: Arc<NanoAccessoryOwnerLifecycle>,
) -> NanoAccessoryWorkerExit {
    let bridge = ProductionFaceRgbBridge(RgbExpressionBridge::new(
        config.stream_epoch,
        config.rgb_expression,
        expression_clock.clone(),
    ));
    run_production_worker_core(
        config,
        channel,
        latest_head_health,
        startup_tx,
        fault_tx,
        expression_clock,
        bridge,
        perception_ready,
        true,
        move || raw_channel.latch_terminal_fault(),
        lifecycle,
    )
}

#[cfg(feature = "nano-agent")]
fn shutdown_pet_evidence_journal(
    journal: Option<NanoPetEvidenceJournal>,
) -> NanoPetEvidenceLifecycleShutdownEvidence {
    match journal {
        Some(journal) => NanoPetEvidenceLifecycleShutdownEvidence::Enabled(
            journal.shutdown(NANO_PET_EVIDENCE_JOURNAL_SHUTDOWN_TIMEOUT),
        ),
        None => NanoPetEvidenceLifecycleShutdownEvidence::Disabled,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_production_worker_core<F, B, LatchFault>(
    mut config: NanoAccessoryWorkerConfig,
    channel: Arc<LatestFrameChannel<F>>,
    latest_head_health: Arc<Mutex<NanoAccessoryHeadHealthState>>,
    startup_tx: std::sync::mpsc::SyncSender<StartupSignal>,
    fault_tx: crossbeam_channel::Sender<NanoAccessoryTerminalFault>,
    expression_clock: NanoAccessoryClock,
    bridge: B,
    perception_ready: NanoAccessoryPerceptionReadyEvidence,
    face_perception_enabled: bool,
    mut latch_fault: LatchFault,
    lifecycle: Arc<NanoAccessoryOwnerLifecycle>,
) -> NanoAccessoryWorkerExit
where
    F: NanoHeadGazeFrame + Send + 'static,
    B: RgbBridgePort<F, Intent = PreparedCharacterFrame, Error = NanoAccessoryRgbProcessingError>,
    LatchFault: FnMut(),
{
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(source) => {
            let _ = startup_tx.send(StartupSignal::Failed);
            return NanoAccessoryWorkerExit::RuntimeBuildFailed {
                message: source.to_string().into_boxed_str(),
            };
        }
    };

    let mut session_generator = OsEyeSessionMaterialGenerator;
    let eye_config = match config.eye.new_session(&mut session_generator) {
        Ok(config) => config,
        Err(source) => {
            let _ = startup_tx.send(StartupSignal::Failed);
            return NanoAccessoryWorkerExit::EyeSessionMaterialFailed(source);
        }
    };
    #[cfg(feature = "nano-agent")]
    let mut pet_evidence_journal = match config.pet_evidence_journal.take() {
        Some(config) => match NanoPetEvidenceJournal::start(config) {
            Ok(journal) => Some(journal),
            Err(source) => {
                let _ = startup_tx.send(StartupSignal::Failed);
                return NanoAccessoryWorkerExit::PetEvidenceJournalStartupFailed(source);
            }
        },
        None => None,
    };
    let eye_clock = expression_clock.clone();
    let eye = SerialKep2EyePort::new(eye_config, eye_clock);
    let head = SerialReviewedNaturalHeadPort::new(
        config.head_return,
        config.head_torque_consent,
        config.head_motion_consent,
        config.head_takeover_consent,
        config.required_hold_target,
        #[cfg(feature = "nano-agent")]
        expression_clock.clone(),
    );
    #[cfg(feature = "nano-agent")]
    let head = head.with_physical_gaze(config.physical_head_gaze);
    let stream_epoch = config.stream_epoch;
    let health_period = config.health_period;
    let rgb_frame_freshness =
        Duration::from_nanos(config.rgb_expression.frame_freshness().as_nanos());
    #[cfg(feature = "nano-agent")]
    let pet_evidence_ready = match pet_evidence_journal.as_ref() {
        Some(journal) => {
            NanoPetEvidenceLifecycleReadyEvidence::Enabled(journal.readiness().clone())
        }
        None => NanoPetEvidenceLifecycleReadyEvidence::Disabled,
    };
    let readiness_head_health = Arc::clone(&latest_head_health);
    let readiness_lifecycle = Arc::clone(&lifecycle);
    let health_lifecycle = Arc::clone(&lifecycle);
    let fault_lifecycle = Arc::clone(&lifecycle);

    let core_exit = runtime.block_on(run_accessory_core(
        head,
        eye,
        bridge,
        channel,
        health_period,
        #[cfg(feature = "nano-agent")]
        pet_evidence_journal.as_mut(),
        CoreObservers {
            ready: move |head: NanoHeadReadyEvidence, eye: NanoEyeReadyEvidence| {
                let initial_health_recorded = readiness_head_health
                    .lock()
                    .map(|mut latest| {
                        latest.record(head.initial_health().clone());
                    })
                    .is_ok();
                if !initial_health_recorded {
                    return false;
                }
                readiness_lifecycle.mark_running();
                startup_tx
                    .send(StartupSignal::Ready(
                        NanoAccessoryReadyEvidence {
                            eye,
                            head,
                            perception: perception_ready,
                            #[cfg(feature = "nano-agent")]
                            pet_evidence: pet_evidence_ready,
                            stream_epoch,
                            health_period,
                            rgb_frame_freshness,
                        }
                        .into(),
                    ))
                    .is_ok()
            },
            record_health: move |evidence| match latest_head_health.lock() {
                Ok(mut latest) => {
                    latest.record(evidence);
                    health_lifecycle.record_completed_health_transaction()
                }
                Err(_) => false,
            },
            publish_fault: move |fault| {
                let fault = map_production_core_fault(fault, face_perception_enabled);
                let _ = fault_tx.try_send(fault);
            },
            latch_fault: move || {
                fault_lifecycle.mark_fault_latched();
                latch_fault();
            },
        },
    ));

    #[cfg(feature = "nano-agent")]
    let pet_evidence = shutdown_pet_evidence_journal(pet_evidence_journal);

    match core_exit {
        CoreExit::EyeStartupFailed(source) => {
            // No ready signal was sent; wake the synchronous starter.
            // Sending can fail only if that starter disappeared.
            // `startup_tx` was moved into the readiness closure, so the channel
            // also disconnects here and wakes `recv`.
            NanoAccessoryWorkerExit::EyeStartupFailed {
                source: Box::new(source),
                #[cfg(feature = "nano-agent")]
                pet_evidence,
            }
        }
        CoreExit::HeadStartupFailed {
            source,
            eye_shutdown,
        } => NanoAccessoryWorkerExit::HeadStartupFailed {
            source: Box::new(source),
            eye_shutdown: Box::new(eye_shutdown),
            #[cfg(feature = "nano-agent")]
            pet_evidence,
        },
        CoreExit::Shutdown {
            terminal_fault,
            fault_recovery_presentation,
            eye_shutdown,
            head_shutdown,
        } => {
            let terminal_fault = terminal_fault
                .map(|fault| map_production_core_fault(fault, face_perception_enabled));
            let fault_recovery_presentation =
                map_production_fault_recovery_presentation(fault_recovery_presentation);
            NanoAccessoryWorkerExit::Shutdown {
                terminal_fault,
                evidence: Box::new(NanoAccessoryShutdownEvidence {
                    eye: eye_shutdown,
                    head: head_shutdown,
                    fault_recovery_presentation,
                    #[cfg(feature = "nano-agent")]
                    pet_evidence,
                    #[cfg(feature = "nano-agent")]
                    face_perception: NanoFacePerceptionShutdownEvidence::Disabled,
                }),
            }
        }
    }
}

fn map_production_fault_recovery_presentation(
    evidence: CoreFaultRecoveryPresentationEvidence<
        NanoAccessoryRgbProcessingError,
        EyeHandleRequestError,
    >,
) -> NanoFaultRecoveryPresentationEvidence {
    match evidence {
        CoreFaultRecoveryPresentationEvidence::NotRequired => {
            NanoFaultRecoveryPresentationEvidence::NotRequired
        }
        CoreFaultRecoveryPresentationEvidence::Presented { frames_applied } => {
            NanoFaultRecoveryPresentationEvidence::Presented { frames_applied }
        }
        CoreFaultRecoveryPresentationEvidence::MissingPresentationDuration => {
            NanoFaultRecoveryPresentationEvidence::Failed {
                frames_applied: 0,
                source: NanoFaultRecoveryPresentationFault::MissingPresentationDuration,
            }
        }
        CoreFaultRecoveryPresentationEvidence::Failed(LifecyclePresentationFault::Bridge {
            frames_applied,
            source: NanoAccessoryRgbProcessingError::Bridge(source),
        }) => NanoFaultRecoveryPresentationEvidence::Failed {
            frames_applied,
            source: NanoFaultRecoveryPresentationFault::ExpressionBridge(source),
        },
        #[cfg(feature = "nano-agent")]
        CoreFaultRecoveryPresentationEvidence::Failed(LifecyclePresentationFault::Bridge {
            frames_applied,
            source: NanoAccessoryRgbProcessingError::FacePerception(source),
        }) => NanoFaultRecoveryPresentationEvidence::Failed {
            frames_applied,
            source: NanoFaultRecoveryPresentationFault::FacePerception(source),
        },
        CoreFaultRecoveryPresentationEvidence::Failed(LifecyclePresentationFault::EyeApply {
            frames_applied,
            source,
        }) => NanoFaultRecoveryPresentationEvidence::Failed {
            frames_applied,
            source: NanoFaultRecoveryPresentationFault::EyeApply(source),
        },
        CoreFaultRecoveryPresentationEvidence::Failed(
            LifecyclePresentationFault::FrameCountExhausted { frames_applied },
        ) => NanoFaultRecoveryPresentationEvidence::Failed {
            frames_applied,
            source: NanoFaultRecoveryPresentationFault::FrameCountExhausted,
        },
    }
}

fn map_production_core_fault(
    fault: CoreTerminalFault<
        NanoHeadHealthError,
        NanoHeadGazePortError,
        NanoAccessoryRgbProcessingError,
        EyeHandleRequestError,
    >,
    face_perception_enabled: bool,
) -> NanoAccessoryTerminalFault {
    #[cfg(not(feature = "nano-agent"))]
    let _ = face_perception_enabled;
    match fault {
        CoreTerminalFault::HeadHealth(source) => NanoAccessoryTerminalFault::HeadHealth(source),
        #[cfg(feature = "nano-agent")]
        CoreTerminalFault::HeadGaze(NanoHeadGazePortError::Physical(source)) => {
            NanoAccessoryTerminalFault::PhysicalHeadGaze(source)
        }
        #[cfg(not(feature = "nano-agent"))]
        CoreTerminalFault::HeadGaze(source) => match source {},
        CoreTerminalFault::HeadHealthStatusPoisoned => {
            NanoAccessoryTerminalFault::HeadHealthStatusPoisoned
        }
        CoreTerminalFault::RgbHealthStatusPoisoned => {
            NanoAccessoryTerminalFault::RgbHealthStatusPoisoned
        }
        CoreTerminalFault::Bridge(NanoAccessoryRgbProcessingError::Bridge(source)) => {
            NanoAccessoryTerminalFault::ExpressionBridge(source)
        }
        #[cfg(feature = "nano-agent")]
        CoreTerminalFault::Bridge(NanoAccessoryRgbProcessingError::FacePerception(source)) => {
            NanoAccessoryTerminalFault::FacePerception(source)
        }
        CoreTerminalFault::EyeApply(source) => NanoAccessoryTerminalFault::EyeApply(source),
        CoreTerminalFault::CharacterCoordination(source) => {
            NanoAccessoryTerminalFault::CharacterCoordination(source)
        }
        CoreTerminalFault::LifecyclePresentationFrameCountExhausted => {
            NanoAccessoryTerminalFault::LifecyclePresentationFrameCountExhausted
        }
        #[cfg(feature = "nano-agent")]
        CoreTerminalFault::PetEvidence(source) => NanoAccessoryTerminalFault::PetEvidence(source),
        CoreTerminalFault::IngressDisconnected => {
            #[cfg(feature = "nano-agent")]
            if face_perception_enabled {
                return NanoAccessoryTerminalFault::FacePerception(
                    NanoFacePerceptionRuntimeError::PerceptionOutputDisconnected,
                );
            }
            NanoAccessoryTerminalFault::RgbIngressDisconnected
        }
        CoreTerminalFault::ChannelPoisoned => {
            #[cfg(feature = "nano-agent")]
            if face_perception_enabled {
                return NanoAccessoryTerminalFault::FacePerception(
                    NanoFacePerceptionRuntimeError::PerceptionOutputChannelPoisoned,
                );
            }
            NanoAccessoryTerminalFault::RgbChannelPoisoned
        }
        CoreTerminalFault::ReadinessObserverDropped => {
            NanoAccessoryTerminalFault::ReadinessObserverDropped
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "nano-agent")]
    use std::fs;
    #[cfg(feature = "nano-agent")]
    use std::path::PathBuf;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    fn health_period(milliseconds: u64) -> NanoAccessoryHealthPeriod {
        NanoAccessoryHealthPeriod::try_from_duration(Duration::from_millis(milliseconds))
            .expect("valid test period")
    }

    #[cfg(feature = "nano-agent")]
    static NEXT_PET_JOURNAL_DIRECTORY: AtomicUsize = AtomicUsize::new(0);

    #[cfg(feature = "nano-agent")]
    struct PetJournalTestDirectory(PathBuf);

    #[cfg(feature = "nano-agent")]
    impl PetJournalTestDirectory {
        fn create() -> Self {
            let base = fs::canonicalize(std::env::temp_dir()).expect("canonical temp root");
            for _ in 0..1_000 {
                let serial = NEXT_PET_JOURNAL_DIRECTORY.fetch_add(1, Ordering::Relaxed);
                let path = base.join(format!(
                    "kiko-accessory-pet-journal-{}-{serial}",
                    std::process::id()
                ));
                match fs::create_dir(&path) {
                    Ok(()) => return Self(path),
                    Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(source) => panic!("create pet-journal test root: {source}"),
                }
            }
            panic!("could not allocate pet-journal test root")
        }
    }

    #[cfg(feature = "nano-agent")]
    impl Drop for PetJournalTestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn health_period_is_nonzero_and_bounded() {
        assert_eq!(
            NanoAccessoryHealthPeriod::try_from_duration(Duration::ZERO),
            Err(NanoAccessoryHealthPeriodError::Zero)
        );
        assert_eq!(
            NanoAccessoryHealthPeriod::try_from_duration(Duration::from_secs(6)),
            Err(NanoAccessoryHealthPeriodError::AboveMaximum {
                actual: Duration::from_secs(6),
                maximum: MAX_NANO_ACCESSORY_HEALTH_PERIOD,
            })
        );
        assert_eq!(health_period(25).get(), Duration::from_millis(25));
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn base_interlock_phase_maps_exhaustively_without_guessing_stationary() {
        assert_eq!(
            adapt_base_motion_state(HeadGazeBaseMotionPhase::ConfirmedZero),
            CharacterBaseMotionState::ConfirmedStationary
        );
        assert_eq!(
            adapt_base_motion_state(HeadGazeBaseMotionPhase::HeadGazeTransaction),
            CharacterBaseMotionState::ConfirmedStationary
        );
        assert_eq!(
            adapt_base_motion_state(HeadGazeBaseMotionPhase::BaseTransaction),
            CharacterBaseMotionState::CommandTransaction
        );
        assert_eq!(
            adapt_base_motion_state(HeadGazeBaseMotionPhase::BaseMayMove),
            CharacterBaseMotionState::MayMove
        );
        assert_eq!(
            adapt_base_motion_state(HeadGazeBaseMotionPhase::Faulted),
            CharacterBaseMotionState::Faulted
        );
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn canonical_face_detector_config_exactly_matches_reviewed_kiko_policy() {
        let config = canonical_nano_face_detector_config().expect("canonical config");
        assert_eq!(config.scale_factor(), 1.15);
        assert_eq!(config.frontal_minimum_neighbors(), 6);
        assert_eq!(config.profile_minimum_neighbors(), 4);
        assert_eq!(config.minimum_face_width(), 30);
        assert_eq!(config.minimum_face_height(), 30);
        assert_eq!(
            config.maximum_retained_detections(),
            u32::try_from(MAX_FACE_DETECTIONS).unwrap()
        );
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn compliant_dispositions_cross_the_character_boundary_without_lossy_fallbacks() {
        assert_eq!(
            adapt_pet_state(CompliantHoldDisposition::FollowingExpression).unwrap(),
            CharacterPetState::Ready
        );
        assert_eq!(
            adapt_pet_state(CompliantHoldDisposition::ContactCandidate {
                consecutive_samples: 2,
            })
            .unwrap(),
            CharacterPetState::ContactCandidate {
                consecutive_samples: NonZeroU8::new(2).unwrap(),
            }
        );
        assert_eq!(
            adapt_pet_state(CompliantHoldDisposition::Yielding {
                envelope_limited: [false; 4],
                command_step_limited: [true; 4],
            })
            .unwrap(),
            CharacterPetState::Yielding
        );
        assert_eq!(
            adapt_pet_state(CompliantHoldDisposition::ReleaseDwell).unwrap(),
            CharacterPetState::ReleaseDwell
        );
        assert_eq!(
            adapt_pet_state(CompliantHoldDisposition::Resting {
                command_step_limited: [false; 4],
                at_rest_target: true,
            })
            .unwrap(),
            CharacterPetState::Resting {
                at_rest_target: true,
            }
        );
        assert_eq!(
            adapt_pet_state(CompliantHoldDisposition::Recovering {
                progress_millionths: 500_000,
                command_step_limited: [false; 4],
            })
            .unwrap(),
            CharacterPetState::Recovering {
                progress: CoreUnitAmount::try_from_basis_points(5_000).unwrap(),
            }
        );
        assert_eq!(
            adapt_pet_state(CompliantHoldDisposition::ReturnedToExpression).unwrap(),
            CharacterPetState::Ready
        );
        assert_eq!(
            adapt_pet_state(CompliantHoldDisposition::Recovering {
                progress_millionths: 999_999,
                command_step_limited: [false; 4],
            })
            .unwrap(),
            CharacterPetState::Recovering {
                progress: CoreUnitAmount::try_from_basis_points(9_999).unwrap(),
            },
            "character recovery cannot become exact before controller recovery"
        );
        assert_eq!(
            adapt_pet_state(CompliantHoldDisposition::ContactCandidate {
                consecutive_samples: 0,
            }),
            Err(NanoPhysicalHeadGazeRuntimeError::PetContactCandidateHasZeroSamples)
        );
        assert_eq!(
            adapt_pet_state(CompliantHoldDisposition::Recovering {
                progress_millionths: 1_000_001,
                command_step_limited: [false; 4],
            }),
            Err(
                NanoPhysicalHeadGazeRuntimeError::PetRecoveryProgressOutOfRange {
                    progress_millionths: 1_000_001,
                }
            )
        );
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn concurrent_head_gaze_diagnostic_health_accounting_is_exact_and_disjoint() {
        const UPDATES_PER_CLASS: u64 = 1_000;
        let counters = Arc::new(NanoFacePerceptionStageCounters::new());
        let actuation = NanoHeadGazeActuationAvailability::UnavailableWithoutBaseZeroExclusiveLease;
        let projection = NanoHeadGazeRgbProjectionEvidence {
            fx_px: 398.0,
            fy_px: 399.0,
            cx_px: 319.5,
            cy_px: 199.5,
            width_px: 640,
            height_px: 400,
        };
        let disabled = NanoHeadGazeDiagnostic::DisabledNoPolicy { actuation };
        let withheld = NanoHeadGazeDiagnostic::ProposalOnly {
            evaluated_at: MonotonicTimestamp::from_nanos_since_epoch(10),
            projection,
            outcome: Ok(HeadGazeFaceProposalOutcome::Withheld(
                crate::navigation::HeadGazeFaceProposalWithheld::NoTarget,
            )),
            actuation,
        };
        let rejected = NanoHeadGazeDiagnostic::ProposalOnly {
            evaluated_at: MonotonicTimestamp::from_nanos_since_epoch(11),
            projection,
            outcome: Err(NanoHeadGazeDiagnosticError::ProjectionGridMismatch {
                frame_width_px: 640,
                frame_height_px: 400,
                intrinsic_width_px: 320,
                intrinsic_height_px: 200,
            }),
            actuation,
        };

        thread::scope(|scope| {
            for diagnostic in [disabled, withheld, rejected] {
                let counters = Arc::clone(&counters);
                scope.spawn(move || {
                    for _ in 0..UPDATES_PER_CLASS {
                        counters.record_head_gaze(diagnostic);
                    }
                });
            }
        });

        let snapshot = counters.snapshot();
        assert_eq!(snapshot.head_gaze_disabled_no_policy, UPDATES_PER_CLASS);
        assert_eq!(snapshot.head_gaze_withheld, UPDATES_PER_CLASS);
        assert_eq!(snapshot.head_gaze_rejected, UPDATES_PER_CLASS);
        assert_eq!(snapshot.head_gaze_proposed, 0);
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn face_shutdown_disabled_is_explicit_and_rejects_impossible_face_fault() {
        let evidence = NanoFacePerceptionShutdownEvidence::Disabled;
        assert!(evidence.join_evidence().is_none());

        let ordinary_fault = NanoAccessoryTerminalFault::ReadinessObserverDropped;
        let class = evidence.classify(Some(&ordinary_fault));
        assert!(matches!(class, NanoFacePerceptionShutdownClass::Disabled));
        assert!(class.is_healthy());
        assert!(class.is_coordinated());
        assert!(!class.is_uncertain_or_unexpected());

        let terminal_source = NanoFacePerceptionRuntimeError::RgbIngressDisconnected;
        let impossible_fault = NanoAccessoryTerminalFault::FacePerception(terminal_source.clone());
        let class = evidence.classify(Some(&impossible_fault));
        assert!(matches!(
            class,
            NanoFacePerceptionShutdownClass::UnexpectedDisabledFaceFault {
                terminal_source: observed,
            } if observed == &terminal_source
        ));
        assert!(!class.is_healthy());
        assert!(!class.is_coordinated());
        assert!(class.is_uncertain_or_unexpected());
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn face_shutdown_runtime_fault_requires_exact_published_terminal_pair() {
        let thread_source = NanoFacePerceptionRuntimeError::RgbIngressDisconnected;
        let evidence = NanoFacePerceptionShutdownEvidence::Join(
            NanoFacePerceptionJoinEvidence::Joined(NanoFacePerceptionThreadExit::RuntimeFault {
                source: thread_source.clone(),
                published_to_accessory: true,
            }),
        );
        let exact_terminal = NanoAccessoryTerminalFault::FacePerception(thread_source.clone());
        let class = evidence.classify(Some(&exact_terminal));
        assert!(matches!(
            class,
            NanoFacePerceptionShutdownClass::PublishedRuntimeFault {
                thread_source: observed_thread,
                terminal_source: observed_terminal,
            } if observed_thread == &thread_source && observed_terminal == &thread_source
        ));
        assert!(!class.is_healthy());
        assert!(class.is_coordinated());

        let mismatched_terminal = NanoAccessoryTerminalFault::FacePerception(
            NanoFacePerceptionRuntimeError::PerceptionOutputDisconnected,
        );
        assert!(matches!(
            evidence.classify(Some(&mismatched_terminal)),
            NanoFacePerceptionShutdownClass::UnexpectedJoined { .. }
        ));

        let unpublished = NanoFacePerceptionShutdownEvidence::Join(
            NanoFacePerceptionJoinEvidence::Joined(NanoFacePerceptionThreadExit::RuntimeFault {
                source: thread_source,
                published_to_accessory: false,
            }),
        );
        assert!(matches!(
            unpublished.classify(Some(&exact_terminal)),
            NanoFacePerceptionShutdownClass::UnexpectedJoined { .. }
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn face_shutdown_classifies_only_evidenced_shutdown_and_fault_followers_as_coordinated() {
        let shutdown = NanoFacePerceptionShutdownEvidence::Join(
            NanoFacePerceptionJoinEvidence::Joined(NanoFacePerceptionThreadExit::Shutdown),
        );
        let class = shutdown.classify(None);
        assert!(matches!(
            class,
            NanoFacePerceptionShutdownClass::CoordinatedShutdown
        ));
        assert!(class.is_healthy());
        assert!(class.is_coordinated());

        let terminal_fault = NanoAccessoryTerminalFault::ReadinessObserverDropped;
        assert!(matches!(
            shutdown.classify(Some(&terminal_fault)),
            NanoFacePerceptionShutdownClass::UnexpectedJoined { .. }
        ));

        for exit in [
            NanoFacePerceptionThreadExit::AccessoryFaultPendingPublication,
            NanoFacePerceptionThreadExit::AccessoryFaultLatched,
        ] {
            let follower = NanoFacePerceptionShutdownEvidence::Join(
                NanoFacePerceptionJoinEvidence::Joined(exit),
            );
            let class = follower.classify(Some(&terminal_fault));
            assert!(matches!(
                class,
                NanoFacePerceptionShutdownClass::AccessoryFaultFollower { .. }
            ));
            assert!(!class.is_healthy());
            assert!(class.is_coordinated());
            assert!(matches!(
                follower.classify(None),
                NanoFacePerceptionShutdownClass::UnexpectedJoined { .. }
            ));
        }
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn detached_face_shutdown_is_always_uncertain_and_retains_both_budgets() {
        let evidence = NanoFacePerceptionShutdownEvidence::Join(
            NanoFacePerceptionJoinEvidence::DetachedAfterTimeout {
                configured_timeout: Duration::from_secs(2),
                active_join_budget: Duration::from_millis(125),
            },
        );
        let terminal_fault = NanoAccessoryTerminalFault::ReadinessObserverDropped;
        let class = evidence.classify(Some(&terminal_fault));
        assert!(matches!(
            class,
            NanoFacePerceptionShutdownClass::DetachedAfterTimeout {
                configured_timeout,
                active_join_budget,
                terminal_fault: Some(NanoAccessoryTerminalFault::ReadinessObserverDropped),
            } if configured_timeout == Duration::from_secs(2)
                && active_join_budget == Duration::from_millis(125)
        ));
        assert!(!class.is_healthy());
        assert!(!class.is_coordinated());
        assert!(class.is_uncertain_or_unexpected());
        assert!(format!("{class}").contains("DetachedAfterTimeout"));
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn perception_handoff_has_lane_local_queue_accounting_and_shared_success_receipts() {
        let input = Arc::new(LatestFrameChannel::new());
        let output = Arc::new(LatestFrameChannel::with_shared_receipts(Arc::clone(
            &input.counters.receipts,
        )));
        assert!(!Arc::ptr_eq(&input.counters, &output.counters));
        assert!(Arc::ptr_eq(
            &input.counters.receipts,
            &output.counters.receipts
        ));
        assert_eq!(
            input.submit(10_u64),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        assert_eq!(
            output.submit_unmetered(20_u64),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        assert_eq!(
            output.submit_unmetered(30_u64),
            NanoAccessoryFrameSubmitOutcome::ReplacedOlderFrame
        );
        assert_eq!(input.counters.snapshot().enqueued, 1);
        assert_eq!(input.counters.snapshot().replaced_older, 0);
        assert!(matches!(
            output.next_event_blocking(),
            LatestFrameEvent::Frame(30)
        ));
        output
            .counters
            .record_processed_successfully()
            .expect("shared health counter");
        assert_eq!(input.counters.snapshot().processed_successfully, 1);
        assert_eq!(output.counters.snapshot().processed_successfully, 1);

        output.latch_terminal_fault();
        assert_eq!(
            input.submit(40),
            NanoAccessoryFrameSubmitOutcome::ReplacedOlderFrame,
            "output-local admission cannot silently latch public raw ingress"
        );
        input.latch_terminal_fault();
        assert_eq!(
            input.submit(50),
            NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched
        );
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn queued_face_output_discard_on_core_fault_cannot_contaminate_ingress_stats() {
        let input = LatestFrameChannel::<u64>::new();
        let output = LatestFrameChannel::with_shared_receipts(Arc::clone(&input.counters.receipts));
        assert_eq!(
            output.submit_unmetered(20_u64),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );

        // This is the exact output-channel operation performed when the
        // accessory core latches a terminal fault while a face result waits.
        output.latch_terminal_fault();

        assert_eq!(input.counters.snapshot().frames_discarded_for_terminal, 0);
        assert_eq!(output.counters.snapshot().frames_discarded_for_terminal, 1);
    }

    #[cfg(feature = "nano-agent")]
    #[tokio::test(flavor = "current_thread")]
    async fn face_output_poison_cannot_contaminate_ingress_poison_stats() {
        let input = LatestFrameChannel::<u64>::new();
        let output = Arc::new(LatestFrameChannel::<u64>::with_shared_receipts(Arc::clone(
            &input.counters.receipts,
        )));
        let poisoned = Arc::clone(&output);
        assert!(
            thread::spawn(move || {
                let _guard = poisoned.slot.lock().expect("initially healthy output slot");
                panic!("poison face-output slot");
            })
            .join()
            .is_err()
        );
        assert!(matches!(
            output.next_event().await,
            LatestFrameEvent::ChannelPoisoned
        ));
        assert_eq!(input.counters.snapshot().channel_poisoned, 0);
        assert_eq!(output.counters.snapshot().channel_poisoned, 1);
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn shared_admission_close_during_handoff_is_classified_as_clean_shutdown() {
        let input = LatestFrameChannel::<u64>::new();
        let output = LatestFrameChannel::with_shared_receipts(Arc::clone(&input.counters.receipts));

        // This is the deterministic narrow window between requesting input
        // shutdown and requesting output shutdown in the public coordinator.
        // Independent lane admission permits this handoff, but the subsequent
        // output shutdown outranks ordinary queued data.
        input.request_shutdown();
        let outcome = output.submit_unmetered(1_u64);
        assert_eq!(outcome, NanoAccessoryFrameSubmitOutcome::Enqueued);

        output.request_shutdown();
        assert!(matches!(
            output.next_event_blocking(),
            LatestFrameEvent::ShutdownRequested
        ));
        let outcome = output.submit_unmetered(2_u64);
        assert_eq!(
            outcome,
            NanoAccessoryFrameSubmitOutcome::IngressDisconnected
        );
        assert!(face_handoff_is_coordinated_shutdown(
            &input, &output, outcome
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn face_join_deadline_reports_detachment_without_claiming_cancellation() {
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let (finished_tx, finished_rx) = std::sync::mpsc::sync_channel(1);
        let handle = thread::spawn(move || {
            release_rx.recv().expect("test releases detached thread");
            finished_tx.send(()).expect("publish detached completion");
            NanoFacePerceptionThreadExit::Shutdown
        });
        let evidence =
            join_face_perception_thread_bounded(handle, Duration::ZERO, Duration::from_secs(2));
        assert!(matches!(
            evidence,
            NanoFacePerceptionJoinEvidence::DetachedAfterTimeout {
                configured_timeout,
                active_join_budget,
            } if configured_timeout == Duration::from_secs(2)
                && active_join_budget == Duration::ZERO
        ));
        release_tx.send(()).expect("release detached test thread");
        finished_rx.recv().expect("detached test thread returned");
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn face_join_returns_exact_finished_thread_exit() {
        let (finished_tx, finished_rx) = std::sync::mpsc::sync_channel(1);
        let handle = thread::spawn(move || {
            finished_tx.send(()).expect("publish completion");
            NanoFacePerceptionThreadExit::Shutdown
        });
        finished_rx.recv().expect("thread reached return");
        let evidence = join_face_perception_thread_bounded(
            handle,
            Duration::from_secs(1),
            Duration::from_secs(2),
        );
        assert!(matches!(
            evidence,
            NanoFacePerceptionJoinEvidence::Joined(NanoFacePerceptionThreadExit::Shutdown)
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn outer_face_lifecycle_guard_closes_both_lanes_after_cleanup_panics() {
        let input: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let output: Arc<LatestFrameChannel<NanoFacePerceptionWork>> = Arc::new(
            LatestFrameChannel::with_shared_receipts(Arc::clone(&input.counters.receipts)),
        );
        let guard =
            NanoFacePerceptionThreadLifecycleGuard::new(Arc::clone(&input), Arc::clone(&output));
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _never_returns = guard.finish_with_post_latch_hook(
                    NanoFacePerceptionThreadExit::Shutdown,
                    || {
                        panic!("post-body cleanup panic");
                    },
                );
            }))
            .is_err()
        );
        assert!(!input.accepting_frames.load(Ordering::Acquire));
        assert!(!output.ingress_alive.load(Ordering::Acquire));
        assert_eq!(
            input.submit(Err(ClockError::ElapsedNanosecondsOutOfRange {
                elapsed_nanoseconds: u128::MAX,
            })),
            NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched
        );
        assert!(matches!(
            output.next_event_blocking(),
            LatestFrameEvent::IngressDisconnected
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[tokio::test(flavor = "current_thread")]
    async fn queued_perception_result_is_drained_before_producer_disconnect() {
        let output = LatestFrameChannel::new();
        assert_eq!(
            output.submit_unmetered(7_u64),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        output.disconnect_ingress();
        assert!(matches!(
            output.next_event().await,
            LatestFrameEvent::Frame(7)
        ));
        assert!(matches!(
            output.next_event().await,
            LatestFrameEvent::IngressDisconnected
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[tokio::test(flavor = "current_thread")]
    async fn first_face_fault_is_published_before_raw_admission_latches() {
        let input: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let output: Arc<LatestFrameChannel<NanoFacePerceptionWork>> = Arc::new(
            LatestFrameChannel::with_shared_receipts(Arc::clone(&input.counters.receipts)),
        );
        let exit = finish_face_perception_fault(
            &input,
            &output,
            NanoFacePerceptionRuntimeError::RgbIngressDisconnected,
        );
        assert!(matches!(
            exit,
            NanoFacePerceptionThreadExit::RuntimeFault {
                source: NanoFacePerceptionRuntimeError::RgbIngressDisconnected,
                published_to_accessory: true,
            }
        ));
        assert!(
            !input.accepting_frames.load(Ordering::Acquire),
            "raw admission must close before the face owner returns"
        );

        // The thread wrapper disconnects its producer immediately after this
        // return. The queued first fault must still win over that disconnect.
        output.disconnect_ingress();
        let event = output.next_event().await;
        assert!(matches!(
            event,
            LatestFrameEvent::Frame(Err(NanoFacePerceptionRuntimeError::RgbIngressDisconnected))
        ));
        // The accessory core's later latch is intentionally idempotent.
        input.latch_terminal_fault();
        assert!(!input.accepting_frames.load(Ordering::Acquire));
        assert!(matches!(
            output.next_event().await,
            LatestFrameEvent::IngressDisconnected
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn pending_face_fault_publication_cannot_leave_runtime_health_ready() {
        let input: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let output: Arc<LatestFrameChannel<NanoFacePerceptionWork>> = Arc::new(
            LatestFrameChannel::with_shared_receipts(Arc::clone(&input.counters.receipts)),
        );
        let lifecycle = Arc::new(NanoAccessoryOwnerLifecycle::starting());
        lifecycle.mark_running();
        let observer = NanoAccessoryHealthObserver {
            channel: Arc::clone(&input),
            head: Arc::new(Mutex::new(NanoAccessoryHeadHealthState::empty())),
            lifecycle: Arc::clone(&lifecycle),
            health_period: health_period(50),
            rgb_frame_freshness: Duration::from_millis(50),
        };

        assert!(publish_face_perception_fault(
            &input,
            &output,
            NanoFacePerceptionRuntimeError::RgbIngressDisconnected,
        ));
        assert_eq!(
            lifecycle.state(),
            NanoAccessoryOwnerState::Running,
            "the actor has intentionally not consumed the published fault"
        );
        assert!(!input.accepting_frames.load(Ordering::Acquire));
        assert_eq!(
            observer
                .snapshot()
                .expect("admission latch is typed health"),
            NanoAccessoryRuntimeHealth {
                head: NanoAccessoryComponentHealth::Faulted,
                eyes: NanoAccessoryComponentHealth::Faulted,
                rgb_expression: NanoAccessoryComponentHealth::Faulted,
                successful_rgb_expression_frames: 0,
            }
        );
        assert!(matches!(
            output.next_event_blocking(),
            LatestFrameEvent::Frame(Err(NanoFacePerceptionRuntimeError::RgbIngressDisconnected))
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn face_fault_publication_has_no_ownerless_raw_admission_window() {
        let input: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let output: Arc<LatestFrameChannel<NanoFacePerceptionWork>> = Arc::new(
            LatestFrameChannel::with_shared_receipts(Arc::clone(&input.counters.receipts)),
        );
        let (raw_locked_tx, raw_locked_rx) = std::sync::mpsc::sync_channel(1);
        let (publish_tx, publish_rx) = std::sync::mpsc::sync_channel(1);
        let publisher_input = Arc::clone(&input);
        let publisher_output = Arc::clone(&output);
        let publisher = thread::spawn(move || {
            publish_face_perception_fault_with_raw_lock_hook(
                &publisher_input,
                &publisher_output,
                NanoFacePerceptionRuntimeError::RgbIngressDisconnected,
                || {
                    raw_locked_tx.send(()).expect("publish raw-slot ownership");
                    publish_rx.recv().expect("release exact fault publication");
                },
            )
        });

        raw_locked_rx
            .recv()
            .expect("publisher owns raw admission mutex");
        let (producer_prechecked_tx, producer_prechecked_rx) = std::sync::mpsc::sync_channel(1);
        let producer_input = Arc::clone(&input);
        let producer = thread::spawn(move || {
            producer_input.submit_inner_with_pre_lock_hook(
                Err(ClockError::ElapsedNanosecondsOutOfRange {
                    elapsed_nanoseconds: u128::MAX,
                }),
                true,
                LatestFrameSubmissionKind::ReplaceLatest,
                || {
                    producer_prechecked_tx
                        .send(())
                        .expect("publish producer precheck");
                },
            )
        });
        producer_prechecked_rx
            .recv()
            .expect("producer reached the raw mutex");
        publish_tx.send(()).expect("publish terminal fault");
        assert!(publisher.join().expect("publisher returned"));
        assert_eq!(
            producer.join().expect("producer returned"),
            NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched
        );
        assert!(matches!(
            output.next_event_blocking(),
            LatestFrameEvent::Frame(Err(NanoFacePerceptionRuntimeError::RgbIngressDisconnected))
        ));
        assert!(
            input
                .slot
                .lock()
                .expect("healthy raw slot")
                .latest
                .is_none()
        );
    }

    #[cfg(feature = "nano-agent")]
    #[tokio::test(flavor = "current_thread")]
    async fn committed_terminal_precedes_later_async_shutdown() {
        let channel = LatestFrameChannel::new();
        assert_eq!(
            channel.submit_first_terminal(7_u64, false),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        channel.request_shutdown();
        assert!(matches!(
            channel.next_event().await,
            LatestFrameEvent::Frame(7)
        ));
        assert!(matches!(
            channel.next_event().await,
            LatestFrameEvent::ShutdownRequested
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[tokio::test(flavor = "current_thread")]
    async fn committed_terminal_precedes_later_async_channel_poison() {
        let channel = Arc::new(LatestFrameChannel::new());
        assert_eq!(
            channel.submit_first_terminal(7_u64, false),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        let poisoned = Arc::clone(&channel);
        assert!(
            thread::spawn(move || {
                let _guard = poisoned.slot.lock().expect("initially healthy slot");
                panic!("poison slot after terminal commit");
            })
            .join()
            .is_err()
        );
        assert!(matches!(
            channel.next_event().await,
            LatestFrameEvent::Frame(7)
        ));
        assert!(matches!(
            channel.next_event().await,
            LatestFrameEvent::ChannelPoisoned
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn committed_terminal_precedes_later_blocking_shutdown() {
        let channel = LatestFrameChannel::new();
        assert_eq!(
            channel.submit_first_terminal(7_u64, false),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        channel.request_shutdown();
        assert!(matches!(
            channel.next_event_blocking(),
            LatestFrameEvent::Frame(7)
        ));
        assert!(matches!(
            channel.next_event_blocking(),
            LatestFrameEvent::ShutdownRequested
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn committed_terminal_precedes_later_blocking_channel_poison() {
        let channel = Arc::new(LatestFrameChannel::new());
        assert_eq!(
            channel.submit_first_terminal(7_u64, false),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        let poisoned = Arc::clone(&channel);
        assert!(
            thread::spawn(move || {
                let _guard = poisoned.slot.lock().expect("initially healthy slot");
                panic!("poison slot after terminal commit");
            })
            .join()
            .is_err()
        );
        assert!(matches!(
            channel.next_event_blocking(),
            LatestFrameEvent::Frame(7)
        ));
        assert!(matches!(
            channel.next_event_blocking(),
            LatestFrameEvent::ChannelPoisoned
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[tokio::test(flavor = "current_thread")]
    async fn raw_poison_is_published_as_the_exact_face_fault() {
        let input: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let output: Arc<LatestFrameChannel<NanoFacePerceptionWork>> = Arc::new(
            LatestFrameChannel::with_shared_receipts(Arc::clone(&input.counters.receipts)),
        );
        let poisoned = Arc::clone(&input);
        assert!(
            thread::spawn(move || {
                let _guard = poisoned.slot.lock().expect("initially healthy raw slot");
                panic!("poison raw slot");
            })
            .join()
            .is_err()
        );
        assert!(matches!(
            input.next_event_blocking(),
            LatestFrameEvent::ChannelPoisoned
        ));
        let exit = finish_face_perception_fault(
            &input,
            &output,
            NanoFacePerceptionRuntimeError::RgbChannelPoisoned,
        );
        assert!(matches!(
            exit,
            NanoFacePerceptionThreadExit::RuntimeFault {
                source: NanoFacePerceptionRuntimeError::RgbChannelPoisoned,
                published_to_accessory: true,
            }
        ));
        output.disconnect_ingress();
        assert!(matches!(
            output.next_event().await,
            LatestFrameEvent::Frame(Err(NanoFacePerceptionRuntimeError::RgbChannelPoisoned))
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn blocking_shutdown_transition_uses_the_wait_predicate_mutex() {
        let channel = Arc::new(LatestFrameChannel::<u64>::new());
        let slot = channel.slot.lock().expect("healthy test slot");
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (completed_tx, completed_rx) = std::sync::mpsc::sync_channel(1);
        let transition = Arc::clone(&channel);
        let task = thread::spawn(move || {
            started_tx.send(()).expect("publish transition start");
            transition.request_shutdown();
            completed_tx
                .send(())
                .expect("publish transition completion");
        });
        started_rx.recv().expect("transition thread started");
        assert_eq!(
            completed_rx.recv_timeout(Duration::from_millis(20)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout),
            "shutdown mutation must wait for the predicate mutex"
        );
        drop(slot);
        completed_rx
            .recv_timeout(Duration::from_millis(100))
            .expect("shutdown transition completes after mutex release");
        task.join().expect("shutdown transition thread");
        assert!(matches!(
            channel.next_event_blocking(),
            LatestFrameEvent::ShutdownRequested
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn blocking_disconnect_transition_uses_the_wait_predicate_mutex() {
        let channel = Arc::new(LatestFrameChannel::<u64>::new());
        let slot = channel.slot.lock().expect("healthy test slot");
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (completed_tx, completed_rx) = std::sync::mpsc::sync_channel(1);
        let transition = Arc::clone(&channel);
        let task = thread::spawn(move || {
            started_tx.send(()).expect("publish transition start");
            transition.disconnect_ingress();
            completed_tx
                .send(())
                .expect("publish transition completion");
        });
        started_rx.recv().expect("transition thread started");
        assert_eq!(
            completed_rx.recv_timeout(Duration::from_millis(20)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout),
            "disconnect mutation must wait for the predicate mutex"
        );
        drop(slot);
        completed_rx
            .recv_timeout(Duration::from_millis(100))
            .expect("disconnect transition completes after mutex release");
        task.join().expect("disconnect transition thread");
        assert!(matches!(
            channel.next_event_blocking(),
            LatestFrameEvent::IngressDisconnected
        ));
    }

    #[cfg(feature = "nano-agent")]
    #[test]
    fn submit_rechecks_disconnect_after_winning_the_slot_mutex() {
        let channel = Arc::new(LatestFrameChannel::<u64>::new());
        let (prechecked_tx, prechecked_rx) = std::sync::mpsc::sync_channel(1);
        let (continue_tx, continue_rx) = std::sync::mpsc::sync_channel(1);
        let producer = Arc::clone(&channel);
        let task = thread::spawn(move || {
            producer.submit_inner_with_pre_lock_hook(
                7,
                true,
                LatestFrameSubmissionKind::ReplaceLatest,
                || {
                    prechecked_tx
                        .send(())
                        .expect("publish completed lock-free precheck");
                    continue_rx.recv().expect("release producer toward mutex");
                },
            )
        });

        prechecked_rx.recv().expect("producer reached pre-lock gap");
        channel.disconnect_ingress();
        continue_tx.send(()).expect("release producer");
        assert_eq!(
            task.join().expect("producer returned"),
            NanoAccessoryFrameSubmitOutcome::IngressDisconnected
        );
        assert!(matches!(
            channel.next_event_blocking(),
            LatestFrameEvent::IngressDisconnected
        ));
        assert_eq!(
            channel.counters.snapshot(),
            NanoAccessoryFrameStats {
                rejected_disconnected: 1,
                ..NanoAccessoryFrameStats::default()
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_one_queue_replaces_without_cloning_the_frame() {
        let channel = Arc::new(LatestFrameChannel::new());
        assert_eq!(
            channel.submit(1_u64),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        assert_eq!(
            channel.submit(2_u64),
            NanoAccessoryFrameSubmitOutcome::ReplacedOlderFrame
        );
        assert_eq!(
            channel.submit(3_u64),
            NanoAccessoryFrameSubmitOutcome::ReplacedOlderFrame
        );
        assert!(matches!(
            channel.next_event().await,
            LatestFrameEvent::Frame(3)
        ));
        assert_eq!(
            channel.counters.snapshot(),
            NanoAccessoryFrameStats {
                enqueued: 3,
                replaced_older: 2,
                ..NanoAccessoryFrameStats::default()
            }
        );
    }

    struct FailingIngressClock;

    impl MonotonicClock for FailingIngressClock {
        fn now(&self) -> Result<kiko_expression_core::MonotonicTimestamp, ClockError> {
            Err(ClockError::ElapsedNanosecondsOutOfRange {
                elapsed_nanoseconds: u128::MAX,
            })
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ingress_clock_failure_is_nonreplaceable_and_has_distinct_accounting() {
        let failure = match observe_rgb_at_ingress(&FailingIngressClock, 1_u64, |_| {}) {
            Ok(_) => panic!("a failed ingress clock cannot construct observed frame evidence"),
            Err(source) => source,
        };
        assert_eq!(
            failure,
            ClockError::ElapsedNanosecondsOutOfRange {
                elapsed_nanoseconds: u128::MAX,
            }
        );

        let channel = Arc::new(LatestFrameChannel::new());
        assert_eq!(
            channel.submit(Ok(IngressObservedRgbFrame::new(
                1_u64,
                kiko_expression_core::MonotonicTimestamp::from_nanos_since_epoch(16),
            ))),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        assert_eq!(
            channel.submit_first_terminal(Err(failure), true),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        assert_eq!(
            channel.submit(Ok(IngressObservedRgbFrame::new(
                2_u64,
                kiko_expression_core::MonotonicTimestamp::from_nanos_since_epoch(17),
            ))),
            NanoAccessoryFrameSubmitOutcome::TerminalFaultPendingPublication
        );
        let LatestFrameEvent::Frame(Err(source)) = channel.next_event().await else {
            panic!("the first typed clock failure must survive later data");
        };
        assert_eq!(
            source,
            ClockError::ElapsedNanosecondsOutOfRange {
                elapsed_nanoseconds: u128::MAX,
            }
        );
        assert_eq!(
            channel.counters.snapshot(),
            NanoAccessoryFrameStats {
                enqueued: 1,
                first_terminal_enqueued: 1,
                frames_discarded_for_terminal: 1,
                rejected_behind_pending_terminal: 1,
                ..NanoAccessoryFrameStats::default()
            }
        );
    }

    struct MutableIngressClock(AtomicU64);

    impl MonotonicClock for MutableIngressClock {
        fn now(&self) -> Result<kiko_expression_core::MonotonicTimestamp, ClockError> {
            Ok(
                kiko_expression_core::MonotonicTimestamp::from_nanos_since_epoch(
                    self.0.load(Ordering::Relaxed),
                ),
            )
        }
    }

    #[test]
    fn ingress_timestamp_precedes_non_authoritative_diagnostic_work() {
        let clock = MutableIngressClock(AtomicU64::new(10));
        let observed = observe_rgb_at_ingress(&clock, 7_u64, |frame| {
            assert_eq!(*frame, 7);
            clock.0.store(90, Ordering::Relaxed);
        })
        .expect("valid ingress observation");

        assert_eq!(observed.observed_at().nanos_since_epoch(), 10);
        assert_eq!(clock.0.load(Ordering::Relaxed), 90);
    }

    #[cfg(feature = "nano-agent")]
    fn fake_tap_evidence() -> NanoPetEpisodeEvidence {
        let line = concat!(
            r#"{"schema_version":1,"wall":1787759000.25,"wall_unix_ms":1787759000250,"completed_monotonic_ns":10600000000,"episode":{"started_at":10.0,"started_monotonic_ns":10000000000,"completed_monotonic_ns":10600000000,"yield_entries":1,"samples":1,"peak_residual":[1,0,0,0],"delta_accum":0,"delta_samples":0,"reached_rest":false,"reached_comfy":false,"tap":true,"duration_s":0.6,"duration_ns":600000000,"mean_delta":0.0}}"#,
            "\n"
        );
        NanoPetEpisodeEvidence::parse_ndjson_line(line.as_bytes()).expect("fake tap evidence")
    }

    #[derive(Clone)]
    struct FakeHead {
        log: Arc<Mutex<Vec<&'static str>>>,
        health_calls: Arc<AtomicUsize>,
        fail_start: bool,
        fail_return: bool,
        fail_gaze: bool,
        emit_pet_episode: bool,
        emit_thermal_derate: bool,
        tracking_eye_gaze: Option<TrackingEyeGaze>,
        fail_health_at: Option<usize>,
    }

    impl<F> ReviewedNaturalHeadPort<F> for FakeHead {
        type Ready = &'static str;
        type StartError = &'static str;
        type Health = usize;
        type HealthError = &'static str;
        type GazeError = &'static str;
        type Shutdown = &'static str;

        async fn start(&mut self) -> Result<Self::Ready, Self::StartError> {
            self.log.lock().unwrap().push("head_start");
            if self.fail_start {
                Err("head_start")
            } else {
                self.log.lock().unwrap().push("head_return");
                if self.fail_return {
                    Err("head_return")
                } else {
                    Ok("head_ready")
                }
            }
        }

        async fn check_health(&mut self) -> Result<Self::Health, Self::HealthError> {
            self.log.lock().unwrap().push("head_health");
            let call = self.health_calls.fetch_add(1, Ordering::SeqCst) + 1;
            if self.fail_health_at == Some(call) {
                Err("head_health")
            } else {
                Ok(call)
            }
        }

        async fn process_gaze_frame(
            &mut self,
            _frame: &F,
            _character_head: CharacterHeadOverlay,
        ) -> Result<Option<AccessoryHeadFeedback>, Self::GazeError> {
            if self.fail_gaze {
                self.log.lock().unwrap().push("head_gaze");
                Err("head_gaze")
            } else if let Some(tracking_eye_gaze) = self.tracking_eye_gaze {
                Ok(Some(AccessoryHeadFeedback {
                    state: None,
                    thermal_state: None,
                    completed_episode: None,
                    tracking_eye_gaze: Some(tracking_eye_gaze),
                }))
            } else if self.emit_pet_episode {
                Ok(Some(AccessoryHeadFeedback {
                    state: Some(CharacterPetState::Ready),
                    thermal_state: None,
                    completed_episode: Some(AccessoryPetEpisode {
                        completed_at: MonotonicTimestamp::from_nanos_since_epoch(10),
                        episode: CharacterPetEpisode::try_new(
                            Duration::from_millis(600),
                            0,
                            0,
                            false,
                            true,
                        )
                        .expect("valid fake tap"),
                        #[cfg(feature = "nano-agent")]
                        controller_evidence: AccessoryPetControllerEvidence::GenericCoreFixture(
                            fake_tap_evidence(),
                        ),
                    }),
                    tracking_eye_gaze: None,
                }))
            } else if self.emit_thermal_derate {
                Ok(Some(AccessoryHeadFeedback {
                    state: None,
                    thermal_state: Some(CharacterThermalState::PitchDerated),
                    completed_episode: None,
                    tracking_eye_gaze: None,
                }))
            } else {
                Ok(None)
            }
        }

        async fn shutdown(&mut self) -> Self::Shutdown {
            self.log.lock().unwrap().push("head_shutdown");
            "head_shutdown"
        }
    }

    struct FakeEye {
        log: Arc<Mutex<Vec<&'static str>>>,
        fail_start: bool,
        fail_apply: bool,
        require_terminal_latch_for_fault_reflex: Option<Arc<LatestFrameChannel<u64>>>,
    }

    impl Kep2EyePort<u64> for FakeEye {
        type Ready = &'static str;
        type StartError = &'static str;
        type ApplyError = &'static str;
        type Shutdown = &'static str;

        async fn start(&mut self) -> Result<Self::Ready, Self::StartError> {
            self.log.lock().unwrap().push("eye_start");
            if self.fail_start {
                Err("eye_start")
            } else {
                Ok("eye_ready")
            }
        }

        async fn apply(&mut self, intent: u64) -> Result<(), Self::ApplyError> {
            if intent == 92
                && let Some(channel) = self.require_terminal_latch_for_fault_reflex.as_ref()
            {
                assert_eq!(
                    channel.submit(u64::MAX),
                    NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched,
                    "the RGB stop latch must precede the first fault-reflex eye write"
                );
            }
            self.log.lock().unwrap().push(if intent == 1_777 {
                "eye_apply_coordinated"
            } else {
                "eye_apply"
            });
            if self.fail_apply {
                Err("eye_apply")
            } else {
                Ok(())
            }
        }

        async fn shutdown(&mut self) -> Self::Shutdown {
            self.log.lock().unwrap().push("eye_shutdown");
            "eye_shutdown"
        }
    }

    struct FakeBridge {
        log: Arc<Mutex<Vec<&'static str>>>,
        fail: bool,
    }

    impl RgbBridgePort<u64> for FakeBridge {
        type Intent = u64;
        type Error = &'static str;

        fn process(&mut self, frame: &u64) -> Result<Self::Intent, Self::Error> {
            self.log.lock().unwrap().push("bridge");
            if self.fail { Err("bridge") } else { Ok(*frame) }
        }

        fn note_pet_state(&mut self, _state: CharacterPetState) {
            self.log.lock().unwrap().push("pet_state");
        }

        fn note_thermal_state(&mut self, _state: CharacterThermalState) {
            self.log.lock().unwrap().push("thermal_state");
        }

        fn prepare_lifecycle_reflex(
            &mut self,
            _state: CharacterLifecycleState,
        ) -> Result<Self::Intent, Self::Error> {
            Ok(0)
        }

        fn note_pet_episode(&mut self, _episode: AccessoryPetEpisode) {
            self.log.lock().unwrap().push("pet_episode");
        }
    }

    struct LifecycleBridge {
        states: Arc<Mutex<Vec<CharacterLifecycleState>>>,
        fail_process: bool,
        fail_prepare_for: Option<CharacterLifecycleState>,
    }

    impl RgbBridgePort<u64> for LifecycleBridge {
        type Intent = u64;
        type Error = &'static str;

        fn process(&mut self, frame: &u64) -> Result<Self::Intent, Self::Error> {
            if self.fail_process {
                Err("lifecycle_process")
            } else {
                Ok(*frame)
            }
        }

        fn note_lifecycle_state(&mut self, state: CharacterLifecycleState) {
            self.states.lock().unwrap().push(state);
        }

        fn prepare_lifecycle_reflex(
            &mut self,
            state: CharacterLifecycleState,
        ) -> Result<Self::Intent, Self::Error> {
            if self.fail_prepare_for == Some(state) {
                Err("lifecycle")
            } else {
                Ok(match state {
                    CharacterLifecycleState::AdmissionRecovery => 91,
                    CharacterLifecycleState::FaultRecovery => 92,
                    CharacterLifecycleState::Parking => 93,
                    _ => 99,
                })
            }
        }

        fn lifecycle_reflex_duration(&self, state: CharacterLifecycleState) -> Duration {
            match state {
                CharacterLifecycleState::AdmissionRecovery | CharacterLifecycleState::Parking => {
                    Duration::from_millis(2)
                }
                CharacterLifecycleState::FaultRecovery => Duration::from_nanos(1),
                _ => Duration::ZERO,
            }
        }
    }

    struct ClockWorkBridge {
        log: Arc<Mutex<Vec<&'static str>>>,
    }

    impl RgbBridgePort<u64> for ClockWorkBridge {
        type Intent = u64;
        type Error = RgbExpressionBridgeError;

        fn process(&mut self, frame: &u64) -> Result<Self::Intent, Self::Error> {
            self.log.lock().unwrap().push("clock_bridge_frame");
            Ok(*frame)
        }

        fn prepare_lifecycle_reflex(
            &mut self,
            _state: CharacterLifecycleState,
        ) -> Result<Self::Intent, Self::Error> {
            Ok(0)
        }
    }

    impl RgbBridgePort<Result<u64, ClockError>> for ClockWorkBridge {
        type Intent = u64;
        type Error = RgbExpressionBridgeError;

        fn process(&mut self, work: &Result<u64, ClockError>) -> Result<Self::Intent, Self::Error> {
            process_rgb_work(self, work)
        }

        fn prepare_lifecycle_reflex(
            &mut self,
            _state: CharacterLifecycleState,
        ) -> Result<Self::Intent, Self::Error> {
            Ok(0)
        }
    }

    type FakePorts = (
        FakeHead,
        FakeEye,
        FakeBridge,
        Arc<Mutex<Vec<&'static str>>>,
        Arc<AtomicUsize>,
    );

    fn fakes(
        fail_eye_start: bool,
        fail_head_start: bool,
        fail_bridge: bool,
        fail_eye_apply: bool,
        fail_health_at: Option<usize>,
    ) -> FakePorts {
        let log = Arc::new(Mutex::new(Vec::new()));
        let health_calls = Arc::new(AtomicUsize::new(0));
        (
            FakeHead {
                log: Arc::clone(&log),
                health_calls: Arc::clone(&health_calls),
                fail_start: fail_head_start,
                fail_return: false,
                fail_gaze: false,
                emit_pet_episode: false,
                emit_thermal_derate: false,
                tracking_eye_gaze: None,
                fail_health_at,
            },
            FakeEye {
                log: Arc::clone(&log),
                fail_start: fail_eye_start,
                fail_apply: fail_eye_apply,
                require_terminal_latch_for_fault_reflex: None,
            },
            FakeBridge {
                log: Arc::clone(&log),
                fail: fail_bridge,
            },
            log,
            health_calls,
        )
    }

    #[tokio::test(flavor = "current_thread")]
    async fn readiness_follows_both_startup_receipts_and_shutdown_is_ordered() {
        let (head, eye, bridge, log, _) = fakes(false, false, false, false, None);
        let channel = Arc::new(LatestFrameChannel::new());
        let ready_seen = Arc::new(AtomicBool::new(false));
        let ready_flag = Arc::clone(&ready_seen);
        let task_channel = Arc::clone(&channel);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(50),
                None,
                CoreObservers {
                    ready: move |head, eye| {
                        assert_eq!(head, "head_ready");
                        assert_eq!(eye, "eye_ready");
                        ready_flag.store(true, Ordering::SeqCst);
                        true
                    },
                    record_health: |_| true,
                    publish_fault: |_| {},
                    latch_fault: || {},
                },
            )
            .await
        });
        tokio::task::yield_now().await;
        assert!(ready_seen.load(Ordering::SeqCst));
        assert_eq!(
            &*log.lock().unwrap(),
            &["eye_start", "head_start", "head_return"]
        );
        channel.request_shutdown();
        assert!(matches!(task.await.unwrap(), CoreExit::Shutdown { .. }));
        assert_eq!(
            &*log.lock().unwrap(),
            &[
                "eye_start",
                "head_start",
                "head_return",
                "eye_shutdown",
                "head_shutdown"
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lifecycle_facts_reach_operational_then_stream_goodnight_before_eye_release() {
        let (head, eye, _, log, _) = fakes(false, false, false, false, None);
        let states = Arc::new(Mutex::new(Vec::new()));
        let bridge = LifecycleBridge {
            states: Arc::clone(&states),
            fail_process: false,
            fail_prepare_for: None,
        };
        let channel = Arc::new(LatestFrameChannel::new());
        let task_channel = Arc::clone(&channel);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(1),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: |_| {},
                    latch_fault: || {},
                },
            )
            .await
        });

        tokio::time::timeout(Duration::from_millis(100), async {
            loop {
                if states
                    .lock()
                    .unwrap()
                    .contains(&CharacterLifecycleState::Operational)
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("completed admission recovery publishes operational");
        channel.request_shutdown();
        let exit = task.await.unwrap();
        assert!(matches!(
            exit,
            CoreExit::Shutdown {
                terminal_fault: None,
                ..
            }
        ));
        assert_eq!(
            &*states.lock().unwrap(),
            &[
                CharacterLifecycleState::Booting,
                CharacterLifecycleState::AdmissionRecovery,
                CharacterLifecycleState::Operational,
                CharacterLifecycleState::Parking,
            ]
        );
        let log = log.lock().unwrap();
        let apply = log
            .iter()
            .position(|entry| *entry == "eye_apply")
            .expect("goodnight applies at least one eye frame");
        let release = log
            .iter()
            .position(|entry| *entry == "eye_shutdown")
            .expect("eye session releases");
        assert!(apply < release);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn goodnight_generation_failure_is_retained_as_a_terminal_fault() {
        let (head, eye, _, _, _) = fakes(false, false, false, false, None);
        let states = Arc::new(Mutex::new(Vec::new()));
        let bridge = LifecycleBridge {
            states,
            fail_process: false,
            fail_prepare_for: Some(CharacterLifecycleState::Parking),
        };
        let channel = Arc::new(LatestFrameChannel::new());
        let task_channel = Arc::clone(&channel);
        let published = Arc::new(Mutex::new(Vec::new()));
        let published_for_task = Arc::clone(&published);
        let latched = Arc::new(AtomicBool::new(false));
        let latched_for_task = Arc::clone(&latched);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(50),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: move |fault| published_for_task.lock().unwrap().push(fault),
                    latch_fault: move || latched_for_task.store(true, Ordering::SeqCst),
                },
            )
            .await
        });
        tokio::task::yield_now().await;
        channel.request_shutdown();

        let exit = task.await.unwrap();
        assert!(matches!(
            exit,
            CoreExit::Shutdown {
                terminal_fault: Some(CoreTerminalFault::Bridge("lifecycle")),
                fault_recovery_presentation:
                    CoreFaultRecoveryPresentationEvidence::Presented { frames_applied },
                ..
            } if frames_applied == NonZeroU64::MIN
        ));
        assert_eq!(
            &*published.lock().unwrap(),
            &[CoreTerminalFault::Bridge("lifecycle")]
        );
        assert!(latched.load(Ordering::SeqCst));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn admission_recovery_failure_prevents_false_readiness_and_closes_both_actors() {
        let (head, eye, _, log, _) = fakes(false, false, false, false, None);
        let states = Arc::new(Mutex::new(Vec::new()));
        let bridge = LifecycleBridge {
            states: Arc::clone(&states),
            fail_process: false,
            fail_prepare_for: Some(CharacterLifecycleState::AdmissionRecovery),
        };
        let channel = Arc::new(LatestFrameChannel::new());
        let ready = Arc::new(AtomicBool::new(false));
        let ready_for_task = Arc::clone(&ready);
        let published = Arc::new(Mutex::new(Vec::new()));
        let published_for_task = Arc::clone(&published);
        let exit = run_accessory_core(
            head,
            eye,
            bridge,
            channel,
            health_period(50),
            None,
            CoreObservers {
                ready: move |_, _| {
                    ready_for_task.store(true, Ordering::SeqCst);
                    true
                },
                record_health: |_| true,
                publish_fault: move |fault| published_for_task.lock().unwrap().push(fault),
                latch_fault: || {},
            },
        )
        .await;

        assert!(!ready.load(Ordering::SeqCst));
        assert!(matches!(
            exit,
            CoreExit::Shutdown {
                terminal_fault: Some(CoreTerminalFault::Bridge("lifecycle")),
                fault_recovery_presentation:
                    CoreFaultRecoveryPresentationEvidence::Presented { frames_applied },
                ..
            } if frames_applied == NonZeroU64::MIN
        ));
        assert_eq!(
            &*states.lock().unwrap(),
            &[
                CharacterLifecycleState::Booting,
                CharacterLifecycleState::AdmissionRecovery,
                CharacterLifecycleState::FaultRecovery,
            ]
        );
        assert_eq!(
            &*published.lock().unwrap(),
            &[CoreTerminalFault::Bridge("lifecycle")]
        );
        assert_eq!(
            &*log.lock().unwrap(),
            &[
                "eye_start",
                "head_start",
                "head_return",
                "eye_apply",
                "eye_shutdown",
                "head_shutdown",
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn terminal_stop_latches_precede_the_first_fresh_fault_reflex_frame() {
        let (head, mut eye, _, log, _) = fakes(false, false, false, false, None);
        let states = Arc::new(Mutex::new(Vec::new()));
        let bridge = LifecycleBridge {
            states: Arc::clone(&states),
            fail_process: true,
            fail_prepare_for: None,
        };
        let channel = Arc::new(LatestFrameChannel::new());
        eye.require_terminal_latch_for_fault_reflex = Some(Arc::clone(&channel));
        let task_channel = Arc::clone(&channel);
        let published = Arc::new(Mutex::new(Vec::new()));
        let published_for_task = Arc::clone(&published);
        let task_log = Arc::clone(&log);
        let latch_log = Arc::clone(&log);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(50),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: move |fault| {
                        published_for_task.lock().unwrap().push(fault);
                        task_log.lock().unwrap().push("fault_published");
                    },
                    latch_fault: move || latch_log.lock().unwrap().push("base_latched"),
                },
            )
            .await
        });

        tokio::time::timeout(Duration::from_millis(100), async {
            while !states
                .lock()
                .unwrap()
                .contains(&CharacterLifecycleState::Operational)
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("accessory owner becomes operational");
        log.lock().unwrap().clear();
        assert_eq!(channel.submit(7), NanoAccessoryFrameSubmitOutcome::Enqueued);
        tokio::time::timeout(Duration::from_millis(100), async {
            while !log.lock().unwrap().contains(&"eye_apply") {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("fault reflex reaches the eye after both stop latches");
        channel.request_shutdown();

        let exit = task.await.unwrap();
        assert!(matches!(
            exit,
            CoreExit::Shutdown {
                terminal_fault: Some(CoreTerminalFault::Bridge("lifecycle_process")),
                fault_recovery_presentation:
                    CoreFaultRecoveryPresentationEvidence::Presented { frames_applied },
                ..
            } if frames_applied == NonZeroU64::MIN
        ));
        assert_eq!(
            &*published.lock().unwrap(),
            &[CoreTerminalFault::Bridge("lifecycle_process")]
        );
        assert_eq!(
            &*states.lock().unwrap(),
            &[
                CharacterLifecycleState::Booting,
                CharacterLifecycleState::AdmissionRecovery,
                CharacterLifecycleState::Operational,
                CharacterLifecycleState::FaultRecovery,
            ]
        );
        assert_eq!(
            &*log.lock().unwrap(),
            &[
                "fault_published",
                "base_latched",
                "eye_apply",
                "eye_shutdown",
                "head_shutdown",
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn fault_reflex_failure_is_secondary_and_shutdown_retains_both_causes() {
        let (head, eye, _, log, _) = fakes(false, false, false, false, None);
        let states = Arc::new(Mutex::new(Vec::new()));
        let bridge = LifecycleBridge {
            states: Arc::clone(&states),
            fail_process: true,
            fail_prepare_for: Some(CharacterLifecycleState::FaultRecovery),
        };
        let channel = Arc::new(LatestFrameChannel::new());
        let task_channel = Arc::clone(&channel);
        let published = Arc::new(Mutex::new(Vec::new()));
        let published_for_task = Arc::clone(&published);
        let latch_count = Arc::new(AtomicUsize::new(0));
        let latch_count_for_task = Arc::clone(&latch_count);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(50),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: move |fault| published_for_task.lock().unwrap().push(fault),
                    latch_fault: move || {
                        latch_count_for_task.fetch_add(1, Ordering::SeqCst);
                    },
                },
            )
            .await
        });

        tokio::time::timeout(Duration::from_millis(100), async {
            while !states
                .lock()
                .unwrap()
                .contains(&CharacterLifecycleState::Operational)
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("accessory owner becomes operational");
        log.lock().unwrap().clear();
        assert_eq!(channel.submit(8), NanoAccessoryFrameSubmitOutcome::Enqueued);
        tokio::time::timeout(Duration::from_millis(100), async {
            while channel.accepting_frames.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("primary fault latches the channel");
        channel.request_shutdown();

        let exit = task.await.unwrap();
        assert!(matches!(
            exit,
            CoreExit::Shutdown {
                terminal_fault: Some(CoreTerminalFault::Bridge("lifecycle_process")),
                fault_recovery_presentation: CoreFaultRecoveryPresentationEvidence::Failed(
                    LifecyclePresentationFault::Bridge {
                        frames_applied: 0,
                        source: "lifecycle",
                    }
                ),
                ..
            }
        ));
        assert_eq!(
            &*published.lock().unwrap(),
            &[CoreTerminalFault::Bridge("lifecycle_process")]
        );
        assert_eq!(latch_count.load(Ordering::SeqCst), 1);
        assert_eq!(
            &*states.lock().unwrap(),
            &[
                CharacterLifecycleState::Booting,
                CharacterLifecycleState::AdmissionRecovery,
                CharacterLifecycleState::Operational,
                CharacterLifecycleState::FaultRecovery,
            ]
        );
        assert_eq!(&*log.lock().unwrap(), &["eye_shutdown", "head_shutdown"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn successful_expression_counter_advances_only_after_eye_acknowledgement() {
        let (head, eye, bridge, _, _) = fakes(false, false, false, false, None);
        let channel = Arc::new(LatestFrameChannel::new());
        let task_channel = Arc::clone(&channel);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(50),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: |_| {},
                    latch_fault: || {},
                },
            )
            .await
        });
        assert_eq!(
            channel.submit(7_u64),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        tokio::time::timeout(Duration::from_millis(100), async {
            while channel.counters.snapshot().processed_successfully == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("successful eye acknowledgement within bounded test time");
        assert_eq!(channel.counters.snapshot().processed_successfully, 1);
        channel.request_shutdown();
        assert!(matches!(task.await.unwrap(), CoreExit::Shutdown { .. }));

        let (head, eye, bridge, _, _) = fakes(false, false, false, true, None);
        let failed_channel = Arc::new(LatestFrameChannel::new());
        let task_channel = Arc::clone(&failed_channel);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(50),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: |_| {},
                    latch_fault: || {},
                },
            )
            .await
        });
        assert_eq!(
            failed_channel.submit(8_u64),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        tokio::time::timeout(Duration::from_millis(100), async {
            while failed_channel.accepting_frames.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("failed eye acknowledgement latches terminal fault");
        assert_eq!(failed_channel.counters.snapshot().processed_successfully, 0);
        failed_channel.request_shutdown();
        assert!(matches!(task.await.unwrap(), CoreExit::Shutdown { .. }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn verified_head_feedback_replaces_tracking_gaze_before_eye_application() {
        let (mut head, eye, bridge, log, _) = fakes(false, false, false, false, None);
        head.tracking_eye_gaze = Some(TrackingEyeGaze::new(
            kiko_eye_protocol::SignedUnit::try_new(777).unwrap(),
            kiko_eye_protocol::SignedUnit::try_new(-250).unwrap(),
        ));
        let channel = Arc::new(LatestFrameChannel::new());
        let task_channel = Arc::clone(&channel);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(50),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: |_| {},
                    latch_fault: || {},
                },
            )
            .await
        });
        assert_eq!(
            channel.submit(7_u64),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        tokio::time::timeout(Duration::from_millis(100), async {
            while channel.counters.snapshot().processed_successfully == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("coordinated eye acknowledgement within bounded test time");
        channel.request_shutdown();
        assert!(matches!(task.await.unwrap(), CoreExit::Shutdown { .. }));
        let log = log.lock().unwrap();
        assert!(log.contains(&"eye_apply_coordinated"));
        assert!(!log.contains(&"eye_apply"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn completed_pet_episode_reaches_character_before_the_current_eye_ack() {
        let (mut head, eye, bridge, log, _) = fakes(false, false, false, false, None);
        head.emit_pet_episode = true;
        let channel = Arc::new(LatestFrameChannel::new());
        let task_channel = Arc::clone(&channel);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(50),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: |_| {},
                    latch_fault: || {},
                },
            )
            .await
        });
        assert_eq!(
            channel.submit(9_u64),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        tokio::time::timeout(Duration::from_millis(100), async {
            while channel.counters.snapshot().processed_successfully == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("pet feedback and eye acknowledgement finish in bounded time");
        assert_eq!(
            &*log.lock().unwrap(),
            &[
                "eye_start",
                "head_start",
                "head_return",
                "bridge",
                "pet_state",
                "pet_episode",
                "eye_apply",
            ]
        );
        channel.request_shutdown();
        assert!(matches!(task.await.unwrap(), CoreExit::Shutdown { .. }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn thermal_derate_fact_reaches_character_before_the_current_eye_ack() {
        let (mut head, eye, bridge, log, _) = fakes(false, false, false, false, None);
        head.emit_thermal_derate = true;
        let channel = Arc::new(LatestFrameChannel::new());
        let task_channel = Arc::clone(&channel);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(50),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: |_| {},
                    latch_fault: || {},
                },
            )
            .await
        });
        assert_eq!(
            channel.submit(10_u64),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        tokio::time::timeout(Duration::from_millis(100), async {
            while channel.counters.snapshot().processed_successfully == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("thermal feedback and eye acknowledgement finish in bounded time");
        assert_eq!(
            &*log.lock().unwrap(),
            &[
                "eye_start",
                "head_start",
                "head_return",
                "bridge",
                "thermal_state",
                "eye_apply",
            ]
        );
        channel.request_shutdown();
        assert!(matches!(task.await.unwrap(), CoreExit::Shutdown { .. }));
    }

    #[cfg(feature = "nano-agent")]
    #[tokio::test(flavor = "current_thread")]
    async fn completed_pet_episode_is_enqueued_before_character_feedback_and_durable_on_shutdown() {
        let root = PetJournalTestDirectory::create();
        let config = NanoPetEvidenceJournalConfig::for_state_root(&root.0).unwrap();
        let path = config.path();
        let mut journal = NanoPetEvidenceJournal::start(config).expect("journal startup");
        let (mut head, eye, bridge, log, _) = fakes(false, false, false, false, None);
        head.emit_pet_episode = true;
        let channel = Arc::new(LatestFrameChannel::new());
        let task_channel = Arc::clone(&channel);

        let core = run_accessory_core(
            head,
            eye,
            bridge,
            task_channel,
            health_period(50),
            Some(&mut journal),
            CoreObservers {
                ready: |_, _| true,
                record_health: |_| true,
                publish_fault: |_| {},
                latch_fault: || {},
            },
        );
        let producer = async {
            assert_eq!(
                channel.submit(10_u64),
                NanoAccessoryFrameSubmitOutcome::Enqueued
            );
            tokio::time::timeout(Duration::from_millis(100), async {
                while channel.counters.snapshot().processed_successfully == 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("episode enqueue and eye acknowledgement complete");
            channel.request_shutdown();
        };
        let (exit, ()) = tokio::join!(core, producer);
        assert!(matches!(exit, CoreExit::Shutdown { .. }));
        assert!(log.lock().unwrap().contains(&"pet_episode"));

        let shutdown = journal.shutdown(Duration::from_secs(1));
        assert!(shutdown.clean());
        let bytes = fs::read(path).expect("durable pet record");
        let record = NanoPetEpisodeEvidence::parse_ndjson_line(&bytes).expect("typed record");
        assert!(record.was_tap());
        assert!(record.replay_comparison().unwrap().matches());
    }

    #[cfg(feature = "nano-agent")]
    #[tokio::test(flavor = "current_thread")]
    async fn asynchronous_pet_journal_fault_latches_the_accessory_owner() {
        let root = PetJournalTestDirectory::create();
        let config = NanoPetEvidenceJournalConfig::for_state_root(&root.0).unwrap();
        let path = config.path();
        let mut journal = NanoPetEvidenceJournal::start(config).expect("journal startup");
        fs::remove_file(&path).expect("replace admitted path");
        fs::write(&path, []).expect("replacement file");
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();

        let (mut head, eye, bridge, _, _) = fakes(false, false, false, false, None);
        head.emit_pet_episode = true;
        let channel = Arc::new(LatestFrameChannel::new());
        let task_channel = Arc::clone(&channel);
        let (fault_tx, mut fault_rx) = tokio::sync::mpsc::unbounded_channel();
        let core = run_accessory_core(
            head,
            eye,
            bridge,
            task_channel,
            health_period(5),
            Some(&mut journal),
            CoreObservers {
                ready: |_, _| true,
                record_health: |_| true,
                publish_fault: move |fault| fault_tx.send(fault).unwrap(),
                latch_fault: || {},
            },
        );
        let producer = async {
            assert_eq!(
                channel.submit(11_u64),
                NanoAccessoryFrameSubmitOutcome::Enqueued
            );
            let fault = tokio::time::timeout(Duration::from_millis(100), fault_rx.recv())
                .await
                .expect("journal fault reaches owner within health cadence")
                .expect("fault publisher remains connected");
            assert!(matches!(
                fault,
                CoreTerminalFault::PetEvidence(
                    NanoPetEvidenceJournalRuntimeError::JournalPathReplaced { .. }
                )
            ));
            assert_eq!(
                channel.submit(12_u64),
                NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched
            );
            channel.request_shutdown();
        };
        let (exit, ()) = tokio::join!(core, producer);
        assert!(matches!(
            exit,
            CoreExit::Shutdown {
                terminal_fault: Some(CoreTerminalFault::PetEvidence(
                    NanoPetEvidenceJournalRuntimeError::JournalPathReplaced { .. }
                )),
                ..
            }
        ));
        let shutdown = journal.shutdown(Duration::from_secs(1));
        assert!(!shutdown.clean());
        assert!(matches!(
            shutdown.join(),
            super::NanoPetEvidenceJournalJoinEvidence::Joined(
                super::NanoPetEvidenceJournalExit::Fault(
                    NanoPetEvidenceJournalRuntimeError::JournalPathReplaced { .. }
                )
            )
        ));
    }

    #[test]
    fn accessory_observer_distinguishes_startup_degraded_expression_and_terminal_fault() {
        let channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let lifecycle = Arc::new(NanoAccessoryOwnerLifecycle::starting());
        lifecycle.mark_running();
        let observer = NanoAccessoryHealthObserver {
            channel: Arc::clone(&channel),
            head: Arc::new(Mutex::new(NanoAccessoryHeadHealthState::empty())),
            lifecycle: Arc::clone(&lifecycle),
            health_period: health_period(50),
            rgb_frame_freshness: Duration::from_millis(50),
        };
        assert_eq!(
            observer.snapshot().unwrap(),
            NanoAccessoryRuntimeHealth {
                head: NanoAccessoryComponentHealth::Degraded,
                eyes: NanoAccessoryComponentHealth::Ready,
                rgb_expression: NanoAccessoryComponentHealth::Degraded,
                successful_rgb_expression_frames: 0,
            }
        );
        channel
            .counters
            .record_processed_successfully()
            .expect("test health receipt");
        assert_eq!(
            observer.snapshot().unwrap().rgb_expression,
            NanoAccessoryComponentHealth::Ready
        );
        channel.latch_terminal_fault();
        lifecycle.mark_fault_latched();
        assert_eq!(
            observer.snapshot().unwrap(),
            NanoAccessoryRuntimeHealth {
                head: NanoAccessoryComponentHealth::Faulted,
                eyes: NanoAccessoryComponentHealth::Faulted,
                rgb_expression: NanoAccessoryComponentHealth::Faulted,
                successful_rgb_expression_frames: 1,
            }
        );
    }

    #[test]
    fn loop_liveness_advances_only_at_readiness_and_completed_health_boundaries() {
        let lifecycle = Arc::new(NanoAccessoryOwnerLifecycle::starting());
        let observer = NanoAccessoryLoopLivenessObserver {
            lifecycle: Arc::clone(&lifecycle),
        };
        assert_eq!(
            observer.snapshot(),
            NanoAccessoryLoopLivenessSnapshot {
                owner_state: NanoAccessoryOwnerState::Starting,
                completed_health_transactions: 0,
            }
        );

        lifecycle.mark_running();
        assert_eq!(
            observer.snapshot(),
            NanoAccessoryLoopLivenessSnapshot {
                owner_state: NanoAccessoryOwnerState::Running,
                completed_health_transactions: 1,
            }
        );
        assert!(lifecycle.record_completed_health_transaction());
        assert_eq!(observer.snapshot().completed_health_transactions, 2);

        lifecycle.mark_fault_latched();
        assert_eq!(
            observer.snapshot().owner_state,
            NanoAccessoryOwnerState::FaultLatched
        );
        assert!(lifecycle.record_completed_health_transaction());
        assert_eq!(
            observer.snapshot().completed_health_transactions,
            3,
            "the retained owner may keep producing diagnostic health evidence after the first fault"
        );
    }

    #[test]
    fn accessory_observer_degrades_when_the_last_rgb_receipt_expires() {
        let channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let lifecycle = Arc::new(NanoAccessoryOwnerLifecycle::starting());
        lifecycle.mark_running();
        let observer = NanoAccessoryHealthObserver {
            channel: Arc::clone(&channel),
            head: Arc::new(Mutex::new(NanoAccessoryHeadHealthState::empty())),
            lifecycle,
            health_period: health_period(50),
            rgb_frame_freshness: Duration::from_millis(50),
        };
        let expired_at = Instant::now()
            .checked_sub(Duration::from_millis(50))
            .expect("test instant has at least 50ms of history");
        channel
            .counters
            .record_processed_successfully_at(expired_at)
            .expect("test health receipt");

        assert_eq!(
            observer.snapshot().unwrap().rgb_expression,
            NanoAccessoryComponentHealth::Degraded
        );
        assert_eq!(
            observer
                .snapshot()
                .unwrap()
                .successful_rgb_expression_frames,
            1
        );
    }

    #[test]
    fn retained_observer_never_reports_released_owner_as_ready() {
        let channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let lifecycle = Arc::new(NanoAccessoryOwnerLifecycle::starting());
        lifecycle.mark_running();
        let observer = NanoAccessoryHealthObserver {
            channel,
            head: Arc::new(Mutex::new(NanoAccessoryHeadHealthState::empty())),
            lifecycle: Arc::clone(&lifecycle),
            health_period: health_period(50),
            rgb_frame_freshness: Duration::from_millis(50),
        };

        lifecycle.mark_shutting_down();
        assert_eq!(
            observer.snapshot(),
            Err(NanoAccessoryHealthStatusError::OwnerNotRunning {
                state: NanoAccessoryOwnerState::ShuttingDown,
            })
        );
        lifecycle.mark_stopped();
        assert_eq!(
            observer.snapshot(),
            Err(NanoAccessoryHealthStatusError::OwnerNotRunning {
                state: NanoAccessoryOwnerState::Stopped,
            })
        );
    }

    #[test]
    fn observer_reports_raw_ingress_disconnect_before_fault_propagation() {
        let channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let lifecycle = Arc::new(NanoAccessoryOwnerLifecycle::starting());
        lifecycle.mark_running();
        let observer = NanoAccessoryHealthObserver {
            channel: Arc::clone(&channel),
            head: Arc::new(Mutex::new(NanoAccessoryHeadHealthState::empty())),
            lifecycle,
            health_period: health_period(50),
            rgb_frame_freshness: Duration::from_millis(50),
        };

        channel.disconnect_ingress();
        assert_eq!(
            observer.snapshot(),
            Err(NanoAccessoryHealthStatusError::IngressDisconnected)
        );
    }

    #[test]
    fn observer_preserves_channel_poison_over_later_disconnect() {
        let channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let lifecycle = Arc::new(NanoAccessoryOwnerLifecycle::starting());
        lifecycle.mark_running();
        let observer = NanoAccessoryHealthObserver {
            channel: Arc::clone(&channel),
            head: Arc::new(Mutex::new(NanoAccessoryHeadHealthState::empty())),
            lifecycle,
            health_period: health_period(50),
            rgb_frame_freshness: Duration::from_millis(50),
        };

        let poisoned_channel = Arc::clone(&channel);
        assert!(
            thread::spawn(move || {
                let _guard = poisoned_channel
                    .slot
                    .lock()
                    .expect("initially healthy slot");
                panic!("poison raw channel slot");
            })
            .join()
            .is_err()
        );
        channel.disconnect_ingress();
        assert_eq!(
            observer.snapshot(),
            Err(NanoAccessoryHealthStatusError::ChannelPoisoned)
        );
    }

    #[test]
    fn observer_preserves_actual_channel_poison_over_later_shutdown_request() {
        let channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let lifecycle = Arc::new(NanoAccessoryOwnerLifecycle::starting());
        lifecycle.mark_running();
        let observer = NanoAccessoryHealthObserver {
            channel: Arc::clone(&channel),
            head: Arc::new(Mutex::new(NanoAccessoryHeadHealthState::empty())),
            lifecycle,
            health_period: health_period(50),
            rgb_frame_freshness: Duration::from_millis(50),
        };

        let poisoned_channel = Arc::clone(&channel);
        assert!(
            thread::spawn(move || {
                let _guard = poisoned_channel
                    .slot
                    .lock()
                    .expect("initially healthy slot");
                panic!("poison raw channel slot");
            })
            .join()
            .is_err()
        );
        channel.request_shutdown();
        assert_eq!(
            observer.snapshot(),
            Err(NanoAccessoryHealthStatusError::ChannelPoisoned)
        );
        assert!(channel.poisoned.load(Ordering::Acquire));
    }

    #[test]
    fn observer_preserves_latched_fault_over_later_disconnect() {
        let channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let lifecycle = Arc::new(NanoAccessoryOwnerLifecycle::starting());
        lifecycle.mark_running();
        let observer = NanoAccessoryHealthObserver {
            channel: Arc::clone(&channel),
            head: Arc::new(Mutex::new(NanoAccessoryHeadHealthState::empty())),
            lifecycle: Arc::clone(&lifecycle),
            health_period: health_period(50),
            rgb_frame_freshness: Duration::from_millis(50),
        };

        lifecycle.mark_fault_latched();
        channel.disconnect_ingress();
        assert_eq!(
            observer
                .snapshot()
                .expect("latched health remains observable"),
            NanoAccessoryRuntimeHealth {
                head: NanoAccessoryComponentHealth::Faulted,
                eyes: NanoAccessoryComponentHealth::Faulted,
                rgb_expression: NanoAccessoryComponentHealth::Faulted,
                successful_rgb_expression_frames: 0,
            }
        );
    }

    #[test]
    fn accessory_thread_panic_invalidates_retained_observer_readiness() {
        let channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let lifecycle = Arc::new(NanoAccessoryOwnerLifecycle::starting());
        lifecycle.mark_running();
        let observer = NanoAccessoryHealthObserver {
            channel,
            head: Arc::new(Mutex::new(NanoAccessoryHeadHealthState::empty())),
            lifecycle: Arc::clone(&lifecycle),
            health_period: health_period(50),
            rgb_frame_freshness: Duration::from_millis(50),
        };

        let panic_lifecycle = Arc::clone(&lifecycle);
        assert!(
            thread::spawn(move || {
                let _guard = NanoAccessoryThreadLifecycleGuard::new(panic_lifecycle);
                panic!("test accessory owner panic after readiness");
            })
            .join()
            .is_err()
        );
        assert_eq!(
            observer.snapshot(),
            Err(NanoAccessoryHealthStatusError::OwnerNotRunning {
                state: NanoAccessoryOwnerState::OwnerExitedUnexpectedly,
            })
        );
    }

    #[test]
    fn poisoned_rgb_health_receipt_is_explicit_and_does_not_increment_count() {
        let counters = Arc::new(LatestFrameCounters::new());
        let poison = Arc::clone(&counters);
        assert!(
            thread::spawn(move || {
                let _guard = poison
                    .receipts
                    .last_processed_successfully_at
                    .lock()
                    .expect("initially healthy test mutex");
                panic!("poison test mutex");
            })
            .join()
            .is_err()
        );

        assert_eq!(
            counters.record_processed_successfully(),
            Err(NanoAccessoryHealthStatusError::Poisoned)
        );
        assert_eq!(counters.snapshot().processed_successfully, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn failed_startup_never_reports_readiness() {
        let (head, eye, bridge, log, _) = fakes(true, false, false, false, None);
        let channel = Arc::new(LatestFrameChannel::new());
        let ready_seen = Arc::new(AtomicBool::new(false));
        let ready_flag = Arc::clone(&ready_seen);
        let exit = run_accessory_core(
            head,
            eye,
            bridge,
            channel,
            health_period(50),
            None,
            CoreObservers {
                ready: move |_, _| {
                    ready_flag.store(true, Ordering::SeqCst);
                    true
                },
                record_health: |_| true,
                publish_fault: |_| {},
                latch_fault: || {},
            },
        )
        .await;
        assert!(matches!(exit, CoreExit::EyeStartupFailed("eye_start")));
        assert!(!ready_seen.load(Ordering::SeqCst));
        assert_eq!(&*log.lock().unwrap(), &["eye_start"]);

        let (head, eye, bridge, log, _) = fakes(false, true, false, false, None);
        let exit = run_accessory_core(
            head,
            eye,
            bridge,
            Arc::new(LatestFrameChannel::new()),
            health_period(50),
            None,
            CoreObservers {
                ready: |_, _| panic!("head startup failure cannot report ready"),
                record_health: |_| true,
                publish_fault: |_| {},
                latch_fault: || {},
            },
        )
        .await;
        assert!(matches!(exit, CoreExit::HeadStartupFailed { .. }));
        assert_eq!(
            &*log.lock().unwrap(),
            &["eye_start", "head_start", "eye_shutdown"]
        );

        let (mut head, eye, bridge, log, _) = fakes(false, false, false, false, None);
        head.fail_return = true;
        let ready_seen = Arc::new(AtomicBool::new(false));
        let ready_flag = Arc::clone(&ready_seen);
        let exit = run_accessory_core(
            head,
            eye,
            bridge,
            Arc::new(LatestFrameChannel::new()),
            health_period(50),
            None,
            CoreObservers {
                ready: move |_, _| {
                    ready_flag.store(true, Ordering::SeqCst);
                    true
                },
                record_health: |_| true,
                publish_fault: |_| {},
                latch_fault: || {},
            },
        )
        .await;
        assert!(matches!(
            exit,
            CoreExit::HeadStartupFailed {
                source: "head_return",
                ..
            }
        ));
        assert!(!ready_seen.load(Ordering::SeqCst));
        assert_eq!(
            &*log.lock().unwrap(),
            &["eye_start", "head_start", "head_return", "eye_shutdown"]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn periodic_health_runs_until_explicit_shutdown() {
        let (head, eye, bridge, _, health_calls) = fakes(false, false, false, false, None);
        let channel = Arc::new(LatestFrameChannel::new());
        let task_channel = Arc::clone(&channel);
        let latest_health = Arc::new(AtomicUsize::new(0));
        let task_latest_health = Arc::clone(&latest_health);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(5),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: move |evidence| {
                        task_latest_health.store(evidence, Ordering::SeqCst);
                        true
                    },
                    publish_fault: |_| {},
                    latch_fault: || {},
                },
            )
            .await
        });
        tokio::time::timeout(Duration::from_millis(100), async {
            while health_calls.load(Ordering::SeqCst) < 3 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("three periodic health observations within bounded test time");
        channel.request_shutdown();
        let _ = task.await.unwrap();
        assert!(health_calls.load(Ordering::SeqCst) >= 3);
        assert_eq!(
            latest_health.load(Ordering::SeqCst),
            health_calls.load(Ordering::SeqCst)
        );
    }

    async fn assert_fault_does_not_shutdown_head(
        fail_head_gaze: bool,
        fail_bridge: bool,
        fail_eye_apply: bool,
        expected: CoreTerminalFault<&'static str, &'static str, &'static str, &'static str>,
    ) {
        let (mut head, eye, bridge, log, health_calls) =
            fakes(false, false, fail_bridge, fail_eye_apply, None);
        head.fail_gaze = fail_head_gaze;
        let channel = Arc::new(LatestFrameChannel::new());
        let (fault_tx, mut fault_rx) = tokio::sync::mpsc::unbounded_channel();
        let task_channel = Arc::clone(&channel);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(5),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: move |fault| {
                        fault_tx.send(fault).unwrap();
                    },
                    latch_fault: || {},
                },
            )
            .await
        });
        assert_eq!(
            channel.submit(7_u64),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        assert_eq!(fault_rx.recv().await, Some(expected));
        if fail_head_gaze {
            let observed = log.lock().unwrap();
            assert!(observed.contains(&"bridge"));
            assert!(observed.contains(&"head_gaze"));
            assert!(!observed.contains(&"eye_apply"));
        }
        tokio::time::sleep(Duration::from_millis(12)).await;
        assert!(!log.lock().unwrap().contains(&"head_shutdown"));
        assert!(health_calls.load(Ordering::SeqCst) >= 1);
        assert_eq!(
            channel.submit(8_u64),
            NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched
        );
        channel.request_shutdown();
        assert!(matches!(task.await.unwrap(), CoreExit::Shutdown { .. }));
        assert!(
            log.lock()
                .unwrap()
                .ends_with(&["eye_shutdown", "head_shutdown"])
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn bridge_fault_is_published_without_implicit_head_teardown() {
        assert_fault_does_not_shutdown_head(
            false,
            true,
            false,
            CoreTerminalFault::Bridge("bridge"),
        )
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn character_bridge_prepares_before_head_and_head_fault_precedes_eye_apply() {
        assert_fault_does_not_shutdown_head(
            true,
            false,
            false,
            CoreTerminalFault::HeadGaze("head_gaze"),
        )
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn consumed_ingress_clock_failure_is_a_typed_terminal_bridge_fault() {
        let (head, eye, _, log, _) = fakes(false, false, false, false, None);
        let bridge = ClockWorkBridge {
            log: Arc::clone(&log),
        };
        let channel = Arc::new(LatestFrameChannel::new());
        let task_channel = Arc::clone(&channel);
        let (fault_tx, mut fault_rx) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(50),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: move |fault| {
                        fault_tx.send(fault).unwrap();
                    },
                    latch_fault: || {},
                },
            )
            .await
        });
        assert_eq!(
            channel.submit_first_terminal(
                Err(ClockError::ElapsedNanosecondsOutOfRange {
                    elapsed_nanoseconds: u128::MAX,
                }),
                true,
            ),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        assert_eq!(
            fault_rx.recv().await,
            Some(CoreTerminalFault::Bridge(RgbExpressionBridgeError::Clock(
                ClockError::ElapsedNanosecondsOutOfRange {
                    elapsed_nanoseconds: u128::MAX,
                }
            )))
        );
        assert!(
            !log.lock().unwrap().contains(&"clock_bridge_frame"),
            "typed clock failure must not be passed to the frame bridge"
        );

        channel.request_shutdown();
        assert!(matches!(task.await.unwrap(), CoreExit::Shutdown { .. }));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn eye_fault_is_published_without_implicit_head_teardown() {
        assert_fault_does_not_shutdown_head(
            false,
            false,
            true,
            CoreTerminalFault::EyeApply("eye_apply"),
        )
        .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn health_and_channel_faults_latch_without_implicit_shutdown() {
        let (head, eye, bridge, log, _) = fakes(false, false, false, false, Some(1));
        let channel = Arc::new(LatestFrameChannel::new());
        let (fault_tx, mut fault_rx) = tokio::sync::mpsc::unbounded_channel();
        let task_channel = Arc::clone(&channel);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(5),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: move |fault| fault_tx.send(fault).unwrap(),
                    latch_fault: || {},
                },
            )
            .await
        });
        assert_eq!(
            fault_rx.recv().await,
            Some(CoreTerminalFault::HeadHealth("head_health"))
        );
        assert!(!log.lock().unwrap().contains(&"head_shutdown"));
        channel.request_shutdown();
        let _ = task.await.unwrap();

        let (head, eye, bridge, log, _) = fakes(false, false, false, false, None);
        let channel = Arc::new(LatestFrameChannel::new());
        let (fault_tx, mut fault_rx) = tokio::sync::mpsc::unbounded_channel();
        let task_channel = Arc::clone(&channel);
        let task = tokio::spawn(async move {
            run_accessory_core(
                head,
                eye,
                bridge,
                task_channel,
                health_period(5),
                None,
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: move |fault| fault_tx.send(fault).unwrap(),
                    latch_fault: || {},
                },
            )
            .await
        });
        channel.disconnect_ingress();
        assert_eq!(
            fault_rx.recv().await,
            Some(CoreTerminalFault::IngressDisconnected)
        );
        assert!(!log.lock().unwrap().contains(&"head_shutdown"));
        channel.request_shutdown();
        let _ = task.await.unwrap();
    }
}
