//! Fail-closed, wheels-off Nano bench bring-up.
//!
//! This module is deliberately a different runtime from navigation. It takes
//! ownership of a [`AgentAuthoritySupervisor`] only while that supervisor is
//! in `ReadyStopped`, retains the exact confirmed-zero evidence, and exposes no
//! base-command or motion-authority API. Camera, head, eye, and telemetry I/O
//! are injected ports so the complete ordering can be tested without hardware.
//!
//! A successful start means only that the configured head actor established
//! its natural-hold contract, the exact OAK produced RGB/depth/IMU samples,
//! Rerun accepted actual RGB, rectified-left depth, and raw sensor-native IMU
//! payloads, and KEP2 admitted the RGB-derived eye intent. It does not prove
//! photons, physical head pose, depth accuracy, IMU calibration, or wheel
//! removal.

use std::fmt;
use std::future::Future;
use std::num::{NonZeroU16, NonZeroU32, NonZeroUsize};
use std::time::Duration;

use kiko_device_inventory::{LoadedExpectedManifestV1, MAX_OAK_MXID_BYTES};
use kiko_expression_core::StreamEpochId;
use kiko_expression_runtime::PreparedEyeIntent;
use kiko_eye_runtime::{
    EyeRuntimeConfig, MonotonicClock as EyeMonotonicClock, OsEyeSessionMaterialError,
    OsEyeSessionMaterialGenerator, StaticEyeRuntimeConfig,
};
pub use kiko_head_runtime::{
    ConfiguredHeadPoseBounds as WheelsOffConfiguredPoseBounds,
    ConfiguredHeadPoseBoundsError as WheelsOffConfiguredPoseBoundsError,
    HeadPoseWithinConfiguredBounds as ObservedPoseWithinConfiguredBounds,
};
use kiko_head_runtime::{HeadRuntimeConfig, PhysicalTorqueEnableConsent};
use kiko_supervisor_core::{
    ConfirmedBaseZero, MonotonicInstant, ReadinessBinding, SupervisorState, ZeroEvidenceError,
};
use oak_sys::{
    ConnectedDeviceIdentityError, ConnectionError, DepthAlignment, DepthError, DepthFrame, Device,
    DeviceConfig, DeviceConfigError, ImageError, ImageFrame, ImuError, ImuSample, StreamId,
    UsbTransportEvidenceError, UsbTransportSpeed,
};

use crate::HostMonotonicTimestamp;

use super::{
    AgentAuthoritySupervisor, NanoAccessoryManifestBindingError, NanoAgentPolicyConfigV1,
    NanoManifestBoundEyePolicy, NanoManifestBoundHeadPolicy, NanoRgbExpressionConfig,
    RgbExpressionBridge, RgbExpressionBridgeError, RgbExpressionBridgeOutcome,
};

/// Maximum individual OAK call deadline in the bench runtime.
pub const MAX_WHEELS_OFF_BENCH_CAPTURE_TIMEOUT_MS: u32 = 5_000;
/// Maximum attempts per required OAK stream during bounded startup.
pub const MAX_WHEELS_OFF_BENCH_CAPTURE_ATTEMPTS: u16 = 100;
/// Maximum aggregate configured wait per required OAK stream.
pub const MAX_WHEELS_OFF_BENCH_CAPTURE_BUDGET_MS: u64 = 30_000;
/// Maximum bounded Rerun flush wait.
pub const MAX_WHEELS_OFF_BENCH_RERUN_FLUSH_MS: u64 = 60_000;

/// Explicit, bounded camera observation policy for a wheels-off start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelsOffBenchCapturePlan {
    timeout_ms: NonZeroU32,
    attempts: NonZeroU16,
}

impl WheelsOffBenchCapturePlan {
    pub fn try_new(timeout_ms: u32, attempts: u16) -> Result<Self, BenchCapturePlanError> {
        let timeout_ms = NonZeroU32::new(timeout_ms).ok_or(BenchCapturePlanError::ZeroTimeout)?;
        if timeout_ms.get() > MAX_WHEELS_OFF_BENCH_CAPTURE_TIMEOUT_MS {
            return Err(BenchCapturePlanError::TimeoutAboveMaximum {
                actual_ms: timeout_ms.get(),
                maximum_ms: MAX_WHEELS_OFF_BENCH_CAPTURE_TIMEOUT_MS,
            });
        }
        let attempts = NonZeroU16::new(attempts).ok_or(BenchCapturePlanError::ZeroAttempts)?;
        if attempts.get() > MAX_WHEELS_OFF_BENCH_CAPTURE_ATTEMPTS {
            return Err(BenchCapturePlanError::AttemptsAboveMaximum {
                actual: attempts.get(),
                maximum: MAX_WHEELS_OFF_BENCH_CAPTURE_ATTEMPTS,
            });
        }
        let aggregate_ms = u64::from(timeout_ms.get()) * u64::from(attempts.get());
        if aggregate_ms > MAX_WHEELS_OFF_BENCH_CAPTURE_BUDGET_MS {
            return Err(BenchCapturePlanError::AggregateBudgetAboveMaximum {
                actual_ms: aggregate_ms,
                maximum_ms: MAX_WHEELS_OFF_BENCH_CAPTURE_BUDGET_MS,
            });
        }
        Ok(Self {
            timeout_ms,
            attempts,
        })
    }

    pub const fn timeout_ms(self) -> NonZeroU32 {
        self.timeout_ms
    }

    pub const fn attempts(self) -> NonZeroU16 {
        self.attempts
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchCapturePlanError {
    ZeroTimeout,
    TimeoutAboveMaximum { actual_ms: u32, maximum_ms: u32 },
    ZeroAttempts,
    AttemptsAboveMaximum { actual: u16, maximum: u16 },
    AggregateBudgetAboveMaximum { actual_ms: u64, maximum_ms: u64 },
}

impl fmt::Display for BenchCapturePlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid wheels-off capture plan: {self:?}")
    }
}

impl std::error::Error for BenchCapturePlanError {}

/// Bench-local exact OAK selection and native stream contract.
///
/// This deliberately does not depend on the unfinished production startup
/// scaffold. It is cross-checked against the already-loaded device inventory
/// before any OAK connection can be attempted.
#[derive(Clone, Debug)]
pub struct WheelsOffBenchOakConfig {
    expected_mxid: Box<str>,
    device: DeviceConfig,
}

impl WheelsOffBenchOakConfig {
    pub fn try_new(
        mut expected_mxid: String,
        device: DeviceConfig,
    ) -> Result<Self, WheelsOffBenchOakConfigError> {
        if expected_mxid.len() < 8 || expected_mxid.len() > MAX_OAK_MXID_BYTES {
            return Err(WheelsOffBenchOakConfigError::MxidLength {
                actual_bytes: expected_mxid.len(),
                minimum_bytes: 8,
                maximum_bytes: MAX_OAK_MXID_BYTES,
            });
        }
        if let Some((index, byte)) = expected_mxid
            .bytes()
            .enumerate()
            .find(|(_, byte)| !byte.is_ascii_hexdigit())
        {
            return Err(WheelsOffBenchOakConfigError::MxidByte { index, byte });
        }
        expected_mxid.make_ascii_uppercase();
        if expected_mxid.bytes().all(|byte| byte == b'0') {
            return Err(WheelsOffBenchOakConfigError::ZeroMxid);
        }
        device
            .validate()
            .map_err(WheelsOffBenchOakConfigError::NativeConfig)?;
        if device.rgb.is_none() {
            return Err(WheelsOffBenchOakConfigError::RequiredStreamDisabled { stream: "RGB" });
        }
        let mono = device
            .mono
            .ok_or(WheelsOffBenchOakConfigError::RequiredStreamDisabled {
                stream: "rectified mono pair",
            })?;
        if !mono.rectified {
            return Err(WheelsOffBenchOakConfigError::MonoNotRectified);
        }
        let depth = device
            .depth
            .ok_or(WheelsOffBenchOakConfigError::RequiredStreamDisabled { stream: "depth" })?;
        if depth.alignment != DepthAlignment::RectifiedLeft {
            return Err(WheelsOffBenchOakConfigError::DepthAlignment {
                actual: depth.alignment,
                required: DepthAlignment::RectifiedLeft,
            });
        }
        if device.imu.is_none() {
            return Err(WheelsOffBenchOakConfigError::RequiredStreamDisabled { stream: "IMU" });
        }
        if device.usb_transport.minimum() < UsbTransportSpeed::Super {
            return Err(WheelsOffBenchOakConfigError::UsbMinimumBelowProduction {
                actual: device.usb_transport.minimum(),
                required: UsbTransportSpeed::Super,
            });
        }
        Ok(Self {
            expected_mxid: expected_mxid.into_boxed_str(),
            device,
        })
    }

    pub fn expected_mxid(&self) -> &str {
        &self.expected_mxid
    }

    pub const fn device(&self) -> &DeviceConfig {
        &self.device
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WheelsOffBenchOakConfigError {
    MxidLength {
        actual_bytes: usize,
        minimum_bytes: usize,
        maximum_bytes: usize,
    },
    MxidByte {
        index: usize,
        byte: u8,
    },
    ZeroMxid,
    NativeConfig(DeviceConfigError),
    RequiredStreamDisabled {
        stream: &'static str,
    },
    MonoNotRectified,
    DepthAlignment {
        actual: DepthAlignment,
        required: DepthAlignment,
    },
    UsbMinimumBelowProduction {
        actual: UsbTransportSpeed,
        required: UsbTransportSpeed,
    },
}

impl fmt::Display for WheelsOffBenchOakConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid wheels-off OAK config: {self:?}")
    }
}

impl std::error::Error for WheelsOffBenchOakConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NativeConfig(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelsOffBenchRerunPlan {
    flush_timeout: Duration,
}

impl WheelsOffBenchRerunPlan {
    pub fn try_from_milliseconds(value: u64) -> Result<Self, WheelsOffBenchRerunPlanError> {
        if value == 0 || value > MAX_WHEELS_OFF_BENCH_RERUN_FLUSH_MS {
            return Err(WheelsOffBenchRerunPlanError::FlushTimeoutOutOfRange {
                actual_ms: value,
                minimum_ms: 1,
                maximum_ms: MAX_WHEELS_OFF_BENCH_RERUN_FLUSH_MS,
            });
        }
        Ok(Self {
            flush_timeout: Duration::from_millis(value),
        })
    }

    pub const fn flush_timeout(self) -> Duration {
        self.flush_timeout
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffBenchRerunPlanError {
    FlushTimeoutOutOfRange {
        actual_ms: u64,
        minimum_ms: u64,
        maximum_ms: u64,
    },
}

impl fmt::Display for WheelsOffBenchRerunPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid wheels-off Rerun plan: {self:?}")
    }
}

impl std::error::Error for WheelsOffBenchRerunPlanError {}

/// Exclusive stopped-base admission retained for the entire bench lifetime.
///
/// Construction consumes the sole authority adapter. No method exposes it or
/// allows a mode request, making nonzero wheel commands unrepresentable through
/// this runtime.
pub struct WheelsOffBaseAdmission {
    authority: AgentAuthoritySupervisor,
    readiness: ReadinessBinding,
    zero: ConfirmedBaseZero,
    admitted_at: HostMonotonicTimestamp,
}

impl WheelsOffBaseAdmission {
    pub const fn readiness(&self) -> ReadinessBinding {
        self.readiness
    }

    pub const fn zero(&self) -> ConfirmedBaseZero {
        self.zero
    }

    pub const fn admitted_at(&self) -> HostMonotonicTimestamp {
        self.admitted_at
    }

    pub const fn supervisor_state(&self) -> SupervisorState {
        self.authority.state()
    }
}

/// One newly applied physical zero plus transport-specific receipt evidence.
///
/// Construction parses the weak host result exactly once. A nonzero request,
/// cached result, nonzero controller output, or faulted result cannot inhabit
/// this type.
#[derive(Debug)]
pub struct RefreshedBaseZero<E> {
    evidence: E,
    confirmed: ConfirmedBaseZero,
}

impl<E> RefreshedBaseZero<E> {
    pub fn try_from_host_result(
        evidence: E,
        result: robot_protocol::v2::HostCommandResult,
        observed_at: HostMonotonicTimestamp,
    ) -> Result<Self, ZeroEvidenceError> {
        let confirmed = ConfirmedBaseZero::try_from_host_command_result(
            result,
            MonotonicInstant::from_nanos_since_process_start(observed_at.as_nanos()),
        )?;
        Ok(Self {
            evidence,
            confirmed,
        })
    }

    pub const fn evidence(&self) -> &E {
        &self.evidence
    }

    pub const fn confirmed(&self) -> ConfirmedBaseZero {
        self.confirmed
    }
}

/// Sole physical-base surface retained by the wheels-off runtime.
///
/// The method has no PWM or target argument: implementations can only return
/// a parsed newly-applied zero. The runtime invokes it before every accessory
/// teardown path.
pub trait WheelsOffBaseCleanupPort {
    type Evidence;
    type HealthEvidence;
    type DisarmEvidence;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Prove that the independently running zero-only keeper and its
    /// controller session are still healthy. The bench polls this before each
    /// runtime RGB cycle so keeper failure enters the normal cleanup path.
    fn check_health(
        &mut self,
    ) -> impl Future<Output = Result<Self::HealthEvidence, Self::Error>> + Send;

    /// Request a newly sequenced zero from a concurrently maintained zero-only
    /// keeper. Implementations must keep the controller's applied-zero lease
    /// alive independently while other bench I/O is awaiting data.
    fn refresh_zero(
        &mut self,
    ) -> impl Future<Output = Result<RefreshedBaseZero<Self::Evidence>, Self::Error>> + Send;

    /// Stop the keeper and consume its physical controller session.
    fn disarm(&mut self) -> impl Future<Output = Result<Self::DisarmEvidence, Self::Error>> + Send;
}

/// Process-level stop signal retained as typed startup-cancellation evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffBenchCancellation {
    Interrupt,
    Terminate,
}

/// Exact startup boundary at which a queued stop signal was observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffBenchCancellationCheckpoint {
    BeforeInitialBaseZero,
    BeforeCameraConnect,
    BeforeDepthCapture { attempt: u16 },
    BeforeImuCapture { attempt: u16 },
    BeforeRgbCapture { attempt: u16 },
    BeforeHeadBaseZero,
    BeforeHeadStart,
    AfterHeadStart,
    BeforeEyeSession,
    BeforeEyeBaseZero,
    BeforeEyeStart,
    AfterEyeStart,
    BeforeExpressionRgbCapture { attempt: u16 },
    BeforeFirstEyeApply,
    BeforeRunning,
}

/// Non-blocking cancellation boundary. Signal acquisition runs independently;
/// startup only polls the already-queued result between bounded operations.
pub trait WheelsOffBenchCancellationPort {
    fn poll_cancellation(&mut self) -> Option<WheelsOffBenchCancellation>;
}

#[derive(Debug)]
pub enum WheelsOffBaseCleanupError<E> {
    Port(E),
    ControllerMismatch,
    ControlEpochMismatch,
    SequenceNotIncreasing { previous: u32, actual: u32 },
    ObservationTimeNotIncreasing { previous_ns: u64, actual_ns: u64 },
}

impl<E: fmt::Debug> fmt::Display for WheelsOffBaseCleanupError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "wheels-off base-zero cleanup failed: {self:?}")
    }
}

impl<E> std::error::Error for WheelsOffBaseCleanupError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Port(source) => Some(source),
            Self::ControllerMismatch
            | Self::ControlEpochMismatch
            | Self::SequenceNotIncreasing { .. }
            | Self::ObservationTimeNotIncreasing { .. } => None,
        }
    }
}

/// Exact manifest-bound configuration admitted for a wheels-off run.
pub struct WheelsOffBenchPlan {
    base: WheelsOffBaseAdmission,
    oak: WheelsOffBenchOakConfig,
    head: HeadRuntimeConfig,
    head_consent: PhysicalTorqueEnableConsent,
    eye: StaticEyeRuntimeConfig,
    rgb_expression: NanoRgbExpressionConfig,
    configured_pose_bounds: WheelsOffConfiguredPoseBounds,
    rerun_flush_timeout: Duration,
    capture: WheelsOffBenchCapturePlan,
}

/// Fully parsed bench-only configuration. Construction performs no I/O.
#[derive(Clone, Debug)]
pub struct WheelsOffBenchConfiguration {
    oak: WheelsOffBenchOakConfig,
    capture: WheelsOffBenchCapturePlan,
    rerun: WheelsOffBenchRerunPlan,
    configured_pose_bounds: WheelsOffConfiguredPoseBounds,
}

impl WheelsOffBenchConfiguration {
    pub const fn new(
        oak: WheelsOffBenchOakConfig,
        capture: WheelsOffBenchCapturePlan,
        rerun: WheelsOffBenchRerunPlan,
        configured_pose_bounds: WheelsOffConfiguredPoseBounds,
    ) -> Self {
        Self {
            oak,
            capture,
            rerun,
            configured_pose_bounds,
        }
    }
}

impl WheelsOffBenchPlan {
    /// Consume a stopped authority owner and bind it to the exact loaded
    /// inventory, accessory policy, and bench-local OAK contract.
    ///
    /// The readiness binding must name the exact loaded inventory
    /// representation and controller identity. Accessory and RGB policies must
    /// be enabled. `rerun` is explicit so cleanup never uses a hidden timeout.
    pub fn admit(
        inventory: &LoadedExpectedManifestV1,
        policy: NanoAgentPolicyConfigV1,
        authority: AgentAuthoritySupervisor,
        now: HostMonotonicTimestamp,
        bench: WheelsOffBenchConfiguration,
    ) -> Result<Self, WheelsOffBenchAdmissionError> {
        let WheelsOffBenchConfiguration {
            oak,
            capture,
            rerun,
            configured_pose_bounds,
        } = bench;
        if oak.expected_mxid() != inventory.manifest().oak().mxid().as_str() {
            return Err(WheelsOffBenchAdmissionError::OakMxidMismatch);
        }
        let policy = policy
            .bind_accessories_to_manifest(inventory.manifest())
            .map_err(WheelsOffBenchAdmissionError::AccessoryManifestBinding)?;

        let (readiness, zero) = match authority.state() {
            SupervisorState::ReadyStopped { readiness, zero } => (readiness, zero),
            state => {
                return Err(WheelsOffBenchAdmissionError::BaseNotReadyStopped {
                    actual: state.kind(),
                });
            }
        };
        if readiness.controller_uid() != *inventory.manifest().stm32().controller_uid() {
            return Err(WheelsOffBenchAdmissionError::ReadinessControllerMismatch);
        }
        if readiness.hardware_manifest().as_bytes() != inventory.content_sha256().as_bytes() {
            return Err(WheelsOffBenchAdmissionError::ReadinessInventoryDigestMismatch);
        }
        require_zero_bound(readiness, zero)?;
        let age_ns = now
            .as_nanos()
            .checked_sub(zero.observed_at().as_nanos())
            .ok_or(WheelsOffBenchAdmissionError::ZeroObservedInFuture {
                observed_at_ns: zero.observed_at().as_nanos(),
                now_ns: now.as_nanos(),
            })?;
        let maximum_zero_age_ns = policy.supervisor().maximum_zero_age().as_nanos();
        if age_ns >= maximum_zero_age_ns {
            return Err(WheelsOffBenchAdmissionError::ZeroEvidenceStale {
                age_ns,
                maximum_exclusive_ns: maximum_zero_age_ns,
            });
        }

        let (head, head_consent) = match policy.head() {
            NanoManifestBoundHeadPolicy::NaturalHold(config) => config.clone().into_parts(),
            NanoManifestBoundHeadPolicy::Disabled => {
                return Err(WheelsOffBenchAdmissionError::HeadDisabled);
            }
        };
        let eye = match policy.eye() {
            NanoManifestBoundEyePolicy::Kep2(config) => config.clone(),
            NanoManifestBoundEyePolicy::Disabled => {
                return Err(WheelsOffBenchAdmissionError::EyeDisabled);
            }
        };
        let rgb_expression = policy
            .rgb_expression()
            .scene_motion()
            .ok_or(WheelsOffBenchAdmissionError::RgbExpressionDisabled)?;
        let rerun_flush_timeout = rerun.flush_timeout();

        Ok(Self {
            base: WheelsOffBaseAdmission {
                authority,
                readiness,
                zero,
                admitted_at: now,
            },
            oak,
            head,
            head_consent,
            eye,
            rgb_expression,
            configured_pose_bounds,
            rerun_flush_timeout,
            capture,
        })
    }

    pub const fn base(&self) -> &WheelsOffBaseAdmission {
        &self.base
    }

    pub const fn oak(&self) -> &WheelsOffBenchOakConfig {
        &self.oak
    }

    pub const fn head(&self) -> &HeadRuntimeConfig {
        &self.head
    }

    pub const fn eye(&self) -> &StaticEyeRuntimeConfig {
        &self.eye
    }

    pub const fn rgb_expression(&self) -> NanoRgbExpressionConfig {
        self.rgb_expression
    }

    pub const fn configured_pose_bounds(&self) -> WheelsOffConfiguredPoseBounds {
        self.configured_pose_bounds
    }

    pub const fn rerun_flush_timeout(&self) -> Duration {
        self.rerun_flush_timeout
    }

    pub const fn capture(&self) -> WheelsOffBenchCapturePlan {
        self.capture
    }
}

fn require_zero_bound(
    readiness: ReadinessBinding,
    zero: ConfirmedBaseZero,
) -> Result<(), WheelsOffBenchAdmissionError> {
    if zero.controller_uid() != readiness.controller_uid()
        || zero.controller_boot_id() != readiness.controller_boot_id()
    {
        return Err(WheelsOffBenchAdmissionError::ZeroControllerMismatch);
    }
    if zero.control_epoch() != readiness.control_epoch() {
        return Err(WheelsOffBenchAdmissionError::ZeroControlEpochMismatch);
    }
    Ok(())
}

fn require_new_cleanup_zero<E>(
    readiness: ReadinessBinding,
    previous: ConfirmedBaseZero,
    actual: ConfirmedBaseZero,
) -> Result<(), WheelsOffBaseCleanupError<E>> {
    if actual.controller_uid() != readiness.controller_uid()
        || actual.controller_boot_id() != readiness.controller_boot_id()
    {
        return Err(WheelsOffBaseCleanupError::ControllerMismatch);
    }
    if actual.control_epoch() != readiness.control_epoch() {
        return Err(WheelsOffBaseCleanupError::ControlEpochMismatch);
    }
    if actual.sequence().get() <= previous.sequence().get() {
        return Err(WheelsOffBaseCleanupError::SequenceNotIncreasing {
            previous: previous.sequence().get(),
            actual: actual.sequence().get(),
        });
    }
    if actual.observed_at().as_nanos() <= previous.observed_at().as_nanos() {
        return Err(WheelsOffBaseCleanupError::ObservationTimeNotIncreasing {
            previous_ns: previous.observed_at().as_nanos(),
            actual_ns: actual.observed_at().as_nanos(),
        });
    }
    Ok(())
}

#[derive(Debug)]
pub enum WheelsOffBenchAdmissionError {
    OakMxidMismatch,
    AccessoryManifestBinding(NanoAccessoryManifestBindingError),
    BaseNotReadyStopped {
        actual: kiko_supervisor_core::SupervisorStateKind,
    },
    ReadinessControllerMismatch,
    ReadinessInventoryDigestMismatch,
    ZeroControllerMismatch,
    ZeroControlEpochMismatch,
    ZeroObservedInFuture {
        observed_at_ns: u64,
        now_ns: u64,
    },
    ZeroEvidenceStale {
        age_ns: u64,
        maximum_exclusive_ns: u64,
    },
    HeadDisabled,
    EyeDisabled,
    RgbExpressionDisabled,
}

impl fmt::Display for WheelsOffBenchAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "wheels-off bench admission failed: {self:?}")
    }
}

impl std::error::Error for WheelsOffBenchAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::AccessoryManifestBinding(source) => Some(source),
            _ => None,
        }
    }
}

/// OAK RGB identity retained independently of the frame buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchRgbFrameEvidence {
    capture_sequence: u64,
    delivery_sequence: u64,
    device_timestamp_ns: i64,
    width_px: u32,
    height_px: u32,
    stride_bytes: u32,
}

impl BenchRgbFrameEvidence {
    pub const fn capture_sequence(self) -> u64 {
        self.capture_sequence
    }

    pub const fn delivery_sequence(self) -> u64 {
        self.delivery_sequence
    }

    pub const fn device_timestamp_ns(self) -> i64 {
        self.device_timestamp_ns
    }

    pub const fn width_px(self) -> u32 {
        self.width_px
    }

    pub const fn height_px(self) -> u32 {
        self.height_px
    }

    pub const fn stride_bytes(self) -> u32 {
        self.stride_bytes
    }
}

pub struct BenchCapturedRgb<F> {
    evidence: BenchRgbFrameEvidence,
    frame: F,
}

impl<F> BenchCapturedRgb<F> {
    pub const fn evidence(&self) -> BenchRgbFrameEvidence {
        self.evidence
    }

    pub const fn frame(&self) -> &F {
        &self.frame
    }
}

pub struct BenchCapturedDepth<F> {
    evidence: BenchDepthReadinessEvidence,
    frame: F,
}

impl<F> BenchCapturedDepth<F> {
    pub const fn evidence(&self) -> BenchDepthReadinessEvidence {
        self.evidence
    }

    pub const fn frame(&self) -> &F {
        &self.frame
    }
}

pub struct BenchCapturedImu<B> {
    evidence: BenchImuReadinessEvidence,
    batch: B,
}

impl<B> BenchCapturedImu<B> {
    pub const fn evidence(&self) -> BenchImuReadinessEvidence {
        self.evidence
    }

    pub const fn batch(&self) -> &B {
        &self.batch
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchDepthReadinessEvidence {
    capture_sequence: u64,
    delivery_sequence: u64,
    device_timestamp_ns: i64,
    width_px: u32,
    height_px: u32,
}

impl BenchDepthReadinessEvidence {
    pub const fn capture_sequence(self) -> u64 {
        self.capture_sequence
    }

    pub const fn delivery_sequence(self) -> u64 {
        self.delivery_sequence
    }

    pub const fn device_timestamp_ns(self) -> i64 {
        self.device_timestamp_ns
    }

    pub const fn width_px(self) -> u32 {
        self.width_px
    }

    pub const fn height_px(self) -> u32 {
        self.height_px
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchImuReadinessEvidence {
    sample_count: NonZeroUsize,
    first_delivery_sequence: u32,
    last_delivery_sequence: u32,
    first_accel_timestamp_ns: i64,
    last_accel_timestamp_ns: i64,
    first_gyro_timestamp_ns: i64,
    last_gyro_timestamp_ns: i64,
}

impl BenchImuReadinessEvidence {
    pub const fn sample_count(self) -> NonZeroUsize {
        self.sample_count
    }

    pub const fn first_delivery_sequence(self) -> u32 {
        self.first_delivery_sequence
    }

    pub const fn last_delivery_sequence(self) -> u32 {
        self.last_delivery_sequence
    }

    pub const fn first_accel_timestamp_ns(self) -> i64 {
        self.first_accel_timestamp_ns
    }

    pub const fn last_accel_timestamp_ns(self) -> i64 {
        self.last_accel_timestamp_ns
    }

    pub const fn first_gyro_timestamp_ns(self) -> i64 {
        self.first_gyro_timestamp_ns
    }

    pub const fn last_gyro_timestamp_ns(self) -> i64 {
        self.last_gyro_timestamp_ns
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchObservationPair<T> {
    first: T,
    second: T,
}

impl<T: Copy> BenchObservationPair<T> {
    pub const fn first(self) -> T {
        self.first
    }

    pub const fn second(self) -> T {
        self.second
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchOakConnectionEvidence {
    requested_mxid: Box<str>,
    opened_mxid: Box<str>,
    discovery_transport_name: Option<Box<str>>,
    eeprom_device_name: Option<Box<str>>,
    product_name: Option<Box<str>>,
    usb_requested_maximum: UsbTransportSpeed,
    usb_required_minimum: UsbTransportSpeed,
    usb_observed: UsbTransportSpeed,
}

impl BenchOakConnectionEvidence {
    pub fn requested_mxid(&self) -> &str {
        &self.requested_mxid
    }

    pub fn opened_mxid(&self) -> &str {
        &self.opened_mxid
    }

    pub fn discovery_transport_name(&self) -> Option<&str> {
        self.discovery_transport_name.as_deref()
    }

    pub fn eeprom_device_name(&self) -> Option<&str> {
        self.eeprom_device_name.as_deref()
    }

    pub fn product_name(&self) -> Option<&str> {
        self.product_name.as_deref()
    }

    pub const fn usb_requested_maximum(&self) -> UsbTransportSpeed {
        self.usb_requested_maximum
    }

    pub const fn usb_required_minimum(&self) -> UsbTransportSpeed {
        self.usb_required_minimum
    }

    pub const fn usb_observed(&self) -> UsbTransportSpeed {
        self.usb_observed
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchCameraReadinessEvidence {
    depth: BenchObservationPair<BenchDepthReadinessEvidence>,
    imu: BenchObservationPair<BenchImuReadinessEvidence>,
    rgb: BenchObservationPair<BenchRgbFrameEvidence>,
}

impl BenchCameraReadinessEvidence {
    pub const fn depth(self) -> BenchObservationPair<BenchDepthReadinessEvidence> {
        self.depth
    }

    pub const fn imu(self) -> BenchObservationPair<BenchImuReadinessEvidence> {
        self.imu
    }

    pub const fn rgb(self) -> BenchObservationPair<BenchRgbFrameEvidence> {
        self.rgb
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchCameraContinuityError {
    DepthCaptureSequenceNotIncreasing { first: u64, second: u64 },
    DepthDeliverySequenceNotIncreasing { first: u64, second: u64 },
    DepthTimestampNotIncreasing { first_ns: i64, second_ns: i64 },
    RgbCaptureSequenceNotIncreasing { first: u64, second: u64 },
    RgbDeliverySequenceNotIncreasing { first: u64, second: u64 },
    RgbTimestampNotIncreasing { first_ns: i64, second_ns: i64 },
    ImuDeliverySequenceNotIncreasing { first: u32, second: u32 },
    ImuAccelTimestampNotIncreasing { first_ns: i64, second_ns: i64 },
    ImuGyroTimestampNotIncreasing { first_ns: i64, second_ns: i64 },
}

impl fmt::Display for BenchCameraContinuityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "OAK bench continuity check failed: {self:?}")
    }
}

impl std::error::Error for BenchCameraContinuityError {}

fn pair_depth(
    first: BenchDepthReadinessEvidence,
    second: BenchDepthReadinessEvidence,
) -> Result<BenchObservationPair<BenchDepthReadinessEvidence>, BenchCameraContinuityError> {
    if second.capture_sequence <= first.capture_sequence {
        return Err(
            BenchCameraContinuityError::DepthCaptureSequenceNotIncreasing {
                first: first.capture_sequence,
                second: second.capture_sequence,
            },
        );
    }
    if second.delivery_sequence <= first.delivery_sequence {
        return Err(
            BenchCameraContinuityError::DepthDeliverySequenceNotIncreasing {
                first: first.delivery_sequence,
                second: second.delivery_sequence,
            },
        );
    }
    if second.device_timestamp_ns <= first.device_timestamp_ns {
        return Err(BenchCameraContinuityError::DepthTimestampNotIncreasing {
            first_ns: first.device_timestamp_ns,
            second_ns: second.device_timestamp_ns,
        });
    }
    Ok(BenchObservationPair { first, second })
}

fn pair_rgb(
    first: BenchRgbFrameEvidence,
    second: BenchRgbFrameEvidence,
) -> Result<BenchObservationPair<BenchRgbFrameEvidence>, BenchCameraContinuityError> {
    if second.capture_sequence <= first.capture_sequence {
        return Err(
            BenchCameraContinuityError::RgbCaptureSequenceNotIncreasing {
                first: first.capture_sequence,
                second: second.capture_sequence,
            },
        );
    }
    if second.delivery_sequence <= first.delivery_sequence {
        return Err(
            BenchCameraContinuityError::RgbDeliverySequenceNotIncreasing {
                first: first.delivery_sequence,
                second: second.delivery_sequence,
            },
        );
    }
    if second.device_timestamp_ns <= first.device_timestamp_ns {
        return Err(BenchCameraContinuityError::RgbTimestampNotIncreasing {
            first_ns: first.device_timestamp_ns,
            second_ns: second.device_timestamp_ns,
        });
    }
    Ok(BenchObservationPair { first, second })
}

fn pair_imu(
    first: BenchImuReadinessEvidence,
    second: BenchImuReadinessEvidence,
) -> Result<BenchObservationPair<BenchImuReadinessEvidence>, BenchCameraContinuityError> {
    if second.first_delivery_sequence <= first.last_delivery_sequence {
        return Err(
            BenchCameraContinuityError::ImuDeliverySequenceNotIncreasing {
                first: first.last_delivery_sequence,
                second: second.first_delivery_sequence,
            },
        );
    }
    if second.first_accel_timestamp_ns <= first.last_accel_timestamp_ns {
        return Err(BenchCameraContinuityError::ImuAccelTimestampNotIncreasing {
            first_ns: first.last_accel_timestamp_ns,
            second_ns: second.first_accel_timestamp_ns,
        });
    }
    if second.first_gyro_timestamp_ns <= first.last_gyro_timestamp_ns {
        return Err(BenchCameraContinuityError::ImuGyroTimestampNotIncreasing {
            first_ns: first.last_gyro_timestamp_ns,
            second_ns: second.first_gyro_timestamp_ns,
        });
    }
    Ok(BenchObservationPair { first, second })
}

/// Camera boundary. Transient queue-empty/timeout results are `Ok(None)`;
/// disconnects, corrupt data, and contract failures remain typed errors.
pub trait WheelsOffOakPort {
    type RgbFrame;
    type DepthFrame;
    type ImuBatch;
    type Error: std::error::Error + Send + Sync + 'static;

    fn connect(
        &mut self,
        config: &WheelsOffBenchOakConfig,
    ) -> Result<BenchOakConnectionEvidence, Self::Error>;

    fn try_depth(
        &mut self,
        timeout_ms: NonZeroU32,
    ) -> Result<Option<BenchCapturedDepth<Self::DepthFrame>>, Self::Error>;

    fn try_imu(&mut self) -> Result<Option<BenchCapturedImu<Self::ImuBatch>>, Self::Error>;

    fn try_rgb(
        &mut self,
        timeout_ms: NonZeroU32,
    ) -> Result<Option<BenchCapturedRgb<Self::RgbFrame>>, Self::Error>;

    fn close(&mut self) -> Result<(), Self::Error>;
}

pub trait WheelsOffHeadPort {
    type StartupEvidence;
    type ShutdownEvidence;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Return evidence admitted by the head actor before it attempted any
    /// goal, torque-limit, or torque-enable write.
    fn configured_pose(evidence: &Self::StartupEvidence) -> ObservedPoseWithinConfiguredBounds;

    fn start(
        &mut self,
        config: HeadRuntimeConfig,
        configured_pose_bounds: WheelsOffConfiguredPoseBounds,
        consent: PhysicalTorqueEnableConsent,
    ) -> impl Future<Output = Result<Self::StartupEvidence, Self::Error>> + Send;

    fn shutdown(
        &mut self,
    ) -> impl Future<Output = Result<Self::ShutdownEvidence, Self::Error>> + Send;
}

pub trait WheelsOffEyePort<C> {
    type StartupEvidence;
    type ApplyEvidence;
    type ShutdownEvidence;
    type Error: std::error::Error + Send + Sync + 'static;

    fn start(
        &mut self,
        config: EyeRuntimeConfig,
        clock: C,
    ) -> impl Future<Output = Result<Self::StartupEvidence, Self::Error>> + Send;

    fn apply(
        &mut self,
        intent: PreparedEyeIntent,
    ) -> impl Future<Output = Result<Self::ApplyEvidence, Self::Error>> + Send;

    fn shutdown(
        &mut self,
    ) -> impl Future<Output = Result<Self::ShutdownEvidence, Self::Error>> + Send;
}

pub trait WheelsOffExpressionPort<F> {
    type Clock: Clone + Send + Sync + 'static;
    type Error: std::error::Error + Send + Sync + 'static;

    fn clone_clock_for_eye_actor(&self) -> Self::Clock;

    fn process(&mut self, frame: &F) -> Result<RgbExpressionBridgeOutcome, Self::Error>;
}

impl<C> WheelsOffExpressionPort<ImageFrame> for RgbExpressionBridge<C>
where
    C: EyeMonotonicClock + Clone,
{
    type Clock = C;
    type Error = RgbExpressionBridgeError;

    fn clone_clock_for_eye_actor(&self) -> Self::Clock {
        RgbExpressionBridge::clone_clock_for_eye_actor(self)
    }

    fn process(&mut self, frame: &ImageFrame) -> Result<RgbExpressionBridgeOutcome, Self::Error> {
        self.process_oak_frame(frame)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffBenchEvent {
    CameraConnected,
    CameraStreamsReady,
    HeadNaturalHoldReady,
    EyeReady,
    FirstRgbExpressionAdmitted,
    Running,
}

pub trait WheelsOffTelemetryPort<O>
where
    O: WheelsOffOakPort,
{
    type Error: std::error::Error + Send + Sync + 'static;

    fn event(&mut self, event: WheelsOffBenchEvent) -> Result<(), Self::Error>;

    fn camera_readiness(
        &mut self,
        connection: &BenchOakConnectionEvidence,
        readiness: BenchCameraReadinessEvidence,
    ) -> Result<(), Self::Error>;

    fn depth(
        &mut self,
        frame: &O::DepthFrame,
        evidence: BenchDepthReadinessEvidence,
    ) -> Result<(), Self::Error>;

    fn imu(
        &mut self,
        batch: &O::ImuBatch,
        evidence: BenchImuReadinessEvidence,
    ) -> Result<(), Self::Error>;

    fn rgb(
        &mut self,
        frame: &O::RgbFrame,
        evidence: BenchRgbFrameEvidence,
    ) -> Result<(), Self::Error>;

    fn flush(&mut self, timeout: Duration) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub enum WheelsOffBenchStartCause<
    BaseError,
    HeadError,
    EyeError,
    OakError,
    TelemetryError,
    ExpressionError,
> {
    BaseZero(WheelsOffBaseCleanupError<BaseError>),
    Oak(OakError),
    RequiredCameraEvidenceMissing {
        depth_observations: u8,
        imu_observations: u8,
        rgb_observations: u8,
        attempts: u16,
    },
    FreshExpressionRgbMissing {
        attempts: u16,
    },
    CameraContinuity(BenchCameraContinuityError),
    Telemetry(TelemetryError),
    Head(HeadError),
    Eye(EyeError),
    EyeSession(OsEyeSessionMaterialError),
    Expression(ExpressionError),
    Cancelled {
        signal: WheelsOffBenchCancellation,
        checkpoint: WheelsOffBenchCancellationCheckpoint,
    },
    AlreadyStarted,
}

#[derive(Debug)]
pub struct WheelsOffBenchCleanupReport<
    BaseEvidence,
    BaseDisarm,
    HeadShutdown,
    EyeShutdown,
    BaseError,
    HeadError,
    EyeError,
    OakError,
    TelemetryError,
> {
    pub base: Result<RefreshedBaseZero<BaseEvidence>, WheelsOffBaseCleanupError<BaseError>>,
    pub eye: Option<Result<EyeShutdown, EyeError>>,
    pub head: Option<Result<HeadShutdown, HeadError>>,
    pub oak: Option<Result<(), OakError>>,
    pub telemetry: Result<(), TelemetryError>,
    pub base_disarm: Result<BaseDisarm, BaseError>,
}

#[derive(Debug)]
pub struct WheelsOffBenchStartFailure<
    BaseEvidence,
    BaseDisarm,
    HeadShutdown,
    EyeShutdown,
    BaseError,
    HeadError,
    EyeError,
    OakError,
    TelemetryError,
    ExpressionError,
> {
    pub cause: WheelsOffBenchStartCause<
        BaseError,
        HeadError,
        EyeError,
        OakError,
        TelemetryError,
        ExpressionError,
    >,
    pub cleanup: WheelsOffBenchCleanupReport<
        BaseEvidence,
        BaseDisarm,
        HeadShutdown,
        EyeShutdown,
        BaseError,
        HeadError,
        EyeError,
        OakError,
        TelemetryError,
    >,
}

#[derive(Debug)]
pub struct WheelsOffBenchStartupEvidence<BaseZero, HeadStartup, EyeStartup, EyeApply> {
    pub before_oak: RefreshedBaseZero<BaseZero>,
    pub connection: BenchOakConnectionEvidence,
    pub camera: BenchCameraReadinessEvidence,
    pub before_head: RefreshedBaseZero<BaseZero>,
    pub head: HeadStartup,
    pub configured_head_pose: ObservedPoseWithinConfiguredBounds,
    pub before_eye: RefreshedBaseZero<BaseZero>,
    pub eye: EyeStartup,
    pub expression_rgb: BenchRgbFrameEvidence,
    pub first_expression: RgbExpressionBridgeOutcome,
    pub first_eye_admission: EyeApply,
}

/// Generic production ordering core. None of its public methods accepts a base
/// command, PWM, velocity, lease, or navigation target.
pub struct WheelsOffBenchRuntime<Base, Head, Eye, Oak, Telemetry, Expression> {
    plan: WheelsOffBenchPlan,
    base_cleanup: Base,
    last_zero: ConfirmedBaseZero,
    head: Head,
    eye: Eye,
    oak: Oak,
    telemetry: Telemetry,
    expression: Expression,
    head_started: bool,
    eye_started: bool,
    oak_connected: bool,
    started: bool,
    consecutive_rgb_misses: u16,
    last_rgb: Option<BenchRgbFrameEvidence>,
}

type RuntimeStartCause<Base, Head, Eye, Oak, Telemetry, Expression> = WheelsOffBenchStartCause<
    <Base as WheelsOffBaseCleanupPort>::Error,
    <Head as WheelsOffHeadPort>::Error,
    <Eye as WheelsOffEyePort<
        <Expression as WheelsOffExpressionPort<<Oak as WheelsOffOakPort>::RgbFrame>>::Clock,
    >>::Error,
    <Oak as WheelsOffOakPort>::Error,
    <Telemetry as WheelsOffTelemetryPort<Oak>>::Error,
    <Expression as WheelsOffExpressionPort<<Oak as WheelsOffOakPort>::RgbFrame>>::Error,
>;

type RuntimeCleanupReport<Base, Head, Eye, Oak, Telemetry, Expression> =
    WheelsOffBenchCleanupReport<
        <Base as WheelsOffBaseCleanupPort>::Evidence,
        <Base as WheelsOffBaseCleanupPort>::DisarmEvidence,
        <Head as WheelsOffHeadPort>::ShutdownEvidence,
        <Eye as WheelsOffEyePort<
            <Expression as WheelsOffExpressionPort<<Oak as WheelsOffOakPort>::RgbFrame>>::Clock,
        >>::ShutdownEvidence,
        <Base as WheelsOffBaseCleanupPort>::Error,
        <Head as WheelsOffHeadPort>::Error,
        <Eye as WheelsOffEyePort<
            <Expression as WheelsOffExpressionPort<<Oak as WheelsOffOakPort>::RgbFrame>>::Clock,
        >>::Error,
        <Oak as WheelsOffOakPort>::Error,
        <Telemetry as WheelsOffTelemetryPort<Oak>>::Error,
    >;

type RuntimeStartFailure<Base, Head, Eye, Oak, Telemetry, Expression> = WheelsOffBenchStartFailure<
    <Base as WheelsOffBaseCleanupPort>::Evidence,
    <Base as WheelsOffBaseCleanupPort>::DisarmEvidence,
    <Head as WheelsOffHeadPort>::ShutdownEvidence,
    <Eye as WheelsOffEyePort<
        <Expression as WheelsOffExpressionPort<<Oak as WheelsOffOakPort>::RgbFrame>>::Clock,
    >>::ShutdownEvidence,
    <Base as WheelsOffBaseCleanupPort>::Error,
    <Head as WheelsOffHeadPort>::Error,
    <Eye as WheelsOffEyePort<
        <Expression as WheelsOffExpressionPort<<Oak as WheelsOffOakPort>::RgbFrame>>::Clock,
    >>::Error,
    <Oak as WheelsOffOakPort>::Error,
    <Telemetry as WheelsOffTelemetryPort<Oak>>::Error,
    <Expression as WheelsOffExpressionPort<<Oak as WheelsOffOakPort>::RgbFrame>>::Error,
>;

impl<Base, Head, Eye, Oak, Telemetry, Expression>
    WheelsOffBenchRuntime<Base, Head, Eye, Oak, Telemetry, Expression>
where
    Base: WheelsOffBaseCleanupPort,
    Head: WheelsOffHeadPort,
    Expression: WheelsOffExpressionPort<Oak::RgbFrame>,
    Eye: WheelsOffEyePort<Expression::Clock>,
    Oak: WheelsOffOakPort,
    Telemetry: WheelsOffTelemetryPort<Oak>,
{
    pub fn new(
        plan: WheelsOffBenchPlan,
        base_cleanup: Base,
        head: Head,
        eye: Eye,
        oak: Oak,
        telemetry: Telemetry,
        expression: Expression,
    ) -> Self {
        let last_zero = plan.base().zero();
        Self {
            plan,
            base_cleanup,
            last_zero,
            head,
            eye,
            oak,
            telemetry,
            expression,
            head_started: false,
            eye_started: false,
            oak_connected: false,
            started: false,
            consecutive_rgb_misses: 0,
            last_rgb: None,
        }
    }

    pub const fn base_admission(&self) -> &WheelsOffBaseAdmission {
        self.plan.base()
    }

    pub async fn start<Cancellation>(
        &mut self,
        cancellation: &mut Cancellation,
    ) -> Result<
        WheelsOffBenchStartupEvidence<
            Base::Evidence,
            Head::StartupEvidence,
            Eye::StartupEvidence,
            Eye::ApplyEvidence,
        >,
        RuntimeStartFailure<Base, Head, Eye, Oak, Telemetry, Expression>,
    >
    where
        Cancellation: WheelsOffBenchCancellationPort,
    {
        if self.started || self.head_started || self.eye_started || self.oak_connected {
            return Err(self.failed(WheelsOffBenchStartCause::AlreadyStarted).await);
        }
        if let Some(failure) = self
            .cancelled(
                cancellation,
                WheelsOffBenchCancellationCheckpoint::BeforeInitialBaseZero,
            )
            .await
        {
            return Err(failure);
        }

        // Prove camera viability before energising the head. The zero-only
        // keeper remains active throughout the bounded camera waits.
        let before_oak = match self.refresh_base_zero().await {
            Ok(evidence) => evidence,
            Err(source) => {
                return Err(self
                    .failed(WheelsOffBenchStartCause::BaseZero(source))
                    .await);
            }
        };
        if let Some(failure) = self
            .cancelled(
                cancellation,
                WheelsOffBenchCancellationCheckpoint::BeforeCameraConnect,
            )
            .await
        {
            return Err(failure);
        }
        let connection = match self.oak.connect(self.plan.oak()) {
            Ok(evidence) => evidence,
            Err(source) => return Err(self.failed(WheelsOffBenchStartCause::Oak(source)).await),
        };
        self.oak_connected = true;
        if let Err(source) = self.telemetry.event(WheelsOffBenchEvent::CameraConnected) {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }

        let capture = self.plan.capture();
        let mut depth_first: Option<BenchCapturedDepth<Oak::DepthFrame>> = None;
        let mut depth_second: Option<BenchCapturedDepth<Oak::DepthFrame>> = None;
        let mut depth_pair = None;
        let mut imu_first: Option<BenchCapturedImu<Oak::ImuBatch>> = None;
        let mut imu_second: Option<BenchCapturedImu<Oak::ImuBatch>> = None;
        let mut imu_pair = None;
        let mut rgb_first: Option<BenchCapturedRgb<Oak::RgbFrame>> = None;
        let mut rgb_second: Option<BenchCapturedRgb<Oak::RgbFrame>> = None;
        let mut rgb_pair = None;
        for attempt_index in 0..capture.attempts().get() {
            let attempt = attempt_index + 1;
            if depth_pair.is_none() {
                if let Some(failure) = self
                    .cancelled(
                        cancellation,
                        WheelsOffBenchCancellationCheckpoint::BeforeDepthCapture { attempt },
                    )
                    .await
                {
                    return Err(failure);
                }
                let sample = match self.oak.try_depth(capture.timeout_ms()) {
                    Ok(value) => value,
                    Err(source) => {
                        return Err(self.failed(WheelsOffBenchStartCause::Oak(source)).await);
                    }
                };
                if let Some(sample) = sample {
                    if let Some(first) = depth_first.as_ref() {
                        depth_pair = match pair_depth(first.evidence(), sample.evidence()) {
                            Ok(pair) => Some(pair),
                            Err(source) => {
                                return Err(self
                                    .failed(WheelsOffBenchStartCause::CameraContinuity(source))
                                    .await);
                            }
                        };
                        depth_second = Some(sample);
                    } else {
                        depth_first = Some(sample);
                    }
                }
            }
            if imu_pair.is_none() {
                if let Some(failure) = self
                    .cancelled(
                        cancellation,
                        WheelsOffBenchCancellationCheckpoint::BeforeImuCapture { attempt },
                    )
                    .await
                {
                    return Err(failure);
                }
                let sample = match self.oak.try_imu() {
                    Ok(value) => value,
                    Err(source) => {
                        return Err(self.failed(WheelsOffBenchStartCause::Oak(source)).await);
                    }
                };
                if let Some(sample) = sample {
                    if let Some(first) = imu_first.as_ref() {
                        imu_pair = match pair_imu(first.evidence(), sample.evidence()) {
                            Ok(pair) => Some(pair),
                            Err(source) => {
                                return Err(self
                                    .failed(WheelsOffBenchStartCause::CameraContinuity(source))
                                    .await);
                            }
                        };
                        imu_second = Some(sample);
                    } else {
                        imu_first = Some(sample);
                    }
                }
            }
            if rgb_pair.is_none() {
                if let Some(failure) = self
                    .cancelled(
                        cancellation,
                        WheelsOffBenchCancellationCheckpoint::BeforeRgbCapture { attempt },
                    )
                    .await
                {
                    return Err(failure);
                }
                let sample = match self.oak.try_rgb(capture.timeout_ms()) {
                    Ok(value) => value,
                    Err(source) => {
                        return Err(self.failed(WheelsOffBenchStartCause::Oak(source)).await);
                    }
                };
                if let Some(sample) = sample {
                    if let Some(first) = rgb_first.as_ref() {
                        rgb_pair = match pair_rgb(first.evidence(), sample.evidence()) {
                            Ok(pair) => Some(pair),
                            Err(source) => {
                                return Err(self
                                    .failed(WheelsOffBenchStartCause::CameraContinuity(source))
                                    .await);
                            }
                        };
                        rgb_second = Some(sample);
                    } else {
                        rgb_first = Some(sample);
                    }
                }
            }
            if depth_pair.is_some() && imu_pair.is_some() && rgb_pair.is_some() {
                break;
            }
        }
        let depth_observations = u8::from(depth_first.is_some()) + u8::from(depth_pair.is_some());
        let imu_observations = u8::from(imu_first.is_some()) + u8::from(imu_pair.is_some());
        let rgb_observations = u8::from(rgb_first.is_some()) + u8::from(rgb_pair.is_some());
        let (
            Some(depth),
            Some(depth_first),
            Some(depth_second),
            Some(imu),
            Some(imu_first),
            Some(imu_second),
            Some(rgb_evidence),
            Some(rgb_first),
            Some(rgb),
        ) = (
            depth_pair,
            depth_first,
            depth_second,
            imu_pair,
            imu_first,
            imu_second,
            rgb_pair,
            rgb_first,
            rgb_second,
        )
        else {
            return Err(self
                .failed(WheelsOffBenchStartCause::RequiredCameraEvidenceMissing {
                    depth_observations,
                    imu_observations,
                    rgb_observations,
                    attempts: capture.attempts().get(),
                })
                .await);
        };
        let camera = BenchCameraReadinessEvidence {
            depth,
            imu,
            rgb: rgb_evidence,
        };
        if let Err(source) = self.telemetry.camera_readiness(&connection, camera) {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }
        if let Err(source) = self
            .telemetry
            .depth(depth_first.frame(), depth_first.evidence())
        {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }
        if let Err(source) = self
            .telemetry
            .depth(depth_second.frame(), depth_second.evidence())
        {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }
        if let Err(source) = self.telemetry.imu(imu_first.batch(), imu_first.evidence()) {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }
        if let Err(source) = self
            .telemetry
            .imu(imu_second.batch(), imu_second.evidence())
        {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }
        if let Err(source) = self.telemetry.rgb(rgb_first.frame(), rgb_first.evidence()) {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }
        if let Err(source) = self.telemetry.rgb(rgb.frame(), rgb.evidence()) {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }
        if let Err(source) = self
            .telemetry
            .event(WheelsOffBenchEvent::CameraStreamsReady)
        {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }

        if let Some(failure) = self
            .cancelled(
                cancellation,
                WheelsOffBenchCancellationCheckpoint::BeforeHeadBaseZero,
            )
            .await
        {
            return Err(failure);
        }
        let before_head = match self.refresh_base_zero().await {
            Ok(evidence) => evidence,
            Err(source) => {
                return Err(self
                    .failed(WheelsOffBenchStartCause::BaseZero(source))
                    .await);
            }
        };
        if let Some(failure) = self
            .cancelled(
                cancellation,
                WheelsOffBenchCancellationCheckpoint::BeforeHeadStart,
            )
            .await
        {
            return Err(failure);
        }
        let head = match self
            .head
            .start(
                self.plan.head.clone(),
                self.plan.configured_pose_bounds(),
                self.plan.head_consent,
            )
            .await
        {
            Ok(evidence) => evidence,
            Err(source) => return Err(self.failed(WheelsOffBenchStartCause::Head(source)).await),
        };
        self.head_started = true;
        if let Some(failure) = self
            .cancelled(
                cancellation,
                WheelsOffBenchCancellationCheckpoint::AfterHeadStart,
            )
            .await
        {
            return Err(failure);
        }
        let configured_head_pose = Head::configured_pose(&head);
        if let Err(source) = self
            .telemetry
            .event(WheelsOffBenchEvent::HeadNaturalHoldReady)
        {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }

        if let Some(failure) = self
            .cancelled(
                cancellation,
                WheelsOffBenchCancellationCheckpoint::BeforeEyeSession,
            )
            .await
        {
            return Err(failure);
        }
        let mut session_generator = OsEyeSessionMaterialGenerator;
        let eye_config = match self.plan.eye().new_session(&mut session_generator) {
            Ok(config) => config,
            Err(source) => {
                return Err(self
                    .failed(WheelsOffBenchStartCause::EyeSession(source))
                    .await);
            }
        };
        let eye_clock = self.expression.clone_clock_for_eye_actor();
        if let Some(failure) = self
            .cancelled(
                cancellation,
                WheelsOffBenchCancellationCheckpoint::BeforeEyeBaseZero,
            )
            .await
        {
            return Err(failure);
        }
        let before_eye = match self.refresh_base_zero().await {
            Ok(evidence) => evidence,
            Err(source) => {
                return Err(self
                    .failed(WheelsOffBenchStartCause::BaseZero(source))
                    .await);
            }
        };
        if let Some(failure) = self
            .cancelled(
                cancellation,
                WheelsOffBenchCancellationCheckpoint::BeforeEyeStart,
            )
            .await
        {
            return Err(failure);
        }
        let eye = match self.eye.start(eye_config, eye_clock).await {
            Ok(evidence) => evidence,
            Err(source) => return Err(self.failed(WheelsOffBenchStartCause::Eye(source)).await),
        };
        self.eye_started = true;
        if let Some(failure) = self
            .cancelled(
                cancellation,
                WheelsOffBenchCancellationCheckpoint::AfterEyeStart,
            )
            .await
        {
            return Err(failure);
        }
        if let Err(source) = self.telemetry.event(WheelsOffBenchEvent::EyeReady) {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }

        let mut expression_rgb = None;
        for attempt_index in 0..capture.attempts().get() {
            let attempt = attempt_index + 1;
            if let Some(failure) = self
                .cancelled(
                    cancellation,
                    WheelsOffBenchCancellationCheckpoint::BeforeExpressionRgbCapture { attempt },
                )
                .await
            {
                return Err(failure);
            }
            let sample = match self.oak.try_rgb(capture.timeout_ms()) {
                Ok(value) => value,
                Err(source) => {
                    return Err(self.failed(WheelsOffBenchStartCause::Oak(source)).await);
                }
            };
            let Some(sample) = sample else {
                continue;
            };
            if let Err(source) = pair_rgb(rgb.evidence(), sample.evidence()) {
                return Err(self
                    .failed(WheelsOffBenchStartCause::CameraContinuity(source))
                    .await);
            }
            expression_rgb = Some(sample);
            break;
        }
        let Some(expression_rgb) = expression_rgb else {
            return Err(self
                .failed(WheelsOffBenchStartCause::FreshExpressionRgbMissing {
                    attempts: capture.attempts().get(),
                })
                .await);
        };
        let first_expression = match self.expression.process(expression_rgb.frame()) {
            Ok(outcome) => outcome,
            Err(source) => {
                return Err(self
                    .failed(WheelsOffBenchStartCause::Expression(source))
                    .await);
            }
        };
        if let Err(source) = self
            .telemetry
            .rgb(expression_rgb.frame(), expression_rgb.evidence())
        {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }
        if let Some(failure) = self
            .cancelled(
                cancellation,
                WheelsOffBenchCancellationCheckpoint::BeforeFirstEyeApply,
            )
            .await
        {
            return Err(failure);
        }
        let first_eye_admission = match self.eye.apply(first_expression.into_prepared()).await {
            Ok(evidence) => evidence,
            Err(source) => return Err(self.failed(WheelsOffBenchStartCause::Eye(source)).await),
        };
        if let Err(source) = self
            .telemetry
            .event(WheelsOffBenchEvent::FirstRgbExpressionAdmitted)
        {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }
        if let Some(failure) = self
            .cancelled(
                cancellation,
                WheelsOffBenchCancellationCheckpoint::BeforeRunning,
            )
            .await
        {
            return Err(failure);
        }
        if let Err(source) = self.telemetry.event(WheelsOffBenchEvent::Running) {
            return Err(self
                .failed(WheelsOffBenchStartCause::Telemetry(source))
                .await);
        }
        self.last_rgb = Some(expression_rgb.evidence());
        self.started = true;
        Ok(WheelsOffBenchStartupEvidence {
            before_oak,
            connection,
            camera,
            before_head,
            head,
            configured_head_pose,
            before_eye,
            eye,
            expression_rgb: expression_rgb.evidence(),
            first_expression,
            first_eye_admission,
        })
    }

    pub async fn process_next_rgb(
        &mut self,
    ) -> Result<
        Option<(
            BenchRgbFrameEvidence,
            RgbExpressionBridgeOutcome,
            Eye::ApplyEvidence,
        )>,
        WheelsOffBenchCycleError<
            Base::Error,
            Oak::Error,
            Telemetry::Error,
            Expression::Error,
            Eye::Error,
        >,
    > {
        if !self.started {
            return Err(WheelsOffBenchCycleError::NotRunning);
        }
        self.base_cleanup
            .check_health()
            .await
            .map_err(WheelsOffBenchCycleError::BaseHealth)?;
        let Some(rgb) = self
            .oak
            .try_rgb(self.plan.capture().timeout_ms())
            .map_err(WheelsOffBenchCycleError::Oak)?
        else {
            self.consecutive_rgb_misses = self.consecutive_rgb_misses.saturating_add(1);
            let maximum_misses = self.plan.capture().attempts().get();
            if self.consecutive_rgb_misses >= maximum_misses {
                return Err(WheelsOffBenchCycleError::RgbLivenessLost {
                    consecutive_misses: self.consecutive_rgb_misses,
                    maximum_misses,
                    per_attempt_timeout_ms: self.plan.capture().timeout_ms().get(),
                });
            }
            return Ok(None);
        };
        self.consecutive_rgb_misses = 0;
        let previous_rgb = self
            .last_rgb
            .ok_or(WheelsOffBenchCycleError::MissingRgbContinuityAnchor)?;
        pair_rgb(previous_rgb, rgb.evidence())
            .map_err(WheelsOffBenchCycleError::CameraContinuity)?;
        let outcome = self
            .expression
            .process(rgb.frame())
            .map_err(WheelsOffBenchCycleError::Expression)?;
        self.telemetry
            .rgb(rgb.frame(), rgb.evidence())
            .map_err(WheelsOffBenchCycleError::Telemetry)?;
        let admission = self
            .eye
            .apply(outcome.into_prepared())
            .await
            .map_err(WheelsOffBenchCycleError::Eye)?;
        self.last_rgb = Some(rgb.evidence());
        Ok(Some((rgb.evidence(), outcome, admission)))
    }

    pub async fn shutdown(
        &mut self,
    ) -> RuntimeCleanupReport<Base, Head, Eye, Oak, Telemetry, Expression> {
        self.started = false;
        self.consecutive_rgb_misses = 0;
        self.last_rgb = None;
        self.cleanup().await
    }

    async fn failed(
        &mut self,
        cause: RuntimeStartCause<Base, Head, Eye, Oak, Telemetry, Expression>,
    ) -> RuntimeStartFailure<Base, Head, Eye, Oak, Telemetry, Expression> {
        WheelsOffBenchStartFailure {
            cause,
            cleanup: self.cleanup().await,
        }
    }

    async fn cancelled<Cancellation>(
        &mut self,
        cancellation: &mut Cancellation,
        checkpoint: WheelsOffBenchCancellationCheckpoint,
    ) -> Option<RuntimeStartFailure<Base, Head, Eye, Oak, Telemetry, Expression>>
    where
        Cancellation: WheelsOffBenchCancellationPort,
    {
        let signal = cancellation.poll_cancellation()?;
        Some(
            self.failed(WheelsOffBenchStartCause::Cancelled { signal, checkpoint })
                .await,
        )
    }

    async fn cleanup(
        &mut self,
    ) -> RuntimeCleanupReport<Base, Head, Eye, Oak, Telemetry, Expression> {
        let base = self.refresh_base_zero().await;
        let eye = if self.eye_started {
            self.eye_started = false;
            Some(self.eye.shutdown().await)
        } else {
            None
        };
        let head = if self.head_started {
            self.head_started = false;
            Some(self.head.shutdown().await)
        } else {
            None
        };
        let oak = if self.oak_connected {
            self.oak_connected = false;
            Some(self.oak.close())
        } else {
            None
        };
        let telemetry = self.telemetry.flush(self.plan.rerun_flush_timeout());
        let base_disarm = self.base_cleanup.disarm().await;
        WheelsOffBenchCleanupReport {
            base,
            eye,
            head,
            oak,
            telemetry,
            base_disarm,
        }
    }

    async fn refresh_base_zero(
        &mut self,
    ) -> Result<RefreshedBaseZero<Base::Evidence>, WheelsOffBaseCleanupError<Base::Error>> {
        let refreshed = self
            .base_cleanup
            .refresh_zero()
            .await
            .map_err(WheelsOffBaseCleanupError::Port)?;
        let actual = refreshed.confirmed();
        require_new_cleanup_zero(self.plan.base().readiness(), self.last_zero, actual)?;
        self.last_zero = actual;
        Ok(refreshed)
    }
}

#[derive(Debug)]
pub enum WheelsOffBenchCycleError<BaseError, OakError, TelemetryError, ExpressionError, EyeError> {
    NotRunning,
    BaseHealth(BaseError),
    MissingRgbContinuityAnchor,
    RgbLivenessLost {
        consecutive_misses: u16,
        maximum_misses: u16,
        per_attempt_timeout_ms: u32,
    },
    CameraContinuity(BenchCameraContinuityError),
    Oak(OakError),
    Telemetry(TelemetryError),
    Expression(ExpressionError),
    Eye(EyeError),
}

/// Production OAK owner. Construction performs no I/O; `connect` opens only
/// the exact MXID from the already-bound OAK pipeline.
#[derive(Default)]
pub struct NativeWheelsOffOakPort {
    device: Option<Device>,
}

#[derive(Debug)]
pub enum NativeWheelsOffOakError {
    AlreadyConnected,
    NotConnected,
    Connect(ConnectionError),
    ConnectedIdentity(ConnectedDeviceIdentityError),
    UsbTransportEvidence(UsbTransportEvidenceError),
    Rgb(ImageError),
    Depth(DepthError),
    Imu(ImuError),
    EmptyImuBatch,
    Close(oak_sys::CloseError),
}

impl fmt::Display for NativeWheelsOffOakError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "native wheels-off OAK operation failed: {self:?}"
        )
    }
}

impl std::error::Error for NativeWheelsOffOakError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Connect(source) => Some(source),
            Self::ConnectedIdentity(source) => Some(source),
            Self::UsbTransportEvidence(source) => Some(source),
            Self::Rgb(source) => Some(source),
            Self::Depth(source) => Some(source),
            Self::Imu(source) => Some(source),
            Self::Close(source) => Some(source),
            Self::AlreadyConnected | Self::NotConnected | Self::EmptyImuBatch => None,
        }
    }
}

impl WheelsOffOakPort for NativeWheelsOffOakPort {
    type RgbFrame = ImageFrame;
    type DepthFrame = DepthFrame;
    type ImuBatch = Vec<ImuSample>;
    type Error = NativeWheelsOffOakError;

    fn connect(
        &mut self,
        config: &WheelsOffBenchOakConfig,
    ) -> Result<BenchOakConnectionEvidence, Self::Error> {
        if self.device.is_some() {
            return Err(NativeWheelsOffOakError::AlreadyConnected);
        }
        let requested_mxid = config.expected_mxid();
        let device = Device::connect(requested_mxid, config.device().clone())
            .map_err(NativeWheelsOffOakError::Connect)?;
        let identity = device
            .connected_identity()
            .map_err(NativeWheelsOffOakError::ConnectedIdentity)?;
        let usb = device
            .usb_transport_evidence()
            .map_err(NativeWheelsOffOakError::UsbTransportEvidence)?;
        let evidence = BenchOakConnectionEvidence {
            requested_mxid: requested_mxid.into(),
            opened_mxid: identity.mxid().into(),
            discovery_transport_name: identity.discovery_transport_name().map(Into::into),
            eeprom_device_name: identity.eeprom_device_name().map(Into::into),
            product_name: identity.product_name().map(Into::into),
            usb_requested_maximum: usb.requested_maximum(),
            usb_required_minimum: usb.required_minimum(),
            usb_observed: usb.observed(),
        };
        self.device = Some(device);
        Ok(evidence)
    }

    fn try_depth(
        &mut self,
        timeout_ms: NonZeroU32,
    ) -> Result<Option<BenchCapturedDepth<Self::DepthFrame>>, Self::Error> {
        let device = self
            .device
            .as_mut()
            .ok_or(NativeWheelsOffOakError::NotConnected)?;
        let frame = match device.depth(timeout_ms.get()) {
            Ok(frame) => frame,
            Err(DepthError::Timeout { .. } | DepthError::QueueEmpty) => return Ok(None),
            Err(source) => return Err(NativeWheelsOffOakError::Depth(source)),
        };
        let evidence = BenchDepthReadinessEvidence {
            capture_sequence: frame.device_capture_sequence.as_u64(),
            delivery_sequence: frame.host_delivery_sequence.as_u64(),
            device_timestamp_ns: frame.timestamp.as_nanos(),
            width_px: frame.width,
            height_px: frame.height,
        };
        Ok(Some(BenchCapturedDepth { evidence, frame }))
    }

    fn try_imu(&mut self) -> Result<Option<BenchCapturedImu<Self::ImuBatch>>, Self::Error> {
        let device = self
            .device
            .as_mut()
            .ok_or(NativeWheelsOffOakError::NotConnected)?;
        let samples = match device.imu() {
            Ok(samples) => samples,
            Err(ImuError::Empty) => return Ok(None),
            Err(source) => return Err(NativeWheelsOffOakError::Imu(source)),
        };
        let sample_count =
            NonZeroUsize::new(samples.len()).ok_or(NativeWheelsOffOakError::EmptyImuBatch)?;
        let first = samples
            .first()
            .expect("nonzero sample count proves a first IMU sample");
        let last = samples
            .last()
            .expect("nonzero sample count proves a last IMU sample");
        if sample_count.get() > 1
            && (last.sequence <= first.sequence
                || last.accel_timestamp <= first.accel_timestamp
                || last.gyro_timestamp <= first.gyro_timestamp)
        {
            return Err(NativeWheelsOffOakError::Imu(ImuError::Corrupt));
        }
        let evidence = BenchImuReadinessEvidence {
            sample_count,
            first_delivery_sequence: first.sequence,
            last_delivery_sequence: last.sequence,
            first_accel_timestamp_ns: first.accel_timestamp.as_nanos(),
            last_accel_timestamp_ns: last.accel_timestamp.as_nanos(),
            first_gyro_timestamp_ns: first.gyro_timestamp.as_nanos(),
            last_gyro_timestamp_ns: last.gyro_timestamp.as_nanos(),
        };
        Ok(Some(BenchCapturedImu {
            evidence,
            batch: samples,
        }))
    }

    fn try_rgb(
        &mut self,
        timeout_ms: NonZeroU32,
    ) -> Result<Option<BenchCapturedRgb<Self::RgbFrame>>, Self::Error> {
        let device = self
            .device
            .as_mut()
            .ok_or(NativeWheelsOffOakError::NotConnected)?;
        let frame = match device.rgb(timeout_ms.get()) {
            Ok(frame) => frame,
            Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => return Ok(None),
            Err(source) => return Err(NativeWheelsOffOakError::Rgb(source)),
        };
        if frame.stream != StreamId::Rgb {
            return Err(NativeWheelsOffOakError::Rgb(ImageError::Corrupt));
        }
        let evidence = BenchRgbFrameEvidence {
            capture_sequence: frame.device_capture_sequence.as_u64(),
            delivery_sequence: frame.host_delivery_sequence.as_u64(),
            device_timestamp_ns: frame.timestamp.as_nanos(),
            width_px: frame.width,
            height_px: frame.height,
            stride_bytes: frame.stride_bytes,
        };
        Ok(Some(BenchCapturedRgb { evidence, frame }))
    }

    fn close(&mut self) -> Result<(), Self::Error> {
        let device = self
            .device
            .take()
            .ok_or(NativeWheelsOffOakError::NotConnected)?;
        device.close().map_err(NativeWheelsOffOakError::Close)
    }
}

/// Rerun adapter for the native tightly-packed BGR OAK stream.
pub struct RerunWheelsOffTelemetry {
    recording: rerun::RecordingStream,
}

impl RerunWheelsOffTelemetry {
    pub fn new(recording: rerun::RecordingStream) -> Self {
        Self { recording }
    }
}

#[derive(Debug)]
pub enum RerunWheelsOffError {
    Recording(rerun::RecordingStreamError),
    Flush(rerun::sink::SinkFlushError),
    TimestampNotExactlyRepresentable { source_ns: i64, encoded_ns: i64 },
    RgbEvidenceMismatch,
    DepthEvidenceMismatch,
    ImuEvidenceMismatch,
    DepthPayloadSizeOverflow,
}

impl fmt::Display for RerunWheelsOffError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "wheels-off Rerun output failed: {self:?}")
    }
}

impl std::error::Error for RerunWheelsOffError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Recording(source) => Some(source),
            Self::Flush(source) => Some(source),
            Self::TimestampNotExactlyRepresentable { .. }
            | Self::RgbEvidenceMismatch
            | Self::DepthEvidenceMismatch
            | Self::ImuEvidenceMismatch
            | Self::DepthPayloadSizeOverflow => None,
        }
    }
}

impl WheelsOffTelemetryPort<NativeWheelsOffOakPort> for RerunWheelsOffTelemetry {
    type Error = RerunWheelsOffError;

    fn event(&mut self, event: WheelsOffBenchEvent) -> Result<(), Self::Error> {
        self.recording
            .log(
                "bench/lifecycle",
                &rerun::TextLog::new(format!("{event:?}")),
            )
            .map_err(RerunWheelsOffError::Recording)
    }

    fn camera_readiness(
        &mut self,
        connection: &BenchOakConnectionEvidence,
        readiness: BenchCameraReadinessEvidence,
    ) -> Result<(), Self::Error> {
        self.recording
            .log_static(
                "bench/camera/opened_mxid",
                &rerun::TextLog::new(connection.opened_mxid()),
            )
            .map_err(RerunWheelsOffError::Recording)?;
        self.recording
            .log_static(
                "bench/camera/usb_transport",
                &rerun::TextLog::new(format!(
                    "requested_maximum={} required_minimum={} observed={}",
                    connection.usb_requested_maximum(),
                    connection.usb_required_minimum(),
                    connection.usb_observed(),
                )),
            )
            .map_err(RerunWheelsOffError::Recording)?;
        self.recording
            .log_static(
                "bench/camera/depth/contract",
                &rerun::TextLog::new(
                    "rectified-left optical frame; uint16 millimetres; 0 is invalid",
                ),
            )
            .map_err(RerunWheelsOffError::Recording)?;
        self.recording
            .log_static(
                "bench/camera/imu/contract",
                &rerun::TextLog::new(
                    "raw sensor-native frame; accelerometer m/s^2; gyroscope rad/s; not calibrated or transformed to base frame",
                ),
            )
            .map_err(RerunWheelsOffError::Recording)?;
        self.recording
            .log(
                "bench/camera/depth/continuity",
                &rerun::TextLog::new(format!(
                    "capture_sequence={}..{} device_timestamp_ns={}..{}",
                    readiness.depth().first().capture_sequence(),
                    readiness.depth().second().capture_sequence(),
                    readiness.depth().first().device_timestamp_ns(),
                    readiness.depth().second().device_timestamp_ns(),
                )),
            )
            .map_err(RerunWheelsOffError::Recording)?;
        self.recording
            .log(
                "bench/camera/imu/continuity",
                &rerun::TextLog::new(format!(
                    "sample_count={}+{} delivery_sequence={}..{} accel_timestamp_ns={}..{} gyro_timestamp_ns={}..{}",
                    readiness.imu().first().sample_count(),
                    readiness.imu().second().sample_count(),
                    readiness.imu().first().first_delivery_sequence(),
                    readiness.imu().second().last_delivery_sequence(),
                    readiness.imu().first().first_accel_timestamp_ns(),
                    readiness.imu().second().last_accel_timestamp_ns(),
                    readiness.imu().first().first_gyro_timestamp_ns(),
                    readiness.imu().second().last_gyro_timestamp_ns(),
                )),
            )
            .map_err(RerunWheelsOffError::Recording)
    }

    fn depth(
        &mut self,
        frame: &DepthFrame,
        evidence: BenchDepthReadinessEvidence,
    ) -> Result<(), Self::Error> {
        if frame.device_capture_sequence.as_u64() != evidence.capture_sequence()
            || frame.host_delivery_sequence.as_u64() != evidence.delivery_sequence()
            || frame.timestamp.as_nanos() != evidence.device_timestamp_ns()
            || frame.width != evidence.width_px()
            || frame.height != evidence.height_px()
            || frame.connected_alignment() != Some(DepthAlignment::RectifiedLeft)
        {
            return Err(RerunWheelsOffError::DepthEvidenceMismatch);
        }
        let byte_capacity = frame
            .depth_mm()
            .len()
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or(RerunWheelsOffError::DepthPayloadSizeOverflow)?;
        let mut bytes = Vec::with_capacity(byte_capacity);
        for value_mm in frame.depth_mm() {
            bytes.extend_from_slice(&value_mm.to_le_bytes());
        }
        let time = exact_rerun_device_time(evidence.device_timestamp_ns())?;
        self.recording.set_time("oak_device_time_ns", time);
        let image =
            rerun::DepthImage::from_gray16(bytes, [frame.width, frame.height]).with_meter(1_000.0);
        let logged = self
            .recording
            .log("bench/camera/depth/rectified_left_mm", &image)
            .map_err(RerunWheelsOffError::Recording);
        self.recording.disable_timeline("oak_device_time_ns");
        logged
    }

    fn imu(
        &mut self,
        samples: &Vec<ImuSample>,
        evidence: BenchImuReadinessEvidence,
    ) -> Result<(), Self::Error> {
        let Some((first, last)) = samples.first().zip(samples.last()) else {
            return Err(RerunWheelsOffError::ImuEvidenceMismatch);
        };
        if samples.len() != evidence.sample_count().get()
            || first.sequence != evidence.first_delivery_sequence()
            || last.sequence != evidence.last_delivery_sequence()
            || first.accel_timestamp.as_nanos() != evidence.first_accel_timestamp_ns()
            || last.accel_timestamp.as_nanos() != evidence.last_accel_timestamp_ns()
            || first.gyro_timestamp.as_nanos() != evidence.first_gyro_timestamp_ns()
            || last.gyro_timestamp.as_nanos() != evidence.last_gyro_timestamp_ns()
        {
            return Err(RerunWheelsOffError::ImuEvidenceMismatch);
        }
        for sample in samples {
            let accel_time = exact_rerun_device_time(sample.accel_timestamp.as_nanos())?;
            self.recording.set_time("oak_device_time_ns", accel_time);
            self.recording
                .log(
                    "bench/camera/imu/sensor_native/accel_m_s2_xyz",
                    &rerun::Scalars::new(sample.accel.as_array().into_iter().map(f64::from)),
                )
                .map_err(RerunWheelsOffError::Recording)?;

            let gyro_time = exact_rerun_device_time(sample.gyro_timestamp.as_nanos())?;
            self.recording.set_time("oak_device_time_ns", gyro_time);
            self.recording
                .log(
                    "bench/camera/imu/sensor_native/gyro_rad_s_xyz",
                    &rerun::Scalars::new(sample.gyro.as_array().into_iter().map(f64::from)),
                )
                .map_err(RerunWheelsOffError::Recording)?;
        }
        self.recording.disable_timeline("oak_device_time_ns");
        Ok(())
    }

    fn rgb(
        &mut self,
        frame: &ImageFrame,
        evidence: BenchRgbFrameEvidence,
    ) -> Result<(), Self::Error> {
        if frame.stream != StreamId::Rgb
            || frame.device_capture_sequence.as_u64() != evidence.capture_sequence()
            || frame.host_delivery_sequence.as_u64() != evidence.delivery_sequence()
            || frame.timestamp.as_nanos() != evidence.device_timestamp_ns()
            || frame.width != evidence.width_px()
            || frame.height != evidence.height_px()
            || frame.stride_bytes != evidence.stride_bytes()
        {
            return Err(RerunWheelsOffError::RgbEvidenceMismatch);
        }
        let time = exact_rerun_device_time(evidence.device_timestamp_ns())?;
        self.recording.set_time("oak_device_time_ns", time);
        let image = rerun::Image::from_color_model_and_bytes(
            frame.pixels(),
            [frame.width, frame.height],
            rerun::ColorModel::BGR,
            rerun::ChannelDatatype::U8,
        );
        let logged = self
            .recording
            .log("bench/camera/rgb", &image)
            .map_err(RerunWheelsOffError::Recording);
        self.recording.disable_timeline("oak_device_time_ns");
        logged
    }

    fn flush(&mut self, timeout: Duration) -> Result<(), Self::Error> {
        self.recording
            .flush_with_timeout(timeout)
            .map_err(RerunWheelsOffError::Flush)
    }
}

fn exact_rerun_device_time(
    device_timestamp_ns: i64,
) -> Result<rerun::TimeCell, RerunWheelsOffError> {
    let time = rerun::TimeCell::from_duration_nanos(device_timestamp_ns);
    if time.as_i64() != device_timestamp_ns {
        return Err(RerunWheelsOffError::TimestampNotExactlyRepresentable {
            source_ns: device_timestamp_ns,
            encoded_ns: time.as_i64(),
        });
    }
    Ok(time)
}

/// Construct the production RGB bridge only from the admitted bench plan.
pub fn wheels_off_rgb_expression_bridge<C>(
    plan: &WheelsOffBenchPlan,
    stream_epoch: StreamEpochId,
    clock: C,
) -> RgbExpressionBridge<C>
where
    C: EyeMonotonicClock,
{
    RgbExpressionBridge::new(stream_epoch, plan.rgb_expression(), clock)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    use kiko_expression_core::{
        ExpressionKind, HeadMotionPolicy, MonotonicTimestamp, ReactionInputs, ReactionMixer,
        UnitAmount,
    };
    use kiko_expression_runtime::{
        EyeRenderStyle, REQUIRED_EYE_CAPABILITIES, adapt_reaction_output,
    };
    use kiko_eye_runtime::StaticEyeRuntimeConfigInput;
    use kiko_head_protocol::{
        HeadJoint, HeadPose, PositionAgreementTicks, PositionTicks, PresentPosition,
        ValidatedPresentPosition,
    };
    use kiko_head_runtime::HeadRuntimeConfigInput;
    use kiko_supervisor_core::{
        AuthorityDuration, ReadinessEpoch, Sha256Digest, StopReason, SupervisorAction,
        SupervisorConfig,
    };
    use oak_sys::{DepthConfig, ImuConfig, MonoConfig, QueueConfig, RgbConfig, UsbTransportPolicy};
    use robot_protocol::ControllerUptimeMsWrapping;
    use robot_protocol::v2::{
        ControlEpoch, ControllerBootId, ControllerDeadlineMsWrapping, ControllerFaults,
        ControllerUid, HostCommandResult, HostCommandResultCode, OutputState, RemainingLeaseMs,
        TimerPwm, V2CommandSequence,
    };
    use serde_json::json;

    use crate::navigation::NavigationClockEpoch;

    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct FakeError(&'static str);

    impl fmt::Display for FakeError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str(self.0)
        }
    }

    impl std::error::Error for FakeError {}

    type EventLog = Arc<Mutex<Vec<&'static str>>>;

    fn record(log: &EventLog, event: &'static str) {
        log.lock().expect("test event log lock").push(event);
    }

    struct NeverCancelled;

    impl WheelsOffBenchCancellationPort for NeverCancelled {
        fn poll_cancellation(&mut self) -> Option<WheelsOffBenchCancellation> {
            None
        }
    }

    struct CancelAfterEvent {
        log: EventLog,
        event: &'static str,
        signal: Option<WheelsOffBenchCancellation>,
    }

    impl WheelsOffBenchCancellationPort for CancelAfterEvent {
        fn poll_cancellation(&mut self) -> Option<WheelsOffBenchCancellation> {
            let observed = self
                .log
                .lock()
                .expect("test event log lock")
                .contains(&self.event);
            observed.then(|| self.signal.take()).flatten()
        }
    }

    struct QueuedCancellation(Option<WheelsOffBenchCancellation>);

    impl WheelsOffBenchCancellationPort for QueuedCancellation {
        fn poll_cancellation(&mut self) -> Option<WheelsOffBenchCancellation> {
            self.0.take()
        }
    }

    fn at(nanoseconds: u64) -> HostMonotonicTimestamp {
        HostMonotonicTimestamp::from_nanos(nanoseconds)
    }

    fn authority_duration(nanoseconds: u64) -> AuthorityDuration {
        AuthorityDuration::try_from_nanos(nanoseconds).expect("nonzero test duration")
    }

    fn controller_uid() -> ControllerUid {
        ControllerUid::try_new([3; 12]).expect("nonzero test controller UID")
    }

    fn controller_boot_id() -> ControllerBootId {
        ControllerBootId::try_new(7).expect("nonzero test controller boot ID")
    }

    fn readiness() -> ReadinessBinding {
        ReadinessBinding::new(
            ReadinessEpoch::try_new(1).expect("nonzero test readiness epoch"),
            controller_uid(),
            controller_boot_id(),
            ControlEpoch::try_new(9).expect("nonzero test control epoch"),
            Sha256Digest::try_new([2; 32]).expect("nonzero hardware digest"),
            Sha256Digest::try_new([3; 32]).expect("nonzero calibration digest"),
        )
    }

    fn host_zero(sequence: u32) -> HostCommandResult {
        HostCommandResult {
            controller_uid: controller_uid(),
            boot_id: controller_boot_id(),
            control_epoch: ControlEpoch::try_new(9).expect("nonzero test control epoch"),
            sequence: V2CommandSequence::new(sequence),
            result: HostCommandResultCode::AppliedNew,
            requested_timer_pwm: TimerPwm::ZERO,
            controller_timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::ZeroPwm,
            controller_applied_at: ControllerUptimeMsWrapping::new(10 + sequence),
            controller_expires_at: ControllerDeadlineMsWrapping::new(20 + sequence),
            remaining_lease: RemainingLeaseMs::ZERO,
            faults: ControllerFaults::NONE,
        }
    }

    fn ready_authority() -> AgentAuthoritySupervisor {
        let config = SupervisorConfig::new(
            authority_duration(1_000_000_000),
            authority_duration(100_000_000),
        )
        .expect("valid test supervisor config");
        let mut authority =
            AgentAuthoritySupervisor::new(config, NavigationClockEpoch::new(at(100)));
        assert_eq!(
            authority.begin_inventory(at(101)),
            Ok(SupervisorAction::InventoryRequired)
        );
        assert_eq!(
            authority.admit_readiness(readiness(), at(102)),
            Ok(SupervisorAction::Disarmed)
        );
        assert_eq!(
            authority.arm(at(103)),
            Ok(SupervisorAction::BaseZeroRequired {
                reason: StopReason::Arming,
            })
        );
        assert_eq!(
            authority.admit_applied_zero(host_zero(0), at(104), at(104)),
            Ok(SupervisorAction::ReadyStopped)
        );
        authority
    }

    fn navigation_oak_device_config() -> DeviceConfig {
        DeviceConfig {
            usb_transport: UsbTransportPolicy::super_speed_required(),
            rgb: Some(RgbConfig {
                width: 640,
                height: 480,
                fps: 30,
            }),
            mono: Some(MonoConfig {
                width: 640,
                height: 480,
                fps: 30,
                rectified: true,
            }),
            depth: Some(DepthConfig {
                width: 640,
                height: 480,
                fps: 30,
                alignment: DepthAlignment::RectifiedLeft,
            }),
            imu: Some(ImuConfig { rate_hz: 400 }),
            queue: QueueConfig::default(),
        }
    }

    fn head_config() -> HeadRuntimeConfig {
        HeadRuntimeConfig::parse(HeadRuntimeConfigInput {
            device_path: "/dev/serial/by-id/test-head".into(),
            response_timeout_ms: 100,
            write_timeout_ms: 10,
            arming_freshness_ms: 100,
            write_attempts: 2,
            noise_budget_bytes: 16,
            redundant_read_tolerance_ticks: 10,
            readback_tolerance_ticks: 20,
            goal_speed_ticks_per_second: 100,
            torque_limit_permille: [600, 400, 400, 400],
        })
        .expect("valid test head config")
    }

    fn eye_config() -> StaticEyeRuntimeConfig {
        StaticEyeRuntimeConfig::parse(StaticEyeRuntimeConfigInput {
            device_path: "/dev/serial/by-id/test-eye".into(),
            baud_rate_bps: 115_200,
            response_timeout_ms: 20,
            write_timeout_ms: 5,
            write_attempts: 2,
            empty_delimiter_budget: 2,
            expected_device_uid: [1; 16],
            expected_firmware_build_id: [2; 32],
            expected_capabilities_bits: REQUIRED_EYE_CAPABILITIES,
            intent_lease_ms: 100,
        })
        .expect("valid test eye config")
    }

    fn rgb_expression_config() -> NanoRgbExpressionConfig {
        let document = json!({
            "schema_version": super::super::NANO_AGENT_POLICY_CONFIG_V1,
            "control": {
                "socket_path": "/tmp/kiko-bench-test.sock",
                "read_timeout_ms": 100,
                "write_timeout_ms": 100,
                "runtime_response_timeout_ms": 500,
                "runtime_queue_capacity": 8
            },
            "inventory": {
                "manifest_path": "/opt/kiko/device.json",
                "artifact_root_path": "/opt/kiko/artifacts",
                "artifact_bindings": [
                    {"kind":"calibration","artifact_id":"stereo-v1","relative_path":"stereo.json"},
                    {"kind":"plant","artifact_id":"drive-v1","relative_path":"drive.json"}
                ]
            },
            "map_persistence": {
                "save_snapshot_path": "/var/lib/kiko/map.kmap",
                "warm_start": {"kind":"none"}
            },
            "eye": {
                "mode":"kep2",
                "device_path":"/dev/serial/by-id/test-eye",
                "baud_rate_bps":115200,
                "response_timeout_ms":20,
                "write_timeout_ms":5,
                "write_attempts":2,
                "empty_delimiter_budget":2,
                "expected_device_uid":vec![1_u8;16],
                "expected_firmware_build_id":vec![2_u8;32],
                "expected_capabilities_bits":REQUIRED_EYE_CAPABILITIES,
                "intent_lease_ms":100
            },
            "head": {
                "mode":"natural_hold",
                "device_path":"/dev/serial/by-id/test-head",
                "response_timeout_ms":100,
                "write_timeout_ms":10,
                "arming_freshness_ms":100,
                "write_attempts":2,
                "noise_budget_bytes":16,
                "redundant_read_tolerance_ticks":10,
                "readback_tolerance_ticks":20,
                "goal_speed_ticks_per_second":100,
                "torque_limit_permille":[600,400,400,400],
                "physical_torque_consent":"natural_hold_at_observed_pose"
            },
            "rgb_expression": {
                "mode":"scene_motion",
                "sampling_columns":16,
                "sampling_rows":12,
                "minimum_residual_luma":24,
                "minimum_active_fraction_basis_points":500,
                "frame_freshness_ms":80,
                "brightness_basis_points":7000,
                "color_rgb":[32,128,255],
                "blink":false,
                "gaze_geometry": {
                    "schema_version":1,
                    "head_origin_in_camera_m":[0.0,-0.25,-0.20],
                    "neutral_head_from_camera_quaternion_xyzw":[0.0,0.0,0.0,1.0]
                }
            },
            "supervisor": {
                "maximum_authority_lease_ms":1000,
                "maximum_zero_age_ms":250
            },
            "live_mode_policy": {
                "startup":"disarmed_map_only",
                "manual":{"permission":"disabled"},
                "point_goal":{"permission":"disabled"},
                "frontier_explore":{"permission":"disabled"}
            }
        });
        let encoded = serde_json::to_vec(&document).expect("serialize test policy");
        NanoAgentPolicyConfigV1::parse_json(&encoded)
            .expect("valid test policy")
            .rgb_expression()
            .scene_motion()
            .expect("enabled test RGB expression")
    }

    fn test_plan() -> WheelsOffBenchPlan {
        let authority = ready_authority();
        let (readiness, zero) = match authority.state() {
            SupervisorState::ReadyStopped { readiness, zero } => (readiness, zero),
            state => panic!("test authority is not ready-stopped: {state:?}"),
        };
        WheelsOffBenchPlan {
            base: WheelsOffBaseAdmission {
                authority,
                readiness,
                zero,
                admitted_at: at(105),
            },
            oak: WheelsOffBenchOakConfig::try_new(
                "ABCDEF1234567890".into(),
                navigation_oak_device_config(),
            )
            .expect("valid test OAK config"),
            head: head_config(),
            head_consent: PhysicalTorqueEnableConsent::explicitly_granted(),
            eye: eye_config(),
            rgb_expression: rgb_expression_config(),
            configured_pose_bounds: WheelsOffConfiguredPoseBounds::try_new([1_900; 4], [2_100; 4])
                .expect("valid configured pose bounds"),
            rerun_flush_timeout: Duration::from_millis(10),
            capture: WheelsOffBenchCapturePlan::try_new(1, 3).expect("valid test capture plan"),
        }
    }

    struct FakeBaseKeeper {
        log: EventLog,
        sequence: u32,
        observed_at_ns: u64,
        running: bool,
        healthy: bool,
    }

    impl WheelsOffBaseCleanupPort for FakeBaseKeeper {
        type Evidence = u32;
        type HealthEvidence = ();
        type DisarmEvidence = ();
        type Error = FakeError;

        fn check_health(
            &mut self,
        ) -> impl Future<Output = Result<Self::HealthEvidence, Self::Error>> + Send {
            record(&self.log, "base_health");
            std::future::ready(
                (self.running && self.healthy)
                    .then_some(())
                    .ok_or(FakeError("zero keeper unhealthy")),
            )
        }

        fn refresh_zero(
            &mut self,
        ) -> impl Future<Output = Result<RefreshedBaseZero<Self::Evidence>, Self::Error>> + Send
        {
            if !self.running {
                return std::future::ready(Err(FakeError("zero keeper stopped")));
            }
            self.sequence += 1;
            self.observed_at_ns += 1;
            record(&self.log, "zero_checkpoint");
            std::future::ready(Ok(RefreshedBaseZero::try_from_host_result(
                self.sequence,
                host_zero(self.sequence),
                at(self.observed_at_ns),
            )
            .expect("valid fake zero keeper receipt")))
        }

        fn disarm(
            &mut self,
        ) -> impl Future<Output = Result<Self::DisarmEvidence, Self::Error>> + Send {
            if !self.running {
                return std::future::ready(Err(FakeError("zero keeper already stopped")));
            }
            self.running = false;
            record(&self.log, "base_disarm");
            std::future::ready(Ok(()))
        }
    }

    fn depth_evidence(sequence: u64) -> BenchDepthReadinessEvidence {
        BenchDepthReadinessEvidence {
            capture_sequence: sequence,
            delivery_sequence: sequence,
            device_timestamp_ns: i64::try_from(sequence * 10).expect("small test timestamp"),
            width_px: 2,
            height_px: 1,
        }
    }

    fn imu_evidence(sequence: u32) -> BenchImuReadinessEvidence {
        BenchImuReadinessEvidence {
            sample_count: NonZeroUsize::new(1).expect("nonzero test sample count"),
            first_delivery_sequence: sequence,
            last_delivery_sequence: sequence,
            first_accel_timestamp_ns: i64::from(sequence) * 10,
            last_accel_timestamp_ns: i64::from(sequence) * 10,
            first_gyro_timestamp_ns: i64::from(sequence) * 10 + 1,
            last_gyro_timestamp_ns: i64::from(sequence) * 10 + 1,
        }
    }

    fn rgb_evidence(sequence: u64) -> BenchRgbFrameEvidence {
        BenchRgbFrameEvidence {
            capture_sequence: sequence,
            delivery_sequence: sequence,
            device_timestamp_ns: i64::try_from(sequence * 10).expect("small test timestamp"),
            width_px: 1,
            height_px: 1,
            stride_bytes: 3,
        }
    }

    struct FakeOak {
        log: EventLog,
        depths: VecDeque<BenchCapturedDepth<u8>>,
        imus: VecDeque<BenchCapturedImu<u8>>,
        rgbs: VecDeque<BenchCapturedRgb<u8>>,
    }

    impl FakeOak {
        fn new(log: EventLog) -> Self {
            Self {
                log,
                depths: [1_u64, 2]
                    .into_iter()
                    .map(|sequence| BenchCapturedDepth {
                        evidence: depth_evidence(sequence),
                        frame: u8::try_from(sequence).expect("small test depth"),
                    })
                    .collect(),
                imus: [1_u32, 2]
                    .into_iter()
                    .map(|sequence| BenchCapturedImu {
                        evidence: imu_evidence(sequence),
                        batch: u8::try_from(sequence).expect("small test IMU"),
                    })
                    .collect(),
                rgbs: [1_u64, 2, 3]
                    .into_iter()
                    .map(|sequence| BenchCapturedRgb {
                        evidence: rgb_evidence(sequence),
                        frame: u8::try_from(sequence).expect("small test RGB"),
                    })
                    .collect(),
            }
        }
    }

    impl WheelsOffOakPort for FakeOak {
        type RgbFrame = u8;
        type DepthFrame = u8;
        type ImuBatch = u8;
        type Error = FakeError;

        fn connect(
            &mut self,
            config: &WheelsOffBenchOakConfig,
        ) -> Result<BenchOakConnectionEvidence, Self::Error> {
            record(&self.log, "oak_connect");
            Ok(BenchOakConnectionEvidence {
                requested_mxid: config.expected_mxid().into(),
                opened_mxid: config.expected_mxid().into(),
                discovery_transport_name: None,
                eeprom_device_name: None,
                product_name: None,
                usb_requested_maximum: config.device().usb_transport.maximum(),
                usb_required_minimum: config.device().usb_transport.minimum(),
                usb_observed: UsbTransportSpeed::Super,
            })
        }

        fn try_depth(
            &mut self,
            _timeout_ms: NonZeroU32,
        ) -> Result<Option<BenchCapturedDepth<Self::DepthFrame>>, Self::Error> {
            Ok(self.depths.pop_front())
        }

        fn try_imu(&mut self) -> Result<Option<BenchCapturedImu<Self::ImuBatch>>, Self::Error> {
            Ok(self.imus.pop_front())
        }

        fn try_rgb(
            &mut self,
            _timeout_ms: NonZeroU32,
        ) -> Result<Option<BenchCapturedRgb<Self::RgbFrame>>, Self::Error> {
            Ok(self.rgbs.pop_front())
        }

        fn close(&mut self) -> Result<(), Self::Error> {
            record(&self.log, "oak_close");
            Ok(())
        }
    }

    struct FakeHead {
        log: EventLog,
        observed: [PositionTicks; 4],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct FakeHeadStartup {
        configured_pose: ObservedPoseWithinConfiguredBounds,
    }

    fn fake_position_response(joint: HeadJoint, position: PositionTicks) -> [u8; 8] {
        let position = position.get().to_le_bytes();
        let mut bytes = [
            0xff,
            0xff,
            joint.servo_id().get(),
            4,
            0,
            position[0],
            position[1],
            0,
        ];
        bytes[7] = !bytes[2..7]
            .iter()
            .fold(0_u8, |sum, byte| sum.wrapping_add(*byte));
        bytes
    }

    fn fake_head_pose(observed: [PositionTicks; 4]) -> HeadPose {
        let validated: [ValidatedPresentPosition; 4] = std::array::from_fn(|index| {
            let joint = HeadJoint::ALL[index];
            let response = fake_position_response(joint, observed[index]);
            let parsed = PresentPosition::parse(&response, joint.servo_id())
                .expect("valid fake present-position response");
            ValidatedPresentPosition::try_from_pair(
                parsed,
                parsed,
                PositionAgreementTicks::try_new(0).expect("zero tolerance is valid"),
            )
            .expect("identical fake observations agree")
        });
        HeadPose::try_from_validated(validated).expect("canonical fake head pose")
    }

    impl WheelsOffHeadPort for FakeHead {
        type StartupEvidence = FakeHeadStartup;
        type ShutdownEvidence = ();
        type Error = FakeError;

        fn configured_pose(evidence: &Self::StartupEvidence) -> ObservedPoseWithinConfiguredBounds {
            evidence.configured_pose
        }

        fn start(
            &mut self,
            _config: HeadRuntimeConfig,
            configured_pose_bounds: WheelsOffConfiguredPoseBounds,
            _consent: PhysicalTorqueEnableConsent,
        ) -> impl Future<Output = Result<Self::StartupEvidence, Self::Error>> + Send {
            record(&self.log, "head_start");
            let result = configured_pose_bounds
                .admit(fake_head_pose(self.observed))
                .map(|configured_pose| FakeHeadStartup { configured_pose })
                .map_err(|_| FakeError("head pose outside configured bounds"));
            std::future::ready(result)
        }

        fn shutdown(
            &mut self,
        ) -> impl Future<Output = Result<Self::ShutdownEvidence, Self::Error>> + Send {
            record(&self.log, "head_shutdown");
            std::future::ready(Ok(()))
        }
    }

    struct FakeEye {
        log: EventLog,
    }

    impl WheelsOffEyePort<()> for FakeEye {
        type StartupEvidence = ();
        type ApplyEvidence = ();
        type ShutdownEvidence = ();
        type Error = FakeError;

        fn start(
            &mut self,
            _config: EyeRuntimeConfig,
            _clock: (),
        ) -> impl Future<Output = Result<Self::StartupEvidence, Self::Error>> + Send {
            record(&self.log, "eye_start");
            std::future::ready(Ok(()))
        }

        fn apply(
            &mut self,
            _intent: PreparedEyeIntent,
        ) -> impl Future<Output = Result<Self::ApplyEvidence, Self::Error>> + Send {
            record(&self.log, "eye_apply");
            std::future::ready(Ok(()))
        }

        fn shutdown(
            &mut self,
        ) -> impl Future<Output = Result<Self::ShutdownEvidence, Self::Error>> + Send {
            record(&self.log, "eye_shutdown");
            std::future::ready(Ok(()))
        }
    }

    struct FakeExpression {
        log: EventLog,
    }

    impl WheelsOffExpressionPort<u8> for FakeExpression {
        type Clock = ();
        type Error = FakeError;

        fn clone_clock_for_eye_actor(&self) -> Self::Clock {}

        fn process(&mut self, _frame: &u8) -> Result<RgbExpressionBridgeOutcome, Self::Error> {
            record(&self.log, "expression_process");
            let now = MonotonicTimestamp::from_nanos_since_epoch(1);
            let output =
                ReactionMixer::new(HeadMotionPolicy::NaturalHold).mix(now, ReactionInputs::empty());
            let prepared = adapt_reaction_output(
                output,
                ExpressionKind::Neutral,
                EyeRenderStyle::new(
                    UnitAmount::try_from_basis_points(7_000).expect("test brightness"),
                    [1, 2, 3],
                    false,
                ),
                now,
            )
            .expect("neutral test expression adapts");
            Ok(RgbExpressionBridgeOutcome::ColdStart(prepared))
        }
    }

    struct FakeTelemetry {
        log: EventLog,
    }

    impl WheelsOffTelemetryPort<FakeOak> for FakeTelemetry {
        type Error = FakeError;

        fn event(&mut self, _event: WheelsOffBenchEvent) -> Result<(), Self::Error> {
            Ok(())
        }

        fn camera_readiness(
            &mut self,
            _connection: &BenchOakConnectionEvidence,
            _readiness: BenchCameraReadinessEvidence,
        ) -> Result<(), Self::Error> {
            Ok(())
        }

        fn depth(
            &mut self,
            _frame: &u8,
            _evidence: BenchDepthReadinessEvidence,
        ) -> Result<(), Self::Error> {
            record(&self.log, "depth_log");
            Ok(())
        }

        fn imu(
            &mut self,
            _batch: &u8,
            _evidence: BenchImuReadinessEvidence,
        ) -> Result<(), Self::Error> {
            record(&self.log, "imu_log");
            Ok(())
        }

        fn rgb(
            &mut self,
            _frame: &u8,
            _evidence: BenchRgbFrameEvidence,
        ) -> Result<(), Self::Error> {
            record(&self.log, "rgb_log");
            Ok(())
        }

        fn flush(&mut self, _timeout: Duration) -> Result<(), Self::Error> {
            record(&self.log, "rerun_flush");
            Ok(())
        }
    }

    fn runtime(
        observed_ticks: [u16; 4],
    ) -> (
        EventLog,
        WheelsOffBenchRuntime<
            FakeBaseKeeper,
            FakeHead,
            FakeEye,
            FakeOak,
            FakeTelemetry,
            FakeExpression,
        >,
    ) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let observed = observed_ticks
            .map(|ticks| PositionTicks::try_new(ticks).expect("valid fake observed position"));
        let runtime = WheelsOffBenchRuntime::new(
            test_plan(),
            FakeBaseKeeper {
                log: Arc::clone(&log),
                sequence: 0,
                observed_at_ns: 104,
                running: true,
                healthy: true,
            },
            FakeHead {
                log: Arc::clone(&log),
                observed,
            },
            FakeEye {
                log: Arc::clone(&log),
            },
            FakeOak::new(Arc::clone(&log)),
            FakeTelemetry {
                log: Arc::clone(&log),
            },
            FakeExpression {
                log: Arc::clone(&log),
            },
        );
        (log, runtime)
    }

    fn event_position(events: &[&str], target: &str) -> usize {
        events
            .iter()
            .position(|event| *event == target)
            .unwrap_or_else(|| panic!("missing event {target}: {events:?}"))
    }

    #[test]
    fn oak_contract_rejects_a_different_pipeline_than_navigation() {
        let mut usb2_diagnostic = navigation_oak_device_config();
        usb2_diagnostic.usb_transport = UsbTransportPolicy::high_speed_diagnostic();
        assert!(matches!(
            WheelsOffBenchOakConfig::try_new("ABCDEF1234567890".into(), usb2_diagnostic),
            Err(WheelsOffBenchOakConfigError::UsbMinimumBelowProduction {
                actual: UsbTransportSpeed::High,
                required: UsbTransportSpeed::Super,
            })
        ));

        let mut rgb_aligned = navigation_oak_device_config();
        rgb_aligned.depth.as_mut().expect("depth enabled").alignment = DepthAlignment::Rgb;
        assert!(matches!(
            WheelsOffBenchOakConfig::try_new("ABCDEF1234567890".into(), rgb_aligned),
            Err(WheelsOffBenchOakConfigError::DepthAlignment {
                actual: DepthAlignment::Rgb,
                required: DepthAlignment::RectifiedLeft,
            })
        ));

        let mut unrectified = navigation_oak_device_config();
        unrectified.mono.as_mut().expect("mono enabled").rectified = false;
        assert!(matches!(
            WheelsOffBenchOakConfig::try_new("ABCDEF1234567890".into(), unrectified),
            Err(WheelsOffBenchOakConfigError::MonoNotRectified)
        ));

        let mut no_mono = navigation_oak_device_config();
        no_mono.mono = None;
        assert!(matches!(
            WheelsOffBenchOakConfig::try_new("ABCDEF1234567890".into(), no_mono),
            Err(WheelsOffBenchOakConfigError::RequiredStreamDisabled {
                stream: "rectified mono pair",
            })
        ));
    }

    #[test]
    fn imu_continuity_checks_accel_and_gyro_clocks_independently() {
        let first = imu_evidence(1);
        let mut bad_accel = imu_evidence(2);
        bad_accel.first_accel_timestamp_ns = first.last_accel_timestamp_ns;
        assert!(matches!(
            pair_imu(first, bad_accel),
            Err(BenchCameraContinuityError::ImuAccelTimestampNotIncreasing { .. })
        ));

        let mut bad_gyro = imu_evidence(2);
        bad_gyro.first_gyro_timestamp_ns = first.last_gyro_timestamp_ns;
        assert!(matches!(
            pair_imu(first, bad_gyro),
            Err(BenchCameraContinuityError::ImuGyroTimestampNotIncreasing { .. })
        ));
    }

    #[tokio::test]
    async fn fake_ports_prove_checkpoint_start_cleanup_and_disarm_order() {
        let (log, mut runtime) = runtime([2_000; 4]);
        let started = runtime
            .start(&mut NeverCancelled)
            .await
            .expect("fake bench starts");
        assert_eq!(started.expression_rgb.capture_sequence(), 3);
        let cleanup = runtime.shutdown().await;
        assert!(cleanup.base.is_ok());
        assert!(cleanup.base_disarm.is_ok());

        let events = log.lock().expect("test event log lock").clone();
        let zeroes: Vec<_> = events
            .iter()
            .enumerate()
            .filter_map(|(index, event)| (*event == "zero_checkpoint").then_some(index))
            .collect();
        assert_eq!(zeroes.len(), 4, "events={events:?}");
        assert!(zeroes[0] < event_position(&events, "oak_connect"));
        assert!(zeroes[1] < event_position(&events, "head_start"));
        assert!(zeroes[2] < event_position(&events, "eye_start"));
        assert!(zeroes[3] < event_position(&events, "eye_shutdown"));
        assert!(event_position(&events, "eye_shutdown") < event_position(&events, "head_shutdown"));
        assert!(event_position(&events, "head_shutdown") < event_position(&events, "oak_close"));
        assert!(event_position(&events, "oak_close") < event_position(&events, "rerun_flush"));
        assert!(event_position(&events, "rerun_flush") < event_position(&events, "base_disarm"));
        assert_eq!(
            events.iter().filter(|event| **event == "rgb_log").count(),
            3
        );
    }

    #[tokio::test]
    async fn configured_pose_refusal_happens_inside_head_start_before_eye_start() {
        let (log, mut runtime) = runtime([2_200, 2_000, 2_000, 2_000]);
        let failure = runtime
            .start(&mut NeverCancelled)
            .await
            .expect_err("pose must be refused");
        assert!(matches!(
            failure.cause,
            WheelsOffBenchStartCause::Head(FakeError("head pose outside configured bounds"))
        ));
        assert!(failure.cleanup.base.is_ok());
        assert!(failure.cleanup.base_disarm.is_ok());

        let events = log.lock().expect("test event log lock").clone();
        let last_zero = events
            .iter()
            .rposition(|event| *event == "zero_checkpoint")
            .expect("cleanup zero checkpoint");
        assert!(last_zero < event_position(&events, "base_disarm"));
        assert!(!events.contains(&"head_shutdown"));
        assert!(!events.contains(&"eye_start"));
    }

    #[tokio::test]
    async fn queued_cancellation_before_start_enters_cleanup_without_accessory_io() {
        let (log, mut runtime) = runtime([2_000; 4]);
        let failure = runtime
            .start(&mut QueuedCancellation(Some(
                WheelsOffBenchCancellation::Terminate,
            )))
            .await
            .expect_err("queued termination must cancel startup");
        assert!(matches!(
            failure.cause,
            WheelsOffBenchStartCause::Cancelled {
                signal: WheelsOffBenchCancellation::Terminate,
                checkpoint: WheelsOffBenchCancellationCheckpoint::BeforeInitialBaseZero,
            }
        ));
        assert!(failure.cleanup.base.is_ok());
        assert!(failure.cleanup.base_disarm.is_ok());

        let events = log.lock().expect("test event log lock").clone();
        assert_eq!(
            events,
            vec!["zero_checkpoint", "rerun_flush", "base_disarm"]
        );
    }

    #[tokio::test]
    async fn cancellation_queued_during_head_start_shuts_head_down_before_eye_start() {
        let (log, mut runtime) = runtime([2_000; 4]);
        let mut cancellation = CancelAfterEvent {
            log: Arc::clone(&log),
            event: "head_start",
            signal: Some(WheelsOffBenchCancellation::Interrupt),
        };
        let failure = runtime
            .start(&mut cancellation)
            .await
            .expect_err("interrupt queued during head start must cancel startup");
        assert!(matches!(
            failure.cause,
            WheelsOffBenchStartCause::Cancelled {
                signal: WheelsOffBenchCancellation::Interrupt,
                checkpoint: WheelsOffBenchCancellationCheckpoint::AfterHeadStart,
            }
        ));

        let events = log.lock().expect("test event log lock").clone();
        let cleanup_zero = events
            .iter()
            .rposition(|event| *event == "zero_checkpoint")
            .expect("cleanup zero checkpoint");
        assert!(cleanup_zero < event_position(&events, "head_shutdown"));
        assert!(!events.contains(&"eye_start"));
        assert!(event_position(&events, "head_shutdown") < event_position(&events, "base_disarm"));
    }

    #[tokio::test]
    async fn runtime_rgb_liveness_is_bounded_by_the_parsed_capture_plan() {
        let (_log, mut runtime) = runtime([2_000; 4]);
        runtime
            .start(&mut NeverCancelled)
            .await
            .expect("fake bench starts");

        assert!(
            runtime
                .process_next_rgb()
                .await
                .expect("first miss")
                .is_none()
        );
        assert!(
            runtime
                .process_next_rgb()
                .await
                .expect("second miss")
                .is_none()
        );
        assert!(matches!(
            runtime.process_next_rgb().await,
            Err(WheelsOffBenchCycleError::RgbLivenessLost {
                consecutive_misses: 3,
                maximum_misses: 3,
                per_attempt_timeout_ms: 1,
            })
        ));
    }

    #[tokio::test]
    async fn runtime_rgb_rejects_a_replayed_frame_as_non_live() {
        let (_log, mut runtime) = runtime([2_000; 4]);
        runtime
            .start(&mut NeverCancelled)
            .await
            .expect("fake bench starts");
        runtime.oak.rgbs.push_back(BenchCapturedRgb {
            evidence: rgb_evidence(3),
            frame: 3,
        });

        assert!(matches!(
            runtime.process_next_rgb().await,
            Err(WheelsOffBenchCycleError::CameraContinuity(
                BenchCameraContinuityError::RgbCaptureSequenceNotIncreasing {
                    first: 3,
                    second: 3,
                }
            ))
        ));
    }

    #[tokio::test]
    async fn keeper_health_failure_stops_a_runtime_cycle_before_camera_work() {
        let (log, mut runtime) = runtime([2_000; 4]);
        runtime
            .start(&mut NeverCancelled)
            .await
            .expect("fake bench starts");
        runtime.base_cleanup.healthy = false;

        assert!(matches!(
            runtime.process_next_rgb().await,
            Err(WheelsOffBenchCycleError::BaseHealth(FakeError(
                "zero keeper unhealthy"
            )))
        ));
        assert_eq!(
            log.lock()
                .expect("test event log lock")
                .iter()
                .filter(|event| **event == "base_health")
                .count(),
            1
        );
    }
}
