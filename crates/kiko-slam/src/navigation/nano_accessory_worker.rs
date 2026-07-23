//! Single-owner Nano head/eye worker fed by the live OAK owner's RGB frames.
//!
//! The worker never opens or owns an OAK device. Its only camera boundary is a
//! capacity-one, replace-latest queue of already-owned [`oak_sys::ImageFrame`]
//! values. One dedicated thread owns one current-thread Tokio runtime, the
//! manifest-bound return-to-natural head actor, the manifest-bound KEP2 eye
//! actor, and one [`RgbExpressionBridge`].
//!
//! A terminal accessory fault is a base-stop signal, not permission to alter
//! the head. After publishing the first terminal fault, the worker stops
//! accepting RGB work and retains the natural-hold actor while continuing
//! bounded health checks. Only [`NanoAccessoryWorker::shutdown`] asks the eye
//! actor to release control and the head actor to release serial ownership.
//! Production head release performs no torque-switch write, preserving the
//! last admitted hold without claiming the physical torque state. Dropping the
//! worker is deliberately not an implicit physical shutdown.

use std::fmt;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use kiko_expression_core::StreamEpochId;
use kiko_expression_runtime::PreparedEyeIntent;
use kiko_eye_runtime::{
    ActorExit as EyeActorExit, ActorTermination as EyeActorTermination, ClockError, EyeActorHandle,
    EyeActorStartError, EyeActorTask, EyeRuntimeConfig, EyeRuntimeFault,
    HandleRequestError as EyeHandleRequestError, MonotonicClock, OsEyeSessionMaterialError,
    OsEyeSessionMaterialGenerator, ReleaseReport, SerialConfigurationEvidence as EyeSerialEvidence,
    StartupEvidence as EyeStartupEvidence, StartupReceiptError as EyeStartupReceiptError,
    StaticEyeRuntimeConfig, TokioClock,
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
use oak_sys::ImageFrame;
use tokio::task::JoinError;

use super::expression_bridge::IngressObservedRgbFrame;
use super::{
    ManifestBoundNanoAgentPolicyConfigV3, NanoRgbExpressionConfig, RgbExpressionBridge,
    RgbExpressionBridgeError,
};

/// A health cadence long enough to avoid a zero-duration busy loop and short
/// enough for the base owner to receive a bounded-latency health result.
pub const MAX_NANO_ACCESSORY_HEALTH_PERIOD: Duration = Duration::from_secs(5);

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
        })
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

/// Readiness is emitted only after eye startup and the head's startup,
/// reviewed return, and immediate exact-target health check all succeeded.
#[derive(Clone, Debug)]
pub struct NanoAccessoryReadyEvidence {
    eye: NanoEyeReadyEvidence,
    head: NanoHeadReadyEvidence,
    stream_epoch: StreamEpochId,
    health_period: NanoAccessoryHealthPeriod,
}

impl NanoAccessoryReadyEvidence {
    pub const fn eye(&self) -> &NanoEyeReadyEvidence {
        &self.eye
    }

    pub const fn head(&self) -> &NanoHeadReadyEvidence {
        &self.head
    }

    pub const fn stream_epoch(&self) -> StreamEpochId {
        self.stream_epoch
    }

    pub const fn health_period(&self) -> NanoAccessoryHealthPeriod {
        self.health_period
    }
}

/// Result of one non-blocking RGB queue-ownership attempt.
///
/// Success means the ingress-owned frame or typed ingress-clock failure was
/// moved into the replace-latest slot. Semantic frame acceptance happens only
/// when the worker consumes that slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAccessoryFrameSubmitOutcome {
    Enqueued,
    ReplacedOlderFrame,
    TerminalFaultLatched,
    IngressDisconnected,
    ChannelPoisoned,
}

/// Saturating capacity-one ingress counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NanoAccessoryFrameStats {
    pub enqueued: u64,
    pub replaced_older: u64,
    pub processed_successfully: u64,
    pub rejected_after_fault: u64,
    pub rejected_disconnected: u64,
    pub channel_poisoned: u64,
}

#[derive(Debug)]
struct LatestFrameCounters {
    enqueued: AtomicU64,
    replaced_older: AtomicU64,
    processed_successfully: AtomicU64,
    rejected_after_fault: AtomicU64,
    rejected_disconnected: AtomicU64,
    channel_poisoned: AtomicU64,
}

impl LatestFrameCounters {
    fn new() -> Self {
        Self {
            enqueued: AtomicU64::new(0),
            replaced_older: AtomicU64::new(0),
            processed_successfully: AtomicU64::new(0),
            rejected_after_fault: AtomicU64::new(0),
            rejected_disconnected: AtomicU64::new(0),
            channel_poisoned: AtomicU64::new(0),
        }
    }

    fn snapshot(&self) -> NanoAccessoryFrameStats {
        NanoAccessoryFrameStats {
            enqueued: self.enqueued.load(Ordering::Relaxed),
            replaced_older: self.replaced_older.load(Ordering::Relaxed),
            processed_successfully: self.processed_successfully.load(Ordering::Relaxed),
            rejected_after_fault: self.rejected_after_fault.load(Ordering::Relaxed),
            rejected_disconnected: self.rejected_disconnected.load(Ordering::Relaxed),
            channel_poisoned: self.channel_poisoned.load(Ordering::Relaxed),
        }
    }
}

fn saturating_increment(counter: &AtomicU64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        value.checked_add(1)
    });
}

struct LatestFrameChannel<F> {
    slot: Mutex<Option<F>>,
    notify: tokio::sync::Notify,
    ingress_alive: AtomicBool,
    accepting_frames: AtomicBool,
    shutdown_requested: AtomicBool,
    poisoned: AtomicBool,
    counters: LatestFrameCounters,
}

impl<F> LatestFrameChannel<F> {
    fn new() -> Self {
        Self {
            slot: Mutex::new(None),
            notify: tokio::sync::Notify::new(),
            ingress_alive: AtomicBool::new(true),
            accepting_frames: AtomicBool::new(true),
            shutdown_requested: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            counters: LatestFrameCounters::new(),
        }
    }

    fn submit(&self, frame: F) -> NanoAccessoryFrameSubmitOutcome {
        if self.poisoned.load(Ordering::Acquire) {
            saturating_increment(&self.counters.channel_poisoned);
            return NanoAccessoryFrameSubmitOutcome::ChannelPoisoned;
        }
        if !self.ingress_alive.load(Ordering::Acquire)
            || self.shutdown_requested.load(Ordering::Acquire)
        {
            saturating_increment(&self.counters.rejected_disconnected);
            return NanoAccessoryFrameSubmitOutcome::IngressDisconnected;
        }
        if !self.accepting_frames.load(Ordering::Acquire) {
            saturating_increment(&self.counters.rejected_after_fault);
            return NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched;
        }

        let mut slot = match self.slot.lock() {
            Ok(slot) => slot,
            Err(_) => {
                self.poisoned.store(true, Ordering::Release);
                self.accepting_frames.store(false, Ordering::Release);
                saturating_increment(&self.counters.channel_poisoned);
                self.notify.notify_one();
                return NanoAccessoryFrameSubmitOutcome::ChannelPoisoned;
            }
        };
        if !self.accepting_frames.load(Ordering::Acquire)
            || self.shutdown_requested.load(Ordering::Acquire)
        {
            saturating_increment(&self.counters.rejected_after_fault);
            return NanoAccessoryFrameSubmitOutcome::TerminalFaultLatched;
        }
        let replaced = slot.replace(frame).is_some();
        saturating_increment(&self.counters.enqueued);
        if replaced {
            saturating_increment(&self.counters.replaced_older);
        }
        drop(slot);
        self.notify.notify_one();
        if replaced {
            NanoAccessoryFrameSubmitOutcome::ReplacedOlderFrame
        } else {
            NanoAccessoryFrameSubmitOutcome::Enqueued
        }
    }

    fn request_shutdown(&self) {
        self.accepting_frames.store(false, Ordering::Release);
        self.shutdown_requested.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    fn latch_terminal_fault(&self) {
        self.accepting_frames.store(false, Ordering::Release);
    }

    fn disconnect_ingress(&self) {
        self.ingress_alive.store(false, Ordering::Release);
        self.notify.notify_waiters();
    }

    async fn next_event(&self) -> LatestFrameEvent<F> {
        loop {
            let notified = self.notify.notified();
            if self.shutdown_requested.load(Ordering::Acquire) {
                return LatestFrameEvent::ShutdownRequested;
            }
            if self.poisoned.load(Ordering::Acquire) {
                return LatestFrameEvent::ChannelPoisoned;
            }
            if !self.ingress_alive.load(Ordering::Acquire) {
                return LatestFrameEvent::IngressDisconnected;
            }
            let frame = match self.slot.lock() {
                Ok(mut slot) => slot.take(),
                Err(_) => {
                    self.poisoned.store(true, Ordering::Release);
                    self.accepting_frames.store(false, Ordering::Release);
                    saturating_increment(&self.counters.channel_poisoned);
                    return LatestFrameEvent::ChannelPoisoned;
                }
            };
            if let Some(frame) = frame {
                return LatestFrameEvent::Frame(frame);
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
}

enum LatestFrameEvent<F> {
    Frame(F),
    ShutdownRequested,
    IngressDisconnected,
    ChannelPoisoned,
}

type NanoAccessoryRgbWork = Result<IngressObservedRgbFrame<ImageFrame>, ClockError>;

/// Sole synchronous producer for the capacity-one RGB handoff.
#[must_use = "dropping the sole RGB ingress publishes a terminal disconnect fault"]
pub struct NanoAccessoryRgbIngress {
    channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>>,
    clock: TokioClock,
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
        let work = observe_rgb_at_ingress(&self.clock, frame);
        self.channel.submit(work)
    }

    pub fn stats(&self) -> NanoAccessoryFrameStats {
        self.channel.counters.snapshot()
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
) -> Result<IngressObservedRgbFrame<F>, ClockError> {
    clock
        .now()
        .map(|observed_at| IngressObservedRgbFrame::new(frame, observed_at))
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
        }
    }
}

#[derive(Clone, Debug)]
pub enum NanoAccessoryTerminalFault {
    HeadHealth(NanoHeadHealthError),
    HeadHealthStatusPoisoned,
    ExpressionBridge(RgbExpressionBridgeError),
    EyeApply(EyeHandleRequestError),
    RgbIngressDisconnected,
    RgbChannelPoisoned,
    ReadinessObserverDropped,
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
            Self::ExpressionBridge(source) => Some(source),
            Self::EyeApply(source) => Some(source),
            Self::HeadHealthStatusPoisoned
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
}

impl NanoAccessoryShutdownEvidence {
    pub const fn eye(&self) -> &NanoEyeShutdownEvidence {
        &self.eye
    }

    pub const fn head(&self) -> &NanoHeadShutdownEvidence {
        &self.head
    }
}

#[derive(Debug)]
pub enum NanoAccessoryWorkerExit {
    RuntimeBuildFailed {
        message: Box<str>,
    },
    EyeSessionMaterialFailed(OsEyeSessionMaterialError),
    EyeStartupFailed(Box<NanoEyeActorStartupError>),
    HeadStartupFailed {
        source: Box<NanoHeadActorStartupError>,
        eye_shutdown: Box<NanoEyeShutdownEvidence>,
    },
    Shutdown {
        terminal_fault: Option<NanoAccessoryTerminalFault>,
        evidence: Box<NanoAccessoryShutdownEvidence>,
    },
}

#[derive(Debug)]
pub enum NanoAccessoryWorkerStartError {
    ThreadSpawn(std::io::Error),
    StartupFailed(Box<NanoAccessoryWorkerExit>),
    ThreadPanickedBeforeReadiness,
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
            Self::StartupFailed(_) | Self::ThreadPanickedBeforeReadiness => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoAccessoryWorkerJoinError {
    ThreadPanicked,
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
pub enum NanoAccessoryHealthStatusError {
    Poisoned,
}

impl fmt::Display for NanoAccessoryHealthStatusError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano accessory head-health status unavailable: {self:?}"
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
    health_period: NanoAccessoryHealthPeriod,
}

impl NanoAccessoryHealthObserver {
    pub fn snapshot(&self) -> Result<NanoAccessoryRuntimeHealth, NanoAccessoryHealthStatusError> {
        let frames = self.channel.counters.snapshot();
        let terminal_fault_latched = !self.channel.accepting_frames.load(Ordering::Acquire)
            && !self.channel.shutdown_requested.load(Ordering::Acquire);
        if terminal_fault_latched {
            return Ok(NanoAccessoryRuntimeHealth {
                head: NanoAccessoryComponentHealth::Faulted,
                eyes: NanoAccessoryComponentHealth::Faulted,
                rgb_expression: NanoAccessoryComponentHealth::Faulted,
                successful_rgb_expression_frames: frames.processed_successfully,
            });
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
        Ok(NanoAccessoryRuntimeHealth {
            head,
            eyes: NanoAccessoryComponentHealth::Ready,
            rgb_expression: if frames.processed_successfully == 0 {
                NanoAccessoryComponentHealth::Degraded
            } else {
                NanoAccessoryComponentHealth::Ready
            },
            successful_rgb_expression_frames: frames.processed_successfully,
        })
    }
}

enum StartupSignal {
    Ready(Box<NanoAccessoryReadyEvidence>),
    Failed,
}

/// Running worker plus its sole RGB ingress.
///
/// # Drop behavior
///
/// Accidental `Drop` disconnects RGB and detaches the worker thread. The
/// detached thread latches the disconnect, keeps owning the head bus, and
/// continues bounded health checks; it does **not** request torque disable.
/// The in-object fault receiver is dropped too, so no caller can observe that
/// publication or later coordinate shutdown through this worker. Process
/// termination still does not prove the resulting physical torque state. A
/// service owner must retain this value and call [`Self::shutdown`].
#[must_use = "the accessory owner must be retained and explicitly shut down"]
pub struct NanoAccessoryWorker {
    ready: NanoAccessoryReadyEvidence,
    ingress: Option<NanoAccessoryRgbIngress>,
    channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>>,
    latest_head_health: Arc<Mutex<NanoAccessoryHeadHealthState>>,
    fault_rx: crossbeam_channel::Receiver<NanoAccessoryTerminalFault>,
    thread: Option<JoinHandle<NanoAccessoryWorkerExit>>,
}

impl NanoAccessoryWorker {
    /// Start the dedicated runtime and block until eye startup plus the head's
    /// reviewed return and immediate exact-target health check have produced
    /// evidence, or startup has failed.
    pub fn start(config: NanoAccessoryWorkerConfig) -> Result<Self, NanoAccessoryWorkerStartError> {
        let expression_clock = TokioClock::new();
        let channel = Arc::new(LatestFrameChannel::new());
        let ingress = NanoAccessoryRgbIngress {
            channel: Arc::clone(&channel),
            clock: expression_clock.clone(),
            connected: true,
        };
        let (startup_tx, startup_rx) = std::sync::mpsc::sync_channel(1);
        let (fault_tx, fault_rx) = crossbeam_channel::bounded(1);
        let worker_channel = Arc::clone(&channel);
        let latest_head_health = Arc::new(Mutex::new(NanoAccessoryHeadHealthState::empty()));
        let worker_head_health = Arc::clone(&latest_head_health);
        let thread = thread::Builder::new()
            .name("kiko-nano-accessories".into())
            .spawn(move || {
                run_production_worker(
                    config,
                    worker_channel,
                    worker_head_health,
                    startup_tx,
                    fault_tx,
                    expression_clock,
                )
            })
            .map_err(NanoAccessoryWorkerStartError::ThreadSpawn)?;

        match startup_rx.recv() {
            Ok(StartupSignal::Ready(ready)) => Ok(Self {
                ready: *ready,
                ingress: Some(ingress),
                channel,
                latest_head_health,
                fault_rx,
                thread: Some(thread),
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

    pub const fn readiness(&self) -> &NanoAccessoryReadyEvidence {
        &self.ready
    }

    pub fn submit_rgb(&mut self, frame: ImageFrame) -> NanoAccessoryFrameSubmitOutcome {
        match &mut self.ingress {
            Some(ingress) => ingress.submit(frame),
            None => NanoAccessoryFrameSubmitOutcome::IngressDisconnected,
        }
    }

    pub fn frame_stats(&self) -> NanoAccessoryFrameStats {
        self.channel.counters.snapshot()
    }

    /// Borrow the running worker's status without duplicating any device owner.
    pub fn health_observer(&self) -> NanoAccessoryHealthObserver {
        NanoAccessoryHealthObserver {
            channel: Arc::clone(&self.channel),
            head: Arc::clone(&self.latest_head_health),
            health_period: self.ready.health_period(),
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
    pub fn shutdown(mut self) -> Result<NanoAccessoryWorkerExit, NanoAccessoryWorkerJoinError> {
        self.channel.request_shutdown();
        self.ingress.take();
        self.thread
            .take()
            .expect("running worker owns one thread")
            .join()
            .map_err(|_| NanoAccessoryWorkerJoinError::ThreadPanicked)
    }
}

// Intentionally no shutdown in Drop. A service owner must coordinate physical
// support and call `shutdown`; an accidental handle drop may not silently
// remove neck torque.

trait ReviewedNaturalHeadPort {
    type Ready;
    type StartError;
    type Health;
    type HealthError: Clone;
    type Shutdown;

    async fn start(&mut self) -> Result<Self::Ready, Self::StartError>;
    async fn check_health(&mut self) -> Result<Self::Health, Self::HealthError>;
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

    fn process(&mut self, frame: F) -> Result<Self::Intent, Self::Error>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CoreTerminalFault<H, B, E> {
    HeadHealth(H),
    HeadHealthStatusPoisoned,
    Bridge(B),
    EyeApply(E),
    IngressDisconnected,
    ChannelPoisoned,
    ReadinessObserverDropped,
}

enum CoreExit<ES, HS, H, B, E, EyeShutdown, HeadShutdown> {
    EyeStartupFailed(ES),
    HeadStartupFailed {
        source: HS,
        eye_shutdown: EyeShutdown,
    },
    Shutdown {
        terminal_fault: Option<CoreTerminalFault<H, B, E>>,
        eye_shutdown: EyeShutdown,
        head_shutdown: HeadShutdown,
    },
}

struct CoreObservers<Ready, RecordHealth, Fault> {
    ready: Ready,
    record_health: RecordHealth,
    publish_fault: Fault,
}

async fn run_accessory_core<F, H, E, B, Ready, RecordHealth, Fault>(
    mut head: H,
    mut eye: E,
    mut bridge: B,
    channel: Arc<LatestFrameChannel<F>>,
    health_period: NanoAccessoryHealthPeriod,
    observers: CoreObservers<Ready, RecordHealth, Fault>,
) -> CoreExit<
    E::StartError,
    H::StartError,
    H::HealthError,
    B::Error,
    E::ApplyError,
    E::Shutdown,
    H::Shutdown,
>
where
    F: Send + 'static,
    H: ReviewedNaturalHeadPort,
    E: Kep2EyePort<B::Intent>,
    B: RgbBridgePort<F>,
    Ready: FnOnce(H::Ready, E::Ready) -> bool,
    RecordHealth: FnMut(H::Health) -> bool,
    Fault: FnMut(CoreTerminalFault<H::HealthError, B::Error, E::ApplyError>),
{
    let CoreObservers {
        ready,
        mut record_health,
        mut publish_fault,
    } = observers;
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

    let mut terminal_fault = if ready(head_ready, eye_ready) {
        None
    } else {
        let fault = CoreTerminalFault::ReadinessObserverDropped;
        publish_fault(fault.clone());
        channel.latch_terminal_fault();
        Some(fault)
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
                match health_result {
                    Ok(evidence) => {
                        if record_health(evidence) {
                            None
                        } else {
                            Some(CoreTerminalFault::HeadHealthStatusPoisoned)
                        }
                    }
                    Err(source) => Some(CoreTerminalFault::HeadHealth(source)),
                }
            }
            event = channel.next_event() => {
                match event {
                    LatestFrameEvent::Frame(frame) => {
                        match bridge.process(frame) {
                            Ok(intent) => match eye.apply(intent).await {
                                Ok(()) => {
                                    saturating_increment(
                                        &channel.counters.processed_successfully,
                                    );
                                    None
                                }
                                Err(source) => Some(CoreTerminalFault::EyeApply(source)),
                            },
                            Err(source) => Some(CoreTerminalFault::Bridge(source)),
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
            channel.latch_terminal_fault();
            terminal_fault = Some(fault);
        }
    }

    // Explicit coordinated shutdown releases eyes first, then preserves the
    // complete result of the head actor's existing shutdown transaction.
    let eye_shutdown = eye.shutdown().await;
    let head_shutdown = head.shutdown().await;
    CoreExit::Shutdown {
        terminal_fault,
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
    clock: Option<TokioClock>,
    active: Option<ActiveEyeActor>,
}

impl SerialKep2EyePort {
    fn new(config: EyeRuntimeConfig, clock: TokioClock) -> Self {
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

struct ActiveHeadActor {
    handle: HeadReturnActorHandle,
    task: HeadActorTask,
    startup: VerifiedNaturalHoldEvidence,
    head_return: VerifiedHeadReturnEvidence,
}

struct SerialReviewedNaturalHeadPort {
    config: Option<ReturnToTargetConfig>,
    torque_consent: PhysicalTorqueEnableConsent,
    motion_consent: PhysicalHeadMotionConsent,
    takeover_consent: ProductionTensionPreservingTakeoverConsent,
    required_hold_target: HeadHoldTarget,
    active: Option<ActiveHeadActor>,
}

impl SerialReviewedNaturalHeadPort {
    fn new(
        config: ReturnToTargetConfig,
        torque_consent: PhysicalTorqueEnableConsent,
        motion_consent: PhysicalHeadMotionConsent,
        takeover_consent: ProductionTensionPreservingTakeoverConsent,
        required_hold_target: HeadHoldTarget,
    ) -> Self {
        Self {
            config: Some(config),
            torque_consent,
            motion_consent,
            takeover_consent,
            required_hold_target,
            active: None,
        }
    }

    fn require_hold_target(
        &self,
        evidence: VerifiedHeadHealthEvidence,
    ) -> Result<VerifiedHeadHealthEvidence, NanoHeadHealthError> {
        let observed = evidence.hold_target();
        if observed == self.required_hold_target {
            Ok(evidence)
        } else {
            Err(NanoHeadHealthError::UnexpectedHoldTarget {
                required: self.required_hold_target,
                observed,
            })
        }
    }
}

impl ReviewedNaturalHeadPort for SerialReviewedNaturalHeadPort {
    type Ready = NanoHeadReadyEvidence;
    type StartError = NanoHeadActorStartupError;
    type Health = VerifiedHeadHealthEvidence;
    type HealthError = NanoHeadHealthError;
    type Shutdown = NanoHeadShutdownEvidence;

    async fn start(&mut self) -> Result<Self::Ready, Self::StartError> {
        let config = self.config.take().expect("head starts exactly once");
        let (serial, handle, receipt, task) =
            kiko_head_runtime::start_serial_tension_preserving_head_return_actor(
                config,
                self.torque_consent,
                self.motion_consent,
                self.takeover_consent,
            )
            .map_err(NanoHeadActorStartupError::Start)?;
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

impl RgbBridgePort<IngressObservedRgbFrame<ImageFrame>> for RgbExpressionBridge<TokioClock> {
    type Intent = PreparedEyeIntent;
    type Error = RgbExpressionBridgeError;

    fn process(
        &mut self,
        frame: IngressObservedRgbFrame<ImageFrame>,
    ) -> Result<Self::Intent, Self::Error> {
        self.process_queued_oak_frame(&frame)
            .map(|outcome| outcome.into_prepared())
    }
}

fn process_rgb_work<F, B>(
    bridge: &mut B,
    work: Result<F, ClockError>,
) -> Result<B::Intent, RgbExpressionBridgeError>
where
    B: RgbBridgePort<F, Error = RgbExpressionBridgeError>,
{
    match work {
        Ok(frame) => bridge.process(frame),
        Err(source) => Err(RgbExpressionBridgeError::Clock(source)),
    }
}

impl RgbBridgePort<NanoAccessoryRgbWork> for RgbExpressionBridge<TokioClock> {
    type Intent = PreparedEyeIntent;
    type Error = RgbExpressionBridgeError;

    fn process(&mut self, work: NanoAccessoryRgbWork) -> Result<Self::Intent, Self::Error> {
        process_rgb_work(self, work)
    }
}

fn run_production_worker(
    config: NanoAccessoryWorkerConfig,
    channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>>,
    latest_head_health: Arc<Mutex<NanoAccessoryHeadHealthState>>,
    startup_tx: std::sync::mpsc::SyncSender<StartupSignal>,
    fault_tx: crossbeam_channel::Sender<NanoAccessoryTerminalFault>,
    expression_clock: TokioClock,
) -> NanoAccessoryWorkerExit {
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
    let eye_clock = expression_clock.clone();
    let bridge =
        RgbExpressionBridge::new(config.stream_epoch, config.rgb_expression, expression_clock);
    let eye = SerialKep2EyePort::new(eye_config, eye_clock);
    let head = SerialReviewedNaturalHeadPort::new(
        config.head_return,
        config.head_torque_consent,
        config.head_motion_consent,
        config.head_takeover_consent,
        config.required_hold_target,
    );
    let stream_epoch = config.stream_epoch;
    let health_period = config.health_period;
    let readiness_head_health = Arc::clone(&latest_head_health);

    let core_exit = runtime.block_on(run_accessory_core(
        head,
        eye,
        bridge,
        channel,
        health_period,
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
                startup_tx
                    .send(StartupSignal::Ready(
                        NanoAccessoryReadyEvidence {
                            eye,
                            head,
                            stream_epoch,
                            health_period,
                        }
                        .into(),
                    ))
                    .is_ok()
            },
            record_health: move |evidence| match latest_head_health.lock() {
                Ok(mut latest) => {
                    latest.record(evidence);
                    true
                }
                Err(_) => false,
            },
            publish_fault: move |fault| {
                let fault = match fault {
                    CoreTerminalFault::HeadHealth(source) => {
                        NanoAccessoryTerminalFault::HeadHealth(source)
                    }
                    CoreTerminalFault::HeadHealthStatusPoisoned => {
                        NanoAccessoryTerminalFault::HeadHealthStatusPoisoned
                    }
                    CoreTerminalFault::Bridge(source) => {
                        NanoAccessoryTerminalFault::ExpressionBridge(source)
                    }
                    CoreTerminalFault::EyeApply(source) => {
                        NanoAccessoryTerminalFault::EyeApply(source)
                    }
                    CoreTerminalFault::IngressDisconnected => {
                        NanoAccessoryTerminalFault::RgbIngressDisconnected
                    }
                    CoreTerminalFault::ChannelPoisoned => {
                        NanoAccessoryTerminalFault::RgbChannelPoisoned
                    }
                    CoreTerminalFault::ReadinessObserverDropped => {
                        NanoAccessoryTerminalFault::ReadinessObserverDropped
                    }
                };
                let _ = fault_tx.try_send(fault);
            },
        },
    ));

    match core_exit {
        CoreExit::EyeStartupFailed(source) => {
            // No ready signal was sent; wake the synchronous starter.
            // Sending can fail only if that starter disappeared.
            // `startup_tx` was moved into the readiness closure, so the channel
            // also disconnects here and wakes `recv`.
            NanoAccessoryWorkerExit::EyeStartupFailed(Box::new(source))
        }
        CoreExit::HeadStartupFailed {
            source,
            eye_shutdown,
        } => NanoAccessoryWorkerExit::HeadStartupFailed {
            source: Box::new(source),
            eye_shutdown: Box::new(eye_shutdown),
        },
        CoreExit::Shutdown {
            terminal_fault,
            eye_shutdown,
            head_shutdown,
        } => {
            let terminal_fault = terminal_fault.map(|fault| match fault {
                CoreTerminalFault::HeadHealth(source) => {
                    NanoAccessoryTerminalFault::HeadHealth(source)
                }
                CoreTerminalFault::HeadHealthStatusPoisoned => {
                    NanoAccessoryTerminalFault::HeadHealthStatusPoisoned
                }
                CoreTerminalFault::Bridge(source) => {
                    NanoAccessoryTerminalFault::ExpressionBridge(source)
                }
                CoreTerminalFault::EyeApply(source) => NanoAccessoryTerminalFault::EyeApply(source),
                CoreTerminalFault::IngressDisconnected => {
                    NanoAccessoryTerminalFault::RgbIngressDisconnected
                }
                CoreTerminalFault::ChannelPoisoned => {
                    NanoAccessoryTerminalFault::RgbChannelPoisoned
                }
                CoreTerminalFault::ReadinessObserverDropped => {
                    NanoAccessoryTerminalFault::ReadinessObserverDropped
                }
            });
            NanoAccessoryWorkerExit::Shutdown {
                terminal_fault,
                evidence: Box::new(NanoAccessoryShutdownEvidence {
                    eye: eye_shutdown,
                    head: head_shutdown,
                }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn health_period(milliseconds: u64) -> NanoAccessoryHealthPeriod {
        NanoAccessoryHealthPeriod::try_from_duration(Duration::from_millis(milliseconds))
            .expect("valid test period")
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
    async fn ingress_clock_failure_stays_typed_and_replacement_counts_as_queue_ownership() {
        let failure = match observe_rgb_at_ingress(&FailingIngressClock, 1_u64) {
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
            channel.submit(Err(failure)),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        assert_eq!(
            channel.submit(Ok(IngressObservedRgbFrame::new(
                2_u64,
                kiko_expression_core::MonotonicTimestamp::from_nanos_since_epoch(17),
            ))),
            NanoAccessoryFrameSubmitOutcome::ReplacedOlderFrame
        );
        let LatestFrameEvent::Frame(Ok(frame)) = channel.next_event().await else {
            panic!("newer observed frame must replace the older queued clock failure");
        };
        assert_eq!(*frame.frame(), 2);
        assert_eq!(frame.observed_at().nanos_since_epoch(), 17);
        assert_eq!(
            channel.counters.snapshot(),
            NanoAccessoryFrameStats {
                enqueued: 2,
                replaced_older: 1,
                ..NanoAccessoryFrameStats::default()
            }
        );
    }

    #[derive(Clone)]
    struct FakeHead {
        log: Arc<Mutex<Vec<&'static str>>>,
        health_calls: Arc<AtomicUsize>,
        fail_start: bool,
        fail_return: bool,
        fail_health_at: Option<usize>,
    }

    impl ReviewedNaturalHeadPort for FakeHead {
        type Ready = &'static str;
        type StartError = &'static str;
        type Health = usize;
        type HealthError = &'static str;
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

        async fn shutdown(&mut self) -> Self::Shutdown {
            self.log.lock().unwrap().push("head_shutdown");
            "head_shutdown"
        }
    }

    struct FakeEye {
        log: Arc<Mutex<Vec<&'static str>>>,
        fail_start: bool,
        fail_apply: bool,
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

        async fn apply(&mut self, _intent: u64) -> Result<(), Self::ApplyError> {
            self.log.lock().unwrap().push("eye_apply");
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

        fn process(&mut self, frame: u64) -> Result<Self::Intent, Self::Error> {
            self.log.lock().unwrap().push("bridge");
            if self.fail { Err("bridge") } else { Ok(frame) }
        }
    }

    struct ClockWorkBridge {
        log: Arc<Mutex<Vec<&'static str>>>,
    }

    impl RgbBridgePort<u64> for ClockWorkBridge {
        type Intent = u64;
        type Error = RgbExpressionBridgeError;

        fn process(&mut self, frame: u64) -> Result<Self::Intent, Self::Error> {
            self.log.lock().unwrap().push("clock_bridge_frame");
            Ok(frame)
        }
    }

    impl RgbBridgePort<Result<u64, ClockError>> for ClockWorkBridge {
        type Intent = u64;
        type Error = RgbExpressionBridgeError;

        fn process(&mut self, work: Result<u64, ClockError>) -> Result<Self::Intent, Self::Error> {
            process_rgb_work(self, work)
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
                fail_health_at,
            },
            FakeEye {
                log: Arc::clone(&log),
                fail_start: fail_eye_start,
                fail_apply: fail_eye_apply,
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
                CoreObservers {
                    ready: move |head, eye| {
                        assert_eq!(head, "head_ready");
                        assert_eq!(eye, "eye_ready");
                        ready_flag.store(true, Ordering::SeqCst);
                        true
                    },
                    record_health: |_| true,
                    publish_fault: |_| {},
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
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: |_| {},
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
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: |_| {},
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

    #[test]
    fn accessory_observer_distinguishes_startup_degraded_expression_and_terminal_fault() {
        let channel: Arc<LatestFrameChannel<NanoAccessoryRgbWork>> =
            Arc::new(LatestFrameChannel::new());
        let observer = NanoAccessoryHealthObserver {
            channel: Arc::clone(&channel),
            head: Arc::new(Mutex::new(NanoAccessoryHeadHealthState::empty())),
            health_period: health_period(50),
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
        saturating_increment(&channel.counters.processed_successfully);
        assert_eq!(
            observer.snapshot().unwrap().rgb_expression,
            NanoAccessoryComponentHealth::Ready
        );
        channel.latch_terminal_fault();
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
            CoreObservers {
                ready: move |_, _| {
                    ready_flag.store(true, Ordering::SeqCst);
                    true
                },
                record_health: |_| true,
                publish_fault: |_| {},
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
            CoreObservers {
                ready: |_, _| panic!("head startup failure cannot report ready"),
                record_health: |_| true,
                publish_fault: |_| {},
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
            CoreObservers {
                ready: move |_, _| {
                    ready_flag.store(true, Ordering::SeqCst);
                    true
                },
                record_health: |_| true,
                publish_fault: |_| {},
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
                CoreObservers {
                    ready: |_, _| true,
                    record_health: move |evidence| {
                        task_latest_health.store(evidence, Ordering::SeqCst);
                        true
                    },
                    publish_fault: |_| {},
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
        fail_bridge: bool,
        fail_eye_apply: bool,
        expected: CoreTerminalFault<&'static str, &'static str, &'static str>,
    ) {
        let (head, eye, bridge, log, health_calls) =
            fakes(false, false, fail_bridge, fail_eye_apply, None);
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
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: move |fault| {
                        fault_tx.send(fault).unwrap();
                    },
                },
            )
            .await
        });
        assert_eq!(
            channel.submit(7_u64),
            NanoAccessoryFrameSubmitOutcome::Enqueued
        );
        assert_eq!(fault_rx.recv().await, Some(expected));
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
        assert_fault_does_not_shutdown_head(true, false, CoreTerminalFault::Bridge("bridge")).await;
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
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: move |fault| {
                        fault_tx.send(fault).unwrap();
                    },
                },
            )
            .await
        });
        assert_eq!(
            channel.submit(Err(ClockError::ElapsedNanosecondsOutOfRange {
                elapsed_nanoseconds: u128::MAX,
            })),
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
        assert_fault_does_not_shutdown_head(false, true, CoreTerminalFault::EyeApply("eye_apply"))
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
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: move |fault| fault_tx.send(fault).unwrap(),
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
                CoreObservers {
                    ready: |_, _| true,
                    record_health: |_| true,
                    publish_fault: move |fault| fault_tx.send(fault).unwrap(),
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
