//! Synchronous bridge from safety-approved navigation decisions to exact V2
//! controller application evidence.
//!
//! This module does not alter MPC, collision checking, or shadow recording.
//! Physical mode consumes the already-recorded safety decision, requires the
//! previous applied receipt before the next solve, and returns only after the
//! exact requested PWM has been acknowledged by the controller.

use std::fmt;
#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
use std::sync::Arc;
#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::Instant;

#[cfg(feature = "agent-runtime")]
use kiko_head_runtime::{
    HeadGazeBaseCommandTransaction, HeadGazeBaseInterlockError, HeadGazeBaseMotionInterlock,
    HeadGazeBaseZeroExclusiveLeaseIssuer,
};
#[cfg(feature = "agent-runtime")]
use kiko_supervisor_core::{
    ConfirmedBaseZero, MonotonicInstant as SupervisorMonotonicInstant, ZeroEvidenceError,
};
use robot_command_client::{
    AcquireFailure, AppliedCommandReceipt, ApplyFailure, ArmedCommandClient, ClientConfig,
    DisarmFailure, DisarmReceipt, MonotonicClock, MonotonicInstant, PendingPhysicalCommand,
    RobotProtocolV2WireAdapter, UdpTransportBuildError, UdpV2Transport,
    VerifiedControllerAcquisition,
};
use robot_protocol::v2::{DomainError, ForceStopReason, TimerPwm, V2CommandLeaseMs};

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
use super::WheelsOffQualificationHostClockFaultInjection;
use super::actuation_config::NavigationActuationConfigV1;
#[cfg(feature = "nano-attended-navigation-trial")]
use super::nano_base_commissioning_bootstrap::MAX_WHEEL_ON_COMMISSIONING_PWM_PERCENT;
use super::{SafetyDecision, SafetyDecisionOutcome, ShadowPwmPair};
use crate::HostMonotonicTimestamp;

type ConcreteTransport = UdpV2Transport<RobotProtocolV2WireAdapter>;
type ConcreteArmed = ArmedCommandClient<ConcreteTransport, ActuationMonotonicClock>;
#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
type ConcreteDisarmed =
    robot_command_client::DisarmedCommandClient<ConcreteTransport, ActuationMonotonicClock>;
type ConcreteAcquireFailure = AcquireFailure<ConcreteTransport, ActuationMonotonicClock>;
type ConcreteApplyFailure = ApplyFailure<ConcreteTransport, ActuationMonotonicClock>;
type ConcreteDisarmFailure = DisarmFailure<ConcreteTransport, ActuationMonotonicClock>;

/// A monotonic domain sharing the live navigation clock's exact origin.
#[derive(Clone, Debug)]
#[doc(hidden)]
pub struct ActuationMonotonicClock {
    origin: Instant,
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    qualification_fault: Option<Arc<QualificationClockFaultState>>,
}

impl ActuationMonotonicClock {
    const fn new(origin: Instant) -> Self {
        Self {
            origin,
            #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
            qualification_fault: None,
        }
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    fn new_candidate(
        origin: Instant,
        fault: Option<WheelsOffQualificationHostClockFaultInjection>,
    ) -> Self {
        Self {
            origin,
            qualification_fault: fault.map(|_| Arc::new(QualificationClockFaultState::new())),
        }
    }

    fn host_now(&self) -> Result<HostMonotonicTimestamp, PhysicalDecisionError> {
        let nanoseconds = u64::try_from(self.origin.elapsed().as_nanos())
            .map_err(|_| PhysicalDecisionError::HostTimestampOutOfRange)?;
        #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
        let nanoseconds = if self.qualification_fault.is_some() {
            nanoseconds
                .checked_add(QUALIFICATION_CLOCK_BIAS_NS)
                .ok_or(PhysicalDecisionError::HostTimestampOutOfRange)?
        } else {
            nanoseconds
        };
        Ok(HostMonotonicTimestamp::from_nanos(nanoseconds))
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    fn arm_qualification_regression(&self) {
        if let Some(fault) = &self.qualification_fault {
            fault.arm();
        }
    }
}

impl MonotonicClock for ActuationMonotonicClock {
    fn now(&self) -> MonotonicInstant {
        #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
        if let Some(fault) = &self.qualification_fault {
            let elapsed = u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX);
            let normal = elapsed.saturating_add(QUALIFICATION_CLOCK_BIAS_NS);
            return MonotonicInstant::from_nanos_since_clock_start(u128::from(fault.now(normal)));
        }
        MonotonicInstant::from_nanos_since_clock_start(self.origin.elapsed().as_nanos())
    }
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
const QUALIFICATION_CLOCK_BIAS_NS: u64 = 1_000_000_000;
#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
const QUALIFICATION_CLOCK_DORMANT: u8 = 0;
#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
const QUALIFICATION_CLOCK_ARMED: u8 = 1;
#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
const QUALIFICATION_CLOCK_FIRED: u8 = 2;

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
#[derive(Debug)]
struct QualificationClockFaultState {
    phase: AtomicU8,
    greatest_normal_ns: AtomicU64,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl QualificationClockFaultState {
    const fn new() -> Self {
        Self {
            phase: AtomicU8::new(QUALIFICATION_CLOCK_DORMANT),
            greatest_normal_ns: AtomicU64::new(QUALIFICATION_CLOCK_BIAS_NS),
        }
    }

    fn arm(&self) {
        let _ = self.phase.compare_exchange(
            QUALIFICATION_CLOCK_DORMANT,
            QUALIFICATION_CLOCK_ARMED,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
    }

    fn now(&self, normal_ns: u64) -> u64 {
        if self
            .phase
            .compare_exchange(
                QUALIFICATION_CLOCK_ARMED,
                QUALIFICATION_CLOCK_FIRED,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok()
        {
            self.greatest_normal_ns
                .load(Ordering::SeqCst)
                .checked_sub(1)
                .expect("qualification clock bias makes regression representable")
        } else {
            let previous = self
                .greatest_normal_ns
                .fetch_max(normal_ns, Ordering::SeqCst);
            previous.max(normal_ns)
        }
    }
}

#[derive(Clone, Copy)]
struct ActuationTiming {
    apply_ack_budget_ns: u64,
    lease: V2CommandLeaseMs,
}

impl ActuationTiming {
    fn from_authority(config: &NavigationActuationConfigV1) -> Self {
        Self {
            apply_ack_budget_ns: config.apply_ack_budget().get(),
            lease: config.controller_motion_lease(),
        }
    }

    #[cfg(all(
        any(
            feature = "nano-wheels-off-qualification",
            feature = "nano-base-commissioning"
        ),
        unix
    ))]
    fn from_zero_client(config: &ClientConfig) -> Self {
        Self {
            apply_ack_budget_ns: config.applied_ack_timeout().get(),
            lease: config.zero_acquisition_lease(),
        }
    }
}

#[derive(Clone, Copy)]
enum PhysicalDecisionInput {
    Controller {
        recorded_pwm: ShadowPwmPair,
        requested_pwm: ShadowPwmPair,
        collision_valid_through: HostMonotonicTimestamp,
    },
    Stopped {
        recorded_pwm: ShadowPwmPair,
    },
}

impl PhysicalDecisionInput {
    fn from_safety(decision: &SafetyDecision) -> Self {
        let recorded_pwm = decision.record().pwm();
        match decision.outcome() {
            SafetyDecisionOutcome::Controller(controller) => Self::Controller {
                recorded_pwm,
                requested_pwm: controller.requested_pwm(),
                collision_valid_through: controller
                    .final_validation()
                    .collision_snapshot()
                    .valid_through(),
            },
            SafetyDecisionOutcome::Stopped(_) => Self::Stopped { recorded_pwm },
        }
    }
}

fn timer_pwm(pwm: ShadowPwmPair) -> Result<TimerPwm, PhysicalDecisionError> {
    TimerPwm::try_new(pwm.left().get(), pwm.right().get()).map_err(PhysicalDecisionError::TimerPwm)
}

fn pending_command(
    timing: ActuationTiming,
    decision: PhysicalDecisionInput,
    now: HostMonotonicTimestamp,
) -> Result<PendingPhysicalCommand, PhysicalDecisionError> {
    let (pwm, acknowledgement_deadline_ns) = match decision {
        PhysicalDecisionInput::Controller {
            recorded_pwm,
            requested_pwm,
            collision_valid_through,
        } => {
            if recorded_pwm != requested_pwm {
                return Err(PhysicalDecisionError::RecordedControllerPwmMismatch {
                    recorded: recorded_pwm,
                    requested: requested_pwm,
                });
            }
            // Collision validity is an exclusive admission deadline for this
            // decision and its exact applied ACK. It does not claim that a
            // dynamic world remains unchanged for the controller lease. The
            // lease independently bounds failover motion while the next
            // periodic decision is expected to refresh obstacle evidence.
            if now >= collision_valid_through {
                return Err(PhysicalDecisionError::CollisionValidityExpired {
                    now,
                    collision_valid_through,
                });
            }
            (recorded_pwm, collision_valid_through.as_nanos())
        }
        PhysicalDecisionInput::Stopped { recorded_pwm } => {
            if !recorded_pwm.is_stop() {
                return Err(PhysicalDecisionError::StoppedDecisionRecordedNonzero(
                    recorded_pwm,
                ));
            }
            let deadline = now
                .as_nanos()
                .checked_add(timing.apply_ack_budget_ns)
                .ok_or(PhysicalDecisionError::DeadlineArithmeticOverflow)?;
            (recorded_pwm, deadline)
        }
    };
    Ok(PendingPhysicalCommand::new(
        timer_pwm(pwm)?,
        timing.lease,
        MonotonicInstant::from_nanos_since_clock_start(u128::from(acknowledgement_deadline_ns)),
    ))
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
fn candidate_pending_command(
    clock: &ActuationMonotonicClock,
    timing: ActuationTiming,
    timer_pwm: TimerPwm,
) -> Result<PendingPhysicalCommand, PhysicalDecisionError> {
    if !timer_pwm.is_zero() {
        clock.arm_qualification_regression();
    }
    let now = clock.host_now()?;
    let acknowledgement_deadline_ns = now
        .as_nanos()
        .checked_add(timing.apply_ack_budget_ns)
        .ok_or(PhysicalDecisionError::DeadlineArithmeticOverflow)?;
    Ok(PendingPhysicalCommand::new(
        timer_pwm,
        timing.lease,
        MonotonicInstant::from_nanos_since_clock_start(u128::from(acknowledgement_deadline_ns)),
    ))
}

/// Sole owner of an armed physical session in the navigation worker.
///
/// `armed == None` means the session was consumed by a terminal failure or an
/// explicit disarm. There is no API that can recreate authority in place.
pub struct PhysicalActuationSession {
    armed: Option<ConcreteArmed>,
    clock: ActuationMonotonicClock,
    timing: ActuationTiming,
    #[cfg(feature = "agent-runtime")]
    head_gaze_interlock: Option<HeadGazeBaseMotionInterlock>,
    #[cfg(feature = "agent-runtime")]
    head_gaze_interlock_installation_open: bool,
    #[cfg(feature = "nano-attended-navigation-trial")]
    attended_trial: Option<AttendedTrialActuationGuard>,
}

/// Immutable second-line guard for the attended navigation trial.
///
/// The STM32 commissioning profile independently enforces the same PWM cap.
/// Keeping the deadline and cap on the physical session also makes it
/// impossible for an MPC call site to forget either check.
#[cfg(feature = "nano-attended-navigation-trial")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct AttendedTrialActuationGuard {
    maximum_abs_pwm_percent: u8,
    issued_at_ns: u64,
    expires_at_ns: u64,
}

#[cfg(feature = "nano-attended-navigation-trial")]
impl AttendedTrialActuationGuard {
    pub(super) fn try_new(
        maximum_abs_pwm_percent: u8,
        issued_at_ns: u64,
        expires_at_ns: u64,
    ) -> Result<Self, AttendedTrialActuationGuardError> {
        if maximum_abs_pwm_percent == 0
            || maximum_abs_pwm_percent > MAX_WHEEL_ON_COMMISSIONING_PWM_PERCENT
        {
            return Err(AttendedTrialActuationGuardError::InvalidMaximumPwm(
                maximum_abs_pwm_percent,
            ));
        }
        if issued_at_ns == 0 || expires_at_ns <= issued_at_ns {
            return Err(AttendedTrialActuationGuardError::InvalidLifetime {
                issued_at_ns,
                expires_at_ns,
            });
        }
        Ok(Self {
            maximum_abs_pwm_percent,
            issued_at_ns,
            expires_at_ns,
        })
    }

    fn require_current(self, now: HostMonotonicTimestamp) -> Result<(), PhysicalDecisionError> {
        if now.as_nanos() < self.issued_at_ns {
            return Err(PhysicalDecisionError::AttendedTrialClockBeforeAttestation {
                now,
                issued_at_ns: self.issued_at_ns,
            });
        }
        if now.as_nanos() >= self.expires_at_ns {
            return Err(PhysicalDecisionError::AttendedTrialExpired {
                now,
                expires_at_ns: self.expires_at_ns,
            });
        }
        Ok(())
    }

    fn require_pwm(self, pwm: ShadowPwmPair) -> Result<(), PhysicalDecisionError> {
        let maximum = i16::from(self.maximum_abs_pwm_percent);
        let left = pwm.left().get();
        let right = pwm.right().get();
        if i16::from(left).abs() > maximum || i16::from(right).abs() > maximum {
            return Err(PhysicalDecisionError::AttendedTrialPwmAboveLimit {
                left,
                right,
                maximum_abs_pwm_percent: self.maximum_abs_pwm_percent,
            });
        }
        Ok(())
    }
}

#[cfg(feature = "nano-attended-navigation-trial")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AttendedTrialActuationGuardError {
    InvalidMaximumPwm(u8),
    InvalidLifetime {
        issued_at_ns: u64,
        expires_at_ns: u64,
    },
}

#[cfg(feature = "nano-attended-navigation-trial")]
impl fmt::Display for AttendedTrialActuationGuardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid attended navigation trial actuation guard: {self:?}"
        )
    }
}

#[cfg(feature = "nano-attended-navigation-trial")]
impl std::error::Error for AttendedTrialActuationGuardError {}

/// Linear ownership of one exactly stopped candidate transport.
///
/// This token is intentionally candidate-only and non-cloneable. Reacquiring
/// consumes it, so a stopped transport cannot create concurrent armed owners.
#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
pub(super) struct StoppedCandidateActuationClient {
    disarmed: ConcreteDisarmed,
    clock: ActuationMonotonicClock,
    timing: ActuationTiming,
}

#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
impl StoppedCandidateActuationClient {
    pub(super) fn reacquire_zero(
        self,
    ) -> Result<(PhysicalActuationSession, AppliedCommandReceipt), LiveActuationError> {
        let Self {
            disarmed,
            clock,
            timing,
        } = self;
        let (armed, receipt) = disarmed
            .acquire_zero()
            .map_err(LiveActuationError::Acquire)?;
        Ok((
            PhysicalActuationSession {
                armed: Some(armed),
                clock,
                timing,
                #[cfg(feature = "agent-runtime")]
                head_gaze_interlock: None,
                #[cfg(feature = "agent-runtime")]
                head_gaze_interlock_installation_open: true,
                #[cfg(feature = "nano-attended-navigation-trial")]
                attended_trial: None,
            },
            receipt,
        ))
    }
}

impl PhysicalActuationSession {
    pub fn acquire(
        config: &NavigationActuationConfigV1,
        clock_origin: Instant,
    ) -> Result<(Self, AppliedCommandReceipt), LiveActuationError> {
        Self::acquire_from_client_config(
            config.client_config(),
            ActuationMonotonicClock::new(clock_origin),
            ActuationTiming::from_authority(config),
        )
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    pub(super) fn acquire_candidate(
        config: ClientConfig,
        clock_origin: Instant,
        clock_fault: Option<WheelsOffQualificationHostClockFaultInjection>,
    ) -> Result<(Self, AppliedCommandReceipt), LiveActuationError> {
        let timing = ActuationTiming::from_zero_client(&config);
        Self::acquire_from_client_config(
            config,
            ActuationMonotonicClock::new_candidate(clock_origin, clock_fault),
            timing,
        )
    }

    #[cfg(all(feature = "nano-base-commissioning", unix))]
    pub(super) fn acquire_commissioning(
        config: ClientConfig,
        clock_origin: Instant,
    ) -> Result<(Self, AppliedCommandReceipt), LiveActuationError> {
        let timing = ActuationTiming::from_zero_client(&config);
        Self::acquire_from_client_config(config, ActuationMonotonicClock::new(clock_origin), timing)
    }

    fn acquire_from_client_config(
        config: ClientConfig,
        clock: ActuationMonotonicClock,
        timing: ActuationTiming,
    ) -> Result<(Self, AppliedCommandReceipt), LiveActuationError> {
        let transport = UdpV2Transport::connect_canonical(config.endpoint())
            .map_err(LiveActuationError::TransportBuild)?;
        let client =
            robot_command_client::DisarmedCommandClient::new(transport, clock.clone(), config);
        let (armed, receipt) = client.acquire_zero().map_err(LiveActuationError::Acquire)?;
        Ok((
            Self {
                armed: Some(armed),
                clock,
                timing,
                #[cfg(feature = "agent-runtime")]
                head_gaze_interlock: None,
                #[cfg(feature = "agent-runtime")]
                head_gaze_interlock_installation_open: true,
                #[cfg(feature = "nano-attended-navigation-trial")]
                attended_trial: None,
            },
            receipt,
        ))
    }

    /// Must run immediately before each MPC tick. This consumes and restores
    /// the armed type state only while the previous applied evidence is live.
    pub fn require_current_before_solve(&mut self) -> Result<(), LiveActuationError> {
        #[cfg(feature = "nano-attended-navigation-trial")]
        if let Some(guard) = self.attended_trial {
            let now = self.clock.host_now().map_err(|source| {
                let stop_reason = source.force_stop_reason();
                self.reject_local_decision(source, stop_reason)
            })?;
            guard.require_current(now).map_err(|source| {
                let stop_reason = source.force_stop_reason();
                self.reject_local_decision(source, stop_reason)
            })?;
        }
        let armed = self.take_armed()?;
        match armed.require_current_applied_evidence() {
            Ok(armed) => {
                self.armed = Some(armed);
                Ok(())
            }
            Err(failure) => Err(LiveActuationError::Preflight(failure)),
        }
    }

    pub fn apply(
        &mut self,
        decision: &SafetyDecision,
    ) -> Result<AppliedCommandReceipt, LiveActuationError> {
        let now = self.clock.host_now().map_err(|source| {
            let stop_reason = source.force_stop_reason();
            self.reject_local_decision(source, stop_reason)
        })?;
        #[cfg(feature = "nano-attended-navigation-trial")]
        if let Some(guard) = self.attended_trial {
            guard.require_current(now).map_err(|source| {
                let stop_reason = source.force_stop_reason();
                self.reject_local_decision(source, stop_reason)
            })?;
            guard
                .require_pwm(decision.record().pwm())
                .map_err(|source| {
                    let stop_reason = source.force_stop_reason();
                    self.reject_local_decision(source, stop_reason)
                })?;
        }
        let pending = pending_command(
            self.timing,
            PhysicalDecisionInput::from_safety(decision),
            now,
        )
        .map_err(|source| {
            let stop_reason = source.force_stop_reason();
            self.reject_local_decision(source, stop_reason)
        })?;
        self.apply_pending(pending)
    }

    /// Install the shared head/base exclusion boundary exactly once, before
    /// this session has attempted any post-acquisition base transaction.
    ///
    /// The supplied evidence must identify this exact acquired controller
    /// boot and control epoch and must be the initial sequence. Callers derive
    /// it by parsing the acquisition receipt once through
    /// `ConfirmedBaseZero::try_from_host_command_result`.
    #[cfg(feature = "agent-runtime")]
    fn install_head_gaze_base_interlock(
        &mut self,
        initial_zero: ConfirmedBaseZero,
    ) -> Result<HeadGazeBaseZeroExclusiveLeaseIssuer, LiveActuationError> {
        if self.head_gaze_interlock.is_some() {
            return Err(LiveActuationError::HeadGazeInterlockInstallation(
                HeadGazeInterlockInstallationError::AlreadyInstalled,
            ));
        }
        if !self.head_gaze_interlock_installation_open {
            return Err(LiveActuationError::HeadGazeInterlockInstallation(
                HeadGazeInterlockInstallationError::BaseTransactionAlreadyAttempted,
            ));
        }
        let acquisition = self.verified_controller_acquisition()?;
        if initial_zero.controller_uid() != acquisition.controller_uid()
            || initial_zero.controller_boot_id() != acquisition.boot_id()
            || initial_zero.control_epoch() != acquisition.control_epoch()
        {
            return Err(LiveActuationError::HeadGazeInterlockInstallation(
                HeadGazeInterlockInstallationError::ControllerIdentityMismatch,
            ));
        }
        if initial_zero.sequence().get() != 1 {
            return Err(LiveActuationError::HeadGazeInterlockInstallation(
                HeadGazeInterlockInstallationError::NotInitialZeroSequence {
                    sequence: initial_zero.sequence().get(),
                },
            ));
        }
        let (interlock, issuer) = HeadGazeBaseMotionInterlock::from_confirmed_zero(initial_zero);
        self.head_gaze_interlock = Some(interlock);
        self.head_gaze_interlock_installation_open = false;
        Ok(issuer)
    }

    /// Parse the verified acquisition receipt exactly once into the zero
    /// evidence which seeds head/base exclusion.
    #[cfg(feature = "agent-runtime")]
    pub fn install_head_gaze_base_interlock_from_initial_receipt(
        &mut self,
        initial_zero: &AppliedCommandReceipt,
    ) -> Result<HeadGazeBaseZeroExclusiveLeaseIssuer, LiveActuationError> {
        let zero = confirmed_head_gaze_zero(initial_zero)?;
        self.install_head_gaze_base_interlock(zero)
    }

    /// Bind a commissioning-class session to the non-bypassable attended
    /// navigation trial cap and deadline before exposing it to MPC.
    #[cfg(feature = "nano-attended-navigation-trial")]
    pub(super) fn bind_attended_navigation_trial(
        mut self,
        guard: AttendedTrialActuationGuard,
    ) -> Self {
        self.attended_trial = Some(guard);
        self
    }

    /// Submit a newly sequenced zero command and retain its exact applied receipt.
    ///
    /// This is the physical half of a supervisor `BaseZeroRequired` obligation.
    /// It does not grant or change supervisor authority by itself. In particular,
    /// callers must pass `receipt.verified_host_result()` through
    /// `ConfirmedBaseZero::try_from_host_command_result` and then admit that
    /// evidence to the same supervisor instance which requested the stop.
    ///
    /// Unlike [`Self::disarm`], success retains the armed command session so a
    /// subsequent supervisor-authorized mode can use the same controller epoch.
    pub fn apply_fresh_zero(&mut self) -> Result<AppliedCommandReceipt, LiveActuationError> {
        let now = self.clock.host_now().map_err(|source| {
            let stop_reason = source.force_stop_reason();
            self.reject_local_decision(source, stop_reason)
        })?;
        let pending = pending_command(
            self.timing,
            PhysicalDecisionInput::Stopped {
                recorded_pwm: ShadowPwmPair::STOP,
            },
            now,
        )
        .map_err(|source| {
            let stop_reason = source.force_stop_reason();
            self.reject_local_decision(source, stop_reason)
        })?;
        self.apply_pending(pending)
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    pub(super) fn apply_candidate_pwm(
        &mut self,
        admitted: super::wheels_off_candidate_actuation::AdmittedCandidatePwm,
    ) -> Result<AppliedCommandReceipt, LiveActuationError> {
        let pending = candidate_pending_command(&self.clock, self.timing, admitted.timer_pwm())
            .map_err(|source| {
                let stop_reason = source.force_stop_reason();
                self.reject_local_decision(source, stop_reason)
            })?;
        self.apply_pending(pending)
    }

    #[cfg(all(feature = "nano-base-commissioning", unix))]
    pub(super) fn apply_commissioning_pwm(
        &mut self,
        requested_pwm: TimerPwm,
    ) -> Result<AppliedCommandReceipt, LiveActuationError> {
        let now = self.clock.host_now().map_err(|source| {
            let stop_reason = source.force_stop_reason();
            self.reject_local_decision(source, stop_reason)
        })?;
        let acknowledgement_deadline_ns = now
            .as_nanos()
            .checked_add(self.timing.apply_ack_budget_ns)
            .ok_or(PhysicalDecisionError::DeadlineArithmeticOverflow)
            .map_err(|source| {
                let stop_reason = source.force_stop_reason();
                self.reject_local_decision(source, stop_reason)
            })?;
        self.apply_pending(PendingPhysicalCommand::new(
            requested_pwm,
            self.timing.lease,
            MonotonicInstant::from_nanos_since_clock_start(u128::from(acknowledgement_deadline_ns)),
        ))
    }

    fn apply_pending(
        &mut self,
        pending: PendingPhysicalCommand,
    ) -> Result<AppliedCommandReceipt, LiveActuationError> {
        #[cfg(feature = "agent-runtime")]
        let requested_zero = pending.requested_timer_pwm().is_zero();
        #[cfg(feature = "agent-runtime")]
        let head_gaze_transaction = self.begin_head_gaze_base_transaction()?;
        let armed = self.take_armed()?;
        match armed.apply(pending) {
            Ok((armed, receipt)) => {
                self.armed = Some(armed);
                #[cfg(feature = "agent-runtime")]
                Self::commit_head_gaze_base_transaction(
                    head_gaze_transaction,
                    &receipt,
                    requested_zero,
                )?;
                Ok(receipt)
            }
            Err(failure) => Err(LiveActuationError::Apply(failure)),
        }
    }

    pub fn disarm(&mut self) -> Result<DisarmReceipt, LiveActuationError> {
        #[cfg(feature = "agent-runtime")]
        let head_gaze_transaction = self.begin_head_gaze_base_transaction()?;
        let armed = self.take_armed()?;
        match armed.disarm() {
            Ok((_disarmed, receipt)) => {
                #[cfg(feature = "agent-runtime")]
                drop(head_gaze_transaction);
                Ok(receipt)
            }
            Err(failure) => Err(LiveActuationError::Disarm(failure)),
        }
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    pub(super) fn stop_candidate(
        mut self,
    ) -> Result<(StoppedCandidateActuationClient, DisarmReceipt), LiveActuationError> {
        #[cfg(feature = "agent-runtime")]
        let head_gaze_transaction = self.begin_head_gaze_base_transaction()?;
        let armed = self.take_armed()?;
        match armed.disarm() {
            Ok((disarmed, receipt)) => {
                #[cfg(feature = "agent-runtime")]
                drop(head_gaze_transaction);
                Ok((
                    StoppedCandidateActuationClient {
                        disarmed,
                        clock: self.clock,
                        timing: self.timing,
                    },
                    receipt,
                ))
            }
            Err(failure) => Err(LiveActuationError::Disarm(failure)),
        }
    }

    pub const fn is_consumed(&self) -> bool {
        self.armed.is_none()
    }

    /// Retain the exact controller identity observed by the acquisition which
    /// created this physical session. A consumed session has no live
    /// acquisition capability and must be reinventoried rather than rebuilt
    /// from expected configuration.
    pub fn verified_controller_acquisition(
        &self,
    ) -> Result<VerifiedControllerAcquisition, LiveActuationError> {
        self.armed
            .as_ref()
            .map(ArmedCommandClient::verified_acquisition)
            .ok_or(LiveActuationError::SessionConsumed)
    }

    fn take_armed(&mut self) -> Result<ConcreteArmed, LiveActuationError> {
        self.armed.take().ok_or(LiveActuationError::SessionConsumed)
    }

    #[cfg(feature = "agent-runtime")]
    fn begin_head_gaze_base_transaction(
        &mut self,
    ) -> Result<Option<HeadGazeBaseCommandTransaction>, LiveActuationError> {
        self.head_gaze_interlock_installation_open = false;
        self.head_gaze_interlock
            .as_mut()
            .map(HeadGazeBaseMotionInterlock::begin_base_transaction)
            .transpose()
            .map_err(LiveActuationError::HeadGazeInterlock)
    }

    #[cfg(feature = "agent-runtime")]
    fn commit_head_gaze_base_transaction(
        transaction: Option<HeadGazeBaseCommandTransaction>,
        receipt: &AppliedCommandReceipt,
        requested_zero: bool,
    ) -> Result<(), LiveActuationError> {
        let Some(transaction) = transaction else {
            return Ok(());
        };
        if requested_zero {
            let zero = confirmed_head_gaze_zero(receipt)?;
            transaction
                .commit_confirmed_zero(zero)
                .map_err(LiveActuationError::HeadGazeInterlock)
        } else {
            transaction
                .commit_verified_motion_application()
                .map_err(LiveActuationError::HeadGazeInterlock)
        }
    }

    fn reject_local_decision(
        &mut self,
        source: PhysicalDecisionError,
        stop_reason: ForceStopReason,
    ) -> LiveActuationError {
        #[cfg(feature = "agent-runtime")]
        let head_gaze_transaction = match self.begin_head_gaze_base_transaction() {
            Ok(transaction) => transaction,
            Err(LiveActuationError::HeadGazeInterlock(interlock)) => {
                return LiveActuationError::DecisionRejected {
                    source,
                    stop: LocalRejectionStop::HeadGazeInterlockBlocked(interlock),
                };
            }
            Err(_) => {
                return LiveActuationError::DecisionRejected {
                    source,
                    stop: LocalRejectionStop::SessionAlreadyConsumed,
                };
            }
        };
        let stop = match self.armed.take() {
            Some(armed) => match armed.disarm_with_reason(stop_reason) {
                Ok((_disarmed, receipt)) => {
                    #[cfg(feature = "agent-runtime")]
                    drop(head_gaze_transaction);
                    LocalRejectionStop::Confirmed(receipt)
                }
                Err(failure) => LocalRejectionStop::DisarmFailed(failure),
            },
            None => LocalRejectionStop::SessionAlreadyConsumed,
        };
        LiveActuationError::DecisionRejected { source, stop }
    }
}

#[cfg(feature = "agent-runtime")]
fn confirmed_head_gaze_zero(
    receipt: &AppliedCommandReceipt,
) -> Result<ConfirmedBaseZero, LiveActuationError> {
    let acknowledged_at_ns = u64::try_from(receipt.acknowledged_at().nanos_since_clock_start())
        .map_err(|_| LiveActuationError::HeadGazeZeroTimestampOutOfRange {
            acknowledged_at_ns: receipt.acknowledged_at().nanos_since_clock_start(),
        })?;
    ConfirmedBaseZero::try_from_host_command_result(
        receipt.verified_host_result(),
        SupervisorMonotonicInstant::from_nanos_since_process_start(acknowledged_at_ns),
    )
    .map_err(LiveActuationError::HeadGazeZeroEvidence)
}

pub enum LocalRejectionStop {
    Confirmed(DisarmReceipt),
    DisarmFailed(ConcreteDisarmFailure),
    #[cfg(feature = "agent-runtime")]
    HeadGazeInterlockBlocked(HeadGazeBaseInterlockError),
    SessionAlreadyConsumed,
}

pub enum LiveActuationError {
    TransportBuild(UdpTransportBuildError),
    Acquire(ConcreteAcquireFailure),
    Preflight(ConcreteApplyFailure),
    DecisionRejected {
        source: PhysicalDecisionError,
        stop: LocalRejectionStop,
    },
    Apply(ConcreteApplyFailure),
    Disarm(ConcreteDisarmFailure),
    #[cfg(feature = "agent-runtime")]
    HeadGazeInterlockInstallation(HeadGazeInterlockInstallationError),
    #[cfg(feature = "agent-runtime")]
    HeadGazeInterlock(HeadGazeBaseInterlockError),
    #[cfg(feature = "agent-runtime")]
    HeadGazeZeroEvidence(ZeroEvidenceError),
    #[cfg(feature = "agent-runtime")]
    HeadGazeZeroTimestampOutOfRange {
        acknowledged_at_ns: u128,
    },
    SessionConsumed,
}

#[cfg(feature = "agent-runtime")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeInterlockInstallationError {
    AlreadyInstalled,
    BaseTransactionAlreadyAttempted,
    ControllerIdentityMismatch,
    NotInitialZeroSequence { sequence: u32 },
}

#[cfg(feature = "agent-runtime")]
impl fmt::Display for HeadGazeInterlockInstallationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot install head/base motion interlock: {self:?}"
        )
    }
}

#[cfg(feature = "agent-runtime")]
impl std::error::Error for HeadGazeInterlockInstallationError {}

impl fmt::Debug for LiveActuationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

fn stop_evidence_text(knowledge: robot_command_client::LatchedStopKnowledge) -> &'static str {
    match knowledge {
        robot_command_client::LatchedStopKnowledge::ConfirmedStop => "controller stop confirmed",
        robot_command_client::LatchedStopKnowledge::Unconfirmed => "controller stop uncertain",
    }
}

impl fmt::Display for LiveActuationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TransportBuild(source) => {
                write!(formatter, "cannot open V2 command transport: {source}")
            }
            Self::Acquire(failure) => write!(
                formatter,
                "physical zero-acquisition failed ({}; {}): {}",
                failure.recovery().attempts_started(),
                stop_evidence_text(failure.stop_knowledge()),
                failure.cause()
            ),
            Self::Preflight(failure) => write!(
                formatter,
                "previous applied evidence cannot authorize another solve ({}): {}",
                stop_evidence_text(failure.stop_knowledge()),
                failure.cause()
            ),
            Self::DecisionRejected { source, stop } => match stop {
                LocalRejectionStop::Confirmed(receipt) => write!(
                    formatter,
                    "physical decision rejected before send ({source}); controller stop confirmed at {} ns",
                    receipt.acknowledged_at().nanos_since_clock_start()
                ),
                LocalRejectionStop::DisarmFailed(failure) => write!(
                    formatter,
                    "physical decision rejected before send ({source}); stop operation failed ({}): {}",
                    stop_evidence_text(failure.stop_knowledge()),
                    failure.cause()
                ),
                #[cfg(feature = "agent-runtime")]
                LocalRejectionStop::HeadGazeInterlockBlocked(interlock) => write!(
                    formatter,
                    "physical decision rejected before send ({source}); base stop was not sent because the head/base interlock rejected it: {interlock}"
                ),
                LocalRejectionStop::SessionAlreadyConsumed => write!(
                    formatter,
                    "physical decision rejected before send ({source}); session was already consumed"
                ),
            },
            Self::Apply(failure) => write!(
                formatter,
                "controller did not prove exact command application ({}): {}",
                stop_evidence_text(failure.stop_knowledge()),
                failure.cause()
            ),
            Self::Disarm(failure) => write!(
                formatter,
                "shutdown did not prove controller stop ({}): {}",
                stop_evidence_text(failure.stop_knowledge()),
                failure.cause()
            ),
            #[cfg(feature = "agent-runtime")]
            Self::HeadGazeInterlockInstallation(source) => fmt::Display::fmt(source, formatter),
            #[cfg(feature = "agent-runtime")]
            Self::HeadGazeInterlock(source) => fmt::Display::fmt(source, formatter),
            #[cfg(feature = "agent-runtime")]
            Self::HeadGazeZeroEvidence(source) => write!(
                formatter,
                "applied base zero could not refresh the head/base interlock: {source}"
            ),
            #[cfg(feature = "agent-runtime")]
            Self::HeadGazeZeroTimestampOutOfRange { acknowledged_at_ns } => write!(
                formatter,
                "applied base-zero acknowledgement time {acknowledged_at_ns} ns exceeds the head/base interlock time domain"
            ),
            Self::SessionConsumed => {
                formatter.write_str("physical actuation session is already consumed")
            }
        }
    }
}

impl std::error::Error for LiveActuationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TransportBuild(source) => Some(source),
            Self::Acquire(failure) => Some(failure.cause()),
            Self::Preflight(failure) | Self::Apply(failure) => Some(failure.cause()),
            Self::DecisionRejected { source, .. } => Some(source),
            Self::Disarm(failure) => Some(failure.cause()),
            #[cfg(feature = "agent-runtime")]
            Self::HeadGazeInterlockInstallation(source) => Some(source),
            #[cfg(feature = "agent-runtime")]
            Self::HeadGazeInterlock(source) => Some(source),
            #[cfg(feature = "agent-runtime")]
            Self::HeadGazeZeroEvidence(source) => Some(source),
            #[cfg(feature = "agent-runtime")]
            Self::HeadGazeZeroTimestampOutOfRange { .. } => None,
            Self::SessionConsumed => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhysicalDecisionError {
    HostTimestampOutOfRange,
    DeadlineArithmeticOverflow,
    RecordedControllerPwmMismatch {
        recorded: ShadowPwmPair,
        requested: ShadowPwmPair,
    },
    StoppedDecisionRecordedNonzero(ShadowPwmPair),
    CollisionValidityExpired {
        now: HostMonotonicTimestamp,
        collision_valid_through: HostMonotonicTimestamp,
    },
    #[cfg(feature = "nano-attended-navigation-trial")]
    AttendedTrialClockBeforeAttestation {
        now: HostMonotonicTimestamp,
        issued_at_ns: u64,
    },
    #[cfg(feature = "nano-attended-navigation-trial")]
    AttendedTrialExpired {
        now: HostMonotonicTimestamp,
        expires_at_ns: u64,
    },
    #[cfg(feature = "nano-attended-navigation-trial")]
    AttendedTrialPwmAboveLimit {
        left: i8,
        right: i8,
        maximum_abs_pwm_percent: u8,
    },
    TimerPwm(DomainError),
}

impl PhysicalDecisionError {
    const fn force_stop_reason(self) -> ForceStopReason {
        match self {
            Self::CollisionValidityExpired { .. } => ForceStopReason::LeaseExpired,
            #[cfg(feature = "nano-attended-navigation-trial")]
            Self::AttendedTrialExpired { .. } => ForceStopReason::LeaseExpired,
            #[cfg(feature = "nano-attended-navigation-trial")]
            Self::AttendedTrialClockBeforeAttestation { .. }
            | Self::AttendedTrialPwmAboveLimit { .. } => ForceStopReason::TransportFault,
            Self::HostTimestampOutOfRange
            | Self::DeadlineArithmeticOverflow
            | Self::RecordedControllerPwmMismatch { .. }
            | Self::StoppedDecisionRecordedNonzero(_)
            | Self::TimerPwm(_) => ForceStopReason::TransportFault,
        }
    }
}

impl fmt::Display for PhysicalDecisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid physical decision boundary: {self:?}")
    }
}

impl std::error::Error for PhysicalDecisionError {}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_command_client::TimeoutNs;
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    use robot_command_client::fake::{FakeClock, FakeStep, FakeTransport};
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    use robot_command_client::{
        ClientConfigInput, DisarmedCommandClient, FailureCause, LatchedStopKnowledge,
    };
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    use robot_protocol::ControllerUptimeMsWrapping;
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    use robot_protocol::v2::{
        AcquireResult, AcquireResultCode, ActuatorConfigFingerprint, ControllerBootId,
        ControllerCapabilities, ControllerDeadlineMsWrapping, ControllerFaults,
        ControllerSessionClass, ControllerUid, HostCommandResult, HostCommandResultCode,
        HostStopResult, Message, MessageKind, OutputState, RemainingLeaseMs, RequestId, StatusCode,
        StatusReport, StopResultCode, TargetBootId, V2CommandSequence,
    };
    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    use std::time::Duration;

    fn pwm(left: i8, right: i8) -> ShadowPwmPair {
        ShadowPwmPair::try_new(left, right).expect("valid fixture PWM")
    }

    fn timing() -> ActuationTiming {
        ActuationTiming {
            apply_ack_budget_ns: TimeoutNs::try_new(20_000_000).expect("valid budget").get(),
            lease: V2CommandLeaseMs::try_new(200).expect("valid lease"),
        }
    }

    #[cfg(feature = "nano-attended-navigation-trial")]
    #[test]
    fn attended_trial_guard_rejects_invalid_lifetimes_caps_and_boundaries() {
        for invalid in [0, MAX_WHEEL_ON_COMMISSIONING_PWM_PERCENT + 1] {
            assert!(matches!(
                AttendedTrialActuationGuard::try_new(invalid, 10, 20),
                Err(AttendedTrialActuationGuardError::InvalidMaximumPwm(actual))
                    if actual == invalid
            ));
        }
        for (issued_at_ns, expires_at_ns) in [(0, 20), (10, 10), (20, 10)] {
            assert!(matches!(
                AttendedTrialActuationGuard::try_new(20, issued_at_ns, expires_at_ns),
                Err(AttendedTrialActuationGuardError::InvalidLifetime {
                    issued_at_ns: actual_issued,
                    expires_at_ns: actual_expires,
                }) if actual_issued == issued_at_ns && actual_expires == expires_at_ns
            ));
        }

        let guard = AttendedTrialActuationGuard::try_new(20, 10, 20).unwrap();
        assert!(matches!(
            guard.require_current(HostMonotonicTimestamp::from_nanos(9)),
            Err(PhysicalDecisionError::AttendedTrialClockBeforeAttestation {
                issued_at_ns: 10,
                ..
            })
        ));
        assert!(
            guard
                .require_current(HostMonotonicTimestamp::from_nanos(10))
                .is_ok()
        );
        assert!(
            guard
                .require_current(HostMonotonicTimestamp::from_nanos(19))
                .is_ok()
        );
        assert!(matches!(
            guard.require_current(HostMonotonicTimestamp::from_nanos(20)),
            Err(PhysicalDecisionError::AttendedTrialExpired {
                expires_at_ns: 20,
                ..
            })
        ));

        assert!(guard.require_pwm(pwm(-20, 20)).is_ok());
        assert!(matches!(
            guard.require_pwm(pwm(-21, 0)),
            Err(PhysicalDecisionError::AttendedTrialPwmAboveLimit {
                left: -21,
                right: 0,
                maximum_abs_pwm_percent: 20,
            })
        ));
    }

    #[test]
    fn exact_wheel_sign_and_percent_survive_decision_conversion() {
        let pending = pending_command(
            timing(),
            PhysicalDecisionInput::Controller {
                recorded_pwm: pwm(-37, 22),
                requested_pwm: pwm(-37, 22),
                collision_valid_through: HostMonotonicTimestamp::from_nanos(500_000_000),
            },
            HostMonotonicTimestamp::from_nanos(100_000_000),
        )
        .expect("sufficient collision horizon");
        assert_eq!(pending.requested_timer_pwm().left().get(), -37);
        assert_eq!(pending.requested_timer_pwm().right().get(), 22);
        assert_eq!(
            pending
                .acknowledgement_deadline_exclusive()
                .nanos_since_clock_start(),
            500_000_000
        );
    }

    #[test]
    fn collision_validity_equality_is_expired_before_send() {
        let error = pending_command(
            timing(),
            PhysicalDecisionInput::Controller {
                recorded_pwm: pwm(1, 1),
                requested_pwm: pwm(1, 1),
                collision_valid_through: HostMonotonicTimestamp::from_nanos(100_000_000),
            },
            HostMonotonicTimestamp::from_nanos(100_000_000),
        )
        .expect_err("equality cannot authorize motion");
        assert!(matches!(
            error,
            PhysicalDecisionError::CollisionValidityExpired { .. }
        ));
    }

    #[test]
    fn collision_freshness_admits_ack_without_promising_a_full_lease_of_static_world() {
        let pending = pending_command(
            timing(),
            PhysicalDecisionInput::Controller {
                recorded_pwm: pwm(8, 9),
                requested_pwm: pwm(8, 9),
                collision_valid_through: HostMonotonicTimestamp::from_nanos(200_000_000),
            },
            HostMonotonicTimestamp::from_nanos(150_000_000),
        )
        .expect("fresh decision can be acknowledged despite a separate 200 ms lease");
        assert_eq!(pending.lease().get(), 200);
        assert_eq!(
            pending
                .acknowledgement_deadline_exclusive()
                .nanos_since_clock_start(),
            200_000_000
        );
    }

    #[test]
    fn stopped_decision_is_zero_and_bounded_only_by_apply_ack() {
        let pending = pending_command(
            timing(),
            PhysicalDecisionInput::Stopped {
                recorded_pwm: ShadowPwmPair::STOP,
            },
            HostMonotonicTimestamp::from_nanos(100_000_000),
        )
        .expect("valid stop command");
        assert!(pending.requested_timer_pwm().is_zero());
        assert_eq!(
            pending
                .acknowledgement_deadline_exclusive()
                .nanos_since_clock_start(),
            120_000_000
        );
    }

    #[test]
    fn recorded_and_controller_pwm_mismatch_never_builds_a_command() {
        assert!(matches!(
            pending_command(
                timing(),
                PhysicalDecisionInput::Controller {
                    recorded_pwm: pwm(10, 10),
                    requested_pwm: pwm(10, 9),
                    collision_valid_through: HostMonotonicTimestamp::from_nanos(500_000_000),
                },
                HostMonotonicTimestamp::from_nanos(100_000_000),
            ),
            Err(PhysicalDecisionError::RecordedControllerPwmMismatch { .. })
        ));
    }

    #[test]
    fn only_collision_expiry_maps_to_the_lease_expired_stop_reason() {
        assert_eq!(
            PhysicalDecisionError::CollisionValidityExpired {
                now: HostMonotonicTimestamp::from_nanos(10),
                collision_valid_through: HostMonotonicTimestamp::from_nanos(10),
            }
            .force_stop_reason(),
            ForceStopReason::LeaseExpired
        );
        for error in [
            PhysicalDecisionError::HostTimestampOutOfRange,
            PhysicalDecisionError::DeadlineArithmeticOverflow,
            PhysicalDecisionError::RecordedControllerPwmMismatch {
                recorded: pwm(1, 2),
                requested: pwm(2, 1),
            },
            PhysicalDecisionError::StoppedDecisionRecordedNonzero(pwm(1, 0)),
            PhysicalDecisionError::TimerPwm(
                TimerPwm::try_new(101, 0).expect_err("out-of-range PWM is rejected"),
            ),
        ] {
            assert_eq!(error.force_stop_reason(), ForceStopReason::TransportFault);
        }
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn qualification_clock_fault_is_one_shot_and_strictly_regresses() {
        let state = QualificationClockFaultState::new();
        let first = state.now(QUALIFICATION_CLOCK_BIAS_NS + 100);
        state.arm();
        let injected = state.now(QUALIFICATION_CLOCK_BIAS_NS + 200);
        let recovered = state.now(QUALIFICATION_CLOCK_BIAS_NS + 300);

        assert_eq!(injected, first - 1);
        assert!(injected < first);
        assert!(recovered > first);
        state.arm();
        assert_eq!(
            state.now(QUALIFICATION_CLOCK_BIAS_NS + 400),
            QUALIFICATION_CLOCK_BIAS_NS + 400,
            "a fired declaration cannot inject twice"
        );
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn dormant_qualification_clock_is_monotonic_across_concurrent_callers() {
        let state = Arc::new(QualificationClockFaultState::new());
        let mut workers = Vec::new();
        for worker in 0..8_u64 {
            let state = Arc::clone(&state);
            workers.push(std::thread::spawn(move || {
                for sample in 0..1_000_u64 {
                    let normal =
                        QUALIFICATION_CLOCK_BIAS_NS + worker.saturating_mul(1_000) + sample;
                    assert!(state.now(normal) >= normal);
                }
            }));
        }
        for worker in workers {
            worker.join().expect("clock worker");
        }
        assert_eq!(
            state.phase.load(Ordering::SeqCst),
            QUALIFICATION_CLOCK_DORMANT
        );
    }

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    #[test]
    fn candidate_apply_clock_fault_latches_before_nonzero_transport_and_confirms_stop() {
        let uid = ControllerUid::try_new([0x11; 12]).expect("fixture UID");
        let boot = ControllerBootId::try_new(17).expect("fixture boot");
        let epoch = robot_protocol::v2::ControlEpoch::try_new(23).expect("fixture epoch");
        let fingerprint =
            ActuatorConfigFingerprint::try_new(*b"KIKO-4PWM-CAND1!").expect("fingerprint");
        let capabilities = ControllerCapabilities::try_from_bits(
            ControllerCapabilities::SOFTWARE_GUARD_BITS
                | ControllerCapabilities::OPERATOR_SUPERVISED_FOUR_PWM_CANDIDATE,
        )
        .expect("candidate capabilities");
        let config = ClientConfig::parse_for_session(
            ClientConfigInput {
                command_endpoint: "127.0.0.1:8080",
                controller_uid_hex: "111111111111111111111111",
                expected_firmware_abi: "2",
                expected_firmware_build_id: "135169",
                expected_actuator_config_fingerprint_hex: "4b494b4f2d3450574d2d43414e443121",
                status_timeout_ns: "50000000",
                acquire_timeout_ns: "50000000",
                applied_ack_timeout_ns: "50000000",
                stop_attempt_timeout_ns: "50000000",
                max_stop_recovery_attempts: "1",
                zero_acquisition_lease_ms: "100",
            },
            ControllerSessionClass::OperatorSupervisedFourPwmCandidate,
        )
        .expect("candidate client config");
        let fake_clock = FakeClock::new(0);
        let (transport, probe) = FakeTransport::scripted(
            fake_clock,
            [
                FakeStep::respond(
                    MessageKind::StatusQuery,
                    Duration::ZERO,
                    Message::StatusReport(StatusReport {
                        controller_uid: uid,
                        observed_boot_id: TargetBootId::Exact(boot),
                        request_id: RequestId::new(0),
                        status: StatusCode::ReadyStopped,
                        control_epoch: None,
                        controller_uptime: ControllerUptimeMsWrapping::new(1_000),
                        capabilities,
                        output_state: OutputState::Disabled,
                        controller_timer_pwm: TimerPwm::ZERO,
                        remaining_lease: RemainingLeaseMs::ZERO,
                        faults: ControllerFaults::NONE,
                    }),
                ),
                FakeStep::respond(
                    MessageKind::AcquireControl,
                    Duration::ZERO,
                    Message::AcquireResult(AcquireResult {
                        controller_uid: uid,
                        boot_id: boot,
                        request_id: RequestId::new(1),
                        control_epoch: Some(epoch),
                        result: AcquireResultCode::Granted,
                        capabilities,
                        faults: ControllerFaults::NONE,
                        observed_firmware_abi: 2,
                        observed_firmware_build_id: 0x0002_1001,
                        observed_actuator_config_fingerprint: fingerprint,
                    }),
                ),
                FakeStep::respond(
                    MessageKind::HostCommand,
                    Duration::ZERO,
                    Message::HostCommandResult(HostCommandResult {
                        controller_uid: uid,
                        boot_id: boot,
                        control_epoch: epoch,
                        sequence: V2CommandSequence::FIRST,
                        result: HostCommandResultCode::AppliedNew,
                        requested_timer_pwm: TimerPwm::ZERO,
                        controller_timer_pwm: TimerPwm::ZERO,
                        output_state: OutputState::ZeroPwm,
                        controller_applied_at: ControllerUptimeMsWrapping::new(2_000),
                        controller_expires_at: ControllerDeadlineMsWrapping::new(2_100),
                        remaining_lease: RemainingLeaseMs::try_new(90).expect("remaining lease"),
                        faults: ControllerFaults::NONE,
                    }),
                ),
                FakeStep::respond(
                    MessageKind::HostStop,
                    Duration::ZERO,
                    Message::HostStopResult(HostStopResult {
                        controller_uid: uid,
                        observed_boot_id: TargetBootId::Exact(boot),
                        request_id: RequestId::new(2),
                        result: StopResultCode::ControllerConfirmed,
                        output_state: OutputState::Disabled,
                        controller_uptime: ControllerUptimeMsWrapping::new(3_000),
                        faults: ControllerFaults::NONE,
                    }),
                ),
            ],
        );
        let clock = ActuationMonotonicClock::new_candidate(
            Instant::now(),
            Some(WheelsOffQualificationHostClockFaultInjection::RegressionOnFirstNonzeroCommand),
        );
        let client = DisarmedCommandClient::new(transport, clock.clone(), config);
        let (armed, _) = match client.acquire_zero() {
            Ok(acquired) => acquired,
            Err(_) => panic!("zero acquisition"),
        };
        let requested = TimerPwm::try_new(1, -1).expect("bounded nonzero candidate PWM");
        let pending =
            candidate_pending_command(&clock, timing(), requested).expect("candidate pending");
        let failure = match armed.apply(pending) {
            Ok(_) => panic!("injected candidate clock regression must latch"),
            Err(failure) => failure,
        };
        assert!(matches!(
            failure.cause(),
            FailureCause::ClockRegressed { previous, observed } if observed < previous
        ));
        assert_eq!(
            failure.stop_knowledge(),
            LatchedStopKnowledge::ConfirmedStop
        );
        assert_eq!(
            probe
                .exchanges()
                .into_iter()
                .map(|exchange| exchange.request().kind())
                .collect::<Vec<_>>(),
            vec![
                MessageKind::StatusQuery,
                MessageKind::AcquireControl,
                MessageKind::HostCommand,
                MessageKind::HostStop,
            ],
            "the only HostCommand was acquisition zero; the nonzero command never reached transport"
        );
        assert_eq!(probe.remaining_steps(), 0);
    }
}
