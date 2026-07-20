//! Synchronous bridge from safety-approved navigation decisions to exact V2
//! controller application evidence.
//!
//! This module does not alter MPC, collision checking, or shadow recording.
//! Physical mode consumes the already-recorded safety decision, requires the
//! previous applied receipt before the next solve, and returns only after the
//! exact requested PWM has been acknowledged by the controller.

use std::fmt;
use std::time::Instant;

use robot_command_client::{
    AcquireFailure, AppliedCommandReceipt, ApplyFailure, ArmedCommandClient, ClientConfig,
    DisarmFailure, DisarmReceipt, DisarmedCommandClient, MonotonicClock, MonotonicInstant,
    PendingPhysicalCommand, RobotProtocolV2WireAdapter, UdpTransportBuildError, UdpV2Transport,
};
use robot_protocol::v2::{DomainError, ForceStopReason, TimerPwm, V2CommandLeaseMs};

use super::actuation_config::NavigationActuationConfigV1;
use super::{SafetyDecision, SafetyDecisionOutcome, ShadowPwmPair};
use crate::HostMonotonicTimestamp;

type ConcreteTransport = UdpV2Transport<RobotProtocolV2WireAdapter>;
type ConcreteArmed = ArmedCommandClient<ConcreteTransport, ActuationMonotonicClock>;
type ConcreteAcquireFailure = AcquireFailure<ConcreteTransport, ActuationMonotonicClock>;
type ConcreteApplyFailure = ApplyFailure<ConcreteTransport, ActuationMonotonicClock>;
type ConcreteDisarmFailure = DisarmFailure<ConcreteTransport, ActuationMonotonicClock>;

/// A monotonic domain sharing the live navigation clock's exact origin.
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct ActuationMonotonicClock {
    origin: Instant,
}

impl ActuationMonotonicClock {
    const fn new(origin: Instant) -> Self {
        Self { origin }
    }

    fn host_now(self) -> Result<HostMonotonicTimestamp, PhysicalDecisionError> {
        let nanoseconds = u64::try_from(self.origin.elapsed().as_nanos())
            .map_err(|_| PhysicalDecisionError::HostTimestampOutOfRange)?;
        Ok(HostMonotonicTimestamp::from_nanos(nanoseconds))
    }
}

impl MonotonicClock for ActuationMonotonicClock {
    fn now(&self) -> MonotonicInstant {
        MonotonicInstant::from_nanos_since_clock_start(self.origin.elapsed().as_nanos())
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

/// Sole owner of an armed physical session in the navigation worker.
///
/// `armed == None` means the session was consumed by a terminal failure or an
/// explicit disarm. There is no API that can recreate authority in place.
pub struct PhysicalActuationSession {
    armed: Option<ConcreteArmed>,
    clock: ActuationMonotonicClock,
    timing: ActuationTiming,
}

/// A zero-only live controller session for non-motion diagnostics.
///
/// The type exposes no PWM-bearing input and can therefore only refresh an
/// already acquired zero or request a terminal stop.
#[must_use = "a zero-only session must be explicitly disarmed and its receipt checked"]
pub struct PhysicalZeroHoldSession {
    inner: PhysicalActuationSession,
}

impl PhysicalZeroHoldSession {
    pub fn acquire_zero_only(
        config: ClientConfig,
        clock_origin: Instant,
    ) -> Result<(Self, AppliedCommandReceipt), LiveActuationError> {
        let timing = ActuationTiming::from_zero_client(&config);
        let (inner, receipt) =
            PhysicalActuationSession::acquire_from_client_config(config, clock_origin, timing)?;
        Ok((Self { inner }, receipt))
    }

    pub fn refresh_zero(&mut self) -> Result<AppliedCommandReceipt, LiveActuationError> {
        self.inner.apply_fresh_zero()
    }

    pub fn disarm(mut self) -> Result<DisarmReceipt, LiveActuationError> {
        self.inner.disarm()
    }

    pub const fn is_consumed(&self) -> bool {
        self.inner.is_consumed()
    }
}

impl PhysicalActuationSession {
    pub fn acquire(
        config: &NavigationActuationConfigV1,
        clock_origin: Instant,
    ) -> Result<(Self, AppliedCommandReceipt), LiveActuationError> {
        Self::acquire_from_client_config(
            config.client_config(),
            clock_origin,
            ActuationTiming::from_authority(config),
        )
    }

    fn acquire_from_client_config(
        config: ClientConfig,
        clock_origin: Instant,
        timing: ActuationTiming,
    ) -> Result<(Self, AppliedCommandReceipt), LiveActuationError> {
        let transport = UdpV2Transport::connect_canonical(config.endpoint())
            .map_err(LiveActuationError::TransportBuild)?;
        let clock = ActuationMonotonicClock::new(clock_origin);
        let client = DisarmedCommandClient::new(transport, clock, config);
        let (armed, receipt) = client.acquire_zero().map_err(LiveActuationError::Acquire)?;
        Ok((
            Self {
                armed: Some(armed),
                clock,
                timing,
            },
            receipt,
        ))
    }

    /// Must run immediately before each MPC tick. This consumes and restores
    /// the armed type state only while the previous applied evidence is live.
    pub fn require_current_before_solve(&mut self) -> Result<(), LiveActuationError> {
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

    fn apply_pending(
        &mut self,
        pending: PendingPhysicalCommand,
    ) -> Result<AppliedCommandReceipt, LiveActuationError> {
        let armed = self.take_armed()?;
        match armed.apply(pending) {
            Ok((armed, receipt)) => {
                self.armed = Some(armed);
                Ok(receipt)
            }
            Err(failure) => Err(LiveActuationError::Apply(failure)),
        }
    }

    pub fn disarm(&mut self) -> Result<DisarmReceipt, LiveActuationError> {
        let armed = self.take_armed()?;
        match armed.disarm() {
            Ok((_disarmed, receipt)) => Ok(receipt),
            Err(failure) => Err(LiveActuationError::Disarm(failure)),
        }
    }

    pub const fn is_consumed(&self) -> bool {
        self.armed.is_none()
    }

    fn take_armed(&mut self) -> Result<ConcreteArmed, LiveActuationError> {
        self.armed.take().ok_or(LiveActuationError::SessionConsumed)
    }

    fn reject_local_decision(
        &mut self,
        source: PhysicalDecisionError,
        stop_reason: ForceStopReason,
    ) -> LiveActuationError {
        let stop = match self.armed.take() {
            Some(armed) => match armed.disarm_with_reason(stop_reason) {
                Ok((_disarmed, receipt)) => LocalRejectionStop::Confirmed(receipt),
                Err(failure) => LocalRejectionStop::Uncertain(failure),
            },
            None => LocalRejectionStop::SessionAlreadyConsumed,
        };
        LiveActuationError::DecisionRejected { source, stop }
    }
}

pub enum LocalRejectionStop {
    Confirmed(DisarmReceipt),
    Uncertain(ConcreteDisarmFailure),
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
    SessionConsumed,
}

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
                LocalRejectionStop::Uncertain(failure) => write!(
                    formatter,
                    "physical decision rejected before send ({source}); controller stop uncertain: {}",
                    failure.cause()
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
    TimerPwm(DomainError),
}

impl PhysicalDecisionError {
    const fn force_stop_reason(self) -> ForceStopReason {
        match self {
            Self::CollisionValidityExpired { .. } => ForceStopReason::LeaseExpired,
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

    fn pwm(left: i8, right: i8) -> ShadowPwmPair {
        ShadowPwmPair::try_new(left, right).expect("valid fixture PWM")
    }

    fn timing() -> ActuationTiming {
        ActuationTiming {
            apply_ack_budget_ns: TimeoutNs::try_new(20_000_000).expect("valid budget").get(),
            lease: V2CommandLeaseMs::try_new(200).expect("valid lease"),
        }
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
}
