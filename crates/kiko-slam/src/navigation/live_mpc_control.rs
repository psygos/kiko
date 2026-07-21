//! Receipt-gated MPC control cycles for the live SLAM owner.
//!
//! The live worker previously performed preflight, coordinator tick, and
//! physical application as three adjacent calls. This type makes that ordering
//! one API invariant and provides the same path for point-goal and manual MPC.

use std::fmt;
use std::time::Instant;

use robot_command_client::{AppliedCommandReceipt, DisarmReceipt};

use super::actuation::{LiveActuationError, PhysicalActuationSession};
use super::mpc::HostMonotonicClock;
use super::{
    CoordinatorTickError, CoordinatorTickOutcome, NavigationActuationConfigV1,
    NavigationIngressSink, SafetyDecision, ShadowNavigationCoordinator,
};
#[cfg(feature = "agent-runtime")]
use super::{
    ManualDriveAcceptedIntent, ManualDriveOutput, ManualMpcCommandError, ManualMpcCommandV1,
    NumericAuthorityLeaseId,
};
use crate::HostMonotonicTimestamp;

trait MpcActuationPort {
    type Receipt;
    type Error;

    fn require_current_before_solve(&mut self) -> Result<(), Self::Error>;
    fn apply(&mut self, decision: &SafetyDecision) -> Result<Self::Receipt, Self::Error>;
}

impl MpcActuationPort for PhysicalActuationSession {
    type Receipt = AppliedCommandReceipt;
    type Error = LiveActuationError;

    fn require_current_before_solve(&mut self) -> Result<(), Self::Error> {
        PhysicalActuationSession::require_current_before_solve(self)
    }

    fn apply(&mut self, decision: &SafetyDecision) -> Result<Self::Receipt, Self::Error> {
        PhysicalActuationSession::apply(self, decision)
    }
}

struct MpcControlCore<Port> {
    port: Port,
}

impl<Port: MpcActuationPort> MpcControlCore<Port> {
    fn execute<Outcome>(
        &mut self,
        tick: impl FnOnce() -> Result<Outcome, CoordinatorTickError>,
        decision: impl FnOnce(&Outcome) -> &SafetyDecision,
    ) -> Result<(Outcome, Port::Receipt), MpcControlCycleError<Port::Error>> {
        self.port
            .require_current_before_solve()
            .map_err(MpcControlCycleError::Preflight)?;
        let outcome = tick().map_err(MpcControlCycleError::Coordinator)?;
        let receipt = self
            .port
            .apply(decision(&outcome))
            .map_err(MpcControlCycleError::Apply)?;
        Ok((outcome, receipt))
    }
}

#[derive(Debug)]
enum MpcControlCycleError<PortError> {
    Preflight(PortError),
    Coordinator(CoordinatorTickError),
    Apply(PortError),
}

/// One coordinator decision and its exact controller application receipt.
///
/// Construction is private: a coordinator result without a matching applied
/// receipt cannot be represented as a successful live control cycle.
pub struct LiveAppliedMpcTick<JournalError> {
    outcome: CoordinatorTickOutcome<JournalError>,
    receipt: AppliedCommandReceipt,
}

impl<JournalError> LiveAppliedMpcTick<JournalError> {
    pub const fn outcome(&self) -> &CoordinatorTickOutcome<JournalError> {
        &self.outcome
    }

    pub const fn receipt(&self) -> &AppliedCommandReceipt {
        &self.receipt
    }

    pub fn into_parts(self) -> (CoordinatorTickOutcome<JournalError>, AppliedCommandReceipt) {
        (self.outcome, self.receipt)
    }
}

/// Sole receipt-gated physical driver used by the live SLAM control thread.
///
/// Every successful tick is ordered as:
/// previous applied evidence -> coordinator/MPC/safety -> exact application
/// receipt. The physical session is never exposed while the driver is live.
pub struct LiveMpcControlDriver {
    core: MpcControlCore<PhysicalActuationSession>,
}

impl LiveMpcControlDriver {
    pub fn acquire(
        config: &NavigationActuationConfigV1,
        clock_origin: Instant,
    ) -> Result<(Self, AppliedCommandReceipt), LiveActuationError> {
        let (session, initial_zero) = PhysicalActuationSession::acquire(config, clock_origin)?;
        Ok((
            Self {
                core: MpcControlCore { port: session },
            },
            initial_zero,
        ))
    }

    pub fn tick_point_goal<J, C>(
        &mut self,
        coordinator: &mut ShadowNavigationCoordinator<J>,
        tick: HostMonotonicTimestamp,
        clock: &mut C,
    ) -> Result<LiveAppliedMpcTick<J::Error>, LiveMpcControlError>
    where
        J: NavigationIngressSink,
        C: HostMonotonicClock,
    {
        let (outcome, receipt) = self
            .core
            .execute(
                || coordinator.tick(tick, clock),
                CoordinatorTickOutcome::decision,
            )
            .map_err(LiveMpcControlError::from_cycle)?;
        Ok(LiveAppliedMpcTick { outcome, receipt })
    }

    /// Apply a newly sequenced zero while retaining the same physical session.
    ///
    /// This exists only for supervisor mode-transition barriers. The receipt
    /// remains evidence, not authority: the live owner must admit its exact
    /// host result to the supervisor which requested the zero.
    pub fn apply_fresh_zero(&mut self) -> Result<AppliedCommandReceipt, LiveActuationError> {
        self.core.port.apply_fresh_zero()
    }

    /// Run one already-admitted manual body-twist, explicit stop, or deadman
    /// stop through the same coordinator, collision, safety, and receipt path.
    #[cfg(feature = "agent-runtime")]
    pub fn tick_manual<J, C, LeaseId>(
        &mut self,
        coordinator: &mut ShadowNavigationCoordinator<J>,
        tick: HostMonotonicTimestamp,
        command: ManualDriveOutput<LeaseId>,
        clock: &mut C,
    ) -> Result<LiveAppliedMpcTick<J::Error>, LiveMpcControlError>
    where
        J: NavigationIngressSink,
        C: HostMonotonicClock,
        LeaseId: NumericAuthorityLeaseId,
    {
        enum ManualTick<LeaseId> {
            Velocity(ManualMpcCommandV1),
            ExplicitStop(super::ManualDriveAcceptedStop<LeaseId>),
            Stopped(super::ManualDriveStopped<LeaseId>),
        }

        let command = match command {
            ManualDriveOutput::Accepted(accepted)
                if accepted.intent() == ManualDriveAcceptedIntent::Velocity =>
            {
                ManualTick::Velocity(
                    ManualMpcCommandV1::try_from_accepted(accepted)
                        .map_err(LiveMpcControlError::ManualCommand)?,
                )
            }
            ManualDriveOutput::Accepted(accepted) => ManualTick::ExplicitStop(
                accepted
                    .into_explicit_stop()
                    .map_err(LiveMpcControlError::ManualStop)?,
            ),
            ManualDriveOutput::Stopped(stopped) => ManualTick::Stopped(stopped),
        };

        let (outcome, receipt) = self
            .core
            .execute(
                || match command {
                    ManualTick::Velocity(command) => coordinator.tick_manual(tick, command, clock),
                    ManualTick::ExplicitStop(stop) => {
                        coordinator.tick_manual_explicit_stop(tick, stop, clock)
                    }
                    ManualTick::Stopped(stopped) => {
                        coordinator.tick_manual_stopped(tick, stopped, clock)
                    }
                },
                CoordinatorTickOutcome::decision,
            )
            .map_err(LiveMpcControlError::from_cycle)?;
        Ok(LiveAppliedMpcTick { outcome, receipt })
    }

    pub fn disarm(&mut self) -> Result<DisarmReceipt, LiveActuationError> {
        self.core.port.disarm()
    }

    pub const fn is_consumed(&self) -> bool {
        self.core.port.is_consumed()
    }
}

#[derive(Debug)]
pub enum LiveMpcControlError {
    Preflight(LiveActuationError),
    Coordinator(CoordinatorTickError),
    #[cfg(feature = "agent-runtime")]
    ManualCommand(ManualMpcCommandError),
    #[cfg(feature = "agent-runtime")]
    ManualStop(super::ManualDriveAcceptedTargetKindError),
    Apply(LiveActuationError),
}

impl LiveMpcControlError {
    fn from_cycle(source: MpcControlCycleError<LiveActuationError>) -> Self {
        match source {
            MpcControlCycleError::Preflight(source) => Self::Preflight(source),
            MpcControlCycleError::Coordinator(source) => Self::Coordinator(source),
            MpcControlCycleError::Apply(source) => Self::Apply(source),
        }
    }
}

impl fmt::Display for LiveMpcControlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preflight(source) => write!(
                formatter,
                "live MPC pre-solve applied-evidence gate failed: {source}"
            ),
            Self::Coordinator(source) => {
                write!(formatter, "live MPC coordinator tick failed: {source}")
            }
            #[cfg(feature = "agent-runtime")]
            Self::ManualCommand(source) => {
                write!(
                    formatter,
                    "live manual MPC command conversion failed: {source}"
                )
            }
            #[cfg(feature = "agent-runtime")]
            Self::ManualStop(source) => {
                write!(formatter, "live manual stop conversion failed: {source}")
            }
            Self::Apply(source) => {
                write!(
                    formatter,
                    "live MPC exact command application failed: {source}"
                )
            }
        }
    }
}

impl std::error::Error for LiveMpcControlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Preflight(source) | Self::Apply(source) => Some(source),
            Self::Coordinator(source) => Some(source),
            #[cfg(feature = "agent-runtime")]
            Self::ManualCommand(source) => Some(source),
            #[cfg(feature = "agent-runtime")]
            Self::ManualStop(source) => Some(source),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::navigation::mpc::{
        MPC_CONFIG_V1, MpcConfigV1, MpcConfigV1Dto, MpcSolver, PLANT_MODEL_V1, PlantEvidenceV1Dto,
        PlantModelV1, PlantModelV1Dto, PlantValidityEnvelopeV1Dto, WheelPlantV1Dto,
    };
    use crate::navigation::{
        SafetyNotReadyReason, SafetyTickInput, ShadowCommandConfig, ShadowCommandConfigDto,
        ShadowSafetySupervisor,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FakeError {
        Preflight,
        Apply,
    }

    struct FakePort {
        events: Rc<RefCell<Vec<&'static str>>>,
        preflight_error: bool,
        apply_error: bool,
    }

    impl MpcActuationPort for FakePort {
        type Receipt = u64;
        type Error = FakeError;

        fn require_current_before_solve(&mut self) -> Result<(), Self::Error> {
            self.events.borrow_mut().push("preflight");
            if self.preflight_error {
                Err(FakeError::Preflight)
            } else {
                Ok(())
            }
        }

        fn apply(&mut self, _decision: &SafetyDecision) -> Result<Self::Receipt, Self::Error> {
            self.events.borrow_mut().push("apply");
            if self.apply_error {
                Err(FakeError::Apply)
            } else {
                Ok(7)
            }
        }
    }

    struct FixedClock(HostMonotonicTimestamp);

    impl HostMonotonicClock for FixedClock {
        fn try_now(
            &mut self,
        ) -> Result<HostMonotonicTimestamp, super::super::mpc::HostMonotonicClockReadError>
        {
            Ok(self.0)
        }
    }

    fn stopped_decision() -> SafetyDecision {
        let plant = PlantModelV1::parse(PlantModelV1Dto {
            schema_version: PLANT_MODEL_V1,
            model_id: "live-driver-test".to_owned(),
            model_version: 1,
            sample_period_s: 0.05,
            wheelbase_m: 0.4,
            left: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.2,
            },
            right: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.2,
            },
            validity: PlantValidityEnvelopeV1Dto {
                left_pwm_min_percent: -20,
                left_pwm_max_percent: 20,
                right_pwm_min_percent: -20,
                right_pwm_max_percent: 20,
                left_velocity_min_mps: -1.0,
                left_velocity_max_mps: 1.0,
                right_velocity_min_mps: -1.0,
                right_velocity_max_mps: 1.0,
                max_abs_yaw_rate_rad_s: 2.0,
                max_abs_lateral_velocity_mps: 0.2,
            },
            evidence: PlantEvidenceV1Dto::SyntheticFixture {
                fixture_id: "live-driver".to_owned(),
                generator_id: "unit-test".to_owned(),
            },
        })
        .unwrap();
        let config = MpcConfigV1::parse(MpcConfigV1Dto {
            schema_version: MPC_CONFIG_V1,
            horizon_steps: 2,
            step_period_s: 0.05,
            integration_substeps: 1,
            optimization_iterations: 1,
            candidates_per_wheel: 3,
            max_rollout_evaluations: 100,
            initial_search_radius_percent: 10,
            search_radius_decay_numerator: 1,
            search_radius_decay_denominator: 2,
            left_pwm_min_percent: -20,
            left_pwm_max_percent: 20,
            right_pwm_min_percent: -20,
            right_pwm_max_percent: 20,
            left_max_slew_percent_per_step: 20,
            right_max_slew_percent_per_step: 20,
            max_integration_tube_radius_m: 1.0,
            position_cost_per_m2: 1.0,
            heading_cost_per_rad2: 1.0,
            forward_velocity_cost_s2_per_m2: 0.0,
            yaw_rate_cost_s2_per_rad2: 0.0,
            pwm_cost_per_percent2: 0.01,
            slew_cost_per_percent2: 0.01,
            terminal_state_cost_multiplier: 1.0,
        })
        .unwrap();
        let solver = MpcSolver::new(plant, config).unwrap();
        let shadow = ShadowCommandConfig::parse(ShadowCommandConfigDto {
            lease_ms: 100,
            retained_records: 4,
            initial_sequence: 1,
        })
        .unwrap();
        let mut safety = ShadowSafetySupervisor::try_new(solver, shadow).unwrap();
        let tick = HostMonotonicTimestamp::from_nanos(10);
        safety
            .decide(
                tick,
                SafetyTickInput::NotReady(SafetyNotReadyReason::NavigationGoalUnavailable),
                &mut FixedClock(tick),
            )
            .unwrap()
    }

    #[test]
    fn control_cycle_orders_preflight_decision_and_exact_application() {
        let decision = stopped_decision();
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut core = MpcControlCore {
            port: FakePort {
                events: Rc::clone(&events),
                preflight_error: false,
                apply_error: false,
            },
        };
        let tick_events = Rc::clone(&events);
        let (outcome, receipt) = core
            .execute(
                || {
                    tick_events.borrow_mut().push("tick");
                    Ok::<_, CoordinatorTickError>(decision)
                },
                |outcome| outcome,
            )
            .unwrap();
        assert!(outcome.record().pwm().is_stop());
        assert_eq!(receipt, 7);
        assert_eq!(*events.borrow(), ["preflight", "tick", "apply"]);
    }

    #[test]
    fn preflight_failure_prevents_both_tick_and_application() {
        let decision = stopped_decision();
        let mut ticked = false;
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut core = MpcControlCore {
            port: FakePort {
                events: Rc::clone(&events),
                preflight_error: true,
                apply_error: false,
            },
        };
        assert!(matches!(
            core.execute(
                || {
                    ticked = true;
                    Ok::<_, CoordinatorTickError>(decision)
                },
                |outcome| outcome,
            ),
            Err(MpcControlCycleError::Preflight(FakeError::Preflight))
        ));
        assert!(!ticked);
        assert_eq!(*events.borrow(), ["preflight"]);
    }

    #[test]
    fn application_failure_is_not_reported_as_a_successful_cycle() {
        let decision = stopped_decision();
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut core = MpcControlCore {
            port: FakePort {
                events: Rc::clone(&events),
                preflight_error: false,
                apply_error: true,
            },
        };
        assert!(matches!(
            core.execute(
                || Ok::<_, CoordinatorTickError>(decision),
                |outcome| outcome,
            ),
            Err(MpcControlCycleError::Apply(FakeError::Apply))
        ));
        assert_eq!(*events.borrow(), ["preflight", "apply"]);
    }
}
