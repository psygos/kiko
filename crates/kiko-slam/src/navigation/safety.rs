//! Fail-closed, transport-free supervision for one shadow navigation tick.
//!
//! The supervisor accepts either an explicit typed not-ready reason or a
//! complete set of already-parsed V1 navigation values. A ready attempt owns
//! one monotonically increasing request identifier, derives collision
//! provenance from the exact immutable costmap view, and records exactly one
//! controller request only after the MPC's unforgeable final revalidation has
//! been checked. Every other non-fatal outcome attempts exactly one zero-PWM
//! shadow record. This module owns no transport, callback, or command encoder.

use std::collections::TryReserveError;
use std::fmt;
use std::num::NonZeroU64;

use crate::HostMonotonicTimestamp;

use super::local_costmap::LocalCostmapView;
use super::mpc::{
    CollisionObservationFailure, CollisionProvenanceError, CollisionQuery,
    CollisionSnapshotProvenanceV1, FinalTrajectoryValidationV1, HostMonotonicClock,
    LocalCostmapCapsuleAdapterError, LocalCostmapCapsuleQueryError, LocalCostmapCapsuleQueryV1,
    MPC_REQUEST_V1, MpcConfigV1, MpcFailureKind, MpcReferenceV1, MpcRequestParseError,
    MpcRequestV1, MpcRequestV1Dto, MpcSolveError, MpcSolveProgressV1, MpcSolver, NavigationEpochV1,
    OdomMotionStateV1, PlantModelV1, PredictedOdomPointV1, SolveStatusV1,
};
use super::shadow_command::{
    MotorPacketsSent, ShadowCommandConfig, ShadowCommandError, ShadowCommandRecord,
    ShadowCommandSession, ShadowDecisionId, ShadowPwmPair,
};

/// A nonzero MPC wall-clock allowance, explicitly measured in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SolverBudgetNs(NonZeroU64);

impl SolverBudgetNs {
    pub fn try_new(nanoseconds: u64) -> Result<Self, SolverBudgetError> {
        NonZeroU64::new(nanoseconds)
            .map(Self)
            .ok_or(SolverBudgetError::Zero)
    }

    pub const fn from_nonzero(nanoseconds: NonZeroU64) -> Self {
        Self(nanoseconds)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverBudgetError {
    Zero,
}

impl fmt::Display for SolverBudgetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("shadow MPC solver budget must be nonzero")
    }
}

impl std::error::Error for SolverBudgetError {}

/// A semantic upstream reason that a complete controller request cannot be
/// formed. Strings and partially populated ready payloads are deliberately not
/// admitted at this boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SafetyNotReadyReason {
    SafetyJournalLatched,
    VisualOdometryUnavailable,
    VisualOdometryRejected,
    NavigationGoalUnavailable,
    NavigationGoalInvalidated,
    GlobalPathUnavailable,
    GlobalPathInvalidated,
    DepthObservationUnavailable,
    DepthObservationUnaligned,
    LocalCostmapUnavailable,
    LocalCostmapExpired,
    ReferenceUnavailable,
    NavigationEpochUnavailable,
    NavigationEpochTransition,
    MotionStateUnavailable,
}

impl fmt::Display for SafetyNotReadyReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "navigation is not ready: {self:?}")
    }
}

impl std::error::Error for SafetyNotReadyReason {}

/// The complete, already-parsed input for exactly one ready control attempt.
pub struct SafetyReadyTick<'reference, 'view> {
    epoch: NavigationEpochV1,
    state: OdomMotionStateV1,
    reference: &'reference MpcReferenceV1<'reference>,
    local_costmap: LocalCostmapView<'view>,
    solver_budget: SolverBudgetNs,
}

impl<'reference, 'view> SafetyReadyTick<'reference, 'view> {
    pub fn new(
        epoch: NavigationEpochV1,
        state: OdomMotionStateV1,
        reference: &'reference MpcReferenceV1<'reference>,
        local_costmap: LocalCostmapView<'view>,
        solver_budget: SolverBudgetNs,
    ) -> Self {
        Self {
            epoch,
            state,
            reference,
            local_costmap,
            solver_budget,
        }
    }

    pub fn epoch(&self) -> NavigationEpochV1 {
        self.epoch
    }

    pub fn state(&self) -> OdomMotionStateV1 {
        self.state
    }

    pub fn reference(&self) -> &'reference MpcReferenceV1<'reference> {
        self.reference
    }

    pub fn local_costmap(&self) -> &LocalCostmapView<'view> {
        &self.local_costmap
    }

    pub fn solver_budget(&self) -> SolverBudgetNs {
        self.solver_budget
    }
}

/// Exactly one of these states must be supplied for every supervisor tick.
// Keeping the ready payload inline avoids a heap allocation on every control
// tick; the not-ready size asymmetry is intentional.
#[allow(clippy::large_enum_variant)]
pub enum SafetyTickInput<'reference, 'view> {
    NotReady(SafetyNotReadyReason),
    Ready(SafetyReadyTick<'reference, 'view>),
}

#[derive(Debug)]
pub enum SafetySupervisorCreateError {
    DiagnosticAllocation {
        elements: usize,
        source: TryReserveError,
    },
}

impl fmt::Display for SafetySupervisorCreateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DiagnosticAllocation { elements, .. } => write!(
                formatter,
                "cannot reserve {elements} shadow MPC trajectory diagnostic elements"
            ),
        }
    }
}

impl std::error::Error for SafetySupervisorCreateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DiagnosticAllocation { source, .. } => Some(source),
        }
    }
}

/// A supervisor-level invariant expected from a successful MPC solution.
///
/// The MPC final-validation value itself cannot be constructed outside its
/// module. These checks bind that value to this exact request and solver
/// configuration before a controller request is recorded.
#[derive(Clone, Debug, PartialEq)]
pub enum FinalValidationMismatch {
    RequestId {
        expected: NonZeroU64,
        actual: NonZeroU64,
    },
    CollisionSnapshot {
        expected: Box<CollisionSnapshotProvenanceV1>,
        actual: Box<CollisionSnapshotProvenanceV1>,
    },
    SegmentCount {
        expected: usize,
        actual: usize,
    },
    FinalQueryCount {
        expected: u64,
        actual: u64,
    },
    ValidationTimestamp {
        validation: HostMonotonicTimestamp,
        solve_status: HostMonotonicTimestamp,
    },
    Deadline {
        request: HostMonotonicTimestamp,
        solve_status: HostMonotonicTimestamp,
    },
    PredictedPointCount {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for FinalValidationMismatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid shadow MPC final validation: {self:?}")
    }
}

impl std::error::Error for FinalValidationMismatch {}

/// Why a ready or not-ready tick was forced to zero PWM.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SafetySolverRequestContext {
    request_id: NonZeroU64,
    submitted_at: HostMonotonicTimestamp,
    deadline: HostMonotonicTimestamp,
    state: OdomMotionStateV1,
    reference_epoch: NavigationEpochV1,
    reference_created_at: HostMonotonicTimestamp,
    previous_pwm: ShadowPwmPair,
    collision_snapshot: CollisionSnapshotProvenanceV1,
}

impl SafetySolverRequestContext {
    pub fn request_id(self) -> NonZeroU64 {
        self.request_id
    }

    pub fn submitted_at(self) -> HostMonotonicTimestamp {
        self.submitted_at
    }

    pub fn deadline(self) -> HostMonotonicTimestamp {
        self.deadline
    }

    pub fn state(self) -> OdomMotionStateV1 {
        self.state
    }

    pub fn reference_epoch(self) -> NavigationEpochV1 {
        self.reference_epoch
    }

    pub fn reference_created_at(self) -> HostMonotonicTimestamp {
        self.reference_created_at
    }

    pub fn previous_pwm(self) -> ShadowPwmPair {
        self.previous_pwm
    }

    pub fn collision_snapshot(self) -> CollisionSnapshotProvenanceV1 {
        self.collision_snapshot
    }
}

/// Lifetime-free snapshot of an exact typed MPC failure.
#[derive(Debug)]
pub struct SafetySolverFailure {
    model: PlantModelV1,
    config: MpcConfigV1,
    request: SafetySolverRequestContext,
    progress: MpcSolveProgressV1,
    kind: MpcFailureKind<LocalCostmapCapsuleQueryError>,
}

impl SafetySolverFailure {
    fn from_mpc(source: MpcSolveError<'_, LocalCostmapCapsuleQueryError>) -> Self {
        let request = source.request();
        Self {
            model: source.model(),
            config: source.config(),
            request: SafetySolverRequestContext {
                request_id: request.request_id(),
                submitted_at: request.submitted_at(),
                deadline: request.deadline(),
                state: request.state(),
                reference_epoch: request.reference().epoch(),
                reference_created_at: request.reference().created_at(),
                previous_pwm: request.previous_pwm(),
                collision_snapshot: request.collision_snapshot(),
            },
            progress: source.progress(),
            kind: own_mpc_failure_kind(source.kind()),
        }
    }

    pub fn model(&self) -> PlantModelV1 {
        self.model
    }

    pub fn config(&self) -> MpcConfigV1 {
        self.config
    }

    pub fn request(&self) -> SafetySolverRequestContext {
        self.request
    }

    pub fn progress(&self) -> MpcSolveProgressV1 {
        self.progress
    }

    pub fn kind(&self) -> &MpcFailureKind<LocalCostmapCapsuleQueryError> {
        &self.kind
    }
}

impl fmt::Display for SafetySolverFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "V1 shadow MPC solve failed: {:?}", self.kind)
    }
}

impl std::error::Error for SafetySolverFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            MpcFailureKind::Clock(source) => Some(source),
            MpcFailureKind::CollisionObservation { source, .. } => source.source(),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum SafetyStopCause {
    NotReady(SafetyNotReadyReason),
    RequestIdExhausted,
    DeadlineOverflow {
        tick: HostMonotonicTimestamp,
        budget: SolverBudgetNs,
    },
    CollisionProvenance(CollisionProvenanceError),
    Request(MpcRequestParseError),
    CollisionAdapter(LocalCostmapCapsuleAdapterError),
    Solver(Box<SafetySolverFailure>),
    FinalValidation(FinalValidationMismatch),
}

impl fmt::Display for SafetyStopCause {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotReady(source) => source.fmt(formatter),
            Self::RequestIdExhausted => {
                formatter.write_str("shadow MPC request identifier domain is exhausted")
            }
            Self::DeadlineOverflow { tick, budget } => write!(
                formatter,
                "shadow MPC deadline overflows host time: tick={} ns, budget={} ns",
                tick.as_nanos(),
                budget.get()
            ),
            Self::CollisionProvenance(source) => source.fmt(formatter),
            Self::Request(source) => source.fmt(formatter),
            Self::CollisionAdapter(source) => source.fmt(formatter),
            Self::Solver(source) => source.fmt(formatter),
            Self::FinalValidation(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for SafetyStopCause {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NotReady(source) => Some(source),
            Self::CollisionProvenance(source) => Some(source),
            Self::Request(source) => Some(source),
            Self::CollisionAdapter(source) => Some(source),
            Self::FinalValidation(source) => Some(source),
            Self::Solver(source) => Some(source),
            Self::RequestIdExhausted | Self::DeadlineOverflow { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SafetyControllerDecision {
    request_id: NonZeroU64,
    requested_pwm: ShadowPwmPair,
    final_validation: FinalTrajectoryValidationV1,
    solve_status: SolveStatusV1,
    objective_cost: f64,
}

impl SafetyControllerDecision {
    pub fn request_id(self) -> NonZeroU64 {
        self.request_id
    }

    pub fn requested_pwm(self) -> ShadowPwmPair {
        self.requested_pwm
    }

    pub fn final_validation(self) -> FinalTrajectoryValidationV1 {
        self.final_validation
    }

    pub fn solve_status(self) -> SolveStatusV1 {
        self.solve_status
    }

    pub fn objective_cost(self) -> f64 {
        self.objective_cost
    }
}

#[derive(Debug)]
pub struct SafetyStoppedDecision {
    request_id: Option<NonZeroU64>,
    cause: Box<SafetyStopCause>,
}

impl SafetyStoppedDecision {
    pub fn request_id(&self) -> Option<NonZeroU64> {
        self.request_id
    }

    pub fn cause(&self) -> &SafetyStopCause {
        &self.cause
    }

    pub fn into_cause(self) -> Box<SafetyStopCause> {
        self.cause
    }
}

#[derive(Debug)]
// A successful decision is the hot path; retaining its exact validation
// provenance inline avoids a per-tick allocation.
#[allow(clippy::large_enum_variant)]
pub enum SafetyDecisionOutcome {
    Controller(SafetyControllerDecision),
    Stopped(SafetyStoppedDecision),
}

#[derive(Debug)]
pub struct SafetyDecision {
    record: ShadowCommandRecord,
    outcome: SafetyDecisionOutcome,
}

impl SafetyDecision {
    pub fn record(&self) -> ShadowCommandRecord {
        self.record
    }

    pub fn request_id(&self) -> Option<NonZeroU64> {
        match &self.outcome {
            SafetyDecisionOutcome::Controller(decision) => Some(decision.request_id),
            SafetyDecisionOutcome::Stopped(decision) => decision.request_id,
        }
    }

    pub fn outcome(&self) -> &SafetyDecisionOutcome {
        &self.outcome
    }

    pub fn into_outcome(self) -> SafetyDecisionOutcome {
        self.outcome
    }

    pub fn motor_packets_sent(&self) -> MotorPacketsSent {
        MotorPacketsSent::ZERO
    }
}

/// A shadow-session write failure is fatal: no decision was recorded, and the
/// caller must not reinterpret the attempted result as a successful STOP.
#[derive(Debug)]
pub enum SafetyFatalError {
    StopRecording {
        recorded_at: HostMonotonicTimestamp,
        request_id: Option<NonZeroU64>,
        cause: Box<SafetyStopCause>,
        source: ShadowCommandError,
    },
    ControllerRecording {
        recorded_at: HostMonotonicTimestamp,
        request_id: NonZeroU64,
        requested_pwm: ShadowPwmPair,
        final_validation: Box<FinalTrajectoryValidationV1>,
        source: ShadowCommandError,
    },
}

pub type SafetyDecideError = Box<SafetyFatalError>;

impl SafetyFatalError {
    pub fn recorded_at(&self) -> HostMonotonicTimestamp {
        match self {
            Self::StopRecording { recorded_at, .. }
            | Self::ControllerRecording { recorded_at, .. } => *recorded_at,
        }
    }

    pub fn request_id(&self) -> Option<NonZeroU64> {
        match self {
            Self::StopRecording { request_id, .. } => *request_id,
            Self::ControllerRecording { request_id, .. } => Some(*request_id),
        }
    }

    pub fn shadow_command_error(&self) -> ShadowCommandError {
        match self {
            Self::StopRecording { source, .. } | Self::ControllerRecording { source, .. } => {
                *source
            }
        }
    }
}

impl fmt::Display for SafetyFatalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StopRecording { cause, source, .. } => {
                write!(
                    formatter,
                    "failed to record fail-closed STOP after {cause}: {source}"
                )
            }
            Self::ControllerRecording {
                request_id, source, ..
            } => write!(
                formatter,
                "failed to record approved shadow MPC request {}: {source}",
                request_id.get()
            ),
        }
    }
}

impl std::error::Error for SafetyFatalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StopRecording { source, .. } | Self::ControllerRecording { source, .. } => {
                Some(source)
            }
        }
    }
}

/// A copied diagnostic whose identity proves which recorded decision produced
/// it. The slice is cleared before every tick and remains absent after STOP or
/// a fatal recording error.
#[derive(Clone, Copy, Debug)]
pub struct BoundPredictedTrajectory<'trajectory> {
    decision_id: ShadowDecisionId,
    points: &'trajectory [PredictedOdomPointV1],
}

impl<'trajectory> BoundPredictedTrajectory<'trajectory> {
    pub fn decision_id(self) -> ShadowDecisionId {
        self.decision_id
    }

    pub fn points(self) -> &'trajectory [PredictedOdomPointV1] {
        self.points
    }
}

/// Owns the concrete solver and concrete transport-free shadow session.
pub struct ShadowSafetySupervisor {
    solver: MpcSolver,
    shadow_session: ShadowCommandSession,
    next_ready_request_id: Option<NonZeroU64>,
    last_success_trajectory: Vec<PredictedOdomPointV1>,
    last_success_decision_id: Option<ShadowDecisionId>,
    #[cfg(test)]
    last_solver_trajectory_address: Option<usize>,
}

impl ShadowSafetySupervisor {
    pub fn try_new(
        solver: MpcSolver,
        shadow_config: ShadowCommandConfig,
    ) -> Result<Self, SafetySupervisorCreateError> {
        let diagnostic_elements = solver.config().horizon_steps();
        let mut last_success_trajectory = Vec::new();
        last_success_trajectory
            .try_reserve_exact(diagnostic_elements)
            .map_err(|source| SafetySupervisorCreateError::DiagnosticAllocation {
                elements: diagnostic_elements,
                source,
            })?;
        Ok(Self {
            solver,
            shadow_session: ShadowCommandSession::new(shadow_config),
            next_ready_request_id: Some(NonZeroU64::MIN),
            last_success_trajectory,
            last_success_decision_id: None,
            #[cfg(test)]
            last_solver_trajectory_address: None,
        })
    }

    pub fn solver(&self) -> &MpcSolver {
        &self.solver
    }

    pub fn shadow_session(&self) -> &ShadowCommandSession {
        &self.shadow_session
    }

    pub fn last_success_trajectory(&self) -> Option<BoundPredictedTrajectory<'_>> {
        self.last_success_decision_id
            .map(|decision_id| BoundPredictedTrajectory {
                decision_id,
                points: &self.last_success_trajectory,
            })
    }

    pub fn motor_packets_sent(&self) -> MotorPacketsSent {
        MotorPacketsSent::ZERO
    }

    /// Make exactly one fail-closed shadow decision for this host timestamp.
    pub fn decide<'reference, C>(
        &mut self,
        tick: HostMonotonicTimestamp,
        input: SafetyTickInput<'reference, '_>,
        clock: &mut C,
    ) -> Result<SafetyDecision, SafetyDecideError>
    where
        C: HostMonotonicClock,
    {
        self.clear_diagnostic();
        match input {
            SafetyTickInput::NotReady(reason) => {
                self.record_stop(tick, None, SafetyStopCause::NotReady(reason))
            }
            SafetyTickInput::Ready(ready) => self.decide_ready(tick, ready, clock),
        }
    }

    fn decide_ready<'reference, C>(
        &mut self,
        tick: HostMonotonicTimestamp,
        ready: SafetyReadyTick<'reference, '_>,
        clock: &mut C,
    ) -> Result<SafetyDecision, SafetyDecideError>
    where
        C: HostMonotonicClock,
    {
        let Some(request_id) = self.take_ready_request_id() else {
            return self.record_stop(tick, None, SafetyStopCause::RequestIdExhausted);
        };
        let collision_snapshot =
            match CollisionSnapshotProvenanceV1::from_runtime(ready.epoch, &ready.local_costmap) {
                Ok(snapshot) => snapshot,
                Err(source) => {
                    return self.record_stop(
                        tick,
                        Some(request_id),
                        SafetyStopCause::CollisionProvenance(source),
                    );
                }
            };
        let Some(budget_deadline_ns) = tick.as_nanos().checked_add(ready.solver_budget.get())
        else {
            return self.record_stop(
                tick,
                Some(request_id),
                SafetyStopCause::DeadlineOverflow {
                    tick,
                    budget: ready.solver_budget,
                },
            );
        };
        let deadline_ns = budget_deadline_ns.min(collision_snapshot.valid_through().as_nanos());
        let previous_pwm = self
            .shadow_session
            .latest()
            .map_or(ShadowPwmPair::STOP, ShadowCommandRecord::pwm);
        let request = match MpcRequestV1::parse(
            MpcRequestV1Dto {
                schema_version: MPC_REQUEST_V1,
                request_id: request_id.get(),
                submitted_at_host_ns: tick.as_nanos(),
                deadline_host_ns: deadline_ns,
            },
            ready.state,
            ready.reference,
            previous_pwm,
            collision_snapshot,
        ) {
            Ok(request) => request,
            Err(source) => {
                return self.record_stop(tick, Some(request_id), SafetyStopCause::Request(source));
            }
        };
        let mut collision =
            match LocalCostmapCapsuleQueryV1::try_new(ready.local_costmap, collision_snapshot) {
                Ok(collision) => collision,
                Err(source) => {
                    return self.record_stop(
                        tick,
                        Some(request_id),
                        SafetyStopCause::CollisionAdapter(source),
                    );
                }
            };
        self.solve_and_record(
            tick,
            request_id,
            request,
            collision_snapshot,
            &mut collision,
            clock,
        )
    }

    fn solve_and_record<'reference, Q, C>(
        &mut self,
        tick: HostMonotonicTimestamp,
        request_id: NonZeroU64,
        request: MpcRequestV1<'reference>,
        collision_snapshot: CollisionSnapshotProvenanceV1,
        collision: &mut Q,
        clock: &mut C,
    ) -> Result<SafetyDecision, SafetyDecideError>
    where
        Q: CollisionQuery<Error = LocalCostmapCapsuleQueryError>,
        C: HostMonotonicClock,
    {
        let solver_config = self.solver.config();
        let solution = match self.solver.solve(request, collision, clock) {
            Ok(solution) => solution,
            Err(source) => {
                let recorded_at = match source.progress() {
                    MpcSolveProgressV1::NotStarted => tick,
                    MpcSolveProgressV1::InProgress(status) => tick.max(status.observed_at()),
                };
                let source = SafetySolverFailure::from_mpc(source);
                return self.record_stop(
                    recorded_at,
                    Some(request_id),
                    SafetyStopCause::Solver(Box::new(source)),
                );
            }
        };
        let requested_pwm = solution.requested_pwm();
        let final_validation = solution.final_validation();
        let solve_status = solution.status();
        let objective_cost = solution.objective_cost();
        if let Err(source) = validate_final_solution(
            request_id,
            collision_snapshot,
            solver_config.horizon_steps(),
            solver_config.integration_substeps(),
            &solution,
        ) {
            let recorded_at = tick.max(solve_status.observed_at());
            return self.record_stop(
                recorded_at,
                Some(request_id),
                SafetyStopCause::FinalValidation(source),
            );
        }

        #[cfg(test)]
        {
            self.last_solver_trajectory_address =
                Some(solution.predicted_trajectory().as_ptr() as usize);
        }
        self.last_success_trajectory
            .extend_from_slice(solution.predicted_trajectory());
        let recorded_at = final_validation.validated_at();
        let record = match self
            .shadow_session
            .record_controller_request(recorded_at, requested_pwm)
        {
            Ok(record) => record,
            Err(source) => {
                self.clear_diagnostic();
                return Err(Box::new(SafetyFatalError::ControllerRecording {
                    recorded_at,
                    request_id,
                    requested_pwm,
                    final_validation: Box::new(final_validation),
                    source,
                }));
            }
        };
        self.last_success_decision_id = Some(record.decision_id());
        Ok(SafetyDecision {
            record,
            outcome: SafetyDecisionOutcome::Controller(SafetyControllerDecision {
                request_id,
                requested_pwm,
                final_validation,
                solve_status,
                objective_cost,
            }),
        })
    }

    fn record_stop(
        &mut self,
        recorded_at: HostMonotonicTimestamp,
        request_id: Option<NonZeroU64>,
        cause: SafetyStopCause,
    ) -> Result<SafetyDecision, SafetyDecideError> {
        self.clear_diagnostic();
        let cause = Box::new(cause);
        match self.shadow_session.record_fail_closed_stop(recorded_at) {
            Ok(record) => Ok(SafetyDecision {
                record,
                outcome: SafetyDecisionOutcome::Stopped(SafetyStoppedDecision {
                    request_id,
                    cause,
                }),
            }),
            Err(source) => Err(Box::new(SafetyFatalError::StopRecording {
                recorded_at,
                request_id,
                cause,
                source,
            })),
        }
    }

    fn take_ready_request_id(&mut self) -> Option<NonZeroU64> {
        let request_id = self.next_ready_request_id?;
        self.next_ready_request_id = request_id.get().checked_add(1).and_then(NonZeroU64::new);
        Some(request_id)
    }

    fn clear_diagnostic(&mut self) {
        self.last_success_trajectory.clear();
        self.last_success_decision_id = None;
        #[cfg(test)]
        {
            self.last_solver_trajectory_address = None;
        }
    }

    #[cfg(test)]
    fn set_next_ready_request_id_for_test(&mut self, request_id: Option<NonZeroU64>) {
        self.next_ready_request_id = request_id;
    }
}

fn validate_final_solution(
    expected_request_id: NonZeroU64,
    expected_collision: CollisionSnapshotProvenanceV1,
    horizon_steps: usize,
    integration_substeps: usize,
    solution: &super::mpc::MpcSolution<'_, '_>,
) -> Result<(), FinalValidationMismatch> {
    let actual_request_id = solution.request().request_id();
    if actual_request_id != expected_request_id {
        return Err(FinalValidationMismatch::RequestId {
            expected: expected_request_id,
            actual: actual_request_id,
        });
    }
    let validation = solution.final_validation();
    if validation.collision_snapshot() != expected_collision {
        return Err(FinalValidationMismatch::CollisionSnapshot {
            expected: Box::new(expected_collision),
            actual: Box::new(validation.collision_snapshot()),
        });
    }
    let expected_segments = horizon_steps
        .checked_mul(integration_substeps)
        .expect("parsed MPC dimensions cannot overflow usize");
    if validation.segment_count() != expected_segments {
        return Err(FinalValidationMismatch::SegmentCount {
            expected: expected_segments,
            actual: validation.segment_count(),
        });
    }
    let status = solution.status();
    let expected_queries = expected_segments as u64;
    if status.final_validation_queries() != expected_queries {
        return Err(FinalValidationMismatch::FinalQueryCount {
            expected: expected_queries,
            actual: status.final_validation_queries(),
        });
    }
    if validation.validated_at() != status.observed_at() {
        return Err(FinalValidationMismatch::ValidationTimestamp {
            validation: validation.validated_at(),
            solve_status: status.observed_at(),
        });
    }
    if solution.request().deadline() != status.deadline() {
        return Err(FinalValidationMismatch::Deadline {
            request: solution.request().deadline(),
            solve_status: status.deadline(),
        });
    }
    if solution.predicted_trajectory().len() != horizon_steps {
        return Err(FinalValidationMismatch::PredictedPointCount {
            expected: horizon_steps,
            actual: solution.predicted_trajectory().len(),
        });
    }
    Ok(())
}

fn own_mpc_failure_kind(
    source: &MpcFailureKind<LocalCostmapCapsuleQueryError>,
) -> MpcFailureKind<LocalCostmapCapsuleQueryError> {
    match source {
        MpcFailureKind::Clock(source) => MpcFailureKind::Clock(*source),
        MpcFailureKind::CollisionSnapshotMismatch { requested, actual } => {
            MpcFailureKind::CollisionSnapshotMismatch {
                requested: Box::new(**requested),
                actual: Box::new(**actual),
            }
        }
        MpcFailureKind::OccupiedStart => MpcFailureKind::OccupiedStart,
        MpcFailureKind::PreviousPwmOutsideEnvelope { wheel, value } => {
            MpcFailureKind::PreviousPwmOutsideEnvelope {
                wheel: *wheel,
                value: *value,
            }
        }
        MpcFailureKind::ReferenceDoesNotMatchConfig {
            expected_steps,
            actual_steps,
            expected_period_s,
            actual_period_s,
        } => MpcFailureKind::ReferenceDoesNotMatchConfig {
            expected_steps: *expected_steps,
            actual_steps: *actual_steps,
            expected_period_s: *expected_period_s,
            actual_period_s: *actual_period_s,
        },
        MpcFailureKind::PlantEnvelope(source) => MpcFailureKind::PlantEnvelope(*source),
        MpcFailureKind::CollisionBlocked {
            horizon_step,
            integration_substep,
        } => MpcFailureKind::CollisionBlocked {
            horizon_step: *horizon_step,
            integration_substep: *integration_substep,
        },
        MpcFailureKind::FinalTrajectoryBlocked {
            horizon_step,
            integration_substep,
        } => MpcFailureKind::FinalTrajectoryBlocked {
            horizon_step: *horizon_step,
            integration_substep: *integration_substep,
        },
        MpcFailureKind::IntegrationTubeExceeded {
            horizon_step,
            integration_substep,
            required_m,
            allowed_m,
        } => MpcFailureKind::IntegrationTubeExceeded {
            horizon_step: *horizon_step,
            integration_substep: *integration_substep,
            required_m: *required_m,
            allowed_m: *allowed_m,
        },
        MpcFailureKind::CollisionObservation {
            horizon_step,
            integration_substep,
            final_revalidation,
            source,
        } => MpcFailureKind::CollisionObservation {
            horizon_step: *horizon_step,
            integration_substep: *integration_substep,
            final_revalidation: *final_revalidation,
            source: match source {
                CollisionObservationFailure::Query(query) => {
                    CollisionObservationFailure::Query(*query)
                }
                CollisionObservationFailure::Clock(clock) => {
                    CollisionObservationFailure::Clock(*clock)
                }
                CollisionObservationFailure::QueryAndClock { query, clock } => {
                    CollisionObservationFailure::QueryAndClock {
                        query: *query,
                        clock: *clock,
                    }
                }
            },
        },
        MpcFailureKind::Numerical {
            stage,
            horizon_step,
            integration_substep,
        } => MpcFailureKind::Numerical {
            stage: *stage,
            horizon_step: *horizon_step,
            integration_substep: *integration_substep,
        },
        MpcFailureKind::EvaluationLimit { configured } => MpcFailureKind::EvaluationLimit {
            configured: *configured,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::f64::consts::FRAC_PI_4;
    use std::time::Duration;

    use crate::dense::occupancy::{
        DepthCameraModel, DepthRangeMeters, DepthToTrackingCamera, HeightRangeMeters,
        OccupancyConfig, OccupancyEvidenceModel, OccupancyGridGeometry, OccupancyMapper,
        WorldToOccupancy,
    };
    use crate::map::SlamMap;
    use crate::{
        DepthImage, DepthObservation, DeviceSessionId, FrameDimensions, FrameId, MapSnapshot,
        PinholeIntrinsics, Pose, Timestamp,
    };

    use super::super::mpc::{
        ClockFault, ConservativeCapsuleSegmentV1, HostMonotonicClockFailure,
        HostMonotonicClockReadError, MpcConfigV1Dto, ODOM_MOTION_STATE_V1, OdomMotionStateV1Dto,
        OdomReferencePointV1Dto, PLANT_MODEL_V1, PlantEvidenceV1Dto, PlantModelV1Dto,
        PlantValidityEnvelopeV1Dto, ReferenceBuilderRevisionV1, WheelPlantV1Dto,
    };
    use super::super::{
        BaseToOdom, GlobalPath, GlobalPlanner, GlobalPlannerConfig, LocalCostmap,
        LocalCostmapConfig, LocalDepthObservation, MapPoint, OdomSegmentId, PlanStart, PointGoal,
        ShadowCommandConfigDto, ShadowCommandDisposition, TimeAlignedOdomPose, TimeAlignment,
        TrackingCameraToBase, UnknownSpacePolicy,
    };
    use super::*;

    fn plant_dto() -> PlantModelV1Dto {
        PlantModelV1Dto {
            schema_version: PLANT_MODEL_V1,
            model_id: "safety-test-plant".into(),
            model_version: 1,
            sample_period_s: 0.1,
            wheelbase_m: 0.5,
            left: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.4,
            },
            right: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.5,
            },
            validity: PlantValidityEnvelopeV1Dto {
                left_pwm_min_percent: -100,
                left_pwm_max_percent: 100,
                right_pwm_min_percent: -100,
                right_pwm_max_percent: 100,
                left_velocity_min_mps: -2.0,
                left_velocity_max_mps: 2.0,
                right_velocity_min_mps: -2.0,
                right_velocity_max_mps: 2.0,
                max_abs_yaw_rate_rad_s: 2.0,
                max_abs_lateral_velocity_mps: 0.2,
            },
            evidence: PlantEvidenceV1Dto::SyntheticFixture {
                fixture_id: "safety-unit".into(),
                generator_id: "hand".into(),
            },
        }
    }

    fn config_dto() -> MpcConfigV1Dto {
        MpcConfigV1Dto {
            schema_version: super::super::mpc::MPC_CONFIG_V1,
            horizon_steps: 1,
            step_period_s: 0.1,
            integration_substeps: 1,
            optimization_iterations: 1,
            candidates_per_wheel: 3,
            max_rollout_evaluations: 100,
            initial_search_radius_percent: 10,
            search_radius_decay_numerator: 1,
            search_radius_decay_denominator: 2,
            left_pwm_min_percent: -50,
            left_pwm_max_percent: 50,
            right_pwm_min_percent: -50,
            right_pwm_max_percent: 50,
            left_max_slew_percent_per_step: 200,
            right_max_slew_percent_per_step: 200,
            max_integration_tube_radius_m: 1.0,
            position_cost_per_m2: 1_000.0,
            heading_cost_per_rad2: 1.0,
            forward_velocity_cost_s2_per_m2: 1_000.0,
            yaw_rate_cost_s2_per_rad2: 1.0,
            pwm_cost_per_percent2: 0.0,
            slew_cost_per_percent2: 0.0,
            terminal_state_cost_multiplier: 2.0,
        }
    }

    fn dimensions(width: u32, height: u32) -> FrameDimensions {
        FrameDimensions::try_new(width, height).expect("test dimensions")
    }

    fn camera(width: u32, height: u32) -> DepthCameraModel {
        DepthCameraModel::new(
            PinholeIntrinsics::try_new(
                4.0,
                4.0,
                width.saturating_sub(1) as f32 * 0.5,
                height.saturating_sub(1) as f32 * 0.5,
            )
            .expect("test intrinsics"),
            dimensions(width, height),
            DepthToTrackingCamera::identity(),
        )
    }

    fn global_path_fixture() -> (MapSnapshot, GlobalPath) {
        let map_snapshot = SlamMap::new().snapshot();
        let config = OccupancyConfig::try_new(
            OccupancyGridGeometry::try_new(1.0, [-2.0, -2.0], 6, 6, 36).expect("global geometry"),
            WorldToOccupancy::level_optical_world(1.0).expect("occupancy frame"),
            camera(9, 5),
            HeightRangeMeters::try_new(0.0, 2.0).expect("height"),
            DepthRangeMeters::try_new(0.1, 8.0).expect("depth"),
            1,
            OccupancyEvidenceModel::try_new(-1, 3, -1, 1).expect("evidence"),
            1,
        )
        .expect("occupancy config");
        let mut mapper = OccupancyMapper::try_new(config).expect("mapper");
        mapper
            .reset_to_map(map_snapshot.instance_id())
            .expect("map identity");
        let occupancy = mapper.snapshot().expect("occupancy snapshot");
        let mut planner = GlobalPlanner::try_new(
            &occupancy,
            GlobalPlannerConfig::try_new(0.0, UnknownSpacePolicy::Traversable)
                .expect("planner config"),
        )
        .expect("planner");
        let start_point = MapPoint::try_new(0.0, 0.0).expect("start");
        let goal_point = MapPoint::try_new(1.0, 0.0).expect("goal");
        let start = PlanStart::for_snapshot(start_point, &occupancy).expect("start provenance");
        let goal = PointGoal::for_snapshot(goal_point, &occupancy).expect("goal provenance");
        (map_snapshot, planner.plan(start, goal).expect("path"))
    }

    fn optical_to_base() -> TrackingCameraToBase {
        TrackingCameraToBase::new(
            Pose::try_from_rt(
                [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [-0.5, 0.0, 0.5],
            )
            .expect("proper optical-to-base transform"),
        )
    }

    fn local_costmap_fixture(
        session: DeviceSessionId,
        odom_segment_id: OdomSegmentId,
        host_ns: u64,
        max_age_ns: u64,
        base_to_odom: BaseToOdom,
    ) -> LocalCostmap {
        let config = LocalCostmapConfig::try_new(
            OccupancyGridGeometry::try_new(0.25, [-1.125, -1.125], 12, 8, 96)
                .expect("local geometry"),
            camera(9, 5),
            optical_to_base(),
            HeightRangeMeters::try_new(0.1, 1.5).expect("obstacle height"),
            DepthRangeMeters::try_new(0.1, 8.0).expect("local depth"),
            1,
            0.1,
            0.0,
            Duration::from_nanos(max_age_ns),
        )
        .expect("local config");
        let mut costmap = LocalCostmap::try_new(config, session).expect("local costmap");
        let image = DepthImage::new(
            FrameId::new(1),
            Timestamp::from_nanos(1),
            9,
            5,
            vec![2.0; 45],
        )
        .expect("depth image");
        let source =
            DepthObservation::parse(session, HostMonotonicTimestamp::from_nanos(host_ns), image)
                .expect("depth provenance");
        let pose = TimeAlignedOdomPose::from_validated_parts_for_test(
            odom_segment_id,
            session,
            source.device_timestamp(),
            base_to_odom,
            TimeAlignment::ExactVisual,
        );
        costmap
            .update(
                LocalDepthObservation::try_from_time_aligned(source, pose)
                    .expect("time-aligned depth"),
            )
            .expect("costmap update");
        costmap
    }

    struct RuntimeFixture {
        path: GlobalPath,
        epoch: NavigationEpochV1,
        collision: CollisionSnapshotProvenanceV1,
        config: MpcConfigV1,
        model: PlantModelV1,
        costmap: LocalCostmap,
    }

    fn runtime_fixture() -> RuntimeFixture {
        let session = DeviceSessionId::try_new(1).expect("session");
        let (map_snapshot, path) = global_path_fixture();
        let epoch = NavigationEpochV1::from_runtime(
            session,
            OdomSegmentId::try_new(1).expect("odom segment"),
            map_snapshot,
            &path,
        )
        .expect("epoch");
        let costmap = local_costmap_fixture(
            session,
            epoch.odom_segment_id(),
            10,
            1_000_000,
            BaseToOdom::try_new(0.0, 0.0, 0.0).expect("capture pose"),
        );
        let view = costmap
            .view_at(HostMonotonicTimestamp::from_nanos(20))
            .expect("current view");
        let collision =
            CollisionSnapshotProvenanceV1::from_runtime(epoch, &view).expect("collision");
        RuntimeFixture {
            path,
            epoch,
            collision,
            config: MpcConfigV1::parse(config_dto()).expect("config"),
            model: PlantModelV1::parse(plant_dto()).expect("model"),
            costmap,
        }
    }

    fn state_at(epoch: NavigationEpochV1, observed_at_ns: u64) -> OdomMotionStateV1 {
        OdomMotionStateV1::parse(
            OdomMotionStateV1Dto {
                schema_version: ODOM_MOTION_STATE_V1,
                observed_at_host_ns: observed_at_ns,
                x_m: 0.0,
                y_m: 0.0,
                yaw_rad: 0.0,
                odom_velocity_x_mps: 0.0,
                odom_velocity_y_mps: 0.0,
                yaw_rate_rad_s: 0.0,
            },
            epoch,
        )
        .expect("state")
    }

    fn reference_at<'path>(
        fixture: &'path RuntimeFixture,
        created_at_ns: u64,
    ) -> MpcReferenceV1<'path> {
        MpcReferenceV1::parse(
            super::super::mpc::MpcReferenceV1Dto {
                schema_version: super::super::mpc::MPC_REFERENCE_V1,
                builder_revision: ReferenceBuilderRevisionV1::TimeParameterizedGlobalPathV1 as u32,
                created_at_host_ns: created_at_ns,
                step_period_s: fixture.config.step_period_s(),
                points: vec![OdomReferencePointV1Dto {
                    x_m: 0.02,
                    y_m: 0.0,
                    yaw_rad: 0.0,
                    forward_velocity_mps: 0.2,
                    yaw_rate_rad_s: 0.0,
                }],
            },
            fixture.config,
            fixture.epoch,
            &fixture.path,
        )
        .expect("reference")
    }

    fn ready<'reference, 'view>(
        fixture: &'view RuntimeFixture,
        reference: &'reference MpcReferenceV1<'reference>,
        state: OdomMotionStateV1,
        view_at_ns: u64,
        budget_ns: u64,
    ) -> SafetyReadyTick<'reference, 'view> {
        SafetyReadyTick::new(
            fixture.epoch,
            state,
            reference,
            fixture
                .costmap
                .view_at(HostMonotonicTimestamp::from_nanos(view_at_ns))
                .expect("monotonic view"),
            SolverBudgetNs::try_new(budget_ns).expect("nonzero budget"),
        )
    }

    fn supervisor(fixture: &RuntimeFixture) -> ShadowSafetySupervisor {
        supervisor_with_capacity(fixture, 64)
    }

    fn supervisor_with_capacity(
        fixture: &RuntimeFixture,
        retained_records: usize,
    ) -> ShadowSafetySupervisor {
        ShadowSafetySupervisor::try_new(
            MpcSolver::new(fixture.model, fixture.config).expect("solver"),
            ShadowCommandConfig::parse(ShadowCommandConfigDto {
                lease_ms: 150,
                retained_records,
                initial_sequence: 0,
            })
            .expect("shadow config"),
        )
        .expect("supervisor")
    }

    fn stopped(decision: &SafetyDecision) -> &SafetyStoppedDecision {
        match decision.outcome() {
            SafetyDecisionOutcome::Stopped(stopped) => stopped,
            SafetyDecisionOutcome::Controller(_) => panic!("expected STOP decision"),
        }
    }

    fn assert_stop_record(supervisor: &ShadowSafetySupervisor, decision: &SafetyDecision) {
        assert_eq!(
            decision.record().disposition(),
            ShadowCommandDisposition::FailClosedStop
        );
        assert!(decision.record().pwm().is_stop());
        assert_eq!(decision.motor_packets_sent(), MotorPacketsSent::ZERO);
        assert_eq!(supervisor.motor_packets_sent(), MotorPacketsSent::ZERO);
        assert_eq!(
            supervisor.shadow_session().motor_packets_sent(),
            MotorPacketsSent::ZERO
        );
    }

    struct ConstantClock(HostMonotonicTimestamp);

    impl HostMonotonicClock for ConstantClock {
        fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
            Ok(self.0)
        }
    }

    struct ScriptedClock {
        times: VecDeque<HostMonotonicTimestamp>,
        last: HostMonotonicTimestamp,
    }

    impl ScriptedClock {
        fn new(times_ns: &[u64]) -> Self {
            Self {
                times: times_ns
                    .iter()
                    .copied()
                    .map(HostMonotonicTimestamp::from_nanos)
                    .collect(),
                last: HostMonotonicTimestamp::from_nanos(
                    *times_ns.last().expect("nonempty clock script"),
                ),
            }
        }
    }

    impl HostMonotonicClock for ScriptedClock {
        fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
            Ok(self.times.pop_front().unwrap_or(self.last))
        }
    }

    const INJECTED_CLOCK_READ_ERROR: HostMonotonicClockReadError =
        HostMonotonicClockReadError::ElapsedNanosecondsOutOfRange {
            elapsed_nanoseconds: 18_446_744_073_709_551_616,
        };

    struct ReadFailingClock {
        now: HostMonotonicTimestamp,
        fail_at_call: usize,
        calls: usize,
    }

    impl ReadFailingClock {
        fn new(now: HostMonotonicTimestamp, fail_at_call: usize) -> Self {
            Self {
                now,
                fail_at_call,
                calls: 0,
            }
        }
    }

    impl HostMonotonicClock for ReadFailingClock {
        fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
            self.calls += 1;
            if self.calls == self.fail_at_call {
                Err(INJECTED_CLOCK_READ_ERROR)
            } else {
                Ok(self.now)
            }
        }
    }

    struct ScriptedQuery {
        snapshot: CollisionSnapshotProvenanceV1,
        calls: usize,
        error_on: Option<usize>,
        block_on: Option<usize>,
    }

    impl ScriptedQuery {
        fn clear(snapshot: CollisionSnapshotProvenanceV1) -> Self {
            Self {
                snapshot,
                calls: 0,
                error_on: None,
                block_on: None,
            }
        }
    }

    impl CollisionQuery for ScriptedQuery {
        type Error = LocalCostmapCapsuleQueryError;

        fn snapshot_provenance(&self) -> CollisionSnapshotProvenanceV1 {
            self.snapshot
        }

        fn is_capsule_traversable(
            &mut self,
            _: ConservativeCapsuleSegmentV1,
        ) -> Result<bool, Self::Error> {
            self.calls += 1;
            if self.error_on == Some(self.calls) {
                Err(LocalCostmapCapsuleQueryError::NumericalBounds)
            } else {
                Ok(self.block_on != Some(self.calls))
            }
        }
    }

    fn request<'reference>(
        fixture: &'reference RuntimeFixture,
        reference: &'reference MpcReferenceV1<'reference>,
        request_id: NonZeroU64,
        submitted_at_ns: u64,
        deadline_ns: u64,
        previous_pwm: ShadowPwmPair,
    ) -> MpcRequestV1<'reference> {
        MpcRequestV1::parse(
            MpcRequestV1Dto {
                schema_version: MPC_REQUEST_V1,
                request_id: request_id.get(),
                submitted_at_host_ns: submitted_at_ns,
                deadline_host_ns: deadline_ns,
            },
            state_at(fixture.epoch, 11),
            reference,
            previous_pwm,
            fixture.collision,
        )
        .expect("request")
    }

    #[test]
    fn budget_is_parsed_once_and_supervisor_is_transport_free_and_sendable() {
        assert_eq!(SolverBudgetNs::try_new(0), Err(SolverBudgetError::Zero));
        assert_eq!(SolverBudgetNs::try_new(7).expect("budget").get(), 7);
        fn assert_send<T: Send>() {}
        assert_send::<ShadowSafetySupervisor>();
        let fixture = runtime_fixture();
        assert_eq!(supervisor(&fixture).motor_packets_sent().get(), 0);
    }

    #[test]
    fn initial_clock_read_failure_records_stop_without_fabricated_progress() {
        let fixture = runtime_fixture();
        let reference = reference_at(&fixture, 12);
        let tick = HostMonotonicTimestamp::from_nanos(20);
        let mut supervisor = supervisor(&fixture);
        let mut clock = ReadFailingClock::new(HostMonotonicTimestamp::from_nanos(30), 1);

        let decision = supervisor
            .decide(
                tick,
                SafetyTickInput::Ready(ready(
                    &fixture,
                    &reference,
                    state_at(fixture.epoch, 11),
                    tick.as_nanos(),
                    1_000,
                )),
                &mut clock,
            )
            .expect("clock read failure must remain an admitted STOP decision");

        assert_eq!(clock.calls, 1);
        assert_stop_record(&supervisor, &decision);
        assert_eq!(decision.record().recorded_at(), tick);
        assert!(supervisor.last_success_trajectory().is_none());
        let SafetyStopCause::Solver(source) = stopped(&decision).cause() else {
            panic!("initial clock failure must retain its solver cause")
        };
        assert_eq!(source.progress(), MpcSolveProgressV1::NotStarted);
        assert!(matches!(
            source.kind(),
            MpcFailureKind::Clock(HostMonotonicClockFailure::Read(actual))
                if *actual == INJECTED_CLOCK_READ_ERROR
        ));
        let clock_failure = std::error::Error::source(source.as_ref())
            .expect("safety failure must expose its typed clock cause");
        assert_eq!(
            clock_failure.downcast_ref::<HostMonotonicClockFailure>(),
            Some(&HostMonotonicClockFailure::Read(INJECTED_CLOCK_READ_ERROR))
        );
    }

    #[test]
    fn second_clock_read_failure_stops_at_last_truthful_observation_with_zero_work() {
        let fixture = runtime_fixture();
        let reference = reference_at(&fixture, 12);
        let tick = HostMonotonicTimestamp::from_nanos(20);
        let first_observation = HostMonotonicTimestamp::from_nanos(30);
        let mut supervisor = supervisor(&fixture);
        let mut clock = ReadFailingClock::new(first_observation, 2);

        let decision = supervisor
            .decide(
                tick,
                SafetyTickInput::Ready(ready(
                    &fixture,
                    &reference,
                    state_at(fixture.epoch, 11),
                    tick.as_nanos(),
                    1_000,
                )),
                &mut clock,
            )
            .expect("second clock read failure must journal STOP");

        assert_eq!(clock.calls, 2);
        assert_stop_record(&supervisor, &decision);
        assert_eq!(decision.record().recorded_at(), first_observation);
        assert!(supervisor.last_success_trajectory().is_none());
        let SafetyStopCause::Solver(source) = stopped(&decision).cause() else {
            panic!("clock failure must retain its solver cause")
        };
        let MpcSolveProgressV1::InProgress(status) = source.progress() else {
            panic!("the successful first read established solver progress")
        };
        assert_eq!(status.started_at(), first_observation);
        assert_eq!(status.observed_at(), first_observation);
        assert_eq!(status.completed_iterations(), 0);
        assert_eq!(status.rollout_evaluations(), 0);
        assert_eq!(status.pre_final_collision_queries(), 0);
        assert_eq!(status.final_validation_queries(), 0);
        assert!(matches!(
            source.kind(),
            MpcFailureKind::Clock(HostMonotonicClockFailure::Read(actual))
                if *actual == INJECTED_CLOCK_READ_ERROR
        ));
    }

    #[test]
    fn every_not_ready_reason_records_exactly_one_bounded_stop_without_a_request_id() {
        struct PanicClock;
        impl HostMonotonicClock for PanicClock {
            fn try_now(&mut self) -> Result<HostMonotonicTimestamp, HostMonotonicClockReadError> {
                panic!("not-ready ticks must not query the solver clock")
            }
        }

        let reasons = [
            SafetyNotReadyReason::SafetyJournalLatched,
            SafetyNotReadyReason::VisualOdometryUnavailable,
            SafetyNotReadyReason::VisualOdometryRejected,
            SafetyNotReadyReason::NavigationGoalUnavailable,
            SafetyNotReadyReason::NavigationGoalInvalidated,
            SafetyNotReadyReason::GlobalPathUnavailable,
            SafetyNotReadyReason::GlobalPathInvalidated,
            SafetyNotReadyReason::DepthObservationUnavailable,
            SafetyNotReadyReason::DepthObservationUnaligned,
            SafetyNotReadyReason::LocalCostmapUnavailable,
            SafetyNotReadyReason::LocalCostmapExpired,
            SafetyNotReadyReason::ReferenceUnavailable,
            SafetyNotReadyReason::NavigationEpochUnavailable,
            SafetyNotReadyReason::NavigationEpochTransition,
            SafetyNotReadyReason::MotionStateUnavailable,
        ];
        let fixture = runtime_fixture();
        let mut supervisor = supervisor_with_capacity(&fixture, 2);
        for (index, reason) in reasons.into_iter().enumerate() {
            let decision = supervisor
                .decide(
                    HostMonotonicTimestamp::from_nanos(index as u64 + 1),
                    SafetyTickInput::NotReady(reason),
                    &mut PanicClock,
                )
                .expect("bounded STOP record");
            assert_stop_record(&supervisor, &decision);
            assert_eq!(decision.request_id(), None);
            assert!(matches!(
                stopped(&decision).cause(),
                SafetyStopCause::NotReady(actual) if *actual == reason
            ));
            assert_eq!(decision.record().decision_id().as_u64(), index as u64 + 1);
            assert_eq!(
                supervisor.shadow_session().retained_len(),
                (index + 1).min(2)
            );
        }
        assert_eq!(
            supervisor
                .shadow_session()
                .retained()
                .map(|record| record.decision_id().as_u64())
                .collect::<Vec<_>>(),
            vec![14, 15]
        );
    }

    #[test]
    fn successful_nonzero_request_then_not_ready_records_one_stop_and_clears_diagnostic() {
        let fixture = runtime_fixture();
        let reference = reference_at(&fixture, 12);
        let mut supervisor = supervisor(&fixture);
        let success = supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(20),
                SafetyTickInput::Ready(ready(
                    &fixture,
                    &reference,
                    state_at(fixture.epoch, 11),
                    20,
                    1_000,
                )),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(30)),
            )
            .expect("controller decision");
        let controller = match success.outcome() {
            SafetyDecisionOutcome::Controller(controller) => *controller,
            SafetyDecisionOutcome::Stopped(stopped) => {
                panic!("unexpected stop: {}", stopped.cause())
            }
        };
        assert!(!controller.requested_pwm().is_stop());
        assert_eq!(controller.request_id(), NonZeroU64::MIN);
        assert_eq!(
            success.record().disposition(),
            ShadowCommandDisposition::ControllerRequest
        );
        let diagnostic = supervisor
            .last_success_trajectory()
            .expect("bound trajectory");
        assert_eq!(diagnostic.decision_id(), success.record().decision_id());
        assert_eq!(diagnostic.points().len(), 1);

        let stop = supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(31),
                SafetyTickInput::NotReady(SafetyNotReadyReason::NavigationGoalUnavailable),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(31)),
            )
            .expect("STOP record");
        assert_stop_record(&supervisor, &stop);
        assert_eq!(stopped(&stop).request_id(), None);
        assert!(matches!(
            stopped(&stop).cause(),
            SafetyStopCause::NotReady(SafetyNotReadyReason::NavigationGoalUnavailable)
        ));
        assert!(supervisor.last_success_trajectory().is_none());
        assert_eq!(supervisor.shadow_session().retained_len(), 2);

        let solver_stop = supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(40),
                SafetyTickInput::Ready(ready(
                    &fixture,
                    &reference,
                    state_at(fixture.epoch, 11),
                    40,
                    10,
                )),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(50)),
            )
            .expect("deadline STOP");
        let SafetyStopCause::Solver(source) = stopped(&solver_stop).cause() else {
            panic!("expected owned solver failure")
        };
        assert!(source.request().previous_pwm().is_stop());
        assert!(matches!(
            source.kind(),
            MpcFailureKind::Clock(HostMonotonicClockFailure::Fault(
                ClockFault::DeadlineReached { .. }
            ))
        ));
        assert_eq!(supervisor.shadow_session().retained_len(), 3);
    }

    #[test]
    fn collision_session_segment_and_staleness_fail_closed_with_advancing_ids() {
        let fixture = runtime_fixture();
        let reference = reference_at(&fixture, 12);
        let mut supervisor = supervisor(&fixture);
        let other_session = DeviceSessionId::try_new(2).expect("other session");
        let session_map = local_costmap_fixture(
            other_session,
            fixture.epoch.odom_segment_id(),
            10,
            1_000_000,
            BaseToOdom::try_new(0.0, 0.0, 0.0).expect("pose"),
        );
        let session_stop = supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(20),
                SafetyTickInput::Ready(SafetyReadyTick::new(
                    fixture.epoch,
                    state_at(fixture.epoch, 11),
                    &reference,
                    session_map
                        .view_at(HostMonotonicTimestamp::from_nanos(20))
                        .expect("view"),
                    SolverBudgetNs::try_new(100).expect("budget"),
                )),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(20)),
            )
            .expect("session STOP");
        assert!(matches!(
            stopped(&session_stop).cause(),
            SafetyStopCause::CollisionProvenance(
                CollisionProvenanceError::DeviceSessionMismatch { .. }
            )
        ));
        assert_eq!(
            stopped(&session_stop).request_id().map(NonZeroU64::get),
            Some(1)
        );

        let other_segment = OdomSegmentId::try_new(2).expect("other segment");
        let segment_map = local_costmap_fixture(
            fixture.epoch.device_session_id(),
            other_segment,
            10,
            1_000_000,
            BaseToOdom::try_new(0.0, 0.0, 0.0).expect("pose"),
        );
        let segment_stop = supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(21),
                SafetyTickInput::Ready(SafetyReadyTick::new(
                    fixture.epoch,
                    state_at(fixture.epoch, 11),
                    &reference,
                    segment_map
                        .view_at(HostMonotonicTimestamp::from_nanos(21))
                        .expect("view"),
                    SolverBudgetNs::try_new(100).expect("budget"),
                )),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(21)),
            )
            .expect("segment STOP");
        assert!(matches!(
            stopped(&segment_stop).cause(),
            SafetyStopCause::CollisionProvenance(
                CollisionProvenanceError::OdomSegmentMismatch { .. }
            )
        ));
        assert_eq!(
            stopped(&segment_stop).request_id().map(NonZeroU64::get),
            Some(2)
        );

        let stale_at = 1_000_011;
        let stale_stop = supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(stale_at),
                SafetyTickInput::Ready(ready(
                    &fixture,
                    &reference,
                    state_at(fixture.epoch, 11),
                    stale_at,
                    100,
                )),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(stale_at)),
            )
            .expect("stale STOP");
        assert!(matches!(
            stopped(&stale_stop).cause(),
            SafetyStopCause::CollisionProvenance(CollisionProvenanceError::ViewNotCurrent(_))
        ));
        assert_eq!(
            stopped(&stale_stop).request_id().map(NonZeroU64::get),
            Some(3)
        );
        assert_eq!(supervisor.shadow_session().retained_len(), 3);
        assert_stop_record(&supervisor, &stale_stop);
    }

    #[test]
    fn request_epoch_state_reference_time_and_expired_deadline_fail_closed() {
        let fixture = runtime_fixture();
        let reference = reference_at(&fixture, 12);

        let other_epoch = NavigationEpochV1::from_runtime(
            DeviceSessionId::try_new(2).expect("other session"),
            fixture.epoch.odom_segment_id(),
            fixture.epoch.map_snapshot(),
            &fixture.path,
        )
        .expect("other epoch");
        let mut epoch_supervisor = supervisor(&fixture);
        let epoch_stop = epoch_supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(20),
                SafetyTickInput::Ready(ready(
                    &fixture,
                    &reference,
                    state_at(other_epoch, 11),
                    20,
                    100,
                )),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(20)),
            )
            .expect("epoch STOP");
        assert!(matches!(
            stopped(&epoch_stop).cause(),
            SafetyStopCause::Request(MpcRequestParseError::ReferenceEpochMismatch { .. })
        ));
        assert_eq!(epoch_supervisor.shadow_session().retained_len(), 1);

        let mut state_supervisor = supervisor(&fixture);
        let state_stop = state_supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(20),
                SafetyTickInput::Ready(ready(
                    &fixture,
                    &reference,
                    state_at(fixture.epoch, 21),
                    20,
                    100,
                )),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(20)),
            )
            .expect("state-time STOP");
        assert!(matches!(
            stopped(&state_stop).cause(),
            SafetyStopCause::Request(MpcRequestParseError::StateAfterSubmission { .. })
        ));

        let late_reference = reference_at(&fixture, 21);
        let mut reference_supervisor = supervisor(&fixture);
        let reference_stop = reference_supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(20),
                SafetyTickInput::Ready(ready(
                    &fixture,
                    &late_reference,
                    state_at(fixture.epoch, 11),
                    20,
                    100,
                )),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(20)),
            )
            .expect("reference-time STOP");
        assert!(matches!(
            stopped(&reference_stop).cause(),
            SafetyStopCause::Request(MpcRequestParseError::ReferenceAfterSubmission { .. })
        ));

        let short_map = local_costmap_fixture(
            fixture.epoch.device_session_id(),
            fixture.epoch.odom_segment_id(),
            10,
            10,
            BaseToOdom::try_new(0.0, 0.0, 0.0).expect("pose"),
        );
        let mut deadline_supervisor = supervisor(&fixture);
        let deadline_stop = deadline_supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(20),
                SafetyTickInput::Ready(SafetyReadyTick::new(
                    fixture.epoch,
                    state_at(fixture.epoch, 11),
                    &reference,
                    short_map
                        .view_at(HostMonotonicTimestamp::from_nanos(20))
                        .expect("current through boundary"),
                    SolverBudgetNs::try_new(100).expect("budget"),
                )),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(20)),
            )
            .expect("deadline STOP");
        assert!(matches!(
            stopped(&deadline_stop).cause(),
            SafetyStopCause::Request(MpcRequestParseError::NonFutureDeadline { .. })
        ));
    }

    #[test]
    fn deadline_overflow_and_request_id_exhaustion_are_distinct_and_never_wrap() {
        let fixture = runtime_fixture();
        let reference = reference_at(&fixture, 12);
        let maximum_age_map = local_costmap_fixture(
            fixture.epoch.device_session_id(),
            fixture.epoch.odom_segment_id(),
            0,
            u64::MAX,
            BaseToOdom::try_new(0.0, 0.0, 0.0).expect("pose"),
        );
        let tick = u64::MAX - 1;
        let mut overflow_supervisor = supervisor(&fixture);
        let overflow = overflow_supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(tick),
                SafetyTickInput::Ready(SafetyReadyTick::new(
                    fixture.epoch,
                    state_at(fixture.epoch, 11),
                    &reference,
                    maximum_age_map
                        .view_at(HostMonotonicTimestamp::from_nanos(tick))
                        .expect("maximum-age view"),
                    SolverBudgetNs::try_new(10).expect("budget"),
                )),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(tick)),
            )
            .expect("overflow STOP");
        assert!(matches!(
            stopped(&overflow).cause(),
            SafetyStopCause::DeadlineOverflow { .. }
        ));

        let other_session = DeviceSessionId::try_new(2).expect("other session");
        let mismatched_map = local_costmap_fixture(
            other_session,
            fixture.epoch.odom_segment_id(),
            10,
            1_000_000,
            BaseToOdom::try_new(0.0, 0.0, 0.0).expect("pose"),
        );
        let mut exhausted_supervisor = supervisor(&fixture);
        exhausted_supervisor.set_next_ready_request_id_for_test(Some(NonZeroU64::MAX));
        let maximum = exhausted_supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(20),
                SafetyTickInput::Ready(SafetyReadyTick::new(
                    fixture.epoch,
                    state_at(fixture.epoch, 11),
                    &reference,
                    mismatched_map
                        .view_at(HostMonotonicTimestamp::from_nanos(20))
                        .expect("view"),
                    SolverBudgetNs::try_new(100).expect("budget"),
                )),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(20)),
            )
            .expect("maximum-id STOP");
        assert_eq!(stopped(&maximum).request_id(), Some(NonZeroU64::MAX));
        let exhausted = exhausted_supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(21),
                SafetyTickInput::Ready(ready(
                    &fixture,
                    &reference,
                    state_at(fixture.epoch, 11),
                    21,
                    100,
                )),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(21)),
            )
            .expect("exhausted-id STOP");
        assert_eq!(stopped(&exhausted).request_id(), None);
        assert!(matches!(
            stopped(&exhausted).cause(),
            SafetyStopCause::RequestIdExhausted
        ));
        assert_eq!(exhausted_supervisor.shadow_session().retained_len(), 2);
    }

    #[test]
    fn noninvertible_capture_transform_is_an_adapter_stop_not_a_fallback() {
        let fixture = runtime_fixture();
        let reference = reference_at(&fixture, 12);
        let extreme_map = local_costmap_fixture(
            fixture.epoch.device_session_id(),
            fixture.epoch.odom_segment_id(),
            10,
            1_000_000,
            BaseToOdom::try_new(f64::MAX, f64::MAX, FRAC_PI_4).expect("finite extreme pose"),
        );
        let mut supervisor = supervisor(&fixture);
        let decision = supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(20),
                SafetyTickInput::Ready(SafetyReadyTick::new(
                    fixture.epoch,
                    state_at(fixture.epoch, 11),
                    &reference,
                    extreme_map
                        .view_at(HostMonotonicTimestamp::from_nanos(20))
                        .expect("view"),
                    SolverBudgetNs::try_new(100).expect("budget"),
                )),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(20)),
            )
            .expect("adapter STOP");
        assert!(matches!(
            stopped(&decision).cause(),
            SafetyStopCause::CollisionAdapter(LocalCostmapCapsuleAdapterError::Transform(_))
        ));
        assert_stop_record(&supervisor, &decision);
        assert_eq!(supervisor.shadow_session().retained_len(), 1);
    }

    #[test]
    fn solver_deadline_regression_and_query_plus_clock_causes_remain_typed_and_owned() {
        let fixture = runtime_fixture();
        let reference = reference_at(&fixture, 12);

        let run = |clock: &mut ScriptedClock, mut query: ScriptedQuery| {
            let mut supervisor = supervisor(&fixture);
            let request_id = supervisor.take_ready_request_id().expect("request ID");
            let request = request(
                &fixture,
                &reference,
                request_id,
                20,
                100,
                ShadowPwmPair::STOP,
            );
            let decision = supervisor
                .solve_and_record(
                    HostMonotonicTimestamp::from_nanos(20),
                    request_id,
                    request,
                    fixture.collision,
                    &mut query,
                    clock,
                )
                .expect("typed STOP");
            assert_stop_record(&supervisor, &decision);
            let SafetyStopCause::Solver(source) = stopped(&decision).cause() else {
                panic!("expected solver stop")
            };
            (own_mpc_failure_kind(source.kind()), source.progress())
        };

        let (regression, regression_progress) = run(
            &mut ScriptedClock::new(&[30, 29]),
            ScriptedQuery::clear(fixture.collision),
        );
        assert!(matches!(
            regression,
            MpcFailureKind::Clock(HostMonotonicClockFailure::Fault(
                ClockFault::Regression { .. }
            ))
        ));
        let MpcSolveProgressV1::InProgress(regression_status) = regression_progress else {
            panic!("the first clock read succeeded")
        };
        assert_eq!(regression_status.observed_at().as_nanos(), 29);

        let mut query = ScriptedQuery::clear(fixture.collision);
        query.error_on = Some(1);
        let (combined, combined_progress) = run(&mut ScriptedClock::new(&[30, 30, 100]), query);
        assert!(matches!(
            combined,
            MpcFailureKind::CollisionObservation {
                source: CollisionObservationFailure::QueryAndClock {
                    query: LocalCostmapCapsuleQueryError::NumericalBounds,
                    clock: HostMonotonicClockFailure::Fault(ClockFault::DeadlineReached { .. }),
                },
                ..
            }
        ));
        let MpcSolveProgressV1::InProgress(combined_status) = combined_progress else {
            panic!("the first clock read succeeded")
        };
        assert_eq!(combined_status.observed_at().as_nanos(), 100);

        let (deadline, _) = run(
            &mut ScriptedClock::new(&[100]),
            ScriptedQuery::clear(fixture.collision),
        );
        assert!(matches!(
            deadline,
            MpcFailureKind::Clock(HostMonotonicClockFailure::Fault(
                ClockFault::DeadlineReached { .. }
            ))
        ));
    }

    #[test]
    fn failed_final_revalidation_records_stop_and_clears_prior_trajectory() {
        let fixture = runtime_fixture();
        let reference = reference_at(&fixture, 12);
        let mut supervisor = supervisor(&fixture);
        let first_id = supervisor.take_ready_request_id().expect("first ID");
        let first_request = request(
            &fixture,
            &reference,
            first_id,
            20,
            1_000,
            ShadowPwmPair::STOP,
        );
        let first = supervisor
            .solve_and_record(
                HostMonotonicTimestamp::from_nanos(20),
                first_id,
                first_request,
                fixture.collision,
                &mut ScriptedQuery::clear(fixture.collision),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(30)),
            )
            .expect("initial solution");
        let pre_final_queries = match first.outcome() {
            SafetyDecisionOutcome::Controller(controller) => {
                controller.solve_status().pre_final_collision_queries()
            }
            SafetyDecisionOutcome::Stopped(stopped) => {
                panic!("unexpected stop: {}", stopped.cause())
            }
        };
        assert!(supervisor.last_success_trajectory().is_some());

        supervisor.clear_diagnostic();
        let second_id = supervisor.take_ready_request_id().expect("second ID");
        let second_request = request(
            &fixture,
            &reference,
            second_id,
            40,
            1_000,
            first.record().pwm(),
        );
        let mut blocking_query = ScriptedQuery::clear(fixture.collision);
        blocking_query.block_on = Some(pre_final_queries as usize + 1);
        let blocked = supervisor
            .solve_and_record(
                HostMonotonicTimestamp::from_nanos(40),
                second_id,
                second_request,
                fixture.collision,
                &mut blocking_query,
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(50)),
            )
            .expect("final-revalidation STOP");
        assert!(matches!(
            stopped(&blocked).cause(),
            SafetyStopCause::Solver(source)
                if matches!(source.kind(), MpcFailureKind::FinalTrajectoryBlocked { .. })
        ));
        assert!(supervisor.last_success_trajectory().is_none());
        assert_stop_record(&supervisor, &blocked);
        assert_eq!(supervisor.shadow_session().retained_len(), 2);
    }

    #[test]
    fn session_recording_regression_is_fatal_and_does_not_claim_a_second_stop() {
        let fixture = runtime_fixture();
        let mut supervisor = supervisor(&fixture);
        supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(100),
                SafetyTickInput::NotReady(SafetyNotReadyReason::SafetyJournalLatched),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(100)),
            )
            .expect("first STOP");
        let error = supervisor
            .decide(
                HostMonotonicTimestamp::from_nanos(99),
                SafetyTickInput::NotReady(SafetyNotReadyReason::SafetyJournalLatched),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(99)),
            )
            .expect_err("record-time regression must be fatal");
        assert!(matches!(
            error.as_ref(),
            SafetyFatalError::StopRecording {
                source: ShadowCommandError::HostClockRegression { .. },
                ..
            }
        ));
        assert_eq!(error.request_id(), None);
        assert_eq!(supervisor.shadow_session().retained_len(), 1);
        assert!(supervisor.last_success_trajectory().is_none());
        assert_eq!(
            error.shadow_command_error(),
            *std::error::Error::source(error.as_ref())
                .expect("shadow source")
                .downcast_ref::<ShadowCommandError>()
                .expect("typed source")
        );
    }

    #[test]
    fn repeated_successes_reuse_solver_and_diagnostic_storage_and_bind_latest_id() {
        let fixture = runtime_fixture();
        let reference = reference_at(&fixture, 12);
        let mut supervisor = supervisor(&fixture);
        let initial_capacity = supervisor.last_success_trajectory.capacity();

        let first_id = supervisor.take_ready_request_id().expect("first ID");
        let first_request = request(
            &fixture,
            &reference,
            first_id,
            20,
            1_000,
            ShadowPwmPair::STOP,
        );
        let first = supervisor
            .solve_and_record(
                HostMonotonicTimestamp::from_nanos(20),
                first_id,
                first_request,
                fixture.collision,
                &mut ScriptedQuery::clear(fixture.collision),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(30)),
            )
            .expect("first success");
        let first_solver_address = supervisor
            .last_solver_trajectory_address
            .expect("solver address evidence");
        let first_diagnostic_address = supervisor.last_success_trajectory.as_ptr() as usize;

        supervisor.clear_diagnostic();
        let second_id = supervisor.take_ready_request_id().expect("second ID");
        let second_request = request(
            &fixture,
            &reference,
            second_id,
            40,
            1_000,
            first.record().pwm(),
        );
        let second = supervisor
            .solve_and_record(
                HostMonotonicTimestamp::from_nanos(40),
                second_id,
                second_request,
                fixture.collision,
                &mut ScriptedQuery::clear(fixture.collision),
                &mut ConstantClock(HostMonotonicTimestamp::from_nanos(50)),
            )
            .expect("second success");
        assert!(matches!(
            second.outcome(),
            SafetyDecisionOutcome::Controller(_)
        ));
        assert_eq!(
            supervisor.last_solver_trajectory_address,
            Some(first_solver_address)
        );
        assert_eq!(
            supervisor.last_success_trajectory.as_ptr() as usize,
            first_diagnostic_address
        );
        assert_eq!(
            supervisor.last_success_trajectory.capacity(),
            initial_capacity
        );
        assert_eq!(
            supervisor
                .last_success_trajectory()
                .expect("latest diagnostic")
                .decision_id(),
            second.record().decision_id()
        );
        assert_eq!(supervisor.shadow_session().retained_len(), 2);
        // Address/capacity equality proves reuse of these two buffers in this
        // test. It is not a process-wide allocation or performance benchmark.
    }
}
