//! One claimed-request dispatcher shared by the live agent's motion modes.
//!
//! The enum returned here is intentionally broader than the manual slice so
//! map-only, point-goal, exploration, persistence, and shutdown can be wired
//! under the same supervisor without replacing the socket or parser.

use std::fmt;

use kiko_supervisor_core::{AuthorityDuration, AuthorityLeaseId, SupervisorAction};
use robot_protocol::v2::HostCommandResult;

use super::agent_manual::{
    AgentAutonomousAuthority, AgentAutonomousControlError, AgentAutonomousMode,
    AgentAutonomousRequest, AgentAutonomousTick, PendingAgentAutonomousGrant,
    PendingAgentAutonomousStop,
};
#[cfg(feature = "operator-console")]
use super::control_socket::AgentControlTypedRequestKey;
use super::{
    AgentControlClaimedRequest, AgentControlClockError, AgentControlCommandV1,
    AgentControlDispatchResponseError, AgentControlMonotonicOrigin, AgentControlRejectionCodeV1,
    AgentControlRuntimeReceiveError, AgentControlRuntimeReceiver, AgentControlStatusV1,
    AgentManualControlCore, AgentManualControlError, AgentManualGlobalStopRequirement,
    AgentMapStateV1, BeginManualTransition, ManualDriveOutput,
};

/// Process-lifetime receiver plus the sole manual authority state.
pub struct AgentControlDispatcher {
    receiver: AgentControlRuntimeReceiver,
    clock: AgentControlMonotonicOrigin,
    manual: AgentManualControlCore,
    #[cfg(feature = "operator-console")]
    ingress_policy: AgentControlIngressPolicy,
    #[cfg(feature = "operator-console")]
    last_claimed_typed_request_key: Option<AgentControlTypedRequestKey>,
}

/// Which authority-bearing commands the legacy local socket may originate.
///
/// The authenticated operator/agent console uses the typed in-process lane.
/// In unified production mode, the legacy socket remains available for
/// observation and safety reduction but cannot bypass console arbitration to
/// acquire or renew motion authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(feature = "operator-console")]
enum AgentControlIngressPolicy {
    Unrestricted,
    UnifiedConsoleAuthority,
}

impl AgentControlDispatcher {
    pub const fn new(
        receiver: AgentControlRuntimeReceiver,
        clock: AgentControlMonotonicOrigin,
        manual: AgentManualControlCore,
    ) -> Self {
        Self {
            receiver,
            clock,
            manual,
            #[cfg(feature = "operator-console")]
            ingress_policy: AgentControlIngressPolicy::Unrestricted,
            #[cfg(feature = "operator-console")]
            last_claimed_typed_request_key: None,
        }
    }

    /// Build the production dispatcher in which the legacy socket can observe
    /// and reduce safety state but only the authenticated typed console lane
    /// may acquire or renew motion authority.
    #[cfg(feature = "operator-console")]
    pub const fn new_with_unified_console_authority(
        receiver: AgentControlRuntimeReceiver,
        clock: AgentControlMonotonicOrigin,
        manual: AgentManualControlCore,
    ) -> Self {
        Self {
            receiver,
            clock,
            manual,
            ingress_policy: AgentControlIngressPolicy::UnifiedConsoleAuthority,
            last_claimed_typed_request_key: None,
        }
    }

    pub const fn manual(&self) -> &AgentManualControlCore {
        &self.manual
    }

    pub const fn manual_mut(&mut self) -> &mut AgentManualControlCore {
        &mut self.manual
    }

    #[cfg(feature = "operator-console")]
    pub(crate) fn take_last_claimed_typed_request_key(
        &mut self,
    ) -> Option<AgentControlTypedRequestKey> {
        self.last_claimed_typed_request_key.take()
    }

    pub(crate) fn request_autonomous(
        &mut self,
        mode: AgentAutonomousMode,
        duration: AuthorityDuration,
        now: crate::HostMonotonicTimestamp,
    ) -> Result<AgentAutonomousRequest, AgentAutonomousControlError> {
        self.manual.request_autonomous(mode, duration, now)
    }

    pub(crate) fn complete_autonomous_grant_with_applied_zero(
        &mut self,
        pending: &PendingAgentAutonomousGrant,
        result: HostCommandResult,
        observed_at: crate::HostMonotonicTimestamp,
        now: crate::HostMonotonicTimestamp,
    ) -> Result<AgentAutonomousAuthority, AgentAutonomousControlError> {
        self.manual
            .complete_autonomous_grant_with_applied_zero(pending, result, observed_at, now)
    }

    pub(crate) fn renew_autonomous(
        &mut self,
        active: &mut AgentAutonomousAuthority,
        duration: AuthorityDuration,
        now: crate::HostMonotonicTimestamp,
    ) -> Result<(), AgentAutonomousControlError> {
        self.manual.renew_autonomous(active, duration, now)
    }

    pub(crate) fn tick_autonomous(
        &mut self,
        active: &AgentAutonomousAuthority,
        now: crate::HostMonotonicTimestamp,
    ) -> Result<AgentAutonomousTick, AgentAutonomousControlError> {
        self.manual.tick_autonomous(active, now)
    }

    pub(crate) fn begin_autonomous_release(
        &mut self,
        active: &AgentAutonomousAuthority,
        now: crate::HostMonotonicTimestamp,
    ) -> Result<PendingAgentAutonomousStop, AgentAutonomousControlError> {
        self.manual.begin_autonomous_release(active, now)
    }

    pub(crate) fn cancel_pending_autonomous_grant(
        &mut self,
        pending: &PendingAgentAutonomousGrant,
        now: crate::HostMonotonicTimestamp,
    ) -> Result<PendingAgentAutonomousStop, AgentAutonomousControlError> {
        self.manual.cancel_pending_autonomous_grant(pending, now)
    }

    pub(crate) fn complete_autonomous_stop_with_applied_zero(
        &mut self,
        pending: &PendingAgentAutonomousStop,
        result: HostCommandResult,
        observed_at: crate::HostMonotonicTimestamp,
        now: crate::HostMonotonicTimestamp,
    ) -> Result<(), AgentAutonomousControlError> {
        self.manual
            .complete_autonomous_stop_with_applied_zero(pending, result, observed_at, now)
    }

    /// Return the sole owner's conservative lifecycle/status projection.
    pub fn control_status(&self, map: AgentMapStateV1) -> AgentControlStatusV1 {
        self.manual.control_status(map)
    }

    /// Begin explicit arming through the sole manual/lifecycle owner.
    pub fn begin_arm(
        &mut self,
        now: crate::HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentManualControlError> {
        self.manual.begin_arm(now)
    }

    /// Complete the arming zero gate with exact applied controller evidence.
    pub fn complete_arm_with_applied_zero(
        &mut self,
        result: HostCommandResult,
        observed_at: crate::HostMonotonicTimestamp,
        now: crate::HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentManualControlError> {
        self.manual
            .complete_arm_with_applied_zero(result, observed_at, now)
    }

    /// Begin explicit disarming through the sole manual/lifecycle owner.
    pub fn begin_disarm(
        &mut self,
        now: crate::HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentManualControlError> {
        self.manual.begin_disarm(now)
    }

    /// Complete the disarming zero gate with exact applied controller evidence.
    pub fn complete_disarm_with_applied_zero(
        &mut self,
        result: HostCommandResult,
        observed_at: crate::HostMonotonicTimestamp,
        now: crate::HostMonotonicTimestamp,
    ) -> Result<SupervisorAction, AgentManualControlError> {
        self.manual
            .complete_disarm_with_applied_zero(result, observed_at, now)
    }

    /// Claim and route at most one parsed request. No transition occurs before
    /// the claim rendezvous succeeds. The observation timestamp is sampled
    /// from the socket's shared host-clock origin only after that successful
    /// claim, so a concurrently enqueued request cannot appear to arrive in
    /// the future relative to its own admission decision.
    pub fn try_dispatch_one(
        &mut self,
        map: AgentMapStateV1,
    ) -> Result<AgentDispatchOutcome, AgentControlDispatcherError> {
        let claimed = match self.try_claim_one()? {
            ClaimedDispatch::None(outcome) => return Ok(outcome),
            ClaimedDispatch::Claimed(claimed) => claimed,
        };
        let observed_at = match self.clock.try_now() {
            Ok(observed_at) => observed_at,
            Err(clock) => {
                return match claimed.reject(AgentControlRejectionCodeV1::InternalFault, false) {
                    Ok(()) => Err(AgentControlDispatcherError::Clock(clock)),
                    Err(response) => {
                        Err(AgentControlDispatcherError::ClockAndResponse { clock, response })
                    }
                };
            }
        };
        self.dispatch_claimed(map, claimed, observed_at)
    }

    /// Claim and route one request using an observation sampled by the sole
    /// outer live owner.
    ///
    /// The caller must sample `observed_at` from the same monotonic epoch as
    /// the socket receipt timestamp, immediately before this call. This
    /// variant exists so lifecycle, coordinator, MPC, and dispatcher decisions
    /// can use one injected clock in deterministic production-owner tests.
    pub(crate) fn try_dispatch_one_at(
        &mut self,
        map: AgentMapStateV1,
        observed_at: crate::HostMonotonicTimestamp,
    ) -> Result<AgentDispatchOutcome, AgentControlDispatcherError> {
        let claimed = match self.try_claim_one()? {
            ClaimedDispatch::None(outcome) => return Ok(outcome),
            ClaimedDispatch::Claimed(claimed) => claimed,
        };
        self.dispatch_claimed(map, claimed, observed_at)
    }

    fn try_claim_one(&mut self) -> Result<ClaimedDispatch, AgentControlDispatcherError> {
        #[cfg(feature = "operator-console")]
        {
            self.last_claimed_typed_request_key = None;
        }
        let dispatch = match self.receiver.try_recv() {
            Ok(dispatch) => dispatch,
            Err(AgentControlRuntimeReceiveError::Empty) => {
                return Ok(ClaimedDispatch::None(AgentDispatchOutcome::Idle));
            }
            Err(source) => return Err(AgentControlDispatcherError::Receive(source)),
        };
        #[cfg(feature = "operator-console")]
        let socket_command_rejected = self.ingress_policy
            == AgentControlIngressPolicy::UnifiedConsoleAuthority
            && dispatch.typed_request_key().is_none()
            && !socket_command_is_observational_or_safety_reducing(dispatch.request().command());
        #[cfg(feature = "operator-console")]
        let typed_request_key = dispatch.typed_request_key();
        match dispatch.claim() {
            #[cfg(feature = "operator-console")]
            Ok(claimed) if socket_command_rejected => {
                self.last_claimed_typed_request_key = typed_request_key;
                match claimed.reject(AgentControlRejectionCodeV1::AuthorityDenied, false) {
                    Ok(()) => Ok(ClaimedDispatch::None(
                        AgentDispatchOutcome::RejectedManual {
                            code: AgentControlRejectionCodeV1::AuthorityDenied,
                            retryable: false,
                        },
                    )),
                    Err(source) => Err(AgentControlDispatcherError::Response(source)),
                }
            }
            Ok(claimed) => {
                #[cfg(feature = "operator-console")]
                {
                    self.last_claimed_typed_request_key = typed_request_key;
                }
                Ok(ClaimedDispatch::Claimed(claimed))
            }
            Err(AgentControlDispatchResponseError::ClientUnavailable) => {
                #[cfg(feature = "operator-console")]
                {
                    self.last_claimed_typed_request_key = None;
                }
                Ok(ClaimedDispatch::None(
                    AgentDispatchOutcome::ClientUnavailableBeforeClaim,
                ))
            }
            Err(source) => Err(AgentControlDispatcherError::Response(source)),
        }
    }

    fn dispatch_claimed(
        &mut self,
        map: AgentMapStateV1,
        claimed: AgentControlClaimedRequest,
        observed_at: crate::HostMonotonicTimestamp,
    ) -> Result<AgentDispatchOutcome, AgentControlDispatcherError> {
        let request = claimed.request();
        match request.command() {
            AgentControlCommandV1::QueryStatus => {
                let status = self.control_status(map);
                claimed
                    .respond_status(status)
                    .map_err(AgentControlDispatcherError::Response)?;
                Ok(AgentDispatchOutcome::RepliedStatus { status })
            }
            AgentControlCommandV1::Arm => Ok(AgentDispatchOutcome::ArmRequested { claimed }),
            AgentControlCommandV1::Disarm => Ok(AgentDispatchOutcome::DisarmRequested {
                manual: self.manual.global_stop_requirement(),
                claimed,
            }),
            AgentControlCommandV1::BeginManual => match self.manual.begin_manual(observed_at) {
                Ok(transition) => Ok(AgentDispatchOutcome::BeginManual {
                    claimed,
                    transition,
                }),
                Err(source) => self.reject_expected_manual(claimed, source),
            },
            AgentControlCommandV1::ManualVelocity(command) => {
                match self
                    .manual
                    .ingest_velocity(command, claimed.received_at(), observed_at)
                {
                    Ok(output) => Ok(AgentDispatchOutcome::ManualCommand { claimed, output }),
                    Err(source) => self.reject_expected_manual(claimed, source),
                }
            }
            AgentControlCommandV1::ManualStop(command) => {
                match self
                    .manual
                    .ingest_stop(command, claimed.received_at(), observed_at)
                {
                    Ok(output) => Ok(AgentDispatchOutcome::ManualCommand { claimed, output }),
                    Err(source) => self.reject_expected_manual(claimed, source),
                }
            }
            AgentControlCommandV1::Stop => Ok(AgentDispatchOutcome::GlobalStopRequested {
                manual: self.manual.global_stop_requirement(),
                claimed,
            }),
            AgentControlCommandV1::Shutdown => Ok(AgentDispatchOutcome::Shutdown { claimed }),
            command @ (AgentControlCommandV1::MapOnly
            | AgentControlCommandV1::FrontierExplore
            | AgentControlCommandV1::SelectMapPoint(_)
            | AgentControlCommandV1::SaveMap) => {
                Ok(AgentDispatchOutcome::OtherMode { claimed, command })
            }
        }
    }

    fn reject_expected_manual(
        &self,
        claimed: AgentControlClaimedRequest,
        source: AgentManualControlError,
    ) -> Result<AgentDispatchOutcome, AgentControlDispatcherError> {
        let (code, retryable) = match &source {
            AgentManualControlError::ManualDisabled => {
                (AgentControlRejectionCodeV1::AuthorityDenied, false)
            }
            AgentManualControlError::ManualNotActive
            | AgentManualControlError::NoPendingBegin
            | AgentManualControlError::NoPendingCancelledBegin
            | AgentManualControlError::NoPendingRelease => {
                (AgentControlRejectionCodeV1::AuthorityDenied, true)
            }
            AgentManualControlError::ModeConflict => {
                (AgentControlRejectionCodeV1::ModeConflict, false)
            }
            AgentManualControlError::AuthorityNotReady { .. }
            | AgentManualControlError::PendingAuthorityExpired => {
                (AgentControlRejectionCodeV1::NotReady, true)
            }
            _ => {
                return Ok(AgentDispatchOutcome::ManualFault { claimed, source });
            }
        };
        if let Err(response) = claimed.reject(code, retryable) {
            return Err(AgentControlDispatcherError::ManualAndResponse { source, response });
        }
        Ok(AgentDispatchOutcome::RejectedManual { code, retryable })
    }
}

#[cfg(feature = "operator-console")]
const fn socket_command_is_observational_or_safety_reducing(
    command: AgentControlCommandV1,
) -> bool {
    matches!(
        command,
        AgentControlCommandV1::QueryStatus
            | AgentControlCommandV1::Disarm
            | AgentControlCommandV1::Stop
            | AgentControlCommandV1::MapOnly
            | AgentControlCommandV1::ManualStop(_)
            | AgentControlCommandV1::Shutdown
    )
}

enum ClaimedDispatch {
    None(AgentDispatchOutcome),
    Claimed(AgentControlClaimedRequest),
}

#[derive(Debug)]
pub enum AgentDispatchOutcome {
    Idle,
    ClientUnavailableBeforeClaim,
    RepliedStatus {
        status: AgentControlStatusV1,
    },
    /// Claimed explicit arm request. Dispatch does not cross the supervisor's
    /// mandatory fresh-zero gate or claim that hardware is ready.
    ArmRequested {
        claimed: AgentControlClaimedRequest,
    },
    /// Claimed explicit disarm request. The runtime must satisfy every active
    /// mode's stop obligation and admit the resulting exact zero before it can
    /// report completion.
    DisarmRequested {
        claimed: AgentControlClaimedRequest,
        manual: AgentManualGlobalStopRequirement,
    },
    RejectedManual {
        code: AgentControlRejectionCodeV1,
        retryable: bool,
    },
    BeginManual {
        claimed: AgentControlClaimedRequest,
        transition: BeginManualTransition,
    },
    ManualCommand {
        claimed: AgentControlClaimedRequest,
        output: ManualDriveOutput<AuthorityLeaseId>,
    },
    /// Claimed process-wide stop intent. The runtime must execute every mode's
    /// stop obligation and obtain the required evidence before responding
    /// completed; dispatch itself has not stopped hardware.
    GlobalStopRequested {
        claimed: AgentControlClaimedRequest,
        manual: AgentManualGlobalStopRequirement,
    },
    OtherMode {
        claimed: AgentControlClaimedRequest,
        command: AgentControlCommandV1,
    },
    Shutdown {
        claimed: AgentControlClaimedRequest,
    },
    ManualFault {
        claimed: AgentControlClaimedRequest,
        source: AgentManualControlError,
    },
}

#[derive(Debug)]
pub enum AgentControlDispatcherError {
    Receive(AgentControlRuntimeReceiveError),
    Clock(AgentControlClockError),
    Response(AgentControlDispatchResponseError),
    ClockAndResponse {
        clock: AgentControlClockError,
        response: AgentControlDispatchResponseError,
    },
    ManualAndResponse {
        source: AgentManualControlError,
        response: AgentControlDispatchResponseError,
    },
}

impl fmt::Display for AgentControlDispatcherError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "agent-control dispatch failed: {self:?}")
    }
}

impl std::error::Error for AgentControlDispatcherError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Receive(source) => Some(source),
            Self::Clock(source) => Some(source),
            Self::Response(source) => Some(source),
            Self::ClockAndResponse { clock, .. } => Some(clock),
            Self::ManualAndResponse { source, .. } => Some(source),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use kiko_supervisor_core::{
        AuthorityDuration, ReadinessBinding, ReadinessEpoch, Sha256Digest, StopReason,
        SupervisorAction, SupervisorConfig,
    };
    use robot_protocol::ControllerUptimeMsWrapping;
    use robot_protocol::v2::{
        ControlEpoch, ControllerBootId, ControllerDeadlineMsWrapping, ControllerFaults,
        ControllerUid, HostCommandResult, HostCommandResultCode, OutputState, RemainingLeaseMs,
        TimerPwm, V2CommandSequence,
    };
    use serde_json::{Value, json};

    use super::super::control_socket::{
        enqueue_agent_control_test_request,
        enqueue_agent_control_test_request_with_expired_response,
    };
    use super::*;
    use crate::navigation::{
        AGENT_CONTROL_SCHEMA_V1, AgentAuthoritySupervisor, AgentControlCompletionV1,
        AgentControlRequestParser, AgentControlResponseKindV1, AgentControlRuntimeQueueCapacity,
        AgentManualRuntimePolicy, MANUAL_DRIVE_CONFIG_V1, ManualDriveConfigV1,
        ManualDriveConfigV1Dto, NavigationClockEpoch, agent_control_runtime_queue,
    };

    const BASE_NS: u64 = 1_000_000_000;

    fn at(nanos: u64) -> crate::HostMonotonicTimestamp {
        crate::HostMonotonicTimestamp::from_nanos(nanos)
    }

    fn duration(nanos: u64) -> AuthorityDuration {
        AuthorityDuration::try_from_nanos(nanos).expect("nonzero fixture duration")
    }

    fn uid() -> ControllerUid {
        ControllerUid::try_new([1; 12]).expect("controller UID")
    }

    fn boot() -> ControllerBootId {
        ControllerBootId::try_new(7).expect("boot ID")
    }

    fn host_zero(sequence: u32) -> HostCommandResult {
        HostCommandResult {
            controller_uid: uid(),
            boot_id: boot(),
            control_epoch: ControlEpoch::try_new(9).expect("control epoch"),
            sequence: V2CommandSequence::new(sequence),
            result: HostCommandResultCode::AppliedNew,
            requested_timer_pwm: TimerPwm::ZERO,
            controller_timer_pwm: TimerPwm::ZERO,
            output_state: OutputState::ZeroPwm,
            controller_applied_at: ControllerUptimeMsWrapping::new(10),
            controller_expires_at: ControllerDeadlineMsWrapping::new(20),
            remaining_lease: RemainingLeaseMs::ZERO,
            faults: ControllerFaults::NONE,
        }
    }

    fn disarmed() -> AgentAuthoritySupervisor {
        let config = SupervisorConfig::new(duration(10_000_000_000), duration(5_000_000_000))
            .expect("supervisor config");
        let mut authority =
            AgentAuthoritySupervisor::new(config, NavigationClockEpoch::new(at(BASE_NS)));
        assert_eq!(
            authority.begin_inventory(at(BASE_NS + 1)).unwrap(),
            SupervisorAction::InventoryRequired
        );
        let readiness = ReadinessBinding::new(
            ReadinessEpoch::try_new(1).expect("readiness epoch"),
            uid(),
            boot(),
            ControlEpoch::try_new(9).expect("control epoch"),
            Sha256Digest::try_new([2; 32]).expect("hardware digest"),
            Sha256Digest::try_new([3; 32]).expect("calibration digest"),
        );
        assert_eq!(
            authority
                .admit_readiness(readiness, at(BASE_NS + 2))
                .unwrap(),
            SupervisorAction::Disarmed
        );
        authority
    }

    fn ready() -> AgentAuthoritySupervisor {
        let mut authority = disarmed();
        assert_eq!(
            authority.arm(at(BASE_NS + 3)).unwrap(),
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::Arming
            }
        );
        assert_eq!(
            authority
                .admit_applied_zero(host_zero(0), at(BASE_NS + 4), at(BASE_NS + 4))
                .unwrap(),
            SupervisorAction::ReadyStopped
        );
        authority
    }

    fn policy() -> AgentManualRuntimePolicy {
        AgentManualRuntimePolicy::for_test(
            duration(2_000_000_000),
            ManualDriveConfigV1::parse(ManualDriveConfigV1Dto {
                schema_version: MANUAL_DRIVE_CONFIG_V1,
                maximum_abs_forward_velocity_mps: 0.5,
                maximum_abs_yaw_rate_rad_s: 1.0,
                maximum_command_age_ns: 1_000_000_000,
                deadman_timeout_ns: 2_000_000_000,
            })
            .expect("manual policy"),
        )
    }

    fn request(
        parser: &mut AgentControlRequestParser,
        request_id: u64,
        command: Value,
    ) -> super::super::AgentControlRequestV1 {
        parser
            .parse_next(
                &serde_json::to_vec(&json!({
                    "schema_version": AGENT_CONTROL_SCHEMA_V1,
                    "request_id": request_id,
                    "command": command,
                }))
                .expect("request JSON"),
            )
            .expect("parsed request")
    }

    fn fixture() -> (
        super::super::AgentControlRuntimeSender,
        AgentControlDispatcher,
        AgentControlMonotonicOrigin,
    ) {
        let (sender, receiver) = agent_control_runtime_queue(
            AgentControlRuntimeQueueCapacity::try_new(8).expect("queue capacity"),
        );
        let clock = AgentControlMonotonicOrigin::new(Instant::now(), at(BASE_NS + 5));
        let dispatcher = AgentControlDispatcher::new(
            receiver,
            clock,
            AgentManualControlCore::new(ready(), Some(policy())),
        );
        (sender, dispatcher, clock)
    }

    fn disarmed_fixture() -> (
        super::super::AgentControlRuntimeSender,
        AgentControlDispatcher,
        AgentControlMonotonicOrigin,
    ) {
        let (sender, receiver) = agent_control_runtime_queue(
            AgentControlRuntimeQueueCapacity::try_new(8).expect("queue capacity"),
        );
        let clock = AgentControlMonotonicOrigin::new(Instant::now(), at(BASE_NS + 5));
        let dispatcher = AgentControlDispatcher::new(
            receiver,
            clock,
            AgentManualControlCore::new(disarmed(), Some(policy())),
        );
        (sender, dispatcher, clock)
    }

    #[cfg(feature = "operator-console")]
    #[test]
    fn unified_console_policy_denies_socket_authority_acquisition_but_keeps_stop() {
        let (sender, receiver) = agent_control_runtime_queue(
            AgentControlRuntimeQueueCapacity::try_new(8).expect("queue capacity"),
        );
        let clock = AgentControlMonotonicOrigin::new(Instant::now(), at(BASE_NS + 5));
        let mut dispatcher = AgentControlDispatcher::new_with_unified_console_authority(
            receiver,
            clock,
            AgentManualControlCore::new(ready(), Some(policy())),
        );
        let mut safety_parser = AgentControlRequestParser::new();
        assert!(socket_command_is_observational_or_safety_reducing(
            request(
                &mut safety_parser,
                99,
                json!({"kind": "manual_stop", "sequence": 1})
            )
            .command()
        ));
        let mut parser = AgentControlRequestParser::new();

        let arm_peer = enqueue_agent_control_test_request(
            &sender,
            request(&mut parser, 1, json!({"kind": "arm"})),
            clock.try_now().expect("arm receipt stamp"),
        );
        assert!(matches!(
            dispatcher
                .try_dispatch_one(AgentMapStateV1::UNAVAILABLE)
                .expect("policy rejection"),
            AgentDispatchOutcome::RejectedManual {
                code: AgentControlRejectionCodeV1::AuthorityDenied,
                retryable: false,
            }
        ));
        assert!(matches!(
            arm_peer.join().expect("arm peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Rejected {
                        code: AgentControlRejectionCodeV1::AuthorityDenied,
                        retryable: false,
                    }
                )
        ));

        let stop_peer = enqueue_agent_control_test_request(
            &sender,
            request(&mut parser, 2, json!({"kind": "stop"})),
            clock.try_now().expect("stop receipt stamp"),
        );
        let AgentDispatchOutcome::GlobalStopRequested { claimed, .. } = dispatcher
            .try_dispatch_one(AgentMapStateV1::UNAVAILABLE)
            .expect("safety-reducing socket command")
        else {
            panic!("stop remains admitted")
        };
        claimed.respond_completed().expect("stop response");
        assert!(matches!(
            stop_peer.join().expect("stop peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Accepted {
                        command: super::super::AgentControlCommandKindV1::Stop,
                        completion: AgentControlCompletionV1::Completed,
                    }
                )
        ));
    }

    #[test]
    fn arm_and_disarm_route_claims_then_cross_only_exact_lifecycle_zero_gates() {
        let (sender, mut dispatcher, clock) = disarmed_fixture();
        let mut parser = AgentControlRequestParser::new();

        let arm_peer = enqueue_agent_control_test_request(
            &sender,
            request(&mut parser, 1, json!({"kind": "arm"})),
            clock.try_now().expect("arm receipt stamp"),
        );
        let AgentDispatchOutcome::ArmRequested { claimed } = dispatcher
            .try_dispatch_one(AgentMapStateV1::UNAVAILABLE)
            .expect("dispatch arm")
        else {
            panic!("arm outcome")
        };
        assert_eq!(
            dispatcher.begin_arm(at(BASE_NS + 6)).expect("begin arm"),
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::Arming,
            }
        );
        assert_eq!(
            dispatcher
                .complete_arm_with_applied_zero(host_zero(0), at(BASE_NS + 7), at(BASE_NS + 7),)
                .expect("complete arm"),
            SupervisorAction::ReadyStopped
        );
        claimed.respond_completed().expect("arm response");
        assert!(matches!(
            arm_peer.join().expect("arm peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Accepted {
                        command: super::super::AgentControlCommandKindV1::Arm,
                        completion: AgentControlCompletionV1::Completed,
                    }
                )
        ));

        let disarm_peer = enqueue_agent_control_test_request(
            &sender,
            request(&mut parser, 2, json!({"kind": "disarm"})),
            clock.try_now().expect("disarm receipt stamp"),
        );
        let AgentDispatchOutcome::DisarmRequested { claimed, manual } = dispatcher
            .try_dispatch_one(AgentMapStateV1::UNAVAILABLE)
            .expect("dispatch disarm")
        else {
            panic!("disarm outcome")
        };
        assert_eq!(manual, AgentManualGlobalStopRequirement::NoManualTransition);
        assert_eq!(
            dispatcher
                .begin_disarm(at(BASE_NS + 8))
                .expect("begin disarm"),
            SupervisorAction::BaseZeroRequired {
                reason: StopReason::ExplicitDisarm,
            }
        );
        assert_eq!(
            dispatcher
                .complete_disarm_with_applied_zero(host_zero(1), at(BASE_NS + 9), at(BASE_NS + 9),)
                .expect("complete disarm"),
            SupervisorAction::Disarmed
        );
        claimed.respond_completed().expect("disarm response");
        assert!(matches!(
            disarm_peer.join().expect("disarm peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Accepted {
                        command: super::super::AgentControlCommandKindV1::Disarm,
                        completion: AgentControlCompletionV1::Completed,
                    }
                )
        ));
        assert_eq!(
            dispatcher
                .control_status(AgentMapStateV1::UNAVAILABLE)
                .runtime(),
            super::super::AgentRuntimeStateV1::Disarmed
        );
    }

    #[test]
    fn begin_velocity_manual_stop_and_global_stop_share_one_claimed_dispatch_stream() {
        let (sender, mut dispatcher, clock) = fixture();
        let mut parser = AgentControlRequestParser::new();

        let begin_received_at = clock.try_now().expect("begin stamp");
        let begin_peer = enqueue_agent_control_test_request(
            &sender,
            request(&mut parser, 1, json!({"kind": "begin_manual"})),
            begin_received_at,
        );
        let AgentDispatchOutcome::BeginManual {
            claimed,
            transition: BeginManualTransition::Granted { .. },
        } = dispatcher
            .try_dispatch_one(AgentMapStateV1::UNAVAILABLE)
            .expect("dispatch begin")
        else {
            panic!("begin outcome")
        };
        claimed.respond_completed().expect("begin response");
        assert!(matches!(
            begin_peer.join().expect("begin peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Accepted {
                        command: super::super::AgentControlCommandKindV1::BeginManual,
                        completion: AgentControlCompletionV1::Completed,
                    }
                )
        ));

        // This stamp is necessarily later than any time a caller could have
        // sampled before the enqueue. Admission still succeeds because the
        // dispatcher samples its observation only after the claim.
        let velocity_received_at = clock.try_now().expect("velocity stamp");
        let velocity_peer = enqueue_agent_control_test_request(
            &sender,
            request(
                &mut parser,
                2,
                json!({
                    "kind": "manual_velocity",
                    "sequence": 1,
                    "forward_velocity_mps": 0.2,
                    "yaw_rate_rad_s": 0.1
                }),
            ),
            velocity_received_at,
        );
        let AgentDispatchOutcome::ManualCommand { claimed, output } = dispatcher
            .try_dispatch_one(AgentMapStateV1::UNAVAILABLE)
            .expect("dispatch velocity")
        else {
            panic!("velocity outcome")
        };
        assert!(matches!(
            output,
            ManualDriveOutput::Accepted(target)
                if target.received_at() == velocity_received_at && !target.target().is_stop()
        ));
        claimed
            .reject(AgentControlRejectionCodeV1::SafetyStopped, true)
            .expect("dispose velocity token without claiming physical completion");
        assert!(velocity_peer.join().expect("velocity peer").is_some());

        let stop_peer = enqueue_agent_control_test_request(
            &sender,
            request(
                &mut parser,
                3,
                json!({"kind": "manual_stop", "sequence": 2}),
            ),
            clock.try_now().expect("manual stop stamp"),
        );
        let AgentDispatchOutcome::ManualCommand { claimed, output } = dispatcher
            .try_dispatch_one(AgentMapStateV1::UNAVAILABLE)
            .expect("dispatch manual stop")
        else {
            panic!("manual stop outcome")
        };
        assert!(matches!(
            output,
            ManualDriveOutput::Accepted(target)
                if target.into_explicit_stop().is_ok()
        ));
        claimed
            .reject(AgentControlRejectionCodeV1::SafetyStopped, true)
            .expect("dispose stop token without claiming physical completion");
        assert!(stop_peer.join().expect("manual stop peer").is_some());

        let global_peer = enqueue_agent_control_test_request(
            &sender,
            request(&mut parser, 4, json!({"kind": "stop"})),
            clock.try_now().expect("global stop stamp"),
        );
        let AgentDispatchOutcome::GlobalStopRequested { claimed, manual } = dispatcher
            .try_dispatch_one(AgentMapStateV1::UNAVAILABLE)
            .expect("dispatch global stop")
        else {
            panic!("global stop outcome")
        };
        assert_eq!(
            manual,
            AgentManualGlobalStopRequirement::ReleaseActive {
                lease_id: dispatcher.manual().active_lease().unwrap().id(),
            }
        );
        drop(claimed);
        assert_eq!(global_peer.join().expect("global stop peer"), None);
    }

    #[test]
    fn velocity_cannot_implicitly_begin_manual_authority() {
        let (sender, mut dispatcher, clock) = fixture();
        let mut parser = AgentControlRequestParser::new();
        let peer = enqueue_agent_control_test_request(
            &sender,
            request(
                &mut parser,
                1,
                json!({
                    "kind": "manual_velocity",
                    "sequence": 1,
                    "forward_velocity_mps": 0.2,
                    "yaw_rate_rad_s": 0.1
                }),
            ),
            clock.try_now().expect("velocity stamp"),
        );
        assert!(matches!(
            dispatcher
                .try_dispatch_one(AgentMapStateV1::UNAVAILABLE)
                .expect("dispatch rejection"),
            AgentDispatchOutcome::RejectedManual {
                code: AgentControlRejectionCodeV1::AuthorityDenied,
                retryable: true,
            }
        ));
        assert!(matches!(
            peer.join().expect("velocity peer"),
            Some(response)
                if matches!(
                    response.response(),
                    AgentControlResponseKindV1::Rejected {
                        code: AgentControlRejectionCodeV1::AuthorityDenied,
                        retryable: true,
                    }
                )
        ));
        assert_eq!(dispatcher.manual().active_lease(), None);
    }

    #[test]
    fn clock_and_rejection_delivery_failures_are_both_retained() {
        let (sender, receiver) = agent_control_runtime_queue(
            AgentControlRuntimeQueueCapacity::try_new(1).expect("queue capacity"),
        );
        let future_origin = Instant::now()
            .checked_add(Duration::from_secs(60))
            .expect("future clock origin");
        let clock = AgentControlMonotonicOrigin::new(future_origin, at(BASE_NS + 5));
        let mut dispatcher = AgentControlDispatcher::new(
            receiver,
            clock,
            AgentManualControlCore::new(ready(), Some(policy())),
        );
        let mut parser = AgentControlRequestParser::new();
        let peer = enqueue_agent_control_test_request_with_expired_response(
            &sender,
            request(&mut parser, 1, json!({"kind": "query_status"})),
            at(BASE_NS + 5),
        );

        assert!(matches!(
            dispatcher.try_dispatch_one(AgentMapStateV1::UNAVAILABLE),
            Err(AgentControlDispatcherError::ClockAndResponse {
                clock: AgentControlClockError::OriginInFuture { .. },
                response: AgentControlDispatchResponseError::ClientUnavailable,
            })
        ));
        peer.join().expect("expired response peer");
    }

    #[test]
    fn expected_manual_and_rejection_delivery_failures_are_both_retained() {
        let (sender, mut dispatcher, clock) = fixture();
        let mut parser = AgentControlRequestParser::new();
        let peer = enqueue_agent_control_test_request_with_expired_response(
            &sender,
            request(
                &mut parser,
                1,
                json!({
                    "kind": "manual_velocity",
                    "sequence": 1,
                    "forward_velocity_mps": 0.2,
                    "yaw_rate_rad_s": 0.1
                }),
            ),
            clock.try_now().expect("velocity stamp"),
        );

        assert!(matches!(
            dispatcher.try_dispatch_one(AgentMapStateV1::UNAVAILABLE),
            Err(AgentControlDispatcherError::ManualAndResponse {
                source: AgentManualControlError::ManualNotActive,
                response: AgentControlDispatchResponseError::ClientUnavailable,
            })
        ));
        peer.join().expect("expired response peer");
    }
}
