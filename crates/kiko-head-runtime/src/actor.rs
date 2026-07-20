use std::fmt;
use std::time::Duration;

use kiko_head_protocol::{
    FullTelemetry, HeadJoint, HeadPose, HeadPoseError, PositionAgreementError,
    PositionAgreementTicks, PositionTicks, PresentPosition, TelemetryParseError, TorqueSwitch,
    ValidatedPresentPosition, build_natural_hold_frames, build_position_read,
    build_torque_switch_write,
};
use tokio::runtime::{Handle, TryCurrentError};
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinError, JoinHandle};

use crate::config::HeadRuntimeConfig;
use crate::framing::{FrameReadError, read_response_frame};
use crate::transport::{
    AsyncByteTransport, MonotonicClock, MonotonicTime, SerialConfigurationEvidence,
    SerialOpenError, SerialTransport, TokioClock, TransportFailure,
};

const ACTOR_MAILBOX_CAPACITY: usize = 1;

/// Deliberate opt-in required before the actor can enable servo torque.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PhysicalTorqueEnableConsent(());

impl PhysicalTorqueEnableConsent {
    /// Acknowledge that natural hold energises physical servos. No calibrated
    /// motion command is exposed by this crate.
    pub const fn explicitly_granted() -> Self {
        Self(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeStage {
    ObserveFirst,
    ObserveSecond,
    WriteObservedGoal,
    WriteTorqueLimit,
    EnableTorque,
    VerifyFirstStoppedPosition,
    VerifySecondStoppedPosition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VerificationSample {
    First,
    Second,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArmingFreshnessCheck {
    BeforeConfigurationWrites,
    BeforeEnableWrite,
    AfterEnableWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WritePurpose {
    PositionReadRequest,
    TelemetryReadRequest,
    ObservedGoal,
    TorqueLimit,
    TorqueEnable,
    TorqueDisable,
}

/// Exact bounded retry history for one completed write.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteEvidence {
    attempts_used: u8,
    recovered_failures: Vec<TransportFailure>,
    completed_at: MonotonicTime,
}

impl WriteEvidence {
    pub const fn attempts_used(&self) -> u8 {
        self.attempts_used
    }

    pub fn recovered_failures(&self) -> impl Iterator<Item = &TransportFailure> {
        self.recovered_failures.iter()
    }

    pub const fn completed_at(&self) -> MonotonicTime {
        self.completed_at
    }
}

/// Typed response plus the request write and framing evidence that admitted it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseEvidence<T> {
    value: T,
    request_write: WriteEvidence,
    discarded_noise_bytes: u16,
    received_at: MonotonicTime,
}

impl<T> ResponseEvidence<T> {
    pub const fn value(&self) -> &T {
        &self.value
    }

    pub const fn request_write(&self) -> &WriteEvidence {
        &self.request_write
    }

    pub const fn discarded_noise_bytes(&self) -> u16 {
        self.discarded_noise_bytes
    }

    pub const fn received_at(&self) -> MonotonicTime {
        self.received_at
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PositionObservationEvidence {
    joint: HeadJoint,
    first: ResponseEvidence<PresentPosition>,
    second: ResponseEvidence<PresentPosition>,
    validated: ValidatedPresentPosition,
}

impl PositionObservationEvidence {
    pub const fn joint(&self) -> HeadJoint {
        self.joint
    }

    pub const fn first(&self) -> &ResponseEvidence<PresentPosition> {
        &self.first
    }

    pub const fn second(&self) -> &ResponseEvidence<PresentPosition> {
        &self.second
    }

    pub const fn validated(&self) -> ValidatedPresentPosition {
        self.validated
    }
}

/// Post-write full telemetry whose position agrees with the observed target.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadbackEvidence {
    joint: HeadJoint,
    target: PositionTicks,
    first_target_difference_ticks: u16,
    second_target_difference_ticks: u16,
    stable_difference_ticks: u16,
    first: ResponseEvidence<FullTelemetry>,
    second: ResponseEvidence<FullTelemetry>,
}

impl ReadbackEvidence {
    pub const fn joint(&self) -> HeadJoint {
        self.joint
    }

    pub const fn target(&self) -> PositionTicks {
        self.target
    }

    pub const fn first_target_difference_ticks(&self) -> u16 {
        self.first_target_difference_ticks
    }

    pub const fn second_target_difference_ticks(&self) -> u16 {
        self.second_target_difference_ticks
    }

    pub const fn stable_difference_ticks(&self) -> u16 {
        self.stable_difference_ticks
    }

    pub const fn first(&self) -> &ResponseEvidence<FullTelemetry> {
        &self.first
    }

    pub const fn second(&self) -> &ResponseEvidence<FullTelemetry> {
        &self.second
    }
}

/// Success is emitted only after two stopped post-write positions per joint
/// parse exactly, agree with each other, and agree with the observed targets.
/// Servo response level zero means the
/// individual writes remain write-completion evidence, not register readback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedNaturalHoldEvidence {
    started_at: MonotonicTime,
    completed_at: MonotonicTime,
    observed_pose: HeadPose,
    observations: [PositionObservationEvidence; 4],
    observed_goal_writes: [WriteEvidence; 4],
    torque_limit_writes: [WriteEvidence; 4],
    torque_enable_writes: [WriteEvidence; 4],
    readbacks: [ReadbackEvidence; 4],
}

impl VerifiedNaturalHoldEvidence {
    pub const fn started_at(&self) -> MonotonicTime {
        self.started_at
    }

    pub const fn completed_at(&self) -> MonotonicTime {
        self.completed_at
    }

    pub const fn observed_pose(&self) -> HeadPose {
        self.observed_pose
    }

    pub const fn observations(&self) -> &[PositionObservationEvidence; 4] {
        &self.observations
    }

    pub const fn observed_goal_writes(&self) -> &[WriteEvidence; 4] {
        &self.observed_goal_writes
    }

    pub const fn torque_limit_writes(&self) -> &[WriteEvidence; 4] {
        &self.torque_limit_writes
    }

    pub const fn torque_enable_writes(&self) -> &[WriteEvidence; 4] {
        &self.torque_enable_writes
    }

    pub const fn readbacks(&self) -> &[ReadbackEvidence; 4] {
        &self.readbacks
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameWriteError {
    pub joint: HeadJoint,
    pub purpose: WritePurpose,
    pub attempts_used: u8,
    pub recovered_failures: Vec<TransportFailure>,
    pub source: TransportFailure,
}

impl fmt::Display for FrameWriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} write for {:?} failed on attempt {}: {}",
            self.purpose, self.joint, self.attempts_used, self.source
        )
    }
}

impl std::error::Error for FrameWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RequestError {
    RequestWrite(FrameWriteError),
    ResponseFrame(FrameReadError),
    Telemetry(TelemetryParseError),
}

impl fmt::Display for RequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "typed STS request/response failed: {self:?}")
    }
}

impl std::error::Error for RequestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RequestWrite(source) => Some(source),
            Self::ResponseFrame(source) => Some(source),
            Self::Telemetry(source) => Some(source),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CancellationCause {
    RequestedShutdown,
    HandleDropped,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadRuntimeError {
    Cancelled {
        cause: CancellationCause,
        stage: RuntimeStage,
        joint: HeadJoint,
    },
    PositionObservation {
        joint: HeadJoint,
        stage: RuntimeStage,
        source: RequestError,
    },
    PositionAgreement {
        joint: HeadJoint,
        source: PositionAgreementError,
    },
    PoseAdmission {
        source: HeadPoseError,
    },
    Write {
        stage: RuntimeStage,
        source: FrameWriteError,
    },
    VerificationRead {
        joint: HeadJoint,
        source: RequestError,
    },
    ReadbackMismatch {
        joint: HeadJoint,
        sample: VerificationSample,
        target: PositionTicks,
        actual: PositionTicks,
        absolute_difference_ticks: u16,
        tolerance: PositionAgreementTicks,
    },
    ReadbackMoving {
        joint: HeadJoint,
        sample: VerificationSample,
        position: PositionTicks,
    },
    ReadbackUnstable {
        joint: HeadJoint,
        first: PositionTicks,
        second: PositionTicks,
        absolute_difference_ticks: u16,
        tolerance: PositionAgreementTicks,
    },
    ObservationClockRegression {
        oldest_observation_at: MonotonicTime,
        checked_at: MonotonicTime,
    },
    ObservationStaleBeforeArming {
        joint: HeadJoint,
        check: ArmingFreshnessCheck,
        oldest_observation_at: MonotonicTime,
        checked_at: MonotonicTime,
        age: Duration,
        maximum_age: Duration,
    },
    ObservationArmingWriteBudgetInsufficient {
        joint: HeadJoint,
        oldest_observation_at: MonotonicTime,
        checked_at: MonotonicTime,
        remaining_freshness: Duration,
        required_write_budget: Duration,
    },
}

impl fmt::Display for HeadRuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Kiko natural-hold startup failed: {self:?}")
    }
}

impl std::error::Error for HeadRuntimeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PositionObservation { source, .. } | Self::VerificationRead { source, .. } => {
                Some(source)
            }
            Self::PositionAgreement { source, .. } => Some(source),
            Self::PoseAdmission { source } => Some(source),
            Self::Write { source, .. } => Some(source),
            Self::Cancelled { .. }
            | Self::ReadbackMismatch { .. }
            | Self::ReadbackMoving { .. }
            | Self::ReadbackUnstable { .. }
            | Self::ObservationClockRegression { .. }
            | Self::ObservationStaleBeforeArming { .. }
            | Self::ObservationArmingWriteBudgetInsufficient { .. } => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TorqueDisableJointOutcome {
    joint: HeadJoint,
    result: Result<WriteEvidence, FrameWriteError>,
}

impl TorqueDisableJointOutcome {
    pub const fn joint(&self) -> HeadJoint {
        self.joint
    }

    pub const fn result(&self) -> &Result<WriteEvidence, FrameWriteError> {
        &self.result
    }
}

/// Every element is present because shutdown always attempts all four joints,
/// even after an earlier disable write fails.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TorqueDisableReport {
    started_at: MonotonicTime,
    completed_at: MonotonicTime,
    outcomes: [TorqueDisableJointOutcome; 4],
}

impl TorqueDisableReport {
    pub const fn started_at(&self) -> MonotonicTime {
        self.started_at
    }

    pub const fn completed_at(&self) -> MonotonicTime {
        self.completed_at
    }

    pub const fn outcomes(&self) -> &[TorqueDisableJointOutcome; 4] {
        &self.outcomes
    }

    pub fn all_writes_completed(&self) -> bool {
        self.outcomes.iter().all(|outcome| outcome.result.is_ok())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActorTermination {
    RequestedShutdown,
    HandleDropped,
    StartupFault,
    StartupFaultWithShutdownRequested,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActorExit {
    startup: Result<VerifiedNaturalHoldEvidence, HeadRuntimeError>,
    termination: ActorTermination,
    torque_disable: TorqueDisableReport,
}

impl ActorExit {
    pub const fn startup(&self) -> &Result<VerifiedNaturalHoldEvidence, HeadRuntimeError> {
        &self.startup
    }

    pub const fn termination(&self) -> &ActorTermination {
        &self.termination
    }

    pub const fn torque_disable(&self) -> &TorqueDisableReport {
        &self.torque_disable
    }
}

enum HeadCommand {
    Shutdown {
        response: oneshot::Sender<TorqueDisableReport>,
    },
}

/// The only public command endpoint. It is intentionally not cloneable: one
/// caller owns shutdown authority, while the actor exclusively owns serial I/O.
pub struct HeadActorHandle {
    commands: mpsc::Sender<HeadCommand>,
}

impl HeadActorHandle {
    pub async fn shutdown(self) -> Result<TorqueDisableReport, ShutdownError> {
        let (response, result) = oneshot::channel();
        self.commands
            .send(HeadCommand::Shutdown { response })
            .await
            .map_err(|_| ShutdownError::ActorAlreadyStopped)?;
        result
            .await
            .map_err(|_| ShutdownError::ActorStoppedBeforeReporting)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShutdownError {
    ActorAlreadyStopped,
    ActorStoppedBeforeReporting,
}

impl fmt::Display for ShutdownError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "head actor shutdown command failed: {self:?}")
    }
}

impl std::error::Error for ShutdownError {}

pub struct StartupReceipt {
    result: oneshot::Receiver<Result<VerifiedNaturalHoldEvidence, HeadRuntimeError>>,
}

impl StartupReceipt {
    pub async fn wait(
        self,
    ) -> Result<Result<VerifiedNaturalHoldEvidence, HeadRuntimeError>, StartupReceiptError> {
        self.result
            .await
            .map_err(|_| StartupReceiptError::ActorStoppedBeforeReporting)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartupReceiptError {
    ActorStoppedBeforeReporting,
}

impl fmt::Display for StartupReceiptError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "head actor startup receipt failed: {self:?}")
    }
}

impl std::error::Error for StartupReceiptError {}

pub struct HeadActorTask {
    task: JoinHandle<ActorExit>,
}

impl HeadActorTask {
    pub async fn join(self) -> Result<ActorExit, JoinError> {
        self.task.await
    }
}

#[derive(Debug)]
pub enum HeadActorSpawnError {
    NoTokioRuntime { source: TryCurrentError },
}

impl fmt::Display for HeadActorSpawnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "could not spawn Kiko head actor: {self:?}")
    }
}

impl std::error::Error for HeadActorSpawnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoTokioRuntime { source } => Some(source),
        }
    }
}

#[derive(Debug)]
pub enum HeadActorStartError {
    NoTokioRuntime { source: TryCurrentError },
    Serial { source: SerialOpenError },
}

impl fmt::Display for HeadActorStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "could not start Kiko head actor: {self:?}")
    }
}

impl std::error::Error for HeadActorStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoTokioRuntime { source } => Some(source),
            Self::Serial { source } => Some(source),
        }
    }
}

/// Spawn the testable core with one transport owner and one injected clock.
pub fn spawn_head_actor<T, C>(
    transport: T,
    clock: C,
    config: HeadRuntimeConfig,
    consent: PhysicalTorqueEnableConsent,
) -> Result<(HeadActorHandle, StartupReceipt, HeadActorTask), HeadActorSpawnError>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let runtime =
        Handle::try_current().map_err(|source| HeadActorSpawnError::NoTokioRuntime { source })?;
    Ok(spawn_head_actor_on(
        &runtime, transport, clock, config, consent,
    ))
}

fn spawn_head_actor_on<T, C>(
    runtime: &Handle,
    transport: T,
    clock: C,
    config: HeadRuntimeConfig,
    _consent: PhysicalTorqueEnableConsent,
) -> (HeadActorHandle, StartupReceipt, HeadActorTask)
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let (commands, receiver) = mpsc::channel(ACTOR_MAILBOX_CAPACITY);
    let (startup_sender, startup_result) = oneshot::channel();
    let actor = HeadActor {
        transport,
        clock,
        config,
    };
    let task = runtime.spawn(actor.run(receiver, startup_sender));
    (
        HeadActorHandle { commands },
        StartupReceipt {
            result: startup_result,
        },
        HeadActorTask { task },
    )
}

/// Open, exclusively claim, configure, and then spawn the production serial
/// actor. No protocol traffic occurs until every serial setting succeeds.
pub fn start_serial_head_actor(
    config: HeadRuntimeConfig,
    consent: PhysicalTorqueEnableConsent,
) -> Result<
    (
        SerialConfigurationEvidence,
        HeadActorHandle,
        StartupReceipt,
        HeadActorTask,
    ),
    HeadActorStartError,
> {
    // Check the runtime before opening or changing any physical serial state.
    let runtime =
        Handle::try_current().map_err(|source| HeadActorStartError::NoTokioRuntime { source })?;
    let transport = SerialTransport::open(config.device())
        .map_err(|source| HeadActorStartError::Serial { source })?;
    let serial_evidence = transport.evidence().clone();
    let (handle, startup, task) =
        spawn_head_actor_on(&runtime, transport, TokioClock::new(), config, consent);
    Ok((serial_evidence, handle, startup, task))
}

struct HeadActor<T, C> {
    transport: T,
    clock: C,
    config: HeadRuntimeConfig,
}

struct ControlState {
    termination: Option<ActorTermination>,
    shutdown_response: Option<oneshot::Sender<TorqueDisableReport>>,
}

impl ControlState {
    const fn new() -> Self {
        Self {
            termination: None,
            shutdown_response: None,
        }
    }
}

impl<T, C> HeadActor<T, C>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    async fn run(
        mut self,
        mut commands: mpsc::Receiver<HeadCommand>,
        startup_sender: oneshot::Sender<Result<VerifiedNaturalHoldEvidence, HeadRuntimeError>>,
    ) -> ActorExit {
        let mut control = ControlState::new();
        let startup = self.startup(&mut commands, &mut control).await;
        // Startup observation is optional to the actor's safety. If the caller
        // dropped its receipt, the complete result still remains in ActorExit.
        let _startup_receiver_present = startup_sender.send(startup.clone()).is_ok();

        let termination = if let Some(termination) = control.termination.clone() {
            termination
        } else if startup.is_err() {
            // Reject future commands, then drain a shutdown that was already
            // queued while the failing operation was in flight.
            commands.close();
            match commands.try_recv() {
                Ok(HeadCommand::Shutdown { response }) => {
                    control.shutdown_response = Some(response);
                    ActorTermination::StartupFaultWithShutdownRequested
                }
                Err(mpsc::error::TryRecvError::Empty | mpsc::error::TryRecvError::Disconnected) => {
                    ActorTermination::StartupFault
                }
            }
        } else {
            match commands.recv().await {
                Some(HeadCommand::Shutdown { response }) => {
                    control.shutdown_response = Some(response);
                    ActorTermination::RequestedShutdown
                }
                None => ActorTermination::HandleDropped,
            }
        };
        commands.close();

        let torque_disable = self.disable_all().await;
        if let Some(response) = control.shutdown_response {
            // The report remains in ActorExit if the requester was cancelled.
            let _requester_present = response.send(torque_disable.clone()).is_ok();
        }
        ActorExit {
            startup,
            termination,
            torque_disable,
        }
    }

    async fn startup(
        &mut self,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<VerifiedNaturalHoldEvidence, HeadRuntimeError> {
        let started_at = self.clock.now();
        let bow = self
            .observe_joint(HeadJoint::Bow, commands, control)
            .await?;
        let curl = self
            .observe_joint(HeadJoint::Curl, commands, control)
            .await?;
        let yaw = self
            .observe_joint(HeadJoint::Yaw, commands, control)
            .await?;
        let roll = self
            .observe_joint(HeadJoint::Roll, commands, control)
            .await?;
        let observations = [bow, curl, yaw, roll];
        let oldest_observation_at = observations
            .iter()
            .map(|observation| observation.second().received_at())
            .min()
            .expect("the exact head pose always has four observations");
        self.ensure_observation_freshness(
            oldest_observation_at,
            HeadJoint::Bow,
            ArmingFreshnessCheck::BeforeConfigurationWrites,
        )?;
        let observed_pose = HeadPose::try_from_validated([
            observations[0].validated(),
            observations[1].validated(),
            observations[2].validated(),
            observations[3].validated(),
        ])
        .map_err(|source| HeadRuntimeError::PoseAdmission { source })?;
        let frames = build_natural_hold_frames(
            observed_pose,
            self.config.torque_limits(),
            self.config.goal_speed(),
        );

        // Clamp torque before writing any goal, including the observed pose.
        let torque_limit_writes = self
            .write_stage(
                frames.torque_limit_writes(),
                RuntimeStage::WriteTorqueLimit,
                WritePurpose::TorqueLimit,
                commands,
                control,
            )
            .await?;
        let observed_goal_writes = self
            .write_stage(
                frames.goal_writes(),
                RuntimeStage::WriteObservedGoal,
                WritePurpose::ObservedGoal,
                commands,
                control,
            )
            .await?;
        let torque_enable_writes = self
            .write_enable_stage(
                frames.torque_enable_writes(),
                oldest_observation_at,
                commands,
                control,
            )
            .await?;

        let readbacks = [
            self.verify_joint(
                HeadJoint::Bow,
                observed_pose,
                &frames.verification_reads()[0],
                commands,
                control,
            )
            .await?,
            self.verify_joint(
                HeadJoint::Curl,
                observed_pose,
                &frames.verification_reads()[1],
                commands,
                control,
            )
            .await?,
            self.verify_joint(
                HeadJoint::Yaw,
                observed_pose,
                &frames.verification_reads()[2],
                commands,
                control,
            )
            .await?,
            self.verify_joint(
                HeadJoint::Roll,
                observed_pose,
                &frames.verification_reads()[3],
                commands,
                control,
            )
            .await?,
        ];

        Ok(VerifiedNaturalHoldEvidence {
            started_at,
            completed_at: self.clock.now(),
            observed_pose,
            observations,
            observed_goal_writes,
            torque_limit_writes,
            torque_enable_writes,
            readbacks,
        })
    }

    async fn observe_joint(
        &mut self,
        joint: HeadJoint,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<PositionObservationEvidence, HeadRuntimeError> {
        self.check_control(commands, control, RuntimeStage::ObserveFirst, joint)?;
        let first = self.read_position(joint).await.map_err(|source| {
            HeadRuntimeError::PositionObservation {
                joint,
                stage: RuntimeStage::ObserveFirst,
                source,
            }
        })?;
        self.check_control(commands, control, RuntimeStage::ObserveSecond, joint)?;
        let second = self.read_position(joint).await.map_err(|source| {
            HeadRuntimeError::PositionObservation {
                joint,
                stage: RuntimeStage::ObserveSecond,
                source,
            }
        })?;
        let validated = ValidatedPresentPosition::try_from_pair(
            *first.value(),
            *second.value(),
            self.config.redundant_read_tolerance(),
        )
        .map_err(|source| HeadRuntimeError::PositionAgreement { joint, source })?;
        Ok(PositionObservationEvidence {
            joint,
            first,
            second,
            validated,
        })
    }

    async fn read_position(
        &mut self,
        joint: HeadJoint,
    ) -> Result<ResponseEvidence<PresentPosition>, RequestError> {
        let id = joint.servo_id();
        let request_write = self
            .write_frame(
                joint,
                WritePurpose::PositionReadRequest,
                build_position_read(id).as_bytes(),
            )
            .await
            .map_err(RequestError::RequestWrite)?;
        let frame = read_response_frame(
            &mut self.transport,
            &self.clock,
            self.config.response_timeout(),
            self.config.noise_budget_bytes(),
        )
        .await
        .map_err(RequestError::ResponseFrame)?;
        let value =
            PresentPosition::parse(frame.as_bytes(), id).map_err(RequestError::Telemetry)?;
        Ok(ResponseEvidence {
            value,
            request_write,
            discarded_noise_bytes: frame.discarded_noise_bytes(),
            received_at: self.clock.now(),
        })
    }

    async fn write_stage(
        &mut self,
        frames: &[kiko_head_protocol::CommandFrame; 4],
        stage: RuntimeStage,
        purpose: WritePurpose,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<[WriteEvidence; 4], HeadRuntimeError> {
        let bow = self
            .write_controlled(
                HeadJoint::Bow,
                &frames[0],
                stage,
                purpose,
                commands,
                control,
            )
            .await?;
        let curl = self
            .write_controlled(
                HeadJoint::Curl,
                &frames[1],
                stage,
                purpose,
                commands,
                control,
            )
            .await?;
        let yaw = self
            .write_controlled(
                HeadJoint::Yaw,
                &frames[2],
                stage,
                purpose,
                commands,
                control,
            )
            .await?;
        let roll = self
            .write_controlled(
                HeadJoint::Roll,
                &frames[3],
                stage,
                purpose,
                commands,
                control,
            )
            .await?;
        Ok([bow, curl, yaw, roll])
    }

    async fn write_enable_stage(
        &mut self,
        frames: &[kiko_head_protocol::CommandFrame; 4],
        oldest_observation_at: MonotonicTime,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<[WriteEvidence; 4], HeadRuntimeError> {
        let bow = self
            .write_enable_controlled(
                HeadJoint::Bow,
                &frames[0],
                oldest_observation_at,
                commands,
                control,
            )
            .await?;
        let curl = self
            .write_enable_controlled(
                HeadJoint::Curl,
                &frames[1],
                oldest_observation_at,
                commands,
                control,
            )
            .await?;
        let yaw = self
            .write_enable_controlled(
                HeadJoint::Yaw,
                &frames[2],
                oldest_observation_at,
                commands,
                control,
            )
            .await?;
        let roll = self
            .write_enable_controlled(
                HeadJoint::Roll,
                &frames[3],
                oldest_observation_at,
                commands,
                control,
            )
            .await?;
        Ok([bow, curl, yaw, roll])
    }

    async fn write_enable_controlled(
        &mut self,
        joint: HeadJoint,
        frame: &kiko_head_protocol::CommandFrame,
        oldest_observation_at: MonotonicTime,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<WriteEvidence, HeadRuntimeError> {
        self.check_control(commands, control, RuntimeStage::EnableTorque, joint)?;
        self.ensure_arming_write_budget(oldest_observation_at, joint)?;
        let evidence = self
            .write_frame(joint, WritePurpose::TorqueEnable, frame.as_bytes())
            .await
            .map_err(|source| HeadRuntimeError::Write {
                stage: RuntimeStage::EnableTorque,
                source,
            })?;
        self.ensure_observation_freshness(
            oldest_observation_at,
            joint,
            ArmingFreshnessCheck::AfterEnableWrite,
        )?;
        Ok(evidence)
    }

    fn ensure_observation_freshness(
        &self,
        oldest_observation_at: MonotonicTime,
        joint: HeadJoint,
        check: ArmingFreshnessCheck,
    ) -> Result<(), HeadRuntimeError> {
        let (checked_at, age) = self.observation_age(oldest_observation_at)?;
        let maximum_age = self.config.arming_freshness().get();
        if age > maximum_age {
            return Err(HeadRuntimeError::ObservationStaleBeforeArming {
                joint,
                check,
                oldest_observation_at,
                checked_at,
                age,
                maximum_age,
            });
        }
        Ok(())
    }

    fn ensure_arming_write_budget(
        &self,
        oldest_observation_at: MonotonicTime,
        joint: HeadJoint,
    ) -> Result<(), HeadRuntimeError> {
        let (checked_at, age) = self.observation_age(oldest_observation_at)?;
        let maximum_age = self.config.arming_freshness().get();
        if age > maximum_age {
            return Err(HeadRuntimeError::ObservationStaleBeforeArming {
                joint,
                check: ArmingFreshnessCheck::BeforeEnableWrite,
                oldest_observation_at,
                checked_at,
                age,
                maximum_age,
            });
        }
        let remaining_freshness = maximum_age
            .checked_sub(age)
            .expect("age was admitted inside the freshness bound");
        let required_write_budget = self
            .config
            .write_timeout()
            .get()
            .checked_mul(u32::from(self.config.write_attempts().get()))
            .expect("parsed timeout and attempt bounds fit Duration");
        if remaining_freshness < required_write_budget {
            return Err(HeadRuntimeError::ObservationArmingWriteBudgetInsufficient {
                joint,
                oldest_observation_at,
                checked_at,
                remaining_freshness,
                required_write_budget,
            });
        }
        Ok(())
    }

    fn observation_age(
        &self,
        oldest_observation_at: MonotonicTime,
    ) -> Result<(MonotonicTime, Duration), HeadRuntimeError> {
        let checked_at = self.clock.now();
        let age = checked_at
            .checked_duration_since(oldest_observation_at)
            .ok_or(HeadRuntimeError::ObservationClockRegression {
                oldest_observation_at,
                checked_at,
            })?;
        Ok((checked_at, age))
    }

    async fn write_controlled(
        &mut self,
        joint: HeadJoint,
        frame: &kiko_head_protocol::CommandFrame,
        stage: RuntimeStage,
        purpose: WritePurpose,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<WriteEvidence, HeadRuntimeError> {
        self.check_control(commands, control, stage, joint)?;
        self.write_frame(joint, purpose, frame.as_bytes())
            .await
            .map_err(|source| HeadRuntimeError::Write { stage, source })
    }

    async fn verify_joint(
        &mut self,
        joint: HeadJoint,
        pose: HeadPose,
        request: &kiko_head_protocol::CommandFrame,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
    ) -> Result<ReadbackEvidence, HeadRuntimeError> {
        self.check_control(
            commands,
            control,
            RuntimeStage::VerifyFirstStoppedPosition,
            joint,
        )?;
        let target = pose.position(joint);
        let first = self
            .read_telemetry(joint, request)
            .await
            .map_err(|source| HeadRuntimeError::VerificationRead { joint, source })?;
        let first_target_difference_ticks =
            self.admit_stopped_readback(joint, VerificationSample::First, target, first.value())?;

        self.check_control(
            commands,
            control,
            RuntimeStage::VerifySecondStoppedPosition,
            joint,
        )?;
        let second = self
            .read_telemetry(joint, request)
            .await
            .map_err(|source| HeadRuntimeError::VerificationRead { joint, source })?;
        let second_target_difference_ticks =
            self.admit_stopped_readback(joint, VerificationSample::Second, target, second.value())?;
        let stable_difference_ticks = first
            .value()
            .position()
            .get()
            .abs_diff(second.value().position().get());
        if stable_difference_ticks > self.config.readback_tolerance().get() {
            return Err(HeadRuntimeError::ReadbackUnstable {
                joint,
                first: first.value().position(),
                second: second.value().position(),
                absolute_difference_ticks: stable_difference_ticks,
                tolerance: self.config.readback_tolerance(),
            });
        }

        Ok(ReadbackEvidence {
            joint,
            target,
            first_target_difference_ticks,
            second_target_difference_ticks,
            stable_difference_ticks,
            first,
            second,
        })
    }

    async fn read_telemetry(
        &mut self,
        joint: HeadJoint,
        request: &kiko_head_protocol::CommandFrame,
    ) -> Result<ResponseEvidence<FullTelemetry>, RequestError> {
        let id = joint.servo_id();
        let request_write = self
            .write_frame(
                joint,
                WritePurpose::TelemetryReadRequest,
                request.as_bytes(),
            )
            .await
            .map_err(RequestError::RequestWrite)?;
        let frame = read_response_frame(
            &mut self.transport,
            &self.clock,
            self.config.response_timeout(),
            self.config.noise_budget_bytes(),
        )
        .await
        .map_err(RequestError::ResponseFrame)?;
        let value = FullTelemetry::parse(frame.as_bytes(), id).map_err(RequestError::Telemetry)?;
        Ok(ResponseEvidence {
            value,
            request_write,
            discarded_noise_bytes: frame.discarded_noise_bytes(),
            received_at: self.clock.now(),
        })
    }

    fn admit_stopped_readback(
        &self,
        joint: HeadJoint,
        sample: VerificationSample,
        target: PositionTicks,
        telemetry: &FullTelemetry,
    ) -> Result<u16, HeadRuntimeError> {
        if telemetry.is_moving() {
            return Err(HeadRuntimeError::ReadbackMoving {
                joint,
                sample,
                position: telemetry.position(),
            });
        }
        let absolute_difference_ticks = target.get().abs_diff(telemetry.position().get());
        if absolute_difference_ticks > self.config.readback_tolerance().get() {
            return Err(HeadRuntimeError::ReadbackMismatch {
                joint,
                sample,
                target,
                actual: telemetry.position(),
                absolute_difference_ticks,
                tolerance: self.config.readback_tolerance(),
            });
        }
        Ok(absolute_difference_ticks)
    }

    async fn write_frame(
        &mut self,
        joint: HeadJoint,
        purpose: WritePurpose,
        bytes: &[u8],
    ) -> Result<WriteEvidence, FrameWriteError> {
        // The common path does not allocate. Capacity grows only when an
        // explicitly configured, retryable zero-progress failure occurs.
        let mut recovered_failures = Vec::new();
        let maximum_attempts = self.config.write_attempts().get();
        let mut attempt = 1_u8;
        loop {
            match self
                .transport
                .write_all(bytes, self.config.write_timeout().get())
                .await
            {
                Ok(()) => {
                    return Ok(WriteEvidence {
                        attempts_used: attempt,
                        recovered_failures,
                        completed_at: self.clock.now(),
                    });
                }
                Err(source)
                    if attempt < maximum_attempts && source.is_retryable_without_progress() =>
                {
                    recovered_failures.push(source);
                    attempt += 1;
                }
                Err(source) => {
                    return Err(FrameWriteError {
                        joint,
                        purpose,
                        attempts_used: attempt,
                        recovered_failures,
                        source,
                    });
                }
            }
        }
    }

    fn check_control(
        &self,
        commands: &mut mpsc::Receiver<HeadCommand>,
        control: &mut ControlState,
        stage: RuntimeStage,
        joint: HeadJoint,
    ) -> Result<(), HeadRuntimeError> {
        match commands.try_recv() {
            Ok(HeadCommand::Shutdown { response }) => {
                control.termination = Some(ActorTermination::RequestedShutdown);
                control.shutdown_response = Some(response);
                Err(HeadRuntimeError::Cancelled {
                    cause: CancellationCause::RequestedShutdown,
                    stage,
                    joint,
                })
            }
            Err(mpsc::error::TryRecvError::Disconnected) => {
                control.termination = Some(ActorTermination::HandleDropped);
                Err(HeadRuntimeError::Cancelled {
                    cause: CancellationCause::HandleDropped,
                    stage,
                    joint,
                })
            }
            Err(mpsc::error::TryRecvError::Empty) => Ok(()),
        }
    }

    async fn disable_all(&mut self) -> TorqueDisableReport {
        let started_at = self.clock.now();
        let bow = self.disable_joint(HeadJoint::Bow).await;
        let curl = self.disable_joint(HeadJoint::Curl).await;
        let yaw = self.disable_joint(HeadJoint::Yaw).await;
        let roll = self.disable_joint(HeadJoint::Roll).await;
        TorqueDisableReport {
            started_at,
            completed_at: self.clock.now(),
            outcomes: [bow, curl, yaw, roll],
        }
    }

    async fn disable_joint(&mut self, joint: HeadJoint) -> TorqueDisableJointOutcome {
        let frame = build_torque_switch_write(joint.servo_id(), TorqueSwitch::Disabled);
        TorqueDisableJointOutcome {
            joint,
            result: self
                .write_frame(joint, WritePurpose::TorqueDisable, frame.as_bytes())
                .await,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};
    use std::io;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use kiko_head_protocol::{ResponseParseError, ServoId, TelemetryParseError};

    use super::*;
    use crate::config::HeadRuntimeConfigInput;
    use crate::transport::{TransportFailureKind, TransportOperation};

    #[derive(Clone, Default)]
    struct TestClock {
        nanoseconds: Arc<AtomicU64>,
    }

    impl TestClock {
        fn advance_one_millisecond(&self) {
            self.nanoseconds.fetch_add(1_000_000, Ordering::Relaxed);
        }
    }

    impl MonotonicClock for TestClock {
        fn now(&self) -> MonotonicTime {
            MonotonicTime::from_duration_since_origin(Duration::from_nanos(
                self.nanoseconds.load(Ordering::Relaxed),
            ))
        }
    }

    #[derive(Clone, Debug)]
    enum ReadAction {
        Bytes(Vec<u8>),
        Eof,
        Failure(TransportFailure),
        GatedFailure {
            entered: Arc<tokio::sync::Notify>,
            release: Arc<tokio::sync::Notify>,
            source: TransportFailure,
        },
    }

    #[derive(Default)]
    struct FakeShared {
        writes: Vec<Vec<u8>>,
        write_failures: BTreeMap<usize, TransportFailure>,
    }

    struct FakeTransport {
        clock: TestClock,
        reads: VecDeque<ReadAction>,
        pending: VecDeque<u8>,
        shared: Arc<Mutex<FakeShared>>,
    }

    impl FakeTransport {
        fn new(clock: TestClock, reads: Vec<ReadAction>) -> (Self, Arc<Mutex<FakeShared>>) {
            let shared = Arc::new(Mutex::new(FakeShared::default()));
            (
                Self {
                    clock,
                    reads: reads.into(),
                    pending: VecDeque::new(),
                    shared: Arc::clone(&shared),
                },
                shared,
            )
        }
    }

    impl AsyncByteTransport for FakeTransport {
        async fn write_all(
            &mut self,
            bytes: &[u8],
            _timeout: Duration,
        ) -> Result<(), TransportFailure> {
            self.clock.advance_one_millisecond();
            let mut shared = self.shared.lock().expect("fake transport mutex");
            let call = shared.writes.len();
            shared.writes.push(bytes.to_vec());
            match shared.write_failures.remove(&call) {
                Some(source) => Err(source),
                None => Ok(()),
            }
        }

        async fn read_some(
            &mut self,
            bytes: &mut [u8],
            _timeout: Duration,
        ) -> Result<usize, TransportFailure> {
            self.clock.advance_one_millisecond();
            loop {
                if !self.pending.is_empty() {
                    let read = bytes.len().min(self.pending.len());
                    for destination in &mut bytes[..read] {
                        *destination = self.pending.pop_front().expect("pending byte");
                    }
                    return Ok(read);
                }
                match self.reads.pop_front() {
                    Some(ReadAction::Bytes(chunk)) => self.pending.extend(chunk),
                    Some(ReadAction::Eof) | None => return Ok(0),
                    Some(ReadAction::Failure(source)) => return Err(source),
                    Some(ReadAction::GatedFailure {
                        entered,
                        release,
                        source,
                    }) => {
                        entered.notify_one();
                        release.notified().await;
                        return Err(source);
                    }
                }
            }
        }
    }

    fn valid_config(write_attempts: u8) -> HeadRuntimeConfig {
        config_with_freshness(write_attempts, 250)
    }

    fn config_with_freshness(write_attempts: u8, arming_freshness_ms: u64) -> HeadRuntimeConfig {
        HeadRuntimeConfig::parse(HeadRuntimeConfigInput {
            device_path: "/dev/serial/by-id/usb-Kiko_head_test".to_owned(),
            response_timeout_ms: 100,
            write_timeout_ms: 1,
            arming_freshness_ms,
            write_attempts,
            noise_budget_bytes: 16,
            redundant_read_tolerance_ticks: 10,
            readback_tolerance_ticks: 20,
            goal_speed_ticks_per_second: 100,
            torque_limit_permille: [600, 400, 400, 400],
        })
        .expect("test configuration")
    }

    fn status(id: ServoId, parameters: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0xff, 0xff, id.get(), 0, 0];
        bytes[3] = u8::try_from(parameters.len() + 2).expect("test response length");
        bytes.extend_from_slice(parameters);
        let checksum = !bytes[2..]
            .iter()
            .fold(0_u8, |sum, byte| sum.wrapping_add(*byte));
        bytes.push(checksum);
        bytes
    }

    fn position_response(joint: HeadJoint, position: u16) -> Vec<u8> {
        status(joint.servo_id(), &position.to_le_bytes())
    }

    fn telemetry_response(joint: HeadJoint, position: u16) -> Vec<u8> {
        telemetry_response_with_moving(joint, position, false)
    }

    fn telemetry_response_with_moving(joint: HeadJoint, position: u16, moving: bool) -> Vec<u8> {
        let mut telemetry = [0_u8; 15];
        telemetry[..2].copy_from_slice(&position.to_le_bytes());
        telemetry[10] = u8::from(moving);
        status(joint.servo_id(), &telemetry)
    }

    fn successful_reads() -> Vec<ReadAction> {
        let positions = [2_127_u16, 2_558, 2_925, 2_930];
        let mut reads = Vec::with_capacity(16);
        for (joint, position) in HeadJoint::ALL.into_iter().zip(positions) {
            reads.push(ReadAction::Bytes(position_response(joint, position - 2)));
            reads.push(ReadAction::Bytes(position_response(joint, position)));
        }
        for (joint, position) in HeadJoint::ALL.into_iter().zip(positions) {
            reads.push(ReadAction::Bytes(telemetry_response(joint, position)));
            reads.push(ReadAction::Bytes(telemetry_response(joint, position)));
        }
        reads
    }

    fn spawn_fake(
        reads: Vec<ReadAction>,
        config: HeadRuntimeConfig,
    ) -> (
        HeadActorHandle,
        StartupReceipt,
        HeadActorTask,
        Arc<Mutex<FakeShared>>,
    ) {
        let clock = TestClock::default();
        let (transport, shared) = FakeTransport::new(clock.clone(), reads);
        let (handle, startup, task) = spawn_head_actor(
            transport,
            clock,
            config,
            PhysicalTorqueEnableConsent::explicitly_granted(),
        )
        .expect("test runtime is active");
        (handle, startup, task, shared)
    }

    async fn run_startup_fault(
        reads: Vec<ReadAction>,
        config: HeadRuntimeConfig,
    ) -> (HeadRuntimeError, ActorExit, Arc<Mutex<FakeShared>>) {
        let (handle, receipt, task, shared) = spawn_fake(reads, config);
        let error = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect_err("startup must fail");
        drop(handle);
        let exit = task.join().await.expect("actor task");
        (error, exit, shared)
    }

    #[tokio::test]
    async fn startup_holds_only_observed_pose_and_shutdown_disables_every_joint() {
        let (handle, receipt, task, shared) = spawn_fake(successful_reads(), valid_config(1));
        let evidence = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("verified natural hold");
        assert_eq!(
            evidence.observed_pose().positions().map(PositionTicks::get),
            [2_127, 2_558, 2_925, 2_930]
        );
        assert!(evidence.readbacks().iter().all(|readback| {
            readback.first_target_difference_ticks() == 0
                && readback.second_target_difference_ticks() == 0
                && readback.stable_difference_ticks() == 0
        }));

        let disable = handle.shutdown().await.expect("shutdown report");
        assert!(disable.all_writes_completed());
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.termination(), &ActorTermination::RequestedShutdown);
        assert_eq!(exit.torque_disable(), &disable);

        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.writes.len(), 32);
        for write in &shared.writes[..8] {
            assert_eq!(&write[4..=6], &[2, 56, 2]);
        }
        assert!(shared.writes[8..12].iter().all(|write| write[5] == 48));
        for write in &shared.writes[12..16] {
            assert_eq!(write[5], 42);
            let id = usize::from(write[2] - 1);
            let target = u16::from_le_bytes([write[6], write[7]]);
            assert_eq!(target, [2_127, 2_558, 2_925, 2_930][id]);
            assert_ne!(u16::from_le_bytes([write[10], write[11]]), 0);
        }
        assert!(
            shared.writes[16..20]
                .iter()
                .all(|write| write[5..=6] == [40, 1])
        );
        assert!(
            shared.writes[20..28]
                .iter()
                .all(|write| write[4..=6] == [2, 56, 15])
        );
        assert!(
            shared.writes[28..]
                .iter()
                .all(|write| write[5..=6] == [40, 0])
        );
    }

    #[tokio::test]
    async fn bounded_noise_is_resynchronised_and_reported() {
        let mut reads = successful_reads();
        let ReadAction::Bytes(first) = &mut reads[0] else {
            unreachable!("successful reads are byte chunks");
        };
        first.splice(0..0, [0x01, 0x7e]);
        let (handle, receipt, task, _) = spawn_fake(reads, valid_config(1));
        let evidence = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("noise within budget");
        assert_eq!(
            evidence.observations()[0].first().discarded_noise_bytes(),
            2
        );
        handle.shutdown().await.expect("shutdown");
        task.join().await.expect("actor task");
    }

    #[tokio::test]
    async fn noise_budget_and_declared_frame_bound_fail_closed() {
        let mut noisy = position_response(HeadJoint::Bow, 2_125);
        noisy.splice(0..0, [0x01; 17]);
        let (error, _, _) =
            run_startup_fault(vec![ReadAction::Bytes(noisy)], valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::ResponseFrame(FrameReadError::NoiseBudgetExceeded {
                    budget_bytes: 16,
                    observed_noise_bytes: 17,
                }),
                ..
            }
        ));

        let oversized = vec![0xff, 0xff, HeadJoint::Bow.servo_id().get(), 0xff];
        let (error, _, _) =
            run_startup_fault(vec![ReadAction::Bytes(oversized)], valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::ResponseFrame(FrameReadError::DeclaredLengthOutOfRange {
                    declared_bytes: 259,
                    maximum_bytes: 21,
                    ..
                }),
                ..
            }
        ));
    }

    #[tokio::test]
    async fn truncation_is_not_relabelled_as_a_protocol_error() {
        let complete = position_response(HeadJoint::Bow, 2_125);
        let reads = vec![ReadAction::Bytes(complete[..4].to_vec()), ReadAction::Eof];
        let (error, exit, _) = run_startup_fault(reads, valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::ResponseFrame(FrameReadError::Truncated {
                    buffered_bytes: 4,
                    expected_bytes: Some(8),
                }),
                ..
            }
        ));
        assert!(exit.torque_disable().all_writes_completed());
    }

    #[tokio::test]
    async fn response_id_length_checksum_and_status_are_propagated_exactly() {
        let wrong_id = vec![ReadAction::Bytes(position_response(HeadJoint::Curl, 2_125))];
        let (error, _, _) = run_startup_fault(wrong_id, valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::Telemetry(TelemetryParseError::Response(
                    ResponseParseError::ServoIdMismatch { actual: 2, .. }
                )),
                ..
            }
        ));

        let mut corrupt = position_response(HeadJoint::Bow, 2_125);
        let last = corrupt.len() - 1;
        corrupt[last] ^= 1;
        let (error, _, _) =
            run_startup_fault(vec![ReadAction::Bytes(corrupt)], valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::Telemetry(TelemetryParseError::Response(
                    ResponseParseError::ChecksumMismatch { .. }
                )),
                ..
            }
        ));

        let wrong_parameter_count = status(HeadJoint::Bow.servo_id(), &[0x01]);
        let (error, _, _) = run_startup_fault(
            vec![ReadAction::Bytes(wrong_parameter_count)],
            valid_config(1),
        )
        .await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::Telemetry(TelemetryParseError::Response(
                    ResponseParseError::ParameterCountMismatch {
                        expected: 2,
                        actual: 1,
                    }
                )),
                ..
            }
        ));

        let mut device_fault = position_response(HeadJoint::Bow, 2_125);
        device_fault[4] = 0x40;
        let last = device_fault.len() - 1;
        device_fault[last] = !device_fault[2..last]
            .iter()
            .fold(0_u8, |sum, byte| sum.wrapping_add(*byte));
        let (error, _, _) =
            run_startup_fault(vec![ReadAction::Bytes(device_fault)], valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::Telemetry(TelemetryParseError::Response(
                    ResponseParseError::DeviceStatus(status)
                )),
                ..
            } if status.bits() == 0x40
        ));
    }

    #[tokio::test]
    async fn response_timeout_is_typed_and_shutdown_is_still_attempted() {
        let timeout = TransportFailure::timed_out(TransportOperation::Read, 0);
        let (error, exit, shared) =
            run_startup_fault(vec![ReadAction::Failure(timeout)], valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::ResponseFrame(FrameReadError::Transport {
                    source,
                    buffered_bytes: 0,
                    ..
                }),
                ..
            } if source.kind() == TransportFailureKind::TimedOut
        ));
        assert!(exit.torque_disable().all_writes_completed());
        assert_eq!(shared.lock().expect("fake state").writes.len(), 5);
    }

    #[tokio::test]
    async fn partial_write_failure_is_never_retried() {
        let (handle, receipt, task, shared) = spawn_fake(successful_reads(), valid_config(8));
        shared
            .lock()
            .expect("fake state")
            .write_failures
            .insert(0, TransportFailure::timed_out(TransportOperation::Write, 3));
        let error = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect_err("partial write must fail startup");
        assert!(matches!(
            error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::RequestWrite(FrameWriteError {
                    attempts_used: 1,
                    source,
                    ..
                }),
                ..
            } if source.bytes_transferred() == 3
        ));
        drop(handle);
        let exit = task.join().await.expect("actor task");
        assert!(exit.torque_disable().all_writes_completed());
        assert_eq!(shared.lock().expect("fake state").writes.len(), 5);
    }

    #[tokio::test]
    async fn zero_progress_retry_is_explicit_in_success_evidence() {
        let (handle, receipt, task, shared) = spawn_fake(successful_reads(), valid_config(2));
        shared
            .lock()
            .expect("fake state")
            .write_failures
            .insert(0, TransportFailure::timed_out(TransportOperation::Write, 0));
        let evidence = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("bounded retry succeeds");
        let write = evidence.observations()[0].first().request_write();
        assert_eq!(write.attempts_used(), 2);
        assert_eq!(write.recovered_failures().count(), 1);
        handle.shutdown().await.expect("shutdown");
        task.join().await.expect("actor task");
    }

    #[tokio::test]
    async fn post_write_position_mismatch_prevents_success() {
        let mut reads = successful_reads();
        reads[8] = ReadAction::Bytes(telemetry_response(HeadJoint::Bow, 2_200));
        let (error, exit, _) = run_startup_fault(reads, valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::ReadbackMismatch {
                joint: HeadJoint::Bow,
                sample: VerificationSample::First,
                target,
                actual,
                absolute_difference_ticks: 73,
                ..
            } if target.get() == 2_127 && actual.get() == 2_200
        ));
        assert!(exit.torque_disable().all_writes_completed());
    }

    #[tokio::test]
    async fn stale_observation_fails_before_any_arming_write() {
        let (error, exit, shared) =
            run_startup_fault(successful_reads(), config_with_freshness(1, 1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::ObservationStaleBeforeArming {
                joint: HeadJoint::Bow,
                check: ArmingFreshnessCheck::BeforeConfigurationWrites,
                age,
                maximum_age,
                ..
            } if age > maximum_age && maximum_age == Duration::from_millis(1)
        ));
        assert_eq!(exit.termination(), &ActorTermination::StartupFault);
        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.writes.len(), 12);
        assert!(shared.writes[..8].iter().all(|write| write[4] == 2));
        assert!(
            shared.writes[8..]
                .iter()
                .all(|write| write[5..=6] == [40, 0])
        );
    }

    #[tokio::test]
    async fn arming_requires_remaining_freshness_for_every_bounded_write_attempt() {
        let (error, exit, shared) =
            run_startup_fault(successful_reads(), config_with_freshness(8, 44)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::ObservationArmingWriteBudgetInsufficient {
                joint: HeadJoint::Bow,
                remaining_freshness,
                required_write_budget,
                ..
            } if remaining_freshness < required_write_budget
                && required_write_budget == Duration::from_millis(8)
        ));
        assert_eq!(exit.termination(), &ActorTermination::StartupFault);
        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.writes.len(), 20);
        assert!(
            shared.writes[16..]
                .iter()
                .all(|write| write[5..=6] == [40, 0])
        );
    }

    #[tokio::test]
    async fn moving_or_unstable_second_readback_never_claims_hold() {
        let mut moving = successful_reads();
        moving[9] = ReadAction::Bytes(telemetry_response_with_moving(HeadJoint::Bow, 2_127, true));
        let (error, _, _) = run_startup_fault(moving, valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::ReadbackMoving {
                joint: HeadJoint::Bow,
                sample: VerificationSample::Second,
                position,
            } if position.get() == 2_127
        ));

        let mut unstable = successful_reads();
        unstable[8] = ReadAction::Bytes(telemetry_response(HeadJoint::Bow, 2_107));
        unstable[9] = ReadAction::Bytes(telemetry_response(HeadJoint::Bow, 2_147));
        let (error, _, _) = run_startup_fault(unstable, valid_config(1)).await;
        assert!(matches!(
            error,
            HeadRuntimeError::ReadbackUnstable {
                joint: HeadJoint::Bow,
                absolute_difference_ticks: 40,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn dropping_handle_before_startup_cancels_then_disables_all() {
        let (handle, receipt, task, shared) = spawn_fake(Vec::new(), valid_config(1));
        drop(handle);
        let error = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect_err("drop cancels startup");
        assert!(matches!(
            error,
            HeadRuntimeError::Cancelled {
                cause: CancellationCause::HandleDropped,
                stage: RuntimeStage::ObserveFirst,
                joint: HeadJoint::Bow,
            }
        ));
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.termination(), &ActorTermination::HandleDropped);
        assert!(exit.torque_disable().all_writes_completed());
        let shared = shared.lock().expect("fake state");
        assert_eq!(shared.writes.len(), 4);
        assert!(shared.writes.iter().all(|write| write[5..=6] == [40, 0]));
    }

    #[tokio::test]
    async fn disable_report_attempts_all_joints_after_individual_failures() {
        let (handle, receipt, task, shared) = spawn_fake(successful_reads(), valid_config(1));
        receipt
            .wait()
            .await
            .expect("startup channel")
            .expect("startup");
        {
            let mut state = shared.lock().expect("fake state");
            let error = io::Error::from(io::ErrorKind::BrokenPipe);
            state.write_failures.insert(
                28,
                TransportFailure::from_io(TransportOperation::Write, &error, 0),
            );
            state.write_failures.insert(
                29,
                TransportFailure::from_io(TransportOperation::Write, &error, 0),
            );
        }

        let report = handle.shutdown().await.expect("shutdown report");
        assert!(!report.all_writes_completed());
        assert!(report.outcomes()[0].result().is_err());
        assert!(report.outcomes()[1].result().is_err());
        assert!(report.outcomes()[2].result().is_ok());
        assert!(report.outcomes()[3].result().is_ok());
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.torque_disable().outcomes().len(), 4);
        assert_eq!(shared.lock().expect("fake state").writes.len(), 32);
    }

    #[tokio::test]
    async fn explicit_shutdown_during_startup_returns_disable_evidence() {
        let (handle, receipt, task, _) = spawn_fake(Vec::new(), valid_config(1));
        let (shutdown, startup) = tokio::join!(handle.shutdown(), receipt.wait());
        let report = shutdown.expect("shutdown report");
        assert!(report.all_writes_completed());
        assert!(matches!(
            startup.expect("startup channel"),
            Err(HeadRuntimeError::Cancelled {
                cause: CancellationCause::RequestedShutdown,
                ..
            })
        ));
        let exit = task.join().await.expect("actor task");
        assert_eq!(exit.termination(), &ActorTermination::RequestedShutdown);
        assert_eq!(exit.torque_disable(), &report);
    }

    #[tokio::test]
    async fn queued_shutdown_during_startup_fault_receives_exact_disable_report() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let source = TransportFailure::timed_out(TransportOperation::Read, 0);
        let reads = vec![ReadAction::GatedFailure {
            entered: Arc::clone(&entered),
            release: Arc::clone(&release),
            source,
        }];
        let (handle, receipt, task, _) = spawn_fake(reads, valid_config(1));
        entered.notified().await;
        let (report_sender, report_receiver) = oneshot::channel();
        handle
            .commands
            .send(HeadCommand::Shutdown {
                response: report_sender,
            })
            .await
            .expect("queue shutdown while read is in flight");
        drop(handle);
        release.notify_one();

        let startup_error = receipt
            .wait()
            .await
            .expect("startup channel")
            .expect_err("in-flight read fault remains primary");
        assert!(matches!(
            startup_error,
            HeadRuntimeError::PositionObservation {
                source: RequestError::ResponseFrame(FrameReadError::Transport { .. }),
                ..
            }
        ));
        let report = report_receiver.await.expect("exact disable report");
        assert!(report.all_writes_completed());
        let exit = task.join().await.expect("actor task");
        assert_eq!(
            exit.termination(),
            &ActorTermination::StartupFaultWithShutdownRequested
        );
        assert_eq!(exit.torque_disable(), &report);
    }

    #[test]
    fn spawning_without_tokio_runtime_is_a_typed_error_before_serial_open() {
        let clock = TestClock::default();
        let (transport, _) = FakeTransport::new(clock.clone(), Vec::new());
        let spawn = spawn_head_actor(
            transport,
            clock,
            valid_config(1),
            PhysicalTorqueEnableConsent::explicitly_granted(),
        );
        assert!(matches!(
            spawn,
            Err(HeadActorSpawnError::NoTokioRuntime { .. })
        ));

        let start = start_serial_head_actor(
            valid_config(1),
            PhysicalTorqueEnableConsent::explicitly_granted(),
        );
        assert!(matches!(
            start,
            Err(HeadActorStartError::NoTokioRuntime { .. })
        ));
    }
}
