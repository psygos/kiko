//! Proactive, zero-only ownership of the base controller lease.
//!
//! The worker in this module is deliberately the only owner of the live KRP2
//! session.  Its public command surface contains no PWM-bearing value: callers
//! can request a newly acknowledged zero, inspect health, or consume the keeper
//! to request a terminal stop.

use std::fmt;
use std::io;
use std::sync::{
    Arc,
    mpsc::{self, Receiver, RecvTimeoutError, Sender},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use robot_command_client::{AppliedCommandReceipt, ClientConfig, DisarmReceipt, MonotonicInstant};
use robot_protocol::v2::HostCommandResult;

use super::actuation::{LiveActuationError, PhysicalZeroHoldSession};

/// Exact, copyable evidence extracted from a verified KRP2 zero receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FreshZeroEvidence {
    host_result: HostCommandResult,
    acknowledged_at: MonotonicInstant,
    known_active_through_exclusive: MonotonicInstant,
}

impl FreshZeroEvidence {
    fn from_receipt(receipt: &AppliedCommandReceipt) -> Self {
        debug_assert!(receipt.is_confirmed_zero());
        Self {
            host_result: receipt.verified_host_result(),
            acknowledged_at: receipt.acknowledged_at(),
            known_active_through_exclusive: receipt.known_active_through_exclusive(),
        }
    }

    pub const fn host_result(self) -> HostCommandResult {
        self.host_result
    }

    pub const fn acknowledged_at(self) -> MonotonicInstant {
        self.acknowledged_at
    }

    pub const fn known_active_through_exclusive(self) -> MonotonicInstant {
        self.known_active_through_exclusive
    }
}

/// A point-in-time health snapshot returned by the keeper worker.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroHoldStatus {
    latest_zero: FreshZeroEvidence,
    automatic_renewals: u64,
    next_renewal_at: MonotonicInstant,
}

impl ZeroHoldStatus {
    pub const fn latest_zero(self) -> FreshZeroEvidence {
        self.latest_zero
    }

    pub const fn automatic_renewals(self) -> u64 {
        self.automatic_renewals
    }

    pub const fn next_renewal_at(self) -> MonotonicInstant {
        self.next_renewal_at
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RenewalSchedule {
    wake_at: MonotonicInstant,
    latest_full_budget_start_exclusive: MonotonicInstant,
}

fn renewal_schedule(
    evidence: FreshZeroEvidence,
    applied_ack_timeout: Duration,
) -> Result<RenewalSchedule, ZeroHoldTimingError> {
    let acknowledged_ns = evidence.acknowledged_at().nanos_since_clock_start();
    let horizon_ns = evidence
        .known_active_through_exclusive()
        .nanos_since_clock_start();
    let ack_budget_ns = applied_ack_timeout.as_nanos();
    let latest_start_ns = horizon_ns.checked_sub(ack_budget_ns).ok_or(
        ZeroHoldTimingError::InsufficientReceiptHorizon {
            acknowledged_at: evidence.acknowledged_at(),
            known_active_through_exclusive: evidence.known_active_through_exclusive(),
            applied_ack_timeout,
        },
    )?;
    if latest_start_ns <= acknowledged_ns {
        return Err(ZeroHoldTimingError::InsufficientReceiptHorizon {
            acknowledged_at: evidence.acknowledged_at(),
            known_active_through_exclusive: evidence.known_active_through_exclusive(),
            applied_ack_timeout,
        });
    }

    // Wake halfway through the slack between acknowledgement and the last
    // instant that still preserves the complete ACK budget.  This derives the
    // cadence from the exact conservative receipt horizon, avoids a magic
    // fixed-rate timer, and retains half of the measured slack for scheduler
    // jitter.
    let slack_ns = latest_start_ns - acknowledged_ns;
    let wake_ns = acknowledged_ns
        .checked_add(slack_ns / 2)
        .ok_or(ZeroHoldTimingError::MonotonicArithmeticOverflow)?;
    Ok(RenewalSchedule {
        wake_at: MonotonicInstant::from_nanos_since_clock_start(wake_ns),
        latest_full_budget_start_exclusive: MonotonicInstant::from_nanos_since_clock_start(
            latest_start_ns,
        ),
    })
}

fn monotonic_now(origin: Instant) -> MonotonicInstant {
    MonotonicInstant::from_nanos_since_clock_start(origin.elapsed().as_nanos())
}

fn wait_until(now: MonotonicInstant, deadline: MonotonicInstant) -> Duration {
    deadline
        .checked_duration_since(now)
        .unwrap_or(Duration::ZERO)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ZeroHoldTimingError {
    InsufficientReceiptHorizon {
        acknowledged_at: MonotonicInstant,
        known_active_through_exclusive: MonotonicInstant,
        applied_ack_timeout: Duration,
    },
    RenewalStartTooLate {
        observed_at: MonotonicInstant,
        latest_full_budget_start_exclusive: MonotonicInstant,
    },
    AutomaticRenewalCounterExhausted,
    MonotonicArithmeticOverflow,
}

impl fmt::Display for ZeroHoldTimingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "zero-hold renewal timing failed: {self:?}")
    }
}

impl std::error::Error for ZeroHoldTimingError {}

#[derive(Debug)]
pub enum ZeroHoldLatchedError {
    Timing(ZeroHoldTimingError),
    Refresh(LiveActuationError),
}

impl fmt::Display for ZeroHoldLatchedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timing(source) => write!(
                formatter,
                "zero-hold keeper missed its safe timing: {source}"
            ),
            Self::Refresh(source) => write!(
                formatter,
                "zero-hold keeper could not refresh zero: {source}"
            ),
        }
    }
}

impl std::error::Error for ZeroHoldLatchedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Timing(source) => Some(source),
            Self::Refresh(source) => Some(source),
        }
    }
}

#[derive(Debug)]
pub enum ZeroHoldKeeperStartError {
    ThreadSpawn(io::Error),
    WorkerExitedBeforeStartup,
    WorkerPanickedDuringStartup,
    Acquire(LiveActuationError),
    InitialTiming {
        source: ZeroHoldTimingError,
        terminal_stop: Box<Result<DisarmReceipt, LiveActuationError>>,
    },
}

impl fmt::Display for ZeroHoldKeeperStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThreadSpawn(source) => {
                write!(formatter, "cannot spawn zero-hold keeper: {source}")
            }
            Self::WorkerExitedBeforeStartup => {
                formatter.write_str("zero-hold keeper exited before reporting startup")
            }
            Self::WorkerPanickedDuringStartup => {
                formatter.write_str("zero-hold keeper panicked during startup")
            }
            Self::Acquire(source) => write!(formatter, "zero-hold acquisition failed: {source}"),
            Self::InitialTiming {
                source,
                terminal_stop,
            } => write!(
                formatter,
                "initial zero receipt cannot support proactive renewal ({source}); terminal_stop={terminal_stop:?}"
            ),
        }
    }
}

impl std::error::Error for ZeroHoldKeeperStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ThreadSpawn(source) => Some(source),
            Self::Acquire(source) => Some(source),
            Self::InitialTiming { source, .. } => Some(source),
            Self::WorkerExitedBeforeStartup | Self::WorkerPanickedDuringStartup => None,
        }
    }
}

#[derive(Debug)]
pub enum ZeroHoldRequestError {
    KeeperLatched(Arc<ZeroHoldLatchedError>),
    WorkerUnavailable,
}

impl fmt::Display for ZeroHoldRequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeeperLatched(source) => {
                write!(formatter, "zero-hold keeper is latched: {source}")
            }
            Self::WorkerUnavailable => {
                formatter.write_str("zero-hold keeper worker is unavailable")
            }
        }
    }
}

impl std::error::Error for ZeroHoldRequestError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::KeeperLatched(source) => Some(source.as_ref()),
            Self::WorkerUnavailable => None,
        }
    }
}

#[derive(Debug)]
pub enum ZeroHoldTerminalError {
    KeeperLatched {
        source: Arc<ZeroHoldLatchedError>,
        terminal_stop: Result<DisarmReceipt, LiveActuationError>,
    },
    Disarm(LiveActuationError),
    WorkerUnavailable,
    WorkerPanicked {
        confirmed_stop: Option<DisarmReceipt>,
    },
}

impl fmt::Display for ZeroHoldTerminalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeeperLatched {
                source,
                terminal_stop,
            } => write!(
                formatter,
                "zero-hold keeper latched before shutdown ({source}); terminal_stop={terminal_stop:?}"
            ),
            Self::Disarm(source) => write!(formatter, "zero-hold terminal stop failed: {source}"),
            Self::WorkerUnavailable => {
                formatter.write_str("zero-hold worker exited without terminal stop evidence")
            }
            Self::WorkerPanicked { confirmed_stop } => write!(
                formatter,
                "zero-hold worker panicked; confirmed_stop={confirmed_stop:?}"
            ),
        }
    }
}

impl std::error::Error for ZeroHoldTerminalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::KeeperLatched { source, .. } => Some(source.as_ref()),
            Self::Disarm(source) => Some(source),
            Self::WorkerUnavailable | Self::WorkerPanicked { .. } => None,
        }
    }
}

#[derive(Debug)]
pub struct ZeroHoldTerminalReport {
    result: Result<DisarmReceipt, ZeroHoldTerminalError>,
}

impl ZeroHoldTerminalReport {
    fn new(result: Result<DisarmReceipt, ZeroHoldTerminalError>) -> Self {
        Self { result }
    }

    pub fn into_disarm_result(self) -> Result<DisarmReceipt, ZeroHoldTerminalError> {
        self.result
    }
}

enum WorkerStartup {
    Ready(FreshZeroEvidence),
    Failed(ZeroHoldKeeperStartError),
}

enum WorkerCommand {
    ForceFresh(Sender<Result<FreshZeroEvidence, Arc<ZeroHoldLatchedError>>>),
    Status(Sender<Result<ZeroHoldStatus, Arc<ZeroHoldLatchedError>>>),
    Disarm(Sender<ZeroHoldTerminalReport>),
}

struct ActiveWorker {
    session: PhysicalZeroHoldSession,
    latest_zero: FreshZeroEvidence,
    schedule: RenewalSchedule,
    automatic_renewals: u64,
}

struct LatchedWorker {
    source: Arc<ZeroHoldLatchedError>,
    terminal_stop: Option<Result<DisarmReceipt, LiveActuationError>>,
}

enum WorkerState {
    Active(Box<ActiveWorker>),
    Latched(LatchedWorker),
}

impl ActiveWorker {
    fn status(&self) -> ZeroHoldStatus {
        ZeroHoldStatus {
            latest_zero: self.latest_zero,
            automatic_renewals: self.automatic_renewals,
            next_renewal_at: self.schedule.wake_at,
        }
    }

    fn refresh(
        &mut self,
        origin: Instant,
        applied_ack_timeout: Duration,
        automatic: bool,
    ) -> Result<FreshZeroEvidence, ZeroHoldLatchedError> {
        let now = monotonic_now(origin);
        if now >= self.schedule.latest_full_budget_start_exclusive {
            return Err(ZeroHoldLatchedError::Timing(
                ZeroHoldTimingError::RenewalStartTooLate {
                    observed_at: now,
                    latest_full_budget_start_exclusive: self
                        .schedule
                        .latest_full_budget_start_exclusive,
                },
            ));
        }
        let receipt = self
            .session
            .refresh_zero()
            .map_err(ZeroHoldLatchedError::Refresh)?;
        let evidence = FreshZeroEvidence::from_receipt(&receipt);
        let schedule = renewal_schedule(evidence, applied_ack_timeout)
            .map_err(ZeroHoldLatchedError::Timing)?;
        if automatic {
            self.automatic_renewals =
                self.automatic_renewals
                    .checked_add(1)
                    .ok_or(ZeroHoldLatchedError::Timing(
                        ZeroHoldTimingError::AutomaticRenewalCounterExhausted,
                    ))?;
        }
        self.latest_zero = evidence;
        self.schedule = schedule;
        Ok(evidence)
    }

    fn latch(self, source: ZeroHoldLatchedError) -> LatchedWorker {
        let source = Arc::new(source);
        let terminal_stop = if self.session.is_consumed() {
            None
        } else {
            Some(self.session.disarm())
        };
        LatchedWorker {
            source,
            terminal_stop,
        }
    }
}

fn handle_active_command(
    mut active: ActiveWorker,
    command: WorkerCommand,
    origin: Instant,
    applied_ack_timeout: Duration,
    retained_terminal: &mut Option<ZeroHoldTerminalReport>,
) -> Option<WorkerState> {
    match command {
        WorkerCommand::ForceFresh(reply) => {
            match active.refresh(origin, applied_ack_timeout, false) {
                Ok(evidence) => {
                    if reply.send(Ok(evidence)).is_ok() {
                        Some(WorkerState::Active(Box::new(active)))
                    } else {
                        *retained_terminal = Some(ZeroHoldTerminalReport::new(
                            active
                                .session
                                .disarm()
                                .map_err(ZeroHoldTerminalError::Disarm),
                        ));
                        None
                    }
                }
                Err(source) => {
                    let latched = active.latch(source);
                    if reply.send(Err(Arc::clone(&latched.source))).is_ok() {
                        Some(WorkerState::Latched(latched))
                    } else {
                        *retained_terminal = Some(latched.into_terminal_report());
                        None
                    }
                }
            }
        }
        WorkerCommand::Status(reply) => {
            if monotonic_now(origin) >= active.schedule.wake_at
                && let Err(source) = active.refresh(origin, applied_ack_timeout, true)
            {
                let latched = active.latch(source);
                return if reply.send(Err(Arc::clone(&latched.source))).is_ok() {
                    Some(WorkerState::Latched(latched))
                } else {
                    *retained_terminal = Some(latched.into_terminal_report());
                    None
                };
            }
            if reply.send(Ok(active.status())).is_ok() {
                Some(WorkerState::Active(Box::new(active)))
            } else {
                *retained_terminal = Some(ZeroHoldTerminalReport::new(
                    active
                        .session
                        .disarm()
                        .map_err(ZeroHoldTerminalError::Disarm),
                ));
                None
            }
        }
        WorkerCommand::Disarm(reply) => {
            let result = active
                .session
                .disarm()
                .map_err(ZeroHoldTerminalError::Disarm);
            let report = ZeroHoldTerminalReport::new(result);
            if let Err(undelivered) = reply.send(report) {
                *retained_terminal = Some(undelivered.0);
            }
            None
        }
    }
}

impl LatchedWorker {
    fn into_terminal_report(self) -> ZeroHoldTerminalReport {
        let terminal_stop = self
            .terminal_stop
            .unwrap_or(Err(LiveActuationError::SessionConsumed));
        ZeroHoldTerminalReport::new(Err(ZeroHoldTerminalError::KeeperLatched {
            source: self.source,
            terminal_stop,
        }))
    }
}

fn handle_latched_command(
    latched: LatchedWorker,
    command: WorkerCommand,
    retained_terminal: &mut Option<ZeroHoldTerminalReport>,
) -> Option<WorkerState> {
    match command {
        WorkerCommand::ForceFresh(reply) => {
            if reply.send(Err(Arc::clone(&latched.source))).is_ok() {
                Some(WorkerState::Latched(latched))
            } else {
                *retained_terminal = Some(latched.into_terminal_report());
                None
            }
        }
        WorkerCommand::Status(reply) => {
            if reply.send(Err(Arc::clone(&latched.source))).is_ok() {
                Some(WorkerState::Latched(latched))
            } else {
                *retained_terminal = Some(latched.into_terminal_report());
                None
            }
        }
        WorkerCommand::Disarm(reply) => {
            let report = latched.into_terminal_report();
            if let Err(undelivered) = reply.send(report) {
                *retained_terminal = Some(undelivered.0);
            }
            None
        }
    }
}

fn worker_main(
    config: ClientConfig,
    origin: Instant,
    startup: Sender<WorkerStartup>,
    commands: Receiver<WorkerCommand>,
) -> Option<ZeroHoldTerminalReport> {
    let applied_ack_timeout = config.applied_ack_timeout().as_duration();
    let (session, receipt) = match PhysicalZeroHoldSession::acquire_zero_only(config, origin) {
        Ok(acquired) => acquired,
        Err(source) => {
            let _ = startup.send(WorkerStartup::Failed(ZeroHoldKeeperStartError::Acquire(
                source,
            )));
            return None;
        }
    };
    let evidence = FreshZeroEvidence::from_receipt(&receipt);
    let schedule = match renewal_schedule(evidence, applied_ack_timeout) {
        Ok(schedule) => schedule,
        Err(source) => {
            let terminal_stop = session.disarm();
            let _ = startup.send(WorkerStartup::Failed(
                ZeroHoldKeeperStartError::InitialTiming {
                    source,
                    terminal_stop: Box::new(terminal_stop),
                },
            ));
            return None;
        }
    };
    let mut state = WorkerState::Active(Box::new(ActiveWorker {
        session,
        latest_zero: evidence,
        schedule,
        automatic_renewals: 0,
    }));
    if startup.send(WorkerStartup::Ready(evidence)).is_err() {
        if let WorkerState::Active(active) = state {
            let active = *active;
            return Some(ZeroHoldTerminalReport::new(
                active
                    .session
                    .disarm()
                    .map_err(ZeroHoldTerminalError::Disarm),
            ));
        }
        return None;
    }

    let mut retained_terminal = None;

    loop {
        state = match state {
            WorkerState::Active(active) => {
                let wait = wait_until(monotonic_now(origin), active.schedule.wake_at);
                match commands.recv_timeout(wait) {
                    Ok(command) => {
                        match handle_active_command(
                            *active,
                            command,
                            origin,
                            applied_ack_timeout,
                            &mut retained_terminal,
                        ) {
                            Some(state) => state,
                            None => return retained_terminal,
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        let mut active = *active;
                        match active.refresh(origin, applied_ack_timeout, true) {
                            Ok(_) => WorkerState::Active(Box::new(active)),
                            Err(source) => WorkerState::Latched(active.latch(source)),
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        let active = *active;
                        return Some(ZeroHoldTerminalReport::new(
                            active
                                .session
                                .disarm()
                                .map_err(ZeroHoldTerminalError::Disarm),
                        ));
                    }
                }
            }
            WorkerState::Latched(latched) => match commands.recv() {
                Ok(command) => {
                    match handle_latched_command(latched, command, &mut retained_terminal) {
                        Some(state) => state,
                        None => return retained_terminal,
                    }
                }
                Err(_) => return Some(latched.into_terminal_report()),
            },
        };
    }
}

#[must_use = "the keeper must be explicitly disarmed and its terminal report checked"]
pub struct ZeroHoldKeeper {
    commands: Option<Sender<WorkerCommand>>,
    worker: Option<JoinHandle<Option<ZeroHoldTerminalReport>>>,
}

impl ZeroHoldKeeper {
    pub fn start(
        config: ClientConfig,
        clock_origin: Instant,
    ) -> Result<(Self, FreshZeroEvidence), ZeroHoldKeeperStartError> {
        let (command_sender, command_receiver) = mpsc::channel();
        let (startup_sender, startup_receiver) = mpsc::channel();
        let worker = thread::Builder::new()
            .name("kiko-zero-hold".to_owned())
            .spawn(move || worker_main(config, clock_origin, startup_sender, command_receiver))
            .map_err(ZeroHoldKeeperStartError::ThreadSpawn)?;

        match startup_receiver.recv() {
            Ok(WorkerStartup::Ready(evidence)) => Ok((
                Self {
                    commands: Some(command_sender),
                    worker: Some(worker),
                },
                evidence,
            )),
            Ok(WorkerStartup::Failed(source)) => {
                let panicked = worker.join().is_err();
                if panicked {
                    Err(ZeroHoldKeeperStartError::WorkerPanickedDuringStartup)
                } else {
                    Err(source)
                }
            }
            Err(_) => {
                if worker.join().is_err() {
                    Err(ZeroHoldKeeperStartError::WorkerPanickedDuringStartup)
                } else {
                    Err(ZeroHoldKeeperStartError::WorkerExitedBeforeStartup)
                }
            }
        }
    }

    pub fn force_fresh_zero(&mut self) -> Result<FreshZeroEvidence, ZeroHoldRequestError> {
        let (reply_sender, reply_receiver) = mpsc::channel();
        self.send(WorkerCommand::ForceFresh(reply_sender))?;
        reply_receiver
            .recv()
            .map_err(|_| ZeroHoldRequestError::WorkerUnavailable)?
            .map_err(ZeroHoldRequestError::KeeperLatched)
    }

    pub fn status(&mut self) -> Result<ZeroHoldStatus, ZeroHoldRequestError> {
        let (reply_sender, reply_receiver) = mpsc::channel();
        self.send(WorkerCommand::Status(reply_sender))?;
        reply_receiver
            .recv()
            .map_err(|_| ZeroHoldRequestError::WorkerUnavailable)?
            .map_err(ZeroHoldRequestError::KeeperLatched)
    }

    fn send(&self, command: WorkerCommand) -> Result<(), ZeroHoldRequestError> {
        self.commands
            .as_ref()
            .ok_or(ZeroHoldRequestError::WorkerUnavailable)?
            .send(command)
            .map_err(|_| ZeroHoldRequestError::WorkerUnavailable)
    }

    pub fn disarm(mut self) -> ZeroHoldTerminalReport {
        let (reply_sender, reply_receiver) = mpsc::channel();
        let send_result = self
            .commands
            .take()
            .ok_or(ZeroHoldTerminalError::WorkerUnavailable)
            .and_then(|sender| {
                sender
                    .send(WorkerCommand::Disarm(reply_sender))
                    .map_err(|_| ZeroHoldTerminalError::WorkerUnavailable)
            });
        let mut report = match send_result {
            Ok(()) => reply_receiver.recv().unwrap_or_else(|_| {
                ZeroHoldTerminalReport::new(Err(ZeroHoldTerminalError::WorkerUnavailable))
            }),
            Err(source) => ZeroHoldTerminalReport::new(Err(source)),
        };

        if let Some(worker) = self.worker.take() {
            match worker.join() {
                Ok(Some(retained)) => report = retained,
                Ok(None) => {}
                Err(_) => {
                    let confirmed_stop = report.result.ok();
                    report =
                        ZeroHoldTerminalReport::new(Err(ZeroHoldTerminalError::WorkerPanicked {
                            confirmed_stop,
                        }));
                }
            }
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use robot_protocol::ControllerUptimeMsWrapping;
    use robot_protocol::v2::{
        ControlEpoch, ControllerBootId, ControllerDeadlineMsWrapping, ControllerFaults,
        ControllerUid, HostCommandResultCode, OutputState, RemainingLeaseMs, TimerPwm,
        V2CommandSequence,
    };

    fn evidence(acknowledged_ns: u128, horizon_ns: u128) -> FreshZeroEvidence {
        FreshZeroEvidence {
            host_result: HostCommandResult {
                controller_uid: ControllerUid::try_new([1; 12]).expect("UID fixture"),
                boot_id: ControllerBootId::try_new(2).expect("boot fixture"),
                control_epoch: ControlEpoch::try_new(3).expect("epoch fixture"),
                sequence: V2CommandSequence::FIRST,
                result: HostCommandResultCode::AppliedNew,
                requested_timer_pwm: TimerPwm::ZERO,
                controller_timer_pwm: TimerPwm::ZERO,
                output_state: OutputState::ZeroPwm,
                controller_applied_at: ControllerUptimeMsWrapping::new(10),
                controller_expires_at: ControllerDeadlineMsWrapping::new(260),
                remaining_lease: RemainingLeaseMs::try_new(250).expect("lease fixture"),
                faults: ControllerFaults::NONE,
            },
            acknowledged_at: MonotonicInstant::from_nanos_since_clock_start(acknowledged_ns),
            known_active_through_exclusive: MonotonicInstant::from_nanos_since_clock_start(
                horizon_ns,
            ),
        }
    }

    fn latched_worker() -> LatchedWorker {
        LatchedWorker {
            source: Arc::new(ZeroHoldLatchedError::Timing(
                ZeroHoldTimingError::AutomaticRenewalCounterExhausted,
            )),
            terminal_stop: Some(Err(LiveActuationError::SessionConsumed)),
        }
    }

    #[test]
    fn renewal_is_midpoint_of_exact_receipt_slack() {
        let schedule =
            renewal_schedule(evidence(10_000_000, 260_000_000), Duration::from_millis(40))
                .expect("250 ms evidence supports a 40 ms ACK budget");
        assert_eq!(
            schedule
                .latest_full_budget_start_exclusive
                .nanos_since_clock_start(),
            220_000_000
        );
        assert_eq!(schedule.wake_at.nanos_since_clock_start(), 115_000_000);
    }

    #[test]
    fn exact_ack_budget_boundary_is_rejected() {
        let source = renewal_schedule(evidence(20_000_000, 60_000_000), Duration::from_millis(40))
            .expect_err("no positive scheduling slack exists");
        assert!(matches!(
            source,
            ZeroHoldTimingError::InsufficientReceiptHorizon { .. }
        ));
    }

    #[test]
    fn horizon_shorter_than_ack_budget_cannot_underflow() {
        let source = renewal_schedule(evidence(1, 39_999_999), Duration::from_millis(40))
            .expect_err("short horizon is invalid");
        assert!(matches!(
            source,
            ZeroHoldTimingError::InsufficientReceiptHorizon { .. }
        ));
    }

    #[test]
    fn elapsed_wait_saturates_at_zero_without_wrapping() {
        assert_eq!(
            wait_until(
                MonotonicInstant::from_nanos_since_clock_start(11),
                MonotonicInstant::from_nanos_since_clock_start(10),
            ),
            Duration::ZERO
        );
    }

    #[test]
    fn dropped_latched_force_reply_retains_terminal_report() {
        let (reply, receiver) = mpsc::channel();
        drop(receiver);
        let mut retained = None;
        let next = handle_latched_command(
            latched_worker(),
            WorkerCommand::ForceFresh(reply),
            &mut retained,
        );
        assert!(next.is_none());
        assert!(matches!(
            retained
                .expect("undelivered terminal report is retained")
                .into_disarm_result(),
            Err(ZeroHoldTerminalError::KeeperLatched { .. })
        ));
    }

    #[test]
    fn dropped_latched_status_reply_retains_terminal_report() {
        let (reply, receiver) = mpsc::channel();
        drop(receiver);
        let mut retained = None;
        let next = handle_latched_command(
            latched_worker(),
            WorkerCommand::Status(reply),
            &mut retained,
        );
        assert!(next.is_none());
        assert!(matches!(
            retained
                .expect("undelivered terminal report is retained")
                .into_disarm_result(),
            Err(ZeroHoldTerminalError::KeeperLatched { .. })
        ));
    }

    #[test]
    fn latched_status_reply_reaches_the_requester() {
        let (reply, receiver) = mpsc::channel();
        let mut retained = None;
        let next = handle_latched_command(
            latched_worker(),
            WorkerCommand::Status(reply),
            &mut retained,
        );
        assert!(matches!(next, Some(WorkerState::Latched(_))));
        assert!(retained.is_none());
        assert!(receiver.recv().expect("worker replied").is_err());
    }

    #[test]
    fn disarm_recovers_terminal_report_returned_by_exited_worker() {
        let (commands, receiver) = mpsc::channel();
        drop(receiver);
        let worker = thread::spawn(|| Some(latched_worker().into_terminal_report()));
        let keeper = ZeroHoldKeeper {
            commands: Some(commands),
            worker: Some(worker),
        };

        assert!(matches!(
            keeper.disarm().into_disarm_result(),
            Err(ZeroHoldTerminalError::KeeperLatched { .. })
        ));
    }
}
