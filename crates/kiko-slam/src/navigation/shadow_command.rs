//! A transport-free command session for replay-bound navigation shadow mode.
//!
//! The session deliberately reuses the canonical STM32 command-domain types,
//! including signed left/right PWM, modular command sequence, and bounded
//! lease duration. It owns no socket, serial handle, transport trait, callback,
//! or byte encoder. Recording a nonzero request therefore cannot emit a motor
//! packet; the value is retained only as shadow evidence.

use std::collections::VecDeque;
use std::num::{NonZeroU64, NonZeroUsize};

use robot_protocol::{
    CommandLeaseError, CommandLeaseMs, CommandSequence, LeasedPwmCommand, PwmPercent,
    PwmPercentError, RobotCommand,
};

use crate::HostMonotonicTimestamp;

/// Hard allocation bound for retained shadow evidence (about two hours at
/// 10 Hz when configured to the maximum).
pub const MAX_SHADOW_COMMAND_RECORDS: usize = 65_536;

/// Weakly typed command-session configuration parsed exactly once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowCommandConfigDto {
    pub lease_ms: u16,
    pub retained_records: usize,
    pub initial_sequence: u32,
}

/// Bounded command-session configuration with the shared protocol domain
/// already established.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowCommandConfig {
    lease: CommandLeaseMs,
    retained_records: NonZeroUsize,
    initial_sequence: CommandSequence,
}

impl ShadowCommandConfig {
    pub fn parse(dto: ShadowCommandConfigDto) -> Result<Self, ShadowCommandConfigError> {
        let lease = CommandLeaseMs::try_new(dto.lease_ms)
            .map_err(ShadowCommandConfigError::InvalidLease)?;
        let retained_records = NonZeroUsize::new(dto.retained_records)
            .ok_or(ShadowCommandConfigError::ZeroRetainedRecords)?;
        if retained_records.get() > MAX_SHADOW_COMMAND_RECORDS {
            return Err(ShadowCommandConfigError::TooManyRetainedRecords {
                actual: retained_records.get(),
                maximum: MAX_SHADOW_COMMAND_RECORDS,
            });
        }
        Ok(Self {
            lease,
            retained_records,
            initial_sequence: CommandSequence::new(dto.initial_sequence),
        })
    }

    pub fn lease(self) -> CommandLeaseMs {
        self.lease
    }

    pub fn retained_records(self) -> usize {
        self.retained_records.get()
    }

    pub fn initial_sequence(self) -> CommandSequence {
        self.initial_sequence
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowCommandConfigError {
    InvalidLease(CommandLeaseError),
    ZeroRetainedRecords,
    TooManyRetainedRecords { actual: usize, maximum: usize },
}

impl std::fmt::Display for ShadowCommandConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLease(source) => write!(f, "invalid shadow command lease: {source}"),
            Self::ZeroRetainedRecords => {
                f.write_str("shadow command retained-record capacity must be nonzero")
            }
            Self::TooManyRetainedRecords { actual, maximum } => write!(
                f,
                "shadow command retained-record capacity {actual} exceeds the maximum {maximum}"
            ),
        }
    }
}

impl std::error::Error for ShadowCommandConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidLease(source) => Some(source),
            Self::ZeroRetainedRecords | Self::TooManyRetainedRecords { .. } => None,
        }
    }
}

/// A left/right PWM request already parsed into the canonical comms domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowPwmPair {
    left: PwmPercent,
    right: PwmPercent,
}

impl ShadowPwmPair {
    pub const STOP: Self = Self {
        left: PwmPercent::ZERO,
        right: PwmPercent::ZERO,
    };

    pub fn try_new(left: i8, right: i8) -> Result<Self, ShadowPwmPairError> {
        Ok(Self {
            left: PwmPercent::try_new(left).map_err(ShadowPwmPairError::Left)?,
            right: PwmPercent::try_new(right).map_err(ShadowPwmPairError::Right)?,
        })
    }

    pub const fn from_validated(left: PwmPercent, right: PwmPercent) -> Self {
        Self { left, right }
    }

    pub const fn left(self) -> PwmPercent {
        self.left
    }

    pub const fn right(self) -> PwmPercent {
        self.right
    }

    pub const fn is_stop(self) -> bool {
        self.left.get() == 0 && self.right.get() == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowPwmPairError {
    Left(PwmPercentError),
    Right(PwmPercentError),
}

impl std::fmt::Display for ShadowPwmPairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Left(source) => write!(f, "invalid left shadow PWM: {source}"),
            Self::Right(source) => write!(f, "invalid right shadow PWM: {source}"),
        }
    }
}

impl std::error::Error for ShadowPwmPairError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Left(source) | Self::Right(source) => Some(source),
        }
    }
}

/// Monotonic identity for every decision admitted to one shadow session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShadowDecisionId(NonZeroU64);

impl ShadowDecisionId {
    const FIRST: Self = Self(NonZeroU64::MIN);

    pub fn as_u64(self) -> u64 {
        self.0.get()
    }

    fn checked_successor(self) -> Option<Self> {
        self.as_u64()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .map(Self)
    }
}

/// Why the session recorded the exact PWM pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowCommandDisposition {
    /// A fresh, safety-approved controller result was retained for inspection.
    ControllerRequest,
    /// Fail-closed supervision required an explicit zero-PWM record.
    FailClosedStop,
}

/// A zero-only counter. There is intentionally no constructor for a nonzero
/// value and no mutable counter inside [`ShadowCommandSession`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MotorPacketsSent;

impl MotorPacketsSent {
    pub const ZERO: Self = Self;

    pub const fn get(self) -> u64 {
        0
    }
}

/// One typed leased command retained as evidence, never transmitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShadowCommandRecord {
    decision_id: ShadowDecisionId,
    recorded_at: HostMonotonicTimestamp,
    disposition: ShadowCommandDisposition,
    command: RobotCommand,
}

impl ShadowCommandRecord {
    pub fn decision_id(self) -> ShadowDecisionId {
        self.decision_id
    }

    pub fn recorded_at(self) -> HostMonotonicTimestamp {
        self.recorded_at
    }

    pub fn disposition(self) -> ShadowCommandDisposition {
        self.disposition
    }

    pub fn command(self) -> RobotCommand {
        self.command
    }

    pub fn pwm(self) -> ShadowPwmPair {
        ShadowPwmPair::from_validated(
            self.command.left_pwm_percent(),
            self.command.right_pwm_percent(),
        )
    }

    pub fn motor_packets_sent(self) -> MotorPacketsSent {
        MotorPacketsSent::ZERO
    }
}

/// A bounded, transport-free leased-command recorder.
#[derive(Debug)]
pub struct ShadowCommandSession {
    config: ShadowCommandConfig,
    next_sequence: CommandSequence,
    next_decision_id: Option<ShadowDecisionId>,
    last_recorded_at: Option<HostMonotonicTimestamp>,
    records: VecDeque<ShadowCommandRecord>,
}

impl ShadowCommandSession {
    pub fn new(config: ShadowCommandConfig) -> Self {
        Self {
            config,
            next_sequence: config.initial_sequence,
            next_decision_id: Some(ShadowDecisionId::FIRST),
            last_recorded_at: None,
            records: VecDeque::with_capacity(config.retained_records.get()),
        }
    }

    pub fn config(&self) -> ShadowCommandConfig {
        self.config
    }

    /// Record a fresh controller request without invoking any I/O path.
    pub fn record_controller_request(
        &mut self,
        recorded_at: HostMonotonicTimestamp,
        pwm: ShadowPwmPair,
    ) -> Result<ShadowCommandRecord, ShadowCommandError> {
        self.record(
            recorded_at,
            pwm,
            ShadowCommandDisposition::ControllerRequest,
        )
    }

    /// Record the only command admitted for a fail-closed decision: zero PWM.
    pub fn record_fail_closed_stop(
        &mut self,
        recorded_at: HostMonotonicTimestamp,
    ) -> Result<ShadowCommandRecord, ShadowCommandError> {
        self.record(
            recorded_at,
            ShadowPwmPair::STOP,
            ShadowCommandDisposition::FailClosedStop,
        )
    }

    fn record(
        &mut self,
        recorded_at: HostMonotonicTimestamp,
        pwm: ShadowPwmPair,
        disposition: ShadowCommandDisposition,
    ) -> Result<ShadowCommandRecord, ShadowCommandError> {
        if let Some(previous) = self.last_recorded_at
            && recorded_at < previous
        {
            return Err(ShadowCommandError::HostClockRegression {
                previous,
                current: recorded_at,
            });
        }
        let decision_id = self
            .next_decision_id
            .ok_or(ShadowCommandError::DecisionIdExhausted)?;
        let command = RobotCommand::from_leased_pwm(
            LeasedPwmCommand::from_validated(pwm.left, pwm.right, self.config.lease),
            self.next_sequence,
        );
        let record = ShadowCommandRecord {
            decision_id,
            recorded_at,
            disposition,
            command,
        };

        // Every check above is complete. Commit the bounded state atomically.
        if self.records.len() == self.config.retained_records.get() {
            self.records.pop_front();
        }
        self.records.push_back(record);
        self.last_recorded_at = Some(recorded_at);
        self.next_sequence = CommandSequence::new(self.next_sequence.get().wrapping_add(1));
        self.next_decision_id = decision_id.checked_successor();
        Ok(record)
    }

    pub fn latest(&self) -> Option<ShadowCommandRecord> {
        self.records.back().copied()
    }

    pub fn retained(&self) -> impl ExactSizeIterator<Item = &ShadowCommandRecord> {
        self.records.iter()
    }

    pub fn retained_len(&self) -> usize {
        self.records.len()
    }

    pub fn last_recorded_at(&self) -> Option<HostMonotonicTimestamp> {
        self.last_recorded_at
    }

    pub fn next_sequence(&self) -> CommandSequence {
        self.next_sequence
    }

    pub fn motor_packets_sent(&self) -> MotorPacketsSent {
        MotorPacketsSent::ZERO
    }

    #[cfg(test)]
    fn set_next_decision_id_for_test(&mut self, id: Option<ShadowDecisionId>) {
        self.next_decision_id = id;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadowCommandError {
    HostClockRegression {
        previous: HostMonotonicTimestamp,
        current: HostMonotonicTimestamp,
    },
    DecisionIdExhausted,
}

impl std::fmt::Display for ShadowCommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HostClockRegression { previous, current } => write!(
                f,
                "shadow command host time regressed from {} ns to {} ns",
                previous.as_nanos(),
                current.as_nanos()
            ),
            Self::DecisionIdExhausted => {
                f.write_str("shadow command decision identifier domain is exhausted")
            }
        }
    }
}

impl std::error::Error for ShadowCommandError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(capacity: usize, initial_sequence: u32) -> ShadowCommandConfig {
        ShadowCommandConfig::parse(ShadowCommandConfigDto {
            lease_ms: 150,
            retained_records: capacity,
            initial_sequence,
        })
        .expect("valid explicit shadow config")
    }

    #[test]
    fn config_reuses_exact_protocol_lease_domain_and_rejects_zero_history() {
        assert!(matches!(
            ShadowCommandConfig::parse(ShadowCommandConfigDto {
                lease_ms: 0,
                retained_records: 1,
                initial_sequence: 0,
            }),
            Err(ShadowCommandConfigError::InvalidLease(
                CommandLeaseError::Zero
            ))
        ));
        assert!(matches!(
            ShadowCommandConfig::parse(ShadowCommandConfigDto {
                lease_ms: u16::MAX,
                retained_records: 1,
                initial_sequence: 0,
            }),
            Err(ShadowCommandConfigError::InvalidLease(
                CommandLeaseError::AboveMaximum { .. }
            ))
        ));
        assert_eq!(
            ShadowCommandConfig::parse(ShadowCommandConfigDto {
                lease_ms: 1,
                retained_records: 0,
                initial_sequence: 0,
            }),
            Err(ShadowCommandConfigError::ZeroRetainedRecords)
        );
        assert_eq!(
            ShadowCommandConfig::parse(ShadowCommandConfigDto {
                lease_ms: 1,
                retained_records: MAX_SHADOW_COMMAND_RECORDS + 1,
                initial_sequence: 0,
            }),
            Err(ShadowCommandConfigError::TooManyRetainedRecords {
                actual: MAX_SHADOW_COMMAND_RECORDS + 1,
                maximum: MAX_SHADOW_COMMAND_RECORDS,
            })
        );
    }

    #[test]
    fn pwm_pair_parses_each_channel_without_clamping() {
        let pair = ShadowPwmPair::try_new(-100, 100).expect("inclusive protocol bounds");
        assert_eq!(pair.left().get(), -100);
        assert_eq!(pair.right().get(), 100);
        assert!(matches!(
            ShadowPwmPair::try_new(-101, 0),
            Err(ShadowPwmPairError::Left(source)) if source.value() == -101
        ));
        assert!(matches!(
            ShadowPwmPair::try_new(0, 101),
            Err(ShadowPwmPairError::Right(source)) if source.value() == 101
        ));
    }

    #[test]
    fn nonzero_request_is_typed_and_retained_but_sends_zero_packets() {
        let mut session = ShadowCommandSession::new(config(4, 7));
        let record = session
            .record_controller_request(
                HostMonotonicTimestamp::from_nanos(10),
                ShadowPwmPair::try_new(-25, 40).expect("valid pair"),
            )
            .expect("first shadow decision");

        assert_eq!(record.decision_id().as_u64(), 1);
        assert_eq!(record.command().sequence().get(), 7);
        assert_eq!(record.command().left_pwm_percent().get(), -25);
        assert_eq!(record.command().right_pwm_percent().get(), 40);
        assert_eq!(record.command().lease_ms().get(), 150);
        assert_eq!(
            record.disposition(),
            ShadowCommandDisposition::ControllerRequest
        );
        assert_eq!(record.motor_packets_sent().get(), 0);
        assert_eq!(session.motor_packets_sent().get(), 0);
        assert_eq!(session.latest(), Some(record));
    }

    #[test]
    fn fail_closed_record_is_always_an_explicit_leased_stop() {
        let mut session = ShadowCommandSession::new(config(2, 0));
        let record = session
            .record_fail_closed_stop(HostMonotonicTimestamp::from_nanos(1))
            .expect("stop record");
        assert!(record.pwm().is_stop());
        assert!(record.command().leased_pwm().is_stop());
        assert_eq!(record.command().lease_ms().get(), 150);
        assert_eq!(
            record.disposition(),
            ShadowCommandDisposition::FailClosedStop
        );
        assert_eq!(session.motor_packets_sent().get(), 0);
    }

    #[test]
    fn bounded_history_evicts_only_the_oldest_evidence() {
        let mut session = ShadowCommandSession::new(config(2, 20));
        for time in 1..=3 {
            session
                .record_controller_request(
                    HostMonotonicTimestamp::from_nanos(time),
                    ShadowPwmPair::try_new(time as i8, -(time as i8)).expect("valid pair"),
                )
                .expect("ordered record");
        }
        let retained = session.retained().copied().collect::<Vec<_>>();
        assert_eq!(retained.len(), 2);
        assert_eq!(retained[0].decision_id().as_u64(), 2);
        assert_eq!(retained[1].decision_id().as_u64(), 3);
        assert_eq!(retained[1].command().sequence().get(), 22);
        assert_eq!(session.motor_packets_sent().get(), 0);
    }

    #[test]
    fn host_clock_regression_rejects_without_mutating_session() {
        let mut session = ShadowCommandSession::new(config(2, 9));
        let first = session
            .record_fail_closed_stop(HostMonotonicTimestamp::from_nanos(10))
            .expect("first stop");
        let before_sequence = session.next_sequence();
        let error = session
            .record_controller_request(
                HostMonotonicTimestamp::from_nanos(9),
                ShadowPwmPair::try_new(1, 1).expect("pair"),
            )
            .expect_err("regression must reject");
        assert_eq!(
            error,
            ShadowCommandError::HostClockRegression {
                previous: HostMonotonicTimestamp::from_nanos(10),
                current: HostMonotonicTimestamp::from_nanos(9),
            }
        );
        assert_eq!(session.retained_len(), 1);
        assert_eq!(session.latest(), Some(first));
        assert_eq!(session.next_sequence(), before_sequence);
        assert_eq!(session.motor_packets_sent().get(), 0);
    }

    #[test]
    fn equal_host_timestamps_are_admitted_and_sequence_wrap_is_canonical() {
        let mut session = ShadowCommandSession::new(config(2, u32::MAX));
        let first = session
            .record_fail_closed_stop(HostMonotonicTimestamp::from_nanos(10))
            .expect("maximum sequence");
        let second = session
            .record_fail_closed_stop(HostMonotonicTimestamp::from_nanos(10))
            .expect("same clock tick and wrapped sequence");
        assert_eq!(first.command().sequence().get(), u32::MAX);
        assert_eq!(second.command().sequence().get(), 0);
        assert_eq!(session.next_sequence().get(), 1);
    }

    #[test]
    fn final_decision_id_is_issued_once_then_exhaustion_is_transactional() {
        let mut session = ShadowCommandSession::new(config(2, 5));
        let maximum =
            ShadowDecisionId(NonZeroU64::new(u64::MAX).expect("maximum integer remains nonzero"));
        session.set_next_decision_id_for_test(Some(maximum));
        let final_record = session
            .record_fail_closed_stop(HostMonotonicTimestamp::from_nanos(1))
            .expect("final ID is usable once");
        assert_eq!(final_record.decision_id(), maximum);
        let sequence_after_final = session.next_sequence();

        assert_eq!(
            session.record_fail_closed_stop(HostMonotonicTimestamp::from_nanos(2)),
            Err(ShadowCommandError::DecisionIdExhausted)
        );
        assert_eq!(session.retained_len(), 1);
        assert_eq!(session.latest(), Some(final_record));
        assert_eq!(session.next_sequence(), sequence_after_final);
        assert_eq!(session.last_recorded_at(), Some(final_record.recorded_at()));
        assert_eq!(session.motor_packets_sent().get(), 0);
    }
}
