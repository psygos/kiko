//! Identity-only commissioning for one exact KEP2 eye serial device.
//!
//! The probe sends exactly one freshly challenged [`Message::IdentityQuery`].
//! It never acquires control, applies an expression, or sends a release frame.

use std::fmt;
use std::time::Duration;

use kiko_expression_core::MonotonicTimestamp;
use kiko_eye_protocol::{
    EncodeError, HandshakeNonce, IdentityReport, MAX_ENCODED_FRAME_BYTES, Message, encode,
};

use crate::config::{BaudRate, ConfigParseError, DeviceIdentity, OperationTimeout};
use crate::framing::{FrameReadError, FrameReader};
use crate::transport::{
    AsyncByteTransport, ClockError, MonotonicClock, SerialConfigurationEvidence, SerialOpenError,
    SerialTransport, TokioClock, TransportFailure, TransportOperation,
};

const KEP2_SERIAL_BAUD_BPS: u32 = 115_200;
const EMPTY_DELIMITER_BUDGET: u8 = 2;
const MAX_CHALLENGE_ATTEMPTS: u8 = 8;

/// Weak command-line values for one identity-only exchange.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdentityProbeConfigInput {
    pub device_path: String,
    /// Exclusive timeout applied independently to the one query write/flush
    /// and the one response read.
    pub operation_timeout_ms: u64,
}

/// Parsed identity-only commissioning policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdentityProbeConfig {
    device: DeviceIdentity,
    baud_rate: BaudRate,
    operation_timeout: OperationTimeout,
    empty_delimiter_budget: u8,
}

impl IdentityProbeConfig {
    /// Parse an exact Linux by-id or macOS callout path and a bounded timeout.
    pub fn parse(input: IdentityProbeConfigInput) -> Result<Self, IdentityProbeConfigError> {
        let device = DeviceIdentity::parse(input.device_path)
            .map_err(IdentityProbeConfigError::RuntimeConfig)?;
        if let Some((index, byte)) = device
            .path()
            .bytes()
            .enumerate()
            .find(|(_, byte)| !byte.is_ascii_graphic())
        {
            return Err(IdentityProbeConfigError::NonGraphicPathByte { index, byte });
        }
        let baud_rate = BaudRate::parse(KEP2_SERIAL_BAUD_BPS)
            .expect("the canonical KEP2 baud rate is inside the parsed runtime range");
        let operation_timeout = OperationTimeout::parse(
            "identity_probe_operation_timeout_ms",
            input.operation_timeout_ms,
        )
        .map_err(IdentityProbeConfigError::RuntimeConfig)?;
        Ok(Self {
            device,
            baud_rate,
            operation_timeout,
            empty_delimiter_budget: EMPTY_DELIMITER_BUDGET,
        })
    }

    pub const fn device(&self) -> &DeviceIdentity {
        &self.device
    }

    pub const fn baud_rate(&self) -> BaudRate {
        self.baud_rate
    }

    pub const fn operation_timeout(&self) -> OperationTimeout {
        self.operation_timeout
    }

    pub const fn empty_delimiter_budget(&self) -> u8 {
        self.empty_delimiter_budget
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IdentityProbeConfigError {
    RuntimeConfig(ConfigParseError),
    NonGraphicPathByte { index: usize, byte: u8 },
}

impl fmt::Display for IdentityProbeConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid KEP2 identity-probe configuration: {self:?}"
        )
    }
}

impl std::error::Error for IdentityProbeConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RuntimeConfig(source) => Some(source),
            Self::NonGraphicPathByte { .. } => None,
        }
    }
}

/// Exact report bound to the challenge transmitted by this process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EyeIdentityObservation {
    challenge: HandshakeNonce,
    report: IdentityReport,
    encoded_query_bytes: usize,
}

impl EyeIdentityObservation {
    pub const fn challenge(self) -> HandshakeNonce {
        self.challenge
    }

    pub const fn report(self) -> IdentityReport {
        self.report
    }

    pub const fn encoded_query_bytes(self) -> usize {
        self.encoded_query_bytes
    }
}

/// Open one exact serial device, challenge its KEP2 identity, and close it.
///
/// Entropy is obtained before the TTY is opened, so an unavailable entropy
/// source cannot change serial-device state. Successful return proves only a
/// syntactically valid, nonce-bound firmware report.
pub async fn probe_serial_eye_identity(
    config: &IdentityProbeConfig,
) -> Result<(SerialConfigurationEvidence, EyeIdentityObservation), IdentityProbeError> {
    let _runtime = tokio::runtime::Handle::try_current()
        .map_err(|source| IdentityProbeError::NoTokioRuntime { source })?;
    let challenge = fresh_identity_challenge()?;
    let mut transport = SerialTransport::open(config.device(), config.baud_rate())
        .map_err(IdentityProbeError::SerialOpen)?;
    let serial_evidence = transport.evidence().clone();
    let clock = TokioClock::new();
    let observation =
        probe_identity_with_challenge(&mut transport, &clock, config, challenge).await?;
    Ok((serial_evidence, observation))
}

fn fresh_identity_challenge() -> Result<HandshakeNonce, IdentityProbeError> {
    for _ in 0..MAX_CHALLENGE_ATTEMPTS {
        let mut bytes = [0_u8; 8];
        getrandom::fill(&mut bytes).map_err(IdentityProbeError::Entropy)?;
        if let Ok(challenge) = HandshakeNonce::try_new(u64::from_le_bytes(bytes)) {
            return Ok(challenge);
        }
    }
    Err(IdentityProbeError::RejectedChallengeSamples {
        attempts: MAX_CHALLENGE_ATTEMPTS,
    })
}

async fn probe_identity_with_challenge<T, C>(
    transport: &mut T,
    clock: &C,
    config: &IdentityProbeConfig,
    challenge: HandshakeNonce,
) -> Result<EyeIdentityObservation, IdentityProbeError>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let mut frame = [0_u8; MAX_ENCODED_FRAME_BYTES];
    let encoded_query_bytes = encode(Message::IdentityQuery { nonce: challenge }, &mut frame)
        .map_err(IdentityProbeError::Encode)?;
    let query_completed_at = write_query_once(
        transport,
        clock,
        config.operation_timeout(),
        &frame[..encoded_query_bytes],
    )
    .await
    .map_err(IdentityProbeError::Write)?;

    let response = FrameReader::new()
        .read_message(
            transport,
            clock,
            config.operation_timeout(),
            config.empty_delimiter_budget(),
        )
        .await
        .map_err(IdentityProbeError::Read)?;
    if response.started_at <= query_completed_at {
        return Err(IdentityProbeError::ResponsePredatesQuery {
            query_completed_at_ns: query_completed_at.nanos_since_epoch(),
            response_started_at_ns: response.started_at.nanos_since_epoch(),
        });
    }
    let Message::IdentityReport(report) = response.message else {
        return Err(IdentityProbeError::UnexpectedMessage {
            actual: response.message,
        });
    };
    if report.nonce != challenge {
        return Err(IdentityProbeError::ChallengeMismatch {
            expected: challenge.get(),
            actual: report.nonce.get(),
        });
    }
    Ok(EyeIdentityObservation {
        challenge,
        report,
        encoded_query_bytes,
    })
}

async fn write_query_once<T, C>(
    transport: &mut T,
    clock: &C,
    timeout: OperationTimeout,
    frame: &[u8],
) -> Result<MonotonicTimestamp, IdentityQueryWriteError>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let started_at = clock.now().map_err(IdentityQueryWriteError::Clock)?;
    let timeout_ns = u64::try_from(timeout.get().as_nanos()).map_err(|_| {
        IdentityQueryWriteError::DeadlineOverflow {
            started_at_ns: started_at.nanos_since_epoch(),
            timeout,
        }
    })?;
    let deadline_ns = started_at
        .nanos_since_epoch()
        .checked_add(timeout_ns)
        .ok_or(IdentityQueryWriteError::DeadlineOverflow {
            started_at_ns: started_at.nanos_since_epoch(),
            timeout,
        })?;
    let mut transferred = 0_usize;

    while transferred < frame.len() {
        let remaining =
            query_remaining(clock, deadline_ns, TransportOperation::Write, transferred)?;
        let offered = frame.len() - transferred;
        match transport.write_some(&frame[transferred..], remaining).await {
            Ok(0) => {
                return Err(IdentityQueryWriteError::Transport(
                    TransportFailure::contract_violation(
                        TransportOperation::Write,
                        "transport reported a zero-progress successful identity-query write",
                        transferred,
                    ),
                ));
            }
            Ok(written) if written <= offered => transferred += written,
            Ok(written) => {
                return Err(IdentityQueryWriteError::Transport(
                    TransportFailure::contract_violation(
                        TransportOperation::Write,
                        format!(
                            "transport reported {written} identity-query bytes for capacity {offered}"
                        ),
                        transferred.saturating_add(written),
                    ),
                ));
            }
            Err(source) => {
                if source.operation() != TransportOperation::Write {
                    return Err(IdentityQueryWriteError::TransportOperationMismatch {
                        expected: TransportOperation::Write,
                        known_total_progress: transferred
                            .saturating_add(source.bytes_transferred()),
                        source,
                    });
                }
                if source.bytes_transferred() > offered {
                    return Err(IdentityQueryWriteError::Transport(
                        TransportFailure::contract_violation(
                            TransportOperation::Write,
                            format!(
                                "failed identity-query write reported {} bytes for capacity {offered}",
                                source.bytes_transferred()
                            ),
                            transferred,
                        ),
                    ));
                }
                let total = transferred
                    .checked_add(source.bytes_transferred())
                    .ok_or_else(|| {
                        IdentityQueryWriteError::Transport(TransportFailure::contract_violation(
                            TransportOperation::Write,
                            "identity-query write progress counter overflowed",
                            usize::MAX,
                        ))
                    })?;
                return Err(IdentityQueryWriteError::Transport(
                    source.with_total_progress(total),
                ));
            }
        }
    }

    let remaining = query_remaining(clock, deadline_ns, TransportOperation::Flush, transferred)?;
    if let Err(source) = transport.flush(remaining).await {
        if source.operation() != TransportOperation::Flush || source.bytes_transferred() != 0 {
            return Err(IdentityQueryWriteError::TransportOperationMismatch {
                expected: TransportOperation::Flush,
                known_total_progress: transferred.saturating_add(source.bytes_transferred()),
                source,
            });
        }
        return Err(IdentityQueryWriteError::Transport(
            source.with_total_progress(transferred),
        ));
    }
    let completed_at = clock.now().map_err(IdentityQueryWriteError::Clock)?;
    if completed_at.nanos_since_epoch() >= deadline_ns {
        return Err(IdentityQueryWriteError::Transport(
            TransportFailure::timed_out(TransportOperation::Flush, transferred),
        ));
    }
    Ok(completed_at)
}

fn query_remaining<C: MonotonicClock>(
    clock: &C,
    deadline_ns: u64,
    operation: TransportOperation,
    transferred: usize,
) -> Result<Duration, IdentityQueryWriteError> {
    let now = clock.now().map_err(IdentityQueryWriteError::Clock)?;
    let Some(remaining_ns) = deadline_ns.checked_sub(now.nanos_since_epoch()) else {
        return Err(IdentityQueryWriteError::Transport(
            TransportFailure::timed_out(operation, transferred),
        ));
    };
    if remaining_ns == 0 {
        return Err(IdentityQueryWriteError::Transport(
            TransportFailure::timed_out(operation, transferred),
        ));
    }
    Ok(Duration::from_nanos(remaining_ns))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IdentityQueryWriteError {
    Clock(ClockError),
    DeadlineOverflow {
        started_at_ns: u64,
        timeout: OperationTimeout,
    },
    Transport(TransportFailure),
    TransportOperationMismatch {
        expected: TransportOperation,
        known_total_progress: usize,
        source: TransportFailure,
    },
}

impl fmt::Display for IdentityQueryWriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "could not write one KEP2 identity query: {self:?}"
        )
    }
}

impl std::error::Error for IdentityQueryWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Clock(source) => Some(source),
            Self::Transport(source) | Self::TransportOperationMismatch { source, .. } => {
                Some(source)
            }
            Self::DeadlineOverflow { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum IdentityProbeError {
    NoTokioRuntime {
        source: tokio::runtime::TryCurrentError,
    },
    Entropy(getrandom::Error),
    RejectedChallengeSamples {
        attempts: u8,
    },
    SerialOpen(SerialOpenError),
    Encode(EncodeError),
    Write(IdentityQueryWriteError),
    Read(FrameReadError),
    ResponsePredatesQuery {
        query_completed_at_ns: u64,
        response_started_at_ns: u64,
    },
    UnexpectedMessage {
        actual: Message,
    },
    ChallengeMismatch {
        expected: u64,
        actual: u64,
    },
}

impl fmt::Display for IdentityProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "identity-only KEP2 probe failed: {self:?}")
    }
}

impl std::error::Error for IdentityProbeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NoTokioRuntime { source } => Some(source),
            Self::Entropy(source) => Some(source),
            Self::SerialOpen(source) => Some(source),
            Self::Encode(source) => Some(source),
            Self::Write(source) => Some(source),
            Self::Read(source) => Some(source),
            Self::RejectedChallengeSamples { .. }
            | Self::ResponsePredatesQuery { .. }
            | Self::UnexpectedMessage { .. }
            | Self::ChallengeMismatch { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    use kiko_eye_protocol::{
        Capabilities, DeviceBootId, DeviceTimestampMs, DeviceUid, FirmwareBuildId, decode,
    };

    use super::*;

    #[derive(Clone, Default)]
    struct FakeClock(Arc<AtomicU64>);

    impl FakeClock {
        fn tick(&self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    impl MonotonicClock for FakeClock {
        fn now(&self) -> Result<MonotonicTimestamp, ClockError> {
            Ok(MonotonicTimestamp::from_nanos_since_epoch(
                self.0.load(Ordering::SeqCst),
            ))
        }
    }

    #[derive(Default)]
    struct FakeState {
        incoming: VecDeque<u8>,
        written: Vec<u8>,
        write_chunk: usize,
        read_chunk: usize,
        write_calls: usize,
        flush_calls: usize,
        read_calls: usize,
        fail_first_write: bool,
    }

    struct FakeTransport {
        state: Arc<Mutex<FakeState>>,
        clock: FakeClock,
    }

    #[derive(Clone)]
    struct FakeProbe(Arc<Mutex<FakeState>>);

    impl FakeProbe {
        fn written_messages(&self) -> Vec<Message> {
            self.0
                .lock()
                .expect("fake state lock")
                .written
                .split(|byte| *byte == 0)
                .filter(|record| !record.is_empty())
                .map(|record| decode(record).expect("probe emitted canonical KEP2"))
                .collect()
        }

        fn write_calls(&self) -> usize {
            self.0.lock().expect("fake state lock").write_calls
        }
    }

    impl FakeTransport {
        fn new(incoming: Vec<u8>, clock: FakeClock) -> (Self, FakeProbe) {
            let state = Arc::new(Mutex::new(FakeState {
                incoming: incoming.into(),
                write_chunk: 3,
                read_chunk: 2,
                ..FakeState::default()
            }));
            (
                Self {
                    state: Arc::clone(&state),
                    clock,
                },
                FakeProbe(state),
            )
        }
    }

    impl AsyncByteTransport for FakeTransport {
        async fn write_some(
            &mut self,
            bytes: &[u8],
            _timeout: Duration,
        ) -> Result<usize, TransportFailure> {
            self.clock.tick();
            let mut state = self.state.lock().expect("fake state lock");
            state.write_calls += 1;
            if state.fail_first_write && state.write_calls == 1 {
                return Err(TransportFailure::timed_out(TransportOperation::Write, 0));
            }
            let written = bytes.len().min(state.write_chunk);
            state.written.extend_from_slice(&bytes[..written]);
            Ok(written)
        }

        async fn flush(&mut self, _timeout: Duration) -> Result<(), TransportFailure> {
            self.clock.tick();
            self.state.lock().expect("fake state lock").flush_calls += 1;
            Ok(())
        }

        async fn read_some(
            &mut self,
            bytes: &mut [u8],
            _timeout: Duration,
        ) -> Result<usize, TransportFailure> {
            self.clock.tick();
            let mut state = self.state.lock().expect("fake state lock");
            state.read_calls += 1;
            let read = bytes.len().min(state.read_chunk).min(state.incoming.len());
            for output in &mut bytes[..read] {
                *output = state.incoming.pop_front().expect("bounded incoming byte");
            }
            Ok(read)
        }
    }

    fn config(path: &str) -> IdentityProbeConfig {
        IdentityProbeConfig::parse(IdentityProbeConfigInput {
            device_path: path.to_owned(),
            operation_timeout_ms: 100,
        })
        .expect("valid probe config")
    }

    fn nonce(value: u64) -> HandshakeNonce {
        HandshakeNonce::try_new(value).expect("nonzero nonce")
    }

    fn report(challenge: HandshakeNonce) -> IdentityReport {
        IdentityReport {
            nonce: challenge,
            device_uid: DeviceUid::try_new([0x12; 16]).expect("nonzero UID"),
            firmware_build_id: FirmwareBuildId::try_new([0x34; 32]).expect("nonzero build ID"),
            boot_id: DeviceBootId::try_new(55).expect("nonzero boot ID"),
            device_uptime: DeviceTimestampMs::from_millis_since_boot(89),
            capabilities: Capabilities::try_from_bits(Capabilities::KNOWN_BITS)
                .expect("known capabilities"),
        }
    }

    fn encoded(message: Message) -> Vec<u8> {
        let mut bytes = [0_u8; MAX_ENCODED_FRAME_BYTES];
        let length = encode(message, &mut bytes).expect("encodable message");
        bytes[..length].to_vec()
    }

    #[test]
    fn config_accepts_only_exact_linux_or_macos_graphic_paths_and_bounded_timeouts() {
        for path in [
            "/dev/serial/by-id/usb-kiko_kiko-eyes-kep2_a1-if00",
            "/dev/cu.usbmodem-kiko-eyes-a1",
        ] {
            assert!(config(path).device().path() == path);
        }
        for path in [
            "/dev/ttyACM2",
            "/dev/serial/by-id/",
            "/dev/serial/by-id/a/b",
            "/dev/serial/by-id/a b",
            "/dev/cu.a\tb",
        ] {
            assert!(
                IdentityProbeConfig::parse(IdentityProbeConfigInput {
                    device_path: path.to_owned(),
                    operation_timeout_ms: 100,
                })
                .is_err(),
                "{path:?}"
            );
        }
        for timeout in [0, 5_001] {
            assert!(
                IdentityProbeConfig::parse(IdentityProbeConfigInput {
                    device_path: "/dev/cu.kiko-eyes".to_owned(),
                    operation_timeout_ms: timeout,
                })
                .is_err()
            );
        }
    }

    #[tokio::test]
    async fn partial_io_sends_only_one_query_and_accepts_its_exact_report() {
        let challenge = nonce(41);
        let expected_report = report(challenge);
        let clock = FakeClock::default();
        let (mut transport, probe) = FakeTransport::new(
            encoded(Message::IdentityReport(expected_report)),
            clock.clone(),
        );

        let observation = probe_identity_with_challenge(
            &mut transport,
            &clock,
            &config("/dev/cu.kiko-eyes"),
            challenge,
        )
        .await
        .expect("identity exchange");

        assert_eq!(observation.challenge(), challenge);
        assert_eq!(observation.report(), expected_report);
        assert_eq!(
            probe.written_messages(),
            vec![Message::IdentityQuery { nonce: challenge }]
        );
    }

    #[tokio::test]
    async fn mismatched_challenge_is_not_skipped_or_treated_as_identity() {
        let challenge = nonce(41);
        let clock = FakeClock::default();
        let (mut transport, probe) = FakeTransport::new(
            encoded(Message::IdentityReport(report(nonce(42)))),
            clock.clone(),
        );

        let result = probe_identity_with_challenge(
            &mut transport,
            &clock,
            &config("/dev/cu.kiko-eyes"),
            challenge,
        )
        .await;

        assert!(matches!(
            result,
            Err(IdentityProbeError::ChallengeMismatch {
                expected: 41,
                actual: 42
            })
        ));
        assert_eq!(
            probe.written_messages(),
            vec![Message::IdentityQuery { nonce: challenge }]
        );
    }

    #[tokio::test]
    async fn any_non_identity_response_fails_without_sending_control() {
        let challenge = nonce(41);
        let clock = FakeClock::default();
        let (mut transport, probe) = FakeTransport::new(
            encoded(Message::IdentityQuery { nonce: nonce(99) }),
            clock.clone(),
        );

        let result = probe_identity_with_challenge(
            &mut transport,
            &clock,
            &config("/dev/cu.kiko-eyes"),
            challenge,
        )
        .await;

        assert!(matches!(
            result,
            Err(IdentityProbeError::UnexpectedMessage {
                actual: Message::IdentityQuery { .. }
            })
        ));
        assert_eq!(
            probe.written_messages(),
            vec![Message::IdentityQuery { nonce: challenge }]
        );
    }

    #[tokio::test]
    async fn zero_progress_write_failure_is_never_retried() {
        let challenge = nonce(41);
        let clock = FakeClock::default();
        let (mut transport, probe) = FakeTransport::new(Vec::new(), clock.clone());
        transport
            .state
            .lock()
            .expect("fake state lock")
            .fail_first_write = true;

        let result = probe_identity_with_challenge(
            &mut transport,
            &clock,
            &config("/dev/cu.kiko-eyes"),
            challenge,
        )
        .await;

        assert!(matches!(result, Err(IdentityProbeError::Write(_))));
        assert_eq!(probe.write_calls(), 1);
        assert!(probe.written_messages().is_empty());
    }
}
