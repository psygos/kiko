use std::fmt;
use std::future::Future;
use std::io;
use std::time::Duration;

use kiko_head_protocol::{ADAPTER_DTR_ASSERTED, ADAPTER_RTS_ASSERTED, BUS_BAUD_RATE_BPS};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_serial::{DataBits, FlowControl, Parity, SerialPort, SerialPortBuilderExt, StopBits};

use crate::config::DeviceIdentity;

/// Duration since one injected monotonic clock's origin.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonotonicTime(Duration);

impl MonotonicTime {
    pub const ZERO: Self = Self(Duration::ZERO);

    pub const fn from_duration_since_origin(value: Duration) -> Self {
        Self(value)
    }

    pub const fn duration_since_origin(self) -> Duration {
        self.0
    }

    pub fn checked_duration_since(self, earlier: Self) -> Option<Duration> {
        self.0.checked_sub(earlier.0)
    }
}

/// Injectable monotonic time source. Test clocks can advance only when a
/// scripted transport operation says time passed.
pub trait MonotonicClock: Send + Sync + 'static {
    fn now(&self) -> MonotonicTime;
}

/// Production clock backed by Tokio's monotonic clock.
#[derive(Clone, Debug)]
pub struct TokioClock {
    origin: tokio::time::Instant,
}

impl TokioClock {
    pub fn new() -> Self {
        Self {
            origin: tokio::time::Instant::now(),
        }
    }
}

impl Default for TokioClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MonotonicClock for TokioClock {
    fn now(&self) -> MonotonicTime {
        MonotonicTime(self.origin.elapsed())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransportOperation {
    Read,
    Write,
    Flush,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TransportFailureKind {
    TimedOut,
    Io(io::ErrorKind),
}

/// Owned, cloneable transport failure retaining the OS classification, raw OS
/// code, message, operation, and known byte progress.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportFailure {
    operation: TransportOperation,
    kind: TransportFailureKind,
    raw_os_error: Option<i32>,
    message: Box<str>,
    bytes_transferred: usize,
}

impl TransportFailure {
    pub fn timed_out(operation: TransportOperation, bytes_transferred: usize) -> Self {
        Self {
            operation,
            kind: TransportFailureKind::TimedOut,
            raw_os_error: None,
            message: "bounded transport operation timed out".into(),
            bytes_transferred,
        }
    }

    pub fn from_io(
        operation: TransportOperation,
        source: &io::Error,
        bytes_transferred: usize,
    ) -> Self {
        Self {
            operation,
            kind: TransportFailureKind::Io(source.kind()),
            raw_os_error: source.raw_os_error(),
            message: source.to_string().into(),
            bytes_transferred,
        }
    }

    pub const fn operation(&self) -> TransportOperation {
        self.operation
    }

    pub const fn kind(&self) -> TransportFailureKind {
        self.kind
    }

    pub const fn raw_os_error(&self) -> Option<i32> {
        self.raw_os_error
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub const fn bytes_transferred(&self) -> usize {
        self.bytes_transferred
    }

    pub const fn is_retryable_without_progress(&self) -> bool {
        if self.bytes_transferred != 0 {
            return false;
        }
        matches!(
            self.kind,
            TransportFailureKind::TimedOut | TransportFailureKind::Io(io::ErrorKind::Interrupted)
        )
    }
}

impl fmt::Display for TransportFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} transport failure after {} bytes ({:?}, os={:?}): {}",
            self.operation, self.bytes_transferred, self.kind, self.raw_os_error, self.message
        )
    }
}

impl std::error::Error for TransportFailure {}

/// Minimal async byte transport owned exclusively by the head actor.
///
/// Each method receives one already-bounded timeout. Implementations must not
/// retry internally.
pub trait AsyncByteTransport: Send + 'static {
    fn write_all(
        &mut self,
        bytes: &[u8],
        timeout: Duration,
    ) -> impl Future<Output = Result<(), TransportFailure>> + Send;

    fn read_some(
        &mut self,
        bytes: &mut [u8],
        timeout: Duration,
    ) -> impl Future<Output = Result<usize, TransportFailure>> + Send;
}

/// Evidence available from the host driver before the first protocol byte is
/// sent. DTR/RTS fields mean the OS setters accepted the requested states;
/// common serial APIs provide no electrical readback for output modem lines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SerialConfigurationEvidence {
    pub device: DeviceIdentity,
    pub exclusive_owner_claimed: bool,
    pub baud_rate_bps_readback: u32,
    pub data_bits_8_readback: bool,
    pub parity_none_readback: bool,
    pub stop_bits_1_readback: bool,
    pub flow_control_none_readback: bool,
    pub dtr_false_setter_accepted: bool,
    pub rts_true_setter_accepted: bool,
}

/// Configured production transport. Construction is the only route to the
/// inner serial stream, preserving one actor owner.
pub(crate) struct SerialTransport {
    port: tokio_serial::SerialStream,
    evidence: SerialConfigurationEvidence,
}

impl SerialTransport {
    pub(crate) fn open(device: &DeviceIdentity) -> Result<Self, SerialOpenError> {
        let builder = tokio_serial::new(device.path(), BUS_BAUD_RATE_BPS)
            .data_bits(DataBits::Eight)
            .parity(Parity::None)
            .stop_bits(StopBits::One)
            .flow_control(FlowControl::None);
        let mut port = builder
            .open_native_async()
            .map_err(|source| SerialOpenError::Open {
                device: device.clone(),
                source,
            })?;

        #[cfg(unix)]
        port.set_exclusive(true)
            .map_err(|source| SerialOpenError::ClaimExclusive { source })?;

        // Opening a TTY can itself change modem-control outputs. Reapply these
        // exact adapter states after open and before any protocol traffic.
        apply_setting(&mut port, SerialSetting::BaudRate, |port| {
            port.set_baud_rate(BUS_BAUD_RATE_BPS)
        })?;
        apply_setting(&mut port, SerialSetting::DataBits, |port| {
            port.set_data_bits(DataBits::Eight)
        })?;
        apply_setting(&mut port, SerialSetting::Parity, |port| {
            port.set_parity(Parity::None)
        })?;
        apply_setting(&mut port, SerialSetting::StopBits, |port| {
            port.set_stop_bits(StopBits::One)
        })?;
        apply_setting(&mut port, SerialSetting::FlowControl, |port| {
            port.set_flow_control(FlowControl::None)
        })?;
        apply_setting(&mut port, SerialSetting::Dtr, |port| {
            port.write_data_terminal_ready(ADAPTER_DTR_ASSERTED)
        })?;
        apply_setting(&mut port, SerialSetting::Rts, |port| {
            port.write_request_to_send(ADAPTER_RTS_ASSERTED)
        })?;

        let baud_rate = read_setting(&port, SerialSetting::BaudRate, SerialPort::baud_rate)?;
        let data_bits = read_setting(&port, SerialSetting::DataBits, SerialPort::data_bits)?;
        let parity = read_setting(&port, SerialSetting::Parity, SerialPort::parity)?;
        let stop_bits = read_setting(&port, SerialSetting::StopBits, SerialPort::stop_bits)?;
        let flow_control =
            read_setting(&port, SerialSetting::FlowControl, SerialPort::flow_control)?;

        if baud_rate != BUS_BAUD_RATE_BPS
            || data_bits != DataBits::Eight
            || parity != Parity::None
            || stop_bits != StopBits::One
            || flow_control != FlowControl::None
        {
            return Err(SerialOpenError::LineConfigurationReadbackMismatch {
                expected_baud_rate_bps: BUS_BAUD_RATE_BPS,
                actual_baud_rate_bps: baud_rate,
                actual_data_bits: data_bits,
                actual_parity: parity,
                actual_stop_bits: stop_bits,
                actual_flow_control: flow_control,
            });
        }

        Ok(Self {
            port,
            evidence: SerialConfigurationEvidence {
                device: device.clone(),
                exclusive_owner_claimed: true,
                baud_rate_bps_readback: baud_rate,
                data_bits_8_readback: true,
                parity_none_readback: true,
                stop_bits_1_readback: true,
                flow_control_none_readback: true,
                dtr_false_setter_accepted: !ADAPTER_DTR_ASSERTED,
                rts_true_setter_accepted: ADAPTER_RTS_ASSERTED,
            },
        })
    }

    pub(crate) fn evidence(&self) -> &SerialConfigurationEvidence {
        &self.evidence
    }
}

fn apply_setting(
    port: &mut tokio_serial::SerialStream,
    setting: SerialSetting,
    apply: impl FnOnce(&mut tokio_serial::SerialStream) -> tokio_serial::Result<()>,
) -> Result<(), SerialOpenError> {
    apply(port).map_err(|source| SerialOpenError::ApplySetting { setting, source })
}

fn read_setting<T>(
    port: &tokio_serial::SerialStream,
    setting: SerialSetting,
    read: impl FnOnce(&tokio_serial::SerialStream) -> tokio_serial::Result<T>,
) -> Result<T, SerialOpenError> {
    read(port).map_err(|source| SerialOpenError::ReadSetting { setting, source })
}

impl AsyncByteTransport for SerialTransport {
    async fn write_all(&mut self, bytes: &[u8], timeout: Duration) -> Result<(), TransportFailure> {
        let mut transferred = 0_usize;
        let mut active_operation = TransportOperation::Write;
        let operation = async {
            while transferred < bytes.len() {
                let written = self
                    .port
                    .write(&bytes[transferred..])
                    .await
                    .map_err(|source| {
                        TransportFailure::from_io(TransportOperation::Write, &source, transferred)
                    })?;
                if written == 0 {
                    let source = io::Error::from(io::ErrorKind::WriteZero);
                    return Err(TransportFailure::from_io(
                        TransportOperation::Write,
                        &source,
                        transferred,
                    ));
                }
                transferred += written;
            }
            active_operation = TransportOperation::Flush;
            self.port.flush().await.map_err(|source| {
                TransportFailure::from_io(TransportOperation::Flush, &source, transferred)
            })
        };

        match tokio::time::timeout(timeout, operation).await {
            Ok(result) => result,
            Err(_) => Err(TransportFailure::timed_out(active_operation, transferred)),
        }
    }

    async fn read_some(
        &mut self,
        bytes: &mut [u8],
        timeout: Duration,
    ) -> Result<usize, TransportFailure> {
        match tokio::time::timeout(timeout, self.port.read(bytes)).await {
            Ok(Ok(read)) => Ok(read),
            Ok(Err(source)) => Err(TransportFailure::from_io(
                TransportOperation::Read,
                &source,
                0,
            )),
            Err(_) => Err(TransportFailure::timed_out(TransportOperation::Read, 0)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SerialSetting {
    BaudRate,
    DataBits,
    Parity,
    StopBits,
    FlowControl,
    Dtr,
    Rts,
}

#[derive(Debug)]
pub enum SerialOpenError {
    Open {
        device: DeviceIdentity,
        source: tokio_serial::Error,
    },
    ClaimExclusive {
        source: tokio_serial::Error,
    },
    ApplySetting {
        setting: SerialSetting,
        source: tokio_serial::Error,
    },
    ReadSetting {
        setting: SerialSetting,
        source: tokio_serial::Error,
    },
    LineConfigurationReadbackMismatch {
        expected_baud_rate_bps: u32,
        actual_baud_rate_bps: u32,
        actual_data_bits: DataBits,
        actual_parity: Parity,
        actual_stop_bits: StopBits,
        actual_flow_control: FlowControl,
    },
}

impl fmt::Display for SerialOpenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "could not establish exact Kiko head serial ownership: {self:?}"
        )
    }
}

impl std::error::Error for SerialOpenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open { source, .. }
            | Self::ClaimExclusive { source }
            | Self::ApplySetting { source, .. }
            | Self::ReadSetting { source, .. } => Some(source),
            Self::LineConfigurationReadbackMismatch { .. } => None,
        }
    }
}
