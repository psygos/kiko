use std::fmt;
use std::future::Future;
use std::io;
use std::time::Duration;

use kiko_expression_core::MonotonicTimestamp;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_serial::{DataBits, FlowControl, Parity, SerialPort, SerialPortBuilderExt, StopBits};

use crate::config::{BaudRate, DeviceIdentity};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClockError {
    ElapsedNanosecondsOutOfRange { elapsed_nanoseconds: u128 },
    SourceUnavailable { message: Box<str> },
}

impl fmt::Display for ClockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "monotonic eye-runtime clock failed: {self:?}")
    }
}

impl std::error::Error for ClockError {}

pub trait MonotonicClock: Send + Sync + 'static {
    fn now(&self) -> Result<MonotonicTimestamp, ClockError>;
}

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
    fn now(&self) -> Result<MonotonicTimestamp, ClockError> {
        let elapsed_nanoseconds = self.origin.elapsed().as_nanos();
        let value = u64::try_from(elapsed_nanoseconds).map_err(|_| {
            ClockError::ElapsedNanosecondsOutOfRange {
                elapsed_nanoseconds,
            }
        })?;
        Ok(MonotonicTimestamp::from_nanos_since_epoch(value))
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
    ContractViolation,
}

/// Owned transport failure preserving operation, OS classification, message,
/// and known progress. Positive write progress makes retransmission unsafe.
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

    pub fn contract_violation(
        operation: TransportOperation,
        message: impl Into<Box<str>>,
        bytes_transferred: usize,
    ) -> Self {
        Self {
            operation,
            kind: TransportFailureKind::ContractViolation,
            raw_os_error: None,
            message: message.into(),
            bytes_transferred,
        }
    }

    pub(crate) fn with_total_progress(mut self, total: usize) -> Self {
        self.bytes_transferred = total;
        self
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
            "{:?} failed after {} bytes ({:?}, os={:?}): {}",
            self.operation, self.bytes_transferred, self.kind, self.raw_os_error, self.message
        )
    }
}

impl std::error::Error for TransportFailure {}

/// Minimal transport owned by exactly one actor. Methods perform one I/O
/// operation and never retry internally.
pub trait AsyncByteTransport: Send + 'static {
    fn write_some(
        &mut self,
        bytes: &[u8],
        timeout: Duration,
    ) -> impl Future<Output = Result<usize, TransportFailure>> + Send;

    fn flush(
        &mut self,
        timeout: Duration,
    ) -> impl Future<Output = Result<(), TransportFailure>> + Send;

    fn read_some(
        &mut self,
        bytes: &mut [u8],
        timeout: Duration,
    ) -> impl Future<Output = Result<usize, TransportFailure>> + Send;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SerialConfigurationEvidence {
    device: DeviceIdentity,
    exclusive_owner_claimed: bool,
    baud_rate_bps_readback: u32,
    data_bits_8_readback: bool,
    parity_none_readback: bool,
    stop_bits_1_readback: bool,
    flow_control_none_readback: bool,
}

impl SerialConfigurationEvidence {
    pub const fn device(&self) -> &DeviceIdentity {
        &self.device
    }

    pub const fn exclusive_owner_claimed(&self) -> bool {
        self.exclusive_owner_claimed
    }

    pub const fn baud_rate_bps_readback(&self) -> u32 {
        self.baud_rate_bps_readback
    }

    pub const fn data_bits_8_readback(&self) -> bool {
        self.data_bits_8_readback
    }

    pub const fn parity_none_readback(&self) -> bool {
        self.parity_none_readback
    }

    pub const fn stop_bits_1_readback(&self) -> bool {
        self.stop_bits_1_readback
    }

    pub const fn flow_control_none_readback(&self) -> bool {
        self.flow_control_none_readback
    }
}

pub(crate) struct SerialTransport {
    port: tokio_serial::SerialStream,
    evidence: SerialConfigurationEvidence,
}

impl SerialTransport {
    pub(crate) fn open(
        device: &DeviceIdentity,
        baud_rate: BaudRate,
    ) -> Result<Self, SerialOpenError> {
        let builder = tokio_serial::new(device.path(), baud_rate.get())
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
            .map_err(|source| SerialOpenError::ClaimExclusive {
                device: device.clone(),
                source,
            })?;

        apply_setting(device, &mut port, SerialSetting::BaudRate, |port| {
            port.set_baud_rate(baud_rate.get())
        })?;
        apply_setting(device, &mut port, SerialSetting::DataBits, |port| {
            port.set_data_bits(DataBits::Eight)
        })?;
        apply_setting(device, &mut port, SerialSetting::Parity, |port| {
            port.set_parity(Parity::None)
        })?;
        apply_setting(device, &mut port, SerialSetting::StopBits, |port| {
            port.set_stop_bits(StopBits::One)
        })?;
        apply_setting(device, &mut port, SerialSetting::FlowControl, |port| {
            port.set_flow_control(FlowControl::None)
        })?;

        let actual_baud = read_setting(
            device,
            &port,
            SerialSetting::BaudRate,
            SerialPort::baud_rate,
        )?;
        let data_bits = read_setting(
            device,
            &port,
            SerialSetting::DataBits,
            SerialPort::data_bits,
        )?;
        let parity = read_setting(device, &port, SerialSetting::Parity, SerialPort::parity)?;
        let stop_bits = read_setting(
            device,
            &port,
            SerialSetting::StopBits,
            SerialPort::stop_bits,
        )?;
        let flow_control = read_setting(
            device,
            &port,
            SerialSetting::FlowControl,
            SerialPort::flow_control,
        )?;

        if actual_baud != baud_rate.get()
            || data_bits != DataBits::Eight
            || parity != Parity::None
            || stop_bits != StopBits::One
            || flow_control != FlowControl::None
        {
            return Err(SerialOpenError::LineConfigurationReadbackMismatch {
                device: device.clone(),
                expected_baud_rate_bps: baud_rate.get(),
                actual_baud_rate_bps: actual_baud,
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
                exclusive_owner_claimed: cfg!(unix),
                baud_rate_bps_readback: actual_baud,
                data_bits_8_readback: true,
                parity_none_readback: true,
                stop_bits_1_readback: true,
                flow_control_none_readback: true,
            },
        })
    }

    pub(crate) const fn evidence(&self) -> &SerialConfigurationEvidence {
        &self.evidence
    }
}

fn apply_setting(
    device: &DeviceIdentity,
    port: &mut tokio_serial::SerialStream,
    setting: SerialSetting,
    apply: impl FnOnce(&mut tokio_serial::SerialStream) -> tokio_serial::Result<()>,
) -> Result<(), SerialOpenError> {
    apply(port).map_err(|source| SerialOpenError::ApplySetting {
        device: device.clone(),
        setting,
        source,
    })
}

fn read_setting<T>(
    device: &DeviceIdentity,
    port: &tokio_serial::SerialStream,
    setting: SerialSetting,
    read: impl FnOnce(&tokio_serial::SerialStream) -> tokio_serial::Result<T>,
) -> Result<T, SerialOpenError> {
    read(port).map_err(|source| SerialOpenError::ReadSetting {
        device: device.clone(),
        setting,
        source,
    })
}

impl AsyncByteTransport for SerialTransport {
    async fn write_some(
        &mut self,
        bytes: &[u8],
        timeout: Duration,
    ) -> Result<usize, TransportFailure> {
        match tokio::time::timeout(timeout, self.port.write(bytes)).await {
            Ok(Ok(written)) => Ok(written),
            Ok(Err(source)) => Err(TransportFailure::from_io(
                TransportOperation::Write,
                &source,
                0,
            )),
            Err(_) => Err(TransportFailure::timed_out(TransportOperation::Write, 0)),
        }
    }

    async fn flush(&mut self, timeout: Duration) -> Result<(), TransportFailure> {
        match tokio::time::timeout(timeout, self.port.flush()).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(source)) => Err(TransportFailure::from_io(
                TransportOperation::Flush,
                &source,
                0,
            )),
            Err(_) => Err(TransportFailure::timed_out(TransportOperation::Flush, 0)),
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
}

#[derive(Debug)]
pub enum SerialOpenError {
    Open {
        device: DeviceIdentity,
        source: tokio_serial::Error,
    },
    ClaimExclusive {
        device: DeviceIdentity,
        source: tokio_serial::Error,
    },
    ApplySetting {
        device: DeviceIdentity,
        setting: SerialSetting,
        source: tokio_serial::Error,
    },
    ReadSetting {
        device: DeviceIdentity,
        setting: SerialSetting,
        source: tokio_serial::Error,
    },
    LineConfigurationReadbackMismatch {
        device: DeviceIdentity,
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
            "could not establish exact KEP2 eye serial ownership: {self:?}"
        )
    }
}

impl std::error::Error for SerialOpenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open { source, .. }
            | Self::ClaimExclusive { source, .. }
            | Self::ApplySetting { source, .. }
            | Self::ReadSetting { source, .. } => Some(source),
            Self::LineConfigurationReadbackMismatch { .. } => None,
        }
    }
}
