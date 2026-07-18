use crate::config::UdpEndpoint;
use crate::domain::MonotonicInstant;
use robot_protocol::v2::{
    EncodeError, FrameError, MAX_RAW_FRAME_BYTES, Message, RawFrame, decode_raw_frame,
};
use std::fmt;
use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

/// Byte encoding is isolated here so the client state machine handles only
/// canonical `robot_protocol::v2::Message` domain values.
pub trait V2WireAdapter {
    type Encoded;
    type Error: std::error::Error + Send + Sync + 'static;

    fn encode(&self, message: Message) -> Result<Self::Encoded, Self::Error>;
    fn encoded_bytes<'a>(&self, encoded: &'a Self::Encoded) -> &'a [u8];
    fn decode(&self, datagram: &[u8]) -> Result<Message, Self::Error>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RobotProtocolV2WireAdapter;

impl V2WireAdapter for RobotProtocolV2WireAdapter {
    type Encoded = RawFrame;
    type Error = RobotProtocolV2CodecError;

    fn encode(&self, message: Message) -> Result<Self::Encoded, Self::Error> {
        RawFrame::encode(message).map_err(RobotProtocolV2CodecError::Encode)
    }

    fn encoded_bytes<'a>(&self, encoded: &'a Self::Encoded) -> &'a [u8] {
        encoded.as_bytes()
    }

    fn decode(&self, datagram: &[u8]) -> Result<Message, Self::Error> {
        decode_raw_frame(datagram).map_err(RobotProtocolV2CodecError::Decode)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RobotProtocolV2CodecError {
    Encode(EncodeError),
    Decode(FrameError),
}

impl fmt::Display for RobotProtocolV2CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encode(source) => write!(formatter, "failed to encode V2 message: {source}"),
            Self::Decode(source) => write!(formatter, "failed to decode V2 message: {source}"),
        }
    }
}

impl std::error::Error for RobotProtocolV2CodecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encode(source) => Some(source),
            Self::Decode(source) => Some(source),
        }
    }
}

pub trait V2CommandTransport {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Sends exactly one V2 message and reads at most one response. No retry,
    /// filtering, or fallback is allowed inside the transport.
    fn exchange_once(
        &mut self,
        request: Message,
        timeout: Duration,
    ) -> Result<Message, Self::Error>;
}

pub trait MonotonicClock {
    fn now(&self) -> MonotonicInstant;
}

#[derive(Debug)]
pub struct SystemMonotonicClock {
    origin: Instant,
}

impl SystemMonotonicClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemMonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MonotonicClock for SystemMonotonicClock {
    fn now(&self) -> MonotonicInstant {
        MonotonicInstant::from_nanos_since_clock_start(self.origin.elapsed().as_nanos())
    }
}

#[derive(Debug)]
pub struct UdpV2Transport<Adapter = RobotProtocolV2WireAdapter> {
    socket: UdpSocket,
    peer: SocketAddr,
    adapter: Adapter,
}

impl UdpV2Transport<RobotProtocolV2WireAdapter> {
    pub fn connect_canonical(endpoint: UdpEndpoint) -> Result<Self, UdpTransportBuildError> {
        Self::connect(endpoint, RobotProtocolV2WireAdapter)
    }
}

impl<Adapter> UdpV2Transport<Adapter> {
    pub fn connect(
        endpoint: UdpEndpoint,
        adapter: Adapter,
    ) -> Result<Self, UdpTransportBuildError> {
        let peer = endpoint.socket_addr();
        let bind_address = match peer {
            SocketAddr::V4(_) => SocketAddr::from(([0, 0, 0, 0], 0)),
            SocketAddr::V6(_) => SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 0], 0)),
        };
        let socket = UdpSocket::bind(bind_address).map_err(UdpTransportBuildError::Bind)?;
        socket
            .connect(peer)
            .map_err(UdpTransportBuildError::Connect)?;
        Ok(Self {
            socket,
            peer,
            adapter,
        })
    }

    pub const fn peer(&self) -> SocketAddr {
        self.peer
    }

    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    fn configure_write_timeout<CodecError>(
        &self,
        timeout: Duration,
    ) -> Result<(), UdpTransportError<CodecError>> {
        self.socket
            .set_write_timeout(Some(timeout))
            .map_err(|source| UdpTransportError::Io {
                phase: UdpTransportPhase::ConfigureWriteTimeout,
                source,
            })
    }

    fn configure_read_timeout<CodecError>(
        &self,
        timeout: Duration,
    ) -> Result<(), UdpTransportError<CodecError>> {
        self.socket
            .set_read_timeout(Some(timeout))
            .map_err(|source| UdpTransportError::Io {
                phase: UdpTransportPhase::ConfigureReadTimeout,
                source,
            })
    }
}

impl<Adapter> V2CommandTransport for UdpV2Transport<Adapter>
where
    Adapter: V2WireAdapter,
{
    type Error = UdpTransportError<Adapter::Error>;

    fn exchange_once(
        &mut self,
        request: Message,
        timeout: Duration,
    ) -> Result<Message, Self::Error> {
        let exchange_started = Instant::now();
        self.configure_write_timeout(timeout)?;
        let encoded = self
            .adapter
            .encode(request)
            .map_err(UdpTransportError::Codec)?;
        let datagram = self.adapter.encoded_bytes(&encoded);
        if datagram.is_empty() {
            return Err(UdpTransportError::EmptyEncodedDatagram);
        }
        if datagram.len() > MAX_RAW_FRAME_BYTES {
            return Err(UdpTransportError::EncodedDatagramTooLarge {
                bytes: datagram.len(),
                maximum_bytes: MAX_RAW_FRAME_BYTES,
            });
        }
        let sent = self
            .socket
            .send(datagram)
            .map_err(|source| UdpTransportError::Io {
                phase: UdpTransportPhase::Send,
                source,
            })?;
        if sent != datagram.len() {
            return Err(UdpTransportError::ShortSend {
                expected_bytes: datagram.len(),
                sent_bytes: sent,
            });
        }

        let remaining = timeout
            .checked_sub(exchange_started.elapsed())
            .filter(|remaining| !remaining.is_zero())
            .ok_or(UdpTransportError::BudgetExhaustedBeforeReceive { timeout })?;
        self.configure_read_timeout(remaining)?;

        let mut receive_buffer = [0_u8; MAX_RAW_FRAME_BYTES + 1];
        let received =
            self.socket
                .recv(&mut receive_buffer)
                .map_err(|source| UdpTransportError::Io {
                    phase: UdpTransportPhase::Receive,
                    source,
                })?;
        if received > MAX_RAW_FRAME_BYTES {
            return Err(UdpTransportError::ReceivedDatagramTooLarge {
                maximum_bytes: MAX_RAW_FRAME_BYTES,
            });
        }
        self.adapter
            .decode(&receive_buffer[..received])
            .map_err(UdpTransportError::Codec)
    }
}

#[derive(Debug)]
pub enum UdpTransportBuildError {
    Bind(io::Error),
    Connect(io::Error),
}

impl fmt::Display for UdpTransportBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bind(source) => write!(
                formatter,
                "failed to bind local UDP command socket: {source}"
            ),
            Self::Connect(source) => {
                write!(formatter, "failed to connect UDP command socket: {source}")
            }
        }
    }
}

impl std::error::Error for UdpTransportBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bind(source) | Self::Connect(source) => Some(source),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UdpTransportPhase {
    ConfigureWriteTimeout,
    ConfigureReadTimeout,
    Send,
    Receive,
}

#[derive(Debug)]
pub enum UdpTransportError<CodecError> {
    Io {
        phase: UdpTransportPhase,
        source: io::Error,
    },
    Codec(CodecError),
    EmptyEncodedDatagram,
    EncodedDatagramTooLarge {
        bytes: usize,
        maximum_bytes: usize,
    },
    ReceivedDatagramTooLarge {
        maximum_bytes: usize,
    },
    ShortSend {
        expected_bytes: usize,
        sent_bytes: usize,
    },
    BudgetExhaustedBeforeReceive {
        timeout: Duration,
    },
}

impl<CodecError: fmt::Display> fmt::Display for UdpTransportError<CodecError> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { phase, source } => {
                write!(
                    formatter,
                    "UDP command transport failed during {phase:?}: {source}"
                )
            }
            Self::Codec(source) => {
                write!(formatter, "V2 command codec rejected a datagram: {source}")
            }
            Self::EmptyEncodedDatagram => {
                formatter.write_str("V2 command codec produced an empty datagram")
            }
            Self::EncodedDatagramTooLarge {
                bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "V2 command codec produced {bytes} bytes; maximum is {maximum_bytes}"
            ),
            Self::ReceivedDatagramTooLarge { maximum_bytes } => write!(
                formatter,
                "received V2 command datagram exceeds the {maximum_bytes}-byte bound"
            ),
            Self::ShortSend {
                expected_bytes,
                sent_bytes,
            } => write!(
                formatter,
                "UDP command send reported {sent_bytes} of {expected_bytes} bytes"
            ),
            Self::BudgetExhaustedBeforeReceive { timeout } => write!(
                formatter,
                "UDP command exchange exhausted its {timeout:?} budget before receive"
            ),
        }
    }
}

impl<CodecError> std::error::Error for UdpTransportError<CodecError>
where
    CodecError: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Codec(source) => Some(source),
            _ => None,
        }
    }
}
