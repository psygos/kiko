use core::fmt;
use std::net::SocketAddr;

pub const MAX_ROBOT_ID_BYTES: usize = 64;
pub const MAX_OAK_MXID_BYTES: usize = 64;
pub const MAX_BUILD_PROVENANCE_BYTES: usize = 96;
pub const MAX_CONTROL_ENDPOINT_ID_BYTES: usize = 192;
pub const MAX_SERIAL_BY_ID_PATH_BYTES: usize = 192;
pub const MAX_ARTIFACT_ID_BYTES: usize = 64;

const SERIAL_BY_ID_PREFIX: &str = "/dev/serial/by-id/";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundedTextError {
    Empty,
    ZeroIdentity,
    TooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    InvalidByte {
        index: usize,
        byte: u8,
    },
    OakMxidTooShort {
        actual_bytes: usize,
        minimum_bytes: usize,
    },
    NotPersistentSerialById,
    SerialByIdHasNestedComponent,
    InvalidSerialByIdComponent,
    InvalidControlEndpoint,
}

impl fmt::Display for BoundedTextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid bounded identity text: {self:?}")
    }
}

impl std::error::Error for BoundedTextError {}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct BoundedAscii<const N: usize> {
    bytes: [u8; N],
    len: u8,
}

impl<const N: usize> BoundedAscii<N> {
    fn parse(value: String, allowed: impl Fn(u8) -> bool) -> Result<Self, BoundedTextError> {
        if value.is_empty() {
            return Err(BoundedTextError::Empty);
        }
        if value.len() > N || value.len() > usize::from(u8::MAX) {
            return Err(BoundedTextError::TooLong {
                actual_bytes: value.len(),
                maximum_bytes: N.min(usize::from(u8::MAX)),
            });
        }
        if let Some((index, byte)) = value.bytes().enumerate().find(|(_, byte)| !allowed(*byte)) {
            return Err(BoundedTextError::InvalidByte { index, byte });
        }
        let mut bytes = [0; N];
        bytes[..value.len()].copy_from_slice(value.as_bytes());
        Ok(Self {
            bytes,
            len: value.len() as u8,
        })
    }

    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.bytes[..usize::from(self.len)])
            .expect("bounded text contains parsed ASCII")
    }
}

impl<const N: usize> fmt::Debug for BoundedAscii<N> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("BoundedAscii")
            .field(&self.as_str())
            .finish()
    }
}

macro_rules! bounded_identity {
    ($name:ident, $maximum:ident, $allowed:expr, $reject_zero:expr) => {
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(BoundedAscii<$maximum>);

        impl $name {
            pub(crate) fn parse(value: String) -> Result<Self, BoundedTextError> {
                let parsed = BoundedAscii::parse(value, $allowed)?;
                if $reject_zero && parsed.as_str().bytes().all(|byte| byte == b'0') {
                    return Err(BoundedTextError::ZeroIdentity);
                }
                Ok(Self(parsed))
            }

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter
                    .debug_tuple(stringify!($name))
                    .field(&self.as_str())
                    .finish()
            }
        }
    };
}

fn stable_id_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':')
}

fn provenance_byte(byte: u8) -> bool {
    stable_id_byte(byte) || matches!(byte, b'+' | b'/' | b'@')
}

fn endpoint_byte(byte: u8) -> bool {
    provenance_byte(byte) || matches!(byte, b'[' | b']')
}

bounded_identity!(RobotId, MAX_ROBOT_ID_BYTES, stable_id_byte, true);
bounded_identity!(
    BuildProvenance,
    MAX_BUILD_PROVENANCE_BYTES,
    provenance_byte,
    false
);
bounded_identity!(ArtifactId, MAX_ARTIFACT_ID_BYTES, stable_id_byte, true);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OakMxid(BoundedAscii<MAX_OAK_MXID_BYTES>);

impl OakMxid {
    pub(crate) fn parse(mut value: String) -> Result<Self, BoundedTextError> {
        const MINIMUM_MXID_BYTES: usize = 8;
        value.make_ascii_uppercase();
        let parsed = BoundedAscii::parse(value, |byte| byte.is_ascii_hexdigit())?;
        if parsed.as_str().len() < MINIMUM_MXID_BYTES {
            return Err(BoundedTextError::OakMxidTooShort {
                actual_bytes: parsed.as_str().len(),
                minimum_bytes: MINIMUM_MXID_BYTES,
            });
        }
        if parsed.as_str().bytes().all(|byte| byte == b'0') {
            return Err(BoundedTextError::ZeroIdentity);
        }
        Ok(Self(parsed))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Debug for OakMxid {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("OakMxid")
            .field(&self.as_str())
            .finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PersistentSerialPath(BoundedAscii<MAX_SERIAL_BY_ID_PATH_BYTES>);

impl PersistentSerialPath {
    pub(crate) fn parse(value: String) -> Result<Self, BoundedTextError> {
        if !value.starts_with(SERIAL_BY_ID_PREFIX) {
            return Err(BoundedTextError::NotPersistentSerialById);
        }
        let suffix = &value[SERIAL_BY_ID_PREFIX.len()..];
        if suffix.is_empty() {
            return Err(BoundedTextError::NotPersistentSerialById);
        }
        if suffix.contains('/') {
            return Err(BoundedTextError::SerialByIdHasNestedComponent);
        }
        if matches!(suffix, "." | "..") {
            return Err(BoundedTextError::InvalidSerialByIdComponent);
        }
        BoundedAscii::parse(value, |byte| stable_id_byte(byte) || byte == b'/').map(Self)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Debug for PersistentSerialPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("PersistentSerialPath")
            .field(&self.as_str())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ControlEndpointTransport {
    Unix,
    Tcp,
    Udp,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ControlEndpointIdentity {
    text: BoundedAscii<MAX_CONTROL_ENDPOINT_ID_BYTES>,
    transport: ControlEndpointTransport,
    socket_addr: Option<SocketAddr>,
}

impl ControlEndpointIdentity {
    pub(crate) fn parse(value: String) -> Result<Self, BoundedTextError> {
        let parsed = BoundedAscii::parse(value, endpoint_byte)?;
        let text = parsed.as_str();
        let (transport, socket_addr) = if text.strip_prefix("unix:").is_some_and(valid_unix_path) {
            (ControlEndpointTransport::Unix, None)
        } else if let Some(socket) = parse_loopback_socket(text, "tcp://") {
            (ControlEndpointTransport::Tcp, Some(socket))
        } else if let Some(socket) = parse_loopback_socket(text, "udp://") {
            (ControlEndpointTransport::Udp, Some(socket))
        } else {
            return Err(BoundedTextError::InvalidControlEndpoint);
        };
        Ok(Self {
            text: parsed,
            transport,
            socket_addr,
        })
    }

    pub fn as_str(&self) -> &str {
        self.text.as_str()
    }

    pub const fn transport(self) -> ControlEndpointTransport {
        self.transport
    }

    pub const fn socket_addr(self) -> Option<SocketAddr> {
        self.socket_addr
    }
}

impl fmt::Debug for ControlEndpointIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("ControlEndpointIdentity")
            .field(&self.as_str())
            .finish()
    }
}

fn parse_loopback_socket(text: &str, scheme: &str) -> Option<SocketAddr> {
    let socket = text.strip_prefix(scheme)?.parse::<SocketAddr>().ok()?;
    (socket.ip().is_loopback() && socket.port() != 0 && format!("{scheme}{socket}") == text)
        .then_some(socket)
}

fn valid_unix_path(value: &str) -> bool {
    value.starts_with('/')
        && value.len() > 1
        && !value.ends_with('/')
        && value[1..]
            .split('/')
            .all(|component| !component.is_empty() && !matches!(component, "." | ".."))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sha256Id([u8; 32]);

impl Sha256Id {
    pub(crate) fn try_new(bytes: [u8; 32]) -> Option<Self> {
        (bytes != [0; 32]).then_some(Self(bytes))
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_identity_accepts_only_one_persistent_by_id_component() {
        assert!(PersistentSerialPath::parse("/dev/serial/by-id/usb-head_1".into()).is_ok());
        for invalid in [
            "ttyUSB0",
            "/dev/ttyUSB0",
            "/dev/serial/by-id/",
            "/dev/serial/by-id/.",
            "/dev/serial/by-id/..",
            "/dev/serial/by-id/usb/head",
        ] {
            assert!(
                PersistentSerialPath::parse(invalid.into()).is_err(),
                "{invalid}"
            );
        }
    }

    #[test]
    fn endpoints_are_explicit_local_identities() {
        for valid in [
            "unix:/run/kiko/robot.sock",
            "tcp://127.0.0.1:5000",
            "tcp://[::1]:5000",
            "udp://127.0.0.1:5000",
            "udp://[::1]:5000",
        ] {
            assert!(
                ControlEndpointIdentity::parse(valid.into()).is_ok(),
                "{valid}"
            );
        }
        for invalid in [
            "/run/kiko/robot.sock",
            "tcp://0.0.0.0:5000",
            "tcp://127.0.0.1:0",
            "tcp://127.0.0.1:05000",
            "tcp://127.0.0.1:70000",
            "udp://0.0.0.0:5000",
            "udp://127.0.0.1:05000",
            "unix:/run/../tmp/robot.sock",
            "unix:/run//kiko.sock",
        ] {
            assert!(
                ControlEndpointIdentity::parse(invalid.into()).is_err(),
                "{invalid}"
            );
        }
    }
}
