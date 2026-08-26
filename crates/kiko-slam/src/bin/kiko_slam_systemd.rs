//! Canonical systemd readiness and watchdog boundary for the Nano agent.
//!
//! The watchdog is deliberately polled by the OAK capture loop itself. A kick
//! is admitted only after the sole accessory owner has completed another
//! four-joint health transaction, so neither an independent timer nor a PID
//! check can report a frozen robot loop as healthy.

use std::ffi::OsString;
use std::fmt;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::net::{SocketAddr, UnixDatagram};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[cfg(target_os = "linux")]
use std::os::linux::net::SocketAddrExt;

use kiko_slam::navigation::{
    NanoAccessoryLoopLivenessObserver, NanoAccessoryLoopLivenessSnapshot, NanoAccessoryOwnerState,
};

const EXPECTED_WATCHDOG_TIMEOUT: Duration = Duration::from_secs(60);
const WATCHDOG_CADENCE_DIVISOR: u32 = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
enum SystemdNotifyAddress {
    Pathname(PathBuf),
    #[cfg(target_os = "linux")]
    Abstract(Vec<u8>),
}

impl SystemdNotifyAddress {
    fn parse(value: OsString) -> Result<Self, NanoSystemdSupervisionConfigError> {
        let bytes = value.as_os_str().as_bytes();
        if bytes.is_empty() {
            return Err(NanoSystemdSupervisionConfigError::EmptyNotifySocket);
        }
        if bytes.contains(&0) {
            return Err(NanoSystemdSupervisionConfigError::NotifySocketContainsNul);
        }
        if bytes[0] == b'@' {
            #[cfg(target_os = "linux")]
            {
                let name = bytes[1..].to_vec();
                if name.is_empty() {
                    return Err(NanoSystemdSupervisionConfigError::EmptyAbstractNotifySocket);
                }
                SocketAddr::from_abstract_name(&name).map_err(|source| {
                    NanoSystemdSupervisionConfigError::InvalidNotifySocket { source }
                })?;
                return Ok(Self::Abstract(name));
            }
            #[cfg(not(target_os = "linux"))]
            {
                return Err(NanoSystemdSupervisionConfigError::AbstractNotifySocketUnsupported);
            }
        }

        let path = PathBuf::from(value);
        SocketAddr::from_pathname(&path)
            .map_err(|source| NanoSystemdSupervisionConfigError::InvalidNotifySocket { source })?;
        Ok(Self::Pathname(path))
    }
}

#[derive(Debug)]
pub(crate) enum NanoSystemdSupervisionConfigError {
    WatchdogWithoutNotifySocket,
    EmptyNotifySocket,
    NotifySocketContainsNul,
    #[cfg(target_os = "linux")]
    EmptyAbstractNotifySocket,
    #[cfg(not(target_os = "linux"))]
    AbstractNotifySocketUnsupported,
    InvalidNotifySocket {
        source: io::Error,
    },
    MissingWatchdogUsec,
    NonUnicodeWatchdogUsec,
    InvalidWatchdogUsec,
    UnexpectedWatchdogTimeout {
        actual: Duration,
        expected: Duration,
    },
    NonUnicodeWatchdogPid,
    InvalidWatchdogPid,
    WatchdogPidMismatch {
        configured: u32,
        actual: u32,
    },
    NotifySocketOpen {
        source: io::Error,
    },
}

impl fmt::Display for NanoSystemdSupervisionConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("invalid Nano systemd supervision boundary: ")?;
        match self {
            Self::WatchdogWithoutNotifySocket => {
                formatter.write_str("WATCHDOG_USEC/WATCHDOG_PID exists without NOTIFY_SOCKET")
            }
            Self::EmptyNotifySocket => formatter.write_str("NOTIFY_SOCKET is empty"),
            Self::NotifySocketContainsNul => {
                formatter.write_str("NOTIFY_SOCKET contains an embedded NUL")
            }
            #[cfg(target_os = "linux")]
            Self::EmptyAbstractNotifySocket => {
                formatter.write_str("abstract NOTIFY_SOCKET has an empty name")
            }
            #[cfg(not(target_os = "linux"))]
            Self::AbstractNotifySocketUnsupported => {
                formatter.write_str("abstract NOTIFY_SOCKET is Linux-only")
            }
            Self::InvalidNotifySocket { source } => {
                write!(
                    formatter,
                    "NOTIFY_SOCKET is not a Unix datagram address: {source}"
                )
            }
            Self::MissingWatchdogUsec => {
                formatter.write_str("NOTIFY_SOCKET exists without WATCHDOG_USEC")
            }
            Self::NonUnicodeWatchdogUsec => {
                formatter.write_str("WATCHDOG_USEC is not Unicode decimal text")
            }
            Self::InvalidWatchdogUsec => {
                formatter.write_str("WATCHDOG_USEC is not an unsigned 64-bit integer")
            }
            Self::UnexpectedWatchdogTimeout { actual, expected } => write!(
                formatter,
                "watchdog timeout {actual:?} does not match the qualified contract {expected:?}"
            ),
            Self::NonUnicodeWatchdogPid => {
                formatter.write_str("WATCHDOG_PID is not Unicode decimal text")
            }
            Self::InvalidWatchdogPid => {
                formatter.write_str("WATCHDOG_PID is not an unsigned 32-bit integer")
            }
            Self::WatchdogPidMismatch { configured, actual } => write!(
                formatter,
                "WATCHDOG_PID {configured} does not name this process {actual}"
            ),
            Self::NotifySocketOpen { source } => {
                write!(
                    formatter,
                    "cannot open the Unix datagram notify socket: {source}"
                )
            }
        }
    }
}

impl std::error::Error for NanoSystemdSupervisionConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidNotifySocket { source } | Self::NotifySocketOpen { source } => {
                Some(source)
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
pub(crate) enum NanoSystemdSupervisionError {
    OwnerNotRunning { state: NanoAccessoryOwnerState },
    MissingInitialHealthTransaction,
    WatchdogPolledBeforeReady,
    AccessoryLoopDidNotAdvance { last_completed: u64 },
    MonotonicDeadlineOverflow,
    NotifySend { source: io::Error },
}

impl fmt::Display for NanoSystemdSupervisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Nano systemd supervision failed: ")?;
        match self {
            Self::OwnerNotRunning { state } => {
                write!(formatter, "accessory owner state is {state:?}, not Running")
            }
            Self::MissingInitialHealthTransaction => formatter
                .write_str("accessory readiness has no completed four-joint health transaction"),
            Self::WatchdogPolledBeforeReady => {
                formatter.write_str("watchdog was polled before READY was published")
            }
            Self::AccessoryLoopDidNotAdvance { last_completed } => write!(
                formatter,
                "accessory loop did not advance beyond health transaction {last_completed}"
            ),
            Self::MonotonicDeadlineOverflow => {
                formatter.write_str("watchdog monotonic deadline overflowed")
            }
            Self::NotifySend { source } => {
                write!(formatter, "systemd notify datagram failed: {source}")
            }
        }
    }
}

impl std::error::Error for NanoSystemdSupervisionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NotifySend { source } => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct ParsedSystemdSupervision {
    address: SystemdNotifyAddress,
    watchdog_cadence: Duration,
}

impl ParsedSystemdSupervision {
    fn parse(
        notify_socket: Option<OsString>,
        watchdog_usec: Option<OsString>,
        watchdog_pid: Option<OsString>,
        actual_pid: u32,
    ) -> Result<Option<Self>, NanoSystemdSupervisionConfigError> {
        let Some(notify_socket) = notify_socket else {
            if watchdog_usec.is_some() || watchdog_pid.is_some() {
                return Err(NanoSystemdSupervisionConfigError::WatchdogWithoutNotifySocket);
            }
            return Ok(None);
        };
        let address = SystemdNotifyAddress::parse(notify_socket)?;
        let watchdog_usec =
            watchdog_usec.ok_or(NanoSystemdSupervisionConfigError::MissingWatchdogUsec)?;
        let watchdog_usec = watchdog_usec
            .to_str()
            .ok_or(NanoSystemdSupervisionConfigError::NonUnicodeWatchdogUsec)?
            .parse::<u64>()
            .map_err(|_| NanoSystemdSupervisionConfigError::InvalidWatchdogUsec)?;
        let watchdog_timeout = Duration::from_micros(watchdog_usec);
        if watchdog_timeout != EXPECTED_WATCHDOG_TIMEOUT {
            return Err(
                NanoSystemdSupervisionConfigError::UnexpectedWatchdogTimeout {
                    actual: watchdog_timeout,
                    expected: EXPECTED_WATCHDOG_TIMEOUT,
                },
            );
        }
        if let Some(watchdog_pid) = watchdog_pid {
            let configured = watchdog_pid
                .to_str()
                .ok_or(NanoSystemdSupervisionConfigError::NonUnicodeWatchdogPid)?
                .parse::<u32>()
                .map_err(|_| NanoSystemdSupervisionConfigError::InvalidWatchdogPid)?;
            if configured != actual_pid {
                return Err(NanoSystemdSupervisionConfigError::WatchdogPidMismatch {
                    configured,
                    actual: actual_pid,
                });
            }
        }
        Ok(Some(Self {
            address,
            watchdog_cadence: watchdog_timeout / WATCHDOG_CADENCE_DIVISOR,
        }))
    }
}

#[derive(Debug)]
pub(crate) struct SystemdNotifyTransport {
    socket: UnixDatagram,
    address: SystemdNotifyAddress,
}

impl SystemdNotifyTransport {
    fn new(address: SystemdNotifyAddress) -> Result<Self, NanoSystemdSupervisionConfigError> {
        let socket = UnixDatagram::unbound()
            .map_err(|source| NanoSystemdSupervisionConfigError::NotifySocketOpen { source })?;
        Ok(Self { socket, address })
    }

    fn send(&self, message: &[u8]) -> Result<(), NanoSystemdSupervisionError> {
        let sent = match &self.address {
            SystemdNotifyAddress::Pathname(path) => self.socket.send_to(message, path),
            #[cfg(target_os = "linux")]
            SystemdNotifyAddress::Abstract(name) => SocketAddr::from_abstract_name(name)
                .and_then(|address| self.socket.send_to_addr(message, &address)),
        }
        .map_err(|source| NanoSystemdSupervisionError::NotifySend { source })?;
        if sent != message.len() {
            return Err(NanoSystemdSupervisionError::NotifySend {
                source: io::Error::new(
                    io::ErrorKind::WriteZero,
                    "systemd notify datagram was only partially sent",
                ),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct WatchdogGate {
    cadence: Duration,
    last_completed: Option<u64>,
    next_due: Option<Instant>,
}

impl WatchdogGate {
    fn new(cadence: Duration) -> Self {
        Self {
            cadence,
            last_completed: None,
            next_due: None,
        }
    }

    fn require_running(
        snapshot: NanoAccessoryLoopLivenessSnapshot,
    ) -> Result<u64, NanoSystemdSupervisionError> {
        if snapshot.owner_state != NanoAccessoryOwnerState::Running {
            return Err(NanoSystemdSupervisionError::OwnerNotRunning {
                state: snapshot.owner_state,
            });
        }
        if snapshot.completed_health_transactions == 0 {
            return Err(NanoSystemdSupervisionError::MissingInitialHealthTransaction);
        }
        Ok(snapshot.completed_health_transactions)
    }

    fn ready(
        &mut self,
        now: Instant,
        snapshot: NanoAccessoryLoopLivenessSnapshot,
    ) -> Result<(), NanoSystemdSupervisionError> {
        let completed = Self::require_running(snapshot)?;
        self.last_completed = Some(completed);
        self.next_due = Some(
            now.checked_add(self.cadence)
                .ok_or(NanoSystemdSupervisionError::MonotonicDeadlineOverflow)?,
        );
        Ok(())
    }

    fn poll(
        &mut self,
        now: Instant,
        snapshot: NanoAccessoryLoopLivenessSnapshot,
    ) -> Result<bool, NanoSystemdSupervisionError> {
        let next_due = self
            .next_due
            .ok_or(NanoSystemdSupervisionError::WatchdogPolledBeforeReady)?;
        if now < next_due {
            return Ok(false);
        }
        let completed = Self::require_running(snapshot)?;
        let last_completed = self
            .last_completed
            .ok_or(NanoSystemdSupervisionError::WatchdogPolledBeforeReady)?;
        if completed <= last_completed {
            return Err(NanoSystemdSupervisionError::AccessoryLoopDidNotAdvance { last_completed });
        }
        self.last_completed = Some(completed);
        self.next_due = Some(
            now.checked_add(self.cadence)
                .ok_or(NanoSystemdSupervisionError::MonotonicDeadlineOverflow)?,
        );
        Ok(true)
    }
}

#[derive(Debug)]
pub(crate) enum NanoSystemdServiceSupervision {
    Disabled,
    Enabled {
        transport: SystemdNotifyTransport,
        gate: WatchdogGate,
    },
}

impl NanoSystemdServiceSupervision {
    pub(crate) fn from_process_environment() -> Result<Self, NanoSystemdSupervisionConfigError> {
        let parsed = ParsedSystemdSupervision::parse(
            std::env::var_os("NOTIFY_SOCKET"),
            std::env::var_os("WATCHDOG_USEC"),
            std::env::var_os("WATCHDOG_PID"),
            std::process::id(),
        )?;
        match parsed {
            None => Ok(Self::Disabled),
            Some(parsed) => Ok(Self::Enabled {
                transport: SystemdNotifyTransport::new(parsed.address)?,
                gate: WatchdogGate::new(parsed.watchdog_cadence),
            }),
        }
    }

    pub(crate) fn bind(
        self,
        liveness: NanoAccessoryLoopLivenessObserver,
    ) -> NanoSystemdRuntimeSupervision {
        NanoSystemdRuntimeSupervision {
            service: self,
            liveness,
        }
    }
}

#[derive(Debug)]
pub(crate) struct NanoSystemdRuntimeSupervision {
    service: NanoSystemdServiceSupervision,
    liveness: NanoAccessoryLoopLivenessObserver,
}

impl NanoSystemdRuntimeSupervision {
    pub(crate) fn notify_ready(&mut self, now: Instant) -> Result<(), NanoSystemdSupervisionError> {
        let Self { service, liveness } = self;
        match service {
            NanoSystemdServiceSupervision::Disabled => Ok(()),
            NanoSystemdServiceSupervision::Enabled { transport, gate } => {
                gate.ready(now, liveness.snapshot())?;
                transport.send(
                    b"READY=1\nSTATUS=Kiko integrated OAK, accessory, SLAM and controller owner ready",
                )
            }
        }
    }

    pub(crate) fn poll_watchdog(
        &mut self,
        now: Instant,
    ) -> Result<(), NanoSystemdSupervisionError> {
        let Self { service, liveness } = self;
        match service {
            NanoSystemdServiceSupervision::Disabled => Ok(()),
            NanoSystemdServiceSupervision::Enabled { transport, gate } => {
                if gate.poll(now, liveness.snapshot())? {
                    transport.send(
                        b"WATCHDOG=1\nSTATUS=Kiko OAK capture and sole accessory owner loops healthy",
                    )?;
                }
                Ok(())
            }
        }
    }

    pub(crate) fn notify_stopping(&mut self) -> Result<(), NanoSystemdSupervisionError> {
        match &mut self.service {
            NanoSystemdServiceSupervision::Disabled => Ok(()),
            NanoSystemdServiceSupervision::Enabled { transport, .. } => transport.send(
                b"STOPPING=1\nSTATUS=Kiko coordinated stop in progress; no replacement owner admitted",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn running(completed: u64) -> NanoAccessoryLoopLivenessSnapshot {
        NanoAccessoryLoopLivenessSnapshot {
            owner_state: NanoAccessoryOwnerState::Running,
            completed_health_transactions: completed,
        }
    }

    #[test]
    fn environment_boundary_is_disabled_only_when_all_systemd_facts_are_absent() {
        assert!(
            ParsedSystemdSupervision::parse(None, None, None, 7)
                .unwrap()
                .is_none()
        );
        assert!(matches!(
            ParsedSystemdSupervision::parse(None, Some(OsString::from("60000000")), None, 7),
            Err(NanoSystemdSupervisionConfigError::WatchdogWithoutNotifySocket)
        ));
        assert!(matches!(
            ParsedSystemdSupervision::parse(
                Some(OsString::from("/run/systemd/notify")),
                None,
                None,
                7,
            ),
            Err(NanoSystemdSupervisionConfigError::MissingWatchdogUsec)
        ));
    }

    #[test]
    fn watchdog_units_pid_and_exact_qualified_timeout_are_parsed_once() {
        let parsed = ParsedSystemdSupervision::parse(
            Some(OsString::from("/run/systemd/notify")),
            Some(OsString::from("60000000")),
            Some(OsString::from("42")),
            42,
        )
        .unwrap()
        .unwrap();
        assert_eq!(parsed.watchdog_cadence, Duration::from_secs(20));
        assert!(matches!(
            ParsedSystemdSupervision::parse(
                Some(OsString::from("/run/systemd/notify")),
                Some(OsString::from("3999999")),
                None,
                42,
            ),
            Err(NanoSystemdSupervisionConfigError::UnexpectedWatchdogTimeout { .. })
        ));
        assert!(matches!(
            ParsedSystemdSupervision::parse(
                Some(OsString::from("/run/systemd/notify")),
                Some(OsString::from("60000000")),
                Some(OsString::from("41")),
                42,
            ),
            Err(NanoSystemdSupervisionConfigError::WatchdogPidMismatch {
                configured: 41,
                actual: 42,
            })
        ));
    }

    #[test]
    fn watchdog_kick_requires_both_capture_poll_and_new_accessory_evidence() {
        let start = Instant::now();
        let mut gate = WatchdogGate::new(Duration::from_secs(20));
        gate.ready(start, running(1)).unwrap();
        assert!(
            !gate
                .poll(start + Duration::from_secs(19), running(99))
                .unwrap()
        );
        assert!(matches!(
            gate.poll(start + Duration::from_secs(20), running(1)),
            Err(NanoSystemdSupervisionError::AccessoryLoopDidNotAdvance { last_completed: 1 })
        ));

        let mut gate = WatchdogGate::new(Duration::from_secs(20));
        gate.ready(start, running(1)).unwrap();
        assert!(
            gate.poll(start + Duration::from_secs(20), running(2))
                .unwrap()
        );
        assert!(
            !gate
                .poll(start + Duration::from_secs(21), running(3))
                .unwrap()
        );
    }

    #[test]
    fn fault_latch_can_never_be_reported_as_watchdog_health() {
        let start = Instant::now();
        let mut gate = WatchdogGate::new(Duration::from_secs(20));
        gate.ready(start, running(1)).unwrap();
        assert!(matches!(
            gate.poll(
                start + Duration::from_secs(20),
                NanoAccessoryLoopLivenessSnapshot {
                    owner_state: NanoAccessoryOwnerState::FaultLatched,
                    completed_health_transactions: 2,
                },
            ),
            Err(NanoSystemdSupervisionError::OwnerNotRunning {
                state: NanoAccessoryOwnerState::FaultLatched,
            })
        ));
    }
}
