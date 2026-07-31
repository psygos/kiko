//! Process-owned HTTP/capability lifecycle for the Nano operator console.
//!
//! This module deliberately owns no camera, controller, serial port, or
//! command receiver. It publishes one fresh per-process capability, starts the
//! loopback-only HTTP frontend against an existing console handle, and removes
//! that exact capability only after the HTTP owner has stopped.

use std::fmt;
use std::net::SocketAddr;
use std::sync::Arc;

use super::OperatorConsoleAccessCapability;
use super::{
    AgentControlMonotonicOrigin, NanoOperatorConsoleConfig, OperatorConsoleBind,
    OperatorConsoleBindError, OperatorConsoleCapabilityCleanupEvidence,
    OperatorConsoleCapabilityPersistError, OperatorConsoleHandle, OperatorConsoleHttpConfigError,
    OperatorConsoleHttpJoinError, OperatorConsoleHttpServer, OperatorConsoleHttpServerConfig,
    OperatorConsoleHttpServerExit, OperatorConsoleHttpServerStartError,
    OperatorConsolePersistedAccessCapability,
};

/// Sole frontend owner for one production process.
///
/// The value is not cloneable. Explicit shutdown is preferred because it
/// returns both HTTP and inode-guarded capability cleanup evidence. `Drop`
/// performs the same ordering as a last-resort fail-closed cleanup.
#[must_use = "the console frontend must be shut down and its cleanup evidence inspected"]
#[derive(Debug)]
pub struct NanoOperatorConsoleFrontend {
    bound_address: SocketAddr,
    http: Option<OperatorConsoleHttpServer>,
    capability: Option<OperatorConsolePersistedAccessCapability>,
    terminal_evidence: Option<NanoOperatorConsoleFrontendShutdownEvidence>,
}

impl NanoOperatorConsoleFrontend {
    /// Publish a fresh capability and start the loopback HTTP owner.
    ///
    /// The supplied handle is a cloneable observational/submission frontend;
    /// command execution remains exclusively in the separately retained
    /// non-clone runtime adapter.
    pub fn start(
        config: &NanoOperatorConsoleConfig,
        clock: AgentControlMonotonicOrigin,
        handle: OperatorConsoleHandle,
    ) -> Result<Self, NanoOperatorConsoleFrontendStartError> {
        let bind = OperatorConsoleBind::parse(config.bind_address())
            .map_err(NanoOperatorConsoleFrontendStartError::Bind)?;
        let capability = OperatorConsoleAccessCapability::generate_and_persist_new(
            config.capability_path().as_path(),
        )
        .map_err(NanoOperatorConsoleFrontendStartError::Capability)?;
        let http_config = match OperatorConsoleHttpServerConfig::parse(
            bind,
            capability.access_capability(),
            clock,
            config.deadman_tick(),
        ) {
            Ok(config) => config,
            Err(source) => {
                return Err(NanoOperatorConsoleFrontendStartError::HttpConfig {
                    source,
                    capability_cleanup: capability.cleanup(),
                });
            }
        };
        let http = match OperatorConsoleHttpServer::start(http_config, Arc::new(handle)) {
            Ok(http) => http,
            Err(source) => {
                return Err(NanoOperatorConsoleFrontendStartError::HttpStart {
                    source,
                    capability_cleanup: capability.cleanup(),
                });
            }
        };
        Ok(Self {
            bound_address: http.bound_addr(),
            http: Some(http),
            capability: Some(capability),
            terminal_evidence: None,
        })
    }

    pub const fn bound_address(&self) -> SocketAddr {
        self.bound_address
    }

    /// Stop accepting new HTTP work without waiting. The production owner uses
    /// this before applying its terminal controller zero/disarm.
    pub fn request_shutdown(&mut self) {
        if let Some(http) = self.http.as_mut() {
            http.request_shutdown();
        }
    }

    /// Probe the HTTP owner without blocking. An exit before production
    /// lifecycle shutdown is a safety-significant frontend loss: the stopped
    /// thread is joined and its exact capability inode is removed immediately.
    pub fn poll_unexpected_exit(&mut self) -> Option<NanoOperatorConsoleFrontendShutdownEvidence> {
        if let Some(evidence) = self.terminal_evidence {
            return Some(evidence);
        }
        let result = match self.http.as_mut()?.try_join() {
            Ok(None) => return None,
            Ok(Some(exit)) => Ok(exit),
            Err(OperatorConsoleHttpJoinError::TimedOut) => return None,
            Err(source) => Err(source),
        };
        self.http.take();
        let capability = NanoOperatorConsoleCapabilityShutdownEvidence::Cleaned(
            self.capability
                .take()
                .expect("live HTTP owner retains its exact capability")
                .cleanup(),
        );
        let evidence = NanoOperatorConsoleFrontendShutdownEvidence {
            http: result,
            capability,
        };
        self.terminal_evidence = Some(evidence);
        Some(evidence)
    }

    /// Join HTTP, then remove only the exact capability inode published by this
    /// owner. On timeout both owners remain in `self` for an explicit retry.
    pub fn shutdown(&mut self) -> NanoOperatorConsoleFrontendShutdownEvidence {
        if let Some(evidence) = self.terminal_evidence {
            return evidence;
        }
        self.request_shutdown();
        let http = self
            .http
            .as_mut()
            .expect("owned console HTTP server is consumed exactly once")
            .shutdown();
        let capability = if matches!(http, Err(OperatorConsoleHttpJoinError::TimedOut)) {
            NanoOperatorConsoleCapabilityShutdownEvidence::RetainedWhileHttpOwnerLive
        } else {
            self.http.take();
            NanoOperatorConsoleCapabilityShutdownEvidence::Cleaned(
                self.capability
                    .take()
                    .expect("owned console capability is consumed exactly once")
                    .cleanup(),
            )
        };
        let evidence = NanoOperatorConsoleFrontendShutdownEvidence { http, capability };
        if !evidence.retains_live_http_owner() {
            self.terminal_evidence = Some(evidence);
        }
        evidence
    }
}

impl Drop for NanoOperatorConsoleFrontend {
    fn drop(&mut self) {
        if self.terminal_evidence.is_some() {
            return;
        }
        self.request_shutdown();
        let http_stopped = match self.http.as_mut().map(OperatorConsoleHttpServer::shutdown) {
            None | Some(Ok(_)) => true,
            Some(Err(OperatorConsoleHttpJoinError::TimedOut)) => false,
            Some(Err(
                OperatorConsoleHttpJoinError::AlreadyJoined
                | OperatorConsoleHttpJoinError::Panicked,
            )) => true,
        };
        if http_stopped {
            self.http.take();
            if let Some(capability) = self.capability.take() {
                let _ = capability.cleanup();
            }
        } else {
            // A last-resort drop cannot safely revoke a capability which a
            // detached live HTTP owner may still accept. Preserve both owners
            // until process exit instead of recording false cleanup.
            if let Some(http) = self.http.take() {
                std::mem::forget(http);
            }
            if let Some(capability) = self.capability.take() {
                std::mem::forget(capability);
            }
        }
    }
}

#[derive(Debug)]
pub enum NanoOperatorConsoleFrontendStartError {
    Bind(OperatorConsoleBindError),
    Capability(OperatorConsoleCapabilityPersistError),
    HttpConfig {
        source: OperatorConsoleHttpConfigError,
        capability_cleanup: OperatorConsoleCapabilityCleanupEvidence,
    },
    HttpStart {
        source: OperatorConsoleHttpServerStartError,
        capability_cleanup: OperatorConsoleCapabilityCleanupEvidence,
    },
}

impl fmt::Display for NanoOperatorConsoleFrontendStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bind(source) => source.fmt(formatter),
            Self::Capability(source) => source.fmt(formatter),
            Self::HttpConfig {
                source,
                capability_cleanup,
            } => write!(
                formatter,
                "operator-console HTTP configuration failed: {source}; capability cleanup: {capability_cleanup:?}"
            ),
            Self::HttpStart {
                source,
                capability_cleanup,
            } => write!(
                formatter,
                "operator-console HTTP startup failed: {source}; capability cleanup: {capability_cleanup:?}"
            ),
        }
    }
}

impl std::error::Error for NanoOperatorConsoleFrontendStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bind(source) => Some(source),
            Self::Capability(source) => Some(source),
            Self::HttpConfig { source, .. } => Some(source),
            Self::HttpStart { source, .. } => Some(source),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoOperatorConsoleFrontendShutdownEvidence {
    http: Result<OperatorConsoleHttpServerExit, OperatorConsoleHttpJoinError>,
    capability: NanoOperatorConsoleCapabilityShutdownEvidence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoOperatorConsoleCapabilityShutdownEvidence {
    Cleaned(OperatorConsoleCapabilityCleanupEvidence),
    RetainedWhileHttpOwnerLive,
}

impl NanoOperatorConsoleFrontendShutdownEvidence {
    pub const fn http(
        &self,
    ) -> &Result<OperatorConsoleHttpServerExit, OperatorConsoleHttpJoinError> {
        &self.http
    }

    pub const fn capability(&self) -> NanoOperatorConsoleCapabilityShutdownEvidence {
        self.capability
    }

    pub const fn retains_live_http_owner(&self) -> bool {
        matches!(
            self.capability,
            NanoOperatorConsoleCapabilityShutdownEvidence::RetainedWhileHttpOwnerLive
        )
    }

    /// A clean shutdown means the request owner joined gracefully and the
    /// exact capability entry was removed and its parent durably synced.
    pub fn is_clean(&self) -> bool {
        matches!(
            self.http,
            Ok(OperatorConsoleHttpServerExit {
                graceful_shutdown: true,
                forced_shutdown: false,
                clock_faulted: false,
                ..
            })
        ) && matches!(
            self.capability,
            NanoOperatorConsoleCapabilityShutdownEvidence::Cleaned(
                OperatorConsoleCapabilityCleanupEvidence::ExactEntryRemovedAndParentSynced
            )
        )
    }
}

impl fmt::Display for NanoOperatorConsoleFrontendShutdownEvidence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "operator-console shutdown: http={:?}; capability={:?}",
            self.http, self.capability
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::net::TcpListener;
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use super::super::{
        ConsoleSnapshotRevision, NanoOperatorConsoleConfig, OperatorConsoleLimits,
        OperatorConsoleSnapshot, operator_console,
    };
    use super::*;
    use crate::HostMonotonicTimestamp;

    struct PrivateTestDirectory(std::path::PathBuf);

    impl PrivateTestDirectory {
        fn create() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            let suffix = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "kiko-console-owner-monitor-test-{}-{suffix}",
                std::process::id()
            ));
            let mut builder = fs::DirBuilder::new();
            builder.mode(0o700);
            builder.create(&path).unwrap();
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
            Self(path)
        }
    }

    impl Drop for PrivateTestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn unexpected_http_owner_exit_is_joined_and_revokes_its_exact_capability() {
        let directory = PrivateTestDirectory::create();
        let capability_path = directory.0.join("operator-console.capability");
        let probe = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = probe.local_addr().unwrap();
        drop(probe);
        let config = NanoOperatorConsoleConfig::for_test(
            address,
            capability_path.clone(),
            Duration::from_millis(20),
        );
        let (handle, _receiver) = operator_console(
            OperatorConsoleLimits::default(),
            OperatorConsoleSnapshot::unknown(
                ConsoleSnapshotRevision::parse(1).unwrap(),
                super::super::ConsoleRuntimeAuthorityKind::ProductionExternalInterlocks,
            ),
        );
        let clock = AgentControlMonotonicOrigin::new(
            Instant::now(),
            HostMonotonicTimestamp::from_nanos(1_000_000),
        );
        let mut frontend = NanoOperatorConsoleFrontend::start(&config, clock, handle).unwrap();
        assert!(capability_path.exists());

        // Simulate an HTTP-owner exit not initiated through the production
        // frontend lifecycle.
        frontend
            .http
            .as_mut()
            .expect("live HTTP owner")
            .request_shutdown();
        let deadline = Instant::now() + Duration::from_secs(2);
        let evidence = loop {
            if let Some(evidence) = frontend.poll_unexpected_exit() {
                break evidence;
            }
            assert!(Instant::now() < deadline, "HTTP owner did not terminate");
            std::thread::sleep(Duration::from_millis(10));
        };

        assert!(evidence.is_clean(), "{evidence}");
        assert!(!capability_path.exists());
        assert_eq!(frontend.shutdown(), evidence);
    }
}
