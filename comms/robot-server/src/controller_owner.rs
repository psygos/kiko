//! Explicit lifecycle ownership for the exact KRP2 serial and UDP resources.

use std::fmt;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::UdpSocket;
use tokio::sync::oneshot;
use tokio::task::{JoinError, JoinHandle};

use crate::actuation_v2::{
    self, ActuationActorError, ActuationShutdownHandle, ActuationShutdownReason,
    ActuationStartError, ActuationTelemetry, NoopActuationTelemetry, UdpServiceError,
};
use crate::config::ControllerServerConfigV1;

/// The sole in-process owner of one exact V2 controller and its loopback
/// command endpoint.
///
/// Construction does not return until the UDP address is bound and the exact
/// configured serial path is open with OS-level exclusive ownership. The
/// owner retains both task handles. Call [`Self::shutdown`] for an intentional
/// bounded stop or [`Self::join`] to supervise an unexpected task exit.
#[must_use = "dropping the owner aborts both tasks; call shutdown or join and inspect the result"]
#[derive(Debug)]
pub struct V2ControllerOwner {
    command_address: SocketAddr,
    udp_shutdown: Option<oneshot::Sender<()>>,
    actuation_shutdown: ActuationShutdownHandle,
    actuation_task: Option<JoinHandle<Result<(), ActuationActorError>>>,
    udp_task: Option<JoinHandle<Result<(), UdpServiceError>>>,
}

impl V2ControllerOwner {
    /// Bind `command_bind`, exclusively claim `controller.serial_device()`,
    /// and start the two owned tasks without a secondary telemetry sink.
    pub async fn start(
        controller: ControllerServerConfigV1,
        command_bind: SocketAddr,
    ) -> Result<Self, V2ControllerOwnerStartError> {
        Self::start_with_telemetry(controller, command_bind, Arc::new(NoopActuationTelemetry)).await
    }

    /// Equivalent to [`Self::start`], with a prompt non-blocking observer for
    /// controller snapshots and observational odometry.
    pub async fn start_with_telemetry(
        controller: ControllerServerConfigV1,
        command_bind: SocketAddr,
        telemetry: Arc<dyn ActuationTelemetry>,
    ) -> Result<Self, V2ControllerOwnerStartError> {
        let socket = actuation_v2::bind_udp_socket(command_bind)
            .await
            .map_err(V2ControllerOwnerStartError::CommandEndpoint)?;
        let command_address = socket
            .local_addr()
            .map_err(V2ControllerOwnerStartError::ReadBoundCommandAddress)?;
        let (actuation, actuation_task) = actuation_v2::start_serial_actor(controller, telemetry)
            .await
            .map_err(V2ControllerOwnerStartError::Controller)?;
        Ok(Self::from_acquired_resources(
            command_address,
            socket,
            actuation,
            actuation_task,
        ))
    }

    /// The exact local address held by this owner.
    pub const fn command_address(&self) -> SocketAddr {
        self.command_address
    }

    /// Request an intentional stop and wait at most `timeout` for each task
    /// to complete its coordinated shutdown.
    ///
    /// The actor receives a priority signal before its mailbox is closed, so
    /// requests already queued by aborted UDP exchanges cannot apply after
    /// shutdown begins.
    ///
    /// If the deadline expires, every unfinished task is aborted and joined
    /// before this method returns.
    pub async fn shutdown(
        mut self,
        timeout: Duration,
    ) -> Result<(), V2ControllerOwnerTerminationError> {
        self.request_shutdown(ActuationShutdownReason::Operator);
        let (actuation, udp) = self.collect_until(timeout, None, None).await;
        if actuation.is_clean() && udp.is_clean() {
            Ok(())
        } else {
            Err(V2ControllerOwnerTerminationError {
                trigger: V2ControllerOwnerExitTrigger::ExplicitShutdown,
                actuation,
                udp,
            })
        }
    }

    /// Supervise both owned tasks until an explicit shutdown request or the
    /// first unexpected task exit.
    ///
    /// This is the process-lifecycle entry point for an embedding service. It
    /// permits that service to retain a small request handle while this owner
    /// remains the sole holder of the task and transport resources. An
    /// explicit request returns `Ok(())` only when both tasks shut down
    /// cleanly. A task exit always reports the exact trigger and both terminal
    /// outcomes after stopping its sibling.
    pub async fn run_until_shutdown(
        mut self,
        shutdown: oneshot::Receiver<()>,
        shutdown_timeout: Duration,
    ) -> Result<(), V2ControllerOwnerTerminationError> {
        tokio::pin!(shutdown);
        let (trigger, actuation, udp) = tokio::select! {
            _ = &mut shutdown => (
                V2ControllerOwnerExitTrigger::ExplicitShutdown,
                None,
                None,
            ),
            result = Self::take_actuation_result(&mut self.actuation_task) => (
                V2ControllerOwnerExitTrigger::ActuationTask,
                Some(result),
                None,
            ),
            result = Self::take_udp_result(&mut self.udp_task) => (
                V2ControllerOwnerExitTrigger::UdpTask,
                None,
                Some(result),
            ),
        };
        let reason = if trigger == V2ControllerOwnerExitTrigger::ExplicitShutdown {
            ActuationShutdownReason::Operator
        } else {
            ActuationShutdownReason::SiblingFailure
        };
        self.request_shutdown(reason);
        let (actuation, udp) = self.collect_until(shutdown_timeout, actuation, udp).await;
        if trigger == V2ControllerOwnerExitTrigger::ExplicitShutdown
            && actuation.is_clean()
            && udp.is_clean()
        {
            Ok(())
        } else {
            Err(V2ControllerOwnerTerminationError {
                trigger,
                actuation,
                udp,
            })
        }
    }

    /// Wait for either owned task to end unexpectedly, stop its sibling, and
    /// collect both terminal outcomes within `shutdown_timeout`.
    ///
    /// A live controller owner has no spontaneous successful terminal state,
    /// so this method always returns a typed combined error after a task exits.
    pub async fn join(
        mut self,
        shutdown_timeout: Duration,
    ) -> Result<(), V2ControllerOwnerTerminationError> {
        let (trigger, actuation, udp) = tokio::select! {
            result = Self::take_actuation_result(&mut self.actuation_task) => (
                V2ControllerOwnerExitTrigger::ActuationTask,
                Some(result),
                None,
            ),
            result = Self::take_udp_result(&mut self.udp_task) => (
                V2ControllerOwnerExitTrigger::UdpTask,
                None,
                Some(result),
            ),
        };
        self.request_shutdown(ActuationShutdownReason::SiblingFailure);
        let (actuation, udp) = self.collect_until(shutdown_timeout, actuation, udp).await;
        Err(V2ControllerOwnerTerminationError {
            trigger,
            actuation,
            udp,
        })
    }

    fn from_acquired_resources(
        command_address: SocketAddr,
        socket: UdpSocket,
        actuation: actuation_v2::ActuationHandle,
        actuation_task: JoinHandle<Result<(), ActuationActorError>>,
    ) -> Self {
        let (udp_shutdown, shutdown) = oneshot::channel();
        let actuation_shutdown = actuation.shutdown_handle();
        let udp_task = tokio::spawn(actuation_v2::udp_service_on_socket_until(
            socket, actuation, shutdown,
        ));
        Self {
            command_address,
            udp_shutdown: Some(udp_shutdown),
            actuation_shutdown,
            actuation_task: Some(actuation_task),
            udp_task: Some(udp_task),
        }
    }

    fn request_shutdown(&mut self, reason: ActuationShutdownReason) {
        self.actuation_shutdown.request(reason);
        if let Some(shutdown) = self.udp_shutdown.take() {
            let _ = shutdown.send(());
        }
    }

    async fn collect_until(
        &mut self,
        timeout: Duration,
        mut actuation: Option<ActuationTaskOutcome>,
        mut udp: Option<UdpTaskOutcome>,
    ) -> (ActuationTaskOutcome, UdpTaskOutcome) {
        let deadline = tokio::time::Instant::now()
            .checked_add(timeout)
            .unwrap_or_else(tokio::time::Instant::now);
        while actuation.is_none() || udp.is_none() {
            tokio::select! {
                result = Self::take_actuation_result(&mut self.actuation_task),
                    if actuation.is_none() =>
                {
                    actuation = Some(result);
                }
                result = Self::take_udp_result(&mut self.udp_task),
                    if udp.is_none() =>
                {
                    udp = Some(result);
                }
                () = tokio::time::sleep_until(deadline) => {
                    break;
                }
            }
        }

        if actuation.is_none() {
            Self::abort_and_join(&mut self.actuation_task).await;
            actuation = Some(ActuationTaskOutcome::TimedOut);
        }
        if udp.is_none() {
            Self::abort_and_join(&mut self.udp_task).await;
            udp = Some(UdpTaskOutcome::TimedOut);
        }

        (
            actuation.expect("actuation outcome is assigned above"),
            udp.expect("UDP outcome is assigned above"),
        )
    }

    async fn take_actuation_result(
        task: &mut Option<JoinHandle<Result<(), ActuationActorError>>>,
    ) -> ActuationTaskOutcome {
        let result = task
            .as_mut()
            .expect("actuation task exists until its result is collected")
            .await;
        let _ = task.take();
        match result {
            Ok(Ok(())) => ActuationTaskOutcome::Clean,
            Ok(Err(source)) => ActuationTaskOutcome::Failed(source),
            Err(source) => ActuationTaskOutcome::JoinFailed(source),
        }
    }

    async fn take_udp_result(
        task: &mut Option<JoinHandle<Result<(), UdpServiceError>>>,
    ) -> UdpTaskOutcome {
        let result = task
            .as_mut()
            .expect("UDP task exists until its result is collected")
            .await;
        let _ = task.take();
        match result {
            Ok(Ok(())) => UdpTaskOutcome::Clean,
            Ok(Err(source)) => UdpTaskOutcome::Failed(source),
            Err(source) => UdpTaskOutcome::JoinFailed(source),
        }
    }

    async fn abort_and_join<T>(task: &mut Option<JoinHandle<T>>) {
        if let Some(task) = task.take() {
            task.abort();
            let _ = task.await;
        }
    }
}

impl Drop for V2ControllerOwner {
    fn drop(&mut self) {
        self.request_shutdown(ActuationShutdownReason::SiblingFailure);
        if let Some(task) = self.udp_task.take() {
            task.abort();
        }
        if let Some(task) = self.actuation_task.take() {
            task.abort();
        }
    }
}

#[derive(Debug)]
pub enum V2ControllerOwnerStartError {
    CommandEndpoint(UdpServiceError),
    ReadBoundCommandAddress(std::io::Error),
    Controller(ActuationStartError),
}

impl fmt::Display for V2ControllerOwnerStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CommandEndpoint(source) => {
                write!(formatter, "cannot acquire V2 command endpoint: {source}")
            }
            Self::ReadBoundCommandAddress(source) => {
                write!(formatter, "cannot read bound V2 command address: {source}")
            }
            Self::Controller(source) => {
                write!(formatter, "cannot acquire exact V2 controller: {source}")
            }
        }
    }
}

impl std::error::Error for V2ControllerOwnerStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CommandEndpoint(source) => Some(source),
            Self::ReadBoundCommandAddress(source) => Some(source),
            Self::Controller(source) => Some(source),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum V2ControllerOwnerExitTrigger {
    ExplicitShutdown,
    ActuationTask,
    UdpTask,
}

#[derive(Debug)]
pub enum ActuationTaskOutcome {
    Clean,
    Failed(ActuationActorError),
    JoinFailed(JoinError),
    TimedOut,
}

impl ActuationTaskOutcome {
    pub const fn is_clean(&self) -> bool {
        matches!(self, Self::Clean)
    }
}

impl fmt::Display for ActuationTaskOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clean => formatter.write_str("clean"),
            Self::Failed(source) => write!(formatter, "failed: {source}"),
            Self::JoinFailed(source) => write!(formatter, "task join failed: {source}"),
            Self::TimedOut => formatter.write_str("timed out and was aborted"),
        }
    }
}

#[derive(Debug)]
pub enum UdpTaskOutcome {
    Clean,
    Failed(UdpServiceError),
    JoinFailed(JoinError),
    TimedOut,
}

impl UdpTaskOutcome {
    pub const fn is_clean(&self) -> bool {
        matches!(self, Self::Clean)
    }
}

impl fmt::Display for UdpTaskOutcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Clean => formatter.write_str("clean"),
            Self::Failed(source) => write!(formatter, "failed: {source}"),
            Self::JoinFailed(source) => write!(formatter, "task join failed: {source}"),
            Self::TimedOut => formatter.write_str("timed out and was aborted"),
        }
    }
}

#[derive(Debug)]
pub struct V2ControllerOwnerTerminationError {
    trigger: V2ControllerOwnerExitTrigger,
    actuation: ActuationTaskOutcome,
    udp: UdpTaskOutcome,
}

impl V2ControllerOwnerTerminationError {
    pub const fn trigger(&self) -> V2ControllerOwnerExitTrigger {
        self.trigger
    }

    pub const fn actuation(&self) -> &ActuationTaskOutcome {
        &self.actuation
    }

    pub const fn udp(&self) -> &UdpTaskOutcome {
        &self.udp
    }
}

impl fmt::Display for V2ControllerOwnerTerminationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "V2 controller owner ended after {:?}: actuation {}; UDP {}",
            self.trigger, self.actuation, self.udp
        )
    }
}

impl std::error::Error for V2ControllerOwnerTerminationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.actuation {
            ActuationTaskOutcome::Failed(source) => return Some(source),
            ActuationTaskOutcome::JoinFailed(source) => return Some(source),
            ActuationTaskOutcome::Clean | ActuationTaskOutcome::TimedOut => {}
        }
        match &self.udp {
            UdpTaskOutcome::Failed(source) => Some(source),
            UdpTaskOutcome::JoinFailed(source) => Some(source),
            UdpTaskOutcome::Clean | UdpTaskOutcome::TimedOut => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;

    use crate::actuation_v2::NoopActuationTelemetry;
    use tokio::io::AsyncReadExt;

    fn exact_controller(serial_device: &Path) -> ControllerServerConfigV1 {
        let json = format!(
            r#"{{
                "schema_version": 1,
                "serial_device": {},
                "controller_uid_hex": "00112233445566778899aabb",
                "firmware_abi": 2,
                "firmware_build_id": 42,
                "actuator_config_fingerprint_hex": "11223344556677889900aabbccddeeff",
                "hardware_profile_claim_id": "kiko-driver-profile-v1",
                "heartbeat_period_ms": 20,
                "maximum_heartbeat_age_ms": 60,
                "serial_applied_ack_timeout_ms": 30,
                "controller_clock_abs_error_ppm_bound": 50000,
                "deadline_quantization_margin_ms": 2,
                "expected_max_abs_pwm_percent": 50,
                "expected_pwm_frequency_hz": 20000,
                "expected_watchdog_nominal_timeout_ms": 250,
                "expected_neutral_output": "both_low",
                "expected_physical_stop_semantics": "coast_verified"
            }}"#,
            serde_json::to_string(serial_device).expect("serial path JSON"),
        );
        ControllerServerConfigV1::parse_json(json.as_bytes()).expect("exact controller fixture")
    }

    async fn fake_owner() -> (V2ControllerOwner, tokio::io::DuplexStream) {
        fake_owner_with_serial_capacity(4_096).await
    }

    async fn fake_owner_with_serial_capacity(
        serial_capacity: usize,
    ) -> (V2ControllerOwner, tokio::io::DuplexStream) {
        let socket = UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("loopback UDP fixture");
        let command_address = socket.local_addr().expect("bound fixture address");
        let (actor_stream, controller_stream) = tokio::io::duplex(serial_capacity);
        let config = exact_controller(Path::new("/dev/fake-kiko-controller"));
        let (actuation, actuation_task) = actuation_v2::spawn_actor_for_test(
            actor_stream,
            &config,
            Arc::new(NoopActuationTelemetry),
        )
        .expect("valid actor config");
        (
            V2ControllerOwner::from_acquired_resources(
                command_address,
                socket,
                actuation,
                actuation_task,
            ),
            controller_stream,
        )
    }

    #[tokio::test]
    async fn explicit_shutdown_joins_both_owned_tasks_and_releases_udp() {
        let (owner, _controller) = fake_owner().await;
        let command_address = owner.command_address();
        assert!(std::net::UdpSocket::bind(command_address).is_err());

        owner
            .shutdown(Duration::from_millis(250))
            .await
            .expect("both owner tasks shut down cleanly");

        let rebound =
            std::net::UdpSocket::bind(command_address).expect("shutdown released UDP ownership");
        drop(rebound);
    }

    #[tokio::test]
    async fn supervised_explicit_shutdown_joins_both_owned_tasks_and_releases_udp() {
        let (owner, _controller) = fake_owner().await;
        let command_address = owner.command_address();
        let (request_shutdown, shutdown) = oneshot::channel();
        let task = tokio::spawn(owner.run_until_shutdown(shutdown, Duration::from_millis(250)));

        request_shutdown
            .send(())
            .expect("live supervision task receives shutdown");
        task.await
            .expect("supervision task joins")
            .expect("both owner tasks shut down cleanly");

        let rebound =
            std::net::UdpSocket::bind(command_address).expect("shutdown released UDP ownership");
        drop(rebound);
    }

    #[tokio::test]
    async fn supervised_actor_failure_stops_udp_and_reports_the_exact_trigger() {
        let (owner, controller) = fake_owner().await;
        let command_address = owner.command_address();
        let (_request_shutdown, shutdown) = oneshot::channel();
        drop(controller);

        let error = owner
            .run_until_shutdown(shutdown, Duration::from_millis(250))
            .await
            .expect_err("serial EOF is an owner failure");
        assert_eq!(error.trigger(), V2ControllerOwnerExitTrigger::ActuationTask);
        assert!(matches!(
            error.actuation(),
            ActuationTaskOutcome::Failed(ActuationActorError::SerialEof)
        ));
        assert!(matches!(error.udp(), UdpTaskOutcome::Clean));

        let rebound =
            std::net::UdpSocket::bind(command_address).expect("failure released UDP ownership");
        drop(rebound);
    }

    #[tokio::test]
    async fn blocked_serial_shutdown_is_aborted_joined_and_reported_without_detaching() {
        let (owner, mut controller) = fake_owner_with_serial_capacity(1).await;
        let command_address = owner.command_address();

        let error = owner
            .shutdown(Duration::from_millis(25))
            .await
            .expect_err("blocked best-effort stop must hit the shutdown bound");
        assert_eq!(
            error.trigger(),
            V2ControllerOwnerExitTrigger::ExplicitShutdown
        );
        assert!(matches!(error.actuation(), ActuationTaskOutcome::TimedOut));
        assert!(matches!(error.udp(), UdpTaskOutcome::Clean));

        let mut serial_tail = Vec::new();
        tokio::time::timeout(
            Duration::from_millis(250),
            controller.read_to_end(&mut serial_tail),
        )
        .await
        .expect("aborted actor released its serial transport")
        .expect("read fake serial EOF");
        let rebound =
            std::net::UdpSocket::bind(command_address).expect("timeout released UDP ownership");
        drop(rebound);
    }

    #[tokio::test]
    async fn actor_failure_stops_udp_and_preserves_both_terminal_outcomes() {
        let (owner, controller) = fake_owner().await;
        let command_address = owner.command_address();
        drop(controller);

        let error = owner
            .join(Duration::from_millis(250))
            .await
            .expect_err("serial EOF is an owner failure");
        assert_eq!(error.trigger(), V2ControllerOwnerExitTrigger::ActuationTask);
        assert!(matches!(
            error.actuation(),
            ActuationTaskOutcome::Failed(ActuationActorError::SerialEof)
        ));
        assert!(matches!(error.udp(), UdpTaskOutcome::Clean));

        let rebound =
            std::net::UdpSocket::bind(command_address).expect("failure released UDP ownership");
        drop(rebound);
    }

    #[tokio::test]
    async fn serial_claim_failure_releases_the_already_bound_udp_endpoint() {
        let reservation =
            std::net::UdpSocket::bind("127.0.0.1:0").expect("reserve loopback address");
        let command_address = reservation.local_addr().expect("reserved address");
        drop(reservation);
        let config = exact_controller(Path::new(
            "/this/path/must/not/exist/kiko-controller-for-owner-test",
        ));

        let error = V2ControllerOwner::start(config, command_address)
            .await
            .expect_err("missing serial device must fail startup");
        assert!(matches!(
            error,
            V2ControllerOwnerStartError::Controller(ActuationStartError::OpenSerial(_))
        ));

        let rebound = std::net::UdpSocket::bind(command_address)
            .expect("failed transactional startup released UDP");
        drop(rebound);
    }

    #[tokio::test]
    async fn non_loopback_bind_rejects_before_touching_the_serial_path() {
        let config = exact_controller(Path::new(
            "/this/path/must/not/exist/kiko-controller-for-owner-test",
        ));
        let command_bind = "0.0.0.0:48123".parse().expect("fixture bind");

        let error = V2ControllerOwner::start(config, command_bind)
            .await
            .expect_err("non-loopback bind must fail");
        assert!(matches!(
            error,
            V2ControllerOwnerStartError::CommandEndpoint(
                UdpServiceError::BindMustBeLoopback(address)
            ) if address == command_bind
        ));
    }
}
