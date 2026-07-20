use std::fmt;

use kiko_head_protocol::{
    CommandFrame, FullTelemetry, HeadJoint, ResponseParseError, TelemetryParseError,
    TorqueSwitchObservation, build_full_telemetry_read, build_torque_switch_read,
};

use crate::config::HeadProbeConfig;
use crate::framing::{FrameReadError, read_response_frame};
use crate::transport::{
    AsyncByteTransport, MonotonicClock, MonotonicTime, SerialConfigurationEvidence,
    SerialOpenError, SerialTransport, TokioClock, TransportFailure,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProbeRequest {
    TorqueSwitch,
    FullTelemetry,
}

/// Timing and framing facts for one successfully delimited response.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeResponseEvidence {
    request_started_at: MonotonicTime,
    request_sent_at: MonotonicTime,
    response_received_at: MonotonicTime,
    discarded_noise_bytes: u16,
}

impl ProbeResponseEvidence {
    pub const fn request_started_at(self) -> MonotonicTime {
        self.request_started_at
    }

    pub const fn request_sent_at(self) -> MonotonicTime {
        self.request_sent_at
    }

    pub const fn response_received_at(self) -> MonotonicTime {
        self.response_received_at
    }

    pub const fn discarded_noise_bytes(self) -> u16 {
        self.discarded_noise_bytes
    }
}

/// Typed observations for one canonical Kiko head joint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServoProbeReport {
    joint: HeadJoint,
    torque_switch: TorqueSwitchObservation,
    torque_response: ProbeResponseEvidence,
    telemetry: FullTelemetry,
    telemetry_response: ProbeResponseEvidence,
}

impl ServoProbeReport {
    pub const fn joint(self) -> HeadJoint {
        self.joint
    }

    pub const fn torque_switch(self) -> TorqueSwitchObservation {
        self.torque_switch
    }

    pub const fn torque_response(self) -> ProbeResponseEvidence {
        self.torque_response
    }

    pub const fn telemetry(self) -> FullTelemetry {
        self.telemetry
    }

    pub const fn telemetry_response(self) -> ProbeResponseEvidence {
        self.telemetry_response
    }
}

/// Complete read-only evidence from one exclusively owned serial session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadProbeReport {
    serial: SerialConfigurationEvidence,
    started_at: MonotonicTime,
    completed_at: MonotonicTime,
    servos: [ServoProbeReport; 4],
}

impl HeadProbeReport {
    pub const fn serial(&self) -> &SerialConfigurationEvidence {
        &self.serial
    }

    pub const fn started_at(&self) -> MonotonicTime {
        self.started_at
    }

    pub const fn completed_at(&self) -> MonotonicTime {
        self.completed_at
    }

    pub const fn servos(&self) -> &[ServoProbeReport; 4] {
        &self.servos
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadProbeError {
    RequestWrite {
        joint: HeadJoint,
        request: ProbeRequest,
        source: TransportFailure,
    },
    ResponseFrame {
        joint: HeadJoint,
        request: ProbeRequest,
        source: FrameReadError,
    },
    TorqueSwitch {
        joint: HeadJoint,
        source: ResponseParseError,
    },
    FullTelemetry {
        joint: HeadJoint,
        source: TelemetryParseError,
    },
}

impl fmt::Display for HeadProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "read-only Kiko head probe failed: {self:?}")
    }
}

impl std::error::Error for HeadProbeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RequestWrite { source, .. } => Some(source),
            Self::ResponseFrame { source, .. } => Some(source),
            Self::TorqueSwitch { source, .. } => Some(source),
            Self::FullTelemetry { source, .. } => Some(source),
        }
    }
}

#[derive(Debug)]
pub enum SerialHeadProbeError {
    Open(SerialOpenError),
    Probe {
        serial: SerialConfigurationEvidence,
        source: HeadProbeError,
    },
}

impl SerialHeadProbeError {
    pub const fn serial(&self) -> Option<&SerialConfigurationEvidence> {
        match self {
            Self::Open(_) => None,
            Self::Probe { serial, .. } => Some(serial),
        }
    }
}

impl fmt::Display for SerialHeadProbeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "could not complete serial head probe: {self:?}")
    }
}

impl std::error::Error for SerialHeadProbeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Open(source) => Some(source),
            Self::Probe { source, .. } => Some(source),
        }
    }
}

/// Open the configured exact device and issue only fixed READ requests for the
/// torque-switch and qualified telemetry windows of IDs 1 through 4.
pub async fn probe_serial_head(
    config: &HeadProbeConfig,
) -> Result<HeadProbeReport, SerialHeadProbeError> {
    let transport = SerialTransport::open(config.device()).map_err(SerialHeadProbeError::Open)?;
    let serial = transport.evidence().clone();
    let (started_at, completed_at, servos) =
        probe_with_transport(transport, TokioClock::new(), config)
            .await
            .map_err(|source| SerialHeadProbeError::Probe {
                serial: serial.clone(),
                source,
            })?;
    Ok(HeadProbeReport {
        serial,
        started_at,
        completed_at,
        servos,
    })
}

async fn probe_with_transport<T, C>(
    mut transport: T,
    clock: C,
    config: &HeadProbeConfig,
) -> Result<(MonotonicTime, MonotonicTime, [ServoProbeReport; 4]), HeadProbeError>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let started_at = clock.now();
    let servos = [
        probe_joint(&mut transport, &clock, config, HeadJoint::Bow).await?,
        probe_joint(&mut transport, &clock, config, HeadJoint::Curl).await?,
        probe_joint(&mut transport, &clock, config, HeadJoint::Yaw).await?,
        probe_joint(&mut transport, &clock, config, HeadJoint::Roll).await?,
    ];
    Ok((started_at, clock.now(), servos))
}

async fn probe_joint<T, C>(
    transport: &mut T,
    clock: &C,
    config: &HeadProbeConfig,
    joint: HeadJoint,
) -> Result<ServoProbeReport, HeadProbeError>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let id = joint.servo_id();
    let torque_frame = build_torque_switch_read(id);
    let (torque_bytes, torque_response) = request(
        transport,
        clock,
        config,
        joint,
        ProbeRequest::TorqueSwitch,
        &torque_frame,
    )
    .await?;
    let torque_switch = TorqueSwitchObservation::parse(torque_bytes.as_bytes(), id)
        .map_err(|source| HeadProbeError::TorqueSwitch { joint, source })?;

    let telemetry_frame = build_full_telemetry_read(id);
    let (telemetry_bytes, telemetry_response) = request(
        transport,
        clock,
        config,
        joint,
        ProbeRequest::FullTelemetry,
        &telemetry_frame,
    )
    .await?;
    let telemetry = FullTelemetry::parse(telemetry_bytes.as_bytes(), id)
        .map_err(|source| HeadProbeError::FullTelemetry { joint, source })?;

    Ok(ServoProbeReport {
        joint,
        torque_switch,
        torque_response,
        telemetry,
        telemetry_response,
    })
}

async fn request<T, C>(
    transport: &mut T,
    clock: &C,
    config: &HeadProbeConfig,
    joint: HeadJoint,
    request: ProbeRequest,
    frame: &CommandFrame,
) -> Result<(crate::framing::ResponseFrame, ProbeResponseEvidence), HeadProbeError>
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    let request_started_at = clock.now();
    transport
        .write_all(frame.as_bytes(), config.request_timeout().get())
        .await
        .map_err(|source| HeadProbeError::RequestWrite {
            joint,
            request,
            source,
        })?;
    let request_sent_at = clock.now();
    let response = read_response_frame(
        transport,
        clock,
        config.response_timeout(),
        config.noise_budget_bytes(),
    )
    .await
    .map_err(|source| HeadProbeError::ResponseFrame {
        joint,
        request,
        source,
    })?;
    let response_received_at = clock.now();
    let evidence = ProbeResponseEvidence {
        request_started_at,
        request_sent_at,
        response_received_at,
        discarded_noise_bytes: response.discarded_noise_bytes(),
    };
    Ok((response, evidence))
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use kiko_head_protocol::{ObservedTorqueSwitch, ServoId};

    use super::*;
    use crate::config::HeadProbeConfigInput;
    use crate::transport::TransportFailure;

    #[derive(Clone, Copy)]
    struct FixedClock;

    impl MonotonicClock for FixedClock {
        fn now(&self) -> MonotonicTime {
            MonotonicTime::ZERO
        }
    }

    struct ScriptedTransport {
        input: VecDeque<u8>,
        writes: Arc<Mutex<Vec<Vec<u8>>>>,
    }

    impl AsyncByteTransport for ScriptedTransport {
        fn write_all(
            &mut self,
            bytes: &[u8],
            _timeout: Duration,
        ) -> impl Future<Output = Result<(), TransportFailure>> + Send {
            self.writes
                .lock()
                .expect("test write log lock")
                .push(bytes.to_vec());
            async { Ok(()) }
        }

        fn read_some(
            &mut self,
            bytes: &mut [u8],
            _timeout: Duration,
        ) -> impl Future<Output = Result<usize, TransportFailure>> + Send {
            let read = bytes.len().min(self.input.len());
            for destination in &mut bytes[..read] {
                *destination = self.input.pop_front().expect("bounded scripted byte");
            }
            async move { Ok(read) }
        }
    }

    fn config() -> HeadProbeConfig {
        HeadProbeConfig::parse(HeadProbeConfigInput {
            device_path: "/dev/serial/by-id/usb-Kiko_STS_adapter_test".to_owned(),
            response_timeout_ms: 100,
            request_timeout_ms: 100,
            noise_budget_bytes: 4,
        })
        .expect("valid test probe configuration")
    }

    fn status(id: ServoId, parameters: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0xff, 0xff, id.get(), 0, 0];
        bytes[3] = u8::try_from(parameters.len() + 2).expect("test parameter count");
        bytes.extend_from_slice(parameters);
        let checksum = !bytes[2..]
            .iter()
            .fold(0_u8, |sum, byte| sum.wrapping_add(*byte));
        bytes.push(checksum);
        bytes
    }

    fn response_stream() -> VecDeque<u8> {
        let mut bytes = Vec::new();
        for (joint, position) in HeadJoint::ALL.into_iter().zip([2000_u16, 2001, 2002, 2003]) {
            bytes.extend(status(joint.servo_id(), &[0]));
            let mut telemetry = [0_u8; 15];
            telemetry[..2].copy_from_slice(&position.to_le_bytes());
            bytes.extend(status(joint.servo_id(), &telemetry));
        }
        bytes.into()
    }

    #[tokio::test]
    async fn probe_emits_only_fixed_reads_and_returns_typed_observations() {
        let writes = Arc::new(Mutex::new(Vec::new()));
        let transport = ScriptedTransport {
            input: response_stream(),
            writes: Arc::clone(&writes),
        };

        let (_, _, reports) = probe_with_transport(transport, FixedClock, &config())
            .await
            .expect("read-only probe");

        for (index, report) in reports.into_iter().enumerate() {
            assert_eq!(report.joint(), HeadJoint::ALL[index]);
            assert_eq!(
                report.torque_switch().state(),
                ObservedTorqueSwitch::Disabled
            );
            assert_eq!(report.telemetry().position().get(), 2000 + index as u16);
        }
        let writes = writes.lock().expect("test write log lock");
        assert_eq!(writes.len(), 8);
        for (index, pair) in writes.chunks_exact(2).enumerate() {
            let id = HeadJoint::ALL[index].servo_id().get();
            assert_eq!(
                pair[0],
                [0xff, 0xff, id, 4, 2, 40, 1, 255_u8.wrapping_sub(id + 47)]
            );
            assert_eq!(pair[1][4], 2, "telemetry command must be READ");
            assert_eq!(&pair[1][5..7], &[56, 15]);
        }
    }

    #[tokio::test]
    async fn checksum_failure_retains_joint_and_request_stage() {
        let writes = Arc::new(Mutex::new(Vec::new()));
        let mut input = response_stream();
        let first_checksum = 6;
        input[first_checksum] ^= 1;
        let transport = ScriptedTransport { input, writes };

        assert!(matches!(
            probe_with_transport(transport, FixedClock, &config()).await,
            Err(HeadProbeError::TorqueSwitch {
                joint: HeadJoint::Bow,
                source: ResponseParseError::ChecksumMismatch { .. },
            })
        ));
    }

    #[tokio::test]
    async fn truncated_response_is_not_retried_or_relabelled() {
        let writes = Arc::new(Mutex::new(Vec::new()));
        let transport = ScriptedTransport {
            input: VecDeque::new(),
            writes,
        };

        assert!(matches!(
            probe_with_transport(transport, FixedClock, &config()).await,
            Err(HeadProbeError::ResponseFrame {
                joint: HeadJoint::Bow,
                request: ProbeRequest::TorqueSwitch,
                source: FrameReadError::Truncated { .. },
            })
        ));
    }
}
