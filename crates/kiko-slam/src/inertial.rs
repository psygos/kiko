//! Hardware-neutral inertial measurements parsed into explicit SI units.
//!
//! OAK reports are expressed in the device's IMU frame. This module does not
//! imply an IMU-to-camera or IMU-to-robot transform; consumers must apply a
//! separately calibrated transform before using measurements in another frame.

use std::marker::PhantomData;
use std::num::NonZeroU64;

/// Identifies one uninterrupted device-clock session.
///
/// Reconnecting or rebooting a device must allocate a new ID even if its clock
/// restarts at the same value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceSessionId(NonZeroU64);

impl DeviceSessionId {
    pub fn try_new(raw: u64) -> Result<Self, InertialValueError> {
        NonZeroU64::new(raw)
            .map(Self)
            .ok_or(InertialValueError::ZeroDeviceSessionId)
    }

    pub fn as_u64(self) -> u64 {
        self.0.get()
    }
}

/// Nanoseconds on a device clock within a [`DeviceSessionId`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceTimestamp(u64);

impl DeviceTimestamp {
    pub fn try_from_nanos(nanos: i64) -> Result<Self, InertialValueError> {
        let nanos = u64::try_from(nanos)
            .map_err(|_| InertialValueError::NegativeDeviceTimestamp { nanos })?;
        Ok(Self(nanos))
    }

    pub fn as_nanos(self) -> u64 {
        self.0
    }
}

/// Nanoseconds on the host process's monotonic clock.
///
/// The epoch is deliberately unspecified. Values are meaningful only for
/// ordering and elapsed-time calculations from the same process session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct HostMonotonicTimestamp(u64);

impl HostMonotonicTimestamp {
    pub fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    pub fn as_nanos(self) -> u64 {
        self.0
    }
}

/// Host-bridge dequeue order for an inertial report.
///
/// Zero is valid. A gap proves only that reports are absent from the observed
/// stream; it does not prove that the physical device dropped measurements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DequeueSequence(u32);

impl DequeueSequence {
    pub fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SensorAccuracy {
    Unreliable,
    Low,
    Medium,
    High,
}

/// The coordinate frame reported by an OAK device's IMU.
///
/// This is intentionally distinct from camera, world, and robot-base frames.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct OakImuFrame;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccelerationMps2<Frame> {
    components: [f64; 3],
    frame: PhantomData<fn() -> Frame>,
}

impl<Frame> AccelerationMps2<Frame> {
    pub fn try_new(x: f64, y: f64, z: f64) -> Result<Self, InertialValueError> {
        let components = [x, y, z];
        validate_components(InertialQuantity::AccelerationMps2, components)?;
        Ok(Self {
            components,
            frame: PhantomData,
        })
    }

    pub fn as_array(self) -> [f64; 3] {
        self.components
    }

    pub fn x(self) -> f64 {
        self.components[0]
    }

    pub fn y(self) -> f64 {
        self.components[1]
    }

    pub fn z(self) -> f64 {
        self.components[2]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularVelocityRadPerSec<Frame> {
    components: [f64; 3],
    frame: PhantomData<fn() -> Frame>,
}

impl<Frame> AngularVelocityRadPerSec<Frame> {
    pub fn try_new(x: f64, y: f64, z: f64) -> Result<Self, InertialValueError> {
        let components = [x, y, z];
        validate_components(InertialQuantity::AngularVelocityRadPerSec, components)?;
        Ok(Self {
            components,
            frame: PhantomData,
        })
    }

    pub fn as_array(self) -> [f64; 3] {
        self.components
    }

    pub fn x(self) -> f64 {
        self.components[0]
    }

    pub fn y(self) -> f64 {
        self.components[1]
    }

    pub fn z(self) -> f64 {
        self.components[2]
    }
}

pub type OakImuAcceleration = AccelerationMps2<OakImuFrame>;
pub type OakImuAngularVelocity = AngularVelocityRadPerSec<OakImuFrame>;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AccelSample {
    timestamp: DeviceTimestamp,
    acceleration: OakImuAcceleration,
    accuracy: SensorAccuracy,
}

impl AccelSample {
    pub fn new(
        timestamp: DeviceTimestamp,
        acceleration: OakImuAcceleration,
        accuracy: SensorAccuracy,
    ) -> Self {
        Self {
            timestamp,
            acceleration,
            accuracy,
        }
    }

    pub fn timestamp(self) -> DeviceTimestamp {
        self.timestamp
    }

    pub fn acceleration(self) -> OakImuAcceleration {
        self.acceleration
    }

    pub fn accuracy(self) -> SensorAccuracy {
        self.accuracy
    }

    pub fn frame(self) -> OakImuFrame {
        OakImuFrame
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GyroSample {
    timestamp: DeviceTimestamp,
    angular_velocity: OakImuAngularVelocity,
    accuracy: SensorAccuracy,
}

impl GyroSample {
    pub fn new(
        timestamp: DeviceTimestamp,
        angular_velocity: OakImuAngularVelocity,
        accuracy: SensorAccuracy,
    ) -> Self {
        Self {
            timestamp,
            angular_velocity,
            accuracy,
        }
    }

    pub fn timestamp(self) -> DeviceTimestamp {
        self.timestamp
    }

    pub fn angular_velocity(self) -> OakImuAngularVelocity {
        self.angular_velocity
    }

    pub fn accuracy(self) -> SensorAccuracy {
        self.accuracy
    }

    pub fn frame(self) -> OakImuFrame {
        OakImuFrame
    }
}

/// One bridge report containing independently timestamped accelerometer and
/// gyroscope measurements.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImuReport {
    session_id: DeviceSessionId,
    sequence: DequeueSequence,
    host_arrival: HostMonotonicTimestamp,
    accel: AccelSample,
    gyro: GyroSample,
}

impl ImuReport {
    pub fn new(
        session_id: DeviceSessionId,
        sequence: DequeueSequence,
        host_arrival: HostMonotonicTimestamp,
        accel: AccelSample,
        gyro: GyroSample,
    ) -> Self {
        Self {
            session_id,
            sequence,
            host_arrival,
            accel,
            gyro,
        }
    }

    pub fn session_id(self) -> DeviceSessionId {
        self.session_id
    }

    pub fn sequence(self) -> DequeueSequence {
        self.sequence
    }

    pub fn host_arrival(self) -> HostMonotonicTimestamp {
        self.host_arrival
    }

    pub fn accel(self) -> AccelSample {
        self.accel
    }

    pub fn gyro(self) -> GyroSample {
        self.gyro
    }

    /// Returns events in device-timestamp order, with accelerometer first when
    /// both timestamps are equal.
    pub fn events(self) -> [ImuEvent; 2] {
        let accel = ImuEvent::Accel {
            session_id: self.session_id,
            sequence: self.sequence,
            host_arrival: self.host_arrival,
            sample: self.accel,
        };
        let gyro = ImuEvent::Gyro {
            session_id: self.session_id,
            sequence: self.sequence,
            host_arrival: self.host_arrival,
            sample: self.gyro,
        };
        if self.gyro.timestamp() < self.accel.timestamp() {
            [gyro, accel]
        } else {
            [accel, gyro]
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InertialSensorKind {
    Accelerometer,
    Gyroscope,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ImuEvent {
    Accel {
        session_id: DeviceSessionId,
        sequence: DequeueSequence,
        host_arrival: HostMonotonicTimestamp,
        sample: AccelSample,
    },
    Gyro {
        session_id: DeviceSessionId,
        sequence: DequeueSequence,
        host_arrival: HostMonotonicTimestamp,
        sample: GyroSample,
    },
}

impl ImuEvent {
    pub fn session_id(self) -> DeviceSessionId {
        match self {
            Self::Accel { session_id, .. } | Self::Gyro { session_id, .. } => session_id,
        }
    }

    pub fn sequence(self) -> DequeueSequence {
        match self {
            Self::Accel { sequence, .. } | Self::Gyro { sequence, .. } => sequence,
        }
    }

    pub fn host_arrival(self) -> HostMonotonicTimestamp {
        match self {
            Self::Accel { host_arrival, .. } | Self::Gyro { host_arrival, .. } => host_arrival,
        }
    }

    pub fn sensor(self) -> InertialSensorKind {
        match self {
            Self::Accel { .. } => InertialSensorKind::Accelerometer,
            Self::Gyro { .. } => InertialSensorKind::Gyroscope,
        }
    }

    pub fn device_timestamp(self) -> DeviceTimestamp {
        match self {
            Self::Accel { sample, .. } => sample.timestamp(),
            Self::Gyro { sample, .. } => sample.timestamp(),
        }
    }

    pub fn accuracy(self) -> SensorAccuracy {
        match self {
            Self::Accel { sample, .. } => sample.accuracy(),
            Self::Gyro { sample, .. } => sample.accuracy(),
        }
    }

    pub fn as_accel(self) -> Option<AccelSample> {
        match self {
            Self::Accel { sample, .. } => Some(sample),
            Self::Gyro { .. } => None,
        }
    }

    pub fn as_gyro(self) -> Option<GyroSample> {
        match self {
            Self::Gyro { sample, .. } => Some(sample),
            Self::Accel { .. } => None,
        }
    }

    pub fn frame(self) -> OakImuFrame {
        OakImuFrame
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InertialOrderOutcome {
    FirstReport,
    Contiguous,
    Gap { missing_reports: u32 },
}

impl InertialOrderOutcome {
    pub fn missing_reports(self) -> u32 {
        match self {
            Self::Gap { missing_reports } => missing_reports,
            Self::FirstReport | Self::Contiguous => 0,
        }
    }
}

/// Stateful validator for one explicitly delimited device session.
///
/// Failed observations never mutate the tracker, so callers can report an
/// error and continue deterministically or reset to a new session.
#[derive(Clone, Debug, Default)]
pub struct InertialOrderTracker {
    session_id: Option<DeviceSessionId>,
    previous_sequence: Option<DequeueSequence>,
    previous_host_arrival: Option<HostMonotonicTimestamp>,
    previous_accel_timestamp: Option<DeviceTimestamp>,
    previous_gyro_timestamp: Option<DeviceTimestamp>,
}

impl InertialOrderTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_session(session_id: DeviceSessionId) -> Self {
        Self {
            session_id: Some(session_id),
            ..Self::default()
        }
    }

    pub fn session_id(&self) -> Option<DeviceSessionId> {
        self.session_id
    }

    pub fn reset_to_session(&mut self, session_id: DeviceSessionId) {
        *self = Self::with_session(session_id);
    }

    pub fn observe(
        &mut self,
        report: &ImuReport,
    ) -> Result<InertialOrderOutcome, InertialOrderingError> {
        let session_id = report.session_id();
        if let Some(expected) = self.session_id
            && session_id != expected
        {
            return Err(InertialOrderingError::SessionMismatch {
                expected,
                actual: session_id,
            });
        }

        let sequence = report.sequence();
        let outcome = match self.previous_sequence {
            None => InertialOrderOutcome::FirstReport,
            Some(previous) if sequence == previous => {
                return Err(InertialOrderingError::DuplicateSequence {
                    session_id,
                    sequence,
                });
            }
            Some(previous) if sequence < previous => {
                return Err(InertialOrderingError::SequenceRegression {
                    session_id,
                    previous,
                    current: sequence,
                });
            }
            Some(previous) => {
                let missing_reports = sequence.as_u32() - previous.as_u32() - 1;
                if missing_reports == 0 {
                    InertialOrderOutcome::Contiguous
                } else {
                    InertialOrderOutcome::Gap { missing_reports }
                }
            }
        };

        let host_arrival = report.host_arrival();
        if let Some(previous) = self.previous_host_arrival
            && host_arrival < previous
        {
            return Err(InertialOrderingError::HostArrivalRegression {
                session_id,
                previous,
                current: host_arrival,
            });
        }

        validate_sensor_timestamp(
            session_id,
            InertialSensorKind::Accelerometer,
            self.previous_accel_timestamp,
            report.accel().timestamp(),
        )?;
        validate_sensor_timestamp(
            session_id,
            InertialSensorKind::Gyroscope,
            self.previous_gyro_timestamp,
            report.gyro().timestamp(),
        )?;

        self.session_id = Some(session_id);
        self.previous_sequence = Some(sequence);
        self.previous_host_arrival = Some(host_arrival);
        self.previous_accel_timestamp = Some(report.accel().timestamp());
        self.previous_gyro_timestamp = Some(report.gyro().timestamp());
        Ok(outcome)
    }
}

fn validate_sensor_timestamp(
    session_id: DeviceSessionId,
    sensor: InertialSensorKind,
    previous: Option<DeviceTimestamp>,
    current: DeviceTimestamp,
) -> Result<(), InertialOrderingError> {
    match previous {
        Some(previous) if current == previous => {
            Err(InertialOrderingError::DuplicateDeviceTimestamp {
                session_id,
                sensor,
                timestamp: current,
            })
        }
        Some(previous) if current < previous => {
            Err(InertialOrderingError::DeviceTimestampRegression {
                session_id,
                sensor,
                previous,
                current,
            })
        }
        _ => Ok(()),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InertialQuantity {
    AccelerationMps2,
    AngularVelocityRadPerSec,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InertialAxis {
    X,
    Y,
    Z,
}

impl InertialAxis {
    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::X,
            1 => Self::Y,
            2 => Self::Z,
            _ => unreachable!("three-axis inertial vector"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InertialValueError {
    ZeroDeviceSessionId,
    NegativeDeviceTimestamp {
        nanos: i64,
    },
    NonFiniteComponent {
        quantity: InertialQuantity,
        axis: InertialAxis,
        value: f64,
    },
    UnknownSensorAccuracy {
        raw: u8,
    },
}

impl std::fmt::Display for InertialValueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDeviceSessionId => write!(f, "device session ID must be nonzero"),
            Self::NegativeDeviceTimestamp { nanos } => {
                write!(f, "device timestamp must be nonnegative, got {nanos} ns")
            }
            Self::NonFiniteComponent {
                quantity,
                axis,
                value,
            } => write!(
                f,
                "{quantity:?} component {axis:?} must be finite, got {value}"
            ),
            Self::UnknownSensorAccuracy { raw } => {
                write!(f, "unknown sensor accuracy value {raw}")
            }
        }
    }
}

impl std::error::Error for InertialValueError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InertialOrderingError {
    SessionMismatch {
        expected: DeviceSessionId,
        actual: DeviceSessionId,
    },
    DuplicateSequence {
        session_id: DeviceSessionId,
        sequence: DequeueSequence,
    },
    SequenceRegression {
        session_id: DeviceSessionId,
        previous: DequeueSequence,
        current: DequeueSequence,
    },
    HostArrivalRegression {
        session_id: DeviceSessionId,
        previous: HostMonotonicTimestamp,
        current: HostMonotonicTimestamp,
    },
    DuplicateDeviceTimestamp {
        session_id: DeviceSessionId,
        sensor: InertialSensorKind,
        timestamp: DeviceTimestamp,
    },
    DeviceTimestampRegression {
        session_id: DeviceSessionId,
        sensor: InertialSensorKind,
        previous: DeviceTimestamp,
        current: DeviceTimestamp,
    },
}

impl std::fmt::Display for InertialOrderingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionMismatch { expected, actual } => write!(
                f,
                "inertial session mismatch: expected {}, got {}",
                expected.as_u64(),
                actual.as_u64()
            ),
            Self::DuplicateSequence {
                session_id,
                sequence,
            } => write!(
                f,
                "duplicate inertial sequence {} in session {}",
                sequence.as_u32(),
                session_id.as_u64()
            ),
            Self::SequenceRegression {
                session_id,
                previous,
                current,
            } => write!(
                f,
                "inertial sequence regressed from {} to {} in session {}",
                previous.as_u32(),
                current.as_u32(),
                session_id.as_u64()
            ),
            Self::HostArrivalRegression {
                session_id,
                previous,
                current,
            } => write!(
                f,
                "host arrival regressed from {} to {} ns in session {}",
                previous.as_nanos(),
                current.as_nanos(),
                session_id.as_u64()
            ),
            Self::DuplicateDeviceTimestamp {
                session_id,
                sensor,
                timestamp,
            } => write!(
                f,
                "duplicate {sensor:?} timestamp {} ns in session {}",
                timestamp.as_nanos(),
                session_id.as_u64()
            ),
            Self::DeviceTimestampRegression {
                session_id,
                sensor,
                previous,
                current,
            } => write!(
                f,
                "{sensor:?} timestamp regressed from {} to {} ns in session {}",
                previous.as_nanos(),
                current.as_nanos(),
                session_id.as_u64()
            ),
        }
    }
}

impl std::error::Error for InertialOrderingError {}

fn validate_components(
    quantity: InertialQuantity,
    components: [f64; 3],
) -> Result<(), InertialValueError> {
    for (index, value) in components.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(InertialValueError::NonFiniteComponent {
                quantity,
                axis: InertialAxis::from_index(index),
                value,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(raw: u64) -> DeviceSessionId {
        DeviceSessionId::try_new(raw).expect("nonzero test session")
    }

    fn timestamp(raw: i64) -> DeviceTimestamp {
        DeviceTimestamp::try_from_nanos(raw).expect("nonnegative test timestamp")
    }

    fn report(
        session_raw: u64,
        sequence: u32,
        arrival: u64,
        accel_timestamp: i64,
        gyro_timestamp: i64,
    ) -> ImuReport {
        ImuReport::new(
            session(session_raw),
            DequeueSequence::new(sequence),
            HostMonotonicTimestamp::from_nanos(arrival),
            AccelSample::new(
                timestamp(accel_timestamp),
                OakImuAcceleration::try_new(1.0, 2.0, 3.0).expect("finite acceleration"),
                SensorAccuracy::High,
            ),
            GyroSample::new(
                timestamp(gyro_timestamp),
                OakImuAngularVelocity::try_new(4.0, 5.0, 6.0).expect("finite angular velocity"),
                SensorAccuracy::Medium,
            ),
        )
    }

    #[test]
    fn identifiers_and_device_timestamps_reject_invalid_domain_values() {
        assert_eq!(
            DeviceSessionId::try_new(0),
            Err(InertialValueError::ZeroDeviceSessionId)
        );
        assert_eq!(session(9).as_u64(), 9);
        assert_eq!(timestamp(0).as_nanos(), 0);
        assert_eq!(
            DeviceTimestamp::try_from_nanos(-1),
            Err(InertialValueError::NegativeDeviceTimestamp { nanos: -1 })
        );
        assert_eq!(DequeueSequence::new(0).as_u32(), 0);
    }

    #[test]
    fn every_nonfinite_vector_component_is_rejected() {
        for quantity in [
            InertialQuantity::AccelerationMps2,
            InertialQuantity::AngularVelocityRadPerSec,
        ] {
            for axis in 0..3 {
                for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                    let mut components = [1.0, 2.0, 3.0];
                    components[axis] = value;
                    let error = match quantity {
                        InertialQuantity::AccelerationMps2 => {
                            OakImuAcceleration::try_new(components[0], components[1], components[2])
                                .expect_err("nonfinite acceleration must fail")
                        }
                        InertialQuantity::AngularVelocityRadPerSec => {
                            OakImuAngularVelocity::try_new(
                                components[0],
                                components[1],
                                components[2],
                            )
                            .expect_err("nonfinite angular velocity must fail")
                        }
                    };
                    assert!(matches!(
                        error,
                        InertialValueError::NonFiniteComponent {
                            quantity: actual_quantity,
                            axis: actual_axis,
                            value: actual_value,
                        } if actual_quantity == quantity
                            && actual_axis == InertialAxis::from_index(axis)
                            && actual_value.to_bits() == value.to_bits()
                    ));
                }
            }
        }
    }

    #[test]
    fn finite_vectors_round_trip_without_unit_or_precision_conversion() {
        for value in [f64::MIN, -1.0, -0.0, 0.0, 1.0, f64::MAX] {
            let acceleration =
                OakImuAcceleration::try_new(value, -2.5, 3.5).expect("all components are finite");
            assert_eq!(acceleration.as_array(), [value, -2.5, 3.5]);
            assert_eq!(
                (acceleration.x(), acceleration.y(), acceleration.z()),
                (value, -2.5, 3.5)
            );
        }
    }

    #[test]
    fn report_preserves_independent_timestamps_and_canonical_event_tie_order() {
        let base_report = report(7, 11, 100, 42, 43);
        assert_eq!(base_report.session_id(), session(7));
        assert_eq!(base_report.sequence(), DequeueSequence::new(11));
        assert_eq!(base_report.accel().timestamp(), timestamp(42));
        assert_eq!(base_report.gyro().timestamp(), timestamp(43));

        let [accel, gyro] = base_report.events();
        assert_eq!(accel.sensor(), InertialSensorKind::Accelerometer);
        assert_eq!(gyro.sensor(), InertialSensorKind::Gyroscope);
        assert_eq!(accel.device_timestamp(), timestamp(42));
        assert_eq!(gyro.device_timestamp(), timestamp(43));
        assert_eq!(accel.session_id(), base_report.session_id());
        assert_eq!(gyro.sequence(), base_report.sequence());
        assert_eq!(accel.host_arrival(), base_report.host_arrival());
        assert_eq!(accel.as_accel(), Some(base_report.accel()));
        assert_eq!(gyro.as_gyro(), Some(base_report.gyro()));
        assert_eq!(accel.as_gyro(), None);
        assert_eq!(gyro.as_accel(), None);

        let [gyro, accel] = report(7, 12, 101, 50, 49).events();
        assert_eq!(gyro.sensor(), InertialSensorKind::Gyroscope);
        assert_eq!(accel.sensor(), InertialSensorKind::Accelerometer);

        let [accel, gyro] = report(7, 13, 102, 50, 50).events();
        assert_eq!(accel.sensor(), InertialSensorKind::Accelerometer);
        assert_eq!(gyro.sensor(), InertialSensorKind::Gyroscope);
    }

    #[test]
    fn monotonic_streams_report_contiguity_and_exact_gaps() {
        let mut tracker = InertialOrderTracker::new();
        for index in 0_u32..256 {
            let current = report(
                1,
                index,
                u64::from(index / 4),
                i64::from(index) * 2,
                i64::from(index) * 2 + 1,
            );
            let expected = if index == 0 {
                InertialOrderOutcome::FirstReport
            } else {
                InertialOrderOutcome::Contiguous
            };
            assert_eq!(tracker.observe(&current), Ok(expected));
        }

        let after_gap = report(1, 260, 100, 520, 521);
        assert_eq!(
            tracker.observe(&after_gap),
            Ok(InertialOrderOutcome::Gap { missing_reports: 4 })
        );
        assert_eq!(tracker.session_id(), Some(session(1)));
    }

    #[test]
    fn ordering_errors_are_specific_and_do_not_mutate_state() {
        let first = report(1, 10, 20, 30, 31);
        let mut tracker = InertialOrderTracker::new();
        assert_eq!(
            tracker.observe(&first),
            Ok(InertialOrderOutcome::FirstReport)
        );

        assert!(matches!(
            tracker.observe(&report(1, 10, 21, 32, 33)),
            Err(InertialOrderingError::DuplicateSequence { .. })
        ));
        assert!(matches!(
            tracker.observe(&report(1, 9, 21, 32, 33)),
            Err(InertialOrderingError::SequenceRegression { .. })
        ));
        assert!(matches!(
            tracker.observe(&report(1, 11, 19, 32, 33)),
            Err(InertialOrderingError::HostArrivalRegression { .. })
        ));
        assert!(matches!(
            tracker.observe(&report(1, 11, 21, 30, 33)),
            Err(InertialOrderingError::DuplicateDeviceTimestamp {
                sensor: InertialSensorKind::Accelerometer,
                ..
            })
        ));
        assert!(matches!(
            tracker.observe(&report(1, 11, 21, 32, 30)),
            Err(InertialOrderingError::DeviceTimestampRegression {
                sensor: InertialSensorKind::Gyroscope,
                ..
            })
        ));
        assert!(matches!(
            tracker.observe(&report(2, 11, 21, 32, 33)),
            Err(InertialOrderingError::SessionMismatch { .. })
        ));

        assert_eq!(
            tracker.observe(&report(1, 11, 21, 32, 33)),
            Ok(InertialOrderOutcome::Contiguous)
        );
    }

    #[test]
    fn explicit_session_reset_accepts_restarted_clocks_and_sequences() {
        let mut tracker = InertialOrderTracker::with_session(session(1));
        tracker
            .observe(&report(1, 100, 500, 900, 901))
            .expect("first session report");
        tracker.reset_to_session(session(2));
        assert_eq!(
            tracker.observe(&report(2, 0, 501, 0, 1)),
            Ok(InertialOrderOutcome::FirstReport)
        );
    }
}
