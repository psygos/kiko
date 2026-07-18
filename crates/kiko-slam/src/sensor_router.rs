use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    ChannelCapacity, ChannelStats, ChannelStatsHandle, DepthImage, DeviceSessionId,
    DeviceTimestamp, DropPolicy, DropReceiver, DropSender, FrameId, HostMonotonicTimestamp,
    ImuReport, InertialOrderOutcome, InertialOrderTracker, InertialOrderingError,
    InertialValueError, SendOutcome, StereoPair, bounded_channel,
};

/// One valid stereo pair bound to its device-clock session and exact
/// coordinator-entry time.
///
/// [`StereoPair`] deliberately retains the image payloads. This boundary
/// parses both weak signed frame timestamps once into [`DeviceTimestamp`] and
/// requires callers, including dataset replay, to supply the real host time;
/// it never substitutes a device timestamp or nominal frame period.
#[derive(Debug)]
pub struct StereoObservation {
    session_id: DeviceSessionId,
    host_arrival: HostMonotonicTimestamp,
    left_device_timestamp: DeviceTimestamp,
    right_device_timestamp: DeviceTimestamp,
    pair: StereoPair,
}

impl StereoObservation {
    pub fn parse(
        session_id: DeviceSessionId,
        host_arrival: HostMonotonicTimestamp,
        pair: StereoPair,
    ) -> Result<Self, StereoObservationError> {
        let left_device_timestamp = DeviceTimestamp::try_from_nanos(
            pair.left().timestamp().as_nanos(),
        )
        .map_err(|source| StereoObservationError::InvalidDeviceTimestamp {
            side: StereoObservationSide::Left,
            source,
        })?;
        let right_device_timestamp = DeviceTimestamp::try_from_nanos(
            pair.right().timestamp().as_nanos(),
        )
        .map_err(|source| StereoObservationError::InvalidDeviceTimestamp {
            side: StereoObservationSide::Right,
            source,
        })?;
        Ok(Self {
            session_id,
            host_arrival,
            left_device_timestamp,
            right_device_timestamp,
            pair,
        })
    }

    pub fn session_id(&self) -> DeviceSessionId {
        self.session_id
    }

    pub fn host_arrival(&self) -> HostMonotonicTimestamp {
        self.host_arrival
    }

    pub fn left_device_timestamp(&self) -> DeviceTimestamp {
        self.left_device_timestamp
    }

    pub fn right_device_timestamp(&self) -> DeviceTimestamp {
        self.right_device_timestamp
    }

    pub fn pair(&self) -> &StereoPair {
        &self.pair
    }

    pub fn into_pair(self) -> StereoPair {
        self.pair
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoObservationSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StereoObservationError {
    InvalidDeviceTimestamp {
        side: StereoObservationSide,
        source: InertialValueError,
    },
}

impl std::fmt::Display for StereoObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDeviceTimestamp { side, source } => {
                write!(f, "invalid {side:?} stereo device timestamp: {source}")
            }
        }
    }
}

impl std::error::Error for StereoObservationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidDeviceTimestamp { source, .. } => Some(source),
        }
    }
}

/// One navigation depth frame bound to its device-clock session and host
/// dequeue time.
///
/// [`DepthImage`] deliberately remains the reusable metric pixel payload. This
/// wrapper parses its weak signed timestamp once and adds the provenance needed
/// for exact pose alignment, freshness checks, recording, and replay. Cloning
/// an observation shares the depth pixel allocation.
#[derive(Clone, Debug)]
pub struct DepthObservation {
    session_id: DeviceSessionId,
    device_timestamp: DeviceTimestamp,
    host_arrival: HostMonotonicTimestamp,
    depth: DepthImage,
}

impl DepthObservation {
    pub fn parse(
        session_id: DeviceSessionId,
        host_arrival: HostMonotonicTimestamp,
        depth: DepthImage,
    ) -> Result<Self, DepthObservationError> {
        let device_timestamp = DeviceTimestamp::try_from_nanos(depth.timestamp().as_nanos())
            .map_err(DepthObservationError::InvalidDeviceTimestamp)?;
        Ok(Self {
            session_id,
            device_timestamp,
            host_arrival,
            depth,
        })
    }

    pub fn session_id(&self) -> DeviceSessionId {
        self.session_id
    }

    pub fn device_timestamp(&self) -> DeviceTimestamp {
        self.device_timestamp
    }

    pub fn host_arrival(&self) -> HostMonotonicTimestamp {
        self.host_arrival
    }

    pub fn frame_id(&self) -> FrameId {
        self.depth.frame_id()
    }

    pub fn depth(&self) -> &DepthImage {
        &self.depth
    }

    pub fn into_depth(self) -> DepthImage {
        self.depth
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DepthObservationError {
    InvalidDeviceTimestamp(InertialValueError),
}

impl std::fmt::Display for DepthObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidDeviceTimestamp(source) => {
                write!(f, "invalid navigation depth device timestamp: {source}")
            }
        }
    }
}

impl std::error::Error for DepthObservationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidDeviceTimestamp(source) => Some(source),
        }
    }
}

/// The two independently backpressured destinations for one depth observation.
#[derive(Debug)]
pub struct DepthRoutes {
    pub slam: DropReceiver<DepthImage>,
    pub navigation: DropReceiver<DepthObservation>,
}

/// Per-destination result for one depth fanout attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DepthRouteOutcome {
    pub slam: SendOutcome,
    pub navigation: SendOutcome,
}

/// Quiescent depth-channel counters, kept separate so one consumer cannot hide
/// loss or disconnection at the other.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DepthRouteStats {
    pub slam: ChannelStats,
    pub navigation: ChannelStats,
}

#[derive(Clone, Debug)]
pub struct DepthRouteStatsHandle {
    slam: ChannelStatsHandle,
    navigation: ChannelStatsHandle,
}

impl DepthRouteStatsHandle {
    pub fn snapshot(&self) -> DepthRouteStats {
        DepthRouteStats {
            slam: self.slam.snapshot(),
            navigation: self.navigation.snapshot(),
        }
    }
}

/// Producer half of a depth fanout.
///
/// [`DepthImage::clone`] only copies its scalar metadata and shares its pixel
/// allocation. Routing therefore needs no additional outer allocation.
#[derive(Debug)]
pub struct DepthRouter {
    slam: DropSender<DepthImage>,
    navigation: DropSender<DepthObservation>,
}

impl DepthRouter {
    /// Attempt both destinations even if either has disconnected or rejects
    /// the observation.
    pub fn route(&self, observation: DepthObservation) -> DepthRouteOutcome {
        let slam = self.slam.try_send(observation.depth.clone());
        let navigation = self.navigation.try_send(observation);
        DepthRouteOutcome { slam, navigation }
    }
}

/// Build the depth producer and its two independently bounded destinations.
///
/// The SLAM/keyframe destination uses its caller-selected capacity and policy.
/// Navigation always retains only the newest depth observation: capacity one,
/// drop oldest.
pub fn depth_router(
    slam_capacity: ChannelCapacity,
    slam_policy: DropPolicy,
) -> (DepthRouter, DepthRoutes, DepthRouteStatsHandle) {
    let (slam, slam_rx, slam_stats) = bounded_channel(slam_capacity, slam_policy);
    let navigation_capacity = ChannelCapacity::new(NonZeroUsize::MIN);
    let (navigation, navigation_rx, navigation_stats) =
        bounded_channel(navigation_capacity, DropPolicy::DropOldest);

    (
        DepthRouter { slam, navigation },
        DepthRoutes {
            slam: slam_rx,
            navigation: navigation_rx,
        },
        DepthRouteStatsHandle {
            slam: slam_stats,
            navigation: navigation_stats,
        },
    )
}

/// Consumer half of the bounded high-rate inertial report route.
///
/// Reports remain intact so downstream replay can split each report into its
/// canonical device-timestamp order (accelerometer first on ties) without
/// losing their shared dequeue identity.
#[derive(Debug)]
pub struct ImuReportRoute {
    pub reports: DropReceiver<ImuReport>,
}

/// Result of accepting one correctly ordered inertial report at the producer
/// boundary and attempting delivery to the bounded route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImuReportRouteOutcome {
    pub ordering: InertialOrderOutcome,
    pub delivery: SendOutcome,
}

/// IMU route counters. `source_*` describes reports already absent before this
/// router; `reports.dropped_*` describes loss caused by this bounded route.
/// Neither kind of gap claims that the physical sensor dropped measurements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImuReportRouteStats {
    pub reports: ChannelStats,
    pub source_gap_events: u64,
    pub source_missing_reports: u64,
    pub ordering_rejected: u64,
}

#[derive(Debug)]
struct ImuReportRouteState {
    source_gap_events: AtomicU64,
    source_missing_reports: AtomicU64,
    ordering_rejected: AtomicU64,
}

impl ImuReportRouteState {
    fn new() -> Self {
        Self {
            source_gap_events: AtomicU64::new(0),
            source_missing_reports: AtomicU64::new(0),
            ordering_rejected: AtomicU64::new(0),
        }
    }

    fn record_gap(&self, missing_reports: u32) {
        saturating_add(&self.source_gap_events, 1);
        saturating_add(&self.source_missing_reports, u64::from(missing_reports));
    }

    fn record_ordering_rejection(&self) {
        saturating_add(&self.ordering_rejected, 1);
    }
}

fn saturating_add(counter: &AtomicU64, increment: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
        let next = value.saturating_add(increment);
        (next != value).then_some(next)
    });
}

#[derive(Clone, Debug)]
pub struct ImuReportRouteStatsHandle {
    reports: ChannelStatsHandle,
    state: Arc<ImuReportRouteState>,
}

impl ImuReportRouteStatsHandle {
    pub fn snapshot(&self) -> ImuReportRouteStats {
        ImuReportRouteStats {
            reports: self.reports.snapshot(),
            source_gap_events: self.state.source_gap_events.load(Ordering::Relaxed),
            source_missing_reports: self.state.source_missing_reports.load(Ordering::Relaxed),
            ordering_rejected: self.state.ordering_rejected.load(Ordering::Relaxed),
        }
    }
}

/// Single-producer route for validated, independently timestamped IMU reports.
///
/// Ordering is checked before enqueueing. Invalid ordering never mutates the
/// order tracker and never reaches the consumer. Overflow drops the oldest
/// queued report so a slow estimator receives the freshest bounded suffix; the
/// exact route loss remains visible in [`ImuReportRouteStats`].
#[derive(Debug)]
pub struct ImuReportRouter {
    reports: DropSender<ImuReport>,
    ordering: InertialOrderTracker,
    state: Arc<ImuReportRouteState>,
}

impl ImuReportRouter {
    pub fn route(
        &mut self,
        report: ImuReport,
    ) -> Result<ImuReportRouteOutcome, InertialOrderingError> {
        let ordering = match self.ordering.observe(&report) {
            Ok(ordering) => ordering,
            Err(error) => {
                self.state.record_ordering_rejection();
                return Err(error);
            }
        };
        let missing_reports = ordering.missing_reports();
        if missing_reports > 0 {
            self.state.record_gap(missing_reports);
        }
        let delivery = self.reports.try_send(report);
        Ok(ImuReportRouteOutcome { ordering, delivery })
    }
}

/// Build one high-rate IMU report route for an explicitly delimited device
/// session. A reconnect must build a new route with its new session ID.
pub fn imu_report_router(
    session_id: DeviceSessionId,
    capacity: ChannelCapacity,
) -> (ImuReportRouter, ImuReportRoute, ImuReportRouteStatsHandle) {
    let (reports, reports_rx, report_stats) = bounded_channel(capacity, DropPolicy::DropOldest);
    let state = Arc::new(ImuReportRouteState::new());
    (
        ImuReportRouter {
            reports,
            ordering: InertialOrderTracker::with_session(session_id),
            state: Arc::clone(&state),
        },
        ImuReportRoute {
            reports: reports_rx,
        },
        ImuReportRouteStatsHandle {
            reports: report_stats,
            state,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AccelSample, DequeueSequence, DeviceSessionId, DeviceTimestamp, Frame, FrameId, GyroSample,
        HostMonotonicTimestamp, OakImuAcceleration, OakImuAngularVelocity, PairingWindowNs,
        SensorAccuracy, SensorId, Timestamp,
    };

    fn capacity(value: usize) -> ChannelCapacity {
        ChannelCapacity::try_from(value).expect("nonzero test capacity")
    }

    fn depth(frame_id: u64, timestamp_ns: i64) -> DepthImage {
        DepthImage::new(
            FrameId::new(frame_id),
            Timestamp::from_nanos(timestamp_ns),
            2,
            1,
            vec![frame_id as f32, timestamp_ns as f32],
        )
        .expect("valid test depth")
    }

    fn session() -> DeviceSessionId {
        DeviceSessionId::try_new(1).expect("nonzero test session")
    }

    fn stereo_pair(left_timestamp_ns: i64, right_timestamp_ns: i64) -> StereoPair {
        let left = Frame::new(
            SensorId::StereoLeft,
            FrameId::new(41),
            Timestamp::from_nanos(left_timestamp_ns),
            2,
            1,
            vec![1, 2],
        )
        .expect("valid left frame");
        let right = Frame::new(
            SensorId::StereoRight,
            FrameId::new(42),
            Timestamp::from_nanos(right_timestamp_ns),
            2,
            1,
            vec![3, 4],
        )
        .expect("valid right frame");
        StereoPair::try_new(
            left,
            right,
            PairingWindowNs::try_from_u64(left_timestamp_ns.abs_diff(right_timestamp_ns))
                .expect("fixture delta fits the pairing domain"),
        )
        .expect("valid stereo pair")
    }

    fn depth_observation(
        frame_id: u64,
        timestamp_ns: i64,
        host_arrival_ns: u64,
    ) -> DepthObservation {
        DepthObservation::parse(
            session(),
            HostMonotonicTimestamp::from_nanos(host_arrival_ns),
            depth(frame_id, timestamp_ns),
        )
        .expect("valid navigation depth observation")
    }

    fn imu_report(sequence: u32) -> ImuReport {
        let timestamp_ns = i64::from(sequence) + 1;
        ImuReport::new(
            session(),
            DequeueSequence::new(sequence),
            HostMonotonicTimestamp::from_nanos(u64::from(sequence) + 100),
            AccelSample::new(
                DeviceTimestamp::try_from_nanos(timestamp_ns).expect("nonnegative timestamp"),
                OakImuAcceleration::try_new(0.0, 9.81, 0.0).expect("finite acceleration"),
                SensorAccuracy::High,
            ),
            GyroSample::new(
                DeviceTimestamp::try_from_nanos(timestamp_ns).expect("nonnegative timestamp"),
                OakImuAngularVelocity::try_new(0.0, 0.0, 0.1).expect("finite angular velocity"),
                SensorAccuracy::High,
            ),
        )
    }

    #[test]
    fn stereo_observation_preserves_exact_session_host_and_device_times() {
        let session_id = DeviceSessionId::try_new(99).expect("nonzero session");
        let host_arrival = HostMonotonicTimestamp::from_nanos(u64::MAX);
        let observation = StereoObservation::parse(session_id, host_arrival, stereo_pair(123, 124))
            .expect("nonnegative pair timestamps");

        assert_eq!(observation.session_id(), session_id);
        assert_eq!(observation.host_arrival(), host_arrival);
        assert_eq!(
            observation.left_device_timestamp(),
            DeviceTimestamp::try_from_nanos(123).expect("nonnegative left timestamp")
        );
        assert_eq!(
            observation.right_device_timestamp(),
            DeviceTimestamp::try_from_nanos(124).expect("nonnegative right timestamp")
        );
        assert_eq!(observation.pair().left().frame_id(), FrameId::new(41));
        assert_eq!(observation.pair().right().frame_id(), FrameId::new(42));
        assert_eq!(
            observation.pair().left().timestamp(),
            Timestamp::from_nanos(123)
        );
        assert_eq!(
            observation.pair().right().timestamp(),
            Timestamp::from_nanos(124)
        );
    }

    #[test]
    fn stereo_observation_rejects_each_negative_timestamp_with_its_side_and_source() {
        for (left_timestamp_ns, right_timestamp_ns, expected_side, expected_nanos) in [
            (
                i64::MIN,
                i64::MIN + 1,
                StereoObservationSide::Left,
                i64::MIN,
            ),
            (0, -1, StereoObservationSide::Right, -1),
        ] {
            let error = StereoObservation::parse(
                session(),
                HostMonotonicTimestamp::from_nanos(10),
                stereo_pair(left_timestamp_ns, right_timestamp_ns),
            )
            .expect_err("device timestamps cannot be negative");
            assert_eq!(
                error,
                StereoObservationError::InvalidDeviceTimestamp {
                    side: expected_side,
                    source: InertialValueError::NegativeDeviceTimestamp {
                        nanos: expected_nanos,
                    },
                }
            );
            assert!(std::error::Error::source(&error).is_some());
        }
    }

    #[test]
    fn stereo_observation_owns_and_returns_the_source_pair_without_copying_frames() {
        let pair = stereo_pair(200, 201);
        let left_pixels = pair.left().data().as_ptr();
        let right_pixels = pair.right().data().as_ptr();
        let observation =
            StereoObservation::parse(session(), HostMonotonicTimestamp::from_nanos(300), pair)
                .expect("valid observation");

        assert_eq!(observation.pair().left().data().as_ptr(), left_pixels);
        assert_eq!(observation.pair().right().data().as_ptr(), right_pixels);
        let pair = observation.into_pair();
        assert_eq!(pair.left().data().as_ptr(), left_pixels);
        assert_eq!(pair.right().data().as_ptr(), right_pixels);
        assert_eq!(pair.left().data(), [1, 2]);
        assert_eq!(pair.right().data(), [3, 4]);
    }

    #[test]
    fn navigation_depth_route_retains_the_newest_observation() {
        let (router, routes, stats) = depth_router(capacity(2), DropPolicy::DropNewest);

        assert_eq!(
            router.route(depth_observation(1, 10, 100)),
            DepthRouteOutcome {
                slam: SendOutcome::Enqueued,
                navigation: SendOutcome::Enqueued,
            }
        );
        assert_eq!(
            router.route(depth_observation(2, 20, 200)),
            DepthRouteOutcome {
                slam: SendOutcome::Enqueued,
                navigation: SendOutcome::DroppedOldest,
            }
        );

        let latest = routes
            .navigation
            .try_recv()
            .expect("newest navigation depth");
        assert_eq!(latest.frame_id(), FrameId::new(2));
        assert_eq!(
            latest.device_timestamp(),
            DeviceTimestamp::try_from_nanos(20).expect("nonnegative fixture time")
        );
        assert_eq!(
            latest.host_arrival(),
            HostMonotonicTimestamp::from_nanos(200)
        );
        assert_eq!(latest.session_id(), session());
        assert!(routes.navigation.try_recv().is_err());
        assert_eq!(stats.snapshot().navigation.dropped_oldest, 1);
    }

    #[test]
    fn depth_destinations_share_the_pixel_allocation_without_an_outer_allocation() {
        let (router, routes, _) = depth_router(capacity(1), DropPolicy::DropNewest);
        let original = depth(7, 70);
        let original_pixels = original.depth_m().as_ptr();
        let observation =
            DepthObservation::parse(session(), HostMonotonicTimestamp::from_nanos(700), original)
                .expect("valid observation");

        assert_eq!(
            router.route(observation),
            DepthRouteOutcome {
                slam: SendOutcome::Enqueued,
                navigation: SendOutcome::Enqueued,
            }
        );
        let slam = routes.slam.try_recv().expect("SLAM depth");
        let navigation = routes.navigation.try_recv().expect("navigation depth");

        assert_eq!(original_pixels, slam.depth_m().as_ptr());
        assert_eq!(original_pixels, navigation.depth().depth_m().as_ptr());
        assert_eq!(
            slam.depth_m().as_ptr(),
            navigation.depth().depth_m().as_ptr()
        );
        assert_eq!(slam.frame_id(), navigation.frame_id());
        assert_eq!(slam.timestamp(), navigation.depth().timestamp());
    }

    #[test]
    fn disconnected_depth_destination_does_not_suppress_the_other() {
        let (router, routes, stats) = depth_router(capacity(1), DropPolicy::DropNewest);
        drop(routes.navigation);

        assert_eq!(
            router.route(depth_observation(3, 30, 300)),
            DepthRouteOutcome {
                slam: SendOutcome::Enqueued,
                navigation: SendOutcome::Disconnected,
            }
        );
        assert_eq!(
            routes
                .slam
                .try_recv()
                .expect("independent SLAM depth")
                .frame_id(),
            FrameId::new(3)
        );
        assert_eq!(stats.snapshot().navigation.disconnected, 1);

        let (router, routes, _) = depth_router(capacity(1), DropPolicy::DropNewest);
        drop(routes.slam);
        assert_eq!(
            router.route(depth_observation(4, 40, 400)),
            DepthRouteOutcome {
                slam: SendOutcome::Disconnected,
                navigation: SendOutcome::Enqueued,
            }
        );
        assert_eq!(
            routes
                .navigation
                .try_recv()
                .expect("independent navigation depth")
                .frame_id(),
            FrameId::new(4)
        );
    }

    #[test]
    fn depth_observation_rejects_negative_device_time_before_routing() {
        let error = DepthObservation::parse(
            session(),
            HostMonotonicTimestamp::from_nanos(1),
            DepthImage::new(
                FrameId::new(1),
                Timestamp::from_nanos(-1),
                2,
                1,
                vec![1.0, 1.0],
            )
            .expect("metric payload is valid independently of weak capture time"),
        )
        .expect_err("navigation cannot join negative depth time to a device session");
        assert!(matches!(
            error,
            DepthObservationError::InvalidDeviceTimestamp(
                InertialValueError::NegativeDeviceTimestamp { nanos: -1 }
            )
        ));
    }

    #[test]
    fn imu_route_overflow_retains_freshest_reports_and_accounts_for_loss() {
        let (mut router, route, stats) = imu_report_router(session(), capacity(2));

        assert_eq!(
            router.route(imu_report(0)),
            Ok(ImuReportRouteOutcome {
                ordering: InertialOrderOutcome::FirstReport,
                delivery: SendOutcome::Enqueued,
            })
        );
        assert_eq!(
            router.route(imu_report(1)),
            Ok(ImuReportRouteOutcome {
                ordering: InertialOrderOutcome::Contiguous,
                delivery: SendOutcome::Enqueued,
            })
        );
        assert_eq!(
            router.route(imu_report(2)),
            Ok(ImuReportRouteOutcome {
                ordering: InertialOrderOutcome::Contiguous,
                delivery: SendOutcome::DroppedOldest,
            })
        );

        assert_eq!(
            route
                .reports
                .try_recv()
                .expect("first retained report")
                .sequence(),
            DequeueSequence::new(1)
        );
        assert_eq!(
            route
                .reports
                .try_recv()
                .expect("freshest retained report")
                .sequence(),
            DequeueSequence::new(2)
        );
        assert!(route.reports.try_recv().is_err());
        assert_eq!(
            stats.snapshot(),
            ImuReportRouteStats {
                reports: ChannelStats {
                    enqueued: 3,
                    dropped_oldest: 1,
                    ..ChannelStats::default()
                },
                ..ImuReportRouteStats::default()
            }
        );
    }

    #[test]
    fn imu_route_accounts_for_source_gaps_separately_from_route_overflow() {
        let (mut router, route, stats) = imu_report_router(session(), capacity(2));
        router.route(imu_report(3)).expect("first report");
        route.reports.try_recv().expect("drain first report");

        assert_eq!(
            router.route(imu_report(6)),
            Ok(ImuReportRouteOutcome {
                ordering: InertialOrderOutcome::Gap { missing_reports: 2 },
                delivery: SendOutcome::Enqueued,
            })
        );
        assert_eq!(stats.snapshot().source_gap_events, 1);
        assert_eq!(stats.snapshot().source_missing_reports, 2);
        assert_eq!(stats.snapshot().reports.dropped_oldest, 0);
    }

    #[test]
    fn imu_ordering_failure_is_counted_and_not_enqueued() {
        let (mut router, route, stats) = imu_report_router(session(), capacity(2));
        router.route(imu_report(4)).expect("first report");
        route.reports.try_recv().expect("drain first report");

        assert!(router.route(imu_report(4)).is_err());
        assert!(route.reports.try_recv().is_err());
        assert_eq!(stats.snapshot().ordering_rejected, 1);
    }
}
