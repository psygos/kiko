use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    ChannelCapacity, ChannelStats, ChannelStatsHandle, DepthImage, DeviceSessionId, DropPolicy,
    DropReceiver, DropSender, ImuReport, InertialOrderOutcome, InertialOrderTracker,
    InertialOrderingError, SendOutcome, bounded_channel,
};

/// The two independently backpressured destinations for one depth observation.
#[derive(Debug)]
pub struct DepthRoutes {
    pub slam: DropReceiver<DepthImage>,
    pub navigation: DropReceiver<DepthImage>,
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
    navigation: DropSender<DepthImage>,
}

impl DepthRouter {
    /// Attempt both destinations even if either has disconnected or rejects
    /// the observation.
    pub fn route(&self, depth: DepthImage) -> DepthRouteOutcome {
        let slam = self.slam.try_send(depth.clone());
        let navigation = self.navigation.try_send(depth);
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
        AccelSample, DequeueSequence, DeviceSessionId, DeviceTimestamp, FrameId, GyroSample,
        HostMonotonicTimestamp, OakImuAcceleration, OakImuAngularVelocity, SensorAccuracy,
        Timestamp,
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
    fn navigation_depth_route_retains_the_newest_observation() {
        let (router, routes, stats) = depth_router(capacity(2), DropPolicy::DropNewest);

        assert_eq!(
            router.route(depth(1, 10)),
            DepthRouteOutcome {
                slam: SendOutcome::Enqueued,
                navigation: SendOutcome::Enqueued,
            }
        );
        assert_eq!(
            router.route(depth(2, 20)),
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
        assert_eq!(latest.timestamp(), Timestamp::from_nanos(20));
        assert!(routes.navigation.try_recv().is_err());
        assert_eq!(stats.snapshot().navigation.dropped_oldest, 1);
    }

    #[test]
    fn depth_destinations_share_the_pixel_allocation_without_an_outer_allocation() {
        let (router, routes, _) = depth_router(capacity(1), DropPolicy::DropNewest);
        let original = depth(7, 70);
        let original_pixels = original.depth_m().as_ptr();

        assert_eq!(
            router.route(original),
            DepthRouteOutcome {
                slam: SendOutcome::Enqueued,
                navigation: SendOutcome::Enqueued,
            }
        );
        let slam = routes.slam.try_recv().expect("SLAM depth");
        let navigation = routes.navigation.try_recv().expect("navigation depth");

        assert_eq!(original_pixels, slam.depth_m().as_ptr());
        assert_eq!(original_pixels, navigation.depth_m().as_ptr());
        assert_eq!(slam.depth_m().as_ptr(), navigation.depth_m().as_ptr());
        assert_eq!(slam.frame_id(), navigation.frame_id());
        assert_eq!(slam.timestamp(), navigation.timestamp());
    }

    #[test]
    fn disconnected_depth_destination_does_not_suppress_the_other() {
        let (router, routes, stats) = depth_router(capacity(1), DropPolicy::DropNewest);
        drop(routes.navigation);

        assert_eq!(
            router.route(depth(3, 30)),
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
            router.route(depth(4, 40)),
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
