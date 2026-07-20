//! Collision-checked MPC references for direct body-twist and frontier-yaw control.
//!
//! This module never produces wheel PWM. It turns already-admitted authority
//! inputs into finite end-of-step odom-frame references. The ordinary MPC,
//! immutable local-costmap provenance, solver deadline, final revalidation,
//! shadow journal, and applied-command evidence path remain the only route to
//! actuation.

use std::fmt;
use std::num::{NonZeroU64, NonZeroUsize};

use crate::{DeviceSessionId, HostMonotonicTimestamp, MapSnapshot};

use super::frames::{MapToOdom, PlanarTransformError};
use super::frontier::{FrontierInPlaceScan, FrontierUnknownDirection, FrontierUnknownDirections};
use super::manual_drive::{
    BodyVelocityTargetV1, ManualDriveAcceptedIntent, ManualDriveAcceptedTarget, ManualDriveSequence,
};
use super::mpc::{
    FrontierYawReferenceIdentityV1, MAX_SUPPORTED_ABS_ODOM_COORDINATE_M, ManualReferenceIdentityV1,
    MotionValueError, MpcConfigV1, MpcReferenceParseError, MpcReferenceSourceV1, MpcReferenceV1,
    NavigationEpochError, NavigationEpochV1, OdomPoseV1, OdomReferencePointV1,
    ReferenceBuilderRevisionV1, ReferenceIdentityError,
};
use super::odometry::OdomSegmentId;
use super::reference::MAX_SUPPORTED_ABS_REFERENCE_YAW_RATE_RAD_S;

/// Projection of the supervisor's nonzero authority ID into navigation.
///
/// The trait is deliberately tiny: callers cannot inject timestamps, mode, or
/// lifecycle claims while converting the exact lease carried by
/// `ManualDriveAcceptedTarget`.
pub trait NumericAuthorityLeaseId: sealed::Sealed + Copy {
    fn get(self) -> u64;
}

mod sealed {
    pub trait Sealed {}

    impl Sealed for std::num::NonZeroU64 {}

    #[cfg(feature = "agent-runtime")]
    impl Sealed for kiko_supervisor_core::AuthorityLeaseId {}
}

impl NumericAuthorityLeaseId for NonZeroU64 {
    fn get(self) -> u64 {
        self.get()
    }
}

#[cfg(feature = "agent-runtime")]
impl NumericAuthorityLeaseId for kiko_supervisor_core::AuthorityLeaseId {
    fn get(self) -> u64 {
        self.get()
    }
}

/// One accepted nonzero manual velocity command, ready for the coordinator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManualMpcCommandV1 {
    identity: ManualReferenceIdentityV1,
    target: BodyVelocityTargetV1,
}

impl ManualMpcCommandV1 {
    pub fn try_from_accepted<LeaseId>(
        accepted: ManualDriveAcceptedTarget<LeaseId>,
    ) -> Result<Self, ManualMpcCommandError>
    where
        LeaseId: NumericAuthorityLeaseId,
    {
        if accepted.intent() != ManualDriveAcceptedIntent::Velocity {
            return Err(ManualMpcCommandError::ExplicitStop);
        }
        let target = accepted.target();
        if target.is_stop() {
            return Err(ManualMpcCommandError::ZeroVelocityInvariant);
        }
        let identity = ManualReferenceIdentityV1::try_new(
            accepted.authority_lease_id().get(),
            accepted.sequence().get(),
            accepted.valid_through_exclusive(),
            target.forward_velocity_mps(),
            target.yaw_rate_rad_s(),
        )
        .map_err(ManualMpcCommandError::Identity)?;
        Ok(Self { identity, target })
    }

    pub fn identity(self) -> ManualReferenceIdentityV1 {
        self.identity
    }

    pub fn authority_lease_id(self) -> NonZeroU64 {
        self.identity.authority_lease_id()
    }

    pub fn sequence(self) -> ManualDriveSequence {
        ManualDriveSequence::from_raw(self.identity.command_sequence())
    }

    pub fn target(self) -> BodyVelocityTargetV1 {
        self.target
    }

    pub fn valid_through_exclusive(self) -> HostMonotonicTimestamp {
        self.identity.valid_through_exclusive()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ManualMpcCommandError {
    ExplicitStop,
    ZeroVelocityInvariant,
    Identity(ReferenceIdentityError),
}

impl fmt::Display for ManualMpcCommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "manual command cannot form an MPC reference: {self:?}"
        )
    }
}

impl std::error::Error for ManualMpcCommandError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Identity(source) => Some(source),
            Self::ExplicitStop | Self::ZeroVelocityInvariant => None,
        }
    }
}

/// Builds an exact body-twist reference bounded by its exclusive validity.
///
/// The commanded twist is integrated only over the remaining valid interval.
/// Every horizon sample at or after the exclusive deadline holds that terminal
/// pose with zero velocity, so an authorized-now command cannot bias MPC toward
/// continued motion after its deadman or authority lease expires.
#[derive(Default)]
pub struct ManualReferenceBuilderV1;

impl ManualReferenceBuilderV1 {
    pub fn build(
        &mut self,
        epoch: NavigationEpochV1,
        command: ManualMpcCommandV1,
        initial_pose: OdomPoseV1,
        mpc_config: MpcConfigV1,
        created_at: HostMonotonicTimestamp,
    ) -> Result<MpcReferenceV1<'static>, ManualReferenceBuildError> {
        if created_at >= command.valid_through_exclusive() {
            return Err(ManualReferenceBuildError::CommandExpired {
                valid_through_exclusive: command.valid_through_exclusive(),
                observed_at: created_at,
            });
        }
        let expected =
            super::mpc::NavigationReferenceIdentityV1::ManualBodyTwist(command.identity());
        if epoch.reference_identity() != expected {
            return Err(ManualReferenceBuildError::EpochIdentityMismatch {
                expected: Box::new(expected),
                actual: Box::new(epoch.reference_identity()),
            });
        }
        let validity_ns = command
            .valid_through_exclusive()
            .as_nanos()
            .checked_sub(created_at.as_nanos())
            .ok_or(ManualReferenceBuildError::CommandExpired {
                valid_through_exclusive: command.valid_through_exclusive(),
                observed_at: created_at,
            })?;
        let validity_s = validity_ns as f64 / 1.0e9;

        let horizon = mpc_config.horizon_steps();
        let mut points = Vec::new();
        points
            .try_reserve_exact(horizon)
            .map_err(|_| ManualReferenceBuildError::Allocation { elements: horizon })?;
        for step in 1..=horizon {
            let elapsed_s = elapsed_seconds(step, mpc_config.step_period_s())
                .ok_or(ManualReferenceBuildError::NumericalTime { step })?;
            let elapsed_ceiling_ns = elapsed_nanoseconds_ceiling(elapsed_s)
                .ok_or(ManualReferenceBuildError::NumericalTime { step })?;
            // Host authority is represented in integer nanoseconds. Round a
            // fractional sample offset outward so binary64 multiplication
            // cannot turn an exact exclusive deadline into a still-active
            // reference point (for example, 0.7 s * 3).
            let active_at_sample = elapsed_ceiling_ns < validity_ns;
            let projected_elapsed_s = if active_at_sample {
                elapsed_s
            } else {
                validity_s
            };
            let projected =
                integrate_constant_body_twist(initial_pose, command.target(), projected_elapsed_s)
                    .map_err(|source| ManualReferenceBuildError::Point { step, source })?;
            let point = if active_at_sample {
                projected
            } else {
                OdomReferencePointV1::try_new(projected.pose(), 0.0, 0.0)
                    .map_err(|source| ManualReferenceBuildError::Point { step, source })?
            };
            points.push(point);
        }

        MpcReferenceV1::from_typed_points(
            ReferenceBuilderRevisionV1::ValidityBoundedBodyTwistV1,
            created_at,
            mpc_config.step_period_s(),
            points,
            mpc_config,
            epoch,
            MpcReferenceSourceV1::ManualBodyTwist(command.identity()),
        )
        .map_err(|source| ManualReferenceBuildError::Reference(Box::new(source)))
    }
}

#[derive(Debug, PartialEq)]
pub enum ManualReferenceBuildError {
    CommandExpired {
        valid_through_exclusive: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
    },
    EpochIdentityMismatch {
        expected: Box<super::mpc::NavigationReferenceIdentityV1>,
        actual: Box<super::mpc::NavigationReferenceIdentityV1>,
    },
    NumericalTime {
        step: usize,
    },
    Point {
        step: usize,
        source: MotionValueError,
    },
    Allocation {
        elements: usize,
    },
    Reference(Box<MpcReferenceParseError>),
}

impl fmt::Display for ManualReferenceBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "cannot build manual MPC reference: {self:?}")
    }
}

impl std::error::Error for ManualReferenceBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Point { source, .. } => Some(source),
            Self::Reference(source) => Some(source.as_ref()),
            Self::CommandExpired { .. }
            | Self::EpochIdentityMismatch { .. }
            | Self::NumericalTime { .. }
            | Self::Allocation { .. } => None,
        }
    }
}

/// Explicit sign of rotation for a frontier scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontierYawTurnDirectionV1 {
    CounterClockwise,
    Clockwise,
}

/// Bounded scan budget parsed once before an explore lease can use it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierYawScanBudgetV1 {
    maximum_abs_yaw_rate_rad_s: f64,
    yaw_travel_limit_exclusive_rad: f64,
    maximum_scan_origin_displacement_m: f64,
    maximum_duration_ns: NonZeroU64,
}

impl FrontierYawScanBudgetV1 {
    pub fn try_new(
        maximum_abs_yaw_rate_rad_s: f64,
        yaw_travel_limit_exclusive_rad: f64,
        maximum_scan_origin_displacement_m: f64,
        maximum_duration_ns: u64,
    ) -> Result<Self, FrontierYawScanBudgetError> {
        if !maximum_abs_yaw_rate_rad_s.is_finite() {
            return Err(FrontierYawScanBudgetError::NonFinite {
                field: "maximum_abs_yaw_rate_rad_s",
                value: maximum_abs_yaw_rate_rad_s,
            });
        }
        if maximum_abs_yaw_rate_rad_s <= 0.0 {
            return Err(FrontierYawScanBudgetError::NotPositive {
                field: "maximum_abs_yaw_rate_rad_s",
                value: maximum_abs_yaw_rate_rad_s,
            });
        }
        if maximum_abs_yaw_rate_rad_s > MAX_SUPPORTED_ABS_REFERENCE_YAW_RATE_RAD_S {
            return Err(FrontierYawScanBudgetError::AboveMaximum {
                field: "maximum_abs_yaw_rate_rad_s",
                value: maximum_abs_yaw_rate_rad_s,
                maximum: MAX_SUPPORTED_ABS_REFERENCE_YAW_RATE_RAD_S,
            });
        }
        if !yaw_travel_limit_exclusive_rad.is_finite() {
            return Err(FrontierYawScanBudgetError::NonFinite {
                field: "yaw_travel_limit_exclusive_rad",
                value: yaw_travel_limit_exclusive_rad,
            });
        }
        if yaw_travel_limit_exclusive_rad <= 0.0
            || yaw_travel_limit_exclusive_rad > std::f64::consts::TAU
        {
            return Err(FrontierYawScanBudgetError::YawTravelLimitOutsideBounds {
                value_rad: yaw_travel_limit_exclusive_rad,
                maximum_rad: std::f64::consts::TAU,
            });
        }
        if !maximum_scan_origin_displacement_m.is_finite() {
            return Err(FrontierYawScanBudgetError::NonFinite {
                field: "maximum_scan_origin_displacement_m",
                value: maximum_scan_origin_displacement_m,
            });
        }
        if maximum_scan_origin_displacement_m < 0.0 {
            return Err(FrontierYawScanBudgetError::Negative {
                field: "maximum_scan_origin_displacement_m",
                value: maximum_scan_origin_displacement_m,
            });
        }
        let maximum_supported_displacement_m =
            2.0 * std::f64::consts::SQRT_2 * MAX_SUPPORTED_ABS_ODOM_COORDINATE_M;
        if maximum_scan_origin_displacement_m > maximum_supported_displacement_m {
            return Err(FrontierYawScanBudgetError::AboveMaximum {
                field: "maximum_scan_origin_displacement_m",
                value: maximum_scan_origin_displacement_m,
                maximum: maximum_supported_displacement_m,
            });
        }
        let maximum_duration_ns =
            NonZeroU64::new(maximum_duration_ns).ok_or(FrontierYawScanBudgetError::ZeroDuration)?;
        Ok(Self {
            maximum_abs_yaw_rate_rad_s,
            yaw_travel_limit_exclusive_rad,
            maximum_scan_origin_displacement_m: canonical_zero(maximum_scan_origin_displacement_m),
            maximum_duration_ns,
        })
    }

    pub fn maximum_abs_yaw_rate_rad_s(self) -> f64 {
        self.maximum_abs_yaw_rate_rad_s
    }
    pub fn yaw_travel_limit_exclusive_rad(self) -> f64 {
        self.yaw_travel_limit_exclusive_rad
    }
    pub fn maximum_scan_origin_displacement_m(self) -> f64 {
        self.maximum_scan_origin_displacement_m
    }
    pub fn maximum_duration_ns(self) -> NonZeroU64 {
        self.maximum_duration_ns
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrontierYawScanBudgetError {
    NonFinite {
        field: &'static str,
        value: f64,
    },
    NotPositive {
        field: &'static str,
        value: f64,
    },
    Negative {
        field: &'static str,
        value: f64,
    },
    AboveMaximum {
        field: &'static str,
        value: f64,
        maximum: f64,
    },
    YawTravelLimitOutsideBounds {
        value_rad: f64,
        maximum_rad: f64,
    },
    ZeroDuration,
}

impl fmt::Display for FrontierYawScanBudgetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid frontier yaw-scan budget: {self:?}")
    }
}

impl std::error::Error for FrontierYawScanBudgetError {}

/// Exact frontier evidence and authority for one deliberate yaw target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontierYawScanCommandV1 {
    authority_lease_id: NonZeroU64,
    scan_sequence: u64,
    scan: FrontierInPlaceScan,
    target_direction: FrontierUnknownDirection,
    turn_direction: FrontierYawTurnDirectionV1,
    started_at: HostMonotonicTimestamp,
    valid_through_exclusive: HostMonotonicTimestamp,
    budget: FrontierYawScanBudgetV1,
}

impl FrontierYawScanCommandV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new<LeaseId: NumericAuthorityLeaseId>(
        authority_lease_id: LeaseId,
        scan_sequence: u64,
        scan: FrontierInPlaceScan,
        target_direction: FrontierUnknownDirection,
        turn_direction: FrontierYawTurnDirectionV1,
        started_at: HostMonotonicTimestamp,
        authority_expires_at_exclusive: HostMonotonicTimestamp,
        budget: FrontierYawScanBudgetV1,
    ) -> Result<Self, FrontierYawScanCommandError> {
        if !scan.unknown_directions().contains(target_direction) {
            return Err(FrontierYawScanCommandError::DirectionNotInFrontier {
                target: target_direction,
                available: scan.unknown_directions(),
            });
        }
        if started_at >= authority_expires_at_exclusive {
            return Err(FrontierYawScanCommandError::AuthorityExpired {
                expires_at_exclusive: authority_expires_at_exclusive,
                observed_at: started_at,
            });
        }
        let scan_deadline_ns = started_at
            .as_nanos()
            .checked_add(budget.maximum_duration_ns().get())
            .ok_or(FrontierYawScanCommandError::DeadlineOverflow {
                started_at,
                maximum_duration_ns: budget.maximum_duration_ns(),
            })?;
        let valid_through_exclusive = HostMonotonicTimestamp::from_nanos(
            scan_deadline_ns.min(authority_expires_at_exclusive.as_nanos()),
        );
        Ok(Self {
            authority_lease_id: NonZeroU64::new(authority_lease_id.get())
                .ok_or(FrontierYawScanCommandError::ZeroAuthorityLeaseId)?,
            scan_sequence,
            scan,
            target_direction,
            turn_direction,
            started_at,
            valid_through_exclusive,
            budget,
        })
    }

    pub fn authority_lease_id(self) -> NonZeroU64 {
        self.authority_lease_id
    }
    pub fn scan_sequence(self) -> u64 {
        self.scan_sequence
    }
    pub fn scan(self) -> FrontierInPlaceScan {
        self.scan
    }
    pub fn target_direction(self) -> FrontierUnknownDirection {
        self.target_direction
    }
    pub fn turn_direction(self) -> FrontierYawTurnDirectionV1 {
        self.turn_direction
    }
    pub fn started_at(self) -> HostMonotonicTimestamp {
        self.started_at
    }
    pub fn valid_through_exclusive(self) -> HostMonotonicTimestamp {
        self.valid_through_exclusive
    }
    pub fn budget(self) -> FrontierYawScanBudgetV1 {
        self.budget
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FrontierYawScanCommandError {
    ZeroAuthorityLeaseId,
    DirectionNotInFrontier {
        target: FrontierUnknownDirection,
        available: FrontierUnknownDirections,
    },
    AuthorityExpired {
        expires_at_exclusive: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
    },
    DeadlineOverflow {
        started_at: HostMonotonicTimestamp,
        maximum_duration_ns: NonZeroU64,
    },
}

impl fmt::Display for FrontierYawScanCommandError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "frontier evidence cannot form a yaw command: {self:?}"
        )
    }
}

impl std::error::Error for FrontierYawScanCommandError {}

#[derive(Default)]
pub struct FrontierYawReferenceBuilderV1;

impl FrontierYawReferenceBuilderV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        &mut self,
        command: FrontierYawScanCommandV1,
        device_session_id: DeviceSessionId,
        odom_segment_id: OdomSegmentId,
        map_snapshot: MapSnapshot,
        map_to_odom: MapToOdom,
        initial_pose: OdomPoseV1,
        mpc_config: MpcConfigV1,
        created_at: HostMonotonicTimestamp,
    ) -> Result<(NavigationEpochV1, MpcReferenceV1<'static>), FrontierYawReferenceBuildError> {
        if created_at < command.started_at() {
            return Err(FrontierYawReferenceBuildError::BeforeScanStart {
                started_at: command.started_at(),
                observed_at: created_at,
            });
        }
        if created_at >= command.valid_through_exclusive() {
            return Err(FrontierYawReferenceBuildError::ScanExpired {
                valid_through_exclusive: command.valid_through_exclusive(),
                observed_at: created_at,
            });
        }
        if map_snapshot.instance_id() != command.scan().map_instance_id() {
            return Err(FrontierYawReferenceBuildError::MapMismatch {
                odometry_map_instance_id: map_snapshot.instance_id(),
                scan_map_instance_id: command.scan().map_instance_id(),
                scan_map_revision: command.scan().map_revision(),
            });
        }

        let odom_to_map = map_to_odom
            .inverse()
            .map_err(FrontierYawReferenceBuildError::ScanOriginTransform)?;
        let current_map_point = odom_to_map
            .transform_point(initial_pose.position())
            .map_err(FrontierYawReferenceBuildError::ScanOriginTransform)?;
        let scan_origin = command.scan().robot_point();
        let displacement_m = (current_map_point.x_m() - scan_origin.x_m())
            .hypot(current_map_point.y_m() - scan_origin.y_m());
        if !displacement_m.is_finite() {
            return Err(FrontierYawReferenceBuildError::ScanOriginDistanceNotFinite);
        }
        if displacement_m > command.budget().maximum_scan_origin_displacement_m() {
            return Err(FrontierYawReferenceBuildError::ScanOriginMoved {
                expected_map_x_m: scan_origin.x_m(),
                expected_map_y_m: scan_origin.y_m(),
                actual_map_x_m: current_map_point.x_m(),
                actual_map_y_m: current_map_point.y_m(),
                displacement_m,
                maximum_m: command.budget().maximum_scan_origin_displacement_m(),
            });
        }

        let target_map_yaw_rad = frontier_direction_yaw(command.target_direction());
        let target_odom_yaw_rad =
            normalize_angle(map_to_odom.source_yaw_in_destination_rad() + target_map_yaw_rad);
        let signed_travel_rad = directed_yaw_delta(
            initial_pose.yaw_rad(),
            target_odom_yaw_rad,
            command.turn_direction(),
        )
        .ok_or(FrontierYawReferenceBuildError::TargetAlreadyReached {
            target_map_yaw_rad,
            target_odom_yaw_rad,
        })?;
        let required_travel_rad = signed_travel_rad.abs();
        if required_travel_rad >= command.budget().yaw_travel_limit_exclusive_rad() {
            return Err(FrontierYawReferenceBuildError::YawBudgetExceeded {
                required_rad: required_travel_rad,
                maximum_exclusive_rad: command.budget().yaw_travel_limit_exclusive_rad(),
            });
        }
        let remaining_exclusive_ns = command
            .valid_through_exclusive()
            .as_nanos()
            .checked_sub(created_at.as_nanos())
            .ok_or(FrontierYawReferenceBuildError::ScanExpired {
                valid_through_exclusive: command.valid_through_exclusive(),
                observed_at: created_at,
            })?;
        let required_s = required_travel_rad / command.budget().maximum_abs_yaw_rate_rad_s();
        let required_ns = required_s * 1.0e9;
        if !required_ns.is_finite() || required_ns >= u64::MAX as f64 {
            return Err(
                FrontierYawReferenceBuildError::DurationComputationOutsideBounds { required_s },
            );
        }
        let required_ceiling_ns = required_ns.ceil() as u64;
        if required_ceiling_ns >= remaining_exclusive_ns {
            return Err(FrontierYawReferenceBuildError::DurationBudgetExceeded {
                required_ceiling_ns,
                remaining_exclusive_ns,
            });
        }
        let signed_yaw_rate_rad_s =
            signed_travel_rad.signum() * command.budget().maximum_abs_yaw_rate_rad_s();
        let identity = FrontierYawReferenceIdentityV1::try_new(
            command.authority_lease_id().get(),
            command.scan_sequence(),
            command.scan().map_instance_id(),
            command.scan().map_revision(),
            command.scan().column(),
            command.scan().row(),
            scan_origin.x_m(),
            scan_origin.y_m(),
            command.budget().maximum_scan_origin_displacement_m(),
            target_map_yaw_rad,
            signed_yaw_rate_rad_s,
            command.budget().yaw_travel_limit_exclusive_rad(),
            command.valid_through_exclusive(),
        )
        .map_err(FrontierYawReferenceBuildError::Identity)?;
        let epoch = NavigationEpochV1::for_frontier_in_place_yaw(
            device_session_id,
            odom_segment_id,
            map_snapshot,
            identity,
        )
        .map_err(FrontierYawReferenceBuildError::Epoch)?;

        let horizon = mpc_config.horizon_steps();
        let mut points = Vec::new();
        points
            .try_reserve_exact(horizon)
            .map_err(|_| FrontierYawReferenceBuildError::Allocation { elements: horizon })?;
        for step in 1..=horizon {
            let elapsed_s = elapsed_seconds(step, mpc_config.step_period_s())
                .ok_or(FrontierYawReferenceBuildError::NumericalTime { step })?;
            let requested_travel = command
                .budget()
                .maximum_abs_yaw_rate_rad_s()
                .mul_add(elapsed_s, 0.0);
            if !requested_travel.is_finite() {
                return Err(FrontierYawReferenceBuildError::NumericalTime { step });
            }
            let travel = requested_travel.min(required_travel_rad);
            let yaw_rad = normalize_angle(
                signed_travel_rad
                    .signum()
                    .mul_add(travel, initial_pose.yaw_rad()),
            );
            let yaw_rate_rad_s = if travel < required_travel_rad {
                signed_yaw_rate_rad_s
            } else {
                0.0
            };
            let pose = OdomPoseV1::try_new(
                initial_pose.position().x_m(),
                initial_pose.position().y_m(),
                yaw_rad,
            )
            .map_err(|source| FrontierYawReferenceBuildError::Point { step, source })?;
            points.push(
                OdomReferencePointV1::try_new(pose, 0.0, yaw_rate_rad_s)
                    .map_err(|source| FrontierYawReferenceBuildError::Point { step, source })?,
            );
        }

        let reference = MpcReferenceV1::from_typed_points(
            ReferenceBuilderRevisionV1::BoundedFrontierYawV1,
            created_at,
            mpc_config.step_period_s(),
            points,
            mpc_config,
            epoch,
            MpcReferenceSourceV1::FrontierInPlaceYaw(identity),
        )
        .map_err(|source| FrontierYawReferenceBuildError::Reference(Box::new(source)))?;
        Ok((epoch, reference))
    }
}

#[derive(Debug, PartialEq)]
pub enum FrontierYawReferenceBuildError {
    BeforeScanStart {
        started_at: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
    },
    ScanExpired {
        valid_through_exclusive: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
    },
    MapMismatch {
        odometry_map_instance_id: crate::map::MapInstanceId,
        scan_map_instance_id: crate::map::MapInstanceId,
        scan_map_revision: u64,
    },
    ScanOriginTransform(PlanarTransformError),
    ScanOriginDistanceNotFinite,
    ScanOriginMoved {
        expected_map_x_m: f64,
        expected_map_y_m: f64,
        actual_map_x_m: f64,
        actual_map_y_m: f64,
        displacement_m: f64,
        maximum_m: f64,
    },
    TargetAlreadyReached {
        target_map_yaw_rad: f64,
        target_odom_yaw_rad: f64,
    },
    YawBudgetExceeded {
        required_rad: f64,
        maximum_exclusive_rad: f64,
    },
    DurationBudgetExceeded {
        required_ceiling_ns: u64,
        remaining_exclusive_ns: u64,
    },
    DurationComputationOutsideBounds {
        required_s: f64,
    },
    NumericalTime {
        step: usize,
    },
    Point {
        step: usize,
        source: MotionValueError,
    },
    Allocation {
        elements: usize,
    },
    Identity(ReferenceIdentityError),
    Epoch(NavigationEpochError),
    Reference(Box<MpcReferenceParseError>),
}

impl fmt::Display for FrontierYawReferenceBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot build frontier yaw MPC reference: {self:?}"
        )
    }
}

impl std::error::Error for FrontierYawReferenceBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ScanOriginTransform(source) => Some(source),
            Self::Identity(source) => Some(source),
            Self::Epoch(source) => Some(source),
            Self::Reference(source) => Some(source.as_ref()),
            Self::Point { source, .. } => Some(source),
            Self::BeforeScanStart { .. }
            | Self::ScanExpired { .. }
            | Self::MapMismatch { .. }
            | Self::ScanOriginDistanceNotFinite
            | Self::ScanOriginMoved { .. }
            | Self::TargetAlreadyReached { .. }
            | Self::YawBudgetExceeded { .. }
            | Self::DurationBudgetExceeded { .. }
            | Self::DurationComputationOutsideBounds { .. }
            | Self::NumericalTime { .. }
            | Self::Allocation { .. } => None,
        }
    }
}

fn elapsed_seconds(step: usize, period_s: f64) -> Option<f64> {
    let step = NonZeroUsize::new(step)?;
    let elapsed_s = period_s * step.get() as f64;
    elapsed_s.is_finite().then_some(elapsed_s)
}

fn elapsed_nanoseconds_ceiling(elapsed_s: f64) -> Option<u64> {
    let elapsed_ns = elapsed_s * 1.0e9;
    if !elapsed_ns.is_finite() || elapsed_ns < 0.0 || elapsed_ns >= u64::MAX as f64 {
        return None;
    }
    Some(elapsed_ns.ceil() as u64)
}

fn integrate_constant_body_twist(
    initial_pose: OdomPoseV1,
    target: BodyVelocityTargetV1,
    elapsed_s: f64,
) -> Result<OdomReferencePointV1, MotionValueError> {
    let yaw_excursion = target.yaw_rate_rad_s() * elapsed_s;
    let (sinc, cosc) = stable_sinc_cosc(yaw_excursion);
    let distance_m = target.forward_velocity_mps() * elapsed_s;
    let body_x_m = distance_m * sinc;
    let body_y_m = distance_m * cosc;
    let (sin_yaw, cos_yaw) = initial_pose.yaw_rad().sin_cos();
    let x_m = cos_yaw.mul_add(
        body_x_m,
        (-sin_yaw).mul_add(body_y_m, initial_pose.position().x_m()),
    );
    let y_m = sin_yaw.mul_add(
        body_x_m,
        cos_yaw.mul_add(body_y_m, initial_pose.position().y_m()),
    );
    let pose = OdomPoseV1::try_new(
        x_m,
        y_m,
        normalize_angle(initial_pose.yaw_rad() + yaw_excursion),
    )?;
    OdomReferencePointV1::try_new(pose, target.forward_velocity_mps(), target.yaw_rate_rad_s())
}

/// Stable `sin(x)/x` and `(1-cos(x))/x` used by the SE(2) exponential.
fn stable_sinc_cosc(value: f64) -> (f64, f64) {
    if value.abs() < 1.0e-4 {
        let value2 = value * value;
        let sinc = 1.0 + value2 * (-1.0 / 6.0 + value2 / 120.0);
        let cosc = value * (0.5 + value2 * (-1.0 / 24.0 + value2 / 720.0));
        (sinc, cosc)
    } else {
        let (sin, cos) = value.sin_cos();
        (sin / value, (1.0 - cos) / value)
    }
}

fn frontier_direction_yaw(direction: FrontierUnknownDirection) -> f64 {
    match direction {
        FrontierUnknownDirection::PositiveMapX => 0.0,
        FrontierUnknownDirection::PositiveMapY => std::f64::consts::FRAC_PI_2,
        FrontierUnknownDirection::NegativeMapX => std::f64::consts::PI,
        FrontierUnknownDirection::NegativeMapY => -std::f64::consts::FRAC_PI_2,
    }
}

fn directed_yaw_delta(
    from_rad: f64,
    to_rad: f64,
    direction: FrontierYawTurnDirectionV1,
) -> Option<f64> {
    let shortest = normalize_angle(to_rad - from_rad);
    if shortest.abs() <= yaw_arithmetic_equivalence_bound(from_rad, to_rad) {
        return None;
    }
    let counter_clockwise = shortest.rem_euclid(std::f64::consts::TAU);
    Some(match direction {
        FrontierYawTurnDirectionV1::CounterClockwise => counter_clockwise,
        FrontierYawTurnDirectionV1::Clockwise => counter_clockwise - std::f64::consts::TAU,
    })
}

/// Roundoff guard for arithmetic on canonical angles, not a sensor tolerance.
///
/// The scale-relative bound covers a small fixed number of binary64 rounding
/// steps in target-frame addition, canonicalization, and subtraction. A delta
/// outside this machine-resolution envelope retains its explicit turn
/// direction, including a deliberate nearly-full revolution.
fn yaw_arithmetic_equivalence_bound(from_rad: f64, to_rad: f64) -> f64 {
    const ROUNDING_STEPS: f64 = 8.0;
    ROUNDING_STEPS * f64::EPSILON * from_rad.abs().max(to_rad.abs()).max(1.0)
}

fn normalize_angle(angle_rad: f64) -> f64 {
    let positive = angle_rad.rem_euclid(std::f64::consts::TAU);
    let normalized = if positive >= std::f64::consts::PI {
        positive - std::f64::consts::TAU
    } else {
        positive
    };
    if normalized == 0.0 { 0.0 } else { normalized }
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::occupancy::{OccupancyCell, OccupancyGridGeometry, OccupancyGridSnapshot};
    use crate::map::SlamMap;
    use crate::navigation::frames::{MapFrame, OdomFrame, PlanarTransform};
    use crate::navigation::frontier::{
        FrontierExplorer, FrontierExplorerConfig, FrontierSearchOutcome,
    };
    use crate::navigation::manual_drive::{
        MANUAL_DRIVE_COMMAND_V1, ManualAuthoritySnapshot, ManualDriveCommandDto,
        ManualDriveCommandKindDto, ManualDriveConfigV1, ManualDriveConfigV1Dto, ManualDriveCore,
        ManualDriveOutput,
    };
    use crate::navigation::mpc::{
        MPC_CONFIG_V1, MPC_REFERENCE_V1, MpcConfigV1Dto, MpcReferenceV1Dto, OdomReferencePointV1Dto,
    };

    fn host(value: u64) -> HostMonotonicTimestamp {
        HostMonotonicTimestamp::from_nanos(value)
    }

    fn config() -> MpcConfigV1 {
        config_with_timing(0.1, 8)
    }

    fn config_with_timing(step_period_s: f64, horizon_steps: u16) -> MpcConfigV1 {
        MpcConfigV1::parse(MpcConfigV1Dto {
            schema_version: MPC_CONFIG_V1,
            horizon_steps,
            step_period_s,
            integration_substeps: 2,
            optimization_iterations: 1,
            candidates_per_wheel: 3,
            max_rollout_evaluations: 10_000,
            initial_search_radius_percent: 10,
            search_radius_decay_numerator: 1,
            search_radius_decay_denominator: 2,
            left_pwm_min_percent: -50,
            left_pwm_max_percent: 50,
            right_pwm_min_percent: -50,
            right_pwm_max_percent: 50,
            left_max_slew_percent_per_step: 100,
            right_max_slew_percent_per_step: 100,
            max_integration_tube_radius_m: 1.0,
            position_cost_per_m2: 1.0,
            heading_cost_per_rad2: 1.0,
            forward_velocity_cost_s2_per_m2: 1.0,
            yaw_rate_cost_s2_per_rad2: 1.0,
            pwm_cost_per_percent2: 0.001,
            slew_cost_per_percent2: 0.001,
            terminal_state_cost_multiplier: 2.0,
        })
        .expect("valid MPC fixture")
    }

    fn accepted(v: f64, w: f64, sequence: u64) -> ManualMpcCommandV1 {
        accepted_with_deadman(v, w, sequence, 2_000_000_000)
    }

    fn accepted_with_deadman(
        v: f64,
        w: f64,
        sequence: u64,
        deadman_timeout_ns: u64,
    ) -> ManualMpcCommandV1 {
        let lease = NonZeroU64::new(7).unwrap();
        let drive_config = ManualDriveConfigV1::parse(ManualDriveConfigV1Dto {
            schema_version: 1,
            maximum_abs_forward_velocity_mps: 2.0,
            maximum_abs_yaw_rate_rad_s: 2.0,
            maximum_command_age_ns: 10,
            deadman_timeout_ns,
        })
        .unwrap();
        let mut core = ManualDriveCore::new(drive_config, lease, host(1_000));
        let output = core.ingest(
            ManualDriveCommandDto {
                schema_version: MANUAL_DRIVE_COMMAND_V1,
                authority_lease_id: lease,
                sequence,
                command: ManualDriveCommandKindDto::Velocity {
                    forward_velocity_mps: v,
                    yaw_rate_rad_s: w,
                },
            },
            host(1_000),
            host(1_000),
            ManualAuthoritySnapshot::active_manual(lease, host(u64::MAX)),
        );
        let ManualDriveOutput::Accepted(target) = output else {
            panic!("fixture command was rejected")
        };
        ManualMpcCommandV1::try_from_accepted(target).unwrap()
    }

    fn manual_reference(v: f64, w: f64) -> MpcReferenceV1<'static> {
        let map = SlamMap::new().snapshot();
        let command = accepted(v, w, 9);
        let epoch = NavigationEpochV1::for_manual_body_twist(
            DeviceSessionId::try_new(1).unwrap(),
            OdomSegmentId::try_new(1).unwrap(),
            map,
            command.identity(),
        );
        ManualReferenceBuilderV1
            .build(
                epoch,
                command,
                OdomPoseV1::try_new(0.0, 0.0, 0.0).unwrap(),
                config(),
                host(1_050),
            )
            .unwrap()
    }

    fn assert_matches_strict_dto_parse(reference: &MpcReferenceV1<'static>) {
        let config = config();
        let reparsed = MpcReferenceV1::parse_for_source(
            MpcReferenceV1Dto {
                schema_version: MPC_REFERENCE_V1,
                builder_revision: reference.builder_revision() as u32,
                created_at_host_ns: reference.created_at().as_nanos(),
                step_period_s: config.step_period_s(),
                points: reference
                    .points()
                    .iter()
                    .map(|point| OdomReferencePointV1Dto {
                        x_m: point.pose().position().x_m(),
                        y_m: point.pose().position().y_m(),
                        yaw_rad: point.pose().yaw_rad(),
                        forward_velocity_mps: point.forward_velocity_mps(),
                        yaw_rate_rad_s: point.yaw_rate_rad_s(),
                    })
                    .collect(),
            },
            config,
            reference.epoch(),
            reference.source(),
        )
        .expect("typed builder output must retain strict DTO parse semantics");
        assert_eq!(&reparsed, reference);
    }

    #[test]
    fn typed_manual_builder_matches_the_strict_dto_parse_contract() {
        assert_matches_strict_dto_parse(&manual_reference(-0.4, 0.3));
    }

    #[test]
    fn zero_yaw_is_exact_straight_motion_and_reverse_is_preserved() {
        for velocity in [0.75, -0.75] {
            let reference = manual_reference(velocity, 0.0);
            for (index, point) in reference.points().iter().enumerate() {
                let time = (index + 1) as f64 * config().step_period_s();
                assert_eq!(point.pose().position().x_m(), velocity * time);
                assert_eq!(point.pose().position().y_m(), 0.0);
                assert_eq!(point.pose().yaw_rad(), 0.0);
                assert_eq!(point.forward_velocity_mps(), velocity);
                assert_eq!(point.yaw_rate_rad_s(), 0.0);
            }
        }
    }

    #[test]
    fn tiny_positive_and_negative_yaw_are_continuous_with_zero() {
        let straight = manual_reference(0.7, 0.0);
        for yaw_rate in [1.0e-12, -1.0e-12, 1.0e-8, -1.0e-8] {
            let curved = manual_reference(0.7, yaw_rate);
            for (baseline, actual) in straight.points().iter().zip(curved.points()) {
                assert!(
                    (actual.pose().position().x_m() - baseline.pose().position().x_m()).abs()
                        < 1.0e-12
                );
                assert!(actual.pose().position().y_m().abs() <= 3.0e-9);
                assert_eq!(actual.pose().yaw_rad().signum(), yaw_rate.signum());
            }
        }
    }

    #[test]
    fn pure_rotation_never_translates() {
        for yaw_rate in [-1.5, 1.5] {
            let reference = manual_reference(0.0, yaw_rate);
            for point in reference.points() {
                assert_eq!(point.pose().position().x_m(), 0.0);
                assert_eq!(point.pose().position().y_m(), 0.0);
            }
        }
    }

    #[test]
    fn manual_reference_stops_at_exclusive_validity_inside_the_horizon() {
        let map = SlamMap::new().snapshot();
        let initial_pose = OdomPoseV1::try_new(0.0, 0.0, 0.0).unwrap();
        let created_at = host(1_000);

        for (sequence, valid_duration_ns, expected_x_m) in [
            (20, 50_000_000, 0.05),
            (21, 100_000_000, 0.1),
            (22, 150_000_000, 0.15),
        ] {
            let command = accepted_with_deadman(1.0, 0.0, sequence, valid_duration_ns);
            let epoch = NavigationEpochV1::for_manual_body_twist(
                DeviceSessionId::try_new(1).unwrap(),
                OdomSegmentId::try_new(1).unwrap(),
                map,
                command.identity(),
            );
            let reference = ManualReferenceBuilderV1
                .build(epoch, command, initial_pose, config(), created_at)
                .unwrap();

            let active_samples = match valid_duration_ns {
                150_000_000 => 1,
                50_000_000 | 100_000_000 => 0,
                _ => unreachable!("table contains only explicit boundary cases"),
            };
            for (index, point) in reference.points().iter().enumerate() {
                if index < active_samples {
                    assert_eq!(point.pose().position().x_m(), 0.1);
                    assert_eq!(point.forward_velocity_mps(), 1.0);
                } else {
                    assert_eq!(point.pose().position().x_m(), expected_x_m);
                    assert_eq!(point.forward_velocity_mps(), 0.0);
                }
                assert_eq!(point.pose().position().y_m(), 0.0);
                assert_eq!(point.pose().yaw_rad(), 0.0);
                assert_eq!(point.yaw_rate_rad_s(), 0.0);
            }
        }
    }

    #[test]
    fn fractional_binary_step_cannot_cross_an_exact_nanosecond_deadline() {
        let map = SlamMap::new().snapshot();
        let command = accepted_with_deadman(1.0, 0.0, 23, 2_100_000_000);
        let epoch = NavigationEpochV1::for_manual_body_twist(
            DeviceSessionId::try_new(1).unwrap(),
            OdomSegmentId::try_new(1).unwrap(),
            map,
            command.identity(),
        );
        let reference = ManualReferenceBuilderV1
            .build(
                epoch,
                command,
                OdomPoseV1::try_new(0.0, 0.0, 0.0).unwrap(),
                config_with_timing(0.7, 4),
                host(1_000),
            )
            .unwrap();

        assert_eq!(reference.points()[0].forward_velocity_mps(), 1.0);
        assert_eq!(reference.points()[1].forward_velocity_mps(), 1.0);
        assert_eq!(reference.points()[2].pose().position().x_m(), 2.1);
        assert_eq!(reference.points()[2].forward_velocity_mps(), 0.0);
        assert_eq!(reference.points()[3].pose(), reference.points()[2].pose());
        assert_eq!(reference.points()[3].forward_velocity_mps(), 0.0);
    }

    #[test]
    fn se2_exponential_matches_high_resolution_integration_across_domain() {
        for velocity_step in -8..=8 {
            for yaw_step in -8..=8 {
                if velocity_step == 0 && yaw_step == 0 {
                    continue;
                }
                let velocity = velocity_step as f64 * 0.125;
                let yaw_rate = yaw_step as f64 * 0.2;
                let elapsed = 0.8;
                let actual = integrate_constant_body_twist(
                    OdomPoseV1::try_new(0.2, -0.3, 0.4).unwrap(),
                    accepted(velocity, yaw_rate, 1).target(),
                    elapsed,
                )
                .unwrap();
                let substeps = 20_000;
                let dt = elapsed / substeps as f64;
                let mut x: f64 = 0.2;
                let mut y: f64 = -0.3;
                let mut yaw: f64 = 0.4;
                for _ in 0..substeps {
                    let midpoint = yaw + 0.5 * yaw_rate * dt;
                    x = velocity.mul_add(dt * midpoint.cos(), x);
                    y = velocity.mul_add(dt * midpoint.sin(), y);
                    yaw += yaw_rate * dt;
                }
                assert!((actual.pose().position().x_m() - x).abs() < 2.0e-9);
                assert!((actual.pose().position().y_m() - y).abs() < 2.0e-9);
                assert!((normalize_angle(actual.pose().yaw_rad() - yaw)).abs() < 2.0e-12);
            }
        }
    }

    #[test]
    fn command_expiry_and_epoch_identity_equality_fail_closed() {
        let map = SlamMap::new().snapshot();
        let command = accepted(0.5, 0.1, 10);
        let wrong = accepted(0.5, 0.1, 11);
        let wrong_epoch = NavigationEpochV1::for_manual_body_twist(
            DeviceSessionId::try_new(1).unwrap(),
            OdomSegmentId::try_new(1).unwrap(),
            map,
            wrong.identity(),
        );
        assert!(matches!(
            ManualReferenceBuilderV1.build(
                wrong_epoch,
                command,
                OdomPoseV1::try_new(0.0, 0.0, 0.0).unwrap(),
                config(),
                host(1_050),
            ),
            Err(ManualReferenceBuildError::EpochIdentityMismatch { .. })
        ));

        let epoch = NavigationEpochV1::for_manual_body_twist(
            DeviceSessionId::try_new(1).unwrap(),
            OdomSegmentId::try_new(1).unwrap(),
            map,
            command.identity(),
        );
        assert!(matches!(
            ManualReferenceBuilderV1.build(
                epoch,
                command,
                OdomPoseV1::try_new(0.0, 0.0, 0.0).unwrap(),
                config(),
                command.valid_through_exclusive(),
            ),
            Err(ManualReferenceBuildError::CommandExpired { .. })
        ));
    }

    #[test]
    fn reference_rejects_coordinates_that_leave_the_supported_odom_domain() {
        let map = SlamMap::new().snapshot();
        let command = accepted(1.0, 0.0, 12);
        let epoch = NavigationEpochV1::for_manual_body_twist(
            DeviceSessionId::try_new(1).unwrap(),
            OdomSegmentId::try_new(1).unwrap(),
            map,
            command.identity(),
        );
        assert!(matches!(
            ManualReferenceBuilderV1.build(
                epoch,
                command,
                OdomPoseV1::try_new(999_999.9, 0.0, std::f64::consts::TAU).unwrap(),
                config(),
                host(1_050),
            ),
            Err(ManualReferenceBuildError::Point {
                source: MotionValueError::CoordinateOutsideSupportedDomain { .. },
                ..
            })
        ));
    }

    fn in_place_scan() -> (MapSnapshot, FrontierInPlaceScan) {
        let map = SlamMap::new().snapshot();
        let geometry = OccupancyGridGeometry::try_new(1.0, [-1.0, -1.0], 3, 3, 9).unwrap();
        let mut cells = vec![OccupancyCell::Unknown; 9];
        cells[4] = OccupancyCell::Free;
        let snapshot =
            OccupancyGridSnapshot::from_test_cells(geometry, &cells, map.instance_id(), 4);
        let mut explorer = FrontierExplorer::try_new(
            &snapshot,
            FrontierExplorerConfig::try_new(0.0, 9, 9, 72).unwrap(),
        )
        .unwrap();
        let start = super::super::global_planner::PlanStart::for_snapshot(
            super::super::global_planner::MapPoint::try_new(0.5, 0.5).unwrap(),
            &snapshot,
        )
        .unwrap();
        let FrontierSearchOutcome::InPlaceScanRequired(scan) = explorer.select(start).unwrap()
        else {
            panic!("fixture must require an in-place scan")
        };
        (map, scan)
    }

    #[test]
    fn yaw_scan_rate_accepts_the_supported_bound_and_rejects_the_next_float() {
        assert!(
            FrontierYawScanBudgetV1::try_new(
                MAX_SUPPORTED_ABS_REFERENCE_YAW_RATE_RAD_S,
                std::f64::consts::PI,
                0.0,
                1,
            )
            .is_ok()
        );
        let above = f64::from_bits(MAX_SUPPORTED_ABS_REFERENCE_YAW_RATE_RAD_S.to_bits() + 1);
        assert!(matches!(
            FrontierYawScanBudgetV1::try_new(above, std::f64::consts::PI, 0.0, 1),
            Err(FrontierYawScanBudgetError::AboveMaximum {
                field: "maximum_abs_yaw_rate_rad_s",
                value,
                maximum,
            }) if value == above && maximum == MAX_SUPPORTED_ABS_REFERENCE_YAW_RATE_RAD_S
        ));
    }

    #[test]
    fn directed_yaw_treats_only_machine_roundoff_as_reached() {
        let base: f64 = 1.25;
        let adjacent = f64::from_bits(base.to_bits() + 1);
        for direction in [
            FrontierYawTurnDirectionV1::CounterClockwise,
            FrontierYawTurnDirectionV1::Clockwise,
        ] {
            assert_eq!(directed_yaw_delta(base, base, direction), None);
            assert_eq!(directed_yaw_delta(base, adjacent, direction), None);
            assert_eq!(directed_yaw_delta(adjacent, base, direction), None);
        }

        let meaningful = 16.0 * f64::EPSILON;
        assert!(meaningful > yaw_arithmetic_equivalence_bound(0.0, meaningful));
        let ccw_short = directed_yaw_delta(
            0.0,
            meaningful,
            FrontierYawTurnDirectionV1::CounterClockwise,
        )
        .unwrap();
        let cw_long =
            directed_yaw_delta(0.0, meaningful, FrontierYawTurnDirectionV1::Clockwise).unwrap();
        assert_eq!(ccw_short, meaningful);
        assert_eq!(cw_long, meaningful - std::f64::consts::TAU);

        let ccw_long = directed_yaw_delta(
            0.0,
            -meaningful,
            FrontierYawTurnDirectionV1::CounterClockwise,
        )
        .unwrap();
        let cw_short =
            directed_yaw_delta(0.0, -meaningful, FrontierYawTurnDirectionV1::Clockwise).unwrap();
        assert_eq!(ccw_long, std::f64::consts::TAU - meaningful);
        assert_eq!(cw_short, -meaningful);
    }

    #[test]
    fn yaw_budget_and_deadline_equalities_fail_closed() {
        let budget =
            FrontierYawScanBudgetV1::try_new(1.0, std::f64::consts::FRAC_PI_2, 0.0, 5_000_000_000)
                .unwrap();
        let (map, scan) = in_place_scan();
        let command = FrontierYawScanCommandV1::try_new(
            NonZeroU64::new(3).unwrap(),
            1,
            scan,
            FrontierUnknownDirection::PositiveMapX,
            FrontierYawTurnDirectionV1::Clockwise,
            host(1_000),
            host(10_000_000_000),
            budget,
        )
        .unwrap();
        assert!(matches!(
            FrontierYawReferenceBuilderV1.build(
                command,
                DeviceSessionId::try_new(1).unwrap(),
                OdomSegmentId::try_new(1).unwrap(),
                map,
                PlanarTransform::<MapFrame, OdomFrame>::try_new(0.0, 0.0, 0.0).unwrap(),
                OdomPoseV1::try_new(0.5, 0.5, std::f64::consts::FRAC_PI_2).unwrap(),
                config(),
                command.valid_through_exclusive(),
            ),
            Err(FrontierYawReferenceBuildError::ScanExpired { .. })
        ));

        assert!(matches!(
            FrontierYawReferenceBuilderV1.build(
                command,
                DeviceSessionId::try_new(1).unwrap(),
                OdomSegmentId::try_new(1).unwrap(),
                map,
                PlanarTransform::<MapFrame, OdomFrame>::try_new(0.0, 0.0, 0.0).unwrap(),
                OdomPoseV1::try_new(0.5, 0.5, std::f64::consts::FRAC_PI_2).unwrap(),
                config(),
                host(1_001),
            ),
            Err(FrontierYawReferenceBuildError::YawBudgetExceeded {
                required_rad,
                maximum_exclusive_rad,
            }) if required_rad == std::f64::consts::FRAC_PI_2
                && maximum_exclusive_rad == std::f64::consts::FRAC_PI_2
        ));

        let (map, scan) = in_place_scan();
        let required_ceiling_ns = (std::f64::consts::FRAC_PI_2 * 1.0e9).ceil() as u64;
        let duration_budget =
            FrontierYawScanBudgetV1::try_new(1.0, std::f64::consts::PI, 0.0, required_ceiling_ns)
                .unwrap();
        let command = FrontierYawScanCommandV1::try_new(
            NonZeroU64::new(3).unwrap(),
            2,
            scan,
            FrontierUnknownDirection::PositiveMapX,
            FrontierYawTurnDirectionV1::Clockwise,
            host(1_000),
            host(10_000_000_000),
            duration_budget,
        )
        .unwrap();
        assert!(matches!(
            FrontierYawReferenceBuilderV1.build(
                command,
                DeviceSessionId::try_new(1).unwrap(),
                OdomSegmentId::try_new(1).unwrap(),
                map,
                PlanarTransform::<MapFrame, OdomFrame>::try_new(0.0, 0.0, 0.0).unwrap(),
                OdomPoseV1::try_new(0.5, 0.5, std::f64::consts::FRAC_PI_2).unwrap(),
                config(),
                host(1_000),
            ),
            Err(FrontierYawReferenceBuildError::DurationBudgetExceeded {
                required_ceiling_ns: required,
                remaining_exclusive_ns: remaining,
            }) if required == required_ceiling_ns && remaining == required_ceiling_ns
        ));
    }

    #[test]
    fn frontier_yaw_reference_is_stationary_map_bound_and_directional() {
        let budget =
            FrontierYawScanBudgetV1::try_new(1.0, std::f64::consts::PI, 1.0e-12, 5_000_000_000)
                .unwrap();
        let (map, scan) = in_place_scan();
        let command = FrontierYawScanCommandV1::try_new(
            NonZeroU64::new(3).unwrap(),
            8,
            scan,
            FrontierUnknownDirection::PositiveMapX,
            FrontierYawTurnDirectionV1::Clockwise,
            host(1_000),
            host(10_000_000_000),
            budget,
        )
        .unwrap();
        let map_to_odom = PlanarTransform::<MapFrame, OdomFrame>::try_new(0.0, 0.0, 0.3).unwrap();
        let initial_position = map_to_odom.transform_point(scan.robot_point()).unwrap();
        let (epoch, reference) = FrontierYawReferenceBuilderV1
            .build(
                command,
                DeviceSessionId::try_new(11).unwrap(),
                OdomSegmentId::try_new(5).unwrap(),
                map,
                map_to_odom,
                OdomPoseV1::try_new(
                    initial_position.x_m(),
                    initial_position.y_m(),
                    0.3 + std::f64::consts::FRAC_PI_2,
                )
                .unwrap(),
                config(),
                host(1_001),
            )
            .unwrap();

        let super::super::mpc::NavigationReferenceIdentityV1::FrontierInPlaceYaw(identity) =
            epoch.reference_identity()
        else {
            panic!("frontier scan must retain frontier reference identity")
        };
        assert_eq!(identity.authority_lease_id().get(), 3);
        assert_eq!(identity.scan_sequence(), 8);
        assert_eq!(identity.occupancy_map_instance_id(), scan.map_instance_id());
        assert_eq!(identity.occupancy_map_revision(), scan.map_revision());
        assert_eq!(
            (identity.frontier_column(), identity.frontier_row()),
            (1, 1)
        );
        assert_eq!(identity.scan_origin_map_x_m(), scan.robot_point().x_m());
        assert_eq!(identity.scan_origin_map_y_m(), scan.robot_point().y_m());
        assert_eq!(identity.maximum_scan_origin_displacement_m(), 1.0e-12);
        assert_eq!(identity.target_map_yaw_rad(), 0.0);
        assert_eq!(identity.signed_yaw_rate_rad_s(), -1.0);
        assert_eq!(
            reference.builder_revision(),
            ReferenceBuilderRevisionV1::BoundedFrontierYawV1
        );
        assert_matches_strict_dto_parse(&reference);
        for (index, point) in reference.points().iter().enumerate() {
            assert_eq!(point.pose().position(), initial_position);
            assert_eq!(point.forward_velocity_mps(), 0.0);
            assert_eq!(point.yaw_rate_rad_s(), -1.0);
            let expected =
                normalize_angle(0.3 + std::f64::consts::FRAC_PI_2 - (index + 1) as f64 * 0.1);
            assert!((normalize_angle(point.pose().yaw_rad() - expected)).abs() < 1.0e-14);
        }
    }

    #[test]
    fn frontier_yaw_rejects_motion_away_from_the_evidenced_scan_origin() {
        let budget =
            FrontierYawScanBudgetV1::try_new(1.0, std::f64::consts::PI, 0.1, 5_000_000_000)
                .unwrap();
        let (map, scan) = in_place_scan();
        let command = FrontierYawScanCommandV1::try_new(
            NonZeroU64::new(3).unwrap(),
            9,
            scan,
            FrontierUnknownDirection::PositiveMapX,
            FrontierYawTurnDirectionV1::Clockwise,
            host(1_000),
            host(10_000_000_000),
            budget,
        )
        .unwrap();
        let moved_pose = OdomPoseV1::try_new(
            scan.robot_point().x_m() + 0.101,
            scan.robot_point().y_m(),
            std::f64::consts::FRAC_PI_2,
        )
        .unwrap();

        assert!(matches!(
            FrontierYawReferenceBuilderV1.build(
                command,
                DeviceSessionId::try_new(11).unwrap(),
                OdomSegmentId::try_new(5).unwrap(),
                map,
                PlanarTransform::<MapFrame, OdomFrame>::try_new(0.0, 0.0, 0.0).unwrap(),
                moved_pose,
                config(),
                host(1_001),
            ),
            Err(FrontierYawReferenceBuildError::ScanOriginMoved {
                displacement_m,
                maximum_m,
                ..
            }) if displacement_m > maximum_m && maximum_m == 0.1
        ));
    }
}
