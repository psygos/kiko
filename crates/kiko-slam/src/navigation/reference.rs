//! Bounded construction of odom-frame MPC references from global map paths.
//!
//! This module performs geometry and time parameterization only. It does not
//! establish obstacle clearance, plant feasibility, or actuator authority.
//! Those contracts remain owned by the global planner, collision snapshot,
//! parsed plant model, and shadow MPC request respectively.

use std::fmt;

use crate::HostMonotonicTimestamp;

use super::frames::{MapToOdom, OdomFrame, PlanarPoint, PlanarTransformError};
use super::global_planner::{GlobalPath, GlobalPlanIdentity, MapPoint};
use super::mpc::{
    MAX_SUPPORTED_ABS_ODOM_COORDINATE_M, MIN_STEP_PERIOD_S, MPC_REFERENCE_V1, MpcConfigV1,
    MpcReferenceParseError, MpcReferenceV1, MpcReferenceV1Dto, NavigationEpochV1, OdomAxisV1,
    OdomPoseV1, OdomReferencePointV1Dto, ReferenceBuilderRevisionV1,
};

pub const PATH_REFERENCE_CONFIG_V1: u32 = 1;
pub const FORWARD_MOST_NEAREST_SEGMENT_V1: u32 = 1;
pub const MAX_PATH_REFERENCE_POINTS: u32 = 1_048_576;
pub const MAX_SUPPORTED_PATH_LENGTH_M: f64 = 2.0
    * std::f64::consts::SQRT_2
    * MAX_SUPPORTED_ABS_ODOM_COORDINATE_M
    * MAX_PATH_REFERENCE_POINTS as f64;
pub const MAX_SUPPORTED_PROJECTION_DISTANCE_M: f64 =
    2.0 * std::f64::consts::SQRT_2 * MAX_SUPPORTED_ABS_ODOM_COORDINATE_M;
pub const MAX_SUPPORTED_ABS_REFERENCE_YAW_RATE_RAD_S: f64 =
    std::f64::consts::PI / MIN_STEP_PERIOD_S;

/// Weak configuration boundary. Every scalar carries its SI unit in its name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathReferenceConfigV1Dto {
    pub schema_version: u32,
    pub maximum_path_points: u32,
    pub minimum_segment_length_m: f64,
    pub maximum_path_length_m: f64,
    pub maximum_projection_distance_m: f64,
    pub target_forward_speed_mps: f64,
    pub goal_stop_distance_m: f64,
    pub maximum_abs_yaw_rate_rad_s: f64,
    pub nearest_segment_tie_policy: u32,
}

/// Exact tie rule for equidistant projections onto an ordered polyline.
///
/// Forward-most means greatest accumulated path distance. If two projections
/// have the same distance and progress (for example at a shared vertex), the
/// later segment wins so its outgoing tangent defines the reference yaw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NearestSegmentTiePolicyV1 {
    ForwardMostThenLatestSegment,
}

/// Parsed, bounded policy for one V1 path-reference builder.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathReferenceConfigV1 {
    maximum_path_points: usize,
    minimum_segment_length_m: f64,
    maximum_path_length_m: f64,
    maximum_projection_distance_m: f64,
    target_forward_speed_mps: f64,
    goal_stop_distance_m: f64,
    maximum_abs_yaw_rate_rad_s: f64,
    nearest_segment_tie_policy: NearestSegmentTiePolicyV1,
}

impl PathReferenceConfigV1 {
    pub fn parse(dto: PathReferenceConfigV1Dto) -> Result<Self, PathReferenceConfigParseError> {
        if dto.schema_version != PATH_REFERENCE_CONFIG_V1 {
            return Err(PathReferenceConfigParseError::UnsupportedSchemaVersion(
                dto.schema_version,
            ));
        }
        if !(1..=MAX_PATH_REFERENCE_POINTS).contains(&dto.maximum_path_points) {
            return Err(
                PathReferenceConfigParseError::MaximumPathPointsOutOfBounds {
                    actual: dto.maximum_path_points,
                    maximum: MAX_PATH_REFERENCE_POINTS,
                },
            );
        }
        for (field, value) in [
            ("minimum_segment_length_m", dto.minimum_segment_length_m),
            ("maximum_path_length_m", dto.maximum_path_length_m),
            (
                "maximum_projection_distance_m",
                dto.maximum_projection_distance_m,
            ),
            ("target_forward_speed_mps", dto.target_forward_speed_mps),
            ("goal_stop_distance_m", dto.goal_stop_distance_m),
            ("maximum_abs_yaw_rate_rad_s", dto.maximum_abs_yaw_rate_rad_s),
        ] {
            require_positive_normal_config(field, value)?;
        }
        for (field, value, maximum) in [
            (
                "maximum_path_length_m",
                dto.maximum_path_length_m,
                MAX_SUPPORTED_PATH_LENGTH_M,
            ),
            (
                "maximum_projection_distance_m",
                dto.maximum_projection_distance_m,
                MAX_SUPPORTED_PROJECTION_DISTANCE_M,
            ),
            (
                "target_forward_speed_mps",
                dto.target_forward_speed_mps,
                dto.maximum_path_length_m / MIN_STEP_PERIOD_S,
            ),
            (
                "maximum_abs_yaw_rate_rad_s",
                dto.maximum_abs_yaw_rate_rad_s,
                MAX_SUPPORTED_ABS_REFERENCE_YAW_RATE_RAD_S,
            ),
        ] {
            if value > maximum {
                return Err(PathReferenceConfigParseError::AboveMaximum {
                    field,
                    value,
                    maximum,
                });
            }
        }
        if dto.minimum_segment_length_m > dto.maximum_path_length_m {
            return Err(
                PathReferenceConfigParseError::MinimumSegmentExceedsMaximumPathLength {
                    minimum_segment_length_m: dto.minimum_segment_length_m,
                    maximum_path_length_m: dto.maximum_path_length_m,
                },
            );
        }
        if dto.goal_stop_distance_m > dto.maximum_path_length_m {
            return Err(
                PathReferenceConfigParseError::GoalStopDistanceExceedsMaximumPathLength {
                    goal_stop_distance_m: dto.goal_stop_distance_m,
                    maximum_path_length_m: dto.maximum_path_length_m,
                },
            );
        }
        let nearest_segment_tie_policy = match dto.nearest_segment_tie_policy {
            FORWARD_MOST_NEAREST_SEGMENT_V1 => {
                NearestSegmentTiePolicyV1::ForwardMostThenLatestSegment
            }
            actual => {
                return Err(
                    PathReferenceConfigParseError::UnsupportedNearestSegmentTiePolicy { actual },
                );
            }
        };
        Ok(Self {
            maximum_path_points: dto.maximum_path_points as usize,
            minimum_segment_length_m: dto.minimum_segment_length_m,
            maximum_path_length_m: dto.maximum_path_length_m,
            maximum_projection_distance_m: dto.maximum_projection_distance_m,
            target_forward_speed_mps: dto.target_forward_speed_mps,
            goal_stop_distance_m: dto.goal_stop_distance_m,
            maximum_abs_yaw_rate_rad_s: dto.maximum_abs_yaw_rate_rad_s,
            nearest_segment_tie_policy,
        })
    }

    pub fn maximum_path_points(self) -> usize {
        self.maximum_path_points
    }

    pub fn minimum_segment_length_m(self) -> f64 {
        self.minimum_segment_length_m
    }

    pub fn maximum_path_length_m(self) -> f64 {
        self.maximum_path_length_m
    }

    pub fn maximum_projection_distance_m(self) -> f64 {
        self.maximum_projection_distance_m
    }

    pub fn target_forward_speed_mps(self) -> f64 {
        self.target_forward_speed_mps
    }

    pub fn goal_stop_distance_m(self) -> f64 {
        self.goal_stop_distance_m
    }

    pub fn maximum_abs_yaw_rate_rad_s(self) -> f64 {
        self.maximum_abs_yaw_rate_rad_s
    }

    pub fn nearest_segment_tie_policy(self) -> NearestSegmentTiePolicyV1 {
        self.nearest_segment_tie_policy
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathReferenceConfigParseError {
    UnsupportedSchemaVersion(u32),
    MaximumPathPointsOutOfBounds {
        actual: u32,
        maximum: u32,
    },
    NonFinite {
        field: &'static str,
        value: f64,
    },
    NotPositive {
        field: &'static str,
        value: f64,
    },
    Subnormal {
        field: &'static str,
        value: f64,
    },
    AboveMaximum {
        field: &'static str,
        value: f64,
        maximum: f64,
    },
    MinimumSegmentExceedsMaximumPathLength {
        minimum_segment_length_m: f64,
        maximum_path_length_m: f64,
    },
    GoalStopDistanceExceedsMaximumPathLength {
        goal_stop_distance_m: f64,
        maximum_path_length_m: f64,
    },
    UnsupportedNearestSegmentTiePolicy {
        actual: u32,
    },
}

impl fmt::Display for PathReferenceConfigParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid V1 path-reference configuration: {self:?}"
        )
    }
}

impl std::error::Error for PathReferenceConfigParseError {}

fn require_positive_normal_config(
    field: &'static str,
    value: f64,
) -> Result<(), PathReferenceConfigParseError> {
    if !value.is_finite() {
        return Err(PathReferenceConfigParseError::NonFinite { field, value });
    }
    if value <= 0.0 {
        return Err(PathReferenceConfigParseError::NotPositive { field, value });
    }
    if !value.is_normal() {
        return Err(PathReferenceConfigParseError::Subnormal { field, value });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OdomPathSegmentV1 {
    start: PlanarPoint<OdomFrame>,
    end: PlanarPoint<OdomFrame>,
    delta_x_m: f64,
    delta_y_m: f64,
    length_m: f64,
    start_distance_m: f64,
    end_distance_m: f64,
    tangent_yaw_rad: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct NearestProjectionV1 {
    distance_to_path_m: f64,
    distance_along_path_m: f64,
    segment_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PathSampleV1 {
    position: PlanarPoint<OdomFrame>,
    tangent_yaw_rad: f64,
}

/// Reusable bounded scratch owner for path transformation and segmentation.
///
/// The final DTO vector cannot be reused: [`MpcReferenceV1::parse`] consumes it
/// and constructs the authoritative domain vector retained by the returned
/// reference. Transformed-point and segment allocations are retained across
/// builds, but this API therefore makes no allocation-free claim.
#[derive(Debug)]
pub struct PathReferenceBuilderV1 {
    config: PathReferenceConfigV1,
    transformed_points: Vec<PlanarPoint<OdomFrame>>,
    segments: Vec<OdomPathSegmentV1>,
    total_path_length_m: f64,
}

impl PathReferenceBuilderV1 {
    pub fn new(config: PathReferenceConfigV1) -> Self {
        Self {
            config,
            transformed_points: Vec::new(),
            segments: Vec::new(),
            total_path_length_m: 0.0,
        }
    }

    pub fn config(&self) -> PathReferenceConfigV1 {
        self.config
    }

    /// Build one end-of-step reference for the exact supplied plan identity.
    ///
    /// `NavigationEpochV1` proves the path identity. `MapToOdom` and
    /// `OdomPoseV1` currently carry frames and numeric bounds but no embedded
    /// epoch token, so their freshness within that epoch remains an explicit
    /// coordinator responsibility rather than a claim made here.
    pub fn build<'path>(
        &mut self,
        epoch: NavigationEpochV1,
        global_path: &'path GlobalPath,
        map_to_odom: MapToOdom,
        current_pose: OdomPoseV1,
        mpc_config: MpcConfigV1,
        created_at: HostMonotonicTimestamp,
    ) -> Result<MpcReferenceV1<'path>, PathReferenceBuildError> {
        let expected = epoch.global_plan_identity();
        let actual = global_path.identity();
        if expected != actual {
            return Err(PathReferenceBuildError::EpochPathMismatch(Box::new(
                EpochPathMismatchV1 { expected, actual },
            )));
        }
        self.prepare_path(global_path.points(), map_to_odom)?;
        let points = self.build_reference_points(current_pose, mpc_config)?;
        MpcReferenceV1::parse(
            MpcReferenceV1Dto {
                schema_version: MPC_REFERENCE_V1,
                builder_revision: ReferenceBuilderRevisionV1::TimeParameterizedGlobalPathV1 as u32,
                created_at_host_ns: created_at.as_nanos(),
                step_period_s: mpc_config.step_period_s(),
                points,
            },
            mpc_config,
            epoch,
            global_path,
        )
        .map_err(|source| PathReferenceBuildError::Reference(Box::new(source)))
    }

    fn prepare_path(
        &mut self,
        map_points: &[MapPoint],
        map_to_odom: MapToOdom,
    ) -> Result<(), PathReferenceBuildError> {
        self.transformed_points.clear();
        self.segments.clear();
        self.total_path_length_m = 0.0;

        if map_points.is_empty() {
            return Err(PathReferenceBuildError::EmptyPath);
        }
        if map_points.len() > self.config.maximum_path_points {
            return Err(PathReferenceBuildError::PathPointLimitExceeded {
                actual: map_points.len(),
                maximum: self.config.maximum_path_points,
            });
        }
        reserve_additional(
            &mut self.transformed_points,
            map_points.len(),
            "transformed path points",
        )?;
        reserve_additional(
            &mut self.segments,
            map_points.len().saturating_sub(1),
            "path segments",
        )?;

        for (index, map_point) in map_points.iter().copied().enumerate() {
            let point = map_to_odom
                .transform_point(map_point)
                .map_err(|source| PathReferenceBuildError::TransformPoint { index, source })?;
            require_supported_odom_point(index, point)?;

            let Some(previous) = self.transformed_points.last().copied() else {
                self.transformed_points.push(point);
                continue;
            };
            let delta_x_m = point.x_m() - previous.x_m();
            let delta_y_m = point.y_m() - previous.y_m();
            let length_m = delta_x_m.hypot(delta_y_m);
            if !length_m.is_finite() {
                return Err(PathReferenceBuildError::NonFiniteSegmentLength {
                    end_point_index: index,
                    value_m: length_m,
                });
            }
            if length_m == 0.0 {
                continue;
            }
            if length_m < self.config.minimum_segment_length_m {
                return Err(PathReferenceBuildError::SegmentBelowMinimumLength {
                    end_point_index: index,
                    length_m,
                    minimum_m: self.config.minimum_segment_length_m,
                });
            }
            let end_distance_m = self.total_path_length_m + length_m;
            if !end_distance_m.is_finite() {
                return Err(PathReferenceBuildError::PathLengthOverflow {
                    end_point_index: index,
                });
            }
            if end_distance_m <= self.total_path_length_m {
                return Err(PathReferenceBuildError::PathLengthResolutionLoss {
                    end_point_index: index,
                    previous_length_m: self.total_path_length_m,
                    segment_length_m: length_m,
                });
            }
            if end_distance_m > self.config.maximum_path_length_m {
                return Err(PathReferenceBuildError::MaximumPathLengthExceeded {
                    actual_m: end_distance_m,
                    maximum_m: self.config.maximum_path_length_m,
                });
            }
            let tangent_yaw_rad = normalize_angle(delta_y_m.atan2(delta_x_m));
            self.segments.push(OdomPathSegmentV1 {
                start: previous,
                end: point,
                delta_x_m,
                delta_y_m,
                length_m,
                start_distance_m: self.total_path_length_m,
                end_distance_m,
                tangent_yaw_rad,
            });
            self.transformed_points.push(point);
            self.total_path_length_m = end_distance_m;
        }

        if map_points.len() > 1 && self.segments.is_empty() {
            return Err(PathReferenceBuildError::DegenerateRepeatedPath {
                point_count: map_points.len(),
            });
        }
        Ok(())
    }

    fn build_reference_points(
        &self,
        current_pose: OdomPoseV1,
        mpc_config: MpcConfigV1,
    ) -> Result<Vec<OdomReferencePointV1Dto>, PathReferenceBuildError> {
        let horizon = mpc_config.horizon_steps();
        let mut points = Vec::new();
        points.try_reserve_exact(horizon).map_err(|_| {
            PathReferenceBuildError::AllocationFailed {
                buffer: "MPC reference DTO points",
                requested: horizon,
            }
        })?;

        if self.segments.is_empty() {
            let goal = *self
                .transformed_points
                .first()
                .ok_or(PathReferenceBuildError::EmptyPreparedPath)?;
            let goal_distance_m = (current_pose.position().x_m() - goal.x_m())
                .hypot(current_pose.position().y_m() - goal.y_m());
            self.require_projection_distance(goal_distance_m, None)?;
            for _ in 0..horizon {
                points.push(OdomReferencePointV1Dto {
                    x_m: goal.x_m(),
                    y_m: goal.y_m(),
                    yaw_rad: current_pose.yaw_rad(),
                    forward_velocity_mps: 0.0,
                    yaw_rate_rad_s: 0.0,
                });
            }
            return Ok(points);
        }

        let projection = self.nearest_projection(current_pose.position())?;
        let step_distance_m = self.config.target_forward_speed_mps * mpc_config.step_period_s();
        if !step_distance_m.is_finite() || step_distance_m <= 0.0 || !step_distance_m.is_normal() {
            return Err(PathReferenceBuildError::UnrepresentableSampleStep {
                target_forward_speed_mps: self.config.target_forward_speed_mps,
                step_period_s: mpc_config.step_period_s(),
                step_distance_m,
            });
        }

        let mut sample_distance_m = projection.distance_along_path_m;
        let mut sample_segment_index = projection.segment_index;
        let mut previous_yaw_rad = current_pose.yaw_rad();
        for index in 0..horizon {
            if sample_distance_m < self.total_path_length_m {
                let next_distance_m = sample_distance_m + step_distance_m;
                if !next_distance_m.is_finite() {
                    return Err(PathReferenceBuildError::SampleDistanceOverflow { index });
                }
                if next_distance_m <= sample_distance_m {
                    return Err(PathReferenceBuildError::SampleDistanceResolutionLoss {
                        index,
                        current_distance_m: sample_distance_m,
                        step_distance_m,
                    });
                }
                sample_distance_m = next_distance_m.min(self.total_path_length_m);
            }

            let sample = self.sample_at(sample_distance_m, &mut sample_segment_index)?;
            let remaining_m = (self.total_path_length_m - sample_distance_m).max(0.0);
            let forward_velocity_mps = if remaining_m == 0.0 {
                0.0
            } else if remaining_m < self.config.goal_stop_distance_m {
                let velocity = self.config.target_forward_speed_mps
                    * (remaining_m / self.config.goal_stop_distance_m);
                if velocity == 0.0 {
                    return Err(PathReferenceBuildError::VelocityTaperUnderflow {
                        index,
                        remaining_m,
                    });
                }
                velocity
            } else {
                self.config.target_forward_speed_mps
            };
            let raw_yaw_rate_rad_s = signed_angle_delta(previous_yaw_rad, sample.tangent_yaw_rad)
                / mpc_config.step_period_s();
            let yaw_rate_rad_s = raw_yaw_rate_rad_s.clamp(
                -self.config.maximum_abs_yaw_rate_rad_s,
                self.config.maximum_abs_yaw_rate_rad_s,
            );
            points.push(OdomReferencePointV1Dto {
                x_m: sample.position.x_m(),
                y_m: sample.position.y_m(),
                yaw_rad: sample.tangent_yaw_rad,
                forward_velocity_mps,
                yaw_rate_rad_s: canonical_zero(yaw_rate_rad_s),
            });
            previous_yaw_rad = sample.tangent_yaw_rad;
        }
        Ok(points)
    }

    fn nearest_projection(
        &self,
        position: PlanarPoint<OdomFrame>,
    ) -> Result<NearestProjectionV1, PathReferenceBuildError> {
        let mut best: Option<NearestProjectionV1> = None;
        for (segment_index, segment) in self.segments.iter().copied().enumerate() {
            let unit_x = segment.delta_x_m / segment.length_m;
            let unit_y = segment.delta_y_m / segment.length_m;
            let relative_x = position.x_m() - segment.start.x_m();
            let relative_y = position.y_m() - segment.start.y_m();
            let along_m = relative_x
                .mul_add(unit_x, relative_y * unit_y)
                .clamp(0.0, segment.length_m);
            let projected_x = unit_x.mul_add(along_m, segment.start.x_m());
            let projected_y = unit_y.mul_add(along_m, segment.start.y_m());
            let distance_to_path_m =
                (position.x_m() - projected_x).hypot(position.y_m() - projected_y);
            let distance_along_path_m = (segment.start_distance_m + along_m)
                .min(segment.end_distance_m)
                .max(segment.start_distance_m);
            if !distance_to_path_m.is_finite() || !distance_along_path_m.is_finite() {
                return Err(PathReferenceBuildError::NonFiniteProjection {
                    segment_index: Some(segment_index),
                });
            }
            let candidate = NearestProjectionV1 {
                distance_to_path_m,
                distance_along_path_m,
                segment_index,
            };
            let replace = match best {
                None => true,
                Some(current) => match candidate
                    .distance_to_path_m
                    .total_cmp(&current.distance_to_path_m)
                {
                    std::cmp::Ordering::Less => true,
                    std::cmp::Ordering::Greater => false,
                    std::cmp::Ordering::Equal => match self.config.nearest_segment_tie_policy {
                        NearestSegmentTiePolicyV1::ForwardMostThenLatestSegment => {
                            candidate.distance_along_path_m > current.distance_along_path_m
                                || (candidate.distance_along_path_m
                                    == current.distance_along_path_m
                                    && candidate.segment_index > current.segment_index)
                        }
                    },
                },
            };
            if replace {
                best = Some(candidate);
            }
        }
        let best = best.ok_or(PathReferenceBuildError::EmptyPreparedPath)?;
        self.require_projection_distance(best.distance_to_path_m, Some(best.segment_index))?;
        Ok(best)
    }

    fn require_projection_distance(
        &self,
        distance_m: f64,
        segment_index: Option<usize>,
    ) -> Result<(), PathReferenceBuildError> {
        if !distance_m.is_finite() {
            return Err(PathReferenceBuildError::NonFiniteProjection { segment_index });
        }
        if distance_m > self.config.maximum_projection_distance_m {
            return Err(PathReferenceBuildError::ProjectionDistanceExceeded {
                actual_m: distance_m,
                maximum_m: self.config.maximum_projection_distance_m,
            });
        }
        Ok(())
    }

    fn sample_at(
        &self,
        distance_m: f64,
        segment_index: &mut usize,
    ) -> Result<PathSampleV1, PathReferenceBuildError> {
        let last = self
            .segments
            .last()
            .copied()
            .ok_or(PathReferenceBuildError::EmptyPreparedPath)?;
        if distance_m >= self.total_path_length_m {
            *segment_index = self.segments.len() - 1;
            return Ok(PathSampleV1 {
                position: last.end,
                tangent_yaw_rad: last.tangent_yaw_rad,
            });
        }
        while (*segment_index).saturating_add(1) < self.segments.len()
            && distance_m >= self.segments[*segment_index].end_distance_m
        {
            *segment_index += 1;
        }
        let segment = self
            .segments
            .get(*segment_index)
            .copied()
            .ok_or(PathReferenceBuildError::SamplingInvariant)?;
        if distance_m <= segment.start_distance_m {
            return Ok(PathSampleV1 {
                position: segment.start,
                tangent_yaw_rad: segment.tangent_yaw_rad,
            });
        }
        let along_m = (distance_m - segment.start_distance_m).clamp(0.0, segment.length_m);
        let fraction = (along_m / segment.length_m).clamp(0.0, 1.0);
        let x_m = segment.delta_x_m.mul_add(fraction, segment.start.x_m());
        let y_m = segment.delta_y_m.mul_add(fraction, segment.start.y_m());
        let position = PlanarPoint::try_new(x_m, y_m)
            .map_err(|_| PathReferenceBuildError::SamplingInvariant)?;
        Ok(PathSampleV1 {
            position,
            tangent_yaw_rad: segment.tangent_yaw_rad,
        })
    }
}

fn reserve_additional<T>(
    values: &mut Vec<T>,
    requested_total: usize,
    buffer: &'static str,
) -> Result<(), PathReferenceBuildError> {
    let additional = requested_total.saturating_sub(values.len());
    values
        .try_reserve_exact(additional)
        .map_err(|_| PathReferenceBuildError::AllocationFailed {
            buffer,
            requested: requested_total,
        })
}

fn require_supported_odom_point(
    index: usize,
    point: PlanarPoint<OdomFrame>,
) -> Result<(), PathReferenceBuildError> {
    for (axis, value_m) in [(OdomAxisV1::X, point.x_m()), (OdomAxisV1::Y, point.y_m())] {
        if value_m.abs() > MAX_SUPPORTED_ABS_ODOM_COORDINATE_M {
            return Err(
                PathReferenceBuildError::TransformedPointOutsideSupportedDomain {
                    index,
                    axis,
                    value_m,
                    maximum_abs_m: MAX_SUPPORTED_ABS_ODOM_COORDINATE_M,
                },
            );
        }
    }
    Ok(())
}

fn normalize_angle(angle_rad: f64) -> f64 {
    let positive = angle_rad.rem_euclid(std::f64::consts::TAU);
    let normalized = if positive >= std::f64::consts::PI {
        positive - std::f64::consts::TAU
    } else {
        positive
    };
    canonical_zero(normalized)
}

fn signed_angle_delta(from_rad: f64, to_rad: f64) -> f64 {
    normalize_angle(to_rad - from_rad)
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

#[derive(Debug, PartialEq)]
pub struct EpochPathMismatchV1 {
    expected: GlobalPlanIdentity,
    actual: GlobalPlanIdentity,
}

impl EpochPathMismatchV1 {
    pub fn expected(&self) -> GlobalPlanIdentity {
        self.expected
    }

    pub fn actual(&self) -> GlobalPlanIdentity {
        self.actual
    }
}

#[derive(Debug, PartialEq)]
pub enum PathReferenceBuildError {
    EpochPathMismatch(Box<EpochPathMismatchV1>),
    EmptyPath,
    PathPointLimitExceeded {
        actual: usize,
        maximum: usize,
    },
    AllocationFailed {
        buffer: &'static str,
        requested: usize,
    },
    TransformPoint {
        index: usize,
        source: PlanarTransformError,
    },
    TransformedPointOutsideSupportedDomain {
        index: usize,
        axis: OdomAxisV1,
        value_m: f64,
        maximum_abs_m: f64,
    },
    NonFiniteSegmentLength {
        end_point_index: usize,
        value_m: f64,
    },
    SegmentBelowMinimumLength {
        end_point_index: usize,
        length_m: f64,
        minimum_m: f64,
    },
    PathLengthOverflow {
        end_point_index: usize,
    },
    PathLengthResolutionLoss {
        end_point_index: usize,
        previous_length_m: f64,
        segment_length_m: f64,
    },
    MaximumPathLengthExceeded {
        actual_m: f64,
        maximum_m: f64,
    },
    DegenerateRepeatedPath {
        point_count: usize,
    },
    EmptyPreparedPath,
    NonFiniteProjection {
        segment_index: Option<usize>,
    },
    ProjectionDistanceExceeded {
        actual_m: f64,
        maximum_m: f64,
    },
    UnrepresentableSampleStep {
        target_forward_speed_mps: f64,
        step_period_s: f64,
        step_distance_m: f64,
    },
    SampleDistanceOverflow {
        index: usize,
    },
    SampleDistanceResolutionLoss {
        index: usize,
        current_distance_m: f64,
        step_distance_m: f64,
    },
    VelocityTaperUnderflow {
        index: usize,
        remaining_m: f64,
    },
    SamplingInvariant,
    Reference(Box<MpcReferenceParseError>),
}

impl fmt::Display for PathReferenceBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "failed to build V1 MPC path reference: {self:?}")
    }
}

impl std::error::Error for PathReferenceBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TransformPoint { source, .. } => Some(source),
            Self::Reference(source) => Some(source.as_ref()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::occupancy::{OccupancyCell, OccupancyGridGeometry, OccupancyGridSnapshot};
    use crate::map::{MapSnapshot, SlamMap};
    use crate::{DeviceSessionId, HostMonotonicTimestamp};

    use super::super::global_planner::{
        GlobalPlanner, GlobalPlannerConfig, PlanStart, PointGoal, UnknownSpacePolicy,
    };
    use super::super::mpc::{MPC_CONFIG_V1, MpcConfigV1Dto};
    use super::super::odometry::OdomSegmentId;

    fn config_dto() -> PathReferenceConfigV1Dto {
        PathReferenceConfigV1Dto {
            schema_version: PATH_REFERENCE_CONFIG_V1,
            maximum_path_points: 256,
            minimum_segment_length_m: 1.0e-9,
            maximum_path_length_m: 1_000.0,
            maximum_projection_distance_m: 2.0,
            target_forward_speed_mps: 1.0,
            goal_stop_distance_m: 1.0,
            maximum_abs_yaw_rate_rad_s: 2.0,
            nearest_segment_tie_policy: FORWARD_MOST_NEAREST_SEGMENT_V1,
        }
    }

    fn config() -> PathReferenceConfigV1 {
        PathReferenceConfigV1::parse(config_dto()).expect("reference config")
    }

    fn mpc_config(dt_s: f64, horizon_steps: u16) -> MpcConfigV1 {
        MpcConfigV1::parse(MpcConfigV1Dto {
            schema_version: MPC_CONFIG_V1,
            horizon_steps,
            step_period_s: dt_s,
            integration_substeps: 1,
            optimization_iterations: 1,
            candidates_per_wheel: 3,
            max_rollout_evaluations: 100_000,
            initial_search_radius_percent: 10,
            search_radius_decay_numerator: 1,
            search_radius_decay_denominator: 2,
            left_pwm_min_percent: -50,
            left_pwm_max_percent: 50,
            right_pwm_min_percent: -50,
            right_pwm_max_percent: 50,
            left_max_slew_percent_per_step: 100,
            right_max_slew_percent_per_step: 100,
            max_integration_tube_radius_m: 0.5,
            position_cost_per_m2: 1.0,
            heading_cost_per_rad2: 1.0,
            forward_velocity_cost_s2_per_m2: 1.0,
            yaw_rate_cost_s2_per_rad2: 1.0,
            pwm_cost_per_percent2: 0.001,
            slew_cost_per_percent2: 0.001,
            terminal_state_cost_multiplier: 2.0,
        })
        .expect("MPC config")
    }

    fn map_point(x_m: f64, y_m: f64) -> MapPoint {
        MapPoint::try_new(x_m, y_m).expect("finite map point")
    }

    fn odom_pose(x_m: f64, y_m: f64, yaw_rad: f64) -> OdomPoseV1 {
        OdomPoseV1::try_new(x_m, y_m, yaw_rad).expect("odom pose")
    }

    fn identity_map_to_odom() -> MapToOdom {
        MapToOdom::try_new(0.0, 0.0, 0.0).expect("identity map-to-odom")
    }

    fn planner_fixture() -> (MapSnapshot, OccupancyGridSnapshot, GlobalPlanner) {
        let map = SlamMap::new();
        let map_snapshot = map.snapshot();
        let geometry =
            OccupancyGridGeometry::try_new(1.0, [-10.0, -10.0], 20, 20, 400).expect("geometry");
        let cells = vec![OccupancyCell::Free; geometry.cell_count()];
        let occupancy =
            OccupancyGridSnapshot::from_test_cells(geometry, &cells, map_snapshot.instance_id(), 1);
        let planner = GlobalPlanner::try_new(
            &occupancy,
            GlobalPlannerConfig::try_new(0.0, UnknownSpacePolicy::Traversable)
                .expect("planner config"),
        )
        .expect("planner");
        (map_snapshot, occupancy, planner)
    }

    fn plan(
        planner: &mut GlobalPlanner,
        occupancy: &OccupancyGridSnapshot,
        start: MapPoint,
        goal: MapPoint,
    ) -> GlobalPath {
        planner
            .plan(
                PlanStart::for_snapshot(start, occupancy).expect("start"),
                PointGoal::for_snapshot(goal, occupancy).expect("goal"),
            )
            .expect("path")
    }

    fn epoch(map_snapshot: MapSnapshot, path: &GlobalPath) -> NavigationEpochV1 {
        NavigationEpochV1::from_runtime(
            DeviceSessionId::try_new(1).expect("session"),
            OdomSegmentId::try_new(1).expect("odom segment"),
            map_snapshot,
            path,
        )
        .expect("navigation epoch")
    }

    fn prepare(builder: &mut PathReferenceBuilderV1, points: &[MapPoint]) {
        builder
            .prepare_path(points, identity_map_to_odom())
            .expect("prepared path");
    }

    #[test]
    fn config_parses_once_and_rejects_weak_invalid_values() {
        let parsed = config();
        assert_eq!(parsed.maximum_path_points(), 256);
        assert_eq!(parsed.minimum_segment_length_m(), 1.0e-9);
        assert_eq!(parsed.maximum_path_length_m(), 1_000.0);
        assert_eq!(parsed.maximum_projection_distance_m(), 2.0);
        assert_eq!(parsed.target_forward_speed_mps(), 1.0);
        assert_eq!(parsed.goal_stop_distance_m(), 1.0);
        assert_eq!(parsed.maximum_abs_yaw_rate_rad_s(), 2.0);
        assert_eq!(
            parsed.nearest_segment_tie_policy(),
            NearestSegmentTiePolicyV1::ForwardMostThenLatestSegment
        );

        let mut dto = config_dto();
        dto.target_forward_speed_mps = f64::NAN;
        assert!(matches!(
            PathReferenceConfigV1::parse(dto),
            Err(PathReferenceConfigParseError::NonFinite {
                field: "target_forward_speed_mps",
                ..
            })
        ));
        let mut dto = config_dto();
        dto.goal_stop_distance_m = 0.0;
        assert!(matches!(
            PathReferenceConfigV1::parse(dto),
            Err(PathReferenceConfigParseError::NotPositive {
                field: "goal_stop_distance_m",
                ..
            })
        ));
        let mut dto = config_dto();
        dto.minimum_segment_length_m = f64::from_bits(1);
        assert!(matches!(
            PathReferenceConfigV1::parse(dto),
            Err(PathReferenceConfigParseError::Subnormal {
                field: "minimum_segment_length_m",
                ..
            })
        ));
        let mut dto = config_dto();
        dto.nearest_segment_tie_policy = 99;
        assert_eq!(
            PathReferenceConfigV1::parse(dto),
            Err(PathReferenceConfigParseError::UnsupportedNearestSegmentTiePolicy { actual: 99 })
        );
    }

    #[test]
    fn straight_path_builds_exact_end_of_step_horizon() {
        let (map_snapshot, occupancy, mut planner) = planner_fixture();
        let path = plan(
            &mut planner,
            &occupancy,
            map_point(0.5, 0.5),
            map_point(5.5, 0.5),
        );
        let epoch = epoch(map_snapshot, &path);
        let mut builder = PathReferenceBuilderV1::new(config());
        assert_eq!(builder.config(), config());
        let reference = builder
            .build(
                epoch,
                &path,
                identity_map_to_odom(),
                odom_pose(0.5, 0.5, 0.0),
                mpc_config(0.1, 5),
                HostMonotonicTimestamp::from_nanos(42),
            )
            .expect("reference");
        assert_eq!(reference.points().len(), 5);
        assert_eq!(reference.created_at().as_nanos(), 42);
        assert_eq!(
            reference.builder_revision(),
            ReferenceBuilderRevisionV1::TimeParameterizedGlobalPathV1
        );
        assert!(std::ptr::eq(reference.source_path(), &path));
        for (index, point) in reference.points().iter().copied().enumerate() {
            let expected_x = 0.6 + index as f64 * 0.1;
            assert!((point.pose().position().x_m() - expected_x).abs() < 1.0e-12);
            assert_eq!(point.pose().position().y_m(), 0.5);
            assert_eq!(point.pose().yaw_rad(), 0.0);
            assert_eq!(point.forward_velocity_mps(), 1.0);
            assert_eq!(point.yaw_rate_rad_s(), 0.0);
        }
    }

    #[test]
    fn current_loop_closure_transform_is_applied_once_before_sampling() {
        let (map_snapshot, occupancy, mut planner) = planner_fixture();
        let path = plan(
            &mut planner,
            &occupancy,
            map_point(0.5, 0.5),
            map_point(2.5, 0.5),
        );
        let epoch = epoch(map_snapshot, &path);
        let map_to_odom =
            MapToOdom::try_new(10.0, -2.0, std::f64::consts::FRAC_PI_2).expect("map-to-odom");
        let mut builder = PathReferenceBuilderV1::new(config());
        let reference = builder
            .build(
                epoch,
                &path,
                map_to_odom,
                odom_pose(9.5, -1.5, std::f64::consts::FRAC_PI_2),
                mpc_config(0.1, 2),
                HostMonotonicTimestamp::from_nanos(10),
            )
            .expect("transformed reference");
        assert!((reference.points()[0].pose().position().x_m() - 9.5).abs() < 1.0e-12);
        assert!((reference.points()[0].pose().position().y_m() + 1.4).abs() < 1.0e-12);
        assert!(
            (reference.points()[0].pose().yaw_rad() - std::f64::consts::FRAC_PI_2).abs() < 1.0e-12
        );
    }

    #[test]
    fn corner_uses_outgoing_tangent_and_bounds_yaw_rate() {
        let mut builder = PathReferenceBuilderV1::new(config());
        prepare(
            &mut builder,
            &[
                map_point(0.0, 0.0),
                map_point(1.0, 0.0),
                map_point(1.0, 1.0),
            ],
        );
        let points = builder
            .build_reference_points(odom_pose(0.9, 0.0, 0.0), mpc_config(0.1, 2))
            .expect("corner points");
        assert!((points[0].x_m - 1.0).abs() < 1.0e-12);
        assert!((points[0].y_m - 0.0).abs() < 1.0e-12);
        assert!((points[0].yaw_rad - std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
        assert_eq!(points[0].yaw_rate_rad_s, 2.0);
        assert!(points[1].y_m > points[0].y_m);
    }

    #[test]
    fn single_point_path_is_an_exact_stationary_goal() {
        let (map_snapshot, occupancy, mut planner) = planner_fixture();
        let goal = map_point(1.5, 1.5);
        let path = plan(&mut planner, &occupancy, goal, goal);
        assert_eq!(path.points().len(), 1);
        let epoch = epoch(map_snapshot, &path);
        let mut builder = PathReferenceBuilderV1::new(config());
        let reference = builder
            .build(
                epoch,
                &path,
                identity_map_to_odom(),
                odom_pose(1.5, 1.5, 0.25),
                mpc_config(0.1, 4),
                HostMonotonicTimestamp::from_nanos(1),
            )
            .expect("stationary reference");
        assert_eq!(reference.points().len(), 4);
        for point in reference.points().iter().copied() {
            assert_eq!(point.pose().position().as_array(), [1.5, 1.5]);
            assert_eq!(point.pose().yaw_rad(), 0.25);
            assert_eq!(point.forward_velocity_mps(), 0.0);
            assert_eq!(point.yaw_rate_rad_s(), 0.0);
        }
    }

    #[test]
    fn duplicate_points_are_skipped_but_an_all_duplicate_path_is_rejected() {
        let mut builder = PathReferenceBuilderV1::new(config());
        prepare(
            &mut builder,
            &[
                map_point(0.0, 0.0),
                map_point(0.0, 0.0),
                map_point(1.0, 0.0),
                map_point(1.0, 0.0),
            ],
        );
        assert_eq!(builder.transformed_points.len(), 2);
        assert_eq!(builder.segments.len(), 1);

        assert_eq!(
            builder.prepare_path(
                &[map_point(2.0, 2.0), map_point(2.0, 2.0)],
                identity_map_to_odom(),
            ),
            Err(PathReferenceBuildError::DegenerateRepeatedPath { point_count: 2 })
        );
    }

    #[test]
    fn equidistant_overlap_selects_forward_most_then_latest_segment() {
        let mut builder = PathReferenceBuilderV1::new(config());
        prepare(
            &mut builder,
            &[
                map_point(-1.0, 0.0),
                map_point(1.0, 0.0),
                map_point(-1.0, 0.0),
            ],
        );
        let projection = builder
            .nearest_projection(odom_pose(0.0, 0.0, 0.0).position())
            .expect("projection");
        assert_eq!(projection.distance_to_path_m, 0.0);
        assert_eq!(projection.distance_along_path_m, 3.0);
        assert_eq!(projection.segment_index, 1);
    }

    #[test]
    fn projection_distance_accepts_exact_boundary_and_rejects_far_robot() {
        let mut builder = PathReferenceBuilderV1::new(config());
        prepare(&mut builder, &[map_point(0.0, 0.0), map_point(1.0, 0.0)]);
        let boundary = builder
            .nearest_projection(odom_pose(0.5, 2.0, 0.0).position())
            .expect("closed projection-distance boundary");
        assert_eq!(boundary.distance_to_path_m, 2.0);
        assert_eq!(boundary.distance_along_path_m, 0.5);
        assert_eq!(boundary.segment_index, 0);

        assert_eq!(
            builder.nearest_projection(odom_pose(0.5, 2.000_001, 0.0).position()),
            Err(PathReferenceBuildError::ProjectionDistanceExceeded {
                actual_m: 2.000_001,
                maximum_m: 2.0,
            })
        );

        prepare(&mut builder, &[map_point(0.0, 0.0)]);
        assert_eq!(
            builder.build_reference_points(odom_pose(3.0, 0.0, 0.0), mpc_config(0.1, 1)),
            Err(PathReferenceBuildError::ProjectionDistanceExceeded {
                actual_m: 3.0,
                maximum_m: 2.0,
            })
        );
    }

    #[test]
    fn yaw_wrap_uses_the_short_signed_delta() {
        let epsilon = 1.0e-6;
        let mut builder = PathReferenceBuilderV1::new(config());
        prepare(
            &mut builder,
            &[map_point(0.0, 0.0), map_point(-1.0, epsilon)],
        );
        let points = builder
            .build_reference_points(
                odom_pose(0.0, 0.0, -std::f64::consts::PI + epsilon),
                mpc_config(0.1, 1),
            )
            .expect("wrapped yaw");
        assert!(points[0].yaw_rad > 3.0);
        assert!(points[0].yaw_rate_rad_s < 0.0);
        assert!(points[0].yaw_rate_rad_s.abs() < 1.0e-3);
        assert!((signed_angle_delta(3.13, -3.13) - 0.023_185_307_179_586_445).abs() < 1.0e-12);
    }

    #[test]
    fn goal_taper_reaches_exact_zero_without_extrapolation() {
        let mut builder = PathReferenceBuilderV1::new(config());
        prepare(&mut builder, &[map_point(0.0, 0.0), map_point(0.15, 0.0)]);
        let points = builder
            .build_reference_points(odom_pose(0.0, 0.0, 0.0), mpc_config(0.1, 3))
            .expect("goal taper");
        assert!((points[0].x_m - 0.1).abs() < 1.0e-12);
        assert!((points[0].forward_velocity_mps - 0.05).abs() < 1.0e-12);
        assert_eq!(points[1].x_m, 0.15);
        assert_eq!(points[1].forward_velocity_mps, 0.0);
        assert_eq!(points[2].x_m, 0.15);
        assert_eq!(points[2].forward_velocity_mps, 0.0);
    }

    #[test]
    fn empty_subnormal_huge_and_overflowing_domains_fail_typed() {
        let mut builder = PathReferenceBuilderV1::new(config());
        assert_eq!(
            builder.prepare_path(&[], identity_map_to_odom()),
            Err(PathReferenceBuildError::EmptyPath)
        );
        assert!(matches!(
            builder.prepare_path(
                &[map_point(0.0, 0.0), map_point(f64::from_bits(1), 0.0)],
                identity_map_to_odom(),
            ),
            Err(PathReferenceBuildError::SegmentBelowMinimumLength { .. })
        ));
        assert!(matches!(
            builder.prepare_path(&[map_point(f64::MAX, 0.0)], identity_map_to_odom()),
            Err(PathReferenceBuildError::TransformedPointOutsideSupportedDomain { .. })
        ));
        assert_eq!(
            builder.prepare_path(
                &[map_point(0.0, 0.0), map_point(1_001.0, 0.0)],
                identity_map_to_odom(),
            ),
            Err(PathReferenceBuildError::MaximumPathLengthExceeded {
                actual_m: 1_001.0,
                maximum_m: 1_000.0,
            })
        );
        assert!(matches!(
            builder.prepare_path(
                &[map_point(f64::MAX, 0.0)],
                MapToOdom::try_new(f64::MAX, 0.0, 0.0).expect("finite transform"),
            ),
            Err(PathReferenceBuildError::TransformPoint { .. })
        ));

        let mut huge_speed = config_dto();
        huge_speed.target_forward_speed_mps = f64::MAX;
        assert!(matches!(
            PathReferenceConfigV1::parse(huge_speed),
            Err(PathReferenceConfigParseError::AboveMaximum {
                field: "target_forward_speed_mps",
                ..
            })
        ));

        let mut one_point = config_dto();
        one_point.maximum_path_points = 1;
        let mut one_point_builder = PathReferenceBuilderV1::new(
            PathReferenceConfigV1::parse(one_point).expect("one-point bound"),
        );
        assert_eq!(
            one_point_builder.prepare_path(
                &[map_point(0.0, 0.0), map_point(1.0, 0.0)],
                identity_map_to_odom(),
            ),
            Err(PathReferenceBuildError::PathPointLimitExceeded {
                actual: 2,
                maximum: 1,
            })
        );
    }

    #[test]
    fn exact_plan_identity_is_required_before_geometry_work() {
        let (map_snapshot, occupancy, mut planner) = planner_fixture();
        let start = map_point(0.5, 0.5);
        let goal = map_point(3.5, 0.5);
        let first = plan(&mut planner, &occupancy, start, goal);
        let second = plan(&mut planner, &occupancy, start, goal);
        assert_ne!(first.identity(), second.identity());
        let epoch = epoch(map_snapshot, &first);
        let mut builder = PathReferenceBuilderV1::new(config());
        let mismatch = builder
            .build(
                epoch,
                &second,
                identity_map_to_odom(),
                odom_pose(0.5, 0.5, 0.0),
                mpc_config(0.1, 2),
                HostMonotonicTimestamp::from_nanos(1),
            )
            .expect_err("identity mismatch");
        let PathReferenceBuildError::EpochPathMismatch(mismatch) = mismatch else {
            panic!("unexpected mismatch error")
        };
        assert_eq!(mismatch.expected(), first.identity());
        assert_eq!(mismatch.actual(), second.identity());
        assert!(builder.transformed_points.is_empty());
    }

    #[test]
    fn sampled_invariants_hold_for_bounded_property_cases() {
        for horizon in [1_u16, 2, 7, 32, 128] {
            for yaw in [
                -std::f64::consts::PI,
                -1.0,
                0.0,
                1.0,
                std::f64::consts::PI - 1.0e-12,
            ] {
                let mut builder = PathReferenceBuilderV1::new(config());
                prepare(
                    &mut builder,
                    &[
                        map_point(-2.0, -1.0),
                        map_point(0.0, -1.0),
                        map_point(0.0, 2.0),
                        map_point(3.0, 2.0),
                    ],
                );
                let points = builder
                    .build_reference_points(odom_pose(-2.0, -1.0, yaw), mpc_config(0.05, horizon))
                    .expect("property sample");
                assert_eq!(points.len(), usize::from(horizon));
                for point in points {
                    assert!(point.x_m.is_finite());
                    assert!(point.y_m.is_finite());
                    assert!(point.yaw_rad >= -std::f64::consts::PI);
                    assert!(point.yaw_rad < std::f64::consts::PI);
                    assert!(point.forward_velocity_mps >= 0.0);
                    assert!(point.forward_velocity_mps <= config().target_forward_speed_mps());
                    assert!(point.yaw_rate_rad_s.abs() <= config().maximum_abs_yaw_rate_rad_s());
                }
            }
        }
    }
}
