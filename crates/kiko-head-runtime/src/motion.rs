use std::fmt;
use std::time::Duration;

use kiko_head_protocol::{
    FullTelemetry, HeadJoint, HeadPose, HeadPoseError, PositionAgreementTicks, PositionStepLimit,
    PositionTicks,
};

use crate::config::{HeadPoseBoundsAdmissionError, HeadReturnPlan};
use crate::transport::MonotonicTime;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FreshHeadTelemetrySet {
    samples: [FullTelemetry; 4],
    received_at: [MonotonicTime; 4],
    admitted_at: MonotonicTime,
}

impl FreshHeadTelemetrySet {
    pub(crate) fn try_new(
        samples: [FullTelemetry; 4],
        received_at: [MonotonicTime; 4],
        admitted_at: MonotonicTime,
        maximum_age: Duration,
    ) -> Result<Self, HeadMotionError> {
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            let sample = samples[index];
            if sample.id() != joint.servo_id() {
                return Err(HeadMotionError::TelemetryOrderMismatch {
                    index,
                    expected_joint: joint,
                    actual_id: sample.id().get(),
                });
            }
            if sample.device_status_raw() != 0 {
                return Err(HeadMotionError::DeviceStatus {
                    joint,
                    raw: sample.device_status_raw(),
                });
            }
        }

        for index in 1..received_at.len() {
            if received_at[index] < received_at[index - 1] {
                return Err(HeadMotionError::TelemetryClockRegression {
                    previous_index: index - 1,
                    previous: received_at[index - 1],
                    observed_index: index,
                    observed: received_at[index],
                });
            }
        }
        let newest = received_at[3];
        if admitted_at < newest {
            return Err(HeadMotionError::ClockRegression {
                previous: newest,
                observed: admitted_at,
            });
        }
        let span = newest
            .checked_duration_since(received_at[0])
            .expect("telemetry timestamp ordering was checked");
        if span > maximum_age {
            return Err(HeadMotionError::TelemetrySetSpanExceeded {
                span,
                maximum: maximum_age,
            });
        }
        let age = admitted_at
            .checked_duration_since(received_at[0])
            .expect("telemetry admission follows every sample");
        if age > maximum_age {
            return Err(HeadMotionError::TelemetrySetStale {
                age,
                maximum: maximum_age,
            });
        }
        Ok(Self {
            samples,
            received_at,
            admitted_at,
        })
    }

    pub(crate) const fn samples(self) -> [FullTelemetry; 4] {
        self.samples
    }

    pub(crate) const fn received_at(self) -> [MonotonicTime; 4] {
        self.received_at
    }

    pub(crate) const fn admitted_at(self) -> MonotonicTime {
        self.admitted_at
    }

    fn ensure_fresh_at(
        self,
        now: MonotonicTime,
        maximum_age: Duration,
    ) -> Result<(), HeadMotionError> {
        let age = now.checked_duration_since(self.received_at[0]).ok_or(
            HeadMotionError::ClockRegression {
                previous: self.received_at[0],
                observed: now,
            },
        )?;
        if age > maximum_age {
            return Err(HeadMotionError::TelemetrySetStale {
                age,
                maximum: maximum_age,
            });
        }
        Ok(())
    }
}

pub(crate) fn admit_stopped_return_start(
    first: FreshHeadTelemetrySet,
    second: FreshHeadTelemetrySet,
    tolerance: PositionAgreementTicks,
) -> Result<HeadPose, HeadMotionError> {
    for (set, samples) in [
        (ReturnStartSample::First, first.samples()),
        (ReturnStartSample::Second, second.samples()),
    ] {
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            if samples[index].is_moving() {
                return Err(HeadMotionError::ReturnStartMoving {
                    joint,
                    sample: set,
                    position: samples[index].position(),
                });
            }
        }
    }
    HeadPose::try_from_telemetry_pair(first.samples(), second.samples(), tolerance)
        .map_err(|source| HeadMotionError::ReturnStartPose { source })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReturnStartSample {
    First,
    Second,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AdmittedRecoveryPose {
    positions: [PositionTicks; 4],
}

impl AdmittedRecoveryPose {
    pub(crate) const fn positions(self) -> [PositionTicks; 4] {
        self.positions
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HeadMotionFault {
    source: HeadMotionError,
    recovery: Option<AdmittedRecoveryPose>,
}

impl HeadMotionFault {
    pub(crate) const fn source(&self) -> &HeadMotionError {
        &self.source
    }

    pub(crate) const fn recovery(&self) -> Option<AdmittedRecoveryPose> {
        self.recovery
    }

    pub(crate) fn into_source(self) -> HeadMotionError {
        self.source
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum HeadReturnAction {
    WriteWaypoints([PositionTicks; 4]),
    AwaitSecondStoppedSample,
    Complete(Box<[FreshHeadTelemetrySet; 2]>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadMotionError {
    StartOutsideTravelLimit {
        joint: HeadJoint,
        start: PositionTicks,
        target: PositionTicks,
        travel_ticks: u16,
        maximum_ticks: u16,
    },
    ReturnStartMoving {
        joint: HeadJoint,
        sample: ReturnStartSample,
        position: PositionTicks,
    },
    ReturnStartPose {
        source: HeadPoseError,
    },
    ReturnStartOutsideConfiguredBounds {
        source: HeadPoseBoundsAdmissionError,
    },
    ClockRegression {
        previous: MonotonicTime,
        observed: MonotonicTime,
    },
    TelemetryClockRegression {
        previous_index: usize,
        previous: MonotonicTime,
        observed_index: usize,
        observed: MonotonicTime,
    },
    TelemetrySetSpanExceeded {
        span: Duration,
        maximum: Duration,
    },
    TelemetrySetStale {
        age: Duration,
        maximum: Duration,
    },
    MotionTimeout {
        elapsed: Duration,
        maximum: Duration,
    },
    TelemetryOrderMismatch {
        index: usize,
        expected_joint: HeadJoint,
        actual_id: u8,
    },
    DeviceStatus {
        joint: HeadJoint,
        raw: u8,
    },
    OutsidePathCorridor {
        joint: HeadJoint,
        actual: PositionTicks,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
    DirectionRegression {
        joint: HeadJoint,
        actual: PositionTicks,
        target: PositionTicks,
        distance_ticks: u16,
        best_distance_ticks: u16,
        tolerance_ticks: u16,
    },
    NoProgress {
        joint: HeadJoint,
        elapsed: Duration,
        maximum: Duration,
        best_distance_ticks: u16,
    },
    FinalSamplesDisagree {
        joint: HeadJoint,
        first: PositionTicks,
        second: PositionTicks,
        difference_ticks: u16,
        tolerance_ticks: u16,
    },
}

impl fmt::Display for HeadMotionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "bounded Kiko head return failed: {self:?}")
    }
}

impl std::error::Error for HeadMotionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ReturnStartPose { source } => Some(source),
            Self::ReturnStartOutsideConfiguredBounds { source } => Some(source),
            Self::StartOutsideTravelLimit { .. }
            | Self::ReturnStartMoving { .. }
            | Self::ClockRegression { .. }
            | Self::TelemetryClockRegression { .. }
            | Self::TelemetrySetSpanExceeded { .. }
            | Self::TelemetrySetStale { .. }
            | Self::MotionTimeout { .. }
            | Self::TelemetryOrderMismatch { .. }
            | Self::DeviceStatus { .. }
            | Self::OutsidePathCorridor { .. }
            | Self::DirectionRegression { .. }
            | Self::NoProgress { .. }
            | Self::FinalSamplesDisagree { .. } => None,
        }
    }
}

impl HeadMotionError {
    /// A rejection which did not invalidate the actor's last complete,
    /// previously admitted goal can keep that goal and serial ownership. This
    /// never authorizes a new goal derived from untrusted telemetry.
    pub(crate) const fn permits_existing_goal_retention(&self) -> bool {
        matches!(
            self,
            Self::StartOutsideTravelLimit { .. }
                | Self::ReturnStartMoving { .. }
                | Self::ReturnStartPose { .. }
                | Self::ReturnStartOutsideConfiguredBounds { .. }
                | Self::MotionTimeout { .. }
                | Self::OutsidePathCorridor { .. }
                | Self::DirectionRegression { .. }
                | Self::NoProgress { .. }
                | Self::FinalSamplesDisagree { .. }
        )
    }
}

pub(crate) struct HeadReturnController {
    plan: HeadReturnPlan,
    started_at: MonotonicTime,
    previous_at: MonotonicTime,
    start: [PositionTicks; 4],
    best_distance_ticks: [u16; 4],
    last_progress_at: [MonotonicTime; 4],
    first_stopped: Option<FreshHeadTelemetrySet>,
    exact_target_commanded: bool,
    step: PositionStepLimit,
}

impl HeadReturnController {
    pub(crate) fn try_new(
        plan: HeadReturnPlan,
        start: HeadPose,
        started_at: MonotonicTime,
        motion_started_at: MonotonicTime,
    ) -> Result<Self, HeadMotionError> {
        if motion_started_at < started_at {
            return Err(HeadMotionError::ClockRegression {
                previous: started_at,
                observed: motion_started_at,
            });
        }
        let start = start.positions();
        let target = plan.target().positions();
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            let travel_ticks = start[index].get().abs_diff(target[index].get());
            let maximum_ticks = plan.maximum_travel_ticks(joint);
            if travel_ticks > maximum_ticks {
                return Err(HeadMotionError::StartOutsideTravelLimit {
                    joint,
                    start: start[index],
                    target: target[index],
                    travel_ticks,
                    maximum_ticks,
                });
            }
        }
        let best_distance_ticks =
            std::array::from_fn(|index| start[index].get().abs_diff(target[index].get()));
        Ok(Self {
            plan,
            started_at,
            previous_at: motion_started_at,
            start,
            best_distance_ticks,
            last_progress_at: [motion_started_at; 4],
            first_stopped: None,
            exact_target_commanded: false,
            step: PositionStepLimit::try_new(plan.position_step_ticks())
                .expect("fixed nonzero position step is in the encoder domain"),
        })
    }

    pub(crate) fn remaining_operation_budget(
        &self,
        now: MonotonicTime,
    ) -> Result<Duration, HeadMotionError> {
        if now < self.previous_at {
            return Err(HeadMotionError::ClockRegression {
                previous: self.previous_at,
                observed: now,
            });
        }
        let elapsed = now
            .checked_duration_since(self.started_at)
            .expect("monotonic ordering was checked");
        if elapsed >= self.plan.motion_timeout() {
            return Err(HeadMotionError::MotionTimeout {
                elapsed,
                maximum: self.plan.motion_timeout(),
            });
        }
        let mut remaining = self
            .plan
            .motion_timeout()
            .checked_sub(elapsed)
            .expect("elapsed is inside the motion deadline");
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            if self.best_distance_ticks[index] <= self.plan.final_target_tolerance().get() {
                continue;
            }
            let stalled_for = now
                .checked_duration_since(self.last_progress_at[index])
                .expect("progress timestamps do not exceed current time");
            if stalled_for >= self.plan.no_progress_timeout() {
                return Err(HeadMotionError::NoProgress {
                    joint,
                    elapsed: stalled_for,
                    maximum: self.plan.no_progress_timeout(),
                    best_distance_ticks: self.best_distance_ticks[index],
                });
            }
            remaining = remaining.min(
                self.plan
                    .no_progress_timeout()
                    .checked_sub(stalled_for)
                    .expect("stall duration is inside its deadline"),
            );
        }
        Ok(remaining)
    }

    pub(crate) fn record_waypoint_written(&mut self, positions: [PositionTicks; 4]) {
        if positions == self.plan.target().positions() {
            self.exact_target_commanded = true;
        }
    }

    pub(crate) fn advance(
        &mut self,
        now: MonotonicTime,
        telemetry: FreshHeadTelemetrySet,
    ) -> Result<HeadReturnAction, HeadMotionFault> {
        if now < self.previous_at {
            return Err(HeadMotionFault {
                source: HeadMotionError::ClockRegression {
                    previous: self.previous_at,
                    observed: now,
                },
                recovery: None,
            });
        }
        if let Err(source) = telemetry.ensure_fresh_at(now, self.plan.telemetry_set_max_age()) {
            return Err(HeadMotionFault {
                source,
                recovery: None,
            });
        }
        self.previous_at = now;

        let samples = telemetry.samples();
        let target = self.plan.target().positions();
        let corridor_tolerance = self.plan.path_corridor_tolerance().get();
        let mut first_corridor_fault = None;
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            let actual = samples[index].position();
            let minimum_raw = self.start[index]
                .get()
                .min(target[index].get())
                .saturating_sub(corridor_tolerance);
            let maximum_raw = self.start[index]
                .get()
                .max(target[index].get())
                .saturating_add(corridor_tolerance)
                .min(PositionTicks::MAX.get());
            let minimum = PositionTicks::try_new(minimum_raw)
                .expect("saturating corridor lower bound is valid");
            let maximum =
                PositionTicks::try_new(maximum_raw).expect("clamped corridor upper bound is valid");
            if first_corridor_fault.is_none() && (actual < minimum || actual > maximum) {
                first_corridor_fault = Some(HeadMotionError::OutsidePathCorridor {
                    joint,
                    actual,
                    minimum,
                    maximum,
                });
            }
        }
        if let Some(source) = first_corridor_fault {
            return Err(HeadMotionFault {
                source,
                recovery: None,
            });
        }
        let recovery = AdmittedRecoveryPose {
            positions: samples.map(FullTelemetry::position),
        };

        let elapsed = now
            .checked_duration_since(self.started_at)
            .expect("monotonic ordering was checked");
        if elapsed >= self.plan.motion_timeout() {
            return Err(HeadMotionFault {
                source: HeadMotionError::MotionTimeout {
                    elapsed,
                    maximum: self.plan.motion_timeout(),
                },
                recovery: Some(recovery),
            });
        }

        let distances: [u16; 4] = std::array::from_fn(|index| {
            samples[index]
                .position()
                .get()
                .abs_diff(target[index].get())
        });
        let direction_tolerance = self.plan.direction_regression_tolerance().get();
        let mut first_direction_fault = None;
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            if first_direction_fault.is_none()
                && distances[index]
                    > self.best_distance_ticks[index].saturating_add(direction_tolerance)
            {
                first_direction_fault = Some(HeadMotionError::DirectionRegression {
                    joint,
                    actual: samples[index].position(),
                    target: target[index],
                    distance_ticks: distances[index],
                    best_distance_ticks: self.best_distance_ticks[index],
                    tolerance_ticks: direction_tolerance,
                });
            }
        }
        if let Some(source) = first_direction_fault {
            return Err(HeadMotionFault {
                source,
                recovery: Some(recovery),
            });
        }

        for (index, distance) in distances.iter().copied().enumerate() {
            if distance < self.best_distance_ticks[index] {
                self.best_distance_ticks[index] = distance;
                self.last_progress_at[index] = now;
            }
        }
        let final_tolerance = self.plan.final_target_tolerance().get();
        for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
            if distances[index] <= final_tolerance {
                continue;
            }
            let stalled_for = now
                .checked_duration_since(self.last_progress_at[index])
                .expect("per-joint progress time never exceeds current time");
            if stalled_for >= self.plan.no_progress_timeout() {
                return Err(HeadMotionFault {
                    source: HeadMotionError::NoProgress {
                        joint,
                        elapsed: stalled_for,
                        maximum: self.plan.no_progress_timeout(),
                        best_distance_ticks: self.best_distance_ticks[index],
                    },
                    recovery: Some(recovery),
                });
            }
        }

        let all_stopped_at_target = HeadJoint::ALL.into_iter().all(|joint| {
            let index = joint as usize;
            distances[index] <= final_tolerance && !samples[index].is_moving()
        });
        if all_stopped_at_target && self.exact_target_commanded {
            if let Some(first) = self.first_stopped.take() {
                let first_samples = first.samples();
                for (index, joint) in HeadJoint::ALL.into_iter().enumerate() {
                    let difference_ticks = first_samples[index]
                        .position()
                        .get()
                        .abs_diff(samples[index].position().get());
                    if difference_ticks > self.plan.final_sample_tolerance().get() {
                        return Err(HeadMotionFault {
                            source: HeadMotionError::FinalSamplesDisagree {
                                joint,
                                first: first_samples[index].position(),
                                second: samples[index].position(),
                                difference_ticks,
                                tolerance_ticks: self.plan.final_sample_tolerance().get(),
                            },
                            recovery: Some(recovery),
                        });
                    }
                }
                return Ok(HeadReturnAction::Complete(Box::new([first, telemetry])));
            }
            self.first_stopped = Some(telemetry);
            return Ok(HeadReturnAction::AwaitSecondStoppedSample);
        }
        self.first_stopped = None;

        let waypoints = if all_stopped_at_target {
            // Tolerance-based completion is never allowed to fabricate evidence
            // that the exact reviewed target was actually written.
            target
        } else {
            std::array::from_fn(|index| self.step.advance(samples[index].position(), target[index]))
        };
        Ok(HeadReturnAction::WriteWaypoints(waypoints))
    }
}

#[cfg(test)]
mod tests {
    use kiko_head_protocol::{ExactHeadTargetPose, PresentPosition, ValidatedPresentPosition};

    use super::*;

    fn response(id: u8, parameters: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0xff, 0xff, id, 0, 0];
        bytes[3] = u8::try_from(parameters.len() + 2).expect("parameter count");
        bytes.extend_from_slice(parameters);
        let checksum = !bytes[2..]
            .iter()
            .fold(0_u8, |sum, byte| sum.wrapping_add(*byte));
        bytes.push(checksum);
        bytes
    }

    fn telemetry(positions: [u16; 4], moving: bool) -> [FullTelemetry; 4] {
        telemetry_with_status(positions, moving, [0; 4])
    }

    fn telemetry_with_status(
        positions: [u16; 4],
        moving: bool,
        device_status: [u8; 4],
    ) -> [FullTelemetry; 4] {
        std::array::from_fn(|index| {
            let joint = HeadJoint::ALL[index];
            let mut data = [0_u8; 15];
            data[..2].copy_from_slice(&positions[index].to_le_bytes());
            data[9] = device_status[index];
            data[10] = u8::from(moving);
            FullTelemetry::parse(&response(joint.servo_id().get(), &data), joint.servo_id())
                .expect("telemetry")
        })
    }

    fn observed_pose(positions: [u16; 4]) -> HeadPose {
        let admitted = std::array::from_fn(|index| {
            let joint = HeadJoint::ALL[index];
            let bytes = response(joint.servo_id().get(), &positions[index].to_le_bytes());
            let position = PresentPosition::parse(&bytes, joint.servo_id()).expect("position");
            ValidatedPresentPosition::try_from_pair(
                position,
                position,
                PositionAgreementTicks::try_new(0).expect("zero tolerance"),
            )
            .expect("validated")
        });
        HeadPose::try_from_validated(admitted).expect("pose")
    }

    fn ticks(value: u16) -> PositionAgreementTicks {
        PositionAgreementTicks::try_new(value).expect("test tolerance")
    }

    fn plan(target: [u16; 4]) -> HeadReturnPlan {
        HeadReturnPlan::for_test(
            ExactHeadTargetPose::try_from_ticks(target).expect("target"),
            [400, 400, 64, 64],
            ticks(20),
            ticks(20),
            ticks(20),
            ticks(10),
        )
    }

    fn at(milliseconds: u64) -> MonotonicTime {
        MonotonicTime::from_duration_since_origin(Duration::from_millis(milliseconds))
    }

    fn set(at_ms: u64, positions: [u16; 4], moving: bool) -> FreshHeadTelemetrySet {
        let received_at = [at(at_ms), at(at_ms + 1), at(at_ms + 2), at(at_ms + 3)];
        FreshHeadTelemetrySet::try_new(
            telemetry(positions, moving),
            received_at,
            at(at_ms + 3),
            Duration::from_millis(100),
        )
        .expect("fresh set")
    }

    fn controller(start: [u16; 4], target: [u16; 4]) -> HeadReturnController {
        HeadReturnController::try_new(plan(target), observed_pose(start), at(0), at(0))
            .expect("admitted start travel")
    }

    #[test]
    fn fresh_set_rejects_status_order_and_time_faults_before_positions_are_usable() {
        let positions = [2_512, 2_916, 2_903, 2_903];
        assert!(matches!(
            FreshHeadTelemetrySet::try_new(
                telemetry_with_status(positions, false, [0, 0, 7, 0]),
                [at(0), at(1), at(2), at(3)],
                at(3),
                Duration::from_millis(100),
            ),
            Err(HeadMotionError::DeviceStatus {
                joint: HeadJoint::Yaw,
                raw: 7,
            })
        ));
        assert!(matches!(
            FreshHeadTelemetrySet::try_new(
                telemetry(positions, false),
                [at(0), at(2), at(1), at(3)],
                at(3),
                Duration::from_millis(100),
            ),
            Err(HeadMotionError::TelemetryClockRegression { .. })
        ));
        assert!(matches!(
            FreshHeadTelemetrySet::try_new(
                telemetry(positions, false),
                [at(0), at(1), at(2), at(101)],
                at(101),
                Duration::from_millis(100),
            ),
            Err(HeadMotionError::TelemetrySetSpanExceeded { .. })
        ));
    }

    #[test]
    fn waypoints_are_fifty_ticks_from_fresh_telemetry_and_never_cross_target() {
        let start = [2_512, 2_916, 2_903, 2_903];
        let target = [2_155, 2_545, 2_943, 2_876];
        let mut controller = controller(start, target);
        let HeadReturnAction::WriteWaypoints(next) = controller
            .advance(at(3), set(0, start, false))
            .expect("first step")
        else {
            panic!("expected waypoint action");
        };
        assert_eq!(next.map(PositionTicks::get), [2_462, 2_866, 2_943, 2_876]);

        let near = [2_160, 2_550, 2_940, 2_880];
        assert!(matches!(
            controller.advance(at(103), set(100, near, true)),
            Ok(HeadReturnAction::WriteWaypoints(next))
                if next.map(PositionTicks::get) == target
        ));
    }

    #[test]
    fn exact_target_write_and_two_stopped_samples_are_required_for_completion() {
        let target = [2_155, 2_545, 2_943, 2_876];
        let mut controller = controller([2_160, 2_550, 2_940, 2_880], target);
        assert!(matches!(
            controller.advance(at(3), set(0, target, false)),
            Ok(HeadReturnAction::WriteWaypoints(positions)) if positions.map(PositionTicks::get) == target
        ));
        controller.record_waypoint_written(plan(target).target().positions());
        assert!(matches!(
            controller.advance(at(103), set(100, target, false)),
            Ok(HeadReturnAction::AwaitSecondStoppedSample)
        ));
        assert!(matches!(
            controller.advance(at(203), set(200, target, false)),
            Ok(HeadReturnAction::Complete(_))
        ));
    }

    #[test]
    fn complete_geometry_is_admitted_before_a_recovery_pose_exists() {
        let start = [2_512, 2_916, 2_903, 2_903];
        let target = [2_155, 2_545, 2_943, 2_876];
        let mut active_controller = controller(start, target);
        let fault = active_controller
            .advance(at(3), set(0, [2_490, 2_960, 2_943, 2_876], false))
            .expect_err("curl is outside the complete path corridor");
        assert!(matches!(
            fault.source(),
            HeadMotionError::OutsidePathCorridor {
                joint: HeadJoint::Curl,
                ..
            }
        ));
        assert_eq!(fault.recovery(), None);

        let mut timed_out = controller(start, target);
        let fault = timed_out
            .advance(at(20_000), set(19_997, start, false))
            .expect_err("whole motion deadline");
        assert!(matches!(
            fault.source(),
            HeadMotionError::MotionTimeout { .. }
        ));
        assert!(fault.recovery().is_some());
    }

    #[test]
    fn runtime_start_admission_defends_against_a_cross_paired_plan() {
        let target = [2_155, 2_545, 2_943, 2_876];
        let result = HeadReturnController::try_new(
            HeadReturnPlan::for_test(
                ExactHeadTargetPose::try_from_ticks(target).expect("target"),
                [100, 400, 64, 64],
                ticks(20),
                ticks(20),
                ticks(20),
                ticks(10),
            ),
            observed_pose([2_512, 2_916, 2_903, 2_903]),
            at(0),
            at(0),
        );
        assert!(matches!(
            result,
            Err(HeadMotionError::StartOutsideTravelLimit {
                joint: HeadJoint::Bow,
                travel_ticks: 357,
                maximum_ticks: 100,
                ..
            })
        ));
    }
}
