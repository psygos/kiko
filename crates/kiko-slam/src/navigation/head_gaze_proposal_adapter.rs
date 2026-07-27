//! Proposal-only seam from current RGB face evidence to head-gaze mapping.
//!
//! This module admits only a fresh, current-frame `Tracked` or `Switched`
//! face. Every other tracker state withholds a proposal. The resulting value
//! remains a [`HeadGazeTargetProposal`], not an actuator command: it cannot be
//! converted into `ExactHeadTargetPose`, `HeadGazeControlConfig`, torque
//! consent, or motion consent.
//!
//! A policy's lifecycle field is deliberately not inspected here. Both
//! lifecycle variants are caller claims, not physical calibration evidence or
//! authority. A later activation boundary must cross-bind retained evidence,
//! prove a shared monotonic-clock origin, construct an exact pose, and submit
//! each bounded planner step through the actor's verified goal-register
//! transaction.

use std::{fmt, time::Duration};

use kiko_expression_core::{
    ImageLayout, ImagePoint, MonotonicTimestamp, PersonTrackId, RgbObservation,
};
use kiko_expression_runtime::{
    CameraGazeTargetError, CameraRayHeadProposalError, FaceTargetState, FaceTrackingUpdate,
    HeadGazeTargetProposal, OakCameraTargetRay,
};
use kiko_head_runtime::gaze_control::HeadProposalTtl;

use crate::PinholeIntrinsics;

use super::HeadGazePolicyV1;

/// Longest accepted silence between the last face observation and the
/// transport-free controller's transition toward its natural target.
///
/// This bounds the return *trigger* when a later activation boundary uses the
/// exact declared timing. It does not claim the physical head reaches natural
/// within this duration.
pub const MAXIMUM_HEAD_GAZE_RETURN_TRIGGER_DELAY: Duration = Duration::from_millis(400);

/// Proof that the policy's exclusive proposal lifetime cannot postpone the
/// controller's natural-return trigger beyond the accepted bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadGazeReturnTriggerBound {
    proposal_ttl: HeadProposalTtl,
}

impl HeadGazeReturnTriggerBound {
    fn try_new(proposal_ttl: HeadProposalTtl) -> Result<Self, HeadGazeFaceProposalAdapterError> {
        if proposal_ttl.get() > MAXIMUM_HEAD_GAZE_RETURN_TRIGGER_DELAY {
            return Err(
                HeadGazeFaceProposalAdapterError::ReturnTriggerDelayAboveMaximum {
                    actual: proposal_ttl.get(),
                    maximum: MAXIMUM_HEAD_GAZE_RETURN_TRIGGER_DELAY,
                },
            );
        }
        Ok(Self { proposal_ttl })
    }

    /// Exclusive lifetime of one admitted face proposal.
    pub const fn proposal_ttl(self) -> HeadProposalTtl {
        self.proposal_ttl
    }

    /// Maximum silence admitted before a matching controller must stop using
    /// the last face proposal and begin returning natural.
    pub const fn maximum_delay(self) -> Duration {
        self.proposal_ttl.get()
    }
}

/// Exact pinhole calibration and pixel grid used to turn one normalized face
/// centre into an OAK-camera ray.
#[derive(Clone, Copy, Debug)]
pub struct RgbFacePinholeProjection {
    intrinsics: PinholeIntrinsics,
    layout: ImageLayout,
}

impl RgbFacePinholeProjection {
    /// Bind already-parsed intrinsics to one already-checked RGB layout.
    ///
    /// The caller must source both from the same delivered OAK stream
    /// configuration. Each proposal is still checked against this exact layout.
    pub const fn new(intrinsics: PinholeIntrinsics, layout: ImageLayout) -> Self {
        Self { intrinsics, layout }
    }

    pub const fn layout(self) -> ImageLayout {
        self.layout
    }

    pub const fn intrinsics(self) -> PinholeIntrinsics {
        self.intrinsics
    }

    fn ray_for(
        self,
        target: FreshCurrentFaceGazeTarget,
    ) -> Result<OakCameraTargetRay, HeadGazeFaceProposalError> {
        let actual = target.observation().layout();
        if actual != self.layout {
            return Err(HeadGazeFaceProposalError::ProjectionLayoutMismatch {
                configured: self.layout,
                actual,
            });
        }

        // ImagePoint represents pixel-centre coordinates normalized against
        // image edges. Undo that convention before applying the pinhole K
        // matrix, whose image coordinates address pixel centres.
        let center = target.center();
        let pixel_x = center.x_right().as_f64() * f64::from(self.layout.width_px()) - 0.5;
        let pixel_y = center.y_down().as_f64() * f64::from(self.layout.height_px()) - 0.5;
        let x = (pixel_x - f64::from(self.intrinsics.cx())) / f64::from(self.intrinsics.fx());
        let y = (pixel_y - f64::from(self.intrinsics.cy())) / f64::from(self.intrinsics.fy());
        OakCameraTargetRay::parse([x, y, 1.0]).map_err(HeadGazeFaceProposalError::CameraRay)
    }
}

/// Why an admitted current face entered this proposal seam.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreshFaceGazeTransition {
    Tracked,
    Switched { previous_track_id: PersonTrackId },
}

/// Fresh current-frame face evidence whose construction is private to the
/// admission check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FreshCurrentFaceGazeTarget {
    observation: RgbObservation,
    track_id: PersonTrackId,
    center: ImagePoint,
    transition: FreshFaceGazeTransition,
}

impl FreshCurrentFaceGazeTarget {
    pub const fn observation(self) -> RgbObservation {
        self.observation
    }

    pub const fn track_id(self) -> PersonTrackId {
        self.track_id
    }

    pub const fn center(self) -> ImagePoint {
        self.center
    }

    pub const fn transition(self) -> FreshFaceGazeTransition {
        self.transition
    }
}

/// A mapped gaze value with the exact face and camera-ray provenance that
/// produced it. This remains proposal-only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadGazeFaceProposal {
    face: FreshCurrentFaceGazeTarget,
    camera_ray: OakCameraTargetRay,
    target: HeadGazeTargetProposal,
}

impl HeadGazeFaceProposal {
    pub const fn face(self) -> FreshCurrentFaceGazeTarget {
        self.face
    }

    pub const fn camera_ray(self) -> OakCameraTargetRay {
        self.camera_ray
    }

    pub const fn target(self) -> HeadGazeTargetProposal {
        self.target
    }
}

/// Tracker states and freshness failures that intentionally produce no head
/// target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeFaceProposalWithheld {
    NoTarget,
    Acquiring,
    Coasting,
    Lost,
    TrackedNotFresh {
        observed_at: MonotonicTimestamp,
        valid_until_exclusive: MonotonicTimestamp,
        evaluated_at: MonotonicTimestamp,
    },
    SwitchedNotFresh {
        observed_at: MonotonicTimestamp,
        valid_until_exclusive: MonotonicTimestamp,
        evaluated_at: MonotonicTimestamp,
    },
    ControllerProposalDeadlineOverflow {
        transition: FreshFaceGazeTransition,
        observed_at: MonotonicTimestamp,
        proposal_ttl: Duration,
    },
    CurrentEvidenceMismatch,
}

/// One face update either yields one non-command proposal or explicitly
/// withholds motion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadGazeFaceProposalOutcome {
    Proposed(HeadGazeFaceProposal),
    Withheld(HeadGazeFaceProposalWithheld),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeFaceProposalAdapterError {
    ReturnTriggerDelayAboveMaximum { actual: Duration, maximum: Duration },
}

impl fmt::Display for HeadGazeFaceProposalAdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "head-gaze proposal adapter policy is not admitted: {self:?}"
        )
    }
}

impl std::error::Error for HeadGazeFaceProposalAdapterError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadGazeFaceProposalError {
    ProjectionLayoutMismatch {
        configured: ImageLayout,
        actual: ImageLayout,
    },
    CameraRay(CameraGazeTargetError),
    Mapping(CameraRayHeadProposalError),
}

impl fmt::Display for HeadGazeFaceProposalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot prepare face head-gaze proposal: {self:?}"
        )
    }
}

impl std::error::Error for HeadGazeFaceProposalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CameraRay(source) => Some(source),
            Self::Mapping(source) => Some(source),
            Self::ProjectionLayoutMismatch { .. } => None,
        }
    }
}

/// Allocation-free proposal adapter for one parsed policy and RGB camera
/// calibration.
///
/// Construction validates only the proposal-silence bound. It deliberately
/// does not activate either lifecycle claim and cannot construct a head
/// controller or actuator command.
#[derive(Debug)]
pub struct HeadGazeFaceProposalAdapter<'policy> {
    policy: &'policy HeadGazePolicyV1,
    projection: RgbFacePinholeProjection,
    return_trigger: HeadGazeReturnTriggerBound,
}

impl<'policy> HeadGazeFaceProposalAdapter<'policy> {
    pub fn try_new(
        policy: &'policy HeadGazePolicyV1,
        projection: RgbFacePinholeProjection,
    ) -> Result<Self, HeadGazeFaceProposalAdapterError> {
        let return_trigger =
            HeadGazeReturnTriggerBound::try_new(policy.controller().timing().proposal_ttl())?;
        Ok(Self {
            policy,
            projection,
            return_trigger,
        })
    }

    pub const fn return_trigger_bound(&self) -> HeadGazeReturnTriggerBound {
        self.return_trigger
    }

    /// Evaluate one accepted face-tracker update at an exact process-local
    /// monotonic timestamp.
    ///
    /// `NoTarget`, `Acquiring`, `Coasting`, and `Lost` never reuse historical
    /// face coordinates. A `Tracked` or `Switched` state also must retain exact
    /// current-frame provenance and be fresh at `evaluated_at`.
    pub fn evaluate(
        &self,
        update: FaceTrackingUpdate,
        evaluated_at: MonotonicTimestamp,
    ) -> Result<HeadGazeFaceProposalOutcome, HeadGazeFaceProposalError> {
        let face = match admit_fresh_current_face(update, evaluated_at, self.return_trigger) {
            Ok(face) => face,
            Err(reason) => return Ok(HeadGazeFaceProposalOutcome::Withheld(reason)),
        };
        let camera_ray = self.projection.ray_for(face)?;
        let target = self
            .policy
            .mapping()
            .proposal_for_camera_ray(camera_ray)
            .map_err(HeadGazeFaceProposalError::Mapping)?;
        Ok(HeadGazeFaceProposalOutcome::Proposed(
            HeadGazeFaceProposal {
                face,
                camera_ray,
                target,
            },
        ))
    }
}

fn admit_fresh_current_face(
    update: FaceTrackingUpdate,
    evaluated_at: MonotonicTimestamp,
    return_trigger: HeadGazeReturnTriggerBound,
) -> Result<FreshCurrentFaceGazeTarget, HeadGazeFaceProposalWithheld> {
    let current = update.observation();
    let (observation, transition) = match update.state() {
        FaceTargetState::Tracked(observation) => (observation, FreshFaceGazeTransition::Tracked),
        FaceTargetState::Switched(switched) => (
            switched.observation(),
            FreshFaceGazeTransition::Switched {
                previous_track_id: switched.previous_track_id(),
            },
        ),
        FaceTargetState::NoTarget => return Err(HeadGazeFaceProposalWithheld::NoTarget),
        FaceTargetState::Acquiring(_) => {
            return Err(HeadGazeFaceProposalWithheld::Acquiring);
        }
        FaceTargetState::Coasting(_) => return Err(HeadGazeFaceProposalWithheld::Coasting),
        FaceTargetState::Lost(_) => return Err(HeadGazeFaceProposalWithheld::Lost),
    };

    let freshness = observation.freshness();
    if observation.frame_id() != current.frame_id()
        || freshness != current.freshness()
        || observation.detection().rectangle().layout() != current.layout()
    {
        return Err(HeadGazeFaceProposalWithheld::CurrentEvidenceMismatch);
    }
    let observed_at = freshness.observed_at();
    let proposal_ttl = return_trigger.proposal_ttl().get();
    let proposal_ttl_ns = u64::try_from(proposal_ttl.as_nanos())
        .expect("an admitted sub-second proposal TTL fits u64 nanoseconds");
    let Some(proposal_deadline_ns) = observed_at.nanos_since_epoch().checked_add(proposal_ttl_ns)
    else {
        return Err(
            HeadGazeFaceProposalWithheld::ControllerProposalDeadlineOverflow {
                transition,
                observed_at,
                proposal_ttl,
            },
        );
    };
    let proposal_deadline = MonotonicTimestamp::from_nanos_since_epoch(proposal_deadline_ns);
    let frame_deadline = freshness.valid_until_exclusive().timestamp();
    let effective_deadline = frame_deadline.min(proposal_deadline);
    if evaluated_at < observed_at || evaluated_at >= effective_deadline {
        return Err(match transition {
            FreshFaceGazeTransition::Tracked => HeadGazeFaceProposalWithheld::TrackedNotFresh {
                observed_at,
                valid_until_exclusive: effective_deadline,
                evaluated_at,
            },
            FreshFaceGazeTransition::Switched { .. } => {
                HeadGazeFaceProposalWithheld::SwitchedNotFresh {
                    observed_at,
                    valid_until_exclusive: effective_deadline,
                    evaluated_at,
                }
            }
        });
    }

    Ok(FreshCurrentFaceGazeTarget {
        observation: current,
        track_id: observation.track_id(),
        center: observation.center(),
        transition,
    })
}

#[cfg(test)]
mod tests {
    use kiko_expression_core::{
        ChannelOrder, FrameId, FreshnessWindow, NonZeroDuration, RgbObservation, StreamEpochId,
    };
    use kiko_expression_runtime::{
        DetectorResultSequence, FaceDetection, FaceDetectionBatch, FaceDetectorSource, FaceTracker,
        FaceTrackingConfig,
    };
    use kiko_head_protocol::HeadJoint;
    use serde_json::{Value, json};

    use super::*;

    const FRAME_TTL_NS: u64 = 5_000_000_000;
    const RESULT_PERIOD_NS: u64 = 100_000_000;

    fn layout(width: u32, height: u32) -> ImageLayout {
        ImageLayout::try_new(
            width,
            height,
            width.checked_mul(3).expect("test row fits"),
            ChannelOrder::Bgr,
        )
        .expect("valid test layout")
    }

    fn observation(sequence: u64, observed_at_ns: u64) -> RgbObservation {
        let observed_at = MonotonicTimestamp::from_nanos_since_epoch(observed_at_ns);
        RgbObservation::new(
            FrameId::new(
                StreamEpochId::try_new(1).expect("nonzero stream epoch"),
                sequence,
            ),
            layout(640, 400),
            FreshnessWindow::from_ttl(
                observed_at,
                NonZeroDuration::try_from_nanos(FRAME_TTL_NS).expect("nonzero frame TTL"),
            )
            .expect("test freshness does not overflow"),
        )
    }

    fn face(left: u32, top: u32, width: u32, height: u32) -> FaceDetection {
        FaceDetection::try_new(
            layout(640, 400),
            left,
            top,
            width,
            height,
            1.0,
            FaceDetectorSource::Frontal,
        )
        .expect("valid test face")
    }

    fn update_at(
        tracker: &mut FaceTracker,
        result_sequence: u64,
        observed_at_ns: u64,
        processed_at_ns: u64,
        detections: &[FaceDetection],
    ) -> FaceTrackingUpdate {
        let batch = FaceDetectionBatch::try_new(
            observation(result_sequence * 10, observed_at_ns),
            DetectorResultSequence::new(result_sequence),
            0,
            detections,
        )
        .expect("valid test detection batch");
        tracker
            .update(
                &batch,
                MonotonicTimestamp::from_nanos_since_epoch(processed_at_ns),
            )
            .expect("valid tracker update")
    }

    fn update(
        tracker: &mut FaceTracker,
        result_sequence: u64,
        detections: &[FaceDetection],
    ) -> FaceTrackingUpdate {
        let timestamp = result_sequence * RESULT_PERIOD_NS;
        update_at(tracker, result_sequence, timestamp, timestamp, detections)
    }

    fn policy_value(proposal_ttl_ns: u64) -> Value {
        json!({
            "schema_version": 1,
            "lifecycle": {
                "kind": "proposal_only",
                "proposal_id": "head-gaze-map-proposal-test",
                "evidence_id": "proposal-evidence-test",
                "evidence_content_sha256_hex":
                    "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"
            },
            "mapping_declaration": {
                "assembly_id": "kiko-head-assembly-01",
                "calibration_provenance_id": "unreviewed-test-proposal",
                "gaze_only_focus_plane": {
                    "camera_forward_depth_m": 1.5
                },
                "camera_to_neutral_head": {
                    "head_origin_in_oak_camera_m": {
                        "x_right_m": 0.0,
                        "y_down_m": -0.25,
                        "z_forward_m": -0.20
                    },
                    "neutral_head_from_oak_camera_quaternion_xyzw": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "w": 1.0
                    }
                },
                "natural_encoder_position_ticks": {
                    "bow_ticks": 2155,
                    "curl_ticks": 2545,
                    "yaw_ticks": 2943,
                    "roll_ticks": 2876
                },
                "hard_encoder_envelopes_ticks": {
                    "bow": {"minimum_ticks": 1955, "maximum_ticks": 2355},
                    "curl": {"minimum_ticks": 2345, "maximum_ticks": 2745},
                    "yaw": {"minimum_ticks": 2743, "maximum_ticks": 3143},
                    "roll": {"minimum_ticks": 2776, "maximum_ticks": 2976}
                },
                "encoder_tick_offsets_per_radian": {
                    "pitch_down_rad": {
                        "bow_ticks_per_radian": -300.0,
                        "curl_ticks_per_radian": 300.0,
                        "yaw_ticks_per_radian": 0.0,
                        "roll_ticks_per_radian": 0.0
                    },
                    "yaw_right_rad": {
                        "bow_ticks_per_radian": 0.0,
                        "curl_ticks_per_radian": 0.0,
                        "yaw_ticks_per_radian": 300.0,
                        "roll_ticks_per_radian": 0.0
                    }
                }
            },
            "controller_declaration": {
                "timing": {
                    "control_period_ns": 20_000_000,
                    "maximum_tick_lateness_ns": 5_000_000,
                    "proposal_ttl_ns": proposal_ttl_ns,
                    "acquisition_proposals": 3
                },
                "error_band": {
                    "settle_deadband_ticks": 2,
                    "resume_threshold_ticks": 5
                },
                "motion_limits": {
                    "bow": {
                        "maximum_velocity_ticks_per_control_tick": 8,
                        "maximum_acceleration_ticks_per_control_tick_squared": 2,
                        "maximum_position_step_ticks": 8
                    },
                    "curl": {
                        "maximum_velocity_ticks_per_control_tick": 8,
                        "maximum_acceleration_ticks_per_control_tick_squared": 2,
                        "maximum_position_step_ticks": 8
                    },
                    "yaw": {
                        "maximum_velocity_ticks_per_control_tick": 8,
                        "maximum_acceleration_ticks_per_control_tick_squared": 2,
                        "maximum_position_step_ticks": 8
                    },
                    "roll": {
                        "maximum_velocity_ticks_per_control_tick": 4,
                        "maximum_acceleration_ticks_per_control_tick_squared": 1,
                        "maximum_position_step_ticks": 4
                    }
                }
            }
        })
    }

    fn policy(proposal_ttl_ns: u64) -> HeadGazePolicyV1 {
        HeadGazePolicyV1::parse_json(
            &serde_json::to_vec(&policy_value(proposal_ttl_ns)).expect("serialize test policy"),
        )
        .expect("valid test policy")
    }

    fn projection(fx: f32, layout: ImageLayout) -> RgbFacePinholeProjection {
        RgbFacePinholeProjection::new(
            PinholeIntrinsics::try_new(fx, 400.0, 319.5, 199.5).expect("valid test intrinsics"),
            layout,
        )
    }

    fn adapter(policy: &HeadGazePolicyV1) -> HeadGazeFaceProposalAdapter<'_> {
        HeadGazeFaceProposalAdapter::try_new(policy, projection(400.0, layout(640, 400)))
            .expect("test policy has bounded return trigger")
    }

    #[test]
    fn only_fresh_current_tracked_face_produces_a_non_command_proposal() {
        let policy = policy(150_000_000);
        let adapter = adapter(&policy);
        assert_eq!(
            adapter.return_trigger_bound().maximum_delay(),
            Duration::from_millis(150)
        );

        let target = face(280, 160, 80, 80);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        let acquiring = update(&mut tracker, 1, &[target]);
        assert_eq!(
            adapter
                .evaluate(
                    acquiring,
                    MonotonicTimestamp::from_nanos_since_epoch(RESULT_PERIOD_NS)
                )
                .expect("withholding cannot fail"),
            HeadGazeFaceProposalOutcome::Withheld(HeadGazeFaceProposalWithheld::Acquiring)
        );

        let tracked = update(&mut tracker, 2, &[target]);
        let HeadGazeFaceProposalOutcome::Proposed(proposal) = adapter
            .evaluate(
                tracked,
                MonotonicTimestamp::from_nanos_since_epoch(2 * RESULT_PERIOD_NS),
            )
            .expect("center face maps")
        else {
            panic!("fresh tracked face must propose");
        };
        assert_eq!(
            proposal.face().transition(),
            FreshFaceGazeTransition::Tracked
        );
        assert_eq!(
            proposal.target().position(HeadJoint::Yaw).get(),
            policy
                .mapping()
                .natural_declaration()
                .position(HeadJoint::Yaw)
                .get()
        );
        assert_eq!(
            proposal.target().position(HeadJoint::Roll).get(),
            policy
                .mapping()
                .natural_declaration()
                .position(HeadJoint::Roll)
                .get()
        );
    }

    #[test]
    fn no_target_coasting_and_lost_never_reuse_historical_face_coordinates() {
        let policy = policy(150_000_000);
        let adapter = adapter(&policy);
        let target = face(280, 160, 80, 80);

        let mut empty_tracker = FaceTracker::new(FaceTrackingConfig::default());
        let no_target = update(&mut empty_tracker, 1, &[]);
        assert_eq!(
            adapter
                .evaluate(
                    no_target,
                    MonotonicTimestamp::from_nanos_since_epoch(RESULT_PERIOD_NS)
                )
                .unwrap(),
            HeadGazeFaceProposalOutcome::Withheld(HeadGazeFaceProposalWithheld::NoTarget)
        );

        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        let _ = update(&mut tracker, 1, &[target]);
        let _ = update(&mut tracker, 2, &[target]);
        let coasting = update(&mut tracker, 3, &[]);
        assert_eq!(
            adapter
                .evaluate(
                    coasting,
                    MonotonicTimestamp::from_nanos_since_epoch(3 * RESULT_PERIOD_NS)
                )
                .unwrap(),
            HeadGazeFaceProposalOutcome::Withheld(HeadGazeFaceProposalWithheld::Coasting)
        );

        let lost_at = 2_300_000_000;
        let lost = update_at(&mut tracker, 4, lost_at, lost_at, &[]);
        assert_eq!(
            adapter
                .evaluate(lost, MonotonicTimestamp::from_nanos_since_epoch(lost_at))
                .unwrap(),
            HeadGazeFaceProposalOutcome::Withheld(HeadGazeFaceProposalWithheld::Lost)
        );
    }

    #[test]
    fn current_tracked_face_is_withheld_at_its_exclusive_freshness_deadline() {
        let policy = policy(150_000_000);
        let adapter = adapter(&policy);
        let target = face(280, 160, 80, 80);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        let _ = update(&mut tracker, 1, &[target]);
        let tracked = update(&mut tracker, 2, &[target]);
        let observed_at = 2 * RESULT_PERIOD_NS;
        let deadline = observed_at + 150_000_000;

        assert_eq!(
            adapter
                .evaluate(
                    tracked,
                    MonotonicTimestamp::from_nanos_since_epoch(deadline)
                )
                .unwrap(),
            HeadGazeFaceProposalOutcome::Withheld(HeadGazeFaceProposalWithheld::TrackedNotFresh {
                observed_at: MonotonicTimestamp::from_nanos_since_epoch(observed_at),
                valid_until_exclusive: MonotonicTimestamp::from_nanos_since_epoch(deadline),
                evaluated_at: MonotonicTimestamp::from_nanos_since_epoch(deadline),
            })
        );
    }

    #[test]
    fn fresh_switched_face_proposes_from_new_track_only() {
        let policy = policy(150_000_000);
        let adapter = adapter(&policy);
        let current = face(80, 150, 80, 80);
        let closer = face(390, 130, 160, 120);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        let _ = update(&mut tracker, 1, &[current]);
        let tracked = update(&mut tracker, 2, &[current]);
        let original_track = match tracked.state() {
            FaceTargetState::Tracked(observation) => observation.track_id(),
            other => panic!("expected tracked state, got {other:?}"),
        };
        for sequence in 3..7 {
            let state = update(&mut tracker, sequence, &[closer, current]);
            assert!(matches!(state.state(), FaceTargetState::Tracked(_)));
        }
        let switched = update(&mut tracker, 7, &[closer, current]);

        let HeadGazeFaceProposalOutcome::Proposed(proposal) = adapter
            .evaluate(
                switched,
                MonotonicTimestamp::from_nanos_since_epoch(7 * RESULT_PERIOD_NS),
            )
            .expect("switched face maps")
        else {
            panic!("fresh switched face must propose");
        };
        assert_eq!(
            proposal.face().transition(),
            FreshFaceGazeTransition::Switched {
                previous_track_id: original_track
            }
        );
        assert_ne!(proposal.face().track_id(), original_track);
    }

    #[test]
    fn policy_above_four_hundred_milliseconds_is_rejected_before_use() {
        let policy = policy(400_000_001);
        assert!(matches!(
            HeadGazeFaceProposalAdapter::try_new(
                &policy,
                projection(400.0, layout(640, 400))
            ),
            Err(
                HeadGazeFaceProposalAdapterError::ReturnTriggerDelayAboveMaximum {
                    actual,
                    maximum
                }
            ) if actual == Duration::from_nanos(400_000_001)
                && maximum == MAXIMUM_HEAD_GAZE_RETURN_TRIGGER_DELAY
        ));
    }

    #[test]
    fn projection_grid_mismatch_and_out_of_envelope_mapping_are_not_clamped() {
        let policy = policy(150_000_000);
        let target = face(560, 160, 80, 80);
        let mut tracker = FaceTracker::new(FaceTrackingConfig::default());
        let _ = update(&mut tracker, 1, &[target]);
        let tracked = update(&mut tracker, 2, &[target]);
        let evaluated_at = MonotonicTimestamp::from_nanos_since_epoch(2 * RESULT_PERIOD_NS);

        let wrong_grid =
            HeadGazeFaceProposalAdapter::try_new(&policy, projection(400.0, layout(320, 200)))
                .unwrap();
        assert!(matches!(
            wrong_grid.evaluate(tracked, evaluated_at),
            Err(HeadGazeFaceProposalError::ProjectionLayoutMismatch { .. })
        ));

        let extreme_projection =
            HeadGazeFaceProposalAdapter::try_new(&policy, projection(100.0, layout(640, 400)))
                .unwrap();
        assert!(matches!(
            extreme_projection.evaluate(tracked, evaluated_at),
            Err(HeadGazeFaceProposalError::Mapping(_))
        ));
    }
}
