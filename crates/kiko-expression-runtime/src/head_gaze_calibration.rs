//! Typed Kiko camera-gaze to physical-head mapping declarations.
//!
//! The weak input names every physical quantity and every joint. Parsing binds
//! it to Kiko's declared camera/head geometry, a fixed camera-forward focus
//! plane, exact natural encoder positions, hard encoder envelopes, and a
//! explicit sparse mapping shape:
//!
//! - `pitch_down` drives bow and curl;
//! - `yaw_right` drives yaw; and
//! - roll never moves for gaze.
//!
//! The retained per-joint conversions reuse
//! [`kiko_head_protocol::JointCalibration`], including its checked,
//! deterministic nearest-encoder-tick conversion. Proposal generation rejects
//! gaze outside the hard envelopes; it never clamps or silently substitutes
//! the natural pose.
//!
//! Parsing proves only schema and numerical domain validity. It does not prove
//! that the declared assembly or provenance identifiers refer to the connected
//! hardware, that the coefficients were physically measured, or that this
//! calibration has been reviewed or activated for motion.

extern crate alloc;

use alloc::boxed::Box;
use core::fmt;

use kiko_head_protocol::{
    AngleRadians, FrameBuildError, HeadJoint, JointCalibration, JointCalibrationError,
    JointDirection, JointLimitsRadians, PositionTicks,
};

use crate::{
    CameraForwardDepthMeters, CameraGazeTargetError, CameraToHeadGazeExtrinsics,
    CameraToHeadGazeExtrinsicsInput, GazeExtrinsicsParseError, HeadRelativeGaze,
    OakCameraTargetRay, RayHeadGazeProjectionError,
};

/// Declared Kiko head-centre origin in OAK coordinates, in metres.
///
/// OAK `+y` points image-down, so `-0.25 m` means the head centre is `0.25 m`
/// above the camera. OAK `+z` points forward, so `-0.20 m` means the head
/// centre is `0.20 m` behind the camera.
pub const DECLARED_HEAD_ORIGIN_IN_OAK_CAMERA_M: [f64; 3] = [0.0, -0.25, -0.20];

/// Declared neutral-head-from-camera rotation: parallel axes.
pub const DECLARED_NEUTRAL_HEAD_FROM_OAK_ROTATION_ROWS: [[f64; 3]; 3] =
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Explicit policy depth used when a two-dimensional camera observation
/// supplies a ray but no measured range.
///
/// This is an assumed camera-forward focus plane, not observed person depth.
pub const HEAD_GAZE_FOCUS_PLANE_CAMERA_FORWARD_DEPTH_M: f64 = 1.5;

/// Largest retained assembly or calibration-provenance identifier.
pub const MAX_HEAD_GAZE_IDENTIFIER_BYTES: usize = 96;

/// Weak, explicitly ordered natural encoder positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedNaturalHeadTicksInput {
    pub bow_ticks: u16,
    pub curl_ticks: u16,
    pub yaw_ticks: u16,
    pub roll_ticks: u16,
}

impl NamedNaturalHeadTicksInput {
    const fn for_joint(self, joint: HeadJoint) -> u16 {
        match joint {
            HeadJoint::Bow => self.bow_ticks,
            HeadJoint::Curl => self.curl_ticks,
            HeadJoint::Yaw => self.yaw_ticks,
            HeadJoint::Roll => self.roll_ticks,
        }
    }
}

/// Parsed natural-position declaration.
///
/// This is neither an observed pose nor a reviewed command target. It has no
/// conversion into `kiko_head_protocol::ExactHeadTargetPose`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadNaturalPoseDeclaration {
    positions: [PositionTicks; 4],
}

impl HeadNaturalPoseDeclaration {
    pub const fn position(self, joint: HeadJoint) -> PositionTicks {
        self.positions[joint_index(joint)]
    }

    pub const fn positions(self) -> [PositionTicks; 4] {
        self.positions
    }
}

/// Numerically valid, non-command gaze-position proposal.
///
/// A deployment-specific activation boundary must separately bind physical
/// evidence and construct any actuator command. This crate intentionally
/// provides no conversion into `kiko_head_protocol::ExactHeadTargetPose`.
///
/// ```compile_fail
/// use kiko_expression_runtime::HeadGazeTargetProposal;
/// use kiko_head_protocol::ExactHeadTargetPose;
///
/// fn incorrectly_activate(proposal: HeadGazeTargetProposal) -> ExactHeadTargetPose {
///     proposal.into()
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadGazeTargetProposal {
    positions: [PositionTicks; 4],
}

impl HeadGazeTargetProposal {
    const fn from_positions(
        bow: PositionTicks,
        curl: PositionTicks,
        yaw: PositionTicks,
        roll: PositionTicks,
    ) -> Self {
        Self {
            positions: [bow, curl, yaw, roll],
        }
    }

    pub const fn position(self, joint: HeadJoint) -> PositionTicks {
        self.positions[joint_index(joint)]
    }

    pub const fn positions(self) -> [PositionTicks; 4] {
        self.positions
    }
}

/// Inclusive absolute encoder envelope for one named joint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadTickEnvelopeInput {
    pub minimum_ticks: u16,
    pub maximum_ticks: u16,
}

/// Weak, named hard envelopes. No array-order convention crosses this
/// boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NamedHeadTickEnvelopesInput {
    pub bow: HeadTickEnvelopeInput,
    pub curl: HeadTickEnvelopeInput,
    pub yaw: HeadTickEnvelopeInput,
    pub roll: HeadTickEnvelopeInput,
}

impl NamedHeadTickEnvelopesInput {
    const fn for_joint(self, joint: HeadJoint) -> HeadTickEnvelopeInput {
        match joint {
            HeadJoint::Bow => self.bow,
            HeadJoint::Curl => self.curl,
            HeadJoint::Yaw => self.yaw,
            HeadJoint::Roll => self.roll,
        }
    }
}

/// Signed encoder-tick offset per radian, explicitly named for every joint.
///
/// A positive value means that a positive gaze angle increases the absolute
/// encoder position. Zero means that the gaze coordinate must not drive that
/// joint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NamedHeadTickOffsetsPerRadianInput {
    pub bow_ticks_per_radian: f64,
    pub curl_ticks_per_radian: f64,
    pub yaw_ticks_per_radian: f64,
    pub roll_ticks_per_radian: f64,
}

impl NamedHeadTickOffsetsPerRadianInput {
    const fn for_joint(self, joint: HeadJoint) -> f64 {
        match joint {
            HeadJoint::Bow => self.bow_ticks_per_radian,
            HeadJoint::Curl => self.curl_ticks_per_radian,
            HeadJoint::Yaw => self.yaw_ticks_per_radian,
            HeadJoint::Roll => self.roll_ticks_per_radian,
        }
    }
}

/// Weak two-column physical gaze mapping.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadGazeTickOffsetsPerRadianInput {
    pub pitch_down: NamedHeadTickOffsetsPerRadianInput,
    pub yaw_right: NamedHeadTickOffsetsPerRadianInput,
}

impl HeadGazeTickOffsetsPerRadianInput {
    const fn coefficient(self, coordinate: HeadGazeCoordinate, joint: HeadJoint) -> f64 {
        match coordinate {
            HeadGazeCoordinate::PitchDown => self.pitch_down.for_joint(joint),
            HeadGazeCoordinate::YawRight => self.yaw_right.for_joint(joint),
        }
    }
}

/// One weak, unreviewed mapping-declaration boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadGazeMappingDeclarationInput<'a> {
    pub assembly_id: &'a str,
    pub calibration_provenance_id: &'a str,
    pub focus_plane_camera_forward_depth_m: f64,
    pub camera_to_head: CameraToHeadGazeExtrinsicsInput,
    pub natural: NamedNaturalHeadTicksInput,
    pub hard_envelopes: NamedHeadTickEnvelopesInput,
    pub tick_offsets_per_radian: HeadGazeTickOffsetsPerRadianInput,
}

/// Bounded identifier declared for a physical assembly.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HeadAssemblyId(Box<str>);

impl HeadAssemblyId {
    fn parse(value: &str) -> Result<Self, HeadGazeIdentifierError> {
        parse_identifier(value)?;
        Ok(Self(Box::from(value)))
    }

    pub fn get(&self) -> &str {
        &self.0
    }
}

/// Bounded identifier declared for calibration provenance.
///
/// Successful parsing does not attest that the named evidence exists or was
/// physically qualified.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HeadCalibrationProvenanceId(Box<str>);

impl HeadCalibrationProvenanceId {
    fn parse(value: &str) -> Result<Self, HeadGazeIdentifierError> {
        parse_identifier(value)?;
        Ok(Self(Box::from(value)))
    }

    pub fn get(&self) -> &str {
        &self.0
    }
}

fn parse_identifier(value: &str) -> Result<(), HeadGazeIdentifierError> {
    if value.is_empty() {
        return Err(HeadGazeIdentifierError::Empty);
    }
    if value.len() > MAX_HEAD_GAZE_IDENTIFIER_BYTES {
        return Err(HeadGazeIdentifierError::TooLong {
            actual_bytes: value.len(),
            maximum_bytes: MAX_HEAD_GAZE_IDENTIFIER_BYTES,
        });
    }
    for (index, byte) in value.bytes().enumerate() {
        if !matches!(
            byte,
            b'a'..=b'z'
                | b'A'..=b'Z'
                | b'0'..=b'9'
                | b'-'
                | b'_'
                | b'.'
                | b':'
                | b'/'
                | b'@'
                | b'+'
        ) {
            return Err(HeadGazeIdentifierError::InvalidByte { index, byte });
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeIdentifierError {
    Empty,
    TooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    InvalidByte {
        index: usize,
        byte: u8,
    },
}

impl fmt::Display for HeadGazeIdentifierError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid head-gaze identifier: {self:?}")
    }
}

impl core::error::Error for HeadGazeIdentifierError {}

/// Inclusive, already parsed absolute encoder envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadTickEnvelope {
    minimum: PositionTicks,
    maximum: PositionTicks,
}

impl HeadTickEnvelope {
    pub const fn minimum(self) -> PositionTicks {
        self.minimum
    }

    pub const fn maximum(self) -> PositionTicks {
        self.maximum
    }

    pub const fn contains(self, position: PositionTicks) -> bool {
        position.get() >= self.minimum.get() && position.get() <= self.maximum.get()
    }
}

/// The only two admitted head-relative gaze coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HeadGazeCoordinate {
    PitchDown,
    YawRight,
}

/// Which identifier failed to parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HeadGazeIdentifierField {
    AssemblyId,
    CalibrationProvenanceId,
}

/// Which hard-envelope endpoint failed to parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HeadTickEnvelopeBound {
    Minimum,
    Maximum,
}

/// Parsed camera-gaze mapping with owned identifiers and numeric invariants.
///
/// This value is not physical calibration evidence or motion authority.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadGazeMappingDeclaration {
    assembly_id: HeadAssemblyId,
    calibration_provenance_id: HeadCalibrationProvenanceId,
    focus_plane: CameraForwardDepthMeters,
    camera_to_head: CameraToHeadGazeExtrinsics,
    natural: HeadNaturalPoseDeclaration,
    hard_envelopes: [HeadTickEnvelope; 4],
    bow_pitch: JointCalibration,
    curl_pitch: JointCalibration,
    yaw_right: JointCalibration,
    tick_offsets_per_radian: HeadGazeTickOffsetsPerRadianInput,
}

impl HeadGazeMappingDeclaration {
    /// Parse all weak metadata and numeric fields exactly once.
    ///
    /// Success establishes domain validity only. A separate deployment gate
    /// must bind the IDs to retained evidence and explicitly activate motion.
    pub fn parse(
        input: HeadGazeMappingDeclarationInput<'_>,
    ) -> Result<Self, HeadGazeMappingDeclarationParseError> {
        let assembly_id = HeadAssemblyId::parse(input.assembly_id).map_err(|source| {
            HeadGazeMappingDeclarationParseError::Identifier {
                field: HeadGazeIdentifierField::AssemblyId,
                source,
            }
        })?;
        let calibration_provenance_id = HeadCalibrationProvenanceId::parse(
            input.calibration_provenance_id,
        )
        .map_err(|source| HeadGazeMappingDeclarationParseError::Identifier {
            field: HeadGazeIdentifierField::CalibrationProvenanceId,
            source,
        })?;

        let focus_plane = CameraForwardDepthMeters::parse(input.focus_plane_camera_forward_depth_m)
            .map_err(HeadGazeMappingDeclarationParseError::FocusPlane)?;
        if focus_plane.get().to_bits() != HEAD_GAZE_FOCUS_PLANE_CAMERA_FORWARD_DEPTH_M.to_bits() {
            return Err(
                HeadGazeMappingDeclarationParseError::FocusPlaneDoesNotMatchPolicy {
                    actual_m: focus_plane.get(),
                    required_m: HEAD_GAZE_FOCUS_PLANE_CAMERA_FORWARD_DEPTH_M,
                },
            );
        }

        let camera_to_head = CameraToHeadGazeExtrinsics::parse(input.camera_to_head)
            .map_err(HeadGazeMappingDeclarationParseError::CameraToHead)?;
        if camera_to_head.head_origin_in_camera_m() != DECLARED_HEAD_ORIGIN_IN_OAK_CAMERA_M {
            return Err(
                HeadGazeMappingDeclarationParseError::HeadOriginDoesNotMatchDeclaration {
                    actual_m: camera_to_head.head_origin_in_camera_m(),
                    required_m: DECLARED_HEAD_ORIGIN_IN_OAK_CAMERA_M,
                },
            );
        }
        let actual_rotation = camera_to_head.neutral_head_from_camera_rotation_rows();
        for row in 0..3 {
            for column in 0..3 {
                let actual = actual_rotation[row][column];
                let required = DECLARED_NEUTRAL_HEAD_FROM_OAK_ROTATION_ROWS[row][column];
                if actual != required {
                    return Err(
                        HeadGazeMappingDeclarationParseError::NeutralRotationDoesNotMatchDeclaration {
                            row,
                            column,
                            actual,
                            required,
                        },
                    );
                }
            }
        }

        let natural = parse_natural_declaration(input.natural)?;
        let hard_envelopes = parse_envelopes(input.hard_envelopes, natural)?;
        validate_sparse_mapping(input.tick_offsets_per_radian)?;

        let bow_pitch = joint_calibration(
            HeadGazeCoordinate::PitchDown,
            HeadJoint::Bow,
            natural.position(HeadJoint::Bow),
            hard_envelopes[joint_index(HeadJoint::Bow)],
            input
                .tick_offsets_per_radian
                .coefficient(HeadGazeCoordinate::PitchDown, HeadJoint::Bow),
        )?;
        let curl_pitch = joint_calibration(
            HeadGazeCoordinate::PitchDown,
            HeadJoint::Curl,
            natural.position(HeadJoint::Curl),
            hard_envelopes[joint_index(HeadJoint::Curl)],
            input
                .tick_offsets_per_radian
                .coefficient(HeadGazeCoordinate::PitchDown, HeadJoint::Curl),
        )?;
        let yaw_right = joint_calibration(
            HeadGazeCoordinate::YawRight,
            HeadJoint::Yaw,
            natural.position(HeadJoint::Yaw),
            hard_envelopes[joint_index(HeadJoint::Yaw)],
            input
                .tick_offsets_per_radian
                .coefficient(HeadGazeCoordinate::YawRight, HeadJoint::Yaw),
        )?;

        Ok(Self {
            assembly_id,
            calibration_provenance_id,
            focus_plane,
            camera_to_head,
            natural,
            hard_envelopes,
            bow_pitch,
            curl_pitch,
            yaw_right,
            tick_offsets_per_radian: input.tick_offsets_per_radian,
        })
    }

    pub const fn assembly_id(&self) -> &HeadAssemblyId {
        &self.assembly_id
    }

    pub const fn calibration_provenance_id(&self) -> &HeadCalibrationProvenanceId {
        &self.calibration_provenance_id
    }

    pub const fn focus_plane(&self) -> CameraForwardDepthMeters {
        self.focus_plane
    }

    pub const fn camera_to_head(&self) -> CameraToHeadGazeExtrinsics {
        self.camera_to_head
    }

    pub const fn natural_declaration(&self) -> HeadNaturalPoseDeclaration {
        self.natural
    }

    pub const fn hard_envelope(&self, joint: HeadJoint) -> HeadTickEnvelope {
        self.hard_envelopes[joint_index(joint)]
    }

    pub const fn tick_offset_per_radian(
        &self,
        coordinate: HeadGazeCoordinate,
        joint: HeadJoint,
    ) -> f64 {
        self.tick_offsets_per_radian.coefficient(coordinate, joint)
    }

    /// Produce one non-command proposal from projected gaze without clamping.
    pub fn proposal_for_gaze(
        &self,
        gaze: HeadRelativeGaze,
    ) -> Result<HeadGazeTargetProposal, HeadGazeProposalMappingError> {
        let pitch = gaze.pitch_down_angle();
        let yaw = gaze.yaw_right_angle();
        let bow = map_joint(
            self.bow_pitch,
            HeadGazeCoordinate::PitchDown,
            pitch,
            self.hard_envelope(HeadJoint::Bow),
        )?;
        let curl = map_joint(
            self.curl_pitch,
            HeadGazeCoordinate::PitchDown,
            pitch,
            self.hard_envelope(HeadJoint::Curl),
        )?;
        let yaw = map_joint(
            self.yaw_right,
            HeadGazeCoordinate::YawRight,
            yaw,
            self.hard_envelope(HeadJoint::Yaw),
        )?;
        let roll = self.natural.position(HeadJoint::Roll);
        debug_assert!(self.hard_envelope(HeadJoint::Roll).contains(roll));
        Ok(HeadGazeTargetProposal::from_positions(bow, curl, yaw, roll))
    }

    /// Intersect one camera ray with the fixed `1.5 m` policy plane, project
    /// it through the bound assembly extrinsic, then map it to encoder ticks.
    pub fn proposal_for_camera_ray(
        &self,
        ray: OakCameraTargetRay,
    ) -> Result<HeadGazeTargetProposal, CameraRayHeadProposalError> {
        let gaze = self
            .camera_to_head
            .project_ray_at_forward_depth(ray, self.focus_plane)
            .map_err(CameraRayHeadProposalError::Projection)?;
        self.proposal_for_gaze(gaze)
            .map_err(CameraRayHeadProposalError::Mapping)
    }
}

const fn joint_index(joint: HeadJoint) -> usize {
    match joint {
        HeadJoint::Bow => 0,
        HeadJoint::Curl => 1,
        HeadJoint::Yaw => 2,
        HeadJoint::Roll => 3,
    }
}

fn parse_natural_declaration(
    input: NamedNaturalHeadTicksInput,
) -> Result<HeadNaturalPoseDeclaration, HeadGazeMappingDeclarationParseError> {
    let mut positions = [PositionTicks::MIN; 4];
    for joint in HeadJoint::ALL {
        let value = input.for_joint(joint);
        positions[joint_index(joint)] = PositionTicks::try_new(value).map_err(|source| {
            HeadGazeMappingDeclarationParseError::NaturalPosition {
                joint,
                value,
                source,
            }
        })?;
    }
    Ok(HeadNaturalPoseDeclaration { positions })
}

fn parse_envelopes(
    input: NamedHeadTickEnvelopesInput,
    natural: HeadNaturalPoseDeclaration,
) -> Result<[HeadTickEnvelope; 4], HeadGazeMappingDeclarationParseError> {
    let mut parsed = [HeadTickEnvelope {
        minimum: PositionTicks::MIN,
        maximum: PositionTicks::MIN,
    }; 4];
    for joint in HeadJoint::ALL {
        let raw = input.for_joint(joint);
        let minimum = PositionTicks::try_new(raw.minimum_ticks).map_err(|source| {
            HeadGazeMappingDeclarationParseError::HardEnvelopePosition {
                joint,
                bound: HeadTickEnvelopeBound::Minimum,
                value: raw.minimum_ticks,
                source,
            }
        })?;
        let maximum = PositionTicks::try_new(raw.maximum_ticks).map_err(|source| {
            HeadGazeMappingDeclarationParseError::HardEnvelopePosition {
                joint,
                bound: HeadTickEnvelopeBound::Maximum,
                value: raw.maximum_ticks,
                source,
            }
        })?;
        if minimum > maximum {
            return Err(HeadGazeMappingDeclarationParseError::HardEnvelopeReversed {
                joint,
                minimum,
                maximum,
            });
        }
        let natural_position = natural.position(joint);
        let envelope = HeadTickEnvelope { minimum, maximum };
        if !envelope.contains(natural_position) {
            return Err(
                HeadGazeMappingDeclarationParseError::NaturalOutsideHardEnvelope {
                    joint,
                    natural: natural_position,
                    minimum,
                    maximum,
                },
            );
        }
        if joint != HeadJoint::Roll && (natural_position == minimum || natural_position == maximum)
        {
            return Err(
                HeadGazeMappingDeclarationParseError::ActiveJointLacksBidirectionalTravel {
                    joint,
                    natural: natural_position,
                    minimum,
                    maximum,
                },
            );
        }
        parsed[joint_index(joint)] = envelope;
    }
    Ok(parsed)
}

fn validate_sparse_mapping(
    mapping: HeadGazeTickOffsetsPerRadianInput,
) -> Result<(), HeadGazeMappingDeclarationParseError> {
    for coordinate in [HeadGazeCoordinate::PitchDown, HeadGazeCoordinate::YawRight] {
        for joint in HeadJoint::ALL {
            let value = mapping.coefficient(coordinate, joint);
            if !value.is_finite() {
                return Err(
                    HeadGazeMappingDeclarationParseError::NonFiniteTickOffsetPerRadian {
                        coordinate,
                        joint,
                        value,
                    },
                );
            }
            let active = matches!(
                (coordinate, joint),
                (HeadGazeCoordinate::PitchDown, HeadJoint::Bow)
                    | (HeadGazeCoordinate::PitchDown, HeadJoint::Curl)
                    | (HeadGazeCoordinate::YawRight, HeadJoint::Yaw)
            );
            if active {
                if value == 0.0 {
                    return Err(
                        HeadGazeMappingDeclarationParseError::MissingActiveTickOffsetPerRadian {
                            coordinate,
                            joint,
                        },
                    );
                }
            } else if value != 0.0 {
                return Err(
                    HeadGazeMappingDeclarationParseError::AmbiguousCrossAxisMapping {
                        coordinate,
                        joint,
                        value,
                    },
                );
            }
        }
    }
    Ok(())
}

fn joint_calibration(
    coordinate: HeadGazeCoordinate,
    joint: HeadJoint,
    natural: PositionTicks,
    envelope: HeadTickEnvelope,
    signed_ticks_per_radian: f64,
) -> Result<JointCalibration, HeadGazeMappingDeclarationParseError> {
    let direction = if signed_ticks_per_radian.is_sign_negative() {
        JointDirection::Negative
    } else {
        JointDirection::Positive
    };
    let ticks_per_radian = signed_ticks_per_radian.abs();
    let lower_tick_angle =
        (f64::from(envelope.minimum.get()) - f64::from(natural.get())) / signed_ticks_per_radian;
    let upper_tick_angle =
        (f64::from(envelope.maximum.get()) - f64::from(natural.get())) / signed_ticks_per_radian;
    let minimum_angle = lower_tick_angle.min(upper_tick_angle);
    let maximum_angle = lower_tick_angle.max(upper_tick_angle);
    let limits = JointLimitsRadians::try_new(minimum_angle, maximum_angle).map_err(|source| {
        HeadGazeMappingDeclarationParseError::JointCalibration {
            coordinate,
            joint,
            source,
        }
    })?;
    JointCalibration::try_new(joint, natural, ticks_per_radian, direction, limits).map_err(
        |source| HeadGazeMappingDeclarationParseError::JointCalibration {
            coordinate,
            joint,
            source,
        },
    )
}

fn map_joint(
    calibration: JointCalibration,
    coordinate: HeadGazeCoordinate,
    angle: AngleRadians,
    envelope: HeadTickEnvelope,
) -> Result<PositionTicks, HeadGazeProposalMappingError> {
    let joint = calibration.joint();
    let position = calibration.position_for_angle(angle).map_err(|source| {
        HeadGazeProposalMappingError::Joint {
            coordinate,
            joint,
            source,
        }
    })?;
    if !envelope.contains(position) {
        return Err(HeadGazeProposalMappingError::MappedOutsideHardEnvelope {
            coordinate,
            joint,
            position,
            minimum: envelope.minimum(),
            maximum: envelope.maximum(),
        });
    }
    Ok(position)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadGazeMappingDeclarationParseError {
    Identifier {
        field: HeadGazeIdentifierField,
        source: HeadGazeIdentifierError,
    },
    FocusPlane(CameraGazeTargetError),
    FocusPlaneDoesNotMatchPolicy {
        actual_m: f64,
        required_m: f64,
    },
    CameraToHead(GazeExtrinsicsParseError),
    HeadOriginDoesNotMatchDeclaration {
        actual_m: [f64; 3],
        required_m: [f64; 3],
    },
    NeutralRotationDoesNotMatchDeclaration {
        row: usize,
        column: usize,
        actual: f64,
        required: f64,
    },
    NaturalPosition {
        joint: HeadJoint,
        value: u16,
        source: FrameBuildError,
    },
    HardEnvelopePosition {
        joint: HeadJoint,
        bound: HeadTickEnvelopeBound,
        value: u16,
        source: FrameBuildError,
    },
    HardEnvelopeReversed {
        joint: HeadJoint,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
    NaturalOutsideHardEnvelope {
        joint: HeadJoint,
        natural: PositionTicks,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
    ActiveJointLacksBidirectionalTravel {
        joint: HeadJoint,
        natural: PositionTicks,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
    NonFiniteTickOffsetPerRadian {
        coordinate: HeadGazeCoordinate,
        joint: HeadJoint,
        value: f64,
    },
    MissingActiveTickOffsetPerRadian {
        coordinate: HeadGazeCoordinate,
        joint: HeadJoint,
    },
    AmbiguousCrossAxisMapping {
        coordinate: HeadGazeCoordinate,
        joint: HeadJoint,
        value: f64,
    },
    JointCalibration {
        coordinate: HeadGazeCoordinate,
        joint: HeadJoint,
        source: JointCalibrationError,
    },
}

impl fmt::Display for HeadGazeMappingDeclarationParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Kiko head-gaze mapping declaration: {self:?}"
        )
    }
}

impl core::error::Error for HeadGazeMappingDeclarationParseError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Identifier { source, .. } => Some(source),
            Self::FocusPlane(source) => Some(source),
            Self::CameraToHead(source) => Some(source),
            Self::NaturalPosition { source, .. } => Some(source),
            Self::HardEnvelopePosition { source, .. } => Some(source),
            Self::JointCalibration { source, .. } => Some(source),
            Self::FocusPlaneDoesNotMatchPolicy { .. }
            | Self::HeadOriginDoesNotMatchDeclaration { .. }
            | Self::NeutralRotationDoesNotMatchDeclaration { .. }
            | Self::HardEnvelopeReversed { .. }
            | Self::NaturalOutsideHardEnvelope { .. }
            | Self::ActiveJointLacksBidirectionalTravel { .. }
            | Self::NonFiniteTickOffsetPerRadian { .. }
            | Self::MissingActiveTickOffsetPerRadian { .. }
            | Self::AmbiguousCrossAxisMapping { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadGazeProposalMappingError {
    Joint {
        coordinate: HeadGazeCoordinate,
        joint: HeadJoint,
        source: JointCalibrationError,
    },
    MappedOutsideHardEnvelope {
        coordinate: HeadGazeCoordinate,
        joint: HeadJoint,
        position: PositionTicks,
        minimum: PositionTicks,
        maximum: PositionTicks,
    },
}

impl fmt::Display for HeadGazeProposalMappingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot map head-relative gaze to a position proposal: {self:?}"
        )
    }
}

impl core::error::Error for HeadGazeProposalMappingError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Joint { source, .. } => Some(source),
            Self::MappedOutsideHardEnvelope { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CameraRayHeadProposalError {
    Projection(RayHeadGazeProjectionError),
    Mapping(HeadGazeProposalMappingError),
}

impl fmt::Display for CameraRayHeadProposalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "cannot map OAK camera ray to a Kiko head proposal: {self:?}"
        )
    }
}

impl core::error::Error for CameraRayHeadProposalError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Projection(source) => Some(source),
            Self::Mapping(source) => Some(source),
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use libm::{cos, sin};
    use std::string::String;

    use super::*;
    use crate::OakCameraTargetPoint;

    fn valid_input() -> HeadGazeMappingDeclarationInput<'static> {
        HeadGazeMappingDeclarationInput {
            assembly_id: "kiko-head:demo-rig@2026-07-27",
            calibration_provenance_id: "sha256:0123456789abcdef",
            focus_plane_camera_forward_depth_m: HEAD_GAZE_FOCUS_PLANE_CAMERA_FORWARD_DEPTH_M,
            camera_to_head: CameraToHeadGazeExtrinsicsInput {
                head_origin_in_camera_m: DECLARED_HEAD_ORIGIN_IN_OAK_CAMERA_M,
                neutral_head_from_camera_quaternion_xyzw: [0.0, 0.0, 0.0, 1.0],
            },
            natural: NamedNaturalHeadTicksInput {
                bow_ticks: 2_155,
                curl_ticks: 2_545,
                yaw_ticks: 2_943,
                roll_ticks: 2_876,
            },
            hard_envelopes: NamedHeadTickEnvelopesInput {
                bow: HeadTickEnvelopeInput {
                    minimum_ticks: 2_110,
                    maximum_ticks: 2_200,
                },
                curl: HeadTickEnvelopeInput {
                    minimum_ticks: 2_455,
                    maximum_ticks: 2_635,
                },
                yaw: HeadTickEnvelopeInput {
                    minimum_ticks: 2_683,
                    maximum_ticks: 3_203,
                },
                roll: HeadTickEnvelopeInput {
                    minimum_ticks: 2_876,
                    maximum_ticks: 2_876,
                },
            },
            tick_offsets_per_radian: HeadGazeTickOffsetsPerRadianInput {
                pitch_down: NamedHeadTickOffsetsPerRadianInput {
                    bow_ticks_per_radian: -100.0,
                    curl_ticks_per_radian: 200.0,
                    yaw_ticks_per_radian: 0.0,
                    roll_ticks_per_radian: 0.0,
                },
                yaw_right: NamedHeadTickOffsetsPerRadianInput {
                    bow_ticks_per_radian: 0.0,
                    curl_ticks_per_radian: 0.0,
                    yaw_ticks_per_radian: 300.0,
                    roll_ticks_per_radian: 0.0,
                },
            },
        }
    }

    fn gaze(yaw_right_rad: f64, pitch_down_rad: f64) -> HeadRelativeGaze {
        let distance_m = 2.0;
        let direction = [
            cos(pitch_down_rad) * sin(yaw_right_rad),
            sin(pitch_down_rad),
            cos(pitch_down_rad) * cos(yaw_right_rad),
        ];
        let target_m = [
            DECLARED_HEAD_ORIGIN_IN_OAK_CAMERA_M[0] + distance_m * direction[0],
            DECLARED_HEAD_ORIGIN_IN_OAK_CAMERA_M[1] + distance_m * direction[1],
            DECLARED_HEAD_ORIGIN_IN_OAK_CAMERA_M[2] + distance_m * direction[2],
        ];
        CameraToHeadGazeExtrinsics::parse(valid_input().camera_to_head)
            .unwrap()
            .project_point(OakCameraTargetPoint::parse(target_m).unwrap())
            .unwrap()
    }

    fn expected_tick(natural: u16, coefficient: f64, angle: f64) -> u16 {
        (f64::from(natural) + coefficient * angle + 0.5) as u16
    }

    #[test]
    fn required_metadata_is_explicitly_above_behind_parallel_and_at_1_5_metres() {
        let calibration = HeadGazeMappingDeclaration::parse(valid_input()).unwrap();
        assert_eq!(
            calibration.camera_to_head().head_origin_in_camera_m(),
            [0.0, -0.25, -0.20]
        );
        assert_eq!(
            calibration
                .camera_to_head()
                .neutral_head_from_camera_rotation_rows(),
            DECLARED_NEUTRAL_HEAD_FROM_OAK_ROTATION_ROWS
        );
        assert_eq!(calibration.focus_plane().get(), 1.5);
        assert_eq!(
            calibration.assembly_id().get(),
            "kiko-head:demo-rig@2026-07-27"
        );
        assert_eq!(
            calibration.calibration_provenance_id().get(),
            "sha256:0123456789abcdef"
        );

        for mismatched_origin in [[0.0, 0.25, -0.20], [0.0, -0.18, -0.15]] {
            let mut input = valid_input();
            input.camera_to_head.head_origin_in_camera_m = mismatched_origin;
            assert!(matches!(
                HeadGazeMappingDeclaration::parse(input),
                Err(HeadGazeMappingDeclarationParseError::HeadOriginDoesNotMatchDeclaration {
                    actual_m,
                    ..
                }) if actual_m == mismatched_origin
            ));
        }

        let mut input = valid_input();
        input
            .camera_to_head
            .neutral_head_from_camera_quaternion_xyzw = [
            0.0,
            0.0,
            core::f64::consts::FRAC_1_SQRT_2,
            core::f64::consts::FRAC_1_SQRT_2,
        ];
        assert!(matches!(
            HeadGazeMappingDeclaration::parse(input),
            Err(
                HeadGazeMappingDeclarationParseError::NeutralRotationDoesNotMatchDeclaration { .. }
            )
        ));

        let mut input = valid_input();
        input.focus_plane_camera_forward_depth_m = 2.0;
        assert!(matches!(
            HeadGazeMappingDeclaration::parse(input),
            Err(
                HeadGazeMappingDeclarationParseError::FocusPlaneDoesNotMatchPolicy {
                    actual_m: 2.0,
                    required_m: 1.5,
                }
            )
        ));
    }

    #[test]
    fn parsed_calibration_owns_manifest_identifiers() {
        let calibration = {
            let assembly_id = String::from("kiko-head:owned-assembly");
            let provenance_id = String::from("sha256:owned-calibration");
            let mut input = valid_input();
            input.assembly_id = &assembly_id;
            input.calibration_provenance_id = &provenance_id;
            HeadGazeMappingDeclaration::parse(input).unwrap()
        };

        assert_eq!(calibration.assembly_id().get(), "kiko-head:owned-assembly");
        assert_eq!(
            calibration.calibration_provenance_id().get(),
            "sha256:owned-calibration"
        );
    }

    #[test]
    fn mapping_signs_follow_pitch_down_and_yaw_right_contract() {
        let calibration = HeadGazeMappingDeclaration::parse(valid_input()).unwrap();
        let natural = calibration.natural_declaration();

        let right = calibration.proposal_for_gaze(gaze(0.2, 0.0)).unwrap();
        assert!(right.position(HeadJoint::Yaw) > natural.position(HeadJoint::Yaw));
        assert_eq!(
            right.position(HeadJoint::Bow),
            natural.position(HeadJoint::Bow)
        );
        assert_eq!(
            right.position(HeadJoint::Curl),
            natural.position(HeadJoint::Curl)
        );

        let down = calibration.proposal_for_gaze(gaze(0.0, 0.2)).unwrap();
        assert!(down.position(HeadJoint::Bow) < natural.position(HeadJoint::Bow));
        assert!(down.position(HeadJoint::Curl) > natural.position(HeadJoint::Curl));
        assert_eq!(
            down.position(HeadJoint::Yaw),
            natural.position(HeadJoint::Yaw)
        );
    }

    #[test]
    fn nearest_tick_rounding_is_stable_at_exact_half_ticks() {
        let mut input = valid_input();
        input
            .tick_offsets_per_radian
            .pitch_down
            .bow_ticks_per_radian = -2.0;
        input
            .tick_offsets_per_radian
            .pitch_down
            .curl_ticks_per_radian = 2.0;
        input.tick_offsets_per_radian.yaw_right.yaw_ticks_per_radian = 2.0;
        let calibration = HeadGazeMappingDeclaration::parse(input).unwrap();

        let proposal = calibration.proposal_for_gaze(gaze(0.25, 0.25)).unwrap();
        assert_eq!(proposal.position(HeadJoint::Bow).get(), 2_155);
        assert_eq!(proposal.position(HeadJoint::Curl).get(), 2_546);
        assert_eq!(proposal.position(HeadJoint::Yaw).get(), 2_944);
    }

    #[test]
    fn every_grid_proposal_respects_envelopes_and_never_moves_roll() {
        let calibration = HeadGazeMappingDeclaration::parse(valid_input()).unwrap();
        for yaw_step in -8..=8 {
            for pitch_step in -4..=4 {
                let yaw = f64::from(yaw_step) / 10.0;
                let pitch = f64::from(pitch_step) / 10.0;
                let proposal = calibration.proposal_for_gaze(gaze(yaw, pitch)).unwrap();
                for joint in HeadJoint::ALL {
                    assert!(
                        calibration
                            .hard_envelope(joint)
                            .contains(proposal.position(joint))
                    );
                }
                assert_eq!(
                    proposal.position(HeadJoint::Bow).get(),
                    expected_tick(2_155, -100.0, pitch)
                );
                assert_eq!(
                    proposal.position(HeadJoint::Curl).get(),
                    expected_tick(2_545, 200.0, pitch)
                );
                assert_eq!(
                    proposal.position(HeadJoint::Yaw).get(),
                    expected_tick(2_943, 300.0, yaw)
                );
                assert_eq!(proposal.position(HeadJoint::Roll).get(), 2_876);
            }
        }
    }

    #[test]
    fn out_of_envelope_gaze_is_rejected_without_clamping() {
        let calibration = HeadGazeMappingDeclaration::parse(valid_input()).unwrap();
        assert!(matches!(
            calibration.proposal_for_gaze(gaze(1.0, 0.0)),
            Err(HeadGazeProposalMappingError::Joint {
                coordinate: HeadGazeCoordinate::YawRight,
                joint: HeadJoint::Yaw,
                source: JointCalibrationError::AngleOutsideJointLimits { .. },
            })
        ));
        assert!(matches!(
            calibration.proposal_for_gaze(gaze(0.0, 0.6)),
            Err(HeadGazeProposalMappingError::Joint {
                coordinate: HeadGazeCoordinate::PitchDown,
                joint: HeadJoint::Bow,
                source: JointCalibrationError::AngleOutsideJointLimits { .. },
            })
        ));
    }

    #[test]
    fn malformed_envelopes_and_pose_are_rejected_once() {
        let mut input = valid_input();
        input.hard_envelopes.yaw.minimum_ticks = 4_096;
        assert!(matches!(
            HeadGazeMappingDeclaration::parse(input),
            Err(HeadGazeMappingDeclarationParseError::HardEnvelopePosition {
                joint: HeadJoint::Yaw,
                bound: HeadTickEnvelopeBound::Minimum,
                value: 4_096,
                ..
            })
        ));

        let mut input = valid_input();
        input.hard_envelopes.curl.minimum_ticks = 2_545;
        assert!(matches!(
            HeadGazeMappingDeclaration::parse(input),
            Err(
                HeadGazeMappingDeclarationParseError::ActiveJointLacksBidirectionalTravel {
                    joint: HeadJoint::Curl,
                    ..
                }
            )
        ));

        let mut input = valid_input();
        input.hard_envelopes.bow.maximum_ticks = 2_100;
        assert!(matches!(
            HeadGazeMappingDeclaration::parse(input),
            Err(HeadGazeMappingDeclarationParseError::HardEnvelopeReversed {
                joint: HeadJoint::Bow,
                ..
            })
        ));

        let mut input = valid_input();
        input.natural.roll_ticks = 4_096;
        assert!(matches!(
            HeadGazeMappingDeclaration::parse(input),
            Err(HeadGazeMappingDeclarationParseError::NaturalPosition {
                joint: HeadJoint::Roll,
                value: 4_096,
                ..
            })
        ));
    }

    #[test]
    fn nonfinite_cross_axis_and_missing_active_columns_are_rejected() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut input = valid_input();
            input
                .tick_offsets_per_radian
                .pitch_down
                .curl_ticks_per_radian = value;
            assert!(matches!(
                HeadGazeMappingDeclaration::parse(input),
                Err(
                    HeadGazeMappingDeclarationParseError::NonFiniteTickOffsetPerRadian {
                        coordinate: HeadGazeCoordinate::PitchDown,
                        joint: HeadJoint::Curl,
                        value: actual,
                    }
                ) if actual.to_bits() == value.to_bits()
            ));
        }

        let mut input = valid_input();
        input
            .tick_offsets_per_radian
            .yaw_right
            .roll_ticks_per_radian = 1.0;
        assert!(matches!(
            HeadGazeMappingDeclaration::parse(input),
            Err(
                HeadGazeMappingDeclarationParseError::AmbiguousCrossAxisMapping {
                    coordinate: HeadGazeCoordinate::YawRight,
                    joint: HeadJoint::Roll,
                    value: 1.0,
                }
            )
        ));

        for value in [0.0, -0.0] {
            let mut input = valid_input();
            input.tick_offsets_per_radian.yaw_right.yaw_ticks_per_radian = value;
            assert!(matches!(
                HeadGazeMappingDeclaration::parse(input),
                Err(
                    HeadGazeMappingDeclarationParseError::MissingActiveTickOffsetPerRadian {
                        coordinate: HeadGazeCoordinate::YawRight,
                        joint: HeadJoint::Yaw,
                    }
                )
            ));
        }
    }

    #[test]
    fn finite_nonzero_column_is_retained_without_a_reachability_claim() {
        for value in [0.25, -0.25] {
            let mut input = valid_input();
            input.tick_offsets_per_radian.yaw_right.yaw_ticks_per_radian = value;
            let declaration = HeadGazeMappingDeclaration::parse(input).unwrap();
            assert_eq!(
                declaration.tick_offset_per_radian(HeadGazeCoordinate::YawRight, HeadJoint::Yaw),
                value
            );
        }
    }

    #[test]
    fn identifiers_reject_empty_whitespace_unicode_and_overlong_values() {
        for value in ["", "has space", "unicode-\u{00e9}"] {
            let mut input = valid_input();
            input.assembly_id = value;
            assert!(matches!(
                HeadGazeMappingDeclaration::parse(input),
                Err(HeadGazeMappingDeclarationParseError::Identifier {
                    field: HeadGazeIdentifierField::AssemblyId,
                    ..
                })
            ));
        }

        let overlong = "a".repeat(MAX_HEAD_GAZE_IDENTIFIER_BYTES + 1);
        let mut input = valid_input();
        input.calibration_provenance_id = &overlong;
        assert!(matches!(
            HeadGazeMappingDeclaration::parse(input),
            Err(HeadGazeMappingDeclarationParseError::Identifier {
                field: HeadGazeIdentifierField::CalibrationProvenanceId,
                source: HeadGazeIdentifierError::TooLong { .. },
            })
        ));
    }

    #[test]
    fn camera_ray_mapping_uses_the_bound_focus_plane() {
        let calibration = HeadGazeMappingDeclaration::parse(valid_input()).unwrap();
        let ray = OakCameraTargetRay::parse([0.15, -0.10, 1.0]).unwrap();
        let from_policy = calibration.proposal_for_camera_ray(ray).unwrap();
        let gaze = calibration
            .camera_to_head()
            .project_ray_at_forward_depth(ray, CameraForwardDepthMeters::parse(1.5).unwrap())
            .unwrap();
        assert_eq!(from_policy, calibration.proposal_for_gaze(gaze).unwrap());
    }
}
