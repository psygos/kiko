//! Strict, non-activating boundary for one head-gaze policy document.
//!
//! Parsing owns and bounds every weak identifier, binds the declared mapping
//! to Kiko's exact OAK/head geometry through `kiko-expression-runtime`, and
//! turns controller numbers into unit-bearing runtime domain values. The
//! resulting value is still only a declaration. In particular, neither an
//! `operator_claimed_physical_review` lifecycle claim nor successful numeric
//! parsing is physical evidence or head-motion authority.
//!
//! This module intentionally does not construct an `ExactHeadTargetPose`,
//! `HeadGazeControlConfig`, torque consent, or motion consent. A later,
//! separately reviewed admission boundary must cross-bind the retained
//! identifiers and exact evidence digest before it can create those types.

use std::{fmt, time::Duration};

use kiko_expression_runtime::{
    CameraToHeadGazeExtrinsicsInput, HeadGazeMappingDeclaration, HeadGazeMappingDeclarationInput,
    HeadGazeMappingDeclarationParseError, HeadGazeTickOffsetsPerRadianInput, HeadTickEnvelope,
    HeadTickEnvelopeInput, NamedHeadTickEnvelopesInput, NamedHeadTickOffsetsPerRadianInput,
    NamedNaturalHeadTicksInput,
};
use kiko_head_protocol::{HeadJoint, JointCalibrationError, PositionStepLimit};
use kiko_head_runtime::gaze_control::{
    HeadAcquisitionProposalCount, HeadAcquisitionProposalCountError, HeadControlPeriod,
    HeadDeadbandTicks, HeadErrorBandValueError, HeadGazeErrorBand, HeadGazeErrorBandError,
    HeadGazeTiming, HeadJointMotionLimits, HeadJointMotionLimitsError, HeadMotionLimits,
    HeadProposalTtl, HeadResumeThresholdTicks, HeadTickLateness, PositiveServoTickLimitError,
    PositiveTimeValueError, ServoAccelerationLimitTicksPerControlTickSquared,
    ServoVelocityLimitTicksPerControlTick,
};
use serde::Deserialize;

/// The only admitted head-gaze policy schema.
pub const HEAD_GAZE_POLICY_V1: u32 = 1;

/// Bound checked before JSON deserialization can allocate caller-sized values.
pub const MAX_HEAD_GAZE_POLICY_JSON_BYTES: usize = 16 * 1_024;

/// Tracking starts only after three distinct, ordered fresh proposals.
pub const REQUIRED_HEAD_GAZE_ACQUISITION_PROPOSALS: u8 = 3;

const MAX_LIFECYCLE_IDENTIFIER_BYTES: usize = 128;
const SHA256_HEX_BYTES: usize = 64;
const SHA256_BYTES: usize = 32;
const MAX_CONTROL_PERIOD_NS: u64 = 1_000_000_000;
const MAX_TICK_LATENESS_NS: u64 = 1_000_000_000;
const MAX_PROPOSAL_TTL_NS: u64 = 5_000_000_000;

/// Parsed schema-V1 mapping, lifecycle claim, and non-activating controller
/// declaration.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadGazePolicyV1 {
    mapping: HeadGazeMappingDeclaration,
    controller: HeadGazeControllerDeclaration,
    lifecycle: HeadGazePolicyLifecycleClaim,
}

impl HeadGazePolicyV1 {
    /// Parse exactly one bounded JSON document.
    ///
    /// Success proves structural and numeric validity only. It does not verify
    /// the connected assembly, attest the retained evidence, or permit motion.
    pub fn parse_json(json: &[u8]) -> Result<Self, HeadGazePolicyParseError> {
        if json.len() > MAX_HEAD_GAZE_POLICY_JSON_BYTES {
            return Err(HeadGazePolicyParseError::InputTooLarge {
                actual_bytes: json.len(),
                maximum_bytes: MAX_HEAD_GAZE_POLICY_JSON_BYTES,
            });
        }

        let mut deserializer = serde_json::Deserializer::from_slice(json);
        let dto = HeadGazePolicyV1Dto::deserialize(&mut deserializer)
            .map_err(HeadGazePolicyParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(HeadGazePolicyParseError::JsonTrailingData)?;

        if dto.schema_version != HEAD_GAZE_POLICY_V1 {
            return Err(HeadGazePolicyParseError::UnsupportedSchemaVersion {
                actual: dto.schema_version,
                supported: HEAD_GAZE_POLICY_V1,
            });
        }

        let mapping =
            parse_mapping(dto.mapping_declaration).map_err(HeadGazePolicyParseError::Mapping)?;
        let controller = HeadGazeControllerDeclaration::parse(dto.controller_declaration, &mapping)
            .map_err(HeadGazePolicyParseError::Controller)?;
        let lifecycle =
            parse_lifecycle(dto.lifecycle).map_err(HeadGazePolicyParseError::Lifecycle)?;

        Ok(Self {
            mapping,
            controller,
            lifecycle,
        })
    }

    /// Numerically valid camera-ray to encoder-proposal mapping.
    ///
    /// Its output remains a non-command `HeadGazeTargetProposal`.
    pub const fn mapping(&self) -> &HeadGazeMappingDeclaration {
        &self.mapping
    }

    /// Typed timing, hysteresis, and motion-limit declaration.
    ///
    /// This deliberately lacks a command pose and cannot itself become a
    /// `HeadGazeControlConfig`.
    pub const fn controller(&self) -> &HeadGazeControllerDeclaration {
        &self.controller
    }

    /// Caller-supplied lifecycle/review claim retained for later evidence
    /// cross-binding.
    pub const fn lifecycle(&self) -> &HeadGazePolicyLifecycleClaim {
        &self.lifecycle
    }
}

/// Typed controller components with no command target and no motion consent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadGazeControllerDeclaration {
    timing: HeadGazeTiming,
    motion_limits: HeadMotionLimits,
    error_band: HeadGazeErrorBand,
}

impl HeadGazeControllerDeclaration {
    fn parse(
        dto: HeadGazeControllerDeclarationDto,
        mapping: &HeadGazeMappingDeclaration,
    ) -> Result<Self, HeadGazeControllerDeclarationParseError> {
        let timing = parse_timing(dto.timing)?;
        let motion_limits = parse_motion_limits(dto.motion_limits, mapping)?;
        let error_band = parse_error_band(dto.error_band)?;
        Ok(Self {
            timing,
            motion_limits,
            error_band,
        })
    }

    pub const fn timing(self) -> HeadGazeTiming {
        self.timing
    }

    pub const fn motion_limits(self) -> HeadMotionLimits {
        self.motion_limits
    }

    pub const fn error_band(self) -> HeadGazeErrorBand {
        self.error_band
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct OwnedLifecycleIdentifier(Box<str>);

impl OwnedLifecycleIdentifier {
    fn parse(
        field: HeadGazeLifecycleIdentifierField,
        value: String,
    ) -> Result<Self, HeadGazeLifecycleClaimParseError> {
        if value.is_empty() {
            return Err(HeadGazeLifecycleClaimParseError::Identifier {
                field,
                source: HeadGazeLifecycleIdentifierError::Empty,
            });
        }
        if value.len() > MAX_LIFECYCLE_IDENTIFIER_BYTES {
            return Err(HeadGazeLifecycleClaimParseError::Identifier {
                field,
                source: HeadGazeLifecycleIdentifierError::TooLong {
                    actual_bytes: value.len(),
                    maximum_bytes: MAX_LIFECYCLE_IDENTIFIER_BYTES,
                },
            });
        }
        if value.bytes().all(|byte| byte == b'0') {
            return Err(HeadGazeLifecycleClaimParseError::Identifier {
                field,
                source: HeadGazeLifecycleIdentifierError::AllZero,
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
                return Err(HeadGazeLifecycleClaimParseError::Identifier {
                    field,
                    source: HeadGazeLifecycleIdentifierError::InvalidByte { index, byte },
                });
            }
        }
        Ok(Self(value.into_boxed_str()))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

macro_rules! lifecycle_identifier {
    ($name:ident, $field:expr) => {
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $name(OwnedLifecycleIdentifier);

        impl $name {
            fn parse(value: String) -> Result<Self, HeadGazeLifecycleClaimParseError> {
                OwnedLifecycleIdentifier::parse($field, value).map(Self)
            }

            pub fn as_str(&self) -> &str {
                self.0.as_str()
            }
        }
    };
}

lifecycle_identifier!(
    HeadGazeProposalClaimId,
    HeadGazeLifecycleIdentifierField::Proposal
);
lifecycle_identifier!(
    HeadGazeReviewClaimId,
    HeadGazeLifecycleIdentifierField::Review
);
lifecycle_identifier!(
    HeadGazeOperatorId,
    HeadGazeLifecycleIdentifierField::Operator
);
lifecycle_identifier!(
    HeadGazeReviewEvidenceId,
    HeadGazeLifecycleIdentifierField::Evidence
);

/// Canonical lowercase SHA-256 identity of the exact evidence bytes claimed by
/// the policy author.
///
/// Parsing this digest does not prove those bytes exist or that they establish
/// a valid physical calibration.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct HeadGazeEvidenceContentSha256 {
    canonical_hex: Box<str>,
    bytes: [u8; SHA256_BYTES],
}

impl HeadGazeEvidenceContentSha256 {
    fn parse(value: String) -> Result<Self, HeadGazeLifecycleClaimParseError> {
        if value.len() != SHA256_HEX_BYTES {
            return Err(HeadGazeLifecycleClaimParseError::EvidenceContentSha256 {
                source: HeadGazeEvidenceContentSha256Error::WrongLength {
                    actual_bytes: value.len(),
                    required_bytes: SHA256_HEX_BYTES,
                },
            });
        }
        let mut bytes = [0_u8; SHA256_BYTES];
        for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
            let high = lowercase_hex_nibble(pair[0]).ok_or(
                HeadGazeLifecycleClaimParseError::EvidenceContentSha256 {
                    source: HeadGazeEvidenceContentSha256Error::NonCanonicalHex {
                        index: index * 2,
                        byte: pair[0],
                    },
                },
            )?;
            let low = lowercase_hex_nibble(pair[1]).ok_or(
                HeadGazeLifecycleClaimParseError::EvidenceContentSha256 {
                    source: HeadGazeEvidenceContentSha256Error::NonCanonicalHex {
                        index: index * 2 + 1,
                        byte: pair[1],
                    },
                },
            )?;
            bytes[index] = (high << 4) | low;
        }
        if bytes == [0; SHA256_BYTES] {
            return Err(HeadGazeLifecycleClaimParseError::EvidenceContentSha256 {
                source: HeadGazeEvidenceContentSha256Error::AllZero,
            });
        }
        Ok(Self {
            canonical_hex: value.into_boxed_str(),
            bytes,
        })
    }

    pub fn as_str(&self) -> &str {
        &self.canonical_hex
    }

    pub const fn as_bytes(&self) -> &[u8; SHA256_BYTES] {
        &self.bytes
    }
}

const fn lowercase_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

/// Explicit proposal-only lifecycle state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadGazeProposalOnlyClaim {
    proposal_id: HeadGazeProposalClaimId,
    evidence_id: HeadGazeReviewEvidenceId,
    evidence_content_sha256: HeadGazeEvidenceContentSha256,
}

impl HeadGazeProposalOnlyClaim {
    pub const fn proposal_id(&self) -> &HeadGazeProposalClaimId {
        &self.proposal_id
    }

    pub const fn evidence_id(&self) -> &HeadGazeReviewEvidenceId {
        &self.evidence_id
    }

    pub const fn evidence_content_sha256(&self) -> &HeadGazeEvidenceContentSha256 {
        &self.evidence_content_sha256
    }
}

/// Caller-asserted physical-review metadata.
///
/// The name intentionally says `OperatorClaimed`: JSON is not an
/// authentication or evidence-verification mechanism.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorClaimedHeadGazePhysicalReview {
    review_id: HeadGazeReviewClaimId,
    operator_id: HeadGazeOperatorId,
    evidence_id: HeadGazeReviewEvidenceId,
    evidence_content_sha256: HeadGazeEvidenceContentSha256,
}

impl OperatorClaimedHeadGazePhysicalReview {
    pub const fn review_id(&self) -> &HeadGazeReviewClaimId {
        &self.review_id
    }

    pub const fn operator_id(&self) -> &HeadGazeOperatorId {
        &self.operator_id
    }

    pub const fn evidence_id(&self) -> &HeadGazeReviewEvidenceId {
        &self.evidence_id
    }

    pub const fn evidence_content_sha256(&self) -> &HeadGazeEvidenceContentSha256 {
        &self.evidence_content_sha256
    }
}

/// Parsed lifecycle claim. Neither variant is motion authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeadGazePolicyLifecycleClaim {
    ProposalOnly(HeadGazeProposalOnlyClaim),
    OperatorClaimedPhysicalReview(OperatorClaimedHeadGazePhysicalReview),
}

fn parse_lifecycle(
    dto: HeadGazePolicyLifecycleDto,
) -> Result<HeadGazePolicyLifecycleClaim, HeadGazeLifecycleClaimParseError> {
    match dto {
        HeadGazePolicyLifecycleDto::ProposalOnly {
            proposal_id,
            evidence_id,
            evidence_content_sha256_hex,
        } => Ok(HeadGazePolicyLifecycleClaim::ProposalOnly(
            HeadGazeProposalOnlyClaim {
                proposal_id: HeadGazeProposalClaimId::parse(proposal_id)?,
                evidence_id: HeadGazeReviewEvidenceId::parse(evidence_id)?,
                evidence_content_sha256: HeadGazeEvidenceContentSha256::parse(
                    evidence_content_sha256_hex,
                )?,
            },
        )),
        HeadGazePolicyLifecycleDto::OperatorClaimedPhysicalReview {
            review_id,
            operator_id,
            evidence_id,
            evidence_content_sha256_hex,
        } => Ok(HeadGazePolicyLifecycleClaim::OperatorClaimedPhysicalReview(
            OperatorClaimedHeadGazePhysicalReview {
                review_id: HeadGazeReviewClaimId::parse(review_id)?,
                operator_id: HeadGazeOperatorId::parse(operator_id)?,
                evidence_id: HeadGazeReviewEvidenceId::parse(evidence_id)?,
                evidence_content_sha256: HeadGazeEvidenceContentSha256::parse(
                    evidence_content_sha256_hex,
                )?,
            },
        )),
    }
}

fn parse_mapping(
    dto: HeadGazeMappingDeclarationDto,
) -> Result<HeadGazeMappingDeclaration, HeadGazeMappingDeclarationParseError> {
    let origin = dto.camera_to_neutral_head.head_origin_in_oak_camera_m;
    let rotation = dto
        .camera_to_neutral_head
        .neutral_head_from_oak_camera_quaternion_xyzw;
    let envelopes = dto.hard_encoder_envelopes_ticks;
    let offsets = dto.encoder_tick_offsets_per_radian;

    HeadGazeMappingDeclaration::parse(HeadGazeMappingDeclarationInput {
        assembly_id: &dto.assembly_id,
        calibration_provenance_id: &dto.calibration_provenance_id,
        focus_plane_camera_forward_depth_m: dto.gaze_only_focus_plane.camera_forward_depth_m,
        camera_to_head: CameraToHeadGazeExtrinsicsInput {
            head_origin_in_camera_m: [origin.x_right_m, origin.y_down_m, origin.z_forward_m],
            neutral_head_from_camera_quaternion_xyzw: [
                rotation.x, rotation.y, rotation.z, rotation.w,
            ],
        },
        natural: NamedNaturalHeadTicksInput {
            bow_ticks: dto.natural_encoder_position_ticks.bow_ticks,
            curl_ticks: dto.natural_encoder_position_ticks.curl_ticks,
            yaw_ticks: dto.natural_encoder_position_ticks.yaw_ticks,
            roll_ticks: dto.natural_encoder_position_ticks.roll_ticks,
        },
        hard_envelopes: NamedHeadTickEnvelopesInput {
            bow: envelopes.bow.into_domain(),
            curl: envelopes.curl.into_domain(),
            yaw: envelopes.yaw.into_domain(),
            roll: envelopes.roll.into_domain(),
        },
        tick_offsets_per_radian: HeadGazeTickOffsetsPerRadianInput {
            pitch_down: offsets.pitch_down_rad.into_domain(),
            yaw_right: offsets.yaw_right_rad.into_domain(),
        },
    })
}

fn parse_timing(
    dto: HeadGazeTimingDto,
) -> Result<HeadGazeTiming, HeadGazeControllerDeclarationParseError> {
    let control_period = parse_positive_time(
        HeadGazeTimingField::ControlPeriod,
        dto.control_period_ns,
        MAX_CONTROL_PERIOD_NS,
        HeadControlPeriod::try_new,
    )?;
    if dto.maximum_tick_lateness_ns > MAX_TICK_LATENESS_NS {
        return Err(
            HeadGazeControllerDeclarationParseError::TimingAboveMaximum {
                field: HeadGazeTimingField::MaximumTickLateness,
                actual_ns: dto.maximum_tick_lateness_ns,
                maximum_ns: MAX_TICK_LATENESS_NS,
            },
        );
    }
    if dto.maximum_tick_lateness_ns >= dto.control_period_ns {
        return Err(
            HeadGazeControllerDeclarationParseError::TickLatenessNotBelowControlPeriod {
                maximum_tick_lateness_ns: dto.maximum_tick_lateness_ns,
                control_period_ns: dto.control_period_ns,
            },
        );
    }
    let maximum_tick_lateness =
        HeadTickLateness::new(Duration::from_nanos(dto.maximum_tick_lateness_ns));
    let proposal_ttl = parse_positive_time(
        HeadGazeTimingField::ProposalTtl,
        dto.proposal_ttl_ns,
        MAX_PROPOSAL_TTL_NS,
        HeadProposalTtl::try_new,
    )?;
    let minimum_useful_ttl_ns = dto
        .control_period_ns
        .checked_add(dto.maximum_tick_lateness_ns)
        .ok_or(
            HeadGazeControllerDeclarationParseError::TimingNanosecondsOverflow {
                left_ns: dto.control_period_ns,
                right_ns: dto.maximum_tick_lateness_ns,
            },
        )?;
    if dto.proposal_ttl_ns <= minimum_useful_ttl_ns {
        return Err(
            HeadGazeControllerDeclarationParseError::ProposalTtlDoesNotCoverOneLateTick {
                proposal_ttl_ns: dto.proposal_ttl_ns,
                required_exclusive_minimum_ns: minimum_useful_ttl_ns,
            },
        );
    }
    if dto.acquisition_proposals != REQUIRED_HEAD_GAZE_ACQUISITION_PROPOSALS {
        return Err(
            HeadGazeControllerDeclarationParseError::AcquisitionProposalCountMismatch {
                actual: dto.acquisition_proposals,
                required: REQUIRED_HEAD_GAZE_ACQUISITION_PROPOSALS,
            },
        );
    }
    let acquisition_proposals = HeadAcquisitionProposalCount::try_new(dto.acquisition_proposals)
        .map_err(
            |source| HeadGazeControllerDeclarationParseError::AcquisitionProposalCount { source },
        )?;

    Ok(HeadGazeTiming::new(
        control_period,
        maximum_tick_lateness,
        proposal_ttl,
        acquisition_proposals,
    ))
}

fn parse_positive_time<T>(
    field: HeadGazeTimingField,
    nanoseconds: u64,
    maximum_ns: u64,
    parse: impl FnOnce(Duration) -> Result<T, PositiveTimeValueError>,
) -> Result<T, HeadGazeControllerDeclarationParseError> {
    if nanoseconds > maximum_ns {
        return Err(
            HeadGazeControllerDeclarationParseError::TimingAboveMaximum {
                field,
                actual_ns: nanoseconds,
                maximum_ns,
            },
        );
    }
    parse(Duration::from_nanos(nanoseconds))
        .map_err(|source| HeadGazeControllerDeclarationParseError::PositiveTiming { field, source })
}

fn parse_error_band(
    dto: HeadGazeErrorBandDto,
) -> Result<HeadGazeErrorBand, HeadGazeControllerDeclarationParseError> {
    let deadband = HeadDeadbandTicks::try_new(dto.settle_deadband_ticks).map_err(|source| {
        HeadGazeControllerDeclarationParseError::ErrorBandValue {
            field: HeadGazeErrorBandField::SettleDeadbandTicks,
            source,
        }
    })?;
    let resume_threshold =
        HeadResumeThresholdTicks::try_new(dto.resume_threshold_ticks).map_err(|source| {
            HeadGazeControllerDeclarationParseError::ErrorBandValue {
                field: HeadGazeErrorBandField::ResumeThresholdTicks,
                source,
            }
        })?;
    HeadGazeErrorBand::try_new(deadband, resume_threshold)
        .map_err(|source| HeadGazeControllerDeclarationParseError::ErrorBand { source })
}

fn parse_motion_limits(
    dto: NamedHeadGazeMotionLimitsDto,
    mapping: &HeadGazeMappingDeclaration,
) -> Result<HeadMotionLimits, HeadGazeControllerDeclarationParseError> {
    Ok(HeadMotionLimits::new(
        parse_joint_motion(
            HeadJoint::Bow,
            dto.bow,
            mapping.hard_envelope(HeadJoint::Bow),
        )?,
        parse_joint_motion(
            HeadJoint::Curl,
            dto.curl,
            mapping.hard_envelope(HeadJoint::Curl),
        )?,
        parse_joint_motion(
            HeadJoint::Yaw,
            dto.yaw,
            mapping.hard_envelope(HeadJoint::Yaw),
        )?,
        parse_joint_motion(
            HeadJoint::Roll,
            dto.roll,
            mapping.hard_envelope(HeadJoint::Roll),
        )?,
    ))
}

fn parse_joint_motion(
    joint: HeadJoint,
    dto: HeadGazeJointMotionLimitsDto,
    envelope: HeadTickEnvelope,
) -> Result<HeadJointMotionLimits, HeadGazeControllerDeclarationParseError> {
    let maximum_velocity =
        ServoVelocityLimitTicksPerControlTick::try_new(dto.maximum_velocity_ticks_per_control_tick)
            .map_err(
                |source| HeadGazeControllerDeclarationParseError::MotionValue {
                    joint,
                    field: HeadGazeMotionField::VelocityPerControlTick,
                    source: HeadGazeMotionValueError::Velocity(source),
                },
            )?;
    let maximum_acceleration = ServoAccelerationLimitTicksPerControlTickSquared::try_new(
        dto.maximum_acceleration_ticks_per_control_tick_squared,
    )
    .map_err(
        |source| HeadGazeControllerDeclarationParseError::MotionValue {
            joint,
            field: HeadGazeMotionField::AccelerationPerControlTickSquared,
            source: HeadGazeMotionValueError::Acceleration(source),
        },
    )?;
    let maximum_position_step = PositionStepLimit::try_new(dto.maximum_position_step_ticks)
        .map_err(
            |source| HeadGazeControllerDeclarationParseError::MotionValue {
                joint,
                field: HeadGazeMotionField::PositionStep,
                source: HeadGazeMotionValueError::PositionStep(source),
            },
        )?;
    HeadJointMotionLimits::try_new(
        envelope.minimum(),
        envelope.maximum(),
        maximum_velocity,
        maximum_acceleration,
        maximum_position_step,
    )
    .map_err(|source| HeadGazeControllerDeclarationParseError::JointMotionLimits { joint, source })
}

#[derive(Debug)]
pub enum HeadGazePolicyParseError {
    InputTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    JsonDecode(serde_json::Error),
    JsonTrailingData(serde_json::Error),
    UnsupportedSchemaVersion {
        actual: u32,
        supported: u32,
    },
    Mapping(HeadGazeMappingDeclarationParseError),
    Controller(HeadGazeControllerDeclarationParseError),
    Lifecycle(HeadGazeLifecycleClaimParseError),
}

impl fmt::Display for HeadGazePolicyParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid head-gaze policy: {self:?}")
    }
}

impl std::error::Error for HeadGazePolicyParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::JsonDecode(source) | Self::JsonTrailingData(source) => Some(source),
            Self::Mapping(source) => Some(source),
            Self::Controller(source) => Some(source),
            Self::Lifecycle(source) => Some(source),
            Self::InputTooLarge { .. } | Self::UnsupportedSchemaVersion { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeTimingField {
    ControlPeriod,
    MaximumTickLateness,
    ProposalTtl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeErrorBandField {
    SettleDeadbandTicks,
    ResumeThresholdTicks,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeMotionField {
    VelocityPerControlTick,
    AccelerationPerControlTickSquared,
    PositionStep,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadGazeMotionValueError {
    Velocity(PositiveServoTickLimitError),
    Acceleration(PositiveServoTickLimitError),
    PositionStep(JointCalibrationError),
}

impl fmt::Display for HeadGazeMotionValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid head-gaze motion value: {self:?}")
    }
}

impl std::error::Error for HeadGazeMotionValueError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Velocity(source) | Self::Acceleration(source) => Some(source),
            Self::PositionStep(source) => Some(source),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HeadGazeControllerDeclarationParseError {
    TimingAboveMaximum {
        field: HeadGazeTimingField,
        actual_ns: u64,
        maximum_ns: u64,
    },
    PositiveTiming {
        field: HeadGazeTimingField,
        source: PositiveTimeValueError,
    },
    TickLatenessNotBelowControlPeriod {
        maximum_tick_lateness_ns: u64,
        control_period_ns: u64,
    },
    TimingNanosecondsOverflow {
        left_ns: u64,
        right_ns: u64,
    },
    ProposalTtlDoesNotCoverOneLateTick {
        proposal_ttl_ns: u64,
        required_exclusive_minimum_ns: u64,
    },
    AcquisitionProposalCountMismatch {
        actual: u8,
        required: u8,
    },
    AcquisitionProposalCount {
        source: HeadAcquisitionProposalCountError,
    },
    ErrorBandValue {
        field: HeadGazeErrorBandField,
        source: HeadErrorBandValueError,
    },
    ErrorBand {
        source: HeadGazeErrorBandError,
    },
    MotionValue {
        joint: HeadJoint,
        field: HeadGazeMotionField,
        source: HeadGazeMotionValueError,
    },
    JointMotionLimits {
        joint: HeadJoint,
        source: HeadJointMotionLimitsError,
    },
}

impl fmt::Display for HeadGazeControllerDeclarationParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid non-activating head-gaze controller declaration: {self:?}"
        )
    }
}

impl std::error::Error for HeadGazeControllerDeclarationParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PositiveTiming { source, .. } => Some(source),
            Self::AcquisitionProposalCount { source } => Some(source),
            Self::ErrorBandValue { source, .. } => Some(source),
            Self::ErrorBand { source } => Some(source),
            Self::MotionValue { source, .. } => Some(source),
            Self::JointMotionLimits { source, .. } => Some(source),
            Self::TimingAboveMaximum { .. }
            | Self::TickLatenessNotBelowControlPeriod { .. }
            | Self::TimingNanosecondsOverflow { .. }
            | Self::ProposalTtlDoesNotCoverOneLateTick { .. }
            | Self::AcquisitionProposalCountMismatch { .. } => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeLifecycleIdentifierField {
    Proposal,
    Review,
    Operator,
    Evidence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeLifecycleIdentifierError {
    Empty,
    TooLong {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    AllZero,
    InvalidByte {
        index: usize,
        byte: u8,
    },
}

impl fmt::Display for HeadGazeLifecycleIdentifierError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid lifecycle identifier: {self:?}")
    }
}

impl std::error::Error for HeadGazeLifecycleIdentifierError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeEvidenceContentSha256Error {
    WrongLength {
        actual_bytes: usize,
        required_bytes: usize,
    },
    NonCanonicalHex {
        index: usize,
        byte: u8,
    },
    AllZero,
}

impl fmt::Display for HeadGazeEvidenceContentSha256Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid evidence content SHA-256: {self:?}")
    }
}

impl std::error::Error for HeadGazeEvidenceContentSha256Error {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadGazeLifecycleClaimParseError {
    Identifier {
        field: HeadGazeLifecycleIdentifierField,
        source: HeadGazeLifecycleIdentifierError,
    },
    EvidenceContentSha256 {
        source: HeadGazeEvidenceContentSha256Error,
    },
}

impl fmt::Display for HeadGazeLifecycleClaimParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid head-gaze lifecycle claim: {self:?}")
    }
}

impl std::error::Error for HeadGazeLifecycleClaimParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Identifier { source, .. } => Some(source),
            Self::EvidenceContentSha256 { source } => Some(source),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeadGazePolicyV1Dto {
    schema_version: u32,
    lifecycle: HeadGazePolicyLifecycleDto,
    mapping_declaration: HeadGazeMappingDeclarationDto,
    controller_declaration: HeadGazeControllerDeclarationDto,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum HeadGazePolicyLifecycleDto {
    ProposalOnly {
        proposal_id: String,
        evidence_id: String,
        evidence_content_sha256_hex: String,
    },
    OperatorClaimedPhysicalReview {
        review_id: String,
        operator_id: String,
        evidence_id: String,
        evidence_content_sha256_hex: String,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeadGazeMappingDeclarationDto {
    assembly_id: String,
    calibration_provenance_id: String,
    gaze_only_focus_plane: GazeOnlyFocusPlaneDto,
    camera_to_neutral_head: CameraToNeutralHeadDto,
    natural_encoder_position_ticks: NamedNaturalHeadTicksDto,
    hard_encoder_envelopes_ticks: NamedHeadTickEnvelopesDto,
    encoder_tick_offsets_per_radian: HeadGazeTickOffsetsPerRadianDto,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GazeOnlyFocusPlaneDto {
    camera_forward_depth_m: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CameraToNeutralHeadDto {
    head_origin_in_oak_camera_m: NamedOakCameraPositionMetersDto,
    neutral_head_from_oak_camera_quaternion_xyzw: NamedQuaternionXyzwDto,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedOakCameraPositionMetersDto {
    x_right_m: f64,
    y_down_m: f64,
    z_forward_m: f64,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedQuaternionXyzwDto {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedNaturalHeadTicksDto {
    bow_ticks: u16,
    curl_ticks: u16,
    yaw_ticks: u16,
    roll_ticks: u16,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedHeadTickEnvelopesDto {
    bow: HeadTickEnvelopeDto,
    curl: HeadTickEnvelopeDto,
    yaw: HeadTickEnvelopeDto,
    roll: HeadTickEnvelopeDto,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeadTickEnvelopeDto {
    minimum_ticks: u16,
    maximum_ticks: u16,
}

impl HeadTickEnvelopeDto {
    const fn into_domain(self) -> HeadTickEnvelopeInput {
        HeadTickEnvelopeInput {
            minimum_ticks: self.minimum_ticks,
            maximum_ticks: self.maximum_ticks,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeadGazeTickOffsetsPerRadianDto {
    pitch_down_rad: NamedHeadTickOffsetsPerRadianDto,
    yaw_right_rad: NamedHeadTickOffsetsPerRadianDto,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedHeadTickOffsetsPerRadianDto {
    bow_ticks_per_radian: f64,
    curl_ticks_per_radian: f64,
    yaw_ticks_per_radian: f64,
    roll_ticks_per_radian: f64,
}

impl NamedHeadTickOffsetsPerRadianDto {
    const fn into_domain(self) -> NamedHeadTickOffsetsPerRadianInput {
        NamedHeadTickOffsetsPerRadianInput {
            bow_ticks_per_radian: self.bow_ticks_per_radian,
            curl_ticks_per_radian: self.curl_ticks_per_radian,
            yaw_ticks_per_radian: self.yaw_ticks_per_radian,
            roll_ticks_per_radian: self.roll_ticks_per_radian,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeadGazeControllerDeclarationDto {
    timing: HeadGazeTimingDto,
    error_band: HeadGazeErrorBandDto,
    motion_limits: NamedHeadGazeMotionLimitsDto,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeadGazeTimingDto {
    control_period_ns: u64,
    maximum_tick_lateness_ns: u64,
    proposal_ttl_ns: u64,
    acquisition_proposals: u8,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeadGazeErrorBandDto {
    settle_deadband_ticks: u16,
    resume_threshold_ticks: u16,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedHeadGazeMotionLimitsDto {
    bow: HeadGazeJointMotionLimitsDto,
    curl: HeadGazeJointMotionLimitsDto,
    yaw: HeadGazeJointMotionLimitsDto,
    roll: HeadGazeJointMotionLimitsDto,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct HeadGazeJointMotionLimitsDto {
    maximum_velocity_ticks_per_control_tick: u32,
    maximum_acceleration_ticks_per_control_tick_squared: u32,
    maximum_position_step_ticks: u16,
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiko_expression_runtime::{
        DECLARED_HEAD_ORIGIN_IN_OAK_CAMERA_M, HEAD_GAZE_FOCUS_PLANE_CAMERA_FORWARD_DEPTH_M,
    };
    use serde_json::{Value, json};

    fn valid_value() -> Value {
        json!({
            "schema_version": 1,
            "lifecycle": {
                "kind": "proposal_only",
                "proposal_id": "head-gaze-map-proposal-2026-07-27",
                "evidence_id": "bench-observations-2026-07-21",
                "evidence_content_sha256_hex":
                    "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"
            },
            "mapping_declaration": {
                "assembly_id": "kiko-head-assembly-01",
                "calibration_provenance_id": "fable-demo-measurements-proposal-v1",
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
                    "bow_ticks": 2174,
                    "curl_ticks": 2570,
                    "yaw_ticks": 1637,
                    "roll_ticks": 3047
                },
                "hard_encoder_envelopes_ticks": {
                    "bow": {"minimum_ticks": 1974, "maximum_ticks": 2374},
                    "curl": {"minimum_ticks": 2370, "maximum_ticks": 2770},
                    "yaw": {"minimum_ticks": 1437, "maximum_ticks": 1837},
                    "roll": {"minimum_ticks": 2847, "maximum_ticks": 3247}
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
                    "proposal_ttl_ns": 150_000_000,
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

    fn parse(value: &Value) -> Result<HeadGazePolicyV1, HeadGazePolicyParseError> {
        HeadGazePolicyV1::parse_json(&serde_json::to_vec(value).expect("fixture JSON"))
    }

    #[test]
    fn parses_exact_geometry_and_typed_non_activating_controller_declaration() {
        let policy = parse(&valid_value()).expect("valid head-gaze policy");
        assert_eq!(
            policy.mapping().focus_plane().get(),
            HEAD_GAZE_FOCUS_PLANE_CAMERA_FORWARD_DEPTH_M
        );
        assert_eq!(
            policy.mapping().camera_to_head().head_origin_in_camera_m(),
            DECLARED_HEAD_ORIGIN_IN_OAK_CAMERA_M
        );
        let timing = policy.controller().timing();
        assert_eq!(timing.control_period().get(), Duration::from_millis(20));
        assert_eq!(
            timing.maximum_tick_lateness().get(),
            Duration::from_millis(5)
        );
        assert_eq!(timing.proposal_ttl().get(), Duration::from_millis(150));
        assert_eq!(
            timing.acquisition_proposals().get(),
            REQUIRED_HEAD_GAZE_ACQUISITION_PROPOSALS
        );
        let yaw = policy.controller().motion_limits().joint(HeadJoint::Yaw);
        assert_eq!(yaw.minimum().get(), 1437);
        assert_eq!(yaw.maximum().get(), 1837);
        assert_eq!(yaw.maximum_velocity().get(), 8);
        assert_eq!(yaw.maximum_acceleration().get(), 2);
        assert_eq!(yaw.maximum_position_step().get(), 8);
        assert_eq!(policy.controller().error_band().deadband().get(), 2);
        assert_eq!(policy.controller().error_band().resume_threshold().get(), 5);
    }

    #[test]
    fn lifecycle_variants_retain_owned_cross_binding_identifiers_without_authority() {
        let proposal = parse(&valid_value()).expect("proposal policy");
        let HeadGazePolicyLifecycleClaim::ProposalOnly(proposal) = proposal.lifecycle() else {
            panic!("expected proposal-only lifecycle");
        };
        assert_eq!(
            proposal.proposal_id().as_str(),
            "head-gaze-map-proposal-2026-07-27"
        );
        assert_eq!(
            proposal.evidence_id().as_str(),
            "bench-observations-2026-07-21"
        );
        assert_eq!(
            proposal.evidence_content_sha256().as_bytes()[0..4],
            [0x11, 0x22, 0x33, 0x44]
        );

        let mut reviewed = valid_value();
        reviewed["lifecycle"] = json!({
            "kind": "operator_claimed_physical_review",
            "review_id": "review-claim-2026-07-27",
            "operator_id": "operator:ttrb",
            "evidence_id": "physical-head-calibration-session-01",
            "evidence_content_sha256_hex":
                "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        });
        let reviewed = parse(&reviewed).expect("claimed-review policy");
        let HeadGazePolicyLifecycleClaim::OperatorClaimedPhysicalReview(reviewed) =
            reviewed.lifecycle()
        else {
            panic!("expected operator-claimed review");
        };
        assert_eq!(reviewed.review_id().as_str(), "review-claim-2026-07-27");
        assert_eq!(reviewed.operator_id().as_str(), "operator:ttrb");
        assert_eq!(
            reviewed.evidence_id().as_str(),
            "physical-head-calibration-session-01"
        );
        assert_eq!(
            reviewed.evidence_content_sha256().as_str(),
            "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        );
    }

    #[test]
    fn malformed_duplicate_unknown_and_trailing_json_are_rejected() {
        assert!(matches!(
            HeadGazePolicyV1::parse_json(b"{"),
            Err(HeadGazePolicyParseError::JsonDecode(_))
        ));

        let canonical = serde_json::to_string(&valid_value()).expect("fixture JSON");
        let duplicate = canonical.replacen(
            "\"schema_version\":1",
            "\"schema_version\":1,\"schema_version\":1",
            1,
        );
        assert!(matches!(
            HeadGazePolicyV1::parse_json(duplicate.as_bytes()),
            Err(HeadGazePolicyParseError::JsonDecode(_))
        ));

        let mut unknown = valid_value();
        unknown["mapping_declaration"]["camera_to_neutral_head"]
            .as_object_mut()
            .expect("camera/head object")
            .insert("frame_guess".to_owned(), json!("maybe"));
        assert!(matches!(
            parse(&unknown),
            Err(HeadGazePolicyParseError::JsonDecode(_))
        ));

        let mut unknown_variant = valid_value();
        unknown_variant["lifecycle"]
            .as_object_mut()
            .expect("lifecycle object")
            .insert("motion_authority".to_owned(), json!(true));
        assert!(matches!(
            parse(&unknown_variant),
            Err(HeadGazePolicyParseError::JsonDecode(_))
        ));

        let mut trailing = serde_json::to_vec(&valid_value()).expect("fixture JSON");
        trailing.extend_from_slice(b" true");
        assert!(matches!(
            HeadGazePolicyV1::parse_json(&trailing),
            Err(HeadGazePolicyParseError::JsonTrailingData(_))
        ));
    }

    #[test]
    fn every_nested_object_shape_denies_unknown_fields() {
        for pointer in [
            "",
            "/mapping_declaration",
            "/mapping_declaration/gaze_only_focus_plane",
            "/mapping_declaration/camera_to_neutral_head",
            "/mapping_declaration/camera_to_neutral_head/head_origin_in_oak_camera_m",
            "/mapping_declaration/camera_to_neutral_head/neutral_head_from_oak_camera_quaternion_xyzw",
            "/mapping_declaration/natural_encoder_position_ticks",
            "/mapping_declaration/hard_encoder_envelopes_ticks",
            "/mapping_declaration/hard_encoder_envelopes_ticks/bow",
            "/mapping_declaration/encoder_tick_offsets_per_radian",
            "/mapping_declaration/encoder_tick_offsets_per_radian/pitch_down_rad",
            "/controller_declaration",
            "/controller_declaration/timing",
            "/controller_declaration/error_band",
            "/controller_declaration/motion_limits",
            "/controller_declaration/motion_limits/bow",
        ] {
            let mut document = valid_value();
            document
                .pointer_mut(pointer)
                .expect("fixture pointer")
                .as_object_mut()
                .expect("object")
                .insert("unknown_field".to_owned(), json!(1));
            assert!(
                matches!(
                    parse(&document),
                    Err(HeadGazePolicyParseError::JsonDecode(_))
                ),
                "object at {pointer} accepted an unknown field"
            );
        }
    }

    #[test]
    fn input_schema_geometry_focus_and_acquisition_are_fail_closed() {
        let oversized = vec![b' '; MAX_HEAD_GAZE_POLICY_JSON_BYTES + 1];
        assert!(matches!(
            HeadGazePolicyV1::parse_json(&oversized),
            Err(HeadGazePolicyParseError::InputTooLarge { .. })
        ));

        let mut schema = valid_value();
        schema["schema_version"] = json!(2);
        assert!(matches!(
            parse(&schema),
            Err(HeadGazePolicyParseError::UnsupportedSchemaVersion {
                actual: 2,
                supported: HEAD_GAZE_POLICY_V1
            })
        ));

        let mut geometry = valid_value();
        geometry["mapping_declaration"]["camera_to_neutral_head"]["head_origin_in_oak_camera_m"]
            ["y_down_m"] = json!(0.25);
        assert!(matches!(
            parse(&geometry),
            Err(HeadGazePolicyParseError::Mapping(
                HeadGazeMappingDeclarationParseError::HeadOriginDoesNotMatchDeclaration { .. }
            ))
        ));

        let mut focus = valid_value();
        focus["mapping_declaration"]["gaze_only_focus_plane"]["camera_forward_depth_m"] =
            json!(1.6);
        assert!(matches!(
            parse(&focus),
            Err(HeadGazePolicyParseError::Mapping(
                HeadGazeMappingDeclarationParseError::FocusPlaneDoesNotMatchPolicy { .. }
            ))
        ));

        let mut acquisition = valid_value();
        acquisition["controller_declaration"]["timing"]["acquisition_proposals"] = json!(2);
        assert!(matches!(
            parse(&acquisition),
            Err(HeadGazePolicyParseError::Controller(
                HeadGazeControllerDeclarationParseError::AcquisitionProposalCountMismatch {
                    actual: 2,
                    required: REQUIRED_HEAD_GAZE_ACQUISITION_PROPOSALS
                }
            ))
        ));
    }

    #[test]
    fn nonfinite_and_invalid_controller_numbers_are_rejected_without_fallbacks() {
        let canonical = serde_json::to_string(&valid_value()).expect("fixture JSON");
        let nonfinite = canonical.replacen("-300.0", "1e400", 1);
        assert!(matches!(
            HeadGazePolicyV1::parse_json(nonfinite.as_bytes()),
            Err(HeadGazePolicyParseError::JsonDecode(_))
        ));

        let mut zero_period = valid_value();
        zero_period["controller_declaration"]["timing"]["control_period_ns"] = json!(0);
        assert!(matches!(
            parse(&zero_period),
            Err(HeadGazePolicyParseError::Controller(
                HeadGazeControllerDeclarationParseError::PositiveTiming {
                    field: HeadGazeTimingField::ControlPeriod,
                    ..
                }
            ))
        ));

        let mut ambiguous_lateness = valid_value();
        ambiguous_lateness["controller_declaration"]["timing"]["maximum_tick_lateness_ns"] =
            json!(20_000_000);
        assert!(matches!(
            parse(&ambiguous_lateness),
            Err(HeadGazePolicyParseError::Controller(
                HeadGazeControllerDeclarationParseError::TickLatenessNotBelowControlPeriod { .. }
            ))
        ));

        let mut zero_velocity = valid_value();
        zero_velocity["controller_declaration"]["motion_limits"]["yaw"]["maximum_velocity_ticks_per_control_tick"] =
            json!(0);
        assert!(matches!(
            parse(&zero_velocity),
            Err(HeadGazePolicyParseError::Controller(
                HeadGazeControllerDeclarationParseError::MotionValue {
                    joint: HeadJoint::Yaw,
                    field: HeadGazeMotionField::VelocityPerControlTick,
                    ..
                }
            ))
        ));

        let mut no_hysteresis = valid_value();
        no_hysteresis["controller_declaration"]["error_band"]["resume_threshold_ticks"] = json!(2);
        assert!(matches!(
            parse(&no_hysteresis),
            Err(HeadGazePolicyParseError::Controller(
                HeadGazeControllerDeclarationParseError::ErrorBand { .. }
            ))
        ));
    }

    #[test]
    fn mapping_and_lifecycle_provenance_are_bounded_owned_and_canonical() {
        let mut invalid_mapping_id = valid_value();
        invalid_mapping_id["mapping_declaration"]["calibration_provenance_id"] =
            json!("untrusted evidence");
        assert!(matches!(
            parse(&invalid_mapping_id),
            Err(HeadGazePolicyParseError::Mapping(
                HeadGazeMappingDeclarationParseError::Identifier { .. }
            ))
        ));

        let mut invalid_review_id = valid_value();
        invalid_review_id["lifecycle"]["proposal_id"] = json!("bad claim!");
        assert!(matches!(
            parse(&invalid_review_id),
            Err(HeadGazePolicyParseError::Lifecycle(
                HeadGazeLifecycleClaimParseError::Identifier {
                    field: HeadGazeLifecycleIdentifierField::Proposal,
                    source: HeadGazeLifecycleIdentifierError::InvalidByte { .. }
                }
            ))
        ));

        let mut zero_digest = valid_value();
        zero_digest["lifecycle"]["evidence_content_sha256_hex"] = json!("0".repeat(64));
        assert!(matches!(
            parse(&zero_digest),
            Err(HeadGazePolicyParseError::Lifecycle(
                HeadGazeLifecycleClaimParseError::EvidenceContentSha256 {
                    source: HeadGazeEvidenceContentSha256Error::AllZero
                }
            ))
        ));

        let mut uppercase_digest = valid_value();
        uppercase_digest["lifecycle"]["evidence_content_sha256_hex"] = json!("A".repeat(64));
        assert!(matches!(
            parse(&uppercase_digest),
            Err(HeadGazePolicyParseError::Lifecycle(
                HeadGazeLifecycleClaimParseError::EvidenceContentSha256 {
                    source: HeadGazeEvidenceContentSha256Error::NonCanonicalHex { .. }
                }
            ))
        ));
    }
}
