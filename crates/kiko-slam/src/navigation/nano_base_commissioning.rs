//! Attended, wheel-on base commissioning adapter.
//!
//! This module is intentionally compiled only by the separate
//! `nano-base-commissioning` feature. Production `nano-agent` builds do not
//! contain an entry point to this lane. The adapter also cannot mint its own
//! controller authority: construction requires a token retained by the
//! commissioning-profile and attended-physical-attestation admission path.
//!
//! The deterministic state machine and fitter live in
//! `kiko-base-commissioning`. This module owns only boundary parsing, durable
//! evidence ordering, a sole injected actuator, lateral-observation evidence,
//! and atomic publication of *proposed* artifacts. It never activates a plant.

use std::fmt;
#[cfg(test)]
use std::fs;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::num::{NonZeroU16, NonZeroU32, NonZeroU64};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use kiko_base_commissioning::{
    BASE_IDENTIFICATION_V1, BoundedId, Cancellation, CanonicalPwmCommand, CommissioningAction,
    CommissioningConfigParseError, CommissioningConfigV1, CommissioningConfigV1Dto,
    CommissioningController, CommissioningEvidence, CommissioningEvidenceParseError,
    CommissioningEvidenceV1Dto, CommissioningState, DatasetParseError, FitError,
    IdentificationDatasetV1, IdentificationDatasetV1Dto, IdentificationSampleV1Dto,
    IdentifiedPlantV1, LateralVelocityEvidence, MonotonicTimestampNs, PlantFitConfigParseError,
    PlantFitConfigV1, PlantFitConfigV1Dto, fit_first_order_plant,
};
use robot_protocol::{AppliedPwm, AppliedPwmError};
use rustix::fs::{
    AtFlags, FileType, Mode, OFlags, RenameFlags, fstat, fsync, openat, renameat_with, statat,
    unlinkat,
};
use rustix::io::Errno;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::mpc::{
    FitResidualsV1Dto, PLANT_MODEL_V1, PlantEvidenceV1Dto, PlantModelParseError, PlantModelV1,
    PlantModelV1Dto, PlantValidityEnvelopeV1Dto, WheelPlantV1Dto,
};

pub const NANO_BASE_COMMISSIONING_POLICY_V1: u32 = 1;
pub const NANO_BASE_COMMISSIONING_ARTIFACT_V1: u32 = 1;
pub const MAX_NANO_BASE_COMMISSIONING_POLICY_JSON_BYTES: usize = 128 * 1_024;
pub const MAX_COMMISSIONING_JOURNAL_RECORD_BYTES: usize = 8 * 1_024;
pub const MAX_COMMISSIONING_ARTIFACT_BYTES: usize = 64 * 1_024 * 1_024;
const CONTENT_DIGEST_BYTES: usize = 32;
const SHA256_PREFIX: &str = "sha256:";
const LATERAL_METHOD_ID: &str = "visual-body-lateral-training-max-margin-holdout-v1";
const PROPOSAL_ACTIVATION_STATUS: &str = "proposed_unapproved";
const BODY_FRAME_ID: &str = "base_body_flu";
const MINIMUM_POLICY_SAMPLES: u32 = 3;
static ARTIFACT_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// Strict immutable policy. The exact source bytes are hashed at parse time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NanoBaseCommissioningPolicyV1 {
    commissioning: CommissioningConfigV1,
    fit: PlantFitConfigV1,
    lateral: LateralHoldoutPolicyV1,
    maximum_aligned_observation_skew_ns: NonZeroU64,
    maximum_sample_gap_ns: NonZeroU64,
    maximum_controller_sequence_gap: NonZeroU64,
    model_version: NonZeroU32,
    model_id: CommissioningLabel,
    content_sha256: [u8; CONTENT_DIGEST_BYTES],
}

impl NanoBaseCommissioningPolicyV1 {
    pub fn parse_json(bytes: &[u8]) -> Result<Self, NanoBaseCommissioningPolicyParseError> {
        if bytes.len() > MAX_NANO_BASE_COMMISSIONING_POLICY_JSON_BYTES {
            return Err(NanoBaseCommissioningPolicyParseError::InputTooLarge {
                actual_bytes: bytes.len(),
                maximum_bytes: MAX_NANO_BASE_COMMISSIONING_POLICY_JSON_BYTES,
            });
        }
        let mut deserializer = serde_json::Deserializer::from_slice(bytes);
        let dto = NanoBaseCommissioningPolicyV1Dto::deserialize(&mut deserializer)
            .map_err(NanoBaseCommissioningPolicyParseError::JsonDecode)?;
        deserializer
            .end()
            .map_err(NanoBaseCommissioningPolicyParseError::JsonTrailingData)?;
        if dto.schema_version != NANO_BASE_COMMISSIONING_POLICY_V1 {
            return Err(NanoBaseCommissioningPolicyParseError::UnsupportedSchema {
                actual: dto.schema_version,
                supported: NANO_BASE_COMMISSIONING_POLICY_V1,
            });
        }
        let commissioning = CommissioningConfigV1::parse(dto.commissioning)
            .map_err(NanoBaseCommissioningPolicyParseError::Commissioning)?;
        let fit =
            PlantFitConfigV1::parse(dto.fit).map_err(NanoBaseCommissioningPolicyParseError::Fit)?;
        for (field, expected, actual) in [
            (
                "controller_session_id",
                commissioning.expected_controller_session_id(),
                fit.expected_controller_session_id(),
            ),
            (
                "visual_velocity_source_id",
                commissioning.expected_visual_velocity_source_id(),
                fit.expected_visual_velocity_source_id(),
            ),
            (
                "imu_calibration_id",
                commissioning.expected_imu_calibration_id(),
                fit.expected_imu_calibration_id(),
            ),
        ] {
            if expected != actual {
                return Err(NanoBaseCommissioningPolicyParseError::CoreIdentityMismatch(
                    Box::new(NanoBaseCommissioningCoreIdentityMismatch {
                        field,
                        commissioning: expected,
                        fit: actual,
                    }),
                ));
            }
        }
        let lateral = LateralHoldoutPolicyV1::parse(dto.lateral)?;
        let lateral_required_samples = lateral.required_samples();
        if lateral_required_samples > fit.max_samples().get() {
            return Err(
                NanoBaseCommissioningPolicyParseError::LateralSampleRequirementAboveFitCapacity {
                    required: lateral_required_samples,
                    maximum: fit.max_samples().get(),
                },
            );
        }
        let maximum_aligned_observation_skew_ns = nonzero_policy(
            "maximum_aligned_observation_skew_ns",
            dto.maximum_aligned_observation_skew_ns,
        )?;
        let maximum_sample_gap_ns =
            nonzero_policy("maximum_sample_gap_ns", dto.maximum_sample_gap_ns)?;
        let maximum_controller_sequence_gap = nonzero_policy(
            "maximum_controller_sequence_gap",
            dto.maximum_controller_sequence_gap,
        )?;
        let model_version = NonZeroU32::new(dto.model_version).ok_or(
            NanoBaseCommissioningPolicyParseError::Zero {
                field: "model_version",
            },
        )?;
        let model_id = CommissioningLabel::parse("model_id", dto.model_id)?;
        Ok(Self {
            commissioning,
            fit,
            lateral,
            maximum_aligned_observation_skew_ns,
            maximum_sample_gap_ns,
            maximum_controller_sequence_gap,
            model_version,
            model_id,
            content_sha256: sha256(bytes),
        })
    }

    pub const fn content_sha256(self) -> [u8; CONTENT_DIGEST_BYTES] {
        self.content_sha256
    }

    pub const fn commissioning(self) -> CommissioningConfigV1 {
        self.commissioning
    }

    pub const fn fit(self) -> PlantFitConfigV1 {
        self.fit
    }

    /// Stable model identity copied into an independently reviewed plant.
    pub fn model_id(&self) -> &str {
        self.model_id.as_str()
    }

    pub const fn model_version(self) -> NonZeroU32 {
        self.model_version
    }

    pub const fn lateral_holdout_stride(self) -> NonZeroU16 {
        self.lateral.holdout_stride
    }

    pub const fn lateral_minimum_training_samples(self) -> NonZeroU32 {
        self.lateral.minimum_training_samples
    }

    pub const fn lateral_minimum_holdout_samples(self) -> NonZeroU32 {
        self.lateral.minimum_holdout_samples
    }

    pub const fn lateral_bound_margin_mps(self) -> f64 {
        self.lateral.bound_margin_mps
    }

    pub const fn lateral_maximum_accepted_bound_mps(self) -> f64 {
        self.lateral.maximum_accepted_bound_mps
    }

    pub fn lateral_scope_label(&self) -> &str {
        self.lateral.scope_label.as_str()
    }

    pub const fn maximum_sample_gap_ns(self) -> NonZeroU64 {
        self.maximum_sample_gap_ns
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NanoBaseCommissioningPolicyV1Dto {
    schema_version: u32,
    commissioning: CommissioningConfigV1Dto,
    fit: PlantFitConfigV1Dto,
    lateral: LateralHoldoutPolicyV1Dto,
    maximum_aligned_observation_skew_ns: u64,
    maximum_sample_gap_ns: u64,
    maximum_controller_sequence_gap: u64,
    model_id: String,
    model_version: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LateralHoldoutPolicyV1 {
    holdout_stride: NonZeroU16,
    minimum_training_samples: NonZeroU32,
    minimum_holdout_samples: NonZeroU32,
    bound_margin_mps: f64,
    maximum_accepted_bound_mps: f64,
    maximum_abs_observed_lateral_velocity_mps: f64,
    scope_label: CommissioningLabel,
}

impl LateralHoldoutPolicyV1 {
    fn parse(
        dto: LateralHoldoutPolicyV1Dto,
    ) -> Result<Self, NanoBaseCommissioningPolicyParseError> {
        if dto.body_frame_id != BODY_FRAME_ID {
            return Err(NanoBaseCommissioningPolicyParseError::UnsupportedBodyFrame(
                dto.body_frame_id,
            ));
        }
        let holdout_stride = NonZeroU16::new(dto.holdout_stride).ok_or(
            NanoBaseCommissioningPolicyParseError::Zero {
                field: "lateral.holdout_stride",
            },
        )?;
        if holdout_stride.get() < 2 {
            return Err(NanoBaseCommissioningPolicyParseError::IntegerOutOfRange {
                field: "lateral.holdout_stride",
                value: u64::from(holdout_stride.get()),
                minimum: 2,
            });
        }
        let minimum_training_samples = NonZeroU32::new(dto.minimum_training_samples).ok_or(
            NanoBaseCommissioningPolicyParseError::Zero {
                field: "lateral.minimum_training_samples",
            },
        )?;
        let minimum_holdout_samples = NonZeroU32::new(dto.minimum_holdout_samples).ok_or(
            NanoBaseCommissioningPolicyParseError::Zero {
                field: "lateral.minimum_holdout_samples",
            },
        )?;
        let required_samples = minimum_training_samples
            .get()
            .checked_add(minimum_holdout_samples.get())
            .ok_or(NanoBaseCommissioningPolicyParseError::SampleRequirementOverflow)?;
        if required_samples < MINIMUM_POLICY_SAMPLES {
            return Err(NanoBaseCommissioningPolicyParseError::IntegerOutOfRange {
                field: "lateral.total_required_samples",
                value: u64::from(required_samples),
                minimum: u64::from(MINIMUM_POLICY_SAMPLES),
            });
        }
        finite_nonnegative_policy("lateral.bound_margin_mps", dto.bound_margin_mps)?;
        finite_positive_policy(
            "lateral.maximum_accepted_bound_mps",
            dto.maximum_accepted_bound_mps,
        )?;
        finite_positive_policy(
            "lateral.maximum_abs_observed_lateral_velocity_mps",
            dto.maximum_abs_observed_lateral_velocity_mps,
        )?;
        if dto.bound_margin_mps > dto.maximum_accepted_bound_mps {
            return Err(
                NanoBaseCommissioningPolicyParseError::LateralMarginAboveBound {
                    margin_mps: dto.bound_margin_mps,
                    maximum_bound_mps: dto.maximum_accepted_bound_mps,
                },
            );
        }
        Ok(Self {
            holdout_stride,
            minimum_training_samples,
            minimum_holdout_samples,
            bound_margin_mps: dto.bound_margin_mps,
            maximum_accepted_bound_mps: dto.maximum_accepted_bound_mps,
            maximum_abs_observed_lateral_velocity_mps: dto
                .maximum_abs_observed_lateral_velocity_mps,
            scope_label: CommissioningLabel::parse("lateral.scope_label", dto.scope_label)?,
        })
    }

    fn required_samples(self) -> u32 {
        self.minimum_training_samples
            .get()
            .checked_add(self.minimum_holdout_samples.get())
            .expect("sum was checked while parsing the lateral policy")
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LateralHoldoutPolicyV1Dto {
    body_frame_id: String,
    holdout_stride: u16,
    minimum_training_samples: u32,
    minimum_holdout_samples: u32,
    bound_margin_mps: f64,
    maximum_accepted_bound_mps: f64,
    maximum_abs_observed_lateral_velocity_mps: f64,
    scope_label: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct CommissioningLabel {
    bytes: [u8; 64],
    length: u8,
}

impl CommissioningLabel {
    fn parse(
        field: &'static str,
        value: String,
    ) -> Result<Self, NanoBaseCommissioningPolicyParseError> {
        if value.is_empty()
            || value.len() > 64
            || !value.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
            })
        {
            return Err(NanoBaseCommissioningPolicyParseError::InvalidLabel { field });
        }
        let mut bytes = [0; 64];
        bytes[..value.len()].copy_from_slice(value.as_bytes());
        Ok(Self {
            bytes,
            length: u8::try_from(value.len()).expect("label length checked"),
        })
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..usize::from(self.length)])
            .expect("commissioning label is checked ASCII")
    }
}

impl fmt::Debug for CommissioningLabel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("CommissioningLabel")
            .field(&self.as_str())
            .finish()
    }
}

#[derive(Debug)]
pub enum NanoBaseCommissioningPolicyParseError {
    InputTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    JsonDecode(serde_json::Error),
    JsonTrailingData(serde_json::Error),
    UnsupportedSchema {
        actual: u32,
        supported: u32,
    },
    Commissioning(CommissioningConfigParseError),
    Fit(PlantFitConfigParseError),
    CoreIdentityMismatch(Box<NanoBaseCommissioningCoreIdentityMismatch>),
    UnsupportedBodyFrame(String),
    Zero {
        field: &'static str,
    },
    IntegerOutOfRange {
        field: &'static str,
        value: u64,
        minimum: u64,
    },
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
    LateralMarginAboveBound {
        margin_mps: f64,
        maximum_bound_mps: f64,
    },
    LateralSampleRequirementAboveFitCapacity {
        required: u32,
        maximum: u32,
    },
    SampleRequirementOverflow,
    InvalidLabel {
        field: &'static str,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoBaseCommissioningCoreIdentityMismatch {
    pub field: &'static str,
    pub commissioning: BoundedId,
    pub fit: BoundedId,
}

impl fmt::Display for NanoBaseCommissioningPolicyParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Nano base-commissioning policy: {self:?}"
        )
    }
}

impl std::error::Error for NanoBaseCommissioningPolicyParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::JsonDecode(source) | Self::JsonTrailingData(source) => Some(source),
            Self::Commissioning(source) => Some(source),
            Self::Fit(source) => Some(source),
            _ => None,
        }
    }
}

fn nonzero_policy(
    field: &'static str,
    value: u64,
) -> Result<NonZeroU64, NanoBaseCommissioningPolicyParseError> {
    NonZeroU64::new(value).ok_or(NanoBaseCommissioningPolicyParseError::Zero { field })
}

fn finite_positive_policy(
    field: &'static str,
    value: f64,
) -> Result<(), NanoBaseCommissioningPolicyParseError> {
    if !value.is_finite() {
        return Err(NanoBaseCommissioningPolicyParseError::NonFinite { field, value });
    }
    if value <= 0.0 {
        return Err(NanoBaseCommissioningPolicyParseError::NotPositive { field, value });
    }
    Ok(())
}

fn finite_nonnegative_policy(
    field: &'static str,
    value: f64,
) -> Result<(), NanoBaseCommissioningPolicyParseError> {
    if !value.is_finite() {
        return Err(NanoBaseCommissioningPolicyParseError::NonFinite { field, value });
    }
    if value < 0.0 {
        return Err(NanoBaseCommissioningPolicyParseError::Negative { field, value });
    }
    Ok(())
}

/// Opaque result of the future commissioning-specific controller-profile and
/// attended physical-attestation admission.
///
/// There is deliberately no public constructor. The production admission
/// path does not manufacture this token, and `nano-agent` does not compile
/// this module.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmittedAttendedCommissioning {
    controller_session_id: BoundedId,
    controller_profile_sha256: [u8; CONTENT_DIGEST_BYTES],
    physical_attestation_sha256: [u8; CONTENT_DIGEST_BYTES],
    maximum_abs_pwm_percent: u8,
    issued_at_ns: u64,
    expires_at_ns: u64,
}

impl AdmittedAttendedCommissioning {
    /// Construct the scoped authority after the immutable controller profile
    /// and one-shot attended physical attestation have both been verified and
    /// consumed.
    #[cfg(any(feature = "nano-base-commissioning", test))]
    pub(super) fn from_verified_attended_admission(
        controller_session_id: BoundedId,
        controller_profile_sha256: [u8; CONTENT_DIGEST_BYTES],
        physical_attestation_sha256: [u8; CONTENT_DIGEST_BYTES],
        maximum_abs_pwm_percent: u8,
        issued_at_ns: u64,
        expires_at_ns: u64,
    ) -> Result<Self, CommissioningAuthorityError> {
        if maximum_abs_pwm_percent == 0 || maximum_abs_pwm_percent > 100 {
            return Err(CommissioningAuthorityError::InvalidMaximumPwm(
                maximum_abs_pwm_percent,
            ));
        }
        if expires_at_ns <= issued_at_ns {
            return Err(CommissioningAuthorityError::InvalidLifetime {
                issued_at_ns,
                expires_at_ns,
            });
        }
        Ok(Self {
            controller_session_id,
            controller_profile_sha256,
            physical_attestation_sha256,
            maximum_abs_pwm_percent,
            issued_at_ns,
            expires_at_ns,
        })
    }

    fn verify_policy(
        self,
        policy: NanoBaseCommissioningPolicyV1,
    ) -> Result<(), CommissioningAuthorityError> {
        if self.controller_session_id != policy.commissioning.expected_controller_session_id() {
            return Err(CommissioningAuthorityError::ControllerSessionMismatch);
        }
        if policy.commissioning.max_abs_pwm_percent().get() > self.maximum_abs_pwm_percent {
            return Err(CommissioningAuthorityError::PolicyPwmAboveAdmittedProfile {
                policy: policy.commissioning.max_abs_pwm_percent().get(),
                admitted: self.maximum_abs_pwm_percent,
            });
        }
        Ok(())
    }

    fn require_fresh(self, now_ns: u64) -> Result<(), CommissioningAuthorityError> {
        if now_ns < self.issued_at_ns {
            return Err(CommissioningAuthorityError::ClockBeforeAttestation {
                now_ns,
                issued_at_ns: self.issued_at_ns,
            });
        }
        if now_ns >= self.expires_at_ns {
            return Err(CommissioningAuthorityError::AttestationExpired {
                now_ns,
                expires_at_ns: self.expires_at_ns,
            });
        }
        Ok(())
    }

    #[cfg(feature = "nano-attended-navigation-trial")]
    pub(super) fn attended_navigation_trial_guard(
        self,
    ) -> Result<
        super::actuation::AttendedTrialActuationGuard,
        super::actuation::AttendedTrialActuationGuardError,
    > {
        super::actuation::AttendedTrialActuationGuard::try_new(
            self.maximum_abs_pwm_percent,
            self.issued_at_ns,
            self.expires_at_ns,
        )
    }

    #[cfg(feature = "nano-attended-navigation-trial")]
    pub const fn maximum_abs_pwm_percent(self) -> u8 {
        self.maximum_abs_pwm_percent
    }

    #[cfg(feature = "nano-attended-navigation-trial")]
    pub const fn expires_at_ns(self) -> u64 {
        self.expires_at_ns
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommissioningAuthorityError {
    InvalidMaximumPwm(u8),
    InvalidLifetime {
        issued_at_ns: u64,
        expires_at_ns: u64,
    },
    ControllerSessionMismatch,
    PolicyPwmAboveAdmittedProfile {
        policy: u8,
        admitted: u8,
    },
    ClockBeforeAttestation {
        now_ns: u64,
        issued_at_ns: u64,
    },
    AttestationExpired {
        now_ns: u64,
        expires_at_ns: u64,
    },
}

impl fmt::Display for CommissioningAuthorityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid attended commissioning authority: {self:?}"
        )
    }
}

impl std::error::Error for CommissioningAuthorityError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExactCommissioningControllerReceipt {
    observed_at_ns: u64,
    applied_command_sequence: u64,
    applied_pwm: AppliedPwm,
}

impl ExactCommissioningControllerReceipt {
    pub fn try_new(
        observed_at_ns: u64,
        applied_command_sequence: u64,
        applied_left_pwm_percent: i8,
        applied_right_pwm_percent: i8,
    ) -> Result<Self, AppliedPwmError> {
        Ok(Self {
            observed_at_ns,
            applied_command_sequence,
            applied_pwm: AppliedPwm::try_new(applied_left_pwm_percent, applied_right_pwm_percent)?,
        })
    }

    pub const fn observed_at_ns(self) -> u64 {
        self.observed_at_ns
    }

    pub const fn applied_command_sequence(self) -> u64 {
        self.applied_command_sequence
    }

    pub const fn applied_pwm(self) -> AppliedPwm {
        self.applied_pwm
    }

    fn is_exact_zero(self) -> bool {
        self.applied_pwm.left().get() == 0 && self.applied_pwm.right().get() == 0
    }
}

/// Sole actuator interface injected only after commissioning admission.
///
/// A successful call is an exact controller receipt, not an acknowledgement
/// that a request was merely queued.
pub trait SoleCommissioningActuator {
    type Error: std::error::Error + Send + Sync + 'static;

    fn apply(
        &mut self,
        command: CanonicalPwmCommand,
    ) -> Result<ExactCommissioningControllerReceipt, Self::Error>;

    fn emergency_zero(&mut self) -> Result<ExactCommissioningControllerReceipt, Self::Error>;
}

/// Journal implementations must make `record` durable before returning.
pub trait DurableCommissioningJournal {
    type Error: std::error::Error + Send + Sync + 'static;

    fn append_durable(&mut self, record: &[u8]) -> Result<(), Self::Error>;
    fn finalize_durable(&mut self) -> Result<CommissioningJournalCommit, Self::Error>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommissioningJournalCommit {
    content_sha256: [u8; CONTENT_DIGEST_BYTES],
    record_count: u64,
    byte_count: u64,
}

impl CommissioningJournalCommit {
    pub const fn content_sha256(self) -> [u8; CONTENT_DIGEST_BYTES] {
        self.content_sha256
    }

    pub const fn record_count(self) -> u64 {
        self.record_count
    }

    pub const fn byte_count(self) -> u64 {
        self.byte_count
    }
}

/// Durable newline-delimited JSON evidence. Every append is synchronized
/// before the adapter may advance the commissioning state machine.
pub struct FileCommissioningJournal {
    file: File,
    hasher: Sha256,
    record_count: u64,
    byte_count: u64,
}

impl FileCommissioningJournal {
    pub fn create_new(path: &Path) -> Result<Self, FileCommissioningJournalError> {
        let parent_path = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let file_name = path
            .file_name()
            .ok_or(FileCommissioningJournalError::InvalidPath)?;
        let parent =
            open_directory_file(parent_path).map_err(FileCommissioningJournalError::Create)?;
        Self::create_new_at(&parent, file_name)
    }

    pub(crate) fn create_new_at(
        parent: &File,
        file_name: &std::ffi::OsStr,
    ) -> Result<Self, FileCommissioningJournalError> {
        if !is_simple_name(file_name) {
            return Err(FileCommissioningJournalError::InvalidPath);
        }
        let file = openat(
            parent,
            file_name,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        )
        .map(File::from)
        .map_err(|source| FileCommissioningJournalError::Create(errno_as_io(source)))?;
        let metadata = file
            .metadata()
            .map_err(FileCommissioningJournalError::Inspect)?;
        if !metadata.file_type().is_file()
            || metadata.nlink() != 1
            || metadata.uid() != current_uid()
            || metadata.mode() & 0o777 != 0o600
        {
            return Err(FileCommissioningJournalError::UnsafeCreatedFile);
        }
        parent
            .sync_all()
            .map_err(FileCommissioningJournalError::SyncParent)?;
        Ok(Self {
            file,
            hasher: Sha256::new(),
            record_count: 0,
            byte_count: 0,
        })
    }
}

impl DurableCommissioningJournal for FileCommissioningJournal {
    type Error = FileCommissioningJournalError;

    fn append_durable(&mut self, record: &[u8]) -> Result<(), Self::Error> {
        if record.len() > MAX_COMMISSIONING_JOURNAL_RECORD_BYTES {
            return Err(FileCommissioningJournalError::RecordTooLarge {
                actual_bytes: record.len(),
                maximum_bytes: MAX_COMMISSIONING_JOURNAL_RECORD_BYTES,
            });
        }
        self.file
            .write_all(record)
            .map_err(FileCommissioningJournalError::Write)?;
        self.file
            .sync_data()
            .map_err(FileCommissioningJournalError::Sync)?;
        self.hasher.update(record);
        self.record_count = self
            .record_count
            .checked_add(1)
            .ok_or(FileCommissioningJournalError::RecordCountOverflow)?;
        self.byte_count = self
            .byte_count
            .checked_add(
                u64::try_from(record.len())
                    .map_err(|_| FileCommissioningJournalError::ByteCountOverflow)?,
            )
            .ok_or(FileCommissioningJournalError::ByteCountOverflow)?;
        Ok(())
    }

    fn finalize_durable(&mut self) -> Result<CommissioningJournalCommit, Self::Error> {
        self.file
            .sync_all()
            .map_err(FileCommissioningJournalError::Sync)?;
        Ok(CommissioningJournalCommit {
            content_sha256: self.hasher.clone().finalize().into(),
            record_count: self.record_count,
            byte_count: self.byte_count,
        })
    }
}

#[derive(Debug)]
pub enum FileCommissioningJournalError {
    InvalidPath,
    Create(io::Error),
    Inspect(io::Error),
    SyncParent(io::Error),
    UnsafeCreatedFile,
    RecordTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    Write(io::Error),
    Sync(io::Error),
    RecordCountOverflow,
    ByteCountOverflow,
}

impl fmt::Display for FileCommissioningJournalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "commissioning journal failure: {self:?}")
    }
}

impl std::error::Error for FileCommissioningJournalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Create(source)
            | Self::Inspect(source)
            | Self::SyncParent(source)
            | Self::Write(source)
            | Self::Sync(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NanoBaseCommissioningSampleV1Dto {
    pub now_ns: u64,
    pub evidence: CommissioningEvidenceV1Dto,
    pub visual_body_lateral_velocity_mps: f64,
    pub visual_body_frame_id: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommissioningExternalSignal {
    Continue,
    CancelledByOperator,
    ControllerFault,
    VisualFault,
    ImuFault,
    SupervisorFault,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NanoBaseCommissioningProgress {
    pub state: CommissioningState,
    pub samples_journaled: u32,
    pub requested_left_pwm_percent: i8,
    pub requested_right_pwm_percent: i8,
    pub last_applied_sequence: u64,
    pub exact_zero_applied: bool,
}

#[derive(Debug)]
pub enum ExactFailClosedStop {
    Applied {
        receipt: ExactCommissioningControllerReceipt,
    },
    ReceiptRejected {
        previous: Option<ExactCommissioningControllerReceipt>,
        receipt: ExactCommissioningControllerReceipt,
    },
    CommandFailed {
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    NotAttempted,
}

impl ExactFailClosedStop {
    pub const fn is_confirmed(&self) -> bool {
        matches!(self, Self::Applied { .. })
    }
}

#[derive(Debug)]
pub enum CommissioningTerminalJournalStatus {
    NotAttempted,
    Durable,
    Failed(Box<NanoBaseCommissioningFault>),
}

#[derive(Debug)]
pub struct NanoBaseCommissioningFailure {
    pub fault: NanoBaseCommissioningFault,
    pub stop: ExactFailClosedStop,
    pub terminal_journal: CommissioningTerminalJournalStatus,
}

impl fmt::Display for NanoBaseCommissioningFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "base commissioning failed closed: {}; stop={:?}; terminal_journal={:?}",
            self.fault, self.stop, self.terminal_journal
        )
    }
}

impl std::error::Error for NanoBaseCommissioningFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.fault)
    }
}

#[derive(Debug)]
pub enum NanoBaseCommissioningFault {
    Authority(CommissioningAuthorityError),
    Evidence(CommissioningEvidenceParseError),
    UnsupportedVisualFrame(String),
    LateralVelocityNonFinite(f64),
    LateralVelocityOutsidePolicy {
        observed_mps: f64,
        maximum_abs_mps: f64,
    },
    AlignedObservationSkew {
        minimum_ns: u64,
        maximum_ns: u64,
        allowed_ns: u64,
    },
    SampleClockDidNotAdvance {
        previous_ns: u64,
        current_ns: u64,
    },
    SampleGapTooShort {
        previous_ns: u64,
        current_ns: u64,
        minimum_ns: u64,
    },
    SampleGap {
        previous_ns: u64,
        current_ns: u64,
        maximum_ns: u64,
    },
    ControllerSequenceGap {
        previous: u64,
        current: u64,
        maximum_gap: u64,
    },
    AppliedReceiptMismatch {
        expected: ExactCommissioningControllerReceipt,
        observed_sequence: u64,
        observed_pwm: AppliedPwm,
    },
    ExternalSignal(CommissioningExternalSignal),
    JournalEncode(serde_json::Error),
    Journal(Box<dyn std::error::Error + Send + Sync>),
    Actuator(Box<dyn std::error::Error + Send + Sync>),
    AppliedCommandMismatch {
        requested: CanonicalPwmCommand,
        receipt: ExactCommissioningControllerReceipt,
    },
    AppliedSequenceDidNotAdvance {
        previous: u64,
        current: u64,
    },
    CoreAborted(kiko_base_commissioning::CommissioningStopReason),
    SampleStorageAllocation,
    SampleCountOverflow,
    SampleLimitReached {
        accepted: u32,
        maximum: u32,
    },
    AlreadyTerminated,
    TerminateRequiresTerminalSignal,
}

impl fmt::Display for NanoBaseCommissioningFault {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Nano base-commissioning fault: {self:?}")
    }
}

impl std::error::Error for NanoBaseCommissioningFault {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Authority(source) => Some(source),
            Self::Evidence(source) => Some(source),
            Self::JournalEncode(source) => Some(source),
            Self::Journal(source) | Self::Actuator(source) => Some(source.as_ref()),
            _ => None,
        }
    }
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CommissioningJournalRecord<'a> {
    SessionStarted {
        schema_version: u32,
        policy_sha256: String,
        controller_profile_sha256: String,
        physical_attestation_sha256: String,
        controller_session_id: &'a str,
        activation_status: &'static str,
    },
    AlignedObservation {
        sample_ordinal: u32,
        now_ns: u64,
        controller_observed_at_ns: u64,
        visual_observed_at_ns: u64,
        imu_observed_at_ns: u64,
        applied_command_sequence: u64,
        applied_left_pwm_percent: i8,
        applied_right_pwm_percent: i8,
        visual_body_forward_velocity_mps: f64,
        visual_body_lateral_velocity_mps: f64,
        calibrated_imu_yaw_rate_rad_s: f64,
        visual_body_frame_id: &'static str,
    },
    RequestedPwm {
        sample_ordinal: u32,
        left_pwm_percent: i8,
        right_pwm_percent: i8,
    },
    AppliedReceipt {
        sample_ordinal: u32,
        observed_at_ns: u64,
        applied_command_sequence: u64,
        applied_left_pwm_percent: i8,
        applied_right_pwm_percent: i8,
    },
    Terminal {
        sample_ordinal: u32,
        reason: &'a str,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct CollectedSample {
    identification: IdentificationSampleV1Dto,
    visual_body_lateral_velocity_mps: f64,
    holdout: bool,
}

/// Feature-isolated attended commissioning runtime.
#[must_use = "an attended commissioning session must be explicitly completed or terminalized"]
pub struct NanoBaseCommissioningSession<A, J>
where
    A: SoleCommissioningActuator,
    J: DurableCommissioningJournal,
{
    policy: NanoBaseCommissioningPolicyV1,
    authority: AdmittedAttendedCommissioning,
    controller: CommissioningController,
    actuator: A,
    journal: J,
    samples: Vec<CollectedSample>,
    expected_receipt: ExactCommissioningControllerReceipt,
    last_sample_observed_at_ns: Option<u64>,
    last_observed_sequence: Option<u64>,
    journal_hasher: Sha256,
    journal_records: u64,
    journal_bytes: u64,
    terminated: bool,
    completed: bool,
}

impl<A, J> NanoBaseCommissioningSession<A, J>
where
    A: SoleCommissioningActuator,
    J: DurableCommissioningJournal,
{
    pub fn start(
        policy: NanoBaseCommissioningPolicyV1,
        authority: AdmittedAttendedCommissioning,
        mut actuator: A,
        mut journal: J,
    ) -> Result<Self, NanoBaseCommissioningFailure> {
        if let Err(source) = authority.verify_policy(policy) {
            let stop = emergency_zero_evidence(&mut actuator, None);
            return Err(NanoBaseCommissioningFailure {
                fault: NanoBaseCommissioningFault::Authority(source),
                stop,
                terminal_journal: CommissioningTerminalJournalStatus::NotAttempted,
            });
        }
        let start = CommissioningJournalRecord::SessionStarted {
            schema_version: NANO_BASE_COMMISSIONING_ARTIFACT_V1,
            policy_sha256: canonical_sha256(policy.content_sha256),
            controller_profile_sha256: canonical_sha256(authority.controller_profile_sha256),
            physical_attestation_sha256: canonical_sha256(authority.physical_attestation_sha256),
            controller_session_id: authority.controller_session_id.as_str(),
            activation_status: PROPOSAL_ACTIVATION_STATUS,
        };
        let mut journal_hasher = Sha256::new();
        let mut journal_records = 0_u64;
        let mut journal_bytes = 0_u64;
        if let Err(fault) = append_journal(
            &mut journal,
            &mut journal_hasher,
            &mut journal_records,
            &mut journal_bytes,
            &start,
        ) {
            let stop = emergency_zero_evidence(&mut actuator, None);
            return Err(NanoBaseCommissioningFailure {
                fault,
                stop,
                terminal_journal: CommissioningTerminalJournalStatus::NotAttempted,
            });
        }
        let requested_zero = CommissioningJournalRecord::RequestedPwm {
            sample_ordinal: 0,
            left_pwm_percent: 0,
            right_pwm_percent: 0,
        };
        if let Err(fault) = append_journal(
            &mut journal,
            &mut journal_hasher,
            &mut journal_records,
            &mut journal_bytes,
            &requested_zero,
        ) {
            let stop = emergency_zero_evidence(&mut actuator, None);
            return Err(NanoBaseCommissioningFailure {
                fault,
                stop,
                terminal_journal: CommissioningTerminalJournalStatus::NotAttempted,
            });
        }
        let initial_zero = match actuator.emergency_zero() {
            Ok(receipt) if receipt.is_exact_zero() => receipt,
            Ok(receipt) => {
                let stop = emergency_zero_evidence(&mut actuator, Some(receipt));
                return Err(NanoBaseCommissioningFailure {
                    fault: NanoBaseCommissioningFault::AppliedCommandMismatch {
                        requested: zero_command_for_error(),
                        receipt,
                    },
                    stop,
                    terminal_journal: CommissioningTerminalJournalStatus::NotAttempted,
                });
            }
            Err(source) => {
                let stop = emergency_zero_evidence(&mut actuator, None);
                return Err(NanoBaseCommissioningFailure {
                    fault: NanoBaseCommissioningFault::Actuator(Box::new(source)),
                    stop,
                    terminal_journal: CommissioningTerminalJournalStatus::NotAttempted,
                });
            }
        };
        let initial_receipt_record = receipt_record(0, initial_zero);
        if let Err(fault) = append_journal(
            &mut journal,
            &mut journal_hasher,
            &mut journal_records,
            &mut journal_bytes,
            &initial_receipt_record,
        ) {
            let stop = emergency_zero_evidence(&mut actuator, Some(initial_zero));
            return Err(NanoBaseCommissioningFailure {
                fault,
                stop,
                terminal_journal: CommissioningTerminalJournalStatus::NotAttempted,
            });
        }
        Ok(Self {
            policy,
            authority,
            controller: CommissioningController::new(policy.commissioning),
            actuator,
            journal,
            samples: Vec::new(),
            expected_receipt: initial_zero,
            last_sample_observed_at_ns: None,
            last_observed_sequence: None,
            journal_hasher,
            journal_records,
            journal_bytes,
            terminated: false,
            completed: false,
        })
    }

    pub fn advance(
        &mut self,
        dto: NanoBaseCommissioningSampleV1Dto,
        signal: CommissioningExternalSignal,
    ) -> Result<NanoBaseCommissioningProgress, NanoBaseCommissioningFailure> {
        if self.terminated {
            return Err(NanoBaseCommissioningFailure {
                fault: NanoBaseCommissioningFault::AlreadyTerminated,
                stop: ExactFailClosedStop::NotAttempted,
                terminal_journal: CommissioningTerminalJournalStatus::NotAttempted,
            });
        }
        if signal != CommissioningExternalSignal::Continue {
            return Err(self.fail_closed(NanoBaseCommissioningFault::ExternalSignal(signal)));
        }
        let accepted_samples = match u32::try_from(self.samples.len()) {
            Ok(value) => value,
            Err(_) => {
                return Err(self.fail_closed(NanoBaseCommissioningFault::SampleCountOverflow));
            }
        };
        let maximum_samples = self.policy.fit.max_samples().get();
        if accepted_samples >= maximum_samples {
            return Err(
                self.fail_closed(NanoBaseCommissioningFault::SampleLimitReached {
                    accepted: accepted_samples,
                    maximum: maximum_samples,
                }),
            );
        }
        if let Err(source) = self.authority.require_fresh(dto.now_ns) {
            return Err(self.fail_closed(NanoBaseCommissioningFault::Authority(source)));
        }

        let now_ns = dto.now_ns;
        let lateral_velocity_mps = dto.visual_body_lateral_velocity_mps;
        if dto.visual_body_frame_id != BODY_FRAME_ID {
            return Err(
                self.fail_closed(NanoBaseCommissioningFault::UnsupportedVisualFrame(
                    dto.visual_body_frame_id,
                )),
            );
        }
        if !lateral_velocity_mps.is_finite() {
            return Err(
                self.fail_closed(NanoBaseCommissioningFault::LateralVelocityNonFinite(
                    lateral_velocity_mps,
                )),
            );
        }
        if lateral_velocity_mps.abs()
            > self
                .policy
                .lateral
                .maximum_abs_observed_lateral_velocity_mps
        {
            return Err(self.fail_closed(
                NanoBaseCommissioningFault::LateralVelocityOutsidePolicy {
                    observed_mps: lateral_velocity_mps,
                    maximum_abs_mps: self
                        .policy
                        .lateral
                        .maximum_abs_observed_lateral_velocity_mps,
                },
            ));
        }

        let evidence = match CommissioningEvidence::parse(dto.evidence, self.policy.commissioning) {
            Ok(evidence) => evidence,
            Err(source) => {
                return Err(self.fail_closed(NanoBaseCommissioningFault::Evidence(source)));
            }
        };
        if let Err(fault) = self.validate_stream(evidence) {
            return Err(self.fail_closed(fault));
        }
        let sample_ordinal = accepted_samples;
        let observation_record = CommissioningJournalRecord::AlignedObservation {
            sample_ordinal,
            now_ns,
            controller_observed_at_ns: evidence.controller_observed_at().as_nanos(),
            visual_observed_at_ns: evidence.visual_observed_at().as_nanos(),
            imu_observed_at_ns: evidence.imu_observed_at().as_nanos(),
            applied_command_sequence: evidence.applied_command_sequence(),
            applied_left_pwm_percent: evidence.applied_pwm().left().get(),
            applied_right_pwm_percent: evidence.applied_pwm().right().get(),
            visual_body_forward_velocity_mps: evidence.visual_forward_velocity_mps(),
            visual_body_lateral_velocity_mps: lateral_velocity_mps,
            calibrated_imu_yaw_rate_rad_s: evidence.calibrated_imu_yaw_rate_rad_s(),
            visual_body_frame_id: BODY_FRAME_ID,
        };
        if let Err(fault) = self.append(&observation_record) {
            return Err(self.fail_closed(fault));
        }
        if self.samples.try_reserve(1).is_err() {
            return Err(self.fail_closed(NanoBaseCommissioningFault::SampleStorageAllocation));
        }
        self.samples.push(CollectedSample {
            identification: IdentificationSampleV1Dto {
                observed_at_ns: evidence.visual_observed_at().as_nanos(),
                applied_command_sequence: evidence.applied_command_sequence(),
                applied_left_pwm_percent: evidence.applied_pwm().left().get(),
                applied_right_pwm_percent: evidence.applied_pwm().right().get(),
                visual_forward_velocity_mps: evidence.visual_forward_velocity_mps(),
                calibrated_imu_yaw_rate_rad_s: evidence.calibrated_imu_yaw_rate_rad_s(),
            },
            visual_body_lateral_velocity_mps: lateral_velocity_mps,
            holdout: sample_ordinal % u32::from(self.policy.lateral.holdout_stride.get()) == 0,
        });
        self.last_sample_observed_at_ns = Some(evidence.visual_observed_at().as_nanos());
        self.last_observed_sequence = Some(evidence.applied_command_sequence());

        let action = self.controller.advance(
            MonotonicTimestampNs::from_nanos(now_ns),
            evidence,
            Cancellation::Continue,
        );
        if let CommissioningState::Aborted(reason) = action.state() {
            return Err(self.fail_closed(NanoBaseCommissioningFault::CoreAborted(reason)));
        }
        if let Err(fault) = self.refresh_requested_pwm(sample_ordinal, action) {
            return Err(self.fail_closed(fault));
        }
        self.completed = action.state() == CommissioningState::Completed
            && self.expected_receipt.is_exact_zero();
        Ok(NanoBaseCommissioningProgress {
            state: action.state(),
            samples_journaled: u32::try_from(self.samples.len()).unwrap_or(u32::MAX),
            requested_left_pwm_percent: action.required_pwm().left().get(),
            requested_right_pwm_percent: action.required_pwm().right().get(),
            last_applied_sequence: self.expected_receipt.applied_command_sequence,
            exact_zero_applied: self.expected_receipt.is_exact_zero(),
        })
    }

    fn validate_stream(
        &self,
        evidence: CommissioningEvidence,
    ) -> Result<(), NanoBaseCommissioningFault> {
        let times = [
            evidence.controller_observed_at().as_nanos(),
            evidence.visual_observed_at().as_nanos(),
            evidence.imu_observed_at().as_nanos(),
        ];
        let minimum_ns = *times.iter().min().expect("fixed nonempty timestamp array");
        let maximum_ns = *times.iter().max().expect("fixed nonempty timestamp array");
        let skew_ns = maximum_ns
            .checked_sub(minimum_ns)
            .expect("maximum timestamp is not below minimum");
        if skew_ns > self.policy.maximum_aligned_observation_skew_ns.get() {
            return Err(NanoBaseCommissioningFault::AlignedObservationSkew {
                minimum_ns,
                maximum_ns,
                allowed_ns: self.policy.maximum_aligned_observation_skew_ns.get(),
            });
        }
        if evidence.applied_command_sequence() != self.expected_receipt.applied_command_sequence
            || evidence.applied_pwm() != self.expected_receipt.applied_pwm
        {
            return Err(NanoBaseCommissioningFault::AppliedReceiptMismatch {
                expected: self.expected_receipt,
                observed_sequence: evidence.applied_command_sequence(),
                observed_pwm: evidence.applied_pwm(),
            });
        }
        if let Some(previous_ns) = self.last_sample_observed_at_ns {
            let current_ns = evidence.visual_observed_at().as_nanos();
            if current_ns <= previous_ns {
                return Err(NanoBaseCommissioningFault::SampleClockDidNotAdvance {
                    previous_ns,
                    current_ns,
                });
            }
            let gap_ns = current_ns - previous_ns;
            if gap_ns < self.policy.fit.min_sample_period_ns().get() {
                return Err(NanoBaseCommissioningFault::SampleGapTooShort {
                    previous_ns,
                    current_ns,
                    minimum_ns: self.policy.fit.min_sample_period_ns().get(),
                });
            }
            let maximum_ns = self
                .policy
                .maximum_sample_gap_ns
                .get()
                .min(self.policy.fit.max_sample_period_ns().get());
            if gap_ns > maximum_ns {
                return Err(NanoBaseCommissioningFault::SampleGap {
                    previous_ns,
                    current_ns,
                    maximum_ns,
                });
            }
        }
        if let Some(previous) = self.last_observed_sequence {
            let current = evidence.applied_command_sequence();
            let gap = current.checked_sub(previous).ok_or(
                NanoBaseCommissioningFault::ControllerSequenceGap {
                    previous,
                    current,
                    maximum_gap: self.policy.maximum_controller_sequence_gap.get(),
                },
            )?;
            if gap > self.policy.maximum_controller_sequence_gap.get() {
                return Err(NanoBaseCommissioningFault::ControllerSequenceGap {
                    previous,
                    current,
                    maximum_gap: self.policy.maximum_controller_sequence_gap.get(),
                });
            }
        }
        Ok(())
    }

    /// Reapply every requested command, including an unchanged one, after
    /// each admitted observation. The controller lease is refreshed at the
    /// observed sample cadence; a long excitation phase never depends on one
    /// command packet outliving the V3 lease.
    fn refresh_requested_pwm(
        &mut self,
        sample_ordinal: u32,
        action: CommissioningAction,
    ) -> Result<(), NanoBaseCommissioningFault> {
        let requested = action.required_pwm();
        let request_record = CommissioningJournalRecord::RequestedPwm {
            sample_ordinal,
            left_pwm_percent: requested.left().get(),
            right_pwm_percent: requested.right().get(),
        };
        self.append(&request_record)?;
        let receipt = self
            .actuator
            .apply(requested)
            .map_err(|source| NanoBaseCommissioningFault::Actuator(Box::new(source)))?;
        if receipt.applied_pwm.left().get() != requested.left().get()
            || receipt.applied_pwm.right().get() != requested.right().get()
        {
            return Err(NanoBaseCommissioningFault::AppliedCommandMismatch { requested, receipt });
        }
        if receipt.applied_command_sequence <= self.expected_receipt.applied_command_sequence {
            return Err(NanoBaseCommissioningFault::AppliedSequenceDidNotAdvance {
                previous: self.expected_receipt.applied_command_sequence,
                current: receipt.applied_command_sequence,
            });
        }
        let record = receipt_record(sample_ordinal, receipt);
        self.append(&record)?;
        self.expected_receipt = receipt;
        Ok(())
    }

    fn append(
        &mut self,
        record: &CommissioningJournalRecord<'_>,
    ) -> Result<(), NanoBaseCommissioningFault> {
        append_journal(
            &mut self.journal,
            &mut self.journal_hasher,
            &mut self.journal_records,
            &mut self.journal_bytes,
            record,
        )
    }

    fn fail_closed(&mut self, fault: NanoBaseCommissioningFault) -> NanoBaseCommissioningFailure {
        let reason = fault.to_string();
        let terminal_record = CommissioningJournalRecord::Terminal {
            sample_ordinal: u32::try_from(self.samples.len()).unwrap_or(u32::MAX),
            reason: &reason,
        };
        let terminal_journal = match self.append(&terminal_record) {
            Ok(()) => CommissioningTerminalJournalStatus::Durable,
            Err(source) => CommissioningTerminalJournalStatus::Failed(Box::new(source)),
        };
        let stop = emergency_zero_evidence(&mut self.actuator, Some(self.expected_receipt));
        self.terminated = true;
        NanoBaseCommissioningFailure {
            fault,
            stop,
            terminal_journal,
        }
    }

    /// Explicitly end an attended run and return compound journal/stop
    /// evidence. Runtime owners must call this on cancellation or source
    /// failure; `Drop` is only a final bounded risk-reduction fallback.
    pub fn terminate(
        &mut self,
        signal: CommissioningExternalSignal,
    ) -> Result<std::convert::Infallible, NanoBaseCommissioningFailure> {
        if signal == CommissioningExternalSignal::Continue {
            return Err(
                self.fail_closed(NanoBaseCommissioningFault::TerminateRequiresTerminalSignal)
            );
        }
        Err(self.fail_closed(NanoBaseCommissioningFault::ExternalSignal(signal)))
    }

    pub const fn expected_receipt(&self) -> ExactCommissioningControllerReceipt {
        self.expected_receipt
    }

    pub fn progress(&self) -> NanoBaseCommissioningProgress {
        let action_state = self.controller.state();
        NanoBaseCommissioningProgress {
            state: action_state,
            samples_journaled: u32::try_from(self.samples.len()).unwrap_or(u32::MAX),
            requested_left_pwm_percent: self.expected_receipt.applied_pwm.left().get(),
            requested_right_pwm_percent: self.expected_receipt.applied_pwm.right().get(),
            last_applied_sequence: self.expected_receipt.applied_command_sequence,
            exact_zero_applied: self.expected_receipt.is_exact_zero(),
        }
    }
}

impl<A, J> Drop for NanoBaseCommissioningSession<A, J>
where
    A: SoleCommissioningActuator,
    J: DurableCommissioningJournal,
{
    fn drop(&mut self) {
        if !self.terminated {
            let _ = self.actuator.emergency_zero();
            self.terminated = true;
        }
    }
}

fn append_journal<J: DurableCommissioningJournal>(
    journal: &mut J,
    hasher: &mut Sha256,
    records: &mut u64,
    bytes: &mut u64,
    record: &CommissioningJournalRecord<'_>,
) -> Result<(), NanoBaseCommissioningFault> {
    let mut encoded =
        serde_json::to_vec(record).map_err(NanoBaseCommissioningFault::JournalEncode)?;
    encoded.push(b'\n');
    if encoded.len() > MAX_COMMISSIONING_JOURNAL_RECORD_BYTES {
        return Err(NanoBaseCommissioningFault::JournalEncode(
            serde_json::Error::io(io::Error::new(
                io::ErrorKind::InvalidData,
                "encoded commissioning journal record exceeds hard bound",
            )),
        ));
    }
    journal
        .append_durable(&encoded)
        .map_err(|source| NanoBaseCommissioningFault::Journal(Box::new(source)))?;
    hasher.update(&encoded);
    *records = records
        .checked_add(1)
        .ok_or(NanoBaseCommissioningFault::SampleCountOverflow)?;
    *bytes = bytes
        .checked_add(
            u64::try_from(encoded.len())
                .map_err(|_| NanoBaseCommissioningFault::SampleCountOverflow)?,
        )
        .ok_or(NanoBaseCommissioningFault::SampleCountOverflow)?;
    Ok(())
}

fn receipt_record(
    sample_ordinal: u32,
    receipt: ExactCommissioningControllerReceipt,
) -> CommissioningJournalRecord<'static> {
    CommissioningJournalRecord::AppliedReceipt {
        sample_ordinal,
        observed_at_ns: receipt.observed_at_ns,
        applied_command_sequence: receipt.applied_command_sequence,
        applied_left_pwm_percent: receipt.applied_pwm.left().get(),
        applied_right_pwm_percent: receipt.applied_pwm.right().get(),
    }
}

fn emergency_zero_evidence<A: SoleCommissioningActuator>(
    actuator: &mut A,
    previous: Option<ExactCommissioningControllerReceipt>,
) -> ExactFailClosedStop {
    match actuator.emergency_zero() {
        Ok(receipt)
            if receipt.is_exact_zero()
                && previous.is_none_or(|previous| {
                    receipt.applied_command_sequence > previous.applied_command_sequence
                }) =>
        {
            ExactFailClosedStop::Applied { receipt }
        }
        Ok(receipt) => ExactFailClosedStop::ReceiptRejected { previous, receipt },
        Err(source) => ExactFailClosedStop::CommandFailed {
            source: Box::new(source),
        },
    }
}

fn zero_command_for_error() -> CanonicalPwmCommand {
    CanonicalPwmCommand::zero()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VerifiedLateralValidityBound {
    maximum_absolute_lateral_velocity_mps: f64,
    training_sample_count: u32,
    holdout_sample_count: u32,
    training_maximum_absolute_lateral_velocity_mps: f64,
    holdout_maximum_absolute_lateral_velocity_mps: f64,
    configured_margin_mps: f64,
    scope_label: CommissioningLabel,
}

impl VerifiedLateralValidityBound {
    pub const fn maximum_absolute_lateral_velocity_mps(self) -> f64 {
        self.maximum_absolute_lateral_velocity_mps
    }

    pub const fn training_sample_count(self) -> u32 {
        self.training_sample_count
    }

    pub const fn holdout_sample_count(self) -> u32 {
        self.holdout_sample_count
    }

    pub fn scope_label(&self) -> &str {
        self.scope_label.as_str()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishedCommissioningArtifact {
    pub path: PathBuf,
    pub content_sha256: [u8; CONTENT_DIGEST_BYTES],
    pub byte_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NanoBaseCommissioningProposal {
    pub dataset: PublishedCommissioningArtifact,
    pub proposed_plant: PublishedCommissioningArtifact,
    pub proposal_evidence: PublishedCommissioningArtifact,
    pub journal: CommissioningJournalCommit,
    pub lateral_validity: VerifiedLateralValidityBound,
    pub activation_status: &'static str,
}

#[derive(Debug)]
pub struct CommissioningArtifactDirectory {
    diagnostic_path: PathBuf,
    directory: File,
    device: u64,
    inode: u64,
}

impl CommissioningArtifactDirectory {
    pub fn inspect(path: &Path) -> Result<Self, CommissioningArtifactDirectoryError> {
        if !path.is_absolute() {
            return Err(CommissioningArtifactDirectoryError::NotAbsolute(
                path.to_path_buf(),
            ));
        }
        let directory =
            open_directory_file(path).map_err(CommissioningArtifactDirectoryError::Inspect)?;
        Self::from_retained_directory(path.to_path_buf(), directory)
    }

    pub(crate) fn from_retained_directory(
        diagnostic_path: PathBuf,
        directory: File,
    ) -> Result<Self, CommissioningArtifactDirectoryError> {
        if !diagnostic_path.is_absolute() {
            return Err(CommissioningArtifactDirectoryError::NotAbsolute(
                diagnostic_path,
            ));
        }
        let metadata = directory
            .metadata()
            .map_err(CommissioningArtifactDirectoryError::Inspect)?;
        if !metadata.file_type().is_dir() {
            return Err(CommissioningArtifactDirectoryError::NotDirectory);
        }
        if metadata.uid() != current_uid() {
            return Err(CommissioningArtifactDirectoryError::OwnerMismatch {
                expected_uid: current_uid(),
                observed_uid: metadata.uid(),
            });
        }
        let mode = metadata.mode() & 0o777;
        if mode != 0o700 {
            return Err(CommissioningArtifactDirectoryError::PermissionsTooBroad { mode });
        }
        let admitted = Self {
            diagnostic_path,
            directory,
            device: metadata.dev(),
            inode: metadata.ino(),
        };
        admitted.verify_binding()?;
        Ok(admitted)
    }

    pub fn as_path(&self) -> &Path {
        &self.diagnostic_path
    }

    pub(crate) fn directory(&self) -> &File {
        &self.directory
    }

    pub(crate) fn verify_binding(&self) -> Result<(), CommissioningArtifactDirectoryError> {
        let retained = self
            .directory
            .metadata()
            .map_err(CommissioningArtifactDirectoryError::Inspect)?;
        if !retained.file_type().is_dir()
            || retained.dev() != self.device
            || retained.ino() != self.inode
        {
            return Err(CommissioningArtifactDirectoryError::RetainedIdentityChanged);
        }
        if retained.uid() != current_uid() {
            return Err(CommissioningArtifactDirectoryError::OwnerMismatch {
                expected_uid: current_uid(),
                observed_uid: retained.uid(),
            });
        }
        let mode = retained.mode() & 0o777;
        if mode != 0o700 {
            return Err(CommissioningArtifactDirectoryError::PermissionsTooBroad { mode });
        }

        let rebound = open_directory_file(&self.diagnostic_path)
            .map_err(CommissioningArtifactDirectoryError::ReopenBinding)?;
        let rebound_metadata = rebound
            .metadata()
            .map_err(CommissioningArtifactDirectoryError::ReopenBinding)?;
        if rebound_metadata.dev() != self.device || rebound_metadata.ino() != self.inode {
            return Err(CommissioningArtifactDirectoryError::PathBindingChanged {
                expected_device: self.device,
                expected_inode: self.inode,
                observed_device: rebound_metadata.dev(),
                observed_inode: rebound_metadata.ino(),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum CommissioningArtifactDirectoryError {
    NotAbsolute(PathBuf),
    Inspect(io::Error),
    ReopenBinding(io::Error),
    NotDirectory,
    RetainedIdentityChanged,
    PathBindingChanged {
        expected_device: u64,
        expected_inode: u64,
        observed_device: u64,
        observed_inode: u64,
    },
    OwnerMismatch {
        expected_uid: u32,
        observed_uid: u32,
    },
    PermissionsTooBroad {
        mode: u32,
    },
}

impl fmt::Display for CommissioningArtifactDirectoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid private commissioning artifact directory: {self:?}"
        )
    }
}

impl std::error::Error for CommissioningArtifactDirectoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Inspect(source) | Self::ReopenBinding(source) => Some(source),
            _ => None,
        }
    }
}

fn open_directory_file(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_DIRECTORY | libc::O_CLOEXEC)
        .open(path)
}

fn is_simple_name(name: &std::ffi::OsStr) -> bool {
    let mut components = Path::new(name).components();
    matches!(components.next(), Some(std::path::Component::Normal(_)))
        && components.next().is_none()
}

fn current_uid() -> u32 {
    rustix::process::geteuid().as_raw()
}

impl<A, J> NanoBaseCommissioningSession<A, J>
where
    A: SoleCommissioningActuator,
    J: DurableCommissioningJournal,
{
    pub fn publish_proposal(
        mut self,
        directory: &CommissioningArtifactDirectory,
    ) -> Result<NanoBaseCommissioningProposal, NanoBaseCommissioningPublishFailure> {
        match self.publish_proposal_inner(directory) {
            Ok(proposal) => {
                self.terminated = true;
                Ok(proposal)
            }
            Err(fault) => {
                let stop = emergency_zero_evidence(&mut self.actuator, Some(self.expected_receipt));
                self.terminated = true;
                Err(NanoBaseCommissioningPublishFailure {
                    fault: Box::new(fault),
                    stop,
                })
            }
        }
    }

    fn publish_proposal_inner(
        &mut self,
        directory: &CommissioningArtifactDirectory,
    ) -> Result<NanoBaseCommissioningProposal, NanoBaseCommissioningPublishError> {
        if !self.completed || self.controller.state() != CommissioningState::Completed {
            return Err(NanoBaseCommissioningPublishError::NotCompleted {
                state: self.controller.state(),
            });
        }
        if !self.expected_receipt.is_exact_zero() {
            return Err(NanoBaseCommissioningPublishError::ExactZeroNotRetained);
        }
        let terminal_record = CommissioningJournalRecord::Terminal {
            sample_ordinal: u32::try_from(self.samples.len()).unwrap_or(u32::MAX),
            reason: "completed; proposed artifacts only; physical approval not granted",
        };
        self.append(&terminal_record)
            .map_err(NanoBaseCommissioningPublishError::JournalRecord)?;
        let journal_commit = self
            .journal
            .finalize_durable()
            .map_err(|source| NanoBaseCommissioningPublishError::Journal(Box::new(source)))?;
        let expected_journal_sha256: [u8; CONTENT_DIGEST_BYTES] =
            self.journal_hasher.clone().finalize().into();
        if journal_commit.content_sha256 != expected_journal_sha256
            || journal_commit.record_count != self.journal_records
            || journal_commit.byte_count != self.journal_bytes
        {
            return Err(NanoBaseCommissioningPublishError::JournalCommitMismatch {
                expected_sha256: expected_journal_sha256,
                observed_sha256: journal_commit.content_sha256,
                expected_records: self.journal_records,
                observed_records: journal_commit.record_count,
                expected_bytes: self.journal_bytes,
                observed_bytes: journal_commit.byte_count,
            });
        }

        let lateral_validity = derive_lateral_validity(&self.samples, self.policy.lateral)?;
        let robot_id = self.policy.fit.expected_robot_id();
        let controller_session_id = self.policy.fit.expected_controller_session_id();
        let visual_velocity_source_id = self.policy.fit.expected_visual_velocity_source_id();
        let imu_calibration_id = self.policy.fit.expected_imu_calibration_id();
        let wheelbase_calibration_id = self.policy.fit.wheelbase_calibration_id();
        let evidence_payload = DatasetEvidenceHashPayload {
            schema_version: NANO_BASE_COMMISSIONING_ARTIFACT_V1,
            robot_id: robot_id.as_str(),
            controller_session_id: controller_session_id.as_str(),
            visual_velocity_source_id: visual_velocity_source_id.as_str(),
            imu_calibration_id: imu_calibration_id.as_str(),
            wheelbase_calibration_id: wheelbase_calibration_id.as_str(),
            body_frame_id: BODY_FRAME_ID,
            samples: &self.samples,
        };
        let evidence_payload_bytes = serde_json::to_vec(&evidence_payload)
            .map_err(NanoBaseCommissioningPublishError::DatasetEncode)?;
        let evidence_content_sha256 = sha256(&evidence_payload_bytes);
        let dataset_content_id = lower_hex(evidence_content_sha256);
        let dataset_dto = IdentificationDatasetV1Dto {
            schema_version: BASE_IDENTIFICATION_V1,
            dataset_content_id,
            robot_id: robot_id.as_str().to_owned(),
            controller_session_id: controller_session_id.as_str().to_owned(),
            visual_velocity_source_id: visual_velocity_source_id.as_str().to_owned(),
            imu_calibration_id: imu_calibration_id.as_str().to_owned(),
            wheelbase_calibration_id: wheelbase_calibration_id.as_str().to_owned(),
            samples: self
                .samples
                .iter()
                .map(|sample| sample.identification)
                .collect(),
        };
        let published_dataset_envelope = PublishedDatasetEnvelopeV1 {
            schema_version: NANO_BASE_COMMISSIONING_ARTIFACT_V1,
            evidence_content_sha256: canonical_sha256(evidence_content_sha256),
            policy_sha256: canonical_sha256(self.policy.content_sha256),
            controller_profile_sha256: canonical_sha256(self.authority.controller_profile_sha256),
            attended_physical_attestation_sha256: canonical_sha256(
                self.authority.physical_attestation_sha256,
            ),
            journal_sha256: canonical_sha256(journal_commit.content_sha256),
            body_frame_id: BODY_FRAME_ID,
            aligned_evidence: &evidence_payload,
        };
        let dataset_bytes = serde_json::to_vec(&published_dataset_envelope)
            .map_err(NanoBaseCommissioningPublishError::DatasetEncode)?;
        let dataset = IdentificationDatasetV1::parse(dataset_dto, self.policy.fit)
            .map_err(NanoBaseCommissioningPublishError::Dataset)?;
        let fit = fit_first_order_plant(&dataset, self.policy.fit)
            .map_err(NanoBaseCommissioningPublishError::Fit)?;
        let published_dataset =
            publish_content_addressed(directory, "base-identification-dataset-v1", &dataset_bytes)?;
        let plant_dto = verified_fit_to_plant_dto(
            fit,
            lateral_validity,
            self.policy,
            published_dataset.content_sha256,
        );
        PlantModelV1::parse(plant_dto.clone())
            .map_err(NanoBaseCommissioningPublishError::PlantVerification)?;
        let plant_bytes = serde_json::to_vec(&plant_dto)
            .map_err(NanoBaseCommissioningPublishError::PlantEncode)?;
        let published_plant =
            publish_content_addressed(directory, "proposed-plant-v1", &plant_bytes)?;
        let proposal_evidence = ProposalEvidenceV1 {
            schema_version: NANO_BASE_COMMISSIONING_ARTIFACT_V1,
            activation_status: PROPOSAL_ACTIVATION_STATUS,
            policy_sha256: canonical_sha256(self.policy.content_sha256),
            controller_profile_sha256: canonical_sha256(self.authority.controller_profile_sha256),
            attended_physical_attestation_sha256: canonical_sha256(
                self.authority.physical_attestation_sha256,
            ),
            journal_sha256: canonical_sha256(journal_commit.content_sha256),
            plant_evidence_dataset_content_id: canonical_sha256(published_dataset.content_sha256),
            proposed_plant_artifact_sha256: canonical_sha256(published_plant.content_sha256),
            controller_session_id: self.authority.controller_session_id.as_str(),
            visual_velocity_source_id: visual_velocity_source_id.as_str(),
            imu_calibration_id: imu_calibration_id.as_str(),
            wheelbase_calibration_id: wheelbase_calibration_id.as_str(),
            lateral_scope: LateralScopeEvidenceV1::from_verified(&lateral_validity),
            remaining_gate: "operator review, physical approval, manifest rebind, and normal production admission",
        };
        let proposal_evidence_bytes = serde_json::to_vec(&proposal_evidence)
            .map_err(NanoBaseCommissioningPublishError::ProposalEvidenceEncode)?;
        let published_proposal_evidence = publish_content_addressed(
            directory,
            "base-commissioning-proposal-evidence-v1",
            &proposal_evidence_bytes,
        )?;
        Ok(NanoBaseCommissioningProposal {
            dataset: published_dataset,
            proposed_plant: published_plant,
            proposal_evidence: published_proposal_evidence,
            journal: journal_commit,
            lateral_validity,
            activation_status: PROPOSAL_ACTIVATION_STATUS,
        })
    }
}

#[derive(Serialize)]
struct DatasetEvidenceHashPayload<'a> {
    schema_version: u32,
    robot_id: &'a str,
    controller_session_id: &'a str,
    visual_velocity_source_id: &'a str,
    imu_calibration_id: &'a str,
    wheelbase_calibration_id: &'a str,
    body_frame_id: &'static str,
    samples: &'a [CollectedSample],
}

#[derive(Serialize)]
struct PublishedDatasetEnvelopeV1<'a> {
    schema_version: u32,
    evidence_content_sha256: String,
    policy_sha256: String,
    controller_profile_sha256: String,
    attended_physical_attestation_sha256: String,
    journal_sha256: String,
    body_frame_id: &'static str,
    aligned_evidence: &'a DatasetEvidenceHashPayload<'a>,
}

impl Serialize for CollectedSample {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        CollectedSampleArtifactV1 {
            observed_at_ns: self.identification.observed_at_ns,
            applied_command_sequence: self.identification.applied_command_sequence,
            applied_left_pwm_percent: self.identification.applied_left_pwm_percent,
            applied_right_pwm_percent: self.identification.applied_right_pwm_percent,
            visual_body_forward_velocity_mps: self.identification.visual_forward_velocity_mps,
            visual_body_lateral_velocity_mps: self.visual_body_lateral_velocity_mps,
            calibrated_imu_yaw_rate_rad_s: self.identification.calibrated_imu_yaw_rate_rad_s,
            lateral_holdout: self.holdout,
        }
        .serialize(serializer)
    }
}

#[derive(Serialize)]
struct CollectedSampleArtifactV1 {
    observed_at_ns: u64,
    applied_command_sequence: u64,
    applied_left_pwm_percent: i8,
    applied_right_pwm_percent: i8,
    visual_body_forward_velocity_mps: f64,
    visual_body_lateral_velocity_mps: f64,
    calibrated_imu_yaw_rate_rad_s: f64,
    lateral_holdout: bool,
}

#[derive(Serialize)]
struct ProposalEvidenceV1<'a> {
    schema_version: u32,
    activation_status: &'static str,
    policy_sha256: String,
    controller_profile_sha256: String,
    attended_physical_attestation_sha256: String,
    journal_sha256: String,
    plant_evidence_dataset_content_id: String,
    proposed_plant_artifact_sha256: String,
    controller_session_id: &'a str,
    visual_velocity_source_id: &'a str,
    imu_calibration_id: &'a str,
    wheelbase_calibration_id: &'a str,
    lateral_scope: LateralScopeEvidenceV1<'a>,
    remaining_gate: &'static str,
}

#[derive(Serialize)]
struct LateralScopeEvidenceV1<'a> {
    method_id: &'static str,
    scope_label: &'a str,
    frame_id: &'static str,
    maximum_absolute_lateral_velocity_mps: f64,
    training_sample_count: u32,
    holdout_sample_count: u32,
    training_maximum_absolute_lateral_velocity_mps: f64,
    holdout_maximum_absolute_lateral_velocity_mps: f64,
    configured_margin_mps: f64,
}

impl<'a> LateralScopeEvidenceV1<'a> {
    fn from_verified(bound: &'a VerifiedLateralValidityBound) -> Self {
        Self {
            method_id: LATERAL_METHOD_ID,
            scope_label: bound.scope_label.as_str(),
            frame_id: BODY_FRAME_ID,
            maximum_absolute_lateral_velocity_mps: bound.maximum_absolute_lateral_velocity_mps,
            training_sample_count: bound.training_sample_count,
            holdout_sample_count: bound.holdout_sample_count,
            training_maximum_absolute_lateral_velocity_mps: bound
                .training_maximum_absolute_lateral_velocity_mps,
            holdout_maximum_absolute_lateral_velocity_mps: bound
                .holdout_maximum_absolute_lateral_velocity_mps,
            configured_margin_mps: bound.configured_margin_mps,
        }
    }
}

#[derive(Debug)]
pub struct NanoBaseCommissioningPublishFailure {
    pub fault: Box<NanoBaseCommissioningPublishError>,
    pub stop: ExactFailClosedStop,
}

impl fmt::Display for NanoBaseCommissioningPublishFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "base-commissioning proposal publication failed: {}; explicit_stop={:?}",
            self.fault, self.stop
        )
    }
}

impl std::error::Error for NanoBaseCommissioningPublishFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.fault.as_ref())
    }
}

#[derive(Debug)]
pub enum NanoBaseCommissioningPublishError {
    NotCompleted {
        state: CommissioningState,
    },
    ExactZeroNotRetained,
    JournalRecord(NanoBaseCommissioningFault),
    Journal(Box<dyn std::error::Error + Send + Sync>),
    JournalCommitMismatch {
        expected_sha256: [u8; CONTENT_DIGEST_BYTES],
        observed_sha256: [u8; CONTENT_DIGEST_BYTES],
        expected_records: u64,
        observed_records: u64,
        expected_bytes: u64,
        observed_bytes: u64,
    },
    Lateral(LateralValidityError),
    DatasetEncode(serde_json::Error),
    Dataset(DatasetParseError),
    Fit(FitError),
    PlantVerification(PlantModelParseError),
    PlantEncode(serde_json::Error),
    ProposalEvidenceEncode(serde_json::Error),
    Publish(AtomicArtifactPublishError),
}

impl fmt::Display for NanoBaseCommissioningPublishError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "could not publish proposed base-commissioning artifacts: {self:?}"
        )
    }
}

impl std::error::Error for NanoBaseCommissioningPublishError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::JournalRecord(source) => Some(source),
            Self::Journal(source) => Some(source.as_ref()),
            Self::Lateral(source) => Some(source),
            Self::DatasetEncode(source)
            | Self::PlantEncode(source)
            | Self::ProposalEvidenceEncode(source) => Some(source),
            Self::Dataset(source) => Some(source),
            Self::Fit(source) => Some(source),
            Self::PlantVerification(source) => Some(source),
            Self::Publish(source) => Some(source),
            _ => None,
        }
    }
}

impl From<LateralValidityError> for NanoBaseCommissioningPublishError {
    fn from(source: LateralValidityError) -> Self {
        Self::Lateral(source)
    }
}

impl From<AtomicArtifactPublishError> for NanoBaseCommissioningPublishError {
    fn from(source: AtomicArtifactPublishError) -> Self {
        Self::Publish(source)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LateralValidityError {
    InsufficientTrainingSamples {
        actual: u32,
        required: u32,
    },
    InsufficientHoldoutSamples {
        actual: u32,
        required: u32,
    },
    DerivedBoundNonFinite,
    DerivedBoundAbovePolicy {
        derived_mps: f64,
        maximum_mps: f64,
    },
    HoldoutOutsideDerivedBound {
        sample_ordinal: u32,
        observed_abs_mps: f64,
        derived_bound_mps: f64,
    },
}

impl fmt::Display for LateralValidityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "lateral validity evidence rejected: {self:?}")
    }
}

impl std::error::Error for LateralValidityError {}

fn derive_lateral_validity(
    samples: &[CollectedSample],
    policy: LateralHoldoutPolicyV1,
) -> Result<VerifiedLateralValidityBound, LateralValidityError> {
    let mut training_count = 0_u32;
    let mut holdout_count = 0_u32;
    let mut training_max = 0.0_f64;
    let mut holdout_max = 0.0_f64;
    for sample in samples {
        let magnitude = sample.visual_body_lateral_velocity_mps.abs();
        if sample.holdout {
            holdout_count = holdout_count.saturating_add(1);
            holdout_max = holdout_max.max(magnitude);
        } else {
            training_count = training_count.saturating_add(1);
            training_max = training_max.max(magnitude);
        }
    }
    if training_count < policy.minimum_training_samples.get() {
        return Err(LateralValidityError::InsufficientTrainingSamples {
            actual: training_count,
            required: policy.minimum_training_samples.get(),
        });
    }
    if holdout_count < policy.minimum_holdout_samples.get() {
        return Err(LateralValidityError::InsufficientHoldoutSamples {
            actual: holdout_count,
            required: policy.minimum_holdout_samples.get(),
        });
    }
    let derived = training_max + policy.bound_margin_mps;
    if !derived.is_finite() {
        return Err(LateralValidityError::DerivedBoundNonFinite);
    }
    if derived > policy.maximum_accepted_bound_mps {
        return Err(LateralValidityError::DerivedBoundAbovePolicy {
            derived_mps: derived,
            maximum_mps: policy.maximum_accepted_bound_mps,
        });
    }
    for (index, sample) in samples.iter().enumerate() {
        let magnitude = sample.visual_body_lateral_velocity_mps.abs();
        if sample.holdout && magnitude > derived {
            return Err(LateralValidityError::HoldoutOutsideDerivedBound {
                sample_ordinal: u32::try_from(index).unwrap_or(u32::MAX),
                observed_abs_mps: magnitude,
                derived_bound_mps: derived,
            });
        }
    }
    Ok(VerifiedLateralValidityBound {
        maximum_absolute_lateral_velocity_mps: derived,
        training_sample_count: training_count,
        holdout_sample_count: holdout_count,
        training_maximum_absolute_lateral_velocity_mps: training_max,
        holdout_maximum_absolute_lateral_velocity_mps: holdout_max,
        configured_margin_mps: policy.bound_margin_mps,
        scope_label: policy.scope_label,
    })
}

fn verified_fit_to_plant_dto(
    fit: IdentifiedPlantV1,
    lateral: VerifiedLateralValidityBound,
    policy: NanoBaseCommissioningPolicyV1,
    dataset_artifact_sha256: [u8; CONTENT_DIGEST_BYTES],
) -> PlantModelV1Dto {
    debug_assert_eq!(
        fit.support().lateral_velocity,
        LateralVelocityEvidence::Unidentified
    );
    let support = fit.support();
    let residuals = fit.holdout_residuals();
    PlantModelV1Dto {
        schema_version: PLANT_MODEL_V1,
        model_id: policy.model_id.as_str().to_owned(),
        model_version: policy.model_version.get(),
        sample_period_s: fit.sample_period_s(),
        wheelbase_m: fit.wheelbase_m(),
        left: WheelPlantV1Dto {
            velocity_gain_mps_per_pwm_percent: fit.left().velocity_gain_mps_per_pwm_percent(),
            time_constant_s: fit.left().time_constant_s(),
        },
        right: WheelPlantV1Dto {
            velocity_gain_mps_per_pwm_percent: fit.right().velocity_gain_mps_per_pwm_percent(),
            time_constant_s: fit.right().time_constant_s(),
        },
        validity: PlantValidityEnvelopeV1Dto {
            left_pwm_min_percent: support.left_pwm_min_percent,
            left_pwm_max_percent: support.left_pwm_max_percent,
            right_pwm_min_percent: support.right_pwm_min_percent,
            right_pwm_max_percent: support.right_pwm_max_percent,
            left_velocity_min_mps: support.left_velocity_min_mps,
            left_velocity_max_mps: support.left_velocity_max_mps,
            right_velocity_min_mps: support.right_velocity_min_mps,
            right_velocity_max_mps: support.right_velocity_max_mps,
            max_abs_yaw_rate_rad_s: support.max_abs_yaw_rate_rad_s,
            max_abs_lateral_velocity_mps: lateral.maximum_absolute_lateral_velocity_mps,
        },
        evidence: PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
            dataset_content_id: canonical_sha256(dataset_artifact_sha256),
            identification_method_id: fit.identification_method_id().as_str().to_owned(),
            sample_count: u64::from(fit.source_sample_count()),
            residuals: FitResidualsV1Dto {
                left_velocity_rmse_mps: residuals.left_velocity_rmse_mps,
                right_velocity_rmse_mps: residuals.right_velocity_rmse_mps,
                yaw_rate_rmse_rad_s: residuals.yaw_rate_rmse_rad_s,
                max_abs_velocity_error_mps: residuals.max_abs_wheel_velocity_error_mps,
            },
        },
    }
}

#[derive(Debug)]
pub enum AtomicArtifactPublishError {
    ArtifactTooLarge {
        actual_bytes: usize,
        maximum_bytes: usize,
    },
    TemporaryNameExhausted,
    Directory(CommissioningArtifactDirectoryError),
    CreateTemporary(io::Error),
    InspectTemporary(io::Error),
    UnsafeTemporary,
    WriteTemporary(io::Error),
    SyncTemporary(io::Error),
    Publish(io::Error),
    CleanupAfterFailure {
        failure: Box<AtomicArtifactPublishError>,
        cleanup: io::Error,
    },
    InspectExisting(io::Error),
    UnsafeExisting,
    ReadExisting(io::Error),
    ExistingContentMismatch,
    RemoveTemporary(io::Error),
    InspectPublished(io::Error),
    PublishedIdentityChanged,
    SyncDirectory(io::Error),
}

impl fmt::Display for AtomicArtifactPublishError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "atomic commissioning artifact publish failed: {self:?}"
        )
    }
}

impl std::error::Error for AtomicArtifactPublishError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Directory(source) => Some(source),
            Self::CreateTemporary(source)
            | Self::InspectTemporary(source)
            | Self::WriteTemporary(source)
            | Self::SyncTemporary(source)
            | Self::Publish(source)
            | Self::InspectExisting(source)
            | Self::ReadExisting(source)
            | Self::RemoveTemporary(source)
            | Self::InspectPublished(source)
            | Self::SyncDirectory(source) => Some(source),
            Self::CleanupAfterFailure { failure, .. } => Some(failure.as_ref()),
            _ => None,
        }
    }
}

fn publish_content_addressed(
    directory: &CommissioningArtifactDirectory,
    stem: &str,
    bytes: &[u8],
) -> Result<PublishedCommissioningArtifact, AtomicArtifactPublishError> {
    if bytes.len() > MAX_COMMISSIONING_ARTIFACT_BYTES {
        return Err(AtomicArtifactPublishError::ArtifactTooLarge {
            actual_bytes: bytes.len(),
            maximum_bytes: MAX_COMMISSIONING_ARTIFACT_BYTES,
        });
    }
    let content_sha256 = sha256(bytes);
    let digest_hex = lower_hex(content_sha256);
    let destination_name = format!("{stem}-{digest_hex}.json");
    let destination = directory.as_path().join(&destination_name);
    directory
        .verify_binding()
        .map_err(AtomicArtifactPublishError::Directory)?;
    let (temporary_name, mut temporary) = create_artifact_temporary(directory, stem)?;
    temporary.write_all(bytes).map_err(|source| {
        artifact_error_with_cleanup(
            directory,
            &temporary_name,
            AtomicArtifactPublishError::WriteTemporary(source),
        )
    })?;
    temporary.sync_all().map_err(|source| {
        artifact_error_with_cleanup(
            directory,
            &temporary_name,
            AtomicArtifactPublishError::SyncTemporary(source),
        )
    })?;
    let temporary_stat = fstat(&temporary).map_err(|source| {
        artifact_error_with_cleanup(
            directory,
            &temporary_name,
            AtomicArtifactPublishError::InspectTemporary(errno_as_io(source)),
        )
    })?;
    let parent_stat = fstat(directory.directory()).map_err(|source| {
        artifact_error_with_cleanup(
            directory,
            &temporary_name,
            AtomicArtifactPublishError::InspectTemporary(errno_as_io(source)),
        )
    })?;
    if FileType::from_raw_mode(temporary_stat.st_mode) != FileType::RegularFile
        || temporary_stat.st_nlink != 1
        || temporary_stat.st_uid != current_uid()
        || u32::from(temporary_stat.st_mode) & 0o777 != 0o600
        || temporary_stat.st_dev != parent_stat.st_dev
    {
        return Err(artifact_error_with_cleanup(
            directory,
            &temporary_name,
            AtomicArtifactPublishError::UnsafeTemporary,
        ));
    }
    directory.verify_binding().map_err(|source| {
        artifact_error_with_cleanup(
            directory,
            &temporary_name,
            AtomicArtifactPublishError::Directory(source),
        )
    })?;
    match renameat_with(
        directory.directory(),
        &temporary_name,
        directory.directory(),
        &destination_name,
        RenameFlags::NOREPLACE,
    ) {
        Ok(()) => {
            require_published_identity(directory, &destination_name, &temporary_stat)?;
            fsync(directory.directory())
                .map_err(|source| AtomicArtifactPublishError::SyncDirectory(errno_as_io(source)))?;
            require_published_identity(directory, &destination_name, &temporary_stat)?;
        }
        Err(Errno::EXIST) => {
            unlink_artifact_temporary(directory, &temporary_name)
                .map_err(AtomicArtifactPublishError::RemoveTemporary)?;
            verify_existing_artifact(directory, &destination_name, bytes)?;
            fsync(directory.directory())
                .map_err(|source| AtomicArtifactPublishError::SyncDirectory(errno_as_io(source)))?;
        }
        Err(source) => {
            return Err(artifact_error_with_cleanup(
                directory,
                &temporary_name,
                AtomicArtifactPublishError::Publish(errno_as_io(source)),
            ));
        }
    }
    directory
        .verify_binding()
        .map_err(AtomicArtifactPublishError::Directory)?;
    Ok(PublishedCommissioningArtifact {
        path: destination,
        content_sha256,
        byte_count: u64::try_from(bytes.len()).expect("artifact hard bound fits u64"),
    })
}

fn create_artifact_temporary(
    directory: &CommissioningArtifactDirectory,
    stem: &str,
) -> Result<(String, File), AtomicArtifactPublishError> {
    for _ in 0..32 {
        let sequence = ARTIFACT_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let name = format!(".{stem}.{}.{}.tmp", std::process::id(), sequence);
        match openat(
            directory.directory(),
            &name,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        ) {
            Ok(file) => return Ok((name, File::from(file))),
            Err(Errno::EXIST) => continue,
            Err(source) => {
                return Err(AtomicArtifactPublishError::CreateTemporary(errno_as_io(
                    source,
                )));
            }
        }
    }
    Err(AtomicArtifactPublishError::TemporaryNameExhausted)
}

fn verify_existing_artifact(
    directory: &CommissioningArtifactDirectory,
    name: &str,
    expected: &[u8],
) -> Result<(), AtomicArtifactPublishError> {
    directory
        .verify_binding()
        .map_err(AtomicArtifactPublishError::Directory)?;
    let file = openat(
        directory.directory(),
        name,
        OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
        Mode::empty(),
    )
    .map(File::from)
    .map_err(|source| AtomicArtifactPublishError::ReadExisting(errno_as_io(source)))?;
    let opened = fstat(&file)
        .map_err(|source| AtomicArtifactPublishError::InspectExisting(errno_as_io(source)))?;
    let named = statat(directory.directory(), name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|source| AtomicArtifactPublishError::InspectExisting(errno_as_io(source)))?;
    if FileType::from_raw_mode(opened.st_mode) != FileType::RegularFile
        || opened.st_nlink != 1
        || opened.st_uid != current_uid()
        || u32::from(opened.st_mode) & 0o777 != 0o600
        || opened.st_dev != named.st_dev
        || opened.st_ino != named.st_ino
    {
        return Err(AtomicArtifactPublishError::UnsafeExisting);
    }
    let mut observed = Vec::new();
    (&file)
        .take(
            u64::try_from(MAX_COMMISSIONING_ARTIFACT_BYTES + 1)
                .expect("artifact hard bound fits u64"),
        )
        .read_to_end(&mut observed)
        .map_err(AtomicArtifactPublishError::ReadExisting)?;
    if observed != expected {
        return Err(AtomicArtifactPublishError::ExistingContentMismatch);
    }
    let named_after = statat(directory.directory(), name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|source| AtomicArtifactPublishError::InspectExisting(errno_as_io(source)))?;
    if opened.st_dev != named_after.st_dev || opened.st_ino != named_after.st_ino {
        return Err(AtomicArtifactPublishError::UnsafeExisting);
    }
    directory
        .verify_binding()
        .map_err(AtomicArtifactPublishError::Directory)?;
    Ok(())
}

fn require_published_identity(
    directory: &CommissioningArtifactDirectory,
    name: &str,
    expected: &rustix::fs::Stat,
) -> Result<(), AtomicArtifactPublishError> {
    let published = statat(directory.directory(), name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|source| AtomicArtifactPublishError::InspectPublished(errno_as_io(source)))?;
    if published.st_dev != expected.st_dev || published.st_ino != expected.st_ino {
        return Err(AtomicArtifactPublishError::PublishedIdentityChanged);
    }
    Ok(())
}

fn unlink_artifact_temporary(
    directory: &CommissioningArtifactDirectory,
    name: &str,
) -> Result<(), io::Error> {
    match unlinkat(directory.directory(), name, AtFlags::empty()) {
        Ok(()) | Err(Errno::NOENT) => Ok(()),
        Err(source) => Err(errno_as_io(source)),
    }
}

fn artifact_error_with_cleanup(
    directory: &CommissioningArtifactDirectory,
    name: &str,
    failure: AtomicArtifactPublishError,
) -> AtomicArtifactPublishError {
    match unlink_artifact_temporary(directory, name) {
        Ok(()) => failure,
        Err(cleanup) => AtomicArtifactPublishError::CleanupAfterFailure {
            failure: Box::new(failure),
            cleanup,
        },
    }
}

fn errno_as_io(source: Errno) -> io::Error {
    io::Error::from_raw_os_error(source.raw_os_error())
}

fn sha256(bytes: &[u8]) -> [u8; CONTENT_DIGEST_BYTES] {
    Sha256::digest(bytes).into()
}

fn lower_hex(digest: [u8; CONTENT_DIGEST_BYTES]) -> String {
    let mut output = String::with_capacity(CONTENT_DIGEST_BYTES * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("formatting into String cannot fail");
    }
    output
}

fn canonical_sha256(digest: [u8; CONTENT_DIGEST_BYTES]) -> String {
    format!("{SHA256_PREFIX}{}", lower_hex(digest))
}

#[cfg(test)]
mod tests {
    use std::fs::Permissions;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::{Arc, Mutex};

    use serde_json::{Value, json};

    use super::*;

    const DT_NS: u64 = 50_000_000;
    const DT_S: f64 = 0.05;
    const WHEELBASE_M: f64 = 0.30;
    const LEFT_GAIN: f64 = 0.009;
    const LEFT_TAU_S: f64 = 0.31;
    const RIGHT_GAIN: f64 = 0.011;
    const RIGHT_TAU_S: f64 = 0.57;

    fn policy_json() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "commissioning": {
                "schema_version": 1,
                "expected_controller_session_id": "stm32-session-commissioning-1",
                "expected_visual_velocity_source_id": "oak-slam-body-velocity-v1",
                "expected_imu_calibration_id": "oak-imu-calibration-1",
                "symmetric_pwm_percent": 30,
                "spin_pwm_percent": 25,
                "max_abs_pwm_percent": 30,
                "excitation_duration_ns": 200_000_000_u64,
                "zero_dwell_duration_ns": 100_000_000_u64,
                "application_timeout_ns": 100_000_000_u64,
                "max_visual_age_ns": 60_000_000_u64,
                "max_imu_age_ns": 60_000_000_u64,
                "max_controller_age_ns": 60_000_000_u64,
                "max_abs_stationary_forward_velocity_mps": 0.03,
                "max_abs_stationary_yaw_rate_rad_s": 0.10,
                "max_total_duration_ns": 20_000_000_000_u64,
                "cycles": 4,
                "max_excitation_steps": 16
            },
            "fit": {
                "schema_version": 1,
                "expected_robot_id": "kiko-1",
                "expected_controller_session_id": "stm32-session-commissioning-1",
                "expected_visual_velocity_source_id": "oak-slam-body-velocity-v1",
                "expected_imu_calibration_id": "oak-imu-calibration-1",
                "wheelbase_calibration_id": "wheelbase-measured-1",
                "wheelbase_m": WHEELBASE_M,
                "min_sample_period_s": 0.049,
                "max_sample_period_s": 0.051,
                "max_sample_period_ratio": 1.01,
                "max_abs_observed_forward_velocity_mps": 3.0,
                "max_abs_observed_yaw_rate_rad_s": 20.0,
                "min_samples": 80,
                "max_samples": 1000,
                "holdout_stride": 5,
                "min_training_transitions": 30,
                "min_holdout_transitions": 5,
                "min_abs_excitation_pwm_percent": 10,
                "min_symmetric_transitions": 8,
                "min_spin_transitions": 8,
                "min_zero_transitions": 8,
                "min_positive_transitions_per_wheel": 8,
                "min_negative_transitions_per_wheel": 8,
                "min_command_changes": 12,
                "min_time_constant_s": 0.05,
                "max_time_constant_s": 2.0,
                "time_constant_bound_margin_fraction": 0.01,
                "min_abs_velocity_gain_mps_per_pwm_percent": 0.001,
                "max_abs_velocity_gain_mps_per_pwm_percent": 0.05,
                "require_positive_velocity_gain": true,
                "max_normal_matrix_condition_number": 1.0e12,
                "min_log_time_constant_sensitivity_mps": 1.0e-8,
                "max_holdout_wheel_velocity_rmse_mps": 1.0e-6,
                "max_holdout_forward_velocity_rmse_mps": 1.0e-6,
                "max_holdout_yaw_rate_rmse_rad_s": 1.0e-5,
                "max_holdout_abs_wheel_velocity_error_mps": 1.0e-5
            },
            "lateral": {
                "body_frame_id": BODY_FRAME_ID,
                "holdout_stride": 5,
                "minimum_training_samples": 60,
                "minimum_holdout_samples": 15,
                "bound_margin_mps": 0.005,
                "maximum_accepted_bound_mps": 0.05,
                "maximum_abs_observed_lateral_velocity_mps": 0.10,
                "scope_label": "indoor-flat-floor-visual-lateral-v1"
            },
            "maximum_aligned_observation_skew_ns": 5_000_000_u64,
            "maximum_sample_gap_ns": 60_000_000_u64,
            "maximum_controller_sequence_gap": 1,
            "model_id": "kiko-base-proposed-v1",
            "model_version": 1
        }))
        .expect("policy JSON")
    }

    fn parsed_policy() -> NanoBaseCommissioningPolicyV1 {
        NanoBaseCommissioningPolicyV1::parse_json(&policy_json()).expect("test policy")
    }

    fn parsed_three_sample_policy() -> NanoBaseCommissioningPolicyV1 {
        let mut value: Value = serde_json::from_slice(&policy_json()).expect("policy value");
        value["fit"]["min_samples"] = json!(3);
        value["fit"]["max_samples"] = json!(3);
        value["fit"]["min_training_transitions"] = json!(1);
        value["fit"]["min_holdout_transitions"] = json!(1);
        value["fit"]["min_symmetric_transitions"] = json!(0);
        value["fit"]["min_spin_transitions"] = json!(0);
        value["fit"]["min_zero_transitions"] = json!(0);
        value["fit"]["min_positive_transitions_per_wheel"] = json!(0);
        value["fit"]["min_negative_transitions_per_wheel"] = json!(0);
        value["fit"]["min_command_changes"] = json!(0);
        value["lateral"]["minimum_training_samples"] = json!(2);
        value["lateral"]["minimum_holdout_samples"] = json!(1);
        let bytes = serde_json::to_vec(&value).expect("bounded policy JSON");
        NanoBaseCommissioningPolicyV1::parse_json(&bytes).expect("three-sample policy")
    }

    fn authority(policy: NanoBaseCommissioningPolicyV1) -> AdmittedAttendedCommissioning {
        AdmittedAttendedCommissioning::from_verified_attended_admission(
            policy.commissioning().expected_controller_session_id(),
            [0x11; 32],
            [0x22; 32],
            policy.commissioning().max_abs_pwm_percent().get(),
            0,
            100_000_000_000,
        )
        .expect("test authority")
    }

    #[derive(Default)]
    struct MemoryJournal {
        bytes: Vec<u8>,
        records: u64,
        fail_after: Option<u64>,
        fail_finalize: bool,
        shared_bytes: Option<Arc<Mutex<Vec<u8>>>>,
    }

    #[derive(Debug)]
    struct MemoryJournalError;

    impl fmt::Display for MemoryJournalError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("injected memory journal failure")
        }
    }

    impl std::error::Error for MemoryJournalError {}

    impl DurableCommissioningJournal for MemoryJournal {
        type Error = MemoryJournalError;

        fn append_durable(&mut self, record: &[u8]) -> Result<(), Self::Error> {
            if self.fail_after == Some(self.records) {
                return Err(MemoryJournalError);
            }
            self.bytes.extend_from_slice(record);
            if let Some(shared) = &self.shared_bytes {
                shared
                    .lock()
                    .expect("shared journal bytes")
                    .extend_from_slice(record);
            }
            self.records += 1;
            Ok(())
        }

        fn finalize_durable(&mut self) -> Result<CommissioningJournalCommit, Self::Error> {
            if self.fail_finalize {
                return Err(MemoryJournalError);
            }
            Ok(CommissioningJournalCommit {
                content_sha256: sha256(&self.bytes),
                record_count: self.records,
                byte_count: u64::try_from(self.bytes.len()).expect("small journal"),
            })
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct FakeActuatorSnapshot {
        sequence: u64,
        pwm: AppliedPwm,
        emergency_zero_count: u32,
    }

    struct FakeActuator {
        shared: Arc<Mutex<FakeActuatorSnapshot>>,
        mismatch_next: bool,
    }

    #[derive(Debug)]
    struct FakeActuatorError;

    impl fmt::Display for FakeActuatorError {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("injected fake actuator failure")
        }
    }

    impl std::error::Error for FakeActuatorError {}

    impl FakeActuator {
        fn new() -> (Self, Arc<Mutex<FakeActuatorSnapshot>>) {
            let shared = Arc::new(Mutex::new(FakeActuatorSnapshot {
                sequence: 0,
                pwm: AppliedPwm::try_new(0, 0).expect("zero"),
                emergency_zero_count: 0,
            }));
            (
                Self {
                    shared: Arc::clone(&shared),
                    mismatch_next: false,
                },
                shared,
            )
        }

        fn receipt(
            state: &mut FakeActuatorSnapshot,
            pwm: AppliedPwm,
        ) -> ExactCommissioningControllerReceipt {
            state.sequence += 1;
            state.pwm = pwm;
            ExactCommissioningControllerReceipt {
                observed_at_ns: state.sequence * 1_000,
                applied_command_sequence: state.sequence,
                applied_pwm: pwm,
            }
        }
    }

    impl SoleCommissioningActuator for FakeActuator {
        type Error = FakeActuatorError;

        fn apply(
            &mut self,
            command: CanonicalPwmCommand,
        ) -> Result<ExactCommissioningControllerReceipt, Self::Error> {
            let mut state = self.shared.lock().expect("fake actuator lock");
            let requested = AppliedPwm::from_validated(command.left(), command.right());
            let requested_is_zero = requested.left().get() == 0 && requested.right().get() == 0;
            let pwm = if self.mismatch_next && !requested_is_zero {
                self.mismatch_next = false;
                AppliedPwm::try_new(0, 0).expect("zero")
            } else {
                requested
            };
            Ok(Self::receipt(&mut state, pwm))
        }

        fn emergency_zero(&mut self) -> Result<ExactCommissioningControllerReceipt, Self::Error> {
            let mut state = self.shared.lock().expect("fake actuator lock");
            state.emergency_zero_count += 1;
            let zero = AppliedPwm::try_new(0, 0).expect("zero");
            Ok(Self::receipt(&mut state, zero))
        }
    }

    fn sample(
        now_ns: u64,
        progress: NanoBaseCommissioningProgress,
        forward_mps: f64,
        lateral_mps: f64,
        yaw_rate_rad_s: f64,
    ) -> NanoBaseCommissioningSampleV1Dto {
        NanoBaseCommissioningSampleV1Dto {
            now_ns,
            evidence: CommissioningEvidenceV1Dto {
                controller_session_id: "stm32-session-commissioning-1".to_owned(),
                visual_velocity_source_id: "oak-slam-body-velocity-v1".to_owned(),
                imu_calibration_id: "oak-imu-calibration-1".to_owned(),
                controller_observed_at_ns: now_ns,
                visual_observed_at_ns: now_ns,
                imu_observed_at_ns: now_ns,
                applied_command_sequence: progress.last_applied_sequence,
                applied_left_pwm_percent: progress.requested_left_pwm_percent,
                applied_right_pwm_percent: progress.requested_right_pwm_percent,
                visual_forward_velocity_mps: forward_mps,
                calibrated_imu_yaw_rate_rad_s: yaw_rate_rad_s,
            },
            visual_body_lateral_velocity_mps: lateral_mps,
            visual_body_frame_id: BODY_FRAME_ID.to_owned(),
        }
    }

    fn advance_wheel(velocity: f64, pwm: i8, gain: f64, tau_s: f64) -> f64 {
        let ratio = DT_S / tau_s;
        (-ratio).exp() * velocity + gain * -(-ratio).exp_m1() * f64::from(pwm)
    }

    fn complete_synthetic_session(
        policy: NanoBaseCommissioningPolicyV1,
        left_gain: f64,
        right_gain: f64,
    ) -> NanoBaseCommissioningSession<FakeActuator, MemoryJournal> {
        complete_synthetic_session_with(
            policy,
            authority(policy),
            MemoryJournal::default(),
            left_gain,
            right_gain,
        )
    }

    fn complete_synthetic_session_with(
        policy: NanoBaseCommissioningPolicyV1,
        authority: AdmittedAttendedCommissioning,
        journal: MemoryJournal,
        left_gain: f64,
        right_gain: f64,
    ) -> NanoBaseCommissioningSession<FakeActuator, MemoryJournal> {
        let (actuator, _) = FakeActuator::new();
        let mut session = NanoBaseCommissioningSession::start(policy, authority, actuator, journal)
            .expect("session starts");
        let mut now_ns = 0_u64;
        let mut left_velocity_mps = 0.0_f64;
        let mut right_velocity_mps = 0.0_f64;
        for _ in 0..500 {
            now_ns += DT_NS;
            let before = session.progress();
            if before.requested_left_pwm_percent == 0 && before.requested_right_pwm_percent == 0 {
                left_velocity_mps = 0.0;
                right_velocity_mps = 0.0;
            } else {
                left_velocity_mps = advance_wheel(
                    left_velocity_mps,
                    before.requested_left_pwm_percent,
                    left_gain,
                    LEFT_TAU_S,
                );
                right_velocity_mps = advance_wheel(
                    right_velocity_mps,
                    before.requested_right_pwm_percent,
                    right_gain,
                    RIGHT_TAU_S,
                );
            }
            let progress = session
                .advance(
                    sample(
                        now_ns,
                        before,
                        0.5 * (left_velocity_mps + right_velocity_mps),
                        0.01,
                        (right_velocity_mps - left_velocity_mps) / WHEELBASE_M,
                    ),
                    CommissioningExternalSignal::Continue,
                )
                .expect("synthetic evidence");
            if progress.state == CommissioningState::Completed {
                break;
            }
        }
        assert_eq!(session.progress().state, CommissioningState::Completed);
        session
    }

    #[test]
    fn policy_is_strict_and_cross_binds_all_core_stream_identities() {
        let policy = parsed_policy();
        assert_eq!(policy.commissioning().program_steps().get(), 16);

        let mut unknown: Value = serde_json::from_slice(&policy_json()).expect("policy value");
        unknown["unexpected"] = json!(true);
        let bytes = serde_json::to_vec(&unknown).expect("unknown policy");
        assert!(matches!(
            NanoBaseCommissioningPolicyV1::parse_json(&bytes),
            Err(NanoBaseCommissioningPolicyParseError::JsonDecode(_))
        ));

        let mut mismatch: Value = serde_json::from_slice(&policy_json()).expect("policy value");
        mismatch["fit"]["expected_imu_calibration_id"] = json!("different-calibration");
        let bytes = serde_json::to_vec(&mismatch).expect("mismatched policy");
        assert!(matches!(
            NanoBaseCommissioningPolicyV1::parse_json(&bytes),
            Err(NanoBaseCommissioningPolicyParseError::CoreIdentityMismatch(source))
                if source.field == "imu_calibration_id"
        ));

        let mut impossible_lateral: Value =
            serde_json::from_slice(&policy_json()).expect("policy value");
        impossible_lateral["fit"]["max_samples"] = json!(80);
        impossible_lateral["lateral"]["minimum_training_samples"] = json!(70);
        impossible_lateral["lateral"]["minimum_holdout_samples"] = json!(15);
        let bytes = serde_json::to_vec(&impossible_lateral).expect("impossible lateral policy");
        assert!(matches!(
            NanoBaseCommissioningPolicyV1::parse_json(&bytes),
            Err(
                NanoBaseCommissioningPolicyParseError::LateralSampleRequirementAboveFitCapacity {
                    required: 85,
                    maximum: 80,
                }
            )
        ));
    }

    #[test]
    fn exact_receipt_mismatch_fails_closed_without_advancing() {
        let policy = parsed_policy();
        let (actuator, shared) = FakeActuator::new();
        let mut session = NanoBaseCommissioningSession::start(
            policy,
            authority(policy),
            actuator,
            MemoryJournal::default(),
        )
        .expect("session starts stopped");
        let mut invalid = sample(DT_NS, session.progress(), 0.0, 0.0, 0.0);
        invalid.evidence.applied_command_sequence += 1;
        let failure = session
            .advance(invalid, CommissioningExternalSignal::Continue)
            .expect_err("mismatched stream must stop");
        assert!(matches!(
            failure.fault,
            NanoBaseCommissioningFault::AppliedReceiptMismatch { .. }
        ));
        assert!(matches!(failure.stop, ExactFailClosedStop::Applied { .. }));
        let snapshot = *shared.lock().expect("fake actuator lock");
        assert_eq!(snapshot.pwm, AppliedPwm::try_new(0, 0).expect("zero"));
        assert!(snapshot.emergency_zero_count >= 2);
    }

    #[test]
    fn unchanged_request_is_reapplied_to_refresh_the_bounded_controller_lease() {
        let policy = parsed_policy();
        let (actuator, shared) = FakeActuator::new();
        let mut session = NanoBaseCommissioningSession::start(
            policy,
            authority(policy),
            actuator,
            MemoryJournal::default(),
        )
        .expect("session starts stopped");
        assert_eq!(session.progress().last_applied_sequence, 1);

        let progress = session
            .advance(
                sample(DT_NS, session.progress(), 0.0, 0.0, 0.0),
                CommissioningExternalSignal::Continue,
            )
            .expect("initial unchanged zero is refreshed");
        assert_eq!(progress.last_applied_sequence, 2);
        let snapshot = *shared.lock().expect("fake actuator lock");
        assert_eq!(snapshot.sequence, 2);
        assert_eq!(snapshot.pwm, AppliedPwm::try_new(0, 0).expect("zero"));
    }

    #[test]
    fn too_short_sample_period_precedes_observation_journal_and_forces_zero() {
        let policy = parsed_policy();
        let (actuator, shared) = FakeActuator::new();
        let journal = MemoryJournal {
            // Start contributes three records; the first admitted sample adds
            // observation, refreshed request, and exact receipt. If the next
            // invalid observation were journaled, it would fail at record 6
            // before the typed sample-period fault could be returned.
            fail_after: Some(6),
            ..MemoryJournal::default()
        };
        let mut session =
            NanoBaseCommissioningSession::start(policy, authority(policy), actuator, journal)
                .expect("session starts stopped");
        session
            .advance(
                sample(DT_NS, session.progress(), 0.0, 0.0, 0.0),
                CommissioningExternalSignal::Continue,
            )
            .expect("first sample");

        let failure = session
            .advance(
                sample(DT_NS + 1, session.progress(), 0.0, 0.0, 0.0),
                CommissioningExternalSignal::Continue,
            )
            .expect_err("sub-minimum period must fail closed");
        assert!(matches!(
            failure.fault,
            NanoBaseCommissioningFault::SampleGapTooShort {
                previous_ns: DT_NS,
                current_ns,
                minimum_ns: 49_000_000,
            } if current_ns == DT_NS + 1
        ));
        assert!(matches!(failure.stop, ExactFailClosedStop::Applied { .. }));
        assert!(matches!(
            failure.terminal_journal,
            CommissioningTerminalJournalStatus::Failed(_)
        ));
        assert_eq!(session.progress().samples_journaled, 1);
        assert_eq!(
            shared.lock().expect("fake actuator lock").pwm,
            AppliedPwm::try_new(0, 0).expect("zero")
        );
    }

    #[test]
    fn journal_failure_precedes_state_advance_and_forces_exact_zero() {
        let policy = parsed_policy();
        let (actuator, shared) = FakeActuator::new();
        let journal = MemoryJournal {
            // Session start, requested initial zero, and its exact receipt are
            // the first three records. Reject the first observation.
            fail_after: Some(3),
            ..MemoryJournal::default()
        };
        let mut session =
            NanoBaseCommissioningSession::start(policy, authority(policy), actuator, journal)
                .expect("session starts stopped");
        let failure = session
            .advance(
                sample(DT_NS, session.progress(), 0.0, 0.0, 0.0),
                CommissioningExternalSignal::Continue,
            )
            .expect_err("journal failure must fail closed");
        assert!(matches!(
            failure.fault,
            NanoBaseCommissioningFault::Journal(_)
        ));
        assert!(matches!(failure.stop, ExactFailClosedStop::Applied { .. }));
        let snapshot = *shared.lock().expect("fake actuator lock");
        assert_eq!(snapshot.pwm, AppliedPwm::try_new(0, 0).expect("zero"));
        assert_eq!(session.progress().samples_journaled, 0);
    }

    #[test]
    fn fit_sample_limit_precedes_observation_journal_and_forces_durable_exact_zero() {
        let policy = parsed_three_sample_policy();
        assert_eq!(policy.fit().max_samples().get(), 3);
        let (actuator, shared) = FakeActuator::new();
        let mut session = NanoBaseCommissioningSession::start(
            policy,
            authority(policy),
            actuator,
            MemoryJournal::default(),
        )
        .expect("session starts stopped");

        for ordinal in 1_u64..=3 {
            session
                .advance(
                    sample(ordinal * DT_NS, session.progress(), 0.0, 0.0, 0.0),
                    CommissioningExternalSignal::Continue,
                )
                .expect("samples through the parsed fit limit are admitted");
        }
        assert_eq!(session.progress().samples_journaled, 3);

        let failure = session
            .advance(
                sample(4 * DT_NS, session.progress(), 0.0, 0.0, 0.0),
                CommissioningExternalSignal::Continue,
            )
            .expect_err("the first observation above the fit limit must fail closed");
        assert!(matches!(
            failure.fault,
            NanoBaseCommissioningFault::SampleLimitReached {
                accepted: 3,
                maximum: 3,
            }
        ));
        assert!(matches!(failure.stop, ExactFailClosedStop::Applied { .. }));
        assert!(matches!(
            failure.terminal_journal,
            CommissioningTerminalJournalStatus::Durable
        ));
        assert_eq!(session.progress().samples_journaled, 3);
        let snapshot = *shared.lock().expect("fake actuator lock");
        assert_eq!(snapshot.pwm, AppliedPwm::try_new(0, 0).expect("zero"));
        assert!(snapshot.emergency_zero_count >= 2);
    }

    #[test]
    fn mismatched_actuator_receipt_for_requested_motion_forces_exact_zero() {
        let policy = parsed_policy();
        let (mut actuator, shared) = FakeActuator::new();
        actuator.mismatch_next = true;
        let mut session = NanoBaseCommissioningSession::start(
            policy,
            authority(policy),
            actuator,
            MemoryJournal::default(),
        )
        .expect("session starts stopped");
        let mut failure = None;
        for step in 1_u64..=10 {
            let result = session.advance(
                sample(step * DT_NS, session.progress(), 0.0, 0.0, 0.0),
                CommissioningExternalSignal::Continue,
            );
            if let Err(observed) = result {
                failure = Some(observed);
                break;
            }
        }
        let failure = failure.expect("an inexact motion receipt must fail closed");
        assert!(matches!(
            failure.fault,
            NanoBaseCommissioningFault::AppliedCommandMismatch { .. }
        ));
        assert!(matches!(failure.stop, ExactFailClosedStop::Applied { .. }));
        assert_eq!(
            shared.lock().expect("fake actuator lock").pwm,
            AppliedPwm::try_new(0, 0).expect("zero")
        );
    }

    #[test]
    fn every_external_terminal_signal_requires_an_exact_zero() {
        for signal in [
            CommissioningExternalSignal::CancelledByOperator,
            CommissioningExternalSignal::ControllerFault,
            CommissioningExternalSignal::VisualFault,
            CommissioningExternalSignal::ImuFault,
            CommissioningExternalSignal::SupervisorFault,
        ] {
            let policy = parsed_policy();
            let (actuator, shared) = FakeActuator::new();
            let mut session = NanoBaseCommissioningSession::start(
                policy,
                authority(policy),
                actuator,
                MemoryJournal::default(),
            )
            .expect("session");
            let failure = session
                .advance(sample(DT_NS, session.progress(), 0.0, 0.0, 0.0), signal)
                .expect_err("terminal signal");
            assert!(matches!(failure.stop, ExactFailClosedStop::Applied { .. }));
            assert!(shared.lock().expect("fake actuator lock").pwm.left().get() == 0);
        }
    }

    #[test]
    fn lateral_holdout_is_never_used_to_inflate_its_own_bound() {
        let policy = LateralHoldoutPolicyV1 {
            holdout_stride: NonZeroU16::new(3).expect("nonzero"),
            minimum_training_samples: NonZeroU32::new(2).expect("nonzero"),
            minimum_holdout_samples: NonZeroU32::new(1).expect("nonzero"),
            bound_margin_mps: 0.005,
            maximum_accepted_bound_mps: 0.05,
            maximum_abs_observed_lateral_velocity_mps: 0.10,
            scope_label: CommissioningLabel::parse("scope", "test-scope".to_owned())
                .expect("label"),
        };
        let identification = IdentificationSampleV1Dto {
            observed_at_ns: 1,
            applied_command_sequence: 1,
            applied_left_pwm_percent: 0,
            applied_right_pwm_percent: 0,
            visual_forward_velocity_mps: 0.0,
            calibrated_imu_yaw_rate_rad_s: 0.0,
        };
        let samples = [
            CollectedSample {
                identification,
                visual_body_lateral_velocity_mps: 0.01,
                holdout: false,
            },
            CollectedSample {
                identification,
                visual_body_lateral_velocity_mps: -0.01,
                holdout: false,
            },
            CollectedSample {
                identification,
                visual_body_lateral_velocity_mps: 0.03,
                holdout: true,
            },
        ];
        assert!(matches!(
            derive_lateral_validity(&samples, policy),
            Err(LateralValidityError::HoldoutOutsideDerivedBound {
                sample_ordinal: 2,
                ..
            })
        ));
    }

    #[test]
    fn file_journal_commits_exact_durable_bytes() {
        let directory = test_directory("journal");
        let path = directory.join("commissioning.ndjson");
        let mut journal = FileCommissioningJournal::create_new(&path).expect("create journal");
        let record = b"{\"kind\":\"test\"}\n";
        journal.append_durable(record).expect("durable append");
        let commit = journal.finalize_durable().expect("durable commit");
        assert_eq!(commit.content_sha256(), sha256(record));
        assert_eq!(commit.record_count(), 1);
        assert_eq!(
            commit.byte_count(),
            u64::try_from(record.len()).expect("small record")
        );
        drop(journal);
        assert_eq!(fs::read(&path).expect("journal bytes"), record);
        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn synthetic_attended_run_publishes_only_content_addressed_proposals() {
        let policy = parsed_policy();
        let (actuator, _shared) = FakeActuator::new();
        let mut session = NanoBaseCommissioningSession::start(
            policy,
            authority(policy),
            actuator,
            MemoryJournal::default(),
        )
        .expect("session starts");
        let mut now_ns = 0_u64;
        let mut left_velocity_mps = 0.0_f64;
        let mut right_velocity_mps = 0.0_f64;
        for _ in 0..500 {
            now_ns += DT_NS;
            let before = session.progress();
            if before.requested_left_pwm_percent == 0 && before.requested_right_pwm_percent == 0 {
                // The fixture models the controller's configured active brake
                // at exact zero. Physical qualification must measure, not
                // assume, this stationarity behavior.
                left_velocity_mps = 0.0;
                right_velocity_mps = 0.0;
            } else {
                left_velocity_mps = advance_wheel(
                    left_velocity_mps,
                    before.requested_left_pwm_percent,
                    LEFT_GAIN,
                    LEFT_TAU_S,
                );
                right_velocity_mps = advance_wheel(
                    right_velocity_mps,
                    before.requested_right_pwm_percent,
                    RIGHT_GAIN,
                    RIGHT_TAU_S,
                );
            }
            let forward_mps = 0.5 * (left_velocity_mps + right_velocity_mps);
            let yaw_rate_rad_s = (right_velocity_mps - left_velocity_mps) / WHEELBASE_M;
            let progress = session
                .advance(
                    sample(now_ns, before, forward_mps, 0.01, yaw_rate_rad_s),
                    CommissioningExternalSignal::Continue,
                )
                .expect("synthetic evidence");
            if progress.state == CommissioningState::Completed {
                break;
            }
        }
        assert_eq!(session.progress().state, CommissioningState::Completed);
        assert!(session.progress().exact_zero_applied);

        let directory = test_directory("publish");
        let admitted_directory =
            CommissioningArtifactDirectory::inspect(&directory).expect("private directory");
        let proposal = session
            .publish_proposal(&admitted_directory)
            .expect("verified proposal");
        assert_eq!(proposal.activation_status, PROPOSAL_ACTIVATION_STATUS);
        assert!(proposal.dataset.path.exists());
        assert!(proposal.proposed_plant.path.exists());
        assert!(proposal.proposal_evidence.path.exists());
        assert!(
            proposal
                .dataset
                .path
                .file_name()
                .expect("dataset filename")
                .to_string_lossy()
                .contains(&lower_hex(proposal.dataset.content_sha256))
        );
        assert_ne!(
            proposal.dataset.content_sha256,
            proposal.proposed_plant.content_sha256
        );
        assert!(
            (proposal
                .lateral_validity
                .maximum_absolute_lateral_velocity_mps()
                - 0.015)
                .abs()
                < 1.0e-12
        );
        let plant: Value =
            serde_json::from_slice(&fs::read(&proposal.proposed_plant.path).expect("plant bytes"))
                .expect("plant JSON");
        assert_eq!(
            plant["evidence"]["kind"],
            Value::String("claimed_physical_identification".to_owned())
        );
        assert_eq!(
            plant["evidence"]["dataset_content_id"],
            Value::String(canonical_sha256(proposal.dataset.content_sha256))
        );
        let dataset: Value =
            serde_json::from_slice(&fs::read(&proposal.dataset.path).expect("dataset bytes"))
                .expect("dataset JSON");
        assert_eq!(
            dataset["aligned_evidence"]["samples"][0]["visual_body_lateral_velocity_mps"],
            json!(0.01)
        );
        let evidence: Value = serde_json::from_slice(
            &fs::read(&proposal.proposal_evidence.path).expect("proposal evidence bytes"),
        )
        .expect("proposal evidence JSON");
        assert_eq!(
            evidence["activation_status"],
            Value::String(PROPOSAL_ACTIVATION_STATUS.to_owned())
        );
        assert_eq!(
            evidence["remaining_gate"],
            Value::String(
                "operator review, physical approval, manifest rebind, and normal production admission"
                    .to_owned()
            )
        );
        assert_eq!(
            evidence["plant_evidence_dataset_content_id"],
            Value::String(canonical_sha256(proposal.dataset.content_sha256))
        );
        assert_eq!(
            evidence["proposed_plant_artifact_sha256"],
            Value::String(canonical_sha256(proposal.proposed_plant.content_sha256))
        );
        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[cfg(feature = "nano-plant-promotion")]
    #[test]
    fn offline_promotion_reuses_completed_proposal_and_rejects_digest_link_and_approval_faults() {
        use crate::navigation::nano_plant_promotion::{PlantPromotionError, promote_review_file};

        let root = test_directory("offline-promotion");
        let proposal_directory = root.join("proposal");
        fs::create_dir(&proposal_directory).expect("proposal directory");
        fs::set_permissions(&proposal_directory, Permissions::from_mode(0o700))
            .expect("proposal mode");
        let output_root = root.join("output");
        fs::create_dir(&output_root).expect("output root");
        fs::set_permissions(&output_root, Permissions::from_mode(0o700)).expect("output mode");

        let policy_bytes = policy_json();
        let policy_path = root.join("policy.json");
        fs::write(&policy_path, &policy_bytes).expect("policy");
        let profile_bytes = b"{\"fixture\":\"commissioning-profile\"}\n";
        let profile_path = root.join("controller-profile.json");
        fs::write(&profile_path, profile_bytes).expect("profile");
        let attestation_bytes = b"{\"fixture\":\"attended-attestation\"}\n";
        let attestation_path = root.join("attestation.json");
        fs::write(&attestation_path, attestation_bytes).expect("attestation");
        let policy = NanoBaseCommissioningPolicyV1::parse_json(&policy_bytes).expect("policy");
        let exact_authority = AdmittedAttendedCommissioning::from_verified_attended_admission(
            policy.commissioning().expected_controller_session_id(),
            sha256(profile_bytes),
            sha256(attestation_bytes),
            policy.commissioning().max_abs_pwm_percent().get(),
            0,
            100_000_000_000,
        )
        .expect("authority");
        let shared_journal = Arc::new(Mutex::new(Vec::new()));
        let session = complete_synthetic_session_with(
            policy,
            exact_authority,
            MemoryJournal {
                shared_bytes: Some(Arc::clone(&shared_journal)),
                ..MemoryJournal::default()
            },
            LEFT_GAIN,
            RIGHT_GAIN,
        );
        let admitted_directory = CommissioningArtifactDirectory::inspect(&proposal_directory)
            .expect("proposal publication directory");
        let proposal = session
            .publish_proposal(&admitted_directory)
            .expect("proposal");
        let journal_bytes = shared_journal.lock().expect("journal").clone();
        let journal_path = root.join("commissioning-evidence-v1.ndjson");
        fs::write(&journal_path, &journal_bytes).expect("journal");

        let binding = |path: &Path| {
            let bytes = fs::read(path).expect("bound bytes");
            json!({
                "path": path,
                "sha256_hex": lower_hex(sha256(&bytes)),
                "bytes": bytes.len()
            })
        };
        let mut review = json!({
            "schema_version": 1,
            "promotion_id": "promotion-test-1",
            "reviewer_id": "reviewer-test",
            "approval_id": "approval-test-1",
            "approver_id": "approver-test",
            "commissioning_session_id": "commissioning-test-1",
            "sources": {
                "policy": binding(&policy_path),
                "controller_profile": binding(&profile_path),
                "attended_attestation": binding(&attestation_path),
                "journal": {
                    "artifact": binding(&journal_path),
                    "records": proposal.journal.record_count()
                },
                "dataset": binding(&proposal.dataset.path),
                "proposed_plant": binding(&proposal.proposed_plant.path),
                "proposal_evidence": binding(&proposal.proposal_evidence.path)
            },
            "physical_review": {
                "complete_journal": "reviewed_and_accepted",
                "dataset_and_reproduced_fit": "reviewed_and_accepted",
                "repeated_run_consistency": "reviewed_and_accepted",
                "wheel_wiring_and_signed_motion": "reviewed_and_accepted",
                "units_and_base_body_flu_frame": "reviewed_and_accepted",
                "surface_payload_and_envelope": "reviewed_and_accepted",
                "default_off_driver_enable": "reviewed_and_accepted",
                "driver_fault_and_estop_feedback": "reviewed_and_accepted",
                "reset_brownout_and_hard_fault": "reviewed_and_accepted",
                "independent_power_cut": "reviewed_and_accepted",
                "verified_physical_stop_semantics": "coast_verified"
            },
            "calibrations": {
                "imu_calibration_id": "oak-imu-calibration-1",
                "stereo_calibration_id": "oak-stereo-calibration-1",
                "tracking_camera_to_base_calibration_id": "oak-to-base-calibration-1"
            },
            "renderer": {
                "plant_artifact_id": "reviewed-plant-test-1",
                "plant_destination_relative_path":
                    "artifacts/reviewed-plant-test-1.json"
            }
        });
        let review_path = root.join("review.json");
        let write_review = |review: &Value| {
            fs::write(
                &review_path,
                serde_json::to_vec(review).expect("review JSON"),
            )
            .expect("review");
        };
        write_review(&review);
        let promoted = promote_review_file(&review_path, &output_root).expect("promotion");
        assert_eq!(
            fs::read(&promoted.production_plant.path).expect("promoted plant"),
            fs::read(&proposal.proposed_plant.path).expect("proposed plant")
        );
        assert!(promoted.completion_marker.path.is_file());

        review["sources"]["dataset"]["sha256_hex"] = json!("00".repeat(32));
        write_review(&review);
        assert!(matches!(
            promote_review_file(&review_path, &output_root),
            Err(PlantPromotionError::Rejected(_))
        ));
        review["sources"]["dataset"] = binding(&proposal.dataset.path);

        let mut bad_evidence: Value = serde_json::from_slice(
            &fs::read(&proposal.proposal_evidence.path).expect("proposal evidence"),
        )
        .expect("proposal evidence JSON");
        bad_evidence["proposed_plant_artifact_sha256"] = json!(canonical_sha256([0x44; 32]));
        let bad_evidence_path = root.join("bad-proposal-evidence.json");
        fs::write(
            &bad_evidence_path,
            serde_json::to_vec(&bad_evidence).expect("bad evidence"),
        )
        .expect("bad evidence");
        review["sources"]["proposal_evidence"] = binding(&bad_evidence_path);
        write_review(&review);
        assert!(matches!(
            promote_review_file(&review_path, &output_root),
            Err(PlantPromotionError::Rejected(_))
        ));

        review["approval_id"] = json!("");
        write_review(&review);
        assert!(matches!(
            promote_review_file(&review_path, &output_root),
            Err(PlantPromotionError::Invalid("approval_id"))
        ));
        fs::remove_dir_all(root).expect("remove test directory");
    }

    #[test]
    fn signed_fitted_gain_survives_proposal_parsing_and_mpc_admission() {
        use crate::navigation::mpc::{MPC_CONFIG_V1, MpcConfigV1, MpcConfigV1Dto, MpcSolver};

        let mut value: Value = serde_json::from_slice(&policy_json()).expect("policy value");
        value["fit"]["require_positive_velocity_gain"] = json!(false);
        let bytes = serde_json::to_vec(&value).expect("signed policy JSON");
        let policy = NanoBaseCommissioningPolicyV1::parse_json(&bytes).expect("signed-gain policy");
        let session = complete_synthetic_session(policy, -LEFT_GAIN, RIGHT_GAIN);
        let directory = test_directory("signed-gain");
        let admitted_directory =
            CommissioningArtifactDirectory::inspect(&directory).expect("private directory");
        let proposal = session
            .publish_proposal(&admitted_directory)
            .expect("signed-gain proposal");
        let plant_bytes = fs::read(&proposal.proposed_plant.path).expect("plant bytes");
        let plant_value: Value = serde_json::from_slice(&plant_bytes).expect("plant JSON");
        assert!(
            plant_value["left"]["velocity_gain_mps_per_pwm_percent"]
                .as_f64()
                .expect("left gain")
                < 0.0
        );
        assert!(
            plant_value["right"]["velocity_gain_mps_per_pwm_percent"]
                .as_f64()
                .expect("right gain")
                > 0.0
        );

        // Plant parsing checks both signed gain*PWM endpoints against their
        // fitted velocity envelopes. MPC construction then consumes that
        // exact parsed model without sign normalization or absolute value.
        let model = PlantModelV1::parse_json(&plant_bytes).expect("signed plant model");
        let mpc = MpcConfigV1::parse(MpcConfigV1Dto {
            schema_version: MPC_CONFIG_V1,
            horizon_steps: 2,
            step_period_s: DT_S,
            integration_substeps: 4,
            optimization_iterations: 1,
            candidates_per_wheel: 3,
            max_rollout_evaluations: 10_000,
            initial_search_radius_percent: 10,
            search_radius_decay_numerator: 1,
            search_radius_decay_denominator: 2,
            left_pwm_min_percent: -20,
            left_pwm_max_percent: 20,
            right_pwm_min_percent: -20,
            right_pwm_max_percent: 20,
            left_max_slew_percent_per_step: 20,
            right_max_slew_percent_per_step: 20,
            max_integration_tube_radius_m: 1.0,
            position_cost_per_m2: 1.0,
            heading_cost_per_rad2: 1.0,
            forward_velocity_cost_s2_per_m2: 0.0,
            yaw_rate_cost_s2_per_rad2: 0.0,
            pwm_cost_per_percent2: 0.001,
            slew_cost_per_percent2: 0.001,
            terminal_state_cost_multiplier: 2.0,
        })
        .expect("bounded MPC policy");
        MpcSolver::new(model, mpc).expect("signed model is admitted unchanged by MPC");
        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn proposal_failure_returns_explicit_stop_evidence_before_session_drop() {
        let policy = parsed_policy();
        let (actuator, shared) = FakeActuator::new();
        let mut session = NanoBaseCommissioningSession::start(
            policy,
            authority(policy),
            actuator,
            MemoryJournal {
                fail_finalize: true,
                ..MemoryJournal::default()
            },
        )
        .expect("session starts");
        let mut now_ns = 0_u64;
        let mut left_velocity_mps = 0.0_f64;
        let mut right_velocity_mps = 0.0_f64;
        for _ in 0..500 {
            now_ns += DT_NS;
            let before = session.progress();
            if before.requested_left_pwm_percent == 0 && before.requested_right_pwm_percent == 0 {
                left_velocity_mps = 0.0;
                right_velocity_mps = 0.0;
            } else {
                left_velocity_mps = advance_wheel(
                    left_velocity_mps,
                    before.requested_left_pwm_percent,
                    LEFT_GAIN,
                    LEFT_TAU_S,
                );
                right_velocity_mps = advance_wheel(
                    right_velocity_mps,
                    before.requested_right_pwm_percent,
                    RIGHT_GAIN,
                    RIGHT_TAU_S,
                );
            }
            let progress = session
                .advance(
                    sample(
                        now_ns,
                        before,
                        0.5 * (left_velocity_mps + right_velocity_mps),
                        0.01,
                        (right_velocity_mps - left_velocity_mps) / WHEELBASE_M,
                    ),
                    CommissioningExternalSignal::Continue,
                )
                .expect("synthetic evidence");
            if progress.state == CommissioningState::Completed {
                break;
            }
        }
        assert_eq!(session.progress().state, CommissioningState::Completed);
        let directory = test_directory("publish-stop");
        let admitted_directory =
            CommissioningArtifactDirectory::inspect(&directory).expect("private directory");
        let failure = session
            .publish_proposal(&admitted_directory)
            .expect_err("injected finalize failure");
        assert!(matches!(
            failure.fault.as_ref(),
            NanoBaseCommissioningPublishError::Journal(_)
        ));
        assert!(matches!(failure.stop, ExactFailClosedStop::Applied { .. }));
        let snapshot = *shared.lock().expect("fake actuator lock");
        assert_eq!(snapshot.pwm, AppliedPwm::try_new(0, 0).expect("zero"));
        assert!(
            snapshot.emergency_zero_count >= 2,
            "start and explicit publication-failure stop must both be observed"
        );
        fs::remove_dir_all(directory).expect("remove test directory");
    }

    #[test]
    fn replaced_artifact_directory_fails_publication_with_explicit_stop() {
        let policy = parsed_policy();
        let session = complete_synthetic_session(policy, LEFT_GAIN, RIGHT_GAIN);
        let directory = test_directory("publish-replaced");
        let admitted_directory =
            CommissioningArtifactDirectory::inspect(&directory).expect("private directory");
        let moved_directory = directory.with_file_name(format!(
            "{}-original",
            directory
                .file_name()
                .expect("test directory name")
                .to_string_lossy()
        ));
        fs::rename(&directory, &moved_directory).expect("rename admitted artifact directory");
        fs::create_dir(&directory).expect("create replacement directory");
        fs::set_permissions(&directory, Permissions::from_mode(0o700))
            .expect("set private replacement permissions");

        let failure = session
            .publish_proposal(&admitted_directory)
            .expect_err("path replacement must invalidate publication");
        assert!(matches!(
            failure.fault.as_ref(),
            NanoBaseCommissioningPublishError::Publish(AtomicArtifactPublishError::Directory(
                CommissioningArtifactDirectoryError::PathBindingChanged { .. }
            ))
        ));
        assert!(matches!(failure.stop, ExactFailClosedStop::Applied { .. }));

        drop(admitted_directory);
        fs::remove_dir_all(directory).expect("remove replacement directory");
        fs::remove_dir_all(moved_directory).expect("remove retained original directory");
    }

    #[test]
    fn temporary_cleanup_uses_retained_directory_after_path_replacement() {
        let directory = test_directory("cleanup-replaced");
        let admitted_directory =
            CommissioningArtifactDirectory::inspect(&directory).expect("private directory");
        let temporary_name = ".commissioning-cleanup.tmp";
        fs::write(directory.join(temporary_name), b"retained").expect("create retained temporary");
        let moved_directory = directory.with_file_name(format!(
            "{}-original",
            directory
                .file_name()
                .expect("test directory name")
                .to_string_lossy()
        ));
        fs::rename(&directory, &moved_directory).expect("rename admitted artifact directory");
        fs::create_dir(&directory).expect("create replacement directory");
        fs::set_permissions(&directory, Permissions::from_mode(0o700))
            .expect("set private replacement permissions");
        fs::write(directory.join(temporary_name), b"replacement")
            .expect("create replacement temporary");

        let failure = artifact_error_with_cleanup(
            &admitted_directory,
            temporary_name,
            AtomicArtifactPublishError::UnsafeTemporary,
        );
        assert!(matches!(
            failure,
            AtomicArtifactPublishError::UnsafeTemporary
        ));
        assert!(!moved_directory.join(temporary_name).exists());
        assert_eq!(
            fs::read(directory.join(temporary_name)).expect("replacement remains"),
            b"replacement"
        );

        drop(admitted_directory);
        fs::remove_dir_all(directory).expect("remove replacement directory");
        fs::remove_dir_all(moved_directory).expect("remove retained original directory");
    }

    fn test_directory(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "kiko-base-commissioning-{label}-{}-{}",
            std::process::id(),
            ARTIFACT_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).expect("create test directory");
        fs::set_permissions(&path, Permissions::from_mode(0o700))
            .expect("set private test permissions");
        path
    }
}
