//! Offline, non-actuating promotion of reviewed base-commissioning output.
//!
//! The tool content-binds the immutable artifacts already emitted by
//! commissioning, re-runs the existing dataset parser/fitter and plant parser,
//! and publishes renderer values. It deliberately does not reinterpret the
//! commissioning journal: its exact bytes/digest/count remain a mandatory
//! human-review boundary.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::num::NonZeroU64;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use kiko_base_commissioning::{
    BASE_IDENTIFICATION_V1, IdentificationDatasetV1, IdentificationDatasetV1Dto,
    IdentificationSampleV1Dto, IdentifiedPlantV1, LateralVelocityEvidence, fit_first_order_plant,
};
use rustix::fs::{
    AtFlags, FileType, Mode, OFlags, RenameFlags, fstat, fsync, mkdirat, openat, renameat_with,
    statat, unlinkat,
};
use rustix::io::Errno;
use serde::{Deserialize, Serialize};
use serde_json::{json, value::RawValue};
use sha2::{Digest, Sha256};

use super::mpc::{
    FitResidualsV1Dto, PLANT_MODEL_V1, PlantEvidenceV1Dto, PlantModelV1, PlantModelV1Dto,
    PlantValidityEnvelopeV1Dto, WheelPlantV1Dto,
};
use super::nano_base_commissioning::{
    CommissioningArtifactDirectory, CommissioningArtifactDirectoryError,
    MAX_COMMISSIONING_ARTIFACT_BYTES, MAX_NANO_BASE_COMMISSIONING_POLICY_JSON_BYTES,
    NANO_BASE_COMMISSIONING_ARTIFACT_V1, NanoBaseCommissioningPolicyV1,
};

pub const NANO_PLANT_PROMOTION_REVIEW_V1: u32 = 1;
const MAX_REVIEW_BYTES: usize = 128 * 1_024;
const MAX_SMALL_EVIDENCE_BYTES: usize = 128 * 1_024;
const MAX_JOURNAL_BYTES: usize = 64 * 1_024 * 1_024;
const BODY_FRAME_ID: &str = "base_body_flu";
const PROPOSED: &str = "proposed_unapproved";
const REVIEWED: &str = "reviewed_bundle_input_motion_authority_withheld";
const MAX_RENDERER_RELATIVE_PATH_BYTES: usize = 1_024;
const STAGING_NAME_ATTEMPTS: u64 = 32;
static STAGING_SEQUENCE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishedPromotionArtifact {
    pub path: PathBuf,
    pub sha256: [u8; 32],
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlantPromotionResult {
    pub directory: PathBuf,
    pub production_plant: PublishedPromotionArtifact,
    pub promotion_evidence: PublishedPromotionArtifact,
    pub renderer_values: PublishedPromotionArtifact,
    pub completion_marker: PublishedPromotionArtifact,
}

pub fn promote_review_file(
    review_path: &Path,
    output_root: &Path,
) -> Result<PlantPromotionResult, PlantPromotionError> {
    let review_bytes = read_file(review_path, MAX_REVIEW_BYTES, "review")?;
    let review_sha = sha256(&review_bytes);
    let review = Review::parse(parse_json(&review_bytes, "review")?)?;

    let policy_bytes = review
        .sources
        .policy
        .load(MAX_NANO_BASE_COMMISSIONING_POLICY_JSON_BYTES, "policy")?;
    let profile_bytes = review
        .sources
        .controller_profile
        .load(MAX_SMALL_EVIDENCE_BYTES, "controller_profile")?;
    let attestation_bytes = review
        .sources
        .attended_attestation
        .load(MAX_SMALL_EVIDENCE_BYTES, "attended_attestation")?;
    let journal_bytes = review
        .sources
        .journal
        .artifact
        .load(MAX_JOURNAL_BYTES, "journal")?;
    let dataset_bytes = review
        .sources
        .dataset
        .load(MAX_COMMISSIONING_ARTIFACT_BYTES, "dataset")?;
    let plant_bytes = review
        .sources
        .proposed_plant
        .load(super::mpc::MAX_PLANT_MODEL_JSON_BYTES, "proposed_plant")?;
    let proposal_bytes = review
        .sources
        .proposal_evidence
        .load(MAX_SMALL_EVIDENCE_BYTES, "proposal_evidence")?;

    let policy = NanoBaseCommissioningPolicyV1::parse_json(&policy_bytes)
        .map_err(|source| PlantPromotionError::Domain("policy", Box::new(source)))?;
    let dataset = DatasetEnvelope::parse(&dataset_bytes)?;
    let plant_dto: PlantModelV1Dto = parse_json(&plant_bytes, "proposed_plant")?;
    let parsed_plant = PlantModelV1::parse(plant_dto.clone())
        .map_err(|source| PlantPromotionError::Domain("proposed_plant", Box::new(source)))?;
    let proposal: ProposalEvidence = parse_json(&proposal_bytes, "proposal_evidence")?;

    let digests = InputDigests {
        policy: sha256(&policy_bytes),
        profile: sha256(&profile_bytes),
        attestation: sha256(&attestation_bytes),
        journal: sha256(&journal_bytes),
        dataset: sha256(&dataset_bytes),
        plant: sha256(&plant_bytes),
        proposal: sha256(&proposal_bytes),
    };
    dataset.verify_links(digests)?;
    proposal.verify(&dataset, policy, digests)?;
    if review.calibrations.imu_calibration_id != dataset.payload.imu_calibration_id {
        return Err(PlantPromotionError::Rejected(
            "review IMU calibration ID does not match the dataset",
        ));
    }

    let lateral = dataset.reproduce_lateral(policy)?;
    let fit_dataset = IdentificationDatasetV1::parse(dataset.identification_dto()?, policy.fit())
        .map_err(|source| PlantPromotionError::Domain("dataset", Box::new(source)))?;
    let fit = fit_first_order_plant(&fit_dataset, policy.fit())
        .map_err(|source| PlantPromotionError::Domain("fit", Box::new(source)))?;
    let reproduced = plant_from_fit(fit, lateral, policy, digests.dataset);
    if !plant_equivalent(&plant_dto, &reproduced) {
        return Err(PlantPromotionError::Rejected(
            "proposed plant does not match the deterministic re-fit/envelope",
        ));
    }
    if parsed_plant.model_id().as_str() != policy.model_id()
        || parsed_plant.model_version() != policy.model_version()
    {
        return Err(PlantPromotionError::Rejected(
            "plant model identity does not match commissioning policy",
        ));
    }
    let physical = match &plant_dto.evidence {
        PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
            dataset_content_id,
            identification_method_id,
            sample_count,
            residuals,
        } => PhysicalPlantEvidence {
            dataset_content_id,
            identification_method_id,
            sample_count: NonZeroU64::new(*sample_count).ok_or(PlantPromotionError::Rejected(
                "plant physical sample count is zero",
            ))?,
            residuals: *residuals,
        },
        PlantEvidenceV1Dto::SyntheticFixture { .. } => {
            return Err(PlantPromotionError::Rejected(
                "synthetic plant cannot be promoted",
            ));
        }
    };

    publish(
        output_root,
        review_sha,
        &review,
        &plant_bytes,
        parsed_plant,
        physical,
        digests,
    )
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReviewDto {
    schema_version: u32,
    promotion_id: String,
    reviewer_id: String,
    approval_id: String,
    approver_id: String,
    commissioning_session_id: String,
    sources: SourcesDto,
    physical_review: PhysicalReview,
    calibrations: Calibrations,
    renderer: RendererInput,
}

#[derive(Debug)]
struct Review {
    promotion_id: String,
    reviewer_id: String,
    approval_id: String,
    approver_id: String,
    commissioning_session_id: String,
    sources: Sources,
    physical_review: PhysicalReview,
    calibrations: Calibrations,
    renderer: RendererInput,
}

impl Review {
    fn parse(dto: ReviewDto) -> Result<Self, PlantPromotionError> {
        if dto.schema_version != NANO_PLANT_PROMOTION_REVIEW_V1 {
            return Err(PlantPromotionError::Rejected(
                "unsupported plant-promotion review schema",
            ));
        }
        Ok(Self {
            promotion_id: id("promotion_id", dto.promotion_id)?,
            reviewer_id: id("reviewer_id", dto.reviewer_id)?,
            approval_id: id("approval_id", dto.approval_id)?,
            approver_id: id("approver_id", dto.approver_id)?,
            commissioning_session_id: id("commissioning_session_id", dto.commissioning_session_id)?,
            sources: Sources::parse(dto.sources)?,
            physical_review: dto.physical_review,
            calibrations: dto.calibrations.parse()?,
            renderer: dto.renderer.parse()?,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourcesDto {
    policy: BindingDto,
    controller_profile: BindingDto,
    attended_attestation: BindingDto,
    journal: JournalBindingDto,
    dataset: BindingDto,
    proposed_plant: BindingDto,
    proposal_evidence: BindingDto,
}

#[derive(Debug)]
struct Sources {
    policy: Binding,
    controller_profile: Binding,
    attended_attestation: Binding,
    journal: JournalBinding,
    dataset: Binding,
    proposed_plant: Binding,
    proposal_evidence: Binding,
}

impl Sources {
    fn parse(dto: SourcesDto) -> Result<Self, PlantPromotionError> {
        Ok(Self {
            policy: Binding::parse("policy", dto.policy)?,
            controller_profile: Binding::parse("controller_profile", dto.controller_profile)?,
            attended_attestation: Binding::parse("attended_attestation", dto.attended_attestation)?,
            journal: JournalBinding {
                artifact: Binding::parse("journal", dto.journal.artifact)?,
                records: NonZeroU64::new(dto.journal.records).ok_or(
                    PlantPromotionError::Rejected("journal record count is zero"),
                )?,
            },
            dataset: Binding::parse("dataset", dto.dataset)?,
            proposed_plant: Binding::parse("proposed_plant", dto.proposed_plant)?,
            proposal_evidence: Binding::parse("proposal_evidence", dto.proposal_evidence)?,
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingDto {
    path: PathBuf,
    sha256_hex: String,
    bytes: u64,
}

#[derive(Debug)]
struct Binding {
    path: PathBuf,
    sha256: [u8; 32],
    bytes: NonZeroU64,
}

impl Binding {
    fn parse(role: &'static str, dto: BindingDto) -> Result<Self, PlantPromotionError> {
        absolute_path(role, &dto.path)?;
        Ok(Self {
            path: dto.path,
            sha256: parse_sha(role, &dto.sha256_hex)?,
            bytes: NonZeroU64::new(dto.bytes)
                .ok_or(PlantPromotionError::Rejected("bound artifact is empty"))?,
        })
    }

    fn load(&self, maximum: usize, role: &'static str) -> Result<Vec<u8>, PlantPromotionError> {
        let bytes = read_file(&self.path, maximum, role)?;
        if u64::try_from(bytes.len()).unwrap_or(u64::MAX) != self.bytes.get()
            || sha256(&bytes) != self.sha256
        {
            return Err(PlantPromotionError::Rejected(
                "artifact byte count or SHA-256 does not match review binding",
            ));
        }
        Ok(bytes)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JournalBindingDto {
    artifact: BindingDto,
    records: u64,
}

#[derive(Debug)]
struct JournalBinding {
    artifact: Binding,
    records: NonZeroU64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum Accepted {
    ReviewedAndAccepted,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum StopSemantics {
    CoastVerified,
    BrakeVerified,
}

impl StopSemantics {
    const fn as_str(self) -> &'static str {
        match self {
            Self::CoastVerified => "coast_verified",
            Self::BrakeVerified => "brake_verified",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PhysicalReview {
    complete_journal: Accepted,
    dataset_and_reproduced_fit: Accepted,
    repeated_run_consistency: Accepted,
    wheel_wiring_and_signed_motion: Accepted,
    units_and_base_body_flu_frame: Accepted,
    surface_payload_and_envelope: Accepted,
    default_off_driver_enable: Accepted,
    driver_fault_and_estop_feedback: Accepted,
    reset_brownout_and_hard_fault: Accepted,
    independent_power_cut: Accepted,
    verified_physical_stop_semantics: StopSemantics,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Calibrations {
    imu_calibration_id: String,
    stereo_calibration_id: String,
    tracking_camera_to_base_calibration_id: String,
}

impl Calibrations {
    fn parse(mut self) -> Result<Self, PlantPromotionError> {
        self.imu_calibration_id = text("imu_calibration_id", self.imu_calibration_id)?;
        self.stereo_calibration_id = text("stereo_calibration_id", self.stereo_calibration_id)?;
        self.tracking_camera_to_base_calibration_id = text(
            "tracking_camera_to_base_calibration_id",
            self.tracking_camera_to_base_calibration_id,
        )?;
        Ok(self)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RendererInput {
    plant_artifact_id: String,
    plant_destination_relative_path: String,
}

impl RendererInput {
    fn parse(mut self) -> Result<Self, PlantPromotionError> {
        self.plant_artifact_id = id("plant_artifact_id", self.plant_artifact_id)?;
        let path = Path::new(&self.plant_destination_relative_path);
        if self.plant_destination_relative_path.is_empty()
            || self.plant_destination_relative_path.len() > MAX_RENDERER_RELATIVE_PATH_BYTES
            || self.plant_destination_relative_path.contains("${")
            || path
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
            || !self
                .plant_destination_relative_path
                .strip_prefix("artifacts/")
                .is_some_and(|suffix| !suffix.is_empty())
        {
            return Err(PlantPromotionError::Rejected(
                "renderer plant destination must be a clean relative path beneath artifacts/",
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetWire {
    schema_version: u32,
    evidence_content_sha256: String,
    policy_sha256: String,
    controller_profile_sha256: String,
    attended_physical_attestation_sha256: String,
    journal_sha256: String,
    body_frame_id: String,
    aligned_evidence: Box<RawValue>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetPayload {
    schema_version: u32,
    robot_id: String,
    controller_session_id: String,
    visual_velocity_source_id: String,
    imu_calibration_id: String,
    wheelbase_calibration_id: String,
    body_frame_id: String,
    samples: Vec<DatasetSample>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DatasetSample {
    observed_at_ns: u64,
    applied_command_sequence: u64,
    applied_left_pwm_percent: i8,
    applied_right_pwm_percent: i8,
    visual_body_forward_velocity_mps: f64,
    visual_body_lateral_velocity_mps: f64,
    calibrated_imu_yaw_rate_rad_s: f64,
    lateral_holdout: bool,
}

#[derive(Debug)]
struct DatasetEnvelope {
    wire: DatasetWire,
    payload: DatasetPayload,
}

impl DatasetEnvelope {
    fn parse(bytes: &[u8]) -> Result<Self, PlantPromotionError> {
        let wire: DatasetWire = parse_json(bytes, "dataset")?;
        let payload: DatasetPayload =
            parse_json(wire.aligned_evidence.get().as_bytes(), "dataset payload")?;
        if canonical_sha(sha256(wire.aligned_evidence.get().as_bytes()))
            != wire.evidence_content_sha256
        {
            return Err(PlantPromotionError::Rejected(
                "dataset payload digest does not match exact embedded JSON",
            ));
        }
        if wire.schema_version != NANO_BASE_COMMISSIONING_ARTIFACT_V1
            || payload.schema_version != NANO_BASE_COMMISSIONING_ARTIFACT_V1
            || wire.body_frame_id != BODY_FRAME_ID
            || payload.body_frame_id != BODY_FRAME_ID
        {
            return Err(PlantPromotionError::Rejected(
                "dataset schema or coordinate frame is unsupported",
            ));
        }
        Ok(Self { wire, payload })
    }

    fn verify_links(&self, digest: InputDigests) -> Result<(), PlantPromotionError> {
        for (actual, expected) in [
            (&self.wire.policy_sha256, digest.policy),
            (&self.wire.controller_profile_sha256, digest.profile),
            (
                &self.wire.attended_physical_attestation_sha256,
                digest.attestation,
            ),
            (&self.wire.journal_sha256, digest.journal),
        ] {
            if actual != &canonical_sha(expected) {
                return Err(PlantPromotionError::Rejected(
                    "dataset provenance digest mismatch",
                ));
            }
        }
        Ok(())
    }

    fn identification_dto(&self) -> Result<IdentificationDatasetV1Dto, PlantPromotionError> {
        let samples = self
            .payload
            .samples
            .iter()
            .map(|sample| IdentificationSampleV1Dto {
                observed_at_ns: sample.observed_at_ns,
                applied_command_sequence: sample.applied_command_sequence,
                applied_left_pwm_percent: sample.applied_left_pwm_percent,
                applied_right_pwm_percent: sample.applied_right_pwm_percent,
                visual_forward_velocity_mps: sample.visual_body_forward_velocity_mps,
                calibrated_imu_yaw_rate_rad_s: sample.calibrated_imu_yaw_rate_rad_s,
            })
            .collect();
        Ok(IdentificationDatasetV1Dto {
            schema_version: BASE_IDENTIFICATION_V1,
            dataset_content_id: self
                .wire
                .evidence_content_sha256
                .strip_prefix("sha256:")
                .ok_or(PlantPromotionError::Rejected(
                    "dataset content ID is not canonical SHA-256",
                ))?
                .to_owned(),
            robot_id: self.payload.robot_id.clone(),
            controller_session_id: self.payload.controller_session_id.clone(),
            visual_velocity_source_id: self.payload.visual_velocity_source_id.clone(),
            imu_calibration_id: self.payload.imu_calibration_id.clone(),
            wheelbase_calibration_id: self.payload.wheelbase_calibration_id.clone(),
            samples,
        })
    }

    fn reproduce_lateral(
        &self,
        policy: NanoBaseCommissioningPolicyV1,
    ) -> Result<Lateral, PlantPromotionError> {
        let mut training = (0_u32, 0.0_f64);
        let mut holdout = (0_u32, 0.0_f64);
        let stride = u32::from(policy.lateral_holdout_stride().get());
        for (index, sample) in self.payload.samples.iter().enumerate() {
            let ordinal = u32::try_from(index)
                .map_err(|_| PlantPromotionError::Rejected("too many dataset samples"))?;
            let expected_holdout = ordinal % stride == 0;
            if sample.lateral_holdout != expected_holdout
                || !sample.visual_body_lateral_velocity_mps.is_finite()
            {
                return Err(PlantPromotionError::Rejected(
                    "lateral holdout marker/value is invalid",
                ));
            }
            let target = if expected_holdout {
                &mut holdout
            } else {
                &mut training
            };
            target.0 += 1;
            target.1 = target.1.max(sample.visual_body_lateral_velocity_mps.abs());
        }
        let bound = training.1 + policy.lateral_bound_margin_mps();
        if training.0 < policy.lateral_minimum_training_samples().get()
            || holdout.0 < policy.lateral_minimum_holdout_samples().get()
            || !bound.is_finite()
            || bound > policy.lateral_maximum_accepted_bound_mps()
            || holdout.1 > bound
        {
            return Err(PlantPromotionError::Rejected(
                "lateral validity envelope cannot be reproduced",
            ));
        }
        Ok(Lateral {
            bound,
            training_count: training.0,
            holdout_count: holdout.0,
            training_max: training.1,
            holdout_max: holdout.1,
            margin: policy.lateral_bound_margin_mps(),
        })
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProposalEvidence {
    schema_version: u32,
    activation_status: String,
    policy_sha256: String,
    controller_profile_sha256: String,
    attended_physical_attestation_sha256: String,
    journal_sha256: String,
    plant_evidence_dataset_content_id: String,
    proposed_plant_artifact_sha256: String,
    controller_session_id: String,
    visual_velocity_source_id: String,
    imu_calibration_id: String,
    wheelbase_calibration_id: String,
    lateral_scope: ProposalLateral,
    remaining_gate: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProposalLateral {
    method_id: String,
    scope_label: String,
    frame_id: String,
    maximum_absolute_lateral_velocity_mps: f64,
    training_sample_count: u32,
    holdout_sample_count: u32,
    training_maximum_absolute_lateral_velocity_mps: f64,
    holdout_maximum_absolute_lateral_velocity_mps: f64,
    configured_margin_mps: f64,
}

impl ProposalEvidence {
    fn verify(
        &self,
        dataset: &DatasetEnvelope,
        policy: NanoBaseCommissioningPolicyV1,
        digest: InputDigests,
    ) -> Result<(), PlantPromotionError> {
        if self.schema_version != NANO_BASE_COMMISSIONING_ARTIFACT_V1
            || self.activation_status != PROPOSED
            || self.remaining_gate
                != "operator review, physical approval, manifest rebind, and normal production admission"
            || self.controller_session_id != dataset.payload.controller_session_id
            || self.visual_velocity_source_id != dataset.payload.visual_velocity_source_id
            || self.imu_calibration_id != dataset.payload.imu_calibration_id
            || self.wheelbase_calibration_id != dataset.payload.wheelbase_calibration_id
        {
            return Err(PlantPromotionError::Rejected(
                "proposal status or stream identities do not match the dataset",
            ));
        }
        for (actual, expected) in [
            (&self.policy_sha256, digest.policy),
            (&self.controller_profile_sha256, digest.profile),
            (
                &self.attended_physical_attestation_sha256,
                digest.attestation,
            ),
            (&self.journal_sha256, digest.journal),
            (&self.plant_evidence_dataset_content_id, digest.dataset),
            (&self.proposed_plant_artifact_sha256, digest.plant),
        ] {
            if actual != &canonical_sha(expected) {
                return Err(PlantPromotionError::Rejected(
                    "proposal provenance digest mismatch",
                ));
            }
        }
        let lateral = dataset.reproduce_lateral(policy)?;
        if self.lateral_scope.method_id != "visual-body-lateral-training-max-margin-holdout-v1"
            || self.lateral_scope.scope_label != policy.lateral_scope_label()
            || self.lateral_scope.frame_id != BODY_FRAME_ID
            || self.lateral_scope.maximum_absolute_lateral_velocity_mps != lateral.bound
            || self.lateral_scope.training_sample_count != lateral.training_count
            || self.lateral_scope.holdout_sample_count != lateral.holdout_count
            || self
                .lateral_scope
                .training_maximum_absolute_lateral_velocity_mps
                != lateral.training_max
            || self
                .lateral_scope
                .holdout_maximum_absolute_lateral_velocity_mps
                != lateral.holdout_max
            || self.lateral_scope.configured_margin_mps != lateral.margin
        {
            return Err(PlantPromotionError::Rejected(
                "proposal lateral evidence does not reproduce",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct InputDigests {
    policy: [u8; 32],
    profile: [u8; 32],
    attestation: [u8; 32],
    journal: [u8; 32],
    dataset: [u8; 32],
    plant: [u8; 32],
    proposal: [u8; 32],
}

#[derive(Clone, Copy)]
struct Lateral {
    bound: f64,
    training_count: u32,
    holdout_count: u32,
    training_max: f64,
    holdout_max: f64,
    margin: f64,
}

struct PhysicalPlantEvidence<'a> {
    dataset_content_id: &'a str,
    identification_method_id: &'a str,
    sample_count: NonZeroU64,
    residuals: FitResidualsV1Dto,
}

fn plant_from_fit(
    fit: IdentifiedPlantV1,
    lateral: Lateral,
    policy: NanoBaseCommissioningPolicyV1,
    dataset_sha: [u8; 32],
) -> PlantModelV1Dto {
    debug_assert_eq!(
        fit.support().lateral_velocity,
        LateralVelocityEvidence::Unidentified
    );
    let support = fit.support();
    let residuals = fit.holdout_residuals();
    PlantModelV1Dto {
        schema_version: PLANT_MODEL_V1,
        model_id: policy.model_id().to_owned(),
        model_version: policy.model_version().get(),
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
            max_abs_lateral_velocity_mps: lateral.bound,
        },
        evidence: PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
            dataset_content_id: canonical_sha(dataset_sha),
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

fn plant_equivalent(left: &PlantModelV1Dto, right: &PlantModelV1Dto) -> bool {
    let exact = left.schema_version == right.schema_version
        && left.model_id == right.model_id
        && left.model_version == right.model_version
        && left.validity.left_pwm_min_percent == right.validity.left_pwm_min_percent
        && left.validity.left_pwm_max_percent == right.validity.left_pwm_max_percent
        && left.validity.right_pwm_min_percent == right.validity.right_pwm_min_percent
        && left.validity.right_pwm_max_percent == right.validity.right_pwm_max_percent
        && match (&left.evidence, &right.evidence) {
            (
                PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
                    dataset_content_id: a,
                    identification_method_id: b,
                    sample_count: c,
                    ..
                },
                PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
                    dataset_content_id: x,
                    identification_method_id: y,
                    sample_count: z,
                    ..
                },
            ) => a == x && b == y && c == z,
            _ => false,
        };
    if !exact {
        return false;
    }
    let numeric = [
        (left.sample_period_s, right.sample_period_s),
        (left.wheelbase_m, right.wheelbase_m),
        (
            left.left.velocity_gain_mps_per_pwm_percent,
            right.left.velocity_gain_mps_per_pwm_percent,
        ),
        (left.left.time_constant_s, right.left.time_constant_s),
        (
            left.right.velocity_gain_mps_per_pwm_percent,
            right.right.velocity_gain_mps_per_pwm_percent,
        ),
        (left.right.time_constant_s, right.right.time_constant_s),
        (
            left.validity.left_velocity_min_mps,
            right.validity.left_velocity_min_mps,
        ),
        (
            left.validity.left_velocity_max_mps,
            right.validity.left_velocity_max_mps,
        ),
        (
            left.validity.right_velocity_min_mps,
            right.validity.right_velocity_min_mps,
        ),
        (
            left.validity.right_velocity_max_mps,
            right.validity.right_velocity_max_mps,
        ),
        (
            left.validity.max_abs_yaw_rate_rad_s,
            right.validity.max_abs_yaw_rate_rad_s,
        ),
        (
            left.validity.max_abs_lateral_velocity_mps,
            right.validity.max_abs_lateral_velocity_mps,
        ),
    ];
    let residuals_equal = if let (
        PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
            residuals: left_residuals,
            ..
        },
        PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
            residuals: right_residuals,
            ..
        },
    ) = (&left.evidence, &right.evidence)
    {
        [
            (
                left_residuals.left_velocity_rmse_mps,
                right_residuals.left_velocity_rmse_mps,
            ),
            (
                left_residuals.right_velocity_rmse_mps,
                right_residuals.right_velocity_rmse_mps,
            ),
            (
                left_residuals.yaw_rate_rmse_rad_s,
                right_residuals.yaw_rate_rmse_rad_s,
            ),
            (
                left_residuals.max_abs_velocity_error_mps,
                right_residuals.max_abs_velocity_error_mps,
            ),
        ]
        .into_iter()
        .all(|(left, right)| left.to_bits() == right.to_bits())
    } else {
        false
    };
    residuals_equal
        && numeric
            .into_iter()
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn publish(
    root: &Path,
    review_sha: [u8; 32],
    review: &Review,
    plant_bytes: &[u8],
    plant: PlantModelV1,
    physical: PhysicalPlantEvidence<'_>,
    digest: InputDigests,
) -> Result<PlantPromotionResult, PlantPromotionError> {
    absolute_path("output_root", root)?;
    let root =
        CommissioningArtifactDirectory::inspect(root).map_err(PlantPromotionError::OutputRoot)?;
    let directory_name = format!("plant-promotion-v1-{}", hex(review_sha));
    let directory = root.as_path().join(&directory_name);
    let (plant_name, production_plant) =
        describe_artifact(&directory, "production-plant-v1", plant_bytes);
    let evidence = json!({
        "schema_version": 1,
        "status": REVIEWED,
        "promotion_id": review.promotion_id,
        "reviewer_id": review.reviewer_id,
        "approval_id": review.approval_id,
        "approver_id": review.approver_id,
        "operator_claimed_commissioning_session_id": review.commissioning_session_id,
        "operator_claimed_journal_records": review.sources.journal.records.get(),
        "exact_inputs": {
            "policy_sha256_hex": hex(digest.policy),
            "controller_profile_sha256_hex": hex(digest.profile),
            "attended_attestation_sha256_hex": hex(digest.attestation),
            "journal_sha256_hex": hex(digest.journal),
            "dataset_sha256_hex": hex(digest.dataset),
            "proposed_plant_sha256_hex": hex(digest.plant),
            "proposal_evidence_sha256_hex": hex(digest.proposal)
        },
        "physical_review": review.physical_review,
        "production_plant": {
            "model_id": plant.model_id().as_str(),
            "model_version": plant.model_version().get(),
            "sha256_hex": hex(digest.plant)
        },
        "remaining_authority_gate":
            "render and qualify the immutable production bundle; live admission and attended fault qualification remain required"
    });
    let evidence_bytes = serde_json::to_vec(&evidence)
        .map_err(|source| PlantPromotionError::Json("promotion_evidence", source))?;
    let (evidence_name, promotion_evidence) =
        describe_artifact(&directory, "plant-promotion-evidence-v1", &evidence_bytes);
    let renderer = json!({
        "schema_version": 1,
        "status": REVIEWED,
        "reviewer_id": review.reviewer_id,
        "verified_physical_stop_semantics":
            review.physical_review.verified_physical_stop_semantics.as_str(),
        "assets": {
            "plant": {
                "artifact_id": review.renderer.plant_artifact_id,
                "source_path": production_plant.path,
                "destination_relative_path":
                    review.renderer.plant_destination_relative_path
            }
        },
        "production_actuation": {
            "plant_model_id": plant.model_id().as_str(),
            "plant_model_version": plant.model_version().get(),
            "operator_claimed_physical_approval": {
                "approval_id": review.approval_id,
                "approver_id": review.approver_id,
                "plant_dataset_content_id": physical.dataset_content_id,
                "plant_identification_method_id": physical.identification_method_id,
                "plant_sample_count": physical.sample_count.get(),
                "plant_fit_residuals": physical.residuals,
                "imu_calibration_id": review.calibrations.imu_calibration_id,
                "stereo_calibration_id": review.calibrations.stereo_calibration_id,
                "tracking_camera_to_base_calibration_id":
                    review.calibrations.tracking_camera_to_base_calibration_id
            }
        },
        "promotion_evidence": {
            "path": promotion_evidence.path,
            "sha256_hex": hex(promotion_evidence.sha256)
        }
    });
    let renderer_bytes = serde_json::to_vec(&renderer)
        .map_err(|source| PlantPromotionError::Json("renderer_values", source))?;
    let (renderer_name, renderer_values) =
        describe_artifact(&directory, "nano-agent-renderer-values-v1", &renderer_bytes);
    let marker = json!({
        "schema_version": 1,
        "status": "complete",
        "production_plant_sha256_hex": hex(production_plant.sha256),
        "promotion_evidence_sha256_hex": hex(promotion_evidence.sha256),
        "renderer_values_sha256_hex": hex(renderer_values.sha256)
    });
    let marker_bytes = serde_json::to_vec(&marker)
        .map_err(|source| PlantPromotionError::Json("completion_marker", source))?;
    let (marker_name, completion_marker) =
        describe_artifact(&directory, "plant-promotion-complete-v1", &marker_bytes);
    let files = [
        (plant_name.as_str(), plant_bytes),
        (evidence_name.as_str(), evidence_bytes.as_slice()),
        (renderer_name.as_str(), renderer_bytes.as_slice()),
        (marker_name.as_str(), marker_bytes.as_slice()),
    ];
    publish_directory_transactionally(
        &root,
        &directory_name,
        &directory,
        &files,
        &mut OsPromotionPublishOps,
    )?;
    Ok(PlantPromotionResult {
        directory,
        production_plant,
        promotion_evidence,
        renderer_values,
        completion_marker,
    })
}

fn describe_artifact(
    directory: &Path,
    stem: &str,
    bytes: &[u8],
) -> (String, PublishedPromotionArtifact) {
    let digest = sha256(bytes);
    let name = format!("{stem}-{}.json", hex(digest));
    (
        name.clone(),
        PublishedPromotionArtifact {
            path: directory.join(name),
            sha256: digest,
            bytes: u64::try_from(bytes.len()).expect("bounded artifact length fits u64"),
        },
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PublishSyncTarget {
    Artifact,
    StagingDirectory,
    OutputRoot,
}

trait PromotionPublishOps {
    fn sync(&mut self, target: PublishSyncTarget, file: &File) -> io::Result<()>;

    fn before_atomic_publish(&mut self, _root: &File, _destination_name: &str) -> io::Result<()> {
        Ok(())
    }

    fn rename_no_replace(
        &mut self,
        root: &File,
        staging_name: &str,
        destination_name: &str,
    ) -> Result<(), Errno>;
}

struct OsPromotionPublishOps;

impl PromotionPublishOps for OsPromotionPublishOps {
    fn sync(&mut self, _target: PublishSyncTarget, file: &File) -> io::Result<()> {
        fsync(file).map_err(errno_as_io)
    }

    fn rename_no_replace(
        &mut self,
        root: &File,
        staging_name: &str,
        destination_name: &str,
    ) -> Result<(), Errno> {
        renameat_with(
            root,
            staging_name,
            root,
            destination_name,
            RenameFlags::NOREPLACE,
        )
    }
}

struct StagingDirectory {
    name: String,
    path: PathBuf,
    directory: File,
    device: u64,
    inode: u64,
}

fn publish_directory_transactionally(
    root: &CommissioningArtifactDirectory,
    destination_name: &str,
    destination: &Path,
    files: &[(&str, &[u8])],
    ops: &mut impl PromotionPublishOps,
) -> Result<(), PlantPromotionError> {
    root.verify_binding()
        .map_err(PlantPromotionError::OutputRoot)?;
    let staging = create_staging_directory(root, destination_name, destination)?;
    let mut created = Vec::with_capacity(files.len());

    for &(name, bytes) in files {
        let mut file = match openat(
            &staging.directory,
            name,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        ) {
            Ok(file) => File::from(file),
            Err(source) => {
                return Err(failure_with_staging_cleanup(
                    root,
                    &staging,
                    &created,
                    publication_io("create staging artifact", &staging.path.join(name), source),
                ));
            }
        };
        created.push(name);
        if let Err(source) = file.write_all(bytes) {
            return Err(failure_with_staging_cleanup(
                root,
                &staging,
                &created,
                publication_std_io("write staging artifact", &staging.path.join(name), source),
            ));
        }
        if let Err(source) = ops.sync(PublishSyncTarget::Artifact, &file) {
            return Err(failure_with_staging_cleanup(
                root,
                &staging,
                &created,
                publication_std_io("sync staging artifact", &staging.path.join(name), source),
            ));
        }
        if let Err(failure) = require_staged_file(root, &staging, name, &file) {
            return Err(failure_with_staging_cleanup(
                root, &staging, &created, failure,
            ));
        }
    }
    if let Err(source) = ops.sync(PublishSyncTarget::StagingDirectory, &staging.directory) {
        return Err(failure_with_staging_cleanup(
            root,
            &staging,
            &created,
            publication_std_io("sync staging directory", &staging.path, source),
        ));
    }
    if let Err(failure) = require_staging_binding(root, &staging) {
        return Err(failure_with_staging_cleanup(
            root, &staging, &created, failure,
        ));
    }
    if let Err(source) = root.verify_binding() {
        return Err(failure_with_staging_cleanup(
            root,
            &staging,
            &created,
            PlantPromotionError::OutputRoot(source),
        ));
    }
    if let Err(source) = ops.before_atomic_publish(root.directory(), destination_name) {
        return Err(failure_with_staging_cleanup(
            root,
            &staging,
            &created,
            publication_std_io("run pre-publication hook", destination, source),
        ));
    }
    match ops.rename_no_replace(root.directory(), &staging.name, destination_name) {
        Ok(()) => {}
        Err(Errno::EXIST) => {
            return Err(failure_with_staging_cleanup(
                root,
                &staging,
                &created,
                PlantPromotionError::OutputAlreadyExists(destination.to_path_buf()),
            ));
        }
        Err(source) => {
            return Err(failure_with_staging_cleanup(
                root,
                &staging,
                &created,
                publication_io("atomically publish staging directory", destination, source),
            ));
        }
    }

    if let Err(source) = require_published_directory(root, destination_name, &staging) {
        return Err(PlantPromotionError::PublishedButIdentityUncertain {
            directory: destination.to_path_buf(),
            source: Box::new(source),
        });
    }
    if let Err(source) = ops.sync(PublishSyncTarget::OutputRoot, root.directory()) {
        return Err(PlantPromotionError::PublishedButDurabilityUncertain {
            directory: destination.to_path_buf(),
            source,
        });
    }
    if let Err(source) = root.verify_binding() {
        return Err(PlantPromotionError::PublishedButIdentityUncertain {
            directory: destination.to_path_buf(),
            source: Box::new(PlantPromotionError::OutputRoot(source)),
        });
    }
    require_published_directory(root, destination_name, &staging).map_err(|source| {
        PlantPromotionError::PublishedButIdentityUncertain {
            directory: destination.to_path_buf(),
            source: Box::new(source),
        }
    })
}

fn create_staging_directory(
    root: &CommissioningArtifactDirectory,
    destination_name: &str,
    destination: &Path,
) -> Result<StagingDirectory, PlantPromotionError> {
    for _ in 0..STAGING_NAME_ATTEMPTS {
        let sequence = STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let name = format!(
            ".{destination_name}.staging-{}-{sequence}",
            std::process::id()
        );
        match mkdirat(root.directory(), &name, Mode::from_raw_mode(0o700)) {
            Ok(()) => {
                let path = root.as_path().join(&name);
                let opened = openat(
                    root.directory(),
                    &name,
                    OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                    Mode::empty(),
                )
                .map(File::from)
                .map_err(|source| {
                    let primary = publication_io("open staging directory", &path, source);
                    cleanup_unopened_staging(root, &name, &path, primary)
                })?;
                let stat = fstat(&opened).map_err(|source| {
                    let primary = publication_io("inspect staging directory", &path, source);
                    cleanup_open_staging(root, &name, &path, &opened, primary)
                })?;
                let staging = StagingDirectory {
                    name,
                    path,
                    directory: opened,
                    device: stat_device(&stat),
                    inode: stat.st_ino,
                };
                if let Err(primary) = require_staging_binding(root, &staging) {
                    return Err(failure_with_staging_cleanup(root, &staging, &[], primary));
                }
                return Ok(staging);
            }
            Err(Errno::EXIST) => continue,
            Err(source) => {
                return Err(publication_io(
                    "create staging directory",
                    &root.as_path().join(name),
                    source,
                ));
            }
        }
    }
    Err(PlantPromotionError::StagingNameExhausted {
        directory: destination.to_path_buf(),
        attempts: STAGING_NAME_ATTEMPTS,
    })
}

fn require_staging_binding(
    root: &CommissioningArtifactDirectory,
    staging: &StagingDirectory,
) -> Result<(), PlantPromotionError> {
    let opened = fstat(&staging.directory).map_err(|source| {
        publication_io("inspect retained staging directory", &staging.path, source)
    })?;
    let named =
        statat(root.directory(), &staging.name, AtFlags::SYMLINK_NOFOLLOW).map_err(|source| {
            publication_io("inspect named staging directory", &staging.path, source)
        })?;
    let root_stat = fstat(root.directory())
        .map_err(|source| publication_io("inspect output root", root.as_path(), source))?;
    if FileType::from_raw_mode(opened.st_mode) != FileType::Directory
        || FileType::from_raw_mode(named.st_mode) != FileType::Directory
        || stat_device(&opened) != staging.device
        || opened.st_ino != staging.inode
        || stat_device(&named) != staging.device
        || named.st_ino != staging.inode
        || opened.st_dev != root_stat.st_dev
        || opened.st_uid != rustix::process::geteuid().as_raw()
        || u32::from(opened.st_mode) & 0o777 != 0o700
    {
        return Err(PlantPromotionError::UnsafePublicationObject {
            path: staging.path.clone(),
            expected: "retained, root-bound, current-user 0700 staging directory",
        });
    }
    Ok(())
}

fn require_staged_file(
    root: &CommissioningArtifactDirectory,
    staging: &StagingDirectory,
    name: &str,
    file: &File,
) -> Result<(), PlantPromotionError> {
    let opened = fstat(file).map_err(|source| {
        publication_io(
            "inspect retained staging artifact",
            &staging.path.join(name),
            source,
        )
    })?;
    let named = statat(&staging.directory, name, AtFlags::SYMLINK_NOFOLLOW).map_err(|source| {
        publication_io(
            "inspect named staging artifact",
            &staging.path.join(name),
            source,
        )
    })?;
    let root_stat = fstat(root.directory())
        .map_err(|source| publication_io("inspect output root", root.as_path(), source))?;
    if FileType::from_raw_mode(opened.st_mode) != FileType::RegularFile
        || opened.st_nlink != 1
        || opened.st_uid != rustix::process::geteuid().as_raw()
        || u32::from(opened.st_mode) & 0o777 != 0o600
        || opened.st_dev != root_stat.st_dev
        || opened.st_dev != named.st_dev
        || opened.st_ino != named.st_ino
    {
        return Err(PlantPromotionError::UnsafePublicationObject {
            path: staging.path.join(name),
            expected: "retained, staging-bound, current-user 0600 regular file",
        });
    }
    Ok(())
}

fn require_published_directory(
    root: &CommissioningArtifactDirectory,
    destination_name: &str,
    staging: &StagingDirectory,
) -> Result<(), PlantPromotionError> {
    let published = statat(
        root.directory(),
        destination_name,
        AtFlags::SYMLINK_NOFOLLOW,
    )
    .map_err(|source| {
        publication_io(
            "inspect published directory",
            &root.as_path().join(destination_name),
            source,
        )
    })?;
    if FileType::from_raw_mode(published.st_mode) != FileType::Directory
        || stat_device(&published) != staging.device
        || published.st_ino != staging.inode
    {
        return Err(PlantPromotionError::UnsafePublicationObject {
            path: root.as_path().join(destination_name),
            expected: "the exact synchronized staging-directory identity",
        });
    }
    Ok(())
}

fn failure_with_staging_cleanup(
    root: &CommissioningArtifactDirectory,
    staging: &StagingDirectory,
    created: &[&str],
    failure: PlantPromotionError,
) -> PlantPromotionError {
    match cleanup_staging(root, staging, created) {
        Ok(()) => failure,
        Err(cleanup) => PlantPromotionError::StagingCleanup {
            failure: Box::new(failure),
            staging_directory: staging.path.clone(),
            cleanup,
        },
    }
}

fn cleanup_staging(
    root: &CommissioningArtifactDirectory,
    staging: &StagingDirectory,
    created: &[&str],
) -> io::Result<()> {
    let mut first_failure = None;
    for &name in created.iter().rev() {
        match unlinkat(&staging.directory, name, AtFlags::empty()) {
            Ok(()) | Err(Errno::NOENT) => {}
            Err(source) => {
                first_failure.get_or_insert_with(|| errno_as_io(source));
            }
        };
    }
    match unlinkat(root.directory(), &staging.name, AtFlags::REMOVEDIR) {
        Ok(()) | Err(Errno::NOENT) => {}
        Err(source) => {
            first_failure.get_or_insert_with(|| errno_as_io(source));
        }
    }
    if let Err(source) = fsync(root.directory()) {
        first_failure.get_or_insert_with(|| errno_as_io(source));
    }
    first_failure.map_or(Ok(()), Err)
}

fn cleanup_unopened_staging(
    root: &CommissioningArtifactDirectory,
    name: &str,
    path: &Path,
    failure: PlantPromotionError,
) -> PlantPromotionError {
    let cleanup = match unlinkat(root.directory(), name, AtFlags::REMOVEDIR) {
        Ok(()) => fsync(root.directory()).map_err(errno_as_io),
        Err(Errno::NOENT) => Ok(()),
        Err(source) => Err(errno_as_io(source)),
    };
    match cleanup {
        Ok(()) => failure,
        Err(source) if source.kind() == io::ErrorKind::NotFound => failure,
        Err(cleanup) => PlantPromotionError::StagingCleanup {
            failure: Box::new(failure),
            staging_directory: path.to_path_buf(),
            cleanup,
        },
    }
}

fn cleanup_open_staging(
    root: &CommissioningArtifactDirectory,
    name: &str,
    path: &Path,
    _directory: &File,
    failure: PlantPromotionError,
) -> PlantPromotionError {
    cleanup_unopened_staging(root, name, path, failure)
}

fn publication_io(operation: &'static str, path: &Path, source: Errno) -> PlantPromotionError {
    publication_std_io(operation, path, errno_as_io(source))
}

fn publication_std_io(
    operation: &'static str,
    path: &Path,
    source: io::Error,
) -> PlantPromotionError {
    PlantPromotionError::PublicationIo {
        operation,
        path: path.to_path_buf(),
        source,
    }
}

fn errno_as_io(source: Errno) -> io::Error {
    io::Error::from_raw_os_error(source.raw_os_error())
}

fn stat_device(stat: &rustix::fs::Stat) -> u64 {
    u64::try_from(stat.st_dev).unwrap_or(u64::MAX)
}

fn read_file(
    path: &Path,
    maximum: usize,
    role: &'static str,
) -> Result<Vec<u8>, PlantPromotionError> {
    read_file_with_hook(path, maximum, role, || {})
}

fn read_file_with_hook(
    path: &Path,
    maximum: usize,
    role: &'static str,
    after_read: impl FnOnce(),
) -> Result<Vec<u8>, PlantPromotionError> {
    absolute_path(role, path)?;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    let before = StableFileMetadata::inspect(&file.metadata()?);
    let named_before = fs::symlink_metadata(path)?;
    if !before.regular
        || !named_before.file_type().is_file()
        || before.device != named_before.dev()
        || before.inode != named_before.ino()
    {
        return Err(PlantPromotionError::Rejected(
            "input must be a path-bound regular file",
        ));
    }
    let read_limit = maximum
        .checked_add(1)
        .ok_or(PlantPromotionError::Rejected("input bound overflow"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(
            usize::try_from(before.bytes)
                .unwrap_or(read_limit)
                .min(read_limit),
        )
        .map_err(|_| PlantPromotionError::Rejected("input allocation failed"))?;
    (&mut file)
        .take(u64::try_from(read_limit).expect("input hard bound fits u64"))
        .read_to_end(&mut bytes)?;
    after_read();
    let after = StableFileMetadata::inspect(&file.metadata()?);
    let named_after = fs::symlink_metadata(path)?;
    if bytes.len() > maximum {
        return Err(PlantPromotionError::Rejected(
            "input exceeds its byte bound",
        ));
    }
    if before != after
        || !named_after.file_type().is_file()
        || after.device != named_after.dev()
        || after.inode != named_after.ino()
        || u64::try_from(bytes.len()).expect("bounded read length fits u64") != after.bytes
    {
        return Err(PlantPromotionError::Rejected(
            "input identity or metadata changed while it was read",
        ));
    }
    Ok(bytes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StableFileMetadata {
    device: u64,
    inode: u64,
    bytes: u64,
    links: u64,
    mode: u32,
    uid: u32,
    gid: u32,
    modified_s: i64,
    modified_ns: i64,
    changed_s: i64,
    changed_ns: i64,
    regular: bool,
}

impl StableFileMetadata {
    fn inspect(metadata: &fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            bytes: metadata.len(),
            links: metadata.nlink(),
            mode: metadata.mode(),
            uid: metadata.uid(),
            gid: metadata.gid(),
            modified_s: metadata.mtime(),
            modified_ns: metadata.mtime_nsec(),
            changed_s: metadata.ctime(),
            changed_ns: metadata.ctime_nsec(),
            regular: metadata.file_type().is_file(),
        }
    }
}

fn parse_json<T: for<'de> Deserialize<'de>>(
    bytes: &[u8],
    role: &'static str,
) -> Result<T, PlantPromotionError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = T::deserialize(&mut deserializer)
        .map_err(|source| PlantPromotionError::Json(role, source))?;
    deserializer
        .end()
        .map_err(|source| PlantPromotionError::Json(role, source))?;
    Ok(value)
}

fn absolute_path(role: &'static str, path: &Path) -> Result<(), PlantPromotionError> {
    if !path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::RootDir | Component::Normal(_)))
    {
        return Err(PlantPromotionError::Invalid(role));
    }
    Ok(())
}

fn id(role: &'static str, value: String) -> Result<String, PlantPromotionError> {
    if value.is_empty()
        || value.len() > 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(PlantPromotionError::Invalid(role));
    }
    Ok(value)
}

fn text(role: &'static str, value: String) -> Result<String, PlantPromotionError> {
    if value.is_empty()
        || value.len() > 256
        || value.contains("${")
        || value.chars().any(char::is_control)
    {
        return Err(PlantPromotionError::Invalid(role));
    }
    Ok(value)
}

fn parse_sha(role: &'static str, value: &str) -> Result<[u8; 32], PlantPromotionError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(PlantPromotionError::Invalid(role));
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let nibble = |byte| match byte {
            b'0'..=b'9' => byte - b'0',
            b'a'..=b'f' => byte - b'a' + 10,
            _ => unreachable!("hex was checked"),
        };
        digest[index] = (nibble(pair[0]) << 4) | nibble(pair[1]);
    }
    Ok(digest)
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn canonical_sha(digest: [u8; 32]) -> String {
    format!("sha256:{}", hex(digest))
}

fn hex(digest: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in digest {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[derive(Debug)]
pub enum PlantPromotionError {
    Io(io::Error),
    Json(&'static str, serde_json::Error),
    Domain(&'static str, Box<dyn std::error::Error + Send + Sync>),
    Invalid(&'static str),
    Rejected(&'static str),
    OutputRoot(CommissioningArtifactDirectoryError),
    OutputAlreadyExists(PathBuf),
    StagingNameExhausted {
        directory: PathBuf,
        attempts: u64,
    },
    PublicationIo {
        operation: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    UnsafePublicationObject {
        path: PathBuf,
        expected: &'static str,
    },
    StagingCleanup {
        failure: Box<PlantPromotionError>,
        staging_directory: PathBuf,
        cleanup: io::Error,
    },
    PublishedButDurabilityUncertain {
        directory: PathBuf,
        source: io::Error,
    },
    PublishedButIdentityUncertain {
        directory: PathBuf,
        source: Box<PlantPromotionError>,
    },
}

impl fmt::Display for PlantPromotionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(source) => write!(formatter, "plant-promotion I/O failed: {source}"),
            Self::Json(role, source) => write!(formatter, "invalid {role} JSON: {source}"),
            Self::Domain(role, source) => write!(formatter, "invalid {role}: {source}"),
            Self::Invalid(role) => write!(formatter, "invalid plant-promotion field: {role}"),
            Self::Rejected(reason) => write!(formatter, "plant promotion rejected: {reason}"),
            Self::OutputRoot(source) => {
                write!(formatter, "invalid promotion output root: {source}")
            }
            Self::OutputAlreadyExists(path) => {
                write!(
                    formatter,
                    "promotion output already exists: {}",
                    path.display()
                )
            }
            Self::StagingNameExhausted {
                directory,
                attempts,
            } => write!(
                formatter,
                "could not reserve a unique staging directory for {} after {attempts} attempts",
                directory.display()
            ),
            Self::PublicationIo {
                operation,
                path,
                source,
            } => write!(
                formatter,
                "failed to {operation} at {}: {source}",
                path.display()
            ),
            Self::UnsafePublicationObject { path, expected } => write!(
                formatter,
                "unsafe promotion publication object at {}; expected {expected}",
                path.display()
            ),
            Self::StagingCleanup {
                failure,
                staging_directory,
                cleanup,
            } => write!(
                formatter,
                "{failure}; additionally failed to clean staging directory {}: {cleanup}",
                staging_directory.display()
            ),
            Self::PublishedButDurabilityUncertain { directory, source } => write!(
                formatter,
                "complete promotion output is visible at {}, but its directory-entry durability is uncertain: {source}",
                directory.display()
            ),
            Self::PublishedButIdentityUncertain { directory, source } => write!(
                formatter,
                "promotion output may be visible at {}, but its post-rename identity could not be confirmed: {source}",
                directory.display()
            ),
        }
    }
}

impl std::error::Error for PlantPromotionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(source) => Some(source),
            Self::Json(_, source) => Some(source),
            Self::Domain(_, source) => Some(source.as_ref()),
            Self::OutputRoot(source) => Some(source),
            Self::PublicationIo { source, .. }
            | Self::PublishedButDurabilityUncertain { source, .. } => Some(source),
            Self::StagingCleanup { failure, .. }
            | Self::PublishedButIdentityUncertain {
                source: failure, ..
            } => Some(failure.as_ref()),
            Self::Invalid(_)
            | Self::Rejected(_)
            | Self::OutputAlreadyExists(_)
            | Self::StagingNameExhausted { .. }
            | Self::UnsafePublicationObject { .. } => None,
        }
    }
}

impl From<io::Error> for PlantPromotionError {
    fn from(source: io::Error) -> Self {
        Self::Io(source)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static TEST_DIRECTORY_SEQUENCE: AtomicU64 = AtomicU64::new(1);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(label: &str) -> Self {
            for _ in 0..32 {
                let sequence = TEST_DIRECTORY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
                let path = std::env::temp_dir().join(format!(
                    "kiko-plant-promotion-{label}-{}-{sequence}",
                    std::process::id()
                ));
                match fs::create_dir(&path) {
                    Ok(()) => {
                        fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
                            .expect("private test directory");
                        return Self(path);
                    }
                    Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {}
                    Err(source) => panic!("create test directory: {source}"),
                }
            }
            panic!("test directory name exhaustion");
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[derive(Default)]
    struct InjectedPublishOps {
        failed_sync: Option<PublishSyncTarget>,
        create_destination_before_rename: bool,
    }

    impl PromotionPublishOps for InjectedPublishOps {
        fn sync(&mut self, target: PublishSyncTarget, file: &File) -> io::Result<()> {
            if self.failed_sync == Some(target) {
                return Err(io::Error::other("injected sync failure"));
            }
            fsync(file).map_err(errno_as_io)
        }

        fn before_atomic_publish(&mut self, root: &File, destination_name: &str) -> io::Result<()> {
            if self.create_destination_before_rename {
                mkdirat(root, destination_name, Mode::from_raw_mode(0o700)).map_err(errno_as_io)?;
            }
            Ok(())
        }

        fn rename_no_replace(
            &mut self,
            root: &File,
            staging_name: &str,
            destination_name: &str,
        ) -> Result<(), Errno> {
            renameat_with(
                root,
                staging_name,
                root,
                destination_name,
                RenameFlags::NOREPLACE,
            )
        }
    }

    fn test_publication(
        root_path: &Path,
        ops: &mut impl PromotionPublishOps,
    ) -> Result<(), PlantPromotionError> {
        let root = CommissioningArtifactDirectory::inspect(root_path).expect("private root");
        publish_directory_transactionally(
            &root,
            "published",
            &root_path.join("published"),
            &[("one.json", b"one"), ("two.json", b"two")],
            ops,
        )
    }

    fn entries(path: &Path) -> Vec<String> {
        let mut entries = fs::read_dir(path)
            .expect("read directory")
            .map(|entry| {
                entry
                    .expect("directory entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();
        entries.sort_unstable();
        entries
    }

    #[test]
    fn bounded_read_rejects_oversize_and_path_replacement() {
        let root = TestDirectory::new("bounded-read");
        let oversized = root.path().join("oversized");
        fs::write(&oversized, [7_u8; 32]).expect("oversized input");
        assert!(matches!(
            read_file(&oversized, 8, "test"),
            Err(PlantPromotionError::Rejected(
                "input exceeds its byte bound"
            ))
        ));

        let input = root.path().join("input");
        let replacement = root.path().join("replacement");
        let displaced = root.path().join("displaced");
        fs::write(&input, b"original").expect("input");
        fs::write(&replacement, b"replaced").expect("replacement");
        let result = read_file_with_hook(&input, 8, "test", || {
            fs::rename(&input, &displaced).expect("displace opened input");
            fs::rename(&replacement, &input).expect("replace named input");
        });
        assert!(matches!(
            result,
            Err(PlantPromotionError::Rejected(
                "input identity or metadata changed while it was read"
            ))
        ));
    }

    #[test]
    fn renderer_destination_matches_renderer_artifact_contract() {
        assert!(
            RendererInput {
                plant_artifact_id: "plant-1".to_owned(),
                plant_destination_relative_path: "artifacts/plant/plant-1.json".to_owned(),
            }
            .parse()
            .is_ok()
        );
        for destination in [
            "plant/plant-1.json".to_owned(),
            format!("artifacts/{}", "x".repeat(MAX_RENDERER_RELATIVE_PATH_BYTES)),
        ] {
            assert!(matches!(
                (RendererInput {
                    plant_artifact_id: "plant-1".to_owned(),
                    plant_destination_relative_path: destination,
                })
                .parse(),
                Err(PlantPromotionError::Rejected(_))
            ));
        }
    }

    #[test]
    fn plant_refit_comparison_rejects_a_single_bit_difference() {
        let reviewed = PlantModelV1Dto {
            schema_version: PLANT_MODEL_V1,
            model_id: "test-model".to_owned(),
            model_version: 1,
            sample_period_s: 0.05,
            wheelbase_m: 0.3,
            left: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.2,
            },
            right: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.2,
            },
            validity: PlantValidityEnvelopeV1Dto {
                left_pwm_min_percent: -20,
                left_pwm_max_percent: 20,
                right_pwm_min_percent: -20,
                right_pwm_max_percent: 20,
                left_velocity_min_mps: -0.2,
                left_velocity_max_mps: 0.2,
                right_velocity_min_mps: -0.2,
                right_velocity_max_mps: 0.2,
                max_abs_yaw_rate_rad_s: 1.0,
                max_abs_lateral_velocity_mps: 0.02,
            },
            evidence: PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
                dataset_content_id: format!("sha256:{}", "0".repeat(64)),
                identification_method_id: "test-method".to_owned(),
                sample_count: 4,
                residuals: FitResidualsV1Dto {
                    left_velocity_rmse_mps: 0.01,
                    right_velocity_rmse_mps: 0.01,
                    yaw_rate_rmse_rad_s: 0.01,
                    max_abs_velocity_error_mps: 0.02,
                },
            },
        };
        let mut reproduced = reviewed.clone();
        reproduced.validity.max_abs_yaw_rate_rad_s =
            f64::from_bits(reviewed.validity.max_abs_yaw_rate_rad_s.to_bits() + 1);
        assert!(!plant_equivalent(&reviewed, &reproduced));
        reproduced.validity.max_abs_yaw_rate_rad_s = reviewed.validity.max_abs_yaw_rate_rad_s;
        assert!(plant_equivalent(&reviewed, &reproduced));
    }

    #[test]
    fn failed_artifact_sync_leaves_no_partial_output_or_staging_directory() {
        let root = TestDirectory::new("sync-failure");
        let error = test_publication(
            root.path(),
            &mut InjectedPublishOps {
                failed_sync: Some(PublishSyncTarget::Artifact),
                ..InjectedPublishOps::default()
            },
        )
        .expect_err("artifact sync must fail");
        assert!(matches!(
            error,
            PlantPromotionError::PublicationIo {
                operation: "sync staging artifact",
                ..
            }
        ));
        assert!(entries(root.path()).is_empty());
    }

    #[test]
    fn no_replace_race_preserves_competing_destination_and_cleans_staging() {
        let root = TestDirectory::new("rename-race");
        let error = test_publication(
            root.path(),
            &mut InjectedPublishOps {
                create_destination_before_rename: true,
                ..InjectedPublishOps::default()
            },
        )
        .expect_err("competing destination must win");
        assert!(matches!(error, PlantPromotionError::OutputAlreadyExists(_)));
        assert_eq!(entries(root.path()), ["published"]);
        assert!(root.path().join("published").is_dir());
    }

    #[test]
    fn post_rename_root_sync_failure_reports_visible_uncertain_output() {
        let root = TestDirectory::new("root-sync-failure");
        let error = test_publication(
            root.path(),
            &mut InjectedPublishOps {
                failed_sync: Some(PublishSyncTarget::OutputRoot),
                ..InjectedPublishOps::default()
            },
        )
        .expect_err("root sync must fail");
        assert!(matches!(
            error,
            PlantPromotionError::PublishedButDurabilityUncertain { .. }
        ));
        assert_eq!(entries(root.path()), ["published"]);
        assert_eq!(
            entries(&root.path().join("published")),
            ["one.json", "two.json"]
        );
    }
}
