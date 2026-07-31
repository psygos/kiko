//! Evidence-binding boundary between head-gaze declarations and commands.
//!
//! A parsed [`HeadGazePolicyV1`] deliberately cannot create a command pose.
//! This module is the one activation boundary: it requires a claimed physical
//! review, binds that claim to the exact bytes of an already loaded deployment
//! asset, requires the mapping's natural declaration to equal the
//! manifest-bound reviewed return target, and reuses the exact parsed head
//! transport timeout. Only the resulting owner can turn its own mapped face
//! proposal into an [`ExactHeadTargetPose`].

use std::fmt;

use kiko_device_inventory::{
    ArtifactRelativePath, DeploymentAssetContentSha256, LoadedDeploymentAsset,
};
use kiko_expression_core::MonotonicTimestamp;
use kiko_expression_runtime::FaceTrackingUpdate;
use kiko_head_protocol::ExactHeadTargetPose;
use kiko_head_runtime::{
    HeadGazeActuationConfig, HeadGazeActuationConfigError, ReturnToTargetConfig,
    gaze_control::{HeadGazeControlConfig, HeadGazeControlConfigError},
};

use super::{
    HeadGazeFaceProposal, HeadGazeFaceProposalAdapter, HeadGazeFaceProposalAdapterError,
    HeadGazeFaceProposalError, HeadGazeFaceProposalOutcome, HeadGazeFaceProposalWithheld,
    HeadGazePolicyLifecycleClaim, HeadGazePolicyV1, OperatorClaimedHeadGazePhysicalReview,
    RgbFacePinholeProjection,
};

/// Exact retained identity of the physical-review evidence used at admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundHeadGazeReviewEvidence {
    relative_path: ArtifactRelativePath,
    byte_len: usize,
    content_sha256: DeploymentAssetContentSha256,
}

impl BoundHeadGazeReviewEvidence {
    pub const fn relative_path(&self) -> &ArtifactRelativePath {
        &self.relative_path
    }

    pub const fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub const fn content_sha256(&self) -> DeploymentAssetContentSha256 {
        self.content_sha256
    }
}

/// One face proposal mapped by the exact evidence-bound policy.
///
/// The command target has no public constructor and is produced only together
/// with the proposal provenance from the adapter retained by the same
/// admission.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AdmittedPhysicalHeadGazeFaceProposal {
    proposal: HeadGazeFaceProposal,
    command_target: ExactHeadTargetPose,
}

impl AdmittedPhysicalHeadGazeFaceProposal {
    pub const fn proposal(self) -> HeadGazeFaceProposal {
        self.proposal
    }

    pub const fn command_target(self) -> ExactHeadTargetPose {
        self.command_target
    }
}

/// Physical-policy evaluation never reuses a stale or withheld target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PhysicalHeadGazeFaceOutcome {
    Proposed(AdmittedPhysicalHeadGazeFaceProposal),
    Withheld(HeadGazeFaceProposalWithheld),
}

/// Sole owner of a head-gaze policy that crossed every static activation
/// boundary available before hardware is opened.
///
/// This is still not a base-zero lease and does not open a head transport.
/// Each physical service transaction must additionally consume the lease
/// minted by the live STM32 owner's interlock.
#[derive(Debug)]
pub struct EvidenceBoundPhysicalHeadGazePolicy {
    adapter: HeadGazeFaceProposalAdapter,
    actuation: HeadGazeActuationConfig,
    review: OperatorClaimedHeadGazePhysicalReview,
    evidence: BoundHeadGazeReviewEvidence,
}

impl EvidenceBoundPhysicalHeadGazePolicy {
    pub fn admit(
        policy: HeadGazePolicyV1,
        review_evidence: &LoadedDeploymentAsset,
        reviewed_return: &ReturnToTargetConfig,
    ) -> Result<Self, EvidenceBoundPhysicalHeadGazePolicyError> {
        let review = match policy.lifecycle() {
            HeadGazePolicyLifecycleClaim::OperatorClaimedPhysicalReview(review) => review.clone(),
            HeadGazePolicyLifecycleClaim::ProposalOnly(_) => {
                return Err(EvidenceBoundPhysicalHeadGazePolicyError::NotPhysicallyReviewed);
            }
        };

        let claimed_evidence_sha256 = *review.evidence_content_sha256().as_bytes();
        let loaded_evidence_sha256 = *review_evidence.content_sha256().as_bytes();
        if claimed_evidence_sha256 != loaded_evidence_sha256 {
            return Err(
                EvidenceBoundPhysicalHeadGazePolicyError::ReviewEvidenceDigestMismatch {
                    claimed: claimed_evidence_sha256,
                    loaded: loaded_evidence_sha256,
                },
            );
        }

        let declared_natural = policy.mapping().natural_declaration().positions();
        let reviewed_natural = reviewed_return.target();
        if declared_natural != reviewed_natural.positions() {
            return Err(
                EvidenceBoundPhysicalHeadGazePolicyError::NaturalPoseDoesNotMatchReviewedReturn {
                    declared: declared_natural.map(|position| position.get()),
                    reviewed: reviewed_natural.positions().map(|position| position.get()),
                },
            );
        }

        let declaration = *policy.controller();
        let controller = HeadGazeControlConfig::try_new(
            declaration.timing(),
            reviewed_natural,
            declaration.motion_limits(),
            declaration.error_band(),
        )
        .map_err(EvidenceBoundPhysicalHeadGazePolicyError::Controller)?;
        let actuation = HeadGazeActuationConfig::try_new_with_transaction_timeout(
            controller,
            reviewed_natural,
            reviewed_return.runtime().write_timeout(),
        )
        .map_err(EvidenceBoundPhysicalHeadGazePolicyError::Actuation)?;
        let adapter = HeadGazeFaceProposalAdapter::try_new(policy)
            .map_err(EvidenceBoundPhysicalHeadGazePolicyError::Adapter)?;

        Ok(Self {
            adapter,
            actuation,
            review,
            evidence: BoundHeadGazeReviewEvidence {
                relative_path: review_evidence.relative_path().clone(),
                byte_len: review_evidence.byte_len(),
                content_sha256: review_evidence.content_sha256(),
            },
        })
    }

    pub const fn actuation_config(&self) -> HeadGazeActuationConfig {
        self.actuation
    }

    pub const fn review(&self) -> &OperatorClaimedHeadGazePhysicalReview {
        &self.review
    }

    pub const fn evidence(&self) -> &BoundHeadGazeReviewEvidence {
        &self.evidence
    }

    /// Evaluate and activate one face update under this exact retained policy.
    ///
    /// The conversion to a command target is private to this owner, so a
    /// proposal from another mapping cannot be injected at this boundary.
    pub fn evaluate(
        &self,
        update: FaceTrackingUpdate,
        evaluated_at: MonotonicTimestamp,
        projection: RgbFacePinholeProjection,
    ) -> Result<PhysicalHeadGazeFaceOutcome, HeadGazeFaceProposalError> {
        match self.adapter.evaluate(update, evaluated_at, projection)? {
            HeadGazeFaceProposalOutcome::Proposed(proposal) => {
                let [bow, curl, yaw, roll] = proposal.target().positions();
                Ok(PhysicalHeadGazeFaceOutcome::Proposed(
                    AdmittedPhysicalHeadGazeFaceProposal {
                        proposal,
                        command_target: ExactHeadTargetPose::from_positions(bow, curl, yaw, roll),
                    },
                ))
            }
            HeadGazeFaceProposalOutcome::Withheld(reason) => {
                Ok(PhysicalHeadGazeFaceOutcome::Withheld(reason))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvidenceBoundPhysicalHeadGazePolicyError {
    NotPhysicallyReviewed,
    ReviewEvidenceDigestMismatch {
        claimed: [u8; 32],
        loaded: [u8; 32],
    },
    NaturalPoseDoesNotMatchReviewedReturn {
        declared: [u16; 4],
        reviewed: [u16; 4],
    },
    Controller(HeadGazeControlConfigError),
    Actuation(HeadGazeActuationConfigError),
    Adapter(HeadGazeFaceProposalAdapterError),
}

impl fmt::Display for EvidenceBoundPhysicalHeadGazePolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "head-gaze physical policy was not admitted: {self:?}"
        )
    }
}

impl std::error::Error for EvidenceBoundPhysicalHeadGazePolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Controller(source) => Some(source),
            Self::Actuation(source) => Some(source),
            Self::Adapter(source) => Some(source),
            Self::NotPhysicallyReviewed
            | Self::ReviewEvidenceDigestMismatch { .. }
            | Self::NaturalPoseDoesNotMatchReviewedReturn { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use kiko_device_inventory::{
        ArtifactRelativePath, DeploymentAssetByteLimit, load_deployment_asset,
    };
    use kiko_head_runtime::{HeadProbeConfig, HeadProbeConfigInput, ReturnToTargetConfigInput};
    use serde_json::{Value, json};

    use super::*;

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(1);
    const NATURAL_TICKS: [u16; 4] = [2_174, 2_570, 1_637, 3_047];

    struct TestDirectory(std::path::PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let nonce = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("test clock after Unix epoch")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "kiko-head-gaze-admission-{}-{timestamp}-{nonce}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create unique test directory");
            Self(fs::canonicalize(path).expect("canonical test directory"))
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn load_evidence(bytes: &[u8]) -> (TestDirectory, LoadedDeploymentAsset) {
        let root = TestDirectory::new();
        fs::write(root.0.join("review.txt"), bytes).expect("write review evidence");
        let asset = load_deployment_asset(
            &root.0,
            ArtifactRelativePath::parse("review.txt".to_owned()).expect("relative asset path"),
            DeploymentAssetByteLimit::try_new(
                u64::try_from(bytes.len()).expect("test evidence length fits u64"),
            )
            .expect("nonempty bounded evidence"),
        )
        .expect("load exact review evidence");
        (root, asset)
    }

    fn lowercase_hex(bytes: &[u8; 32]) -> String {
        let mut result = String::with_capacity(64);
        for byte in bytes {
            use std::fmt::Write as _;
            write!(&mut result, "{byte:02x}").expect("writing to String cannot fail");
        }
        result
    }

    fn reviewed_return() -> ReturnToTargetConfig {
        let probe = HeadProbeConfig::parse(HeadProbeConfigInput {
            device_path: "/dev/serial/by-id/usb-Kiko_STS_adapter_0001".to_owned(),
            response_timeout_ms: 100,
            request_timeout_ms: 100,
            noise_budget_bytes: 32,
        })
        .expect("valid probe policy");
        ReturnToTargetConfig::parse(
            &probe,
            ReturnToTargetConfigInput {
                write_timeout_ms: 100,
                arming_freshness_ms: 250,
                write_attempts: 2,
                redundant_read_tolerance_ticks: 10,
                readback_tolerance_ticks: 20,
                final_target_tolerance_ticks: 20,
                path_corridor_tolerance_ticks: 20,
                direction_regression_tolerance_ticks: 20,
                goal_speed_ticks_per_second: 50,
                torque_limit_permille: [600, 400, 400, 400],
                minimum_start_ticks: [2_133, 2_550, 1_617, 3_023],
                maximum_start_ticks: [2_194, 2_660, 1_852, 3_067],
                target_ticks: NATURAL_TICKS,
                maximum_travel_ticks: [48, 96, 224, 32],
            },
        )
        .expect("valid reviewed return")
    }

    fn policy_value(evidence_sha256: &str, control_period_ns: u64) -> Value {
        json!({
            "schema_version": 1,
            "lifecycle": {
                "kind": "operator_claimed_physical_review",
                "review_id": "review-claim-2026-07-31",
                "operator_id": "operator:ttrb",
                "evidence_id": "physical-head-calibration-session-01",
                "evidence_content_sha256_hex": evidence_sha256
            },
            "mapping_declaration": {
                "assembly_id": "kiko-head-assembly-01",
                "calibration_provenance_id": "physical-head-calibration-session-01",
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
                    "bow_ticks": NATURAL_TICKS[0],
                    "curl_ticks": NATURAL_TICKS[1],
                    "yaw_ticks": NATURAL_TICKS[2],
                    "roll_ticks": NATURAL_TICKS[3]
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
                    "control_period_ns": control_period_ns,
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

    fn parse_policy(value: &Value) -> HeadGazePolicyV1 {
        HeadGazePolicyV1::parse_json(&serde_json::to_vec(value).expect("serialize policy"))
            .expect("parse policy")
    }

    #[test]
    fn exact_review_bytes_natural_pose_and_transport_budget_activate_once() {
        let (_root, evidence) = load_evidence(b"retained physical review evidence");
        let digest = lowercase_hex(evidence.content_sha256().as_bytes());
        let policy = parse_policy(&policy_value(&digest, 100_000_000));
        let reviewed_return = reviewed_return();

        let admitted =
            EvidenceBoundPhysicalHeadGazePolicy::admit(policy, &evidence, &reviewed_return)
                .expect("evidence-bound policy");

        assert_eq!(
            admitted.actuation_config().controller().natural_pose(),
            reviewed_return.target()
        );
        assert_eq!(
            admitted
                .actuation_config()
                .goal_register_transaction_timeout(),
            reviewed_return.runtime().write_timeout()
        );
        assert_eq!(
            admitted.evidence().content_sha256(),
            evidence.content_sha256()
        );
        assert_eq!(
            admitted.evidence().relative_path(),
            evidence.relative_path()
        );
    }

    #[test]
    fn proposal_only_claim_cannot_cross_physical_boundary() {
        let (_root, evidence) = load_evidence(b"retained physical review evidence");
        let digest = lowercase_hex(evidence.content_sha256().as_bytes());
        let mut value = policy_value(&digest, 100_000_000);
        value["lifecycle"] = json!({
            "kind": "proposal_only",
            "proposal_id": "head-gaze-proposal-01",
            "evidence_id": "physical-head-calibration-session-01",
            "evidence_content_sha256_hex": digest
        });

        assert!(matches!(
            EvidenceBoundPhysicalHeadGazePolicy::admit(
                parse_policy(&value),
                &evidence,
                &reviewed_return(),
            ),
            Err(EvidenceBoundPhysicalHeadGazePolicyError::NotPhysicallyReviewed)
        ));
    }

    #[test]
    fn exact_review_digest_and_natural_pose_are_required() {
        let (_root, evidence) = load_evidence(b"retained physical review evidence");
        let wrong_digest = "11".repeat(32);
        assert!(matches!(
            EvidenceBoundPhysicalHeadGazePolicy::admit(
                parse_policy(&policy_value(&wrong_digest, 100_000_000)),
                &evidence,
                &reviewed_return(),
            ),
            Err(EvidenceBoundPhysicalHeadGazePolicyError::ReviewEvidenceDigestMismatch { .. })
        ));

        let digest = lowercase_hex(evidence.content_sha256().as_bytes());
        let mut wrong_natural = policy_value(&digest, 100_000_000);
        wrong_natural["mapping_declaration"]["natural_encoder_position_ticks"]["yaw_ticks"] =
            json!(1_638);
        assert!(matches!(
            EvidenceBoundPhysicalHeadGazePolicy::admit(
                parse_policy(&wrong_natural),
                &evidence,
                &reviewed_return(),
            ),
            Err(
                EvidenceBoundPhysicalHeadGazePolicyError::NaturalPoseDoesNotMatchReviewedReturn { .. }
            )
        ));
    }

    #[test]
    fn controller_period_shorter_than_exact_transport_timeout_is_rejected() {
        let (_root, evidence) = load_evidence(b"retained physical review evidence");
        let digest = lowercase_hex(evidence.content_sha256().as_bytes());

        assert!(matches!(
            EvidenceBoundPhysicalHeadGazePolicy::admit(
                parse_policy(&policy_value(&digest, 20_000_000)),
                &evidence,
                &reviewed_return(),
            ),
            Err(EvidenceBoundPhysicalHeadGazePolicyError::Actuation(
                HeadGazeActuationConfigError::TransactionTimeoutExceedsControlPeriod {
                    transaction_timeout,
                    control_period,
                }
            )) if transaction_timeout == std::time::Duration::from_millis(100)
                && control_period == std::time::Duration::from_millis(20)
        ));
    }
}
