#![cfg(feature = "agent-runtime")]

use kiko_slam::navigation::{
    HeadGazePolicyLifecycleClaim, HeadGazePolicyV1, MAX_HEAD_GAZE_POLICY_JSON_BYTES,
};
#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
use kiko_slam::navigation::{
    QualificationHeadGazePolicyAdmissionError, QualificationHeadGazeProposalOnlyPolicy,
};
#[cfg(all(feature = "nano-wheels-off-qualification", unix))]
use serde_json::json;

const TEMPLATE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../configs/nano-wheels-off-qualification-template/head-gaze-policy-v1.json.template"
));

const UNRESOLVED_SENTINELS: [&str; 44] = [
    "${HEAD_ASSEMBLY_DECLARATION_ID}",
    "${HEAD_GAZE_UNVALIDATED_PROPOSAL_EVIDENCE_ID}",
    "${HEAD_GAZE_UNVALIDATED_PROPOSAL_EVIDENCE_CONTENT_SHA256}",
    "${HEAD_GAZE_UNVALIDATED_CONTROL_PERIOD_NS}",
    "${HEAD_GAZE_UNVALIDATED_MAXIMUM_TICK_LATENESS_NS}",
    "${HEAD_GAZE_UNVALIDATED_PROPOSAL_TTL_NS}",
    "${HEAD_GAZE_UNVALIDATED_SETTLE_DEADBAND_TICKS}",
    "${HEAD_GAZE_UNVALIDATED_RESUME_THRESHOLD_TICKS}",
    "${HEAD_GAZE_UNVALIDATED_BOW_MAXIMUM_VELOCITY_TICKS_PER_CONTROL_TICK}",
    "${HEAD_GAZE_UNVALIDATED_BOW_MAXIMUM_ACCELERATION_TICKS_PER_CONTROL_TICK_SQUARED}",
    "${HEAD_GAZE_UNVALIDATED_BOW_MAXIMUM_POSITION_STEP_TICKS}",
    "${HEAD_GAZE_UNVALIDATED_CURL_MAXIMUM_VELOCITY_TICKS_PER_CONTROL_TICK}",
    "${HEAD_GAZE_UNVALIDATED_CURL_MAXIMUM_ACCELERATION_TICKS_PER_CONTROL_TICK_SQUARED}",
    "${HEAD_GAZE_UNVALIDATED_CURL_MAXIMUM_POSITION_STEP_TICKS}",
    "${HEAD_GAZE_UNVALIDATED_YAW_MAXIMUM_VELOCITY_TICKS_PER_CONTROL_TICK}",
    "${HEAD_GAZE_UNVALIDATED_YAW_MAXIMUM_ACCELERATION_TICKS_PER_CONTROL_TICK_SQUARED}",
    "${HEAD_GAZE_UNVALIDATED_YAW_MAXIMUM_POSITION_STEP_TICKS}",
    "${HEAD_GAZE_UNVALIDATED_ROLL_MAXIMUM_VELOCITY_TICKS_PER_CONTROL_TICK}",
    "${HEAD_GAZE_UNVALIDATED_ROLL_MAXIMUM_ACCELERATION_TICKS_PER_CONTROL_TICK_SQUARED}",
    "${HEAD_GAZE_UNVALIDATED_ROLL_MAXIMUM_POSITION_STEP_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_OBSERVATION_TRANSACTION_TIMEOUT_NS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_MAXIMUM_OBSERVATION_SPAN_NS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_OBSERVATION_TTL_NS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_CONTACT_ARM_DWELL_NS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_CONTACT_ACQUISITION_SAMPLES}",
    "${HEAD_COMPLIANCE_UNVALIDATED_RELEASE_DWELL_NS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_RECOVERY_DURATION_NS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_FOLLOW_PERMILLE}",
    "${HEAD_COMPLIANCE_UNVALIDATED_BOW_CONTACT_ENTRY_ERROR_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_BOW_CONTACT_RELEASE_ERROR_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_BOW_MAXIMUM_YIELD_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_BOW_MAXIMUM_OBSERVED_STEP_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_CURL_CONTACT_ENTRY_ERROR_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_CURL_CONTACT_RELEASE_ERROR_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_CURL_MAXIMUM_YIELD_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_CURL_MAXIMUM_OBSERVED_STEP_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_YAW_CONTACT_ENTRY_ERROR_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_YAW_CONTACT_RELEASE_ERROR_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_YAW_MAXIMUM_YIELD_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_YAW_MAXIMUM_OBSERVED_STEP_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_ROLL_CONTACT_ENTRY_ERROR_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_ROLL_CONTACT_RELEASE_ERROR_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_ROLL_MAXIMUM_YIELD_TICKS}",
    "${HEAD_COMPLIANCE_UNVALIDATED_ROLL_MAXIMUM_OBSERVED_STEP_TICKS}",
];

fn render_parser_fixture() -> String {
    let substitutions = [
        (
            "${HEAD_ASSEMBLY_DECLARATION_ID}",
            "kiko-head-test-unverified",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_PROPOSAL_EVIDENCE_ID}",
            "synthetic-template-contract-test",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_PROPOSAL_EVIDENCE_CONTENT_SHA256}",
            "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff",
        ),
        ("${HEAD_GAZE_UNVALIDATED_CONTROL_PERIOD_NS}", "20000000"),
        (
            "${HEAD_GAZE_UNVALIDATED_MAXIMUM_TICK_LATENESS_NS}",
            "5000000",
        ),
        ("${HEAD_GAZE_UNVALIDATED_PROPOSAL_TTL_NS}", "150000000"),
        ("${HEAD_GAZE_UNVALIDATED_SETTLE_DEADBAND_TICKS}", "2"),
        ("${HEAD_GAZE_UNVALIDATED_RESUME_THRESHOLD_TICKS}", "5"),
        (
            "${HEAD_GAZE_UNVALIDATED_BOW_MAXIMUM_VELOCITY_TICKS_PER_CONTROL_TICK}",
            "8",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_BOW_MAXIMUM_ACCELERATION_TICKS_PER_CONTROL_TICK_SQUARED}",
            "2",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_BOW_MAXIMUM_POSITION_STEP_TICKS}",
            "8",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_CURL_MAXIMUM_VELOCITY_TICKS_PER_CONTROL_TICK}",
            "8",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_CURL_MAXIMUM_ACCELERATION_TICKS_PER_CONTROL_TICK_SQUARED}",
            "2",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_CURL_MAXIMUM_POSITION_STEP_TICKS}",
            "8",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_YAW_MAXIMUM_VELOCITY_TICKS_PER_CONTROL_TICK}",
            "8",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_YAW_MAXIMUM_ACCELERATION_TICKS_PER_CONTROL_TICK_SQUARED}",
            "2",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_YAW_MAXIMUM_POSITION_STEP_TICKS}",
            "8",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_ROLL_MAXIMUM_VELOCITY_TICKS_PER_CONTROL_TICK}",
            "4",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_ROLL_MAXIMUM_ACCELERATION_TICKS_PER_CONTROL_TICK_SQUARED}",
            "1",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_ROLL_MAXIMUM_POSITION_STEP_TICKS}",
            "4",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_OBSERVATION_TRANSACTION_TIMEOUT_NS}",
            "5000000",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_MAXIMUM_OBSERVATION_SPAN_NS}",
            "4000000",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_OBSERVATION_TTL_NS}",
            "30000000",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_CONTACT_ARM_DWELL_NS}",
            "40000000",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_CONTACT_ACQUISITION_SAMPLES}",
            "2",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_RELEASE_DWELL_NS}",
            "100000000",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_RECOVERY_DURATION_NS}",
            "1000000000",
        ),
        ("${HEAD_COMPLIANCE_UNVALIDATED_FOLLOW_PERMILLE}", "800"),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_BOW_CONTACT_ENTRY_ERROR_TICKS}",
            "20",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_BOW_CONTACT_RELEASE_ERROR_TICKS}",
            "6",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_BOW_MAXIMUM_YIELD_TICKS}",
            "80",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_BOW_MAXIMUM_OBSERVED_STEP_TICKS}",
            "100",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_CURL_CONTACT_ENTRY_ERROR_TICKS}",
            "20",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_CURL_CONTACT_RELEASE_ERROR_TICKS}",
            "6",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_CURL_MAXIMUM_YIELD_TICKS}",
            "100",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_CURL_MAXIMUM_OBSERVED_STEP_TICKS}",
            "100",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_YAW_CONTACT_ENTRY_ERROR_TICKS}",
            "20",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_YAW_CONTACT_RELEASE_ERROR_TICKS}",
            "6",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_YAW_MAXIMUM_YIELD_TICKS}",
            "180",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_YAW_MAXIMUM_OBSERVED_STEP_TICKS}",
            "100",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_ROLL_CONTACT_ENTRY_ERROR_TICKS}",
            "20",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_ROLL_CONTACT_RELEASE_ERROR_TICKS}",
            "6",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_ROLL_MAXIMUM_YIELD_TICKS}",
            "90",
        ),
        (
            "${HEAD_COMPLIANCE_UNVALIDATED_ROLL_MAXIMUM_OBSERVED_STEP_TICKS}",
            "100",
        ),
    ];

    let mut rendered = TEMPLATE.to_owned();
    for (sentinel, fixture_value) in substitutions {
        assert!(
            rendered.contains(sentinel),
            "template lost required sentinel {sentinel}"
        );
        rendered = rendered.replace(sentinel, fixture_value);
    }
    assert!(
        !rendered.contains("${"),
        "the parser fixture did not resolve every template sentinel"
    );
    rendered
}

#[test]
fn checked_in_template_keeps_every_remaining_unknown_explicitly_unvalidated() {
    assert!(TEMPLATE.contains("\"kind\": \"proposal_only\""));
    assert!(
        TEMPLATE
            .contains("\"calibration_provenance_id\": \"kiko-follow-config-0d98af8c-2026-07-31\"")
    );
    assert!(!TEMPLATE.contains("operator_claimed_physical_review"));
    for sentinel in UNRESOLVED_SENTINELS {
        assert!(
            TEMPLATE.contains(sentinel),
            "template lost unresolved deployment sentinel {sentinel}"
        );
    }
}

#[test]
fn rendered_contract_is_typed_proposal_only_with_exact_known_geometry() {
    let rendered = render_parser_fixture();
    assert!(rendered.len() <= MAX_HEAD_GAZE_POLICY_JSON_BYTES);
    let policy = HeadGazePolicyV1::parse_json(rendered.as_bytes())
        .expect("rendered parser fixture is valid");

    let HeadGazePolicyLifecycleClaim::ProposalOnly(proposal) = policy.lifecycle() else {
        panic!("the qualification template must never claim physical review");
    };
    assert_eq!(
        proposal.proposal_id().as_str(),
        "kiko-head-gaze-proposal-only-unvalidated-v1"
    );

    let mapping = policy.mapping();
    assert_eq!(mapping.focus_plane().get(), 1.5);
    assert_eq!(
        mapping.camera_to_head().head_origin_in_camera_m(),
        [0.0, -0.25, -0.20]
    );
    assert_eq!(
        mapping
            .camera_to_head()
            .neutral_head_from_camera_rotation_rows(),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    );
    assert_eq!(
        mapping
            .natural_declaration()
            .positions()
            .map(|position| position.get()),
        [2174, 2570, 1637, 3047]
    );
    assert_eq!(
        kiko_head_protocol::HeadJoint::ALL.map(|joint| {
            let envelope = mapping.hard_envelope(joint);
            [envelope.minimum().get(), envelope.maximum().get()]
        }),
        [[2064, 2284], [2390, 2750], [1157, 2117], [2887, 3207]]
    );
    assert_eq!(
        [
            mapping.tick_offset_per_radian(
                kiko_expression_runtime::HeadGazeCoordinate::PitchDown,
                kiko_head_protocol::HeadJoint::Bow,
            ),
            mapping.tick_offset_per_radian(
                kiko_expression_runtime::HeadGazeCoordinate::PitchDown,
                kiko_head_protocol::HeadJoint::Curl,
            ),
            mapping.tick_offset_per_radian(
                kiko_expression_runtime::HeadGazeCoordinate::YawRight,
                kiko_head_protocol::HeadJoint::Yaw,
            ),
        ],
        [-93.0, 465.0, -1050.0]
    );
    let recruitment = mapping
        .dynamic_pitch_recruitment()
        .expect("template carries Fable's dynamic serial-pitch recruitment");
    assert_eq!(recruitment.maximum_bow_share_permille(), 600);
    assert_eq!(recruitment.full_recruitment_total_pitch_demand_ticks(), 140);
    let character = policy
        .character_mapping()
        .expect("template retains an explicit proposal-only four-joint mapping");
    assert_eq!(
        [
            character.full_scale_tick_offset(kiko_head_protocol::HeadJoint::Bow),
            character.full_scale_tick_offset(kiko_head_protocol::HeadJoint::Curl),
            character.full_scale_tick_offset(kiko_head_protocol::HeadJoint::Yaw),
            character.full_scale_tick_offset(kiko_head_protocol::HeadJoint::Roll),
        ],
        [110, -180, 480, 160]
    );
    let organic = policy
        .controller()
        .organic_motion()
        .expect("template carries the field-derived organic motion policy");
    assert_eq!(
        kiko_head_protocol::HeadJoint::ALL.map(|joint| organic.joint(joint).response_millihertz()),
        [400, 850, 1050, 900]
    );
    assert_eq!(
        kiko_head_protocol::HeadJoint::ALL.map(|joint| organic.joint(joint).damping_permille()),
        [1400, 1400, 1150, 850]
    );
    let compliant = policy
        .compliant_hold()
        .expect("rendered fixture carries typed but explicitly unvalidated compliance dynamics");
    assert_eq!(
        compliant.control_period(),
        std::time::Duration::from_millis(20)
    );
    assert_eq!(
        compliant.observation_transaction_timeout(),
        std::time::Duration::from_millis(5)
    );
    assert_eq!(
        compliant.maximum_observation_span(),
        std::time::Duration::from_millis(4)
    );
    assert_eq!(
        compliant.observation_ttl(),
        std::time::Duration::from_millis(30)
    );
    assert_eq!(
        compliant.contact_arm_dwell(),
        std::time::Duration::from_millis(40)
    );
    assert_eq!(compliant.contact_acquisition_samples(), 2);
    assert_eq!(
        compliant.release_dwell(),
        std::time::Duration::from_millis(100)
    );
    assert_eq!(
        compliant.recovery_duration(),
        std::time::Duration::from_secs(1)
    );
    assert_eq!(compliant.follow_permille(), 800);
    assert_eq!(
        kiko_head_protocol::HeadJoint::ALL
            .map(|joint| { compliant.holding_torque_limits().for_joint(joint).get() }),
        [600, 400, 400, 400]
    );

    #[cfg(all(feature = "nano-wheels-off-qualification", unix))]
    {
        let policy = HeadGazePolicyV1::parse_json(rendered.as_bytes())
            .expect("rendered parser fixture remains valid");
        QualificationHeadGazeProposalOnlyPolicy::try_from(policy)
            .expect("qualification admits proposal-only metadata");

        let mut reviewed: serde_json::Value =
            serde_json::from_str(&rendered).expect("rendered policy JSON");
        reviewed["lifecycle"] = json!({
            "kind": "operator_claimed_physical_review",
            "review_id": "synthetic-review-claim",
            "operator_id": "operator:test",
            "evidence_id": "synthetic-review-evidence",
            "evidence_content_sha256_hex":
                "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd"
        });
        let reviewed = HeadGazePolicyV1::parse_json(
            &serde_json::to_vec(&reviewed).expect("reviewed fixture JSON"),
        )
        .expect("reviewed policy remains valid in the general policy domain");
        assert_eq!(
            QualificationHeadGazeProposalOnlyPolicy::try_from(reviewed),
            Err(QualificationHeadGazePolicyAdmissionError::NotProposalOnly)
        );
    }
}
