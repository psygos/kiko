#![cfg(feature = "agent-runtime")]

use kiko_slam::navigation::{
    HeadGazePolicyLifecycleClaim, HeadGazePolicyV1, KIKO_REVIEWED_NATURAL_HEAD_TARGET_TICKS,
    KIKO_REVIEWED_NATURAL_HEAD_TORQUE_LIMIT_PERMILLE, MAX_HEAD_GAZE_POLICY_JSON_BYTES,
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
const PRODUCTION_TEMPLATE: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../configs/nano-agent-template/head-gaze-policy-v1.json.template"
));
const FABLE_CONFIG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../deploy/expression/config.json"
));
const CURRENT_FABLE_CONFIG_PROVENANCE_ID: &str =
    "fable-config-sha256-46d69519425caba5ace1920d39ff8a07101bf86b79eacb1cdeb53f1dd8957a56";

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
    assert!(TEMPLATE.contains(&format!(
        "\"calibration_provenance_id\": \"{CURRENT_FABLE_CONFIG_PROVENANCE_ID}\""
    )));
    assert!(!TEMPLATE.contains("operator_claimed_physical_review"));
    for sentinel in UNRESOLVED_SENTINELS {
        assert!(
            TEMPLATE.contains(sentinel),
            "template lost unresolved deployment sentinel {sentinel}"
        );
    }
}

#[test]
fn production_and_qualification_templates_share_the_current_fable_assembly_literals() {
    for template in [TEMPLATE, PRODUCTION_TEMPLATE] {
        for required in [
            CURRENT_FABLE_CONFIG_PROVENANCE_ID,
            "\"bow_ticks\": 1505",
            "\"curl_ticks\": 3937",
            "\"yaw_ticks\": 1551",
            "\"roll_ticks\": 3018",
            "\"bow_ticks_per_radian\": -217.0",
            "\"curl_ticks_per_radian\": 403.0",
            "\"bow\": 650",
            "\"curl\": 550",
        ] {
            assert!(
                template.contains(required),
                "head policy template lost current Fable literal {required}"
            );
        }
        for obsolete in [
            "kiko-follow-config-0d98af8c-2026-07-31",
            "\"bow_ticks\": 2174",
            "\"curl_ticks\": 2570",
            "\"bow_ticks_per_radian\": -93.0",
            "\"curl_ticks_per_radian\": 465.0",
        ] {
            assert!(
                !template.contains(obsolete),
                "head policy template retained obsolete assembly literal {obsolete}"
            );
        }
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
        KIKO_REVIEWED_NATURAL_HEAD_TARGET_TICKS
    );
    assert_eq!(
        kiko_head_protocol::HeadJoint::ALL.map(|joint| {
            let envelope = mapping.hard_envelope(joint);
            [envelope.minimum().get(), envelope.maximum().get()]
        }),
        [[1395, 1615], [3787, 4087], [1071, 2031], [2858, 3178]]
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
        [-217.0, 403.0, -1050.0]
    );
    let recruitment = mapping
        .dynamic_pitch_recruitment()
        .expect("template carries Fable's dynamic serial-pitch recruitment");
    assert_eq!(recruitment.maximum_bow_share_permille(), 600);
    assert_eq!(recruitment.full_recruitment_total_pitch_demand_ticks(), 140);
    let turn_dip = policy
        .turn_dip_posture()
        .expect("template carries Fable's gaze-neutral turn posture");
    assert_eq!(turn_dip.turn_rate_deadband_ticks_per_second(), 120);
    assert_eq!(turn_dip.maximum_dip_ticks(), 26);
    assert_eq!(turn_dip.excess_turn_rate_to_dip_milliseconds(), 80);
    assert_eq!(turn_dip.decay_retention_permille(), 850);
    assert_eq!(
        turn_dip.decay_reference_period(),
        std::time::Duration::from_millis(50)
    );
    assert_eq!(
        turn_dip.maximum_rate_interval(),
        std::time::Duration::from_millis(500)
    );
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
        [110, -150, 480, 160]
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
        KIKO_REVIEWED_NATURAL_HEAD_TORQUE_LIMIT_PERMILLE
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

#[test]
fn checked_in_policy_mapping_is_derived_from_the_current_fable_configuration() {
    let config: serde_json::Value =
        serde_json::from_str(FABLE_CONFIG).expect("current Fable configuration is JSON");
    let rendered = render_parser_fixture();
    let policy = HeadGazePolicyV1::parse_json(rendered.as_bytes())
        .expect("rendered parser fixture is valid");
    let mapping = policy.mapping();

    let natural: [u16; 4] = config["natural_ticks"]
        .as_array()
        .expect("natural_ticks array")
        .iter()
        .map(|value| {
            u16::try_from(value.as_u64().expect("natural tick is an unsigned integer"))
                .expect("natural tick fits u16")
        })
        .collect::<Vec<_>>()
        .try_into()
        .expect("natural_ticks has four joints");
    assert_eq!(natural, KIKO_REVIEWED_NATURAL_HEAD_TARGET_TICKS);
    assert_eq!(
        mapping
            .natural_declaration()
            .positions()
            .map(|position| position.get()),
        natural
    );

    let limits = [
        config["bow_limit_ticks"].as_u64().expect("bow limit"),
        config["curl_limit_ticks"].as_u64().expect("curl limit"),
        config["yaw_limit_ticks"].as_u64().expect("yaw limit"),
        config["roll_limit_ticks"].as_u64().expect("roll limit"),
    ]
    .map(|value| u16::try_from(value).expect("joint limit fits u16"));
    for joint in kiko_head_protocol::HeadJoint::ALL {
        let index = joint as usize;
        let envelope = mapping.hard_envelope(joint);
        assert_eq!(
            [envelope.minimum().get(), envelope.maximum().get()],
            [
                natural[index].saturating_sub(limits[index]),
                natural[index].saturating_add(limits[index]),
            ]
        );
    }

    let pitch_ticks_per_rad = config["pitch_ticks_per_rad"].as_f64().expect("pitch scale");
    let bow_share = config["bow_pitch_share"].as_f64().expect("bow share");
    let curl_share = config["curl_pitch_share"].as_f64().expect("curl share");
    let curl_sign = config["curl_sign"].as_i64().expect("curl sign") as f64;
    let expected_pitch_down = [
        (curl_sign * bow_share * pitch_ticks_per_rad).round(),
        (-curl_sign * curl_share * pitch_ticks_per_rad).round(),
    ];
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
        ],
        expected_pitch_down
    );
    assert_eq!(
        mapping.tick_offset_per_radian(
            kiko_expression_runtime::HeadGazeCoordinate::YawRight,
            kiko_head_protocol::HeadJoint::Yaw,
        ),
        config["yaw_sign"].as_i64().expect("yaw sign") as f64
            * config["yaw_ticks_per_rad"].as_f64().expect("yaw scale")
    );

    let character = policy
        .character_mapping()
        .expect("current policy retains four-axis character mapping");
    assert_eq!(
        kiko_head_protocol::HeadJoint::ALL
            .map(|joint| character.full_scale_tick_offset(joint).unsigned_abs()),
        limits
    );
    let torque: [u16; 4] = config["torque_limit_permille"]
        .as_array()
        .expect("torque array")
        .iter()
        .map(|value| {
            u16::try_from(value.as_u64().expect("torque is an unsigned integer"))
                .expect("torque fits u16")
        })
        .collect::<Vec<_>>()
        .try_into()
        .expect("torque array has four joints");
    assert_eq!(torque, KIKO_REVIEWED_NATURAL_HEAD_TORQUE_LIMIT_PERMILLE);
    let compliant = policy
        .compliant_hold()
        .expect("current policy retains compliant hold");
    assert_eq!(
        kiko_head_protocol::HeadJoint::ALL
            .map(|joint| compliant.holding_torque_limits().for_joint(joint).get()),
        torque
    );
}
