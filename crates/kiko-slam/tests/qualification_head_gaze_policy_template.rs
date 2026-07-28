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

const UNRESOLVED_SENTINELS: [&str; 31] = [
    "${HEAD_ASSEMBLY_DECLARATION_ID}",
    "${HEAD_GAZE_UNVALIDATED_PROPOSAL_EVIDENCE_ID}",
    "${HEAD_GAZE_UNVALIDATED_PROPOSAL_EVIDENCE_CONTENT_SHA256}",
    "${HEAD_GAZE_UNVALIDATED_BOW_MINIMUM_TICKS}",
    "${HEAD_GAZE_UNVALIDATED_BOW_MAXIMUM_TICKS}",
    "${HEAD_GAZE_UNVALIDATED_CURL_MINIMUM_TICKS}",
    "${HEAD_GAZE_UNVALIDATED_CURL_MAXIMUM_TICKS}",
    "${HEAD_GAZE_UNVALIDATED_YAW_MINIMUM_TICKS}",
    "${HEAD_GAZE_UNVALIDATED_YAW_MAXIMUM_TICKS}",
    "${HEAD_GAZE_UNVALIDATED_ROLL_MINIMUM_TICKS}",
    "${HEAD_GAZE_UNVALIDATED_ROLL_MAXIMUM_TICKS}",
    "${HEAD_GAZE_UNVALIDATED_PITCH_DOWN_BOW_TICKS_PER_RADIAN}",
    "${HEAD_GAZE_UNVALIDATED_PITCH_DOWN_CURL_TICKS_PER_RADIAN}",
    "${HEAD_GAZE_UNVALIDATED_YAW_RIGHT_YAW_TICKS_PER_RADIAN}",
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
        ("${HEAD_GAZE_UNVALIDATED_BOW_MINIMUM_TICKS}", "1974"),
        ("${HEAD_GAZE_UNVALIDATED_BOW_MAXIMUM_TICKS}", "2374"),
        ("${HEAD_GAZE_UNVALIDATED_CURL_MINIMUM_TICKS}", "2370"),
        ("${HEAD_GAZE_UNVALIDATED_CURL_MAXIMUM_TICKS}", "2770"),
        ("${HEAD_GAZE_UNVALIDATED_YAW_MINIMUM_TICKS}", "1437"),
        ("${HEAD_GAZE_UNVALIDATED_YAW_MAXIMUM_TICKS}", "1837"),
        ("${HEAD_GAZE_UNVALIDATED_ROLL_MINIMUM_TICKS}", "2847"),
        ("${HEAD_GAZE_UNVALIDATED_ROLL_MAXIMUM_TICKS}", "3247"),
        (
            "${HEAD_GAZE_UNVALIDATED_PITCH_DOWN_BOW_TICKS_PER_RADIAN}",
            "-300.0",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_PITCH_DOWN_CURL_TICKS_PER_RADIAN}",
            "300.0",
        ),
        (
            "${HEAD_GAZE_UNVALIDATED_YAW_RIGHT_YAW_TICKS_PER_RADIAN}",
            "300.0",
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
fn checked_in_template_keeps_every_unknown_value_explicitly_unvalidated() {
    assert!(TEMPLATE.contains("\"kind\": \"proposal_only\""));
    assert!(TEMPLATE.contains("\"calibration_provenance_id\": \"unvalidated-"));
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
