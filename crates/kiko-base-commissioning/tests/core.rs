use kiko_base_commissioning::{
    BASE_IDENTIFICATION_V1, Cancellation, CommissioningAction, CommissioningConfigV1,
    CommissioningConfigV1Dto, CommissioningController, CommissioningEvidence,
    CommissioningEvidenceV1Dto, CommissioningState, CommissioningStopReason, CoverageGate,
    DatasetParseError, EvidenceKind, FitError, IdentificationDatasetV1, IdentificationDatasetV1Dto,
    IdentificationSampleV1Dto, LateralVelocityEvidence, MonotonicTimestampNs, PlantFitConfigV1,
    PlantFitConfigV1Dto, fit_first_order_plant,
};

const WHEELBASE_M: f64 = 0.30;

fn fit_config_dto() -> PlantFitConfigV1Dto {
    PlantFitConfigV1Dto {
        schema_version: BASE_IDENTIFICATION_V1,
        expected_robot_id: "kiko-1".to_owned(),
        expected_controller_session_id: "stm32-session-7".to_owned(),
        expected_visual_velocity_source_id: "slam-forward-v1".to_owned(),
        expected_imu_calibration_id: "imu-cal-4".to_owned(),
        wheelbase_calibration_id: "wheelbase-cal-2".to_owned(),
        wheelbase_m: WHEELBASE_M,
        min_sample_period_s: 0.049,
        max_sample_period_s: 0.051,
        max_sample_period_ratio: 1.01,
        max_abs_observed_forward_velocity_mps: 5.0,
        max_abs_observed_yaw_rate_rad_s: 20.0,
        min_samples: 100,
        max_samples: 1_000,
        holdout_stride: 5,
        min_training_transitions: 100,
        min_holdout_transitions: 20,
        min_abs_excitation_pwm_percent: 10,
        min_symmetric_transitions: 20,
        min_spin_transitions: 20,
        min_zero_transitions: 20,
        min_positive_transitions_per_wheel: 20,
        min_negative_transitions_per_wheel: 20,
        min_command_changes: 8,
        min_time_constant_s: 0.05,
        max_time_constant_s: 2.0,
        time_constant_bound_margin_fraction: 0.02,
        min_abs_velocity_gain_mps_per_pwm_percent: 0.001,
        max_abs_velocity_gain_mps_per_pwm_percent: 0.05,
        require_positive_velocity_gain: true,
        max_normal_matrix_condition_number: 100_000.0,
        min_log_time_constant_sensitivity_mps: 1.0e-5,
        max_holdout_wheel_velocity_rmse_mps: 1.0e-6,
        max_holdout_forward_velocity_rmse_mps: 1.0e-6,
        max_holdout_yaw_rate_rmse_rad_s: 1.0e-5,
        max_holdout_abs_wheel_velocity_error_mps: 1.0e-5,
    }
}

fn advance_wheel(velocity: f64, pwm: i8, gain: f64, tau_s: f64, dt_s: f64) -> f64 {
    let ratio = dt_s / tau_s;
    let decay = (-ratio).exp();
    let response = -(-ratio).exp_m1();
    decay * velocity + gain * response * f64::from(pwm)
}

fn synthetic_dataset_dto(
    left_gain: f64,
    left_tau_s: f64,
    right_gain: f64,
    right_tau_s: f64,
) -> IdentificationDatasetV1Dto {
    let dt_ns = 50_000_000_u64;
    let dt_s = dt_ns as f64 * 1.0e-9;
    let commands = [
        (0_i8, 0_i8),
        (30, 30),
        (0, 0),
        (-30, -30),
        (0, 0),
        (-25, 25),
        (0, 0),
        (25, -25),
        (0, 0),
        (45, 45),
        (0, 0),
        (-45, -45),
        (0, 0),
        (-35, 35),
        (0, 0),
        (35, -35),
        (0, 0),
    ];
    let mut samples = Vec::with_capacity(commands.len() * 24);
    let mut observed_at_ns = 0_u64;
    let mut left_velocity_mps = 0.0_f64;
    let mut right_velocity_mps = 0.0_f64;
    for (segment, (left_pwm, right_pwm)) in commands.into_iter().enumerate() {
        let sequence = u64::try_from(segment + 1).expect("small fixture");
        for _ in 0..24 {
            if !samples.is_empty() {
                left_velocity_mps =
                    advance_wheel(left_velocity_mps, left_pwm, left_gain, left_tau_s, dt_s);
                right_velocity_mps =
                    advance_wheel(right_velocity_mps, right_pwm, right_gain, right_tau_s, dt_s);
                observed_at_ns += dt_ns;
            }
            samples.push(IdentificationSampleV1Dto {
                observed_at_ns,
                applied_command_sequence: sequence,
                applied_left_pwm_percent: left_pwm,
                applied_right_pwm_percent: right_pwm,
                visual_forward_velocity_mps: 0.5 * (left_velocity_mps + right_velocity_mps),
                calibrated_imu_yaw_rate_rad_s: (right_velocity_mps - left_velocity_mps)
                    / WHEELBASE_M,
            });
        }
    }
    IdentificationDatasetV1Dto {
        schema_version: BASE_IDENTIFICATION_V1,
        dataset_content_id: "sha256:synthetic-a".to_owned(),
        robot_id: "kiko-1".to_owned(),
        controller_session_id: "stm32-session-7".to_owned(),
        visual_velocity_source_id: "slam-forward-v1".to_owned(),
        imu_calibration_id: "imu-cal-4".to_owned(),
        wheelbase_calibration_id: "wheelbase-cal-2".to_owned(),
        samples,
    }
}

fn parse_fixture(
    left_gain: f64,
    left_tau_s: f64,
    right_gain: f64,
    right_tau_s: f64,
) -> (PlantFitConfigV1, IdentificationDatasetV1) {
    let config = PlantFitConfigV1::parse(fit_config_dto()).expect("fit config");
    let dataset = IdentificationDatasetV1::parse(
        synthetic_dataset_dto(left_gain, left_tau_s, right_gain, right_tau_s),
        config,
    )
    .expect("synthetic dataset");
    (config, dataset)
}

#[test]
fn exact_unequal_wheel_fixture_recovers_the_existing_first_order_convention() {
    let (config, dataset) = parse_fixture(0.009, 0.31, 0.011, 0.57);
    let fit = fit_first_order_plant(&dataset, config).expect("identifiable exact fixture");

    assert!((fit.left().velocity_gain_mps_per_pwm_percent() - 0.009).abs() < 1.0e-8);
    assert!((fit.right().velocity_gain_mps_per_pwm_percent() - 0.011).abs() < 1.0e-8);
    assert!((fit.left().time_constant_s() - 0.31).abs() < 1.0e-6);
    assert!((fit.right().time_constant_s() - 0.57).abs() < 1.0e-6);
    assert!(fit.holdout_residuals().left_velocity_rmse_mps < 1.0e-8);
    assert!(fit.holdout_residuals().right_velocity_rmse_mps < 1.0e-8);
    assert_eq!(
        fit.support().lateral_velocity,
        LateralVelocityEvidence::Unidentified
    );
    assert!(fit.support().left_pwm_min_percent < 0);
    assert!(fit.support().left_pwm_max_percent > 0);
}

#[test]
fn recovery_is_deterministic_and_holds_across_unequal_parameter_grid() {
    for (left_gain, left_tau, right_gain, right_tau) in [
        (0.006, 0.17, 0.008, 0.28),
        (0.012, 0.42, 0.010, 0.73),
        (0.015, 0.91, 0.007, 0.36),
        (0.004, 1.20, 0.016, 0.22),
    ] {
        let (config, dataset) = parse_fixture(left_gain, left_tau, right_gain, right_tau);
        let first = fit_first_order_plant(&dataset, config).expect("grid fit");
        let second = fit_first_order_plant(&dataset, config).expect("repeat grid fit");
        assert_eq!(first, second);
        assert!((first.left().velocity_gain_mps_per_pwm_percent() - left_gain).abs() < 2.0e-8);
        assert!((first.left().time_constant_s() - left_tau).abs() < 2.0e-5);
        assert!((first.right().velocity_gain_mps_per_pwm_percent() - right_gain).abs() < 2.0e-8);
        assert!((first.right().time_constant_s() - right_tau).abs() < 2.0e-5);
    }
}

#[test]
fn support_envelope_does_not_invent_unobserved_zero_velocity() {
    let left_gain = 0.009;
    let left_tau_s = 0.31;
    let right_gain = 0.011;
    let right_tau_s = 0.57;
    let dt_ns = 50_000_000_u64;
    let dt_s = 0.05;
    let commands = [
        (20_i8, 20_i8),
        (35, 35),
        (25, 25),
        (45, 45),
        (30, 30),
        (40, 40),
    ];
    let mut samples = Vec::with_capacity(commands.len() * 24);
    let mut observed_at_ns = 0_u64;
    let mut left_velocity_mps = left_gain * f64::from(commands[0].0);
    let mut right_velocity_mps = right_gain * f64::from(commands[0].1);
    for (segment, (left_pwm, right_pwm)) in commands.into_iter().enumerate() {
        let sequence = u64::try_from(segment + 1).expect("small fixture");
        for _ in 0..24 {
            if !samples.is_empty() {
                left_velocity_mps =
                    advance_wheel(left_velocity_mps, left_pwm, left_gain, left_tau_s, dt_s);
                right_velocity_mps =
                    advance_wheel(right_velocity_mps, right_pwm, right_gain, right_tau_s, dt_s);
                observed_at_ns += dt_ns;
            }
            samples.push(IdentificationSampleV1Dto {
                observed_at_ns,
                applied_command_sequence: sequence,
                applied_left_pwm_percent: left_pwm,
                applied_right_pwm_percent: right_pwm,
                visual_forward_velocity_mps: 0.5 * (left_velocity_mps + right_velocity_mps),
                calibrated_imu_yaw_rate_rad_s: (right_velocity_mps - left_velocity_mps)
                    / WHEELBASE_M,
            });
        }
    }

    let mut config_dto = fit_config_dto();
    config_dto.holdout_stride = 3;
    config_dto.min_training_transitions = 80;
    config_dto.min_symmetric_transitions = 20;
    config_dto.min_spin_transitions = 0;
    config_dto.min_zero_transitions = 0;
    config_dto.min_negative_transitions_per_wheel = 0;
    config_dto.min_command_changes = 5;
    let config = PlantFitConfigV1::parse(config_dto).expect("positive-only fit config");
    let dataset = IdentificationDatasetV1::parse(
        IdentificationDatasetV1Dto {
            schema_version: BASE_IDENTIFICATION_V1,
            dataset_content_id: "sha256:positive-only".to_owned(),
            robot_id: "kiko-1".to_owned(),
            controller_session_id: "stm32-session-7".to_owned(),
            visual_velocity_source_id: "slam-forward-v1".to_owned(),
            imu_calibration_id: "imu-cal-4".to_owned(),
            wheelbase_calibration_id: "wheelbase-cal-2".to_owned(),
            samples,
        },
        config,
    )
    .expect("positive-only dataset");

    let support = fit_first_order_plant(&dataset, config)
        .expect("identifiable positive-only fit")
        .support();
    assert!(support.left_velocity_min_mps > 0.0);
    assert!(support.right_velocity_min_mps > 0.0);
}

#[test]
fn imu_yaw_rate_alone_cannot_identify_translation() {
    let yaw_rate = 1.2;
    let first_forward = 0.0;
    let second_forward = 0.8;
    let first = (
        first_forward - 0.5 * WHEELBASE_M * yaw_rate,
        first_forward + 0.5 * WHEELBASE_M * yaw_rate,
    );
    let second = (
        second_forward - 0.5 * WHEELBASE_M * yaw_rate,
        second_forward + 0.5 * WHEELBASE_M * yaw_rate,
    );

    assert!(((first.1 - first.0) / WHEELBASE_M - yaw_rate).abs() < 1.0e-12);
    assert!(((second.1 - second.0) / WHEELBASE_M - yaw_rate).abs() < 1.0e-12);
    assert_ne!(first, second);
}

#[test]
fn dataset_rejects_changed_pwm_under_one_applied_sequence() {
    let config = PlantFitConfigV1::parse(fit_config_dto()).expect("config");
    let mut dto = synthetic_dataset_dto(0.009, 0.31, 0.011, 0.57);
    dto.samples[1].applied_left_pwm_percent = 12;
    let error = IdentificationDatasetV1::parse(dto, config).expect_err("ambiguous command hold");
    assert!(matches!(
        error,
        DatasetParseError::ChangedPwmForSameCommand { index: 1, .. }
    ));
}

#[test]
fn fitter_rejects_missing_spin_coverage_before_claiming_a_model() {
    let config = PlantFitConfigV1::parse(fit_config_dto()).expect("config");
    let mut dto = synthetic_dataset_dto(0.009, 0.31, 0.011, 0.57);
    for sample in &mut dto.samples {
        if sample.applied_left_pwm_percent == -sample.applied_right_pwm_percent {
            sample.applied_right_pwm_percent = sample.applied_left_pwm_percent;
        }
    }
    let dataset = IdentificationDatasetV1::parse(dto, config).expect("valid weak dataset");
    let error = fit_first_order_plant(&dataset, config).expect_err("spin gate");
    assert!(matches!(
        error,
        FitError::InsufficientCoverage {
            gate: CoverageGate::SpinTransitions,
            ..
        }
    ));
}

#[test]
fn fitter_rejects_sample_period_outside_the_parsed_contract() {
    let config = PlantFitConfigV1::parse(fit_config_dto()).expect("config");
    let mut dto = synthetic_dataset_dto(0.009, 0.31, 0.011, 0.57);
    for (index, sample) in dto.samples.iter_mut().enumerate() {
        sample.observed_at_ns = u64::try_from(index).expect("small") * 60_000_000;
    }
    let dataset = IdentificationDatasetV1::parse(dto, config).expect("typed dataset");
    assert!(matches!(
        fit_first_order_plant(&dataset, config),
        Err(FitError::SamplePeriodOutsideConfiguredRange {
            interval_start_index: 0,
            ..
        })
    ));
}

#[test]
fn fitter_rejects_an_ill_conditioned_parameter_pair() {
    let mut config_dto = fit_config_dto();
    config_dto.max_normal_matrix_condition_number = 1.0;
    let config = PlantFitConfigV1::parse(config_dto).expect("strict conditioning config");
    let dataset =
        IdentificationDatasetV1::parse(synthetic_dataset_dto(0.009, 0.31, 0.011, 0.57), config)
            .expect("dataset");
    assert!(matches!(
        fit_first_order_plant(&dataset, config),
        Err(FitError::IllConditionedNormalMatrix {
            maximum_condition_number: 1.0,
            ..
        })
    ));
}

#[test]
fn corrupted_held_out_endpoint_fails_without_contaminating_training() {
    let config = PlantFitConfigV1::parse(fit_config_dto()).expect("config");
    let mut dto = synthetic_dataset_dto(0.009, 0.31, 0.011, 0.57);
    dto.samples
        .iter_mut()
        .rfind(|sample| sample.applied_command_sequence == 5)
        .expect("fifth command hold is the first held-out segment")
        .visual_forward_velocity_mps += 0.1;
    let dataset = IdentificationDatasetV1::parse(dto, config).expect("bounded corrupt evidence");
    assert!(matches!(
        fit_first_order_plant(&dataset, config),
        Err(FitError::HoldoutResidualExceeded { .. })
    ));
}

fn commissioning_config() -> CommissioningConfigV1 {
    CommissioningConfigV1::parse(CommissioningConfigV1Dto {
        schema_version: BASE_IDENTIFICATION_V1,
        expected_controller_session_id: "stm32-session-7".to_owned(),
        expected_visual_velocity_source_id: "slam-forward-v1".to_owned(),
        expected_imu_calibration_id: "imu-cal-4".to_owned(),
        symmetric_pwm_percent: 30,
        spin_pwm_percent: 25,
        max_abs_pwm_percent: 35,
        excitation_duration_ns: 20,
        zero_dwell_duration_ns: 10,
        application_timeout_ns: 8,
        max_visual_age_ns: 5,
        max_imu_age_ns: 5,
        max_controller_age_ns: 5,
        max_abs_stationary_forward_velocity_mps: 0.02,
        max_abs_stationary_yaw_rate_rad_s: 0.05,
        max_total_duration_ns: 1_000,
        cycles: 1,
        max_excitation_steps: 4,
    })
    .expect("commissioning config")
}

fn evidence(
    config: CommissioningConfigV1,
    observed_at_ns: u64,
    sequence: u64,
    left: i8,
    right: i8,
) -> CommissioningEvidence {
    CommissioningEvidence::parse(
        CommissioningEvidenceV1Dto {
            controller_session_id: "stm32-session-7".to_owned(),
            visual_velocity_source_id: "slam-forward-v1".to_owned(),
            imu_calibration_id: "imu-cal-4".to_owned(),
            controller_observed_at_ns: observed_at_ns,
            visual_observed_at_ns: observed_at_ns,
            imu_observed_at_ns: observed_at_ns,
            applied_command_sequence: sequence,
            applied_left_pwm_percent: left,
            applied_right_pwm_percent: right,
            visual_forward_velocity_mps: 0.0,
            calibrated_imu_yaw_rate_rad_s: 0.0,
        },
        config,
    )
    .expect("evidence")
}

#[test]
fn excitation_requires_a_fresh_applied_zero_and_zero_dwell() {
    let config = commissioning_config();
    let mut controller = CommissioningController::new(config);
    let nonzero = evidence(config, 0, 1, 4, 4);
    assert!(matches!(
        controller.advance(
            MonotonicTimestampNs::from_nanos(0),
            nonzero,
            Cancellation::Continue
        ),
        CommissioningAction::RequiredZero {
            state: CommissioningState::AwaitingInitialZero
        }
    ));
    let zero = evidence(config, 1, 2, 0, 0);
    assert!(matches!(
        controller.advance(
            MonotonicTimestampNs::from_nanos(1),
            zero,
            Cancellation::Continue
        ),
        CommissioningAction::RequiredZero {
            state: CommissioningState::ZeroDwell { next_step_index: 0 }
        }
    ));
    let almost = evidence(config, 10, 2, 0, 0);
    assert!(matches!(
        controller.advance(
            MonotonicTimestampNs::from_nanos(10),
            almost,
            Cancellation::Continue
        ),
        CommissioningAction::RequiredZero { .. }
    ));
    let ready = evidence(config, 11, 2, 0, 0);
    let action = controller.advance(
        MonotonicTimestampNs::from_nanos(11),
        ready,
        Cancellation::Continue,
    );
    let CommissioningAction::Excitation { step, .. } = action else {
        panic!("zero dwell completion must issue the first bounded step")
    };
    assert_eq!(step.index(), 0);
    assert_eq!(step.pwm().left().get(), 30);
    assert_eq!(step.pwm().right().get(), 30);
}

#[test]
fn stale_evidence_aborts_to_latched_required_zero() {
    let config = commissioning_config();
    let mut controller = CommissioningController::new(config);
    let stale = evidence(config, 4, 1, 0, 0);
    let action = controller.advance(
        MonotonicTimestampNs::from_nanos(10),
        stale,
        Cancellation::Continue,
    );
    assert!(matches!(
        action,
        CommissioningAction::RequiredZero {
            state: CommissioningState::Aborted(CommissioningStopReason::StaleEvidence {
                kind: EvidenceKind::Controller,
                age_ns: 6,
                maximum_age_ns: 5,
            })
        }
    ));
    assert!(action.required_pwm().left().get() == 0 && action.required_pwm().right().get() == 0);
    let fresh = evidence(config, 11, 2, 0, 0);
    assert_eq!(
        controller.advance(
            MonotonicTimestampNs::from_nanos(11),
            fresh,
            Cancellation::Continue
        ),
        action
    );
}

#[test]
fn stale_visual_evidence_aborts_even_when_controller_and_imu_are_fresh() {
    let config = commissioning_config();
    let mut dto = CommissioningEvidenceV1Dto {
        controller_session_id: "stm32-session-7".to_owned(),
        visual_velocity_source_id: "slam-forward-v1".to_owned(),
        imu_calibration_id: "imu-cal-4".to_owned(),
        controller_observed_at_ns: 10,
        visual_observed_at_ns: 4,
        imu_observed_at_ns: 10,
        applied_command_sequence: 1,
        applied_left_pwm_percent: 0,
        applied_right_pwm_percent: 0,
        visual_forward_velocity_mps: 0.0,
        calibrated_imu_yaw_rate_rad_s: 0.0,
    };
    let parsed = CommissioningEvidence::parse(dto.clone(), config).expect("typed evidence");
    let mut controller = CommissioningController::new(config);
    assert!(matches!(
        controller.advance(
            MonotonicTimestampNs::from_nanos(10),
            parsed,
            Cancellation::Continue
        ),
        CommissioningAction::RequiredZero {
            state: CommissioningState::Aborted(CommissioningStopReason::StaleEvidence {
                kind: EvidenceKind::VisualVelocity,
                ..
            })
        }
    ));

    dto.visual_observed_at_ns = 10;
    dto.imu_observed_at_ns = 4;
    let parsed = CommissioningEvidence::parse(dto, config).expect("typed evidence");
    let mut controller = CommissioningController::new(config);
    assert!(matches!(
        controller.advance(
            MonotonicTimestampNs::from_nanos(10),
            parsed,
            Cancellation::Continue
        ),
        CommissioningAction::RequiredZero {
            state: CommissioningState::Aborted(CommissioningStopReason::StaleEvidence {
                kind: EvidenceKind::ImuYawRate,
                ..
            })
        }
    ));
}

#[test]
fn applied_zero_with_observed_motion_cannot_unlock_excitation() {
    let config = commissioning_config();
    let moving = CommissioningEvidence::parse(
        CommissioningEvidenceV1Dto {
            controller_session_id: "stm32-session-7".to_owned(),
            visual_velocity_source_id: "slam-forward-v1".to_owned(),
            imu_calibration_id: "imu-cal-4".to_owned(),
            controller_observed_at_ns: 0,
            visual_observed_at_ns: 0,
            imu_observed_at_ns: 0,
            applied_command_sequence: 1,
            applied_left_pwm_percent: 0,
            applied_right_pwm_percent: 0,
            visual_forward_velocity_mps: 0.021,
            calibrated_imu_yaw_rate_rad_s: 0.0,
        },
        config,
    )
    .expect("finite moving evidence");
    let mut controller = CommissioningController::new(config);
    assert!(matches!(
        controller.advance(
            MonotonicTimestampNs::from_nanos(0),
            moving,
            Cancellation::Continue
        ),
        CommissioningAction::RequiredZero {
            state: CommissioningState::Aborted(
                CommissioningStopReason::MotionWhileZeroRequired { .. }
            )
        }
    ));
}

#[test]
fn one_controller_sequence_cannot_change_its_applied_pwm() {
    let config = commissioning_config();
    let mut controller = CommissioningController::new(config);
    let _ = controller.advance(
        MonotonicTimestampNs::from_nanos(0),
        evidence(config, 0, 7, 4, 4),
        Cancellation::Continue,
    );
    let changed = evidence(config, 1, 7, 0, 0);
    assert!(matches!(
        controller.advance(
            MonotonicTimestampNs::from_nanos(1),
            changed,
            Cancellation::Continue
        ),
        CommissioningAction::RequiredZero {
            state: CommissioningState::Aborted(
                CommissioningStopReason::ChangedPwmForSameControllerSequence { sequence: 7, .. }
            )
        }
    ));
}

#[test]
fn clock_regression_and_cancellation_each_latch_zero() {
    let config = commissioning_config();
    let mut clock_controller = CommissioningController::new(config);
    let first = evidence(config, 10, 1, 0, 0);
    let _ = clock_controller.advance(
        MonotonicTimestampNs::from_nanos(10),
        first,
        Cancellation::Continue,
    );
    let regressed = evidence(config, 9, 1, 0, 0);
    assert!(matches!(
        clock_controller.advance(
            MonotonicTimestampNs::from_nanos(9),
            regressed,
            Cancellation::Continue
        ),
        CommissioningAction::RequiredZero {
            state: CommissioningState::Aborted(CommissioningStopReason::ClockRegression {
                previous_ns: 10,
                current_ns: 9,
            })
        }
    ));

    let mut cancelled = CommissioningController::new(config);
    let evidence = evidence(config, 0, 1, 0, 0);
    assert!(matches!(
        cancelled.advance(
            MonotonicTimestampNs::from_nanos(0),
            evidence,
            Cancellation::Requested
        ),
        CommissioningAction::RequiredZero {
            state: CommissioningState::Aborted(CommissioningStopReason::Cancelled)
        }
    ));
}

#[test]
fn application_timeout_is_fail_closed_at_the_exact_limit() {
    let config = commissioning_config();
    let mut controller = CommissioningController::new(config);
    let zero = evidence(config, 0, 1, 0, 0);
    let _ = controller.advance(
        MonotonicTimestampNs::from_nanos(0),
        zero,
        Cancellation::Continue,
    );
    let zero = evidence(config, 10, 1, 0, 0);
    let issued = controller.advance(
        MonotonicTimestampNs::from_nanos(10),
        zero,
        Cancellation::Continue,
    );
    assert!(matches!(issued, CommissioningAction::Excitation { .. }));
    let still_zero = evidence(config, 18, 1, 0, 0);
    assert!(matches!(
        controller.advance(
            MonotonicTimestampNs::from_nanos(18),
            still_zero,
            Cancellation::Continue
        ),
        CommissioningAction::RequiredZero {
            state: CommissioningState::Aborted(CommissioningStopReason::ApplicationTimeout {
                step_index: 0,
                elapsed_ns: 8,
                maximum_ns: 8,
            })
        }
    ));
}

#[test]
fn total_duration_limit_is_fail_closed_at_the_exact_deadline() {
    let config = commissioning_config();
    let mut controller = CommissioningController::new(config);
    let _ = controller.advance(
        MonotonicTimestampNs::from_nanos(0),
        evidence(config, 0, 1, 0, 0),
        Cancellation::Continue,
    );
    let at_deadline = evidence(config, 1_000, 1, 0, 0);
    assert!(matches!(
        controller.advance(
            MonotonicTimestampNs::from_nanos(1_000),
            at_deadline,
            Cancellation::Continue
        ),
        CommissioningAction::RequiredZero {
            state: CommissioningState::Aborted(
                CommissioningStopReason::TotalDurationLimitReached {
                    elapsed_ns: 1_000,
                    maximum_ns: 1_000,
                }
            )
        }
    ));
}

#[test]
fn every_emitted_program_step_is_bounded_symmetric_or_spin() {
    let config = commissioning_config();
    let mut controller = CommissioningController::new(config);
    let mut now = 0_u64;
    let mut sequence = 1_u64;
    let mut applied = (0_i8, 0_i8);
    let mut emitted = Vec::new();

    for _ in 0..200 {
        let current_evidence = evidence(config, now, sequence, applied.0, applied.1);
        let action = controller.advance(
            MonotonicTimestampNs::from_nanos(now),
            current_evidence,
            Cancellation::Continue,
        );
        match action {
            CommissioningAction::Excitation { step, .. } => {
                if emitted.last().copied() != Some(step.index()) {
                    emitted.push(step.index());
                }
                let left = step.pwm().left().get();
                let right = step.pwm().right().get();
                assert!(left.unsigned_abs() <= config.max_abs_pwm_percent().get());
                assert!(right.unsigned_abs() <= config.max_abs_pwm_percent().get());
                assert!(left == right || i16::from(left) == -i16::from(right));
                if applied != (left, right) {
                    applied = (left, right);
                    sequence += 1;
                }
                now += 10;
            }
            CommissioningAction::RequiredZero { state } => {
                if applied != (0, 0) {
                    applied = (0, 0);
                    sequence += 1;
                }
                if state == CommissioningState::Completed {
                    break;
                }
                assert!(!matches!(state, CommissioningState::Aborted(_)));
                now += 10;
            }
        }
    }
    assert_eq!(emitted, vec![0, 1, 2, 3]);
    assert_eq!(controller.state(), CommissioningState::Completed);
}
