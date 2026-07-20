use robot_command_client::fake::{FakeClock, FakeStep, FakeTransport};
use robot_command_client::{
    ClientConfig, ClientConfigInput, DisarmedCommandClient, EvidenceError, FailureCause,
    LatchedStopKnowledge, MonotonicClock, PendingPhysicalCommand, RobotProtocolV2WireAdapter,
    UdpV2Transport, V2CommandTransport, V2WireAdapter,
};
use robot_protocol::ControllerUptimeMsWrapping;
use robot_protocol::v2::{
    AcquireResult, AcquireResultCode, ActuatorConfigFingerprint, ControlEpoch, ControllerBootId,
    ControllerCapabilities, ControllerDeadlineMsWrapping, ControllerFaults, ControllerUid,
    ForceStopReason, HostCommandResult, HostCommandResultCode, HostStopResult, MAX_RAW_FRAME_BYTES,
    Message, MessageKind, OutputState, RawFrame, RemainingLeaseMs, RequestId, StatusCode,
    StatusReport, StopResultCode, TargetBootId, TimerPwm, V2CommandLeaseMs, V2CommandSequence,
};
use std::net::UdpSocket;
use std::time::Duration;

const UID_BYTES: [u8; 12] = [0x11; 12];
const FINGERPRINT_BYTES: [u8; 16] = [0x22; 16];
const FIRMWARE_ABI: u16 = 7;
const FIRMWARE_BUILD_ID: u32 = 9;
const RESPONSE_DELAY: Duration = Duration::from_millis(1);

fn uid() -> ControllerUid {
    ControllerUid::try_new(UID_BYTES).expect("nonzero fixture UID")
}

fn boot() -> ControllerBootId {
    ControllerBootId::try_new(17).expect("nonzero fixture boot")
}

fn other_boot() -> ControllerBootId {
    ControllerBootId::try_new(18).expect("nonzero fixture boot")
}

fn epoch() -> ControlEpoch {
    ControlEpoch::try_new(23).expect("nonzero fixture epoch")
}

fn fingerprint() -> ActuatorConfigFingerprint {
    ActuatorConfigFingerprint::try_new(FINGERPRINT_BYTES).expect("nonzero fixture fingerprint")
}

fn capabilities() -> ControllerCapabilities {
    ControllerCapabilities::try_from_bits(ControllerCapabilities::REQUIRED_BITS)
        .expect("known required capability bits")
}

fn lease() -> V2CommandLeaseMs {
    V2CommandLeaseMs::try_new(100).expect("bounded fixture lease")
}

fn input() -> ClientConfigInput<'static> {
    ClientConfigInput {
        command_endpoint: "127.0.0.1:8080",
        controller_uid_hex: "111111111111111111111111",
        expected_firmware_abi: "7",
        expected_firmware_build_id: "9",
        expected_actuator_config_fingerprint_hex: "22222222222222222222222222222222",
        status_timeout_ns: "50000000",
        acquire_timeout_ns: "50000000",
        applied_ack_timeout_ns: "50000000",
        stop_attempt_timeout_ns: "50000000",
        max_stop_recovery_attempts: "3",
        zero_acquisition_lease_ms: "100",
    }
}

fn config() -> ClientConfig {
    ClientConfig::parse(input()).expect("valid fixture client config")
}

fn status_report(request_id: u32) -> Message {
    Message::StatusReport(StatusReport {
        controller_uid: uid(),
        observed_boot_id: TargetBootId::Exact(boot()),
        request_id: RequestId::new(request_id),
        status: StatusCode::ReadyStopped,
        control_epoch: None,
        controller_uptime: ControllerUptimeMsWrapping::new(1_000),
        capabilities: capabilities(),
        output_state: OutputState::Disabled,
        controller_timer_pwm: TimerPwm::ZERO,
        remaining_lease: RemainingLeaseMs::ZERO,
        faults: ControllerFaults::NONE,
    })
}

fn acquire_result(request_id: u32) -> Message {
    Message::AcquireResult(AcquireResult {
        controller_uid: uid(),
        boot_id: boot(),
        request_id: RequestId::new(request_id),
        control_epoch: Some(epoch()),
        result: AcquireResultCode::Granted,
        capabilities: capabilities(),
        faults: ControllerFaults::NONE,
        observed_firmware_abi: FIRMWARE_ABI,
        observed_firmware_build_id: FIRMWARE_BUILD_ID,
        observed_actuator_config_fingerprint: fingerprint(),
    })
}

fn command_result(sequence: u32, pwm: TimerPwm, result: HostCommandResultCode) -> Message {
    Message::HostCommandResult(HostCommandResult {
        controller_uid: uid(),
        boot_id: boot(),
        control_epoch: epoch(),
        sequence: V2CommandSequence::new(sequence),
        result,
        requested_timer_pwm: pwm,
        controller_timer_pwm: pwm,
        output_state: if pwm.is_zero() {
            OutputState::ZeroPwm
        } else {
            OutputState::NonzeroPwm
        },
        controller_applied_at: ControllerUptimeMsWrapping::new(2_000),
        controller_expires_at: ControllerDeadlineMsWrapping::new(2_100),
        remaining_lease: RemainingLeaseMs::try_new(90).expect("bounded remaining lease"),
        faults: ControllerFaults::NONE,
    })
}

fn stop_result(request_id: u32) -> Message {
    Message::HostStopResult(HostStopResult {
        controller_uid: uid(),
        observed_boot_id: TargetBootId::Exact(boot()),
        request_id: RequestId::new(request_id),
        result: StopResultCode::ControllerConfirmed,
        output_state: OutputState::Disabled,
        controller_uptime: ControllerUptimeMsWrapping::new(3_000),
        faults: ControllerFaults::NONE,
    })
}

fn acquisition_steps() -> Vec<FakeStep> {
    vec![
        FakeStep::respond(MessageKind::StatusQuery, RESPONSE_DELAY, status_report(0)),
        FakeStep::respond(
            MessageKind::AcquireControl,
            RESPONSE_DELAY,
            acquire_result(1),
        ),
        FakeStep::respond(
            MessageKind::HostCommand,
            RESPONSE_DELAY,
            command_result(0, TimerPwm::ZERO, HostCommandResultCode::AppliedNew),
        ),
    ]
}

fn pending(clock: &FakeClock, pwm: TimerPwm) -> PendingPhysicalCommand {
    let acknowledgement_deadline = clock
        .now()
        .checked_add(Duration::from_millis(200))
        .expect("fixture deadline does not overflow");
    PendingPhysicalCommand::new(pwm, lease(), acknowledgement_deadline)
}

#[test]
fn config_parser_rejects_ambiguous_or_unsafe_boundary_values() {
    let parsed = config();
    assert_eq!(parsed.controller_uid(), uid());
    assert_eq!(parsed.expected_actuator_config_fingerprint(), fingerprint());

    let mut invalid = input();
    invalid.command_endpoint = "192.168.50.2:8080";
    assert!(ClientConfig::parse(invalid).is_err());

    invalid = input();
    invalid.controller_uid_hex = "11111111111111111111111A";
    assert!(ClientConfig::parse(invalid).is_err());

    invalid = input();
    invalid.applied_ack_timeout_ns = "050000000";
    assert!(ClientConfig::parse(invalid).is_err());

    invalid = input();
    invalid.expected_actuator_config_fingerprint_hex = "00000000000000000000000000000000";
    assert!(ClientConfig::parse(invalid).is_err());
}

#[test]
fn exact_status_acquire_zero_motion_and_disarm_path() {
    let clock = FakeClock::default();
    let motion = TimerPwm::try_new(31, -27).expect("valid fixture PWM");
    let mut steps = acquisition_steps();
    steps.push(FakeStep::respond(
        MessageKind::HostCommand,
        RESPONSE_DELAY,
        command_result(1, motion, HostCommandResultCode::AppliedNew),
    ));
    steps.push(FakeStep::respond(
        MessageKind::HostStop,
        RESPONSE_DELAY,
        stop_result(2),
    ));
    let (transport, probe) = FakeTransport::scripted(clock.clone(), steps);
    let client = DisarmedCommandClient::new(transport, clock.clone(), config());

    let (armed, initial_receipt) = client
        .acquire_zero()
        .ok()
        .expect("zero acquisition succeeds");
    assert!(initial_receipt.is_confirmed_zero());
    assert_eq!(initial_receipt.sequence(), V2CommandSequence::FIRST);
    assert_eq!(initial_receipt.result(), HostCommandResultCode::AppliedNew);
    assert_eq!(initial_receipt.output_state(), OutputState::ZeroPwm);
    assert_eq!(
        initial_receipt.controller_applied_at(),
        ControllerUptimeMsWrapping::new(2_000)
    );
    assert_eq!(
        initial_receipt.controller_expires_at(),
        ControllerDeadlineMsWrapping::new(2_100)
    );
    assert!(initial_receipt.controller_faults().is_clear());
    let retained_zero = initial_receipt.verified_host_result();
    assert_eq!(retained_zero.requested_timer_pwm, TimerPwm::ZERO);
    assert_eq!(retained_zero.controller_timer_pwm, TimerPwm::ZERO);
    let (armed, motion_receipt) = armed
        .apply(pending(&clock, motion))
        .ok()
        .expect("motion applies once");
    assert_eq!(motion_receipt.sequence(), V2CommandSequence::new(1));
    assert_eq!(motion_receipt.applied_timer_pwm(), motion);
    assert_eq!(motion_receipt.output_state(), OutputState::NonzeroPwm);
    assert!(!motion_receipt.is_confirmed_zero());
    let (_disarmed, stop_receipt) = armed.disarm().ok().expect("explicit stop is confirmed");
    assert_eq!(stop_receipt.controller_uid(), uid());

    let exchanges = probe.exchanges();
    assert_eq!(exchanges.len(), 5);
    let commands: Vec<_> = exchanges
        .iter()
        .filter_map(|exchange| match exchange.request() {
            Message::HostCommand(command) => Some(command),
            _ => None,
        })
        .collect();
    assert_eq!(commands.len(), 2);
    assert!(commands[0].is_initial_zero_acquisition());
    assert_eq!(commands[1].sequence, V2CommandSequence::new(1));
    assert_eq!(commands[1].requested_timer_pwm, motion);
}

#[test]
fn uncertain_nonzero_is_never_retried_and_latches_after_confirmed_stop() {
    let clock = FakeClock::default();
    let motion = TimerPwm::try_new(40, 35).expect("valid fixture PWM");
    let mut steps = acquisition_steps();
    steps.push(FakeStep::fail(
        MessageKind::HostCommand,
        RESPONSE_DELAY,
        "lost applied result",
    ));
    steps.push(FakeStep::respond(
        MessageKind::HostStop,
        RESPONSE_DELAY,
        stop_result(2),
    ));
    let (transport, probe) = FakeTransport::scripted(clock.clone(), steps);
    let client = DisarmedCommandClient::new(transport, clock.clone(), config());
    let (armed, _) = client
        .acquire_zero()
        .ok()
        .expect("zero acquisition succeeds");

    let failure = match armed.apply(pending(&clock, motion)) {
        Ok(_) => panic!("transport uncertainty must not continue armed"),
        Err(failure) => failure,
    };
    assert!(matches!(failure.cause(), FailureCause::Transport(_)));
    let latched = failure.into_latched();
    assert_eq!(
        latched.stop_knowledge(),
        LatchedStopKnowledge::ConfirmedStop
    );

    let exchanges = probe.exchanges();
    let motion_commands = exchanges
        .iter()
        .filter(|exchange| {
            matches!(
                exchange.request(),
                Message::HostCommand(command) if command.requested_timer_pwm == motion
            )
        })
        .count();
    assert_eq!(
        motion_commands, 1,
        "uncertain nonzero must never be retried"
    );
    assert_eq!(
        exchanges
            .iter()
            .filter(|exchange| matches!(exchange.request(), Message::HostStop(_)))
            .count(),
        1
    );
    assert!(exchanges.iter().any(|exchange| matches!(
        exchange.request(),
        Message::HostStop(stop) if stop.reason == ForceStopReason::TransportFault
    )));
}

#[test]
fn controller_restart_ack_mismatch_permanently_latches() {
    let clock = FakeClock::default();
    let motion = TimerPwm::try_new(20, 20).expect("valid fixture PWM");
    let mut mismatched = match command_result(1, motion, HostCommandResultCode::AppliedNew) {
        Message::HostCommandResult(result) => result,
        _ => unreachable!(),
    };
    mismatched.boot_id = other_boot();
    let mut steps = acquisition_steps();
    steps.push(FakeStep::respond(
        MessageKind::HostCommand,
        RESPONSE_DELAY,
        Message::HostCommandResult(mismatched),
    ));
    steps.push(FakeStep::respond(
        MessageKind::HostStop,
        RESPONSE_DELAY,
        stop_result(2),
    ));
    let (transport, probe) = FakeTransport::scripted(clock.clone(), steps);
    let client = DisarmedCommandClient::new(transport, clock.clone(), config());
    let (armed, _) = client
        .acquire_zero()
        .ok()
        .expect("zero acquisition succeeds");
    let failure = match armed.apply(pending(&clock, motion)) {
        Ok(_) => panic!("boot mismatch must latch"),
        Err(failure) => failure,
    };
    assert!(matches!(
        failure.cause(),
        FailureCause::Evidence(EvidenceError::ControllerBootIdMismatch { .. })
    ));
    assert_eq!(
        failure.into_latched().stop_knowledge(),
        LatchedStopKnowledge::ConfirmedStop
    );
    assert!(probe.exchanges().iter().any(|exchange| matches!(
        exchange.request(),
        Message::HostStop(stop) if stop.reason == ForceStopReason::SessionReset
    )));
}

#[test]
fn applied_pwm_mismatch_latches_without_retrying_motion() {
    let clock = FakeClock::default();
    let requested = TimerPwm::try_new(22, -19).expect("valid fixture PWM");
    let different = TimerPwm::try_new(21, -19).expect("valid fixture PWM");
    let mut mismatched = match command_result(1, requested, HostCommandResultCode::AppliedNew) {
        Message::HostCommandResult(result) => result,
        _ => unreachable!(),
    };
    mismatched.controller_timer_pwm = different;
    let mut steps = acquisition_steps();
    steps.push(FakeStep::respond(
        MessageKind::HostCommand,
        RESPONSE_DELAY,
        Message::HostCommandResult(mismatched),
    ));
    steps.push(FakeStep::respond(
        MessageKind::HostStop,
        RESPONSE_DELAY,
        stop_result(2),
    ));
    let (transport, probe) = FakeTransport::scripted(clock.clone(), steps);
    let client = DisarmedCommandClient::new(transport, clock.clone(), config());
    let (armed, _) = client
        .acquire_zero()
        .ok()
        .expect("zero acquisition succeeds");
    let failure = match armed.apply(pending(&clock, requested)) {
        Ok(_) => panic!("applied PWM mismatch must latch"),
        Err(failure) => failure,
    };
    assert!(matches!(
        failure.cause(),
        FailureCause::Evidence(EvidenceError::ControllerTimerPwmMismatch { .. })
    ));
    assert_eq!(
        probe
            .exchanges()
            .iter()
            .filter(|exchange| matches!(
                exchange.request(),
                Message::HostCommand(command) if command.requested_timer_pwm == requested
            ))
            .count(),
        1
    );
}

#[test]
fn acknowledgement_at_exclusive_deadline_is_rejected() {
    let clock = FakeClock::default();
    let motion = TimerPwm::try_new(15, 10).expect("valid fixture PWM");
    let mut steps = acquisition_steps();
    steps.push(FakeStep::respond(
        MessageKind::HostCommand,
        Duration::from_millis(50),
        command_result(1, motion, HostCommandResultCode::AppliedNew),
    ));
    steps.push(FakeStep::respond(
        MessageKind::HostStop,
        RESPONSE_DELAY,
        stop_result(2),
    ));
    let (transport, _) = FakeTransport::scripted(clock.clone(), steps);
    let client = DisarmedCommandClient::new(transport, clock.clone(), config());
    let (armed, _) = client
        .acquire_zero()
        .ok()
        .expect("zero acquisition succeeds");
    let failure = match armed.apply(pending(&clock, motion)) {
        Ok(_) => panic!("deadline equality must be expired"),
        Err(failure) => failure,
    };
    assert!(matches!(
        failure.cause(),
        FailureCause::ResponseAtOrAfterDeadline { .. }
    ));
}

#[test]
fn acknowledgement_deadline_does_not_truncate_applied_lease_evidence() {
    let clock = FakeClock::default();
    let motion = TimerPwm::try_new(15, 10).expect("valid fixture PWM");
    let mut steps = acquisition_steps();
    steps.push(FakeStep::respond(
        MessageKind::HostCommand,
        RESPONSE_DELAY,
        command_result(1, motion, HostCommandResultCode::AppliedNew),
    ));
    steps.push(FakeStep::respond(
        MessageKind::HostStop,
        RESPONSE_DELAY,
        stop_result(2),
    ));
    let (transport, _) = FakeTransport::scripted(clock.clone(), steps);
    let client = DisarmedCommandClient::new(transport, clock.clone(), config());
    let (armed, _) = client
        .acquire_zero()
        .ok()
        .expect("zero acquisition succeeds");
    let acknowledgement_deadline = clock
        .now()
        .checked_add(Duration::from_millis(10))
        .expect("fixture deadline does not overflow");
    let command = PendingPhysicalCommand::new(motion, lease(), acknowledgement_deadline);

    let (armed, receipt) = armed.apply(command).ok().expect("motion applies in time");
    assert!(receipt.acknowledged_at() < acknowledgement_deadline);
    assert!(receipt.known_active_through_exclusive() > acknowledgement_deadline);
    clock.set_nanos(
        u64::try_from(acknowledgement_deadline.nanos_since_clock_start())
            .expect("fixture timestamp fits u64"),
    );
    let armed = armed
        .require_current_applied_evidence()
        .ok()
        .expect("old admission deadline cannot expire controller lease evidence");
    let _ = armed.disarm().ok().expect("fixture disarms");
}

#[test]
fn expired_previous_applied_evidence_prevents_next_motion_send() {
    let clock = FakeClock::default();
    let motion = TimerPwm::try_new(15, 10).expect("valid fixture PWM");
    let mut steps = acquisition_steps();
    steps.push(FakeStep::respond(
        MessageKind::HostStop,
        RESPONSE_DELAY,
        stop_result(2),
    ));
    let (transport, probe) = FakeTransport::scripted(clock.clone(), steps);
    let client = DisarmedCommandClient::new(transport, clock.clone(), config());
    let (armed, initial) = client
        .acquire_zero()
        .ok()
        .expect("zero acquisition succeeds");
    clock.set_nanos(
        u64::try_from(
            initial
                .known_active_through_exclusive()
                .nanos_since_clock_start(),
        )
        .expect("fixture timestamp fits u64"),
    );
    let failure = match armed.apply(pending(&clock, motion)) {
        Ok(_) => panic!("expired prior receipt must latch"),
        Err(failure) => failure,
    };
    assert!(matches!(
        failure.cause(),
        FailureCause::PreviousAppliedEvidenceExpired { .. }
    ));
    let motion_sends = probe
        .exchanges()
        .iter()
        .filter(|exchange| {
            matches!(
                exchange.request(),
                Message::HostCommand(command) if command.requested_timer_pwm == motion
            )
        })
        .count();
    assert_eq!(motion_sends, 0);
    assert!(probe.exchanges().iter().any(|exchange| matches!(
        exchange.request(),
        Message::HostStop(stop) if stop.reason == ForceStopReason::LeaseExpired
    )));
}

#[test]
fn expired_previous_applied_evidence_prevents_pre_solve_admission() {
    let clock = FakeClock::default();
    let mut steps = acquisition_steps();
    steps.push(FakeStep::respond(
        MessageKind::HostStop,
        RESPONSE_DELAY,
        stop_result(2),
    ));
    let (transport, probe) = FakeTransport::scripted(clock.clone(), steps);
    let client = DisarmedCommandClient::new(transport, clock.clone(), config());
    let (armed, initial) = client
        .acquire_zero()
        .ok()
        .expect("zero acquisition succeeds");
    clock.set_nanos(
        u64::try_from(
            initial
                .known_active_through_exclusive()
                .nanos_since_clock_start(),
        )
        .expect("fixture timestamp fits u64"),
    );

    let failure = match armed.require_current_applied_evidence() {
        Ok(_) => panic!("expired evidence must prevent the next solve"),
        Err(failure) => failure,
    };
    assert!(matches!(
        failure.cause(),
        FailureCause::PreviousAppliedEvidenceExpired { .. }
    ));
    assert_eq!(
        probe
            .exchanges()
            .iter()
            .filter(|exchange| matches!(exchange.request(), Message::HostCommand(_)))
            .count(),
        1,
        "only the acquisition zero may have crossed the command boundary"
    );
    assert!(probe.exchanges().iter().any(|exchange| matches!(
        exchange.request(),
        Message::HostStop(stop) if stop.reason == ForceStopReason::LeaseExpired
    )));
}

#[test]
fn regressed_clock_still_sends_bounded_stops_but_claims_no_receipt() {
    let clock = FakeClock::default();
    let motion = TimerPwm::try_new(10, 10).expect("valid fixture PWM");
    let mut steps = acquisition_steps();
    steps.extend([
        FakeStep::respond(MessageKind::HostStop, RESPONSE_DELAY, stop_result(2)),
        FakeStep::respond(MessageKind::HostStop, RESPONSE_DELAY, stop_result(3)),
        FakeStep::respond(MessageKind::HostStop, RESPONSE_DELAY, stop_result(4)),
    ]);
    let (transport, probe) = FakeTransport::scripted(clock.clone(), steps);
    let client = DisarmedCommandClient::new(transport, clock.clone(), config());
    let (armed, _) = client
        .acquire_zero()
        .ok()
        .expect("zero acquisition succeeds");
    clock.set_nanos(0);
    let failure = match armed.apply(pending(&clock, motion)) {
        Ok(_) => panic!("clock regression must latch"),
        Err(failure) => failure,
    };
    assert!(matches!(
        failure.cause(),
        FailureCause::ClockRegressed { .. }
    ));
    let latched = failure.into_latched();
    assert_eq!(latched.stop_knowledge(), LatchedStopKnowledge::Unconfirmed);
    assert_eq!(latched.recovery().attempts_started(), 3);
    assert_eq!(
        probe
            .exchanges()
            .iter()
            .filter(|exchange| matches!(exchange.request(), Message::HostStop(_)))
            .count(),
        3
    );
}

#[test]
fn disarm_failure_uses_at_most_configured_total_stop_attempts() {
    let clock = FakeClock::default();
    let mut steps = acquisition_steps();
    steps.push(FakeStep::fail(
        MessageKind::HostStop,
        RESPONSE_DELAY,
        "first stop ack lost",
    ));
    steps.push(FakeStep::fail(
        MessageKind::HostStop,
        RESPONSE_DELAY,
        "second stop ack lost",
    ));
    steps.push(FakeStep::respond(
        MessageKind::HostStop,
        RESPONSE_DELAY,
        stop_result(4),
    ));
    let (transport, probe) = FakeTransport::scripted(clock.clone(), steps);
    let client = DisarmedCommandClient::new(transport, clock, config());
    let (armed, _) = client
        .acquire_zero()
        .ok()
        .expect("zero acquisition succeeds");
    let failure = match armed.disarm() {
        Ok(_) => panic!("first uncertain disarm must latch even if recovery later stops"),
        Err(failure) => failure,
    };
    assert_eq!(
        failure.into_latched().stop_knowledge(),
        LatchedStopKnowledge::ConfirmedStop
    );
    assert_eq!(
        probe
            .exchanges()
            .iter()
            .filter(|exchange| matches!(exchange.request(), Message::HostStop(_)))
            .count(),
        3
    );
}

#[test]
fn dropping_armed_client_makes_only_bounded_best_effort_stops() {
    let clock = FakeClock::default();
    let mut steps = acquisition_steps();
    for _ in 0..3 {
        steps.push(FakeStep::fail(
            MessageKind::HostStop,
            RESPONSE_DELAY,
            "drop stop ack lost",
        ));
    }
    let (transport, probe) = FakeTransport::scripted(clock.clone(), steps);
    let client = DisarmedCommandClient::new(transport, clock, config());
    let (armed, _) = client
        .acquire_zero()
        .ok()
        .expect("zero acquisition succeeds");
    drop(armed);
    assert_eq!(
        probe
            .exchanges()
            .iter()
            .filter(|exchange| matches!(exchange.request(), Message::HostStop(_)))
            .count(),
        3
    );
    assert_eq!(probe.remaining_steps(), 0);
}

#[test]
fn property_style_exact_pwm_round_trips_never_change_sign_or_wheel() {
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    for _case in 0..64 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let left = i8::try_from((state % 201) as i16 - 100).expect("bounded PWM fixture");
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let right = i8::try_from((state % 201) as i16 - 100).expect("bounded PWM fixture");
        let pwm = TimerPwm::try_new(left, right).expect("property generator stays in bounds");
        let clock = FakeClock::default();
        let mut steps = acquisition_steps();
        steps.push(FakeStep::respond(
            MessageKind::HostCommand,
            RESPONSE_DELAY,
            command_result(1, pwm, HostCommandResultCode::AppliedNew),
        ));
        steps.push(FakeStep::respond(
            MessageKind::HostStop,
            RESPONSE_DELAY,
            stop_result(2),
        ));
        let (transport, _) = FakeTransport::scripted(clock.clone(), steps);
        let client = DisarmedCommandClient::new(transport, clock.clone(), config());
        let (armed, _) = client
            .acquire_zero()
            .ok()
            .expect("zero acquisition succeeds");
        let (armed, receipt) = armed
            .apply(pending(&clock, pwm))
            .ok()
            .expect("exact generated PWM applies");
        assert_eq!(receipt.applied_timer_pwm().left().get(), left);
        assert_eq!(receipt.applied_timer_pwm().right().get(), right);
        let _ = armed.disarm().ok().expect("property case stops");
    }
}

#[test]
fn canonical_wire_adapter_and_udp_transport_round_trip_v2_only() {
    let server = UdpSocket::bind("127.0.0.1:0").expect("loopback test server binds");
    let server_address = server.local_addr().expect("server has address");
    let server_thread = std::thread::spawn(move || {
        let mut buffer = [0_u8; MAX_RAW_FRAME_BYTES + 1];
        let (received, peer) = server
            .recv_from(&mut buffer)
            .expect("server receives V2 frame");
        let request = robot_protocol::v2::decode_raw_frame(&buffer[..received])
            .expect("server parses canonical V2 frame");
        assert!(matches!(request, Message::StatusQuery(_)));
        let response = RawFrame::encode(status_report(0)).expect("status fixture encodes");
        server
            .send_to(response.as_bytes(), peer)
            .expect("server sends canonical V2 frame");
    });

    let endpoint = server_address
        .to_string()
        .parse()
        .expect("loopback endpoint parses");
    let mut transport = UdpV2Transport::connect_canonical(endpoint).expect("UDP client connects");
    let response = transport
        .exchange_once(
            Message::StatusQuery(robot_protocol::v2::StatusQuery {
                expected_controller_uid: uid(),
                request_id: RequestId::new(0),
            }),
            Duration::from_millis(100),
        )
        .expect("one UDP exchange succeeds");
    assert_eq!(response, status_report(0));
    server_thread.join().expect("loopback server exits cleanly");

    let adapter = RobotProtocolV2WireAdapter;
    let encoded = adapter
        .encode(response)
        .expect("canonical adapter encodes response");
    assert_eq!(
        adapter
            .decode(adapter.encoded_bytes(&encoded))
            .expect("canonical adapter decodes response"),
        response
    );
}
