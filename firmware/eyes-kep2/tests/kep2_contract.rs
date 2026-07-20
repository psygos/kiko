use kiko_eye_protocol::{
    AcquireControl, AcquireResultCode, ApplyIntent, ControlEpoch, DeviceBootId, DeviceTimestampMs,
    Expression, EyeFlags, EyeIntent, HandshakeNonce, IntentLeaseMs, IntentResultCode,
    IntentSequence, MAX_ENCODED_FRAME_BYTES, Message, ReleaseControl, ReleaseReason, SignedUnit,
    UnitAmount, decode, encode,
};
use kiko_eyes_kep2_firmware::{
    Controller, ControllerError, EndpointEvent, FallbackCause, FirmwareIdentity, Kep2Endpoint,
    OutputState, SUPPORTED_CAPABILITIES_BITS,
};

const BOOT_ID: u64 = 0x1020_3040_5060_7080;
const EPOCH: u32 = 41;

fn timestamp(milliseconds: u64) -> DeviceTimestampMs {
    DeviceTimestampMs::from_millis_since_boot(milliseconds)
}

fn identity() -> FirmwareIdentity {
    FirmwareIdentity::try_new([0x11; 16], [0x22; 32], BOOT_ID).unwrap()
}

fn boot_id() -> DeviceBootId {
    DeviceBootId::try_new(BOOT_ID).unwrap()
}

fn epoch() -> ControlEpoch {
    ControlEpoch::try_new(EPOCH).unwrap()
}

fn nonce(value: u64) -> HandshakeNonce {
    HandshakeNonce::try_new(value).unwrap()
}

fn intent(gaze_x: i16, brightness: u16, blink: bool) -> EyeIntent {
    EyeIntent::new(
        SignedUnit::try_new(gaze_x).unwrap(),
        SignedUnit::ZERO,
        UnitAmount::try_new(100).unwrap(),
        UnitAmount::try_new(500).unwrap(),
        UnitAmount::try_new(brightness).unwrap(),
        Expression::Curious,
        EyeFlags::try_from_bits(if blink { EyeFlags::BLINK } else { 0 }).unwrap(),
        [0x10, 0x80, 0x20],
    )
}

fn acquire(controller: &mut Controller, now_ms: u64) -> Message {
    controller
        .handle_received(
            Message::AcquireControl(AcquireControl {
                expected_boot_id: boot_id(),
                requested_epoch: epoch(),
                nonce: nonce(9),
            }),
            timestamp(now_ms),
            timestamp(now_ms),
        )
        .unwrap()
}

fn command(sequence: u32, lease_ms: u16, gaze_x: i16) -> ApplyIntent {
    ApplyIntent {
        boot_id: boot_id(),
        control_epoch: epoch(),
        sequence: IntentSequence::new(sequence),
        lease: IntentLeaseMs::try_new(lease_ms).unwrap(),
        intent: intent(gaze_x, 800, false),
    }
}

fn intent_result(message: Message) -> kiko_eye_protocol::IntentResult {
    let Message::IntentResult(result) = message else {
        panic!("expected IntentResult, got {message:?}");
    };
    result
}

#[test]
fn identity_round_trips_through_the_canonical_stream_codec() {
    let mut endpoint = Kep2Endpoint::new(identity());
    let query = Message::IdentityQuery { nonce: nonce(7) };
    let mut encoded = [0_u8; MAX_ENCODED_FRAME_BYTES];
    let length = encode(query, &mut encoded).unwrap();

    let mut response = None;
    for byte in &encoded[..length] {
        match endpoint.push(*byte, timestamp(123), timestamp(123)) {
            EndpointEvent::Pending => {}
            EndpointEvent::Response(frame) => response = Some(frame),
            event => panic!("unexpected endpoint event: {event:?}"),
        }
    }
    let response = response.expect("delimiter produced one response");
    assert_eq!(response.as_bytes().last(), Some(&0));
    let report = decode(&response.as_bytes()[..response.len() - 1]).unwrap();
    let Message::IdentityReport(report) = report else {
        panic!("expected identity report");
    };
    assert_eq!(report.nonce, nonce(7));
    assert_eq!(report.device_uid, identity().device_uid());
    assert_eq!(report.firmware_build_id, identity().firmware_build_id());
    assert_eq!(report.boot_id, boot_id());
    assert_eq!(report.device_uptime, timestamp(123));
    assert_eq!(report.capabilities.bits(), SUPPORTED_CAPABILITIES_BITS);
}

#[test]
fn endpoint_does_not_refresh_a_frame_delayed_after_usb_receipt() {
    let mut endpoint = Kep2Endpoint::new(identity());
    acquire(endpoint.controller_mut(), 0);
    let mut encoded = [0_u8; MAX_ENCODED_FRAME_BYTES];
    let length = encode(Message::ApplyIntent(command(0, 20, 0)), &mut encoded).unwrap();

    let mut response = None;
    for byte in &encoded[..length] {
        match endpoint.push(*byte, timestamp(100), timestamp(120)) {
            EndpointEvent::Pending => {}
            EndpointEvent::Response(frame) => response = Some(frame),
            event => panic!("unexpected endpoint event: {event:?}"),
        }
    }
    let response = response.expect("delimiter produced one response");
    let result = intent_result(
        decode(&response.as_bytes()[..response.len() - 1]).expect("valid response frame"),
    );
    assert_eq!(result.result(), IntentResultCode::RejectedExpired);
    assert_eq!(
        endpoint.controller().expected_sequence(),
        Some(IntentSequence::FIRST)
    );
}

#[test]
fn acquisition_is_exclusive_and_exact_retry_does_not_extend_deadline() {
    let mut controller = Controller::new(identity());
    let Message::AcquireResult(granted) = acquire(&mut controller, 10) else {
        panic!("expected acquire result");
    };
    assert_eq!(granted.result, AcquireResultCode::Granted);

    let Message::AcquireResult(retried) = acquire(&mut controller, 1_000) else {
        panic!("expected acquire result");
    };
    assert_eq!(retried.result, AcquireResultCode::Granted);

    let Message::AcquireResult(busy) = controller
        .handle_received(
            Message::AcquireControl(AcquireControl {
                expected_boot_id: boot_id(),
                requested_epoch: ControlEpoch::try_new(EPOCH + 1).unwrap(),
                nonce: nonce(10),
            }),
            timestamp(1_001),
            timestamp(1_001),
        )
        .unwrap()
    else {
        panic!("expected acquire result");
    };
    assert_eq!(busy.result, AcquireResultCode::Busy);

    controller.poll(timestamp(2_010)).unwrap();
    assert!(!controller.is_owned());
    assert_eq!(
        controller.output(),
        OutputState::Autonomous {
            cause: FallbackCause::LeaseExpired
        }
    );
}

#[test]
fn admission_uses_handling_time_and_expires_at_the_exclusive_deadline() {
    let mut controller = Controller::new(identity());
    acquire(&mut controller, 0);
    let result = intent_result(
        controller
            .handle_received(
                Message::ApplyIntent(command(0, 100, 250)),
                timestamp(20),
                timestamp(25),
            )
            .unwrap(),
    );
    assert_eq!(result.result(), IntentResultCode::AppliedNew);
    assert_eq!(result.applied_at(), timestamp(25));
    assert_eq!(result.expires_at(), timestamp(125));
    assert_eq!(result.device_interval_ms(), 100);
    assert_eq!(controller.expected_sequence(), Some(IntentSequence::new(1)));

    assert!(matches!(
        controller.output_at(timestamp(124)).unwrap(),
        OutputState::Commanded { .. }
    ));
    assert_eq!(
        controller.output_at(timestamp(125)).unwrap(),
        OutputState::Autonomous {
            cause: FallbackCause::LeaseExpired
        }
    );
    assert!(!controller.is_owned());
}

#[test]
fn a_queued_command_is_expired_before_admission() {
    let mut controller = Controller::new(identity());
    acquire(&mut controller, 0);
    let result = intent_result(
        controller
            .handle_received(
                Message::ApplyIntent(command(0, 20, 0)),
                timestamp(100),
                timestamp(120),
            )
            .unwrap(),
    );
    assert_eq!(result.result(), IntentResultCode::RejectedExpired);
    assert_eq!(result.applied_at(), timestamp(120));
    assert_eq!(result.expires_at(), timestamp(120));
    assert_eq!(controller.expected_sequence(), Some(IntentSequence::FIRST));
}

#[test]
fn exact_duplicate_is_cached_without_rerender_or_lease_extension() {
    let mut controller = Controller::new(identity());
    acquire(&mut controller, 0);
    let apply = command(0, 200, 300);
    let first = intent_result(
        controller
            .handle_received(Message::ApplyIntent(apply), timestamp(10), timestamp(10))
            .unwrap(),
    );
    let duplicate = intent_result(
        controller
            .handle_received(Message::ApplyIntent(apply), timestamp(50), timestamp(50))
            .unwrap(),
    );
    assert_eq!(duplicate.result(), IntentResultCode::DuplicateCached);
    assert_eq!(duplicate.applied_at(), first.applied_at());
    assert_eq!(duplicate.expires_at(), first.expires_at());
    assert_eq!(
        duplicate.rendered_frame_sequence(),
        first.rendered_frame_sequence()
    );
    assert_eq!(controller.expected_sequence(), Some(IntentSequence::new(1)));
}

#[test]
fn reused_sequence_with_different_content_is_rejected() {
    let mut controller = Controller::new(identity());
    acquire(&mut controller, 0);
    controller
        .handle_received(
            Message::ApplyIntent(command(0, 200, 100)),
            timestamp(10),
            timestamp(10),
        )
        .unwrap();
    let result = intent_result(
        controller
            .handle_received(
                Message::ApplyIntent(command(0, 200, -100)),
                timestamp(20),
                timestamp(20),
            )
            .unwrap(),
    );
    assert_eq!(result.result(), IntentResultCode::RejectedSequence);
    assert_eq!(controller.expected_sequence(), Some(IntentSequence::new(1)));
}

#[test]
fn graceful_release_requires_the_exact_next_sequence() {
    let mut controller = Controller::new(identity());
    acquire(&mut controller, 0);
    controller
        .handle_received(
            Message::ApplyIntent(command(0, 200, 0)),
            timestamp(10),
            timestamp(10),
        )
        .unwrap();

    let wrong = intent_result(
        controller
            .handle_received(
                Message::ReleaseControl(ReleaseControl {
                    boot_id: boot_id(),
                    control_epoch: epoch(),
                    sequence: IntentSequence::FIRST,
                    reason: ReleaseReason::HostShutdown,
                }),
                timestamp(20),
                timestamp(20),
            )
            .unwrap(),
    );
    assert_eq!(wrong.result(), IntentResultCode::RejectedSequence);
    assert!(controller.is_owned());

    let released = intent_result(
        controller
            .handle_received(
                Message::ReleaseControl(ReleaseControl {
                    boot_id: boot_id(),
                    control_epoch: epoch(),
                    sequence: IntentSequence::new(1),
                    reason: ReleaseReason::HostShutdown,
                }),
                timestamp(21),
                timestamp(21),
            )
            .unwrap(),
    );
    assert_eq!(released.result(), IntentResultCode::Released);
    assert_eq!(released.applied_at(), released.expires_at());
    assert!(!controller.is_owned());
    assert_eq!(
        controller.output(),
        OutputState::Autonomous {
            cause: FallbackCause::Released(ReleaseReason::HostShutdown)
        }
    );
}

#[test]
fn malformed_record_and_disconnect_both_relinquish_control() {
    let mut endpoint = Kep2Endpoint::new(identity());
    acquire(endpoint.controller_mut(), 0);
    assert!(endpoint.controller().is_owned());
    assert!(matches!(
        endpoint.push(0, timestamp(1), timestamp(1)),
        EndpointEvent::Dropped(_)
    ));
    assert!(!endpoint.controller().is_owned());
    assert_eq!(
        endpoint.controller().output(),
        OutputState::Autonomous {
            cause: FallbackCause::MalformedFrame
        }
    );

    acquire(endpoint.controller_mut(), 2);
    endpoint.on_disconnect(timestamp(3)).unwrap();
    assert!(!endpoint.controller().is_owned());
    assert_eq!(
        endpoint.controller().output(),
        OutputState::Autonomous {
            cause: FallbackCause::Disconnected
        }
    );
}

#[test]
fn clock_regression_latches_fault_and_future_acquire_reports_faulted() {
    let mut controller = Controller::new(identity());
    acquire(&mut controller, 100);
    assert_eq!(
        controller.poll(timestamp(99)),
        Err(ControllerError::ClockRegressed {
            previous_ms: 100,
            actual_ms: 99
        })
    );
    assert!(controller.is_faulted());
    assert!(!controller.is_owned());

    let Message::AcquireResult(result) = controller
        .handle_received(
            Message::AcquireControl(AcquireControl {
                expected_boot_id: boot_id(),
                requested_epoch: epoch(),
                nonce: nonce(20),
            }),
            timestamp(101),
            timestamp(101),
        )
        .unwrap()
    else {
        panic!("expected acquire result");
    };
    assert_eq!(result.result, AcquireResultCode::Faulted);
}

#[test]
fn startup_rejects_every_zero_identity_component() {
    assert!(FirmwareIdentity::try_new([0; 16], [0x22; 32], BOOT_ID).is_err());
    assert!(FirmwareIdentity::try_new([0x11; 16], [0; 32], BOOT_ID).is_err());
    assert!(FirmwareIdentity::try_new([0x11; 16], [0x22; 32], 0).is_err());
}

#[test]
fn acquisition_for_a_different_boot_is_rejected_without_ownership() {
    let mut controller = Controller::new(identity());
    let Message::AcquireResult(result) = controller
        .handle_received(
            Message::AcquireControl(AcquireControl {
                expected_boot_id: DeviceBootId::try_new(BOOT_ID + 1).unwrap(),
                requested_epoch: epoch(),
                nonce: nonce(30),
            }),
            timestamp(1),
            timestamp(1),
        )
        .unwrap()
    else {
        panic!("expected acquire result");
    };
    assert_eq!(result.boot_id, boot_id());
    assert_eq!(result.result, AcquireResultCode::IdentityMismatch);
    assert!(!controller.is_owned());
}

#[test]
fn a_device_direction_message_on_the_host_boundary_fails_closed() {
    let mut controller = Controller::new(identity());
    acquire(&mut controller, 0);
    let unexpected = Message::AcquireResult(kiko_eye_protocol::AcquireResult {
        boot_id: boot_id(),
        control_epoch: epoch(),
        nonce: nonce(9),
        result: AcquireResultCode::Granted,
        device_uptime: timestamp(1),
    });
    assert!(matches!(
        controller.handle_received(unexpected, timestamp(1), timestamp(1)),
        Err(ControllerError::UnexpectedInbound(_))
    ));
    assert!(!controller.is_owned());
    assert_eq!(
        controller.output(),
        OutputState::Autonomous {
            cause: FallbackCause::ProtocolViolation
        }
    );
}

#[test]
fn an_unrepresentable_acquisition_deadline_latches_internal_fallback() {
    let mut controller = Controller::new(identity());
    let now = timestamp(u64::MAX - 1_000);
    assert_eq!(
        controller.handle_received(
            Message::AcquireControl(AcquireControl {
                expected_boot_id: boot_id(),
                requested_epoch: epoch(),
                nonce: nonce(40),
            }),
            now,
            now,
        ),
        Err(ControllerError::DeadlineOverflow {
            now_ms: u64::MAX - 1_000,
            duration_ms: 2_000,
        })
    );
    assert!(controller.is_faulted());
    assert_eq!(
        controller.output(),
        OutputState::Autonomous {
            cause: FallbackCause::InternalFault
        }
    );
}
