use std::collections::{BTreeMap, VecDeque};
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use kiko_expression_core::{
    ExpressionIntent, ExpressionKind, ExpressionPriority, FreshnessWindow, MonotonicTimestamp,
    NonZeroDuration, PositiveUnitAmount, ReactionInputs, ReactionMixer,
    UnitAmount as CoreUnitAmount,
};
use kiko_expression_runtime::{
    EyeRenderStyle, EyeSessionFaultKind, PreparedEyeIntent, REQUIRED_EYE_CAPABILITIES,
    adapt_reaction_output,
};
use kiko_eye_protocol::{
    AcquireResult, AcquireResultCode, Capabilities, ControlEpoch, DeviceBootId, DeviceTimestampMs,
    DeviceUid, FirmwareBuildId, HandshakeNonce, IdentityReport, IntentResult, IntentResultCode,
    IntentSequence, MAX_ENCODED_FRAME_BYTES, Message, ReleaseReason, RenderedFrameSequence, decode,
    encode,
};

use crate::{
    ActorTermination, AsyncByteTransport, COMMISSIONING_INTENT_LEASE_MS, CancellationCause,
    CleanupOutcome, EyeRuntimeConfig, EyeSessionMaterial, EyeSessionMaterialError,
    EyeSessionMaterialGenerator, EyeSessionMaterialInput, FrameReadError, FrameWriteFailure,
    HandleRequestError, MonotonicClock, ProtocolExchange, ReleaseReport, RuntimeFaultCause,
    StaticEyeRuntimeConfig, StaticEyeRuntimeConfigInput, TransportFailure, TransportOperation,
    eye_commissioning_steps, spawn_eye_actor as try_spawn_eye_actor,
};

fn spawn_eye_actor<T, C>(
    transport: T,
    clock: C,
    config: EyeRuntimeConfig,
) -> (
    crate::EyeActorHandle,
    crate::StartupReceipt,
    crate::EyeActorTask,
)
where
    T: AsyncByteTransport,
    C: MonotonicClock,
{
    try_spawn_eye_actor(transport, clock, config).expect("test owns an active Tokio runtime")
}

#[derive(Clone, Default)]
struct FakeClock {
    nanos: Arc<AtomicU64>,
}

impl FakeClock {
    fn tick(&self) {
        self.advance(1);
    }

    fn advance(&self, nanoseconds: u64) {
        self.nanos.fetch_add(nanoseconds, Ordering::SeqCst);
    }
}

impl MonotonicClock for FakeClock {
    fn now(&self) -> Result<MonotonicTimestamp, crate::ClockError> {
        Ok(MonotonicTimestamp::from_nanos_since_epoch(
            self.nanos.load(Ordering::SeqCst),
        ))
    }
}

#[derive(Default)]
struct FakeState {
    incoming: VecDeque<u8>,
    written: Vec<u8>,
    write_chunk: usize,
    read_chunk: usize,
    read_advance_ns: u64,
    flush_advance_ns: u64,
    write_calls: usize,
    flush_calls: usize,
    read_calls: usize,
    write_failures: BTreeMap<usize, TransportFailure>,
    flush_failures: BTreeMap<usize, TransportFailure>,
    read_failures: BTreeMap<usize, TransportFailure>,
}

#[derive(Clone)]
struct FakeProbe(Arc<Mutex<FakeState>>);

impl FakeProbe {
    fn written_messages(&self) -> Vec<Message> {
        let state = self.0.lock().expect("fake state lock");
        state
            .written
            .split(|byte| *byte == 0)
            .filter(|record| !record.is_empty())
            .map(|record| decode(record).expect("host emitted valid KEP2"))
            .collect()
    }
}

struct FakeTransport {
    state: Arc<Mutex<FakeState>>,
    clock: FakeClock,
}

impl FakeTransport {
    fn new(incoming: Vec<u8>, clock: FakeClock) -> (Self, FakeProbe) {
        let state = Arc::new(Mutex::new(FakeState {
            incoming: incoming.into(),
            write_chunk: usize::MAX,
            read_chunk: 1,
            read_advance_ns: 1,
            flush_advance_ns: 1,
            ..FakeState::default()
        }));
        (
            Self {
                state: Arc::clone(&state),
                clock,
            },
            FakeProbe(state),
        )
    }
}

impl AsyncByteTransport for FakeTransport {
    async fn write_some(
        &mut self,
        bytes: &[u8],
        _timeout: Duration,
    ) -> Result<usize, TransportFailure> {
        self.clock.tick();
        let mut state = self.state.lock().expect("fake state lock");
        state.write_calls += 1;
        let call = state.write_calls;
        if let Some(source) = state.write_failures.remove(&call) {
            let progress = source.bytes_transferred().min(bytes.len());
            state.written.extend_from_slice(&bytes[..progress]);
            return Err(source);
        }
        let written = bytes.len().min(state.write_chunk);
        state.written.extend_from_slice(&bytes[..written]);
        Ok(written)
    }

    async fn flush(&mut self, _timeout: Duration) -> Result<(), TransportFailure> {
        let mut state = self.state.lock().expect("fake state lock");
        self.clock.advance(state.flush_advance_ns);
        state.flush_calls += 1;
        let call = state.flush_calls;
        match state.flush_failures.remove(&call) {
            Some(source) => Err(source),
            None => Ok(()),
        }
    }

    async fn read_some(
        &mut self,
        bytes: &mut [u8],
        _timeout: Duration,
    ) -> Result<usize, TransportFailure> {
        let mut state = self.state.lock().expect("fake state lock");
        self.clock.advance(state.read_advance_ns);
        state.read_calls += 1;
        let call = state.read_calls;
        if let Some(source) = state.read_failures.remove(&call) {
            return Err(source);
        }
        if state.incoming.is_empty() {
            return Err(TransportFailure::timed_out(TransportOperation::Read, 0));
        }
        let read = bytes.len().min(state.read_chunk).min(state.incoming.len());
        for output in &mut bytes[..read] {
            *output = state.incoming.pop_front().expect("bounded incoming byte");
        }
        Ok(read)
    }
}

fn protocol_nonce(value: u64) -> HandshakeNonce {
    HandshakeNonce::try_new(value).expect("non-zero nonce")
}

#[test]
fn spawning_without_tokio_runtime_is_a_typed_error() {
    let clock = FakeClock::default();
    let (transport, _) = FakeTransport::new(Vec::new(), clock.clone());
    assert!(matches!(
        try_spawn_eye_actor(transport, clock, config()),
        Err(crate::EyeActorSpawnError::NoTokioRuntime { .. })
    ));
}

fn uid(byte: u8) -> DeviceUid {
    DeviceUid::try_new([byte; 16]).expect("non-zero UID")
}

fn build(byte: u8) -> FirmwareBuildId {
    FirmwareBuildId::try_new([byte; 32]).expect("non-zero build")
}

fn boot(value: u64) -> DeviceBootId {
    DeviceBootId::try_new(value).expect("non-zero boot")
}

fn epoch(value: u32) -> ControlEpoch {
    ControlEpoch::try_new(value).expect("non-zero epoch")
}

fn identity_report() -> IdentityReport {
    IdentityReport {
        nonce: protocol_nonce(11),
        device_uid: uid(1),
        firmware_build_id: build(2),
        boot_id: boot(7),
        device_uptime: DeviceTimestampMs::from_millis_since_boot(100),
        capabilities: Capabilities::try_from_bits(REQUIRED_EYE_CAPABILITIES)
            .expect("known capabilities"),
    }
}

fn acquire_result() -> AcquireResult {
    AcquireResult {
        boot_id: boot(7),
        control_epoch: epoch(13),
        nonce: protocol_nonce(12),
        result: AcquireResultCode::Granted,
        device_uptime: DeviceTimestampMs::from_millis_since_boot(110),
    }
}

fn intent_result(
    boot_id: DeviceBootId,
    control_epoch: ControlEpoch,
    sequence: u32,
    code: IntentResultCode,
    at_ms: u64,
    expires_ms: u64,
    rendered_sequence: u32,
) -> IntentResult {
    IntentResult::try_new(
        boot_id,
        control_epoch,
        IntentSequence::new(sequence),
        code,
        DeviceTimestampMs::from_millis_since_boot(at_ms),
        DeviceTimestampMs::from_millis_since_boot(expires_ms),
        RenderedFrameSequence::new(rendered_sequence),
    )
    .expect("valid result timing")
}

fn encode_messages(messages: impl IntoIterator<Item = Message>) -> Vec<u8> {
    let mut bytes = Vec::new();
    for message in messages {
        let mut frame = [0_u8; MAX_ENCODED_FRAME_BYTES];
        let length = encode(message, &mut frame).expect("encode fake firmware frame");
        bytes.extend_from_slice(&frame[..length]);
    }
    bytes
}

fn config() -> EyeRuntimeConfig {
    config_with_lease(100)
}

fn config_with_lease(intent_lease_ms: u16) -> EyeRuntimeConfig {
    let policy = StaticEyeRuntimeConfig::parse(StaticEyeRuntimeConfigInput {
        device_path: "/dev/serial/by-id/kiko-eye-001".to_owned(),
        baud_rate_bps: 115_200,
        response_timeout_ms: 20,
        write_timeout_ms: 5,
        write_attempts: 2,
        empty_delimiter_budget: 2,
        expected_device_uid: [1; 16],
        expected_firmware_build_id: [2; 32],
        expected_capabilities_bits: REQUIRED_EYE_CAPABILITIES,
        intent_lease_ms,
    })
    .expect("test policy");
    policy
        .new_session(&mut FixedSessionMaterial)
        .expect("test session material")
}

struct FixedSessionMaterial;

impl EyeSessionMaterialGenerator for FixedSessionMaterial {
    type Error = EyeSessionMaterialError;

    fn generate(&mut self) -> Result<EyeSessionMaterial, Self::Error> {
        EyeSessionMaterial::parse(EyeSessionMaterialInput {
            identity_nonce: 11,
            acquire_nonce: 12,
            control_epoch: 13,
        })
    }
}

fn prepared(at_ns: u64) -> PreparedEyeIntent {
    let at = MonotonicTimestamp::from_nanos_since_epoch(at_ns);
    let output = ReactionMixer::default().mix(at, ReactionInputs::empty());
    adapt_reaction_output(
        output,
        ExpressionKind::Neutral,
        EyeRenderStyle::new(
            CoreUnitAmount::try_from_basis_points(5_000).expect("brightness"),
            [3, 4, 5],
            false,
        ),
        at,
    )
    .expect("prepared eye intent")
}

fn expiring_prepared(observed_at_ns: u64, ttl_ns: u64) -> PreparedEyeIntent {
    let observed_at = MonotonicTimestamp::from_nanos_since_epoch(observed_at_ns);
    let freshness = FreshnessWindow::from_ttl(
        observed_at,
        NonZeroDuration::try_from_nanos(ttl_ns).expect("positive ttl"),
    )
    .expect("freshness");
    let intent = ExpressionIntent::new(
        ExpressionKind::Attentive,
        PositiveUnitAmount::ONE,
        ExpressionPriority::Normal,
        None,
        freshness,
    );
    let output = ReactionMixer::default().mix(
        observed_at,
        ReactionInputs {
            rgb: None,
            people: &[],
            scene: None,
            intents: &[intent],
        },
    );
    adapt_reaction_output(
        output,
        ExpressionKind::Attentive,
        EyeRenderStyle::new(
            CoreUnitAmount::try_from_basis_points(5_000).expect("brightness"),
            [3, 4, 5],
            false,
        ),
        observed_at,
    )
    .expect("expiring prepared eye intent")
}

fn handshake_messages() -> [Message; 2] {
    [
        Message::IdentityReport(identity_report()),
        Message::AcquireResult(acquire_result()),
    ]
}

fn session_fault_kind(fault: &crate::EyeRuntimeFault) -> EyeSessionFaultKind {
    match fault.cause() {
        RuntimeFaultCause::Session(source) => source.kind(),
        other => panic!("expected session fault, got {other:?}"),
    }
}

#[tokio::test]
async fn partial_io_nominal_handshake_apply_and_release_are_exact() {
    let inbound = encode_messages([
        Message::IdentityReport(identity_report()),
        Message::AcquireResult(acquire_result()),
        Message::IntentResult(intent_result(
            boot(7),
            epoch(13),
            0,
            IntentResultCode::AppliedNew,
            120,
            220,
            1,
        )),
        Message::IntentResult(intent_result(
            boot(7),
            epoch(13),
            1,
            IntentResultCode::Released,
            130,
            130,
            2,
        )),
    ]);
    let clock = FakeClock::default();
    let (transport, probe) = FakeTransport::new(inbound, clock.clone());
    {
        let mut state = probe.0.lock().expect("fake state lock");
        state.write_chunk = 3;
        state.read_chunk = 1;
    }
    let (mut handle, startup, task) = spawn_eye_actor(transport, clock, config());
    let startup = startup
        .wait()
        .await
        .expect("startup receipt")
        .expect("startup");
    assert_eq!(startup.identity(), identity_report());
    assert_eq!(startup.binding().boot_id(), boot(7));

    let admission = handle.apply_intent(prepared(0)).await.expect("admission");
    assert_eq!(admission.admission().sequence(), IntentSequence::new(0));
    assert_eq!(admission.admission().rendered_frame_sequence().get(), 1);

    let release = handle.shutdown().await.expect("shutdown report");
    assert!(matches!(release, ReleaseReport::Released(_)));
    let exit = task.join().await.expect("actor join");
    assert_eq!(exit.termination(), &ActorTermination::RequestedShutdown);
    assert_eq!(exit.admitted_intent_count(), 1);

    let written = probe.written_messages();
    assert_eq!(written.len(), 4);
    assert!(matches!(written[0], Message::IdentityQuery { .. }));
    assert!(matches!(written[1], Message::AcquireControl(_)));
    assert!(matches!(written[2], Message::ApplyIntent(_)));
    assert!(matches!(written[3], Message::ReleaseControl(_)));
}

#[tokio::test]
async fn fixed_commissioning_recipe_acquires_applies_and_releases_consistently() {
    let mut inbound = Vec::from(handshake_messages());
    for sequence in 0..u32::try_from(crate::COMMISSIONING_STEP_COUNT).expect("bounded steps") {
        let applied_at_ms = 120 + u64::from(sequence) * 10;
        inbound.push(Message::IntentResult(intent_result(
            boot(7),
            epoch(13),
            sequence,
            IntentResultCode::AppliedNew,
            applied_at_ms,
            applied_at_ms + u64::from(COMMISSIONING_INTENT_LEASE_MS),
            sequence + 1,
        )));
    }
    inbound.push(Message::IntentResult(intent_result(
        boot(7),
        epoch(13),
        u32::try_from(crate::COMMISSIONING_STEP_COUNT).expect("bounded steps"),
        IntentResultCode::Released,
        180,
        180,
        u32::try_from(crate::COMMISSIONING_STEP_COUNT + 1).expect("bounded steps"),
    )));

    let clock = FakeClock::default();
    let (transport, probe) = FakeTransport::new(encode_messages(inbound), clock.clone());
    let (mut handle, startup, task) = spawn_eye_actor(
        transport,
        clock.clone(),
        config_with_lease(COMMISSIONING_INTENT_LEASE_MS),
    );
    startup.wait().await.expect("receipt").expect("startup");

    let mut expected_intents = Vec::new();
    for step in eye_commissioning_steps() {
        let prepared = step.prepare(clock.now().expect("clock")).expect("recipe");
        expected_intents.push(prepared.intent());
        handle.apply_intent(prepared).await.expect("admission");
    }
    let release = handle.shutdown().await.expect("release report");
    assert!(matches!(release, ReleaseReport::Released(_)));
    let exit = task.join().await.expect("actor join");
    assert_eq!(exit.termination(), &ActorTermination::RequestedShutdown);
    assert_eq!(
        exit.admitted_intent_count(),
        u64::try_from(crate::COMMISSIONING_STEP_COUNT).expect("bounded steps")
    );
    assert_eq!(
        exit.last_admission()
            .expect("last admission")
            .admission()
            .sequence(),
        IntentSequence::new(
            u32::try_from(crate::COMMISSIONING_STEP_COUNT - 1).expect("bounded steps")
        )
    );

    let written = probe.written_messages();
    assert_eq!(written.len(), crate::COMMISSIONING_STEP_COUNT + 3);
    for (index, expected) in expected_intents.into_iter().enumerate() {
        let Message::ApplyIntent(actual) = written[index + 2] else {
            panic!("commissioning write {index} was not an ApplyIntent")
        };
        assert_eq!(
            actual.sequence,
            IntentSequence::new(u32::try_from(index).expect("bounded steps"))
        );
        assert_eq!(actual.lease.get(), COMMISSIONING_INTENT_LEASE_MS);
        assert_eq!(actual.intent, expected);
    }
    assert!(matches!(
        written.last(),
        Some(Message::ReleaseControl(release))
            if release.reason == ReleaseReason::HostShutdown
    ));
}

#[tokio::test]
async fn a_response_record_that_started_before_its_request_is_rejected() {
    let clock = FakeClock::default();
    let (transport, probe) =
        FakeTransport::new(encode_messages(handshake_messages()), clock.clone());
    probe.0.lock().expect("state").read_chunk = crate::MAX_READ_CHUNK_BYTES;
    let (_handle, startup, task) = spawn_eye_actor(transport, clock, config());
    let fault = startup
        .wait()
        .await
        .expect("receipt")
        .expect_err("pre-request response prefix");
    assert!(matches!(
        fault.cause(),
        RuntimeFaultCause::ResponsePredatesRequest {
            exchange: ProtocolExchange::Acquire,
            ..
        }
    ));
    assert!(matches!(
        fault.cleanup(),
        CleanupOutcome::WriteCompleted { .. }
    ));
    let _exit = task.join().await.expect("join");
}

#[tokio::test]
async fn release_response_failure_retains_the_completed_release_attempt() {
    let clock = FakeClock::default();
    let (transport, probe) =
        FakeTransport::new(encode_messages(handshake_messages()), clock.clone());
    let (handle, startup, task) = spawn_eye_actor(transport, clock, config());
    startup.wait().await.expect("receipt").expect("startup");
    let report = handle.shutdown().await.expect("shutdown report");
    let ReleaseReport::Fallback(fault) = report else {
        panic!("missing release response must fallback")
    };
    let prior = fault
        .prior_release_attempt()
        .expect("completed release write retained");
    assert_eq!(prior.write().exchange(), ProtocolExchange::Release);
    assert_eq!(prior.request().reason, ReleaseReason::HostShutdown);
    assert_eq!(
        fault.cleanup(),
        &CleanupOutcome::SessionProvidedNoAdditionalRelease
    );
    assert!(matches!(
        fault.cause(),
        RuntimeFaultCause::Read {
            exchange: ProtocolExchange::Release,
            ..
        }
    ));
    assert!(matches!(
        probe.written_messages().last(),
        Some(Message::ReleaseControl(release))
            if release.reason == ReleaseReason::HostShutdown
    ));
    let _exit = task.join().await.expect("join");
}

#[tokio::test]
async fn bounded_empty_delimiter_noise_is_retained_but_malformed_record_is_fatal() {
    let mut tolerated = vec![0, 0];
    tolerated.extend(encode_messages(handshake_messages()));
    let clock = FakeClock::default();
    let (transport, _probe) = FakeTransport::new(tolerated, clock.clone());
    let (handle, startup, task) = spawn_eye_actor(transport, clock, config());
    assert!(startup.wait().await.expect("receipt").is_ok());
    drop(handle);
    let _exit = task.join().await.expect("join");

    let mut malformed = vec![1, 2, 0];
    malformed.extend(encode_messages(handshake_messages()));
    let clock = FakeClock::default();
    let (transport, _probe) = FakeTransport::new(malformed, clock.clone());
    let (_handle, startup, task) = spawn_eye_actor(transport, clock, config());
    let fault = startup
        .wait()
        .await
        .expect("receipt")
        .expect_err("malformed");
    assert!(matches!(
        fault.cause(),
        RuntimeFaultCause::Read {
            exchange: ProtocolExchange::Identity,
            source: FrameReadError::Malformed { .. }
        }
    ));
    let _exit = task.join().await.expect("join");
}

#[tokio::test]
async fn malformed_active_response_enters_fallback_and_attempts_release() {
    let mut inbound = encode_messages(handshake_messages());
    inbound.extend_from_slice(&[1, 2, 0]);
    let clock = FakeClock::default();
    let (transport, _probe) = FakeTransport::new(inbound, clock.clone());
    let (mut handle, startup, task) = spawn_eye_actor(transport, clock, config());
    startup.wait().await.expect("receipt").expect("startup");
    let fault = match handle.apply_intent(prepared(0)).await {
        Err(HandleRequestError::Runtime(fault)) => fault,
        other => panic!("expected malformed response fault, got {other:?}"),
    };
    assert!(matches!(
        fault.cause(),
        RuntimeFaultCause::Read {
            exchange: ProtocolExchange::Intent,
            source: FrameReadError::Malformed { .. }
        }
    ));
    assert!(matches!(
        fault.cleanup(),
        CleanupOutcome::WriteCompleted { .. }
    ));
    let _exit = task.join().await.expect("join");
}

#[tokio::test]
async fn empty_delimiter_budget_and_oversized_record_fail_explicitly() {
    let clock = FakeClock::default();
    let (transport, _probe) = FakeTransport::new(vec![0, 0, 0], clock.clone());
    let (_handle, startup, task) = spawn_eye_actor(transport, clock, config());
    let fault = startup
        .wait()
        .await
        .expect("receipt")
        .expect_err("noise budget");
    assert!(matches!(
        fault.cause(),
        RuntimeFaultCause::Read {
            source: FrameReadError::EmptyDelimiterBudgetExceeded {
                budget: 2,
                observed: 3
            },
            ..
        }
    ));
    let _exit = task.join().await.expect("join");

    let mut oversized = vec![1; MAX_ENCODED_FRAME_BYTES + 5];
    oversized.push(0);
    let clock = FakeClock::default();
    let (transport, _probe) = FakeTransport::new(oversized, clock.clone());
    let (_handle, startup, task) = spawn_eye_actor(transport, clock, config());
    let fault = startup
        .wait()
        .await
        .expect("receipt")
        .expect_err("oversized");
    assert!(matches!(
        fault.cause(),
        RuntimeFaultCause::Read {
            source: FrameReadError::Malformed { .. },
            ..
        }
    ));
    let _exit = task.join().await.expect("join");
}

#[tokio::test]
async fn response_timeout_is_not_converted_to_absence_or_success() {
    let clock = FakeClock::default();
    let (transport, _probe) = FakeTransport::new(Vec::new(), clock.clone());
    let (_handle, startup, task) = spawn_eye_actor(transport, clock, config());
    let fault = startup.wait().await.expect("receipt").expect_err("timeout");
    assert!(matches!(
        fault.cause(),
        RuntimeFaultCause::Read {
            source: FrameReadError::Transport { source, .. },
            ..
        } if source.kind() == crate::TransportFailureKind::TimedOut
    ));
    assert_eq!(
        fault.cleanup(),
        &CleanupOutcome::SessionProvidedNoAdditionalRelease
    );
    let _exit = task.join().await.expect("join");
}

#[tokio::test]
async fn identity_uid_build_capabilities_and_nonce_are_all_exact() {
    let mut reports = Vec::new();
    let mut wrong_nonce = identity_report();
    wrong_nonce.nonce = protocol_nonce(99);
    reports.push((wrong_nonce, "nonce"));
    let mut wrong_uid = identity_report();
    wrong_uid.device_uid = uid(9);
    reports.push((wrong_uid, "uid"));
    let mut wrong_build = identity_report();
    wrong_build.firmware_build_id = build(9);
    reports.push((wrong_build, "build"));
    let mut wrong_capabilities = identity_report();
    wrong_capabilities.capabilities =
        Capabilities::try_from_bits(REQUIRED_EYE_CAPABILITIES & !Capabilities::BLINK)
            .expect("known reduced capabilities");
    reports.push((wrong_capabilities, "capabilities"));

    for (report, label) in reports {
        let clock = FakeClock::default();
        let (transport, _probe) = FakeTransport::new(
            encode_messages([Message::IdentityReport(report)]),
            clock.clone(),
        );
        let (_handle, startup, task) = spawn_eye_actor(transport, clock, config());
        let fault = startup.wait().await.expect("receipt").unwrap_err();
        assert!(
            matches!(
                session_fault_kind(&fault),
                EyeSessionFaultKind::IdentityNonceMismatch { .. }
                    | EyeSessionFaultKind::DeviceUidMismatch { .. }
                    | EyeSessionFaultKind::FirmwareBuildMismatch { .. }
                    | EyeSessionFaultKind::CapabilityMismatch { .. }
            ),
            "unexpected {label} result"
        );
        assert_eq!(
            fault.cleanup(),
            &CleanupOutcome::SessionProvidedNoAdditionalRelease
        );
        let _exit = task.join().await.expect("join");
    }
}

#[tokio::test]
async fn acquisition_boot_epoch_and_nonce_are_all_exact() {
    let mut results = Vec::new();
    let mut wrong_boot = acquire_result();
    wrong_boot.boot_id = boot(8);
    results.push(wrong_boot);
    let mut wrong_epoch = acquire_result();
    wrong_epoch.control_epoch = epoch(14);
    results.push(wrong_epoch);
    let mut wrong_nonce = acquire_result();
    wrong_nonce.nonce = protocol_nonce(99);
    results.push(wrong_nonce);

    for result in results {
        let clock = FakeClock::default();
        let (transport, _probe) = FakeTransport::new(
            encode_messages([
                Message::IdentityReport(identity_report()),
                Message::AcquireResult(result),
            ]),
            clock.clone(),
        );
        let (_handle, startup, task) = spawn_eye_actor(transport, clock, config());
        let fault = startup
            .wait()
            .await
            .expect("receipt")
            .expect_err("wrong acquire binding");
        assert!(matches!(
            session_fault_kind(&fault),
            EyeSessionFaultKind::DeviceRebooted { .. }
                | EyeSessionFaultKind::AcquireEpochMismatch { .. }
                | EyeSessionFaultKind::AcquireNonceMismatch { .. }
        ));
        assert_eq!(
            fault.cleanup(),
            &CleanupOutcome::SessionProvidedNoAdditionalRelease
        );
        let _exit = task.join().await.expect("join");
    }
}

#[tokio::test]
async fn denied_acquisition_is_not_softened_into_readiness() {
    let mut denied = acquire_result();
    denied.result = AcquireResultCode::Busy;
    let clock = FakeClock::default();
    let (transport, _probe) = FakeTransport::new(
        encode_messages([
            Message::IdentityReport(identity_report()),
            Message::AcquireResult(denied),
        ]),
        clock.clone(),
    );
    let (_handle, startup, task) = spawn_eye_actor(transport, clock, config());
    let fault = startup
        .wait()
        .await
        .expect("receipt")
        .expect_err("busy controller");
    assert!(matches!(
        session_fault_kind(&fault),
        EyeSessionFaultKind::AcquireRejected {
            result: AcquireResultCode::Busy
        }
    ));
    let _exit = task.join().await.expect("join");
}

#[tokio::test]
async fn intent_boot_epoch_and_sequence_mismatch_fail_closed() {
    let results = [
        intent_result(
            boot(8),
            epoch(13),
            0,
            IntentResultCode::AppliedNew,
            120,
            220,
            1,
        ),
        intent_result(
            boot(7),
            epoch(14),
            0,
            IntentResultCode::AppliedNew,
            120,
            220,
            1,
        ),
        intent_result(
            boot(7),
            epoch(13),
            9,
            IntentResultCode::AppliedNew,
            120,
            220,
            1,
        ),
    ];

    for result in results {
        let clock = FakeClock::default();
        let (transport, _probe) = FakeTransport::new(
            encode_messages([
                Message::IdentityReport(identity_report()),
                Message::AcquireResult(acquire_result()),
                Message::IntentResult(result),
            ]),
            clock.clone(),
        );
        let (mut handle, startup, task) = spawn_eye_actor(transport, clock, config());
        startup.wait().await.expect("receipt").expect("startup");
        let fault = match handle.apply_intent(prepared(0)).await {
            Err(HandleRequestError::Runtime(fault)) => fault,
            other => panic!("expected runtime fault, got {other:?}"),
        };
        assert!(matches!(
            session_fault_kind(&fault),
            EyeSessionFaultKind::DeviceRebooted { .. }
                | EyeSessionFaultKind::ResultEpochMismatch { .. }
                | EyeSessionFaultKind::OutOfOrderResult { .. }
        ));
        let _exit = task.join().await.expect("join");
    }
}

#[tokio::test]
async fn stale_and_future_intents_never_reach_firmware() {
    for prepared in [expiring_prepared(0, 1), prepared(1_000_000_000)] {
        let clock = FakeClock::default();
        let (transport, probe) =
            FakeTransport::new(encode_messages(handshake_messages()), clock.clone());
        let (mut handle, startup, task) = spawn_eye_actor(transport, clock, config());
        startup.wait().await.expect("receipt").expect("startup");
        let fault = match handle.apply_intent(prepared).await {
            Err(HandleRequestError::Runtime(fault)) => fault,
            other => panic!("expected freshness fault, got {other:?}"),
        };
        assert!(matches!(
            session_fault_kind(&fault),
            EyeSessionFaultKind::IntentSourceStale { .. }
                | EyeSessionFaultKind::IntentSourceFromFuture { .. }
        ));
        assert!(matches!(
            fault.cleanup(),
            CleanupOutcome::WriteCompleted { .. }
        ));
        assert_eq!(
            probe
                .written_messages()
                .iter()
                .filter(|message| matches!(message, Message::ApplyIntent(_)))
                .count(),
            0
        );
        let _exit = task.join().await.expect("join");
    }
}

#[tokio::test]
async fn source_freshness_is_rechecked_after_the_response_arrives() {
    let inbound = encode_messages([
        Message::IdentityReport(identity_report()),
        Message::AcquireResult(acquire_result()),
        Message::IntentResult(intent_result(
            boot(7),
            epoch(13),
            0,
            IntentResultCode::AppliedNew,
            120,
            220,
            1,
        )),
    ]);
    let clock = FakeClock::default();
    let (transport, probe) = FakeTransport::new(inbound, clock.clone());
    let (mut handle, startup, task) = spawn_eye_actor(transport, clock.clone(), config());
    startup.wait().await.expect("receipt").expect("startup");
    let observed_at_ns = clock.now().expect("clock").nanos_since_epoch();
    probe.0.lock().expect("state").read_advance_ns = 5;

    let fault = match handle
        .apply_intent(expiring_prepared(observed_at_ns, 4))
        .await
    {
        Err(HandleRequestError::Runtime(fault)) => fault,
        other => panic!("expected in-flight freshness fault, got {other:?}"),
    };
    assert!(matches!(
        session_fault_kind(&fault),
        EyeSessionFaultKind::IntentSourceStale { .. }
    ));
    assert!(matches!(
        fault.cleanup(),
        CleanupOutcome::WriteCompleted { .. }
    ));
    let _exit = task.join().await.expect("join");
}

#[tokio::test]
async fn transport_success_after_an_exclusive_deadline_is_still_a_timeout() {
    let clock = FakeClock::default();
    let (transport, probe) = FakeTransport::new(
        encode_messages([Message::IdentityReport(identity_report())]),
        clock.clone(),
    );
    probe.0.lock().expect("state").read_advance_ns = 20_000_000;
    let (_handle, startup, task) = spawn_eye_actor(transport, clock, config());
    let fault = startup
        .wait()
        .await
        .expect("receipt")
        .expect_err("late read");
    assert!(matches!(
        fault.cause(),
        RuntimeFaultCause::Read {
            source: FrameReadError::Transport { source, .. },
            ..
        } if source.kind() == crate::TransportFailureKind::TimedOut
    ));
    let _exit = task.join().await.expect("join");

    let clock = FakeClock::default();
    let (transport, probe) = FakeTransport::new(Vec::new(), clock.clone());
    probe.0.lock().expect("state").flush_advance_ns = 5_000_000;
    let (_handle, startup, task) = spawn_eye_actor(transport, clock, config());
    let fault = startup
        .wait()
        .await
        .expect("receipt")
        .expect_err("late flush");
    let RuntimeFaultCause::Write(source) = fault.cause() else {
        panic!("expected write timeout")
    };
    assert!(source.transmission_uncertain());
    assert!(matches!(
        source.source(),
        FrameWriteFailure::Transport(failure)
            if failure.kind() == crate::TransportFailureKind::TimedOut
                && failure.operation() == TransportOperation::Flush
    ));
    let _exit = task.join().await.expect("join");
}

#[tokio::test]
async fn device_reboot_after_one_admission_is_detected() {
    let clock = FakeClock::default();
    let inbound = encode_messages([
        Message::IdentityReport(identity_report()),
        Message::AcquireResult(acquire_result()),
        Message::IntentResult(intent_result(
            boot(7),
            epoch(13),
            0,
            IntentResultCode::AppliedNew,
            120,
            220,
            1,
        )),
        Message::IntentResult(intent_result(
            boot(8),
            epoch(13),
            1,
            IntentResultCode::AppliedNew,
            130,
            230,
            2,
        )),
    ]);
    let (transport, _probe) = FakeTransport::new(inbound, clock.clone());
    let (mut handle, startup, task) = spawn_eye_actor(transport, clock, config());
    startup.wait().await.expect("receipt").expect("startup");
    handle
        .apply_intent(prepared(0))
        .await
        .expect("first admission");
    let fault = match handle.apply_intent(prepared(0)).await {
        Err(HandleRequestError::Runtime(fault)) => fault,
        other => panic!("expected reboot fault, got {other:?}"),
    };
    assert!(matches!(
        session_fault_kind(&fault),
        EyeSessionFaultKind::DeviceRebooted { .. }
    ));
    assert_eq!(
        fault.cleanup(),
        &CleanupOutcome::SessionProvidedNoAdditionalRelease
    );
    let exit = task.join().await.expect("join");
    assert_eq!(exit.admitted_intent_count(), 1);
}

#[tokio::test]
async fn zero_progress_interrupt_retries_but_partial_progress_never_does() {
    let clock = FakeClock::default();
    let (transport, probe) =
        FakeTransport::new(encode_messages(handshake_messages()), clock.clone());
    probe.0.lock().expect("state").write_failures.insert(
        1,
        TransportFailure::from_io(
            TransportOperation::Write,
            &io::Error::from(io::ErrorKind::Interrupted),
            0,
        ),
    );
    let (handle, startup, task) = spawn_eye_actor(transport, clock, config());
    let startup = startup
        .wait()
        .await
        .expect("receipt")
        .expect("retry startup");
    assert_eq!(startup.identity_query_write().attempts_used(), 2);
    assert_eq!(startup.identity_query_write().recovered_failures().len(), 1);
    drop(handle);
    let _exit = task.join().await.expect("join");

    let clock = FakeClock::default();
    let (transport, probe) = FakeTransport::new(Vec::new(), clock.clone());
    {
        let mut state = probe.0.lock().expect("state");
        state.write_chunk = 3;
        state.write_failures.insert(
            2,
            TransportFailure::from_io(
                TransportOperation::Write,
                &io::Error::from(io::ErrorKind::Interrupted),
                0,
            ),
        );
    }
    let (_handle, startup, task) = spawn_eye_actor(transport, clock, config());
    let fault = startup
        .wait()
        .await
        .expect("receipt")
        .expect_err("partial write");
    let RuntimeFaultCause::Write(source) = fault.cause() else {
        panic!("expected write fault")
    };
    assert_eq!(source.attempts_used(), 1);
    assert!(source.transmission_uncertain());
    let _exit = task.join().await.expect("join");
}

#[tokio::test]
async fn failed_flush_is_reported_as_uncertain_and_not_retried() {
    let clock = FakeClock::default();
    let (transport, probe) = FakeTransport::new(Vec::new(), clock.clone());
    probe.0.lock().expect("state").flush_failures.insert(
        1,
        TransportFailure::from_io(
            TransportOperation::Flush,
            &io::Error::from(io::ErrorKind::BrokenPipe),
            0,
        ),
    );
    let (_handle, startup, task) = spawn_eye_actor(transport, clock, config());
    let fault = startup.wait().await.expect("receipt").expect_err("flush");
    let RuntimeFaultCause::Write(source) = fault.cause() else {
        panic!("expected write fault")
    };
    assert_eq!(source.attempts_used(), 1);
    assert!(source.transmission_uncertain());
    assert!(matches!(
        source.source(),
        FrameWriteFailure::Transport(failure)
            if failure.operation() == TransportOperation::Flush
    ));
    let _exit = task.join().await.expect("join");
}

#[tokio::test]
async fn handle_drop_and_requested_cancel_both_fallback_and_release_best_effort() {
    for requested in [false, true] {
        let clock = FakeClock::default();
        let (transport, probe) =
            FakeTransport::new(encode_messages(handshake_messages()), clock.clone());
        let (handle, startup, task) = spawn_eye_actor(transport, clock, config());
        startup.wait().await.expect("receipt").expect("startup");
        if requested {
            let fault = handle.cancel().await.expect("cancel report");
            assert!(matches!(
                fault.cleanup(),
                CleanupOutcome::WriteCompleted { .. }
            ));
        } else {
            drop(handle);
        }
        let exit = task.join().await.expect("join");
        assert_eq!(
            exit.termination(),
            &ActorTermination::Cancellation(if requested {
                CancellationCause::Requested
            } else {
                CancellationCause::HandleDropped
            })
        );
        assert!(matches!(
            exit.release(),
            Some(ReleaseReport::Fallback(fault))
                if matches!(fault.cleanup(), CleanupOutcome::WriteCompleted { .. })
        ));
        assert!(matches!(
            probe.written_messages().last(),
            Some(Message::ReleaseControl(release)) if release.reason == ReleaseReason::Fault
        ));
    }
}

#[tokio::test]
async fn handle_dropped_before_startup_never_opens_a_protocol_session() {
    let clock = FakeClock::default();
    let (transport, probe) =
        FakeTransport::new(encode_messages(handshake_messages()), clock.clone());
    let (handle, startup, task) = spawn_eye_actor(transport, clock, config());
    drop(handle);
    let fault = startup
        .wait()
        .await
        .expect("receipt")
        .expect_err("owner dropped");
    assert!(matches!(
        fault.cause(),
        RuntimeFaultCause::Cancellation(CancellationCause::HandleDropped)
    ));
    assert_eq!(
        fault.cleanup(),
        &CleanupOutcome::SessionProvidedNoAdditionalRelease
    );
    let exit = task.join().await.expect("join");
    assert_eq!(
        exit.termination(),
        &ActorTermination::Cancellation(CancellationCause::HandleDropped)
    );
    assert!(probe.written_messages().is_empty());
}

#[tokio::test]
async fn cleanup_reports_no_release_completed_write_and_failed_write_distinctly() {
    let clock = FakeClock::default();
    let wrong = {
        let mut report = identity_report();
        report.device_uid = uid(9);
        report
    };
    let (transport, _probe) = FakeTransport::new(
        encode_messages([Message::IdentityReport(wrong)]),
        clock.clone(),
    );
    let (_handle, startup, task) = spawn_eye_actor(transport, clock, config());
    let fault = startup
        .wait()
        .await
        .expect("receipt")
        .expect_err("identity");
    assert_eq!(
        fault.cleanup(),
        &CleanupOutcome::SessionProvidedNoAdditionalRelease
    );
    let _exit = task.join().await.expect("join");

    let clock = FakeClock::default();
    let (transport, probe) =
        FakeTransport::new(encode_messages(handshake_messages()), clock.clone());
    probe.0.lock().expect("state").write_failures.insert(
        3,
        TransportFailure::from_io(
            TransportOperation::Write,
            &io::Error::from(io::ErrorKind::PermissionDenied),
            0,
        ),
    );
    let (handle, startup, task) = spawn_eye_actor(transport, clock, config());
    startup.wait().await.expect("receipt").expect("startup");
    drop(handle);
    let exit = task.join().await.expect("join");
    assert!(matches!(
        exit.release(),
        Some(ReleaseReport::Fallback(fault))
            if matches!(fault.cleanup(), CleanupOutcome::WriteFailed { .. })
    ));
}

#[tokio::test]
async fn rejected_intent_preserves_exact_session_error_and_cleanup() {
    let inbound = encode_messages([
        Message::IdentityReport(identity_report()),
        Message::AcquireResult(acquire_result()),
        Message::IntentResult(intent_result(
            boot(7),
            epoch(13),
            0,
            IntentResultCode::RejectedDomain,
            120,
            120,
            1,
        )),
    ]);
    let clock = FakeClock::default();
    let (transport, _probe) = FakeTransport::new(inbound, clock.clone());
    let (mut handle, startup, task) = spawn_eye_actor(transport, clock, config());
    startup.wait().await.expect("receipt").expect("startup");
    let fault = match handle.apply_intent(prepared(0)).await {
        Err(HandleRequestError::Runtime(fault)) => fault,
        other => panic!("expected rejection, got {other:?}"),
    };
    assert!(matches!(
        session_fault_kind(&fault),
        EyeSessionFaultKind::IntentRejected {
            result: IntentResultCode::RejectedDomain,
            ..
        }
    ));
    assert!(matches!(
        fault.cleanup(),
        CleanupOutcome::WriteCompleted { .. }
    ));
    let _exit = task.join().await.expect("join");
}

#[tokio::test]
async fn firmware_admission_lease_must_equal_the_requested_lease() {
    let inbound = encode_messages([
        Message::IdentityReport(identity_report()),
        Message::AcquireResult(acquire_result()),
        Message::IntentResult(intent_result(
            boot(7),
            epoch(13),
            0,
            IntentResultCode::AppliedNew,
            120,
            219,
            1,
        )),
    ]);
    let clock = FakeClock::default();
    let (transport, _probe) = FakeTransport::new(inbound, clock.clone());
    let (mut handle, startup, task) = spawn_eye_actor(transport, clock, config());
    startup.wait().await.expect("receipt").expect("startup");
    let fault = match handle.apply_intent(prepared(0)).await {
        Err(HandleRequestError::Runtime(fault)) => fault,
        other => panic!("expected lease mismatch, got {other:?}"),
    };
    assert!(matches!(
        session_fault_kind(&fault),
        EyeSessionFaultKind::LeaseMismatch {
            expected_ms: 100,
            actual_ms: 99
        }
    ));
    assert!(matches!(
        fault.cleanup(),
        CleanupOutcome::WriteCompleted { .. }
    ));
    let _exit = task.join().await.expect("join");
}
