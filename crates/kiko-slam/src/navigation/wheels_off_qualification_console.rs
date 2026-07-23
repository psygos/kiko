//! Qualification-only manual-control boundary.
//!
//! This module deliberately does not reuse production body-frame velocity
//! commands. A qualification request names the two signed timer-duty
//! percentages exactly, and its types cannot be converted into
//! `AgentControlCommandV1`, `ManualDriveOutput`, or `LiveMotionActuationPort`.
//! The sole consumer must translate [`WheelsOffQualificationIngressEvent`]
//! directly into the separately admitted candidate-controller path.
//!
//! Browser operators and agents use the same session table and the same
//! authority arbiter. Manual release, deadman expiry, connection loss, global
//! stop, and the one-way software safety stop all clear queued duty and create
//! a typed terminal-stop event. No new authority can be acquired until the
//! sole consumer reports an observed applied zero for that stop barrier.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::num::{NonZeroU8, NonZeroU64};
use std::sync::{Arc, Mutex, MutexGuard};

use serde::{Deserialize, Serialize, Serializer};

use super::{
    ConsoleAppliedReceipt, ConsoleIdempotencyKey, ConsoleSessionCapability, ConsoleSessionId,
    ConsoleSourceKind, ConsoleSourceSequence,
};
use crate::HostMonotonicTimestamp;

pub const WHEELS_OFF_QUALIFICATION_SCHEMA_V1: u32 = 1;
pub const WHEELS_OFF_QUALIFICATION_PROFILE_KIND: &str = "wheels_off_raw_timer_pwm_qualification";
pub const WHEELS_OFF_QUALIFICATION_INTENT_ENDPOINT: &str =
    "/api/v1/wheels-off-qualification/intents";
pub const WHEELS_OFF_QUALIFICATION_BANNER: &str =
    "WHEELS-OFF QUALIFICATION — RAW TIMER DUTY ONLY — AUTONOMOUS ACTUATION DISABLED";
pub const MAX_WHEELS_OFF_QUALIFICATION_SESSIONS: usize = 32;
pub const MAX_WHEELS_OFF_QUALIFICATION_IDEMPOTENCY_RECORDS: usize = 64;
const MAX_DEADMAN_MILLISECONDS: u64 = 5_000;

fn serialize_u64_as_decimal_string<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&value.to_string())
}

fn serialize_nonzero_u64_as_decimal_string<S>(
    value: &NonZeroU64,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(&value.get().to_string())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WheelsOffQualificationProfileKind {
    WheelsOffRawTimerPwmQualification,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WheelsOffQualificationCommandUnits {
    SignedTimerDutyPercent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WheelsOffQualificationRequiredWheelState {
    Removed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WheelsOffQualificationAutonomousActuation {
    DisabledShadowOnly,
}

/// One explicit raw test pattern.
///
/// Pattern names describe electrical sign only. They make no claim about
/// forward travel or yaw direction because those plant signs are not known
/// until physical wheel calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct WheelsOffQualificationPattern {
    pub left_timer_pwm_percent: i8,
    pub right_timer_pwm_percent: i8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct WheelsOffQualificationPatterns {
    pub both_positive: WheelsOffQualificationPattern,
    pub both_negative: WheelsOffQualificationPattern,
    pub left_negative_right_positive: WheelsOffQualificationPattern,
    pub left_positive_right_negative: WheelsOffQualificationPattern,
}

/// Immutable, serialized admission facts for the qualification UI.
///
/// All fields are private and the only constructor checks their relationships.
/// A running console therefore cannot silently change units, command bounds,
/// deadman semantics, or autonomous-output policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct WheelsOffQualificationControlProfile {
    kind: WheelsOffQualificationProfileKind,
    banner: &'static str,
    command_units: WheelsOffQualificationCommandUnits,
    required_wheel_state: WheelsOffQualificationRequiredWheelState,
    autonomous_actuation: WheelsOffQualificationAutonomousActuation,
    intent_endpoint: &'static str,
    maximum_abs_timer_pwm_percent: NonZeroU8,
    manual_test_magnitude_timer_pwm_percent: NonZeroU8,
    manual_deadman_ms: NonZeroU64,
    patterns: WheelsOffQualificationPatterns,
}

impl WheelsOffQualificationControlProfile {
    pub fn parse(
        maximum_abs_timer_pwm_percent: u8,
        manual_test_magnitude_timer_pwm_percent: u8,
        manual_deadman_ms: u64,
    ) -> Result<Self, WheelsOffQualificationProfileError> {
        let maximum_abs_timer_pwm_percent = NonZeroU8::new(maximum_abs_timer_pwm_percent)
            .ok_or(WheelsOffQualificationProfileError::ZeroMaximumTimerDuty)?;
        if maximum_abs_timer_pwm_percent.get() > 100 {
            return Err(
                WheelsOffQualificationProfileError::MaximumTimerDutyExceedsOneHundred(
                    maximum_abs_timer_pwm_percent.get(),
                ),
            );
        }
        let manual_test_magnitude_timer_pwm_percent =
            NonZeroU8::new(manual_test_magnitude_timer_pwm_percent)
                .ok_or(WheelsOffQualificationProfileError::ZeroTestTimerDuty)?;
        if manual_test_magnitude_timer_pwm_percent > maximum_abs_timer_pwm_percent {
            return Err(
                WheelsOffQualificationProfileError::TestTimerDutyExceedsMaximum {
                    test: manual_test_magnitude_timer_pwm_percent.get(),
                    maximum: maximum_abs_timer_pwm_percent.get(),
                },
            );
        }
        let manual_deadman_ms = NonZeroU64::new(manual_deadman_ms)
            .ok_or(WheelsOffQualificationProfileError::ZeroDeadman)?;
        if manual_deadman_ms.get() > MAX_DEADMAN_MILLISECONDS {
            return Err(WheelsOffQualificationProfileError::DeadmanTooLong {
                actual_ms: manual_deadman_ms.get(),
                maximum_ms: MAX_DEADMAN_MILLISECONDS,
            });
        }
        let magnitude = i8::try_from(manual_test_magnitude_timer_pwm_percent.get())
            .expect("timer duty is bounded to 100");
        Ok(Self {
            kind: WheelsOffQualificationProfileKind::WheelsOffRawTimerPwmQualification,
            banner: WHEELS_OFF_QUALIFICATION_BANNER,
            command_units: WheelsOffQualificationCommandUnits::SignedTimerDutyPercent,
            required_wheel_state: WheelsOffQualificationRequiredWheelState::Removed,
            autonomous_actuation: WheelsOffQualificationAutonomousActuation::DisabledShadowOnly,
            intent_endpoint: WHEELS_OFF_QUALIFICATION_INTENT_ENDPOINT,
            maximum_abs_timer_pwm_percent,
            manual_test_magnitude_timer_pwm_percent,
            manual_deadman_ms,
            patterns: WheelsOffQualificationPatterns {
                both_positive: WheelsOffQualificationPattern {
                    left_timer_pwm_percent: magnitude,
                    right_timer_pwm_percent: magnitude,
                },
                both_negative: WheelsOffQualificationPattern {
                    left_timer_pwm_percent: -magnitude,
                    right_timer_pwm_percent: -magnitude,
                },
                left_negative_right_positive: WheelsOffQualificationPattern {
                    left_timer_pwm_percent: -magnitude,
                    right_timer_pwm_percent: magnitude,
                },
                left_positive_right_negative: WheelsOffQualificationPattern {
                    left_timer_pwm_percent: magnitude,
                    right_timer_pwm_percent: -magnitude,
                },
            },
        })
    }

    pub const fn kind(self) -> WheelsOffQualificationProfileKind {
        self.kind
    }

    pub const fn maximum_abs_timer_pwm_percent(self) -> u8 {
        self.maximum_abs_timer_pwm_percent.get()
    }

    pub const fn manual_test_magnitude_timer_pwm_percent(self) -> u8 {
        self.manual_test_magnitude_timer_pwm_percent.get()
    }

    pub const fn manual_deadman_ms(self) -> u64 {
        self.manual_deadman_ms.get()
    }

    pub const fn patterns(self) -> WheelsOffQualificationPatterns {
        self.patterns
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationProfileError {
    ZeroMaximumTimerDuty,
    MaximumTimerDutyExceedsOneHundred(u8),
    ZeroTestTimerDuty,
    TestTimerDutyExceedsMaximum { test: u8, maximum: u8 },
    ZeroDeadman,
    DeadmanTooLong { actual_ms: u64, maximum_ms: u64 },
}

impl fmt::Display for WheelsOffQualificationProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid wheels-off qualification control profile: {self:?}"
        )
    }
}

impl std::error::Error for WheelsOffQualificationProfileError {}

/// A checked signed timer-duty percentage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct QualificationTimerPwmPercent(i8);

impl QualificationTimerPwmPercent {
    pub fn parse(
        raw: i16,
        profile: WheelsOffQualificationControlProfile,
    ) -> Result<Self, QualificationTimerPwmParseError> {
        let maximum = i16::from(profile.maximum_abs_timer_pwm_percent());
        if raw < -maximum || raw > maximum {
            return Err(QualificationTimerPwmParseError::OutsideAdmittedMagnitude {
                actual: raw,
                maximum: profile.maximum_abs_timer_pwm_percent(),
            });
        }
        Ok(Self(
            i8::try_from(raw).expect("admitted magnitude is bounded to 100"),
        ))
    }

    pub const fn get(self) -> i8 {
        self.0
    }

    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualificationTimerPwmParseError {
    OutsideAdmittedMagnitude { actual: i16, maximum: u8 },
}

impl fmt::Display for QualificationTimerPwmParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid qualification timer-duty percentage: {self:?}"
        )
    }
}

impl std::error::Error for QualificationTimerPwmParseError {}

/// One nonzero candidate command. A two-wheel zero is deliberately
/// unrepresentable here; callers must use a typed stop intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct QualificationTimerPwmPair {
    pub left_timer_pwm_percent: QualificationTimerPwmPercent,
    pub right_timer_pwm_percent: QualificationTimerPwmPercent,
}

impl QualificationTimerPwmPair {
    pub fn parse(
        left_timer_pwm_percent: i16,
        right_timer_pwm_percent: i16,
        profile: WheelsOffQualificationControlProfile,
    ) -> Result<Self, QualificationTimerPwmPairParseError> {
        let left_timer_pwm_percent =
            QualificationTimerPwmPercent::parse(left_timer_pwm_percent, profile)
                .map_err(QualificationTimerPwmPairParseError::Left)?;
        let right_timer_pwm_percent =
            QualificationTimerPwmPercent::parse(right_timer_pwm_percent, profile)
                .map_err(QualificationTimerPwmPairParseError::Right)?;
        if left_timer_pwm_percent.is_zero() && right_timer_pwm_percent.is_zero() {
            return Err(QualificationTimerPwmPairParseError::ZeroMustUseStopIntent);
        }
        Ok(Self {
            left_timer_pwm_percent,
            right_timer_pwm_percent,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualificationTimerPwmPairParseError {
    Left(QualificationTimerPwmParseError),
    Right(QualificationTimerPwmParseError),
    ZeroMustUseStopIntent,
}

impl fmt::Display for QualificationTimerPwmPairParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid qualification timer-duty pair: {self:?}")
    }
}

impl std::error::Error for QualificationTimerPwmPairParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Left(source) | Self::Right(source) => Some(source),
            Self::ZeroMustUseStopIntent => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WheelsOffQualificationEventId(NonZeroU64);

impl WheelsOffQualificationEventId {
    pub const fn get(self) -> u64 {
        self.0.get()
    }

    pub const fn as_nonzero(self) -> NonZeroU64 {
        self.0
    }
}

impl Serialize for WheelsOffQualificationEventId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serialize_nonzero_u64_as_decimal_string(&self.0, serializer)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WheelsOffQualificationIntentKind {
    BeginManual,
    ManualPwm,
    ReleaseManual,
    Stop,
    SoftwareSafetyStop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationIntent {
    BeginManual,
    ManualPwm(QualificationTimerPwmPair),
    ReleaseManual,
    Stop,
    SoftwareSafetyStop,
}

impl WheelsOffQualificationIntent {
    pub const fn kind(self) -> WheelsOffQualificationIntentKind {
        match self {
            Self::BeginManual => WheelsOffQualificationIntentKind::BeginManual,
            Self::ManualPwm(_) => WheelsOffQualificationIntentKind::ManualPwm,
            Self::ReleaseManual => WheelsOffQualificationIntentKind::ReleaseManual,
            Self::Stop => WheelsOffQualificationIntentKind::Stop,
            Self::SoftwareSafetyStop => WheelsOffQualificationIntentKind::SoftwareSafetyStop,
        }
    }
}

/// Weak HTTP DTO for the distinct qualification endpoint.
///
/// Production `manual_velocity` cannot deserialize into this enum, while
/// `manual_pwm` cannot deserialize into the production console DTO.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct WheelsOffQualificationIntentRequestDto {
    pub schema_version: u32,
    pub control_profile: WheelsOffQualificationProfileKind,
    pub session_id: String,
    pub source_sequence: String,
    pub idempotency_key: String,
    pub intent: WheelsOffQualificationIntentDto,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum WheelsOffQualificationIntentDto {
    BeginManual {},
    ManualPwm {
        left_timer_pwm_percent: i16,
        right_timer_pwm_percent: i16,
    },
    ReleaseManual {},
    Stop {},
    SoftwareSafetyStop {},
}

impl WheelsOffQualificationIntentRequestDto {
    pub(crate) fn parse(
        self,
        profile: WheelsOffQualificationControlProfile,
    ) -> Result<WheelsOffQualificationRequest, WheelsOffQualificationRequestParseError> {
        if self.schema_version != WHEELS_OFF_QUALIFICATION_SCHEMA_V1 {
            return Err(WheelsOffQualificationRequestParseError::UnsupportedSchema(
                self.schema_version,
            ));
        }
        if self.control_profile != profile.kind() {
            return Err(WheelsOffQualificationRequestParseError::WrongControlProfile);
        }
        let session_id = ConsoleSessionId::parse(parse_canonical_decimal(&self.session_id)?)
            .map_err(|_| WheelsOffQualificationRequestParseError::ZeroIdentity)?;
        let source_sequence =
            ConsoleSourceSequence::parse(parse_canonical_decimal(&self.source_sequence)?)
                .map_err(|_| WheelsOffQualificationRequestParseError::ZeroIdentity)?;
        let idempotency_key =
            ConsoleIdempotencyKey::parse(parse_canonical_decimal(&self.idempotency_key)?)
                .map_err(|_| WheelsOffQualificationRequestParseError::ZeroIdentity)?;
        let intent = match self.intent {
            WheelsOffQualificationIntentDto::BeginManual {} => {
                WheelsOffQualificationIntent::BeginManual
            }
            WheelsOffQualificationIntentDto::ManualPwm {
                left_timer_pwm_percent,
                right_timer_pwm_percent,
            } => WheelsOffQualificationIntent::ManualPwm(
                QualificationTimerPwmPair::parse(
                    left_timer_pwm_percent,
                    right_timer_pwm_percent,
                    profile,
                )
                .map_err(WheelsOffQualificationRequestParseError::TimerPwm)?,
            ),
            WheelsOffQualificationIntentDto::ReleaseManual {} => {
                WheelsOffQualificationIntent::ReleaseManual
            }
            WheelsOffQualificationIntentDto::Stop {} => WheelsOffQualificationIntent::Stop,
            WheelsOffQualificationIntentDto::SoftwareSafetyStop {} => {
                WheelsOffQualificationIntent::SoftwareSafetyStop
            }
        };
        Ok(WheelsOffQualificationRequest {
            session_id,
            source_sequence,
            idempotency_key,
            intent,
        })
    }
}

fn parse_canonical_decimal(raw: &str) -> Result<u64, WheelsOffQualificationRequestParseError> {
    if raw.is_empty()
        || !raw.bytes().all(|byte| byte.is_ascii_digit())
        || (raw.len() > 1 && raw.starts_with('0'))
    {
        return Err(WheelsOffQualificationRequestParseError::NonCanonicalDecimalIdentity);
    }
    raw.parse()
        .map_err(|_| WheelsOffQualificationRequestParseError::NonCanonicalDecimalIdentity)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WheelsOffQualificationRequest {
    pub session_id: ConsoleSessionId,
    pub source_sequence: ConsoleSourceSequence,
    pub idempotency_key: ConsoleIdempotencyKey,
    pub intent: WheelsOffQualificationIntent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationRequestParseError {
    UnsupportedSchema(u32),
    WrongControlProfile,
    NonCanonicalDecimalIdentity,
    ZeroIdentity,
    TimerPwm(QualificationTimerPwmPairParseError),
}

impl fmt::Display for WheelsOffQualificationRequestParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid wheels-off qualification request: {self:?}"
        )
    }
}

impl std::error::Error for WheelsOffQualificationRequestParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::TimerPwm(source) => Some(source),
            Self::UnsupportedSchema(_)
            | Self::WrongControlProfile
            | Self::NonCanonicalDecimalIdentity
            | Self::ZeroIdentity => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct WheelsOffQualificationManualAuthority {
    pub source: ConsoleSourceKind,
    pub session_id: ConsoleSessionId,
    pub authority_generation: WheelsOffQualificationEventId,
    #[serde(serialize_with = "serialize_u64_as_decimal_string")]
    pub deadman_deadline_host_monotonic_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct WheelsOffQualificationRequestedCommand {
    pub event_id: WheelsOffQualificationEventId,
    pub source: ConsoleSourceKind,
    pub session_id: ConsoleSessionId,
    pub kind: WheelsOffQualificationIntentKind,
    pub requested_pwm: Option<QualificationTimerPwmPair>,
}

/// One controller-applied qualification step correlated to both its operator
/// event and the durable navigation-ingress record that owns its evidence.
///
/// Fields are private so this value can only be minted by the runtime after a
/// matching journal append has succeeded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct WheelsOffQualificationAppliedStep {
    event_id: WheelsOffQualificationEventId,
    #[serde(serialize_with = "serialize_u64_as_decimal_string")]
    navigation_ingress_sequence: u64,
    requested_target: QualificationTimerPwmPair,
    target_reached: bool,
    receipt: ConsoleAppliedReceipt,
}

impl WheelsOffQualificationAppliedStep {
    pub(super) const fn from_journaled_parts(
        event_id: WheelsOffQualificationEventId,
        navigation_ingress_sequence: u64,
        requested_target: QualificationTimerPwmPair,
        target_reached: bool,
        receipt: ConsoleAppliedReceipt,
    ) -> Self {
        Self {
            event_id,
            navigation_ingress_sequence,
            requested_target,
            target_reached,
            receipt,
        }
    }

    pub const fn event_id(self) -> WheelsOffQualificationEventId {
        self.event_id
    }

    pub const fn navigation_ingress_sequence(self) -> u64 {
        self.navigation_ingress_sequence
    }

    pub const fn requested_target(self) -> QualificationTimerPwmPair {
        self.requested_target
    }

    pub const fn target_reached(self) -> bool {
        self.target_reached
    }

    pub const fn receipt(self) -> ConsoleAppliedReceipt {
        self.receipt
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QualificationCandidatePwmEvent {
    event_id: WheelsOffQualificationEventId,
    authority_generation: WheelsOffQualificationEventId,
    source: ConsoleSourceKind,
    session_id: ConsoleSessionId,
    received_at: HostMonotonicTimestamp,
    requested_pwm: QualificationTimerPwmPair,
}

impl QualificationCandidatePwmEvent {
    pub const fn event_id(self) -> WheelsOffQualificationEventId {
        self.event_id
    }

    pub const fn authority_generation(self) -> WheelsOffQualificationEventId {
        self.authority_generation
    }

    pub const fn source(self) -> ConsoleSourceKind {
        self.source
    }

    pub const fn session_id(self) -> ConsoleSessionId {
        self.session_id
    }

    pub const fn received_at(self) -> HostMonotonicTimestamp {
        self.received_at
    }

    pub const fn requested_pwm(self) -> QualificationTimerPwmPair {
        self.requested_pwm
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateMotionTerminalCause {
    ManualRelease,
    ExplicitGlobalStop,
    ManualDeadmanExpired,
    SessionClosed,
    ConnectionLost,
    FrontendConnectionLost,
    SoftwareSafetyStop,
    InternalBoundaryFault,
    RuntimeReceiverDisconnected,
}

impl CandidateMotionTerminalCause {
    const fn bit(self) -> u16 {
        1 << (self as u16)
    }

    const fn severity(self) -> u8 {
        match self {
            Self::ManualRelease => 0,
            Self::SessionClosed => 1,
            Self::ManualDeadmanExpired => 2,
            Self::ConnectionLost | Self::FrontendConnectionLost => 3,
            Self::ExplicitGlobalStop => 4,
            Self::InternalBoundaryFault | Self::RuntimeReceiverDisconnected => 5,
            Self::SoftwareSafetyStop => 6,
        }
    }
}

/// Allocation-free set preserving every reason coalesced into one pending
/// terminal transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CandidateMotionTerminalCauses(u16);

impl CandidateMotionTerminalCauses {
    const fn one(cause: CandidateMotionTerminalCause) -> Self {
        Self(cause.bit())
    }

    fn insert(&mut self, cause: CandidateMotionTerminalCause) {
        self.0 |= cause.bit();
    }

    pub const fn contains(self, cause: CandidateMotionTerminalCause) -> bool {
        self.0 & cause.bit() != 0
    }

    pub fn primary(self) -> CandidateMotionTerminalCause {
        const ALL: [CandidateMotionTerminalCause; 9] = [
            CandidateMotionTerminalCause::ManualRelease,
            CandidateMotionTerminalCause::ExplicitGlobalStop,
            CandidateMotionTerminalCause::ManualDeadmanExpired,
            CandidateMotionTerminalCause::SessionClosed,
            CandidateMotionTerminalCause::ConnectionLost,
            CandidateMotionTerminalCause::FrontendConnectionLost,
            CandidateMotionTerminalCause::SoftwareSafetyStop,
            CandidateMotionTerminalCause::InternalBoundaryFault,
            CandidateMotionTerminalCause::RuntimeReceiverDisconnected,
        ];
        ALL.into_iter()
            .filter(|cause| self.contains(*cause))
            .max_by_key(|cause| cause.severity())
            .expect("terminal cause set is constructed nonempty")
    }
}

impl Serialize for CandidateMotionTerminalCauses {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let values = [
            CandidateMotionTerminalCause::ManualRelease,
            CandidateMotionTerminalCause::ExplicitGlobalStop,
            CandidateMotionTerminalCause::ManualDeadmanExpired,
            CandidateMotionTerminalCause::SessionClosed,
            CandidateMotionTerminalCause::ConnectionLost,
            CandidateMotionTerminalCause::FrontendConnectionLost,
            CandidateMotionTerminalCause::SoftwareSafetyStop,
            CandidateMotionTerminalCause::InternalBoundaryFault,
            CandidateMotionTerminalCause::RuntimeReceiverDisconnected,
        ]
        .into_iter()
        .filter(|cause| self.contains(*cause))
        .collect::<Vec<_>>();
        values.serialize(serializer)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct CandidateMotionTerminalEvent {
    event_id: WheelsOffQualificationEventId,
    authority_generation: Option<WheelsOffQualificationEventId>,
    source_session_id: Option<ConsoleSessionId>,
    causes: CandidateMotionTerminalCauses,
    first_received_at_host_monotonic_ns: Option<u64>,
    latest_received_at_host_monotonic_ns: Option<u64>,
}

impl CandidateMotionTerminalEvent {
    pub const fn event_id(self) -> WheelsOffQualificationEventId {
        self.event_id
    }

    pub const fn authority_generation(self) -> Option<WheelsOffQualificationEventId> {
        self.authority_generation
    }

    pub const fn source_session_id(self) -> Option<ConsoleSessionId> {
        self.source_session_id
    }

    pub const fn causes(self) -> CandidateMotionTerminalCauses {
        self.causes
    }

    pub fn primary_cause(self) -> CandidateMotionTerminalCause {
        self.causes.primary()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationIngressEvent {
    TerminalStop(CandidateMotionTerminalEvent),
    CandidatePwm(QualificationCandidatePwmEvent),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationReceiveError {
    Empty,
    RuntimeReceiverDisconnected,
}

/// Minimal controller observation required to clear a stop barrier.
///
/// This proves only that the caller supplied an observed zero bound to the
/// exact host stop request; it does not claim physical wheel motion or
/// stopping distance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct QualificationObservedAppliedZero {
    #[serde(serialize_with = "serialize_u64_as_decimal_string")]
    controller_stop_request_id: u64,
    applied_left_timer_pwm_percent: i8,
    applied_right_timer_pwm_percent: i8,
}

impl QualificationObservedAppliedZero {
    pub fn parse(
        controller_stop_request_id: u64,
        applied_left_timer_pwm_percent: i8,
        applied_right_timer_pwm_percent: i8,
    ) -> Result<Self, QualificationObservedAppliedZeroError> {
        if applied_left_timer_pwm_percent != 0 || applied_right_timer_pwm_percent != 0 {
            return Err(QualificationObservedAppliedZeroError::Nonzero {
                left_timer_pwm_percent: applied_left_timer_pwm_percent,
                right_timer_pwm_percent: applied_right_timer_pwm_percent,
            });
        }
        Ok(Self {
            controller_stop_request_id,
            applied_left_timer_pwm_percent,
            applied_right_timer_pwm_percent,
        })
    }

    pub const fn controller_stop_request_id(self) -> u64 {
        self.controller_stop_request_id
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualificationObservedAppliedZeroError {
    Nonzero {
        left_timer_pwm_percent: i8,
        right_timer_pwm_percent: i8,
    },
}

impl fmt::Display for QualificationObservedAppliedZeroError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "qualification stop completion is not an observed applied zero: {self:?}"
        )
    }
}

impl std::error::Error for QualificationObservedAppliedZeroError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct WheelsOffQualificationTerminalCompletion {
    pub event_id: WheelsOffQualificationEventId,
    pub observed_applied_zero: QualificationObservedAppliedZero,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum WheelsOffQualificationSubmitOutcome {
    ManualAuthorityAcquired {
        event_id: WheelsOffQualificationEventId,
    },
    CandidatePwmQueued {
        event_id: WheelsOffQualificationEventId,
        superseded_event_id: Option<WheelsOffQualificationEventId>,
    },
    TerminalStopQueued {
        event_id: WheelsOffQualificationEventId,
        coalesced: bool,
    },
    SoftwareSafetyStopLatched {
        event_id: WheelsOffQualificationEventId,
        coalesced: bool,
    },
    IdempotentReplay {
        event_id: WheelsOffQualificationEventId,
    },
}

impl WheelsOffQualificationSubmitOutcome {
    pub const fn event_id(self) -> WheelsOffQualificationEventId {
        match self {
            Self::ManualAuthorityAcquired { event_id }
            | Self::CandidatePwmQueued { event_id, .. }
            | Self::TerminalStopQueued { event_id, .. }
            | Self::SoftwareSafetyStopLatched { event_id, .. }
            | Self::IdempotentReplay { event_id } => event_id,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationSubmitError {
    UnknownSession(ConsoleSessionId),
    SessionCapabilityMismatch,
    SessionCapacityReached,
    SourceSequenceNotIncreasing {
        previous: ConsoleSourceSequence,
        current: ConsoleSourceSequence,
    },
    IdempotencyConflict(ConsoleIdempotencyKey),
    AuthorityConflict {
        requested_by: ConsoleSessionId,
        held_by: ConsoleSessionId,
    },
    ManualAuthorityRequired(ConsoleSessionId),
    StopBarrierPending,
    SoftwareSafetyStopLatched,
    RuntimeReceiverDisconnected,
    DeadmanDeadlineOverflow,
    EventIdentityExhausted,
}

impl fmt::Display for WheelsOffQualificationSubmitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "wheels-off qualification request rejected: {self:?}"
        )
    }
}

impl std::error::Error for WheelsOffQualificationSubmitError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WheelsOffQualificationRuntimeIngressState {
    Connected,
    DisconnectedStopConfirmed,
    DisconnectedStopUnconfirmed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct WheelsOffQualificationSnapshot {
    pub schema_version: u32,
    pub control_profile: WheelsOffQualificationControlProfile,
    #[serde(serialize_with = "serialize_nonzero_u64_as_decimal_string")]
    pub revision: NonZeroU64,
    pub manual_authority: Option<WheelsOffQualificationManualAuthority>,
    pub last_requested: Option<WheelsOffQualificationRequestedCommand>,
    pub last_applied_step: Option<WheelsOffQualificationAppliedStep>,
    pub last_terminal_stop: Option<CandidateMotionTerminalEvent>,
    pub last_terminal_completion: Option<WheelsOffQualificationTerminalCompletion>,
    pub stop_barrier_pending: bool,
    pub software_safety_stop_latched: bool,
    pub runtime_ingress_state: WheelsOffQualificationRuntimeIngressState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationAppliedStepRecordError {
    JournalSequenceNotIncreasing { previous: u64, actual: u64 },
}

impl fmt::Display for WheelsOffQualificationAppliedStepRecordError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "qualification applied-step evidence rejected: {self:?}"
        )
    }
}

impl std::error::Error for WheelsOffQualificationAppliedStepRecordError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationCompletionError {
    NoTerminalStopInFlight,
    WrongTerminalEvent {
        expected: WheelsOffQualificationEventId,
        actual: WheelsOffQualificationEventId,
    },
}

impl fmt::Display for WheelsOffQualificationCompletionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid qualification terminal-stop completion: {self:?}"
        )
    }
}

impl std::error::Error for WheelsOffQualificationCompletionError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationDisconnectError {
    AlreadyDisconnected(WheelsOffQualificationRuntimeIngressState),
    EventIdentityExhausted,
}

impl fmt::Display for WheelsOffQualificationDisconnectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "could not record confirmed qualification runtime shutdown: {self:?}"
        )
    }
}

impl std::error::Error for WheelsOffQualificationDisconnectError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CachedSubmission {
    key: ConsoleIdempotencyKey,
    intent: WheelsOffQualificationIntent,
    outcome: WheelsOffQualificationSubmitOutcome,
}

#[derive(Debug)]
struct QualificationSession {
    source: ConsoleSourceKind,
    capability: ConsoleSessionCapability,
    last_source_sequence: Option<ConsoleSourceSequence>,
    idempotency: VecDeque<CachedSubmission>,
}

#[derive(Debug)]
struct QualificationState {
    profile: WheelsOffQualificationControlProfile,
    maximum_sessions: usize,
    idempotency_records_per_session: usize,
    next_session_id: Option<NonZeroU64>,
    next_event_id: Option<NonZeroU64>,
    revision: NonZeroU64,
    sessions: HashMap<ConsoleSessionId, QualificationSession>,
    authority: Option<WheelsOffQualificationManualAuthority>,
    pending_pwm: Option<QualificationCandidatePwmEvent>,
    pending_terminal: Option<CandidateMotionTerminalEvent>,
    terminal_in_flight: Option<CandidateMotionTerminalEvent>,
    safety_stop_event_id: Option<WheelsOffQualificationEventId>,
    software_safety_stop_latched: bool,
    runtime_ingress_state: WheelsOffQualificationRuntimeIngressState,
    last_requested: Option<WheelsOffQualificationRequestedCommand>,
    last_applied_step: Option<WheelsOffQualificationAppliedStep>,
    last_terminal_stop: Option<CandidateMotionTerminalEvent>,
    last_terminal_completion: Option<WheelsOffQualificationTerminalCompletion>,
}

impl QualificationState {
    fn allocate_event(
        &mut self,
    ) -> Result<WheelsOffQualificationEventId, WheelsOffQualificationSubmitError> {
        let raw = self
            .next_event_id
            .ok_or(WheelsOffQualificationSubmitError::EventIdentityExhausted)?;
        self.next_event_id = raw.get().checked_add(1).and_then(NonZeroU64::new);
        Ok(WheelsOffQualificationEventId(raw))
    }

    fn bump_revision(&mut self) {
        if let Some(next) = self.revision.get().checked_add(1).and_then(NonZeroU64::new) {
            self.revision = next;
        } else {
            self.software_safety_stop_latched = true;
            self.pending_pwm = None;
            self.authority = None;
        }
    }

    fn stop_barrier_pending(&self) -> bool {
        self.pending_terminal.is_some() || self.terminal_in_flight.is_some()
    }

    fn queue_terminal(
        &mut self,
        cause: CandidateMotionTerminalCause,
        source_session_id: Option<ConsoleSessionId>,
        received_at: Option<HostMonotonicTimestamp>,
    ) -> Result<(WheelsOffQualificationEventId, bool), WheelsOffQualificationSubmitError> {
        self.pending_pwm = None;
        let authority_generation = self
            .authority
            .map(|authority| authority.authority_generation);
        self.authority = None;
        if let Some(pending) = self.pending_terminal.as_mut() {
            pending.causes.insert(cause);
            pending.latest_received_at_host_monotonic_ns =
                received_at.map(HostMonotonicTimestamp::as_nanos);
            self.last_terminal_stop = Some(*pending);
            return Ok((pending.event_id, true));
        }
        let event_id = self.allocate_event()?;
        let timestamp = received_at.map(HostMonotonicTimestamp::as_nanos);
        let event = CandidateMotionTerminalEvent {
            event_id,
            authority_generation,
            source_session_id,
            causes: CandidateMotionTerminalCauses::one(cause),
            first_received_at_host_monotonic_ns: timestamp,
            latest_received_at_host_monotonic_ns: timestamp,
        };
        self.pending_terminal = Some(event);
        self.last_terminal_stop = Some(event);
        Ok((event_id, false))
    }

    fn latch_internal_fault(&mut self) {
        self.software_safety_stop_latched = true;
        let _ = self.queue_terminal(
            CandidateMotionTerminalCause::InternalBoundaryFault,
            None,
            None,
        );
        self.bump_revision();
    }
}

#[derive(Debug)]
struct QualificationShared {
    state: Mutex<QualificationState>,
}

#[derive(Clone, Debug)]
pub struct WheelsOffQualificationConsoleHandle {
    shared: Arc<QualificationShared>,
}

#[derive(Debug)]
pub struct WheelsOffQualificationIngressReceiver {
    shared: Arc<QualificationShared>,
    terminal: bool,
}

pub fn wheels_off_qualification_console(
    profile: WheelsOffQualificationControlProfile,
) -> (
    WheelsOffQualificationConsoleHandle,
    WheelsOffQualificationIngressReceiver,
) {
    wheels_off_qualification_console_with_limits(
        profile,
        16,
        MAX_WHEELS_OFF_QUALIFICATION_IDEMPOTENCY_RECORDS,
    )
    .expect("static qualification-console limits are valid")
}

pub fn wheels_off_qualification_console_with_limits(
    profile: WheelsOffQualificationControlProfile,
    maximum_sessions: usize,
    idempotency_records_per_session: usize,
) -> Result<
    (
        WheelsOffQualificationConsoleHandle,
        WheelsOffQualificationIngressReceiver,
    ),
    WheelsOffQualificationLimitsError,
> {
    if maximum_sessions == 0 || maximum_sessions > MAX_WHEELS_OFF_QUALIFICATION_SESSIONS {
        return Err(WheelsOffQualificationLimitsError::MaximumSessions {
            actual: maximum_sessions,
            maximum: MAX_WHEELS_OFF_QUALIFICATION_SESSIONS,
        });
    }
    if idempotency_records_per_session == 0
        || idempotency_records_per_session > MAX_WHEELS_OFF_QUALIFICATION_IDEMPOTENCY_RECORDS
    {
        return Err(
            WheelsOffQualificationLimitsError::IdempotencyRecordsPerSession {
                actual: idempotency_records_per_session,
                maximum: MAX_WHEELS_OFF_QUALIFICATION_IDEMPOTENCY_RECORDS,
            },
        );
    }
    let shared = Arc::new(QualificationShared {
        state: Mutex::new(QualificationState {
            profile,
            maximum_sessions,
            idempotency_records_per_session,
            next_session_id: NonZeroU64::new(1),
            next_event_id: NonZeroU64::new(1),
            revision: NonZeroU64::new(1).expect("one is nonzero"),
            sessions: HashMap::new(),
            authority: None,
            pending_pwm: None,
            pending_terminal: None,
            terminal_in_flight: None,
            safety_stop_event_id: None,
            software_safety_stop_latched: false,
            runtime_ingress_state: WheelsOffQualificationRuntimeIngressState::Connected,
            last_requested: None,
            last_applied_step: None,
            last_terminal_stop: None,
            last_terminal_completion: None,
        }),
    });
    Ok((
        WheelsOffQualificationConsoleHandle {
            shared: Arc::clone(&shared),
        },
        WheelsOffQualificationIngressReceiver {
            shared,
            terminal: false,
        },
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WheelsOffQualificationLimitsError {
    MaximumSessions { actual: usize, maximum: usize },
    IdempotencyRecordsPerSession { actual: usize, maximum: usize },
}

impl fmt::Display for WheelsOffQualificationLimitsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid wheels-off qualification console limits: {self:?}"
        )
    }
}

impl std::error::Error for WheelsOffQualificationLimitsError {}

impl WheelsOffQualificationConsoleHandle {
    fn lock_state(&self) -> MutexGuard<'_, QualificationState> {
        match self.shared.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                guard.latch_internal_fault();
                guard
            }
        }
    }

    pub fn open_session(
        &self,
        source: ConsoleSourceKind,
        capability: ConsoleSessionCapability,
    ) -> Result<ConsoleSessionId, WheelsOffQualificationSubmitError> {
        let mut state = self.lock_state();
        if state.runtime_ingress_state != WheelsOffQualificationRuntimeIngressState::Connected {
            return Err(WheelsOffQualificationSubmitError::RuntimeReceiverDisconnected);
        }
        if state.sessions.len() == state.maximum_sessions {
            return Err(WheelsOffQualificationSubmitError::SessionCapacityReached);
        }
        let raw = state
            .next_session_id
            .ok_or(WheelsOffQualificationSubmitError::EventIdentityExhausted)?;
        state.next_session_id = raw.get().checked_add(1).and_then(NonZeroU64::new);
        let session_id =
            ConsoleSessionId::parse(raw.get()).expect("allocated session ID is nonzero");
        state.sessions.insert(
            session_id,
            QualificationSession {
                source,
                capability,
                last_source_sequence: None,
                idempotency: VecDeque::new(),
            },
        );
        state.bump_revision();
        Ok(session_id)
    }

    pub fn session_capability_matches(
        &self,
        session_id: ConsoleSessionId,
        capability: ConsoleSessionCapability,
    ) -> bool {
        self.lock_state()
            .sessions
            .get(&session_id)
            .is_some_and(|session| constant_time_capability_matches(session.capability, capability))
    }

    pub fn submit(
        &self,
        request: WheelsOffQualificationRequest,
        capability: ConsoleSessionCapability,
        received_at: HostMonotonicTimestamp,
    ) -> Result<WheelsOffQualificationSubmitOutcome, WheelsOffQualificationSubmitError> {
        let mut state = self.lock_state();
        if state.runtime_ingress_state != WheelsOffQualificationRuntimeIngressState::Connected {
            return Err(WheelsOffQualificationSubmitError::RuntimeReceiverDisconnected);
        }
        let session = state.sessions.get(&request.session_id).ok_or(
            WheelsOffQualificationSubmitError::UnknownSession(request.session_id),
        )?;
        if !constant_time_capability_matches(session.capability, capability) {
            return Err(WheelsOffQualificationSubmitError::SessionCapabilityMismatch);
        }
        if let Some(cached) = session
            .idempotency
            .iter()
            .find(|cached| cached.key == request.idempotency_key)
        {
            if cached.intent == request.intent {
                return Ok(WheelsOffQualificationSubmitOutcome::IdempotentReplay {
                    event_id: cached.outcome.event_id(),
                });
            }
            return Err(WheelsOffQualificationSubmitError::IdempotencyConflict(
                request.idempotency_key,
            ));
        }
        if let Some(previous) = session.last_source_sequence
            && request.source_sequence <= previous
        {
            return Err(
                WheelsOffQualificationSubmitError::SourceSequenceNotIncreasing {
                    previous,
                    current: request.source_sequence,
                },
            );
        }
        if state.software_safety_stop_latched {
            if request.intent == WheelsOffQualificationIntent::SoftwareSafetyStop
                && let Some(event_id) = state.safety_stop_event_id
            {
                return Ok(
                    WheelsOffQualificationSubmitOutcome::SoftwareSafetyStopLatched {
                        event_id,
                        coalesced: true,
                    },
                );
            }
            return Err(WheelsOffQualificationSubmitError::SoftwareSafetyStopLatched);
        }
        if state.stop_barrier_pending()
            && !matches!(
                request.intent,
                WheelsOffQualificationIntent::Stop
                    | WheelsOffQualificationIntent::SoftwareSafetyStop
            )
        {
            return Err(WheelsOffQualificationSubmitError::StopBarrierPending);
        }

        let source = session.source;
        let held_by = state.authority.map(|authority| authority.session_id);
        match request.intent {
            WheelsOffQualificationIntent::BeginManual => {
                if let Some(held_by) = held_by {
                    return Err(WheelsOffQualificationSubmitError::AuthorityConflict {
                        requested_by: request.session_id,
                        held_by,
                    });
                }
            }
            WheelsOffQualificationIntent::ManualPwm(_)
            | WheelsOffQualificationIntent::ReleaseManual => {
                if held_by != Some(request.session_id) {
                    return Err(WheelsOffQualificationSubmitError::ManualAuthorityRequired(
                        request.session_id,
                    ));
                }
            }
            WheelsOffQualificationIntent::Stop
            | WheelsOffQualificationIntent::SoftwareSafetyStop => {}
        }

        let deadline = if matches!(
            request.intent,
            WheelsOffQualificationIntent::BeginManual | WheelsOffQualificationIntent::ManualPwm(_)
        ) {
            let duration_ns = state
                .profile
                .manual_deadman_ms()
                .checked_mul(1_000_000)
                .ok_or(WheelsOffQualificationSubmitError::DeadmanDeadlineOverflow)?;
            Some(
                received_at
                    .as_nanos()
                    .checked_add(duration_ns)
                    .ok_or(WheelsOffQualificationSubmitError::DeadmanDeadlineOverflow)?,
            )
        } else {
            None
        };

        let (outcome, requested_pwm) = match request.intent {
            WheelsOffQualificationIntent::BeginManual => {
                let event_id = state.allocate_event()?;
                state.authority = Some(WheelsOffQualificationManualAuthority {
                    source,
                    session_id: request.session_id,
                    authority_generation: event_id,
                    deadman_deadline_host_monotonic_ns: deadline
                        .ok_or(WheelsOffQualificationSubmitError::DeadmanDeadlineOverflow)?,
                });
                (
                    WheelsOffQualificationSubmitOutcome::ManualAuthorityAcquired { event_id },
                    None,
                )
            }
            WheelsOffQualificationIntent::ManualPwm(requested_pwm) => {
                let authority_generation = state
                    .authority
                    .filter(|authority| authority.session_id == request.session_id)
                    .map(|authority| authority.authority_generation)
                    .ok_or(WheelsOffQualificationSubmitError::ManualAuthorityRequired(
                        request.session_id,
                    ))?;
                let event_id = state.allocate_event()?;
                let superseded_event_id = state.pending_pwm.map(|pending| pending.event_id);
                state.pending_pwm = Some(QualificationCandidatePwmEvent {
                    event_id,
                    authority_generation,
                    source,
                    session_id: request.session_id,
                    received_at,
                    requested_pwm,
                });
                state.authority = Some(WheelsOffQualificationManualAuthority {
                    source,
                    session_id: request.session_id,
                    authority_generation,
                    deadman_deadline_host_monotonic_ns: deadline
                        .ok_or(WheelsOffQualificationSubmitError::DeadmanDeadlineOverflow)?,
                });
                (
                    WheelsOffQualificationSubmitOutcome::CandidatePwmQueued {
                        event_id,
                        superseded_event_id,
                    },
                    Some(requested_pwm),
                )
            }
            WheelsOffQualificationIntent::ReleaseManual => {
                let (event_id, coalesced) = state.queue_terminal(
                    CandidateMotionTerminalCause::ManualRelease,
                    Some(request.session_id),
                    Some(received_at),
                )?;
                (
                    WheelsOffQualificationSubmitOutcome::TerminalStopQueued {
                        event_id,
                        coalesced,
                    },
                    None,
                )
            }
            WheelsOffQualificationIntent::Stop => {
                let (event_id, coalesced) = state.queue_terminal(
                    CandidateMotionTerminalCause::ExplicitGlobalStop,
                    Some(request.session_id),
                    Some(received_at),
                )?;
                (
                    WheelsOffQualificationSubmitOutcome::TerminalStopQueued {
                        event_id,
                        coalesced,
                    },
                    None,
                )
            }
            WheelsOffQualificationIntent::SoftwareSafetyStop => {
                state.software_safety_stop_latched = true;
                let (event_id, coalesced) = state.queue_terminal(
                    CandidateMotionTerminalCause::SoftwareSafetyStop,
                    Some(request.session_id),
                    Some(received_at),
                )?;
                state.safety_stop_event_id = Some(event_id);
                (
                    WheelsOffQualificationSubmitOutcome::SoftwareSafetyStopLatched {
                        event_id,
                        coalesced,
                    },
                    None,
                )
            }
        };
        state.last_requested = Some(WheelsOffQualificationRequestedCommand {
            event_id: outcome.event_id(),
            source,
            session_id: request.session_id,
            kind: request.intent.kind(),
            requested_pwm,
        });
        let idempotency_limit = state.idempotency_records_per_session;
        let session = state.sessions.get_mut(&request.session_id).ok_or(
            WheelsOffQualificationSubmitError::UnknownSession(request.session_id),
        )?;
        session.last_source_sequence = Some(request.source_sequence);
        if session.idempotency.len() == idempotency_limit {
            session.idempotency.pop_front();
        }
        session.idempotency.push_back(CachedSubmission {
            key: request.idempotency_key,
            intent: request.intent,
            outcome,
        });
        state.bump_revision();
        Ok(outcome)
    }

    /// Explicitly close one authenticated session. If it owns manual
    /// authority, closure creates a stop barrier instead of refusing cleanup.
    pub fn close_session(
        &self,
        session_id: ConsoleSessionId,
        capability: ConsoleSessionCapability,
        received_at: HostMonotonicTimestamp,
    ) -> Result<bool, WheelsOffQualificationSubmitError> {
        self.remove_session(
            session_id,
            capability,
            received_at,
            CandidateMotionTerminalCause::SessionClosed,
        )
    }

    /// Report loss of one authenticated transport/session.
    pub fn report_connection_lost(
        &self,
        session_id: ConsoleSessionId,
        capability: ConsoleSessionCapability,
        received_at: HostMonotonicTimestamp,
    ) -> Result<bool, WheelsOffQualificationSubmitError> {
        self.remove_session(
            session_id,
            capability,
            received_at,
            CandidateMotionTerminalCause::ConnectionLost,
        )
    }

    fn remove_session(
        &self,
        session_id: ConsoleSessionId,
        capability: ConsoleSessionCapability,
        received_at: HostMonotonicTimestamp,
        terminal_cause: CandidateMotionTerminalCause,
    ) -> Result<bool, WheelsOffQualificationSubmitError> {
        let mut state = self.lock_state();
        let session = state.sessions.get(&session_id).ok_or(
            WheelsOffQualificationSubmitError::UnknownSession(session_id),
        )?;
        if !constant_time_capability_matches(session.capability, capability) {
            return Err(WheelsOffQualificationSubmitError::SessionCapabilityMismatch);
        }
        let owned = state
            .authority
            .is_some_and(|authority| authority.session_id == session_id);
        state.sessions.remove(&session_id);
        if owned {
            state.queue_terminal(terminal_cause, Some(session_id), Some(received_at))?;
        }
        state.bump_revision();
        Ok(owned)
    }

    /// Report loss of the whole HTTP/control frontend. All sessions are
    /// invalidated and any possible candidate authority gets a typed stop.
    pub fn report_frontend_connection_lost(
        &self,
        received_at: HostMonotonicTimestamp,
    ) -> Result<bool, WheelsOffQualificationSubmitError> {
        let mut state = self.lock_state();
        let could_have_candidate_motion = state.authority.is_some() || state.pending_pwm.is_some();
        state.sessions.clear();
        if could_have_candidate_motion {
            state.queue_terminal(
                CandidateMotionTerminalCause::FrontendConnectionLost,
                None,
                Some(received_at),
            )?;
        }
        state.bump_revision();
        Ok(could_have_candidate_motion)
    }

    pub fn tick_deadman(
        &self,
        now: HostMonotonicTimestamp,
    ) -> Result<bool, WheelsOffQualificationSubmitError> {
        let mut state = self.lock_state();
        if state.software_safety_stop_latched || state.stop_barrier_pending() {
            return Ok(false);
        }
        let Some(authority) = state.authority else {
            return Ok(false);
        };
        if now.as_nanos() < authority.deadman_deadline_host_monotonic_ns {
            return Ok(false);
        }
        state.queue_terminal(
            CandidateMotionTerminalCause::ManualDeadmanExpired,
            Some(authority.session_id),
            Some(now),
        )?;
        state.bump_revision();
        Ok(true)
    }

    /// One-way process-lifetime fail-closed hook for clock, HTTP, or ownership
    /// faults outside an authenticated request.
    pub fn signal_internal_fail_closed(&self, observed_at: Option<HostMonotonicTimestamp>) {
        let mut state = self.lock_state();
        state.software_safety_stop_latched = true;
        if let Ok((event_id, _)) = state.queue_terminal(
            CandidateMotionTerminalCause::InternalBoundaryFault,
            None,
            observed_at,
        ) {
            state.safety_stop_event_id = Some(event_id);
        }
        state.bump_revision();
    }

    pub(super) fn record_applied_step(
        &self,
        step: WheelsOffQualificationAppliedStep,
    ) -> Result<(), WheelsOffQualificationAppliedStepRecordError> {
        let mut state = self.lock_state();
        if let Some(previous) = state.last_applied_step
            && step.navigation_ingress_sequence() <= previous.navigation_ingress_sequence()
        {
            return Err(
                WheelsOffQualificationAppliedStepRecordError::JournalSequenceNotIncreasing {
                    previous: previous.navigation_ingress_sequence(),
                    actual: step.navigation_ingress_sequence(),
                },
            );
        }
        state.last_applied_step = Some(step);
        state.bump_revision();
        Ok(())
    }

    pub fn snapshot(&self) -> WheelsOffQualificationSnapshot {
        let state = self.lock_state();
        WheelsOffQualificationSnapshot {
            schema_version: WHEELS_OFF_QUALIFICATION_SCHEMA_V1,
            control_profile: state.profile,
            revision: state.revision,
            manual_authority: state.authority,
            last_requested: state.last_requested,
            last_applied_step: state.last_applied_step,
            last_terminal_stop: state.last_terminal_stop,
            last_terminal_completion: state.last_terminal_completion,
            stop_barrier_pending: state.stop_barrier_pending(),
            software_safety_stop_latched: state.software_safety_stop_latched,
            runtime_ingress_state: state.runtime_ingress_state,
        }
    }
}

fn constant_time_capability_matches(
    expected: ConsoleSessionCapability,
    candidate: ConsoleSessionCapability,
) -> bool {
    let mut difference = 0_u8;
    for (expected, actual) in expected.as_bytes().iter().zip(candidate.as_bytes()) {
        difference |= expected ^ actual;
    }
    difference == 0
}

impl WheelsOffQualificationIngressReceiver {
    fn lock_state(&self) -> MutexGuard<'_, QualificationState> {
        match self.shared.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => {
                let mut guard = poisoned.into_inner();
                guard.latch_internal_fault();
                guard
            }
        }
    }

    /// Stops always drain before candidate duty. An in-flight stop is a
    /// barrier, so no duty can be observed until its applied zero is reported.
    pub fn try_recv(
        &mut self,
    ) -> Result<WheelsOffQualificationIngressEvent, WheelsOffQualificationReceiveError> {
        if self.terminal {
            return Err(WheelsOffQualificationReceiveError::RuntimeReceiverDisconnected);
        }
        let mut state = self.lock_state();
        if state.runtime_ingress_state != WheelsOffQualificationRuntimeIngressState::Connected {
            return Err(WheelsOffQualificationReceiveError::RuntimeReceiverDisconnected);
        }
        if state.terminal_in_flight.is_none()
            && let Some(event) = state.pending_terminal.take()
        {
            state.terminal_in_flight = Some(event);
            state.bump_revision();
            return Ok(WheelsOffQualificationIngressEvent::TerminalStop(event));
        }
        if state.terminal_in_flight.is_some() {
            return Err(WheelsOffQualificationReceiveError::Empty);
        }
        if let Some(event) = state.pending_pwm.take() {
            state.bump_revision();
            return Ok(WheelsOffQualificationIngressEvent::CandidatePwm(event));
        }
        Err(WheelsOffQualificationReceiveError::Empty)
    }

    pub fn complete_terminal_stop(
        &mut self,
        event_id: WheelsOffQualificationEventId,
        observed_applied_zero: QualificationObservedAppliedZero,
    ) -> Result<WheelsOffQualificationTerminalCompletion, WheelsOffQualificationCompletionError>
    {
        let mut state = self.lock_state();
        let in_flight = state
            .terminal_in_flight
            .ok_or(WheelsOffQualificationCompletionError::NoTerminalStopInFlight)?;
        if in_flight.event_id != event_id {
            return Err(WheelsOffQualificationCompletionError::WrongTerminalEvent {
                expected: in_flight.event_id,
                actual: event_id,
            });
        }
        let completion = WheelsOffQualificationTerminalCompletion {
            event_id,
            observed_applied_zero,
        };
        state.terminal_in_flight = None;
        state.last_terminal_completion = Some(completion);
        state.bump_revision();
        Ok(completion)
    }

    /// Record a direct controller `HostStop` that returned an observed applied
    /// zero, then permanently disconnect this sole runtime ingress.
    ///
    /// The transition is atomic with respect to request submission. Any
    /// pending or in-flight console stop is subsumed into the recorded
    /// terminal event, queued PWM and authority are cleared, and all sessions
    /// are invalidated before the ingress becomes disconnected. The checked
    /// zero type cannot carry a nonzero applied duty.
    pub fn disconnect_after_confirmed_zero(
        &mut self,
        observed_applied_zero: QualificationObservedAppliedZero,
    ) -> Result<WheelsOffQualificationTerminalCompletion, WheelsOffQualificationDisconnectError>
    {
        if self.terminal {
            return Err(WheelsOffQualificationDisconnectError::AlreadyDisconnected(
                WheelsOffQualificationRuntimeIngressState::DisconnectedStopConfirmed,
            ));
        }
        let mut state = self.lock_state();
        if state.runtime_ingress_state != WheelsOffQualificationRuntimeIngressState::Connected {
            let ingress_state = state.runtime_ingress_state;
            drop(state);
            self.terminal = true;
            return Err(WheelsOffQualificationDisconnectError::AlreadyDisconnected(
                ingress_state,
            ));
        }

        state.sessions.clear();
        state.pending_pwm = None;
        state.authority = None;

        let mut event = match (
            state.pending_terminal.take(),
            state.terminal_in_flight.take(),
        ) {
            (Some(mut pending), Some(in_flight)) => {
                pending.causes.0 |= in_flight.causes.0;
                pending.first_received_at_host_monotonic_ns = match (
                    pending.first_received_at_host_monotonic_ns,
                    in_flight.first_received_at_host_monotonic_ns,
                ) {
                    (Some(left), Some(right)) => Some(left.min(right)),
                    (left, right) => left.or(right),
                };
                pending.latest_received_at_host_monotonic_ns = match (
                    pending.latest_received_at_host_monotonic_ns,
                    in_flight.latest_received_at_host_monotonic_ns,
                ) {
                    (Some(left), Some(right)) => Some(left.max(right)),
                    (left, right) => left.or(right),
                };
                pending
            }
            (Some(pending), None) => pending,
            (None, Some(in_flight)) => in_flight,
            (None, None) => {
                let event_id = match state.allocate_event() {
                    Ok(event_id) => event_id,
                    Err(_) => {
                        state.runtime_ingress_state =
                            WheelsOffQualificationRuntimeIngressState::DisconnectedStopUnconfirmed;
                        state.software_safety_stop_latched = true;
                        state.bump_revision();
                        drop(state);
                        self.terminal = true;
                        return Err(WheelsOffQualificationDisconnectError::EventIdentityExhausted);
                    }
                };
                CandidateMotionTerminalEvent {
                    event_id,
                    authority_generation: None,
                    source_session_id: None,
                    causes: CandidateMotionTerminalCauses::one(
                        CandidateMotionTerminalCause::RuntimeReceiverDisconnected,
                    ),
                    first_received_at_host_monotonic_ns: None,
                    latest_received_at_host_monotonic_ns: None,
                }
            }
        };
        event
            .causes
            .insert(CandidateMotionTerminalCause::RuntimeReceiverDisconnected);
        let completion = WheelsOffQualificationTerminalCompletion {
            event_id: event.event_id,
            observed_applied_zero,
        };
        state.last_terminal_stop = Some(event);
        state.last_terminal_completion = Some(completion);
        state.runtime_ingress_state =
            WheelsOffQualificationRuntimeIngressState::DisconnectedStopConfirmed;
        state.bump_revision();
        drop(state);
        self.terminal = true;
        Ok(completion)
    }
}

impl Drop for WheelsOffQualificationIngressReceiver {
    fn drop(&mut self) {
        if self.terminal {
            return;
        }
        let mut state = self.lock_state();
        state.runtime_ingress_state =
            WheelsOffQualificationRuntimeIngressState::DisconnectedStopUnconfirmed;
        state.sessions.clear();
        state.software_safety_stop_latched = true;
        state.pending_pwm = None;
        state.authority = None;
        if state.pending_terminal.is_none() {
            let _ = state.queue_terminal(
                CandidateMotionTerminalCause::RuntimeReceiverDisconnected,
                None,
                None,
            );
        } else if let Some(pending) = state.pending_terminal.as_mut() {
            pending
                .causes
                .insert(CandidateMotionTerminalCause::RuntimeReceiverDisconnected);
            state.last_terminal_stop = Some(*pending);
        }
        state.bump_revision();
        drop(state);
        self.terminal = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> WheelsOffQualificationControlProfile {
        WheelsOffQualificationControlProfile::parse(30, 10, 250).unwrap()
    }

    fn capability(byte: u8) -> ConsoleSessionCapability {
        ConsoleSessionCapability::from_bytes([byte; 32])
    }

    fn request(
        session_id: ConsoleSessionId,
        sequence: u64,
        idempotency: u64,
        intent: WheelsOffQualificationIntent,
    ) -> WheelsOffQualificationRequest {
        WheelsOffQualificationRequest {
            session_id,
            source_sequence: ConsoleSourceSequence::parse(sequence).unwrap(),
            idempotency_key: ConsoleIdempotencyKey::parse(idempotency).unwrap(),
            intent,
        }
    }

    fn pwm(left: i16, right: i16) -> QualificationTimerPwmPair {
        QualificationTimerPwmPair::parse(left, right, profile()).unwrap()
    }

    #[test]
    fn profile_is_explicitly_wheels_off_raw_and_shadow_only() {
        let value = serde_json::to_value(profile()).unwrap();
        assert_eq!(value["kind"], "wheels_off_raw_timer_pwm_qualification");
        assert_eq!(value["command_units"], "signed_timer_duty_percent");
        assert_eq!(value["required_wheel_state"], "removed");
        assert_eq!(value["autonomous_actuation"], "disabled_shadow_only");
        assert_eq!(value["banner"], WHEELS_OFF_QUALIFICATION_BANNER);
        assert_eq!(
            value["intent_endpoint"],
            WHEELS_OFF_QUALIFICATION_INTENT_ENDPOINT
        );
        assert_eq!(value["maximum_abs_timer_pwm_percent"], 30);
        assert_eq!(
            value["patterns"]["both_positive"]["left_timer_pwm_percent"],
            10
        );
    }

    #[test]
    fn parser_rejects_zero_pair_and_out_of_profile_magnitude() {
        assert_eq!(
            QualificationTimerPwmPair::parse(0, 0, profile()),
            Err(QualificationTimerPwmPairParseError::ZeroMustUseStopIntent)
        );
        assert!(matches!(
            QualificationTimerPwmPair::parse(31, 0, profile()),
            Err(QualificationTimerPwmPairParseError::Left(
                QualificationTimerPwmParseError::OutsideAdmittedMagnitude {
                    actual: 31,
                    maximum: 30
                }
            ))
        ));
    }

    #[test]
    fn qualification_and_production_weak_schemas_reject_each_others_motion_kind() {
        let qualification_raw = br#"{
            "schema_version":1,
            "control_profile":"wheels_off_raw_timer_pwm_qualification",
            "session_id":"1",
            "source_sequence":"1",
            "idempotency_key":"1",
            "intent":{"kind":"manual_pwm","left_timer_pwm_percent":5,"right_timer_pwm_percent":5}
        }"#;
        assert!(
            serde_json::from_slice::<super::super::operator_console::ConsoleIntentRequestDto>(
                qualification_raw
            )
            .is_err()
        );

        let production_raw = br#"{
            "schema_version":1,
            "session_id":"1",
            "source_sequence":"1",
            "idempotency_key":"1",
            "intent":{"kind":"manual_velocity","forward_velocity_mps":0.1,"yaw_rate_rad_s":0.0}
        }"#;
        assert!(
            serde_json::from_slice::<WheelsOffQualificationIntentRequestDto>(production_raw)
                .is_err()
        );
    }

    #[test]
    fn dto_parses_once_and_requires_canonical_decimal_identities() {
        let valid = br#"{
            "schema_version":1,
            "control_profile":"wheels_off_raw_timer_pwm_qualification",
            "session_id":"1",
            "source_sequence":"2",
            "idempotency_key":"3",
            "intent":{"kind":"manual_pwm","left_timer_pwm_percent":-5,"right_timer_pwm_percent":5}
        }"#;
        let parsed = serde_json::from_slice::<WheelsOffQualificationIntentRequestDto>(valid)
            .unwrap()
            .parse(profile())
            .unwrap();
        assert_eq!(
            parsed.intent,
            WheelsOffQualificationIntent::ManualPwm(pwm(-5, 5))
        );

        let noncanonical = br#"{
            "schema_version":1,
            "control_profile":"wheels_off_raw_timer_pwm_qualification",
            "session_id":"01",
            "source_sequence":"2",
            "idempotency_key":"3",
            "intent":{"kind":"stop"}
        }"#;
        assert_eq!(
            serde_json::from_slice::<WheelsOffQualificationIntentRequestDto>(noncanonical)
                .unwrap()
                .parse(profile()),
            Err(WheelsOffQualificationRequestParseError::NonCanonicalDecimalIdentity)
        );
    }

    #[test]
    fn operator_and_agent_sessions_share_one_authority_arbiter() {
        let (console, _receiver) = wheels_off_qualification_console(profile());
        let operator = console
            .open_session(ConsoleSourceKind::Operator, capability(1))
            .unwrap();
        let agent = console
            .open_session(ConsoleSourceKind::Agent, capability(2))
            .unwrap();
        console
            .submit(
                request(operator, 1, 1, WheelsOffQualificationIntent::BeginManual),
                capability(1),
                HostMonotonicTimestamp::from_nanos(10),
            )
            .unwrap();
        assert_eq!(
            console.submit(
                request(agent, 1, 1, WheelsOffQualificationIntent::BeginManual),
                capability(2),
                HostMonotonicTimestamp::from_nanos(11),
            ),
            Err(WheelsOffQualificationSubmitError::AuthorityConflict {
                requested_by: agent,
                held_by: operator,
            })
        );
    }

    #[test]
    fn latest_pwm_is_bounded_and_stop_preempts_it() {
        let (console, mut receiver) = wheels_off_qualification_console(profile());
        let session = console
            .open_session(ConsoleSourceKind::Operator, capability(1))
            .unwrap();
        console
            .submit(
                request(session, 1, 1, WheelsOffQualificationIntent::BeginManual),
                capability(1),
                HostMonotonicTimestamp::from_nanos(1),
            )
            .unwrap();
        let first = console
            .submit(
                request(
                    session,
                    2,
                    2,
                    WheelsOffQualificationIntent::ManualPwm(pwm(5, 5)),
                ),
                capability(1),
                HostMonotonicTimestamp::from_nanos(2),
            )
            .unwrap();
        let second = console
            .submit(
                request(
                    session,
                    3,
                    3,
                    WheelsOffQualificationIntent::ManualPwm(pwm(10, 10)),
                ),
                capability(1),
                HostMonotonicTimestamp::from_nanos(3),
            )
            .unwrap();
        assert_eq!(
            second,
            WheelsOffQualificationSubmitOutcome::CandidatePwmQueued {
                event_id: second.event_id(),
                superseded_event_id: Some(first.event_id()),
            }
        );
        let stop = console
            .submit(
                request(session, 4, 4, WheelsOffQualificationIntent::ReleaseManual),
                capability(1),
                HostMonotonicTimestamp::from_nanos(4),
            )
            .unwrap();
        assert_eq!(
            receiver.try_recv(),
            Ok(WheelsOffQualificationIngressEvent::TerminalStop(
                console.snapshot().last_terminal_stop.unwrap()
            ))
        );
        assert_eq!(
            receiver.try_recv(),
            Err(WheelsOffQualificationReceiveError::Empty)
        );
        assert_ne!(stop.event_id(), first.event_id());
    }

    #[test]
    fn stop_barrier_requires_observed_applied_zero_before_reacquisition() {
        let (console, mut receiver) = wheels_off_qualification_console(profile());
        let session = console
            .open_session(ConsoleSourceKind::Operator, capability(1))
            .unwrap();
        console
            .submit(
                request(session, 1, 1, WheelsOffQualificationIntent::BeginManual),
                capability(1),
                HostMonotonicTimestamp::from_nanos(1),
            )
            .unwrap();
        let stop = console
            .submit(
                request(session, 2, 2, WheelsOffQualificationIntent::ReleaseManual),
                capability(1),
                HostMonotonicTimestamp::from_nanos(2),
            )
            .unwrap();
        let event = receiver.try_recv().unwrap();
        assert!(matches!(
            event,
            WheelsOffQualificationIngressEvent::TerminalStop(_)
        ));
        assert_eq!(
            console.submit(
                request(session, 3, 3, WheelsOffQualificationIntent::BeginManual),
                capability(1),
                HostMonotonicTimestamp::from_nanos(3),
            ),
            Err(WheelsOffQualificationSubmitError::StopBarrierPending)
        );
        assert!(QualificationObservedAppliedZero::parse(9, 1, 0).is_err());
        receiver
            .complete_terminal_stop(
                stop.event_id(),
                QualificationObservedAppliedZero::parse(9, 0, 0).unwrap(),
            )
            .unwrap();
        console
            .submit(
                request(session, 3, 3, WheelsOffQualificationIntent::BeginManual),
                capability(1),
                HostMonotonicTimestamp::from_nanos(3),
            )
            .unwrap();
    }

    #[test]
    fn deadman_connection_loss_and_safety_stop_are_typed_terminal_events() {
        let (console, mut receiver) = wheels_off_qualification_console(profile());
        let session = console
            .open_session(ConsoleSourceKind::Agent, capability(1))
            .unwrap();
        console
            .submit(
                request(session, 1, 1, WheelsOffQualificationIntent::BeginManual),
                capability(1),
                HostMonotonicTimestamp::from_nanos(1),
            )
            .unwrap();
        assert!(
            console
                .tick_deadman(HostMonotonicTimestamp::from_nanos(250_000_001))
                .unwrap()
        );
        let deadman = match receiver.try_recv().unwrap() {
            WheelsOffQualificationIngressEvent::TerminalStop(stop) => stop,
            WheelsOffQualificationIngressEvent::CandidatePwm(_) => panic!("stop must preempt PWM"),
        };
        assert!(
            deadman
                .causes()
                .contains(CandidateMotionTerminalCause::ManualDeadmanExpired)
        );
        receiver
            .complete_terminal_stop(
                deadman.event_id(),
                QualificationObservedAppliedZero::parse(1, 0, 0).unwrap(),
            )
            .unwrap();

        console
            .submit(
                request(session, 2, 2, WheelsOffQualificationIntent::BeginManual),
                capability(1),
                HostMonotonicTimestamp::from_nanos(300_000_000),
            )
            .unwrap();
        assert!(
            console
                .report_connection_lost(
                    session,
                    capability(1),
                    HostMonotonicTimestamp::from_nanos(300_000_001),
                )
                .unwrap()
        );
        let disconnected = match receiver.try_recv().unwrap() {
            WheelsOffQualificationIngressEvent::TerminalStop(stop) => stop,
            WheelsOffQualificationIngressEvent::CandidatePwm(_) => panic!("expected stop"),
        };
        assert!(
            disconnected
                .causes()
                .contains(CandidateMotionTerminalCause::ConnectionLost)
        );
        receiver
            .complete_terminal_stop(
                disconnected.event_id(),
                QualificationObservedAppliedZero::parse(2, 0, 0).unwrap(),
            )
            .unwrap();

        let safety_session = console
            .open_session(ConsoleSourceKind::Operator, capability(2))
            .unwrap();
        console
            .submit(
                request(
                    safety_session,
                    1,
                    1,
                    WheelsOffQualificationIntent::SoftwareSafetyStop,
                ),
                capability(2),
                HostMonotonicTimestamp::from_nanos(400_000_000),
            )
            .unwrap();
        let safety = match receiver.try_recv().unwrap() {
            WheelsOffQualificationIngressEvent::TerminalStop(stop) => stop,
            WheelsOffQualificationIngressEvent::CandidatePwm(_) => panic!("expected stop"),
        };
        assert!(
            safety
                .causes()
                .contains(CandidateMotionTerminalCause::SoftwareSafetyStop)
        );
        assert!(console.snapshot().software_safety_stop_latched);
    }

    #[test]
    fn idempotency_replay_cannot_change_the_raw_pwm_pair() {
        let (console, _receiver) = wheels_off_qualification_console(profile());
        let session = console
            .open_session(ConsoleSourceKind::Operator, capability(1))
            .unwrap();
        console
            .submit(
                request(session, 1, 1, WheelsOffQualificationIntent::BeginManual),
                capability(1),
                HostMonotonicTimestamp::from_nanos(1),
            )
            .unwrap();
        let first = console
            .submit(
                request(
                    session,
                    2,
                    2,
                    WheelsOffQualificationIntent::ManualPwm(pwm(5, 5)),
                ),
                capability(1),
                HostMonotonicTimestamp::from_nanos(2),
            )
            .unwrap();
        assert_eq!(
            console
                .submit(
                    request(
                        session,
                        2,
                        2,
                        WheelsOffQualificationIntent::ManualPwm(pwm(5, 5)),
                    ),
                    capability(1),
                    HostMonotonicTimestamp::from_nanos(3),
                )
                .unwrap(),
            WheelsOffQualificationSubmitOutcome::IdempotentReplay {
                event_id: first.event_id()
            }
        );
        assert_eq!(
            console.submit(
                request(
                    session,
                    3,
                    2,
                    WheelsOffQualificationIntent::ManualPwm(pwm(6, 6)),
                ),
                capability(1),
                HostMonotonicTimestamp::from_nanos(4),
            ),
            Err(WheelsOffQualificationSubmitError::IdempotencyConflict(
                ConsoleIdempotencyKey::parse(2).unwrap()
            ))
        );
    }

    #[test]
    fn dropping_sole_receiver_latches_and_rejects_future_motion() {
        let (console, receiver) = wheels_off_qualification_console(profile());
        let session = console
            .open_session(ConsoleSourceKind::Operator, capability(1))
            .unwrap();
        drop(receiver);
        assert_eq!(
            console.submit(
                request(session, 1, 1, WheelsOffQualificationIntent::BeginManual),
                capability(1),
                HostMonotonicTimestamp::from_nanos(1),
            ),
            Err(WheelsOffQualificationSubmitError::RuntimeReceiverDisconnected)
        );
        let snapshot = console.snapshot();
        assert!(snapshot.software_safety_stop_latched);
        assert_eq!(
            snapshot.runtime_ingress_state,
            WheelsOffQualificationRuntimeIngressState::DisconnectedStopUnconfirmed
        );
        assert!(
            snapshot
                .last_terminal_stop
                .unwrap()
                .causes()
                .contains(CandidateMotionTerminalCause::RuntimeReceiverDisconnected)
        );
    }

    #[test]
    fn confirmed_direct_host_stop_disconnects_without_false_unconfirmed_latch() {
        let (console, mut receiver) = wheels_off_qualification_console(profile());
        let session = console
            .open_session(ConsoleSourceKind::Agent, capability(1))
            .unwrap();
        console
            .submit(
                request(session, 1, 1, WheelsOffQualificationIntent::BeginManual),
                capability(1),
                HostMonotonicTimestamp::from_nanos(1),
            )
            .unwrap();
        console
            .submit(
                request(
                    session,
                    2,
                    2,
                    WheelsOffQualificationIntent::ManualPwm(pwm(10, -10)),
                ),
                capability(1),
                HostMonotonicTimestamp::from_nanos(2),
            )
            .unwrap();

        let observed_zero = QualificationObservedAppliedZero::parse(42, 0, 0).unwrap();
        let completion = receiver
            .disconnect_after_confirmed_zero(observed_zero)
            .unwrap();
        assert_eq!(completion.observed_applied_zero, observed_zero);
        assert_eq!(
            receiver.try_recv(),
            Err(WheelsOffQualificationReceiveError::RuntimeReceiverDisconnected)
        );
        assert!(!console.session_capability_matches(session, capability(1)));
        assert_eq!(
            console.submit(
                request(
                    session,
                    3,
                    3,
                    WheelsOffQualificationIntent::ManualPwm(pwm(5, 5))
                ),
                capability(1),
                HostMonotonicTimestamp::from_nanos(3),
            ),
            Err(WheelsOffQualificationSubmitError::RuntimeReceiverDisconnected)
        );

        let snapshot = console.snapshot();
        assert_eq!(
            snapshot.runtime_ingress_state,
            WheelsOffQualificationRuntimeIngressState::DisconnectedStopConfirmed
        );
        assert!(!snapshot.stop_barrier_pending);
        assert!(!snapshot.software_safety_stop_latched);
        assert_eq!(snapshot.last_terminal_completion, Some(completion));
        assert!(
            snapshot
                .last_terminal_stop
                .unwrap()
                .causes()
                .contains(CandidateMotionTerminalCause::RuntimeReceiverDisconnected)
        );
        drop(receiver);
        assert_eq!(
            console.snapshot().runtime_ingress_state,
            WheelsOffQualificationRuntimeIngressState::DisconnectedStopConfirmed
        );
    }
}
