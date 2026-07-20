#![no_std]
#![forbid(unsafe_code)]

//! Strict, allocation-free Kiko host-to-eye-firmware protocol.

use core::fmt;
use core::num::{NonZeroU16, NonZeroU32, NonZeroU64};

pub const PROTOCOL_VERSION: u8 = 2;
pub const MAX_INTENT_LEASE_MS: u16 = 2_000;
pub const MIN_INTENT_LEASE_MS: u16 = 20;
pub const NORMALIZED_SCALE: i16 = 1_000;
const NORMALIZED_SCALE_U16: u16 = 1_000;
pub const HEADER_BYTES: usize = 8;
pub const CHECKSUM_BYTES: usize = 4;
pub const MAX_PAYLOAD_BYTES: usize = 80;
pub const MAX_RAW_FRAME_BYTES: usize = HEADER_BYTES + MAX_PAYLOAD_BYTES + CHECKSUM_BYTES;
pub const MAX_ENCODED_FRAME_BYTES: usize = MAX_RAW_FRAME_BYTES + MAX_RAW_FRAME_BYTES / 254 + 2;

const MAGIC: [u8; 2] = *b"KE";

/// Non-zero challenge bound to exactly one identity or acquisition exchange.
///
/// Zero is excluded so a cleared or never-initialized challenge cannot become
/// an encodable KEP2 message.
///
/// ```compile_fail
/// use core::num::NonZeroU64;
/// use kiko_eye_protocol::HandshakeNonce;
///
/// let value = NonZeroU64::new(1).unwrap();
/// let _ = HandshakeNonce(value);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HandshakeNonce(NonZeroU64);

impl HandshakeNonce {
    pub const fn try_new(value: u64) -> Result<Self, DomainError> {
        match NonZeroU64::new(value) {
            Some(value) => Ok(Self::from_nonzero(value)),
            None => Err(DomainError::ZeroHandshakeNonce),
        }
    }

    pub const fn from_nonzero(value: NonZeroU64) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceUid([u8; 16]);

impl DeviceUid {
    pub fn try_new(bytes: [u8; 16]) -> Result<Self, DomainError> {
        if bytes == [0; 16] {
            Err(DomainError::ZeroDeviceUid)
        } else {
            Ok(Self(bytes))
        }
    }

    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FirmwareBuildId([u8; 32]);

impl FirmwareBuildId {
    pub fn try_new(bytes: [u8; 32]) -> Result<Self, DomainError> {
        if bytes == [0; 32] {
            Err(DomainError::ZeroFirmwareBuildId)
        } else {
            Ok(Self(bytes))
        }
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceBootId(NonZeroU64);

impl DeviceBootId {
    pub fn try_new(value: u64) -> Result<Self, DomainError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(DomainError::ZeroDeviceBootId)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Milliseconds on the reporting device's monotonic clock since this boot.
///
/// This is deliberately distinct from host timestamps and durations. Zero is
/// valid at the start of a boot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceTimestampMs(u64);

impl DeviceTimestampMs {
    pub const ZERO: Self = Self(0);

    pub const fn from_millis_since_boot(value: u64) -> Self {
        Self(value)
    }

    pub const fn millis_since_boot(self) -> u64 {
        self.0
    }

    pub const fn checked_millis_since(self, earlier: Self) -> Option<u64> {
        self.0.checked_sub(earlier.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ControlEpoch(NonZeroU32);

impl ControlEpoch {
    pub fn try_new(value: u32) -> Result<Self, DomainError> {
        NonZeroU32::new(value)
            .map(Self)
            .ok_or(DomainError::ZeroControlEpoch)
    }

    pub const fn get(self) -> u32 {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IntentSequence(u32);

impl IntentSequence {
    pub const FIRST: Self = Self(0);

    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }

    pub const fn checked_successor(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

/// Wrapping sequence of frames admitted by the eye renderer.
///
/// This counter has RFC-1982-style serial-number semantics at the host session
/// layer and is not interchangeable with [`IntentSequence`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RenderedFrameSequence(u32);

impl RenderedFrameSequence {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntentLeaseMs(NonZeroU16);

impl IntentLeaseMs {
    pub fn try_new(value: u16) -> Result<Self, DomainError> {
        let Some(value) = NonZeroU16::new(value) else {
            return Err(DomainError::IntentLeaseOutOfRange {
                value: 0,
                minimum: MIN_INTENT_LEASE_MS,
                maximum: MAX_INTENT_LEASE_MS,
            });
        };
        if !(MIN_INTENT_LEASE_MS..=MAX_INTENT_LEASE_MS).contains(&value.get()) {
            return Err(DomainError::IntentLeaseOutOfRange {
                value: value.get(),
                minimum: MIN_INTENT_LEASE_MS,
                maximum: MAX_INTENT_LEASE_MS,
            });
        }
        Ok(Self(value))
    }

    pub const fn get(self) -> u16 {
        self.0.get()
    }
}

/// Signed normalized coordinate in the inclusive range `[-1000, 1000]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignedUnit(i16);

impl SignedUnit {
    pub const ZERO: Self = Self(0);

    pub fn try_new(value: i16) -> Result<Self, DomainError> {
        if (-NORMALIZED_SCALE..=NORMALIZED_SCALE).contains(&value) {
            Ok(Self(value))
        } else {
            Err(DomainError::SignedUnitOutOfRange { value })
        }
    }

    pub const fn get(self) -> i16 {
        self.0
    }
}

/// Unsigned normalized amount in the inclusive range `[0, 1000]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitAmount(u16);

impl UnitAmount {
    pub const ZERO: Self = Self(0);
    pub const FULL: Self = Self(NORMALIZED_SCALE_U16);

    pub fn try_new(value: u16) -> Result<Self, DomainError> {
        if value <= NORMALIZED_SCALE_U16 {
            Ok(Self(value))
        } else {
            Err(DomainError::UnitAmountOutOfRange { value })
        }
    }

    pub const fn get(self) -> u16 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Expression {
    Neutral = 0,
    Curious = 1,
    Greet = 2,
    Concerned = 3,
    Sleepy = 4,
}

impl Expression {
    fn parse(value: u8) -> Result<Self, PayloadError> {
        match value {
            0 => Ok(Self::Neutral),
            1 => Ok(Self::Curious),
            2 => Ok(Self::Greet),
            3 => Ok(Self::Concerned),
            4 => Ok(Self::Sleepy),
            _ => Err(PayloadError::UnknownEnum {
                field: "expression",
                value,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EyeFlags(u8);

impl EyeFlags {
    pub const NONE: Self = Self(0);
    pub const BLINK: u8 = 1 << 0;
    pub const KNOWN_BITS: u8 = Self::BLINK;

    pub fn try_from_bits(bits: u8) -> Result<Self, DomainError> {
        if bits & !Self::KNOWN_BITS == 0 {
            Ok(Self(bits))
        } else {
            Err(DomainError::UnknownEyeFlagBits { bits })
        }
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn requests_blink(self) -> bool {
        self.0 & Self::BLINK != 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities(u32);

impl Capabilities {
    pub const GAZE: u32 = 1 << 0;
    pub const LID: u32 = 1 << 1;
    pub const PUPIL: u32 = 1 << 2;
    pub const COLOR: u32 = 1 << 3;
    pub const BRIGHTNESS: u32 = 1 << 4;
    pub const BLINK: u32 = 1 << 5;
    pub const AUTONOMOUS_FALLBACK: u32 = 1 << 6;
    pub const APPLIED_REPORT: u32 = 1 << 7;
    pub const KNOWN_BITS: u32 = (1 << 8) - 1;

    pub fn try_from_bits(bits: u32) -> Result<Self, DomainError> {
        if bits & !Self::KNOWN_BITS == 0 {
            Ok(Self(bits))
        } else {
            Err(DomainError::UnknownCapabilityBits { bits })
        }
    }

    pub const fn bits(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EyeIntent {
    gaze_x: SignedUnit,
    gaze_y: SignedUnit,
    lid: UnitAmount,
    pupil: UnitAmount,
    brightness: UnitAmount,
    expression: Expression,
    flags: EyeFlags,
    color_rgb: [u8; 3],
}

impl EyeIntent {
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        gaze_x: SignedUnit,
        gaze_y: SignedUnit,
        lid: UnitAmount,
        pupil: UnitAmount,
        brightness: UnitAmount,
        expression: Expression,
        flags: EyeFlags,
        color_rgb: [u8; 3],
    ) -> Self {
        Self {
            gaze_x,
            gaze_y,
            lid,
            pupil,
            brightness,
            expression,
            flags,
            color_rgb,
        }
    }

    pub const fn gaze_x(self) -> SignedUnit {
        self.gaze_x
    }
    pub const fn gaze_y(self) -> SignedUnit {
        self.gaze_y
    }
    pub const fn lid(self) -> UnitAmount {
        self.lid
    }
    pub const fn pupil(self) -> UnitAmount {
        self.pupil
    }
    pub const fn brightness(self) -> UnitAmount {
        self.brightness
    }
    pub const fn expression(self) -> Expression {
        self.expression
    }
    pub const fn flags(self) -> EyeFlags {
        self.flags
    }
    pub const fn color_rgb(self) -> [u8; 3] {
        self.color_rgb
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ApplyIntent {
    pub boot_id: DeviceBootId,
    pub control_epoch: ControlEpoch,
    pub sequence: IntentSequence,
    pub lease: IntentLeaseMs,
    pub intent: EyeIntent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcquireControl {
    pub expected_boot_id: DeviceBootId,
    pub requested_epoch: ControlEpoch,
    pub nonce: HandshakeNonce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AcquireResultCode {
    Granted = 0,
    Busy = 1,
    IdentityMismatch = 2,
    Faulted = 3,
}

impl AcquireResultCode {
    fn parse(value: u8) -> Result<Self, PayloadError> {
        match value {
            0 => Ok(Self::Granted),
            1 => Ok(Self::Busy),
            2 => Ok(Self::IdentityMismatch),
            3 => Ok(Self::Faulted),
            _ => Err(PayloadError::UnknownEnum {
                field: "acquire result",
                value,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AcquireResult {
    pub boot_id: DeviceBootId,
    pub control_epoch: ControlEpoch,
    pub nonce: HandshakeNonce,
    pub result: AcquireResultCode,
    pub device_uptime: DeviceTimestampMs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ReleaseReason {
    Operator = 0,
    HostShutdown = 1,
    PerceptionStale = 2,
    Fault = 3,
}

impl ReleaseReason {
    fn parse(value: u8) -> Result<Self, PayloadError> {
        match value {
            0 => Ok(Self::Operator),
            1 => Ok(Self::HostShutdown),
            2 => Ok(Self::PerceptionStale),
            3 => Ok(Self::Fault),
            _ => Err(PayloadError::UnknownEnum {
                field: "release reason",
                value,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReleaseControl {
    pub boot_id: DeviceBootId,
    pub control_epoch: ControlEpoch,
    pub sequence: IntentSequence,
    pub reason: ReleaseReason,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum IntentResultCode {
    AppliedNew = 0,
    DuplicateCached = 1,
    Released = 2,
    RejectedExpired = 3,
    RejectedSession = 4,
    RejectedSequence = 5,
    RejectedDomain = 6,
    FaultedFallback = 7,
}

impl IntentResultCode {
    fn parse(value: u8) -> Result<Self, PayloadError> {
        match value {
            0 => Ok(Self::AppliedNew),
            1 => Ok(Self::DuplicateCached),
            2 => Ok(Self::Released),
            3 => Ok(Self::RejectedExpired),
            4 => Ok(Self::RejectedSession),
            5 => Ok(Self::RejectedSequence),
            6 => Ok(Self::RejectedDomain),
            7 => Ok(Self::FaultedFallback),
            _ => Err(PayloadError::UnknownEnum {
                field: "intent result",
                value,
            }),
        }
    }

    pub const fn proves_admission(self) -> bool {
        matches!(
            self,
            Self::AppliedNew | Self::DuplicateCached | Self::Released
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IdentityReport {
    pub nonce: HandshakeNonce,
    pub device_uid: DeviceUid,
    pub firmware_build_id: FirmwareBuildId,
    pub boot_id: DeviceBootId,
    pub device_uptime: DeviceTimestampMs,
    pub capabilities: Capabilities,
}

/// Firmware result bound to one exact boot, epoch, and intent sequence.
///
/// The fields are private because the result code and device-clock interval
/// form one invariant: admitted results carry a bounded KEP2 lease, while all
/// other results are instantaneous. Use [`IntentResult::try_new`] to construct
/// a value; decoded frames pass through the same parser exactly once.
///
/// ```compile_fail
/// use kiko_eye_protocol::IntentResult;
///
/// let _ = IntentResult {
///     boot_id: todo!(),
///     control_epoch: todo!(),
///     sequence: todo!(),
///     result: todo!(),
///     applied_at: todo!(),
///     expires_at: todo!(),
///     admitted_lease: todo!(),
///     rendered_frame_sequence: todo!(),
/// };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntentResult {
    boot_id: DeviceBootId,
    control_epoch: ControlEpoch,
    sequence: IntentSequence,
    result: IntentResultCode,
    applied_at: DeviceTimestampMs,
    expires_at: DeviceTimestampMs,
    admitted_lease: Option<IntentLeaseMs>,
    rendered_frame_sequence: RenderedFrameSequence,
}

impl IntentResult {
    pub fn try_new(
        boot_id: DeviceBootId,
        control_epoch: ControlEpoch,
        sequence: IntentSequence,
        result: IntentResultCode,
        applied_at: DeviceTimestampMs,
        expires_at: DeviceTimestampMs,
        rendered_frame_sequence: RenderedFrameSequence,
    ) -> Result<Self, DomainError> {
        let interval_ms = expires_at.checked_millis_since(applied_at);
        let invalid_timing = || DomainError::InvalidIntentResultTiming {
            result,
            applied_at_ms: applied_at.millis_since_boot(),
            expires_at_ms: expires_at.millis_since_boot(),
        };
        let admitted_lease = match result {
            IntentResultCode::AppliedNew | IntentResultCode::DuplicateCached => {
                let interval_ms = interval_ms.ok_or_else(invalid_timing)?;
                let interval_ms = u16::try_from(interval_ms).map_err(|_| invalid_timing())?;
                // Preserve the richer result/timestamp context instead of
                // surfacing the derived lease value as if it came from wire.
                let lease = IntentLeaseMs::try_new(interval_ms).map_err(|_| invalid_timing())?;
                Some(lease)
            }
            IntentResultCode::Released
            | IntentResultCode::RejectedExpired
            | IntentResultCode::RejectedSession
            | IntentResultCode::RejectedSequence
            | IntentResultCode::RejectedDomain
            | IntentResultCode::FaultedFallback => {
                if interval_ms != Some(0) {
                    return Err(invalid_timing());
                }
                None
            }
        };
        Ok(Self {
            boot_id,
            control_epoch,
            sequence,
            result,
            applied_at,
            expires_at,
            admitted_lease,
            rendered_frame_sequence,
        })
    }

    pub const fn boot_id(self) -> DeviceBootId {
        self.boot_id
    }

    pub const fn control_epoch(self) -> ControlEpoch {
        self.control_epoch
    }

    pub const fn sequence(self) -> IntentSequence {
        self.sequence
    }

    pub const fn result(self) -> IntentResultCode {
        self.result
    }

    pub const fn applied_at(self) -> DeviceTimestampMs {
        self.applied_at
    }

    pub const fn expires_at(self) -> DeviceTimestampMs {
        self.expires_at
    }

    /// The checked device-relative lease for admitted results.
    ///
    /// This is `Some` exactly for `AppliedNew` and `DuplicateCached` and `None`
    /// for every instantaneous result.
    pub const fn admitted_lease(self) -> Option<IntentLeaseMs> {
        self.admitted_lease
    }

    pub const fn device_interval_ms(self) -> u16 {
        match self.admitted_lease {
            Some(lease) => lease.get(),
            None => 0,
        }
    }

    pub const fn rendered_frame_sequence(self) -> RenderedFrameSequence {
        self.rendered_frame_sequence
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Message {
    IdentityQuery { nonce: HandshakeNonce },
    IdentityReport(IdentityReport),
    AcquireControl(AcquireControl),
    AcquireResult(AcquireResult),
    ApplyIntent(ApplyIntent),
    ReleaseControl(ReleaseControl),
    IntentResult(IntentResult),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum MessageKind {
    IdentityQuery = 1,
    IdentityReport = 2,
    AcquireControl = 3,
    AcquireResult = 4,
    ApplyIntent = 5,
    ReleaseControl = 6,
    IntentResult = 7,
}

impl MessageKind {
    fn parse(value: u8) -> Result<Self, FrameError> {
        match value {
            1 => Ok(Self::IdentityQuery),
            2 => Ok(Self::IdentityReport),
            3 => Ok(Self::AcquireControl),
            4 => Ok(Self::AcquireResult),
            5 => Ok(Self::ApplyIntent),
            6 => Ok(Self::ReleaseControl),
            7 => Ok(Self::IntentResult),
            _ => Err(FrameError::UnknownMessageKind { value }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainError {
    ZeroHandshakeNonce,
    ZeroDeviceUid,
    ZeroFirmwareBuildId,
    ZeroDeviceBootId,
    ZeroControlEpoch,
    IntentLeaseOutOfRange {
        value: u16,
        minimum: u16,
        maximum: u16,
    },
    SignedUnitOutOfRange {
        value: i16,
    },
    UnitAmountOutOfRange {
        value: u16,
    },
    UnknownEyeFlagBits {
        bits: u8,
    },
    UnknownCapabilityBits {
        bits: u32,
    },
    InvalidIntentResultTiming {
        result: IntentResultCode,
        applied_at_ms: u64,
        expires_at_ms: u64,
    },
}

impl fmt::Display for DomainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid eye-protocol domain value: {self:?}")
    }
}

impl core::error::Error for DomainError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PayloadError {
    Domain(DomainError),
    LengthMismatch {
        expected: usize,
        actual: usize,
    },
    FieldOutOfBounds {
        offset: usize,
        width: usize,
        payload_length: usize,
    },
    UnknownEnum {
        field: &'static str,
        value: u8,
    },
    ReservedNonzero {
        offset: usize,
        value: u8,
    },
}

impl From<DomainError> for PayloadError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameError {
    EmptyRecord,
    EncodedRecordTooLong {
        observed_at_least: usize,
        maximum: usize,
    },
    CobsMalformed,
    RawFrameTooShort {
        length: usize,
        minimum: usize,
    },
    BadMagic {
        actual: [u8; 2],
    },
    UnsupportedVersion {
        actual: u8,
    },
    UnknownMessageKind {
        value: u8,
    },
    HeaderReservedNonzero {
        bytes: [u8; 2],
    },
    PayloadTooLong {
        length: usize,
        maximum: usize,
    },
    LengthMismatch {
        declared_payload: usize,
        actual_frame: usize,
    },
    ChecksumMismatch {
        expected: u32,
        actual: u32,
    },
    Payload(PayloadError),
}

impl fmt::Display for FrameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid eye-protocol frame: {self:?}")
    }
}

impl core::error::Error for FrameError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncodeError {
    OutputTooSmall { required: usize, available: usize },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "cannot encode eye-protocol frame: {self:?}")
    }
}

impl core::error::Error for EncodeError {}

pub fn encode(message: Message, output: &mut [u8]) -> Result<usize, EncodeError> {
    let mut raw = [0_u8; MAX_RAW_FRAME_BYTES];
    raw[..2].copy_from_slice(&MAGIC);
    raw[2] = PROTOCOL_VERSION;
    let (kind, payload_len) = encode_payload(message, &mut raw[HEADER_BYTES..]);
    let payload_len_bytes = usize::from(payload_len);
    raw[3] = kind as u8;
    raw[4..6].copy_from_slice(&payload_len.to_le_bytes());
    raw[6..8].fill(0);
    let checksum_offset = HEADER_BYTES + payload_len_bytes;
    let checksum = crc32c(&raw[..checksum_offset]);
    raw[checksum_offset..checksum_offset + CHECKSUM_BYTES].copy_from_slice(&checksum.to_le_bytes());
    cobs_encode(&raw[..checksum_offset + CHECKSUM_BYTES], output)
}

pub fn decode(encoded_record: &[u8]) -> Result<Message, FrameError> {
    if encoded_record.is_empty() {
        return Err(FrameError::EmptyRecord);
    }
    if encoded_record.len() > MAX_ENCODED_FRAME_BYTES - 1 {
        return Err(FrameError::EncodedRecordTooLong {
            observed_at_least: encoded_record.len(),
            maximum: MAX_ENCODED_FRAME_BYTES - 1,
        });
    }
    let mut raw = [0_u8; MAX_RAW_FRAME_BYTES];
    let raw_len = cobs_decode(encoded_record, &mut raw).map_err(|_| FrameError::CobsMalformed)?;
    if raw_len < HEADER_BYTES + CHECKSUM_BYTES {
        return Err(FrameError::RawFrameTooShort {
            length: raw_len,
            minimum: HEADER_BYTES + CHECKSUM_BYTES,
        });
    }
    if raw[..2] != MAGIC {
        return Err(FrameError::BadMagic {
            actual: [raw[0], raw[1]],
        });
    }
    if raw[2] != PROTOCOL_VERSION {
        return Err(FrameError::UnsupportedVersion { actual: raw[2] });
    }
    let kind = MessageKind::parse(raw[3])?;
    if raw[6] != 0 || raw[7] != 0 {
        return Err(FrameError::HeaderReservedNonzero {
            bytes: [raw[6], raw[7]],
        });
    }
    let payload_len = usize::from(u16::from_le_bytes([raw[4], raw[5]]));
    if payload_len > MAX_PAYLOAD_BYTES {
        return Err(FrameError::PayloadTooLong {
            length: payload_len,
            maximum: MAX_PAYLOAD_BYTES,
        });
    }
    let expected_len = HEADER_BYTES + payload_len + CHECKSUM_BYTES;
    if raw_len != expected_len {
        return Err(FrameError::LengthMismatch {
            declared_payload: payload_len,
            actual_frame: raw_len,
        });
    }
    let checksum_offset = HEADER_BYTES + payload_len;
    let Some(checksum_bytes) = raw
        .get(checksum_offset..raw_len)
        .and_then(|bytes| <[u8; CHECKSUM_BYTES]>::try_from(bytes).ok())
    else {
        return Err(FrameError::LengthMismatch {
            declared_payload: payload_len,
            actual_frame: raw_len,
        });
    };
    let actual = u32::from_le_bytes(checksum_bytes);
    let expected = crc32c(&raw[..checksum_offset]);
    if actual != expected {
        return Err(FrameError::ChecksumMismatch { expected, actual });
    }
    decode_payload(kind, &raw[HEADER_BYTES..checksum_offset]).map_err(FrameError::Payload)
}

fn encode_payload(message: Message, output: &mut [u8]) -> (MessageKind, u16) {
    match message {
        Message::IdentityQuery { nonce } => {
            put_u64(output, 0, nonce.get());
            (MessageKind::IdentityQuery, 8)
        }
        Message::IdentityReport(report) => {
            put_u64(output, 0, report.nonce.get());
            output[8..24].copy_from_slice(report.device_uid.as_bytes());
            output[24..56].copy_from_slice(report.firmware_build_id.as_bytes());
            put_u64(output, 56, report.boot_id.get());
            put_u64(output, 64, report.device_uptime.millis_since_boot());
            put_u32(output, 72, report.capabilities.bits());
            (MessageKind::IdentityReport, 76)
        }
        Message::AcquireControl(acquire) => {
            put_u64(output, 0, acquire.expected_boot_id.get());
            put_u32(output, 8, acquire.requested_epoch.get());
            put_u64(output, 12, acquire.nonce.get());
            (MessageKind::AcquireControl, 20)
        }
        Message::AcquireResult(result) => {
            put_u64(output, 0, result.boot_id.get());
            put_u32(output, 8, result.control_epoch.get());
            put_u64(output, 12, result.nonce.get());
            output[20] = result.result as u8;
            output[21..24].fill(0);
            put_u64(output, 24, result.device_uptime.millis_since_boot());
            (MessageKind::AcquireResult, 32)
        }
        Message::ApplyIntent(command) => {
            put_u64(output, 0, command.boot_id.get());
            put_u32(output, 8, command.control_epoch.get());
            put_u32(output, 12, command.sequence.get());
            put_u16(output, 16, command.lease.get());
            put_i16(output, 18, command.intent.gaze_x().get());
            put_i16(output, 20, command.intent.gaze_y().get());
            put_u16(output, 22, command.intent.lid().get());
            put_u16(output, 24, command.intent.pupil().get());
            put_u16(output, 26, command.intent.brightness().get());
            output[28] = command.intent.expression() as u8;
            output[29] = command.intent.flags().bits();
            output[30..33].copy_from_slice(&command.intent.color_rgb());
            output[33] = 0;
            (MessageKind::ApplyIntent, 34)
        }
        Message::ReleaseControl(release) => {
            put_u64(output, 0, release.boot_id.get());
            put_u32(output, 8, release.control_epoch.get());
            put_u32(output, 12, release.sequence.get());
            output[16] = release.reason as u8;
            output[17..20].fill(0);
            (MessageKind::ReleaseControl, 20)
        }
        Message::IntentResult(result) => {
            put_u64(output, 0, result.boot_id().get());
            put_u32(output, 8, result.control_epoch().get());
            put_u32(output, 12, result.sequence().get());
            output[16] = result.result() as u8;
            output[17..20].fill(0);
            put_u64(output, 20, result.applied_at().millis_since_boot());
            put_u64(output, 28, result.expires_at().millis_since_boot());
            put_u32(output, 36, result.rendered_frame_sequence().get());
            (MessageKind::IntentResult, 40)
        }
    }
}

fn decode_payload(kind: MessageKind, payload: &[u8]) -> Result<Message, PayloadError> {
    match kind {
        MessageKind::IdentityQuery => {
            require_len(payload, 8)?;
            Ok(Message::IdentityQuery {
                nonce: HandshakeNonce::try_new(read_u64(payload, 0)?)?,
            })
        }
        MessageKind::IdentityReport => {
            require_len(payload, 76)?;
            let device_uid = DeviceUid::try_new(read_array(payload, 8)?)?;
            let firmware_build_id = FirmwareBuildId::try_new(read_array(payload, 24)?)?;
            Ok(Message::IdentityReport(IdentityReport {
                nonce: HandshakeNonce::try_new(read_u64(payload, 0)?)?,
                device_uid,
                firmware_build_id,
                boot_id: DeviceBootId::try_new(read_u64(payload, 56)?)?,
                device_uptime: DeviceTimestampMs::from_millis_since_boot(read_u64(payload, 64)?),
                capabilities: Capabilities::try_from_bits(read_u32(payload, 72)?)?,
            }))
        }
        MessageKind::AcquireControl => {
            require_len(payload, 20)?;
            Ok(Message::AcquireControl(AcquireControl {
                expected_boot_id: DeviceBootId::try_new(read_u64(payload, 0)?)?,
                requested_epoch: ControlEpoch::try_new(read_u32(payload, 8)?)?,
                nonce: HandshakeNonce::try_new(read_u64(payload, 12)?)?,
            }))
        }
        MessageKind::AcquireResult => {
            require_len(payload, 32)?;
            require_reserved_zero(payload, 21..24)?;
            Ok(Message::AcquireResult(AcquireResult {
                boot_id: DeviceBootId::try_new(read_u64(payload, 0)?)?,
                control_epoch: ControlEpoch::try_new(read_u32(payload, 8)?)?,
                nonce: HandshakeNonce::try_new(read_u64(payload, 12)?)?,
                result: AcquireResultCode::parse(read_u8(payload, 20)?)?,
                device_uptime: DeviceTimestampMs::from_millis_since_boot(read_u64(payload, 24)?),
            }))
        }
        MessageKind::ApplyIntent => {
            require_len(payload, 34)?;
            require_zero(payload, 33)?;
            let intent = EyeIntent::new(
                SignedUnit::try_new(read_i16(payload, 18)?)?,
                SignedUnit::try_new(read_i16(payload, 20)?)?,
                UnitAmount::try_new(read_u16(payload, 22)?)?,
                UnitAmount::try_new(read_u16(payload, 24)?)?,
                UnitAmount::try_new(read_u16(payload, 26)?)?,
                Expression::parse(read_u8(payload, 28)?)?,
                EyeFlags::try_from_bits(read_u8(payload, 29)?)?,
                read_array(payload, 30)?,
            );
            Ok(Message::ApplyIntent(ApplyIntent {
                boot_id: DeviceBootId::try_new(read_u64(payload, 0)?)?,
                control_epoch: ControlEpoch::try_new(read_u32(payload, 8)?)?,
                sequence: IntentSequence::new(read_u32(payload, 12)?),
                lease: IntentLeaseMs::try_new(read_u16(payload, 16)?)?,
                intent,
            }))
        }
        MessageKind::ReleaseControl => {
            require_len(payload, 20)?;
            require_reserved_zero(payload, 17..20)?;
            Ok(Message::ReleaseControl(ReleaseControl {
                boot_id: DeviceBootId::try_new(read_u64(payload, 0)?)?,
                control_epoch: ControlEpoch::try_new(read_u32(payload, 8)?)?,
                sequence: IntentSequence::new(read_u32(payload, 12)?),
                reason: ReleaseReason::parse(read_u8(payload, 16)?)?,
            }))
        }
        MessageKind::IntentResult => {
            require_len(payload, 40)?;
            require_reserved_zero(payload, 17..20)?;
            Ok(Message::IntentResult(IntentResult::try_new(
                DeviceBootId::try_new(read_u64(payload, 0)?)?,
                ControlEpoch::try_new(read_u32(payload, 8)?)?,
                IntentSequence::new(read_u32(payload, 12)?),
                IntentResultCode::parse(read_u8(payload, 16)?)?,
                DeviceTimestampMs::from_millis_since_boot(read_u64(payload, 20)?),
                DeviceTimestampMs::from_millis_since_boot(read_u64(payload, 28)?),
                RenderedFrameSequence::new(read_u32(payload, 36)?),
            )?))
        }
    }
}

fn require_len(payload: &[u8], expected: usize) -> Result<(), PayloadError> {
    if payload.len() == expected {
        Ok(())
    } else {
        Err(PayloadError::LengthMismatch {
            expected,
            actual: payload.len(),
        })
    }
}

fn require_zero(payload: &[u8], offset: usize) -> Result<(), PayloadError> {
    let value = read_u8(payload, offset)?;
    if value == 0 {
        Ok(())
    } else {
        Err(PayloadError::ReservedNonzero { offset, value })
    }
}

fn require_reserved_zero(
    payload: &[u8],
    range: core::ops::Range<usize>,
) -> Result<(), PayloadError> {
    for offset in range {
        require_zero(payload, offset)?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamEvent {
    Pending,
    Frame(Message),
    Dropped(FrameError),
}

/// Incremental decoder for a byte-oriented USB CDC stream.
pub struct StreamDecoder {
    encoded: [u8; MAX_ENCODED_FRAME_BYTES - 1],
    length: usize,
    discarding_oversized_record: bool,
}

impl StreamDecoder {
    pub const fn new() -> Self {
        Self {
            encoded: [0; MAX_ENCODED_FRAME_BYTES - 1],
            length: 0,
            discarding_oversized_record: false,
        }
    }

    pub fn push(&mut self, byte: u8) -> StreamEvent {
        if byte == 0 {
            if self.discarding_oversized_record {
                self.discarding_oversized_record = false;
                self.length = 0;
                return StreamEvent::Dropped(FrameError::EncodedRecordTooLong {
                    observed_at_least: MAX_ENCODED_FRAME_BYTES,
                    maximum: MAX_ENCODED_FRAME_BYTES - 1,
                });
            }
            if self.length == 0 {
                return StreamEvent::Dropped(FrameError::EmptyRecord);
            }
            let result = decode(&self.encoded[..self.length]);
            self.length = 0;
            return match result {
                Ok(message) => StreamEvent::Frame(message),
                Err(error) => StreamEvent::Dropped(error),
            };
        }

        if self.discarding_oversized_record {
            return StreamEvent::Pending;
        }
        if self.length == self.encoded.len() {
            self.length = 0;
            self.discarding_oversized_record = true;
            return StreamEvent::Pending;
        }
        self.encoded[self.length] = byte;
        self.length += 1;
        StreamEvent::Pending
    }
}

impl Default for StreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

fn cobs_encode(input: &[u8], output: &mut [u8]) -> Result<usize, EncodeError> {
    let required = input.len() + input.len() / 254 + 2;
    if output.len() < required {
        return Err(EncodeError::OutputTooSmall {
            required,
            available: output.len(),
        });
    }
    let mut read = 0;
    let mut write = 1;
    let mut code_index = 0;
    let mut code = 1_u8;
    while read < input.len() {
        let byte = input[read];
        read += 1;
        if byte == 0 {
            output[code_index] = code;
            code_index = write;
            write += 1;
            code = 1;
        } else {
            output[write] = byte;
            write += 1;
            code = code.wrapping_add(1);
            if code == 0xff {
                output[code_index] = code;
                code_index = write;
                write += 1;
                code = 1;
            }
        }
    }
    output[code_index] = code;
    output[write] = 0;
    Ok(write + 1)
}

fn cobs_decode(input: &[u8], output: &mut [u8]) -> Result<usize, ()> {
    let mut read = 0_usize;
    let mut write = 0_usize;
    while read < input.len() {
        let code = input[read];
        if code == 0 {
            return Err(());
        }
        read += 1;
        let count = usize::from(code - 1);
        if read.checked_add(count).is_none_or(|end| end > input.len()) {
            return Err(());
        }
        if write
            .checked_add(count)
            .is_none_or(|end| end > output.len())
        {
            return Err(());
        }
        if input[read..read + count].contains(&0) {
            return Err(());
        }
        output[write..write + count].copy_from_slice(&input[read..read + count]);
        read += count;
        write += count;
        if code != 0xff && read < input.len() {
            if write == output.len() {
                return Err(());
            }
            output[write] = 0;
            write += 1;
        }
    }
    Ok(write)
}

fn crc32c(bytes: &[u8]) -> u32 {
    let mut crc = !0_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0x82f6_3b78 & mask);
        }
    }
    !crc
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put_i16(bytes: &mut [u8], offset: usize, value: i16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
fn read_array<const WIDTH: usize>(
    payload: &[u8],
    offset: usize,
) -> Result<[u8; WIDTH], PayloadError> {
    let Some(end) = offset.checked_add(WIDTH) else {
        return Err(PayloadError::FieldOutOfBounds {
            offset,
            width: WIDTH,
            payload_length: payload.len(),
        });
    };
    payload
        .get(offset..end)
        .and_then(|bytes| <[u8; WIDTH]>::try_from(bytes).ok())
        .ok_or(PayloadError::FieldOutOfBounds {
            offset,
            width: WIDTH,
            payload_length: payload.len(),
        })
}

fn read_u8(payload: &[u8], offset: usize) -> Result<u8, PayloadError> {
    payload
        .get(offset)
        .copied()
        .ok_or(PayloadError::FieldOutOfBounds {
            offset,
            width: 1,
            payload_length: payload.len(),
        })
}

fn read_u16(payload: &[u8], offset: usize) -> Result<u16, PayloadError> {
    Ok(u16::from_le_bytes(read_array(payload, offset)?))
}

fn read_i16(payload: &[u8], offset: usize) -> Result<i16, PayloadError> {
    Ok(i16::from_le_bytes(read_array(payload, offset)?))
}

fn read_u32(payload: &[u8], offset: usize) -> Result<u32, PayloadError> {
    Ok(u32::from_le_bytes(read_array(payload, offset)?))
}

fn read_u64(payload: &[u8], offset: usize) -> Result<u64, PayloadError> {
    Ok(u64::from_le_bytes(read_array(payload, offset)?))
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::*;

    fn boot() -> DeviceBootId {
        DeviceBootId::try_new(7).expect("nonzero boot")
    }
    fn epoch() -> ControlEpoch {
        ControlEpoch::try_new(11).expect("nonzero epoch")
    }
    fn nonce(value: u64) -> HandshakeNonce {
        HandshakeNonce::try_new(value).expect("nonzero nonce")
    }
    const fn device_time(value: u64) -> DeviceTimestampMs {
        DeviceTimestampMs::from_millis_since_boot(value)
    }
    const fn rendered_sequence(value: u32) -> RenderedFrameSequence {
        RenderedFrameSequence::new(value)
    }

    fn decode_after_raw_mutation(
        message: Message,
        mutate: impl FnOnce(&mut [u8], usize),
    ) -> Result<Message, FrameError> {
        let mut encoded = [0_u8; MAX_ENCODED_FRAME_BYTES];
        let encoded_len = encode(message, &mut encoded).expect("encode source frame");
        let mut raw = [0_u8; MAX_RAW_FRAME_BYTES];
        let raw_len = cobs_decode(&encoded[..encoded_len - 1], &mut raw).expect("decode COBS");
        let payload_len = usize::from(u16::from_le_bytes([raw[4], raw[5]]));
        mutate(&mut raw[..raw_len], payload_len);
        let checksum_offset = HEADER_BYTES + payload_len;
        let checksum = crc32c(&raw[..checksum_offset]);
        raw[checksum_offset..checksum_offset + CHECKSUM_BYTES]
            .copy_from_slice(&checksum.to_le_bytes());
        let rewritten_len =
            cobs_encode(&raw[..raw_len], &mut encoded).expect("re-encode mutated frame");
        decode(&encoded[..rewritten_len - 1])
    }

    fn decode_after_payload_mutation(
        message: Message,
        mutate: impl FnOnce(&mut [u8]),
    ) -> Result<Message, FrameError> {
        decode_after_raw_mutation(message, |raw, payload_len| {
            mutate(&mut raw[HEADER_BYTES..HEADER_BYTES + payload_len]);
        })
    }
    fn intent() -> EyeIntent {
        EyeIntent::new(
            SignedUnit::try_new(-250).expect("gaze"),
            SignedUnit::try_new(500).expect("gaze"),
            UnitAmount::try_new(800).expect("lid"),
            UnitAmount::try_new(600).expect("pupil"),
            UnitAmount::FULL,
            Expression::Curious,
            EyeFlags::try_from_bits(EyeFlags::BLINK).expect("flags"),
            [10, 20, 30],
        )
    }

    fn messages() -> [Message; 7] {
        [
            Message::IdentityQuery { nonce: nonce(123) },
            Message::IdentityReport(IdentityReport {
                nonce: nonce(123),
                device_uid: DeviceUid::try_new([1; 16]).expect("uid"),
                firmware_build_id: FirmwareBuildId::try_new([2; 32]).expect("build"),
                boot_id: boot(),
                device_uptime: device_time(456),
                capabilities: Capabilities::try_from_bits(Capabilities::KNOWN_BITS).expect("caps"),
            }),
            Message::AcquireControl(AcquireControl {
                expected_boot_id: boot(),
                requested_epoch: epoch(),
                nonce: nonce(456),
            }),
            Message::AcquireResult(AcquireResult {
                boot_id: boot(),
                control_epoch: epoch(),
                nonce: nonce(456),
                result: AcquireResultCode::Granted,
                device_uptime: device_time(789),
            }),
            Message::ApplyIntent(ApplyIntent {
                boot_id: boot(),
                control_epoch: epoch(),
                sequence: IntentSequence::new(3),
                lease: IntentLeaseMs::try_new(500).expect("lease"),
                intent: intent(),
            }),
            Message::ReleaseControl(ReleaseControl {
                boot_id: boot(),
                control_epoch: epoch(),
                sequence: IntentSequence::new(4),
                reason: ReleaseReason::HostShutdown,
            }),
            Message::IntentResult(
                IntentResult::try_new(
                    boot(),
                    epoch(),
                    IntentSequence::new(3),
                    IntentResultCode::AppliedNew,
                    device_time(1_000),
                    device_time(1_500),
                    rendered_sequence(88),
                )
                .expect("result"),
            ),
        ]
    }

    #[test]
    fn every_message_round_trips_exactly() {
        for message in messages() {
            let mut encoded = [0; MAX_ENCODED_FRAME_BYTES];
            let length = encode(message, &mut encoded).expect("encode");
            assert_eq!(encoded[length - 1], 0);
            assert_eq!(decode(&encoded[..length - 1]), Ok(message));
        }
    }

    #[test]
    fn every_result_code_and_lease_boundary_round_trips() {
        let cases = [
            (IntentResultCode::AppliedNew, MIN_INTENT_LEASE_MS),
            (IntentResultCode::AppliedNew, MAX_INTENT_LEASE_MS),
            (IntentResultCode::DuplicateCached, MIN_INTENT_LEASE_MS),
            (IntentResultCode::DuplicateCached, MAX_INTENT_LEASE_MS),
            (IntentResultCode::Released, 0),
            (IntentResultCode::RejectedExpired, 0),
            (IntentResultCode::RejectedSession, 0),
            (IntentResultCode::RejectedSequence, 0),
            (IntentResultCode::RejectedDomain, 0),
            (IntentResultCode::FaultedFallback, 0),
        ];
        for (index, (result, interval_ms)) in cases.into_iter().enumerate() {
            let applied_at_ms = u64::try_from(index).expect("small index") * 10_000;
            let expires_at_ms = applied_at_ms + u64::from(interval_ms);
            let message = Message::IntentResult(
                IntentResult::try_new(
                    boot(),
                    epoch(),
                    IntentSequence::new(u32::try_from(index).expect("small index")),
                    result,
                    device_time(applied_at_ms),
                    device_time(expires_at_ms),
                    rendered_sequence(u32::try_from(index).expect("small index")),
                )
                .expect("valid result case"),
            );
            let mut encoded = [0_u8; MAX_ENCODED_FRAME_BYTES];
            let encoded_len = encode(message, &mut encoded).expect("encode");
            assert_eq!(decode(&encoded[..encoded_len - 1]), Ok(message));
        }
    }

    #[test]
    fn identity_query_wire_bytes_remain_stable() {
        let message = Message::IdentityQuery {
            nonce: nonce(0x0102_0304_0506_0708),
        };
        let mut encoded = [0_u8; MAX_ENCODED_FRAME_BYTES];
        let length = encode(message, &mut encoded).expect("encode");
        assert_eq!(
            &encoded[..length],
            &[
                0x06, 0x4b, 0x45, 0x02, 0x01, 0x08, 0x01, 0x01, 0x0d, 0x08, 0x07, 0x06, 0x05, 0x04,
                0x03, 0x02, 0x01, 0x11, 0x13, 0x34, 0x57, 0x00,
            ]
        );
    }

    #[test]
    fn stream_decoder_handles_bytewise_frames_and_empty_delimiters() {
        let message = messages()[2];
        let mut encoded = [0; MAX_ENCODED_FRAME_BYTES];
        let length = encode(message, &mut encoded).expect("encode");
        let mut decoder = StreamDecoder::new();
        for &byte in &encoded[..length - 1] {
            assert_eq!(decoder.push(byte), StreamEvent::Pending);
        }
        assert_eq!(decoder.push(0), StreamEvent::Frame(message));
        assert_eq!(
            decoder.push(0),
            StreamEvent::Dropped(FrameError::EmptyRecord)
        );
    }

    #[test]
    fn oversized_stream_record_drops_its_complete_suffix() {
        let mut decoder = StreamDecoder::new();
        for _ in 0..MAX_ENCODED_FRAME_BYTES + 20 {
            assert_eq!(decoder.push(1), StreamEvent::Pending);
        }
        assert!(matches!(
            decoder.push(0),
            StreamEvent::Dropped(FrameError::EncodedRecordTooLong { .. })
        ));

        let message = Message::IdentityQuery { nonce: nonce(9) };
        let mut encoded = [0; MAX_ENCODED_FRAME_BYTES];
        let length = encode(message, &mut encoded).expect("encode");
        let mut result = StreamEvent::Pending;
        for &byte in &encoded[..length] {
            result = decoder.push(byte);
        }
        assert_eq!(result, StreamEvent::Frame(message));
    }

    #[test]
    fn every_single_bit_corruption_is_rejected_or_changes_no_message() {
        for message in messages() {
            let mut encoded = [0; MAX_ENCODED_FRAME_BYTES];
            let length = encode(message, &mut encoded).expect("encode");
            for byte_index in 0..length - 1 {
                for bit in 0..8 {
                    let mut changed = encoded;
                    changed[byte_index] ^= 1 << bit;
                    assert_ne!(decode(&changed[..length - 1]), Ok(message));
                }
            }
        }
    }

    #[test]
    fn constructors_reject_invalid_domains() {
        assert_eq!(
            HandshakeNonce::try_new(0),
            Err(DomainError::ZeroHandshakeNonce)
        );
        assert_eq!(DeviceUid::try_new([0; 16]), Err(DomainError::ZeroDeviceUid));
        assert_eq!(
            FirmwareBuildId::try_new([0; 32]),
            Err(DomainError::ZeroFirmwareBuildId)
        );
        assert!(IntentLeaseMs::try_new(MIN_INTENT_LEASE_MS - 1).is_err());
        assert!(IntentLeaseMs::try_new(MAX_INTENT_LEASE_MS + 1).is_err());
        assert!(SignedUnit::try_new(-NORMALIZED_SCALE - 1).is_err());
        assert!(SignedUnit::try_new(NORMALIZED_SCALE + 1).is_err());
        assert!(UnitAmount::try_new(NORMALIZED_SCALE_U16 + 1).is_err());
        assert!(EyeFlags::try_from_bits(0x80).is_err());
        assert!(Capabilities::try_from_bits(1 << 31).is_err());
    }

    #[test]
    fn zero_nonce_is_rejected_at_every_inbound_handshake_boundary() {
        let cases = [
            (Message::IdentityQuery { nonce: nonce(1) }, 0),
            (messages()[1], 0),
            (messages()[2], 12),
            (messages()[3], 12),
        ];
        for (message, nonce_offset) in cases {
            assert_eq!(
                decode_after_payload_mutation(message, |payload| {
                    payload[nonce_offset..nonce_offset + 8].fill(0);
                }),
                Err(FrameError::Payload(PayloadError::Domain(
                    DomainError::ZeroHandshakeNonce
                )))
            );
        }
    }

    #[test]
    fn checksum_valid_invalid_primitives_are_rejected_at_decode() {
        assert_eq!(
            decode_after_payload_mutation(messages()[1], |payload| payload[8..24].fill(0)),
            Err(FrameError::Payload(PayloadError::Domain(
                DomainError::ZeroDeviceUid
            )))
        );
        assert_eq!(
            decode_after_payload_mutation(messages()[1], |payload| payload[24..56].fill(0)),
            Err(FrameError::Payload(PayloadError::Domain(
                DomainError::ZeroFirmwareBuildId
            )))
        );
        assert_eq!(
            decode_after_payload_mutation(messages()[1], |payload| payload[56..64].fill(0)),
            Err(FrameError::Payload(PayloadError::Domain(
                DomainError::ZeroDeviceBootId
            )))
        );
        assert_eq!(
            decode_after_payload_mutation(messages()[1], |payload| {
                payload[72..76].copy_from_slice(&(1_u32 << 31).to_le_bytes());
            }),
            Err(FrameError::Payload(PayloadError::Domain(
                DomainError::UnknownCapabilityBits { bits: 1_u32 << 31 }
            )))
        );
        assert_eq!(
            decode_after_payload_mutation(messages()[2], |payload| payload[8..12].fill(0)),
            Err(FrameError::Payload(PayloadError::Domain(
                DomainError::ZeroControlEpoch
            )))
        );
        assert!(matches!(
            decode_after_payload_mutation(messages()[4], |payload| {
                payload[16..18].copy_from_slice(&(MIN_INTENT_LEASE_MS - 1).to_le_bytes());
            }),
            Err(FrameError::Payload(PayloadError::Domain(
                DomainError::IntentLeaseOutOfRange { .. }
            )))
        ));
        assert_eq!(
            decode_after_payload_mutation(messages()[4], |payload| {
                payload[18..20].copy_from_slice(&(NORMALIZED_SCALE + 1).to_le_bytes());
            }),
            Err(FrameError::Payload(PayloadError::Domain(
                DomainError::SignedUnitOutOfRange {
                    value: NORMALIZED_SCALE + 1
                }
            )))
        );
        assert_eq!(
            decode_after_payload_mutation(messages()[4], |payload| {
                payload[22..24].copy_from_slice(&(NORMALIZED_SCALE_U16 + 1).to_le_bytes());
            }),
            Err(FrameError::Payload(PayloadError::Domain(
                DomainError::UnitAmountOutOfRange {
                    value: NORMALIZED_SCALE_U16 + 1
                }
            )))
        );
        assert_eq!(
            decode_after_payload_mutation(messages()[4], |payload| payload[29] = 0x80),
            Err(FrameError::Payload(PayloadError::Domain(
                DomainError::UnknownEyeFlagBits { bits: 0x80 }
            )))
        );
    }

    #[test]
    fn checksum_valid_unknown_enums_are_rejected_at_decode() {
        let cases = [
            (messages()[3], 20, "acquire result"),
            (messages()[4], 28, "expression"),
            (messages()[5], 16, "release reason"),
            (messages()[6], 16, "intent result"),
        ];
        for (message, offset, field) in cases {
            assert_eq!(
                decode_after_payload_mutation(message, |payload| payload[offset] = u8::MAX),
                Err(FrameError::Payload(PayloadError::UnknownEnum {
                    field,
                    value: u8::MAX,
                }))
            );
        }
    }

    #[test]
    fn intent_result_timing_is_exact_and_bounded() {
        assert!(matches!(
            IntentResult::try_new(
                boot(),
                epoch(),
                IntentSequence::FIRST,
                IntentResultCode::AppliedNew,
                device_time(10),
                device_time(9),
                rendered_sequence(0)
            ),
            Err(DomainError::InvalidIntentResultTiming { .. })
        ));
        assert!(
            IntentResult::try_new(
                boot(),
                epoch(),
                IntentSequence::FIRST,
                IntentResultCode::AppliedNew,
                device_time(10),
                device_time(10 + u64::from(MAX_INTENT_LEASE_MS) + 1),
                rendered_sequence(0)
            )
            .is_err()
        );
        assert!(
            IntentResult::try_new(
                boot(),
                epoch(),
                IntentSequence::FIRST,
                IntentResultCode::RejectedExpired,
                device_time(10),
                device_time(10),
                rendered_sequence(0)
            )
            .is_ok()
        );

        let admitted = IntentResult::try_new(
            boot(),
            epoch(),
            IntentSequence::FIRST,
            IntentResultCode::AppliedNew,
            device_time(u64::MAX - u64::from(MIN_INTENT_LEASE_MS)),
            device_time(u64::MAX),
            rendered_sequence(u32::MAX),
        )
        .expect("upper-bound timestamp");
        assert_eq!(
            admitted.admitted_lease(),
            Some(IntentLeaseMs::try_new(MIN_INTENT_LEASE_MS).expect("lease"))
        );
        assert_eq!(admitted.device_interval_ms(), MIN_INTENT_LEASE_MS);
        assert_eq!(admitted.applied_at().millis_since_boot(), u64::MAX - 20);
        assert_eq!(admitted.expires_at().millis_since_boot(), u64::MAX);
        assert_eq!(admitted.rendered_frame_sequence().get(), u32::MAX);
    }

    #[test]
    fn checksum_valid_invalid_result_timing_is_rejected_during_decode() {
        let admitted = messages()[6];
        let reversed = decode_after_payload_mutation(admitted, |payload| {
            payload[28..36].copy_from_slice(&999_u64.to_le_bytes());
        });
        assert!(matches!(
            reversed,
            Err(FrameError::Payload(PayloadError::Domain(
                DomainError::InvalidIntentResultTiming {
                    result: IntentResultCode::AppliedNew,
                    applied_at_ms: 1_000,
                    expires_at_ms: 999,
                }
            )))
        ));

        let released = Message::IntentResult(
            IntentResult::try_new(
                boot(),
                epoch(),
                IntentSequence::new(4),
                IntentResultCode::Released,
                device_time(1_600),
                device_time(1_600),
                rendered_sequence(89),
            )
            .expect("released"),
        );
        let nonzero = decode_after_payload_mutation(released, |payload| {
            payload[28..36].copy_from_slice(&1_601_u64.to_le_bytes());
        });
        assert!(matches!(
            nonzero,
            Err(FrameError::Payload(PayloadError::Domain(
                DomainError::InvalidIntentResultTiming {
                    result: IntentResultCode::Released,
                    applied_at_ms: 1_600,
                    expires_at_ms: 1_601,
                }
            )))
        ));
    }

    #[test]
    fn checksum_valid_reserved_fields_are_still_rejected() {
        let cases = [
            (messages()[3], 21),
            (messages()[4], 33),
            (messages()[5], 17),
            (messages()[6], 17),
        ];
        for (message, reserved_offset) in cases {
            assert_eq!(
                decode_after_payload_mutation(message, |payload| {
                    payload[reserved_offset] = 1;
                }),
                Err(FrameError::Payload(PayloadError::ReservedNonzero {
                    offset: reserved_offset,
                    value: 1,
                }))
            );
        }
        assert_eq!(
            decode_after_raw_mutation(messages()[0], |raw, _| raw[6] = 1),
            Err(FrameError::HeaderReservedNonzero { bytes: [1, 0] })
        );
    }

    #[test]
    fn arbitrary_bounded_records_never_panic_the_decoder() {
        let mut state = 0x4b45_5032_5eed_u64;
        let mut record = [0_u8; MAX_ENCODED_FRAME_BYTES + 4];
        for length in 0..=record.len() {
            for _ in 0..32 {
                for byte in &mut record[..length] {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    *byte = state as u8;
                }
                let _ = decode(&record[..length]);
            }
        }
    }

    #[test]
    fn truncated_extended_and_delimited_records_are_rejected() {
        let message = messages()[2];
        let mut encoded = [0; MAX_ENCODED_FRAME_BYTES];
        let length = encode(message, &mut encoded).expect("encode");
        for end in 0..length - 1 {
            assert!(decode(&encoded[..end]).is_err());
        }
        assert!(decode(&encoded[..length]).is_err());
        let mut extended = [0; MAX_ENCODED_FRAME_BYTES];
        extended[..length - 1].copy_from_slice(&encoded[..length - 1]);
        extended[length - 1] = 1;
        assert!(decode(&extended[..length]).is_err());
    }

    #[test]
    fn crc32c_matches_the_standard_check_value() {
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
    }

    #[test]
    fn cobs_round_trips_zero_dense_and_full_code_blocks() {
        let mut source = [0_u8; MAX_RAW_FRAME_BYTES];
        for (index, byte) in source.iter_mut().enumerate() {
            *byte = if index % 7 == 0 { 0 } else { index as u8 };
        }
        let mut encoded = [0; MAX_ENCODED_FRAME_BYTES];
        let encoded_len = cobs_encode(&source, &mut encoded).expect("encode");
        let mut decoded = [0; MAX_RAW_FRAME_BYTES];
        let decoded_len = cobs_decode(&encoded[..encoded_len - 1], &mut decoded).expect("decode");
        assert_eq!(&decoded[..decoded_len], &source);
    }
}
