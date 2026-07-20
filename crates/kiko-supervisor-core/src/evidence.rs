use core::fmt;
use core::num::NonZeroU64;

use robot_protocol::v2::{
    AppliedResult, AppliedResultCode, ControlEpoch, ControllerBootId, ControllerFaults,
    ControllerUid, HostCommandResult, HostCommandResultCode, OutputState, TimerPwm,
    V2CommandSequence,
};

use crate::MonotonicInstant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Sha256Digest([u8; 32]);

impl Sha256Digest {
    pub fn try_new(bytes: [u8; 32]) -> Result<Self, EvidenceValueError> {
        if bytes == [0; 32] {
            Err(EvidenceValueError::ZeroSha256Digest)
        } else {
            Ok(Self(bytes))
        }
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReadinessEpoch(NonZeroU64);

impl ReadinessEpoch {
    pub fn try_new(value: u64) -> Result<Self, EvidenceValueError> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(EvidenceValueError::ZeroReadinessEpoch)
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Identity and evidence hashes admitted by the hardware-inventory boundary.
///
/// This type binds later zero receipts to the inventory that qualified them.
/// Its constructor does not claim that the contents of either manifest are
/// physically true; the inventory owner must establish that separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadinessBinding {
    epoch: ReadinessEpoch,
    controller_uid: ControllerUid,
    controller_boot_id: ControllerBootId,
    control_epoch: ControlEpoch,
    hardware_manifest: Sha256Digest,
    calibration_bundle: Sha256Digest,
}

impl ReadinessBinding {
    pub const fn new(
        epoch: ReadinessEpoch,
        controller_uid: ControllerUid,
        controller_boot_id: ControllerBootId,
        control_epoch: ControlEpoch,
        hardware_manifest: Sha256Digest,
        calibration_bundle: Sha256Digest,
    ) -> Self {
        Self {
            epoch,
            controller_uid,
            controller_boot_id,
            control_epoch,
            hardware_manifest,
            calibration_bundle,
        }
    }

    pub const fn epoch(self) -> ReadinessEpoch {
        self.epoch
    }

    pub const fn controller_uid(self) -> ControllerUid {
        self.controller_uid
    }

    pub const fn controller_boot_id(self) -> ControllerBootId {
        self.controller_boot_id
    }

    pub const fn control_epoch(self) -> ControlEpoch {
        self.control_epoch
    }

    pub const fn hardware_manifest(self) -> Sha256Digest {
        self.hardware_manifest
    }

    pub const fn calibration_bundle(self) -> Sha256Digest {
        self.calibration_bundle
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfirmedBaseZero {
    controller_uid: ControllerUid,
    controller_boot_id: ControllerBootId,
    control_epoch: ControlEpoch,
    sequence: V2CommandSequence,
    observed_at: MonotonicInstant,
    result: AppliedResultCode,
    output_state: OutputState,
}

impl ConfirmedBaseZero {
    pub fn try_from_applied_result(
        result: AppliedResult,
        observed_at: MonotonicInstant,
    ) -> Result<Self, ZeroEvidenceError> {
        // A cached duplicate proves that some earlier command was applied, but
        // it cannot prove the post-stop application required for an authority
        // handoff. Admit only a newly applied command or the controller's
        // explicit stopped result as fresh zero evidence.
        if !matches!(
            result.result,
            AppliedResultCode::AppliedNew | AppliedResultCode::Stopped
        ) {
            return Err(ZeroEvidenceError::ResultDoesNotProveApplication {
                result: result.result,
            });
        }
        Self::try_from_parts(
            result.controller_uid,
            result.boot_id,
            result.control_epoch,
            result.sequence,
            result.result,
            result.timer_pwm,
            result.output_state,
            result.faults,
            observed_at,
        )
    }

    /// Admit the exact host-server result retained by a verified command
    /// receipt without reconstructing a controller wire message.
    ///
    /// The host request itself must have requested zero. A nonzero command
    /// that happened to stop cannot satisfy the supervisor's deliberate
    /// post-stop zero-command gate.
    pub fn try_from_host_command_result(
        result: HostCommandResult,
        observed_at: MonotonicInstant,
    ) -> Result<Self, ZeroEvidenceError> {
        if !result.requested_timer_pwm.is_zero() {
            return Err(ZeroEvidenceError::RequestedNonzeroPwm {
                left: result.requested_timer_pwm.left().get(),
                right: result.requested_timer_pwm.right().get(),
            });
        }
        let controller_result = match result.result {
            HostCommandResultCode::AppliedNew => AppliedResultCode::AppliedNew,
            HostCommandResultCode::Stopped => AppliedResultCode::Stopped,
            result => {
                return Err(ZeroEvidenceError::HostResultDoesNotProveFreshApplication { result });
            }
        };
        Self::try_from_parts(
            result.controller_uid,
            result.boot_id,
            result.control_epoch,
            result.sequence,
            controller_result,
            result.controller_timer_pwm,
            result.output_state,
            result.faults,
            observed_at,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn try_from_parts(
        controller_uid: ControllerUid,
        controller_boot_id: ControllerBootId,
        control_epoch: ControlEpoch,
        sequence: V2CommandSequence,
        result: AppliedResultCode,
        timer_pwm: TimerPwm,
        output_state: OutputState,
        faults: ControllerFaults,
        observed_at: MonotonicInstant,
    ) -> Result<Self, ZeroEvidenceError> {
        if !timer_pwm.is_zero() {
            return Err(ZeroEvidenceError::NonzeroPwm {
                left: timer_pwm.left().get(),
                right: timer_pwm.right().get(),
            });
        }
        if !output_state.is_safe() {
            return Err(ZeroEvidenceError::UnsafeOutputState { output_state });
        }
        if !faults.is_clear() {
            return Err(ZeroEvidenceError::ControllerFaults {
                bits: faults.bits(),
            });
        }
        Ok(Self {
            controller_uid,
            controller_boot_id,
            control_epoch,
            sequence,
            observed_at,
            result,
            output_state,
        })
    }

    pub const fn controller_uid(self) -> ControllerUid {
        self.controller_uid
    }

    pub const fn controller_boot_id(self) -> ControllerBootId {
        self.controller_boot_id
    }

    pub const fn control_epoch(self) -> ControlEpoch {
        self.control_epoch
    }

    pub const fn sequence(self) -> V2CommandSequence {
        self.sequence
    }

    pub const fn observed_at(self) -> MonotonicInstant {
        self.observed_at
    }

    pub const fn result(self) -> AppliedResultCode {
        self.result
    }

    pub const fn output_state(self) -> OutputState {
        self.output_state
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceValueError {
    ZeroSha256Digest,
    ZeroReadinessEpoch,
}

impl fmt::Display for EvidenceValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid supervisor evidence identity: {self:?}")
    }
}

impl core::error::Error for EvidenceValueError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZeroEvidenceError {
    ResultDoesNotProveApplication { result: AppliedResultCode },
    HostResultDoesNotProveFreshApplication { result: HostCommandResultCode },
    RequestedNonzeroPwm { left: i8, right: i8 },
    NonzeroPwm { left: i8, right: i8 },
    UnsafeOutputState { output_state: OutputState },
    ControllerFaults { bits: u32 },
}

impl fmt::Display for ZeroEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "invalid confirmed-base-zero evidence: {self:?}")
    }
}

impl core::error::Error for ZeroEvidenceError {}
