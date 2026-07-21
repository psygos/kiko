use core::num::{NonZeroU16, NonZeroU32};

use kiko_eye_protocol::{
    Capabilities as EyeCapabilities, DeviceBootId as EyeBootId, DeviceUid as EyeDeviceUid,
    FirmwareBuildId as EyeFirmwareBuildId,
};
use kiko_head_protocol::ServoId;
use robot_protocol::v2::{
    ActuatorConfigFingerprint, ControllerBootId, ControllerCapabilities, ControllerUid,
};

use crate::{BuildProvenance, ControlEndpointIdentity, OakMxid, PersistentSerialPath};

pub const MAX_OBSERVED_HEAD_SERVOS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceRole {
    Stm32,
    Head,
    Eye,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OakIdentity {
    mxid: OakMxid,
    compiled_depthai_header_sdk_version: BuildProvenance,
    compiled_depthai_header_sdk_commit: BuildProvenance,
    compiled_depthai_header_embedded_device_artifact_version: BuildProvenance,
    compiled_depthai_header_embedded_bootloader_artifact_version: BuildProvenance,
}

impl OakIdentity {
    pub(crate) const fn new(
        mxid: OakMxid,
        compiled_depthai_header_sdk_version: BuildProvenance,
        compiled_depthai_header_sdk_commit: BuildProvenance,
        compiled_depthai_header_embedded_device_artifact_version: BuildProvenance,
        compiled_depthai_header_embedded_bootloader_artifact_version: BuildProvenance,
    ) -> Self {
        Self {
            mxid,
            compiled_depthai_header_sdk_version,
            compiled_depthai_header_sdk_commit,
            compiled_depthai_header_embedded_device_artifact_version,
            compiled_depthai_header_embedded_bootloader_artifact_version,
        }
    }

    pub fn mxid(&self) -> &OakMxid {
        &self.mxid
    }

    pub fn compiled_depthai_header_sdk_version(&self) -> &BuildProvenance {
        &self.compiled_depthai_header_sdk_version
    }

    pub fn compiled_depthai_header_sdk_commit(&self) -> &BuildProvenance {
        &self.compiled_depthai_header_sdk_commit
    }

    /// Device-artifact version reported by the compiled DepthAI header.
    ///
    /// This proves neither the identity of a linked/runtime DepthAI library nor
    /// firmware currently executing on the connected OAK device.
    pub fn compiled_depthai_header_embedded_device_artifact_version(&self) -> &BuildProvenance {
        &self.compiled_depthai_header_embedded_device_artifact_version
    }

    /// Bootloader-artifact version reported by the compiled DepthAI header.
    ///
    /// This proves neither the identity of a linked/runtime DepthAI library nor
    /// the bootloader installed on the connected OAK device.
    pub fn compiled_depthai_header_embedded_bootloader_artifact_version(&self) -> &BuildProvenance {
        &self.compiled_depthai_header_embedded_bootloader_artifact_version
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stm32StaticIdentity {
    serial_path: PersistentSerialPath,
    control_endpoint: ControlEndpointIdentity,
    controller_uid: ControllerUid,
    firmware_abi: NonZeroU16,
    firmware_build_id: NonZeroU32,
    hardware_profile: ActuatorConfigFingerprint,
    capabilities: ControllerCapabilities,
}

impl Stm32StaticIdentity {
    pub(crate) const fn new(
        serial_path: PersistentSerialPath,
        control_endpoint: ControlEndpointIdentity,
        controller_uid: ControllerUid,
        firmware_abi: NonZeroU16,
        firmware_build_id: NonZeroU32,
        hardware_profile: ActuatorConfigFingerprint,
        capabilities: ControllerCapabilities,
    ) -> Self {
        Self {
            serial_path,
            control_endpoint,
            controller_uid,
            firmware_abi,
            firmware_build_id,
            hardware_profile,
            capabilities,
        }
    }

    pub fn serial_path(&self) -> &PersistentSerialPath {
        &self.serial_path
    }

    pub fn control_endpoint(&self) -> &ControlEndpointIdentity {
        &self.control_endpoint
    }

    pub fn controller_uid(&self) -> &ControllerUid {
        &self.controller_uid
    }

    pub fn firmware_abi(&self) -> u16 {
        self.firmware_abi.get()
    }

    pub fn firmware_build_id(&self) -> u32 {
        self.firmware_build_id.get()
    }

    pub fn hardware_profile(&self) -> &ActuatorConfigFingerprint {
        &self.hardware_profile
    }

    pub fn capabilities(&self) -> ControllerCapabilities {
        self.capabilities
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedStm32 {
    static_identity: Stm32StaticIdentity,
    boot_id: ControllerBootId,
}

impl ObservedStm32 {
    pub(crate) const fn new(
        static_identity: Stm32StaticIdentity,
        boot_id: ControllerBootId,
    ) -> Self {
        Self {
            static_identity,
            boot_id,
        }
    }

    pub fn static_identity(&self) -> &Stm32StaticIdentity {
        &self.static_identity
    }

    pub fn boot_id(&self) -> ControllerBootId {
        self.boot_id
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadExpectedIdentity {
    serial_path: PersistentSerialPath,
    servo_ids: [ServoId; 4],
}

impl HeadExpectedIdentity {
    pub(crate) const fn new(serial_path: PersistentSerialPath, servo_ids: [ServoId; 4]) -> Self {
        Self {
            serial_path,
            servo_ids,
        }
    }

    pub fn serial_path(&self) -> &PersistentSerialPath {
        &self.serial_path
    }

    pub fn servo_ids(&self) -> &[ServoId; 4] {
        &self.servo_ids
    }

    pub const fn baud_rate_bps(&self) -> u32 {
        kiko_head_protocol::BUS_BAUD_RATE_BPS
    }

    pub const fn dtr_asserted(&self) -> bool {
        kiko_head_protocol::ADAPTER_DTR_ASSERTED
    }

    pub const fn rts_asserted(&self) -> bool {
        kiko_head_protocol::ADAPTER_RTS_ASSERTED
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservedServoIds {
    entries: [Option<ServoId>; MAX_OBSERVED_HEAD_SERVOS],
    len: u8,
}

impl ObservedServoIds {
    pub(crate) fn from_parsed(
        mut entries: [Option<ServoId>; MAX_OBSERVED_HEAD_SERVOS],
        len: usize,
    ) -> Self {
        entries[..len].sort_unstable();
        Self {
            entries,
            len: len as u8,
        }
    }

    pub fn len(self) -> usize {
        usize::from(self.len)
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = ServoId> + '_ {
        self.entries[..self.len()]
            .iter()
            .map(|entry| entry.expect("parsed observed-servo prefix is initialized"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedHead {
    serial_path: PersistentSerialPath,
    baud_rate_bps: u32,
    dtr_asserted: bool,
    rts_asserted: bool,
    servo_ids: ObservedServoIds,
}

impl ObservedHead {
    pub(crate) const fn new(
        serial_path: PersistentSerialPath,
        baud_rate_bps: u32,
        dtr_asserted: bool,
        rts_asserted: bool,
        servo_ids: ObservedServoIds,
    ) -> Self {
        Self {
            serial_path,
            baud_rate_bps,
            dtr_asserted,
            rts_asserted,
            servo_ids,
        }
    }

    pub fn serial_path(&self) -> &PersistentSerialPath {
        &self.serial_path
    }

    pub fn baud_rate_bps(&self) -> u32 {
        self.baud_rate_bps
    }

    pub fn dtr_asserted(&self) -> bool {
        self.dtr_asserted
    }

    pub fn rts_asserted(&self) -> bool {
        self.rts_asserted
    }

    pub fn servo_ids(&self) -> &ObservedServoIds {
        &self.servo_ids
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EyeStaticIdentity {
    serial_path: PersistentSerialPath,
    protocol_version: u8,
    device_uid: EyeDeviceUid,
    firmware_build_id: EyeFirmwareBuildId,
    capabilities: EyeCapabilities,
}

impl EyeStaticIdentity {
    pub(crate) const fn new(
        serial_path: PersistentSerialPath,
        protocol_version: u8,
        device_uid: EyeDeviceUid,
        firmware_build_id: EyeFirmwareBuildId,
        capabilities: EyeCapabilities,
    ) -> Self {
        Self {
            serial_path,
            protocol_version,
            device_uid,
            firmware_build_id,
            capabilities,
        }
    }

    pub fn serial_path(&self) -> &PersistentSerialPath {
        &self.serial_path
    }

    pub fn protocol_version(&self) -> u8 {
        self.protocol_version
    }

    pub fn device_uid(&self) -> &EyeDeviceUid {
        &self.device_uid
    }

    pub fn firmware_build_id(&self) -> &EyeFirmwareBuildId {
        &self.firmware_build_id
    }

    pub fn capabilities(&self) -> EyeCapabilities {
        self.capabilities
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedEye {
    static_identity: EyeStaticIdentity,
    boot_id: EyeBootId,
}

impl ObservedEye {
    pub(crate) const fn new(static_identity: EyeStaticIdentity, boot_id: EyeBootId) -> Self {
        Self {
            static_identity,
            boot_id,
        }
    }

    pub fn static_identity(&self) -> &EyeStaticIdentity {
        &self.static_identity
    }

    pub fn boot_id(&self) -> EyeBootId {
        self.boot_id
    }
}
