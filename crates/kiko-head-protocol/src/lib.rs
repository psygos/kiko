//! Transport-independent contract for Kiko's four Feetech ST3215 head servos.
//!
//! This crate intentionally owns neither a serial port nor motion authority. It
//! turns typed requests into exact SCS/STS frames, parses one already delimited
//! reply exactly once, and provides the redundant-present-pose evidence needed
//! by a higher-level head actor. Unacknowledged writes are never represented as
//! applied hardware state; callers must read telemetry after writing.

#![no_std]
#![forbid(unsafe_code)]

mod packet;
mod pose;
mod telemetry;

pub use packet::{
    CommandFrame, FrameBuildError, GoalSpeedTicksPerSecond, PositionTicks, ResponseParseError,
    ServoId, ServoStatus, TorqueLimitPermille, TorqueSwitch, build_full_telemetry_read,
    build_goal_position_read, build_goal_with_speed_write, build_ping, build_position_read,
    build_torque_limit_write, build_torque_switch_read, build_torque_switch_write,
    parse_status_response,
};
pub use pose::{
    AngleRadians, ExactHeadTargetPose, ExactHeadTargetPoseError, HeadJoint, HeadPose,
    HeadPoseError, HeadTorqueLimits, JointCalibration, JointCalibrationError, JointDirection,
    JointLimitsRadians, NaturalHoldFrames, PositionStepLimit, TelemetryPoseSample,
    build_natural_hold_frames,
};
pub use telemetry::{
    FullTelemetry, GoalPositionObservation, ObservedTorqueSwitch, PositionAgreementError,
    PositionAgreementTicks, PresentPosition, TelemetryParseError, TorqueSwitchObservation,
    ValidatedPresentPosition,
};

/// Proven electrical configuration of the Waveshare adapter used by the demo
/// rig. A transport must apply and verify these line states before sending any
/// servo traffic; opening a USB serial port can itself change modem controls.
pub const BUS_BAUD_RATE_BPS: u32 = 1_000_000;
pub const ADAPTER_DTR_ASSERTED: bool = false;
pub const ADAPTER_RTS_ASSERTED: bool = true;
