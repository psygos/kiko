//! Hardware-independent binding between one production controller contract
//! and one parsed navigation/actuation graph.
//!
//! Runtime startup and offline deployment proof both use this constructor so
//! controller identity, PWM envelope, and cadence semantics cannot drift.

use std::fmt;
use std::net::SocketAddr;
use std::time::Duration;

use robot_server::config::ControllerServerConfigV1;

use super::mpc::{MpcConfigV1, WheelSide};
use super::{ControlPeriodNs, NavigationActuationConfigV1};

/// Values admitted by the typed controller contract plus its launch-owned
/// loopback endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductionNavigationControllerContractV1<'controller> {
    command_endpoint: SocketAddr,
    controller: &'controller ControllerServerConfigV1,
}

impl<'controller> ProductionNavigationControllerContractV1<'controller> {
    pub const fn new(
        command_endpoint: SocketAddr,
        controller: &'controller ControllerServerConfigV1,
    ) -> Self {
        Self {
            command_endpoint,
            controller,
        }
    }
}

/// Proof that one exact parsed controller contract can execute one exact
/// parsed production navigation graph without exceeding its command envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductionNavigationControllerBindingV1 {
    _private: (),
}

impl ProductionNavigationControllerBindingV1 {
    pub fn bind(
        controller: ProductionNavigationControllerContractV1<'_>,
        actuation: &NavigationActuationConfigV1,
        mpc: MpcConfigV1,
        control_period: ControlPeriodNs,
    ) -> Result<Self, ProductionNavigationControllerBindingError> {
        let actuation_endpoint = actuation.command_endpoint().socket_addr();
        if controller.command_endpoint != actuation_endpoint {
            return Err(
                ProductionNavigationControllerBindingError::CommandEndpointMismatch {
                    controller: controller.command_endpoint,
                    actuation: actuation_endpoint,
                },
            );
        }
        if controller.controller.controller_uid() != actuation.controller_uid() {
            return Err(ProductionNavigationControllerBindingError::ControllerUidMismatch);
        }
        if controller.controller.firmware_abi() != actuation.firmware_abi() {
            return Err(
                ProductionNavigationControllerBindingError::ControllerFirmwareAbiMismatch {
                    controller: controller.controller.firmware_abi().get(),
                    actuation: actuation.firmware_abi().get(),
                },
            );
        }
        if controller.controller.firmware_build_id() != actuation.firmware_build_id() {
            return Err(
                ProductionNavigationControllerBindingError::ControllerFirmwareBuildMismatch {
                    controller: controller.controller.firmware_build_id().get(),
                    actuation: actuation.firmware_build_id().get(),
                },
            );
        }
        if controller.controller.actuator_config_fingerprint()
            != actuation.actuator_config_fingerprint()
        {
            return Err(ProductionNavigationControllerBindingError::ControllerFingerprintMismatch);
        }
        bind_mpc_pwm_to_controller_envelope(
            controller.controller.expected_max_abs_pwm_percent().get(),
            mpc,
        )?;
        bind_navigation_cadence_to_controller(
            control_period.as_duration(),
            controller.controller.minimum_host_command_interval(),
            Duration::from_nanos(actuation.scheduling_guard_ns().get()),
        )?;
        Ok(Self { _private: () })
    }
}

pub(super) fn bind_navigation_cadence_to_controller(
    control_period: Duration,
    controller_minimum_interval: Duration,
    scheduling_margin: Duration,
) -> Result<(), ProductionNavigationControllerBindingError> {
    let required_exclusive_lower_bound = controller_minimum_interval
        .checked_add(scheduling_margin)
        .ok_or(ProductionNavigationControllerBindingError::CadenceArithmeticOverflow)?;
    if control_period <= required_exclusive_lower_bound {
        return Err(
            ProductionNavigationControllerBindingError::ControlPeriodHasNoControllerRateMargin {
                control_period,
                controller_minimum_interval,
                scheduling_margin,
                required_exclusive_lower_bound,
            },
        );
    }
    Ok(())
}

fn bind_mpc_pwm_to_controller_envelope(
    controller_max_abs_percent: u8,
    mpc: MpcConfigV1,
) -> Result<(), ProductionNavigationControllerBindingError> {
    for (wheel, (configured_min, configured_max)) in [
        (WheelSide::Left, mpc.left_pwm_bounds_percent()),
        (WheelSide::Right, mpc.right_pwm_bounds_percent()),
    ] {
        bind_one_mpc_pwm_range(
            wheel,
            configured_min,
            configured_max,
            controller_max_abs_percent,
        )?;
    }
    Ok(())
}

pub(super) fn bind_one_mpc_pwm_range(
    wheel: WheelSide,
    configured_min_percent: i8,
    configured_max_percent: i8,
    controller_max_abs_percent: u8,
) -> Result<(), ProductionNavigationControllerBindingError> {
    let controller_max = i16::from(controller_max_abs_percent);
    if i16::from(configured_min_percent) < -controller_max
        || i16::from(configured_max_percent) > controller_max
    {
        return Err(
            ProductionNavigationControllerBindingError::MpcPwmOutsideControllerEnvelope {
                wheel,
                configured_min_percent,
                configured_max_percent,
                controller_max_abs_percent,
            },
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProductionNavigationControllerBindingError {
    CommandEndpointMismatch {
        controller: SocketAddr,
        actuation: SocketAddr,
    },
    ControllerUidMismatch,
    ControllerFirmwareAbiMismatch {
        controller: u16,
        actuation: u16,
    },
    ControllerFirmwareBuildMismatch {
        controller: u32,
        actuation: u32,
    },
    ControllerFingerprintMismatch,
    MpcPwmOutsideControllerEnvelope {
        wheel: WheelSide,
        configured_min_percent: i8,
        configured_max_percent: i8,
        controller_max_abs_percent: u8,
    },
    CadenceArithmeticOverflow,
    ControlPeriodHasNoControllerRateMargin {
        control_period: Duration,
        controller_minimum_interval: Duration,
        scheduling_margin: Duration,
        required_exclusive_lower_bound: Duration,
    },
}

impl fmt::Display for ProductionNavigationControllerBindingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "production navigation/controller binding rejected input: {self:?}"
        )
    }
}

impl std::error::Error for ProductionNavigationControllerBindingError {}
