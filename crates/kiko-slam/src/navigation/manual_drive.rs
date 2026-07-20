//! Typed, transport-free manual-drive ingress and deadman state.
//!
//! Weak command DTOs are consumed exactly once here. Successful output is a
//! body-frame velocity target in SI units, never wheel PWM. This module does
//! not bypass local collision validation, MPC, the safety journal, or physical
//! actuation authority.
//!
//! `kiko-slam` intentionally does not depend on the supervisor crate. The core
//! is therefore generic over the supervisor's real lease-ID type instead of
//! defining a competing ID. The sole supervisor adapter must convert its
//! current state to [`ManualAuthoritySnapshot`] for every ingress and tick.

use std::fmt;
use std::num::NonZeroU64;

use crate::HostMonotonicTimestamp;

pub const MANUAL_DRIVE_CONFIG_V1: u32 = 1;
pub const MANUAL_DRIVE_COMMAND_V1: u32 = 1;
pub const BODY_VELOCITY_TARGET_V1: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManualDriveConfigV1Dto {
    pub schema_version: u32,
    pub maximum_abs_forward_velocity_mps: f64,
    pub maximum_abs_yaw_rate_rad_s: f64,
    pub maximum_command_age_ns: u64,
    pub deadman_timeout_ns: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ManualDriveConfigParseError {
    UnsupportedSchemaVersion {
        actual: u32,
        supported: u32,
    },
    NonFiniteLimit {
        field: &'static str,
        value: f64,
    },
    NonPositiveLimit {
        field: &'static str,
        value: f64,
    },
    ZeroMaximumCommandAge,
    ZeroDeadmanTimeout,
    CommandAgeExceedsDeadman {
        maximum_command_age_ns: u64,
        deadman_timeout_ns: u64,
    },
}

impl fmt::Display for ManualDriveConfigParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { actual, supported } => write!(
                formatter,
                "unsupported manual-drive config schema {actual}; supported schema is {supported}"
            ),
            Self::NonFiniteLimit { field, value } => {
                write!(
                    formatter,
                    "manual-drive {field} must be finite, got {value}"
                )
            }
            Self::NonPositiveLimit { field, value } => write!(
                formatter,
                "manual-drive {field} must be strictly positive, got {value}"
            ),
            Self::ZeroMaximumCommandAge => {
                formatter.write_str("manual-drive maximum command age must be nonzero")
            }
            Self::ZeroDeadmanTimeout => {
                formatter.write_str("manual-drive deadman timeout must be nonzero")
            }
            Self::CommandAgeExceedsDeadman {
                maximum_command_age_ns,
                deadman_timeout_ns,
            } => write!(
                formatter,
                "manual-drive maximum command age {maximum_command_age_ns} ns exceeds deadman timeout {deadman_timeout_ns} ns"
            ),
        }
    }
}

impl std::error::Error for ManualDriveConfigParseError {}

/// Fully parsed manual velocity and timing limits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManualDriveConfigV1 {
    maximum_abs_forward_velocity_mps: f64,
    maximum_abs_yaw_rate_rad_s: f64,
    maximum_command_age_ns: NonZeroU64,
    deadman_timeout_ns: NonZeroU64,
}

impl ManualDriveConfigV1 {
    pub fn parse(dto: ManualDriveConfigV1Dto) -> Result<Self, ManualDriveConfigParseError> {
        if dto.schema_version != MANUAL_DRIVE_CONFIG_V1 {
            return Err(ManualDriveConfigParseError::UnsupportedSchemaVersion {
                actual: dto.schema_version,
                supported: MANUAL_DRIVE_CONFIG_V1,
            });
        }
        parse_positive_limit(
            "maximum_abs_forward_velocity_mps",
            dto.maximum_abs_forward_velocity_mps,
        )?;
        parse_positive_limit("maximum_abs_yaw_rate_rad_s", dto.maximum_abs_yaw_rate_rad_s)?;
        let maximum_command_age_ns = NonZeroU64::new(dto.maximum_command_age_ns)
            .ok_or(ManualDriveConfigParseError::ZeroMaximumCommandAge)?;
        let deadman_timeout_ns = NonZeroU64::new(dto.deadman_timeout_ns)
            .ok_or(ManualDriveConfigParseError::ZeroDeadmanTimeout)?;
        if maximum_command_age_ns > deadman_timeout_ns {
            return Err(ManualDriveConfigParseError::CommandAgeExceedsDeadman {
                maximum_command_age_ns: maximum_command_age_ns.get(),
                deadman_timeout_ns: deadman_timeout_ns.get(),
            });
        }
        Ok(Self {
            maximum_abs_forward_velocity_mps: dto.maximum_abs_forward_velocity_mps,
            maximum_abs_yaw_rate_rad_s: dto.maximum_abs_yaw_rate_rad_s,
            maximum_command_age_ns,
            deadman_timeout_ns,
        })
    }

    pub fn maximum_abs_forward_velocity_mps(self) -> f64 {
        self.maximum_abs_forward_velocity_mps
    }

    pub fn maximum_abs_yaw_rate_rad_s(self) -> f64 {
        self.maximum_abs_yaw_rate_rad_s
    }

    pub fn maximum_command_age_ns(self) -> u64 {
        self.maximum_command_age_ns.get()
    }

    pub fn deadman_timeout_ns(self) -> u64 {
        self.deadman_timeout_ns.get()
    }
}

fn parse_positive_limit(
    field: &'static str,
    value: f64,
) -> Result<(), ManualDriveConfigParseError> {
    if !value.is_finite() {
        return Err(ManualDriveConfigParseError::NonFiniteLimit { field, value });
    }
    if value <= 0.0 {
        return Err(ManualDriveConfigParseError::NonPositiveLimit { field, value });
    }
    Ok(())
}

/// Weak command intent. A stop is a distinct variant; a zero-valued velocity
/// is rejected instead of being silently reinterpreted as a stop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ManualDriveCommandKindDto {
    Velocity {
        forward_velocity_mps: f64,
        yaw_rate_rad_s: f64,
    },
    Stop,
}

/// One weak boundary command carrying the supervisor's already parsed lease ID.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManualDriveCommandDto<LeaseId> {
    pub schema_version: u32,
    pub authority_lease_id: LeaseId,
    pub sequence: u64,
    pub command: ManualDriveCommandKindDto,
}

/// Strictly ordered command identity within one authority lease.
///
/// Zero is valid. The core admits only strictly increasing values and never
/// wraps, so an exhausted sender must acquire a new supervisor lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ManualDriveSequence(u64);

impl ManualDriveSequence {
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Supervisor state projected into this dependency-neutral navigation core.
///
/// `ActiveManual` may only be constructed by the sole supervisor adapter when
/// its state is active, its mode is manual, and the deadline/ID are copied from
/// that exact lease. The deadline is exclusive, matching supervisor semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManualAuthoritySnapshot<LeaseId> {
    NotActiveManual,
    ActiveManual {
        lease_id: LeaseId,
        expires_at_exclusive: HostMonotonicTimestamp,
    },
}

impl<LeaseId> ManualAuthoritySnapshot<LeaseId> {
    pub const fn active_manual(
        lease_id: LeaseId,
        expires_at_exclusive: HostMonotonicTimestamp,
    ) -> Self {
        Self::ActiveManual {
            lease_id,
            expires_at_exclusive,
        }
    }
}

/// Differential-drive body target: +x forward in m/s and +z yaw in rad/s.
///
/// There is no lateral component and no wheel/PWM representation. Values are
/// finite, canonicalize signed zero, and have passed the configured manual
/// envelope. Existing MPC/local-safety adapters remain responsible for turning
/// this target into collision-validated actuation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyVelocityTargetV1 {
    schema_version: u32,
    forward_velocity_mps: f64,
    yaw_rate_rad_s: f64,
}

impl BodyVelocityTargetV1 {
    pub const STOP: Self = Self {
        schema_version: BODY_VELOCITY_TARGET_V1,
        forward_velocity_mps: 0.0,
        yaw_rate_rad_s: 0.0,
    };

    pub fn forward_velocity_mps(self) -> f64 {
        self.forward_velocity_mps
    }

    pub fn yaw_rate_rad_s(self) -> f64 {
        self.yaw_rate_rad_s
    }

    pub fn is_stop(self) -> bool {
        self.forward_velocity_mps == 0.0 && self.yaw_rate_rad_s == 0.0
    }

    fn parsed_velocity(forward_velocity_mps: f64, yaw_rate_rad_s: f64) -> Self {
        Self {
            schema_version: BODY_VELOCITY_TARGET_V1,
            forward_velocity_mps: canonical_zero(forward_velocity_mps),
            yaw_rate_rad_s: canonical_zero(yaw_rate_rad_s),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManualDriveAcceptedIntent {
    Velocity,
    ExplicitStop,
}

/// An accepted target bound to one lease, sequence, receipt, and exclusive
/// validity deadline. The deadline is the earlier of the manual deadman and
/// the supervisor authority expiry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManualDriveAcceptedTarget<LeaseId> {
    authority_lease_id: LeaseId,
    sequence: ManualDriveSequence,
    received_at: HostMonotonicTimestamp,
    valid_through_exclusive: HostMonotonicTimestamp,
    intent: ManualDriveAcceptedIntent,
    target: BodyVelocityTargetV1,
}

impl<LeaseId: Copy> ManualDriveAcceptedTarget<LeaseId> {
    pub fn authority_lease_id(self) -> LeaseId {
        self.authority_lease_id
    }

    pub fn sequence(self) -> ManualDriveSequence {
        self.sequence
    }

    pub fn received_at(self) -> HostMonotonicTimestamp {
        self.received_at
    }

    pub fn valid_through_exclusive(self) -> HostMonotonicTimestamp {
        self.valid_through_exclusive
    }

    pub fn intent(self) -> ManualDriveAcceptedIntent {
        self.intent
    }

    pub fn target(self) -> BodyVelocityTargetV1 {
        self.target
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ManualDriveStopCause<LeaseId> {
    NoCommand,
    ClockRegression {
        previous: HostMonotonicTimestamp,
        current: HostMonotonicTimestamp,
    },
    ClockFaultLatched,
    AuthorityNotActiveManual,
    ActiveAuthorityLeaseMismatch {
        bound: LeaseId,
        active: LeaseId,
    },
    AuthorityLeaseExpired {
        expires_at_exclusive: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
    },
    CommandAuthorityLeaseMismatch {
        bound: LeaseId,
        command: LeaseId,
    },
    UnsupportedCommandSchema {
        actual: u32,
        supported: u32,
    },
    DuplicateSequence {
        sequence: ManualDriveSequence,
    },
    SequenceRegression {
        previous: ManualDriveSequence,
        current: ManualDriveSequence,
    },
    ReceiptTimeRegression {
        previous: HostMonotonicTimestamp,
        current: HostMonotonicTimestamp,
    },
    ReceiptAfterObservation {
        received_at: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
    },
    NonFiniteVelocity {
        field: &'static str,
        value: f64,
    },
    VelocityOutsideEnvelope {
        field: &'static str,
        value: f64,
        maximum_abs: f64,
    },
    AmbiguousZeroVelocity,
    CommandStale {
        received_at: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
        maximum_age_ns: u64,
    },
    DeadmanDeadlineOverflow {
        received_at: HostMonotonicTimestamp,
        deadman_timeout_ns: u64,
    },
    DeadmanExpired {
        sequence: ManualDriveSequence,
        deadline_exclusive: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
    },
}

/// Fail-closed output. Its target is always the canonical zero body velocity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ManualDriveStopped<LeaseId> {
    bound_authority_lease_id: LeaseId,
    observed_at: HostMonotonicTimestamp,
    cause: ManualDriveStopCause<LeaseId>,
}

impl<LeaseId: Copy> ManualDriveStopped<LeaseId> {
    pub fn bound_authority_lease_id(self) -> LeaseId {
        self.bound_authority_lease_id
    }

    pub fn observed_at(self) -> HostMonotonicTimestamp {
        self.observed_at
    }

    pub fn cause(self) -> ManualDriveStopCause<LeaseId> {
        self.cause
    }

    pub fn target(self) -> BodyVelocityTargetV1 {
        BodyVelocityTargetV1::STOP
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ManualDriveOutput<LeaseId> {
    Accepted(ManualDriveAcceptedTarget<LeaseId>),
    Stopped(ManualDriveStopped<LeaseId>),
}

impl<LeaseId: Copy> ManualDriveOutput<LeaseId> {
    pub fn target(self) -> BodyVelocityTargetV1 {
        match self {
            Self::Accepted(accepted) => accepted.target(),
            Self::Stopped(stopped) => stopped.target(),
        }
    }
}

/// Stateful ordering, authority, freshness, and deadman gate for one lease.
pub struct ManualDriveCore<LeaseId> {
    config: ManualDriveConfigV1,
    bound_authority_lease_id: LeaseId,
    last_sequence: Option<ManualDriveSequence>,
    last_received_at: Option<HostMonotonicTimestamp>,
    last_observed_at: HostMonotonicTimestamp,
    clock_fault_latched: bool,
    current: Option<ManualDriveAcceptedTarget<LeaseId>>,
}

impl<LeaseId> ManualDriveCore<LeaseId>
where
    LeaseId: Copy + Eq,
{
    pub fn new(
        config: ManualDriveConfigV1,
        bound_authority_lease_id: LeaseId,
        created_at: HostMonotonicTimestamp,
    ) -> Self {
        Self {
            config,
            bound_authority_lease_id,
            last_sequence: None,
            last_received_at: None,
            last_observed_at: created_at,
            clock_fault_latched: false,
            current: None,
        }
    }

    pub fn config(&self) -> ManualDriveConfigV1 {
        self.config
    }

    pub fn bound_authority_lease_id(&self) -> LeaseId {
        self.bound_authority_lease_id
    }

    pub fn last_sequence(&self) -> Option<ManualDriveSequence> {
        self.last_sequence
    }

    pub fn current(&self) -> Option<ManualDriveAcceptedTarget<LeaseId>> {
        self.current
    }

    /// Consume one weak command exactly once and return the resulting target.
    ///
    /// `received_at` must be stamped by the trusted host receiver, not supplied
    /// by a remote operator. `observed_at` is the current host-monotonic time.
    pub fn ingest(
        &mut self,
        dto: ManualDriveCommandDto<LeaseId>,
        received_at: HostMonotonicTimestamp,
        observed_at: HostMonotonicTimestamp,
        authority: ManualAuthoritySnapshot<LeaseId>,
    ) -> ManualDriveOutput<LeaseId> {
        if let Some(cause) = self.observe_time(observed_at) {
            return self.stop(observed_at, cause);
        }
        if let Some(cause) = self.authority_stop_cause(authority, observed_at) {
            return self.stop(observed_at, cause);
        }
        let authority_expires_at_exclusive = match authority {
            ManualAuthoritySnapshot::ActiveManual {
                expires_at_exclusive,
                ..
            } => expires_at_exclusive,
            ManualAuthoritySnapshot::NotActiveManual => {
                unreachable!("non-manual authority returned a stop cause")
            }
        };
        if dto.authority_lease_id != self.bound_authority_lease_id {
            return self.stop(
                observed_at,
                ManualDriveStopCause::CommandAuthorityLeaseMismatch {
                    bound: self.bound_authority_lease_id,
                    command: dto.authority_lease_id,
                },
            );
        }
        if dto.schema_version != MANUAL_DRIVE_COMMAND_V1 {
            return self.stop(
                observed_at,
                ManualDriveStopCause::UnsupportedCommandSchema {
                    actual: dto.schema_version,
                    supported: MANUAL_DRIVE_COMMAND_V1,
                },
            );
        }

        let sequence = ManualDriveSequence::from_raw(dto.sequence);
        if let Some(previous) = self.last_sequence {
            if sequence == previous {
                return self.stop(
                    observed_at,
                    ManualDriveStopCause::DuplicateSequence { sequence },
                );
            }
            if sequence < previous {
                return self.stop(
                    observed_at,
                    ManualDriveStopCause::SequenceRegression {
                        previous,
                        current: sequence,
                    },
                );
            }
        }
        self.last_sequence = Some(sequence);

        if let Some(previous) = self.last_received_at
            && received_at < previous
        {
            return self.stop(
                observed_at,
                ManualDriveStopCause::ReceiptTimeRegression {
                    previous,
                    current: received_at,
                },
            );
        }
        if received_at > observed_at {
            return self.stop(
                observed_at,
                ManualDriveStopCause::ReceiptAfterObservation {
                    received_at,
                    observed_at,
                },
            );
        }
        self.last_received_at = Some(received_at);

        let (intent, target) = match self.parse_target(dto.command) {
            Ok(parsed) => parsed,
            Err(cause) => return self.stop(observed_at, cause),
        };
        let age_ns = observed_at.as_nanos() - received_at.as_nanos();
        if age_ns >= self.config.maximum_command_age_ns() {
            return self.stop(
                observed_at,
                ManualDriveStopCause::CommandStale {
                    received_at,
                    observed_at,
                    maximum_age_ns: self.config.maximum_command_age_ns(),
                },
            );
        }
        let Some(deadline_ns) = received_at
            .as_nanos()
            .checked_add(self.config.deadman_timeout_ns())
        else {
            return self.stop(
                observed_at,
                ManualDriveStopCause::DeadmanDeadlineOverflow {
                    received_at,
                    deadman_timeout_ns: self.config.deadman_timeout_ns(),
                },
            );
        };
        let deadman_deadline = HostMonotonicTimestamp::from_nanos(deadline_ns);
        let accepted = ManualDriveAcceptedTarget {
            authority_lease_id: self.bound_authority_lease_id,
            sequence,
            received_at,
            valid_through_exclusive: deadman_deadline.min(authority_expires_at_exclusive),
            intent,
            target,
        };
        self.current = Some(accepted);
        ManualDriveOutput::Accepted(accepted)
    }

    /// Re-evaluate authority and the deadman without accepting a new command.
    pub fn tick(
        &mut self,
        observed_at: HostMonotonicTimestamp,
        authority: ManualAuthoritySnapshot<LeaseId>,
    ) -> ManualDriveOutput<LeaseId> {
        if let Some(cause) = self.observe_time(observed_at) {
            return self.stop(observed_at, cause);
        }
        if let Some(cause) = self.authority_stop_cause(authority, observed_at) {
            return self.stop(observed_at, cause);
        }
        let Some(current) = self.current else {
            return self.stop(observed_at, ManualDriveStopCause::NoCommand);
        };
        if observed_at >= current.valid_through_exclusive {
            return self.stop(
                observed_at,
                ManualDriveStopCause::DeadmanExpired {
                    sequence: current.sequence,
                    deadline_exclusive: current.valid_through_exclusive,
                    observed_at,
                },
            );
        }
        ManualDriveOutput::Accepted(current)
    }

    fn observe_time(
        &mut self,
        observed_at: HostMonotonicTimestamp,
    ) -> Option<ManualDriveStopCause<LeaseId>> {
        if self.clock_fault_latched {
            return Some(ManualDriveStopCause::ClockFaultLatched);
        }
        if observed_at < self.last_observed_at {
            let previous = self.last_observed_at;
            self.clock_fault_latched = true;
            return Some(ManualDriveStopCause::ClockRegression {
                previous,
                current: observed_at,
            });
        }
        self.last_observed_at = observed_at;
        None
    }

    fn authority_stop_cause(
        &self,
        authority: ManualAuthoritySnapshot<LeaseId>,
        observed_at: HostMonotonicTimestamp,
    ) -> Option<ManualDriveStopCause<LeaseId>> {
        match authority {
            ManualAuthoritySnapshot::NotActiveManual => {
                Some(ManualDriveStopCause::AuthorityNotActiveManual)
            }
            ManualAuthoritySnapshot::ActiveManual {
                lease_id,
                expires_at_exclusive: _,
            } if lease_id != self.bound_authority_lease_id => {
                Some(ManualDriveStopCause::ActiveAuthorityLeaseMismatch {
                    bound: self.bound_authority_lease_id,
                    active: lease_id,
                })
            }
            ManualAuthoritySnapshot::ActiveManual {
                expires_at_exclusive,
                ..
            } if observed_at >= expires_at_exclusive => {
                Some(ManualDriveStopCause::AuthorityLeaseExpired {
                    expires_at_exclusive,
                    observed_at,
                })
            }
            ManualAuthoritySnapshot::ActiveManual { .. } => None,
        }
    }

    fn parse_target(
        &self,
        command: ManualDriveCommandKindDto,
    ) -> Result<(ManualDriveAcceptedIntent, BodyVelocityTargetV1), ManualDriveStopCause<LeaseId>>
    {
        let ManualDriveCommandKindDto::Velocity {
            forward_velocity_mps,
            yaw_rate_rad_s,
        } = command
        else {
            return Ok((
                ManualDriveAcceptedIntent::ExplicitStop,
                BodyVelocityTargetV1::STOP,
            ));
        };
        for (field, value, maximum_abs) in [
            (
                "forward_velocity_mps",
                forward_velocity_mps,
                self.config.maximum_abs_forward_velocity_mps(),
            ),
            (
                "yaw_rate_rad_s",
                yaw_rate_rad_s,
                self.config.maximum_abs_yaw_rate_rad_s(),
            ),
        ] {
            if !value.is_finite() {
                return Err(ManualDriveStopCause::NonFiniteVelocity { field, value });
            }
            if value.abs() > maximum_abs {
                return Err(ManualDriveStopCause::VelocityOutsideEnvelope {
                    field,
                    value,
                    maximum_abs,
                });
            }
        }
        let target = BodyVelocityTargetV1::parsed_velocity(forward_velocity_mps, yaw_rate_rad_s);
        if target.is_stop() {
            return Err(ManualDriveStopCause::AmbiguousZeroVelocity);
        }
        Ok((ManualDriveAcceptedIntent::Velocity, target))
    }

    fn stop(
        &mut self,
        observed_at: HostMonotonicTimestamp,
        cause: ManualDriveStopCause<LeaseId>,
    ) -> ManualDriveOutput<LeaseId> {
        self.current = None;
        ManualDriveOutput::Stopped(ManualDriveStopped {
            bound_authority_lease_id: self.bound_authority_lease_id,
            observed_at,
            cause,
        })
    }
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct LeaseId(u64);

    fn at(nanos: u64) -> HostMonotonicTimestamp {
        HostMonotonicTimestamp::from_nanos(nanos)
    }

    fn config_dto() -> ManualDriveConfigV1Dto {
        ManualDriveConfigV1Dto {
            schema_version: MANUAL_DRIVE_CONFIG_V1,
            maximum_abs_forward_velocity_mps: 0.5,
            maximum_abs_yaw_rate_rad_s: 1.25,
            maximum_command_age_ns: 10,
            deadman_timeout_ns: 20,
        }
    }

    fn config() -> ManualDriveConfigV1 {
        ManualDriveConfigV1::parse(config_dto()).expect("test config")
    }

    fn authority(lease_id: LeaseId, expires_at: u64) -> ManualAuthoritySnapshot<LeaseId> {
        ManualAuthoritySnapshot::active_manual(lease_id, at(expires_at))
    }

    fn velocity(
        lease_id: LeaseId,
        sequence: u64,
        forward_velocity_mps: f64,
        yaw_rate_rad_s: f64,
    ) -> ManualDriveCommandDto<LeaseId> {
        ManualDriveCommandDto {
            schema_version: MANUAL_DRIVE_COMMAND_V1,
            authority_lease_id: lease_id,
            sequence,
            command: ManualDriveCommandKindDto::Velocity {
                forward_velocity_mps,
                yaw_rate_rad_s,
            },
        }
    }

    fn stop(lease_id: LeaseId, sequence: u64) -> ManualDriveCommandDto<LeaseId> {
        ManualDriveCommandDto {
            schema_version: MANUAL_DRIVE_COMMAND_V1,
            authority_lease_id: lease_id,
            sequence,
            command: ManualDriveCommandKindDto::Stop,
        }
    }

    fn accepted(output: ManualDriveOutput<LeaseId>) -> ManualDriveAcceptedTarget<LeaseId> {
        match output {
            ManualDriveOutput::Accepted(target) => target,
            ManualDriveOutput::Stopped(stopped) => {
                panic!("expected accepted target, got {:?}", stopped.cause())
            }
        }
    }

    fn stopped(output: ManualDriveOutput<LeaseId>) -> ManualDriveStopped<LeaseId> {
        match output {
            ManualDriveOutput::Accepted(_) => panic!("expected fail-closed stop"),
            ManualDriveOutput::Stopped(stopped) => stopped,
        }
    }

    #[test]
    fn config_parses_units_and_rejects_invalid_domains() {
        let parsed = config();
        assert_eq!(parsed.maximum_abs_forward_velocity_mps(), 0.5);
        assert_eq!(parsed.maximum_abs_yaw_rate_rad_s(), 1.25);
        assert_eq!(parsed.maximum_command_age_ns(), 10);
        assert_eq!(parsed.deadman_timeout_ns(), 20);

        let mut dto = config_dto();
        dto.schema_version = 2;
        assert!(matches!(
            ManualDriveConfigV1::parse(dto),
            Err(ManualDriveConfigParseError::UnsupportedSchemaVersion { .. })
        ));
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut dto = config_dto();
            dto.maximum_abs_forward_velocity_mps = value;
            assert!(matches!(
                ManualDriveConfigV1::parse(dto),
                Err(ManualDriveConfigParseError::NonFiniteLimit { .. })
            ));
        }
        for value in [0.0, -0.0, -f64::EPSILON] {
            let mut dto = config_dto();
            dto.maximum_abs_yaw_rate_rad_s = value;
            assert!(matches!(
                ManualDriveConfigV1::parse(dto),
                Err(ManualDriveConfigParseError::NonPositiveLimit { .. })
            ));
        }
        let mut dto = config_dto();
        dto.maximum_command_age_ns = 0;
        assert_eq!(
            ManualDriveConfigV1::parse(dto),
            Err(ManualDriveConfigParseError::ZeroMaximumCommandAge)
        );
        let mut dto = config_dto();
        dto.deadman_timeout_ns = 0;
        assert_eq!(
            ManualDriveConfigV1::parse(dto),
            Err(ManualDriveConfigParseError::ZeroDeadmanTimeout)
        );
        let mut dto = config_dto();
        dto.maximum_command_age_ns = 21;
        assert!(matches!(
            ManualDriveConfigV1::parse(dto),
            Err(ManualDriveConfigParseError::CommandAgeExceedsDeadman { .. })
        ));
    }

    #[test]
    fn finite_velocity_is_canonical_bounded_and_never_pwm() {
        let lease = LeaseId(1);
        let mut core = ManualDriveCore::new(config(), lease, at(0));
        let target = accepted(core.ingest(
            velocity(lease, 0, -0.5, 1.25),
            at(5),
            at(5),
            authority(lease, 100),
        ));
        assert_eq!(target.authority_lease_id(), lease);
        assert_eq!(target.sequence().get(), 0);
        assert_eq!(target.received_at(), at(5));
        assert_eq!(target.valid_through_exclusive(), at(25));
        assert_eq!(target.intent(), ManualDriveAcceptedIntent::Velocity);
        assert_eq!(target.target().forward_velocity_mps(), -0.5);
        assert_eq!(target.target().yaw_rate_rad_s(), 1.25);
        assert!(!target.target().is_stop());

        let rejected = stopped(core.ingest(
            velocity(lease, 1, 0.500_000_000_000_000_1, 0.0),
            at(6),
            at(6),
            authority(lease, 100),
        ));
        assert!(matches!(
            rejected.cause(),
            ManualDriveStopCause::VelocityOutsideEnvelope {
                field: "forward_velocity_mps",
                ..
            }
        ));
        assert!(rejected.target().is_stop());
    }

    #[test]
    fn explicit_stop_is_distinct_from_ambiguous_zero_velocity() {
        let lease = LeaseId(1);
        let mut core = ManualDriveCore::new(config(), lease, at(0));
        let rejected = stopped(core.ingest(
            velocity(lease, 1, -0.0, 0.0),
            at(1),
            at(1),
            authority(lease, 100),
        ));
        assert_eq!(
            rejected.cause(),
            ManualDriveStopCause::AmbiguousZeroVelocity
        );
        assert!(rejected.target().is_stop());

        let explicit = accepted(core.ingest(stop(lease, 2), at(2), at(2), authority(lease, 100)));
        assert_eq!(explicit.intent(), ManualDriveAcceptedIntent::ExplicitStop);
        assert_eq!(explicit.target(), BodyVelocityTargetV1::STOP);
        assert_eq!(
            accepted(core.tick(at(10), authority(lease, 100))).intent(),
            ManualDriveAcceptedIntent::ExplicitStop
        );
    }

    #[test]
    fn exact_authority_identity_mode_and_exclusive_deadline_fail_closed() {
        let lease = LeaseId(7);
        let other = LeaseId(8);
        let mut core = ManualDriveCore::new(config(), lease, at(0));
        assert!(matches!(
            stopped(core.ingest(
                velocity(lease, 1, 0.1, 0.0),
                at(1),
                at(1),
                ManualAuthoritySnapshot::NotActiveManual,
            ))
            .cause(),
            ManualDriveStopCause::AuthorityNotActiveManual
        ));
        assert!(matches!(
            stopped(core.ingest(
                velocity(lease, 1, 0.1, 0.0),
                at(2),
                at(2),
                authority(other, 100),
            ))
            .cause(),
            ManualDriveStopCause::ActiveAuthorityLeaseMismatch {
                bound: LeaseId(7),
                active: LeaseId(8)
            }
        ));
        assert!(matches!(
            stopped(core.ingest(
                velocity(other, 1, 0.1, 0.0),
                at(3),
                at(3),
                authority(lease, 100),
            ))
            .cause(),
            ManualDriveStopCause::CommandAuthorityLeaseMismatch { .. }
        ));
        assert!(matches!(
            stopped(core.ingest(
                velocity(lease, 1, 0.1, 0.0),
                at(4),
                at(4),
                authority(lease, 4),
            ))
            .cause(),
            ManualDriveStopCause::AuthorityLeaseExpired { .. }
        ));
    }

    #[test]
    fn authority_loss_or_handover_stops_an_existing_motion_target() {
        let lease = LeaseId(7);
        let other = LeaseId(8);
        let mut core = ManualDriveCore::new(config(), lease, at(0));
        accepted(core.ingest(
            velocity(lease, 1, 0.2, 0.1),
            at(1),
            at(1),
            authority(lease, 100),
        ));
        let lost = stopped(core.tick(at(2), ManualAuthoritySnapshot::NotActiveManual));
        assert_eq!(lost.cause(), ManualDriveStopCause::AuthorityNotActiveManual);
        assert!(lost.target().is_stop());

        accepted(core.ingest(
            velocity(lease, 2, 0.2, 0.1),
            at(3),
            at(3),
            authority(lease, 100),
        ));
        let handover = stopped(core.tick(at(4), authority(other, 100)));
        assert_eq!(
            handover.cause(),
            ManualDriveStopCause::ActiveAuthorityLeaseMismatch {
                bound: lease,
                active: other,
            }
        );
        assert!(handover.target().is_stop());
    }

    #[test]
    fn sequence_is_strict_and_invalid_payload_consumes_its_identity() {
        let lease = LeaseId(1);
        let mut core = ManualDriveCore::new(config(), lease, at(0));
        accepted(core.ingest(
            velocity(lease, 10, 0.1, 0.0),
            at(1),
            at(1),
            authority(lease, 100),
        ));
        assert!(matches!(
            stopped(core.ingest(
                velocity(lease, 10, 0.2, 0.0),
                at(2),
                at(2),
                authority(lease, 100),
            ))
            .cause(),
            ManualDriveStopCause::DuplicateSequence { .. }
        ));
        assert!(matches!(
            stopped(core.ingest(
                velocity(lease, 9, 0.2, 0.0),
                at(3),
                at(3),
                authority(lease, 100),
            ))
            .cause(),
            ManualDriveStopCause::SequenceRegression { .. }
        ));
        assert!(matches!(
            stopped(core.ingest(
                velocity(lease, 11, f64::NAN, 0.0),
                at(4),
                at(4),
                authority(lease, 100),
            ))
            .cause(),
            ManualDriveStopCause::NonFiniteVelocity { .. }
        ));
        assert!(matches!(
            stopped(core.ingest(
                velocity(lease, 11, 0.2, 0.0),
                at(5),
                at(5),
                authority(lease, 100),
            ))
            .cause(),
            ManualDriveStopCause::DuplicateSequence { .. }
        ));
        assert_eq!(
            accepted(core.ingest(
                velocity(lease, 12, 0.2, 0.0),
                at(6),
                at(6),
                authority(lease, 100),
            ))
            .sequence()
            .get(),
            12
        );

        let mut exhausted = ManualDriveCore::new(config(), lease, at(0));
        accepted(exhausted.ingest(stop(lease, u64::MAX), at(1), at(1), authority(lease, 100)));
        assert!(matches!(
            stopped(exhausted.ingest(stop(lease, 0), at(2), at(2), authority(lease, 100),)).cause(),
            ManualDriveStopCause::SequenceRegression { .. }
        ));
    }

    #[test]
    fn ingress_freshness_and_deadman_deadlines_are_exclusive() {
        let lease = LeaseId(1);
        let mut core = ManualDriveCore::new(config(), lease, at(0));
        let target = accepted(core.ingest(
            velocity(lease, 1, 0.1, 0.0),
            at(100),
            at(109),
            authority(lease, 1_000),
        ));
        assert_eq!(target.valid_through_exclusive(), at(120));
        assert!(
            !core
                .tick(at(119), authority(lease, 1_000))
                .target()
                .is_stop()
        );
        let expired = stopped(core.tick(at(120), authority(lease, 1_000)));
        assert!(matches!(
            expired.cause(),
            ManualDriveStopCause::DeadmanExpired {
                deadline_exclusive,
                observed_at,
                ..
            } if deadline_exclusive == at(120) && observed_at == at(120)
        ));

        let stale = stopped(core.ingest(
            velocity(lease, 2, 0.1, 0.0),
            at(130),
            at(140),
            authority(lease, 1_000),
        ));
        assert!(matches!(
            stale.cause(),
            ManualDriveStopCause::CommandStale {
                maximum_age_ns: 10,
                ..
            }
        ));
        assert!(stale.target().is_stop());
        assert!(
            accepted(core.ingest(
                velocity(lease, 3, 0.1, 0.0),
                at(141),
                at(141),
                authority(lease, 1_000),
            ))
            .target()
            .forward_velocity_mps()
                > 0.0
        );

        let lease_limited = accepted(core.ingest(
            velocity(lease, 4, 0.1, 0.0),
            at(150),
            at(150),
            authority(lease, 155),
        ));
        assert_eq!(
            lease_limited.valid_through_exclusive(),
            at(155),
            "an accepted target must never claim validity beyond authority"
        );
    }

    #[test]
    fn host_clock_regression_latches_stop_without_resumption() {
        let lease = LeaseId(1);
        let mut core = ManualDriveCore::new(config(), lease, at(10));
        accepted(core.ingest(
            velocity(lease, 1, 0.1, 0.0),
            at(11),
            at(11),
            authority(lease, 100),
        ));
        let regressed = stopped(core.tick(at(10), authority(lease, 100)));
        assert!(matches!(
            regressed.cause(),
            ManualDriveStopCause::ClockRegression { .. }
        ));
        assert!(regressed.target().is_stop());
        assert!(matches!(
            stopped(core.ingest(
                velocity(lease, 2, 0.2, 0.0),
                at(12),
                at(12),
                authority(lease, 100),
            ))
            .cause(),
            ManualDriveStopCause::ClockFaultLatched
        ));
    }

    #[test]
    fn receipt_order_future_receipt_and_deadline_overflow_are_typed_stops() {
        let lease = LeaseId(1);
        let mut core = ManualDriveCore::new(config(), lease, at(0));
        accepted(core.ingest(
            velocity(lease, 1, 0.1, 0.0),
            at(10),
            at(10),
            authority(lease, u64::MAX),
        ));
        assert!(matches!(
            stopped(core.ingest(
                velocity(lease, 2, 0.1, 0.0),
                at(9),
                at(11),
                authority(lease, u64::MAX),
            ))
            .cause(),
            ManualDriveStopCause::ReceiptTimeRegression { .. }
        ));
        assert!(matches!(
            stopped(core.ingest(
                velocity(lease, 3, 0.1, 0.0),
                at(13),
                at(12),
                authority(lease, u64::MAX),
            ))
            .cause(),
            ManualDriveStopCause::ReceiptAfterObservation { .. }
        ));
        assert!(matches!(
            stopped(core.ingest(
                velocity(lease, 4, 0.1, 0.0),
                at(u64::MAX - 10),
                at(u64::MAX - 10),
                authority(lease, u64::MAX),
            ))
            .cause(),
            ManualDriveStopCause::DeadmanDeadlineOverflow { .. }
        ));
    }

    #[test]
    fn deterministic_stream_never_emits_motion_without_fresh_authority_and_command() {
        let lease = LeaseId(42);
        let mut first = ManualDriveCore::new(config(), lease, at(0));
        let mut second = ManualDriveCore::new(config(), lease, at(0));
        for sequence in 0..1_000_u64 {
            let now = sequence + 1;
            let command = if sequence % 17 == 0 {
                stop(lease, sequence)
            } else {
                velocity(
                    lease,
                    sequence,
                    ((sequence % 9) as f64 - 4.0) * 0.1,
                    ((sequence % 7) as f64 - 3.0) * 0.2,
                )
            };
            let left = first.ingest(command, at(now), at(now), authority(lease, 2_000));
            let right = second.ingest(command, at(now), at(now), authority(lease, 2_000));
            assert_eq!(left, right);
            match left {
                ManualDriveOutput::Accepted(target) => {
                    assert_eq!(target.authority_lease_id(), lease);
                    assert_eq!(target.sequence().get(), sequence);
                    assert!(target.target().forward_velocity_mps().is_finite());
                    assert!(target.target().yaw_rate_rad_s().is_finite());
                }
                ManualDriveOutput::Stopped(stopped) => {
                    assert!(stopped.target().is_stop());
                }
            }
        }
        assert!(matches!(
            stopped(first.tick(at(2_000), authority(lease, 2_000))).cause(),
            ManualDriveStopCause::AuthorityLeaseExpired { .. }
        ));
    }
}
