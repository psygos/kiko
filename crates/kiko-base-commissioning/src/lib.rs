#![forbid(unsafe_code)]

//! Hardware- and transport-independent encoderless base commissioning.
//!
//! The crate has two deliberately separate responsibilities:
//! - parse an already time-aligned physical evidence dataset and identify the
//!   unequal-wheel first-order plant consumed by Kiko's MPC; and
//! - produce a small, bounded commissioning excitation program whose caller
//!   remains responsible for transport, authorization, collision safety, and
//!   recording the applied controller result.
//!
//! Translation is never inferred from IMU yaw rate. Differential-drive yaw
//! identifies only the wheel-speed difference; visual forward velocity is
//! required to identify their sum.

mod commissioning;
mod data;
mod fit;
mod identity;
mod time;

pub use commissioning::{
    Cancellation, CanonicalPwmCommand, CommissioningAction, CommissioningConfigParseError,
    CommissioningConfigV1, CommissioningConfigV1Dto, CommissioningController,
    CommissioningEvidence, CommissioningEvidenceParseError, CommissioningEvidenceV1Dto,
    CommissioningState, CommissioningStep, CommissioningStopReason, EvidenceKind, ExcitationKind,
};
pub use data::{
    DatasetParseError, IdentificationDatasetV1, IdentificationDatasetV1Dto,
    IdentificationSampleV1Dto, PlantFitConfigParseError, PlantFitConfigV1, PlantFitConfigV1Dto,
};
pub use fit::{
    CoverageGate, FitError, HoldoutResidualsV1, IdentifiedPlantV1, IdentifiedWheelPlantV1,
    LateralVelocityEvidence, NumericalStage, PlantSupportV1, ResidualMetric, WheelSide,
    fit_first_order_plant,
};
pub use identity::{BoundedId, IdentifierError};
pub use time::MonotonicTimestampNs;

/// Schema version shared by the dataset, fitter configuration, and output.
pub const BASE_IDENTIFICATION_V1: u32 = 1;

/// Stable method identity suitable for downstream provenance.
pub const IDENTIFICATION_METHOD_ID: &str = "kiko.encoderless.first_order.v1";
