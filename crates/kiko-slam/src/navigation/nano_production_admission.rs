//! Fail-closed production admission for the sole Nano motion owner.
//!
//! This boundary deliberately accepts only already-parsed policy/configuration
//! and already-loaded no-follow file evidence. It performs the one exact
//! expected/observed inventory comparison, binds physical actuation to the
//! selected plant artifact, enters the process supervisor in `Disarmed`, and
//! only then promotes the zero-only controller owner. It exposes no arm or
//! motion operation of its own.

use std::fmt;

use kiko_device_inventory::{
    ArtifactId, ArtifactRelativePath, InventoryMismatchReport, LoadedExpectedManifestV1,
    ManifestArtifactHashes, ObservedDeviceInventoryV1, admit_exact_inventory,
};
use kiko_head_runtime::HeadGazeBaseZeroExclusiveLeaseIssuer;
use kiko_supervisor_core::ReadinessEpoch;
use robot_command_client::{
    AppliedCommandReceipt, ControllerSession, DisarmReceipt, VerifiedControllerAcquisition,
};

use super::actuation::LiveActuationError;
use super::{
    ActuationAdmissionError, AdmittedNavigationActuationConfigV1, DisarmedNanoStartupParts,
    LiveMpcControlDriver, NanoAgentPolicyConfigV3, NanoStartupAdmissionError,
    NanoStartupSupervisorError, NavigationActuationConfigV1, NavigationClockEpoch,
    PendingLiveMpcAdmissionError, PendingLiveMpcControlDriver, ProductionObservedDeviceInventoryV1,
};
use crate::HostMonotonicTimestamp;

/// Ordered timestamps for one process-lifetime production admission.
///
/// Construction proves that both lifecycle observations belong at or after
/// the selected host clock origin and that readiness was not observed before
/// inventory began. Equal timestamps are valid because the supervisor permits
/// multiple ordered transitions at one monotonic observation. Construct this
/// value before acquiring controller ownership: this parser owns no controller
/// and therefore cannot produce stop evidence on failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanoProductionAdmissionTimeline {
    readiness_epoch: ReadinessEpoch,
    clock_epoch: NavigationClockEpoch,
    inventory_started_at: HostMonotonicTimestamp,
    readiness_admitted_at: HostMonotonicTimestamp,
}

impl NanoProductionAdmissionTimeline {
    pub fn try_new(
        readiness_epoch: ReadinessEpoch,
        clock_epoch: NavigationClockEpoch,
        inventory_started_at: HostMonotonicTimestamp,
        readiness_admitted_at: HostMonotonicTimestamp,
    ) -> Result<Self, NanoProductionAdmissionTimelineError> {
        if inventory_started_at < clock_epoch.origin() {
            return Err(
                NanoProductionAdmissionTimelineError::InventoryBeforeClockOrigin {
                    clock_origin_ns: clock_epoch.origin().as_nanos(),
                    inventory_started_at_ns: inventory_started_at.as_nanos(),
                },
            );
        }
        if readiness_admitted_at < inventory_started_at {
            return Err(
                NanoProductionAdmissionTimelineError::ReadinessBeforeInventory {
                    inventory_started_at_ns: inventory_started_at.as_nanos(),
                    readiness_admitted_at_ns: readiness_admitted_at.as_nanos(),
                },
            );
        }
        Ok(Self {
            readiness_epoch,
            clock_epoch,
            inventory_started_at,
            readiness_admitted_at,
        })
    }

    pub const fn readiness_epoch(self) -> ReadinessEpoch {
        self.readiness_epoch
    }

    pub const fn clock_epoch(self) -> NavigationClockEpoch {
        self.clock_epoch
    }

    pub const fn inventory_started_at(self) -> HostMonotonicTimestamp {
        self.inventory_started_at
    }

    pub const fn readiness_admitted_at(self) -> HostMonotonicTimestamp {
        self.readiness_admitted_at
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanoProductionAdmissionTimelineError {
    InventoryBeforeClockOrigin {
        clock_origin_ns: u64,
        inventory_started_at_ns: u64,
    },
    ReadinessBeforeInventory {
        inventory_started_at_ns: u64,
        readiness_admitted_at_ns: u64,
    },
}

impl fmt::Display for NanoProductionAdmissionTimelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid Nano production admission timeline: {self:?}"
        )
    }
}

impl std::error::Error for NanoProductionAdmissionTimelineError {}

/// Fully admitted, disarmed inputs for construction of the sole Nano runtime.
///
/// The physical driver is present but this type intentionally has no arm,
/// command, or mutable-driver accessor. Runtime construction consumes the
/// value through [`Self::into_parts`].
#[must_use = "dropping an admitted physical session cannot prove that the controller stopped"]
pub struct PreparedNanoProductionRuntime {
    startup: DisarmedNanoStartupParts,
    actuation: AdmittedNavigationActuationConfigV1,
    physical_driver: LiveMpcControlDriver,
    initial_zero: AppliedCommandReceipt,
    head_gaze_lease_issuer: HeadGazeBaseZeroExclusiveLeaseIssuer,
}

impl PreparedNanoProductionRuntime {
    /// Perform the complete production startup transition.
    ///
    /// `pending_controller` is already a live, zero-only controller session,
    /// so every failure from this point explicitly consumes it through a stop.
    #[allow(clippy::too_many_arguments)]
    pub fn admit(
        policy: NanoAgentPolicyConfigV3,
        loaded_manifest: LoadedExpectedManifestV1,
        artifact_hashes: ManifestArtifactHashes,
        observed_inventory: ProductionObservedDeviceInventoryV1,
        actuation_config: NavigationActuationConfigV1,
        plant_artifact_id: ArtifactId,
        plant_artifact_relative_path: ArtifactRelativePath,
        mut pending_controller: PendingLiveMpcControlDriver,
        initial_zero: AppliedCommandReceipt,
        timeline: NanoProductionAdmissionTimeline,
    ) -> Result<Self, NanoProductionAdmissionError> {
        let head_gaze_lease_issuer =
            match pending_controller.install_head_gaze_base_interlock(&initial_zero) {
                Ok(issuer) => issuer,
                Err(source) => {
                    let stop = match pending_controller.disarm() {
                        Ok(receipt) => NanoProductionAdmissionStop::Confirmed(receipt),
                        Err(stop) => NanoProductionAdmissionStop::Uncertain(stop),
                    };
                    return Err(NanoProductionAdmissionError::PrePromotion {
                        primary: Box::new(NanoProductionAdmissionPrimaryError::HeadGazeInterlock(
                            source,
                        )),
                        stop,
                    });
                }
            };
        prepare_with_pending_controller(
            policy,
            loaded_manifest,
            artifact_hashes,
            observed_inventory.into_inventory(),
            actuation_config,
            plant_artifact_id,
            plant_artifact_relative_path,
            pending_controller,
            initial_zero,
            timeline,
        )
        .map(
            |PreparedNanoProductionRuntimeWithDriver {
                 startup,
                 actuation,
                 physical_driver,
                 initial_zero,
             }| Self {
                startup,
                actuation,
                physical_driver,
                initial_zero,
                head_gaze_lease_issuer,
            },
        )
        .map_err(NanoProductionAdmissionError::from_internal)
    }

    pub const fn startup(&self) -> &DisarmedNanoStartupParts {
        &self.startup
    }

    pub const fn actuation(&self) -> &AdmittedNavigationActuationConfigV1 {
        &self.actuation
    }

    pub const fn initial_zero(&self) -> &AppliedCommandReceipt {
        &self.initial_zero
    }

    /// Borrow the cloneable request endpoint retained by this admitted base
    /// owner. It mints no lease by itself and becomes permanently faulted when
    /// the non-cloneable base interlock owner is dropped.
    pub const fn head_gaze_lease_issuer(&self) -> &HeadGazeBaseZeroExclusiveLeaseIssuer {
        &self.head_gaze_lease_issuer
    }

    /// Consume an admitted runtime before a live owner has been constructed.
    ///
    /// This is the only truthful post-bootstrap cancellation path: the exact
    /// controller session is explicitly disarmed and its receipt or stop
    /// uncertainty is returned to the launch owner.
    pub fn abort_before_owner(mut self) -> Result<DisarmReceipt, LiveActuationError> {
        self.physical_driver.disarm()
    }

    pub fn into_parts(self) -> PreparedNanoProductionRuntimeParts {
        PreparedNanoProductionRuntimeParts {
            startup: self.startup,
            actuation: self.actuation,
            physical_driver: self.physical_driver,
            initial_zero: self.initial_zero,
            head_gaze_lease_issuer: self.head_gaze_lease_issuer,
        }
    }
}

/// Owned output for handing the completed admission to the sole live owner.
///
/// Receiving these parts does not itself arm Kiko. The supervisor remains in
/// `Disarmed`, and the retained zero receipt is evidence rather than authority.
#[must_use = "the sole live owner must retain or explicitly disarm the physical driver"]
pub struct PreparedNanoProductionRuntimeParts {
    pub startup: DisarmedNanoStartupParts,
    pub actuation: AdmittedNavigationActuationConfigV1,
    pub physical_driver: LiveMpcControlDriver,
    pub initial_zero: AppliedCommandReceipt,
    pub head_gaze_lease_issuer: HeadGazeBaseZeroExclusiveLeaseIssuer,
}

/// Primary reason a pre-promotion production admission failed.
#[derive(Debug)]
pub enum NanoProductionAdmissionPrimaryError {
    ControllerEvidence(LiveActuationError),
    HeadGazeInterlock(LiveActuationError),
    InitialZeroSessionMismatch {
        acquisition: ControllerSession,
        receipt: Box<AppliedCommandReceipt>,
    },
    InitialReceiptWasNotConfirmedZero {
        receipt: Box<AppliedCommandReceipt>,
    },
    ExactInventory(InventoryMismatchReport),
    Actuation(ActuationAdmissionError),
    Startup(NanoStartupAdmissionError),
    Supervisor(NanoStartupSupervisorError),
}

impl fmt::Display for NanoProductionAdmissionPrimaryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Nano production startup evidence failed: {self:?}"
        )
    }
}

impl std::error::Error for NanoProductionAdmissionPrimaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ControllerEvidence(source) | Self::HeadGazeInterlock(source) => Some(source),
            Self::ExactInventory(source) => Some(source),
            Self::Actuation(source) => Some(source),
            Self::Startup(source) => Some(source),
            Self::Supervisor(source) => Some(source),
            Self::InitialZeroSessionMismatch { .. }
            | Self::InitialReceiptWasNotConfirmedZero { .. } => None,
        }
    }
}

/// What is known about the explicit stop attempted after a pre-promotion
/// failure.
pub enum NanoProductionAdmissionStop {
    Confirmed(DisarmReceipt),
    Uncertain(LiveActuationError),
}

impl fmt::Debug for NanoProductionAdmissionStop {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for NanoProductionAdmissionStop {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Confirmed(receipt) => write!(
                formatter,
                "controller stop confirmed at {} ns",
                receipt.acknowledged_at().nanos_since_clock_start()
            ),
            Self::Uncertain(source) => write!(formatter, "controller stop uncertain: {source}"),
        }
    }
}

/// Fail-closed production admission error.
///
/// Pre-promotion failures preserve both the primary evidence error and the
/// result of the explicit stop. Promotion failures retain
/// [`PendingLiveMpcAdmissionError`], whose configuration-mismatch variant
/// already carries its own confirmed-versus-uncertain explicit stop evidence.
pub enum NanoProductionAdmissionError {
    PrePromotion {
        primary: Box<NanoProductionAdmissionPrimaryError>,
        stop: NanoProductionAdmissionStop,
    },
    Promotion(PendingLiveMpcAdmissionError),
}

impl NanoProductionAdmissionError {
    fn from_internal(
        source: PrepareWithPendingControllerError<
            LiveActuationError,
            DisarmReceipt,
            PendingLiveMpcAdmissionError,
        >,
    ) -> Self {
        match source {
            PrepareWithPendingControllerError::PrePromotion { primary, stop } => {
                Self::PrePromotion {
                    primary: Box::new(match *primary {
                        PreparePrimaryError::ControllerEvidence(source) => {
                            NanoProductionAdmissionPrimaryError::ControllerEvidence(source)
                        }
                        PreparePrimaryError::InitialZeroSessionMismatch {
                            acquisition,
                            receipt,
                        } => NanoProductionAdmissionPrimaryError::InitialZeroSessionMismatch {
                            acquisition,
                            receipt,
                        },
                        PreparePrimaryError::InitialReceiptWasNotConfirmedZero { receipt } => {
                            NanoProductionAdmissionPrimaryError::InitialReceiptWasNotConfirmedZero {
                                receipt,
                            }
                        }
                        PreparePrimaryError::ExactInventory(source) => {
                            NanoProductionAdmissionPrimaryError::ExactInventory(source)
                        }
                        PreparePrimaryError::Actuation(source) => {
                            NanoProductionAdmissionPrimaryError::Actuation(source)
                        }
                        PreparePrimaryError::Startup(source) => {
                            NanoProductionAdmissionPrimaryError::Startup(source)
                        }
                        PreparePrimaryError::Supervisor(source) => {
                            NanoProductionAdmissionPrimaryError::Supervisor(source)
                        }
                    }),
                    stop: match stop {
                        PendingControllerStop::Confirmed(receipt) => {
                            NanoProductionAdmissionStop::Confirmed(receipt)
                        }
                        PendingControllerStop::Uncertain(source) => {
                            NanoProductionAdmissionStop::Uncertain(source)
                        }
                    },
                }
            }
            PrepareWithPendingControllerError::Promotion(source) => Self::Promotion(source),
        }
    }
}

impl fmt::Debug for NanoProductionAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

impl fmt::Display for NanoProductionAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrePromotion { primary, stop } => {
                write!(
                    formatter,
                    "Nano production admission failed: {primary}; {stop}"
                )
            }
            Self::Promotion(source) => {
                write!(formatter, "Nano physical-driver promotion failed: {source}")
            }
        }
    }
}

impl std::error::Error for NanoProductionAdmissionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PrePromotion { primary, .. } => Some(primary.as_ref()),
            Self::Promotion(source) => Some(source),
        }
    }
}

trait PendingControllerPort {
    type Driver;
    type Error;
    type StopEvidence;
    type AdmissionError;

    fn verified_controller_acquisition(&self)
    -> Result<VerifiedControllerAcquisition, Self::Error>;
    fn disarm(self) -> Result<Self::StopEvidence, Self::Error>;
    fn admit(
        self,
        admitted: &AdmittedNavigationActuationConfigV1,
    ) -> Result<Self::Driver, Self::AdmissionError>;
}

impl PendingControllerPort for PendingLiveMpcControlDriver {
    type Driver = LiveMpcControlDriver;
    type Error = LiveActuationError;
    type StopEvidence = DisarmReceipt;
    type AdmissionError = PendingLiveMpcAdmissionError;

    fn verified_controller_acquisition(
        &self,
    ) -> Result<VerifiedControllerAcquisition, Self::Error> {
        PendingLiveMpcControlDriver::verified_controller_acquisition(self)
    }

    fn disarm(self) -> Result<Self::StopEvidence, Self::Error> {
        PendingLiveMpcControlDriver::disarm(self)
    }

    fn admit(
        self,
        admitted: &AdmittedNavigationActuationConfigV1,
    ) -> Result<Self::Driver, Self::AdmissionError> {
        PendingLiveMpcControlDriver::admit(self, admitted)
    }
}

struct PreparedNanoProductionRuntimeWithDriver<Driver> {
    startup: DisarmedNanoStartupParts,
    actuation: AdmittedNavigationActuationConfigV1,
    physical_driver: Driver,
    initial_zero: AppliedCommandReceipt,
}

enum PreparePrimaryError<ControllerError> {
    ControllerEvidence(ControllerError),
    InitialZeroSessionMismatch {
        acquisition: ControllerSession,
        receipt: Box<AppliedCommandReceipt>,
    },
    InitialReceiptWasNotConfirmedZero {
        receipt: Box<AppliedCommandReceipt>,
    },
    ExactInventory(InventoryMismatchReport),
    Actuation(ActuationAdmissionError),
    Startup(NanoStartupAdmissionError),
    Supervisor(NanoStartupSupervisorError),
}

enum PendingControllerStop<StopEvidence, ControllerError> {
    Confirmed(StopEvidence),
    Uncertain(ControllerError),
}

enum PrepareWithPendingControllerError<ControllerError, StopEvidence, AdmissionError> {
    PrePromotion {
        primary: Box<PreparePrimaryError<ControllerError>>,
        stop: PendingControllerStop<StopEvidence, ControllerError>,
    },
    Promotion(AdmissionError),
}

type PrepareWithPendingControllerResult<Pending> = Result<
    PreparedNanoProductionRuntimeWithDriver<<Pending as PendingControllerPort>::Driver>,
    PrepareWithPendingControllerError<
        <Pending as PendingControllerPort>::Error,
        <Pending as PendingControllerPort>::StopEvidence,
        <Pending as PendingControllerPort>::AdmissionError,
    >,
>;

#[allow(clippy::too_many_arguments)]
fn prepare_with_pending_controller<Pending>(
    policy: NanoAgentPolicyConfigV3,
    loaded_manifest: LoadedExpectedManifestV1,
    artifact_hashes: ManifestArtifactHashes,
    observed_inventory: ObservedDeviceInventoryV1,
    actuation_config: NavigationActuationConfigV1,
    plant_artifact_id: ArtifactId,
    plant_artifact_relative_path: ArtifactRelativePath,
    pending_controller: Pending,
    initial_zero: AppliedCommandReceipt,
    timeline: NanoProductionAdmissionTimeline,
) -> PrepareWithPendingControllerResult<Pending>
where
    Pending: PendingControllerPort,
{
    let controller_acquisition = match pending_controller.verified_controller_acquisition() {
        Ok(acquisition) => acquisition,
        Err(source) => {
            return Err(stop_after_failure(
                pending_controller,
                PreparePrimaryError::ControllerEvidence(source),
            ));
        }
    };

    if initial_zero.controller_session() != controller_acquisition.controller_session() {
        return Err(stop_after_failure(
            pending_controller,
            PreparePrimaryError::InitialZeroSessionMismatch {
                acquisition: controller_acquisition.controller_session(),
                receipt: Box::new(initial_zero),
            },
        ));
    }
    if !initial_zero.is_confirmed_zero() {
        return Err(stop_after_failure(
            pending_controller,
            PreparePrimaryError::InitialReceiptWasNotConfirmedZero {
                receipt: Box::new(initial_zero),
            },
        ));
    }

    // This is the sole expected/observed comparison in this transition.
    let exact_inventory =
        match admit_exact_inventory(loaded_manifest.manifest().clone(), observed_inventory) {
            Ok(admission) => admission,
            Err(source) => {
                return Err(stop_after_failure(
                    pending_controller,
                    PreparePrimaryError::ExactInventory(source),
                ));
            }
        };

    let actuation = match AdmittedNavigationActuationConfigV1::admit(
        actuation_config,
        &loaded_manifest,
        &exact_inventory,
        &artifact_hashes,
        &plant_artifact_id,
        &plant_artifact_relative_path,
    ) {
        Ok(admission) => admission,
        Err(source) => {
            return Err(stop_after_failure(
                pending_controller,
                PreparePrimaryError::Actuation(source),
            ));
        }
    };

    let startup = match super::AdmittedNanoStartup::admit(
        policy,
        loaded_manifest,
        artifact_hashes,
        exact_inventory,
        controller_acquisition,
        timeline.readiness_epoch(),
    ) {
        Ok(startup) => startup,
        Err(source) => {
            return Err(stop_after_failure(
                pending_controller,
                PreparePrimaryError::Startup(source),
            ));
        }
    };
    let startup = match startup.enter_disarmed(
        timeline.clock_epoch(),
        timeline.inventory_started_at(),
        timeline.readiness_admitted_at(),
    ) {
        Ok(startup) => startup.into_parts(),
        Err(source) => {
            return Err(stop_after_failure(
                pending_controller,
                PreparePrimaryError::Supervisor(source),
            ));
        }
    };

    let physical_driver = pending_controller
        .admit(&actuation)
        .map_err(PrepareWithPendingControllerError::Promotion)?;
    Ok(PreparedNanoProductionRuntimeWithDriver {
        startup,
        actuation,
        physical_driver,
        initial_zero,
    })
}

fn stop_after_failure<Pending>(
    pending_controller: Pending,
    primary: PreparePrimaryError<Pending::Error>,
) -> PrepareWithPendingControllerError<Pending::Error, Pending::StopEvidence, Pending::AdmissionError>
where
    Pending: PendingControllerPort,
{
    let stop = match pending_controller.disarm() {
        Ok(evidence) => PendingControllerStop::Confirmed(evidence),
        Err(source) => PendingControllerStop::Uncertain(source),
    };
    PrepareWithPendingControllerError::PrePromotion {
        primary: Box::new(primary),
        stop,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::fs;
    use std::num::NonZeroU64;
    use std::path::PathBuf;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    use kiko_device_inventory::{
        ArtifactDigestDto, ArtifactFileBindingInput, ArtifactFileBindingSet,
        ObservedDeviceInventoryV1Dto, ObservedOakV1Dto, ObservedStm32V1Dto,
        hash_manifest_artifacts, load_expected_manifest_v1_file,
        load_expected_manifest_v1_from_slice,
    };
    use kiko_supervisor_core::SupervisorState;
    use robot_command_client::fake::{FakeClock, FakeStep, FakeTransport};
    use robot_command_client::{
        ClientConfig, ClientConfigInput, DisarmedCommandClient, MonotonicClock,
        PendingPhysicalCommand, V2CommandLeaseMs,
    };
    use robot_protocol::ControllerUptimeMsWrapping;
    use robot_protocol::v2::{
        AcquireResult, AcquireResultCode, ActuatorConfigFingerprint, ControlEpoch,
        ControllerBootId, ControllerCapabilities, ControllerDeadlineMsWrapping, ControllerFaults,
        ControllerUid, HostCommandResult, HostCommandResultCode, HostStopResult, Message,
        MessageKind, OutputState, RemainingLeaseMs, RequestId, StatusCode, StatusReport,
        StopResultCode, TargetBootId, TimerPwm, V2CommandSequence,
    };
    use serde_json::json;
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::navigation::mpc::{
        FitResidualsV1Dto, PLANT_MODEL_V1, PlantEvidenceV1Dto, PlantModelV1, PlantModelV1Dto,
        PlantValidityEnvelopeV1Dto, WheelPlantV1Dto,
    };
    use crate::navigation::{ControlPeriodNs, SolverBudgetNs};

    const ROBOT_ID: &str = "kiko-production-test";
    const UID_BYTES: [u8; 12] = [0x11; 12];
    const FINGERPRINT_BYTES: [u8; 16] = [0x22; 16];
    const FIRMWARE_ABI: u16 = 2;
    const FIRMWARE_BUILD_ID: u32 = 42;
    const BOOT_ID: u64 = 7;
    const OTHER_BOOT_ID: u64 = 8;
    const CONTROL_EPOCH: u32 = 23;
    const ENDPOINT: &str = "127.0.0.1:8080";
    const INVENTORY_ENDPOINT: &str = "udp://127.0.0.1:8080";
    const NAVIGATION_BYTES: &[u8] = b"production navigation config";
    const CALIBRATION_BYTES: &[u8] = b"production calibration";
    const PLANT_BYTES: &[u8] = b"production physical plant dataset";
    const PLANT_EVIDENCE_DATASET_BYTES: &[u8] = b"distinct production physical evidence dataset";
    const RESPONSE_DELAY: Duration = Duration::from_millis(1);

    static NEXT_TEMP_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        sequence: u64,
        root: PathBuf,
        manifest_path: PathBuf,
        artifact_root: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let sequence = NEXT_TEMP_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let root = fs::canonicalize(std::env::temp_dir())
                .expect("canonical temp root")
                .join(format!(
                    "kiko-nano-production-admission-{}-{sequence}",
                    std::process::id()
                ));
            let artifact_root = root.join("artifacts");
            fs::create_dir_all(artifact_root.join("calibration")).expect("calibration directory");
            fs::create_dir_all(artifact_root.join("plant")).expect("plant directory");
            fs::write(
                artifact_root.join("calibration/main.bin"),
                CALIBRATION_BYTES,
            )
            .expect("calibration artifact");
            fs::write(artifact_root.join("plant/main.bin"), PLANT_BYTES).expect("plant artifact");
            let manifest_path = root.join("device-inventory-v1.json");
            fs::write(&manifest_path, manifest_json()).expect("manifest file");
            Self {
                sequence,
                root,
                manifest_path,
                artifact_root,
            }
        }

        fn loaded(&self) -> LoadedExpectedManifestV1 {
            load_expected_manifest_v1_file(&self.manifest_path).expect("no-follow manifest")
        }

        fn hashes(&self, loaded: &LoadedExpectedManifestV1) -> ManifestArtifactHashes {
            hash_manifest_artifacts(loaded.manifest(), &self.artifact_root, artifact_bindings())
                .expect("no-follow artifact hashes")
        }

        fn policy(&self) -> NanoAgentPolicyConfigV3 {
            let socket_path = PathBuf::from(format!(
                "/tmp/kiko-npa-{}-{}.sock",
                std::process::id(),
                self.sequence
            ));
            let encoded = serde_json::to_vec(&json!({
                "schema_version": 3,
                "control": {
                    "socket_path": socket_path,
                    "read_timeout_ms": 100,
                    "write_timeout_ms": 100,
                    "runtime_response_timeout_ms": 500,
                    "terminal_response_timeout_ms": 300000,
                    "runtime_queue_capacity": 8,
                    "operator_console": {
                        "bind_address": "127.0.0.1:9877",
                        "capability_path": socket_path.with_extension("capability"),
                        "deadman_tick_ms": 20,
                        "manual_command_forward_mm_per_s": 100,
                        "manual_command_yaw_millirad_per_s": 500
                    }
                },
                "inventory": {
                    "manifest_path": self.manifest_path,
                    "artifact_root_path": self.artifact_root,
                    "artifact_bindings": [
                        {
                            "kind": "calibration",
                            "artifact_id": "camera-main",
                            "relative_path": "calibration/main.bin"
                        },
                        {
                            "kind": "plant",
                            "artifact_id": "drive-main",
                            "relative_path": "plant/main.bin"
                        }
                    ]
                },
                "map_persistence": {
                    "save_snapshot_path": self.root.join("current.kmap"),
                    "warm_start": {"kind": "none"}
                },
                "eye": {"mode": "disabled"},
                "head": {"mode": "disabled"},
                "rgb_expression": {"mode": "disabled"},
                "supervisor": {
                    "maximum_authority_lease_ms": 1000,
                    "maximum_zero_age_ms": 250
                },
                "live_mode_policy": {
                    "startup": "disarmed_map_only",
                    "manual": {"permission": "disabled"},
                    "point_goal": {"permission": "disabled"},
                    "frontier_explore": {"permission": "disabled"}
                }
            }))
            .expect("policy JSON");
            NanoAgentPolicyConfigV3::parse_json(&encoded).expect("parsed policy")
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn sha256(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn canonical_sha256_id(bytes: &[u8]) -> String {
        let mut output = String::from("sha256:");
        for byte in sha256(bytes) {
            use fmt::Write;
            write!(output, "{byte:02x}").expect("write SHA-256");
        }
        output
    }

    fn manifest_json() -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "robot_id": ROBOT_ID,
            "oak": {
                "mxid": "A1B2C3D4E5F60708",
                "compiled_depthai_header_sdk_version": "3.6.1",
                "compiled_depthai_header_sdk_commit": "abc123",
                "compiled_depthai_header_embedded_device_artifact_version": "device-1",
                "compiled_depthai_header_embedded_bootloader_artifact_version": "bootloader-1"
            },
            "stm32": {
                "serial_by_id_path": "/dev/serial/by-id/usb-Kiko_STM32_A1-if00",
                "control_endpoint_identity": INVENTORY_ENDPOINT,
                "controller_uid": UID_BYTES,
                "firmware_abi": FIRMWARE_ABI,
                "firmware_build_id": FIRMWARE_BUILD_ID,
                "hardware_profile_fingerprint": FINGERPRINT_BYTES,
                "capabilities_bits": ControllerCapabilities::REQUIRED_BITS
            },
            "head": null,
            "eye": null,
            "calibration_artifacts": [{
                "artifact_id": "camera-main",
                "sha256": sha256(CALIBRATION_BYTES)
            }],
            "plant_artifacts": [{
                "artifact_id": "drive-main",
                "sha256": sha256(PLANT_BYTES)
            }]
        }))
        .expect("manifest JSON")
    }

    fn artifact_bindings() -> ArtifactFileBindingSet {
        ArtifactFileBindingSet::parse(vec![
            ArtifactFileBindingInput {
                kind: kiko_device_inventory::ArtifactKind::Calibration,
                artifact_id: "camera-main".into(),
                relative_path: "calibration/main.bin".into(),
            },
            ArtifactFileBindingInput {
                kind: kiko_device_inventory::ArtifactKind::Plant,
                artifact_id: "drive-main".into(),
                relative_path: "plant/main.bin".into(),
            },
        ])
        .expect("artifact bindings")
    }

    fn observed_inventory(
        robot_id: &str,
        controller_uid: [u8; 12],
        boot_id: u64,
    ) -> ObservedDeviceInventoryV1 {
        ObservedDeviceInventoryV1::parse(ObservedDeviceInventoryV1Dto {
            schema_version: kiko_device_inventory::OBSERVED_DEVICE_INVENTORY_V1,
            robot_id: robot_id.into(),
            oak: Some(ObservedOakV1Dto {
                mxid: "A1B2C3D4E5F60708".into(),
                compiled_depthai_header_sdk_version: "3.6.1".into(),
                compiled_depthai_header_sdk_commit: "abc123".into(),
                compiled_depthai_header_embedded_device_artifact_version: "device-1".into(),
                compiled_depthai_header_embedded_bootloader_artifact_version: "bootloader-1".into(),
            }),
            stm32: Some(ObservedStm32V1Dto {
                serial_by_id_path: "/dev/serial/by-id/usb-Kiko_STM32_A1-if00".into(),
                control_endpoint_identity: INVENTORY_ENDPOINT.into(),
                controller_uid,
                controller_boot_id: boot_id,
                firmware_abi: FIRMWARE_ABI,
                firmware_build_id: FIRMWARE_BUILD_ID,
                hardware_profile_fingerprint: FINGERPRINT_BYTES,
                capabilities_bits: ControllerCapabilities::REQUIRED_BITS,
            }),
            head: None,
            eye: None,
            calibration_artifacts: vec![ArtifactDigestDto {
                artifact_id: "camera-main".into(),
                sha256: sha256(CALIBRATION_BYTES),
            }],
            plant_artifacts: vec![ArtifactDigestDto {
                artifact_id: "drive-main".into(),
                sha256: sha256(PLANT_BYTES),
            }],
        })
        .expect("observed inventory")
    }

    fn plant_model() -> PlantModelV1 {
        PlantModelV1::parse(PlantModelV1Dto {
            schema_version: PLANT_MODEL_V1,
            model_id: "kiko-physical-v1".into(),
            model_version: 1,
            sample_period_s: 0.1,
            wheelbase_m: 0.3,
            left: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.2,
            },
            right: WheelPlantV1Dto {
                velocity_gain_mps_per_pwm_percent: 0.01,
                time_constant_s: 0.2,
            },
            validity: PlantValidityEnvelopeV1Dto {
                left_pwm_min_percent: -50,
                left_pwm_max_percent: 50,
                right_pwm_min_percent: -50,
                right_pwm_max_percent: 50,
                left_velocity_min_mps: -0.5,
                left_velocity_max_mps: 0.5,
                right_velocity_min_mps: -0.5,
                right_velocity_max_mps: 0.5,
                max_abs_yaw_rate_rad_s: 3.0,
                max_abs_lateral_velocity_mps: 0.1,
            },
            evidence: PlantEvidenceV1Dto::ClaimedPhysicalIdentification {
                dataset_content_id: canonical_sha256_id(PLANT_EVIDENCE_DATASET_BYTES),
                identification_method_id: "method-v1".into(),
                sample_count: 100,
                residuals: FitResidualsV1Dto {
                    left_velocity_rmse_mps: 0.01,
                    right_velocity_rmse_mps: 0.02,
                    yaw_rate_rmse_rad_s: 0.03,
                    max_abs_velocity_error_mps: 0.04,
                },
            },
        })
        .expect("physical model")
    }

    fn actuation_config(endpoint: &str, uid_hex: &str) -> NavigationActuationConfigV1 {
        let navigation_sha256: String = sha256(NAVIGATION_BYTES)
            .into_iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let encoded = serde_json::to_vec(&json!({
            "schema_version": 2,
            "robot_id": ROBOT_ID,
            "command_endpoint": endpoint,
            "navigation_config_sha256_hex": navigation_sha256,
            "controller_uid_hex": uid_hex,
            "firmware_abi": FIRMWARE_ABI,
            "firmware_build_id": FIRMWARE_BUILD_ID,
            "actuator_config_fingerprint_hex": "22222222222222222222222222222222",
            "plant_model_id": "kiko-physical-v1",
            "plant_model_version": 1,
            "plant_artifact_sha256_hex": sha256(PLANT_BYTES)
                .into_iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>(),
            "operator_claimed_physical_approval": {
                "approval_id": "approval-v1",
                "approver_id": "operator@example.com",
                "plant_dataset_content_id": canonical_sha256_id(PLANT_EVIDENCE_DATASET_BYTES),
                "plant_identification_method_id": "method-v1",
                "plant_sample_count": 100,
                "plant_fit_residuals": {
                    "left_velocity_rmse_mps": 0.01,
                    "right_velocity_rmse_mps": 0.02,
                    "yaw_rate_rmse_rad_s": 0.03,
                    "max_abs_velocity_error_mps": 0.04
                },
                "imu_calibration_id": "imu-cal-v1",
                "stereo_calibration_id": "stereo-cal-v1",
                "tracking_camera_to_base_calibration_id": "extrinsic-v1"
            },
            "apply_ack_budget_ns": 20_000_000,
            "stop_ack_budget_ns": 30_000_000,
            "scheduling_guard_ns": 5_000_000,
            "controller_motion_lease_ms": 200,
            "controller_deadline_tolerance_ns": 2_000_000,
            "maximum_uncommanded_motion_ns": 222_000_000
        }))
        .expect("actuation JSON");
        NavigationActuationConfigV1::parse_and_authorize(
            &encoded,
            ROBOT_ID,
            NAVIGATION_BYTES,
            plant_model(),
            SolverBudgetNs::try_new(50_000_000).expect("solver budget"),
            ControlPeriodNs::from_nonzero(NonZeroU64::new(100_000_000).expect("control period")),
        )
        .expect("parsed actuation config")
    }

    fn uid() -> ControllerUid {
        ControllerUid::try_new(UID_BYTES).expect("UID")
    }

    fn boot(raw: u64) -> ControllerBootId {
        ControllerBootId::try_new(raw).expect("boot ID")
    }

    fn control_epoch() -> ControlEpoch {
        ControlEpoch::try_new(CONTROL_EPOCH).expect("control epoch")
    }

    fn fingerprint() -> ActuatorConfigFingerprint {
        ActuatorConfigFingerprint::try_new(FINGERPRINT_BYTES).expect("fingerprint")
    }

    fn capabilities() -> ControllerCapabilities {
        ControllerCapabilities::try_from_bits(ControllerCapabilities::REQUIRED_BITS)
            .expect("capabilities")
    }

    fn timer_pwm_result(
        boot_id: ControllerBootId,
        sequence: V2CommandSequence,
        pwm: TimerPwm,
    ) -> Message {
        Message::HostCommandResult(HostCommandResult {
            controller_uid: uid(),
            boot_id,
            control_epoch: control_epoch(),
            sequence,
            result: HostCommandResultCode::AppliedNew,
            requested_timer_pwm: pwm,
            controller_timer_pwm: pwm,
            output_state: if pwm.is_zero() {
                OutputState::ZeroPwm
            } else {
                OutputState::NonzeroPwm
            },
            controller_applied_at: ControllerUptimeMsWrapping::new(2_000 + sequence.get()),
            controller_expires_at: ControllerDeadlineMsWrapping::new(2_100 + sequence.get()),
            remaining_lease: RemainingLeaseMs::try_new(90).expect("remaining lease"),
            faults: ControllerFaults::NONE,
        })
    }

    struct ControllerEvidence {
        acquisition: VerifiedControllerAcquisition,
        zero: AppliedCommandReceipt,
        motion: Option<AppliedCommandReceipt>,
    }

    fn controller_evidence(boot_raw: u64, include_motion: bool) -> ControllerEvidence {
        let boot_id = boot(boot_raw);
        let motion = TimerPwm::try_new(10, 10).expect("motion PWM");
        let mut steps = vec![
            FakeStep::respond(
                MessageKind::StatusQuery,
                RESPONSE_DELAY,
                Message::StatusReport(StatusReport {
                    controller_uid: uid(),
                    observed_boot_id: TargetBootId::Exact(boot_id),
                    request_id: RequestId::new(0),
                    status: StatusCode::ReadyStopped,
                    control_epoch: None,
                    controller_uptime: ControllerUptimeMsWrapping::new(1_000),
                    capabilities: capabilities(),
                    output_state: OutputState::Disabled,
                    controller_timer_pwm: TimerPwm::ZERO,
                    remaining_lease: RemainingLeaseMs::ZERO,
                    faults: ControllerFaults::NONE,
                }),
            ),
            FakeStep::respond(
                MessageKind::AcquireControl,
                RESPONSE_DELAY,
                Message::AcquireResult(AcquireResult {
                    controller_uid: uid(),
                    boot_id,
                    request_id: RequestId::new(1),
                    control_epoch: Some(control_epoch()),
                    result: AcquireResultCode::Granted,
                    capabilities: capabilities(),
                    faults: ControllerFaults::NONE,
                    observed_firmware_abi: FIRMWARE_ABI,
                    observed_firmware_build_id: FIRMWARE_BUILD_ID,
                    observed_actuator_config_fingerprint: fingerprint(),
                }),
            ),
            FakeStep::respond(
                MessageKind::HostCommand,
                RESPONSE_DELAY,
                timer_pwm_result(boot_id, V2CommandSequence::FIRST, TimerPwm::ZERO),
            ),
        ];
        if include_motion {
            steps.push(FakeStep::respond(
                MessageKind::HostCommand,
                RESPONSE_DELAY,
                timer_pwm_result(boot_id, V2CommandSequence::new(1), motion),
            ));
        }
        steps.push(FakeStep::respond(
            MessageKind::HostStop,
            RESPONSE_DELAY,
            Message::HostStopResult(HostStopResult {
                controller_uid: uid(),
                observed_boot_id: TargetBootId::Exact(boot_id),
                request_id: RequestId::new(2),
                result: StopResultCode::ControllerConfirmed,
                output_state: OutputState::Disabled,
                controller_uptime: ControllerUptimeMsWrapping::new(3_000),
                faults: ControllerFaults::NONE,
            }),
        ));
        let clock = FakeClock::default();
        let (transport, _probe) = FakeTransport::scripted(clock.clone(), steps);
        let client = DisarmedCommandClient::new(
            transport,
            clock.clone(),
            ClientConfig::parse(ClientConfigInput {
                command_endpoint: ENDPOINT,
                controller_uid_hex: "111111111111111111111111",
                expected_firmware_abi: "2",
                expected_firmware_build_id: "42",
                expected_actuator_config_fingerprint_hex: "22222222222222222222222222222222",
                status_timeout_ns: "50000000",
                acquire_timeout_ns: "50000000",
                applied_ack_timeout_ns: "50000000",
                stop_attempt_timeout_ns: "50000000",
                max_stop_recovery_attempts: "3",
                zero_acquisition_lease_ms: "100",
            })
            .expect("client config"),
        );
        let (armed, zero) = client.acquire_zero().ok().expect("controller acquisition");
        let acquisition = armed.verified_acquisition();
        let (armed, motion_receipt) = if include_motion {
            let acknowledgement_deadline_exclusive = clock
                .now()
                .checked_add(Duration::from_millis(50))
                .expect("deadline");
            let pending = PendingPhysicalCommand::new(
                motion,
                V2CommandLeaseMs::try_new(100).expect("lease"),
                acknowledgement_deadline_exclusive,
            );
            let (armed, receipt) = armed.apply(pending).ok().expect("motion receipt");
            (armed, Some(receipt))
        } else {
            (armed, None)
        };
        let (_disarmed, _stop) = armed.disarm().ok().expect("explicit stop");
        ControllerEvidence {
            acquisition,
            zero,
            motion: motion_receipt,
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FakeControllerError {
        EvidenceUnavailable,
        StopFailed,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum FakeAdmissionError {
        ConfigMismatchStopConfirmed,
        ConfigMismatchStopUncertain,
    }

    struct FakePendingController {
        events: Rc<RefCell<Vec<&'static str>>>,
        acquisition: Result<VerifiedControllerAcquisition, FakeControllerError>,
        stop: Result<u64, FakeControllerError>,
        admission: Result<u64, FakeAdmissionError>,
    }

    impl PendingControllerPort for FakePendingController {
        type Driver = u64;
        type Error = FakeControllerError;
        type StopEvidence = u64;
        type AdmissionError = FakeAdmissionError;

        fn verified_controller_acquisition(
            &self,
        ) -> Result<VerifiedControllerAcquisition, Self::Error> {
            self.events.borrow_mut().push("controller-evidence");
            self.acquisition
        }

        fn disarm(self) -> Result<Self::StopEvidence, Self::Error> {
            self.events.borrow_mut().push("disarm");
            self.stop
        }

        fn admit(
            self,
            _admitted: &AdmittedNavigationActuationConfigV1,
        ) -> Result<Self::Driver, Self::AdmissionError> {
            self.events.borrow_mut().push("admit");
            self.admission
        }
    }

    fn fake_pending(
        events: &Rc<RefCell<Vec<&'static str>>>,
        acquisition: Result<VerifiedControllerAcquisition, FakeControllerError>,
    ) -> FakePendingController {
        FakePendingController {
            events: Rc::clone(events),
            acquisition,
            stop: Ok(91),
            admission: Ok(73),
        }
    }

    type TestPrepareError =
        PrepareWithPendingControllerError<FakeControllerError, u64, FakeAdmissionError>;

    fn prepare(
        fixture: &Fixture,
        pending: FakePendingController,
        initial_zero: AppliedCommandReceipt,
        observed: ObservedDeviceInventoryV1,
        actuation: NavigationActuationConfigV1,
        inventory_started_at: HostMonotonicTimestamp,
        readiness_admitted_at: HostMonotonicTimestamp,
    ) -> Result<PreparedNanoProductionRuntimeWithDriver<u64>, TestPrepareError> {
        prepare_with_plant_path(
            fixture,
            pending,
            initial_zero,
            observed,
            actuation,
            ArtifactRelativePath::parse("plant/main.bin".into()).expect("plant path"),
            inventory_started_at,
            readiness_admitted_at,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_with_plant_path(
        fixture: &Fixture,
        pending: FakePendingController,
        initial_zero: AppliedCommandReceipt,
        observed: ObservedDeviceInventoryV1,
        actuation: NavigationActuationConfigV1,
        plant_artifact_relative_path: ArtifactRelativePath,
        inventory_started_at: HostMonotonicTimestamp,
        readiness_admitted_at: HostMonotonicTimestamp,
    ) -> Result<PreparedNanoProductionRuntimeWithDriver<u64>, TestPrepareError> {
        let loaded = fixture.loaded();
        let hashes = fixture.hashes(&loaded);
        let plant_artifact_id = *loaded
            .manifest()
            .artifacts()
            .iter()
            .find(|artifact| artifact.kind() == kiko_device_inventory::ArtifactKind::Plant)
            .expect("plant artifact")
            .artifact_id();
        prepare_with_pending_controller(
            fixture.policy(),
            loaded,
            hashes,
            observed,
            actuation,
            plant_artifact_id,
            plant_artifact_relative_path,
            pending,
            initial_zero,
            NanoProductionAdmissionTimeline::try_new(
                ReadinessEpoch::try_new(1).expect("readiness epoch"),
                NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(10)),
                inventory_started_at,
                readiness_admitted_at,
            )
            .expect("ordered test timeline"),
        )
    }

    fn assert_confirmed_pre_promotion_failure(
        result: Result<PreparedNanoProductionRuntimeWithDriver<u64>, TestPrepareError>,
        predicate: impl FnOnce(&PreparePrimaryError<FakeControllerError>) -> bool,
    ) {
        match result {
            Err(PrepareWithPendingControllerError::PrePromotion {
                primary,
                stop: PendingControllerStop::Confirmed(91),
            }) if predicate(&primary) => {}
            Err(_) => panic!("unexpected production admission failure"),
            Ok(_) => panic!("invalid evidence must not produce a runtime"),
        }
    }

    #[test]
    fn exact_evidence_orders_controller_check_before_promotion_and_returns_disarmed_runtime() {
        let fixture = Fixture::new();
        let evidence = controller_evidence(BOOT_ID, false);
        let events = Rc::new(RefCell::new(Vec::new()));
        let result = prepare(
            &fixture,
            fake_pending(&events, Ok(evidence.acquisition)),
            evidence.zero,
            observed_inventory(ROBOT_ID, UID_BYTES, BOOT_ID),
            actuation_config(ENDPOINT, "111111111111111111111111"),
            HostMonotonicTimestamp::from_nanos(11),
            HostMonotonicTimestamp::from_nanos(12),
        );
        let prepared = match result {
            Ok(prepared) => prepared,
            Err(_) => panic!("exact production evidence must admit"),
        };

        assert_eq!(*events.borrow(), ["controller-evidence", "admit"]);
        assert_eq!(prepared.physical_driver, 73);
        assert!(prepared.initial_zero.is_confirmed_zero());
        assert_eq!(
            prepared.startup.authority.state(),
            SupervisorState::Disarmed {
                readiness: prepared.startup.readiness
            }
        );
        assert_eq!(
            prepared.actuation.config().controller_uid(),
            prepared.startup.controller_acquisition.controller_uid()
        );
    }

    #[test]
    fn unavailable_controller_evidence_and_failed_stop_are_both_preserved() {
        let fixture = Fixture::new();
        let evidence = controller_evidence(BOOT_ID, false);
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut pending = fake_pending(&events, Err(FakeControllerError::EvidenceUnavailable));
        pending.stop = Err(FakeControllerError::StopFailed);
        let result = prepare(
            &fixture,
            pending,
            evidence.zero,
            observed_inventory(ROBOT_ID, UID_BYTES, BOOT_ID),
            actuation_config(ENDPOINT, "111111111111111111111111"),
            HostMonotonicTimestamp::from_nanos(11),
            HostMonotonicTimestamp::from_nanos(12),
        );
        match result {
            Err(PrepareWithPendingControllerError::PrePromotion { primary, stop }) => {
                assert!(matches!(
                    *primary,
                    PreparePrimaryError::ControllerEvidence(
                        FakeControllerError::EvidenceUnavailable
                    )
                ));
                assert!(matches!(
                    stop,
                    PendingControllerStop::Uncertain(FakeControllerError::StopFailed)
                ));
            }
            Err(_) => panic!("unexpected production admission failure"),
            Ok(_) => panic!("controller-evidence failure must not produce a runtime"),
        }
        assert_eq!(*events.borrow(), ["controller-evidence", "disarm"]);
    }

    #[test]
    fn initial_zero_must_name_the_pending_session_and_be_exact_zero() {
        let fixture = Fixture::new();
        let pending_evidence = controller_evidence(BOOT_ID, false);
        let other_session = controller_evidence(OTHER_BOOT_ID, false);
        let events = Rc::new(RefCell::new(Vec::new()));
        let result = prepare(
            &fixture,
            fake_pending(&events, Ok(pending_evidence.acquisition)),
            other_session.zero,
            observed_inventory(ROBOT_ID, UID_BYTES, BOOT_ID),
            actuation_config(ENDPOINT, "111111111111111111111111"),
            HostMonotonicTimestamp::from_nanos(11),
            HostMonotonicTimestamp::from_nanos(12),
        );
        assert_confirmed_pre_promotion_failure(result, |primary| {
            matches!(
                primary,
                PreparePrimaryError::InitialZeroSessionMismatch { .. }
            )
        });
        assert_eq!(*events.borrow(), ["controller-evidence", "disarm"]);

        let evidence = controller_evidence(BOOT_ID, true);
        let motion = evidence.motion.expect("motion receipt");
        let events = Rc::new(RefCell::new(Vec::new()));
        let result = prepare(
            &fixture,
            fake_pending(&events, Ok(evidence.acquisition)),
            motion,
            observed_inventory(ROBOT_ID, UID_BYTES, BOOT_ID),
            actuation_config(ENDPOINT, "111111111111111111111111"),
            HostMonotonicTimestamp::from_nanos(11),
            HostMonotonicTimestamp::from_nanos(12),
        );
        assert_confirmed_pre_promotion_failure(result, |primary| {
            matches!(
                primary,
                PreparePrimaryError::InitialReceiptWasNotConfirmedZero { receipt }
                    if !receipt.applied_timer_pwm().is_zero()
            )
        });
        assert_eq!(*events.borrow(), ["controller-evidence", "disarm"]);
    }

    #[test]
    fn exact_inventory_identity_mismatch_stops_without_promotion() {
        let fixture = Fixture::new();
        let evidence = controller_evidence(BOOT_ID, false);
        let events = Rc::new(RefCell::new(Vec::new()));
        let result = prepare(
            &fixture,
            fake_pending(&events, Ok(evidence.acquisition)),
            evidence.zero,
            observed_inventory("different-kiko", UID_BYTES, BOOT_ID),
            actuation_config(ENDPOINT, "111111111111111111111111"),
            HostMonotonicTimestamp::from_nanos(11),
            HostMonotonicTimestamp::from_nanos(12),
        );
        assert_confirmed_pre_promotion_failure(
            result,
            |primary| matches!(primary, PreparePrimaryError::ExactInventory(report) if report.len() == 1),
        );
        assert_eq!(*events.borrow(), ["controller-evidence", "disarm"]);
    }

    #[test]
    fn admitted_inventory_cannot_authorize_a_different_actuation_identity_or_endpoint() {
        let fixture = Fixture::new();
        for config in [
            actuation_config("127.0.0.1:8081", "111111111111111111111111"),
            actuation_config(ENDPOINT, "333333333333333333333333"),
        ] {
            let evidence = controller_evidence(BOOT_ID, false);
            let events = Rc::new(RefCell::new(Vec::new()));
            let result = prepare(
                &fixture,
                fake_pending(&events, Ok(evidence.acquisition)),
                evidence.zero,
                observed_inventory(ROBOT_ID, UID_BYTES, BOOT_ID),
                config,
                HostMonotonicTimestamp::from_nanos(11),
                HostMonotonicTimestamp::from_nanos(12),
            );
            assert_confirmed_pre_promotion_failure(result, |primary| {
                matches!(
                    primary,
                    PreparePrimaryError::Actuation(
                        ActuationAdmissionError::ControllerEndpointMismatch { .. }
                            | ActuationAdmissionError::ControllerUidMismatch
                    )
                )
            });
            assert_eq!(*events.borrow(), ["controller-evidence", "disarm"]);
        }
    }

    #[test]
    fn selected_plant_path_is_exact_and_mismatch_explicitly_stops() {
        let fixture = Fixture::new();
        let evidence = controller_evidence(BOOT_ID, false);
        let events = Rc::new(RefCell::new(Vec::new()));
        let result = prepare_with_plant_path(
            &fixture,
            fake_pending(&events, Ok(evidence.acquisition)),
            evidence.zero,
            observed_inventory(ROBOT_ID, UID_BYTES, BOOT_ID),
            actuation_config(ENDPOINT, "111111111111111111111111"),
            ArtifactRelativePath::parse("plant/other.bin".into()).expect("different plant path"),
            HostMonotonicTimestamp::from_nanos(11),
            HostMonotonicTimestamp::from_nanos(12),
        );
        assert_confirmed_pre_promotion_failure(result, |primary| {
            matches!(
                primary,
                PreparePrimaryError::Actuation(
                    ActuationAdmissionError::SelectedPlantPathMismatch { .. }
                )
            )
        });
        assert_eq!(*events.borrow(), ["controller-evidence", "disarm"]);
    }

    #[test]
    fn in_memory_manifest_cannot_replace_no_follow_file_evidence() {
        let fixture = Fixture::new();
        let evidence = controller_evidence(BOOT_ID, false);
        let events = Rc::new(RefCell::new(Vec::new()));
        let loaded = load_expected_manifest_v1_from_slice(&manifest_json())
            .expect("structurally valid in-memory manifest");
        let hashes = fixture.hashes(&loaded);
        let plant_artifact_id = *loaded
            .manifest()
            .artifacts()
            .iter()
            .find(|artifact| artifact.kind() == kiko_device_inventory::ArtifactKind::Plant)
            .expect("plant artifact")
            .artifact_id();
        let result = prepare_with_pending_controller(
            fixture.policy(),
            loaded,
            hashes,
            observed_inventory(ROBOT_ID, UID_BYTES, BOOT_ID),
            actuation_config(ENDPOINT, "111111111111111111111111"),
            plant_artifact_id,
            ArtifactRelativePath::parse("plant/main.bin".into()).expect("plant path"),
            fake_pending(&events, Ok(evidence.acquisition)),
            evidence.zero,
            NanoProductionAdmissionTimeline::try_new(
                ReadinessEpoch::try_new(1).expect("readiness epoch"),
                NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(10)),
                HostMonotonicTimestamp::from_nanos(11),
                HostMonotonicTimestamp::from_nanos(12),
            )
            .expect("ordered test timeline"),
        );
        assert_confirmed_pre_promotion_failure(result, |primary| {
            matches!(
                primary,
                PreparePrimaryError::Actuation(
                    ActuationAdmissionError::ManifestWasNotLoadedFromFile
                )
            )
        });
        assert_eq!(*events.borrow(), ["controller-evidence", "disarm"]);
    }

    #[test]
    fn acquisition_boot_identity_mismatch_fails_closed() {
        let fixture = Fixture::new();
        let evidence = controller_evidence(OTHER_BOOT_ID, false);
        let events = Rc::new(RefCell::new(Vec::new()));
        let result = prepare(
            &fixture,
            fake_pending(&events, Ok(evidence.acquisition)),
            evidence.zero,
            observed_inventory(ROBOT_ID, UID_BYTES, BOOT_ID),
            actuation_config(ENDPOINT, "111111111111111111111111"),
            HostMonotonicTimestamp::from_nanos(11),
            HostMonotonicTimestamp::from_nanos(12),
        );
        assert_confirmed_pre_promotion_failure(result, |primary| {
            matches!(
                primary,
                PreparePrimaryError::Startup(
                    NanoStartupAdmissionError::ControllerBootIdMismatch { .. }
                )
            )
        });
        assert_eq!(*events.borrow(), ["controller-evidence", "disarm"]);
    }

    #[test]
    fn admission_timeline_rejects_pre_epoch_and_regressed_observations() {
        let readiness_epoch = ReadinessEpoch::try_new(1).expect("readiness epoch");
        let clock_epoch = NavigationClockEpoch::new(HostMonotonicTimestamp::from_nanos(10));
        assert_eq!(
            NanoProductionAdmissionTimeline::try_new(
                readiness_epoch,
                clock_epoch,
                HostMonotonicTimestamp::from_nanos(9),
                HostMonotonicTimestamp::from_nanos(12),
            ),
            Err(
                NanoProductionAdmissionTimelineError::InventoryBeforeClockOrigin {
                    clock_origin_ns: 10,
                    inventory_started_at_ns: 9,
                }
            )
        );
        assert_eq!(
            NanoProductionAdmissionTimeline::try_new(
                readiness_epoch,
                clock_epoch,
                HostMonotonicTimestamp::from_nanos(12),
                HostMonotonicTimestamp::from_nanos(11),
            ),
            Err(
                NanoProductionAdmissionTimelineError::ReadinessBeforeInventory {
                    inventory_started_at_ns: 12,
                    readiness_admitted_at_ns: 11,
                }
            )
        );
        assert!(
            NanoProductionAdmissionTimeline::try_new(
                readiness_epoch,
                clock_epoch,
                HostMonotonicTimestamp::from_nanos(10),
                HostMonotonicTimestamp::from_nanos(10),
            )
            .is_ok(),
            "equal ordered observations are valid"
        );
    }

    #[test]
    fn promotion_is_last_and_preserves_its_own_confirmed_or_uncertain_stop() {
        let fixture = Fixture::new();
        for admission in [
            Err(FakeAdmissionError::ConfigMismatchStopConfirmed),
            Err(FakeAdmissionError::ConfigMismatchStopUncertain),
        ] {
            let evidence = controller_evidence(BOOT_ID, false);
            let events = Rc::new(RefCell::new(Vec::new()));
            let mut pending = fake_pending(&events, Ok(evidence.acquisition));
            pending.admission = admission;
            let result = prepare(
                &fixture,
                pending,
                evidence.zero,
                observed_inventory(ROBOT_ID, UID_BYTES, BOOT_ID),
                actuation_config(ENDPOINT, "111111111111111111111111"),
                HostMonotonicTimestamp::from_nanos(11),
                HostMonotonicTimestamp::from_nanos(12),
            );
            assert!(matches!(
                result,
                Err(PrepareWithPendingControllerError::Promotion(
                    FakeAdmissionError::ConfigMismatchStopConfirmed
                        | FakeAdmissionError::ConfigMismatchStopUncertain
                ))
            ));
            assert_eq!(
                *events.borrow(),
                ["controller-evidence", "admit"],
                "the consuming admit operation owns mismatch stopping"
            );
        }
    }
}
