//! One-shot attended wheel-on base commissioning.
//!
//! Standard output is newline-delimited JSON evidence for external tooling;
//! the production web console does not currently launch or consume this
//! process. Human diagnostics go to standard error and attended confirmations
//! use the controlling terminal directly. Successful completion publishes
//! only a proposed plant and evidence; it never grants manual, MPC, or mapping
//! authority.

use std::fmt;
use std::future::Future;
use std::io::{self, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::Parser;
use kiko_base_commissioning::CommissioningState;
use kiko_device_inventory::ArtifactRelativePath;
use kiko_expression_core::StreamEpochId;
use kiko_slam::navigation::nano_base_commissioning::{
    CommissioningExternalSignal, NanoBaseCommissioningProposal,
};
use kiko_slam::navigation::nano_base_commissioning_bootstrap::{
    CommissioningClockEpoch, CommissioningObservationKind, CommissioningProgressReporter,
    CommissioningSamplingRequest, prepare_nano_base_commissioning,
};
use kiko_slam::navigation::nano_base_commissioning_live::{
    CommissioningLiveCloseError, prepare_commissioning_live_observation,
};
use serde_json::{Value, json};

const OUTPUT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Parser)]
#[command(
    name = "kiko-nano-base-commission",
    about = "Attended, one-shot, wheel-on base calibration (proposal-only)"
)]
struct Args {
    /// Canonical absolute deployment bundle root.
    #[arg(long)]
    deployment_root: PathBuf,

    /// Commissioning launch document relative to deployment-root.
    #[arg(long)]
    launch: String,

    /// Existing private (0700), absolute commissioning state root.
    #[arg(long)]
    state_root: PathBuf,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    let args = Args::parse();
    let reporter = NdjsonReporter::stdout();
    let result = run(args, reporter.clone()).await;
    match result {
        Ok(()) => {}
        Err(error) => {
            let _ = reporter.emit(json!({
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "event": "failed",
                "stage": error.stage(),
                "message": error.to_string(),
                "motion_authority_granted": false,
                "proposal_activated": false,
            }));
            eprintln!(
                "kiko-nano-base-commission failed at {}: {error}",
                error.stage()
            );
            std::process::exit(1);
        }
    }
}

async fn run(args: Args, reporter: NdjsonReporter) -> Result<(), AppError> {
    let running = Arc::new(AtomicBool::new(true));
    let signal_running = Arc::clone(&running);
    ctrlc::set_handler(move || {
        signal_running.store(false, Ordering::Release);
    })
    .map_err(AppError::Signal)?;
    let clock_origin = Instant::now();

    reporter.emit(json!({
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "event": "starting",
        "lane": "attended_wheel_on_commissioning_v3",
        "maximum_pwm_percent": 20,
        "motion_authority_granted": false,
        "proposal_activated": false,
    }))?;

    let launch = ArtifactRelativePath::parse(args.launch).map_err(AppError::LaunchRelativePath)?;
    let prepared = prepare_nano_base_commissioning(&args.deployment_root, launch, &args.state_root)
        .map_err(|source| AppError::StaticAdmission(Box::new(source)))?;
    reporter.emit(json!({
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "event": "static_admission_ready",
        "session_id": prepared.launch().session_id(),
        "controller_class": "attended_wheel_on_commissioning",
        "physical_stop_semantics": "unverified",
        "motion_authority_granted": false,
    }))?;

    let mut epoch_bytes = [0_u8; 16];
    getrandom::fill(&mut epoch_bytes).map_err(AppError::Entropy)?;
    let clock_epoch =
        CommissioningClockEpoch::try_new(epoch_bytes).map_err(AppError::ClockEpoch)?;
    let mut stream_epoch_value =
        u64::from_le_bytes(epoch_bytes[..8].try_into().expect("fixed eight-byte slice"));
    if stream_epoch_value == 0 {
        stream_epoch_value =
            u64::from_le_bytes(epoch_bytes[8..].try_into().expect("fixed eight-byte slice"));
    }
    if stream_epoch_value == 0 {
        // The 128-bit clock epoch parser already proved at least one byte is
        // nonzero; this deterministic fallback remains a distinct nonzero
        // expression-stream identity without another weak boundary.
        stream_epoch_value = 1;
    }
    let accessory_stream_epoch =
        StreamEpochId::try_new(stream_epoch_value).map_err(AppError::AccessoryStreamEpoch)?;
    let live = prepare_commissioning_live_observation(
        &prepared,
        clock_origin,
        Arc::clone(&running),
        clock_epoch,
        accessory_stream_epoch,
    )
    .map_err(|source| AppError::LiveGraph(Box::new(source)))?;
    let accessory_stats = live.source.accessory_frame_stats();
    let live_stream = live.stream;
    let live_source = live.source;
    let live_source = match emit_or_cleanup(
        || {
            reporter.emit(json!({
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "event": "oak_live_ready",
            "session_id": prepared.launch().session_id(),
            "same_process_owner": true,
            "visual_source": prepared.expected_visual_velocity_source_id().as_str(),
            "motion_authority_granted": false,
            }))
        },
        live_source,
        |source| async move { source.close().err() },
    )
    .await
    {
        Ok(source) => source,
        Err(failure) => {
            return Err(AppError::OutputAfterLiveOwnership {
                source: Box::new(failure.primary),
                live_close: failure.cleanup,
            });
        }
    };
    let live_source = match emit_or_cleanup(
        || {
            reporter.emit(json!({
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "event": "accessory_live_ready",
            "natural_head_hold_verified": true,
            "eye_owner_ready": true,
            "rgb_expression_owner": "same_process_oak_borrowed_frames",
            "successful_rgb_expression_frames": accessory_stats.processed_successfully,
            "stream_epoch": accessory_stream_epoch.get(),
            "motion_authority_granted": false,
            }))
        },
        live_source,
        |source| async move { source.close().err() },
    )
    .await
    {
        Ok(source) => source,
        Err(failure) => {
            return Err(AppError::OutputAfterLiveOwnership {
                source: Box::new(failure.primary),
                live_close: failure.cleanup,
            });
        }
    };

    let admitted =
        match prepared.consume_fresh_attended_attestation(live_stream, clock_origin, &running) {
            Ok(value) => value,
            Err(source) => {
                let live_close = live_source.close().err();
                return Err(AppError::AttendedAdmission {
                    source: Box::new(source),
                    live_close,
                });
            }
        };
    let issued_at_ns = admitted.confirmation_issued_at_ns();
    let live_source = match emit_or_cleanup(
        || {
            reporter.emit(json!({
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "event": "attestation_consumed",
            "issued_at_ns": issued_at_ns,
            "wheels_attached": true,
            "clear_area_confirmed": true,
            "operator_attending": true,
            "independent_power_cut_tested_and_reachable": true,
            "motion_authority_granted": false,
            }))
        },
        live_source,
        |source| async move { source.close().err() },
    )
    .await
    {
        Ok(source) => source,
        Err(failure) => {
            return Err(AppError::OutputAfterLiveOwnership {
                source: Box::new(failure.primary),
                live_close: failure.cleanup,
            });
        }
    };

    let owned = match admitted.start_controller(clock_origin).await {
        Ok(value) => value,
        Err(source) => {
            let live_close = live_source.close().err();
            return Err(AppError::ControllerStart {
                source: Box::new(source),
                live_close,
            });
        }
    };
    let (owned, mut source) = match emit_or_cleanup(
        || {
            reporter.emit(json!({
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "event": "controller_owned_at_exact_zero",
                "controller_owner": "in_process_v3",
                "motion_authority_granted": "calibration_schedule_only",
                "manual_authority_granted": false,
                "mpc_authority_granted": false,
            }))
        },
        (owned, live_source),
        |(owned, source)| async move {
            let terminal = owned
                .terminate_before_execution(CommissioningExternalSignal::SupervisorFault)
                .await;
            let live_close = source.close().err();
            PostControllerOwnershipCleanup {
                terminal: Box::new(terminal),
                live_close,
            }
        },
    )
    .await
    {
        Ok(resources) => resources,
        Err(failure) => {
            return Err(AppError::OutputAfterControllerOwnership {
                source: Box::new(failure.primary),
                cleanup: failure.cleanup,
            });
        }
    };
    let mut progress_reporter = NdjsonProgressReporter {
        reporter: reporter.clone(),
    };
    let operation = owned.execute(&mut source, &mut progress_reporter).await;
    let live_close = source.close().err();
    let proposal = match operation {
        Ok(proposal) if live_close.is_none() => proposal,
        Ok(_) => {
            return Err(AppError::LiveCloseAfterSuccess(
                live_close.expect("checked Some"),
            ));
        }
        Err(source) => {
            return Err(AppError::CommissioningRun {
                source: Box::new(source),
                live_close,
            });
        }
    };

    emit_proposal(&reporter, &proposal)?;
    Ok(())
}

struct EmitCleanupFailure<Primary, Cleanup> {
    primary: Primary,
    cleanup: Cleanup,
}

async fn emit_or_cleanup<Resource, Primary, Cleanup, CleanupFuture, CleanupEvidence>(
    emit: impl FnOnce() -> Result<(), Primary>,
    resources: Resource,
    cleanup: Cleanup,
) -> Result<Resource, EmitCleanupFailure<Primary, CleanupEvidence>>
where
    Cleanup: FnOnce(Resource) -> CleanupFuture,
    CleanupFuture: Future<Output = CleanupEvidence>,
{
    match emit() {
        Ok(()) => Ok(resources),
        Err(primary) => Err(EmitCleanupFailure {
            primary,
            cleanup: cleanup(resources).await,
        }),
    }
}

#[derive(Debug)]
struct PostControllerOwnershipCleanup {
    terminal:
        Box<kiko_slam::navigation::nano_base_commissioning_bootstrap::OwnedCommissioningRunError>,
    live_close: Option<CommissioningLiveCloseError>,
}

#[derive(Clone)]
struct NdjsonReporter {
    output: Arc<Mutex<BufWriter<io::Stdout>>>,
}

impl NdjsonReporter {
    fn stdout() -> Self {
        Self {
            output: Arc::new(Mutex::new(BufWriter::new(io::stdout()))),
        }
    }

    fn emit(&self, value: Value) -> Result<(), AppError> {
        let mut output = self.output.lock().map_err(|_| AppError::OutputPoisoned)?;
        serde_json::to_writer(&mut *output, &value).map_err(AppError::OutputEncode)?;
        output.write_all(b"\n").map_err(AppError::Output)?;
        output.flush().map_err(AppError::Output)
    }
}

struct NdjsonProgressReporter {
    reporter: NdjsonReporter,
}

#[derive(Debug)]
struct ReportingOutputError(Box<AppError>);

impl fmt::Display for ReportingOutputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl std::error::Error for ReportingOutputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

impl CommissioningProgressReporter for NdjsonProgressReporter {
    type Error = ReportingOutputError;

    fn before_observation(
        &mut self,
        request: CommissioningSamplingRequest,
    ) -> Result<(), Self::Error> {
        self.reporter
            .emit(progress_event(request))
            .map_err(|source| ReportingOutputError(Box::new(source)))
    }

    fn after_observation(
        &mut self,
        request: CommissioningSamplingRequest,
        outcome: CommissioningObservationKind,
    ) -> Result<(), Self::Error> {
        let event_name = match outcome {
            CommissioningObservationKind::SampleReady => "sample_ready",
            CommissioningObservationKind::TerminalSignal => "terminal_signal",
        };
        self.reporter
            .emit(json!({
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "event": event_name,
                "sample_ordinal": request.progress.samples_journaled,
            }))
            .map_err(|source| ReportingOutputError(Box::new(source)))
    }
}

fn progress_event(request: CommissioningSamplingRequest) -> Value {
    let progress = request.progress;
    let receipt = request.expected_receipt;
    let (state, step_index) = commissioning_state(progress.state);
    json!({
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "event": "progress",
        "state": state,
        "step_index": step_index,
        "samples_journaled": progress.samples_journaled,
        "requested_left_pwm_percent": progress.requested_left_pwm_percent,
        "requested_right_pwm_percent": progress.requested_right_pwm_percent,
        "last_applied_sequence": progress.last_applied_sequence,
        "exact_zero_applied": progress.exact_zero_applied,
        "receipt_observed_at_ns": receipt.observed_at_ns(),
        "receipt_left_pwm_percent": receipt.applied_pwm().left().get(),
        "receipt_right_pwm_percent": receipt.applied_pwm().right().get(),
    })
}

fn commissioning_state(state: CommissioningState) -> (&'static str, Option<u16>) {
    match state {
        CommissioningState::AwaitingInitialZero => ("awaiting_initial_zero", None),
        CommissioningState::ZeroDwell { next_step_index } => ("zero_dwell", Some(next_step_index)),
        CommissioningState::AwaitingApplication { step_index } => {
            ("awaiting_application", Some(step_index))
        }
        CommissioningState::Exciting { step_index } => ("exciting", Some(step_index)),
        CommissioningState::AwaitingInterstepZero {
            completed_step_index,
        } => ("awaiting_interstep_zero", Some(completed_step_index)),
        CommissioningState::Completed => ("completed", None),
        CommissioningState::Aborted(_) => ("aborted", None),
    }
}

fn emit_proposal(
    reporter: &NdjsonReporter,
    proposal: &NanoBaseCommissioningProposal,
) -> Result<(), AppError> {
    reporter.emit(json!({
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "event": "completed",
        "activation_status": proposal.activation_status,
        "proposal_activated": false,
        "manual_authority_granted": false,
        "mpc_authority_granted": false,
        "mapping_authority_granted": false,
        "dataset": artifact_json(&proposal.dataset),
        "proposed_plant": artifact_json(&proposal.proposed_plant),
        "proposal_evidence": artifact_json(&proposal.proposal_evidence),
        "journal": {
            "sha256": lower_hex(proposal.journal.content_sha256()),
            "records": proposal.journal.record_count(),
            "bytes": proposal.journal.byte_count(),
        },
        "lateral_validity": {
            "maximum_absolute_lateral_velocity_mps":
                proposal.lateral_validity.maximum_absolute_lateral_velocity_mps(),
            "training_samples": proposal.lateral_validity.training_sample_count(),
            "holdout_samples": proposal.lateral_validity.holdout_sample_count(),
            "scope": proposal.lateral_validity.scope_label(),
        },
    }))
}

fn artifact_json(
    artifact: &kiko_slam::navigation::nano_base_commissioning::PublishedCommissioningArtifact,
) -> Value {
    json!({
        "path": artifact.path,
        "sha256": lower_hex(artifact.content_sha256),
        "bytes": artifact.byte_count,
    })
}

fn lower_hex(bytes: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

#[derive(Debug)]
enum AppError {
    Signal(ctrlc::Error),
    LaunchRelativePath(kiko_device_inventory::ArtifactRelativePathError),
    StaticAdmission(
        Box<
            kiko_slam::navigation::nano_base_commissioning_bootstrap::NanoBaseCommissioningPreparationError,
        >,
    ),
    Entropy(getrandom::Error),
    ClockEpoch(
        kiko_slam::navigation::nano_base_commissioning_bootstrap::CommissioningClockEpochError,
    ),
    AccessoryStreamEpoch(kiko_expression_core::ObservationValueError),
    LiveGraph(
        Box<kiko_slam::navigation::nano_base_commissioning_live::CommissioningLiveOpenError>,
    ),
    OutputAfterLiveOwnership {
        source: Box<AppError>,
        live_close: Option<CommissioningLiveCloseError>,
    },
    AttendedAdmission {
        source: Box<
            kiko_slam::navigation::nano_base_commissioning_bootstrap::FreshAttendedCommissioningAdmissionError,
        >,
        live_close: Option<CommissioningLiveCloseError>,
    },
    ControllerStart {
        source: Box<
            kiko_slam::navigation::nano_base_commissioning_bootstrap::CommissioningControllerStartError,
        >,
        live_close: Option<CommissioningLiveCloseError>,
    },
    OutputAfterControllerOwnership {
        source: Box<AppError>,
        cleanup: PostControllerOwnershipCleanup,
    },
    CommissioningRun {
        source: Box<
            kiko_slam::navigation::nano_base_commissioning_bootstrap::OwnedCommissioningRunError,
        >,
        live_close: Option<CommissioningLiveCloseError>,
    },
    LiveCloseAfterSuccess(CommissioningLiveCloseError),
    OutputPoisoned,
    OutputEncode(serde_json::Error),
    Output(io::Error),
}

impl AppError {
    const fn stage(&self) -> &'static str {
        match self {
            Self::Signal(_) => "signal_handler",
            Self::LaunchRelativePath(_) | Self::StaticAdmission(_) => "static_admission",
            Self::Entropy(_) | Self::ClockEpoch(_) | Self::AccessoryStreamEpoch(_) => "clock_epoch",
            Self::LiveGraph(_) => "oak_live_graph",
            Self::OutputAfterLiveOwnership { .. } | Self::OutputAfterControllerOwnership { .. } => {
                "ndjson_output_cleanup"
            }
            Self::AttendedAdmission { .. } => "attestation",
            Self::ControllerStart { .. } => "controller_start",
            Self::CommissioningRun { .. } => "commissioning_run",
            Self::LiveCloseAfterSuccess(_) => "live_resource_close",
            Self::OutputPoisoned | Self::OutputEncode(_) | Self::Output(_) => "ndjson_output",
        }
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Signal(source) => {
                write!(formatter, "failed to install shutdown handler: {source}")
            }
            Self::LaunchRelativePath(source) => write!(formatter, "invalid launch path: {source}"),
            Self::StaticAdmission(source) => source.fmt(formatter),
            Self::Entropy(source) => write!(formatter, "failed to create clock epoch: {source}"),
            Self::ClockEpoch(source) => source.fmt(formatter),
            Self::AccessoryStreamEpoch(source) => source.fmt(formatter),
            Self::LiveGraph(source) => source.fmt(formatter),
            Self::OutputAfterLiveOwnership { source, live_close } => write!(
                formatter,
                "NDJSON output failed after live ownership: {source}; live_close={live_close:?}"
            ),
            Self::AttendedAdmission { source, live_close } => write!(
                formatter,
                "fresh attended admission failed: {source}; live_close={live_close:?}"
            ),
            Self::ControllerStart { source, live_close } => write!(
                formatter,
                "controller ownership failed: {source}; live_close={live_close:?}"
            ),
            Self::OutputAfterControllerOwnership { source, cleanup } => write!(
                formatter,
                "NDJSON output failed after controller ownership: {source}; terminal={}; owner_shutdown={:?}; live_close={:?}",
                cleanup.terminal.runtime, cleanup.terminal.owner_shutdown, cleanup.live_close
            ),
            Self::CommissioningRun { source, live_close } => write!(
                formatter,
                "commissioning failed: {source}; live_close={live_close:?}"
            ),
            Self::LiveCloseAfterSuccess(source) => {
                write!(
                    formatter,
                    "commissioning proposal exists but live OAK/accessory cleanup failed: {source}"
                )
            }
            Self::OutputPoisoned => formatter.write_str("NDJSON output mutex was poisoned"),
            Self::OutputEncode(source) => write!(formatter, "NDJSON encode failed: {source}"),
            Self::Output(source) => write!(formatter, "NDJSON output failed: {source}"),
        }
    }
}

impl std::error::Error for AppError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn required_arguments() -> Vec<&'static str> {
        vec![
            "kiko-nano-base-commission",
            "--deployment-root",
            "/opt/kiko/deployment",
            "--launch",
            "commissioning/nano-base-commissioning-launch-v1.json",
            "--state-root",
            "/var/lib/kiko/base-commissioning",
        ]
    }

    #[test]
    fn old_reusable_physical_claim_flags_are_rejected_by_clap() {
        for obsolete in [
            "--confirm-wheels-attached",
            "--confirm-clear-area",
            "--confirm-operator-attending",
            "--confirm-power-cut-tested-and-reachable",
        ] {
            let mut arguments = required_arguments();
            arguments.push(obsolete);
            let error = Args::try_parse_from(arguments).expect_err("obsolete flag must not parse");
            assert_eq!(error.kind(), clap::error::ErrorKind::UnknownArgument);
        }
    }

    #[test]
    fn cli_has_no_physical_claim_or_environment_argument_surface() {
        let arguments = required_arguments();
        Args::try_parse_from(arguments).expect("three non-physical boundaries parse");

        let command = <Args as clap::CommandFactory>::command();
        let argument_ids: Vec<_> = command
            .get_arguments()
            .map(|argument| argument.get_id().as_str())
            .collect();
        assert_eq!(argument_ids, ["deployment_root", "launch", "state_root"]);
        assert!(
            command
                .get_arguments()
                .all(|argument| argument.get_env().is_none())
        );
    }

    #[tokio::test]
    async fn output_failure_consumes_resources_and_retains_cleanup_evidence() {
        let failure = emit_or_cleanup(
            || Err::<(), _>("injected output failure"),
            41_u8,
            |resource| async move { resource + 1 },
        )
        .await
        .expect_err("injected output failure must run cleanup");

        assert_eq!(failure.primary, "injected output failure");
        assert_eq!(failure.cleanup, 42);
    }
}
