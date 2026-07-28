//! Camera-only qualification of Kiko's canonical Jetson Orin Nano OAK stream
//! graph.
//!
//! Usage:
//! `oak_stream_qualify EXACT_MXID MAXIMUM_USB_SPEED FRAMES_PER_IMAGE_STREAM MAX_DURATION_SECONDS`
//!
//! Successful stdout is one JSON document. Native sequence gaps describe
//! missing host deliveries only; they do not attribute loss to USB.
//! The measured window begins at a bounded candidate instant recorded before
//! every enabled queue is observed empty by one nonblocking check.

use oak_sys::{
    CloseError, ConnectedDeviceIdentityError, DepthAlignment, DepthConfig, DepthError, Device,
    DeviceConfig, DeviceFrameSequence, ImageError, ImageFrame, ImuConfig, ImuError, MonoConfig,
    QueueConfig, RgbConfig, StreamId, UsbTransportEvidenceError, UsbTransportPolicy,
    UsbTransportSpeed,
};
use std::ffi::OsString;
use std::fmt;
use std::num::ParseIntError;
use std::process::ExitCode;
use std::time::{Duration, Instant};
use thiserror::Error;

const RGB_WIDTH: u32 = 640;
const RGB_HEIGHT: u32 = 400;
const IMAGE_RATE_HZ: u32 = 15;
const IMU_RATE_HZ: u32 = 200;
/// Qualification admits at least 4/5 of the requested 200 Hz IMU rate.
const MINIMUM_IMU_RATE_NUMERATOR: u32 = 4;
const MINIMUM_IMU_RATE_DENOMINATOR: u32 = 5;
const QUEUE_SIZE: u32 = 4;
const RGB_STRIDE_BYTES: u32 = RGB_WIDTH * 3;
const MONO_STRIDE_BYTES: u32 = RGB_WIDTH;
const DEPTH_STRIDE_BYTES: u32 = RGB_WIDTH * 2;
const RGB_PAYLOAD_BYTES: u64 = 640 * 400 * 3;
const MONO_PAYLOAD_BYTES: u64 = 640 * 400;
const DEPTH_PAYLOAD_BYTES: u64 = 640 * 400 * 2;
const MAXIMUM_FRAMES_PER_STREAM: u32 = IMAGE_RATE_HZ * 60 * 60;
const MAXIMUM_DURATION_SECONDS: u64 = 60 * 60;
const IDLE_POLL_SLEEP: Duration = Duration::from_millis(1);
const EMPTY_EPOCH_MAXIMUM_DURATION: Duration = Duration::from_secs(5);
const EMPTY_EPOCH_MAXIMUM_CANDIDATES: u64 = 100_000;

fn main() -> ExitCode {
    match Command::parse(std::env::args_os()) {
        Ok(Command::Help { usage }) => {
            println!("{usage}");
            ExitCode::SUCCESS
        }
        Ok(Command::Run(arguments)) => match execute(arguments) {
            Ok(report) => {
                println!("{}", report.json("complete"));
                ExitCode::SUCCESS
            }
            Err(error) => {
                if let Some(report) = error.report() {
                    println!("{}", report.json(error.report_status()));
                }
                eprintln!("oak_stream_qualify: {error}");
                ExitCode::FAILURE
            }
        },
        Err(error) => {
            eprintln!("oak_stream_qualify: {error}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    Help { usage: String },
    Run(Arguments),
}

impl Command {
    fn parse(raw_arguments: impl IntoIterator<Item = OsString>) -> Result<Self, ArgumentError> {
        let mut arguments = raw_arguments.into_iter();
        let program = parse_unicode(
            arguments
                .next()
                .unwrap_or_else(|| OsString::from("oak_stream_qualify")),
            0,
        )?;
        let values = arguments
            .enumerate()
            .map(|(index, value)| parse_unicode(value, index + 1))
            .collect::<Result<Vec<_>, _>>()?;

        if values.as_slice() == ["--help"] || values.as_slice() == ["-h"] {
            return Ok(Self::Help {
                usage: usage(&program),
            });
        }
        if values.len() != 4 {
            return Err(ArgumentError::WrongArity {
                usage: usage(&program),
                actual: values.len(),
            });
        }

        let mxid = values[0].clone();
        if mxid.is_empty() {
            return Err(ArgumentError::EmptyMxid);
        }
        let maximum_usb_speed = QualificationUsbMaximum::parse(&values[1])?;
        let frames_per_image_stream = parse_bounded_u32(
            &values[2],
            "FRAMES_PER_IMAGE_STREAM",
            MAXIMUM_FRAMES_PER_STREAM,
        )?;
        let maximum_duration_seconds =
            parse_bounded_u64(&values[3], "MAX_DURATION_SECONDS", MAXIMUM_DURATION_SECONDS)?;
        Ok(Self::Run(Arguments {
            mxid,
            maximum_usb_speed,
            frames_per_image_stream,
            maximum_duration: Duration::from_secs(maximum_duration_seconds),
            maximum_duration_seconds,
        }))
    }
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} EXACT_MXID MAXIMUM_USB_SPEED FRAMES_PER_IMAGE_STREAM MAX_DURATION_SECONDS\n\
         MAXIMUM_USB_SPEED: SUPER or SUPER_PLUS\n\
         FRAMES_PER_IMAGE_STREAM: 1..={MAXIMUM_FRAMES_PER_STREAM}\n\
         MAX_DURATION_SECONDS: 1..={MAXIMUM_DURATION_SECONDS}"
    )
}

fn parse_unicode(value: OsString, position: usize) -> Result<String, ArgumentError> {
    value
        .into_string()
        .map_err(|_| ArgumentError::NonUnicode { position })
}

fn parse_bounded_u32(value: &str, field: &'static str, maximum: u32) -> Result<u32, ArgumentError> {
    let parsed = value
        .parse::<u32>()
        .map_err(|source| ArgumentError::InvalidInteger {
            field,
            value: value.to_owned(),
            source,
        })?;
    if !(1..=maximum).contains(&parsed) {
        return Err(ArgumentError::OutsideBounds {
            field,
            value: u64::from(parsed),
            maximum: u64::from(maximum),
        });
    }
    Ok(parsed)
}

fn parse_bounded_u64(value: &str, field: &'static str, maximum: u64) -> Result<u64, ArgumentError> {
    let parsed = value
        .parse::<u64>()
        .map_err(|source| ArgumentError::InvalidInteger {
            field,
            value: value.to_owned(),
            source,
        })?;
    if !(1..=maximum).contains(&parsed) {
        return Err(ArgumentError::OutsideBounds {
            field,
            value: parsed,
            maximum,
        });
    }
    Ok(parsed)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Arguments {
    mxid: String,
    maximum_usb_speed: QualificationUsbMaximum,
    frames_per_image_stream: u32,
    maximum_duration: Duration,
    maximum_duration_seconds: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum QualificationUsbMaximum {
    Super,
    SuperPlus,
}

impl QualificationUsbMaximum {
    fn parse(value: &str) -> Result<Self, ArgumentError> {
        match value {
            "SUPER" => Ok(Self::Super),
            "SUPER_PLUS" => Ok(Self::SuperPlus),
            _ => Err(ArgumentError::UnsupportedMaximumUsbSpeed {
                value: value.to_owned(),
            }),
        }
    }

    fn policy(self) -> UsbTransportPolicy {
        let maximum = match self {
            Self::Super => UsbTransportSpeed::Super,
            Self::SuperPlus => UsbTransportSpeed::SuperPlus,
        };
        UsbTransportPolicy::try_new(maximum, UsbTransportSpeed::Super)
            .expect("qualification USB maximum always admits the fixed SUPER minimum")
    }
}

#[derive(Error, Debug)]
enum ArgumentError {
    #[error("argument {position} is not valid Unicode")]
    NonUnicode { position: usize },
    #[error("{usage}; received {actual} positional arguments")]
    WrongArity { usage: String, actual: usize },
    #[error("EXACT_MXID must be nonempty")]
    EmptyMxid,
    #[error("MAXIMUM_USB_SPEED must be exactly SUPER or SUPER_PLUS, got '{value}'")]
    UnsupportedMaximumUsbSpeed { value: String },
    #[error("{field} must be an unsigned decimal integer, got '{value}': {source}")]
    InvalidInteger {
        field: &'static str,
        value: String,
        #[source]
        source: ParseIntError,
    },
    #[error("{field} must be in 1..={maximum}, got {value}")]
    OutsideBounds {
        field: &'static str,
        value: u64,
        maximum: u64,
    },
}

fn canonical_config(maximum_usb_speed: QualificationUsbMaximum) -> DeviceConfig {
    DeviceConfig {
        usb_transport: maximum_usb_speed.policy(),
        rgb: Some(RgbConfig {
            width: RGB_WIDTH,
            height: RGB_HEIGHT,
            fps: IMAGE_RATE_HZ,
        }),
        mono: Some(MonoConfig {
            width: RGB_WIDTH,
            height: RGB_HEIGHT,
            fps: IMAGE_RATE_HZ,
            rectified: true,
        }),
        depth: Some(DepthConfig {
            width: RGB_WIDTH,
            height: RGB_HEIGHT,
            fps: IMAGE_RATE_HZ,
            alignment: DepthAlignment::RectifiedLeft,
        }),
        imu: Some(ImuConfig {
            rate_hz: IMU_RATE_HZ,
        }),
        queue: QueueConfig {
            size: QUEUE_SIZE,
            blocking: false,
        },
    }
}

fn execute(arguments: Arguments) -> Result<QualificationReport, ProgramError> {
    let mut device = Device::connect(
        &arguments.mxid,
        canonical_config(arguments.maximum_usb_speed),
    )
    .map_err(ProgramError::Connect)?;
    let operation = qualify(&mut device, &arguments);
    let close = device.close();

    match (operation, close) {
        (Ok(report), Ok(())) => Ok(report),
        (Err(operation), Ok(())) => Err(ProgramError::Operation { operation }),
        (Ok(report), Err(close)) => Err(ProgramError::Close {
            report: Box::new(report),
            close,
        }),
        (Err(operation), Err(close)) => Err(ProgramError::OperationAndClose { operation, close }),
    }
}

fn qualify(
    device: &mut Device,
    arguments: &Arguments,
) -> Result<QualificationReport, OperationFailure> {
    let identity = device
        .connected_identity()
        .map_err(|source| OperationFailure::without_report(OperationError::Identity(source)))?;
    let connected_mxid = identity.mxid().to_owned();
    let usb_transport = *device
        .usb_transport_evidence()
        .map_err(|source| OperationFailure::without_report(OperationError::UsbTransport(source)))?;

    let attempt = capture(device, arguments);
    let report = QualificationReport {
        connected_mxid,
        requested_maximum_usb_speed: usb_transport.requested_maximum(),
        required_minimum_usb_speed: usb_transport.required_minimum(),
        observed_usb_speed: usb_transport.observed(),
        frames_per_image_stream: arguments.frames_per_image_stream,
        maximum_duration_seconds: arguments.maximum_duration_seconds,
        elapsed: attempt.elapsed,
        required_minimum_imu_samples: attempt.required_minimum_imu_samples,
        empty_queue_epoch: attempt.empty_queue_epoch,
        stats: attempt.stats,
    };
    if let Some(source) = attempt.failure {
        return Err(OperationFailure::with_report(
            OperationError::Capture(source),
            report,
        ));
    }
    if !report.targets_reached() {
        return Err(OperationFailure::with_report(
            OperationError::DurationLimit {
                target: arguments.frames_per_image_stream,
                rgb: report.stats.rgb.delivered,
                mono_left: report.stats.mono_left.delivered,
                mono_right: report.stats.mono_right.delivered,
                depth: report.stats.depth.delivered,
                imu_delivered: report.stats.imu_samples,
                imu_required: report.required_minimum_imu_samples,
            },
            report,
        ));
    }
    if let Some(evidence) = report.stats.sequence_anomalies() {
        return Err(OperationFailure::with_report(
            OperationError::NativeSequenceAnomalies { evidence },
            report,
        ));
    }
    Ok(report)
}

#[derive(Debug)]
struct CaptureAttempt {
    stats: CaptureStats,
    elapsed: Duration,
    required_minimum_imu_samples: u64,
    empty_queue_epoch: EmptyQueueEpochStats,
    failure: Option<CaptureError>,
}

fn capture(device: &mut Device, arguments: &Arguments) -> CaptureAttempt {
    let epoch = match establish_empty_queue_epoch(device) {
        Ok(epoch) => epoch,
        Err(failure) => {
            return CaptureAttempt {
                stats: CaptureStats::default(),
                elapsed: Duration::ZERO,
                required_minimum_imu_samples: 0,
                empty_queue_epoch: failure.stats,
                failure: Some(CaptureError::EmptyQueueEpoch(failure.source)),
            };
        }
    };
    let started = epoch.measurement_started;
    let empty_queue_epoch = epoch.stats;
    let mut stats = CaptureStats::default();
    let target = u64::from(arguments.frames_per_image_stream);

    loop {
        let elapsed = started.elapsed();
        let required_minimum_imu_samples = minimum_imu_samples_for_elapsed(elapsed);
        if stats.image_targets_reached(target) && stats.imu_samples >= required_minimum_imu_samples
        {
            return CaptureAttempt {
                stats,
                elapsed,
                required_minimum_imu_samples,
                empty_queue_epoch,
                failure: None,
            };
        }
        if elapsed >= arguments.maximum_duration {
            return CaptureAttempt {
                stats,
                elapsed,
                required_minimum_imu_samples,
                empty_queue_epoch,
                failure: None,
            };
        }

        let mut delivered_anything = false;
        match poll_image(device.rgb(0), ImageContract::rgb(), &mut stats.rgb) {
            Ok(delivered) => delivered_anything |= delivered,
            Err(source) => {
                return failed_attempt(
                    stats,
                    started.elapsed(),
                    empty_queue_epoch,
                    CaptureError::Rgb(source),
                );
            }
        }
        match poll_image(
            device.mono_left(0),
            ImageContract::mono_left(),
            &mut stats.mono_left,
        ) {
            Ok(delivered) => delivered_anything |= delivered,
            Err(source) => {
                return failed_attempt(
                    stats,
                    started.elapsed(),
                    empty_queue_epoch,
                    CaptureError::MonoLeft(source),
                );
            }
        }
        match poll_image(
            device.mono_right(0),
            ImageContract::mono_right(),
            &mut stats.mono_right,
        ) {
            Ok(delivered) => delivered_anything |= delivered,
            Err(source) => {
                return failed_attempt(
                    stats,
                    started.elapsed(),
                    empty_queue_epoch,
                    CaptureError::MonoRight(source),
                );
            }
        }
        match poll_depth(device.depth(0), &mut stats.depth) {
            Ok(delivered) => delivered_anything |= delivered,
            Err(source) => {
                return failed_attempt(
                    stats,
                    started.elapsed(),
                    empty_queue_epoch,
                    CaptureError::Depth(source),
                );
            }
        }
        match device.imu() {
            Ok(samples) => {
                if !samples.is_empty() {
                    delivered_anything = true;
                }
                match u64::try_from(samples.len())
                    .ok()
                    .and_then(|count| stats.imu_samples.checked_add(count))
                {
                    Some(total) => stats.imu_samples = total,
                    None => {
                        return failed_attempt(
                            stats,
                            started.elapsed(),
                            empty_queue_epoch,
                            CaptureError::Accounting(AccountingError::CounterOverflow {
                                counter: "imu.samples",
                            }),
                        );
                    }
                }
            }
            Err(ImuError::Empty) => {
                if let Some(total) = stats.imu_empty_polls.checked_add(1) {
                    stats.imu_empty_polls = total;
                } else {
                    return failed_attempt(
                        stats,
                        started.elapsed(),
                        empty_queue_epoch,
                        CaptureError::Accounting(AccountingError::CounterOverflow {
                            counter: "imu.empty_polls",
                        }),
                    );
                }
            }
            Err(source) => {
                return failed_attempt(
                    stats,
                    started.elapsed(),
                    empty_queue_epoch,
                    CaptureError::Imu(source),
                );
            }
        }

        if !delivered_anything {
            let remaining = arguments.maximum_duration.saturating_sub(started.elapsed());
            std::thread::sleep(IDLE_POLL_SLEEP.min(remaining));
        }
    }
}

fn failed_attempt(
    stats: CaptureStats,
    elapsed: Duration,
    empty_queue_epoch: EmptyQueueEpochStats,
    failure: CaptureError,
) -> CaptureAttempt {
    CaptureAttempt {
        stats,
        elapsed,
        required_minimum_imu_samples: minimum_imu_samples_for_elapsed(elapsed),
        empty_queue_epoch,
        failure: Some(failure),
    }
}

#[derive(Debug)]
struct EmptyQueueEpoch {
    /// Recorded before all five queue checks whose empty result admitted it.
    measurement_started: Instant,
    stats: EmptyQueueEpochStats,
}

#[derive(Debug)]
struct EmptyQueueEpochFailure {
    stats: EmptyQueueEpochStats,
    source: Box<EmptyQueueEpochError>,
}

#[derive(Debug, Clone, Copy, Default)]
struct EmptyQueueEpochStats {
    candidate_epochs: u64,
    discarded_rgb_frames: u64,
    discarded_mono_left_frames: u64,
    discarded_mono_right_frames: u64,
    discarded_depth_frames: u64,
    discarded_imu_samples: u64,
    elapsed: Duration,
    accepted: bool,
}

impl EmptyQueueEpochStats {
    fn begin_candidate(&mut self) -> Result<(), AccountingError> {
        self.candidate_epochs =
            self.candidate_epochs
                .checked_add(1)
                .ok_or(AccountingError::CounterOverflow {
                    counter: "empty_queue_epoch.candidate_epochs",
                })?;
        Ok(())
    }

    fn record_discarded(
        &mut self,
        rgb_delivered: bool,
        mono_left_delivered: bool,
        mono_right_delivered: bool,
        depth_delivered: bool,
        imu_samples: Option<u64>,
    ) -> Result<(), AccountingError> {
        for (counter, delivered) in [
            (&mut self.discarded_rgb_frames, rgb_delivered),
            (&mut self.discarded_mono_left_frames, mono_left_delivered),
            (&mut self.discarded_mono_right_frames, mono_right_delivered),
            (&mut self.discarded_depth_frames, depth_delivered),
        ] {
            if delivered {
                *counter = counter
                    .checked_add(1)
                    .ok_or(AccountingError::CounterOverflow {
                        counter: "empty_queue_epoch.discarded_image_frames",
                    })?;
            }
        }
        if let Some(samples) = imu_samples {
            self.discarded_imu_samples = self.discarded_imu_samples.checked_add(samples).ok_or(
                AccountingError::CounterOverflow {
                    counter: "empty_queue_epoch.discarded_imu_samples",
                },
            )?;
        }
        Ok(())
    }
}

fn establish_empty_queue_epoch(
    device: &mut Device,
) -> Result<EmptyQueueEpoch, EmptyQueueEpochFailure> {
    let phase_started = Instant::now();
    let mut stats = EmptyQueueEpochStats::default();

    loop {
        if phase_started.elapsed() >= EMPTY_EPOCH_MAXIMUM_DURATION {
            return Err(empty_epoch_failure(
                stats,
                phase_started,
                EmptyQueueEpochError::MaximumDuration {
                    maximum_milliseconds: EMPTY_EPOCH_MAXIMUM_DURATION.as_millis(),
                },
            ));
        }
        if stats.candidate_epochs >= EMPTY_EPOCH_MAXIMUM_CANDIDATES {
            return Err(empty_epoch_failure(
                stats,
                phase_started,
                EmptyQueueEpochError::MaximumCandidates {
                    maximum: EMPTY_EPOCH_MAXIMUM_CANDIDATES,
                },
            ));
        }
        if let Err(source) = stats.begin_candidate() {
            return Err(empty_epoch_failure(
                stats,
                phase_started,
                EmptyQueueEpochError::Accounting(source),
            ));
        }

        // This instant precedes every nonblocking check. If an earlier queue
        // becomes nonempty after its check, that data is still post-candidate
        // and belongs to the subsequently measured window.
        let candidate = Instant::now();
        let rgb_delivered = drain_image_for_empty_epoch(device.rgb(0), ImageContract::rgb())
            .map_err(|source| {
                empty_epoch_failure(stats, phase_started, EmptyQueueEpochError::Rgb(source))
            })?;
        if let Err(source) = stats.record_discarded(rgb_delivered, false, false, false, None) {
            return Err(empty_epoch_failure(
                stats,
                phase_started,
                EmptyQueueEpochError::Accounting(source),
            ));
        }
        let mono_left_delivered = drain_image_for_empty_epoch(
            device.mono_left(0),
            ImageContract::mono_left(),
        )
        .map_err(|source| {
            empty_epoch_failure(stats, phase_started, EmptyQueueEpochError::MonoLeft(source))
        })?;
        if let Err(source) = stats.record_discarded(false, mono_left_delivered, false, false, None)
        {
            return Err(empty_epoch_failure(
                stats,
                phase_started,
                EmptyQueueEpochError::Accounting(source),
            ));
        }
        let mono_right_delivered =
            drain_image_for_empty_epoch(device.mono_right(0), ImageContract::mono_right())
                .map_err(|source| {
                    empty_epoch_failure(
                        stats,
                        phase_started,
                        EmptyQueueEpochError::MonoRight(source),
                    )
                })?;
        if let Err(source) = stats.record_discarded(false, false, mono_right_delivered, false, None)
        {
            return Err(empty_epoch_failure(
                stats,
                phase_started,
                EmptyQueueEpochError::Accounting(source),
            ));
        }
        let depth_delivered = drain_depth_for_empty_epoch(device.depth(0)).map_err(|source| {
            empty_epoch_failure(stats, phase_started, EmptyQueueEpochError::Depth(source))
        })?;
        if let Err(source) = stats.record_discarded(false, false, false, depth_delivered, None) {
            return Err(empty_epoch_failure(
                stats,
                phase_started,
                EmptyQueueEpochError::Accounting(source),
            ));
        }
        let imu_samples = match device.imu() {
            Ok(samples) if samples.is_empty() => {
                return Err(empty_epoch_failure(
                    stats,
                    phase_started,
                    EmptyQueueEpochError::EmptySuccessfulImuBatch,
                ));
            }
            Ok(samples) => Some(u64::try_from(samples.len()).map_err(|_| {
                empty_epoch_failure(
                    stats,
                    phase_started,
                    EmptyQueueEpochError::Accounting(AccountingError::CounterOverflow {
                        counter: "empty_queue_epoch.imu_batch_samples",
                    }),
                )
            })?),
            Err(ImuError::Empty) => None,
            Err(source) => {
                return Err(empty_epoch_failure(
                    stats,
                    phase_started,
                    EmptyQueueEpochError::Imu(source),
                ));
            }
        };
        if let Err(source) = stats.record_discarded(false, false, false, false, imu_samples) {
            return Err(empty_epoch_failure(
                stats,
                phase_started,
                EmptyQueueEpochError::Accounting(source),
            ));
        }
        stats.elapsed = phase_started.elapsed();
        if stats.elapsed >= EMPTY_EPOCH_MAXIMUM_DURATION {
            return Err(EmptyQueueEpochFailure {
                stats,
                source: Box::new(EmptyQueueEpochError::MaximumDuration {
                    maximum_milliseconds: EMPTY_EPOCH_MAXIMUM_DURATION.as_millis(),
                }),
            });
        }

        if candidate_is_empty_epoch([
            rgb_delivered,
            mono_left_delivered,
            mono_right_delivered,
            depth_delivered,
            imu_samples.is_some(),
        ]) {
            stats.accepted = true;
            return Ok(EmptyQueueEpoch {
                measurement_started: candidate,
                stats,
            });
        }
    }
}

/// Pure admission predicate for one ordered set of candidate-epoch checks.
fn candidate_is_empty_epoch(delivered: [bool; 5]) -> bool {
    delivered.into_iter().all(|was_delivered| !was_delivered)
}

fn empty_epoch_failure(
    mut stats: EmptyQueueEpochStats,
    phase_started: Instant,
    source: EmptyQueueEpochError,
) -> EmptyQueueEpochFailure {
    stats.elapsed = phase_started.elapsed();
    EmptyQueueEpochFailure {
        stats,
        source: Box::new(source),
    }
}

fn minimum_imu_samples_for_elapsed(elapsed: Duration) -> u64 {
    let scaled =
        elapsed.as_nanos() * u128::from(IMU_RATE_HZ) * u128::from(MINIMUM_IMU_RATE_NUMERATOR);
    let divisor = 1_000_000_000_u128 * u128::from(MINIMUM_IMU_RATE_DENOMINATOR);
    let required = scaled.div_ceil(divisor);
    u64::try_from(required)
        .expect("parsed one-hour qualification duration keeps the IMU requirement in u64")
}

#[derive(Debug, Clone, Copy)]
struct ImageContract {
    name: &'static str,
    stream: StreamId,
    width: u32,
    height: u32,
    stride_bytes: u32,
    payload_bytes: u64,
}

impl ImageContract {
    const fn rgb() -> Self {
        Self {
            name: "RGB",
            stream: StreamId::Rgb,
            width: RGB_WIDTH,
            height: RGB_HEIGHT,
            stride_bytes: RGB_STRIDE_BYTES,
            payload_bytes: RGB_PAYLOAD_BYTES,
        }
    }

    const fn mono_left() -> Self {
        Self {
            name: "rectified-left mono",
            stream: StreamId::MonoLeft,
            width: RGB_WIDTH,
            height: RGB_HEIGHT,
            stride_bytes: MONO_STRIDE_BYTES,
            payload_bytes: MONO_PAYLOAD_BYTES,
        }
    }

    const fn mono_right() -> Self {
        Self {
            name: "rectified-right mono",
            stream: StreamId::MonoRight,
            width: RGB_WIDTH,
            height: RGB_HEIGHT,
            stride_bytes: MONO_STRIDE_BYTES,
            payload_bytes: MONO_PAYLOAD_BYTES,
        }
    }

    fn validate(self, frame: &ImageFrame, payload_bytes: u64) -> Result<(), ImageContractError> {
        if frame.stream != self.stream {
            return Err(ImageContractError::UnexpectedStream {
                name: self.name,
                expected: self.stream,
                actual: frame.stream,
            });
        }
        if (frame.width, frame.height, frame.stride_bytes, payload_bytes)
            != (
                self.width,
                self.height,
                self.stride_bytes,
                self.payload_bytes,
            )
        {
            return Err(ImageContractError::LayoutMismatch {
                name: self.name,
                expected_width: self.width,
                expected_height: self.height,
                expected_stride_bytes: self.stride_bytes,
                expected_payload_bytes: self.payload_bytes,
                actual_width: frame.width,
                actual_height: frame.height,
                actual_stride_bytes: frame.stride_bytes,
                actual_payload_bytes: payload_bytes,
            });
        }
        Ok(())
    }
}

fn drain_image_for_empty_epoch(
    result: Result<ImageFrame, ImageError>,
    contract: ImageContract,
) -> Result<bool, CaptureImageError> {
    match result {
        Ok(frame) => {
            let payload_bytes = u64::try_from(frame.pixels().len()).map_err(|_| {
                AccountingError::CounterOverflow {
                    counter: "empty_queue_epoch.image_payload_bytes",
                }
            })?;
            contract.validate(&frame, payload_bytes)?;
            Ok(true)
        }
        Err(ImageError::QueueEmpty) => Ok(false),
        // A zero-timeout native check must report QueueEmpty, not Timeout.
        Err(source) => Err(CaptureImageError::Acquisition(source)),
    }
}

fn drain_depth_for_empty_epoch(
    result: Result<oak_sys::DepthFrame, DepthError>,
) -> Result<bool, CaptureDepthError> {
    match result {
        Ok(frame) => {
            let payload_bytes = u64::try_from(frame.depth_mm().len())
                .ok()
                .and_then(|samples| samples.checked_mul(2))
                .ok_or(AccountingError::CounterOverflow {
                    counter: "empty_queue_epoch.depth_payload_bytes",
                })?;
            validate_depth_contract(&frame, payload_bytes)?;
            Ok(true)
        }
        Err(DepthError::QueueEmpty) => Ok(false),
        // A zero-timeout native check must report QueueEmpty, not Timeout.
        Err(source) => Err(CaptureDepthError::Acquisition(source)),
    }
}

fn poll_image(
    result: Result<ImageFrame, ImageError>,
    contract: ImageContract,
    stats: &mut ImageStreamStats,
) -> Result<bool, CaptureImageError> {
    match result {
        Ok(frame) => {
            let payload_bytes = u64::try_from(frame.pixels().len()).map_err(|_| {
                AccountingError::CounterOverflow {
                    counter: "image.payload_bytes",
                }
            })?;
            contract.validate(&frame, payload_bytes)?;
            stats.record(frame.device_capture_sequence, payload_bytes)?;
            Ok(true)
        }
        Err(ImageError::QueueEmpty) => {
            stats.record_queue_empty()?;
            Ok(false)
        }
        Err(ImageError::Timeout { .. }) => {
            stats.record_timeout()?;
            Ok(false)
        }
        Err(source) => Err(CaptureImageError::Acquisition(source)),
    }
}

fn poll_depth(
    result: Result<oak_sys::DepthFrame, DepthError>,
    stats: &mut ImageStreamStats,
) -> Result<bool, CaptureDepthError> {
    match result {
        Ok(frame) => {
            let payload_bytes = u64::try_from(frame.depth_mm().len())
                .ok()
                .and_then(|samples| samples.checked_mul(2))
                .ok_or(AccountingError::CounterOverflow {
                    counter: "depth.payload_bytes",
                })?;
            validate_depth_contract(&frame, payload_bytes)?;
            stats.record(frame.device_capture_sequence, payload_bytes)?;
            Ok(true)
        }
        Err(DepthError::QueueEmpty) => {
            stats.record_queue_empty()?;
            Ok(false)
        }
        Err(DepthError::Timeout { .. }) => {
            stats.record_timeout()?;
            Ok(false)
        }
        Err(source) => Err(CaptureDepthError::Acquisition(source)),
    }
}

fn validate_depth_contract(
    frame: &oak_sys::DepthFrame,
    payload_bytes: u64,
) -> Result<(), DepthContractError> {
    let observed_packed_stride_bytes = frame.width.checked_mul(2);
    let observed_alignment = frame.connected_alignment();
    if (
        frame.width,
        frame.height,
        observed_packed_stride_bytes,
        payload_bytes,
        observed_alignment,
    ) != (
        RGB_WIDTH,
        RGB_HEIGHT,
        Some(DEPTH_STRIDE_BYTES),
        DEPTH_PAYLOAD_BYTES,
        Some(DepthAlignment::RectifiedLeft),
    ) {
        return Err(DepthContractError::LayoutOrAlignmentMismatch {
            expected_width: RGB_WIDTH,
            expected_height: RGB_HEIGHT,
            expected_packed_stride_bytes: DEPTH_STRIDE_BYTES,
            expected_payload_bytes: DEPTH_PAYLOAD_BYTES,
            expected_alignment: DepthAlignment::RectifiedLeft,
            actual_width: frame.width,
            actual_height: frame.height,
            actual_packed_stride_bytes: observed_packed_stride_bytes,
            actual_payload_bytes: payload_bytes,
            actual_alignment: observed_alignment,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, Default)]
struct CaptureStats {
    rgb: ImageStreamStats,
    mono_left: ImageStreamStats,
    mono_right: ImageStreamStats,
    depth: ImageStreamStats,
    imu_samples: u64,
    imu_empty_polls: u64,
}

impl CaptureStats {
    fn image_targets_reached(self, target: u64) -> bool {
        [
            self.rgb.delivered,
            self.mono_left.delivered,
            self.mono_right.delivered,
            self.depth.delivered,
        ]
        .into_iter()
        .all(|delivered| delivered >= target)
    }

    fn sequence_anomalies(self) -> Option<SequenceAnomalyEvidence> {
        let evidence = SequenceAnomalyEvidence {
            rgb: self.rgb.sequence,
            mono_left: self.mono_left.sequence,
            mono_right: self.mono_right.sequence,
            depth: self.depth.sequence,
        };
        evidence.has_any().then_some(evidence)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ImageStreamStats {
    delivered: u64,
    payload_bytes: u64,
    queue_empty_polls: u64,
    timeout_polls: u64,
    sequence: SequenceAccounting,
}

impl ImageStreamStats {
    fn record(
        &mut self,
        sequence: DeviceFrameSequence,
        payload_bytes: u64,
    ) -> Result<(), AccountingError> {
        let mut updated = *self;
        updated.delivered =
            updated
                .delivered
                .checked_add(1)
                .ok_or(AccountingError::CounterOverflow {
                    counter: "image.delivered",
                })?;
        updated.payload_bytes = updated.payload_bytes.checked_add(payload_bytes).ok_or(
            AccountingError::CounterOverflow {
                counter: "image.payload_bytes",
            },
        )?;
        updated.sequence.observe(sequence.as_u64())?;
        *self = updated;
        Ok(())
    }

    fn record_queue_empty(&mut self) -> Result<(), AccountingError> {
        self.queue_empty_polls =
            self.queue_empty_polls
                .checked_add(1)
                .ok_or(AccountingError::CounterOverflow {
                    counter: "image.queue_empty_polls",
                })?;
        Ok(())
    }

    fn record_timeout(&mut self) -> Result<(), AccountingError> {
        self.timeout_polls =
            self.timeout_polls
                .checked_add(1)
                .ok_or(AccountingError::CounterOverflow {
                    counter: "image.timeout_polls",
                })?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SequenceAccounting {
    first: Option<u64>,
    previous: Option<u64>,
    gaps: u64,
    duplicates: u64,
    regressions: u64,
}

impl SequenceAccounting {
    fn observe(&mut self, current: u64) -> Result<(), AccountingError> {
        let mut updated = *self;
        match updated.previous {
            None => updated.first = Some(current),
            Some(previous) if current > previous => {
                updated.gaps = updated.gaps.checked_add(current - previous - 1).ok_or(
                    AccountingError::CounterOverflow {
                        counter: "image.native_sequence_gaps",
                    },
                )?;
            }
            Some(previous) if current == previous => {
                updated.duplicates =
                    updated
                        .duplicates
                        .checked_add(1)
                        .ok_or(AccountingError::CounterOverflow {
                            counter: "image.native_sequence_duplicates",
                        })?;
            }
            Some(_) => {
                updated.regressions =
                    updated
                        .regressions
                        .checked_add(1)
                        .ok_or(AccountingError::CounterOverflow {
                            counter: "image.native_sequence_regressions",
                        })?;
            }
        }
        updated.previous = Some(current);
        *self = updated;
        Ok(())
    }

    const fn has_anomaly(self) -> bool {
        self.gaps != 0 || self.duplicates != 0 || self.regressions != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SequenceAnomalyEvidence {
    rgb: SequenceAccounting,
    mono_left: SequenceAccounting,
    mono_right: SequenceAccounting,
    depth: SequenceAccounting,
}

impl SequenceAnomalyEvidence {
    const fn has_any(self) -> bool {
        self.rgb.has_anomaly()
            || self.mono_left.has_anomaly()
            || self.mono_right.has_anomaly()
            || self.depth.has_anomaly()
    }
}

impl fmt::Display for SequenceAnomalyEvidence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let write_stream = |formatter: &mut fmt::Formatter<'_>,
                            name: &str,
                            sequence: SequenceAccounting|
         -> fmt::Result {
            write!(
                formatter,
                "{name}=gaps:{}/duplicates:{}/regressions:{}",
                sequence.gaps, sequence.duplicates, sequence.regressions
            )
        };
        write_stream(formatter, "rgb", self.rgb)?;
        formatter.write_str(", ")?;
        write_stream(formatter, "mono_left", self.mono_left)?;
        formatter.write_str(", ")?;
        write_stream(formatter, "mono_right", self.mono_right)?;
        formatter.write_str(", ")?;
        write_stream(formatter, "depth", self.depth)
    }
}

#[derive(Debug)]
struct QualificationReport {
    connected_mxid: String,
    requested_maximum_usb_speed: UsbTransportSpeed,
    required_minimum_usb_speed: UsbTransportSpeed,
    observed_usb_speed: UsbTransportSpeed,
    frames_per_image_stream: u32,
    maximum_duration_seconds: u64,
    elapsed: Duration,
    required_minimum_imu_samples: u64,
    empty_queue_epoch: EmptyQueueEpochStats,
    stats: CaptureStats,
}

impl QualificationReport {
    fn targets_reached(&self) -> bool {
        self.stats
            .image_targets_reached(u64::from(self.frames_per_image_stream))
            && self.stats.imu_samples >= self.required_minimum_imu_samples
    }

    fn json<'a>(&'a self, status: &'a str) -> QualificationJson<'a> {
        QualificationJson {
            report: self,
            status,
        }
    }
}

struct QualificationJson<'a> {
    report: &'a QualificationReport,
    status: &'a str,
}

impl fmt::Display for QualificationJson<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let report = self.report;
        let elapsed_seconds = report.elapsed.as_secs_f64();
        write!(
            formatter,
            "{{\"schema_version\":1,\"status\":{},\"connected_mxid\":{},\
             \"usb_transport\":{{\"requested_maximum\":{},\"required_minimum\":{},\"observed\":{}}},\
             \"graph\":{{\"rgb\":{{\"width\":{RGB_WIDTH},\"height\":{RGB_HEIGHT},\"rate_hz\":{IMAGE_RATE_HZ}}},\
             \"mono\":{{\"width\":{RGB_WIDTH},\"height\":{RGB_HEIGHT},\"rate_hz\":{IMAGE_RATE_HZ},\"rectified\":true}},\
             \"depth\":{{\"width\":{RGB_WIDTH},\"height\":{RGB_HEIGHT},\"rate_hz\":{IMAGE_RATE_HZ},\"alignment\":\"RECTIFIED_LEFT\",\"unit\":\"millimetres_u16\",\"invalid_value\":0}},\
             \"imu\":{{\"rate_hz\":{IMU_RATE_HZ}}},\"queue\":{{\"size\":{QUEUE_SIZE},\"blocking\":false}}}},\
             \"limits\":{{\"frames_per_image_stream\":{},\"maximum_duration_seconds\":{}}},\
             \"empty_queue_epoch\":{{\"accepted\":{},\"candidate_epochs\":{},\
             \"elapsed_seconds\":{:.9},\"maximum_duration_milliseconds\":{},\
             \"maximum_candidate_epochs\":{EMPTY_EPOCH_MAXIMUM_CANDIDATES},\
             \"discarded\":{{\"rgb_frames\":{},\"mono_left_frames\":{},\
             \"mono_right_frames\":{},\"depth_frames\":{},\"imu_samples\":{}}},\
             \"measurement_start_semantics\":\"candidate monotonic instant recorded before one empty nonblocking check of every queue\"}},\
             \"elapsed_seconds\":{elapsed_seconds:.9},\"image_streams\":{{",
            JsonString(self.status),
            JsonString(&report.connected_mxid),
            JsonString(report.requested_maximum_usb_speed.as_depthai_name()),
            JsonString(report.required_minimum_usb_speed.as_depthai_name()),
            JsonString(report.observed_usb_speed.as_depthai_name()),
            report.frames_per_image_stream,
            report.maximum_duration_seconds,
            report.empty_queue_epoch.accepted,
            report.empty_queue_epoch.candidate_epochs,
            report.empty_queue_epoch.elapsed.as_secs_f64(),
            EMPTY_EPOCH_MAXIMUM_DURATION.as_millis(),
            report.empty_queue_epoch.discarded_rgb_frames,
            report.empty_queue_epoch.discarded_mono_left_frames,
            report.empty_queue_epoch.discarded_mono_right_frames,
            report.empty_queue_epoch.discarded_depth_frames,
            report.empty_queue_epoch.discarded_imu_samples,
        )?;
        write_image_stream(formatter, "rgb", report.stats.rgb, elapsed_seconds)?;
        formatter.write_str(",")?;
        write_image_stream(
            formatter,
            "mono_left",
            report.stats.mono_left,
            elapsed_seconds,
        )?;
        formatter.write_str(",")?;
        write_image_stream(
            formatter,
            "mono_right",
            report.stats.mono_right,
            elapsed_seconds,
        )?;
        formatter.write_str(",")?;
        write_image_stream(formatter, "depth", report.stats.depth, elapsed_seconds)?;
        let imu_rate = rate(report.stats.imu_samples, elapsed_seconds);
        write!(
            formatter,
            "}},\"imu\":{{\"requested_rate_hz\":{IMU_RATE_HZ},\
             \"minimum_delivery_rate_fraction\":{{\"numerator\":{MINIMUM_IMU_RATE_NUMERATOR},\"denominator\":{MINIMUM_IMU_RATE_DENOMINATOR}}},\
             \"required_minimum_samples_for_elapsed\":{},\
             \"delivered_samples\":{},\"empty_polls\":{},\
             \"measured_host_delivery_samples_per_second\":{imu_rate:.6}}},\
             \"interpretation\":{{\
             \"native_sequence_anomalies\":\"gaps prove missing deliveries and duplicates/regressions prove invalid adjacent native capture ordering; none attributes cause to USB\",\
             \"measured_rates\":\"host delivery only; not USB link capacity or sustained line-rate utilisation\"}}}}",
            report.required_minimum_imu_samples,
            report.stats.imu_samples,
            report.stats.imu_empty_polls,
        )
    }
}

fn write_image_stream(
    formatter: &mut fmt::Formatter<'_>,
    name: &str,
    stats: ImageStreamStats,
    elapsed_seconds: f64,
) -> fmt::Result {
    write!(
        formatter,
        "{}:{{\"delivered_frames\":{},\"delivered_payload_bytes\":{},\
         \"queue_empty_polls\":{},\"timeout_polls\":{},\
         \"native_sequence\":{{\"first\":",
        JsonString(name),
        stats.delivered,
        stats.payload_bytes,
        stats.queue_empty_polls,
        stats.timeout_polls,
    )?;
    write_optional_u64(formatter, stats.sequence.first)?;
    formatter.write_str(",\"last\":")?;
    write_optional_u64(formatter, stats.sequence.previous)?;
    write!(
        formatter,
        ",\"gaps\":{},\"duplicates\":{},\"regressions\":{}}},\
         \"measured_host_delivery_frames_per_second\":{:.6},\
         \"measured_host_delivery_payload_bytes_per_second\":{:.6}}}",
        stats.sequence.gaps,
        stats.sequence.duplicates,
        stats.sequence.regressions,
        rate(stats.delivered, elapsed_seconds),
        rate(stats.payload_bytes, elapsed_seconds),
    )
}

fn write_optional_u64(formatter: &mut fmt::Formatter<'_>, value: Option<u64>) -> fmt::Result {
    match value {
        Some(value) => write!(formatter, "{value}"),
        None => formatter.write_str("null"),
    }
}

fn rate(count: u64, elapsed_seconds: f64) -> f64 {
    if elapsed_seconds > 0.0 {
        count as f64 / elapsed_seconds
    } else {
        0.0
    }
}

struct JsonString<'a>(&'a str);

impl fmt::Display for JsonString<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("\"")?;
        for character in self.0.chars() {
            match character {
                '"' => formatter.write_str("\\\"")?,
                '\\' => formatter.write_str("\\\\")?,
                '\u{08}' => formatter.write_str("\\b")?,
                '\u{0c}' => formatter.write_str("\\f")?,
                '\n' => formatter.write_str("\\n")?,
                '\r' => formatter.write_str("\\r")?,
                '\t' => formatter.write_str("\\t")?,
                character if character <= '\u{1f}' => {
                    write!(formatter, "\\u{:04x}", u32::from(character))?;
                }
                character => formatter.write_str(character.encode_utf8(&mut [0; 4]))?,
            }
        }
        formatter.write_str("\"")
    }
}

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
enum ImageContractError {
    #[error("{name} returned stream {actual:?}, expected {expected:?}")]
    UnexpectedStream {
        name: &'static str,
        expected: StreamId,
        actual: StreamId,
    },
    #[error(
        "{name} delivered layout {actual_width}x{actual_height}, stride={actual_stride_bytes} bytes, payload={actual_payload_bytes} bytes; expected {expected_width}x{expected_height}, stride={expected_stride_bytes} bytes, payload={expected_payload_bytes} bytes"
    )]
    LayoutMismatch {
        name: &'static str,
        expected_width: u32,
        expected_height: u32,
        expected_stride_bytes: u32,
        expected_payload_bytes: u64,
        actual_width: u32,
        actual_height: u32,
        actual_stride_bytes: u32,
        actual_payload_bytes: u64,
    },
}

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
enum DepthContractError {
    #[error(
        "depth delivered layout {actual_width}x{actual_height}, packed_stride={actual_packed_stride_bytes:?} bytes, payload={actual_payload_bytes} bytes, alignment={actual_alignment:?}; expected {expected_width}x{expected_height}, packed_stride={expected_packed_stride_bytes} bytes, payload={expected_payload_bytes} bytes, alignment={expected_alignment:?}"
    )]
    LayoutOrAlignmentMismatch {
        expected_width: u32,
        expected_height: u32,
        expected_packed_stride_bytes: u32,
        expected_payload_bytes: u64,
        expected_alignment: DepthAlignment,
        actual_width: u32,
        actual_height: u32,
        actual_packed_stride_bytes: Option<u32>,
        actual_payload_bytes: u64,
        actual_alignment: Option<DepthAlignment>,
    },
}

#[derive(Error, Debug)]
enum CaptureImageError {
    #[error("{0}")]
    Acquisition(#[source] ImageError),
    #[error("{0}")]
    Contract(#[from] ImageContractError),
    #[error("{0}")]
    Accounting(#[from] AccountingError),
}

#[derive(Error, Debug)]
enum CaptureDepthError {
    #[error("{0}")]
    Acquisition(#[source] DepthError),
    #[error("{0}")]
    Contract(#[from] DepthContractError),
    #[error("{0}")]
    Accounting(#[from] AccountingError),
}

#[derive(Error, Debug)]
enum EmptyQueueEpochError {
    #[error("RGB drain check failed: {0}")]
    Rgb(#[source] CaptureImageError),
    #[error("rectified-left mono drain check failed: {0}")]
    MonoLeft(#[source] CaptureImageError),
    #[error("rectified-right mono drain check failed: {0}")]
    MonoRight(#[source] CaptureImageError),
    #[error("rectified-left-aligned depth drain check failed: {0}")]
    Depth(#[source] CaptureDepthError),
    #[error("IMU drain check failed: {0}")]
    Imu(#[source] ImuError),
    #[error("IMU returned a successful but empty batch during the drain check")]
    EmptySuccessfulImuBatch,
    #[error("no all-empty queue epoch was established within {maximum_milliseconds} milliseconds")]
    MaximumDuration { maximum_milliseconds: u128 },
    #[error("no all-empty queue epoch was established within {maximum} candidate epochs")]
    MaximumCandidates { maximum: u64 },
    #[error("empty-queue epoch accounting failed: {0}")]
    Accounting(#[source] AccountingError),
}

#[derive(Error, Debug)]
enum CaptureError {
    #[error("bounded empty-queue epoch establishment failed: {0}")]
    EmptyQueueEpoch(#[source] Box<EmptyQueueEpochError>),
    #[error("RGB stream failed: {0}")]
    Rgb(#[source] CaptureImageError),
    #[error("rectified-left mono stream failed: {0}")]
    MonoLeft(#[source] CaptureImageError),
    #[error("rectified-right mono stream failed: {0}")]
    MonoRight(#[source] CaptureImageError),
    #[error("rectified-left-aligned depth stream failed: {0}")]
    Depth(#[source] CaptureDepthError),
    #[error("IMU stream failed: {0}")]
    Imu(#[source] ImuError),
    #[error("qualification accounting failed: {0}")]
    Accounting(#[source] AccountingError),
}

impl CaptureError {
    const fn status(&self) -> &'static str {
        match self {
            Self::EmptyQueueEpoch(_) => "empty_queue_epoch_error",
            Self::Rgb(_)
            | Self::MonoLeft(_)
            | Self::MonoRight(_)
            | Self::Depth(_)
            | Self::Imu(_)
            | Self::Accounting(_) => "capture_error",
        }
    }
}

#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
enum AccountingError {
    #[error("counter '{counter}' overflowed")]
    CounterOverflow { counter: &'static str },
}

#[derive(Error, Debug)]
enum OperationError {
    #[error("connected-device identity readback failed: {0}")]
    Identity(#[source] ConnectedDeviceIdentityError),
    #[error("USB transport evidence readback failed: {0}")]
    UsbTransport(#[source] UsbTransportEvidenceError),
    #[error("{0}")]
    Capture(#[source] CaptureError),
    #[error(
        "duration limit reached before all targets: image target={target}, rgb={rgb}, mono_left={mono_left}, mono_right={mono_right}, depth={depth}, IMU delivered={imu_delivered}, IMU required={imu_required}"
    )]
    DurationLimit {
        target: u32,
        rgb: u64,
        mono_left: u64,
        mono_right: u64,
        depth: u64,
        imu_delivered: u64,
        imu_required: u64,
    },
    #[error(
        "native image sequence anomalies observed; successful qualification requires zero: {evidence}"
    )]
    NativeSequenceAnomalies { evidence: SequenceAnomalyEvidence },
}

impl OperationError {
    fn status(&self) -> &'static str {
        match self {
            Self::DurationLimit { .. } => "duration_limit",
            Self::Capture(source) => source.status(),
            Self::NativeSequenceAnomalies { .. } => "native_sequence_anomaly",
            Self::Identity(_) | Self::UsbTransport(_) => "evidence_error",
        }
    }
}

#[derive(Error, Debug)]
#[error("{source}")]
struct OperationFailure {
    #[source]
    source: Box<OperationError>,
    report: Option<Box<QualificationReport>>,
}

impl OperationFailure {
    fn without_report(source: OperationError) -> Self {
        Self {
            source: Box::new(source),
            report: None,
        }
    }

    fn with_report(source: OperationError, report: QualificationReport) -> Self {
        Self {
            source: Box::new(source),
            report: Some(Box::new(report)),
        }
    }
}

#[derive(Error, Debug)]
enum ProgramError {
    #[error("failed to connect the exact OAK MXID: {0}")]
    Connect(#[source] oak_sys::ConnectionError),
    #[error("qualification operation failed: {operation}")]
    Operation {
        #[source]
        operation: OperationFailure,
    },
    #[error("qualification operation completed but explicit device close failed: {close}")]
    Close {
        report: Box<QualificationReport>,
        #[source]
        close: CloseError,
    },
    #[error(
        "qualification operation failed ({operation}); explicit device close also failed ({close})"
    )]
    OperationAndClose {
        #[source]
        operation: OperationFailure,
        close: CloseError,
    },
}

impl ProgramError {
    fn report(&self) -> Option<&QualificationReport> {
        match self {
            Self::Connect(_) => None,
            Self::Operation { operation } | Self::OperationAndClose { operation, .. } => {
                operation.report.as_deref()
            }
            Self::Close { report, .. } => Some(report),
        }
    }

    fn report_status(&self) -> &str {
        match self {
            Self::Connect(_) => "connect_error",
            Self::Operation { operation } => operation.source.status(),
            Self::Close { .. } => "close_error",
            Self::OperationAndClose { operation, .. } => match operation.source.status() {
                "duration_limit" => "duration_limit_and_close_error",
                "capture_error" => "capture_and_close_error",
                "empty_queue_epoch_error" => "empty_queue_epoch_and_close_error",
                "native_sequence_anomaly" => "native_sequence_anomaly_and_close_error",
                _ => "evidence_and_close_error",
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn parse(values: &[&str]) -> Result<Command, ArgumentError> {
        Command::parse(values.iter().map(OsString::from))
    }

    fn image_stats(
        delivered: u64,
        payload_bytes_per_frame: u64,
        first_sequence: u64,
    ) -> ImageStreamStats {
        ImageStreamStats {
            delivered,
            payload_bytes: delivered * payload_bytes_per_frame,
            queue_empty_polls: 3,
            timeout_polls: 0,
            sequence: if delivered == 0 {
                SequenceAccounting::default()
            } else {
                SequenceAccounting {
                    first: Some(first_sequence),
                    previous: Some(first_sequence + delivered - 1),
                    gaps: 0,
                    duplicates: 0,
                    regressions: 0,
                }
            },
        }
    }

    fn complete_report(mxid: &str) -> QualificationReport {
        QualificationReport {
            connected_mxid: mxid.to_owned(),
            requested_maximum_usb_speed: UsbTransportSpeed::SuperPlus,
            required_minimum_usb_speed: UsbTransportSpeed::Super,
            observed_usb_speed: UsbTransportSpeed::SuperPlus,
            frames_per_image_stream: 2,
            maximum_duration_seconds: 10,
            elapsed: Duration::from_secs(2),
            required_minimum_imu_samples: 320,
            empty_queue_epoch: EmptyQueueEpochStats {
                candidate_epochs: 3,
                discarded_rgb_frames: 1,
                discarded_mono_left_frames: 1,
                discarded_mono_right_frames: 1,
                discarded_depth_frames: 1,
                discarded_imu_samples: 12,
                elapsed: Duration::from_millis(4),
                accepted: true,
            },
            stats: CaptureStats {
                rgb: image_stats(2, RGB_PAYLOAD_BYTES, 10),
                mono_left: image_stats(2, MONO_PAYLOAD_BYTES, 20),
                mono_right: image_stats(2, MONO_PAYLOAD_BYTES, 30),
                depth: image_stats(2, DEPTH_PAYLOAD_BYTES, 40),
                imu_samples: 400,
                imu_empty_polls: 4,
            },
        }
    }

    #[test]
    fn arguments_parse_once_into_bounded_domain_values() {
        let command = parse(&["qualify", "19443010F1B43A2E00", "SUPER_PLUS", "150", "30"])
            .expect("valid command");
        assert_eq!(
            command,
            Command::Run(Arguments {
                mxid: "19443010F1B43A2E00".to_owned(),
                maximum_usb_speed: QualificationUsbMaximum::SuperPlus,
                frames_per_image_stream: 150,
                maximum_duration: Duration::from_secs(30),
                maximum_duration_seconds: 30,
            })
        );
        assert!(matches!(
            parse(&["qualify", "", "SUPER", "1", "1"]),
            Err(ArgumentError::EmptyMxid)
        ));
        assert!(matches!(
            parse(&["qualify", "mxid", "SUPER", "0", "1"]),
            Err(ArgumentError::OutsideBounds {
                field: "FRAMES_PER_IMAGE_STREAM",
                ..
            })
        ));
        assert!(matches!(
            parse(&["qualify", "mxid", "SUPER", "1", "3601"]),
            Err(ArgumentError::OutsideBounds {
                field: "MAX_DURATION_SECONDS",
                ..
            })
        ));
        assert!(matches!(
            parse(&["qualify", "mxid", "SUPER", "one", "1"]),
            Err(ArgumentError::InvalidInteger {
                field: "FRAMES_PER_IMAGE_STREAM",
                ..
            })
        ));
        assert!(matches!(
            parse(&["qualify", "mxid", "SUPER", "1"]),
            Err(ArgumentError::WrongArity { actual: 3, .. })
        ));
        for unsupported in ["HIGH", "super", "SUPER_PLUS "] {
            assert!(matches!(
                parse(&["qualify", "mxid", unsupported, "1", "1"]),
                Err(ArgumentError::UnsupportedMaximumUsbSpeed { .. })
            ));
        }
        let capped = parse(&["qualify", "mxid", "SUPER", "1", "1"])
            .expect("explicit USB 3 5 Gbit/s maximum parses");
        assert!(matches!(
            capped,
            Command::Run(Arguments {
                maximum_usb_speed: QualificationUsbMaximum::Super,
                ..
            })
        ));
        assert!(matches!(
            parse(&["qualify", "--help"]),
            Ok(Command::Help { .. })
        ));
    }

    #[test]
    fn native_sequence_accounting_distinguishes_adjacent_anomalies() {
        let mut accounting = SequenceAccounting::default();
        for sequence in [10, 11, 14, 14, 9, 10] {
            accounting.observe(sequence).expect("bounded test counts");
        }
        assert_eq!(
            accounting,
            SequenceAccounting {
                first: Some(10),
                previous: Some(10),
                gaps: 2,
                duplicates: 1,
                regressions: 1,
            }
        );
    }

    #[test]
    fn minimum_imu_delivery_is_exactly_four_fifths_of_elapsed_200_hz() {
        assert_eq!(minimum_imu_samples_for_elapsed(Duration::ZERO), 0);
        assert_eq!(minimum_imu_samples_for_elapsed(Duration::from_nanos(1)), 1);
        assert_eq!(minimum_imu_samples_for_elapsed(Duration::from_secs(1)), 160);
        assert_eq!(
            minimum_imu_samples_for_elapsed(Duration::from_millis(1_500)),
            240
        );
    }

    #[test]
    fn empty_epoch_requires_every_nonblocking_check_to_report_empty() {
        assert!(candidate_is_empty_epoch([false; 5]));
        for delivered_index in 0..5 {
            let mut delivered = [false; 5];
            delivered[delivered_index] = true;
            assert!(
                !candidate_is_empty_epoch(delivered),
                "queue {delivered_index} delivered data"
            );
        }
    }

    #[test]
    fn complete_json_round_trips_and_escapes_exact_mxid() {
        let mxid = "mxid\"with\\escapes\nand\u{1}control";
        let report = complete_report(mxid);
        assert!(report.targets_reached());
        assert!(report.stats.sequence_anomalies().is_none());

        let rendered = report.json("complete").to_string();
        let parsed: Value = serde_json::from_str(&rendered).expect("complete report is valid JSON");
        assert_eq!(parsed["status"], "complete");
        assert_eq!(parsed["connected_mxid"], mxid);
        assert_eq!(parsed["usb_transport"]["requested_maximum"], "SUPER_PLUS");
        assert_eq!(parsed["usb_transport"]["required_minimum"], "SUPER");
        assert_eq!(parsed["usb_transport"]["observed"], "SUPER_PLUS");
        assert_eq!(parsed["empty_queue_epoch"]["accepted"], true);
        assert_eq!(parsed["empty_queue_epoch"]["candidate_epochs"], 3);
        assert_eq!(parsed["empty_queue_epoch"]["discarded"]["imu_samples"], 12);
        assert_eq!(
            parsed["imu"]["minimum_delivery_rate_fraction"]["numerator"],
            4
        );
        assert_eq!(
            parsed["imu"]["minimum_delivery_rate_fraction"]["denominator"],
            5
        );
        assert_eq!(parsed["imu"]["required_minimum_samples_for_elapsed"], 320);
        assert_eq!(parsed["imu"]["delivered_samples"], 400);
        assert_eq!(
            parsed["image_streams"]["rgb"]["delivered_payload_bytes"],
            2 * RGB_PAYLOAD_BYTES
        );
    }

    #[test]
    fn partial_json_round_trips_with_required_and_delivered_counts() {
        let mut report = complete_report("partial-mxid");
        report.elapsed = Duration::from_secs(1);
        report.required_minimum_imu_samples = 160;
        report.stats.imu_samples = 100;
        report.stats.depth = ImageStreamStats::default();
        assert!(!report.targets_reached());

        let rendered = report.json("duration_limit").to_string();
        let parsed: Value = serde_json::from_str(&rendered).expect("partial report is valid JSON");
        assert_eq!(parsed["status"], "duration_limit");
        assert_eq!(parsed["imu"]["required_minimum_samples_for_elapsed"], 160);
        assert_eq!(parsed["imu"]["delivered_samples"], 100);
        assert_eq!(
            parsed["image_streams"]["depth"]["native_sequence"]["first"],
            Value::Null
        );
        assert_eq!(
            parsed["image_streams"]["depth"]["native_sequence"]["last"],
            Value::Null
        );
    }

    #[test]
    fn any_native_sequence_anomaly_is_disqualifying_evidence() {
        let mut stats = complete_report("mxid").stats;
        assert!(stats.sequence_anomalies().is_none());
        stats.mono_right.sequence.gaps = 1;
        let evidence = stats
            .sequence_anomalies()
            .expect("one observed gap is disqualifying");
        assert!(evidence.has_any());
        assert_eq!(evidence.mono_right.gaps, 1);
    }

    #[test]
    fn canonical_graph_is_fixed_and_nonblocking() {
        for (maximum, expected_policy) in [
            (
                QualificationUsbMaximum::Super,
                UsbTransportPolicy::try_new(UsbTransportSpeed::Super, UsbTransportSpeed::Super)
                    .expect("ordered capped USB 3 policy"),
            ),
            (
                QualificationUsbMaximum::SuperPlus,
                UsbTransportPolicy::try_new(UsbTransportSpeed::SuperPlus, UsbTransportSpeed::Super)
                    .expect("ordered explicit 10 Gbit/s diagnostic policy"),
            ),
        ] {
            let config = canonical_config(maximum);
            assert_eq!(config.usb_transport, expected_policy);
            let rgb = config.rgb.expect("RGB enabled");
            assert_eq!((rgb.width, rgb.height, rgb.fps), (640, 400, 15));
            let mono = config.mono.expect("mono enabled");
            assert_eq!(
                (mono.width, mono.height, mono.fps, mono.rectified),
                (640, 400, 15, true)
            );
            let depth = config.depth.expect("depth enabled");
            assert_eq!(
                (depth.width, depth.height, depth.fps, depth.alignment),
                (640, 400, 15, DepthAlignment::RectifiedLeft)
            );
            assert_eq!(config.imu.expect("IMU enabled").rate_hz, 200);
            assert_eq!((config.queue.size, config.queue.blocking), (4, false));
            config.validate().expect("canonical graph is valid");
        }
    }
}
