//! Camera-only qualification of Kiko's canonical Jetson Orin Nano OAK stream
//! graph.
//!
//! Usage:
//! `oak_stream_qualify EXACT_MXID FRAMES_PER_IMAGE_STREAM MAX_DURATION_SECONDS`
//!
//! Successful stdout is one JSON document. Native sequence gaps describe
//! missing host deliveries only; they do not attribute loss to USB.

use oak_sys::{
    CloseError, ConnectedDeviceIdentityError, DepthAlignment, DepthConfig, DepthError, Device,
    DeviceConfig, DeviceFrameSequence, ImageError, ImageFrame, ImuConfig, ImuError, MonoConfig,
    QueueConfig, RgbConfig, UsbTransportAdmissionEvidence, UsbTransportEvidenceError,
    UsbTransportPolicy,
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
const QUEUE_SIZE: u32 = 4;
const MAXIMUM_FRAMES_PER_STREAM: u32 = IMAGE_RATE_HZ * 60 * 60;
const MAXIMUM_DURATION_SECONDS: u64 = 60 * 60;
const IDLE_POLL_SLEEP: Duration = Duration::from_millis(1);

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
        if values.len() != 3 {
            return Err(ArgumentError::WrongArity {
                usage: usage(&program),
                actual: values.len(),
            });
        }

        let mxid = values[0].clone();
        if mxid.is_empty() {
            return Err(ArgumentError::EmptyMxid);
        }
        let frames_per_image_stream = parse_bounded_u32(
            &values[1],
            "FRAMES_PER_IMAGE_STREAM",
            MAXIMUM_FRAMES_PER_STREAM,
        )?;
        let maximum_duration_seconds =
            parse_bounded_u64(&values[2], "MAX_DURATION_SECONDS", MAXIMUM_DURATION_SECONDS)?;
        Ok(Self::Run(Arguments {
            mxid,
            frames_per_image_stream,
            maximum_duration: Duration::from_secs(maximum_duration_seconds),
            maximum_duration_seconds,
        }))
    }
}

fn usage(program: &str) -> String {
    format!(
        "usage: {program} EXACT_MXID FRAMES_PER_IMAGE_STREAM MAX_DURATION_SECONDS\n\
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
    frames_per_image_stream: u32,
    maximum_duration: Duration,
    maximum_duration_seconds: u64,
}

#[derive(Error, Debug)]
enum ArgumentError {
    #[error("argument {position} is not valid Unicode")]
    NonUnicode { position: usize },
    #[error("{usage}; received {actual} positional arguments")]
    WrongArity { usage: String, actual: usize },
    #[error("EXACT_MXID must be nonempty")]
    EmptyMxid,
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

fn canonical_config() -> DeviceConfig {
    DeviceConfig {
        usb_transport: UsbTransportPolicy::super_speed_required(),
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
    let mut device =
        Device::connect(&arguments.mxid, canonical_config()).map_err(ProgramError::Connect)?;
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
        usb_transport,
        frames_per_image_stream: arguments.frames_per_image_stream,
        maximum_duration_seconds: arguments.maximum_duration_seconds,
        elapsed: attempt.elapsed,
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
                imu: report.stats.imu_samples,
            },
            report,
        ));
    }
    Ok(report)
}

#[derive(Debug)]
struct CaptureAttempt {
    stats: CaptureStats,
    elapsed: Duration,
    failure: Option<CaptureError>,
}

fn capture(device: &mut Device, arguments: &Arguments) -> CaptureAttempt {
    let started = Instant::now();
    let mut stats = CaptureStats::default();
    let target = u64::from(arguments.frames_per_image_stream);

    loop {
        if stats.image_targets_reached(target) && stats.imu_samples > 0 {
            return CaptureAttempt {
                stats,
                elapsed: started.elapsed(),
                failure: None,
            };
        }
        let elapsed = started.elapsed();
        if elapsed >= arguments.maximum_duration {
            return CaptureAttempt {
                stats,
                elapsed,
                failure: None,
            };
        }

        let mut delivered_anything = false;
        if stats.rgb.delivered < target {
            match poll_image(device.rgb(0), &mut stats.rgb) {
                Ok(delivered) => delivered_anything |= delivered,
                Err(source) => {
                    return failed_attempt(stats, started.elapsed(), CaptureError::Rgb(source));
                }
            }
        }
        if stats.mono_left.delivered < target {
            match poll_image(device.mono_left(0), &mut stats.mono_left) {
                Ok(delivered) => delivered_anything |= delivered,
                Err(source) => {
                    return failed_attempt(
                        stats,
                        started.elapsed(),
                        CaptureError::MonoLeft(source),
                    );
                }
            }
        }
        if stats.mono_right.delivered < target {
            match poll_image(device.mono_right(0), &mut stats.mono_right) {
                Ok(delivered) => delivered_anything |= delivered,
                Err(source) => {
                    return failed_attempt(
                        stats,
                        started.elapsed(),
                        CaptureError::MonoRight(source),
                    );
                }
            }
        }
        if stats.depth.delivered < target {
            match poll_depth(device.depth(0), &mut stats.depth) {
                Ok(delivered) => delivered_anything |= delivered,
                Err(source) => {
                    return failed_attempt(stats, started.elapsed(), CaptureError::Depth(source));
                }
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
                        CaptureError::Accounting(AccountingError::CounterOverflow {
                            counter: "imu.empty_polls",
                        }),
                    );
                }
            }
            Err(source) => {
                return failed_attempt(stats, started.elapsed(), CaptureError::Imu(source));
            }
        }

        if !delivered_anything {
            let remaining = arguments.maximum_duration.saturating_sub(started.elapsed());
            std::thread::sleep(IDLE_POLL_SLEEP.min(remaining));
        }
    }
}

fn failed_attempt(stats: CaptureStats, elapsed: Duration, failure: CaptureError) -> CaptureAttempt {
    CaptureAttempt {
        stats,
        elapsed,
        failure: Some(failure),
    }
}

fn poll_image(
    result: Result<ImageFrame, ImageError>,
    stats: &mut ImageStreamStats,
) -> Result<bool, CaptureImageError> {
    match result {
        Ok(frame) => {
            stats.record(
                frame.device_capture_sequence,
                u64::try_from(frame.pixels().len()).map_err(|_| {
                    AccountingError::CounterOverflow {
                        counter: "image.payload_bytes",
                    }
                })?,
            )?;
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
}

#[derive(Debug)]
struct QualificationReport {
    connected_mxid: String,
    usb_transport: UsbTransportAdmissionEvidence,
    frames_per_image_stream: u32,
    maximum_duration_seconds: u64,
    elapsed: Duration,
    stats: CaptureStats,
}

impl QualificationReport {
    fn targets_reached(&self) -> bool {
        self.stats
            .image_targets_reached(u64::from(self.frames_per_image_stream))
            && self.stats.imu_samples > 0
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
             \"elapsed_seconds\":{elapsed_seconds:.9},\"image_streams\":{{",
            JsonString(self.status),
            JsonString(&report.connected_mxid),
            JsonString(
                report
                    .usb_transport
                    .requested_maximum()
                    .as_depthai_name()
            ),
            JsonString(
                report
                    .usb_transport
                    .required_minimum()
                    .as_depthai_name()
            ),
            JsonString(report.usb_transport.observed().as_depthai_name()),
            report.frames_per_image_stream,
            report.maximum_duration_seconds,
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
            "}},\"imu\":{{\"delivered_samples\":{},\"empty_polls\":{},\
             \"measured_host_delivery_samples_per_second\":{imu_rate:.6}}},\
             \"interpretation\":{{\
             \"native_sequence_gaps\":\"missing host deliveries; no USB-cause attribution\",\
             \"measured_rates\":\"host delivery only; not USB link capacity or sustained line-rate utilisation\"}}}}",
            report.stats.imu_samples, report.stats.imu_empty_polls,
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

#[derive(Error, Debug)]
enum CaptureImageError {
    #[error("{0}")]
    Acquisition(#[source] ImageError),
    #[error("{0}")]
    Accounting(#[from] AccountingError),
}

#[derive(Error, Debug)]
enum CaptureDepthError {
    #[error("{0}")]
    Acquisition(#[source] DepthError),
    #[error("{0}")]
    Accounting(#[from] AccountingError),
}

#[derive(Error, Debug)]
enum CaptureError {
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
        "duration limit reached before all targets: target={target}, rgb={rgb}, mono_left={mono_left}, mono_right={mono_right}, depth={depth}, imu_samples={imu}"
    )]
    DurationLimit {
        target: u32,
        rgb: u64,
        mono_left: u64,
        mono_right: u64,
        depth: u64,
        imu: u64,
    },
}

impl OperationError {
    fn status(&self) -> &'static str {
        match self {
            Self::DurationLimit { .. } => "duration_limit",
            Self::Capture(_) => "capture_error",
            Self::Identity(_) | Self::UsbTransport(_) => "evidence_error",
        }
    }
}

#[derive(Error, Debug)]
#[error("{source}")]
struct OperationFailure {
    #[source]
    source: OperationError,
    report: Option<Box<QualificationReport>>,
}

impl OperationFailure {
    fn without_report(source: OperationError) -> Self {
        Self {
            source,
            report: None,
        }
    }

    fn with_report(source: OperationError, report: QualificationReport) -> Self {
        Self {
            source,
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
                _ => "evidence_and_close_error",
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(values: &[&str]) -> Result<Command, ArgumentError> {
        Command::parse(values.iter().map(OsString::from))
    }

    #[test]
    fn arguments_parse_once_into_bounded_domain_values() {
        let command =
            parse(&["qualify", "19443010F1B43A2E00", "150", "30"]).expect("valid command");
        assert_eq!(
            command,
            Command::Run(Arguments {
                mxid: "19443010F1B43A2E00".to_owned(),
                frames_per_image_stream: 150,
                maximum_duration: Duration::from_secs(30),
                maximum_duration_seconds: 30,
            })
        );
        assert!(matches!(
            parse(&["qualify", "", "1", "1"]),
            Err(ArgumentError::EmptyMxid)
        ));
        assert!(matches!(
            parse(&["qualify", "mxid", "0", "1"]),
            Err(ArgumentError::OutsideBounds {
                field: "FRAMES_PER_IMAGE_STREAM",
                ..
            })
        ));
        assert!(matches!(
            parse(&["qualify", "mxid", "1", "3601"]),
            Err(ArgumentError::OutsideBounds {
                field: "MAX_DURATION_SECONDS",
                ..
            })
        ));
        assert!(matches!(
            parse(&["qualify", "mxid", "one", "1"]),
            Err(ArgumentError::InvalidInteger {
                field: "FRAMES_PER_IMAGE_STREAM",
                ..
            })
        ));
        assert!(matches!(
            parse(&["qualify", "mxid", "1"]),
            Err(ArgumentError::WrongArity { actual: 2, .. })
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
    fn canonical_graph_is_fixed_and_nonblocking() {
        let config = canonical_config();
        assert_eq!(
            config.usb_transport,
            UsbTransportPolicy::super_speed_required()
        );
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
