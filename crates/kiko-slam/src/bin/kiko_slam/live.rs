use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use clap::Args;

use kiko_slam::env::{try_env_bool, try_env_u32, try_env_usize};
use kiko_slam::{
    CalibrationBundle, CaptureBundle, CaptureId, CaptureImu, CaptureInterval, ChannelCapacity,
    DepthImage, DiagnosticEvent, DropPolicy, DropReceiver, Frame, FrameDiagnostics, FrameId,
    ImuAccumulator, ImuBatch, PairingDropReason, PairingOutcome, Point3, PoseStatus, Raw,
    RerunSink, SendOutcome, SensorId, SlamTracker, StereoPairer, SystemHealth, VizPacket,
    bounded_channel, oak_to_depth_image, oak_to_frame, oak_to_imu_batch,
};
use kiko_slam::{PinholeIntrinsics, RectifiedStereo};
use oak_sys::{
    DepthConfig, DepthError, DeviceConfig, ImageError, ImuConfig, ImuError, MonoConfig, QueueConfig,
};

use crate::args::{CameraArgs, InferenceArgs, InferenceConfig, InferencePurpose, RerunArgs};
use crate::config::{TrackerDefaults, build_tracker_config};
use crate::record::{
    build_calibration, load_oak_read_timeout_ms, load_pairer_max_pending_per_side,
    load_pairing_window,
};
use crate::rerun_recording;

#[derive(Args, Clone, Debug)]
#[command(about = "Run live SLAM from OAK-D camera")]
pub struct LiveArgs {
    #[command(flatten)]
    pub camera: CameraArgs,
    #[command(flatten)]
    pub inference: InferenceArgs,
    #[command(flatten)]
    pub rerun: RerunArgs,
}

struct LiveVizMsg {
    left: Frame,
    right: Frame,
    depth: Option<DepthImage>,
    imu: Option<ImuBatch>,
    pose: PoseStatus,
    vio_telemetry: Option<kiko_slam::VioTelemetry>,
    packet: Option<VizPacket<Raw>>,
    points: Option<Vec<Point3>>,
    covisibility_snapshot: Option<kiko_slam::CovisibilitySnapshot>,
    health: SystemHealth,
    diagnostics: FrameDiagnostics,
    events: Vec<DiagnosticEvent>,
}

#[derive(Debug)]
enum LiveThreadError {
    TrackerInit {
        source: kiko_slam::TrackerInitError,
    },
    VizInit {
        source: kiko_slam::RerunSinkInitError,
    },
    VizChannelDisconnected,
    Tracker {
        source: kiko_slam::TrackerError,
    },
    VizPacket {
        source: kiko_slam::VizError,
    },
    FrameProcessingPanic {
        detail: String,
    },
}

impl std::fmt::Display for LiveThreadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiveThreadError::TrackerInit { source } => {
                write!(f, "failed to initialize tracker: {source}")
            }
            LiveThreadError::VizInit { source } => {
                write!(f, "failed to initialize visualization: {source}")
            }
            LiveThreadError::VizChannelDisconnected => write!(f, "viz channel disconnected"),
            LiveThreadError::Tracker { source } => {
                write!(f, "tracker frame processing failed: {source}")
            }
            LiveThreadError::VizPacket { source } => {
                write!(f, "visualization packet construction failed: {source}")
            }
            LiveThreadError::FrameProcessingPanic { detail } => {
                write!(f, "inference panic while processing frame: {detail}")
            }
        }
    }
}

impl std::error::Error for LiveThreadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LiveThreadError::TrackerInit { source } => Some(source),
            LiveThreadError::VizInit { source } => Some(source),
            LiveThreadError::Tracker { source } => Some(source),
            LiveThreadError::VizPacket { source } => Some(source),
            LiveThreadError::VizChannelDisconnected
            | LiveThreadError::FrameProcessingPanic { .. } => None,
        }
    }
}

#[derive(Debug)]
enum LiveCaptureError {
    Image {
        sensor: SensorId,
        source: ImageError,
    },
    Frame {
        sensor: SensorId,
        source: kiko_slam::FrameError,
    },
    Depth {
        source: DepthError,
    },
    DepthImage {
        source: kiko_slam::DepthImageError,
    },
    Imu {
        source: ImuError,
    },
    OakImu {
        source: kiko_slam::OakImuError,
    },
    ImuTimestampShift {
        source: kiko_slam::ImuTimestampShiftError,
    },
    Pairing {
        source: kiko_slam::PairError,
    },
    ImuBatch {
        source: kiko_slam::ImuBatchError,
    },
    ImuAccumulator {
        source: kiko_slam::ImuAccumulatorError,
    },
    CaptureInterval {
        source: kiko_slam::CaptureIntervalError,
    },
    CaptureBundle {
        source: kiko_slam::CaptureBundleError,
    },
    DepthConsumerDisconnected,
    InferenceConsumerDisconnected,
}

impl std::fmt::Display for LiveCaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Image { sensor, source } => {
                write!(f, "{sensor:?} image stream failed: {source}")
            }
            Self::Frame { sensor, source } => {
                write!(f, "{sensor:?} frame conversion failed: {source}")
            }
            Self::Depth { source } => write!(f, "depth stream failed: {source}"),
            Self::DepthImage { source } => write!(f, "depth frame conversion failed: {source}"),
            Self::Imu { source } => write!(f, "IMU stream failed: {source}"),
            Self::OakImu { source } => write!(f, "IMU sample conversion failed: {source}"),
            Self::ImuTimestampShift { source } => {
                write!(f, "IMU timestamp calibration failed: {source}")
            }
            Self::Pairing { source } => write!(f, "stereo pairing failed: {source}"),
            Self::ImuBatch { source } => write!(f, "capture IMU batch is invalid: {source}"),
            Self::ImuAccumulator { source } => {
                write!(f, "capture IMU stream ordering is invalid: {source}")
            }
            Self::CaptureInterval { source } => {
                write!(f, "capture interval is invalid: {source}")
            }
            Self::CaptureBundle { source } => write!(f, "capture bundle is invalid: {source}"),
            Self::DepthConsumerDisconnected => write!(f, "depth consumer disconnected"),
            Self::InferenceConsumerDisconnected => write!(f, "inference consumer disconnected"),
        }
    }
}

impl std::error::Error for LiveCaptureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Image { source, .. } => Some(source),
            Self::Frame { source, .. } => Some(source),
            Self::Depth { source } => Some(source),
            Self::DepthImage { source } => Some(source),
            Self::Imu { source } => Some(source),
            Self::OakImu { source } => Some(source),
            Self::ImuTimestampShift { source } => Some(source),
            Self::Pairing { source } => Some(source),
            Self::ImuBatch { source } => Some(source),
            Self::ImuAccumulator { source } => Some(source),
            Self::CaptureInterval { source } => Some(source),
            Self::CaptureBundle { source } => Some(source),
            Self::DepthConsumerDisconnected | Self::InferenceConsumerDisconnected => None,
        }
    }
}

#[derive(Debug)]
enum LiveWorkerError {
    Failed {
        worker: &'static str,
        source: LiveThreadError,
    },
    Panicked {
        worker: &'static str,
        detail: String,
    },
}

impl std::fmt::Display for LiveWorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed { worker, source } => write!(f, "{worker} worker failed: {source}"),
            Self::Panicked { worker, detail } => write!(f, "{worker} worker panicked: {detail}"),
        }
    }
}

impl std::error::Error for LiveWorkerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Failed { source, .. } => Some(source),
            Self::Panicked { .. } => None,
        }
    }
}

fn join_live_worker(
    worker: &'static str,
    handle: thread::JoinHandle<Result<(), LiveThreadError>>,
) -> Result<(), LiveWorkerError> {
    match handle.join() {
        Ok(Ok(())) => Ok(()),
        Ok(Err(source)) => Err(LiveWorkerError::Failed { worker, source }),
        Err(payload) => Err(LiveWorkerError::Panicked {
            worker,
            detail: kiko_slam::panic_payload_to_string(payload.as_ref()),
        }),
    }
}

fn drain_latest_depth(rx: &DropReceiver<DepthImage>) -> Option<DepthImage> {
    let mut latest = None;
    while let Ok(depth) = rx.try_recv() {
        latest = Some(depth);
    }
    latest
}

pub fn run_live(args: &LiveArgs) -> Result<(), Box<dyn std::error::Error>> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nreceived ctrl+c, stopping...");
        r.store(false, Ordering::SeqCst);
    })?;

    let mono_config = MonoConfig {
        width: args.camera.width,
        height: args.camera.height,
        fps: args.camera.fps,
        rectified: args.camera.rectified(),
    };
    let depth_enabled = try_env_bool("KIKO_LIVE_DEPTH")?.unwrap_or(false);
    let imu_enabled = try_env_bool("KIKO_LIVE_IMU")?.unwrap_or(false);
    let vio_enabled = try_env_bool("KIKO_VIO")?.unwrap_or(false);
    #[cfg(feature = "vio")]
    if vio_enabled && !imu_enabled {
        return Err("KIKO_VIO=true requires KIKO_LIVE_IMU=true".into());
    }
    #[cfg(not(feature = "vio"))]
    let _ = vio_enabled;
    let depth_queue_depth = try_env_usize("KIKO_LIVE_DEPTH_QUEUE_DEPTH")?.unwrap_or(8);

    let config = DeviceConfig {
        rgb: None,
        mono: Some(mono_config),
        depth: depth_enabled.then_some(DepthConfig {
            width: mono_config.width,
            height: mono_config.height,
            fps: mono_config.fps,
            align_to_rgb: false,
        }),
        imu: imu_enabled.then_some(ImuConfig {
            rate_hz: try_env_u32("KIKO_LIVE_IMU_RATE_HZ")?.unwrap_or(400),
        }),
        queue: QueueConfig {
            size: 8,
            blocking: false,
        },
    };

    eprintln!("connecting to oak-d...");
    let mut device = oak_sys::Device::connect("", config)?;

    let pairing_window = load_pairing_window()?;
    let pairer_max_pending = load_pairer_max_pending_per_side()?;
    let mut pairer = StereoPairer::new_with_max_pending(pairing_window, pairer_max_pending);
    let read_timeout_ms = load_oak_read_timeout_ms()?;

    let pair_queue_depth = try_env_usize("KIKO_LIVE_PAIR_QUEUE_DEPTH")?.unwrap_or(12);
    let pair_capacity = ChannelCapacity::try_from(pair_queue_depth)?;
    let (pair_tx, pair_rx, pair_stats) =
        bounded_channel::<CaptureBundle>(pair_capacity, DropPolicy::DropOldest);

    let viz_queue_depth = try_env_usize("KIKO_LIVE_VIZ_QUEUE_DEPTH")?.unwrap_or(12);
    let viz_capacity = ChannelCapacity::try_from(viz_queue_depth)?;
    let (viz_tx, viz_rx, viz_stats) = bounded_channel(viz_capacity, DropPolicy::DropNewest);
    let (depth_tx, depth_rx, depth_stats_handle) = if depth_enabled {
        let depth_capacity = ChannelCapacity::try_from(depth_queue_depth)?;
        let (depth_tx, depth_rx, depth_stats) =
            bounded_channel::<DepthImage>(depth_capacity, DropPolicy::DropOldest);
        (Some(depth_tx), Some(depth_rx), Some(depth_stats))
    } else {
        (None, None, None)
    };

    let inference = InferenceConfig::from_args(&args.inference, InferencePurpose::Slam)?;
    let InferenceConfig {
        superpoint,
        superpoint_right,
        lightglue,
        lightglue_prefetch,
        end2end,
        key_limit,
        downscale,
    } = inference;
    if end2end.is_some() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "--pipeline-model is not supported by live SLAM",
        )
        .into());
    }
    let superpoint_right = superpoint_right.into_option();
    let lightglue_prefetch = lightglue_prefetch.into_option();

    let dataset_calibration = build_calibration(&device, device.stereo_baseline_m(), &mono_config)?;
    let dataset_calibration =
        kiko_slam::apply_runtime_imu_calibration_override(&dataset_calibration)?;
    let rectified = RectifiedStereo::from_calibration(&dataset_calibration)?;
    let intrinsics = PinholeIntrinsics::try_from(&dataset_calibration.left)?;
    let calibration =
        CalibrationBundle::from_dataset_calibration(intrinsics, rectified, &dataset_calibration)?;
    #[cfg(feature = "vio")]
    if vio_enabled && !calibration.has_imu() {
        return Err(
            "KIKO_VIO=true requires IMU calibration via calibration.json, KIKO_IMU_CALIBRATION_FILE, or KIKO_IMU_* env".into(),
        );
    }
    let imu_time_offset_ns = calibration
        .imu_extrinsics()
        .map(|extrinsics| extrinsics.time_offset_ns())
        .unwrap_or(0);

    let tracker_config = build_tracker_config(
        TrackerDefaults {
            min_keyframe_points: 80,
            refresh_inliers: 20,
            min_inliers: 15,
        },
        key_limit,
        downscale,
    )?;
    let use_speculative_lg = tracker_config.tracking_matcher.uses_speculative_lightglue();

    eprintln!(
        "live: pair_queue_depth={} viz_queue_depth={} depth_enabled={} depth_queue_depth={} pairing_window_ns={} pairer_max_pending_per_side={}",
        pair_queue_depth,
        viz_queue_depth,
        depth_enabled,
        depth_queue_depth,
        pairer.window().as_ns(),
        pairer.max_pending_per_side().get()
    );

    let inference_handle = thread::spawn(move || -> Result<(), LiveThreadError> {
        let mut tracker = SlamTracker::try_new(superpoint, lightglue, calibration, tracker_config)
            .map_err(|source| LiveThreadError::TrackerInit { source })?;
        if let Some(sp_right) = superpoint_right {
            tracker.return_prefetch_sp(sp_right);
        }
        if let Some(lg_prefetch) = lightglue_prefetch.filter(|_| use_speculative_lg) {
            tracker.return_prefetch_lg(lg_prefetch);
        }
        let depth_rx = depth_rx;

        for capture in pair_rx.iter() {
            let imu = capture.imu().batch().cloned();
            let left = capture.pair().left().clone();
            let right = capture.pair().right().clone();
            let depth = depth_rx.as_ref().and_then(drain_latest_depth);
            let process_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                tracker.process_capture(capture)
            }));
            match process_result {
                Ok(Ok(output)) => {
                    let health = output.health.clone();
                    let mut packet = None;
                    let mut points = None;
                    let log_covisibility = output.keyframe.is_some()
                        || output.diagnostics.loop_closure_status.is_some();
                    let covisibility_snapshot = if log_covisibility {
                        Some(tracker.covisibility_snapshot())
                    } else {
                        None
                    };
                    if let Some(matches) = output.stereo_matches {
                        if let Some(keyframe) = output.keyframe.as_ref() {
                            points = Some(keyframe.landmarks().to_vec());
                        }
                        packet = Some(
                            VizPacket::try_new(left.clone(), right.clone(), matches)
                                .map_err(|source| LiveThreadError::VizPacket { source })?,
                        );
                    }
                    let msg = LiveVizMsg {
                        left,
                        right,
                        depth,
                        imu,
                        pose: output.pose,
                        vio_telemetry: output.vio_telemetry,
                        packet,
                        points,
                        covisibility_snapshot,
                        health,
                        diagnostics: output.diagnostics,
                        events: output.events,
                    };
                    if matches!(viz_tx.try_send(msg), SendOutcome::Disconnected) {
                        return Err(LiveThreadError::VizChannelDisconnected);
                    }
                }
                Ok(Err(err)) => {
                    return Err(LiveThreadError::Tracker { source: err });
                }
                Err(payload) => {
                    return Err(LiveThreadError::FrameProcessingPanic {
                        detail: kiko_slam::panic_payload_to_string(payload.as_ref()),
                    });
                }
            }
        }
        Ok(())
    });

    let decimation = args.rerun.rerun_decimation;
    let rerun = args.rerun.clone();
    let live_viz_enabled = try_env_bool("KIKO_LIVE_VIZ")?.unwrap_or(true);
    let viz_handle = thread::spawn(move || -> Result<(), LiveThreadError> {
        let mut sink = if live_viz_enabled {
            match rerun_recording(&rerun, "kiko-slam-live") {
                Ok(rec) => Some(
                    RerunSink::try_new(rec, decimation)
                        .map_err(|source| LiveThreadError::VizInit { source })?,
                ),
                Err(err) => {
                    eprintln!("failed to connect to rerun viewer; continuing headless: {err}");
                    None
                }
            }
        } else {
            eprintln!("live viz disabled; continuing headless");
            None
        };
        for msg in viz_rx.iter() {
            if let Some(sink) = sink.as_mut() {
                if let Some(packet) = msg.packet.as_ref() {
                    if let Err(err) = sink.log_with_points(packet, msg.points.as_deref()) {
                        eprintln!("rerun log error: {err}");
                    }
                } else if let Err(err) = sink.log_frames(&msg.left, &msg.right) {
                    eprintln!("rerun log error: {err}");
                }
                if let Some(depth) = msg.depth.as_ref() {
                    if let Err(err) = sink.log_depth(depth) {
                        eprintln!("rerun log error: {err}");
                    }
                }
                if let Some(imu) = msg.imu.as_ref() {
                    if let Err(err) = sink.log_imu_batch(imu) {
                        eprintln!("rerun imu log error: {err}");
                    }
                }

                if let Some(pose) = msg.pose.current_estimate() {
                    if let Err(err) = sink.log_tracking_pose(msg.left.timestamp(), pose) {
                        eprintln!("rerun log error: {err}");
                    }
                }
                if let Some(vio_telemetry) = msg.vio_telemetry.as_ref() {
                    if let Err(err) = sink.log_vio_telemetry(msg.left.timestamp(), vio_telemetry) {
                        eprintln!("rerun imu log error: {err}");
                    }
                }
                if let Some(snapshot) = msg.covisibility_snapshot.as_ref() {
                    if let Err(err) = sink.log_covisibility_graph(msg.left.timestamp(), snapshot) {
                        eprintln!("rerun log error: {err}");
                    }
                }
                if let Err(err) = sink.log_system_health(msg.left.timestamp(), &msg.health) {
                    eprintln!("rerun health error: {err}");
                }
                if let Err(err) = sink.log_diagnostics(msg.left.timestamp(), &msg.diagnostics) {
                    eprintln!("rerun diagnostics error: {err}");
                }
                for event in &msg.events {
                    if let Err(err) = sink.log_event(msg.left.timestamp(), event) {
                        eprintln!("rerun event error: {err}");
                    }
                }
            }
        }
        Ok(())
    });

    let mut left_seq = 0u64;
    let mut right_seq = 0u64;
    let mut capture_seq = 0u64;
    let mut previous_capture_time = None;
    let mut pending_imu = ImuAccumulator::new();
    let mut pending_capacity_left_drops = 0u64;
    let mut pending_capacity_right_drops = 0u64;

    eprintln!("streaming matches... press ctrl+c to stop");

    let capture_result = (|| -> Result<(), LiveCaptureError> {
        while running.load(Ordering::Relaxed) {
            let mut got_any = false;

            match device.mono_left(read_timeout_ms) {
                Ok(frame) => {
                    match oak_to_frame(frame, SensorId::StereoLeft, FrameId::new(left_seq)) {
                        Ok(frame) => {
                            if let Some(PairingOutcome::Dropped {
                                sensor: SensorId::StereoLeft,
                                reason: PairingDropReason::PendingCapacity,
                            }) = pairer.push_left(frame)
                            {
                                pending_capacity_left_drops =
                                    pending_capacity_left_drops.saturating_add(1);
                            }
                            left_seq += 1;
                            got_any = true;
                        }
                        Err(source) => {
                            return Err(LiveCaptureError::Frame {
                                sensor: SensorId::StereoLeft,
                                source,
                            });
                        }
                    }
                }
                Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
                Err(source) => {
                    return Err(LiveCaptureError::Image {
                        sensor: SensorId::StereoLeft,
                        source,
                    });
                }
            }

            match device.mono_right(read_timeout_ms) {
                Ok(frame) => {
                    match oak_to_frame(frame, SensorId::StereoRight, FrameId::new(right_seq)) {
                        Ok(frame) => {
                            if let Some(PairingOutcome::Dropped {
                                sensor: SensorId::StereoRight,
                                reason: PairingDropReason::PendingCapacity,
                            }) = pairer.push_right(frame)
                            {
                                pending_capacity_right_drops =
                                    pending_capacity_right_drops.saturating_add(1);
                            }
                            right_seq += 1;
                            got_any = true;
                        }
                        Err(source) => {
                            return Err(LiveCaptureError::Frame {
                                sensor: SensorId::StereoRight,
                                source,
                            });
                        }
                    }
                }
                Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
                Err(source) => {
                    return Err(LiveCaptureError::Image {
                        sensor: SensorId::StereoRight,
                        source,
                    });
                }
            }

            if depth_enabled {
                match device.depth(read_timeout_ms) {
                    Ok(depth_frame) => match oak_to_depth_image(depth_frame) {
                        Ok(depth_image) => {
                            got_any = true;
                            if let Some(depth_tx) = depth_tx.as_ref() {
                                if matches!(
                                    depth_tx.try_send(depth_image),
                                    SendOutcome::Disconnected
                                ) {
                                    return Err(LiveCaptureError::DepthConsumerDisconnected);
                                }
                            }
                        }
                        Err(source) => return Err(LiveCaptureError::DepthImage { source }),
                    },
                    Err(DepthError::Timeout { .. } | DepthError::QueueEmpty) => {}
                    Err(source) => return Err(LiveCaptureError::Depth { source }),
                }
            }

            if imu_enabled {
                match device.imu() {
                    Ok(samples) => match oak_to_imu_batch(samples) {
                        Ok(batch) => match batch.shifted_timestamp_ns(imu_time_offset_ns) {
                            Ok(batch) => {
                                pending_imu.extend_batch(&batch).map_err(|source| {
                                    LiveCaptureError::ImuAccumulator { source }
                                })?;
                                got_any = true;
                            }
                            Err(source) => {
                                return Err(LiveCaptureError::ImuTimestampShift { source });
                            }
                        },
                        Err(source) => return Err(LiveCaptureError::OakImu { source }),
                    },
                    Err(ImuError::Empty) => {}
                    Err(ImuError::Overflow { dropped }) => {
                        eprintln!("imu overflow: dropped {dropped} samples");
                    }
                    Err(source @ ImuError::Disconnected) => {
                        return Err(LiveCaptureError::Imu { source });
                    }
                }
            }

            loop {
                match pairer
                    .next_outcome()
                    .map_err(|source| LiveCaptureError::Pairing { source })?
                {
                    PairingOutcome::Produced(pair) => {
                        let capture_time = pair.capture_time();
                        let interval =
                            match CaptureInterval::new(previous_capture_time, capture_time) {
                                Ok(interval) => interval,
                                Err(source) => {
                                    return Err(LiveCaptureError::CaptureInterval { source });
                                }
                            };
                        let imu = pending_imu
                            .drain_interval(
                                previous_capture_time,
                                capture_time,
                                NonZeroUsize::new(2).expect("two is nonzero"),
                            )
                            .map_err(|source| LiveCaptureError::ImuBatch { source })?
                            .map(CaptureImu::present)
                            .unwrap_or_else(CaptureImu::absent);
                        let capture = match CaptureBundle::new(
                            CaptureId::new(capture_seq),
                            pair,
                            interval,
                            imu,
                        ) {
                            Ok(capture) => capture,
                            Err(source) => {
                                return Err(LiveCaptureError::CaptureBundle { source });
                            }
                        };
                        capture_seq = capture_seq.saturating_add(1);
                        previous_capture_time = Some(capture_time);
                        if matches!(pair_tx.try_send(capture), SendOutcome::Disconnected) {
                            running.store(false, Ordering::SeqCst);
                            return Err(LiveCaptureError::InferenceConsumerDisconnected);
                        }
                    }
                    PairingOutcome::Dropped { .. } => continue,
                    PairingOutcome::Waiting => break,
                }
            }

            if !got_any {
                thread::sleep(Duration::from_micros(500));
            }
        }
        Ok(())
    })();

    drop(pair_tx);
    drop(depth_tx);
    let inference_result = join_live_worker("inference", inference_handle);
    let viz_result = join_live_worker("visualization", viz_handle);

    let mut capture_error = capture_result.err();
    let downstream_disconnected = matches!(
        capture_error.as_ref(),
        Some(
            LiveCaptureError::DepthConsumerDisconnected
                | LiveCaptureError::InferenceConsumerDisconnected
        )
    );
    if !downstream_disconnected {
        if let Some(error) = capture_error.take() {
            if let Err(worker_error) = &inference_result {
                eprintln!("secondary live shutdown error: {worker_error}");
            }
            if let Err(worker_error) = &viz_result {
                eprintln!("secondary live shutdown error: {worker_error}");
            }
            return Err(Box::new(error));
        }
    }
    if let Err(error) = inference_result {
        if let Err(worker_error) = &viz_result {
            eprintln!("secondary live shutdown error: {worker_error}");
        }
        return Err(Box::new(error));
    }
    if let Err(error) = viz_result {
        return Err(Box::new(error));
    }
    if let Some(error) = capture_error {
        return Err(Box::new(error));
    }

    let pair_snapshot = pair_stats.snapshot();
    let viz_snapshot = viz_stats.snapshot();
    eprintln!(
        "pair queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}, current_depth={}, max_depth={}",
        pair_snapshot.enqueued,
        pair_snapshot.dropped_oldest,
        pair_snapshot.dropped_newest,
        pair_snapshot.disconnected,
        pair_snapshot.current_depth,
        pair_snapshot.max_depth
    );
    eprintln!(
        "viz queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}, current_depth={}, max_depth={}",
        viz_snapshot.enqueued,
        viz_snapshot.dropped_oldest,
        viz_snapshot.dropped_newest,
        viz_snapshot.disconnected,
        viz_snapshot.current_depth,
        viz_snapshot.max_depth
    );
    if let Some(depth_stats_handle) = depth_stats_handle {
        let depth_snapshot = depth_stats_handle.snapshot();
        eprintln!(
            "depth queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}, current_depth={}, max_depth={}",
            depth_snapshot.enqueued,
            depth_snapshot.dropped_oldest,
            depth_snapshot.dropped_newest,
            depth_snapshot.disconnected,
            depth_snapshot.current_depth,
            depth_snapshot.max_depth
        );
    }
    let pairer_stats = pairer.stats();
    eprintln!(
        "pairer stats: paired={} dropped_left={} dropped_right={} outside_window={}",
        pairer_stats.paired,
        pairer_stats.dropped_left,
        pairer_stats.dropped_right,
        pairer_stats.outside_window
    );
    if pending_capacity_left_drops > 0 || pending_capacity_right_drops > 0 {
        eprintln!(
            "pairer pending-capacity drops: left={pending_capacity_left_drops} right={pending_capacity_right_drops}"
        );
    }

    Ok(())
}
