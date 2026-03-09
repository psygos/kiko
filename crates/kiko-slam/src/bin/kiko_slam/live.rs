use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use clap::Args;

use kiko_slam::env::{env_bool, env_usize};
use kiko_slam::{
    CalibrationBundle, CaptureBundle, CaptureId, CaptureImu, CaptureInterval, ChannelCapacity,
    DepthImage, DiagnosticEvent, DropPolicy, DropReceiver, Frame, FrameDiagnostics, FrameId,
    ImuBatch, ImuSample, PairingDropReason, PairingOutcome, Point3, Raw, RerunSink, SendOutcome,
    SensorId, SlamTracker, StereoPairer, SystemHealth, TrackingPose, VizPacket, bounded_channel,
    oak_to_depth_image, oak_to_frame, oak_to_imu_batch,
};
use kiko_slam::{PinholeIntrinsics, RectifiedStereo};
use oak_sys::{
    DepthConfig, DepthError, DeviceConfig, ImageError, ImuConfig, ImuError, MonoConfig, QueueConfig,
};

use crate::args::{CameraArgs, InferenceArgs, InferenceConfig, RerunArgs};
use crate::config::{TrackerDefaults, build_tracker_config};
use crate::record::{
    build_calibration, load_oak_read_timeout_ms, load_pairer_max_pending_per_side,
    load_pairing_window,
};

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
    pose: Option<TrackingPose>,
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
    TrackerInit { detail: String },
    VizChannelDisconnected,
    FrameProcessingPanic { detail: String },
}

impl std::fmt::Display for LiveThreadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LiveThreadError::TrackerInit { detail } => {
                write!(f, "failed to initialize tracker: {detail}")
            }
            LiveThreadError::VizChannelDisconnected => write!(f, "viz channel disconnected"),
            LiveThreadError::FrameProcessingPanic { detail } => {
                write!(f, "inference panic while processing frame: {detail}")
            }
        }
    }
}

impl std::error::Error for LiveThreadError {}

fn drain_latest_depth(rx: &DropReceiver<DepthImage>) -> Option<DepthImage> {
    let mut latest = None;
    while let Ok(depth) = rx.try_recv() {
        latest = Some(depth);
    }
    latest
}

fn drain_imu_until(
    pending: &mut VecDeque<ImuSample>,
    start_exclusive: Option<kiko_slam::Timestamp>,
    end_inclusive: kiko_slam::Timestamp,
) -> Option<ImuBatch> {
    let mut samples = Vec::new();
    while let Some(sample) = pending.front() {
        if sample.timestamp().as_nanos() > end_inclusive.as_nanos() {
            break;
        }
        let sample = pending.pop_front().expect("front existed");
        if start_exclusive.is_some_and(|start| sample.timestamp().as_nanos() <= start.as_nanos()) {
            continue;
        }
        samples.push(sample);
    }
    if samples.len() < 2 {
        return None;
    }
    ImuBatch::new(samples).ok()
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
    let depth_enabled = env_bool("KIKO_LIVE_DEPTH").unwrap_or(false);
    let imu_enabled = env_bool("KIKO_LIVE_IMU").unwrap_or(false);
    #[cfg(feature = "vio")]
    if env_bool("KIKO_VIO").unwrap_or(false) && !imu_enabled {
        return Err("KIKO_VIO=true requires KIKO_LIVE_IMU=true".into());
    }
    let depth_queue_depth = env_usize("KIKO_LIVE_DEPTH_QUEUE_DEPTH").unwrap_or(8);

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
            rate_hz: u32::try_from(env_usize("KIKO_LIVE_IMU_RATE_HZ").unwrap_or(400))
                .unwrap_or(400),
        }),
        queue: QueueConfig {
            size: 8,
            blocking: false,
        },
    };

    eprintln!("connecting to oak-d...");
    let mut device = oak_sys::Device::connect("", config)?;

    let pairing_window = load_pairing_window()?;
    let pairer_max_pending = load_pairer_max_pending_per_side();
    let mut pairer = StereoPairer::new_with_max_pending(pairing_window, pairer_max_pending);
    let read_timeout_ms = load_oak_read_timeout_ms();

    let pair_queue_depth = env_usize("KIKO_LIVE_PAIR_QUEUE_DEPTH").unwrap_or(12);
    let pair_capacity = ChannelCapacity::try_from(pair_queue_depth)?;
    let (pair_tx, pair_rx, pair_stats) =
        bounded_channel::<CaptureBundle>(pair_capacity, DropPolicy::DropOldest);

    let viz_queue_depth = env_usize("KIKO_LIVE_VIZ_QUEUE_DEPTH").unwrap_or(12);
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

    let inference = InferenceConfig::from_args(&args.inference)?;
    let InferenceConfig {
        superpoint,
        lightglue,
        key_limit,
        downscale,
    } = inference;

    let dataset_calibration = build_calibration(&device, device.stereo_baseline_m(), &mono_config)?;
    let rectified = RectifiedStereo::from_calibration(&dataset_calibration)?;
    let intrinsics = PinholeIntrinsics::try_from(&dataset_calibration.left)?;
    let calibration =
        CalibrationBundle::from_dataset_calibration(intrinsics, rectified, &dataset_calibration)?;
    #[cfg(feature = "vio")]
    if env_bool("KIKO_VIO").unwrap_or(false) && !calibration.has_imu() {
        return Err(
            "KIKO_VIO=true requires IMU calibration via calibration.json or KIKO_IMU_* env".into(),
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
            .map_err(|err| LiveThreadError::TrackerInit {
                detail: err.to_string(),
            })?;
        let depth_rx = depth_rx;

        for capture in pair_rx.iter() {
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
                        if let Ok(viz_packet) =
                            VizPacket::try_new(left.clone(), right.clone(), matches)
                        {
                            packet = Some(viz_packet);
                        }
                    }
                    let msg = LiveVizMsg {
                        left,
                        right,
                        depth,
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
                    eprintln!("tracker error: {err}");
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
    let live_viz_enabled = env_bool("KIKO_LIVE_VIZ").unwrap_or(true);
    let viz_handle = thread::spawn(move || -> Result<(), LiveThreadError> {
        let mut sink = if live_viz_enabled {
            match rerun::RecordingStreamBuilder::new("kiko-slam-live").connect_grpc() {
                Ok(rec) => Some(RerunSink::new(rec, decimation)),
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

                if let Some(pose) = msg.pose.as_ref() {
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
    let mut pending_imu = VecDeque::new();
    let mut pending_capacity_left_drops = 0u64;
    let mut pending_capacity_right_drops = 0u64;

    eprintln!("streaming matches... press ctrl+c to stop");

    'capture: while running.load(Ordering::Relaxed) {
        let mut got_any = false;

        match device.mono_left(read_timeout_ms) {
            Ok(frame) => match oak_to_frame(frame, SensorId::StereoLeft, FrameId::new(left_seq)) {
                Ok(frame) => {
                    if let Some(PairingOutcome::Dropped {
                        sensor: SensorId::StereoLeft,
                        reason: PairingDropReason::PendingCapacity,
                    }) = pairer.push_left(frame)
                    {
                        pending_capacity_left_drops = pending_capacity_left_drops.saturating_add(1);
                    }
                    left_seq += 1;
                    got_any = true;
                }
                Err(err) => {
                    eprintln!("left frame dropped (invalid dimensions): {err}");
                }
            },
            Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
            Err(e) => {
                eprintln!("left error: {e:?}");
                break;
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
                    Err(err) => {
                        eprintln!("right frame dropped (invalid dimensions): {err}");
                    }
                }
            }
            Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
            Err(e) => {
                eprintln!("right error: {e:?}");
                break;
            }
        }

        if depth_enabled {
            match device.depth(read_timeout_ms) {
                Ok(depth_frame) => match oak_to_depth_image(depth_frame) {
                    Ok(depth_image) => {
                        got_any = true;
                        if let Some(depth_tx) = depth_tx.as_ref() {
                            if matches!(depth_tx.try_send(depth_image), SendOutcome::Disconnected) {
                                break;
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!("depth frame dropped (invalid dimensions): {err}");
                    }
                },
                Err(DepthError::Timeout { .. } | DepthError::QueueEmpty) => {}
                Err(e) => {
                    eprintln!("depth error: {e:?}");
                    break;
                }
            }
        }

        if imu_enabled {
            match device.imu() {
                Ok(samples) => match oak_to_imu_batch(samples) {
                    Ok(batch) => match batch.shifted_timestamp_ns(imu_time_offset_ns) {
                        Ok(batch) => {
                            pending_imu.extend(batch.samples().iter().cloned());
                            got_any = true;
                        }
                        Err(err) => {
                            eprintln!("imu batch dropped (invalid timing): {err}");
                        }
                    },
                    Err(err) => {
                        eprintln!("imu batch dropped (invalid values): {err}");
                    }
                },
                Err(ImuError::Empty) => {}
                Err(ImuError::Overflow { dropped }) => {
                    eprintln!("imu overflow: dropped {dropped} samples");
                }
                Err(ImuError::Disconnected) => {
                    eprintln!("imu error: disconnected");
                    break;
                }
            }
        }

        loop {
            match pairer.next_outcome()? {
                PairingOutcome::Produced(pair) => {
                    let capture_time = pair.capture_time();
                    let interval = match CaptureInterval::new(previous_capture_time, capture_time) {
                        Ok(interval) => interval,
                        Err(err) => {
                            eprintln!("capture interval error: {err}");
                            continue;
                        }
                    };
                    let imu =
                        drain_imu_until(&mut pending_imu, previous_capture_time, capture_time)
                            .map(CaptureImu::present)
                            .unwrap_or_else(CaptureImu::absent);
                    let capture = match CaptureBundle::new(
                        CaptureId::new(capture_seq),
                        pair,
                        interval,
                        imu,
                    ) {
                        Ok(capture) => capture,
                        Err(err) => {
                            eprintln!("capture bundle error: {err}");
                            continue;
                        }
                    };
                    capture_seq = capture_seq.saturating_add(1);
                    previous_capture_time = Some(capture_time);
                    if matches!(pair_tx.try_send(capture), SendOutcome::Disconnected) {
                        running.store(false, Ordering::SeqCst);
                        break 'capture;
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

    drop(pair_tx);
    drop(depth_tx);
    let inference_result = inference_handle.join().map_err(|payload| {
        std::io::Error::other(format!(
            "inference thread panicked: {}",
            kiko_slam::panic_payload_to_string(payload.as_ref())
        ))
    })?;
    if let Err(err) = inference_result {
        return Err(std::io::Error::other(err).into());
    }

    let viz_result = viz_handle.join().map_err(|payload| {
        std::io::Error::other(format!(
            "viz thread panicked: {}",
            kiko_slam::panic_payload_to_string(payload.as_ref())
        ))
    })?;
    if let Err(err) = viz_result {
        return Err(std::io::Error::other(err).into());
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
