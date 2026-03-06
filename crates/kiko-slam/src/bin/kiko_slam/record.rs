use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use clap::Args;

use kiko_slam::dataset::{
    Calibration, CameraIntrinsics, DatasetWriter, DepthMeta, ImuMeta, Meta, MonoMeta,
};
use kiko_slam::env::{env_bool, env_usize};
use kiko_slam::{
    oak_to_depth_image, oak_to_frame, FrameId, PairingConfigError, PairingOutcome,
    PairingDropReason, PairingWindowNs, PendingFramesCapacity, SensorId, StereoPairer,
};
use oak_sys::{
    DepthConfig, DepthError, Device, DeviceConfig, ImageError, ImuConfig, MonoConfig, QueueConfig,
};

use crate::args::CameraArgs;

const DEFAULT_PAIRING_WINDOW_NS: i64 = 5_000_000;
const DEFAULT_PAIRER_MAX_PENDING_PER_SIDE: usize = 64;

#[derive(Args, Clone, Debug)]
#[command(about = "Record stereo dataset from OAK-D camera")]
pub struct RecordArgs {
    /// Output path for the recorded dataset
    #[arg(value_name = "OUTPUT_PATH")]
    pub output_path: std::path::PathBuf,
    #[command(flatten)]
    pub camera: CameraArgs,
}

pub fn run_record(args: &RecordArgs) -> Result<(), Box<dyn std::error::Error>> {
    let output_path = &args.output_path;

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
    let depth_enabled = env_bool("KIKO_RECORD_DEPTH").unwrap_or(false);
    let depth_config = depth_enabled.then_some(DepthConfig {
        width: mono_config.width,
        height: mono_config.height,
        fps: mono_config.fps,
        align_to_rgb: false,
    });

    let config = DeviceConfig {
        rgb: None,
        mono: Some(mono_config),
        depth: depth_config,
        imu: None,
        queue: QueueConfig {
            size: 8,
            blocking: false,
        },
    };

    eprintln!("connecting to oak-d...");
    let mut device = Device::connect("", config)?;
    let baseline_m = device.stereo_baseline_m();

    let meta = build_meta(&mono_config, depth_config.as_ref(), None);
    let calibration = build_calibration(&device, baseline_m, &mono_config);

    eprintln!("creating dataset at {}", output_path.display());
    let (writer, writer_handle) = DatasetWriter::create(output_path, &meta, &calibration)?;

    let mut pair_count = 0u64;
    let mut left_count = 0u64;
    let mut right_count = 0u64;
    let mut depth_count = 0u64;
    let mut left_seq = 0u64;
    let mut right_seq = 0u64;
    let mut pending_capacity_left_drops = 0u64;
    let mut pending_capacity_right_drops = 0u64;
    let pairing_window = load_pairing_window()?;
    let pairer_max_pending = load_pairer_max_pending_per_side();
    let mut pairer = StereoPairer::new_with_max_pending(pairing_window, pairer_max_pending);
    let read_timeout_ms = load_oak_read_timeout_ms();
    let start = Instant::now();

    eprintln!("recording... press ctrl+c to stop");

    while running.load(Ordering::Relaxed) {
        let mut got_any = false;

        match device.mono_left(read_timeout_ms) {
            Ok(frame) => match oak_to_frame(frame, SensorId::StereoLeft, FrameId::new(left_seq)) {
                Ok(frame) => {
                    if let Some(PairingOutcome::Dropped {
                        sensor: SensorId::StereoLeft,
                        reason: PairingDropReason::PendingCapacity,
                    }) = pairer.push_left(frame)
                    {
                        pending_capacity_left_drops =
                            pending_capacity_left_drops.saturating_add(1);
                    }
                    left_count += 1;
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
                        right_count += 1;
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
                    Ok(depth) => {
                        writer.write_depth(&depth);
                        depth_count = depth_count.saturating_add(1);
                        got_any = true;
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

        loop {
            match pairer.next_outcome()? {
                PairingOutcome::Produced(pair) => {
                    writer.write_frame(pair.left());
                    writer.write_frame(pair.right());
                    pair_count += 1;

                    if pair_count % 30 == 0 {
                        eprintln!("captured {pair_count} stereo pairs");
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

    let elapsed = start.elapsed().as_secs_f64();
    let pairer_stats = pairer.stats();
    drop(writer);
    let stats = writer_handle.finish()?;
    eprintln!(
        "finished in {:.1}s: pairs={}, left={} ({:.1}fps), right={} ({:.1}fps), depth={} ({:.1}fps), written={}, dropped={}",
        elapsed,
        pair_count,
        left_count,
        left_count as f64 / elapsed,
        right_count,
        right_count as f64 / elapsed,
        depth_count,
        depth_count as f64 / elapsed,
        stats.frames_written,
        stats.frames_dropped
    );
    eprintln!(
        "pairer stats: window_ns={} max_pending_per_side={} paired={} dropped_left={} dropped_right={} outside_window={}",
        pairer.window().as_ns(),
        pairer.max_pending_per_side().get(),
        pairer_stats.paired,
        pairer_stats.dropped_left,
        pairer_stats.dropped_right,
        pairer_stats.outside_window
    );
    if pending_capacity_left_drops > 0 || pending_capacity_right_drops > 0 {
        eprintln!(
            "pairer pending-capacity drops: left={} right={}",
            pending_capacity_left_drops,
            pending_capacity_right_drops
        );
    }
    Ok(())
}

fn build_meta(
    config: &MonoConfig,
    depth_config: Option<&DepthConfig>,
    imu_config: Option<&ImuConfig>,
) -> Meta {
    Meta {
        created: chrono::Utc::now().to_rfc3339(),
        device: "OAK-D".to_string(),
        mono: Some(MonoMeta {
            width: config.width,
            height: config.height,
            fps: config.fps,
        }),
        depth: depth_config.map(|c| DepthMeta {
            width: c.width,
            height: c.height,
            fps: c.fps,
            encoding: "f32_meters_le".to_string(),
        }),
        imu: imu_config.map(|c| ImuMeta { rate_hz: c.rate_hz }),
    }
}

pub(crate) fn build_calibration(
    device: &Device,
    baseline_m: f32,
    config: &MonoConfig,
) -> Calibration {
    let left = device.left_intrinsics();
    let right = device.right_intrinsics();

    Calibration {
        left: CameraIntrinsics {
            fx: left.fx,
            fy: left.fy,
            cx: left.cx,
            cy: left.cy,
            width: left.width,
            height: left.height,
        },
        right: CameraIntrinsics {
            fx: right.fx,
            fy: right.fy,
            cx: right.cx,
            cy: right.cy,
            width: right.width,
            height: right.height,
        },
        baseline_m,
        rectified: config.rectified,
    }
}

pub(crate) fn load_pairing_window() -> Result<PairingWindowNs, PairingConfigError> {
    let window_ns = match env_usize("KIKO_PAIRING_WINDOW_NS") {
        Some(raw) => match i64::try_from(raw) {
            Ok(value) => value,
            Err(_) => {
                eprintln!(
                    "invalid KIKO_PAIRING_WINDOW_NS={raw}, exceeds i64::MAX, using default {DEFAULT_PAIRING_WINDOW_NS}"
                );
                DEFAULT_PAIRING_WINDOW_NS
            }
        },
        None => DEFAULT_PAIRING_WINDOW_NS,
    };
    match PairingWindowNs::new(window_ns) {
        Ok(window) => Ok(window),
        Err(err) => {
            eprintln!("invalid pairing window from env ({err}); using default");
            PairingWindowNs::new(DEFAULT_PAIRING_WINDOW_NS)
        }
    }
}

pub(crate) fn load_pairer_max_pending_per_side() -> PendingFramesCapacity {
    let raw = env_usize("KIKO_PAIRER_MAX_PENDING_PER_SIDE")
        .unwrap_or(DEFAULT_PAIRER_MAX_PENDING_PER_SIDE);
    match PendingFramesCapacity::try_from(raw) {
        Ok(capacity) => capacity,
        Err(err) => {
            eprintln!("invalid pairer capacity from env ({err}); using default");
            PendingFramesCapacity::try_from(DEFAULT_PAIRER_MAX_PENDING_PER_SIDE)
                .expect("default pairer capacity")
        }
    }
}

pub(crate) fn load_oak_read_timeout_ms() -> u32 {
    env_usize("KIKO_OAK_READ_TIMEOUT_MS")
        .unwrap_or(2)
        .min(u32::MAX as usize) as u32
}
