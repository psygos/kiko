use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use clap::Args;

use kiko_slam::dataset::{
    Calibration, CameraIntrinsics, DatasetWriter, DepthMeta, ImuMeta, Meta, MonoMeta, WriteOutcome,
};
use kiko_slam::env::{try_env_bool, try_env_i64, try_env_u32, try_env_usize};
use kiko_slam::{
    FrameId, PairingDropReason, PairingOutcome, PairingWindowNs, PendingFramesCapacity, SensorId,
    StereoPairer, load_runtime_imu_calibration_from_env, oak_to_depth_image, oak_to_frame,
    oak_to_imu_batch,
};
use oak_sys::{
    DepthConfig, DepthError, Device, DeviceConfig, ImageError, ImuConfig, ImuError, MonoConfig,
    QueueConfig,
};

use crate::args::CameraArgs;

const DEFAULT_PAIRING_WINDOW_NS: i64 = 5_000_000;
const DEFAULT_PAIRER_MAX_PENDING_PER_SIDE: usize = 64;

#[derive(Debug)]
struct RuntimeSettingError<E> {
    key: &'static str,
    value: String,
    source: E,
}

impl<E: std::fmt::Display> std::fmt::Display for RuntimeSettingError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "environment variable {} has invalid value {:?}: {}",
            self.key, self.value, self.source
        )
    }
}

impl<E: std::error::Error + 'static> std::error::Error for RuntimeSettingError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

#[derive(Args, Clone, Debug)]
#[command(about = "Record stereo dataset from OAK-D camera")]
pub struct RecordArgs {
    /// Output path for the recorded dataset
    #[arg(value_name = "OUTPUT_PATH")]
    pub output_path: std::path::PathBuf,
    #[command(flatten)]
    pub camera: CameraArgs,
}

#[derive(Debug)]
enum RecordCaptureError {
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
    Pairing {
        source: kiko_slam::PairError,
    },
    WriterDropped {
        item: &'static str,
    },
    WriterFailed {
        item: &'static str,
    },
}

impl std::fmt::Display for RecordCaptureError {
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
            Self::Pairing { source } => write!(f, "stereo pairing failed: {source}"),
            Self::WriterDropped { item } => {
                write!(f, "dataset writer dropped {item} due to backpressure")
            }
            Self::WriterFailed { item } => write!(f, "dataset writer failed while queuing {item}"),
        }
    }
}

impl std::error::Error for RecordCaptureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Image { source, .. } => Some(source),
            Self::Frame { source, .. } => Some(source),
            Self::Depth { source } => Some(source),
            Self::DepthImage { source } => Some(source),
            Self::Imu { source } => Some(source),
            Self::OakImu { source } => Some(source),
            Self::Pairing { source } => Some(source),
            Self::WriterDropped { .. } | Self::WriterFailed { .. } => None,
        }
    }
}

fn require_enqueued(outcome: WriteOutcome, item: &'static str) -> Result<(), RecordCaptureError> {
    match outcome {
        WriteOutcome::Enqueued => Ok(()),
        WriteOutcome::Dropped => Err(RecordCaptureError::WriterDropped { item }),
        WriteOutcome::WriterFailed => Err(RecordCaptureError::WriterFailed { item }),
    }
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
    let depth_enabled = try_env_bool("KIKO_RECORD_DEPTH")?.unwrap_or(false);
    let imu_enabled = try_env_bool("KIKO_RECORD_IMU")?.unwrap_or(false);
    let depth_config = depth_enabled.then_some(DepthConfig {
        width: mono_config.width,
        height: mono_config.height,
        fps: mono_config.fps,
        align_to_rgb: false,
    });
    let imu_config = imu_enabled.then_some(ImuConfig {
        rate_hz: try_env_u32("KIKO_RECORD_IMU_RATE_HZ")?.unwrap_or(400),
    });

    let config = DeviceConfig {
        rgb: None,
        mono: Some(mono_config),
        depth: depth_config,
        imu: imu_config,
        queue: QueueConfig {
            size: 8,
            blocking: false,
        },
    };

    eprintln!("connecting to oak-d...");
    let mut device = Device::connect("", config)?;
    let baseline_m = device.stereo_baseline_m();

    let meta = build_meta(&mono_config, depth_config.as_ref(), imu_config.as_ref());
    let calibration = build_calibration(&device, baseline_m, &mono_config)?;

    eprintln!("creating dataset at {}", output_path.display());
    let (writer, writer_handle) = DatasetWriter::create(output_path, &meta, &calibration)?;

    let mut pair_count = 0u64;
    let mut left_count = 0u64;
    let mut right_count = 0u64;
    let mut depth_count = 0u64;
    let mut imu_batch_count = 0u64;
    let mut imu_sample_count = 0u64;
    let mut left_seq = 0u64;
    let mut right_seq = 0u64;
    let mut pending_capacity_left_drops = 0u64;
    let mut pending_capacity_right_drops = 0u64;
    let pairing_window = load_pairing_window()?;
    let pairer_max_pending = load_pairer_max_pending_per_side()?;
    let mut pairer = StereoPairer::new_with_max_pending(pairing_window, pairer_max_pending);
    let read_timeout_ms = load_oak_read_timeout_ms()?;
    let start = Instant::now();

    eprintln!("recording... press ctrl+c to stop");

    let capture_result = (|| -> Result<(), RecordCaptureError> {
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
                            left_count += 1;
                            left_seq += 1;
                            got_any = true;
                        }
                        Err(source) => {
                            return Err(RecordCaptureError::Frame {
                                sensor: SensorId::StereoLeft,
                                source,
                            });
                        }
                    }
                }
                Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
                Err(source) => {
                    return Err(RecordCaptureError::Image {
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
                            right_count += 1;
                            right_seq += 1;
                            got_any = true;
                        }
                        Err(source) => {
                            return Err(RecordCaptureError::Frame {
                                sensor: SensorId::StereoRight,
                                source,
                            });
                        }
                    }
                }
                Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
                Err(source) => {
                    return Err(RecordCaptureError::Image {
                        sensor: SensorId::StereoRight,
                        source,
                    });
                }
            }

            if depth_enabled {
                match device.depth(read_timeout_ms) {
                    Ok(depth_frame) => match oak_to_depth_image(depth_frame) {
                        Ok(depth) => {
                            require_enqueued(writer.write_depth(&depth), "depth frame")?;
                            depth_count = depth_count.saturating_add(1);
                            got_any = true;
                        }
                        Err(source) => return Err(RecordCaptureError::DepthImage { source }),
                    },
                    Err(DepthError::Timeout { .. } | DepthError::QueueEmpty) => {}
                    Err(source) => return Err(RecordCaptureError::Depth { source }),
                }
            }

            if imu_enabled {
                match device.imu() {
                    Ok(samples) => match oak_to_imu_batch(samples) {
                        Ok(batch) => {
                            require_enqueued(writer.write_imu(&batch), "IMU batch")?;
                            imu_sample_count = imu_sample_count.saturating_add(batch.len() as u64);
                            imu_batch_count = imu_batch_count.saturating_add(1);
                            got_any = true;
                        }
                        Err(source) => return Err(RecordCaptureError::OakImu { source }),
                    },
                    Err(ImuError::Empty) => {}
                    Err(ImuError::Overflow { dropped }) => {
                        eprintln!("imu overflow: dropped {dropped} samples");
                    }
                    Err(source @ ImuError::Disconnected) => {
                        return Err(RecordCaptureError::Imu { source });
                    }
                }
            }

            loop {
                match pairer
                    .next_outcome()
                    .map_err(|source| RecordCaptureError::Pairing { source })?
                {
                    PairingOutcome::Produced(pair) => {
                        require_enqueued(writer.write_stereo_pair(&pair), "stereo pair")?;
                        pair_count = pair_count.saturating_add(1);

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
        Ok(())
    })();

    let elapsed = start.elapsed().as_secs_f64();
    let pairer_stats = pairer.stats();
    drop(writer);
    let writer_result = writer_handle.finish();
    let stats = match writer_result {
        Ok(stats) => stats,
        Err(error) => {
            if let Err(capture_error) = capture_result {
                eprintln!("secondary recording capture error: {capture_error}");
            }
            return Err(Box::new(error));
        }
    };
    eprintln!(
        "finished in {:.1}s: pairs={}, left={} ({:.1}fps), right={} ({:.1}fps), depth={} ({:.1}fps), imu_batches={} imu_samples={}, written={}, dropped={}",
        elapsed,
        pair_count,
        left_count,
        left_count as f64 / elapsed,
        right_count,
        right_count as f64 / elapsed,
        depth_count,
        depth_count as f64 / elapsed,
        imu_batch_count,
        imu_sample_count,
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
            "pairer pending-capacity drops: left={pending_capacity_left_drops} right={pending_capacity_right_drops}"
        );
    }
    if let Err(error) = capture_result {
        return Err(Box::new(error));
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
) -> Result<Calibration, Box<dyn std::error::Error>> {
    let left = device.left_intrinsics();
    let right = device.right_intrinsics();

    Ok(Calibration {
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
        imu: load_runtime_imu_calibration_from_env()?,
    })
}

pub(crate) fn load_pairing_window() -> Result<PairingWindowNs, Box<dyn std::error::Error>> {
    const KEY: &str = "KIKO_PAIRING_WINDOW_NS";
    let window_ns = try_env_i64(KEY)?.unwrap_or(DEFAULT_PAIRING_WINDOW_NS);
    PairingWindowNs::new(window_ns).map_err(|source| {
        Box::new(RuntimeSettingError {
            key: KEY,
            value: window_ns.to_string(),
            source,
        }) as Box<dyn std::error::Error>
    })
}

pub(crate) fn load_pairer_max_pending_per_side()
-> Result<PendingFramesCapacity, Box<dyn std::error::Error>> {
    const KEY: &str = "KIKO_PAIRER_MAX_PENDING_PER_SIDE";
    let raw = try_env_usize(KEY)?.unwrap_or(DEFAULT_PAIRER_MAX_PENDING_PER_SIDE);
    PendingFramesCapacity::try_from(raw).map_err(|source| {
        Box::new(RuntimeSettingError {
            key: KEY,
            value: raw.to_string(),
            source,
        }) as Box<dyn std::error::Error>
    })
}

pub(crate) fn load_oak_read_timeout_ms() -> Result<u32, Box<dyn std::error::Error>> {
    Ok(try_env_u32("KIKO_OAK_READ_TIMEOUT_MS")?.unwrap_or(2))
}
