use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use kiko_slam::dataset::{Calibration, CameraIntrinsics, DatasetWriter, ImuMeta, Meta, MonoMeta};
use kiko_slam::{Frame, SensorId, StereoSource, StereoTap, Timestamp};
use oak_sys::{Device, DeviceConfig, ImageError, ImageFrame, ImuConfig, MonoConfig, QueueConfig};

struct OakStereo {
    device: Device,
    timeout_ms: u32,
    running: Arc<AtomicBool>,
}

impl OakStereo {
    fn new(device: Device, running: Arc<AtomicBool>) -> Self {
        Self {
            device,
            timeout_ms: 1000,
            running,
        }
    }
}

fn oak_to_frame(oak_frame: ImageFrame, sensor: SensorId) -> Frame {
    Frame::new(
        sensor,
        Timestamp::from_nanos(oak_frame.timestamp.as_nanos()),
        oak_frame.width,
        oak_frame.height,
        oak_frame.into_pixels(),
    )
    .expect("oak frame dimensions should be valid")
}

impl StereoSource for OakStereo {
    fn left(&mut self) -> Option<Frame> {
        while self.running.load(Ordering::Relaxed) {
            match self.device.mono_left(self.timeout_ms) {
                Ok(frame) => return Some(oak_to_frame(frame, SensorId::StereoLeft)),
                Err(ImageError::Timeout { .. }) => continue,
                Err(ImageError::QueueEmpty) => continue,
                Err(ImageError::QueueOverflow) => {
                    eprintln!("warning: left camera queue overflow");
                    continue;
                }
                Err(_) => return None,
            }
        }
        None
    }

    fn right(&mut self) -> Option<Frame> {
        while self.running.load(Ordering::Relaxed) {
            match self.device.mono_right(self.timeout_ms) {
                Ok(frame) => return Some(oak_to_frame(frame, SensorId::StereoRight)),
                Err(ImageError::Timeout { .. }) => continue,
                Err(ImageError::QueueEmpty) => continue,
                Err(ImageError::QueueOverflow) => {
                    eprintln!("warning: right camera queue overflow");
                    continue;
                }
                Err(_) => return None,
            }
        }
        None
    }
}

fn build_meta(config: &MonoConfig, imu_config: Option<&ImuConfig>) -> Meta {
    Meta {
        created: chrono::Utc::now().to_rfc3339(),
        device: "OAK-D".to_string(),
        mono: Some(MonoMeta {
            width: config.width,
            height: config.height,
            fps: config.fps,
        }),
        imu: imu_config.map(|c| ImuMeta { rate_hz: c.rate_hz }),
    }
}

fn build_calibration(device: &Device, baseline_m: f32) -> Calibration {
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
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: record <output_path>");
        std::process::exit(1);
    }
    let output_path = &args[1];

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        eprintln!("\nreceived ctrl+c, stopping...");
        r.store(false, Ordering::SeqCst);
    })?;

    let mono_config = MonoConfig {
        width: 640,
        height: 480,
        fps: 30,
    };

    let config = DeviceConfig {
        rgb: None,
        mono: Some(mono_config),
        depth: None,
        imu: None,
        queue: QueueConfig::default(),
    };

    eprintln!("connecting to oak-d...");
    let device = Device::connect("", config)?;
    let baseline_m = device.stereo_baseline_m();

    let meta = build_meta(&mono_config, None);
    let calibration = build_calibration(&device, baseline_m);

    eprintln!("creating dataset at {}", output_path);
    let writer = DatasetWriter::create(output_path, &meta, &calibration)?;

    let source = OakStereo::new(device, running);
    let mut pair_count = 0u64;

    eprintln!("recording... press ctrl+c to stop");

    let recording = {
        let writer = &writer;
        StereoTap::new(source, move |pair| {
            writer.write_frame(&pair.left);
            writer.write_frame(&pair.right);
        })
    };

    for _pair in recording {
        pair_count += 1;
        if pair_count % 30 == 0 {
            eprintln!("captured {} stereo pairs", pair_count);
        }
    }

    let stats = writer.finish()?;
    eprintln!(
        "finished. captured: {}, frames written: {}, frames dropped: {}",
        pair_count, stats.frames_written, stats.frames_dropped
    );
    Ok(())
}
