use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use kiko_slam::dataset::{Calibration, CameraIntrinsics, DatasetWriter, ImuMeta, Meta, MonoMeta};
use kiko_slam::{oak_to_frame, FrameId, PairingWindowNs, SensorId, StereoPairer};
use oak_sys::{Device, DeviceConfig, ImageError, ImuConfig, MonoConfig, QueueConfig};

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
        queue: QueueConfig {
            size: 8,
            blocking: false,
        },
    };

    eprintln!("connecting to oak-d...");
    let mut device = Device::connect("", config)?;
    let baseline_m = device.stereo_baseline_m();

    let meta = build_meta(&mono_config, None);
    let calibration = build_calibration(&device, baseline_m);

    eprintln!("creating dataset at {}", output_path);
    let (writer, writer_handle) = DatasetWriter::create(output_path, &meta, &calibration)?;

    let mut pair_count = 0u64;
    let mut left_count = 0u64;
    let mut right_count = 0u64;
    let mut left_seq = 0u64;
    let mut right_seq = 0u64;
    let pairing_window =
        PairingWindowNs::new(5_000_000).expect("pairing window must be positive");
    let mut pairer = StereoPairer::new(pairing_window);
    let start = std::time::Instant::now();

    eprintln!("recording... press ctrl+c to stop");

    while running.load(Ordering::Relaxed) {
        let mut got_any = false;

        // Poll left (non-blocking)
        match device.mono_left(0) {
            Ok(frame) => {
                pairer.push_left(oak_to_frame(
                    frame,
                    SensorId::StereoLeft,
                    FrameId::new(left_seq),
                ));
                left_count += 1;
                left_seq += 1;
                got_any = true;
            }
            Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
            Err(e) => {
                eprintln!("left error: {:?}", e);
                break;
            }
        }

        // Poll right (non-blocking)
        match device.mono_right(0) {
            Ok(frame) => {
                pairer.push_right(oak_to_frame(
                    frame,
                    SensorId::StereoRight,
                    FrameId::new(right_seq),
                ));
                right_count += 1;
                right_seq += 1;
                got_any = true;
            }
            Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
            Err(e) => {
                eprintln!("right error: {:?}", e);
                break;
            }
        }

        // Try to pair and write
        while let Some(pair) = pairer.next_pair()? {
            writer.write_frame(&pair.left);
            writer.write_frame(&pair.right);
            pair_count += 1;

            if pair_count % 30 == 0 {
                eprintln!("captured {} stereo pairs", pair_count);
            }
        }

        // Small sleep if no frames to avoid busy spin
        if !got_any {
            std::thread::sleep(std::time::Duration::from_micros(500));
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    drop(writer);
    let stats = writer_handle.finish()?;
    eprintln!(
        "finished in {:.1}s: pairs={}, left={} ({:.1}fps), right={} ({:.1}fps), written={}, dropped={}",
        elapsed,
        pair_count,
        left_count,
        left_count as f64 / elapsed,
        right_count,
        right_count as f64 / elapsed,
        stats.frames_written,
        stats.frames_dropped
    );
    Ok(())
}
