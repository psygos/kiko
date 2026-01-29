use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use kiko_slam::{
    bounded_channel, oak_to_frame, ChannelCapacity, DropPolicy, InferencePipeline, KeypointLimit,
    LightGlue, PairingWindowNs, RerunSink, SendOutcome, StereoPairer, SuperPoint, VizDecimation,
};
use oak_sys::{Device, DeviceConfig, ImageError, MonoConfig, QueueConfig};

const MAX_KEYPOINTS: usize = 1024;
const PAIR_QUEUE_DEPTH: usize = 4;
const VIZ_QUEUE_DEPTH: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let pairing_window =
        PairingWindowNs::new(5_000_000).expect("pairing window must be positive");
    let mut pairer = StereoPairer::new(pairing_window);

    let pair_capacity = ChannelCapacity::try_from(PAIR_QUEUE_DEPTH)?;
    let (pair_tx, pair_rx, pair_stats) = bounded_channel(pair_capacity, DropPolicy::DropOldest);

    let viz_capacity = ChannelCapacity::try_from(VIZ_QUEUE_DEPTH)?;
    let (viz_tx, viz_rx, viz_stats) = bounded_channel(viz_capacity, DropPolicy::DropNewest);

    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    let superpoint = SuperPoint::new(model_dir.join("sp.onnx"))?;
    let lightglue = LightGlue::new(model_dir.join("lg.onnx"))?;
    let key_limit = KeypointLimit::try_from(MAX_KEYPOINTS)?;

    let inference_handle = thread::spawn(move || {
        let mut pipeline = InferencePipeline::new(superpoint, lightglue, key_limit);
        for pair in pair_rx.iter() {
            match pipeline.process_pair(pair) {
                Ok(packet) => {
                    if matches!(viz_tx.try_send(packet), SendOutcome::Disconnected) {
                        break;
                    }
                }
                Err(err) => {
                    eprintln!("inference error: {err}");
                }
            }
        }
    });

    let viz_handle = thread::spawn(move || {
        let rec = match rerun::RecordingStreamBuilder::new("kiko-slam-live").connect_grpc() {
            Ok(rec) => rec,
            Err(err) => {
                eprintln!("failed to connect to rerun viewer: {err}");
                return;
            }
        };

        let mut sink = RerunSink::new(rec, VizDecimation::default());
        for packet in viz_rx.iter() {
            if let Err(err) = sink.log(&packet) {
                eprintln!("rerun log error: {err}");
            }
        }
    });

    let mut left_seq = 0u64;
    let mut right_seq = 0u64;

    eprintln!("streaming matches... press ctrl+c to stop");

    while running.load(Ordering::Relaxed) {
        let mut got_any = false;

        match device.mono_left(0) {
            Ok(frame) => {
                pairer.push_left(oak_to_frame(
                    frame,
                    kiko_slam::SensorId::StereoLeft,
                    kiko_slam::FrameId::new(left_seq),
                ));
                left_seq += 1;
                got_any = true;
            }
            Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
            Err(e) => {
                eprintln!("left error: {:?}", e);
                break;
            }
        }

        match device.mono_right(0) {
            Ok(frame) => {
                pairer.push_right(oak_to_frame(
                    frame,
                    kiko_slam::SensorId::StereoRight,
                    kiko_slam::FrameId::new(right_seq),
                ));
                right_seq += 1;
                got_any = true;
            }
            Err(ImageError::Timeout { .. } | ImageError::QueueEmpty) => {}
            Err(e) => {
                eprintln!("right error: {:?}", e);
                break;
            }
        }

        while let Some(pair) = pairer.next_pair()? {
            if matches!(pair_tx.try_send(pair), SendOutcome::Disconnected) {
                break;
            }
        }

        if !got_any {
            thread::sleep(Duration::from_micros(500));
        }
    }

    drop(pair_tx);
    inference_handle.join().ok();
    viz_handle.join().ok();

    let pair_snapshot = pair_stats.snapshot();
    let viz_snapshot = viz_stats.snapshot();
    eprintln!(
        "pair queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}",
        pair_snapshot.enqueued,
        pair_snapshot.dropped_oldest,
        pair_snapshot.dropped_newest,
        pair_snapshot.disconnected
    );
    eprintln!(
        "viz queue stats: enqueued={}, dropped_oldest={}, dropped_newest={}, disconnected={}",
        viz_snapshot.enqueued,
        viz_snapshot.dropped_oldest,
        viz_snapshot.dropped_newest,
        viz_snapshot.disconnected
    );

    Ok(())
}
