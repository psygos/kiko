use std::path::{Path, PathBuf};
use std::time::Instant;

use kiko_slam::dataset::DatasetReader;
use kiko_slam::{
    DownscaleFactor, InferenceBackend, InferencePipeline, KeypointLimit, LightGlue, RerunSink,
    SuperPoint, VizDecimation,
};

const MAX_KEYPOINTS: usize = 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: viz_dataset <dataset_path> [max_pairs]");
        std::process::exit(1);
    }

    let dataset_path = &args[1];
    let max_pairs = args.get(2).and_then(|s| s.parse::<usize>().ok());

    let mut reader = DatasetReader::open(dataset_path)?;
    let stats = reader.stats()?;

    eprintln!("dataset: {dataset_path}");
    eprintln!(
        "camera fps: left={:.2?} right={:.2?} paired={:.2?} (left={}, right={})",
        stats.left_fps, stats.right_fps, stats.paired_fps, stats.left_count, stats.right_count
    );

    let default_backend = env_backend("KIKO_BACKEND").unwrap_or(InferenceBackend::auto());
    let superpoint_backend = env_backend("KIKO_SUPERPOINT_BACKEND").unwrap_or(default_backend);
    let lightglue_backend = env_backend("KIKO_LIGHTGLUE_BACKEND").unwrap_or(default_backend);

    let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models");
    let sp_path = model_path(&model_dir, "KIKO_SUPERPOINT_MODEL", "sp.onnx");
    let lg_path = model_path(&model_dir, "KIKO_LIGHTGLUE_MODEL", "lg.onnx");
    eprintln!(
        "models: superpoint={} lightglue={}",
        sp_path.display(),
        lg_path.display()
    );

    let superpoint_left = SuperPoint::new_with_backend(&sp_path, superpoint_backend)?;
    let superpoint_right = SuperPoint::new_with_backend(&sp_path, superpoint_backend)?;
    let lightglue = LightGlue::new_with_backend(&lg_path, lightglue_backend)?;
    eprintln!(
        "inference backend: superpoint={:?}, lightglue={:?}",
        superpoint_left.backend(),
        lightglue.backend()
    );

    let max_keypoints = env_usize("KIKO_MAX_KEYPOINTS").unwrap_or(MAX_KEYPOINTS);
    let key_limit = KeypointLimit::try_from(max_keypoints)?;
    let downscale = env_usize("KIKO_DOWNSCALE")
        .map(DownscaleFactor::try_from)
        .transpose()
        .map_err(|e| format!("invalid KIKO_DOWNSCALE: {e}"))?
        .unwrap_or_else(DownscaleFactor::identity);
    eprintln!("max_keypoints: {}", max_keypoints);
    eprintln!("downscale: {}", downscale.get());

    let decimation = env_usize("KIKO_RERUN_DECIMATION")
        .map(VizDecimation::try_from)
        .transpose()
        .map_err(|e| format!("invalid KIKO_RERUN_DECIMATION: {e}"))?
        .unwrap_or_default();

    let rec = rerun::RecordingStreamBuilder::new("kiko-slam-dataset").connect_grpc()?;
    let mut sink = RerunSink::new(rec, decimation);

    let mut pipeline = InferencePipeline::new(
        superpoint_left,
        superpoint_right,
        lightglue,
        key_limit,
    )
    .with_downscale(downscale);

    let start = Instant::now();
    let mut processed = 0usize;
    let mut inference_errors = 0usize;
    let mut read_errors = 0usize;

    for pair in reader.pairs() {
        let pair = match pair {
            Ok(pair) => pair,
            Err(err) => {
                read_errors += 1;
                eprintln!("read error: {err}");
                continue;
            }
        };

        match pipeline.process_pair(pair) {
            Ok(packet) => {
                if let Err(err) = sink.log(&packet) {
                    eprintln!("rerun log error: {err}");
                }
                processed += 1;
            }
            Err(err) => {
                inference_errors += 1;
                eprintln!("inference error: {err}");
            }
        }

        if let Some(limit) = max_pairs {
            if processed >= limit {
                break;
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let fps = if elapsed > 0.0 {
        processed as f64 / elapsed
    } else {
        0.0
    };

    eprintln!(
        "done: processed={}, elapsed={:.2}s, fps={:.2}, read_errors={}, inference_errors={}",
        processed, elapsed, fps, read_errors, inference_errors
    );

    Ok(())
}

fn env_backend(key: &str) -> Option<InferenceBackend> {
    let raw = std::env::var(key).ok()?;
    match InferenceBackend::parse(&raw) {
        Some(backend) => Some(backend),
        None => {
            eprintln!("invalid {key}={raw}, ignoring");
            None
        }
    }
}

fn env_usize(key: &str) -> Option<usize> {
    let raw = std::env::var(key).ok()?;
    match raw.parse::<usize>() {
        Ok(value) => Some(value),
        Err(_) => {
            eprintln!("invalid {key}={raw}, ignoring");
            None
        }
    }
}

fn model_path(model_dir: &Path, key: &str, default_name: &str) -> PathBuf {
    match std::env::var(key) {
        Ok(value) => {
            let candidate = PathBuf::from(value);
            if candidate.is_absolute() {
                candidate
            } else {
                model_dir.join(candidate)
            }
        }
        Err(_) => model_dir.join(default_name),
    }
}
