use std::time::Instant;

use clap::Args;

use kiko_slam::dataset::DatasetReader;
use kiko_slam::{
    RectifiedStereo, RerunSink, TriangulationConfig, TriangulationError, Triangulator,
};

use crate::args::{DatasetArgs, InferenceArgs, InferenceConfig, RectifyArgs, RerunArgs};
use crate::config::build_rectified_stereo_config;
use crate::rerun_recording;

#[derive(Args, Clone, Debug)]
#[command(about = "Visualize stereo feature matches on a recorded dataset")]
pub struct VizArgs {
    #[command(flatten)]
    pub inference: InferenceArgs,
    #[command(flatten)]
    pub rerun: RerunArgs,
    #[command(flatten)]
    pub rectify: RectifyArgs,
    #[command(flatten)]
    pub dataset: DatasetArgs,
}

pub fn run_viz(args: &VizArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = DatasetReader::open(&args.dataset.path)?;
    let stats = reader.stats()?;

    eprintln!("dataset: {}", args.dataset.path.display());
    eprintln!(
        "camera fps: left={:.2?} right={:.2?} paired={:.2?} (left={}, right={})",
        stats.left_fps, stats.right_fps, stats.paired_fps, stats.left_count, stats.right_count
    );

    let inference = InferenceConfig::from_args(&args.inference)?;

    let rectified = RectifiedStereo::from_calibration_with_config(
        reader.calibration(),
        build_rectified_stereo_config(&args.rectify),
    )?;
    let triangulator = Triangulator::new(rectified, TriangulationConfig::default());

    let mut sink = match rerun_recording(&args.rerun, "kiko-slam-dataset") {
        Ok(rec) => Some(RerunSink::new(rec, args.rerun.rerun_decimation)),
        Err(err) => {
            eprintln!("failed to initialize rerun; continuing headless: {err}");
            None
        }
    };

    let mut pipeline = inference.into_pipeline();

    let start = Instant::now();
    let mut attempted = 0usize;
    let mut processed = 0usize;
    let mut inference_errors = 0usize;
    let mut read_errors = 0usize;
    let mut triangulation_empty = 0usize;
    let mut triangulation_errors = 0usize;
    let mut triangulated_points = 0usize;
    let mut total_matches = 0usize;

    for pair in reader.pairs() {
        let pair = match pair {
            Ok(pair) => pair,
            Err(err) => {
                read_errors += 1;
                eprintln!("read error: {err}");
                continue;
            }
        };
        attempted += 1;

        match pipeline.process_pair(pair) {
            Ok(packet) => {
                total_matches += packet.matches().len();
                let mut keyframe = None;
                match triangulator.triangulate(packet.matches()) {
                    Ok(result) => {
                        triangulated_points += result.keyframe.landmarks().len();
                        keyframe = Some(result.keyframe);
                    }
                    Err(TriangulationError::NoLandmarks { .. }) => {
                        triangulation_empty += 1;
                    }
                    Err(err) => {
                        triangulation_errors += 1;
                        eprintln!("triangulation error: {err}");
                    }
                };

                let points = keyframe.as_ref().map(|kf| kf.landmarks());
                if let Some(sink) = sink.as_mut() {
                    if let Err(err) = sink.log_with_points(&packet, points) {
                        eprintln!("rerun log error: {err}");
                    }
                }
                processed += 1;
            }
            Err(err) => {
                inference_errors += 1;
                eprintln!("inference error: {err}");
            }
        }

        if let Some(limit) = args.dataset.max_pairs {
            if attempted >= limit {
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
    let avg_matches = if processed > 0 {
        total_matches as f64 / processed as f64
    } else {
        0.0
    };
    let avg_triangulated = if processed > 0 {
        triangulated_points as f64 / processed as f64
    } else {
        0.0
    };

    eprintln!(
        "done: attempted={attempted}, processed={processed}, elapsed={elapsed:.2}s, fps={fps:.2}, read_errors={read_errors}, inference_errors={inference_errors}, triangulation_empty={triangulation_empty}, triangulation_errors={triangulation_errors}, triangulated_points={triangulated_points}"
    );
    eprintln!("summary: avg_matches={avg_matches:.1}, avg_triangulated={avg_triangulated:.1}");

    Ok(())
}
