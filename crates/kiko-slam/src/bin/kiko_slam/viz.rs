use std::time::Instant;

use clap::Args;

use kiko_slam::dataset::DatasetReader;
use kiko_slam::{
    RerunSink, RerunSinkConfig, TriangulationConfig, TriangulationError, Triangulator, VizLogError,
};

use crate::args::{
    DatasetArgs, InferenceArgs, InferenceConfig, InferencePurpose, RerunArgs, RerunOutput,
};
use crate::{
    RunFailureCapture, RunMetricsStatus, SkippedDatasetEntryError, combine_rerun_results,
    next_selected_success, rerun_recording, verify_run_integrity,
};

#[derive(Args, Clone, Debug)]
#[command(about = "Visualize stereo feature matches on a recorded dataset")]
pub struct VizArgs {
    #[command(flatten)]
    pub inference: InferenceArgs,
    #[command(flatten)]
    pub rerun: RerunArgs,
    #[command(flatten)]
    pub dataset: DatasetArgs,
}

pub fn run_viz(args: &VizArgs) -> Result<(), Box<dyn std::error::Error>> {
    let rerun_output = RerunOutput::try_from_args(&args.rerun)?;
    let sink_config = RerunSinkConfig::from_environment()?;
    let mut reader = DatasetReader::open(&args.dataset.path)?;
    let rectified = reader.calibration().rectified_stereo()?;
    let stats = reader.stats();

    eprintln!("dataset: {}", args.dataset.path.display());
    eprintln!(
        "camera fps: left={:.2?} right={:.2?} paired={:.2?} (left={}, right={})",
        stats.left_fps, stats.right_fps, stats.paired_fps, stats.left_count, stats.right_count
    );
    let expected_pairs = args.dataset.selected_pair_count(stats.paired_count)?;
    eprintln!(
        "visualization selection: available_pairs={} skip_frames={} expected_pairs={}",
        stats.paired_count, args.dataset.skip_frames, expected_pairs
    );

    let inference = InferenceConfig::from_args(&args.inference, InferencePurpose::Visualization)?;

    let triangulator = Triangulator::new(rectified, TriangulationConfig::default());

    let rec = rerun_recording(rerun_output.destination(), "kiko-slam-dataset")?;
    let mut sink = RerunSink::from_config(rec, rerun_output.decimation(), sink_config);

    let mut pipeline = inference.into_pipeline()?;

    let mut pairs = reader.pairs();
    for skipped in 0..args.dataset.skip_frames {
        match pairs.next() {
            Some(Ok(_)) => {}
            Some(Err(source)) => {
                return Err(Box::new(SkippedDatasetEntryError {
                    command: "visualization",
                    entry_number: skipped + 1,
                    requested_skip: args.dataset.skip_frames,
                    source,
                }));
            }
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "visualization dataset ended while skipping entry {} of {}",
                        skipped + 1,
                        args.dataset.skip_frames
                    ),
                )
                .into());
            }
        }
    }

    let start = Instant::now();
    let mut entries_consumed = 0usize;
    let mut inference_attempts = 0usize;
    let mut processed = 0usize;
    let mut inference_errors = 0usize;
    let mut read_errors = 0usize;
    let mut triangulation_empty = 0usize;
    let mut triangulation_errors = 0usize;
    let mut triangulated_points = 0usize;
    let mut total_matches = 0usize;
    let mut failure_sources = RunFailureCapture::default();

    let mut next_pair =
        next_selected_success(&mut pairs, &mut entries_consumed, expected_pairs, |err| {
            failure_sources.report_and_record(
                &mut read_errors,
                "dataset_read",
                "visualization input error",
                err,
            );
        });

    let processing = (|| -> Result<(), VizLogError> {
        while let Some(pair) = next_pair.take() {
            inference_attempts += 1;

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
                            failure_sources.report_and_record(
                                &mut triangulation_errors,
                                "triangulation",
                                "triangulation error",
                                err,
                            );
                        }
                    };

                    let points = keyframe.as_ref().map(|kf| kf.landmarks());
                    sink.log_with_points(&packet, points)?;
                    processed += 1;
                }
                Err(err) => {
                    failure_sources.report_and_record(
                        &mut inference_errors,
                        "inference",
                        "visualization inference error",
                        err,
                    );
                }
            }

            next_pair =
                next_selected_success(&mut pairs, &mut entries_consumed, expected_pairs, |err| {
                    failure_sources.report_and_record(
                        &mut read_errors,
                        "dataset_read",
                        "visualization input error",
                        err,
                    );
                });
        }
        Ok(())
    })();
    let finalization = sink.finish_with_timeout(rerun_output.finish_timeout().get());
    combine_rerun_results(processing, finalization)?;

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

    let integrity = verify_run_integrity(
        "viz",
        "pairs",
        processed,
        &[
            ("dataset_read", read_errors),
            ("inference", inference_errors),
            ("triangulation", triangulation_errors),
        ],
        failure_sources,
    );
    let exact_completion = entries_consumed == expected_pairs
        && inference_attempts == expected_pairs
        && processed == expected_pairs;
    let metrics_status = RunMetricsStatus::from_outcome(integrity.is_ok(), exact_completion, true);
    eprintln!("visualization metrics status: {}", metrics_status.as_str());
    eprintln!(
        "done: expected={expected_pairs}, entries_consumed={entries_consumed}, inference_attempts={inference_attempts}, processed={processed}, elapsed={elapsed:.2}s, fps={fps:.2}, read_errors={read_errors}, inference_errors={inference_errors}, triangulation_empty={triangulation_empty}, triangulation_errors={triangulation_errors}, triangulated_points={triangulated_points}"
    );
    eprintln!("summary: avg_matches={avg_matches:.1}, avg_triangulated={avg_triangulated:.1}");

    integrity?;
    if !exact_completion {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "visualization did not consume the selected dataset exactly: expected_pairs={expected_pairs}, entries_consumed={entries_consumed}, inference_attempts={inference_attempts}, processed={processed}"
            ),
        )
        .into());
    }

    Ok(())
}
