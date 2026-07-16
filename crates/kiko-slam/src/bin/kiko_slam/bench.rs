use std::time::{Duration, Instant};

use clap::Args;

use kiko_slam::InferencePipeline;
use kiko_slam::dataset::DatasetReader;

use crate::args::{DatasetArgs, InferenceArgs, InferenceConfig, InferencePurpose};
use crate::{
    RunFailureCapture, RunMetricsStatus, SkippedDatasetEntryError, report_error_chain,
    verify_run_integrity,
};

#[derive(Args, Clone, Debug)]
#[command(about = "Benchmark inference pipeline throughput")]
pub struct BenchArgs {
    #[command(flatten)]
    pub inference: InferenceArgs,
    #[command(flatten)]
    pub dataset: DatasetArgs,
    /// Successful pairs processed for warm-up but excluded from steady-state metrics
    #[arg(long, env = "KIKO_BENCH_WARMUP_PAIRS", default_value_t = 4)]
    pub warmup_pairs: usize,
}

#[derive(Default)]
pub(crate) struct BenchAccum {
    pub entries_attempted: usize,
    pub read_samples: usize,
    pub processed: usize,
    pub matches_nonzero: usize,
    pub total_matches: usize,
    pub sum_left_keypoints: usize,
    pub sum_right_keypoints: usize,
    pub read_errors: usize,
    pub pairing_errors: usize,
    pub inference_errors: usize,
    pub sum_read_left: Duration,
    pub sum_read_right: Duration,
    pub sum_pairing: Duration,
    pub sum_read_bytes: usize,
    pub sum_sp_left: Duration,
    pub sum_sp_right: Duration,
    pub sum_stereo_sp_wall: Duration,
    pub sum_lightglue: Duration,
    pub sum_fused_ort_invocation: Duration,
    pub sum_total_success: Duration,
    pub sum_inference_attempt: Duration,
    pub timings: Vec<BenchPairTimings>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BenchPairTimings {
    pub read_pair: Duration,
    pub read_bytes: usize,
    pub inference_attempt: Duration,
    pub superpoint_left: Duration,
    pub superpoint_right: Duration,
    pub stereo_superpoint_wall: Duration,
    pub lightglue: Duration,
    pub fused_ort_invocation: Duration,
    pub total_success: Duration,
}

impl BenchPairTimings {
    fn overhead(self) -> Duration {
        self.total_success.saturating_sub(
            self.stereo_superpoint_wall + self.lightglue + self.fused_ort_invocation,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DurationPercentiles {
    pub median_ms: f64,
    pub p95_ms: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BenchLatencySummary {
    pub samples: usize,
    pub read_pair: DurationPercentiles,
    pub inference_attempt: DurationPercentiles,
    pub superpoint_left: DurationPercentiles,
    pub superpoint_right: DurationPercentiles,
    pub stereo_superpoint_wall: DurationPercentiles,
    pub lightglue: DurationPercentiles,
    pub fused_ort_invocation: DurationPercentiles,
    pub overhead: DurationPercentiles,
    pub total_success: DurationPercentiles,
}

pub(crate) struct BenchSummary {
    pub read_samples: usize,
    pub processed: usize,
    pub wall_seconds: f64,
    pub wall_fps: f64,
    pub reader_stage_seconds: f64,
    pub reader_stage_fps: f64,
    pub reader_throughput_mb_s: f64,
    pub inference_attempt_seconds: f64,
    pub inference_attempt_fps: f64,
    pub successful_inference_seconds: f64,
    pub successful_inference_fps: f64,
    pub match_rate: f64,
    pub avg_matches_per_processed_pair: f64,
    pub avg_matches_per_nonzero_pair: f64,
    pub avg_left_keypoints: f64,
    pub avg_right_keypoints: f64,
    pub avg_sp_left_ms: f64,
    pub avg_sp_right_ms: f64,
    pub avg_stereo_sp_wall_ms: f64,
    pub avg_lightglue_ms: f64,
    pub avg_fused_ort_invocation_ms: f64,
    pub avg_total_success_ms: f64,
    pub avg_overhead_ms: f64,
    pub pct_stereo_sp_wall: f64,
    pub pct_lightglue: f64,
    pub pct_fused_ort_invocation: f64,
    pub pct_overhead: f64,
}

pub(crate) fn summarize_bench(accum: &BenchAccum, elapsed: Duration) -> BenchSummary {
    let wall_seconds = elapsed.as_secs_f64();
    let wall_fps = if wall_seconds > 0.0 {
        accum.processed as f64 / wall_seconds
    } else {
        0.0
    };

    let reader_stage = accum.sum_read_left + accum.sum_read_right + accum.sum_pairing;
    let reader_stage_seconds = reader_stage.as_secs_f64();
    let reader_stage_fps = if reader_stage_seconds > 0.0 {
        accum.read_samples as f64 / reader_stage_seconds
    } else {
        0.0
    };
    let reader_throughput_mb_s = if reader_stage_seconds > 0.0 {
        (accum.sum_read_bytes as f64 / (1024.0 * 1024.0)) / reader_stage_seconds
    } else {
        0.0
    };

    let inference_attempt_seconds = accum.sum_inference_attempt.as_secs_f64();
    let inference_attempt_fps = if inference_attempt_seconds > 0.0 {
        accum.read_samples as f64 / inference_attempt_seconds
    } else {
        0.0
    };

    let successful_inference_seconds = accum.sum_total_success.as_secs_f64();
    let successful_inference_fps = if successful_inference_seconds > 0.0 {
        accum.processed as f64 / successful_inference_seconds
    } else {
        0.0
    };

    let match_rate = if accum.processed > 0 {
        accum.matches_nonzero as f64 / accum.processed as f64
    } else {
        0.0
    };
    let avg_matches_per_processed_pair = if accum.processed > 0 {
        accum.total_matches as f64 / accum.processed as f64
    } else {
        0.0
    };
    let avg_matches_per_nonzero_pair = if accum.matches_nonzero > 0 {
        accum.total_matches as f64 / accum.matches_nonzero as f64
    } else {
        0.0
    };
    let avg_left_keypoints = if accum.processed > 0 {
        accum.sum_left_keypoints as f64 / accum.processed as f64
    } else {
        0.0
    };
    let avg_right_keypoints = if accum.processed > 0 {
        accum.sum_right_keypoints as f64 / accum.processed as f64
    } else {
        0.0
    };

    let denom = accum.processed as f64;
    let avg_sp_left_ms = if accum.processed > 0 {
        (accum.sum_sp_left.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let avg_sp_right_ms = if accum.processed > 0 {
        (accum.sum_sp_right.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let avg_stereo_sp_wall_ms = if accum.processed > 0 {
        (accum.sum_stereo_sp_wall.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let avg_lightglue_ms = if accum.processed > 0 {
        (accum.sum_lightglue.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let avg_fused_ort_invocation_ms = if accum.processed > 0 {
        (accum.sum_fused_ort_invocation.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let avg_total_success_ms = if accum.processed > 0 {
        (accum.sum_total_success.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let overhead = accum.sum_total_success.saturating_sub(
        accum.sum_stereo_sp_wall + accum.sum_lightglue + accum.sum_fused_ort_invocation,
    );
    let avg_overhead_ms = if accum.processed > 0 {
        (overhead.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let total_ms = accum.sum_total_success.as_secs_f64().max(1e-9);
    let pct_stereo_sp_wall = (accum.sum_stereo_sp_wall.as_secs_f64() / total_ms) * 100.0;
    let pct_lightglue = (accum.sum_lightglue.as_secs_f64() / total_ms) * 100.0;
    let pct_fused_ort_invocation =
        (accum.sum_fused_ort_invocation.as_secs_f64() / total_ms) * 100.0;
    let pct_overhead = (overhead.as_secs_f64() / total_ms) * 100.0;

    BenchSummary {
        read_samples: accum.read_samples,
        processed: accum.processed,
        wall_seconds,
        wall_fps,
        reader_stage_seconds,
        reader_stage_fps,
        reader_throughput_mb_s,
        inference_attempt_seconds,
        inference_attempt_fps,
        successful_inference_seconds,
        successful_inference_fps,
        match_rate,
        avg_matches_per_processed_pair,
        avg_matches_per_nonzero_pair,
        avg_left_keypoints,
        avg_right_keypoints,
        avg_sp_left_ms,
        avg_sp_right_ms,
        avg_stereo_sp_wall_ms,
        avg_lightglue_ms,
        avg_fused_ort_invocation_ms,
        avg_total_success_ms,
        avg_overhead_ms,
        pct_stereo_sp_wall,
        pct_lightglue,
        pct_fused_ort_invocation,
        pct_overhead,
    }
}

pub(crate) fn summarize_latencies(
    samples: &[BenchPairTimings],
    warmup_pairs: usize,
) -> Option<BenchLatencySummary> {
    let measured = samples.get(warmup_pairs..)?;
    if measured.is_empty() {
        return None;
    }

    Some(BenchLatencySummary {
        samples: measured.len(),
        read_pair: duration_percentiles(measured.iter().map(|sample| sample.read_pair))?,
        inference_attempt: duration_percentiles(
            measured.iter().map(|sample| sample.inference_attempt),
        )?,
        superpoint_left: duration_percentiles(
            measured.iter().map(|sample| sample.superpoint_left),
        )?,
        superpoint_right: duration_percentiles(
            measured.iter().map(|sample| sample.superpoint_right),
        )?,
        stereo_superpoint_wall: duration_percentiles(
            measured.iter().map(|sample| sample.stereo_superpoint_wall),
        )?,
        lightglue: duration_percentiles(measured.iter().map(|sample| sample.lightglue))?,
        fused_ort_invocation: duration_percentiles(
            measured.iter().map(|sample| sample.fused_ort_invocation),
        )?,
        overhead: duration_percentiles(measured.iter().copied().map(BenchPairTimings::overhead))?,
        total_success: duration_percentiles(measured.iter().map(|sample| sample.total_success))?,
    })
}

fn summarize_steady_stages(
    samples: &[BenchPairTimings],
    warmup_pairs: usize,
    elapsed: Duration,
) -> Option<BenchSummary> {
    let measured = samples.get(warmup_pairs..)?;
    if measured.is_empty() {
        return None;
    }
    let mut accum = BenchAccum {
        read_samples: measured.len(),
        processed: measured.len(),
        ..BenchAccum::default()
    };
    for sample in measured {
        accum.sum_read_left += sample.read_pair;
        accum.sum_read_bytes += sample.read_bytes;
        accum.sum_inference_attempt += sample.inference_attempt;
        accum.sum_sp_left += sample.superpoint_left;
        accum.sum_sp_right += sample.superpoint_right;
        accum.sum_stereo_sp_wall += sample.stereo_superpoint_wall;
        accum.sum_lightglue += sample.lightglue;
        accum.sum_fused_ort_invocation += sample.fused_ort_invocation;
        accum.sum_total_success += sample.total_success;
    }
    Some(summarize_bench(&accum, elapsed))
}

pub(crate) fn duration_percentiles(
    samples: impl Iterator<Item = Duration>,
) -> Option<DurationPercentiles> {
    let mut milliseconds: Vec<f64> = samples
        .map(|sample| sample.as_secs_f64() * 1000.0)
        .collect();
    milliseconds.sort_by(f64::total_cmp);

    if milliseconds.is_empty() {
        return None;
    }
    let middle = milliseconds.len() / 2;
    let median_ms = if milliseconds.len().is_multiple_of(2) {
        (milliseconds[middle - 1] + milliseconds[middle]) / 2.0
    } else {
        milliseconds[middle]
    };
    // Nearest-rank p95: rank=ceil(0.95*N), converted to a zero-based index.
    let p95_rank = milliseconds.len().saturating_mul(95).div_ceil(100);
    let p95_ms = milliseconds[p95_rank.saturating_sub(1)];

    Some(DurationPercentiles { median_ms, p95_ms })
}

#[derive(Debug)]
struct BenchmarkSelectionError {
    warmup_pairs: usize,
    selected_pairs: usize,
}

impl std::fmt::Display for BenchmarkSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "benchmark warm-up must leave at least one steady-state pair: selected={}, warmup_pairs={}",
            self.selected_pairs, self.warmup_pairs
        )
    }
}

impl std::error::Error for BenchmarkSelectionError {}

fn validate_benchmark_warmup(
    selected_pairs: usize,
    warmup_pairs: usize,
) -> Result<(), BenchmarkSelectionError> {
    if warmup_pairs >= selected_pairs {
        return Err(BenchmarkSelectionError {
            warmup_pairs,
            selected_pairs,
        });
    }
    Ok(())
}

#[derive(Debug)]
struct IncompleteBenchmarkError {
    expected_pairs: usize,
    entries_attempted: usize,
    read_samples: usize,
    processed: usize,
}

impl std::fmt::Display for IncompleteBenchmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "benchmark did not consume the selected dataset exactly: expected_pairs={}, entries_attempted={}, read_samples={}, processed={}",
            self.expected_pairs, self.entries_attempted, self.read_samples, self.processed
        )
    }
}

impl std::error::Error for IncompleteBenchmarkError {}

pub fn run_bench(args: &BenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    let dataset_path = &args.dataset.path;
    let open_start = Instant::now();
    let mut reader = DatasetReader::open(dataset_path)?;
    let open_time = open_start.elapsed();

    let stats_start = Instant::now();
    let stats = reader.stats();
    let stats_time = stats_start.elapsed();

    eprintln!("dataset: {}", dataset_path.display());
    eprintln!(
        "dataset validation + open: {:.2}ms",
        open_time.as_secs_f64() * 1000.0
    );
    eprintln!(
        "cached stats lookup: {:.2}ms",
        stats_time.as_secs_f64() * 1000.0
    );
    eprintln!(
        "camera fps: left={:.2?} right={:.2?} paired={:.2?} (left={}, right={})",
        stats.left_fps, stats.right_fps, stats.paired_fps, stats.left_count, stats.right_count
    );
    let expected_pairs = args.dataset.selected_pair_count(stats.paired_count)?;
    validate_benchmark_warmup(expected_pairs, args.warmup_pairs)?;
    eprintln!(
        "benchmark selection: available_pairs={} skip_frames={} expected_pairs={} warmup_pairs={} steady_pairs={}",
        stats.paired_count,
        args.dataset.skip_frames,
        expected_pairs,
        args.warmup_pairs,
        expected_pairs - args.warmup_pairs
    );

    let inference_init_start = Instant::now();
    let inference = InferenceConfig::from_args(&args.inference, InferencePurpose::Benchmark)?;
    let max_keypoints = inference.key_limit.get();
    let mut end2end_pipeline = inference.end2end;
    let stereo_superpoint = inference.superpoint_right.require_ready_if_applicable()?;
    let mut pipeline: Option<InferencePipeline> = if end2end_pipeline.is_none() {
        Some(
            InferencePipeline::new(
                inference.superpoint,
                inference.lightglue,
                inference.key_limit,
            )
            .with_downscale(inference.downscale)
            .with_stereo_superpoint_opt(stereo_superpoint),
        )
    } else {
        None
    };
    let inference_init = inference_init_start.elapsed();
    eprintln!(
        "inference session initialization: {:.2}ms",
        inference_init.as_secs_f64() * 1000.0
    );

    let mut accum = BenchAccum {
        timings: Vec::with_capacity(expected_pairs),
        ..BenchAccum::default()
    };
    let mut failure_sources = RunFailureCapture::default();

    let mut samples = reader.timed_pairs();
    for skipped in 0..args.dataset.skip_frames {
        match samples.next() {
            Some(Ok(_)) => {}
            Some(Err(source)) => {
                return Err(Box::new(SkippedDatasetEntryError {
                    command: "benchmark",
                    entry_number: skipped + 1,
                    requested_skip: args.dataset.skip_frames,
                    source,
                }));
            }
            None => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!(
                        "dataset ended while skipping benchmark pair {} of {}",
                        skipped + 1,
                        args.dataset.skip_frames
                    ),
                )
                .into());
            }
        }
    }

    let cpu_start = process_usage();
    let start = Instant::now();
    let mut steady_start = (args.warmup_pairs == 0).then(Instant::now);

    for _ in 0..expected_pairs {
        let Some(sample) = samples.next() else {
            break;
        };
        accum.entries_attempted += 1;
        let sample = match sample {
            Ok(sample) => sample,
            Err(err) => {
                let stage = match &err {
                    kiko_slam::dataset::DatasetError::PairingFailed { .. } => {
                        accum.pairing_errors += 1;
                        "stereo_pairing"
                    }
                    _ => {
                        accum.read_errors += 1;
                        "dataset_read"
                    }
                };
                report_error_chain("benchmark input error", &err);
                failure_sources.record(stage, err);
                continue;
            }
        };
        accum.read_samples += 1;
        let pair = sample.pair;
        accum.sum_read_left += sample.timings.left_read;
        accum.sum_read_right += sample.timings.right_read;
        accum.sum_pairing += sample.timings.pairing;
        accum.sum_read_bytes += sample.timings.left_bytes + sample.timings.right_bytes;

        let inference_attempt_start = Instant::now();
        let read_pair =
            sample.timings.left_read + sample.timings.right_read + sample.timings.pairing;
        let read_bytes = sample.timings.left_bytes + sample.timings.right_bytes;
        let mut successful_timings = None;

        if let Some(ref mut end2end) = end2end_pipeline {
            // End-to-end pipeline: single call for SP+LG fused
            let (left_frame, right_frame) = pair.into_parts();
            match end2end.match_pair(&left_frame, &right_frame, max_keypoints) {
                Ok((matches, timings)) => {
                    accum.sum_left_keypoints += matches.source_a().len();
                    accum.sum_right_keypoints += matches.source_b().len();
                    accum.total_matches += matches.len();
                    if !matches.is_empty() {
                        accum.matches_nonzero += 1;
                    }
                    accum.sum_total_success += timings.total;
                    accum.sum_fused_ort_invocation += timings.ort_invocation;
                    accum.processed += 1;
                    successful_timings = Some(BenchPairTimings {
                        read_pair,
                        read_bytes,
                        fused_ort_invocation: timings.ort_invocation,
                        total_success: timings.total,
                        ..BenchPairTimings::default()
                    });
                }
                Err(err) => {
                    accum.inference_errors += 1;
                    report_error_chain("benchmark inference error", &err);
                    failure_sources.record("inference", err);
                }
            }
        } else if let Some(ref mut p) = pipeline {
            match p.process_pair_timed(pair) {
                Ok((packet, timings)) => {
                    let matches = packet.matches();
                    accum.sum_left_keypoints += matches.source_a().len();
                    accum.sum_right_keypoints += matches.source_b().len();
                    accum.total_matches += matches.len();
                    if !matches.is_empty() {
                        accum.matches_nonzero += 1;
                    }
                    accum.sum_sp_left += timings.superpoint_left;
                    accum.sum_sp_right += timings.superpoint_right;
                    accum.sum_stereo_sp_wall += timings.stereo_superpoint_wall;
                    accum.sum_lightglue += timings.lightglue;
                    accum.sum_total_success += timings.total;
                    accum.processed += 1;
                    successful_timings = Some(BenchPairTimings {
                        read_pair,
                        read_bytes,
                        superpoint_left: timings.superpoint_left,
                        superpoint_right: timings.superpoint_right,
                        stereo_superpoint_wall: timings.stereo_superpoint_wall,
                        lightglue: timings.lightglue,
                        total_success: timings.total,
                        ..BenchPairTimings::default()
                    });
                }
                Err(err) => {
                    accum.inference_errors += 1;
                    report_error_chain("benchmark inference error", &err);
                    failure_sources.record("inference", err);
                }
            }
        }
        let inference_attempt = inference_attempt_start.elapsed();
        accum.sum_inference_attempt += inference_attempt;
        if let Some(mut timings) = successful_timings {
            timings.inference_attempt = inference_attempt;
            accum.timings.push(timings);
            if accum.processed == args.warmup_pairs {
                steady_start = Some(Instant::now());
            }
        }
    }
    let elapsed = start.elapsed();
    let steady_elapsed = steady_start.map(|start| start.elapsed());
    let cpu_end = process_usage();
    let summary = summarize_bench(&accum, elapsed);
    let latency_summary = summarize_latencies(&accum.timings, args.warmup_pairs);
    let steady_stage_summary = steady_elapsed
        .and_then(|elapsed| summarize_steady_stages(&accum.timings, args.warmup_pairs, elapsed));
    let steady_processed = accum.processed.saturating_sub(args.warmup_pairs);
    let steady_wall_seconds = steady_elapsed.map_or(0.0, |elapsed| elapsed.as_secs_f64());
    let steady_wall_fps = if steady_wall_seconds > 0.0 {
        steady_processed as f64 / steady_wall_seconds
    } else {
        0.0
    };

    let integrity = verify_run_integrity(
        "bench",
        "pairs",
        accum.processed,
        &[
            ("dataset_read", accum.read_errors),
            ("stereo_pairing", accum.pairing_errors),
            ("inference", accum.inference_errors),
        ],
        failure_sources,
    );
    let exact_completion = accum.entries_attempted == expected_pairs
        && accum.read_samples == expected_pairs
        && accum.processed == expected_pairs;
    let has_matches = accum.matches_nonzero > 0;
    let metrics_status =
        RunMetricsStatus::from_outcome(integrity.is_ok(), exact_completion, has_matches);
    eprintln!("benchmark metrics status: {}", metrics_status.as_str());

    eprintln!(
        "completion: expected_pairs={} entries_attempted={} read_samples={} processed={} warmup_processed={} steady_processed={}",
        expected_pairs,
        accum.entries_attempted,
        accum.read_samples,
        accum.processed,
        accum.processed.min(args.warmup_pairs),
        steady_processed
    );

    eprintln!(
        "selected-run pipeline wall fps (session initialization excluded): {:.2} (processed={}, elapsed={:.2}s)",
        summary.wall_fps, summary.processed, summary.wall_seconds
    );
    eprintln!(
        "steady pipeline wall fps: {:.2} (processed={}, warmup_excluded={}, elapsed={:.2}s)",
        steady_wall_fps, steady_processed, args.warmup_pairs, steady_wall_seconds
    );
    if let Some(ref steady) = steady_stage_summary {
        eprintln!(
            "steady reader stage fps: {:.2} (successful_samples={}, read_stage_time={:.2}s, throughput={:.2} MB/s)",
            steady.reader_stage_fps,
            steady.read_samples,
            steady.reader_stage_seconds,
            steady.reader_throughput_mb_s
        );
        eprintln!(
            "steady inference-attempt wall fps (successful pairs only): {:.2} (samples={}, attempt_time={:.2}s)",
            steady.inference_attempt_fps, steady.read_samples, steady.inference_attempt_seconds
        );
        eprintln!(
            "steady model-pipeline timing fps: {:.2} (samples={}, model_pipeline_time={:.2}s)",
            steady.successful_inference_fps, steady.processed, steady.successful_inference_seconds
        );
    } else {
        eprintln!("steady stage metrics: unavailable (no successful post-warm-up samples)");
    }
    eprintln!(
        "full-run matching: nonzero_pairs={}, match_rate={:.2} avg_matches_processed={:.1} avg_matches_nonzero={:.1}",
        accum.matches_nonzero,
        summary.match_rate,
        summary.avg_matches_per_processed_pair,
        summary.avg_matches_per_nonzero_pair
    );
    eprintln!(
        "full-run features: avg_left_keypoints={:.1} avg_right_keypoints={:.1}",
        summary.avg_left_keypoints, summary.avg_right_keypoints
    );
    eprintln!(
        "errors: read={} pairing={} inference={}",
        accum.read_errors, accum.pairing_errors, accum.inference_errors
    );

    if let Some(ref steady) = steady_stage_summary {
        eprintln!(
            "steady timings avg ms: sp_left_call={:.2} sp_right_call={:.2} stereo_sp_wall={:.2} lightglue_call={:.2} fused_ort_invocation={:.2} overhead={:.2} total_success={:.2}",
            steady.avg_sp_left_ms,
            steady.avg_sp_right_ms,
            steady.avg_stereo_sp_wall_ms,
            steady.avg_lightglue_ms,
            steady.avg_fused_ort_invocation_ms,
            steady.avg_overhead_ms,
            steady.avg_total_success_ms
        );
        eprintln!(
            "steady timings pct of successful call wall time: stereo_sp_wall={:.1}% lightglue_call={:.1}% fused_ort_invocation={:.1}% overhead={:.1}%",
            steady.pct_stereo_sp_wall,
            steady.pct_lightglue,
            steady.pct_fused_ort_invocation,
            steady.pct_overhead
        );
    }

    if let Some(latency) = latency_summary {
        eprintln!(
            "steady latency ms (median/p95, samples={}): read_pair={:.2}/{:.2} inference_attempt={:.2}/{:.2} sp_left_call={:.2}/{:.2} sp_right_call={:.2}/{:.2} stereo_sp_wall={:.2}/{:.2} lightglue_call={:.2}/{:.2} fused_ort_invocation={:.2}/{:.2} overhead={:.2}/{:.2} total_success={:.2}/{:.2}",
            latency.samples,
            latency.read_pair.median_ms,
            latency.read_pair.p95_ms,
            latency.inference_attempt.median_ms,
            latency.inference_attempt.p95_ms,
            latency.superpoint_left.median_ms,
            latency.superpoint_left.p95_ms,
            latency.superpoint_right.median_ms,
            latency.superpoint_right.p95_ms,
            latency.stereo_superpoint_wall.median_ms,
            latency.stereo_superpoint_wall.p95_ms,
            latency.lightglue.median_ms,
            latency.lightglue.p95_ms,
            latency.fused_ort_invocation.median_ms,
            latency.fused_ort_invocation.p95_ms,
            latency.overhead.median_ms,
            latency.overhead.p95_ms,
            latency.total_success.median_ms,
            latency.total_success.p95_ms,
        );
    }

    if let (Some(start_usage), Some(end_usage)) = (cpu_start, cpu_end) {
        let cpu_time = end_usage.cpu_time.saturating_sub(start_usage.cpu_time);
        let cpu_s = cpu_time.user.as_secs_f64() + cpu_time.sys.as_secs_f64();
        let core_equiv = if summary.wall_seconds > 0.0 {
            cpu_s / summary.wall_seconds
        } else {
            0.0
        };
        eprintln!(
            "cpu: user={:.2}ms sys={:.2}ms total={:.2}ms cpu_time_over_wall_pct={:.1} core_equiv={:.2}",
            cpu_time.user.as_secs_f64() * 1000.0,
            cpu_time.sys.as_secs_f64() * 1000.0,
            cpu_s * 1000.0,
            core_equiv * 100.0,
            core_equiv
        );
        if let Some(rss) = end_usage.max_rss_bytes {
            eprintln!("memory: max_rss={:.2} MB", (rss as f64) / (1024.0 * 1024.0));
        }
    }

    integrity?;
    if !exact_completion {
        return Err(Box::new(IncompleteBenchmarkError {
            expected_pairs,
            entries_attempted: accum.entries_attempted,
            read_samples: accum.read_samples,
            processed: accum.processed,
        }));
    }
    if !has_matches {
        return Err("no nonzero matches; check models/data".into());
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct CpuSnapshot {
    cpu_time: CpuTime,
    max_rss_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
struct CpuTime {
    user: Duration,
    sys: Duration,
}

impl CpuTime {
    fn saturating_sub(self, other: CpuTime) -> CpuTime {
        CpuTime {
            user: self.user.saturating_sub(other.user),
            sys: self.sys.saturating_sub(other.sys),
        }
    }
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn process_usage() -> Option<CpuSnapshot> {
    // SAFETY: `libc::rusage` is a plain-old-data C struct; zeroed is a valid
    // representation. `getrusage` writes into the provided pointer.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            return None;
        }
        let user = timeval_to_duration(usage.ru_utime);
        let sys = timeval_to_duration(usage.ru_stime);
        let max_rss_bytes = max_rss_bytes(usage.ru_maxrss);
        Some(CpuSnapshot {
            cpu_time: CpuTime { user, sys },
            max_rss_bytes,
        })
    }
}

#[cfg(not(unix))]
fn process_usage() -> Option<CpuSnapshot> {
    None
}

#[cfg(unix)]
fn timeval_to_duration(tv: libc::timeval) -> Duration {
    let secs = tv.tv_sec.max(0) as u64;
    let micros = tv.tv_usec.max(0) as u32;
    Duration::new(secs, micros * 1000)
}

#[cfg(unix)]
fn max_rss_bytes(raw: libc::c_long) -> Option<u64> {
    if raw <= 0 {
        return None;
    }
    let rss = raw as u64;
    if cfg!(target_os = "macos") {
        Some(rss)
    } else {
        Some(rss * 1024)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BenchPairTimings, duration_percentiles, summarize_latencies, summarize_steady_stages,
        validate_benchmark_warmup,
    };
    use std::time::Duration;

    #[test]
    fn latency_summary_excludes_warmup_and_uses_documented_percentiles() {
        let samples: Vec<_> = (1..=32)
            .map(|milliseconds| {
                let duration = Duration::from_millis(milliseconds);
                BenchPairTimings {
                    read_pair: duration,
                    read_bytes: 0,
                    inference_attempt: duration,
                    superpoint_left: duration,
                    superpoint_right: duration,
                    stereo_superpoint_wall: duration,
                    lightglue: duration,
                    fused_ort_invocation: duration,
                    total_success: duration,
                }
            })
            .collect();

        let summary = summarize_latencies(&samples, 4).expect("28 measured samples");
        assert_eq!(summary.samples, 28);
        assert_eq!(summary.total_success.median_ms, 18.5);
        assert_eq!(summary.total_success.p95_ms, 31.0);
    }

    #[test]
    fn steady_stage_summary_excludes_cold_samples() {
        let samples: Vec<_> = (0..6)
            .map(|index| {
                let cold = index < 2;
                BenchPairTimings {
                    read_pair: Duration::from_millis(if cold { 30 } else { 3 }),
                    read_bytes: if cold { 10_000 } else { 1_000 },
                    inference_attempt: Duration::from_millis(if cold { 100 } else { 10 }),
                    stereo_superpoint_wall: Duration::from_millis(if cold { 50 } else { 5 }),
                    lightglue: Duration::from_millis(if cold { 20 } else { 2 }),
                    fused_ort_invocation: Duration::from_millis(if cold { 10 } else { 1 }),
                    total_success: Duration::from_millis(if cold { 100 } else { 10 }),
                    ..BenchPairTimings::default()
                }
            })
            .collect();

        let summary = summarize_steady_stages(&samples, 2, Duration::from_millis(80))
            .expect("four steady samples");
        assert_eq!(summary.processed, 4);
        assert_eq!(summary.read_samples, 4);
        assert_eq!(summary.avg_total_success_ms, 10.0);
        assert_eq!(summary.successful_inference_seconds, 0.04);
        assert_eq!(summary.reader_stage_seconds, 0.012);
        assert_eq!(summary.wall_fps, 50.0);
    }

    #[test]
    fn benchmark_warmup_must_leave_a_measured_pair() {
        assert!(validate_benchmark_warmup(2_084, 4).is_ok());
        assert!(validate_benchmark_warmup(4, 4).is_err());
    }

    #[test]
    fn percentile_summary_rejects_empty_input() {
        assert_eq!(duration_percentiles(std::iter::empty()), None);
    }

    #[test]
    fn parallel_invocation_latencies_are_not_added_to_overhead() {
        let sample = BenchPairTimings {
            superpoint_left: Duration::from_millis(10),
            superpoint_right: Duration::from_millis(9),
            stereo_superpoint_wall: Duration::from_millis(10),
            lightglue: Duration::from_millis(1),
            total_success: Duration::from_millis(12),
            ..BenchPairTimings::default()
        };
        assert_eq!(sample.overhead(), Duration::from_millis(1));
    }
}
