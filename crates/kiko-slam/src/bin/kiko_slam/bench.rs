use std::time::{Duration, Instant};

use clap::Args;

use kiko_slam::InferencePipeline;
use kiko_slam::dataset::DatasetReader;

use crate::args::{DatasetArgs, InferenceArgs, InferenceConfig};

#[derive(Args, Clone, Debug)]
#[command(about = "Benchmark inference pipeline throughput")]
pub struct BenchArgs {
    #[command(flatten)]
    pub inference: InferenceArgs,
    #[command(flatten)]
    pub dataset: DatasetArgs,
}

#[derive(Default)]
pub(crate) struct BenchAccum {
    pub read_samples: usize,
    pub processed: usize,
    pub matches_nonzero: usize,
    pub total_matches: usize,
    pub read_errors: usize,
    pub pairing_errors: usize,
    pub inference_errors: usize,
    pub sum_read_left: Duration,
    pub sum_read_right: Duration,
    pub sum_pairing: Duration,
    pub sum_read_bytes: usize,
    pub sum_sp_left: Duration,
    pub sum_sp_right: Duration,
    pub sum_lightglue: Duration,
    pub sum_total_success: Duration,
    pub sum_inference_attempt: Duration,
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
    pub avg_sp_left_ms: f64,
    pub avg_sp_right_ms: f64,
    pub avg_lightglue_ms: f64,
    pub avg_total_success_ms: f64,
    pub avg_overhead_ms: f64,
    pub pct_sp_left: f64,
    pub pct_sp_right: f64,
    pub pct_lightglue: f64,
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
    let avg_lightglue_ms = if accum.processed > 0 {
        (accum.sum_lightglue.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let avg_total_success_ms = if accum.processed > 0 {
        (accum.sum_total_success.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let overhead = accum
        .sum_total_success
        .saturating_sub(accum.sum_sp_left + accum.sum_sp_right + accum.sum_lightglue);
    let avg_overhead_ms = if accum.processed > 0 {
        (overhead.as_secs_f64() * 1000.0) / denom
    } else {
        0.0
    };
    let total_ms = accum.sum_total_success.as_secs_f64().max(1e-9);
    let pct_sp_left = (accum.sum_sp_left.as_secs_f64() / total_ms) * 100.0;
    let pct_sp_right = (accum.sum_sp_right.as_secs_f64() / total_ms) * 100.0;
    let pct_lightglue = (accum.sum_lightglue.as_secs_f64() / total_ms) * 100.0;
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
        avg_sp_left_ms,
        avg_sp_right_ms,
        avg_lightglue_ms,
        avg_total_success_ms,
        avg_overhead_ms,
        pct_sp_left,
        pct_sp_right,
        pct_lightglue,
        pct_overhead,
    }
}

pub fn run_bench(args: &BenchArgs) -> Result<(), Box<dyn std::error::Error>> {
    let dataset_path = &args.dataset.path;
    let open_start = Instant::now();
    let mut reader = DatasetReader::open(dataset_path)?;
    let open_time = open_start.elapsed();

    let stats_start = Instant::now();
    let stats = reader.stats()?;
    let stats_time = stats_start.elapsed();

    eprintln!("dataset: {}", dataset_path.display());
    eprintln!("dataset open: {:.2}ms", open_time.as_secs_f64() * 1000.0);
    eprintln!("scan frames: {:.2}ms", stats_time.as_secs_f64() * 1000.0);
    eprintln!(
        "camera fps: left={:.2?} right={:.2?} paired={:.2?} (left={}, right={})",
        stats.left_fps, stats.right_fps, stats.paired_fps, stats.left_count, stats.right_count
    );

    let inference = InferenceConfig::from_args(&args.inference)?;
    let downscale = inference.downscale;
    let max_keypoints = inference.key_limit.get();
    let mut end2end_pipeline = inference.end2end;
    let mut pipeline: Option<InferencePipeline> = if end2end_pipeline.is_none() {
        Some(
            InferencePipeline::new(
                inference.superpoint,
                inference.lightglue,
                inference.key_limit,
            )
            .with_downscale(inference.downscale)
            .with_stereo_superpoint_opt(inference.superpoint_right),
        )
    } else {
        None
    };

    let cpu_start = process_usage();
    let mut accum = BenchAccum::default();

    let start = Instant::now();
    for sample in reader.timed_pairs() {
        let sample = match sample {
            Ok(sample) => sample,
            Err(err) => {
                match err {
                    kiko_slam::dataset::DatasetError::PairingFailed { .. } => {
                        accum.pairing_errors += 1;
                    }
                    _ => accum.read_errors += 1,
                }
                eprintln!("read error: {err}");
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

        if let Some(ref mut end2end) = end2end_pipeline {
            // End-to-end pipeline: single call for SP+LG fused
            let (left_frame, right_frame) = pair.into_parts();
            match end2end.match_pair(&left_frame, &right_frame, max_keypoints) {
                Ok((matches, timings)) => {
                    accum.total_matches += matches.len();
                    if !matches.is_empty() {
                        accum.matches_nonzero += 1;
                    }
                    accum.sum_total_success += timings.total;
                    accum.sum_sp_left += timings.total; // attribute all to "pipeline"
                    accum.processed += 1;
                }
                Err(err) => {
                    accum.inference_errors += 1;
                    eprintln!("inference error: {err}");
                }
            }
        } else if let Some(ref mut p) = pipeline {
            match p.process_pair_timed(pair) {
                Ok((packet, timings)) => {
                    let matches = packet.matches();
                    accum.total_matches += matches.len();
                    if !matches.is_empty() {
                        accum.matches_nonzero += 1;
                    }
                    accum.sum_sp_left += timings.superpoint_left;
                    accum.sum_sp_right += timings.superpoint_right;
                    accum.sum_lightglue += timings.lightglue;
                    accum.sum_total_success += timings.total;
                    accum.processed += 1;
                }
                Err(err) => {
                    accum.inference_errors += 1;
                    eprintln!("inference error: {err}");
                }
            }
        }
        accum.sum_inference_attempt += inference_attempt_start.elapsed();

        if let Some(limit) = args.dataset.max_pairs {
            if accum.read_samples >= limit {
                break;
            }
        }
    }
    let elapsed = start.elapsed();
    let cpu_end = process_usage();
    let summary = summarize_bench(&accum, elapsed);

    eprintln!(
        "pipeline wall fps: {:.2} (processed={}, elapsed={:.2}s)",
        summary.wall_fps, summary.processed, summary.wall_seconds
    );
    eprintln!(
        "reader stage fps: {:.2} (read_samples={}, read_stage_time={:.2}s, throughput={:.2} MB/s)",
        summary.reader_stage_fps,
        summary.read_samples,
        summary.reader_stage_seconds,
        summary.reader_throughput_mb_s
    );
    eprintln!(
        "inference attempt fps: {:.2} (attempts={}, attempt_time={:.2}s)",
        summary.inference_attempt_fps, summary.read_samples, summary.inference_attempt_seconds
    );
    eprintln!(
        "successful inference fps: {:.2} (processed={}, successful_infer_time={:.2}s)",
        summary.successful_inference_fps, summary.processed, summary.successful_inference_seconds
    );
    eprintln!(
        "matching: nonzero_pairs={}, match_rate={:.2} avg_matches_processed={:.1} avg_matches_nonzero={:.1}",
        accum.matches_nonzero,
        summary.match_rate,
        summary.avg_matches_per_processed_pair,
        summary.avg_matches_per_nonzero_pair
    );
    eprintln!(
        "errors: read={} pairing={} inference={}",
        accum.read_errors, accum.pairing_errors, accum.inference_errors
    );

    if accum.processed > 0 {
        eprintln!(
            "timings avg ms: sp_left={:.2} sp_right={:.2} lightglue={:.2} overhead={:.2} total_success={:.2}",
            summary.avg_sp_left_ms,
            summary.avg_sp_right_ms,
            summary.avg_lightglue_ms,
            summary.avg_overhead_ms,
            summary.avg_total_success_ms
        );
        eprintln!(
            "timings pct of successful inference time: sp_left={:.1}% sp_right={:.1}% lightglue={:.1}% overhead={:.1}%",
            summary.pct_sp_left, summary.pct_sp_right, summary.pct_lightglue, summary.pct_overhead
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

    if accum.processed == 0 {
        return Err("no paired frames processed".into());
    }
    if accum.matches_nonzero == 0 {
        return Err("no nonzero matches; check models/data".into());
    }
    if accum.inference_errors > 0 {
        return Err("inference errors encountered during run".into());
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
