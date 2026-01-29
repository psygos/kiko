use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use kiko_slam::dataset::DatasetReader;
use kiko_slam::{
    DownscaleFactor, InferenceBackend, InferencePipeline, KeypointLimit, LightGlue, SuperPoint,
};

const MAX_KEYPOINTS: usize = 1024;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: bench_dataset <dataset_path> [max_pairs]");
        std::process::exit(1);
    }

    let dataset_path = &args[1];
    let max_pairs = args.get(2).and_then(|s| s.parse::<usize>().ok());

    let open_start = Instant::now();
    let mut reader = DatasetReader::open(dataset_path)?;
    let open_time = open_start.elapsed();

    let stats_start = Instant::now();
    let stats = reader.stats()?;
    let stats_time = stats_start.elapsed();

    eprintln!("dataset: {dataset_path}");
    eprintln!("dataset open: {:.2}ms", open_time.as_secs_f64() * 1000.0);
    eprintln!("scan frames: {:.2}ms", stats_time.as_secs_f64() * 1000.0);
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
    let mut pipeline = InferencePipeline::new(
        superpoint_left,
        superpoint_right,
        lightglue,
        key_limit,
    )
    .with_downscale(downscale);

    let cpu_start = process_usage();
    let mut processed = 0usize;
    let mut matches_nonzero = 0usize;
    let mut total_matches = 0usize;
    let mut read_errors = 0usize;
    let mut pairing_errors = 0usize;
    let mut inference_errors = 0usize;
    let mut sum_read_left = Duration::ZERO;
    let mut sum_read_right = Duration::ZERO;
    let mut sum_pairing = Duration::ZERO;
    let mut sum_read_bytes = 0usize;
    let mut sum_sp_left = Duration::ZERO;
    let mut sum_sp_right = Duration::ZERO;
    let mut sum_lightglue = Duration::ZERO;
    let mut sum_total = Duration::ZERO;

    let start = Instant::now();
    for sample in reader.timed_pairs() {
        let sample = match sample {
            Ok(sample) => sample,
            Err(err) => {
                match err {
                    kiko_slam::dataset::DatasetError::PairingFailed { .. } => {
                        pairing_errors += 1;
                    }
                    _ => read_errors += 1,
                }
                eprintln!("read error: {err}");
                continue;
            }
        };
        let pair = sample.pair;
        sum_read_left += sample.timings.left_read;
        sum_read_right += sample.timings.right_read;
        sum_pairing += sample.timings.pairing;
        sum_read_bytes += sample.timings.left_bytes + sample.timings.right_bytes;

        match pipeline.process_pair_timed(pair) {
            Ok((packet, timings)) => {
                let matches = packet.matches();
                if !matches.is_empty() {
                    matches_nonzero += 1;
                    total_matches += matches.len();
                }
                sum_sp_left += timings.superpoint_left;
                sum_sp_right += timings.superpoint_right;
                sum_lightglue += timings.lightglue;
                sum_total += timings.total;
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
    let elapsed = start.elapsed();
    let cpu_end = process_usage();
    let elapsed_s = elapsed.as_secs_f64();
    let fps = if elapsed_s > 0.0 {
        processed as f64 / elapsed_s
    } else {
        0.0
    };
    let infer_s = sum_total.as_secs_f64();
    let infer_fps = if infer_s > 0.0 {
        processed as f64 / infer_s
    } else {
        0.0
    };

    let match_rate = if processed > 0 {
        matches_nonzero as f64 / processed as f64
    } else {
        0.0
    };
    let avg_matches = if matches_nonzero > 0 {
        total_matches as f64 / matches_nonzero as f64
    } else {
        0.0
    };

    let read_total = sum_read_left + sum_read_right + sum_pairing;
    let read_s = read_total.as_secs_f64();
    let read_fps = if read_s > 0.0 {
        processed as f64 / read_s
    } else {
        0.0
    };
    let read_mb_s = if read_s > 0.0 {
        (sum_read_bytes as f64 / (1024.0 * 1024.0)) / read_s
    } else {
        0.0
    };

    eprintln!(
        "pipeline fps: {:.2} (processed={}, elapsed={:.2}s)",
        fps, processed, elapsed_s
    );
    eprintln!(
        "reader fps: {:.2} (read_time={:.2}s, throughput={:.2} MB/s)",
        read_fps, read_s, read_mb_s
    );
    eprintln!(
        "inference fps: {:.2} (sum_infer_time={:.2}s)",
        infer_fps, infer_s
    );
    eprintln!(
        "matching: nonzero_pairs={}, match_rate={:.2} avg_matches={:.1}",
        matches_nonzero, match_rate, avg_matches
    );
    eprintln!(
        "errors: read={} pairing={} inference={}",
        read_errors, pairing_errors, inference_errors
    );

    if processed > 0 {
        let denom = processed as f64;
        let avg_sp_left_ms = (sum_sp_left.as_secs_f64() * 1000.0) / denom;
        let avg_sp_right_ms = (sum_sp_right.as_secs_f64() * 1000.0) / denom;
        let avg_lightglue_ms = (sum_lightglue.as_secs_f64() * 1000.0) / denom;
        let avg_total_ms = (sum_total.as_secs_f64() * 1000.0) / denom;
        let overhead = sum_total
            .saturating_sub(sum_sp_left + sum_sp_right + sum_lightglue);
        let avg_overhead_ms = (overhead.as_secs_f64() * 1000.0) / denom;
        let total_ms = sum_total.as_secs_f64().max(1e-9);
        let pct_sp_left = (sum_sp_left.as_secs_f64() / total_ms) * 100.0;
        let pct_sp_right = (sum_sp_right.as_secs_f64() / total_ms) * 100.0;
        let pct_lightglue = (sum_lightglue.as_secs_f64() / total_ms) * 100.0;
        let pct_overhead = (overhead.as_secs_f64() / total_ms) * 100.0;

        eprintln!("timings avg ms: sp_left={:.2} sp_right={:.2} lightglue={:.2} overhead={:.2} total={:.2}",
            avg_sp_left_ms, avg_sp_right_ms, avg_lightglue_ms, avg_overhead_ms, avg_total_ms);
        eprintln!(
            "timings pct: sp_left={:.1}% sp_right={:.1}% lightglue={:.1}% overhead={:.1}%",
            pct_sp_left, pct_sp_right, pct_lightglue, pct_overhead
        );
    }

    if let (Some(start_usage), Some(end_usage)) = (cpu_start, cpu_end) {
        let cpu_time = end_usage.cpu_time.saturating_sub(start_usage.cpu_time);
        let cpu_s = cpu_time.user.as_secs_f64() + cpu_time.sys.as_secs_f64();
        let cpu_pct = if elapsed_s > 0.0 {
            (cpu_s / elapsed_s) * 100.0
        } else {
            0.0
        };
        eprintln!(
            "cpu: user={:.2}ms sys={:.2}ms total={:.2}ms cpu%={:.1}",
            cpu_time.user.as_secs_f64() * 1000.0,
            cpu_time.sys.as_secs_f64() * 1000.0,
            cpu_s * 1000.0,
            cpu_pct
        );
        if let Some(rss) = end_usage.max_rss_bytes {
            eprintln!(
                "memory: max_rss={:.2} MB",
                (rss as f64) / (1024.0 * 1024.0)
            );
        }
    }

    if processed == 0 {
        return Err("no paired frames processed".into());
    }
    if matches_nonzero == 0 {
        return Err("no nonzero matches; check models/data".into());
    }
    if inference_errors > 0 {
        return Err("inference errors encountered during run".into());
    }

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
fn process_usage() -> Option<CpuSnapshot> {
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
