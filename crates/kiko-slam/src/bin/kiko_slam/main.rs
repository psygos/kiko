use clap::{Parser, Subcommand};

mod args;
mod bench;
mod config;
#[cfg(feature = "record")]
mod live;
#[cfg(feature = "record")]
mod record;
mod slam;
mod viz;

#[derive(Parser, Debug)]
#[command(name = "kiko-slam", about = "Kiko SLAM tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run full SLAM pipeline on a recorded dataset
    Slam(slam::SlamArgs),
    /// Visualize stereo feature matches on a recorded dataset
    Viz(viz::VizArgs),
    /// Benchmark inference pipeline throughput
    Bench(bench::BenchArgs),
    /// Record stereo dataset from OAK-D camera
    #[cfg(feature = "record")]
    Record(record::RecordArgs),
    /// Run live SLAM from OAK-D camera
    #[cfg(feature = "record")]
    Live(live::LiveArgs),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Slam(args) => slam::run_slam(&args),
        Command::Viz(args) => viz::run_viz(&args),
        Command::Bench(args) => bench::run_bench(&args),
        #[cfg(feature = "record")]
        Command::Record(args) => record::run_record(&args),
        #[cfg(feature = "record")]
        Command::Live(args) => live::run_live(&args),
    }
}

pub fn rerun_recording(
    args: &args::RerunArgs,
    name: &str,
) -> Result<rerun::RecordingStream, Box<dyn std::error::Error>> {
    if let Some(path) = &args.save_rrd {
        let path = if path.is_dir() {
            path.join(format!("{name}.rrd"))
        } else {
            path.clone()
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        eprintln!("rerun: saving to {}", path.display());
        let rec = rerun::RecordingStreamBuilder::new(name).save(&path)?;
        Ok(rec)
    } else {
        Ok(rerun::RecordingStreamBuilder::new(name).connect_grpc()?)
    }
}

#[cfg(test)]
mod tests {
    use super::bench::{BenchAccum, summarize_bench};
    use super::config::{TrackerDefaults, build_ba_config, build_tracker_config};
    use kiko_slam::{DownscaleFactor, KeypointLimit, LoopSubsystemConfig};
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn set_env(key: &str, value: &str) {
        // Safety: tests hold a process-wide lock while mutating environment vars.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var(key, value);
        }
    }

    fn restore_env(key: &str, value: Option<OsString>) {
        // Safety: tests hold a process-wide lock while mutating environment vars.
        #[allow(unsafe_code)]
        unsafe {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }

    #[test]
    fn build_ba_config_reads_lm_env_settings() {
        let _guard = env_lock().lock().expect("env lock");
        let keys = [
            "KIKO_BA_WINDOW",
            "KIKO_BA_ITERS",
            "KIKO_BA_MIN_OBS",
            "KIKO_BA_HUBER_PX",
            "KIKO_BA_DAMPING",
            "KIKO_LM_FACTOR",
            "KIKO_LM_MIN",
            "KIKO_LM_MAX",
            "KIKO_BA_MOTION_WEIGHT",
        ];
        let saved: Vec<(String, Option<OsString>)> = keys
            .iter()
            .map(|&key| (key.to_string(), std::env::var_os(key)))
            .collect();

        set_env("KIKO_BA_WINDOW", "12");
        set_env("KIKO_BA_ITERS", "7");
        set_env("KIKO_BA_MIN_OBS", "9");
        set_env("KIKO_BA_HUBER_PX", "2.5");
        set_env("KIKO_BA_DAMPING", "0.002");
        set_env("KIKO_LM_FACTOR", "12.0");
        set_env("KIKO_LM_MIN", "0.000001");
        set_env("KIKO_LM_MAX", "5000");
        set_env("KIKO_BA_MOTION_WEIGHT", "0.25");

        let config = build_ba_config().expect("build config");
        assert_eq!(config.window(), 12);
        assert_eq!(config.max_iterations(), 7);
        assert_eq!(config.min_observations(), 9);
        assert!((config.huber_delta_px() - 2.5).abs() < 1e-6);
        assert!((config.lm().initial_lambda() - 0.002).abs() < 1e-9);
        assert!((config.lm().lambda_factor() - 12.0).abs() < 1e-9);
        assert!((config.lm().min_lambda() - 1e-6).abs() < 1e-12);
        assert!((config.lm().max_lambda() - 5000.0).abs() < 1e-6);
        assert!((config.motion_prior_weight() - 0.25).abs() < 1e-6);

        for (key, value) in saved {
            restore_env(&key, value);
        }
    }

    #[test]
    fn build_tracker_config_reads_loop_env_settings() {
        let _guard = env_lock().lock().expect("env lock");
        let keys = [
            "KIKO_LOOP_SIMILARITY_THRESHOLD",
            "KIKO_LOOP_DESCRIPTOR_MATCH_THRESHOLD",
            "KIKO_LOOP_MIN_INLIERS",
            "KIKO_LOOP_MAX_CANDIDATES",
            "KIKO_LOOP_TEMPORAL_GAP",
            "KIKO_LOOP_MIN_STREAK",
            "KIKO_LOOP_MAX_CORRECTION_TRANSLATION_M",
            "KIKO_LOOP_MAX_CORRECTION_ROTATION_DEG",
            "KIKO_LOOP_RANSAC_MAX_ITERATIONS",
            "KIKO_LOOP_RANSAC_THRESHOLD_PX",
            "KIKO_LOOP_RANSAC_MIN_INLIERS",
        ];
        let saved: Vec<(String, Option<OsString>)> = keys
            .iter()
            .map(|&key| (key.to_string(), std::env::var_os(key)))
            .collect();

        set_env("KIKO_LOOP_SIMILARITY_THRESHOLD", "0.80");
        set_env("KIKO_LOOP_DESCRIPTOR_MATCH_THRESHOLD", "0.72");
        set_env("KIKO_LOOP_MIN_INLIERS", "18");
        set_env("KIKO_LOOP_MAX_CANDIDATES", "5");
        set_env("KIKO_LOOP_TEMPORAL_GAP", "25");
        set_env("KIKO_LOOP_MIN_STREAK", "2");
        set_env("KIKO_LOOP_MAX_CORRECTION_TRANSLATION_M", "4.5");
        set_env("KIKO_LOOP_MAX_CORRECTION_ROTATION_DEG", "25");
        set_env("KIKO_LOOP_RANSAC_MAX_ITERATIONS", "150");
        set_env("KIKO_LOOP_RANSAC_THRESHOLD_PX", "1.75");
        set_env("KIKO_LOOP_RANSAC_MIN_INLIERS", "18");

        let config = build_tracker_config(
            TrackerDefaults {
                min_keyframe_points: 12,
                refresh_inliers: 12,
                min_inliers: 8,
            },
            KeypointLimit::try_from(1024).expect("keypoint limit"),
            DownscaleFactor::try_from(1).expect("downscale"),
        )
        .expect("tracker config");

        let loop_cfg = match config.loop_subsystem {
            LoopSubsystemConfig::LoopClosureOnly { loop_closure, .. }
            | LoopSubsystemConfig::LoopClosureAndRelocalization { loop_closure, .. } => {
                loop_closure
            }
            LoopSubsystemConfig::Disabled => panic!("loop subsystem should be enabled"),
        };
        assert!((loop_cfg.similarity_threshold() - 0.80).abs() < 1e-6);
        assert!((loop_cfg.descriptor_match_threshold() - 0.72).abs() < 1e-6);
        assert_eq!(loop_cfg.min_inliers(), 18);
        assert_eq!(loop_cfg.max_candidates(), 5);
        assert_eq!(loop_cfg.temporal_gap(), 25);
        assert_eq!(loop_cfg.min_streak(), 2);
        assert!((loop_cfg.max_correction_translation() - 4.5).abs() < 1e-6);
        assert!((loop_cfg.max_correction_rotation_deg() - 25.0).abs() < 1e-6);
        assert_eq!(loop_cfg.ransac().max_iterations, 150);
        assert!((loop_cfg.ransac().reprojection_threshold_px - 1.75).abs() < 1e-6);
        assert_eq!(loop_cfg.ransac().min_inliers, 18);

        for (key, value) in saved {
            restore_env(&key, value);
        }
    }

    #[test]
    fn build_tracker_config_rejects_invalid_loop_env() {
        let _guard = env_lock().lock().expect("env lock");
        let key = "KIKO_LOOP_MAX_CORRECTION_ROTATION_DEG";
        let saved = std::env::var_os(key);
        set_env(key, "181.0");

        let result = build_tracker_config(
            TrackerDefaults {
                min_keyframe_points: 12,
                refresh_inliers: 12,
                min_inliers: 8,
            },
            KeypointLimit::try_from(1024).expect("keypoint limit"),
            DownscaleFactor::try_from(1).expect("downscale"),
        );
        assert!(
            result.is_err(),
            "invalid loop config should return an error"
        );

        restore_env(key, saved);
    }

    #[test]
    fn build_tracker_config_disables_loop_closure_without_descriptors() {
        let _guard = env_lock().lock().expect("env lock");
        let keys = [
            "KIKO_LOOP_CLOSURE",
            "KIKO_LEARNED_DESCRIPTORS",
            "KIKO_RELOCALIZATION",
        ];
        let saved: Vec<(String, Option<OsString>)> = keys
            .iter()
            .map(|&key| (key.to_string(), std::env::var_os(key)))
            .collect();

        set_env("KIKO_LOOP_CLOSURE", "true");
        set_env("KIKO_LEARNED_DESCRIPTORS", "false");
        set_env("KIKO_RELOCALIZATION", "true");

        let config = build_tracker_config(
            TrackerDefaults {
                min_keyframe_points: 12,
                refresh_inliers: 12,
                min_inliers: 8,
            },
            KeypointLimit::try_from(1024).expect("keypoint limit"),
            DownscaleFactor::try_from(1).expect("downscale"),
        )
        .expect("tracker config");

        assert!(matches!(config.loop_subsystem, LoopSubsystemConfig::Disabled));

        for (key, value) in saved {
            restore_env(&key, value);
        }
    }

    #[cfg(feature = "vio")]
    #[test]
    fn build_tracker_config_reads_vio_env_settings() {
        let _guard = env_lock().lock().expect("env lock");
        let keys = [
            "KIKO_VIO",
            "KIKO_VIO_WINDOW",
            "KIKO_VIO_MAX_ITERS",
            "KIKO_VIO_POSE_PRIOR_WEIGHT",
        ];
        let saved: Vec<(String, Option<OsString>)> = keys
            .iter()
            .map(|&key| (key.to_string(), std::env::var_os(key)))
            .collect();

        set_env("KIKO_VIO", "true");
        set_env("KIKO_VIO_WINDOW", "9");
        set_env("KIKO_VIO_MAX_ITERS", "6");
        set_env("KIKO_VIO_POSE_PRIOR_WEIGHT", "55.5");

        let config = build_tracker_config(
            TrackerDefaults {
                min_keyframe_points: 12,
                refresh_inliers: 12,
                min_inliers: 8,
            },
            KeypointLimit::try_from(1024).expect("keypoint limit"),
            DownscaleFactor::try_from(1).expect("downscale"),
        )
        .expect("tracker config");

        let vio = config.vio.expect("vio config should be present");
        assert_eq!(vio.window_size(), 9);
        assert_eq!(vio.max_iterations(), 6);
        assert!((vio.pose_prior_weight() - 55.5).abs() < 1e-9);

        for (key, value) in saved {
            restore_env(&key, value);
        }
    }

    #[test]
    fn summarize_bench_reports_exact_stage_metrics() {
        let accum = BenchAccum {
            read_samples: 4,
            processed: 3,
            matches_nonzero: 2,
            total_matches: 12,
            sum_read_left: Duration::from_millis(20),
            sum_read_right: Duration::from_millis(20),
            sum_pairing: Duration::from_millis(10),
            sum_read_bytes: 8 * 1024 * 1024,
            sum_sp_left: Duration::from_millis(9),
            sum_sp_right: Duration::from_millis(12),
            sum_lightglue: Duration::from_millis(15),
            sum_total_success: Duration::from_millis(45),
            sum_inference_attempt: Duration::from_millis(60),
            ..BenchAccum::default()
        };
        let summary = summarize_bench(&accum, Duration::from_secs(2));

        assert!((summary.wall_fps - 1.5).abs() < 1e-9);
        assert!((summary.reader_stage_fps - 80.0).abs() < 1e-9);
        assert!((summary.inference_attempt_fps - (4.0 / 0.06)).abs() < 1e-9);
        assert!((summary.successful_inference_fps - (3.0 / 0.045)).abs() < 1e-9);
        assert!((summary.match_rate - (2.0 / 3.0)).abs() < 1e-9);
        assert!((summary.avg_matches_per_processed_pair - 4.0).abs() < 1e-9);
        assert!((summary.avg_matches_per_nonzero_pair - 6.0).abs() < 1e-9);
    }
}
