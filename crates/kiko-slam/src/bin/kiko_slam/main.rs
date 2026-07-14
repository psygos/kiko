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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RunIntegrityError {
    command: &'static str,
    successful_item: &'static str,
    successful_items: usize,
    total_failures: u128,
    first_failed_stage: Option<&'static str>,
    first_stage_failures: usize,
}

impl std::fmt::Display for RunIntegrityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(stage) = self.first_failed_stage {
            write!(
                f,
                "{} did not complete cleanly: successful_{}={}, total_failures={}, first_failed_stage={} ({})",
                self.command,
                self.successful_item,
                self.successful_items,
                self.total_failures,
                stage,
                self.first_stage_failures
            )
        } else {
            write!(
                f,
                "{} produced no successful {}",
                self.command, self.successful_item
            )
        }
    }
}

impl std::error::Error for RunIntegrityError {}

fn verify_run_integrity(
    command: &'static str,
    successful_item: &'static str,
    successful_items: usize,
    failures: &[(&'static str, usize)],
) -> Result<(), RunIntegrityError> {
    let first_failure = failures.iter().copied().find(|(_, count)| *count > 0);
    let total_failures = failures
        .iter()
        .map(|(_, count)| *count as u128)
        .sum::<u128>();
    if successful_items > 0 && total_failures == 0 {
        return Ok(());
    }
    Err(RunIntegrityError {
        command,
        successful_item,
        successful_items,
        total_failures,
        first_failed_stage: first_failure.map(|(stage, _)| stage),
        first_stage_failures: first_failure.map_or(0, |(_, count)| count),
    })
}

#[derive(Parser, Debug)]
#[command(name = "kiko-slam", about = "Kiko SLAM tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run full SLAM pipeline on a recorded dataset
    #[command(visible_alias = "run")]
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

#[derive(Debug)]
enum RerunRecordingInitError {
    CreateParent {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    Save {
        path: std::path::PathBuf,
        source: rerun::RecordingStreamError,
    },
    Serve {
        port: u16,
        source: rerun::RecordingStreamError,
    },
    Connect {
        url: String,
        source: rerun::RecordingStreamError,
    },
    SpawnLocalViewer {
        source: rerun::RecordingStreamError,
    },
}

impl std::fmt::Display for RerunRecordingInitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreateParent { path, source } => write!(
                f,
                "failed to create Rerun output directory {}: {source}",
                path.display()
            ),
            Self::Save { path, source } => {
                write!(
                    f,
                    "failed to initialize Rerun recording at {}: {source}",
                    path.display()
                )
            }
            Self::Serve { port, source } => {
                write!(f, "failed to serve Rerun output on port {port}: {source}")
            }
            Self::Connect { url, source } => {
                write!(f, "failed to connect Rerun output to {url}: {source}")
            }
            Self::SpawnLocalViewer { source } => {
                write!(f, "failed to spawn or reuse a local Rerun viewer: {source}")
            }
        }
    }
}

impl std::error::Error for RerunRecordingInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CreateParent { source, .. } => Some(source),
            Self::Save { source, .. }
            | Self::Serve { source, .. }
            | Self::Connect { source, .. }
            | Self::SpawnLocalViewer { source } => Some(source),
        }
    }
}

fn rerun_recording(
    destination: &args::RerunDestination,
    name: &str,
) -> Result<rerun::RecordingStream, RerunRecordingInitError> {
    match destination {
        args::RerunDestination::Save(path) => {
            let path = if path.is_dir() {
                path.join(format!("{name}.rrd"))
            } else {
                path.clone()
            };
            if let Some(parent) = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent).map_err(|source| {
                    RerunRecordingInitError::CreateParent {
                        path: parent.to_owned(),
                        source,
                    }
                })?;
            }
            let rec = rerun::RecordingStreamBuilder::new(name)
                .save(&path)
                .map_err(|source| RerunRecordingInitError::Save {
                    path: path.clone(),
                    source,
                })?;
            eprintln!("rerun: saving to {}", path.display());
            Ok(rec)
        }
        args::RerunDestination::Serve { port } => {
            let rec = rerun::RecordingStreamBuilder::new(name)
                .serve_grpc_opts("0.0.0.0", *port, Default::default())
                .map_err(|source| RerunRecordingInitError::Serve {
                    port: *port,
                    source,
                })?;
            eprintln!("rerun: serving gRPC on 0.0.0.0:{port}");
            eprintln!(
                "rerun: on your laptop run:  rerun --connect rerun+http://192.168.50.2:{port}/proxy"
            );
            Ok(rec)
        }
        args::RerunDestination::Connect(url) => {
            let rec = rerun::RecordingStreamBuilder::new(name)
                .connect_grpc_opts(url)
                .map_err(|source| RerunRecordingInitError::Connect {
                    url: url.clone(),
                    source,
                })?;
            eprintln!("rerun: connecting to {url}");
            Ok(rec)
        }
        args::RerunDestination::ImplicitLocalViewer => {
            let rec = rerun::RecordingStreamBuilder::new(name)
                .spawn()
                .map_err(|source| RerunRecordingInitError::SpawnLocalViewer { source })?;
            eprintln!("rerun: spawning or reusing a local viewer");
            Ok(rec)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::args::{PrefetchSession, RerunArgsError, RerunDestination, RerunOutput};
    use super::bench::{BenchAccum, summarize_bench};
    use super::config::{TrackerDefaults, build_ba_config, build_tracker_config};
    use super::{Cli, Command, RerunRecordingInitError, rerun_recording, verify_run_integrity};
    use clap::Parser;
    use kiko_slam::{DownscaleFactor, KeypointLimit, LoopSubsystemConfig, TrackingMatcher};
    use std::ffi::OsString;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn run_integrity_requires_success_and_zero_stage_failures() {
        verify_run_integrity("bench", "pairs", 3, &[("read", 0), ("inference", 0)])
            .expect("clean run");

        let partial = verify_run_integrity("bench", "pairs", 2, &[("read", 1), ("inference", 2)])
            .expect_err("partial run");
        assert_eq!(partial.total_failures, 3);
        assert_eq!(partial.first_failed_stage, Some("read"));
        assert_eq!(partial.first_stage_failures, 1);

        let empty = verify_run_integrity("bench", "pairs", 0, &[]).expect_err("empty run");
        assert_eq!(empty.total_failures, 0);
        assert_eq!(empty.first_failed_stage, None);
    }

    #[test]
    fn prefetch_session_distinguishes_ready_from_not_applicable() {
        let ready = PrefetchSession::Ready(42_u8);
        assert!(ready.is_ready());
        assert_eq!(ready.into_option(), Some(42));

        let not_applicable = PrefetchSession::<u8>::NotApplicable;
        assert!(!not_applicable.is_ready());
        assert_eq!(not_applicable.into_option(), None);
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
        restore_env("KIKO_BA_MOTION_WEIGHT", None);

        let config = build_ba_config().expect("build config");
        assert_eq!(config.window(), 12);
        assert_eq!(config.max_iterations(), 7);
        assert_eq!(config.min_observations(), 9);
        assert!((config.huber_delta_px() - 2.5).abs() < 1e-6);
        assert!((config.lm().initial_lambda() - 0.002).abs() < 1e-9);
        assert!((config.lm().lambda_factor() - 12.0).abs() < 1e-9);
        assert!((config.lm().min_lambda() - 1e-6).abs() < 1e-12);
        assert!((config.lm().max_lambda() - 5000.0).abs() < 1e-6);

        for (key, value) in saved {
            restore_env(&key, value);
        }
    }

    #[test]
    fn build_ba_config_rejects_removed_mixed_unit_motion_regularizer() {
        let _guard = env_lock().lock().expect("env lock");
        let key = "KIKO_BA_MOTION_WEIGHT";
        let saved = std::env::var_os(key);

        for value in ["0", "0.25"] {
            set_env(key, value);
            let error = build_ba_config().expect_err("legacy motion regularizer must fail closed");
            let message = error.to_string();
            assert!(message.contains(key));
            assert!(message.contains("mixed metres and radians"));
            assert!(message.contains("zero relative motion"));
        }

        restore_env(key, saved);
    }

    #[test]
    fn build_ba_config_rejects_malformed_integer_with_source() {
        let _guard = env_lock().lock().expect("env lock");
        let key = "KIKO_BA_WINDOW";
        let saved = std::env::var_os(key);
        set_env(key, "not-an-integer");

        let error = build_ba_config().expect_err("malformed integer must fail");

        assert!(error.to_string().contains(key));
        assert!(error.source().is_some());
        restore_env(key, saved);
    }

    #[test]
    fn build_tracker_config_rejects_unknown_tracking_matcher() {
        let _guard = env_lock().lock().expect("env lock");
        let primary = "KIKO_TRACKING_MATCHER";
        let legacy = "KIKO_TRACK_MATCHER";
        let saved_primary = std::env::var_os(primary);
        let saved_legacy = std::env::var_os(legacy);
        set_env(primary, "mystery");
        restore_env(legacy, None);

        let error = build_tracker_config(
            TrackerDefaults {
                min_keyframe_points: 12,
                refresh_inliers: 12,
                min_inliers: 8,
            },
            KeypointLimit::try_from(1024).expect("keypoint limit"),
            DownscaleFactor::try_from(1).expect("downscale"),
        )
        .expect_err("unknown matcher must fail");

        assert!(error.to_string().contains("unknown tracking matcher"));
        restore_env(primary, saved_primary);
        restore_env(legacy, saved_legacy);
    }

    #[test]
    fn build_tracker_config_reads_truthful_projected_dot_product_setting() {
        let _guard = env_lock().lock().expect("env lock");
        let keys = [
            "KIKO_TRACKING_MATCHER",
            "KIKO_TRACK_MATCHER",
            "KIKO_PROJECTED_MATCH_MIN_DOT_PRODUCT",
            "KIKO_PROJECTED_MATCH_MIN_SIMILARITY",
        ];
        let saved: Vec<(String, Option<OsString>)> = keys
            .iter()
            .map(|&key| (key.to_string(), std::env::var_os(key)))
            .collect();
        set_env("KIKO_TRACKING_MATCHER", "projected");
        restore_env("KIKO_TRACK_MATCHER", None);
        set_env("KIKO_PROJECTED_MATCH_MIN_DOT_PRODUCT", "1.25");
        restore_env("KIKO_PROJECTED_MATCH_MIN_SIMILARITY", None);

        let config = build_tracker_config(
            TrackerDefaults {
                min_keyframe_points: 12,
                refresh_inliers: 12,
                min_inliers: 8,
            },
            KeypointLimit::try_from(1024).expect("keypoint limit"),
            DownscaleFactor::try_from(1).expect("downscale"),
        )
        .expect("projected matcher configuration");

        let TrackingMatcher::Projected(projected) = config.tracking_matcher else {
            panic!("projected matcher setting must select projected tracking");
        };
        assert_eq!(projected.min_descriptor_dot_product(), 1.25);

        for (key, value) in saved {
            restore_env(&key, value);
        }
    }

    #[test]
    fn build_tracker_config_rejects_conflicting_projected_dot_product_aliases() {
        let _guard = env_lock().lock().expect("env lock");
        let keys = [
            "KIKO_TRACKING_MATCHER",
            "KIKO_TRACK_MATCHER",
            "KIKO_PROJECTED_MATCH_MIN_DOT_PRODUCT",
            "KIKO_PROJECTED_MATCH_MIN_SIMILARITY",
        ];
        let saved: Vec<(String, Option<OsString>)> = keys
            .iter()
            .map(|&key| (key.to_string(), std::env::var_os(key)))
            .collect();
        set_env("KIKO_TRACKING_MATCHER", "projected");
        restore_env("KIKO_TRACK_MATCHER", None);
        set_env("KIKO_PROJECTED_MATCH_MIN_DOT_PRODUCT", "0.45");
        set_env("KIKO_PROJECTED_MATCH_MIN_SIMILARITY", "0.8");

        let error = build_tracker_config(
            TrackerDefaults {
                min_keyframe_points: 12,
                refresh_inliers: 12,
                min_inliers: 8,
            },
            KeypointLimit::try_from(1024).expect("keypoint limit"),
            DownscaleFactor::try_from(1).expect("downscale"),
        )
        .expect_err("conflicting raw-dot-product aliases must fail");

        assert!(error.to_string().contains("conflicts"));
        assert!(error.to_string().contains("DOT_PRODUCT"));

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
        assert!((loop_cfg.max_correction_translation_m() - 4.5).abs() < 1e-6);
        assert!((loop_cfg.max_correction_rotation_deg() - 25.0).abs() < 1e-6);
        assert_eq!(loop_cfg.ransac().max_iterations(), 150);
        assert!((loop_cfg.ransac().reprojection_threshold_px() - 1.75).abs() < 1e-6);
        assert_eq!(loop_cfg.ransac().min_inliers(), 18);

        for (key, value) in saved {
            restore_env(&key, value);
        }
    }

    #[test]
    fn rerun_save_alias_parses_for_viz() {
        let cli = Cli::try_parse_from([
            "kiko-slam",
            "viz",
            "--backend",
            "cpu",
            "--rerun-save",
            "/tmp/debug.rrd",
            "/tmp/dataset",
        ])
        .expect("parse cli");

        match cli.command {
            Command::Viz(args) => {
                assert_eq!(args.rerun.save_rrd, Some(PathBuf::from("/tmp/debug.rrd")));
                assert_eq!(args.dataset.path, PathBuf::from("/tmp/dataset"));
                let output = RerunOutput::try_from_args(&args.rerun).expect("rerun output");
                assert_eq!(
                    output.destination(),
                    &RerunDestination::Save(PathBuf::from("/tmp/debug.rrd"))
                );
                assert!(output.has_explicit_destination());

                let mut local_args = args.rerun;
                local_args.save_rrd = None;
                let local = RerunOutput::try_from_args(&local_args).expect("local output");
                assert_eq!(local.destination(), &RerunDestination::ImplicitLocalViewer);
                assert!(!local.has_explicit_destination());
            }
            _ => panic!("expected viz command"),
        }
    }

    #[test]
    fn rerun_destinations_are_mutually_exclusive_at_cli_boundary() {
        let error = Cli::try_parse_from([
            "kiko-slam",
            "viz",
            "--backend",
            "cpu",
            "--save-rrd",
            "/tmp/debug.rrd",
            "--rerun-serve",
            "/tmp/dataset",
        ])
        .expect_err("conflicting Rerun destinations must be rejected");

        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn rerun_output_parser_rejects_forged_conflicting_args() {
        let cli = Cli::try_parse_from([
            "kiko-slam",
            "viz",
            "--backend",
            "cpu",
            "--save-rrd",
            "/tmp/debug.rrd",
            "/tmp/dataset",
        ])
        .expect("parse cli");
        let Command::Viz(mut args) = cli.command else {
            panic!("expected viz command");
        };
        args.rerun.rerun_serve = true;

        assert_eq!(
            RerunOutput::try_from_args(&args.rerun),
            Err(RerunArgsError::ConflictingDestinations {
                destination_count: 2
            })
        );
    }

    #[test]
    fn rerun_recording_parent_failure_preserves_path_and_io_source() {
        let blocker =
            std::env::temp_dir().join(format!("kiko-rerun-parent-test-{}", std::process::id()));
        let _ = std::fs::remove_file(&blocker);
        std::fs::write(&blocker, b"not a directory").expect("create parent blocker");
        let output_path = blocker.join("recording.rrd");

        let error = rerun_recording(
            &RerunDestination::Save(output_path),
            "rerun-parent-error-test",
        )
        .expect_err("file parent must reject directory creation");
        std::fs::remove_file(&blocker).expect("remove parent blocker");

        match error {
            RerunRecordingInitError::CreateParent { path, source } => {
                assert_eq!(path, blocker);
                assert_ne!(source.kind(), std::io::ErrorKind::NotFound);
            }
            other => panic!("expected parent creation error, got {other}"),
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

        assert!(matches!(
            config.loop_subsystem,
            LoopSubsystemConfig::Disabled
        ));

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
