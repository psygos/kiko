use std::time::Instant;

use clap::Args;

use kiko_slam::dataset::DatasetReader;
use kiko_slam::{ProjectedMatcherConfig, TrackingMatcher};
use kiko_slam::{RerunSink, SlamTracker, VizPacket};

use crate::args::{
    DatasetArgs, InferenceArgs, InferenceConfig, InferencePurpose, RerunArgs, RunProfileArg,
};
use crate::config::{TrackerDefaults, TrackerOverrides, build_tracker_config_with_overrides};
use crate::rerun_recording;

const SLAM_ENV_HELP: &str = "\
ENVIRONMENT VARIABLES (expert tuning):

  Bundle Adjustment:
    KIKO_BA_WINDOW=10           Sliding window size
    KIKO_BA_ITERS=6             Max LM iterations
    KIKO_BA_MIN_OBS=8           Min observations per landmark
    KIKO_BA_HUBER_PX=3.0        Huber robust kernel delta (pixels)
    KIKO_BA_DAMPING=0.001       Initial LM lambda
    KIKO_BA_MOTION_WEIGHT       Unsupported legacy mixed-unit regularizer (startup error)
    KIKO_LM_FACTOR=10.0         LM lambda scale factor
    KIKO_LM_MIN=1e-8            LM lambda floor
    KIKO_LM_MAX=10000           LM lambda ceiling

  Keyframe Policy:
    KIKO_KEYFRAME_MIN_POINTS=12          Min triangulated points to accept keyframe
    KIKO_KEYFRAME_REFRESH_INLIERS=12     Min inliers to skip keyframe insertion
    KIKO_KEYFRAME_PARALLAX_PX=40.0       Parallax threshold (pixels)
    KIKO_KEYFRAME_COVISIBILITY=0.6       Min covisibility ratio for connection
    KIKO_KEYFRAME_REDUNDANT_COVISIBILITY=0.9  Covisibility ratio for redundancy culling
    KIKO_TRACK_MIN_INLIERS=8             Min PnP RANSAC inliers
    KIKO_TRACKING_MATCHER=projected      projected or lightglue tracking matcher
    KIKO_PROJECTED_MATCH_RADIUS_PX=32    Projected matcher search radius
    KIKO_PROJECTED_MATCH_MIN_SIMILARITY=0.45
    KIKO_PROJECTED_MATCH_MIN_MATCHES=32  Fallback to LightGlue below this match count
    KIKO_PROJECTED_MATCH_MIN_INLIERS=24  Fallback to LightGlue below this inlier count

  Loop Closure:
    KIKO_LOOP_CLOSURE=true                   Enable loop closure detection
    KIKO_RELOCALIZATION=true                 Enable relocalization
    KIKO_LEARNED_DESCRIPTORS=true            Use EigenPlaces global descriptors
    KIKO_LOOP_SIMILARITY_THRESHOLD           Cosine similarity threshold
    KIKO_LOOP_DESCRIPTOR_MATCH_THRESHOLD     Local descriptor match threshold
    KIKO_LOOP_MIN_INLIERS                    Min geometric verification inliers
    KIKO_LOOP_MAX_CANDIDATES                 Max candidates per query
    KIKO_LOOP_TEMPORAL_GAP                   Min keyframe gap for loop candidates
    KIKO_LOOP_MIN_STREAK                     Min consecutive detections to accept
    KIKO_LOOP_MAX_CORRECTION_TRANSLATION_M   Max correction translation (meters)
    KIKO_LOOP_MAX_CORRECTION_ROTATION_DEG    Max correction rotation (degrees)
    KIKO_LOOP_RANSAC_MAX_ITERATIONS          RANSAC iterations for verification
    KIKO_LOOP_RANSAC_THRESHOLD_PX            RANSAC reprojection threshold (pixels)
    KIKO_LOOP_RANSAC_MIN_INLIERS             RANSAC min inliers

  Backend:
    KIKO_BACKEND_ASYNC=true       Run BA on a background thread
    KIKO_BACKEND_QUEUE_DEPTH=2    BA work queue depth
    KIKO_DESCRIPTOR_QUEUE_DEPTH=2 Descriptor extraction queue depth

  Stable Surface Map:
    KIKO_DENSE_CLOUD=true                 Enable the stable sparse stereo surface map
    KIKO_SURFACE_VOXEL_SIZE_M=0.05        Voxel size for fused stable surface points
    KIKO_SURFACE_MIN_SUPPORT_VIEWS=3      Minimum support views to confirm a voxel
    KIKO_SURFACE_MAX_CONSISTENCY_SCORE=8.0 Max residual consistency score for a confirmed voxel
    KIKO_SURFACE_MAX_CONFIRMED_STD_DEV_M=0.05 Max posterior sigma allowed for a confirmed voxel
    KIKO_SURFACE_MAX_PREDICTIVE_CONSISTENCY_SCORE=12.0 Max predictive consistency score allowed for a novel support view
    KIKO_SURFACE_MAX_RENDER_POINTS=250000 Max confirmed voxels rendered to Rerun
    KIKO_SURFACE_MAX_POINT_SIGMA_M=0.05   Max per-observation positional sigma accepted
    KIKO_SURFACE_MIN_PROJECTABLE_TRACKED_OBSERVATIONS=8 Min projectable tracked observations required before surface fusion
    KIKO_SURFACE_MAX_TRACKED_REPROJECTION_RMSE_PX=1.5 Max tracked reprojection RMSE allowed to fuse surface observations
    KIKO_DENSE_MAX_POINTS=30000           Max stable surface observations per keyframe

  Runtime IMU Override:
    KIKO_IMU_CALIBRATION_FILE=/path/to/file.json  Override only the IMU block at runtime
    KIKO_IMU_ROTATION=...                 3x3 row-major camera-from-imu rotation
    KIKO_IMU_TRANSLATION=...              3-vector camera-from-imu translation (meters)
    KIKO_IMU_ACCEL_NOISE_DENSITY=...      Accelerometer noise density
    KIKO_IMU_GYRO_NOISE_DENSITY=...       Gyroscope noise density
    KIKO_IMU_ACCEL_RANDOM_WALK=...        Accelerometer random walk
    KIKO_IMU_GYRO_RANDOM_WALK=...         Gyroscope random walk
    KIKO_IMU_TIME_OFFSET_NS=...           Camera-IMU time offset in nanoseconds
    KIKO_IMU_GRAVITY_MPS2=...             Gravity magnitude
    KIKO_IMU_INITIAL_ACCEL_BIAS=...       3-vector initial accel bias
    KIKO_IMU_INITIAL_GYRO_BIAS=...        3-vector initial gyro bias";

#[derive(Args, Clone, Debug)]
#[command(
    about = "Run full SLAM pipeline on a recorded dataset",
    after_long_help = SLAM_ENV_HELP
)]
pub struct SlamArgs {
    /// Apply a named run profile before explicit flags/env overrides.
    #[arg(long, value_enum, default_value_t = RunProfileArg::Default)]
    pub profile: RunProfileArg,
    /// Enable visual-inertial SLAM when built with the `vio` feature.
    #[arg(long, env = "KIKO_VIO", default_value_t = false)]
    pub vio: bool,
    /// Force visual-only tracking even if a profile or env enables VIO.
    #[arg(long, default_value_t = false)]
    pub visual_only: bool,
    #[command(flatten)]
    pub inference: InferenceArgs,
    #[command(flatten)]
    pub rerun: RerunArgs,
    #[command(flatten)]
    pub dataset: DatasetArgs,
}

fn vio_enabled(args: &SlamArgs) -> bool {
    (args.vio || args.profile == RunProfileArg::Jetson) && !args.visual_only
}

fn tracker_overrides(args: &SlamArgs, vio_enabled: bool) -> TrackerOverrides {
    match args.profile {
        RunProfileArg::Default => TrackerOverrides {
            vio_enabled: Some(vio_enabled),
            ..TrackerOverrides::default()
        },
        RunProfileArg::Jetson => TrackerOverrides {
            vio_enabled: Some(vio_enabled),
            ba_window: Some(6),
            ba_iters: Some(4),
            ba_min_obs: Some(4),
            tracking_matcher: Some(TrackingMatcher::Projected(
                ProjectedMatcherConfig::jetson_default(),
            )),
            loop_closure: Some(false),
            learned_descriptors: Some(false),
            relocalization: Some(false),
        },
    }
}

pub fn run_slam(args: &SlamArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vio_enabled = vio_enabled(args);
    let runtime_imu_override = kiko_slam::load_runtime_imu_calibration_from_env()?;
    let mut reader = DatasetReader::open_with_imu_calibration_override(
        &args.dataset.path,
        runtime_imu_override,
    )?;
    #[cfg(feature = "vio")]
    if vio_enabled && !reader.has_imu_data() {
        return Err("--vio requires IMU data in the dataset".into());
    }
    let stats = reader.stats();

    eprintln!("dataset: {}", args.dataset.path.display());
    eprintln!(
        "camera fps: left={:.2?} right={:.2?} paired={:.2?} (left={}, right={})",
        stats.left_fps, stats.right_fps, stats.paired_fps, stats.left_count, stats.right_count
    );

    let inference_args = args.inference.with_profile_defaults(args.profile)?;
    let inference = InferenceConfig::from_args(&inference_args, InferencePurpose::Slam)?;

    let calibration = reader.calibration().clone();
    let rectified = calibration.stereo().clone();
    let dense_cloud_enabled = kiko_slam::env::try_env_bool("KIKO_DENSE_CLOUD")?.unwrap_or(false);
    let dense_config = kiko_slam::DenseCloudConfig::try_from_env()?;
    let dense_triangulator = if dense_cloud_enabled {
        Some(kiko_slam::Triangulator::new(
            rectified.clone(),
            kiko_slam::TriangulationConfig::default(),
        ))
    } else {
        None
    };
    if dense_cloud_enabled {
        eprintln!(
            "stable surface map: enabled (measured sparse stereo -> surface belief); TSDF remains disabled in stereo-only slam mode"
        );
    }
    #[cfg(feature = "vio")]
    if vio_enabled && calibration.inertial().is_none() {
        return Err(
            "VIO requires IMU calibration via calibration.json, KIKO_IMU_CALIBRATION_FILE, or KIKO_IMU_* env".into(),
        );
    }

    let InferenceConfig {
        superpoint,
        superpoint_right,
        lightglue,
        lightglue_prefetch,
        end2end: _,
        key_limit,
        downscale,
    } = inference;
    let superpoint_right = superpoint_right.into_option();
    let lightglue_prefetch = lightglue_prefetch.into_option();

    let tracker_config = build_tracker_config_with_overrides(
        TrackerDefaults {
            min_keyframe_points: 12,
            refresh_inliers: 12,
            min_inliers: 8,
        },
        key_limit,
        downscale,
        tracker_overrides(args, vio_enabled),
    )?;
    let use_speculative_lg = tracker_config.tracking_matcher.uses_speculative_lightglue();
    let mut sink = match rerun_recording(&args.rerun, "kiko-slam-dataset-odometry") {
        Ok(rec) => Some(RerunSink::try_new(rec, args.rerun.rerun_decimation)?),
        Err(err) => {
            eprintln!("failed to initialize rerun; continuing headless: {err}");
            None
        }
    };
    let mut tracker = SlamTracker::try_new(superpoint, lightglue, calibration, tracker_config)?;
    let has_prefetch_sp = superpoint_right.is_some();
    let has_prefetch_lg = lightglue_prefetch.is_some() && use_speculative_lg;
    if let Some(sp_right) = superpoint_right {
        tracker.return_prefetch_sp(sp_right);
    }
    if let Some(lg_prefetch) = lightglue_prefetch.filter(|_| use_speculative_lg) {
        tracker.return_prefetch_lg(lg_prefetch);
    }
    eprintln!(
        "detection prefetch pipeline: {}",
        if has_prefetch_sp {
            "enabled"
        } else {
            "disabled"
        }
    );
    eprintln!(
        "speculative LightGlue prefetch: {}",
        if has_prefetch_lg {
            "enabled"
        } else {
            "disabled"
        }
    );

    let start = Instant::now();
    let mut attempted = 0usize;
    let mut processed = 0usize;
    let mut inference_errors = 0usize;
    let mut prefetch_errors = 0usize;
    let mut read_errors = 0usize;
    let mut poses_logged = 0usize;
    let mut keyframes = 0usize;

    // Prefetch pipeline: overlap SP(N+1) + speculative LG(N+1) with frame N.
    enum PrefetchInferenceOutcome {
        Detected {
            detections: std::sync::Arc<kiko_slam::Detections>,
            speculative_matches: Result<
                Option<(kiko_slam::KeyframeId, kiko_slam::Matches<kiko_slam::Raw>)>,
                kiko_slam::InferenceError,
            >,
        },
        DetectionFailed {
            source: kiko_slam::InferenceError,
        },
    }

    struct PrefetchResult {
        sp: kiko_slam::SuperPoint,
        lg: Option<kiko_slam::LightGlue>,
        frame_id: kiko_slam::FrameId,
        outcome: PrefetchInferenceOutcome,
    }
    let mut pending_prefetch: Option<std::thread::JoinHandle<PrefetchResult>> = None;
    let ds = downscale;
    let max_kp = key_limit.get();

    let mut bundles_iter = reader.bundles();
    // Skip initial frames (camera white balance + IMU settling)
    let skip = args.dataset.skip_frames;
    if skip > 0 {
        eprintln!("skipping first {skip} frames");
        for _ in 0..skip {
            let _ = bundles_iter.next();
        }
    }
    // Read-ahead: grab the first bundle
    let mut next_bundle: Option<kiko_slam::CaptureBundle> = loop {
        match bundles_iter.next() {
            Some(Ok(b)) => break Some(b),
            Some(Err(err)) => {
                read_errors += 1;
                eprintln!("read error: {err}");
                continue;
            }
            None => break None,
        }
    };

    while let Some(bundle) = next_bundle.take() {
        attempted += 1;
        let left = bundle.pair().left().clone();
        let right = bundle.pair().right().clone();
        let imu = bundle.imu().batch().cloned();

        // Collect prefetched SP+LG for THIS frame (launched during previous iteration)
        let (prefetched_left, prefetched_matches) = if let Some(handle) = pending_prefetch.take() {
            match handle.join() {
                Ok(result) => {
                    tracker.return_prefetch_sp(result.sp);
                    if let Some(lg) = result.lg {
                        tracker.return_prefetch_lg(lg);
                    }
                    match result.outcome {
                        PrefetchInferenceOutcome::Detected {
                            detections,
                            speculative_matches,
                        } => {
                            let speculative_matches = match speculative_matches {
                                Ok(matches) => matches,
                                Err(source) => {
                                    prefetch_errors = prefetch_errors.saturating_add(1);
                                    report_error_chain(
                                        "speculative LightGlue prefetch failed; tracker will match synchronously",
                                        &source,
                                    );
                                    None
                                }
                            };
                            (Some((result.frame_id, detections)), speculative_matches)
                        }
                        PrefetchInferenceOutcome::DetectionFailed { source } => {
                            prefetch_errors = prefetch_errors.saturating_add(1);
                            report_error_chain(
                                "SuperPoint prefetch failed; tracker will detect synchronously",
                                &source,
                            );
                            (None, None)
                        }
                    }
                }
                Err(payload) => {
                    return Err(std::io::Error::other(format!(
                        "inference prefetch thread panicked: {}",
                        kiko_slam::panic_payload_to_string(payload.as_ref())
                    ))
                    .into());
                }
            }
        } else {
            (None, None)
        };

        // Read-ahead: grab the NEXT bundle from the iterator
        let lookahead_bundle: Option<kiko_slam::CaptureBundle> = loop {
            match bundles_iter.next() {
                Some(Ok(b)) => break Some(b),
                Some(Err(err)) => {
                    read_errors += 1;
                    eprintln!("read error: {err}");
                    continue;
                }
                None => break None,
            }
        };

        // Launch SP + speculative LG prefetch for the NEXT frame on a background thread
        let below_pair_limit = should_prefetch_lookahead(attempted, args.dataset.max_pairs);
        if below_pair_limit && let Some(ref next_b) = lookahead_bundle {
            if let Some(mut sp) = tracker.take_prefetch_sp() {
                let mut lg = if use_speculative_lg {
                    tracker.take_prefetch_lg()
                } else {
                    None
                };
                let keyframe_info = tracker.current_tracking_keyframe_detections();
                let next_left = next_b.pair().left().clone();
                let next_fid = next_left.frame_id();
                pending_prefetch = Some(std::thread::spawn(move || {
                    let outcome = match sp.detect_with_downscale_limited(&next_left, ds, max_kp) {
                        Ok(detections) => {
                            let detections = std::sync::Arc::new(detections.top_k(max_kp));
                            let speculative_matches = match (&mut lg, &keyframe_info) {
                                (Some(lg_session), Some((keyframe_id, keyframe_dets))) => {
                                    lg_session
                                        .match_these(detections.clone(), keyframe_dets.clone())
                                        .map(|matches| Some((*keyframe_id, matches)))
                                }
                                _ => Ok(None),
                            };
                            PrefetchInferenceOutcome::Detected {
                                detections,
                                speculative_matches,
                            }
                        }
                        Err(source) => PrefetchInferenceOutcome::DetectionFailed { source },
                    };
                    PrefetchResult {
                        sp,
                        lg,
                        frame_id: next_fid,
                        outcome,
                    }
                }));
            }
        }

        // Process THIS frame while SP+LG prefetch for NEXT frame runs in parallel.
        match tracker.process_capture_with_prefetch(bundle, prefetched_left, prefetched_matches) {
            Ok(output) => {
                let timestamp = left.timestamp();
                let loop_applied = output.events.iter().any(|event| {
                    matches!(
                        event,
                        kiko_slam::DiagnosticEvent::LoopClosureDetected { .. }
                    )
                });
                if let Some(matches) = output.stereo_matches {
                    // Extract measured stereo samples before matches are consumed.
                    let surface_samples = dense_triangulator
                        .as_ref()
                        .map(|tri| tri.extract_stereo_samples(&matches))
                        .transpose()?;

                    let points = output
                        .keyframe
                        .as_ref()
                        .map(|kf| kf.landmarks())
                        .filter(|pts| !pts.is_empty());
                    if let Some(sink) = sink.as_mut() {
                        match VizPacket::try_new(left.clone(), right.clone(), matches) {
                            Ok(packet) => {
                                if let Err(err) = sink.log_with_points(&packet, points) {
                                    eprintln!("rerun log error: {err}");
                                }
                            }
                            Err(err) => {
                                eprintln!(
                                    "viz packet error: {err}; falling back to raw stereo views"
                                );
                                if let Err(log_err) = sink.log_frames(&left, &right) {
                                    eprintln!("rerun log error: {log_err}");
                                }
                            }
                        }
                    }
                    // Generate stable sparse surface observations for the low-resolution voxel map.
                    if let Some(sink) = sink.as_mut() {
                        if let (Some(samples), Some(triangulator), Some(pose)) = (
                            surface_samples.as_ref(),
                            dense_triangulator.as_ref(),
                            output.pose.current_estimate(),
                        ) {
                            let stereo = triangulator.stereo();
                            let raw_frame_points: Vec<[f32; 3]> = samples
                                .samples
                                .iter()
                                .map(|sample| {
                                    let z = sample.depth_m;
                                    let x = (sample.u - stereo.left().cx) * z / stereo.fx();
                                    let y = (sample.v - stereo.left().cy) * z / stereo.fy();
                                    [x, y, z]
                                })
                                .collect();
                            let surface = kiko_slam::generate_stable_surface_points(
                                &samples.samples,
                                stereo.fx(),
                                stereo.fy(),
                                stereo.left().cx,
                                stereo.left().cy,
                                stereo.baseline_m(),
                                left.data(),
                                left.width(),
                                left.height(),
                                &dense_config,
                            );
                            if let Err(err) = sink.log_surface_observations(
                                left.timestamp(),
                                &raw_frame_points,
                                &surface.points,
                                &surface.stats,
                                pose.cam_from_map_pose32(),
                                &output.diagnostics,
                                true,
                                output.keyframe.is_some(),
                            ) {
                                eprintln!("stable surface: {err}");
                            }
                        }
                    }
                    if output.keyframe.is_some() {
                        keyframes += 1;
                    }
                    if let Some(sink) = sink.as_mut() {
                        if output.keyframe.is_some() || loop_applied {
                            let snapshot = tracker.covisibility_snapshot();
                            if let Err(err) =
                                sink.log_covisibility_graph(left.timestamp(), &snapshot)
                            {
                                eprintln!("rerun log error: {err}");
                            }
                        }
                    }
                } else if let Some(sink) = sink.as_mut() {
                    if let Err(err) = sink.log_frames(&left, &right) {
                        eprintln!("rerun log error: {err}");
                    }
                }

                if let Some(sink) = sink.as_mut() {
                    if let Some(batch) = imu.as_ref() {
                        if let Err(err) = sink.log_imu_batch(batch) {
                            eprintln!("rerun imu log error: {err}");
                        }
                    }
                    if let Some(pose) = output.pose.current_estimate() {
                        if let Err(err) = sink.log_tracking_pose(timestamp, pose) {
                            eprintln!("rerun log error: {err}");
                        } else {
                            poses_logged += 1;
                        }
                    }
                    if let Some(vio_telemetry) = output.vio_telemetry.as_ref() {
                        if let Err(err) = sink.log_vio_telemetry(timestamp, vio_telemetry) {
                            eprintln!("rerun imu log error: {err}");
                        }
                    }
                    if let Err(err) = sink.log_system_health(timestamp, &output.health) {
                        eprintln!("rerun health error: {err}");
                    }
                    if let Err(err) = sink.log_diagnostics(timestamp, &output.diagnostics) {
                        eprintln!("rerun diagnostics error: {err}");
                    }
                    for event in &output.events {
                        if let Err(err) = sink.log_event(timestamp, event) {
                            eprintln!("rerun event error: {err}");
                        }
                    }
                }
                processed += 1;
            }
            Err(err) => {
                inference_errors += 1;
                eprintln!("tracker error: {err}");
            }
        }

        if let Some(limit) = args.dataset.max_pairs {
            if attempted >= limit {
                break;
            }
        }

        next_bundle = lookahead_bundle;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let fps = if elapsed > 0.0 {
        processed as f64 / elapsed
    } else {
        0.0
    };

    eprintln!(
        "done: attempted={attempted}, processed={processed}, elapsed={elapsed:.2}s, fps={fps:.2}, read_errors={read_errors}, tracker_errors={inference_errors}, prefetch_errors={prefetch_errors}, poses_logged={poses_logged}, keyframes={keyframes}"
    );

    Ok(())
}

fn report_error_chain(context: &str, error: &(dyn std::error::Error + 'static)) {
    eprintln!("{context}: {error}");
    let mut nested = error.source();
    while let Some(cause) = nested {
        eprintln!("  caused by: {cause}");
        nested = cause.source();
    }
}

fn should_prefetch_lookahead(attempted: usize, max_pairs: Option<usize>) -> bool {
    max_pairs.is_none_or(|limit| attempted < limit)
}

#[cfg(test)]
mod tests {
    use super::{SLAM_ENV_HELP, should_prefetch_lookahead};

    #[test]
    fn pair_limit_does_not_spawn_unused_prefetch_work() {
        assert!(should_prefetch_lookahead(10, None));
        assert!(should_prefetch_lookahead(9, Some(10)));
        assert!(!should_prefetch_lookahead(10, Some(10)));
        assert!(!should_prefetch_lookahead(11, Some(10)));
    }

    #[test]
    fn slam_env_help_mentions_confirmed_surface_sigma_env() {
        assert!(SLAM_ENV_HELP.contains("KIKO_SURFACE_MAX_CONFIRMED_STD_DEV_M"));
    }

    #[test]
    fn slam_env_help_mentions_surface_predictive_consistency_env() {
        assert!(SLAM_ENV_HELP.contains("KIKO_SURFACE_MAX_PREDICTIVE_CONSISTENCY_SCORE"));
    }

    #[test]
    fn slam_env_help_mentions_surface_pose_quality_gate_env() {
        assert!(SLAM_ENV_HELP.contains("KIKO_SURFACE_MAX_TRACKED_REPROJECTION_RMSE_PX"));
    }

    #[test]
    fn slam_env_help_mentions_surface_pose_quality_support_env() {
        assert!(SLAM_ENV_HELP.contains("KIKO_SURFACE_MIN_PROJECTABLE_TRACKED_OBSERVATIONS"));
    }

    #[test]
    fn slam_env_help_mentions_runtime_imu_override_file_env() {
        assert!(SLAM_ENV_HELP.contains("KIKO_IMU_CALIBRATION_FILE"));
    }
}
