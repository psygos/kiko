use std::time::Instant;

use clap::Args;

use kiko_slam::dataset::DatasetReader;
use kiko_slam::{ProjectedMatcherConfig, TrackingMatcher};
use kiko_slam::{RerunSink, RerunSinkConfig, SlamTracker, VizPacket};

use crate::args::{
    DatasetArgs, InferenceArgs, InferenceConfig, InferencePurpose, RerunArgs, RerunOutput,
    RunProfileArg,
};
use crate::bench::duration_percentiles;
use crate::config::{TrackerDefaults, TrackerOverrides, build_tracker_config_with_overrides};
use crate::{RunFailureCapture, RunMetricsStatus, SkippedDatasetEntryError, combine_rerun_results};
use crate::{next_selected_success, rerun_recording, verify_run_integrity};

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

  Visual-Inertial Optimization (requires `vio` feature):
    KIKO_VIO_WINDOW=5           Max frames; minimum 2, dense workspace allocated at startup
    KIKO_VIO_ITERS=3            Max LM iterations per VIO solve

  Keyframe Policy:
    KIKO_KEYFRAME_MIN_POINTS=12          Min triangulated points to accept keyframe
    KIKO_KEYFRAME_REFRESH_INLIERS=12     Min inliers to skip keyframe insertion
    KIKO_KEYFRAME_PARALLAX_PX=40.0       Parallax threshold (pixels)
    KIKO_KEYFRAME_COVISIBILITY=0.6       Min covisibility ratio for connection
    KIKO_KEYFRAME_REDUNDANT_COVISIBILITY=0.9  Covisibility ratio for redundancy culling
    KIKO_TRACK_MIN_INLIERS=8             Min PnP RANSAC inliers
    KIKO_TRACKING_MATCHER=projected      projected or lightglue tracking matcher
    KIKO_PROJECTED_MATCH_RADIUS_PX=32    Projected matcher search radius
    KIKO_PROJECTED_MATCH_MIN_DOT_PRODUCT=0.45  Raw descriptor dot-product gate
    KIKO_PROJECTED_MATCH_MIN_MATCHES=32  Fallback to LightGlue below this match count
    KIKO_PROJECTED_MATCH_MIN_INLIERS=24  Fallback to LightGlue below this inlier count
    KIKO_PROJECTED_MATCH_MIN_SIMILARITY  Deprecated misnamed alias for MIN_DOT_PRODUCT

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
    /// Disable Rerun output without attempting to spawn or connect to a viewer.
    #[arg(
        long,
        env = "KIKO_HEADLESS",
        default_value_t = false,
        conflicts_with_all = ["save_rrd", "rerun_url", "rerun_laptop", "rerun_serve"]
    )]
    pub headless: bool,
    #[command(flatten)]
    pub inference: InferenceArgs,
    #[command(flatten)]
    pub rerun: RerunArgs,
    #[command(flatten)]
    pub dataset: DatasetArgs,
    /// Successful pipeline pairs processed for warm-up but excluded from steady-state metrics
    #[arg(long, env = "KIKO_SLAM_WARMUP_PAIRS", default_value_t = 4)]
    pub warmup_pairs: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PoseOutcomeCounts {
    current: usize,
    predicted: usize,
    stale: usize,
    unavailable: usize,
}

impl PoseOutcomeCounts {
    fn record(&mut self, status: &kiko_slam::PoseStatus) {
        let count = match status {
            kiko_slam::PoseStatus::Current(_) => &mut self.current,
            kiko_slam::PoseStatus::Predicted(_) => &mut self.predicted,
            kiko_slam::PoseStatus::Stale { .. } => &mut self.stale,
            kiko_slam::PoseStatus::Unavailable => &mut self.unavailable,
        };
        *count = count.saturating_add(1);
    }

    fn all_have_current_estimates(self, expected: usize) -> bool {
        expected > 0 && self.current.checked_add(self.predicted) == Some(expected)
    }
}

#[derive(Debug)]
struct UnusableSteadyPoseStreamError {
    steady_processed: usize,
    outcomes: PoseOutcomeCounts,
}

impl std::fmt::Display for UnusableSteadyPoseStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "steady SLAM output did not provide a current estimate for every processed pair: processed={}, current={}, predicted={}, stale={}, unavailable={}",
            self.steady_processed,
            self.outcomes.current,
            self.outcomes.predicted,
            self.outcomes.stale,
            self.outcomes.unavailable
        )
    }
}

impl std::error::Error for UnusableSteadyPoseStreamError {}

#[derive(Debug)]
enum SlamProcessingError {
    PrefetchThread(std::io::Error),
    Triangulation(kiko_slam::TriangulationError),
    StableSurface(kiko_slam::StableSurfaceGenerationError),
}

impl std::fmt::Display for SlamProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PrefetchThread(source) => write!(f, "inference prefetch failed: {source}"),
            Self::Triangulation(source) => write!(f, "stereo sample extraction failed: {source}"),
            Self::StableSurface(source) => {
                write!(f, "stable surface generation failed: {source}")
            }
        }
    }
}

impl std::error::Error for SlamProcessingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::PrefetchThread(source) => Some(source),
            Self::Triangulation(source) => Some(source),
            Self::StableSurface(source) => Some(source),
        }
    }
}

impl From<kiko_slam::TriangulationError> for SlamProcessingError {
    fn from(source: kiko_slam::TriangulationError) -> Self {
        Self::Triangulation(source)
    }
}

impl From<kiko_slam::StableSurfaceGenerationError> for SlamProcessingError {
    fn from(source: kiko_slam::StableSurfaceGenerationError) -> Self {
        Self::StableSurface(source)
    }
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
    let rerun_output = RerunOutput::try_from_args(&args.rerun)?;
    let rerun_sink_config = if args.headless {
        None
    } else {
        Some(RerunSinkConfig::from_environment()?)
    };
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
    let expected_pairs = args.dataset.selected_pair_count(stats.paired_count)?;
    if args.warmup_pairs >= expected_pairs {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "SLAM warm-up must leave at least one steady-state pair: selected={expected_pairs}, warmup_pairs={}",
                args.warmup_pairs
            ),
        )
        .into());
    }
    eprintln!(
        "slam selection: available_pairs={} skip_frames={} expected_pairs={} warmup_pairs={} steady_pairs={}",
        stats.paired_count,
        args.dataset.skip_frames,
        expected_pairs,
        args.warmup_pairs,
        expected_pairs - args.warmup_pairs
    );

    let inference_args = args.inference.with_profile_defaults(args.profile)?;
    let inference_init_start = Instant::now();
    let inference = InferenceConfig::from_args(&inference_args, InferencePurpose::Slam)?;
    let inference_init = inference_init_start.elapsed();
    eprintln!(
        "inference session initialization: {:.2}ms",
        inference_init.as_secs_f64() * 1000.0
    );

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
            "stable surface map: enabled (measured sparse stereo -> surface belief; TSDF integration unsupported)"
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
    let superpoint_right = superpoint_right.require_ready_if_applicable()?;
    let lightglue_prefetch = lightglue_prefetch.require_ready_if_applicable()?;

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
    let mut sink = if args.headless {
        eprintln!("rerun: disabled explicitly (--headless)");
        None
    } else {
        match rerun_recording(rerun_output.destination(), "kiko-slam-dataset-odometry") {
            Ok(rec) => Some(RerunSink::from_config(
                rec,
                rerun_output.decimation(),
                rerun_sink_config.expect("non-headless runs parse Rerun sink settings"),
            )),
            Err(err) => {
                if rerun_output.has_explicit_destination() {
                    return Err(Box::new(err));
                }
                eprintln!("local Rerun viewer unavailable; continuing headless: {err}");
                None
            }
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

    let mut entries_consumed = 0usize;
    let mut tracker_attempts = 0usize;
    let mut processed = 0usize;
    let mut tracker_errors = 0usize;
    let mut prefetch_fallbacks = 0usize;
    let mut read_errors = 0usize;
    let mut visualization_errors = 0usize;
    let mut visualization_fallbacks = 0usize;
    let mut poses_logged = 0usize;
    let mut keyframes = 0usize;
    let mut service_intervals = Vec::with_capacity(expected_pairs);
    let mut failure_sources = RunFailureCapture::default();
    let mut pose_outcomes = PoseOutcomeCounts::default();
    let mut steady_pose_outcomes = PoseOutcomeCounts::default();

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
        for skipped in 0..skip {
            match bundles_iter.next() {
                Some(Ok(_)) => {}
                Some(Err(err)) => {
                    return Err(Box::new(SkippedDatasetEntryError {
                        command: "SLAM",
                        entry_number: skipped + 1,
                        requested_skip: skip,
                        source: err,
                    }));
                }
                None => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        format!(
                            "requested --skip-frames={skip}, but the dataset ended after {skipped} entries"
                        ),
                    )
                    .into());
                }
            }
        }
    }
    let start = Instant::now();
    let mut steady_start = (args.warmup_pairs == 0).then(Instant::now);
    // Read-ahead: grab the first bundle
    let mut next_bundle = next_selected_success(
        &mut bundles_iter,
        &mut entries_consumed,
        expected_pairs,
        |err| {
            failure_sources.report_and_record(&mut read_errors, "dataset_read", "read error", err);
        },
    );

    let processing = (|| -> Result<(), SlamProcessingError> {
        while let Some(bundle) = next_bundle.take() {
            let service_interval_start = Instant::now();
            let processed_before = processed;
            tracker_attempts += 1;
            let left = bundle.pair().left().clone();
            let right = bundle.pair().right().clone();
            let imu = bundle.imu().batch().cloned();

            // Collect prefetched SP+LG for THIS frame (launched during previous iteration)
            let (prefetched_left, prefetched_matches) = if let Some(handle) =
                pending_prefetch.take()
            {
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
                                        failure_sources.report_and_record(
                                        &mut prefetch_fallbacks,
                                        "inference_prefetch",
                                        "speculative LightGlue prefetch failed; tracker will match synchronously",
                                        source,
                                    );
                                        None
                                    }
                                };
                                (Some((result.frame_id, detections)), speculative_matches)
                            }
                            PrefetchInferenceOutcome::DetectionFailed { source } => {
                                failure_sources.report_and_record(
                                    &mut prefetch_fallbacks,
                                    "inference_prefetch",
                                    "SuperPoint prefetch failed; tracker will detect synchronously",
                                    source,
                                );
                                (None, None)
                            }
                        }
                    }
                    Err(payload) => {
                        return Err(SlamProcessingError::PrefetchThread(std::io::Error::other(
                            format!(
                                "inference prefetch thread panicked: {}",
                                kiko_slam::panic_payload_to_string(payload.as_ref())
                            ),
                        )));
                    }
                }
            } else {
                (None, None)
            };

            // Count every iterator entry so a read error cannot move the selected boundary.
            let lookahead_bundle = next_selected_success(
                &mut bundles_iter,
                &mut entries_consumed,
                expected_pairs,
                |err| {
                    failure_sources.report_and_record(
                        &mut read_errors,
                        "dataset_read",
                        "read error",
                        err,
                    );
                },
            );

            // Launch SP + speculative LG prefetch for the NEXT frame on a background thread
            if let Some(ref next_b) = lookahead_bundle {
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
                        let outcome = match sp.detect_with_downscale_limited(&next_left, ds, max_kp)
                        {
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
            match tracker.process_capture_with_prefetch(bundle, prefetched_left, prefetched_matches)
            {
                Ok(output) => {
                    pose_outcomes.record(&output.pose);
                    if processed >= args.warmup_pairs {
                        steady_pose_outcomes.record(&output.pose);
                    }
                    if !matches!(
                        &output.pose,
                        kiko_slam::PoseStatus::Current(_) | kiko_slam::PoseStatus::Predicted(_)
                    ) {
                        eprintln!(
                            "non-current pose: frame_id={:?} pose={:?} health={:?} diagnostics={:?} events={:?}",
                            output.frame_id,
                            output.pose,
                            output.health,
                            output.diagnostics,
                            output.events
                        );
                    }
                    let timestamp = left.timestamp();
                    // A new mapping-session identity is a causal boundary for the
                    // accumulated surface map, so clear it before logging any output
                    // that belongs to the new session.
                    if let Some(sink) = sink.as_mut() {
                        for event in output.events.iter().filter(|event| {
                            matches!(
                                event,
                                kiko_slam::DiagnosticEvent::MappingSessionReset { .. }
                            )
                        }) {
                            if let Err(err) = sink.log_event(timestamp, event) {
                                failure_sources.report_and_record(
                                    &mut visualization_errors,
                                    "visualization_output",
                                    "rerun mapping-session reset error",
                                    err,
                                );
                            }
                        }
                    }
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
                                        failure_sources.report_and_record(
                                            &mut visualization_errors,
                                            "visualization_output",
                                            "rerun point log error",
                                            err,
                                        );
                                    }
                                }
                                Err(err) => {
                                    failure_sources.report_and_record(
                                        &mut visualization_fallbacks,
                                        "visualization_fallback",
                                        "viz packet error; falling back to raw stereo views",
                                        err,
                                    );
                                    if let Err(log_err) = sink.log_frames(&left, &right) {
                                        failure_sources.report_and_record(
                                            &mut visualization_errors,
                                            "visualization_output",
                                            "rerun fallback frame log error",
                                            log_err,
                                        );
                                    }
                                }
                            }
                        }
                        // Generate stable sparse surface observations for the low-resolution voxel map.
                        if let Some(sink) = sink.as_mut() {
                            if let (Some(samples), Some(pose)) =
                                (surface_samples.as_ref(), output.pose.current_estimate())
                            {
                                let surface = kiko_slam::generate_stable_surface_points(
                                    samples,
                                    &left,
                                    &dense_config,
                                )?;
                                if let Err(err) = sink.log_surface_observations(
                                    left.timestamp(),
                                    &surface.measured_camera_points_m,
                                    &surface.points,
                                    &surface.stats,
                                    pose.cam_from_map_pose32(),
                                    &output.diagnostics,
                                    true,
                                    output.keyframe.is_some(),
                                ) {
                                    failure_sources.report_and_record(
                                        &mut visualization_errors,
                                        "visualization_output",
                                        "stable surface log error",
                                        err,
                                    );
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
                                    failure_sources.report_and_record(
                                        &mut visualization_errors,
                                        "visualization_output",
                                        "rerun covisibility graph log error",
                                        err,
                                    );
                                }
                            }
                        }
                    } else if let Some(sink) = sink.as_mut() {
                        if let Err(err) = sink.log_frames(&left, &right) {
                            failure_sources.report_and_record(
                                &mut visualization_errors,
                                "visualization_output",
                                "rerun frame log error",
                                err,
                            );
                        }
                    }

                    if let Some(sink) = sink.as_mut() {
                        if let Some(batch) = imu.as_ref() {
                            if let Err(err) = sink.log_imu_batch(batch) {
                                failure_sources.report_and_record(
                                    &mut visualization_errors,
                                    "visualization_output",
                                    "rerun IMU log error",
                                    err,
                                );
                            }
                        }
                        if let Some(pose) = output.pose.current_estimate() {
                            if let Err(err) = sink.log_tracking_pose(timestamp, pose) {
                                failure_sources.report_and_record(
                                    &mut visualization_errors,
                                    "visualization_output",
                                    "rerun tracking pose log error",
                                    err,
                                );
                            } else {
                                poses_logged += 1;
                            }
                        }
                        if let Some(vio_telemetry) = output.vio_telemetry.as_ref() {
                            if let Err(err) = sink.log_vio_telemetry(timestamp, vio_telemetry) {
                                failure_sources.report_and_record(
                                    &mut visualization_errors,
                                    "visualization_output",
                                    "rerun VIO telemetry log error",
                                    err,
                                );
                            }
                        }
                        if let Err(err) = sink.log_system_health(timestamp, &output.health) {
                            failure_sources.report_and_record(
                                &mut visualization_errors,
                                "visualization_output",
                                "rerun health log error",
                                err,
                            );
                        }
                        if let Err(err) = sink.log_diagnostics(timestamp, &output.diagnostics) {
                            failure_sources.report_and_record(
                                &mut visualization_errors,
                                "visualization_output",
                                "rerun diagnostics log error",
                                err,
                            );
                        }
                        for event in output.events.iter().filter(|event| {
                            !matches!(
                                event,
                                kiko_slam::DiagnosticEvent::MappingSessionReset { .. }
                            )
                        }) {
                            if let Err(err) = sink.log_event(timestamp, event) {
                                failure_sources.report_and_record(
                                    &mut visualization_errors,
                                    "visualization_output",
                                    "rerun event log error",
                                    err,
                                );
                            }
                        }
                    }
                    processed += 1;
                }
                Err(err) => {
                    failure_sources.report_and_record(
                        &mut tracker_errors,
                        "tracker",
                        "tracker error",
                        err,
                    );
                }
            }

            if processed > processed_before {
                service_intervals.push(service_interval_start.elapsed());
                if processed == args.warmup_pairs {
                    steady_start = Some(Instant::now());
                }
            }
            next_bundle = lookahead_bundle;
        }
        Ok(())
    })();
    let finalization = sink
        .take()
        .map(|sink| sink.finish_with_timeout(rerun_output.finish_timeout().get()))
        .unwrap_or(Ok(()));
    combine_rerun_results(processing, finalization)?;

    let elapsed = start.elapsed().as_secs_f64();
    let steady_elapsed = steady_start.map_or(0.0, |start| start.elapsed().as_secs_f64());
    let steady_processed = processed.saturating_sub(args.warmup_pairs);
    let steady_fps = if steady_elapsed > 0.0 {
        steady_processed as f64 / steady_elapsed
    } else {
        0.0
    };
    let fps = if elapsed > 0.0 {
        processed as f64 / elapsed
    } else {
        0.0
    };
    let integrity = verify_run_integrity(
        "slam",
        "pairs",
        processed,
        &[
            ("dataset_read", read_errors),
            ("tracker", tracker_errors),
            ("inference_prefetch", prefetch_fallbacks),
            ("visualization_fallback", visualization_fallbacks),
            ("visualization_output", visualization_errors),
        ],
        failure_sources,
    );
    let exact_completion = entries_consumed == expected_pairs
        && tracker_attempts == expected_pairs
        && processed == expected_pairs;
    let usable_pose_output = steady_pose_outcomes.all_have_current_estimates(steady_processed);
    let metrics_status =
        RunMetricsStatus::from_outcome(integrity.is_ok(), exact_completion, usable_pose_output);
    eprintln!("slam metrics status: {}", metrics_status.as_str());

    eprintln!(
        "done: expected={expected_pairs}, entries_consumed={entries_consumed}, tracker_attempts={tracker_attempts}, processed={processed}, elapsed={elapsed:.2}s, fps={fps:.2}, warmup_processed={}, steady_processed={steady_processed}, steady_elapsed={steady_elapsed:.2}s, steady_fps={steady_fps:.2}, read_errors={read_errors}, tracker_errors={tracker_errors}, prefetch_fallbacks={prefetch_fallbacks}, visualization_fallbacks={visualization_fallbacks}, visualization_errors={visualization_errors}, poses_logged={poses_logged}, keyframes={keyframes}",
        processed.min(args.warmup_pairs)
    );
    eprintln!(
        "pose outcomes: total_current={} total_predicted={} total_stale={} total_unavailable={} steady_current={} steady_predicted={} steady_stale={} steady_unavailable={}",
        pose_outcomes.current,
        pose_outcomes.predicted,
        pose_outcomes.stale,
        pose_outcomes.unavailable,
        steady_pose_outcomes.current,
        steady_pose_outcomes.predicted,
        steady_pose_outcomes.stale,
        steady_pose_outcomes.unavailable
    );
    if let Some(measured) = service_intervals.get(args.warmup_pairs..)
        && !measured.is_empty()
    {
        if let Some(latency) = duration_percentiles(measured.iter().copied()) {
            eprintln!(
                "steady pipeline service interval ms (median/p95, samples={}): {:.2}/{:.2}",
                measured.len(),
                latency.median_ms,
                latency.p95_ms
            );
        }
    }

    integrity?;
    if !exact_completion {
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "SLAM did not consume the selected dataset exactly: expected_pairs={expected_pairs}, entries_consumed={entries_consumed}, tracker_attempts={tracker_attempts}, processed={processed}"
            ),
        )
        .into());
    }
    if !usable_pose_output {
        return Err(Box::new(UnusableSteadyPoseStreamError {
            steady_processed,
            outcomes: steady_pose_outcomes,
        }));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{PoseOutcomeCounts, SLAM_ENV_HELP};
    use crate::next_selected_success;

    #[test]
    fn selected_entry_limit_counts_errors_and_never_reads_beyond_bound() {
        let mut entries = [Ok(0), Err("bad entry"), Ok(2), Ok(3)].into_iter();
        let mut consumed = 0;
        let mut errors = Vec::new();

        assert_eq!(
            next_selected_success(&mut entries, &mut consumed, 3, |error| errors.push(error)),
            Some(0)
        );
        assert_eq!(
            next_selected_success(&mut entries, &mut consumed, 3, |error| errors.push(error)),
            Some(2)
        );
        assert_eq!(
            next_selected_success(&mut entries, &mut consumed, 3, |error| errors.push(error)),
            None
        );
        assert_eq!(consumed, 3);
        assert_eq!(errors, ["bad entry"]);
        assert_eq!(entries.next(), Some(Ok(3)));
    }

    #[test]
    fn steady_pose_stream_requires_a_current_estimate_for_every_pair() {
        assert!(
            PoseOutcomeCounts {
                current: 27,
                predicted: 1,
                ..PoseOutcomeCounts::default()
            }
            .all_have_current_estimates(28)
        );
        assert!(
            !PoseOutcomeCounts {
                current: 27,
                stale: 1,
                ..PoseOutcomeCounts::default()
            }
            .all_have_current_estimates(28)
        );
        assert!(!PoseOutcomeCounts::default().all_have_current_estimates(0));
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

    #[test]
    fn slam_env_help_describes_vio_workspace_controls() {
        assert!(SLAM_ENV_HELP.contains("KIKO_VIO_WINDOW=5"));
        assert!(SLAM_ENV_HELP.contains("minimum 2"));
        assert!(SLAM_ENV_HELP.contains("KIKO_VIO_ITERS=3"));
    }

    #[test]
    fn slam_env_help_names_raw_projected_descriptor_dot_product_truthfully() {
        assert!(SLAM_ENV_HELP.contains("KIKO_PROJECTED_MATCH_MIN_DOT_PRODUCT=0.45"));
        assert!(SLAM_ENV_HELP.contains("Raw descriptor dot-product gate"));
        assert!(SLAM_ENV_HELP.contains("Deprecated misnamed alias"));
    }
}
