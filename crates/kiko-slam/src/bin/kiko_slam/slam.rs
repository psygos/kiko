use std::time::Instant;

use clap::Args;

use kiko_slam::dataset::DatasetReader;
use kiko_slam::{
    CalibrationBundle, PinholeIntrinsics, RectifiedStereo, RerunSink, SlamTracker, VizPacket,
};

use crate::args::{DatasetArgs, InferenceArgs, InferenceConfig, RectifyArgs, RerunArgs};
use crate::config::{TrackerDefaults, build_rectified_stereo_config, build_tracker_config};
use crate::rerun_recording;

const SLAM_ENV_HELP: &str = "\
ENVIRONMENT VARIABLES (expert tuning):

  Bundle Adjustment:
    KIKO_BA_WINDOW=10           Sliding window size
    KIKO_BA_ITERS=6             Max LM iterations
    KIKO_BA_MIN_OBS=8           Min observations per landmark
    KIKO_BA_HUBER_PX=3.0        Huber robust kernel delta (pixels)
    KIKO_BA_DAMPING=0.001       Initial LM lambda
    KIKO_BA_MOTION_WEIGHT=0.0   Motion prior regularization weight
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
    KIKO_DENSE_MAX_POINTS=30000           Max stable surface observations per keyframe";

#[derive(Args, Clone, Debug)]
#[command(
    about = "Run full SLAM pipeline on a recorded dataset",
    after_long_help = SLAM_ENV_HELP
)]
pub struct SlamArgs {
    #[command(flatten)]
    pub inference: InferenceArgs,
    #[command(flatten)]
    pub rerun: RerunArgs,
    #[command(flatten)]
    pub rectify: RectifyArgs,
    #[command(flatten)]
    pub dataset: DatasetArgs,
}

pub fn run_slam(args: &SlamArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut reader = DatasetReader::open(&args.dataset.path)?;
    #[cfg(feature = "vio")]
    if kiko_slam::env::env_bool("KIKO_VIO").unwrap_or(false) && reader.meta().imu.is_none() {
        return Err("KIKO_VIO=true requires IMU data in the dataset".into());
    }
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
    let dense_cloud_enabled = kiko_slam::env::env_bool("KIKO_DENSE_CLOUD").unwrap_or(false);
    let dense_config = kiko_slam::DenseCloudConfig::from_env();
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
    let intrinsics = PinholeIntrinsics::try_from(&reader.calibration().left)?;
    let calibration =
        CalibrationBundle::from_dataset_calibration(intrinsics, rectified, reader.calibration())?;
    #[cfg(feature = "vio")]
    if kiko_slam::env::env_bool("KIKO_VIO").unwrap_or(false) && !calibration.has_imu() {
        return Err(
            "KIKO_VIO=true requires IMU calibration via calibration.json or KIKO_IMU_* env".into(),
        );
    }

    let InferenceConfig {
        superpoint,
        superpoint_right,
        lightglue,
        end2end: _,
        key_limit,
        downscale,
    } = inference;

    let tracker_config = build_tracker_config(
        TrackerDefaults {
            min_keyframe_points: 12,
            refresh_inliers: 12,
            min_inliers: 8,
        },
        key_limit,
        downscale,
    )?;
    let mut sink = match rerun_recording(&args.rerun, "kiko-slam-dataset-odometry") {
        Ok(rec) => Some(RerunSink::new(rec, args.rerun.rerun_decimation)),
        Err(err) => {
            eprintln!("failed to initialize rerun; continuing headless: {err}");
            None
        }
    };
    let mut tracker = SlamTracker::try_new(superpoint, lightglue, calibration, tracker_config)?;
    // Create a second SP session for detection prefetch pipelining
    if let Some(sp_right) = superpoint_right {
        tracker.return_prefetch_sp(sp_right);
        eprintln!("detection prefetch pipeline: enabled");
    }

    let start = Instant::now();
    let mut attempted = 0usize;
    let mut processed = 0usize;
    let mut inference_errors = 0usize;
    let mut read_errors = 0usize;
    let mut poses_logged = 0usize;
    let mut keyframes = 0usize;

    // SP prefetch pipeline: overlap SP(N+1) with tracker processing of frame N
    type PrefetchResult = (
        kiko_slam::SuperPoint,
        kiko_slam::FrameId,
        Option<std::sync::Arc<kiko_slam::Detections>>,
    );
    let mut pending_prefetch: Option<std::thread::JoinHandle<PrefetchResult>> = None;
    let ds = inference.downscale;
    let max_kp = inference.key_limit.get();

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

        // Collect prefetched SP for THIS frame (launched during previous iteration)
        let prefetched_left = if let Some(handle) = pending_prefetch.take() {
            match handle.join() {
                Ok((sp_session, fid, Some(dets))) => {
                    tracker.return_prefetch_sp(sp_session);
                    Some((fid, dets))
                }
                Ok((sp_session, _fid, None)) => {
                    tracker.return_prefetch_sp(sp_session);
                    None
                }
                Err(_) => None,
            }
        } else {
            None
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

        // Launch SP prefetch for the NEXT frame on a background thread
        if let Some(ref next_b) = lookahead_bundle {
            if let Some(mut sp) = tracker.take_prefetch_sp() {
                let next_left = next_b.pair().left().clone();
                let next_fid = next_left.frame_id();
                pending_prefetch = Some(std::thread::spawn(move || {
                    let result = sp
                        .detect_with_downscale(&next_left, ds)
                        .ok()
                        .map(|d| std::sync::Arc::new(d.top_k(max_kp)));
                    (sp, next_fid, result)
                }));
            }
        }

        // Process THIS frame — SP prefetch for NEXT frame runs in parallel
        match tracker.process_capture_with_prefetch(bundle, prefetched_left) {
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
                    let surface_samples = if output.keyframe.is_some() {
                        dense_triangulator
                            .as_ref()
                            .map(|tri| tri.extract_stereo_samples(&matches))
                    } else {
                        None
                    };

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
                        if let Some(samples) = surface_samples {
                            if let Some(pose) = output.pose.as_ref() {
                                let stereo = dense_triangulator.as_ref().unwrap().stereo();
                                let surface = kiko_slam::generate_stable_surface_points(
                                    &samples,
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
                                    &surface.points,
                                    &surface.stats,
                                    pose.cam_from_map_pose32(),
                                    &output.diagnostics,
                                ) {
                                    eprintln!("stable surface: {err}");
                                }
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
                    if let Some(pose) = output.pose.as_ref() {
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
        "done: attempted={attempted}, processed={processed}, elapsed={elapsed:.2}s, fps={fps:.2}, read_errors={read_errors}, tracker_errors={inference_errors}, poses_logged={poses_logged}, keyframes={keyframes}"
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::SLAM_ENV_HELP;

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
}
