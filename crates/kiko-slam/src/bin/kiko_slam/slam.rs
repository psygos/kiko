use std::time::Instant;

use clap::Args;

use kiko_slam::dataset::DatasetReader;
use kiko_slam::{
    CalibrationBundle, PinholeIntrinsics, RectifiedStereo, RerunSink, SlamTracker, VizPacket,
};

use crate::args::{DatasetArgs, InferenceArgs, InferenceConfig, RectifyArgs, RerunArgs};
use crate::config::{build_rectified_stereo_config, build_tracker_config, TrackerDefaults};
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
    KIKO_DESCRIPTOR_QUEUE_DEPTH=2 Descriptor extraction queue depth";

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
    let intrinsics = PinholeIntrinsics::try_from(&reader.calibration().left)?;
    let calibration = CalibrationBundle::visual_only(intrinsics, rectified);

    let InferenceConfig {
        superpoint,
        lightglue,
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

    let rec = rerun_recording(&args.rerun, "kiko-slam-dataset-odometry")?;
    let mut sink = RerunSink::new(rec, args.rerun.rerun_decimation);
    let mut tracker = SlamTracker::try_new(
        superpoint,
        lightglue,
        calibration,
        tracker_config,
    )?;

    let start = Instant::now();
    let mut attempted = 0usize;
    let mut processed = 0usize;
    let mut inference_errors = 0usize;
    let mut read_errors = 0usize;
    let mut poses_logged = 0usize;
    let mut keyframes = 0usize;

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

        let left = pair.left().clone();
        let right = pair.right().clone();

        match tracker.process(pair) {
            Ok(output) => {
                let timestamp = left.timestamp();
                let loop_applied = output.events.iter().any(|event| {
                    matches!(
                        event,
                        kiko_slam::DiagnosticEvent::LoopClosureDetected { .. }
                    )
                });
                if let Some(matches) = output.stereo_matches {
                    let points = output
                        .keyframe
                        .as_ref()
                        .map(|kf| kf.landmarks())
                        .filter(|pts| !pts.is_empty());
                    if let Ok(packet) = VizPacket::try_new(left.clone(), right.clone(), matches) {
                        if let Err(err) = sink.log_with_points(&packet, points) {
                            eprintln!("rerun log error: {err}");
                        }
                    }
                    if output.keyframe.is_some() {
                        keyframes += 1;
                    }
                    if output.keyframe.is_some() || loop_applied {
                        let snapshot = tracker.covisibility_snapshot();
                        if let Err(err) = sink.log_covisibility_graph(left.timestamp(), &snapshot) {
                            eprintln!("rerun log error: {err}");
                        }
                    }
                } else if let Err(err) = sink.log_frames(&left, &right) {
                    eprintln!("rerun log error: {err}");
                }

                if let Some(pose) = output.pose.as_ref() {
                    let pose_map = pose.cam_from_map_pose32();
                    if let Err(err) = sink.log_pose(timestamp, &pose_map) {
                        eprintln!("rerun log error: {err}");
                    } else {
                        poses_logged += 1;
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
