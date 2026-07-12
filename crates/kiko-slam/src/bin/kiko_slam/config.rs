use kiko_slam::{
    BackendConfig, DownscaleFactor, GlobalDescriptorConfig, KeyframePolicy, KeypointLimit,
    LmConfig, LocalBaConfig, LoopClosureConfig, LoopClosureConfigInput, LoopSubsystemConfig,
    ProjectedMatcherConfig, RansacConfig, RectificationMode, RectifiedStereoConfig,
    RedundancyPolicy, RelocalizationConfig, TrackingMatcher, TriangulationConfig,
};

use kiko_slam::env::{env_bool, env_f32, env_usize};

use crate::args::RectifyArgs;

// BA defaults (overridable via KIKO_BA_* / KIKO_LM_* env vars)
const DEFAULT_BA_WINDOW: usize = 10;
const DEFAULT_BA_ITERS: usize = 6;
const DEFAULT_BA_MIN_OBS: usize = 8;
const DEFAULT_BA_HUBER_PX: f32 = 3.0;
const DEFAULT_BA_DAMPING: f32 = 1e-3;
const DEFAULT_LM_FACTOR: f32 = 10.0;
const DEFAULT_LM_MIN: f32 = 1e-8;
const DEFAULT_LM_MAX: f32 = 1e4;
const DEFAULT_BA_MOTION_WEIGHT: f32 = 0.0;

// Keyframe policy defaults
const DEFAULT_KEYFRAME_PARALLAX_PX: f32 = 40.0;
const DEFAULT_KEYFRAME_COVISIBILITY: f32 = 0.3;
const DEFAULT_KEYFRAME_REDUNDANT_COVISIBILITY: f32 = 0.9;
const MIN_PNP_CORRESPONDENCES: usize = 4;

pub struct TrackerDefaults {
    pub min_keyframe_points: usize,
    pub refresh_inliers: usize,
    pub min_inliers: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TrackerOverrides {
    #[allow(dead_code)]
    pub vio_enabled: Option<bool>,
    pub ba_window: Option<usize>,
    pub ba_iters: Option<usize>,
    pub ba_min_obs: Option<usize>,
    pub tracking_matcher: Option<TrackingMatcher>,
    pub loop_closure: Option<bool>,
    pub learned_descriptors: Option<bool>,
    pub relocalization: Option<bool>,
}

#[allow(dead_code)]
pub fn build_tracker_config(
    defaults: TrackerDefaults,
    key_limit: KeypointLimit,
    downscale: DownscaleFactor,
) -> Result<kiko_slam::TrackerConfig, Box<dyn std::error::Error>> {
    build_tracker_config_with_overrides(defaults, key_limit, downscale, TrackerOverrides::default())
}

pub fn build_tracker_config_with_overrides(
    defaults: TrackerDefaults,
    key_limit: KeypointLimit,
    downscale: DownscaleFactor,
    overrides: TrackerOverrides,
) -> Result<kiko_slam::TrackerConfig, Box<dyn std::error::Error>> {
    let min_keyframe_points =
        env_usize("KIKO_KEYFRAME_MIN_POINTS").unwrap_or(defaults.min_keyframe_points);
    let refresh_inliers =
        env_usize("KIKO_KEYFRAME_REFRESH_INLIERS").unwrap_or(defaults.refresh_inliers);
    let parallax_px = env_f32("KIKO_KEYFRAME_PARALLAX_PX").unwrap_or(DEFAULT_KEYFRAME_PARALLAX_PX);
    let min_covisibility =
        env_f32("KIKO_KEYFRAME_COVISIBILITY").unwrap_or(DEFAULT_KEYFRAME_COVISIBILITY);
    let redundant_covisibility = env_f32("KIKO_KEYFRAME_REDUNDANT_COVISIBILITY")
        .unwrap_or(DEFAULT_KEYFRAME_REDUNDANT_COVISIBILITY);
    let min_inliers = env_usize("KIKO_TRACK_MIN_INLIERS").unwrap_or(defaults.min_inliers);
    let ransac = RansacConfig::default().try_with_min_inliers(min_inliers)?;
    let tracking_matcher = tracking_matcher_from_env(
        overrides
            .tracking_matcher
            .unwrap_or(TrackingMatcher::LightGlue),
    );
    let ba_config = build_ba_config_with_overrides(overrides)?;
    let keyframe_policy = KeyframePolicy::new(refresh_inliers, parallax_px, min_covisibility)?;
    let redundancy = Some(RedundancyPolicy::new(redundant_covisibility)?);
    let backend = if env_bool("KIKO_BACKEND_ASYNC").unwrap_or(true) {
        Some(BackendConfig::new(
            env_usize("KIKO_BACKEND_QUEUE_DEPTH").unwrap_or(2),
        )?)
    } else {
        None
    };
    let loop_closure_requested = overrides
        .loop_closure
        .unwrap_or_else(|| env_bool("KIKO_LOOP_CLOSURE").unwrap_or(true));
    let learned_descriptors_enabled = overrides
        .learned_descriptors
        .unwrap_or_else(|| env_bool("KIKO_LEARNED_DESCRIPTORS").unwrap_or(true));
    let loop_closure_enabled = loop_closure_requested && learned_descriptors_enabled;
    let relocalization_enabled = overrides
        .relocalization
        .unwrap_or_else(|| env_bool("KIKO_RELOCALIZATION").unwrap_or(true));
    let loop_subsystem = if loop_closure_enabled {
        let loop_cfg = build_loop_closure_config_from_env()?;
        let descriptor_cfg =
            GlobalDescriptorConfig::new(env_usize("KIKO_DESCRIPTOR_QUEUE_DEPTH").unwrap_or(2))?;
        eprintln!(
            "loop config: similarity={:.3} descriptor_match={:.3} min_inliers={} max_candidates={} temporal_gap={} min_streak={} max_translation_m={:.3} max_rotation_deg={:.3} ransac_iters={} ransac_px={:.3} ransac_min_inliers={}",
            loop_cfg.similarity_threshold(),
            loop_cfg.descriptor_match_threshold(),
            loop_cfg.min_inliers(),
            loop_cfg.max_candidates(),
            loop_cfg.temporal_gap(),
            loop_cfg.min_streak(),
            loop_cfg.max_correction_translation(),
            loop_cfg.max_correction_rotation_deg(),
            loop_cfg.ransac().max_iterations(),
            loop_cfg.ransac().reprojection_threshold_px(),
            loop_cfg.ransac().min_inliers(),
        );
        if relocalization_enabled {
            LoopSubsystemConfig::with_relocalization(
                loop_cfg,
                descriptor_cfg,
                RelocalizationConfig::default(),
            )
        } else {
            LoopSubsystemConfig::loop_closure_only(loop_cfg, descriptor_cfg)
        }
    } else {
        if loop_closure_requested && !learned_descriptors_enabled {
            eprintln!("learned descriptors disabled; disabling loop closure");
        }
        if relocalization_enabled {
            eprintln!(
                "relocalization requested but loop closure is disabled; disabling relocalization"
            );
        }
        LoopSubsystemConfig::Disabled
    };

    eprintln!(
        "tracker: keyframe_min_points={min_keyframe_points} refresh_inliers={refresh_inliers} parallax_px={parallax_px:.1} min_covisibility={min_covisibility:.2} redundant_covisibility={redundant_covisibility:.2} min_inliers={min_inliers} downscale={downscale} max_keypoints={key_limit} tracking_matcher={tracking_matcher:?} loop_closure={loop_closure_enabled} learned_descriptors={} relocalization={}",
        learned_descriptors_enabled && loop_closure_enabled,
        relocalization_enabled && loop_closure_enabled,
    );

    Ok(kiko_slam::TrackerConfig {
        max_keypoints: key_limit,
        downscale,
        tracking_matcher,
        min_keyframe_points,
        ransac,
        triangulation: TriangulationConfig::default(),
        keyframe_policy,
        ba: ba_config,
        redundancy,
        backend,
        loop_subsystem,
        #[cfg(feature = "vio")]
        vio_enabled: overrides
            .vio_enabled
            .unwrap_or_else(|| env_bool("KIKO_VIO").unwrap_or(false)),
    })
}

fn tracking_matcher_from_env(default: TrackingMatcher) -> TrackingMatcher {
    let matcher = std::env::var("KIKO_TRACKING_MATCHER")
        .or_else(|_| std::env::var("KIKO_TRACK_MATCHER"))
        .ok()
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_else(|| match default {
            TrackingMatcher::LightGlue => "lightglue".to_string(),
            TrackingMatcher::Projected(_) => "projected".to_string(),
        });

    match matcher.as_str() {
        "projected" | "projection" | "local" => {
            let defaults = match default {
                TrackingMatcher::Projected(config) => config,
                TrackingMatcher::LightGlue => ProjectedMatcherConfig::jetson_default(),
            };
            TrackingMatcher::Projected(ProjectedMatcherConfig {
                search_radius_px: env_f32("KIKO_PROJECTED_MATCH_RADIUS_PX")
                    .filter(|value| value.is_finite() && *value > 0.0)
                    .unwrap_or(defaults.search_radius_px),
                min_similarity: env_f32("KIKO_PROJECTED_MATCH_MIN_SIMILARITY")
                    .filter(|value| value.is_finite())
                    .unwrap_or(defaults.min_similarity),
                min_matches: env_usize("KIKO_PROJECTED_MATCH_MIN_MATCHES")
                    .unwrap_or(defaults.min_matches)
                    .max(MIN_PNP_CORRESPONDENCES),
                min_inliers: env_usize("KIKO_PROJECTED_MATCH_MIN_INLIERS")
                    .unwrap_or(defaults.min_inliers)
                    .max(MIN_PNP_CORRESPONDENCES),
            })
        }
        _ => TrackingMatcher::LightGlue,
    }
}

fn build_loop_closure_config_from_env() -> Result<LoopClosureConfig, Box<dyn std::error::Error>> {
    let mut input = LoopClosureConfigInput::default();
    if let Some(v) = env_f32("KIKO_LOOP_SIMILARITY_THRESHOLD") {
        input.similarity_threshold = v;
    }
    if let Some(v) = env_f32("KIKO_LOOP_DESCRIPTOR_MATCH_THRESHOLD") {
        input.descriptor_match_threshold = v;
    }
    if let Some(v) = env_usize("KIKO_LOOP_MIN_INLIERS") {
        input.min_inliers = v;
    }
    if let Some(v) = env_usize("KIKO_LOOP_MAX_CANDIDATES") {
        input.max_candidates = v;
    }
    if let Some(v) = env_usize("KIKO_LOOP_TEMPORAL_GAP") {
        input.temporal_gap = v;
    }
    if let Some(v) = env_usize("KIKO_LOOP_MIN_STREAK") {
        input.min_streak = v;
    }
    if let Some(v) = env_f32("KIKO_LOOP_MAX_CORRECTION_TRANSLATION_M") {
        input.max_correction_translation = v;
    }
    if let Some(v) = env_f32("KIKO_LOOP_MAX_CORRECTION_ROTATION_DEG") {
        input.max_correction_rotation_deg = v;
    }

    let ransac = input.ransac;
    let max_iterations =
        env_usize("KIKO_LOOP_RANSAC_MAX_ITERATIONS").unwrap_or(ransac.max_iterations());
    let reprojection_threshold_px =
        env_f32("KIKO_LOOP_RANSAC_THRESHOLD_PX").unwrap_or(ransac.reprojection_threshold_px());
    let min_inliers = env_usize("KIKO_LOOP_RANSAC_MIN_INLIERS").unwrap_or(ransac.min_inliers());
    input.ransac = RansacConfig::new(
        max_iterations,
        reprojection_threshold_px,
        min_inliers,
        ransac.seed(),
    )?;

    LoopClosureConfig::new(input).map_err(Into::into)
}

#[allow(dead_code)]
pub fn build_ba_config() -> Result<LocalBaConfig, Box<dyn std::error::Error>> {
    build_ba_config_with_overrides(TrackerOverrides::default())
}

pub fn build_ba_config_with_overrides(
    overrides: TrackerOverrides,
) -> Result<LocalBaConfig, Box<dyn std::error::Error>> {
    let window = overrides
        .ba_window
        .unwrap_or_else(|| env_usize("KIKO_BA_WINDOW").unwrap_or(DEFAULT_BA_WINDOW));
    let iters = overrides
        .ba_iters
        .unwrap_or_else(|| env_usize("KIKO_BA_ITERS").unwrap_or(DEFAULT_BA_ITERS));
    let min_obs = overrides
        .ba_min_obs
        .unwrap_or_else(|| env_usize("KIKO_BA_MIN_OBS").unwrap_or(DEFAULT_BA_MIN_OBS));
    let huber = env_f32("KIKO_BA_HUBER_PX").unwrap_or(DEFAULT_BA_HUBER_PX);
    let initial_lambda = env_f32("KIKO_BA_DAMPING").unwrap_or(DEFAULT_BA_DAMPING);
    let lambda_factor = env_f32("KIKO_LM_FACTOR").unwrap_or(DEFAULT_LM_FACTOR);
    let min_lambda = env_f32("KIKO_LM_MIN").unwrap_or(DEFAULT_LM_MIN);
    let max_lambda = env_f32("KIKO_LM_MAX").unwrap_or(DEFAULT_LM_MAX);
    let motion = env_f32("KIKO_BA_MOTION_WEIGHT").unwrap_or(DEFAULT_BA_MOTION_WEIGHT);
    let default_lm = LmConfig::default();
    let lm = LmConfig::new(
        initial_lambda,
        lambda_factor,
        min_lambda,
        max_lambda,
        default_lm.rho_accept(),
        default_lm.rho_good(),
    )?;
    let config = LocalBaConfig::new(window, iters, min_obs, huber, lm, motion)?;
    eprintln!(
        "local BA: window={} iters={} min_obs={} huber_px={} lm_init={} lm_factor={} lm_min={} lm_max={} motion_weight={}",
        config.window(),
        config.max_iterations(),
        config.min_observations(),
        config.huber_delta_px(),
        config.lm().initial_lambda(),
        config.lm().lambda_factor(),
        config.lm().min_lambda(),
        config.lm().max_lambda(),
        config.motion_prior_weight()
    );
    Ok(config)
}

pub fn build_rectified_stereo_config(args: &RectifyArgs) -> RectifiedStereoConfig {
    RectifiedStereoConfig {
        max_principal_delta_px: args.rectify_tolerance,
        rectification: if args.allow_unrectified {
            RectificationMode::AllowUnrectified
        } else {
            RectificationMode::RequireRectified
        },
    }
}
