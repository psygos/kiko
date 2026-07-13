use kiko_slam::{
    BackendConfig, DownscaleFactor, GlobalDescriptorConfig, KeyframePolicy, KeypointLimit,
    LmConfig, LocalBaConfig, LoopClosureConfig, LoopClosureConfigInput, LoopSubsystemConfig,
    ProjectedMatcherConfig, RansacConfig, RedundancyPolicy, RelocalizationConfig,
    TrackerRuntimeConfig, TrackingMatcher, TriangulationConfig,
};

use kiko_slam::env::{try_env_bool, try_env_f32, try_env_string, try_env_usize};

// BA defaults (overridable via KIKO_BA_* / KIKO_LM_* env vars)
const DEFAULT_BA_WINDOW: usize = 10;
const DEFAULT_BA_ITERS: usize = 6;
const DEFAULT_BA_MIN_OBS: usize = 8;
const DEFAULT_BA_HUBER_PX: f32 = 3.0;
const DEFAULT_BA_DAMPING: f32 = 1e-3;
const DEFAULT_LM_FACTOR: f32 = 10.0;
const DEFAULT_LM_MIN: f32 = 1e-8;
const DEFAULT_LM_MAX: f32 = 1e4;

// Keyframe policy defaults
const DEFAULT_KEYFRAME_PARALLAX_PX: f32 = 40.0;
const DEFAULT_KEYFRAME_COVISIBILITY: f32 = 0.3;
const DEFAULT_KEYFRAME_REDUNDANT_COVISIBILITY: f32 = 0.9;
#[derive(Debug)]
enum TrackingMatcherParseError {
    ConflictingAliases { primary: String, legacy: String },
    Unknown { value: String },
}

impl std::fmt::Display for TrackingMatcherParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConflictingAliases { primary, legacy } => write!(
                f,
                "KIKO_TRACKING_MATCHER ({primary:?}) conflicts with legacy KIKO_TRACK_MATCHER ({legacy:?})"
            ),
            Self::Unknown { value } => write!(
                f,
                "unknown tracking matcher {value:?}; expected `lightglue` or `projected`"
            ),
        }
    }
}

impl std::error::Error for TrackingMatcherParseError {}

#[derive(Debug)]
struct UnsupportedBaMotionRegularizerError {
    value: f32,
}

impl std::fmt::Display for UnsupportedBaMotionRegularizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KIKO_BA_MOTION_WEIGHT={} is unsupported: the removed heuristic mixed metres and radians and biased consecutive keyframes toward zero relative motion; use measured VIO/IMU constraints instead",
            self.value
        )
    }
}

impl std::error::Error for UnsupportedBaMotionRegularizerError {}

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
        try_env_usize("KIKO_KEYFRAME_MIN_POINTS")?.unwrap_or(defaults.min_keyframe_points);
    let refresh_inliers =
        try_env_usize("KIKO_KEYFRAME_REFRESH_INLIERS")?.unwrap_or(defaults.refresh_inliers);
    let parallax_px =
        try_env_f32("KIKO_KEYFRAME_PARALLAX_PX")?.unwrap_or(DEFAULT_KEYFRAME_PARALLAX_PX);
    let min_covisibility =
        try_env_f32("KIKO_KEYFRAME_COVISIBILITY")?.unwrap_or(DEFAULT_KEYFRAME_COVISIBILITY);
    let redundant_covisibility = try_env_f32("KIKO_KEYFRAME_REDUNDANT_COVISIBILITY")?
        .unwrap_or(DEFAULT_KEYFRAME_REDUNDANT_COVISIBILITY);
    let min_inliers = try_env_usize("KIKO_TRACK_MIN_INLIERS")?.unwrap_or(defaults.min_inliers);
    let ransac = RansacConfig::default().try_with_min_inliers(min_inliers)?;
    let tracking_matcher = tracking_matcher_from_env(
        overrides
            .tracking_matcher
            .unwrap_or(TrackingMatcher::LightGlue),
    )?;
    let ba_config = build_ba_config_with_overrides(overrides)?;
    let keyframe_policy = KeyframePolicy::new(refresh_inliers, parallax_px, min_covisibility)?;
    let redundancy = Some(RedundancyPolicy::new(redundant_covisibility)?);
    let backend = if try_env_bool("KIKO_BACKEND_ASYNC")?.unwrap_or(true) {
        Some(BackendConfig::new(
            try_env_usize("KIKO_BACKEND_QUEUE_DEPTH")?.unwrap_or(2),
        )?)
    } else {
        None
    };
    let loop_closure_requested = match overrides.loop_closure {
        Some(value) => value,
        None => try_env_bool("KIKO_LOOP_CLOSURE")?.unwrap_or(true),
    };
    let learned_descriptors_enabled = match overrides.learned_descriptors {
        Some(value) => value,
        None => try_env_bool("KIKO_LEARNED_DESCRIPTORS")?.unwrap_or(true),
    };
    let loop_closure_enabled = loop_closure_requested && learned_descriptors_enabled;
    let relocalization_enabled = match overrides.relocalization {
        Some(value) => value,
        None => try_env_bool("KIKO_RELOCALIZATION")?.unwrap_or(true),
    };
    let loop_subsystem = if loop_closure_enabled {
        let loop_cfg = build_loop_closure_config_from_env()?;
        let descriptor_cfg = GlobalDescriptorConfig::new(
            try_env_usize("KIKO_DESCRIPTOR_QUEUE_DEPTH")?.unwrap_or(2),
        )?;
        eprintln!(
            "loop config: similarity={:.3} descriptor_match={:.3} min_inliers={} max_candidates={} temporal_gap={} min_streak={} max_translation_m={:.3} max_rotation_deg={:.3} ransac_iters={} ransac_px={:.3} ransac_min_inliers={}",
            loop_cfg.similarity_threshold(),
            loop_cfg.descriptor_match_threshold(),
            loop_cfg.min_inliers(),
            loop_cfg.max_candidates(),
            loop_cfg.temporal_gap(),
            loop_cfg.min_streak(),
            loop_cfg.max_correction_translation_m(),
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
    let runtime_defaults = TrackerRuntimeConfig::default();
    let runtime = TrackerRuntimeConfig::try_new(
        try_env_usize("KIKO_BACKEND_MAX_RESPAWNS")?
            .unwrap_or(runtime_defaults.backend_max_respawns() as usize),
        try_env_usize("KIKO_DESCRIPTOR_MAX_RESPAWNS")?
            .unwrap_or(runtime_defaults.descriptor_max_respawns() as usize),
        try_env_bool("KIKO_TRACK_TRACE")?.unwrap_or(runtime_defaults.trace_transitions()),
        try_env_usize("KIKO_MAP_CULL_MIN_OBSERVATIONS")?
            .unwrap_or(runtime_defaults.cull_min_observations().get()),
    )?;
    #[cfg(feature = "vio")]
    let runtime = runtime.try_with_vio(
        try_env_usize("KIKO_VIO_WINDOW")?
            .unwrap_or(runtime_defaults.vio_window_capacity().frames().get()),
        try_env_usize("KIKO_VIO_ITERS")?.unwrap_or(runtime_defaults.vio_max_iterations().get()),
    )?;

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
        runtime,
        #[cfg(feature = "vio")]
        vio_enabled: match overrides.vio_enabled {
            Some(value) => value,
            None => try_env_bool("KIKO_VIO")?.unwrap_or(false),
        },
    })
}

fn tracking_matcher_from_env(
    default: TrackingMatcher,
) -> Result<TrackingMatcher, Box<dyn std::error::Error>> {
    let primary = try_env_string("KIKO_TRACKING_MATCHER")?;
    let legacy = try_env_string("KIKO_TRACK_MATCHER")?;
    let matcher = match (primary, legacy) {
        (Some(primary), Some(legacy)) if !primary.eq_ignore_ascii_case(&legacy) => {
            return Err(Box::new(TrackingMatcherParseError::ConflictingAliases {
                primary,
                legacy,
            }));
        }
        (Some(value), _) | (None, Some(value)) => value,
        (None, None) => match default {
            TrackingMatcher::LightGlue => "lightglue".to_string(),
            TrackingMatcher::Projected(_) => "projected".to_string(),
        },
    };
    let matcher = matcher.trim().to_ascii_lowercase();

    match matcher.as_str() {
        "projected" | "projection" | "local" => {
            let defaults = match default {
                TrackingMatcher::Projected(config) => config,
                TrackingMatcher::LightGlue => ProjectedMatcherConfig::jetson_default(),
            };
            Ok(TrackingMatcher::Projected(ProjectedMatcherConfig::new(
                try_env_f32("KIKO_PROJECTED_MATCH_RADIUS_PX")?
                    .unwrap_or(defaults.search_radius_px()),
                try_env_f32("KIKO_PROJECTED_MATCH_MIN_SIMILARITY")?
                    .unwrap_or(defaults.min_similarity()),
                try_env_usize("KIKO_PROJECTED_MATCH_MIN_MATCHES")?
                    .unwrap_or(defaults.min_matches()),
                try_env_usize("KIKO_PROJECTED_MATCH_MIN_INLIERS")?
                    .unwrap_or(defaults.min_inliers()),
            )?))
        }
        "lightglue" | "learned" => Ok(TrackingMatcher::LightGlue),
        _ => Err(Box::new(TrackingMatcherParseError::Unknown {
            value: matcher,
        })),
    }
}

fn build_loop_closure_config_from_env() -> Result<LoopClosureConfig, Box<dyn std::error::Error>> {
    let mut input = LoopClosureConfigInput::default();
    if let Some(v) = try_env_f32("KIKO_LOOP_SIMILARITY_THRESHOLD")? {
        input.similarity_threshold = v;
    }
    if let Some(v) = try_env_f32("KIKO_LOOP_DESCRIPTOR_MATCH_THRESHOLD")? {
        input.descriptor_match_threshold = v;
    }
    if let Some(v) = try_env_usize("KIKO_LOOP_MIN_INLIERS")? {
        input.min_inliers = v;
    }
    if let Some(v) = try_env_usize("KIKO_LOOP_MAX_CANDIDATES")? {
        input.max_candidates = v;
    }
    if let Some(v) = try_env_usize("KIKO_LOOP_TEMPORAL_GAP")? {
        input.temporal_gap = v;
    }
    if let Some(v) = try_env_usize("KIKO_LOOP_MIN_STREAK")? {
        input.min_streak = v;
    }
    if let Some(v) = try_env_f32("KIKO_LOOP_MAX_CORRECTION_TRANSLATION_M")? {
        input.max_correction_translation_m = v;
    }
    if let Some(v) = try_env_f32("KIKO_LOOP_MAX_CORRECTION_ROTATION_DEG")? {
        input.max_correction_rotation_deg = v;
    }

    let ransac = input.ransac;
    let max_iterations =
        try_env_usize("KIKO_LOOP_RANSAC_MAX_ITERATIONS")?.unwrap_or(ransac.max_iterations());
    let reprojection_threshold_px =
        try_env_f32("KIKO_LOOP_RANSAC_THRESHOLD_PX")?.unwrap_or(ransac.reprojection_threshold_px());
    let min_inliers =
        try_env_usize("KIKO_LOOP_RANSAC_MIN_INLIERS")?.unwrap_or(ransac.min_inliers());
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
    let window = match overrides.ba_window {
        Some(value) => value,
        None => try_env_usize("KIKO_BA_WINDOW")?.unwrap_or(DEFAULT_BA_WINDOW),
    };
    let iters = match overrides.ba_iters {
        Some(value) => value,
        None => try_env_usize("KIKO_BA_ITERS")?.unwrap_or(DEFAULT_BA_ITERS),
    };
    let min_obs = match overrides.ba_min_obs {
        Some(value) => value,
        None => try_env_usize("KIKO_BA_MIN_OBS")?.unwrap_or(DEFAULT_BA_MIN_OBS),
    };
    let huber = try_env_f32("KIKO_BA_HUBER_PX")?.unwrap_or(DEFAULT_BA_HUBER_PX);
    let initial_lambda = try_env_f32("KIKO_BA_DAMPING")?.unwrap_or(DEFAULT_BA_DAMPING);
    let lambda_factor = try_env_f32("KIKO_LM_FACTOR")?.unwrap_or(DEFAULT_LM_FACTOR);
    let min_lambda = try_env_f32("KIKO_LM_MIN")?.unwrap_or(DEFAULT_LM_MIN);
    let max_lambda = try_env_f32("KIKO_LM_MAX")?.unwrap_or(DEFAULT_LM_MAX);
    if let Some(value) = try_env_f32("KIKO_BA_MOTION_WEIGHT")? {
        return Err(Box::new(UnsupportedBaMotionRegularizerError { value }));
    }
    let default_lm = LmConfig::default();
    let lm = LmConfig::new(
        initial_lambda,
        lambda_factor,
        min_lambda,
        max_lambda,
        default_lm.rho_accept(),
        default_lm.rho_good(),
    )?;
    let config = LocalBaConfig::new(window, iters, min_obs, huber, lm)?;
    eprintln!(
        "local BA: window={} iters={} min_obs={} huber_px={} lm_init={} lm_factor={} lm_min={} lm_max={}",
        config.window(),
        config.max_iterations(),
        config.min_observations(),
        config.huber_delta_px(),
        config.lm().initial_lambda(),
        config.lm().lambda_factor(),
        config.lm().min_lambda(),
        config.lm().max_lambda()
    );
    Ok(config)
}
