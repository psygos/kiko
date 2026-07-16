use std::cmp::Ordering;
use std::num::NonZeroUsize;

use crate::map::{KeyframeId, MapError, MapSnapshot, SlamMap};
use crate::pnp::MIN_PNP_POINTS;
use crate::pose_graph::{EssentialGraphError, PoseGraphError};
use crate::{
    CompactDescriptor, Descriptor, Keypoint, Observation, PinholeIntrinsics, PnpError, Pose,
    RansacConfig, RansacConfigError, solve_pnp_ransac,
};
pub use crate::{CosineSimilarity, CosineSimilarityError};

pub(crate) const GLOBAL_DESCRIPTOR_DIM: usize = crate::DESCRIPTOR_DIM * 2;

const DEFAULT_RELOCALIZATION_MAX_ATTEMPTS: NonZeroUsize = NonZeroUsize::MIN.saturating_add(29);
const DEFAULT_RELOCALIZATION_MIN_INLIERS: NonZeroUsize = NonZeroUsize::MIN.saturating_add(19);
const DEFAULT_RELOCALIZATION_MAX_CANDIDATES: NonZeroUsize = NonZeroUsize::MIN.saturating_add(2);
const DEFAULT_RELOCALIZATION_MIN_CONFIRMATIONS: NonZeroUsize = NonZeroUsize::MIN.saturating_add(1);
const DEFAULT_RELOCALIZATION_DESCRIPTOR_MATCH_THRESHOLD: f32 = 0.7;
const DEFAULT_RELOCALIZATION_MAX_TRANSLATION_DELTA_M: f32 = 1.5;
const DEFAULT_RELOCALIZATION_MAX_ROTATION_DELTA_DEG: f32 = 10.0;

const DEFAULT_LOOP_SIMILARITY_THRESHOLD: f32 = 0.75;
const DEFAULT_LOOP_DESCRIPTOR_MATCH_THRESHOLD: f32 = 0.7;
const DEFAULT_LOOP_MIN_INLIERS: NonZeroUsize = NonZeroUsize::MIN.saturating_add(19);
const DEFAULT_LOOP_MAX_CANDIDATES: NonZeroUsize = NonZeroUsize::MIN.saturating_add(2);
const DEFAULT_LOOP_TEMPORAL_GAP: NonZeroUsize = NonZeroUsize::MIN.saturating_add(29);
const DEFAULT_LOOP_MIN_STREAK: NonZeroUsize = NonZeroUsize::MIN.saturating_add(2);
const DEFAULT_LOOP_MAX_CORRECTION_TRANSLATION_M: f32 = 5.0;
const DEFAULT_LOOP_MAX_CORRECTION_ROTATION_DEG: f32 = 30.0;

#[derive(Clone, Copy, Debug)]
pub struct LoopClosureConfig {
    similarity_threshold: f32,
    descriptor_match_threshold: f32,
    min_inliers: NonZeroUsize,
    max_candidates: NonZeroUsize,
    temporal_gap: NonZeroUsize,
    min_streak: NonZeroUsize,
    max_correction_translation_m: f32,
    max_correction_rotation_deg: f32,
    ransac: RansacConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct LoopClosureConfigInput {
    pub similarity_threshold: f32,
    pub descriptor_match_threshold: f32,
    pub min_inliers: usize,
    pub max_candidates: usize,
    pub temporal_gap: usize,
    pub min_streak: usize,
    pub max_correction_translation_m: f32,
    pub max_correction_rotation_deg: f32,
    pub ransac: RansacConfig,
}

impl Default for LoopClosureConfigInput {
    fn default() -> Self {
        Self {
            similarity_threshold: DEFAULT_LOOP_SIMILARITY_THRESHOLD,
            descriptor_match_threshold: DEFAULT_LOOP_DESCRIPTOR_MATCH_THRESHOLD,
            min_inliers: DEFAULT_LOOP_MIN_INLIERS.get(),
            max_candidates: DEFAULT_LOOP_MAX_CANDIDATES.get(),
            temporal_gap: DEFAULT_LOOP_TEMPORAL_GAP.get(),
            min_streak: DEFAULT_LOOP_MIN_STREAK.get(),
            max_correction_translation_m: DEFAULT_LOOP_MAX_CORRECTION_TRANSLATION_M,
            max_correction_rotation_deg: DEFAULT_LOOP_MAX_CORRECTION_ROTATION_DEG,
            ransac: RansacConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RelocalizationConfig {
    max_attempts: NonZeroUsize,
    min_inliers: NonZeroUsize,
    max_candidates: NonZeroUsize,
    descriptor_match_threshold: f32,
    min_confirmations: NonZeroUsize,
    max_translation_delta_m: f32,
    max_rotation_delta_deg: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct RelocalizationConfigInput {
    pub max_attempts: usize,
    pub min_inliers: usize,
    pub max_candidates: usize,
    pub descriptor_match_threshold: f32,
    pub min_confirmations: usize,
    pub max_translation_delta_m: f32,
    pub max_rotation_delta_deg: f32,
}

impl Default for RelocalizationConfigInput {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_RELOCALIZATION_MAX_ATTEMPTS.get(),
            min_inliers: DEFAULT_RELOCALIZATION_MIN_INLIERS.get(),
            max_candidates: DEFAULT_RELOCALIZATION_MAX_CANDIDATES.get(),
            descriptor_match_threshold: DEFAULT_RELOCALIZATION_DESCRIPTOR_MATCH_THRESHOLD,
            min_confirmations: DEFAULT_RELOCALIZATION_MIN_CONFIRMATIONS.get(),
            max_translation_delta_m: DEFAULT_RELOCALIZATION_MAX_TRANSLATION_DELTA_M,
            max_rotation_delta_deg: DEFAULT_RELOCALIZATION_MAX_ROTATION_DELTA_DEG,
        }
    }
}

#[derive(Debug)]
pub enum RelocalizationConfigError {
    ZeroMaxAttempts,
    ZeroMinInliers,
    ZeroMaxCandidates,
    ZeroMinConfirmations,
    TooFewMaxAttempts {
        max_attempts: usize,
        min_confirmations: usize,
    },
    TooFewMinInliers {
        value: usize,
        min: usize,
    },
    DescriptorMatchThresholdOutOfRange {
        value: f32,
    },
    NonPositiveMaxTranslationDelta {
        value: f32,
    },
    InvalidMaxRotationDeltaDeg {
        value: f32,
    },
}

impl std::fmt::Display for RelocalizationConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RelocalizationConfigError::ZeroMaxAttempts => {
                write!(f, "relocalization max attempts must be > 0")
            }
            RelocalizationConfigError::ZeroMinInliers => {
                write!(f, "relocalization min inliers must be > 0")
            }
            RelocalizationConfigError::ZeroMaxCandidates => {
                write!(f, "relocalization max candidates must be > 0")
            }
            RelocalizationConfigError::ZeroMinConfirmations => {
                write!(f, "relocalization min confirmations must be > 0")
            }
            RelocalizationConfigError::TooFewMaxAttempts {
                max_attempts,
                min_confirmations,
            } => write!(
                f,
                "relocalization max attempts must be >= min confirmations ({min_confirmations}), got {max_attempts}"
            ),
            RelocalizationConfigError::TooFewMinInliers { value, min } => {
                write!(
                    f,
                    "relocalization min inliers must be >= {min}, got {value}"
                )
            }
            RelocalizationConfigError::DescriptorMatchThresholdOutOfRange { value } => write!(
                f,
                "relocalization descriptor match threshold must be in (0, 1], got {value}"
            ),
            RelocalizationConfigError::NonPositiveMaxTranslationDelta { value } => write!(
                f,
                "relocalization max translation delta must be > 0, got {value}"
            ),
            RelocalizationConfigError::InvalidMaxRotationDeltaDeg { value } => write!(
                f,
                "relocalization max rotation delta must be in (0, 180], got {value}"
            ),
        }
    }
}

impl std::error::Error for RelocalizationConfigError {}

impl RelocalizationConfig {
    pub fn new(input: RelocalizationConfigInput) -> Result<Self, RelocalizationConfigError> {
        let RelocalizationConfigInput {
            max_attempts,
            min_inliers,
            max_candidates,
            descriptor_match_threshold,
            min_confirmations,
            max_translation_delta_m,
            max_rotation_delta_deg,
        } = input;
        let max_attempts =
            NonZeroUsize::new(max_attempts).ok_or(RelocalizationConfigError::ZeroMaxAttempts)?;
        let min_inliers =
            NonZeroUsize::new(min_inliers).ok_or(RelocalizationConfigError::ZeroMinInliers)?;
        let max_candidates = NonZeroUsize::new(max_candidates)
            .ok_or(RelocalizationConfigError::ZeroMaxCandidates)?;
        let min_confirmations = NonZeroUsize::new(min_confirmations)
            .ok_or(RelocalizationConfigError::ZeroMinConfirmations)?;
        if max_attempts < min_confirmations {
            return Err(RelocalizationConfigError::TooFewMaxAttempts {
                max_attempts: max_attempts.get(),
                min_confirmations: min_confirmations.get(),
            });
        }
        if min_inliers.get() < MIN_PNP_POINTS {
            return Err(RelocalizationConfigError::TooFewMinInliers {
                value: min_inliers.get(),
                min: MIN_PNP_POINTS,
            });
        }
        if !descriptor_match_threshold.is_finite()
            || descriptor_match_threshold <= 0.0
            || descriptor_match_threshold > 1.0
        {
            return Err(
                RelocalizationConfigError::DescriptorMatchThresholdOutOfRange {
                    value: descriptor_match_threshold,
                },
            );
        }
        if !max_translation_delta_m.is_finite() || max_translation_delta_m <= 0.0 {
            return Err(RelocalizationConfigError::NonPositiveMaxTranslationDelta {
                value: max_translation_delta_m,
            });
        }
        if !max_rotation_delta_deg.is_finite()
            || max_rotation_delta_deg <= 0.0
            || max_rotation_delta_deg > 180.0
        {
            return Err(RelocalizationConfigError::InvalidMaxRotationDeltaDeg {
                value: max_rotation_delta_deg,
            });
        }

        Ok(Self {
            max_attempts,
            min_inliers,
            max_candidates,
            descriptor_match_threshold,
            min_confirmations,
            max_translation_delta_m,
            max_rotation_delta_deg,
        })
    }

    pub fn max_attempts(self) -> usize {
        self.max_attempts.get()
    }

    pub fn min_inliers(self) -> usize {
        self.min_inliers.get()
    }

    pub fn max_candidates(self) -> usize {
        self.max_candidates.get()
    }

    pub fn descriptor_match_threshold(self) -> f32 {
        self.descriptor_match_threshold
    }

    pub fn min_confirmations(self) -> usize {
        self.min_confirmations.get()
    }

    pub fn max_translation_delta_m(self) -> f32 {
        self.max_translation_delta_m
    }

    pub fn max_rotation_delta_deg(self) -> f32 {
        self.max_rotation_delta_deg
    }
}

impl Default for RelocalizationConfig {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_RELOCALIZATION_MAX_ATTEMPTS,
            min_inliers: DEFAULT_RELOCALIZATION_MIN_INLIERS,
            max_candidates: DEFAULT_RELOCALIZATION_MAX_CANDIDATES,
            descriptor_match_threshold: DEFAULT_RELOCALIZATION_DESCRIPTOR_MATCH_THRESHOLD,
            min_confirmations: DEFAULT_RELOCALIZATION_MIN_CONFIRMATIONS,
            max_translation_delta_m: DEFAULT_RELOCALIZATION_MAX_TRANSLATION_DELTA_M,
            max_rotation_delta_deg: DEFAULT_RELOCALIZATION_MAX_ROTATION_DELTA_DEG,
        }
    }
}

#[derive(Debug)]
pub enum LoopClosureConfigError {
    SimilarityThresholdOutOfRange { value: f32 },
    DescriptorMatchThresholdOutOfRange { value: f32 },
    ZeroMinInliers,
    ZeroMaxCandidates,
    ZeroTemporalGap,
    ZeroMinStreak,
    TooFewMinInliers { value: usize, min: usize },
    InvalidMaxCorrectionTranslationM { value: f32 },
    InvalidMaxCorrectionRotationDeg { value: f32 },
}

impl std::fmt::Display for LoopClosureConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopClosureConfigError::SimilarityThresholdOutOfRange { value } => {
                write!(
                    f,
                    "loop similarity threshold must be in (0, 1], got {value}"
                )
            }
            LoopClosureConfigError::DescriptorMatchThresholdOutOfRange { value } => write!(
                f,
                "loop descriptor match threshold must be in (0, 1], got {value}"
            ),
            LoopClosureConfigError::ZeroMinInliers => {
                write!(f, "loop min inliers must be > 0")
            }
            LoopClosureConfigError::ZeroMaxCandidates => {
                write!(f, "loop max candidates must be > 0")
            }
            LoopClosureConfigError::ZeroTemporalGap => {
                write!(f, "loop temporal gap must be > 0")
            }
            LoopClosureConfigError::ZeroMinStreak => {
                write!(f, "loop min streak must be > 0")
            }
            LoopClosureConfigError::TooFewMinInliers { value, min } => {
                write!(f, "loop min inliers must be >= {min}, got {value}")
            }
            LoopClosureConfigError::InvalidMaxCorrectionTranslationM { value } => write!(
                f,
                "loop max correction translation must be positive finite meters, got {value}"
            ),
            LoopClosureConfigError::InvalidMaxCorrectionRotationDeg { value } => write!(
                f,
                "loop max correction rotation must be finite degrees in (0, 180], got {value}"
            ),
        }
    }
}

impl std::error::Error for LoopClosureConfigError {}

impl LoopClosureConfig {
    pub fn new(input: LoopClosureConfigInput) -> Result<Self, LoopClosureConfigError> {
        let LoopClosureConfigInput {
            similarity_threshold,
            descriptor_match_threshold,
            min_inliers,
            max_candidates,
            temporal_gap,
            min_streak,
            max_correction_translation_m,
            max_correction_rotation_deg,
            ransac,
        } = input;
        if !similarity_threshold.is_finite()
            || similarity_threshold <= 0.0
            || similarity_threshold > 1.0
        {
            return Err(LoopClosureConfigError::SimilarityThresholdOutOfRange {
                value: similarity_threshold,
            });
        }
        if !descriptor_match_threshold.is_finite()
            || descriptor_match_threshold <= 0.0
            || descriptor_match_threshold > 1.0
        {
            return Err(LoopClosureConfigError::DescriptorMatchThresholdOutOfRange {
                value: descriptor_match_threshold,
            });
        }
        let min_inliers =
            NonZeroUsize::new(min_inliers).ok_or(LoopClosureConfigError::ZeroMinInliers)?;
        let max_candidates =
            NonZeroUsize::new(max_candidates).ok_or(LoopClosureConfigError::ZeroMaxCandidates)?;
        let temporal_gap =
            NonZeroUsize::new(temporal_gap).ok_or(LoopClosureConfigError::ZeroTemporalGap)?;
        let min_streak =
            NonZeroUsize::new(min_streak).ok_or(LoopClosureConfigError::ZeroMinStreak)?;
        if min_inliers.get() < MIN_PNP_POINTS {
            return Err(LoopClosureConfigError::TooFewMinInliers {
                value: min_inliers.get(),
                min: MIN_PNP_POINTS,
            });
        }
        if !max_correction_translation_m.is_finite() || max_correction_translation_m <= 0.0 {
            return Err(LoopClosureConfigError::InvalidMaxCorrectionTranslationM {
                value: max_correction_translation_m,
            });
        }
        if !max_correction_rotation_deg.is_finite()
            || max_correction_rotation_deg <= 0.0
            || max_correction_rotation_deg > 180.0
        {
            return Err(LoopClosureConfigError::InvalidMaxCorrectionRotationDeg {
                value: max_correction_rotation_deg,
            });
        }
        Ok(Self {
            similarity_threshold,
            descriptor_match_threshold,
            min_inliers,
            max_candidates,
            temporal_gap,
            min_streak,
            max_correction_translation_m,
            max_correction_rotation_deg,
            ransac,
        })
    }

    pub fn similarity_threshold(self) -> f32 {
        self.similarity_threshold
    }

    pub fn descriptor_match_threshold(self) -> f32 {
        self.descriptor_match_threshold
    }

    pub fn min_inliers(self) -> usize {
        self.min_inliers.get()
    }

    pub fn max_candidates(self) -> usize {
        self.max_candidates.get()
    }

    pub fn temporal_gap(self) -> usize {
        self.temporal_gap.get()
    }

    pub fn min_streak(self) -> usize {
        self.min_streak.get()
    }

    pub fn max_correction_translation_m(self) -> f32 {
        self.max_correction_translation_m
    }

    pub fn max_correction_rotation_deg(self) -> f32 {
        self.max_correction_rotation_deg
    }

    pub fn ransac(self) -> RansacConfig {
        self.ransac
    }
}

impl Default for LoopClosureConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: DEFAULT_LOOP_SIMILARITY_THRESHOLD,
            descriptor_match_threshold: DEFAULT_LOOP_DESCRIPTOR_MATCH_THRESHOLD,
            min_inliers: DEFAULT_LOOP_MIN_INLIERS,
            max_candidates: DEFAULT_LOOP_MAX_CANDIDATES,
            temporal_gap: DEFAULT_LOOP_TEMPORAL_GAP,
            min_streak: DEFAULT_LOOP_MIN_STREAK,
            max_correction_translation_m: DEFAULT_LOOP_MAX_CORRECTION_TRANSLATION_M,
            max_correction_rotation_deg: DEFAULT_LOOP_MAX_CORRECTION_ROTATION_DEG,
            ransac: RansacConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopApplyErrorKind {
    StaleCorrection,
    MissingKeyframe,
    MissingMapPoint,
    MapMutation,
    EssentialGraph,
    PoseGraph,
    PoseConversion,
    MapFrameAlignment,
}

impl std::fmt::Display for LoopApplyErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StaleCorrection => write!(f, "stale correction"),
            Self::MissingKeyframe => write!(f, "missing keyframe"),
            Self::MissingMapPoint => write!(f, "missing map point"),
            Self::MapMutation => write!(f, "map mutation"),
            Self::EssentialGraph => write!(f, "essential-graph mutation"),
            Self::PoseGraph => write!(f, "pose-graph optimization"),
            Self::PoseConversion => write!(f, "pose conversion"),
            Self::MapFrameAlignment => write!(f, "map/odometry frame alignment"),
        }
    }
}

#[derive(Debug)]
pub enum LoopApplyError {
    StaleCorrection {
        proof: MapSnapshot,
        current: MapSnapshot,
    },
    Map {
        source: MapError,
    },
    EssentialGraph {
        source: EssentialGraphError,
    },
    PoseGraph {
        source: PoseGraphError,
    },
    PoseConversion {
        operation: &'static str,
        keyframe_id: Option<KeyframeId>,
        source: crate::Pose64Error,
    },
    MapFrameAlignment {
        source: crate::GeometryError,
    },
}

impl LoopApplyError {
    pub fn kind(&self) -> LoopApplyErrorKind {
        match self {
            Self::StaleCorrection { .. } => LoopApplyErrorKind::StaleCorrection,
            Self::Map {
                source: MapError::KeyframeNotFound(_),
            } => LoopApplyErrorKind::MissingKeyframe,
            Self::Map {
                source: MapError::MapPointNotFound(_),
            } => LoopApplyErrorKind::MissingMapPoint,
            Self::Map { .. } => LoopApplyErrorKind::MapMutation,
            Self::EssentialGraph { .. } => LoopApplyErrorKind::EssentialGraph,
            Self::PoseGraph { .. } => LoopApplyErrorKind::PoseGraph,
            Self::PoseConversion { .. } => LoopApplyErrorKind::PoseConversion,
            Self::MapFrameAlignment { .. } => LoopApplyErrorKind::MapFrameAlignment,
        }
    }
}

impl std::error::Error for LoopApplyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Map { source } => Some(source),
            Self::EssentialGraph { source } => Some(source),
            Self::PoseGraph { source } => Some(source),
            Self::PoseConversion { source, .. } => Some(source),
            Self::MapFrameAlignment { source } => Some(source),
            Self::StaleCorrection { .. } => None,
        }
    }
}

impl std::fmt::Display for LoopApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StaleCorrection { proof, current } => write!(
                f,
                "loop proof belongs to map instance {} generation {}, but current map is instance {} generation {}",
                proof.instance_id().as_u64(),
                proof.generation().as_u64(),
                current.instance_id().as_u64(),
                current.generation().as_u64(),
            ),
            Self::Map { source } => write!(f, "loop-closure map operation failed: {source}"),
            Self::EssentialGraph { source } => {
                write!(f, "loop-closure essential-graph mutation failed: {source}")
            }
            Self::PoseGraph { source } => {
                write!(f, "loop-closure pose-graph optimization failed: {source}")
            }
            Self::PoseConversion {
                operation,
                keyframe_id,
                source,
            } => {
                if let Some(keyframe_id) = keyframe_id {
                    write!(
                        f,
                        "loop-closure {operation} failed for keyframe {keyframe_id:?}: {source}"
                    )
                } else {
                    write!(f, "loop-closure {operation} failed: {source}")
                }
            }
            Self::MapFrameAlignment { source } => {
                write!(f, "loop-closure map/odometry alignment failed: {source}")
            }
        }
    }
}

impl From<MapError> for LoopApplyError {
    fn from(source: MapError) -> Self {
        Self::Map { source }
    }
}

impl From<PoseGraphError> for LoopApplyError {
    fn from(source: PoseGraphError) -> Self {
        Self::PoseGraph { source }
    }
}

#[derive(Debug)]
pub enum LoopDetectError {
    TooFewCorrespondences {
        count: usize,
    },
    VerificationFailed(LoopVerificationError),
    CorrectionEvaluation {
        source: crate::Pose64Error,
    },
    CorrectionTooLarge {
        translation_m: f64,
        rotation_deg: f64,
    },
    ApplyFailed(LoopApplyError),
}

impl LoopDetectError {
    pub fn is_candidate_rejection(&self) -> bool {
        match self {
            Self::TooFewCorrespondences { .. } | Self::CorrectionTooLarge { .. } => true,
            Self::VerificationFailed(source) => source.is_candidate_rejection(),
            Self::CorrectionEvaluation { .. } | Self::ApplyFailed(_) => false,
        }
    }
}

impl std::fmt::Display for LoopDetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopDetectError::TooFewCorrespondences { count } => {
                write!(
                    f,
                    "loop closure rejected: too few correspondences ({count})"
                )
            }
            LoopDetectError::VerificationFailed(err) => {
                write!(f, "loop closure verification failed: {err}")
            }
            LoopDetectError::CorrectionEvaluation { source } => {
                write!(f, "loop closure correction evaluation failed: {source}")
            }
            LoopDetectError::CorrectionTooLarge {
                translation_m,
                rotation_deg,
            } => write!(
                f,
                "loop closure rejected: correction too large (translation={translation_m:.3}m, rotation={rotation_deg:.2}deg)"
            ),
            LoopDetectError::ApplyFailed(err) => {
                write!(f, "loop closure apply failed: {err}")
            }
        }
    }
}

impl std::error::Error for LoopDetectError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoopDetectError::VerificationFailed(source) => Some(source),
            LoopDetectError::CorrectionEvaluation { source } => Some(source),
            LoopDetectError::ApplyFailed(source) => Some(source),
            LoopDetectError::TooFewCorrespondences { .. }
            | LoopDetectError::CorrectionTooLarge { .. } => None,
        }
    }
}

#[derive(Debug)]
pub enum GlobalDescriptorError {
    EmptyInput,
    NonFiniteValue { index: usize, value: f32 },
    ZeroNorm,
}

impl std::fmt::Display for GlobalDescriptorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GlobalDescriptorError::EmptyInput => {
                write!(
                    f,
                    "global descriptor requires at least one local descriptor"
                )
            }
            GlobalDescriptorError::NonFiniteValue { index, value } => write!(
                f,
                "global descriptor contains non-finite value at index {index}: {value}"
            ),
            GlobalDescriptorError::ZeroNorm => write!(f, "global descriptor norm must be > 0"),
        }
    }
}

impl std::error::Error for GlobalDescriptorError {}

#[derive(Clone, Debug, PartialEq)]
pub struct GlobalDescriptor {
    values: [f32; GLOBAL_DESCRIPTOR_DIM],
    inverse_norm: f64,
}

impl GlobalDescriptor {
    pub fn try_new(values: [f32; GLOBAL_DESCRIPTOR_DIM]) -> Result<Self, GlobalDescriptorError> {
        let mut norm_sq = 0.0_f64;
        for (idx, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(GlobalDescriptorError::NonFiniteValue { index: idx, value });
            }
            let value = f64::from(value);
            norm_sq += value * value;
        }
        if norm_sq <= 0.0 {
            return Err(GlobalDescriptorError::ZeroNorm);
        }

        let inv_norm = 1.0 / norm_sq.sqrt();
        let mut normalized = values;
        for v in &mut normalized {
            *v = (f64::from(*v) * inv_norm) as f32;
        }
        // A nonzero input has a normalized component of at least 1/sqrt(DIM),
        // so conversion to f32 cannot turn every component into zero.
        let normalized_norm_sq = normalized.iter().fold(0.0_f64, |sum, &value| {
            let value = f64::from(value);
            sum + value * value
        });
        Ok(Self {
            values: normalized,
            inverse_norm: normalized_norm_sq.sqrt().recip(),
        })
    }

    pub fn as_array(&self) -> &[f32; GLOBAL_DESCRIPTOR_DIM] {
        &self.values
    }

    pub fn clamped_cosine_similarity(&self, other: &Self) -> CosineSimilarity {
        let dot = self
            .values
            .iter()
            .zip(other.values.iter())
            .fold(0.0_f64, |sum, (&a, &b)| sum + f64::from(a) * f64::from(b));
        CosineSimilarity::from_computed_clamped(dot * self.inverse_norm * other.inverse_norm)
    }

    pub fn from_local_descriptors(
        descriptors: &[Descriptor],
    ) -> Result<Self, GlobalDescriptorError> {
        if descriptors.is_empty() {
            return Err(GlobalDescriptorError::EmptyInput);
        }

        let mut mean = [0.0_f64; crate::DESCRIPTOR_DIM];
        let mut max = [f32::NEG_INFINITY; crate::DESCRIPTOR_DIM];
        for d in descriptors {
            for (idx, value) in d.as_slice().iter().copied().enumerate() {
                mean[idx] += f64::from(value);
                if value > max[idx] {
                    max[idx] = value;
                }
            }
        }
        let count = descriptors.len() as f64;
        let mut out = [0.0_f32; GLOBAL_DESCRIPTOR_DIM];
        for idx in 0..crate::DESCRIPTOR_DIM {
            out[idx] = (mean[idx] / count) as f32;
            out[crate::DESCRIPTOR_DIM + idx] = max[idx];
        }

        Self::try_new(out)
    }
}

pub fn aggregate_global_descriptor(
    descriptors: &[Descriptor],
) -> Result<GlobalDescriptor, GlobalDescriptorError> {
    GlobalDescriptor::from_local_descriptors(descriptors)
}

pub fn match_descriptors_for_loop(
    query_descriptors: &[Descriptor],
    candidate_kf: KeyframeId,
    map: &SlamMap,
    similarity_threshold: f32,
) -> Result<LoopDescriptorMatchResult, DescriptorMatchError> {
    let query_quantized = quantize_loop_descriptors(query_descriptors);
    match_quantized_descriptors_for_loop(&query_quantized, candidate_kf, map, similarity_threshold)
}

pub(crate) fn quantize_loop_descriptors(descriptors: &[Descriptor]) -> Vec<CompactDescriptor> {
    descriptors.iter().map(Descriptor::quantize).collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopDescriptorMatchResult {
    correspondences: Vec<(usize, usize)>,
    zero_norm_query_descriptors: usize,
    zero_norm_candidate_descriptors: usize,
}

impl LoopDescriptorMatchResult {
    pub fn correspondences(&self) -> &[(usize, usize)] {
        &self.correspondences
    }

    pub fn into_correspondences(self) -> Vec<(usize, usize)> {
        self.correspondences
    }

    pub fn zero_norm_query_descriptors(&self) -> usize {
        self.zero_norm_query_descriptors
    }

    pub fn zero_norm_candidate_descriptors(&self) -> usize {
        self.zero_norm_candidate_descriptors
    }
}

#[derive(Debug)]
pub enum DescriptorMatchError {
    InvalidSimilarityThreshold { value: f32 },
    Map(MapError),
}

impl std::fmt::Display for DescriptorMatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidSimilarityThreshold { value } => write!(
                f,
                "descriptor similarity threshold must be finite and in (0, 1], got {value}"
            ),
            Self::Map(source) => write!(f, "descriptor matching map lookup failed: {source}"),
        }
    }
}

impl std::error::Error for DescriptorMatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Map(source) => Some(source),
            Self::InvalidSimilarityThreshold { .. } => None,
        }
    }
}

impl From<MapError> for DescriptorMatchError {
    fn from(source: MapError) -> Self {
        Self::Map(source)
    }
}

pub(crate) fn match_quantized_descriptors_for_loop(
    query_quantized: &[CompactDescriptor],
    candidate_kf: KeyframeId,
    map: &SlamMap,
    similarity_threshold: f32,
) -> Result<LoopDescriptorMatchResult, DescriptorMatchError> {
    if !similarity_threshold.is_finite()
        || similarity_threshold <= 0.0
        || similarity_threshold > 1.0
    {
        return Err(DescriptorMatchError::InvalidSimilarityThreshold {
            value: similarity_threshold,
        });
    }
    let zero_norm_query_descriptors = query_quantized
        .iter()
        .filter(|descriptor| descriptor.has_zero_norm())
        .count();

    let mut candidate_count = 0usize;
    let mut zero_norm_candidate_descriptors = 0usize;
    map.for_each_keyframe_point_descriptor(candidate_kf, |_, descriptor| {
        candidate_count += 1;
        if descriptor.has_zero_norm() {
            zero_norm_candidate_descriptors += 1;
        }
    })?;
    if query_quantized.is_empty() || candidate_count == 0 {
        return Ok(LoopDescriptorMatchResult {
            correspondences: Vec::new(),
            zero_norm_query_descriptors,
            zero_norm_candidate_descriptors,
        });
    }

    let mut query_best: Vec<Option<(usize, f32)>> = vec![None; query_quantized.len()];
    map.for_each_keyframe_point_descriptor(candidate_kf, |keypoint, descriptor| {
        for (query_idx, query_descriptor) in query_quantized.iter().enumerate() {
            let Some(similarity) = query_descriptor.clamped_cosine_similarity(descriptor) else {
                continue;
            };
            let similarity = similarity.value();
            if similarity < similarity_threshold {
                continue;
            }
            match &mut query_best[query_idx] {
                Some((_, best_similarity)) if *best_similarity >= similarity => {}
                slot => {
                    *slot = Some((keypoint.index(), similarity));
                }
            }
        }
    })?;

    let mut correspondences = Vec::new();
    map.for_each_keyframe_point_descriptor(candidate_kf, |keypoint, descriptor| {
        let mut best_query: Option<(usize, f32)> = None;
        for (query_idx, query_descriptor) in query_quantized.iter().enumerate() {
            let Some(similarity) = query_descriptor.clamped_cosine_similarity(descriptor) else {
                continue;
            };
            let similarity = similarity.value();
            if similarity < similarity_threshold {
                continue;
            }
            match best_query {
                Some((_, best_similarity)) if best_similarity >= similarity => {}
                _ => best_query = Some((query_idx, similarity)),
            }
        }
        let Some((query_idx, _)) = best_query else {
            return;
        };
        let Some((best_candidate_index, _)) = query_best[query_idx] else {
            return;
        };
        if best_candidate_index == keypoint.index() {
            correspondences.push((query_idx, keypoint.index()));
        }
    })?;
    Ok(LoopDescriptorMatchResult {
        correspondences,
        zero_norm_query_descriptors,
        zero_norm_candidate_descriptors,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct PlaceMatch {
    query: KeyframeId,
    candidate: KeyframeId,
    cosine_similarity: CosineSimilarity,
}

impl PlaceMatch {
    pub fn query(self) -> KeyframeId {
        self.query
    }

    pub fn candidate(self) -> KeyframeId {
        self.candidate
    }

    pub fn cosine_similarity(self) -> CosineSimilarity {
        self.cosine_similarity
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RelocalizationMatch {
    candidate: KeyframeId,
    cosine_similarity: CosineSimilarity,
}

impl RelocalizationMatch {
    pub fn candidate(self) -> KeyframeId {
        self.candidate
    }

    pub fn cosine_similarity(self) -> CosineSimilarity {
        self.cosine_similarity
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorSource {
    Bootstrap,
    Learned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyframeDescriptorUpdate {
    Initialized,
    Replaced,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyframeDatabaseError {
    KeyframeAlreadyRegistered { keyframe_id: KeyframeId },
    KeyframeNotRegistered { keyframe_id: KeyframeId },
    SequenceExhausted { next_sequence: u64 },
}

impl std::fmt::Display for KeyframeDatabaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeyframeAlreadyRegistered { keyframe_id } => write!(
                f,
                "keyframe {keyframe_id:?} is already registered in the descriptor database"
            ),
            Self::KeyframeNotRegistered { keyframe_id } => {
                write!(
                    f,
                    "keyframe {keyframe_id:?} is not registered in the descriptor database"
                )
            }
            Self::SequenceExhausted { next_sequence } => write!(
                f,
                "keyframe descriptor sequence exhausted at {next_sequence}; refusing to reuse an ordering value"
            ),
        }
    }
}

impl std::error::Error for KeyframeDatabaseError {}

#[derive(Clone, Debug)]
struct StoredKeyframeDescriptor {
    value: GlobalDescriptor,
    source: DescriptorSource,
}

#[derive(Clone, Debug)]
struct KeyframeDescriptorEntry {
    keyframe_id: KeyframeId,
    descriptor: Option<StoredKeyframeDescriptor>,
    sequence: u64,
}

struct RankedMatch<T> {
    value: T,
    candidate_sequence: u64,
    candidate: KeyframeId,
    similarity: CosineSimilarity,
}

const DEFAULT_TEMPORAL_GAP: usize = 30;

#[derive(Clone, Debug)]
pub struct KeyframeDatabase {
    entries: Vec<KeyframeDescriptorEntry>,
    registered_keyframe_gap: u64,
    next_sequence: Option<u64>,
}

impl KeyframeDatabase {
    pub fn new(temporal_gap: usize) -> Self {
        Self {
            entries: Vec::new(),
            registered_keyframe_gap: u64::try_from(temporal_gap)
                .expect("usize fits in u64 on supported macOS and Linux hosts"),
            next_sequence: Some(0),
        }
    }

    /// Number of immediately preceding registered keyframes excluded from
    /// loop matching. The retained API name is historical; this is not time.
    pub fn temporal_gap(&self) -> usize {
        usize::try_from(self.registered_keyframe_gap)
            .expect("registered-keyframe gap originated from usize on this host")
    }

    pub fn registered_len(&self) -> usize {
        self.entries.len()
    }

    pub fn descriptor_len(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.descriptor.is_some())
            .count()
    }

    pub fn register_keyframe(&mut self, id: KeyframeId) -> Result<(), KeyframeDatabaseError> {
        if self.entries.iter().any(|entry| entry.keyframe_id == id) {
            return Err(KeyframeDatabaseError::KeyframeAlreadyRegistered { keyframe_id: id });
        }
        let sequence = self
            .next_sequence
            .ok_or(KeyframeDatabaseError::SequenceExhausted {
                next_sequence: u64::MAX,
            })?;
        let next_sequence = sequence.checked_add(1);
        self.entries.push(KeyframeDescriptorEntry {
            keyframe_id: id,
            descriptor: None,
            sequence,
        });
        self.next_sequence = next_sequence;
        Ok(())
    }

    pub fn set_descriptor(
        &mut self,
        id: KeyframeId,
        descriptor: GlobalDescriptor,
        source: DescriptorSource,
    ) -> Result<KeyframeDescriptorUpdate, KeyframeDatabaseError> {
        let entry = self
            .entries
            .iter_mut()
            .find(|entry| entry.keyframe_id == id)
            .ok_or(KeyframeDatabaseError::KeyframeNotRegistered { keyframe_id: id })?;
        let update = if entry.descriptor.is_some() {
            KeyframeDescriptorUpdate::Replaced
        } else {
            KeyframeDescriptorUpdate::Initialized
        };
        entry.descriptor = Some(StoredKeyframeDescriptor {
            value: descriptor,
            source,
        });
        Ok(update)
    }

    pub fn remove(&mut self, id: KeyframeId) -> Result<(), KeyframeDatabaseError> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.keyframe_id == id)
            .ok_or(KeyframeDatabaseError::KeyframeNotRegistered { keyframe_id: id })?;
        self.entries.remove(index);
        Ok(())
    }

    pub fn descriptor_source(&self, id: KeyframeId) -> Option<DescriptorSource> {
        self.entries
            .iter()
            .find(|entry| entry.keyframe_id == id)
            .and_then(|entry| entry.descriptor.as_ref())
            .map(|descriptor| descriptor.source)
    }

    pub fn query_loop_candidates(
        &self,
        query_id: KeyframeId,
        descriptor: &GlobalDescriptor,
        top_k: usize,
    ) -> Result<Vec<PlaceMatch>, KeyframeDatabaseError> {
        let query_entry = self
            .entries
            .iter()
            .find(|entry| entry.keyframe_id == query_id)
            .ok_or(KeyframeDatabaseError::KeyframeNotRegistered {
                keyframe_id: query_id,
            })?;
        if top_k == 0 {
            return Ok(Vec::new());
        }
        let query_id = query_entry.keyframe_id;
        let query_sequence = query_entry.sequence;
        let mut matches = Vec::new();
        for candidate in &self.entries {
            let Some(candidate_descriptor) = candidate.descriptor.as_ref() else {
                continue;
            };
            let Some(age) = query_sequence.checked_sub(candidate.sequence) else {
                continue;
            };
            if age == 0 || age <= self.registered_keyframe_gap {
                continue;
            }
            let similarity = descriptor.clamped_cosine_similarity(&candidate_descriptor.value);
            matches.push(RankedMatch {
                value: PlaceMatch {
                    query: query_id,
                    candidate: candidate.keyframe_id,
                    cosine_similarity: similarity,
                },
                candidate_sequence: candidate.sequence,
                candidate: candidate.keyframe_id,
                similarity,
            });
        }
        matches.sort_by(compare_ranked_matches);
        matches.truncate(top_k);
        Ok(matches.into_iter().map(|ranked| ranked.value).collect())
    }

    pub fn query_for_relocalization(
        &self,
        descriptor: &GlobalDescriptor,
        top_k: usize,
    ) -> Vec<RelocalizationMatch> {
        if top_k == 0 {
            return Vec::new();
        }
        let mut matches: Vec<RankedMatch<RelocalizationMatch>> = self
            .entries
            .iter()
            .filter_map(|entry| {
                entry.descriptor.as_ref().map(|candidate_descriptor| {
                    let similarity =
                        descriptor.clamped_cosine_similarity(&candidate_descriptor.value);
                    RankedMatch {
                        value: RelocalizationMatch {
                            candidate: entry.keyframe_id,
                            cosine_similarity: similarity,
                        },
                        candidate_sequence: entry.sequence,
                        candidate: entry.keyframe_id,
                        similarity,
                    }
                })
            })
            .collect();
        matches.sort_by(compare_ranked_matches);
        matches.truncate(top_k);
        matches.into_iter().map(|ranked| ranked.value).collect()
    }
}

fn compare_ranked_matches<T>(left: &RankedMatch<T>, right: &RankedMatch<T>) -> Ordering {
    right
        .similarity
        .value()
        .total_cmp(&left.similarity.value())
        .then_with(|| left.candidate_sequence.cmp(&right.candidate_sequence))
        .then_with(|| left.candidate.cmp(&right.candidate))
}

impl Default for KeyframeDatabase {
    fn default() -> Self {
        Self::new(DEFAULT_TEMPORAL_GAP)
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedLoop {
    map_snapshot: MapSnapshot,
    query_kf: KeyframeId,
    match_kf: KeyframeId,
    query_pose_world: Pose,
    inlier_count: usize,
}

impl VerifiedLoop {
    pub fn map_snapshot(&self) -> MapSnapshot {
        self.map_snapshot
    }

    pub fn query_kf(&self) -> KeyframeId {
        self.query_kf
    }

    pub fn match_kf(&self) -> KeyframeId {
        self.match_kf
    }

    pub fn query_pose_world(&self) -> Pose {
        self.query_pose_world
    }

    pub fn inlier_count(&self) -> usize {
        self.inlier_count
    }

    #[cfg(test)]
    pub(crate) fn from_parts(
        map: &SlamMap,
        query_kf: KeyframeId,
        match_kf: KeyframeId,
        query_pose_world: Pose,
        inlier_count: usize,
    ) -> Self {
        Self {
            map_snapshot: map.snapshot(),
            query_kf,
            match_kf,
            query_pose_world,
            inlier_count,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VerifiedRelocalization {
    match_kf: KeyframeId,
    pose_world: Pose,
    inlier_count: usize,
}

impl VerifiedRelocalization {
    pub fn match_kf(&self) -> KeyframeId {
        self.match_kf
    }

    pub fn pose_world(&self) -> Pose {
        self.pose_world
    }

    pub fn inlier_count(&self) -> usize {
        self.inlier_count
    }
}

#[derive(Debug)]
pub enum LoopVerificationError {
    TooFewMatches { count: usize },
    QueryIndexOutOfBounds { index: usize, len: usize },
    DescriptorMatch(DescriptorMatchError),
    Map(MapError),
    InvalidRansacConfig { source: RansacConfigError },
    PnpFailed(PnpError),
    InsufficientInliers { inliers: usize, required: usize },
}

impl LoopVerificationError {
    pub fn is_candidate_rejection(&self) -> bool {
        match self {
            Self::TooFewMatches { .. } | Self::InsufficientInliers { .. } => true,
            Self::PnpFailed(error) => error.rejection().is_some(),
            Self::QueryIndexOutOfBounds { .. }
            | Self::DescriptorMatch(_)
            | Self::Map(_)
            | Self::InvalidRansacConfig { .. } => false,
        }
    }
}

impl std::fmt::Display for LoopVerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopVerificationError::TooFewMatches { count } => {
                write!(f, "loop verification needs at least 4 matches, got {count}")
            }
            LoopVerificationError::QueryIndexOutOfBounds { index, len } => {
                write!(
                    f,
                    "loop query keypoint index {index} is out of bounds for {len} keypoints"
                )
            }
            LoopVerificationError::Map(source) => {
                write!(f, "loop verification map lookup failed: {source}")
            }
            LoopVerificationError::DescriptorMatch(source) => {
                write!(f, "loop descriptor matching failed: {source}")
            }
            LoopVerificationError::PnpFailed(err) => {
                write!(f, "loop verification PnP failed: {err}")
            }
            LoopVerificationError::InvalidRansacConfig { source } => {
                write!(f, "loop verification RANSAC config failed: {source}")
            }
            LoopVerificationError::InsufficientInliers { inliers, required } => write!(
                f,
                "loop verification inliers below threshold: inliers={inliers}, required={required}"
            ),
        }
    }
}

impl std::error::Error for LoopVerificationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoopVerificationError::DescriptorMatch(source) => Some(source),
            LoopVerificationError::Map(source) => Some(source),
            LoopVerificationError::InvalidRansacConfig { source } => Some(source),
            LoopVerificationError::PnpFailed(source) => Some(source),
            LoopVerificationError::TooFewMatches { .. }
            | LoopVerificationError::QueryIndexOutOfBounds { .. }
            | LoopVerificationError::InsufficientInliers { .. } => None,
        }
    }
}

fn verify_pose_from_keyframe(
    query_keypoints: &[Keypoint],
    correspondences: &[(usize, usize)],
    map: &SlamMap,
    match_kf: KeyframeId,
    intrinsics: PinholeIntrinsics,
    ransac_config: RansacConfig,
    min_inliers: usize,
) -> Result<(Pose, usize), LoopVerificationError> {
    let required_inliers = min_inliers.max(MIN_PNP_POINTS);

    if correspondences.len() < MIN_PNP_POINTS {
        return Err(LoopVerificationError::TooFewMatches {
            count: correspondences.len(),
        });
    }

    let mut observations = Vec::with_capacity(correspondences.len());
    for &(query_idx, match_idx) in correspondences {
        let pixel = *query_keypoints.get(query_idx).ok_or(
            LoopVerificationError::QueryIndexOutOfBounds {
                index: query_idx,
                len: query_keypoints.len(),
            },
        )?;
        let match_ref = map
            .keyframe_keypoint(match_kf, match_idx)
            .map_err(LoopVerificationError::Map)?;
        let Some(point_id) = map
            .map_point_for_keypoint(match_ref)
            .map_err(LoopVerificationError::Map)?
        else {
            continue;
        };
        let point =
            map.point(point_id)
                .ok_or(LoopVerificationError::Map(MapError::MapPointNotFound(
                    point_id,
                )))?;
        let obs = Observation::try_new(point.position(), pixel, intrinsics)
            .map_err(LoopVerificationError::PnpFailed)?;
        observations.push(obs);
    }

    if observations.len() < MIN_PNP_POINTS {
        return Err(LoopVerificationError::TooFewMatches {
            count: observations.len(),
        });
    }

    // Use a relaxed inlier threshold for the RANSAC solver itself so it can
    // find a valid pose even with some outlier correspondences.  The actual
    // quality gate is the post-PnP inlier check below.
    let pnp_config = ransac_config
        .try_with_min_inliers(MIN_PNP_POINTS)
        .map_err(|source| LoopVerificationError::InvalidRansacConfig { source })?;

    let result = solve_pnp_ransac(&observations, intrinsics, pnp_config)
        .map_err(LoopVerificationError::PnpFailed)?;
    if result.inliers().len() < required_inliers {
        return Err(LoopVerificationError::InsufficientInliers {
            inliers: result.inliers().len(),
            required: required_inliers,
        });
    }
    Ok((result.pose(), result.inliers().len()))
}

impl PlaceMatch {
    pub fn verify(
        &self,
        query_keypoints: &[Keypoint],
        correspondences: &[(usize, usize)],
        map: &SlamMap,
        intrinsics: PinholeIntrinsics,
        ransac_config: RansacConfig,
        min_inliers: usize,
    ) -> Result<VerifiedLoop, LoopVerificationError> {
        map.keyframe(self.query)
            .ok_or(LoopVerificationError::Map(MapError::KeyframeNotFound(
                self.query,
            )))?;
        map.keyframe(self.candidate)
            .ok_or(LoopVerificationError::Map(MapError::KeyframeNotFound(
                self.candidate,
            )))?;
        let (query_pose_world, inlier_count) = verify_pose_from_keyframe(
            query_keypoints,
            correspondences,
            map,
            self.candidate,
            intrinsics,
            ransac_config,
            min_inliers,
        )?;
        Ok(VerifiedLoop {
            map_snapshot: map.snapshot(),
            query_kf: self.query,
            match_kf: self.candidate,
            query_pose_world,
            inlier_count,
        })
    }
}

impl RelocalizationMatch {
    pub fn verify(
        &self,
        query_keypoints: &[Keypoint],
        correspondences: &[(usize, usize)],
        map: &SlamMap,
        intrinsics: PinholeIntrinsics,
        ransac_config: RansacConfig,
        min_inliers: usize,
    ) -> Result<VerifiedRelocalization, LoopVerificationError> {
        let (pose_world, inlier_count) = verify_pose_from_keyframe(
            query_keypoints,
            correspondences,
            map,
            self.candidate,
            intrinsics,
            ransac_config,
            min_inliers,
        )?;
        Ok(VerifiedRelocalization {
            match_kf: self.candidate,
            pose_world,
            inlier_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DescriptorMatchError, DescriptorSource, GlobalDescriptor, GlobalDescriptorError,
        KeyframeDatabase, KeyframeDatabaseError, KeyframeDescriptorUpdate, LoopApplyError,
        LoopApplyErrorKind, LoopClosureConfig, LoopClosureConfigInput, LoopVerificationError,
        PlaceMatch, RelocalizationConfig, RelocalizationConfigError, RelocalizationConfigInput,
        RelocalizationMatch, aggregate_global_descriptor, match_descriptors_for_loop,
    };
    use crate::map::{KeyframeId, MapError, SlamMap};
    use crate::pose_graph::PoseGraphError;
    use crate::test_helpers::{make_pinhole_intrinsics, project_world_point};
    use crate::{
        CompactDescriptor, CosineSimilarity, CosineSimilarityError, Descriptor, FrameDimensions,
        FrameId, Keypoint, Point3, Pose, RansacConfig, Timestamp,
    };
    use std::error::Error as _;

    type LoopFixture = (
        SlamMap,
        KeyframeId,
        KeyframeId,
        Vec<Keypoint>,
        Vec<(usize, usize)>,
        crate::PinholeIntrinsics,
        Pose,
    );

    fn descriptor(values: [f32; crate::DESCRIPTOR_DIM]) -> Descriptor {
        Descriptor::try_new(values).expect("valid test descriptor")
    }

    fn descriptor_with_basis(idx: usize) -> GlobalDescriptor {
        let mut d = [0.0_f32; 512];
        d[idx] = 1.0;
        GlobalDescriptor::try_new(d).expect("valid basis descriptor")
    }

    fn place_match(query: KeyframeId, candidate: KeyframeId) -> PlaceMatch {
        PlaceMatch {
            query,
            candidate,
            cosine_similarity: CosineSimilarity::try_new(0.9).expect("valid similarity"),
        }
    }

    fn relocalization_match(candidate: KeyframeId) -> RelocalizationMatch {
        RelocalizationMatch {
            candidate,
            cosine_similarity: CosineSimilarity::try_new(0.9).expect("valid similarity"),
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct BidirectionalMatchBehavior {
        query_best: Vec<Option<(usize, u32)>>,
        candidate_best: Vec<Option<(usize, u32)>>,
        mutual_matches: Vec<(usize, usize, u32)>,
    }

    fn generated_compact_descriptor(mut state: u64) -> CompactDescriptor {
        let mut values = [0_u8; crate::DESCRIPTOR_DIM];
        for value in &mut values {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *value = state.to_be_bytes()[0];
        }
        CompactDescriptor(values)
    }

    fn reference_bidirectional_behavior(
        query: &[CompactDescriptor],
        candidates: &[CompactDescriptor],
        threshold: f32,
    ) -> BidirectionalMatchBehavior {
        fn best_matches(
            source_len: usize,
            target_len: usize,
            threshold: f32,
            similarity: impl Fn(usize, usize) -> Option<f32>,
        ) -> Vec<Option<(usize, f32)>> {
            (0..source_len)
                .map(|source_index| {
                    let mut best = None;
                    for target_index in 0..target_len {
                        let Some(score) = similarity(source_index, target_index) else {
                            continue;
                        };
                        if score < threshold
                            || best.is_some_and(|(_, best_score)| best_score >= score)
                        {
                            continue;
                        }
                        best = Some((target_index, score));
                    }
                    best
                })
                .collect()
        }

        let score = |query_index: usize, candidate_index: usize| {
            query[query_index]
                .clamped_cosine_similarity(&candidates[candidate_index])
                .map(CosineSimilarity::value)
        };
        let query_best = best_matches(query.len(), candidates.len(), threshold, score);
        let candidate_best = best_matches(candidates.len(), query.len(), threshold, |ci, qi| {
            score(qi, ci)
        });
        let mutual_matches = candidate_best
            .iter()
            .enumerate()
            .filter_map(|(candidate_index, best)| {
                let &(query_index, score) = best.as_ref()?;
                (query_best[query_index].map(|(index, _)| index) == Some(candidate_index))
                    .then_some((query_index, candidate_index, score.to_bits()))
            })
            .collect();
        BidirectionalMatchBehavior {
            query_best: query_best
                .into_iter()
                .map(|best| best.map(|(index, score)| (index, score.to_bits())))
                .collect(),
            candidate_best: candidate_best
                .into_iter()
                .map(|best| best.map(|(index, score)| (index, score.to_bits())))
                .collect(),
            mutual_matches,
        }
    }

    #[test]
    fn local_match_threshold_is_inclusive_and_first_tied_indices_win() {
        let descriptor = generated_compact_descriptor(7);
        let query = [descriptor.clone(), descriptor.clone()];
        let candidates = [descriptor.clone(), descriptor.clone(), descriptor];
        let behavior = reference_bidirectional_behavior(&query, &candidates, 1.0);
        let identical = 1.0_f32.to_bits();
        assert_eq!(
            behavior,
            BidirectionalMatchBehavior {
                query_best: vec![Some((0, identical)), Some((0, identical))],
                candidate_best: vec![Some((0, identical)); 3],
                mutual_matches: vec![(0, 0, identical)],
            }
        );
    }

    #[test]
    fn local_match_oracle_covers_zero_norm_extremes_and_small_matrices() {
        let mut corpus = vec![
            CompactDescriptor([128; crate::DESCRIPTOR_DIM]),
            CompactDescriptor([0; crate::DESCRIPTOR_DIM]),
            CompactDescriptor([255; crate::DESCRIPTOR_DIM]),
        ];
        corpus.extend((1..=12).map(generated_compact_descriptor));
        assert!(corpus[0].clamped_cosine_similarity(&corpus[1]).is_none());
        assert_eq!(
            corpus[1]
                .clamped_cosine_similarity(&corpus[1])
                .expect("nonzero extreme")
                .value()
                .to_bits(),
            1.0_f32.to_bits()
        );

        for query_len in 0..=4 {
            for candidate_len in 0..=4 {
                let query = &corpus[..query_len];
                let candidates = &corpus[3..3 + candidate_len];
                for threshold in [f32::from_bits(1), 0.35, 1.0] {
                    let behavior = reference_bidirectional_behavior(query, candidates, threshold);
                    assert_eq!(behavior.query_best.len(), query_len);
                    assert_eq!(behavior.candidate_best.len(), candidate_len);
                    assert!(
                        behavior
                            .mutual_matches
                            .iter()
                            .all(|(qi, ci, _)| { *qi < query_len && *ci < candidate_len })
                    );
                }
            }
        }
    }

    fn insert_bootstrap(
        database: &mut KeyframeDatabase,
        keyframe_id: KeyframeId,
        descriptor: GlobalDescriptor,
    ) -> KeyframeDescriptorUpdate {
        register_descriptor(
            database,
            keyframe_id,
            descriptor,
            DescriptorSource::Bootstrap,
        )
    }

    fn register_descriptor(
        database: &mut KeyframeDatabase,
        keyframe_id: KeyframeId,
        descriptor: GlobalDescriptor,
        source: DescriptorSource,
    ) -> KeyframeDescriptorUpdate {
        database
            .register_keyframe(keyframe_id)
            .expect("descriptor sequence available");
        database
            .set_descriptor(keyframe_id, descriptor, source)
            .expect("registered keyframe")
    }

    fn make_keyframe_ids(n: usize) -> Vec<KeyframeId> {
        let mut map = SlamMap::new();
        let size = FrameDimensions::try_new(320, 240).expect("size");
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            let id = map
                .add_keyframe(
                    FrameId::new((i + 1) as u64),
                    Timestamp::from_nanos(i as i64 + 1),
                    Pose::identity(),
                    size,
                    vec![Keypoint { x: 10.0, y: 10.0 }],
                )
                .expect("keyframe");
            ids.push(id);
        }
        ids
    }

    fn make_loop_fixture() -> LoopFixture {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let match_pose = Pose::identity();
        let query_pose = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.18, -0.04, 0.06],
        );

        let world_points = vec![
            Point3 {
                x: -0.3,
                y: -0.2,
                z: 3.2,
            },
            Point3 {
                x: -0.1,
                y: -0.2,
                z: 3.5,
            },
            Point3 {
                x: 0.1,
                y: -0.1,
                z: 3.8,
            },
            Point3 {
                x: 0.3,
                y: -0.1,
                z: 3.4,
            },
            Point3 {
                x: -0.2,
                y: 0.1,
                z: 3.6,
            },
            Point3 {
                x: 0.2,
                y: 0.2,
                z: 3.9,
            },
        ];

        let mut match_keypoints = Vec::new();
        let mut query_keypoints = Vec::new();
        for &point in &world_points {
            match_keypoints
                .push(project_world_point(match_pose, point, intrinsics).expect("match kp"));
            query_keypoints
                .push(project_world_point(query_pose, point, intrinsics).expect("query kp"));
        }

        let mut map = SlamMap::new();
        let size = FrameDimensions::try_new(640, 480).expect("size");
        let match_kf = map
            .add_keyframe(
                FrameId::new(10),
                Timestamp::from_nanos(10),
                match_pose,
                size,
                match_keypoints,
            )
            .expect("match kf");
        let query_kf = map
            .add_keyframe(
                FrameId::new(11),
                Timestamp::from_nanos(11),
                query_pose,
                size,
                query_keypoints.clone(),
            )
            .expect("query kf");

        for (idx, &point) in world_points.iter().enumerate() {
            let kp_ref = map.keyframe_keypoint(match_kf, idx).expect("kp ref");
            map.add_map_point(point, CompactDescriptor([128; 256]), kp_ref)
                .expect("map point");
        }

        let correspondences = (0..world_points.len()).map(|i| (i, i)).collect::<Vec<_>>();
        (
            map,
            query_kf,
            match_kf,
            query_keypoints,
            correspondences,
            intrinsics,
            query_pose,
        )
    }

    #[test]
    fn global_descriptor_identical_similarity_is_one() {
        let d = descriptor_with_basis(3);
        let sim = d.clamped_cosine_similarity(&d).value();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_parser_rejects_nonfinite_and_out_of_domain_values() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                CosineSimilarity::try_new(value),
                Err(CosineSimilarityError::NonFinite { .. })
            ));
        }
        for value in [1.0 + f32::EPSILON, -1.0 - f32::EPSILON] {
            assert_eq!(
                CosineSimilarity::try_new(value),
                Err(CosineSimilarityError::OutOfRange { value })
            );
        }
        for value in [-1.0, -0.0, 0.0, 1.0] {
            assert_eq!(
                CosineSimilarity::try_new(value)
                    .expect("bounded finite similarity")
                    .value(),
                value
            );
        }
    }

    #[test]
    fn computed_cosine_clamping_is_explicit_and_bounded() {
        assert_eq!(
            CosineSimilarity::from_computed_clamped(1.0 + f64::EPSILON).value(),
            1.0
        );
        assert_eq!(
            CosineSimilarity::from_computed_clamped(-1.0 - f64::EPSILON).value(),
            -1.0
        );
    }

    #[test]
    fn global_descriptor_similarity_is_bounded_and_symmetric_for_generated_inputs() {
        let mut state = 0x517c_c1b7_2722_0a95_u64;
        for _case in 0..128 {
            let mut a = [0.0_f32; super::GLOBAL_DESCRIPTOR_DIM];
            let mut b = [0.0_f32; super::GLOBAL_DESCRIPTOR_DIM];
            for (a_value, b_value) in a.iter_mut().zip(&mut b) {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                *a_value =
                    (f64::from((state >> 32) as u32) / f64::from(u32::MAX) * 2.0 - 1.0) as f32;
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                *b_value =
                    (f64::from((state >> 32) as u32) / f64::from(u32::MAX) * 2.0 - 1.0) as f32;
            }
            let a = GlobalDescriptor::try_new(a).expect("generated descriptor a");
            let b = GlobalDescriptor::try_new(b).expect("generated descriptor b");
            let ab = a.clamped_cosine_similarity(&b).value();
            let ba = b.clamped_cosine_similarity(&a).value();
            let (dot, norm_a, norm_b) = a.as_array().iter().zip(b.as_array()).fold(
                (0.0_f64, 0.0_f64, 0.0_f64),
                |(dot, norm_a, norm_b), (&a, &b)| {
                    let a = f64::from(a);
                    let b = f64::from(b);
                    (dot + a * b, norm_a + a * a, norm_b + b * b)
                },
            );
            let reference = (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32;
            assert!((-1.0..=1.0).contains(&ab));
            assert!((ab - ba).abs() <= f32::EPSILON);
            assert!((ab - reference).abs() <= f32::EPSILON);
        }
    }

    #[test]
    fn loop_detection_error_preserves_verification_and_pnp_sources() {
        let error = super::LoopDetectError::VerificationFailed(
            super::LoopVerificationError::PnpFailed(crate::PnpRejection::NoSolution.into()),
        );

        let verification = error.source().expect("verification source");
        let pnp = verification.source().expect("pnp source");
        assert_eq!(pnp.to_string(), "pnp rejected: no valid pose solution");
        assert_eq!(
            pnp.source().expect("typed rejection source").to_string(),
            "no valid pose solution"
        );
    }

    #[test]
    fn loop_candidate_rejections_are_distinct_from_operational_failures() {
        for rejection in [
            super::LoopDetectError::TooFewCorrespondences { count: 3 },
            super::LoopDetectError::CorrectionTooLarge {
                translation_m: 2.0,
                rotation_deg: 15.0,
            },
            super::LoopDetectError::VerificationFailed(super::LoopVerificationError::PnpFailed(
                crate::PnpRejection::NoSolution.into(),
            )),
            super::LoopDetectError::VerificationFailed(super::LoopVerificationError::PnpFailed(
                crate::PnpRejection::InsufficientObservationsForRequiredInliers {
                    required_inliers: 5,
                    observations: 4,
                }
                .into(),
            )),
            super::LoopDetectError::VerificationFailed(
                super::LoopVerificationError::InsufficientInliers {
                    inliers: 3,
                    required: 4,
                },
            ),
        ] {
            assert!(rejection.is_candidate_rejection(), "{rejection}");
        }

        for failure in [
            super::LoopDetectError::VerificationFailed(
                super::LoopVerificationError::QueryIndexOutOfBounds { index: 4, len: 4 },
            ),
            super::LoopDetectError::VerificationFailed(super::LoopVerificationError::PnpFailed(
                crate::PnpError::NonFiniteObservation {
                    field: "world.x",
                    value: f32::NAN,
                },
            )),
            super::LoopDetectError::VerificationFailed(super::LoopVerificationError::PnpFailed(
                crate::PnpError::RansacRejectionCountOverflow {
                    kind: crate::PnpRansacRejectionKind::CandidateProjection,
                },
            )),
            super::LoopDetectError::ApplyFailed(LoopApplyError::from(MapError::KeyframeNotFound(
                KeyframeId::default(),
            ))),
        ] {
            assert!(!failure.is_candidate_rejection(), "{failure}");
        }
    }

    #[test]
    fn keyframe_database_temporal_gap_filters_recent_frames() {
        let ids = make_keyframe_ids(5);
        let mut db = KeyframeDatabase::new(2);
        for (i, id) in ids.iter().enumerate() {
            insert_bootstrap(&mut db, *id, descriptor_with_basis(i));
        }

        let matches = db
            .query_loop_candidates(ids[4], &descriptor_with_basis(0), 10)
            .expect("registered query keyframe");
        // Query is the latest keyframe; with gap=2, only the first two are eligible.
        assert_eq!(matches.len(), 2);
        assert!(
            matches
                .iter()
                .all(|m| m.candidate() == ids[0] || m.candidate() == ids[1])
        );
    }

    #[test]
    fn keyframe_database_returns_top_k_by_similarity() {
        let ids = make_keyframe_ids(4);
        let mut db = KeyframeDatabase::new(0);

        let mut q = [0.0_f32; 512];
        q[0] = 1.0;
        q[1] = 1.0;
        let query = GlobalDescriptor::try_new(q).expect("valid query descriptor");

        insert_bootstrap(&mut db, ids[0], descriptor_with_basis(0)); // sim ~= 0.707
        insert_bootstrap(&mut db, ids[1], descriptor_with_basis(1)); // sim ~= 0.707
        insert_bootstrap(&mut db, ids[2], descriptor_with_basis(2)); // sim = 0
        insert_bootstrap(&mut db, ids[3], query.clone()); // query entry

        let matches = db
            .query_loop_candidates(ids[3], &query, 2)
            .expect("registered query keyframe");
        assert_eq!(matches.len(), 2);
        assert!(matches[0].cosine_similarity().value() >= matches[1].cosine_similarity().value());
        assert!(matches.iter().all(|m| m.candidate() != ids[2]));
    }

    #[test]
    fn keyframe_database_similarity_ties_follow_registration_sequence() {
        let ids = make_keyframe_ids(3);
        let mut db = KeyframeDatabase::new(0);
        insert_bootstrap(&mut db, ids[1], descriptor_with_basis(0));
        insert_bootstrap(&mut db, ids[0], descriptor_with_basis(0));
        insert_bootstrap(&mut db, ids[2], descriptor_with_basis(0));

        let matches = db
            .query_loop_candidates(ids[2], &descriptor_with_basis(0), 2)
            .expect("registered query keyframe");

        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].candidate(), ids[1]);
        assert_eq!(matches[1].candidate(), ids[0]);
    }

    #[test]
    fn keyframe_database_remove_deletes_entry() {
        let ids = make_keyframe_ids(3);
        let mut db = KeyframeDatabase::new(0);
        insert_bootstrap(&mut db, ids[0], descriptor_with_basis(0));
        insert_bootstrap(&mut db, ids[1], descriptor_with_basis(1));
        insert_bootstrap(&mut db, ids[2], descriptor_with_basis(2));
        assert_eq!(db.registered_len(), 3);
        assert_eq!(db.descriptor_len(), 3);
        db.remove(ids[1]).expect("registered keyframe");
        assert_eq!(db.registered_len(), 2);
        assert_eq!(db.descriptor_len(), 2);
        assert!(matches!(
            db.remove(ids[1]),
            Err(KeyframeDatabaseError::KeyframeNotRegistered { keyframe_id })
                if keyframe_id == ids[1]
        ));
    }

    #[test]
    fn keyframe_database_temporal_gap_uses_sequence_after_removal() {
        let ids = make_keyframe_ids(5);
        let mut db = KeyframeDatabase::new(2);
        for (i, id) in ids.iter().enumerate() {
            insert_bootstrap(&mut db, *id, descriptor_with_basis(i));
        }

        // Remove a middle entry; sequence distance must still be respected.
        db.remove(ids[2]).expect("registered keyframe");

        let matches = db
            .query_loop_candidates(ids[4], &descriptor_with_basis(4), 10)
            .expect("registered query keyframe");
        // kf3 is seq distance 1 (filtered), kf1 is distance 3 (kept), kf0 is distance 4 (kept).
        assert_eq!(matches.len(), 2);
        assert!(
            matches
                .iter()
                .all(|m| m.candidate() == ids[0] || m.candidate() == ids[1])
        );
    }

    #[test]
    fn keyframe_database_reports_registration_initialization_and_replacement() {
        let ids = make_keyframe_ids(2);
        let mut db = KeyframeDatabase::new(0);
        assert_eq!(
            register_descriptor(
                &mut db,
                ids[0],
                descriptor_with_basis(0),
                DescriptorSource::Bootstrap,
            ),
            KeyframeDescriptorUpdate::Initialized
        );
        assert_eq!(
            db.descriptor_source(ids[0]),
            Some(DescriptorSource::Bootstrap)
        );
        assert_eq!(
            db.register_keyframe(ids[0])
                .expect_err("duplicate registration must be rejected"),
            KeyframeDatabaseError::KeyframeAlreadyRegistered {
                keyframe_id: ids[0]
            }
        );
        assert_eq!(db.registered_len(), 1);
        assert_eq!(db.descriptor_len(), 1);

        assert_eq!(
            db.set_descriptor(ids[0], descriptor_with_basis(1), DescriptorSource::Learned)
                .expect("registered keyframe"),
            KeyframeDescriptorUpdate::Replaced
        );
        assert_eq!(db.registered_len(), 1);
        assert_eq!(db.descriptor_len(), 1);
        assert_eq!(
            db.descriptor_source(ids[0]),
            Some(DescriptorSource::Learned)
        );

        assert_eq!(
            register_descriptor(
                &mut db,
                ids[1],
                descriptor_with_basis(2),
                DescriptorSource::Learned,
            ),
            KeyframeDescriptorUpdate::Initialized
        );
        assert_eq!(
            db.descriptor_source(ids[1]),
            Some(DescriptorSource::Learned)
        );
    }

    #[test]
    fn keyframe_database_sequence_exhaustion_is_atomic() {
        let ids = make_keyframe_ids(3);
        let mut db = KeyframeDatabase::new(0);
        insert_bootstrap(&mut db, ids[0], descriptor_with_basis(0));
        db.next_sequence = Some(u64::MAX);

        db.register_keyframe(ids[1])
            .expect("the final u64 sequence value remains representable");

        let error = db
            .register_keyframe(ids[2])
            .expect_err("exhausted sequence must reject a new entry");
        assert_eq!(
            error,
            KeyframeDatabaseError::SequenceExhausted {
                next_sequence: u64::MAX
            }
        );
        assert_eq!(db.registered_len(), 2);
        assert_eq!(db.descriptor_len(), 1);
        assert_eq!(db.descriptor_source(ids[2]), None);
        assert_eq!(db.next_sequence, None);

        assert_eq!(
            db.set_descriptor(ids[0], descriptor_with_basis(2), DescriptorSource::Learned,)
                .expect("replacing an existing entry consumes no sequence"),
            KeyframeDescriptorUpdate::Replaced
        );
        assert_eq!(
            db.descriptor_source(ids[0]),
            Some(DescriptorSource::Learned)
        );
    }

    #[test]
    fn late_learned_descriptor_preserves_registered_temporal_order() {
        let ids = make_keyframe_ids(3);
        let mut db = KeyframeDatabase::new(0);
        insert_bootstrap(&mut db, ids[0], descriptor_with_basis(0));
        db.register_keyframe(ids[1])
            .expect("descriptor sequence available");
        insert_bootstrap(&mut db, ids[2], descriptor_with_basis(2));
        assert_eq!(db.registered_len(), 3);
        assert_eq!(db.descriptor_len(), 2);

        assert_eq!(
            db.set_descriptor(ids[1], descriptor_with_basis(1), DescriptorSource::Learned,)
                .expect("registered keyframe"),
            KeyframeDescriptorUpdate::Initialized
        );

        let matches = db
            .query_loop_candidates(ids[2], &descriptor_with_basis(2), 3)
            .expect("registered query keyframe");
        assert_eq!(matches.len(), 2);
        assert!(
            matches
                .iter()
                .any(|candidate| candidate.candidate() == ids[1])
        );
    }

    #[test]
    fn setting_unregistered_descriptor_is_rejected_without_mutation() {
        let id = make_keyframe_ids(1)[0];
        let mut db = KeyframeDatabase::new(0);

        let error = db
            .set_descriptor(id, descriptor_with_basis(0), DescriptorSource::Learned)
            .expect_err("unregistered keyframe must be rejected");

        assert_eq!(
            error,
            KeyframeDatabaseError::KeyframeNotRegistered { keyframe_id: id }
        );
        assert_eq!(db.registered_len(), 0);
        assert_eq!(db.descriptor_len(), 0);
    }

    #[test]
    fn keyframe_database_relocalization_query_ignores_temporal_gap() {
        let ids = make_keyframe_ids(4);
        let mut db = KeyframeDatabase::new(1000);
        for (i, id) in ids.iter().enumerate() {
            insert_bootstrap(&mut db, *id, descriptor_with_basis(i));
        }

        let matches = db.query_for_relocalization(&descriptor_with_basis(0), 4);
        assert_eq!(matches.len(), 4);
        assert_eq!(matches[0].candidate(), ids[0]);
        assert!(matches[0].cosine_similarity().value() >= matches[1].cosine_similarity().value());
    }

    #[test]
    fn keyframe_database_remove_does_not_affect_other_entries() {
        let ids = make_keyframe_ids(4);
        let mut db = KeyframeDatabase::new(0);
        register_descriptor(
            &mut db,
            ids[0],
            descriptor_with_basis(0),
            DescriptorSource::Bootstrap,
        );
        register_descriptor(
            &mut db,
            ids[1],
            descriptor_with_basis(1),
            DescriptorSource::Learned,
        );
        register_descriptor(
            &mut db,
            ids[2],
            descriptor_with_basis(2),
            DescriptorSource::Bootstrap,
        );
        register_descriptor(
            &mut db,
            ids[3],
            descriptor_with_basis(3),
            DescriptorSource::Learned,
        );

        db.remove(ids[1]).expect("registered keyframe");
        assert_eq!(db.registered_len(), 3);
        assert_eq!(db.descriptor_len(), 3);
        assert_eq!(db.descriptor_source(ids[1]), None);
        assert_eq!(
            db.descriptor_source(ids[0]),
            Some(DescriptorSource::Bootstrap)
        );
        assert_eq!(
            db.descriptor_source(ids[2]),
            Some(DescriptorSource::Bootstrap)
        );
        assert_eq!(
            db.descriptor_source(ids[3]),
            Some(DescriptorSource::Learned)
        );

        let relocalization = db.query_for_relocalization(&descriptor_with_basis(3), 10);
        assert_eq!(relocalization.len(), 3);
        assert!(
            relocalization
                .iter()
                .all(|match_| match_.candidate() != ids[1])
        );
    }

    #[test]
    fn keyframe_database_can_query_explicit_identity_after_newer_removal() {
        let ids = make_keyframe_ids(3);
        let mut db = KeyframeDatabase::new(0);
        insert_bootstrap(&mut db, ids[0], descriptor_with_basis(0));
        insert_bootstrap(&mut db, ids[1], descriptor_with_basis(1));
        insert_bootstrap(&mut db, ids[2], descriptor_with_basis(2));

        db.remove(ids[2]).expect("registered keyframe");
        let matches = db
            .query_loop_candidates(ids[1], &descriptor_with_basis(1), 10)
            .expect("registered query keyframe");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].query(), ids[1]);
        assert_eq!(matches[0].candidate(), ids[0]);
    }

    #[test]
    fn keyframe_database_removal_preserves_explicit_query_identity() {
        let ids = make_keyframe_ids(5);
        let mut db = KeyframeDatabase::new(0);
        for (i, id) in ids.iter().enumerate() {
            insert_bootstrap(&mut db, *id, descriptor_with_basis(i));
        }

        db.remove(ids[1]).expect("registered keyframe");
        db.remove(ids[3]).expect("registered keyframe");
        let matches = db
            .query_loop_candidates(ids[4], &descriptor_with_basis(4), 10)
            .expect("registered query keyframe");
        assert!(!matches.is_empty());
        assert!(matches.iter().all(|m| m.query() == ids[4]));
        assert!(
            matches
                .iter()
                .all(|m| m.candidate() != ids[1] && m.candidate() != ids[3])
        );
    }

    #[test]
    fn relocalization_config_rejects_invalid_values() {
        let err = RelocalizationConfig::new(RelocalizationConfigInput {
            max_attempts: 0,
            ..RelocalizationConfigInput::default()
        })
        .expect_err("zero attempts");
        assert!(matches!(err, RelocalizationConfigError::ZeroMaxAttempts));
        let err = RelocalizationConfig::new(RelocalizationConfigInput {
            max_attempts: 1,
            min_confirmations: 2,
            ..RelocalizationConfigInput::default()
        })
        .expect_err("confirmation requirement exceeds attempt budget");
        assert!(matches!(
            err,
            RelocalizationConfigError::TooFewMaxAttempts {
                max_attempts: 1,
                min_confirmations: 2,
            }
        ));
        let err = RelocalizationConfig::new(RelocalizationConfigInput {
            max_attempts: 10,
            min_inliers: 3,
            ..RelocalizationConfigInput::default()
        })
        .expect_err("too few inliers");
        assert!(matches!(
            err,
            RelocalizationConfigError::TooFewMinInliers { .. }
        ));
        let err = RelocalizationConfig::new(RelocalizationConfigInput {
            max_attempts: 10,
            min_inliers: 20,
            descriptor_match_threshold: 0.0,
            ..RelocalizationConfigInput::default()
        })
        .expect_err("invalid threshold");
        assert!(matches!(
            err,
            RelocalizationConfigError::DescriptorMatchThresholdOutOfRange { .. }
        ));
        let err = RelocalizationConfig::new(RelocalizationConfigInput {
            max_attempts: 10,
            min_inliers: 20,
            max_translation_delta_m: 0.0,
            ..RelocalizationConfigInput::default()
        })
        .expect_err("invalid translation threshold");
        assert!(matches!(
            err,
            RelocalizationConfigError::NonPositiveMaxTranslationDelta { .. }
        ));
    }

    #[test]
    fn relocalization_config_default_values() {
        let input = RelocalizationConfigInput::default();
        assert_eq!(input.max_attempts, 30);
        assert_eq!(input.min_inliers, 20);
        assert_eq!(input.max_candidates, 3);
        assert_eq!(input.descriptor_match_threshold, 0.7);
        assert_eq!(input.min_confirmations, 2);
        assert_eq!(input.max_translation_delta_m, 1.5);
        assert_eq!(input.max_rotation_delta_deg, 10.0);
        let cfg = RelocalizationConfig::default();
        assert_eq!(cfg.max_attempts(), input.max_attempts);
        assert_eq!(cfg.min_inliers(), input.min_inliers);
        assert_eq!(cfg.max_candidates(), input.max_candidates);
        assert_eq!(
            cfg.descriptor_match_threshold(),
            input.descriptor_match_threshold
        );
        assert_eq!(cfg.min_confirmations(), input.min_confirmations);
        assert_eq!(cfg.max_translation_delta_m(), input.max_translation_delta_m);
        assert_eq!(cfg.max_rotation_delta_deg(), input.max_rotation_delta_deg);
        RelocalizationConfig::new(input).expect("default input must parse without repair");
    }

    #[test]
    fn place_match_verify_succeeds_on_synthetic_geometry() {
        let (
            map,
            query_kf,
            match_kf,
            query_keypoints,
            correspondences,
            intrinsics,
            query_pose_world,
        ) = make_loop_fixture();
        let candidate = place_match(query_kf, match_kf);
        let verified = candidate
            .verify(
                &query_keypoints,
                &correspondences,
                &map,
                intrinsics,
                RansacConfig::default(),
                4,
            )
            .expect("verified loop");
        assert_eq!(verified.match_kf(), match_kf);
        assert_eq!(verified.query_kf(), query_kf);
        assert_eq!(verified.map_snapshot(), map.snapshot());
        assert!(verified.inlier_count() >= 4);
        let actual = verified.query_pose_world().translation();
        let expected = query_pose_world.translation();
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 1e-3,
                "translation mismatch on axis {axis}: actual={}, expected={}",
                actual[axis],
                expected[axis]
            );
        }
    }

    #[test]
    fn place_match_verify_rejects_insufficient_inliers() {
        let (map, query_kf, match_kf, query_keypoints, correspondences, intrinsics, _) =
            make_loop_fixture();
        let candidate = place_match(query_kf, match_kf);
        let err = candidate
            .verify(
                &query_keypoints,
                &correspondences,
                &map,
                intrinsics,
                RansacConfig::default(),
                100,
            )
            .expect_err("expected inlier threshold failure");
        assert!(matches!(
            err,
            LoopVerificationError::InsufficientInliers { .. }
        ));
    }

    #[test]
    fn place_match_verify_propagates_map_failure() {
        let ids = make_keyframe_ids(1);
        let candidate = place_match(ids[0], ids[0]);
        let map = SlamMap::new();
        let query_keypoints = vec![Keypoint { x: 10.0, y: 10.0 }; 4];
        let correspondences = vec![(0, 0), (1, 1), (2, 2), (3, 3)];
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let err = candidate
            .verify(
                &query_keypoints,
                &correspondences,
                &map,
                intrinsics,
                RansacConfig::default(),
                4,
            )
            .expect_err("expected pnp failure");
        assert!(matches!(
            err,
            LoopVerificationError::Map(MapError::KeyframeNotFound(_))
        ));
    }

    #[test]
    fn place_match_verify_rejects_out_of_bounds_query_index() {
        let (map, query_kf, match_kf, query_keypoints, mut correspondences, intrinsics, _) =
            make_loop_fixture();
        correspondences[0].0 = query_keypoints.len();
        let candidate = place_match(query_kf, match_kf);

        let error = candidate
            .verify(
                &query_keypoints,
                &correspondences,
                &map,
                intrinsics,
                RansacConfig::default(),
                4,
            )
            .expect_err("invalid query index must fail");
        assert!(matches!(
            error,
            LoopVerificationError::QueryIndexOutOfBounds { index, len }
                if index == len && len == query_keypoints.len()
        ));
    }

    #[test]
    fn relocalization_match_verify_succeeds_without_map_mutation() {
        let (map, _, match_kf, query_keypoints, correspondences, intrinsics, query_pose_world) =
            make_loop_fixture();
        let before_generation = map.generation();
        let candidate = relocalization_match(match_kf);
        let verified = candidate
            .verify(
                &query_keypoints,
                &correspondences,
                &map,
                intrinsics,
                RansacConfig::default(),
                4,
            )
            .expect("relocalization should verify");
        assert_eq!(verified.match_kf(), match_kf);
        assert!(verified.inlier_count() >= 4);
        let actual = verified.pose_world().translation();
        let expected = query_pose_world.translation();
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 1e-3,
                "translation mismatch on axis {axis}: actual={}, expected={}",
                actual[axis],
                expected[axis]
            );
        }
        assert_eq!(map.generation(), before_generation);
    }

    #[test]
    fn aggregate_empty_descriptors_is_rejected() {
        let err = aggregate_global_descriptor(&[]).expect_err("empty descriptor set should fail");
        assert!(matches!(err, GlobalDescriptorError::EmptyInput));
    }

    #[test]
    fn aggregate_single_descriptor_produces_unit_norm() {
        let mut data = [0.0_f32; 256];
        data[4] = 1.0;
        data[17] = 0.5;
        let descriptor =
            aggregate_global_descriptor(&[descriptor(data)]).expect("aggregated descriptor");
        let values = descriptor.as_array();
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "descriptor norm should be 1, got {norm}"
        );
        assert!(values[4] > 0.0);
        assert!(values[256 + 4] > 0.0);
    }

    #[test]
    fn global_descriptor_normalization_handles_extreme_finite_inputs() {
        let descriptor = GlobalDescriptor::try_new([f32::MAX; super::GLOBAL_DESCRIPTOR_DIM])
            .expect("finite descriptor must not overflow during normalization");
        let norm = descriptor
            .as_array()
            .iter()
            .map(|&value| {
                let value = f64::from(value);
                value * value
            })
            .sum::<f64>()
            .sqrt();

        assert!(descriptor.as_array().iter().all(|value| value.is_finite()));
        assert!((norm - 1.0).abs() < 1e-6, "normalized norm was {norm}");
        let similarity = descriptor.clamped_cosine_similarity(&descriptor).value();
        assert!((similarity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn global_descriptor_max_pool_preserves_all_negative_dimensions() {
        let descriptor = aggregate_global_descriptor(&[
            descriptor([-2.0; crate::DESCRIPTOR_DIM]),
            descriptor([-1.0; crate::DESCRIPTOR_DIM]),
        ])
        .expect("negative finite descriptors remain valid");

        assert!(
            descriptor.as_array()[crate::DESCRIPTOR_DIM..]
                .iter()
                .all(|&value| value < 0.0)
        );
    }

    #[test]
    fn match_descriptors_finds_mutual_matches() {
        let mut map = SlamMap::new();
        let keypoints = vec![Keypoint { x: 20.0, y: 20.0 }, Keypoint { x: 40.0, y: 20.0 }];
        let image_size = FrameDimensions::try_new(80, 60).expect("image size");
        let kf = map
            .add_keyframe(
                FrameId::new(11),
                Timestamp::from_nanos(11),
                Pose::identity(),
                image_size,
                keypoints,
            )
            .expect("keyframe");

        let mut q0 = [0.0_f32; 256];
        q0[7] = 1.0;
        let mut q1 = [0.0_f32; 256];
        q1[23] = 1.0;
        let query = vec![descriptor(q0), descriptor(q1)];

        let kp0 = map.keyframe_keypoint(kf, 0).expect("kp0");
        map.add_map_point(
            Point3 {
                x: -0.1,
                y: 0.0,
                z: 3.0,
            },
            query[0].quantize(),
            kp0,
        )
        .expect("point0");
        let kp1 = map.keyframe_keypoint(kf, 1).expect("kp1");
        map.add_map_point(
            Point3 {
                x: 0.1,
                y: 0.0,
                z: 3.0,
            },
            query[1].quantize(),
            kp1,
        )
        .expect("point1");

        let matches =
            match_descriptors_for_loop(&query, kf, &map, 0.95).expect("descriptor matches");
        assert_eq!(matches.correspondences().len(), 2);
        assert!(matches.correspondences().contains(&(0, 0)));
        assert!(matches.correspondences().contains(&(1, 1)));
        assert_eq!(matches.zero_norm_query_descriptors(), 0);
        assert_eq!(matches.zero_norm_candidate_descriptors(), 0);
    }

    #[test]
    fn match_descriptors_includes_threshold_and_preserves_first_tie() {
        let mut map = SlamMap::new();
        let image_size = FrameDimensions::try_new(80, 60).expect("image size");
        let keyframe = map
            .add_keyframe(
                FrameId::new(13),
                Timestamp::from_nanos(13),
                Pose::identity(),
                image_size,
                vec![Keypoint { x: 20.0, y: 20.0 }, Keypoint { x: 40.0, y: 20.0 }],
            )
            .expect("keyframe");
        let mut values = [0.0_f32; crate::DESCRIPTOR_DIM];
        values[9] = 1.0;
        let descriptor = descriptor(values);
        for index in 0..2 {
            let keypoint = map.keyframe_keypoint(keyframe, index).expect("keypoint");
            map.add_map_point(
                Point3 {
                    x: index as f32,
                    y: 0.0,
                    z: 3.0,
                },
                descriptor.quantize(),
                keypoint,
            )
            .expect("map point");
        }

        let matches = match_descriptors_for_loop(&[descriptor, descriptor], keyframe, &map, 1.0)
            .expect("inclusive threshold");
        assert_eq!(matches.correspondences(), [(0, 0)]);
    }

    #[test]
    fn match_descriptors_reports_zero_norm_skips_without_fabricating_similarity() {
        let mut map = SlamMap::new();
        let image_size = FrameDimensions::try_new(80, 60).expect("image size");
        let keyframe = map
            .add_keyframe(
                FrameId::new(12),
                Timestamp::from_nanos(12),
                Pose::identity(),
                image_size,
                vec![Keypoint { x: 20.0, y: 20.0 }, Keypoint { x: 40.0, y: 20.0 }],
            )
            .expect("keyframe");
        let zero = Descriptor::ZERO;
        let mut basis_values = [0.0; crate::DESCRIPTOR_DIM];
        basis_values[7] = 1.0;
        let basis = descriptor(basis_values);
        for (index, descriptor) in [zero, basis].iter().enumerate() {
            let keypoint = map.keyframe_keypoint(keyframe, index).expect("keypoint");
            map.add_map_point(
                Point3 {
                    x: index as f32 * 0.1,
                    y: 0.0,
                    z: 3.0,
                },
                descriptor.quantize(),
                keypoint,
            )
            .expect("map point");
        }

        let result = match_descriptors_for_loop(&[zero, basis], keyframe, &map, 0.95)
            .expect("descriptor matching");

        assert_eq!(result.correspondences(), [(1, 1)]);
        assert_eq!(result.zero_norm_query_descriptors(), 1);
        assert_eq!(result.zero_norm_candidate_descriptors(), 1);
    }

    #[test]
    fn match_descriptors_skips_keypoints_without_map_points() {
        let mut map = SlamMap::new();
        let keypoints = vec![
            Keypoint { x: 20.0, y: 20.0 },
            Keypoint { x: 40.0, y: 20.0 },
            Keypoint { x: 60.0, y: 20.0 },
        ];
        let image_size = FrameDimensions::try_new(80, 60).expect("image size");
        let kf = map
            .add_keyframe(
                FrameId::new(21),
                Timestamp::from_nanos(21),
                Pose::identity(),
                image_size,
                keypoints,
            )
            .expect("keyframe");

        let mut q0 = [0.0_f32; 256];
        q0[3] = 1.0;
        let mut q1 = [0.0_f32; 256];
        q1[8] = 1.0;
        let mut q2 = [0.0_f32; 256];
        q2[13] = 1.0;
        let query = vec![descriptor(q0), descriptor(q1), descriptor(q2)];

        let only_observed = map.keyframe_keypoint(kf, 1).expect("kp1");
        map.add_map_point(
            Point3 {
                x: 0.0,
                y: 0.1,
                z: 3.0,
            },
            query[1].quantize(),
            only_observed,
        )
        .expect("point");

        let matches =
            match_descriptors_for_loop(&query, kf, &map, 0.95).expect("descriptor matches");
        assert_eq!(matches.correspondences(), [(1, 1)]);
    }

    #[test]
    fn match_descriptors_returns_original_sparse_candidate_keypoint_indices() {
        let mut map = SlamMap::new();
        let image_size = FrameDimensions::try_new(80, 60).expect("image size");
        let keyframe = map
            .add_keyframe(
                FrameId::new(31),
                Timestamp::from_nanos(31),
                Pose::identity(),
                image_size,
                (0..5)
                    .map(|index| Keypoint {
                        x: 10.0 + index as f32 * 10.0,
                        y: 20.0,
                    })
                    .collect(),
            )
            .expect("keyframe");
        let mut first = [0.0_f32; crate::DESCRIPTOR_DIM];
        first[3] = 1.0;
        let mut second = [0.0_f32; crate::DESCRIPTOR_DIM];
        second[17] = 1.0;
        let descriptors = [descriptor(first), descriptor(second)];

        for (keypoint_index, descriptor) in [(1, &descriptors[0]), (4, &descriptors[1])] {
            let observation = map
                .keyframe_keypoint(keyframe, keypoint_index)
                .expect("sparse keypoint");
            map.add_map_point(
                Point3 {
                    x: keypoint_index as f32 * 0.1,
                    y: 0.0,
                    z: 3.0,
                },
                descriptor.quantize(),
                observation,
            )
            .expect("map point");
        }

        let matches = match_descriptors_for_loop(&descriptors, keyframe, &map, 0.95)
            .expect("descriptor matching");
        assert_eq!(matches.correspondences(), [(0, 1), (1, 4)]);
    }

    #[test]
    fn match_descriptors_propagates_map_error_with_source() {
        let map = SlamMap::new();
        let mut query = [0.0_f32; 256];
        query[7] = 1.0;
        let err =
            match_descriptors_for_loop(&[descriptor(query)], KeyframeId::default(), &map, 0.95)
                .expect_err("missing keyframe should return map error");
        assert!(matches!(
            err,
            DescriptorMatchError::Map(MapError::KeyframeNotFound(_))
        ));
        assert!(err.source().is_some());
    }

    #[test]
    fn match_descriptors_rejects_invalid_similarity_threshold() {
        let map = SlamMap::new();
        for value in [0.0, -0.1, 1.1, f32::NAN] {
            assert!(matches!(
                match_descriptors_for_loop(&[], KeyframeId::default(), &map, value),
                Err(DescriptorMatchError::InvalidSimilarityThreshold { value: actual })
                    if actual.to_bits() == value.to_bits()
            ));
        }
    }

    #[test]
    fn loop_closure_config_default_values() {
        let input = LoopClosureConfigInput::default();
        assert_eq!(input.similarity_threshold, 0.75);
        assert_eq!(input.descriptor_match_threshold, 0.7);
        assert_eq!(input.min_inliers, 20);
        assert_eq!(input.max_candidates, 3);
        assert_eq!(input.temporal_gap, 30);
        assert_eq!(input.min_streak, 3);
        assert_eq!(input.max_correction_translation_m, 5.0);
        assert_eq!(input.max_correction_rotation_deg, 30.0);
        let cfg = LoopClosureConfig::default();
        assert_eq!(cfg.similarity_threshold(), input.similarity_threshold);
        assert_eq!(
            cfg.descriptor_match_threshold(),
            input.descriptor_match_threshold
        );
        assert_eq!(cfg.min_inliers(), input.min_inliers);
        assert_eq!(cfg.max_candidates(), input.max_candidates);
        assert_eq!(cfg.temporal_gap(), input.temporal_gap);
        assert_eq!(cfg.min_streak(), input.min_streak);
        assert_eq!(
            cfg.max_correction_translation_m(),
            input.max_correction_translation_m
        );
        assert_eq!(
            cfg.max_correction_rotation_deg(),
            input.max_correction_rotation_deg
        );
        LoopClosureConfig::new(input).expect("default input must parse without repair");
    }

    #[test]
    fn loop_correction_translation_limit_requires_positive_finite_meters() {
        for value in [0.0, f32::NAN, f32::INFINITY] {
            let input = LoopClosureConfigInput {
                max_correction_translation_m: value,
                ..LoopClosureConfigInput::default()
            };
            assert!(matches!(
                LoopClosureConfig::new(input),
                Err(super::LoopClosureConfigError::InvalidMaxCorrectionTranslationM {
                    value: actual,
                }) if actual.to_bits() == value.to_bits()
            ));
        }
    }

    #[test]
    fn loop_apply_error_preserves_pose_graph_source() {
        let error = LoopApplyError::PoseGraph {
            source: PoseGraphError::NotConverged {
                outer_iterations: 4,
                last_linear_solve_residual_norm: 2.5,
            },
        };

        assert_eq!(error.kind(), LoopApplyErrorKind::PoseGraph);
        assert!(error.to_string().contains("pose-graph optimization"));
        let detected = super::LoopDetectError::ApplyFailed(error);
        let apply = detected.source().expect("loop-apply source");
        let pose_graph = apply.source().expect("pose-graph source");
        assert!(pose_graph.to_string().contains("did not converge"));
    }

    #[test]
    fn loop_apply_error_preserves_essential_graph_source() {
        let keyframe_id = KeyframeId::default();
        let source = super::EssentialGraphError::KeyframeNotFound { keyframe_id };
        let error = LoopApplyError::EssentialGraph { source };

        assert_eq!(error.kind(), LoopApplyErrorKind::EssentialGraph);
        assert!(error.to_string().contains("essential-graph mutation"));
        assert_eq!(
            error.source().expect("essential-graph source").to_string(),
            source.to_string()
        );
    }

    #[test]
    fn loop_apply_error_preserves_map_frame_alignment_source() {
        let source = crate::GeometryError::NonFiniteTransformTranslation {
            operation: "test map/odometry alignment",
            axis: 2,
            value: f64::INFINITY,
        };
        let error = LoopApplyError::MapFrameAlignment { source };

        assert_eq!(error.kind(), LoopApplyErrorKind::MapFrameAlignment);
        assert!(error.to_string().contains("map/odometry alignment failed"));
        let preserved = error.source().expect("geometry source");
        assert_eq!(preserved.to_string(), source.to_string());
    }

    #[test]
    fn loop_pose_conversion_errors_preserve_source_and_are_not_candidate_rejections() {
        let source = crate::Pose64Error::TranslationOutOfF32Range {
            axis: 1,
            value: f64::MAX,
        };
        let apply = LoopApplyError::PoseConversion {
            operation: "test optimized pose narrowing",
            keyframe_id: None,
            source,
        };
        assert_eq!(apply.kind(), LoopApplyErrorKind::PoseConversion);
        assert_eq!(
            apply.source().expect("pose conversion source").to_string(),
            source.to_string()
        );

        let detect = super::LoopDetectError::CorrectionEvaluation { source };
        assert!(!detect.is_candidate_rejection());
        assert_eq!(
            detect
                .source()
                .expect("correction evaluation source")
                .to_string(),
            source.to_string()
        );
    }

    #[test]
    fn loop_apply_error_classifies_map_failure_without_discarding_source() {
        let keyframe_id = KeyframeId::default();
        let error = LoopApplyError::from(MapError::KeyframeNotFound(keyframe_id));

        assert_eq!(error.kind(), LoopApplyErrorKind::MissingKeyframe);
        assert!(error.source().is_some());
        assert!(error.to_string().contains("keyframe not found"));
    }
}
