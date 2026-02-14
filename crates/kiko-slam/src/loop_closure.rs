use std::num::NonZeroUsize;

use crate::map::{KeyframeId, SlamMap};
use crate::{
    solve_pnp_ransac, CompactDescriptor, Descriptor, Keypoint, Observation, PinholeIntrinsics,
    PnpError, Pose, RansacConfig,
};

const GLOBAL_DESCRIPTOR_DIM: usize = 512;

#[derive(Clone, Copy, Debug)]
pub struct LoopClosureConfig {
    similarity_threshold: f32,
    descriptor_match_threshold: f32,
    min_inliers: NonZeroUsize,
    max_candidates: NonZeroUsize,
    temporal_gap: NonZeroUsize,
    min_streak: NonZeroUsize,
    max_correction_translation: f32,
    max_correction_rotation_deg: f32,
    ransac: RansacConfig,
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
    NonPositiveMaxCorrectionTranslation { value: f32 },
    InvalidMaxCorrectionRotationDeg { value: f32 },
    ZeroRansacIterations,
    NonPositiveRansacThresholdPx { value: f32 },
}

impl std::fmt::Display for LoopClosureConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopClosureConfigError::SimilarityThresholdOutOfRange { value } => {
                write!(f, "loop similarity threshold must be in (0, 1], got {value}")
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
            LoopClosureConfigError::NonPositiveMaxCorrectionTranslation { value } => write!(
                f,
                "loop max correction translation must be > 0, got {value}"
            ),
            LoopClosureConfigError::InvalidMaxCorrectionRotationDeg { value } => write!(
                f,
                "loop max correction rotation must be in (0, 180], got {value}"
            ),
            LoopClosureConfigError::ZeroRansacIterations => {
                write!(f, "loop ransac max iterations must be > 0")
            }
            LoopClosureConfigError::NonPositiveRansacThresholdPx { value } => write!(
                f,
                "loop ransac reprojection threshold must be > 0, got {value}"
            ),
        }
    }
}

impl std::error::Error for LoopClosureConfigError {}

impl LoopClosureConfig {
    pub fn new(
        similarity_threshold: f32,
        descriptor_match_threshold: f32,
        min_inliers: usize,
        max_candidates: usize,
        temporal_gap: usize,
        min_streak: usize,
        max_correction_translation: f32,
        max_correction_rotation_deg: f32,
        ransac: RansacConfig,
    ) -> Result<Self, LoopClosureConfigError> {
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
            return Err(
                LoopClosureConfigError::DescriptorMatchThresholdOutOfRange {
                    value: descriptor_match_threshold,
                },
            );
        }
        let min_inliers =
            NonZeroUsize::new(min_inliers).ok_or(LoopClosureConfigError::ZeroMinInliers)?;
        let max_candidates =
            NonZeroUsize::new(max_candidates).ok_or(LoopClosureConfigError::ZeroMaxCandidates)?;
        let temporal_gap =
            NonZeroUsize::new(temporal_gap).ok_or(LoopClosureConfigError::ZeroTemporalGap)?;
        let min_streak =
            NonZeroUsize::new(min_streak).ok_or(LoopClosureConfigError::ZeroMinStreak)?;
        if min_inliers.get() < 4 {
            return Err(LoopClosureConfigError::TooFewMinInliers {
                value: min_inliers.get(),
                min: 4,
            });
        }
        if !max_correction_translation.is_finite() || max_correction_translation <= 0.0 {
            return Err(
                LoopClosureConfigError::NonPositiveMaxCorrectionTranslation {
                    value: max_correction_translation,
                },
            );
        }
        if !max_correction_rotation_deg.is_finite()
            || max_correction_rotation_deg <= 0.0
            || max_correction_rotation_deg > 180.0
        {
            return Err(LoopClosureConfigError::InvalidMaxCorrectionRotationDeg {
                value: max_correction_rotation_deg,
            });
        }
        if ransac.max_iterations == 0 {
            return Err(LoopClosureConfigError::ZeroRansacIterations);
        }
        if !ransac.reprojection_threshold_px.is_finite() || ransac.reprojection_threshold_px <= 0.0
        {
            return Err(LoopClosureConfigError::NonPositiveRansacThresholdPx {
                value: ransac.reprojection_threshold_px,
            });
        }

        Ok(Self {
            similarity_threshold,
            descriptor_match_threshold,
            min_inliers,
            max_candidates,
            temporal_gap,
            min_streak,
            max_correction_translation,
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

    pub fn max_correction_translation(self) -> f32 {
        self.max_correction_translation
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
        Self::new(0.75, 0.7, 20, 3, 30, 3, 5.0, 30.0, RansacConfig::default())
            .expect("default loop closure config should be valid")
    }
}

#[derive(Debug)]
pub enum LoopDetectError {
    TooFewCorrespondences { count: usize },
    VerificationFailed(LoopVerificationError),
    CorrectionTooLarge { translation: f32, rotation_deg: f32 },
    ApplyFailed(String),
}

impl std::fmt::Display for LoopDetectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopDetectError::TooFewCorrespondences { count } => {
                write!(f, "loop closure rejected: too few correspondences ({count})")
            }
            LoopDetectError::VerificationFailed(err) => {
                write!(f, "loop closure verification failed: {err}")
            }
            LoopDetectError::CorrectionTooLarge {
                translation,
                rotation_deg,
            } => write!(
                f,
                "loop closure rejected: correction too large (translation={translation:.3}m, rotation={rotation_deg:.2}deg)"
            ),
            LoopDetectError::ApplyFailed(err) => {
                write!(f, "loop closure apply failed: {err}")
            }
        }
    }
}

impl std::error::Error for LoopDetectError {}

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
                write!(f, "global descriptor requires at least one local descriptor")
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
pub struct GlobalDescriptor([f32; GLOBAL_DESCRIPTOR_DIM]);

impl GlobalDescriptor {
    pub fn try_new(values: [f32; GLOBAL_DESCRIPTOR_DIM]) -> Result<Self, GlobalDescriptorError> {
        let mut norm_sq = 0.0_f32;
        for (idx, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(GlobalDescriptorError::NonFiniteValue { index: idx, value });
            }
            norm_sq += value * value;
        }
        if norm_sq <= 0.0 {
            return Err(GlobalDescriptorError::ZeroNorm);
        }

        let inv_norm = 1.0 / norm_sq.sqrt();
        let mut normalized = [0.0_f32; GLOBAL_DESCRIPTOR_DIM];
        for (idx, value) in values.into_iter().enumerate() {
            normalized[idx] = value * inv_norm;
        }
        Ok(Self(normalized))
    }

    pub fn as_array(&self) -> &[f32; GLOBAL_DESCRIPTOR_DIM] {
        &self.0
    }

    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;
        for i in 0..GLOBAL_DESCRIPTOR_DIM {
            let a = self.0[i] as f64;
            let b = other.0[i] as f64;
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        if norm_a <= 0.0 || norm_b <= 0.0 {
            return 0.0;
        }
        (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
    }

    pub fn from_local_descriptors(
        descriptors: &[Descriptor],
    ) -> Result<Self, GlobalDescriptorError> {
        if descriptors.is_empty() {
            return Err(GlobalDescriptorError::EmptyInput);
        }

        let mut out = [0.0_f32; GLOBAL_DESCRIPTOR_DIM];
        let count = descriptors.len() as f32;
        for d in descriptors {
            for (idx, value) in d.0.iter().copied().enumerate() {
                out[idx] += value;
                let max_slot = &mut out[256 + idx];
                if value > *max_slot {
                    *max_slot = value;
                }
            }
        }
        for value in &mut out[..256] {
            *value /= count;
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
) -> Vec<(usize, usize)> {
    if query_descriptors.is_empty()
        || !similarity_threshold.is_finite()
        || similarity_threshold <= 0.0
        || similarity_threshold > 1.0
    {
        return Vec::new();
    }

    let candidate_descriptors = match map.keyframe_point_descriptors(candidate_kf) {
        Ok(values) => values,
        Err(_) => return Vec::new(),
    };
    if candidate_descriptors.is_empty() {
        return Vec::new();
    }

    let query_quantized: Vec<CompactDescriptor> =
        query_descriptors.iter().map(Descriptor::quantize).collect();

    let mut query_best = vec![None::<(usize, f32)>; query_quantized.len()];
    for (query_idx, query_desc) in query_quantized.iter().enumerate() {
        let mut best = similarity_threshold;
        let mut best_candidate = None;
        for (candidate_pos, (_, candidate_desc)) in candidate_descriptors.iter().enumerate() {
            let sim = query_desc.cosine_similarity(candidate_desc);
            if sim >= best {
                best = sim;
                best_candidate = Some((candidate_pos, sim));
            }
        }
        query_best[query_idx] = best_candidate;
    }

    let mut candidate_best = vec![None::<(usize, f32)>; candidate_descriptors.len()];
    for (candidate_pos, (_, candidate_desc)) in candidate_descriptors.iter().enumerate() {
        let mut best = similarity_threshold;
        let mut best_query = None;
        for (query_idx, query_desc) in query_quantized.iter().enumerate() {
            let sim = query_desc.cosine_similarity(candidate_desc);
            if sim >= best {
                best = sim;
                best_query = Some((query_idx, sim));
            }
        }
        candidate_best[candidate_pos] = best_query;
    }

    let mut correspondences = Vec::new();
    for (query_idx, best) in query_best.iter().enumerate() {
        let Some((candidate_pos, _)) = best else {
            continue;
        };
        let Some((back_query_idx, _)) = candidate_best[*candidate_pos] else {
            continue;
        };
        if back_query_idx == query_idx {
            correspondences.push((query_idx, candidate_descriptors[*candidate_pos].0.index()));
        }
    }
    correspondences
}

#[derive(Clone, Debug)]
pub struct PlaceMatch {
    pub query: KeyframeId,
    pub candidate: KeyframeId,
    pub similarity: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescriptorSource {
    Bootstrap,
    Learned,
}

#[derive(Clone, Debug)]
struct KeyframeDescriptorEntry {
    keyframe_id: KeyframeId,
    descriptor: GlobalDescriptor,
    source: DescriptorSource,
    seq: u64,
}

#[derive(Clone, Debug)]
pub struct KeyframeDatabase {
    entries: Vec<KeyframeDescriptorEntry>,
    temporal_gap: usize,
    next_seq: u64,
}

impl KeyframeDatabase {
    pub fn new(temporal_gap: usize) -> Self {
        Self {
            entries: Vec::new(),
            temporal_gap,
            next_seq: 0,
        }
    }

    pub fn temporal_gap(&self) -> usize {
        self.temporal_gap
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn insert(&mut self, id: KeyframeId, descriptor: GlobalDescriptor) {
        self.insert_with_source(id, descriptor, DescriptorSource::Bootstrap);
    }

    pub fn insert_with_source(
        &mut self,
        id: KeyframeId,
        descriptor: GlobalDescriptor,
        source: DescriptorSource,
    ) {
        if let Some(existing) = self.entries.iter_mut().find(|entry| entry.keyframe_id == id) {
            existing.descriptor = descriptor;
            existing.source = source;
            return;
        }
        let seq = self.next_seq;
        self.next_seq = self.next_seq.saturating_add(1);
        self.entries.push(KeyframeDescriptorEntry {
            keyframe_id: id,
            descriptor,
            source,
            seq,
        });
    }

    pub fn replace_descriptor(
        &mut self,
        id: KeyframeId,
        descriptor: GlobalDescriptor,
        source: DescriptorSource,
    ) -> bool {
        let Some(existing) = self.entries.iter_mut().find(|entry| entry.keyframe_id == id) else {
            return false;
        };
        existing.descriptor = descriptor;
        existing.source = source;
        true
    }

    pub fn remove(&mut self, id: KeyframeId) -> bool {
        let before = self.entries.len();
        self.entries.retain(|entry| entry.keyframe_id != id);
        self.entries.len() != before
    }

    pub fn descriptor_source(&self, id: KeyframeId) -> Option<DescriptorSource> {
        self.entries
            .iter()
            .find(|entry| entry.keyframe_id == id)
            .map(|entry| entry.source)
    }

    pub fn query(&self, descriptor: &GlobalDescriptor, top_k: usize) -> Vec<PlaceMatch> {
        if top_k == 0 || self.entries.is_empty() {
            return Vec::new();
        }
        let query_entry = self.entries.last().expect("non-empty checked");
        let query_id = query_entry.keyframe_id;
        let query_seq = query_entry.seq;
        let mut matches = Vec::new();
        for candidate in &self.entries {
            if candidate.keyframe_id == query_id {
                continue;
            }
            if query_seq.saturating_sub(candidate.seq) <= self.temporal_gap as u64 {
                continue;
            }
            matches.push(PlaceMatch {
                query: query_id,
                candidate: candidate.keyframe_id,
                similarity: descriptor.cosine_similarity(&candidate.descriptor),
            });
        }
        matches.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches.truncate(top_k);
        matches
    }
}

impl Default for KeyframeDatabase {
    fn default() -> Self {
        Self::new(30)
    }
}

#[derive(Clone, Debug)]
pub struct LoopCandidate {
    pub query_kf: KeyframeId,
    pub match_kf: KeyframeId,
    pub similarity: f32,
}

#[derive(Clone, Debug)]
pub struct VerifiedLoop {
    query_kf: KeyframeId,
    match_kf: KeyframeId,
    relative_pose: Pose,
    inlier_count: usize,
}

impl VerifiedLoop {
    pub fn query_kf(&self) -> KeyframeId {
        self.query_kf
    }

    pub fn match_kf(&self) -> KeyframeId {
        self.match_kf
    }

    pub fn relative_pose(&self) -> Pose {
        self.relative_pose
    }

    pub fn inlier_count(&self) -> usize {
        self.inlier_count
    }

    #[cfg(test)]
    pub(crate) fn from_parts(
        query_kf: KeyframeId,
        match_kf: KeyframeId,
        relative_pose: Pose,
        inlier_count: usize,
    ) -> Self {
        Self {
            query_kf,
            match_kf,
            relative_pose,
            inlier_count,
        }
    }
}

#[derive(Debug)]
pub enum LoopVerificationError {
    TooFewMatches { count: usize },
    PnpFailed(PnpError),
    InsufficientInliers { inliers: usize, required: usize },
}

impl std::fmt::Display for LoopVerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopVerificationError::TooFewMatches { count } => {
                write!(f, "loop verification needs at least 4 matches, got {count}")
            }
            LoopVerificationError::PnpFailed(err) => write!(f, "loop verification PnP failed: {err}"),
            LoopVerificationError::InsufficientInliers { inliers, required } => write!(
                f,
                "loop verification inliers below threshold: inliers={inliers}, required={required}"
            ),
        }
    }
}

impl std::error::Error for LoopVerificationError {}

impl LoopCandidate {
    pub fn verify(
        &self,
        query_keypoints: &[Keypoint],
        correspondences: &[(usize, usize)],
        map: &SlamMap,
        intrinsics: PinholeIntrinsics,
        ransac_config: RansacConfig,
        min_inliers: usize,
    ) -> Result<VerifiedLoop, LoopVerificationError> {
        let required_inliers = min_inliers.max(4);

        if correspondences.len() < 4 {
            return Err(LoopVerificationError::TooFewMatches {
                count: correspondences.len(),
            });
        }

        let mut observations = Vec::with_capacity(correspondences.len());
        for &(query_idx, match_idx) in correspondences {
            let Some(&pixel) = query_keypoints.get(query_idx) else {
                continue;
            };
            let Ok(match_ref) = map.keyframe_keypoint(self.match_kf, match_idx) else {
                continue;
            };
            let Some(point_id) = map.map_point_for_keypoint(match_ref).ok().flatten() else {
                continue;
            };
            let Some(point) = map.point(point_id) else {
                continue;
            };
            let obs = Observation::try_new(point.position(), pixel, intrinsics)
                .map_err(LoopVerificationError::PnpFailed)?;
            observations.push(obs);
        }

        if observations.len() < 4 {
            return Err(LoopVerificationError::TooFewMatches {
                count: observations.len(),
            });
        }

        let pnp_min_inliers = ransac_config
            .min_inliers
            .min(required_inliers)
            .min(observations.len())
            .max(4);
        let pnp_config = RansacConfig {
            min_inliers: pnp_min_inliers,
            ..ransac_config
        };

        let result = solve_pnp_ransac(&observations, intrinsics, pnp_config)
            .map_err(LoopVerificationError::PnpFailed)?;
        if result.inliers.len() < required_inliers {
            return Err(LoopVerificationError::InsufficientInliers {
                inliers: result.inliers.len(),
                required: required_inliers,
            });
        }
        Ok(VerifiedLoop {
            query_kf: self.query_kf,
            match_kf: self.match_kf,
            relative_pose: result.pose,
            inlier_count: result.inliers.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        aggregate_global_descriptor, match_descriptors_for_loop, DescriptorSource,
        GlobalDescriptor, GlobalDescriptorError, KeyframeDatabase, LoopCandidate,
        LoopClosureConfig, LoopVerificationError,
    };
    use crate::map::{ImageSize, KeyframeId, SlamMap};
    use crate::test_helpers::{make_pinhole_intrinsics, project_world_point};
    use crate::{
        CompactDescriptor, Descriptor, FrameId, Keypoint, Point3, Pose, RansacConfig, Timestamp,
    };

    fn descriptor_with_basis(idx: usize) -> GlobalDescriptor {
        let mut d = [0.0_f32; 512];
        d[idx] = 1.0;
        GlobalDescriptor::try_new(d).expect("valid basis descriptor")
    }

    fn make_keyframe_ids(n: usize) -> Vec<KeyframeId> {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(320, 240).expect("size");
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

    fn make_loop_fixture() -> (
        SlamMap,
        KeyframeId,
        Vec<Keypoint>,
        Vec<(usize, usize)>,
        crate::PinholeIntrinsics,
    ) {
        let intrinsics =
            make_pinhole_intrinsics(640, 480, 420.0, 418.0, 320.0, 240.0).expect("intrinsics");
        let match_pose = Pose::identity();
        let query_pose = Pose::identity();

        let world_points = vec![
            Point3 { x: -0.3, y: -0.2, z: 3.2 },
            Point3 { x: -0.1, y: -0.2, z: 3.5 },
            Point3 { x: 0.1, y: -0.1, z: 3.8 },
            Point3 { x: 0.3, y: -0.1, z: 3.4 },
            Point3 { x: -0.2, y: 0.1, z: 3.6 },
            Point3 { x: 0.2, y: 0.2, z: 3.9 },
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
        let size = ImageSize::try_new(640, 480).expect("size");
        let match_kf = map
            .add_keyframe(
                FrameId::new(10),
                Timestamp::from_nanos(10),
                match_pose,
                size,
                match_keypoints,
            )
            .expect("match kf");

        for (idx, &point) in world_points.iter().enumerate() {
            let kp_ref = map.keyframe_keypoint(match_kf, idx).expect("kp ref");
            map.add_map_point(point, CompactDescriptor([128; 256]), kp_ref)
                .expect("map point");
        }

        let correspondences = (0..world_points.len()).map(|i| (i, i)).collect::<Vec<_>>();
        (map, match_kf, query_keypoints, correspondences, intrinsics)
    }

    #[test]
    fn global_descriptor_identical_similarity_is_one() {
        let d = descriptor_with_basis(3);
        let sim = d.cosine_similarity(&d);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn keyframe_database_temporal_gap_filters_recent_frames() {
        let ids = make_keyframe_ids(5);
        let mut db = KeyframeDatabase::new(2);
        for (i, id) in ids.iter().enumerate() {
            db.insert(*id, descriptor_with_basis(i));
        }

        let matches = db.query(&descriptor_with_basis(0), 10);
        // Query is the latest keyframe; with gap=2, only the first two are eligible.
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|m| m.candidate == ids[0] || m.candidate == ids[1]));
    }

    #[test]
    fn keyframe_database_returns_top_k_by_similarity() {
        let ids = make_keyframe_ids(4);
        let mut db = KeyframeDatabase::new(0);

        let mut q = [0.0_f32; 512];
        q[0] = 1.0;
        q[1] = 1.0;
        let query = GlobalDescriptor::try_new(q).expect("valid query descriptor");

        db.insert(ids[0], descriptor_with_basis(0)); // sim ~= 0.707
        db.insert(ids[1], descriptor_with_basis(1)); // sim ~= 0.707
        db.insert(ids[2], descriptor_with_basis(2)); // sim = 0
        db.insert(ids[3], query.clone()); // query entry

        let matches = db.query(&query, 2);
        assert_eq!(matches.len(), 2);
        assert!(matches[0].similarity >= matches[1].similarity);
        assert!(matches.iter().all(|m| m.candidate != ids[2]));
    }

    #[test]
    fn keyframe_database_remove_deletes_entry() {
        let ids = make_keyframe_ids(3);
        let mut db = KeyframeDatabase::new(0);
        db.insert(ids[0], descriptor_with_basis(0));
        db.insert(ids[1], descriptor_with_basis(1));
        db.insert(ids[2], descriptor_with_basis(2));
        assert_eq!(db.len(), 3);
        assert!(db.remove(ids[1]));
        assert_eq!(db.len(), 2);
        assert!(!db.remove(ids[1]));
    }

    #[test]
    fn keyframe_database_temporal_gap_uses_sequence_after_removal() {
        let ids = make_keyframe_ids(5);
        let mut db = KeyframeDatabase::new(2);
        for (i, id) in ids.iter().enumerate() {
            db.insert(*id, descriptor_with_basis(i));
        }

        // Remove a middle entry; sequence distance must still be respected.
        assert!(db.remove(ids[2]));

        let matches = db.query(&descriptor_with_basis(4), 10);
        // kf3 is seq distance 1 (filtered), kf1 is distance 3 (kept), kf0 is distance 4 (kept).
        assert_eq!(matches.len(), 2);
        assert!(matches.iter().all(|m| m.candidate == ids[0] || m.candidate == ids[1]));
    }

    #[test]
    fn keyframe_database_replace_descriptor_updates_source() {
        let ids = make_keyframe_ids(1);
        let mut db = KeyframeDatabase::new(0);
        db.insert_with_source(ids[0], descriptor_with_basis(0), DescriptorSource::Bootstrap);
        assert_eq!(
            db.descriptor_source(ids[0]),
            Some(DescriptorSource::Bootstrap)
        );

        assert!(db.replace_descriptor(
            ids[0],
            descriptor_with_basis(1),
            DescriptorSource::Learned
        ));
        assert_eq!(db.descriptor_source(ids[0]), Some(DescriptorSource::Learned));
        assert!(!db.replace_descriptor(
            make_keyframe_ids(2)[1],
            descriptor_with_basis(2),
            DescriptorSource::Learned
        ));
    }

    #[test]
    fn loop_candidate_verify_succeeds_on_synthetic_geometry() {
        let (map, match_kf, query_keypoints, correspondences, intrinsics) = make_loop_fixture();
        let candidate = LoopCandidate {
            query_kf: match_kf,
            match_kf,
            similarity: 0.95,
        };
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
        assert!(verified.inlier_count() >= 4);
    }

    #[test]
    fn loop_candidate_verify_rejects_insufficient_inliers() {
        let (map, match_kf, query_keypoints, correspondences, intrinsics) = make_loop_fixture();
        let candidate = LoopCandidate {
            query_kf: match_kf,
            match_kf,
            similarity: 0.95,
        };
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
    fn loop_candidate_verify_propagates_pnp_failure() {
        let ids = make_keyframe_ids(1);
        let candidate = LoopCandidate {
            query_kf: ids[0],
            match_kf: ids[0],
            similarity: 0.5,
        };
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
            LoopVerificationError::PnpFailed(_) | LoopVerificationError::TooFewMatches { .. }
        ));
    }

    #[test]
    fn aggregate_empty_descriptors_returns_zero() {
        let err = aggregate_global_descriptor(&[]).expect_err("empty descriptor set should fail");
        assert!(matches!(err, GlobalDescriptorError::EmptyInput));
    }

    #[test]
    fn aggregate_single_descriptor_produces_unit_norm() {
        let mut data = [0.0_f32; 256];
        data[4] = 1.0;
        data[17] = 0.5;
        let descriptor =
            aggregate_global_descriptor(&[Descriptor(data)]).expect("aggregated descriptor");
        let values = descriptor.as_array();
        let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "descriptor norm should be 1, got {norm}");
        assert!(values[4] > 0.0);
        assert!(values[256 + 4] > 0.0);
    }

    #[test]
    fn match_descriptors_finds_mutual_matches() {
        let mut map = SlamMap::new();
        let keypoints = vec![Keypoint { x: 20.0, y: 20.0 }, Keypoint { x: 40.0, y: 20.0 }];
        let image_size = ImageSize::try_new(80, 60).expect("image size");
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
        let query = vec![Descriptor(q0), Descriptor(q1)];

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

        let matches = match_descriptors_for_loop(&query, kf, &map, 0.95);
        assert_eq!(matches.len(), 2);
        assert!(matches.contains(&(0, 0)));
        assert!(matches.contains(&(1, 1)));
    }

    #[test]
    fn match_descriptors_skips_keypoints_without_map_points() {
        let mut map = SlamMap::new();
        let keypoints = vec![
            Keypoint { x: 20.0, y: 20.0 },
            Keypoint { x: 40.0, y: 20.0 },
            Keypoint { x: 60.0, y: 20.0 },
        ];
        let image_size = ImageSize::try_new(80, 60).expect("image size");
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
        let query = vec![Descriptor(q0), Descriptor(q1), Descriptor(q2)];

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

        let matches = match_descriptors_for_loop(&query, kf, &map, 0.95);
        assert_eq!(matches, vec![(1, 1)]);
    }

    #[test]
    fn loop_closure_config_default_values() {
        let cfg = LoopClosureConfig::default();
        assert!((cfg.similarity_threshold() - 0.75).abs() < 1e-6);
        assert!((cfg.descriptor_match_threshold() - 0.7).abs() < 1e-6);
        assert_eq!(cfg.min_inliers(), 20);
        assert_eq!(cfg.max_candidates(), 3);
        assert_eq!(cfg.temporal_gap(), 30);
        assert_eq!(cfg.min_streak(), 3);
        assert!((cfg.max_correction_translation() - 5.0).abs() < 1e-6);
        assert!((cfg.max_correction_rotation_deg() - 30.0).abs() < 1e-6);
    }
}
