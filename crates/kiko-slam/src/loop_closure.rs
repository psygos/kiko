use crate::map::KeyframeId;
use crate::{solve_pnp_ransac, Keypoint, Observation, PinholeIntrinsics, PnpError, Pose, RansacConfig};
use crate::map::SlamMap;

#[derive(Clone, Debug)]
pub struct GlobalDescriptor(pub [f32; 512]);

impl GlobalDescriptor {
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let mut dot = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;
        for i in 0..512 {
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
}

#[derive(Clone, Debug)]
pub struct PlaceMatch {
    pub query: KeyframeId,
    pub candidate: KeyframeId,
    pub similarity: f32,
}

#[derive(Clone, Debug)]
pub struct KeyframeDatabase {
    entries: Vec<(KeyframeId, GlobalDescriptor)>,
    temporal_gap: usize,
}

impl KeyframeDatabase {
    pub fn new(temporal_gap: usize) -> Self {
        Self {
            entries: Vec::new(),
            temporal_gap,
        }
    }

    pub fn temporal_gap(&self) -> usize {
        self.temporal_gap
    }

    pub fn insert(&mut self, id: KeyframeId, descriptor: GlobalDescriptor) {
        self.entries.push((id, descriptor));
    }

    pub fn query(&self, descriptor: &GlobalDescriptor, top_k: usize) -> Vec<PlaceMatch> {
        if top_k == 0 || self.entries.is_empty() {
            return Vec::new();
        }
        let query_idx = self.entries.len() - 1;
        let query_id = self.entries[query_idx].0;
        let mut matches = Vec::new();
        for (idx, (candidate, candidate_desc)) in self.entries.iter().enumerate() {
            if idx == query_idx {
                continue;
            }
            if query_idx.saturating_sub(idx) <= self.temporal_gap {
                continue;
            }
            matches.push(PlaceMatch {
                query: query_id,
                candidate: *candidate,
                similarity: descriptor.cosine_similarity(candidate_desc),
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
        GlobalDescriptor, KeyframeDatabase, LoopCandidate, LoopVerificationError,
    };
    use crate::map::{ImageSize, KeyframeId, SlamMap};
    use crate::test_helpers::{make_pinhole_intrinsics, project_world_point};
    use crate::{CompactDescriptor, FrameId, Keypoint, Point3, Pose, RansacConfig, Timestamp};

    fn descriptor_with_basis(idx: usize) -> GlobalDescriptor {
        let mut d = [0.0_f32; 512];
        d[idx] = 1.0;
        GlobalDescriptor(d)
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
        let query = GlobalDescriptor(q);

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
}
