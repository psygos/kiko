use std::cmp::Ordering;
use std::sync::Arc;

use crate::inference::InferenceError;
use crate::map::SlamMap;
use crate::{
    Detections, DownscaleFactor, Frame, LightGlue, Matches, Observation, PinholeIntrinsics,
    PnpError, PnpResult, RansacConfig, Raw, SuperPoint, TriangulationResult, Triangulator,
    Verified, solve_pnp_ransac,
};

pub(crate) struct StereoFrontend {
    superpoint: SuperPoint,
    /// Optional second SuperPoint session for background prefetch
    prefetch_sp: Option<SuperPoint>,
    lightglue: LightGlue,
    /// Optional second LightGlue session for speculative match prefetch
    prefetch_lg: Option<LightGlue>,
    triangulator: Triangulator,
    intrinsics: PinholeIntrinsics,
}

impl StereoFrontend {
    pub(crate) fn new(
        superpoint: SuperPoint,
        lightglue: LightGlue,
        triangulator: Triangulator,
        intrinsics: PinholeIntrinsics,
    ) -> Self {
        Self {
            superpoint,
            prefetch_sp: None,
            lightglue,
            prefetch_lg: None,
            triangulator,
            intrinsics,
        }
    }

    pub(crate) fn intrinsics(&self) -> PinholeIntrinsics {
        self.intrinsics
    }

    /// Detect with optional prefetched result. If `prefetched` matches this frame, skip SP.
    pub(crate) fn detect_or_use_prefetched(
        &mut self,
        frame: &Frame,
        downscale: DownscaleFactor,
        max_keypoints: usize,
        prefetched: Option<(crate::FrameId, Arc<Detections>)>,
    ) -> Result<Arc<Detections>, InferenceError> {
        if let Some((fid, dets)) = prefetched {
            if fid == frame.frame_id() {
                return Ok(dets);
            }
        }
        self.detect(frame, downscale, max_keypoints)
    }

    pub(crate) fn detect(
        &mut self,
        frame: &Frame,
        downscale: DownscaleFactor,
        max_keypoints: usize,
    ) -> Result<Arc<Detections>, InferenceError> {
        let detections = self
            .superpoint
            .detect_with_downscale_limited(frame, downscale, max_keypoints)?
            .top_k(max_keypoints);
        Ok(Arc::new(detections))
    }

    /// Take the prefetch SP session out for use on a background thread.
    /// Call `return_prefetch_sp` to put it back.
    pub(crate) fn take_prefetch_sp(&mut self) -> Option<SuperPoint> {
        self.prefetch_sp.take()
    }

    pub(crate) fn return_prefetch_sp(&mut self, sp: SuperPoint) {
        self.prefetch_sp = Some(sp);
    }

    pub(crate) fn take_prefetch_lg(&mut self) -> Option<LightGlue> {
        self.prefetch_lg.take()
    }

    pub(crate) fn return_prefetch_lg(&mut self, lg: LightGlue) {
        self.prefetch_lg = Some(lg);
    }

    pub(crate) fn match_tracking(
        &mut self,
        current: Arc<Detections>,
        keyframe: Arc<Detections>,
    ) -> Result<Matches<Raw>, InferenceError> {
        self.lightglue.match_these(current, keyframe)
    }

    pub(crate) fn match_stereo(
        &mut self,
        left: Arc<Detections>,
        right: Arc<Detections>,
    ) -> Result<Matches<Raw>, InferenceError> {
        self.lightglue.match_these(left, right)
    }

    pub(crate) fn triangulate_matches(
        &self,
        matches: &Matches<Raw>,
    ) -> Result<TriangulationResult, crate::TriangulationError> {
        self.triangulator.triangulate(matches)
    }

    pub(crate) fn build_map_observations(
        &self,
        map: &SlamMap,
        matches: &Matches<Verified>,
    ) -> Result<TrackedMapObservations, MapObservationError> {
        build_map_observations(map, matches, self.intrinsics)
    }

    pub(crate) fn solve_tracking_pose(
        &self,
        observations: &[Observation],
        ransac: RansacConfig,
    ) -> Result<PnpResult, PnpError> {
        solve_pnp_ransac(observations, self.intrinsics, ransac)
    }
}

#[derive(Debug)]
pub enum MapObservationError {
    MissingMatchProvenance,
    KeyframeProvenanceMismatch {
        matches_frame: crate::FrameId,
        map_frame: crate::FrameId,
        matches_detections: usize,
        map_keypoints: usize,
    },
    Map(crate::map::MapError),
    Pnp(crate::PnpError),
    NotEnoughPoints {
        required: usize,
        actual: usize,
    },
}

impl std::fmt::Display for MapObservationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MapObservationError::MissingMatchProvenance => {
                write!(f, "verified matches are missing keyframe provenance")
            }
            MapObservationError::KeyframeProvenanceMismatch {
                matches_frame,
                map_frame,
                matches_detections,
                map_keypoints,
            } => write!(
                f,
                "verified matches reference frame {} with {matches_detections} detections, but the map keyframe is frame {} with {map_keypoints} keypoints",
                matches_frame.as_u64(),
                map_frame.as_u64()
            ),
            MapObservationError::Map(source) => {
                write!(f, "map observation lookup failed: {source}")
            }
            MapObservationError::Pnp(source) => {
                write!(f, "map observation bearing failed: {source}")
            }
            MapObservationError::NotEnoughPoints { required, actual } => write!(
                f,
                "map observation resolution requires {required} points, got {actual}"
            ),
        }
    }
}

impl std::error::Error for MapObservationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MapObservationError::Map(source) => Some(source),
            MapObservationError::Pnp(source) => Some(source),
            MapObservationError::MissingMatchProvenance
            | MapObservationError::KeyframeProvenanceMismatch { .. }
            | MapObservationError::NotEnoughPoints { .. } => None,
        }
    }
}

#[derive(Debug)]
pub(crate) struct TrackedMapObservations {
    pub(crate) observations: Vec<crate::Observation>,
    pub(crate) verified_match_indices: Vec<usize>,
    pub(crate) missing_map_point_associations: usize,
}

impl TrackedMapObservations {
    pub(crate) fn len(&self) -> usize {
        self.observations.len()
    }
}

pub(crate) fn build_map_observations(
    map: &SlamMap,
    matches: &Matches<Verified>,
    intrinsics: PinholeIntrinsics,
) -> Result<TrackedMapObservations, MapObservationError> {
    const MIN_PNP_CORRESPONDENCES: usize = 4;
    let keyframe_id = matches
        .keyframe_id()
        .ok_or(MapObservationError::MissingMatchProvenance)?;
    let map_keyframe = map.keyframe(keyframe_id).ok_or(MapObservationError::Map(
        crate::map::MapError::KeyframeNotFound(keyframe_id),
    ))?;
    let matches_frame = matches.source_b().frame_id();
    if map_keyframe.frame_id() != matches_frame || map_keyframe.len() != matches.source_b().len() {
        return Err(MapObservationError::KeyframeProvenanceMismatch {
            matches_frame,
            map_frame: map_keyframe.frame_id(),
            matches_detections: matches.source_b().len(),
            map_keypoints: map_keyframe.len(),
        });
    }

    let mut observations = Vec::with_capacity(matches.len());
    let mut verified_match_indices = Vec::with_capacity(matches.len());
    let mut missing_map_point_associations = 0;
    let current = matches.source_a();

    for (verified_match_idx, &(ci, ki)) in matches.indices().iter().enumerate() {
        let keypoint_ref = map
            .keyframe_keypoint(keyframe_id, ki)
            .map_err(MapObservationError::Map)?;
        let Some(point_id) = map
            .map_point_for_keypoint(keypoint_ref)
            .map_err(MapObservationError::Map)?
        else {
            missing_map_point_associations += 1;
            continue;
        };
        let point = map.point(point_id).ok_or(MapObservationError::Map(
            crate::map::MapError::MapPointNotFound(point_id),
        ))?;
        let pixel = current.keypoints()[ci];
        let obs = crate::Observation::try_new(point.position(), pixel, intrinsics)
            .map_err(MapObservationError::Pnp)?;
        observations.push(obs);
        verified_match_indices.push(verified_match_idx);
    }

    if observations.len() < MIN_PNP_CORRESPONDENCES {
        return Err(MapObservationError::NotEnoughPoints {
            required: MIN_PNP_CORRESPONDENCES,
            actual: observations.len(),
        });
    }
    Ok(TrackedMapObservations {
        observations,
        verified_match_indices,
        missing_map_point_associations,
    })
}

pub(crate) fn median_parallax_px(
    matches: &Matches<Verified>,
    verified_match_indices: &[usize],
    inliers: &[usize],
) -> Option<f32> {
    if inliers.is_empty() {
        return None;
    }

    let left_kps = matches.source_a().keypoints();
    let key_kps = matches.source_b().keypoints();
    let mut parallax = Vec::with_capacity(inliers.len());

    for &idx in inliers {
        let Some(&verified_idx) = verified_match_indices.get(idx) else {
            continue;
        };
        let Some(&(li, ki)) = matches.indices().get(verified_idx) else {
            continue;
        };
        let (Some(left), Some(key)) = (left_kps.get(li), key_kps.get(ki)) else {
            continue;
        };
        let dx = left.x - key.x;
        let dy = left.y - key.y;
        parallax.push((dx * dx + dy * dy).sqrt());
    }

    if parallax.is_empty() {
        return None;
    }

    parallax.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mid = parallax.len() / 2;
    let median = if parallax.len() % 2 == 0 {
        (parallax[mid - 1] + parallax[mid]) * 0.5
    } else {
        parallax[mid]
    };

    Some(median)
}
