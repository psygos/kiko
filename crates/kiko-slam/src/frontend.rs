use std::cmp::Ordering;
use std::sync::Arc;

use crate::inference::InferenceError;
use crate::map::{KeyframeId, SlamMap};
use crate::{
    Detections, DownscaleFactor, Frame, Keyframe, LightGlue, Matches, Observation,
    PinholeIntrinsics, PnpError, PnpResult, RansacConfig, Raw, SuperPoint, TriangulationResult,
    Triangulator, Verified, solve_pnp_ransac,
};

pub(crate) struct StereoFrontend {
    superpoint: SuperPoint,
    /// Optional second SuperPoint session for background prefetch
    prefetch_sp: Option<SuperPoint>,
    lightglue: LightGlue,
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
            triangulator,
            intrinsics,
        }
    }

    pub(crate) fn set_prefetch_sp(&mut self, sp: SuperPoint) {
        self.prefetch_sp = Some(sp);
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
            .detect_with_downscale(frame, downscale)?
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
        keyframe_id: KeyframeId,
        matches: &Matches<Verified>,
        current: &Detections,
    ) -> Result<TrackedMapObservations, crate::PnpError> {
        build_map_observations(map, keyframe_id, matches, current, self.intrinsics)
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
pub(crate) struct TrackedMapObservations {
    pub(crate) observations: Vec<crate::Observation>,
    pub(crate) verified_match_indices: Vec<usize>,
}

impl TrackedMapObservations {
    pub(crate) fn len(&self) -> usize {
        self.observations.len()
    }
}

pub(crate) fn build_map_observations(
    map: &SlamMap,
    keyframe_id: KeyframeId,
    matches: &Matches<Verified>,
    current: &Detections,
    intrinsics: PinholeIntrinsics,
) -> Result<TrackedMapObservations, crate::PnpError> {
    const MIN_PNP_CORRESPONDENCES: usize = 4;

    let mut observations = Vec::with_capacity(matches.len());
    let mut verified_match_indices = Vec::with_capacity(matches.len());
    let current_len = current.len();

    for (verified_match_idx, &(ci, ki)) in matches.indices().iter().enumerate() {
        if ci >= current_len {
            return Err(crate::PnpError::IndexOutOfBounds {
                current_len,
                keyframe_len: 0,
                current_index: ci,
                keyframe_index: ki,
            });
        }
        let keypoint_ref = match map.keyframe_keypoint(keyframe_id, ki) {
            Ok(kp) => kp,
            Err(_) => continue,
        };
        let Some(point_id) = map.map_point_for_keypoint(keypoint_ref).ok().flatten() else {
            continue;
        };
        let Some(point) = map.point(point_id) else {
            continue;
        };
        let pixel = current.keypoints()[ci];
        let obs = crate::Observation::try_new(point.position(), pixel, intrinsics)?;
        observations.push(obs);
        verified_match_indices.push(verified_match_idx);
    }

    if observations.len() < MIN_PNP_CORRESPONDENCES {
        return Err(crate::PnpError::NotEnoughPoints {
            required: MIN_PNP_CORRESPONDENCES,
            actual: observations.len(),
        });
    }
    Ok(TrackedMapObservations {
        observations,
        verified_match_indices,
    })
}

pub(crate) fn median_parallax_px(
    matches: &Matches<Verified>,
    verified_match_indices: &[usize],
    inliers: &[usize],
    keyframe: &Keyframe,
) -> Option<f32> {
    if inliers.is_empty() {
        return None;
    }

    let left_kps = matches.source_a().keypoints();
    let key_kps = keyframe.detections().keypoints();
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
