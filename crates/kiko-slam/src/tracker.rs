use std::sync::Arc;

use crate::{
    build_observations, solve_pnp_ransac, DownscaleFactor, Frame, FrameId, Keyframe, KeypointLimit,
    LightGlue, LocalBaConfig, LocalBundleAdjuster, Matches, MapObservation, ObservationSet,
    PinholeIntrinsics, Pose, Raw, RansacConfig, RectifiedStereo, StereoPair, SuperPoint, Timestamp,
    TriangulationConfig, TriangulationError, Triangulator, Verified,
    map::{KeyframeId, SlamMap},
};

use std::cmp::Ordering;
use std::num::NonZeroUsize;
use crate::inference::InferenceError;

#[derive(Clone, Copy, Debug)]
pub struct TrackerConfig {
    pub max_keypoints: KeypointLimit,
    pub downscale: DownscaleFactor,
    pub min_keyframe_points: usize,
    pub ransac: RansacConfig,
    pub triangulation: TriangulationConfig,
    pub keyframe_policy: KeyframePolicy,
    pub ba: LocalBaConfig,
}

impl TrackerConfig {
    pub fn max_keypoints(&self) -> usize {
        self.max_keypoints.get()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct KeyframePolicy {
    min_inliers: NonZeroUsize,
    parallax_px: ParallaxPx,
    min_covisibility: CovisibilityRatio,
}

#[derive(Clone, Copy, Debug)]
pub struct ParallaxPx(f32);

#[derive(Clone, Copy, Debug)]
pub struct CovisibilityRatio(f32);

#[derive(Debug)]
pub enum KeyframePolicyError {
    ZeroInliers,
    NonPositiveParallax { value: f32 },
    CovisibilityOutOfRange { value: f32 },
}

impl std::fmt::Display for KeyframePolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeyframePolicyError::ZeroInliers => write!(f, "keyframe inlier threshold must be > 0"),
            KeyframePolicyError::NonPositiveParallax { value } => {
                write!(f, "parallax threshold must be > 0 (got {value})")
            }
            KeyframePolicyError::CovisibilityOutOfRange { value } => write!(
                f,
                "covisibility ratio must be within [0, 1] (got {value})"
            ),
        }
    }
}

impl std::error::Error for KeyframePolicyError {}

impl KeyframePolicy {
    pub fn new(
        min_inliers: usize,
        parallax_px: f32,
        min_covisibility: f32,
    ) -> Result<Self, KeyframePolicyError> {
        let min_inliers = NonZeroUsize::new(min_inliers).ok_or(KeyframePolicyError::ZeroInliers)?;
        if !parallax_px.is_finite() || parallax_px <= 0.0 {
            return Err(KeyframePolicyError::NonPositiveParallax { value: parallax_px });
        }
        if !min_covisibility.is_finite() || min_covisibility < 0.0 || min_covisibility > 1.0 {
            return Err(KeyframePolicyError::CovisibilityOutOfRange {
                value: min_covisibility,
            });
        }
        Ok(Self {
            min_inliers,
            parallax_px: ParallaxPx(parallax_px),
            min_covisibility: CovisibilityRatio(min_covisibility),
        })
    }

    pub fn min_inliers(&self) -> usize {
        self.min_inliers.get()
    }

    pub fn parallax_px(&self) -> f32 {
        self.parallax_px.0
    }

    pub fn min_covisibility(&self) -> f32 {
        self.min_covisibility.0
    }

    pub fn should_refresh(
        &self,
        inliers: usize,
        parallax_px: Option<f32>,
        covisibility: f32,
    ) -> bool {
        if inliers < self.min_inliers.get() {
            return true;
        }
        if let Some(parallax) = parallax_px {
            if parallax > self.parallax_px.0 {
                return true;
            }
        }
        if covisibility < self.min_covisibility.0 {
            return true;
        }
        false
    }
}

#[derive(Debug)]
pub enum TrackerError {
    Inference(InferenceError),
    Triangulation(TriangulationError),
    Pnp(crate::PnpError),
    Map(crate::map::MapError),
    KeyframeRejected { landmarks: usize },
}

impl std::fmt::Display for TrackerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackerError::Inference(err) => write!(f, "inference error: {err}"),
            TrackerError::Triangulation(err) => write!(f, "triangulation error: {err}"),
            TrackerError::Pnp(err) => write!(f, "pnp error: {err}"),
            TrackerError::Map(err) => write!(f, "map error: {err}"),
            TrackerError::KeyframeRejected { landmarks } => {
                write!(f, "keyframe rejected: only {landmarks} landmarks")
            }
        }
    }
}

impl std::error::Error for TrackerError {}

impl From<InferenceError> for TrackerError {
    fn from(err: InferenceError) -> Self {
        TrackerError::Inference(err)
    }
}

impl From<TriangulationError> for TrackerError {
    fn from(err: TriangulationError) -> Self {
        TrackerError::Triangulation(err)
    }
}

impl From<crate::PnpError> for TrackerError {
    fn from(err: crate::PnpError) -> Self {
        TrackerError::Pnp(err)
    }
}

impl From<crate::map::MapError> for TrackerError {
    fn from(err: crate::map::MapError) -> Self {
        TrackerError::Map(err)
    }
}

#[derive(Debug)]
pub struct TrackerOutput {
    pub pose: Option<Pose>,
    pub inliers: usize,
    pub keyframe: Option<Arc<Keyframe>>,
    pub stereo_matches: Option<Matches<Raw>>,
    pub frame_id: FrameId,
}

#[derive(Debug)]
enum TrackerState {
    NeedKeyframe,
    Tracking {
        keyframe: Arc<Keyframe>,
        keyframe_pose: Pose,
        keyframe_id: KeyframeId,
    },
}

pub struct SlamTracker {
    superpoint_left: SuperPoint,
    superpoint_right: SuperPoint,
    lightglue: LightGlue,
    triangulator: Triangulator,
    intrinsics: PinholeIntrinsics,
    config: TrackerConfig,
    state: TrackerState,
    ba: LocalBundleAdjuster,
    map: SlamMap,
}

impl SlamTracker {
    pub fn new(
        superpoint_left: SuperPoint,
        superpoint_right: SuperPoint,
        lightglue: LightGlue,
        stereo: RectifiedStereo,
        intrinsics: PinholeIntrinsics,
        config: TrackerConfig,
    ) -> Self {
        let triangulator = Triangulator::new(stereo, config.triangulation);
        let ba = LocalBundleAdjuster::new(intrinsics, config.ba);
        Self {
            superpoint_left,
            superpoint_right,
            lightglue,
            triangulator,
            intrinsics,
            config,
            state: TrackerState::NeedKeyframe,
            ba,
            map: SlamMap::new(),
        }
    }

    pub fn process(&mut self, pair: StereoPair) -> Result<TrackerOutput, TrackerError> {
        let tracking = match &self.state {
            TrackerState::NeedKeyframe => None,
            TrackerState::Tracking {
                keyframe,
                keyframe_pose,
                keyframe_id,
            } => Some((Arc::clone(keyframe), *keyframe_pose, *keyframe_id)),
        };

        if let Some((keyframe, keyframe_pose, keyframe_id)) = tracking {
            self.track(pair, &keyframe, keyframe_pose, keyframe_id)
        } else {
            self.create_keyframe(pair, Pose::identity())
        }
    }

    fn track(
        &mut self,
        pair: StereoPair,
        keyframe: &Arc<Keyframe>,
        keyframe_pose: Pose,
        keyframe_id: KeyframeId,
    ) -> Result<TrackerOutput, TrackerError> {
        let StereoPair { left, right } = pair;
        let frame_id = left.frame_id();

        let current = self
            .superpoint_left
            .detect_with_downscale(&left, self.config.downscale)?
            .top_k(self.config.max_keypoints());
        let current = Arc::new(current);

        let matches = if current.is_empty() || keyframe.detections().is_empty() {
            return Ok(TrackerOutput {
                pose: None,
                inliers: 0,
                keyframe: None,
                stereo_matches: None,
                frame_id,
            });
        } else {
            self.lightglue
                .match_these(current.clone(), keyframe.detections().clone())?
        };

        let verified = match matches.with_landmarks(keyframe) {
            Ok(verified) => verified,
            Err(err) => return Err(TrackerError::Inference(InferenceError::Domain(format!("{err:?}")))),
        };

        let observations = match build_observations(keyframe, &verified, self.intrinsics) {
            Ok(obs) => obs,
            Err(crate::PnpError::NotEnoughPoints { .. }) => {
                return Ok(TrackerOutput {
                    pose: None,
                    inliers: 0,
                    keyframe: None,
                    stereo_matches: None,
                    frame_id,
                })
            }
            Err(err) => return Err(TrackerError::Pnp(err)),
        };

        let result = match solve_pnp_ransac(&observations, self.intrinsics, self.config.ransac) {
            Ok(result) => result,
            Err(crate::PnpError::NotEnoughPoints { .. } | crate::PnpError::NoSolution) => {
                return Ok(TrackerOutput {
                    pose: None,
                    inliers: 0,
                    keyframe: None,
                    stereo_matches: None,
                    frame_id,
                })
            }
            Err(err) => return Err(TrackerError::Pnp(err)),
        };

        let mut map_observations = Vec::with_capacity(result.inliers.len());
        for &idx in &result.inliers {
            let (ci, ki) = *verified
                .indices()
                .get(idx)
                .ok_or_else(|| {
                    TrackerError::Inference(InferenceError::Domain(
                        "verified match index out of bounds".to_string(),
                    ))
                })?;
            let pixel = *current.keypoints().get(ci).ok_or_else(|| {
                TrackerError::Inference(InferenceError::Domain(
                    "current keypoint index out of bounds".to_string(),
                ))
            })?;
            let keypoint_ref = self.map.keyframe_keypoint(keyframe_id, ki)?;
            map_observations.push(MapObservation::new(keypoint_ref, pixel));
        }

        let parallax_px = median_parallax_px(&verified, &result.inliers, keyframe);
        let covisibility = if keyframe.landmarks().is_empty() {
            0.0
        } else {
            result.inliers.len() as f32 / keyframe.landmarks().len() as f32
        };

        let refined_rel = ObservationSet::new(map_observations, self.ba.min_observations())
            .ok()
            .and_then(|set| self.ba.push_frame(&self.map, result.pose, set));

        let pose_rel = refined_rel.unwrap_or(result.pose);
        let pose_world = keyframe_pose.compose(pose_rel);

        let mut output = TrackerOutput {
            pose: Some(pose_world),
            inliers: result.inliers.len(),
            keyframe: None,
            stereo_matches: None,
            frame_id,
        };

        let should_refresh = self
            .config
            .keyframe_policy
            .should_refresh(result.inliers.len(), parallax_px, covisibility);

        if should_refresh {
            let new_pose = pose_world;
            if let Ok((keyframe_output, keyframe_id)) =
                self.create_keyframe_internal(left, right, new_pose)
            {
                if let Some(keyframe) = keyframe_output.keyframe {
                    self.state = TrackerState::Tracking {
                        keyframe: keyframe.clone(),
                        keyframe_pose: new_pose,
                        keyframe_id,
                    };
                    self.ba.reset();
                    output.keyframe = Some(keyframe);
                    output.stereo_matches = keyframe_output.stereo_matches;
                }
            }
        }

        Ok(output)
    }

    fn create_keyframe(
        &mut self,
        pair: StereoPair,
        pose_world: Pose,
    ) -> Result<TrackerOutput, TrackerError> {
        let StereoPair { left, right } = pair;
        let frame_id = left.frame_id();
        let (output, keyframe_id) = self.create_keyframe_internal(left, right, pose_world)?;
        let keyframe = output.keyframe.clone().expect("keyframe");
        self.state = TrackerState::Tracking {
            keyframe,
            keyframe_pose: pose_world,
            keyframe_id,
        };
        self.ba.reset();
        Ok(TrackerOutput {
            pose: Some(pose_world),
            inliers: 0,
            keyframe: output.keyframe,
            stereo_matches: output.stereo_matches,
            frame_id,
        })
    }

    fn create_keyframe_internal(
        &mut self,
        left: Frame,
        right: Frame,
        pose_world: Pose,
    ) -> Result<(TrackerOutput, KeyframeId), TrackerError> {
        let frame_id = left.frame_id();
        let max_keypoints = self.config.max_keypoints();

        let (left_det, right_det) = std::thread::scope(|scope| {
            let left_sp = &mut self.superpoint_left;
            let right_sp = &mut self.superpoint_right;
            let left_ref = &left;
            let right_ref = &right;
            let downscale = self.config.downscale;

            let left_handle = scope.spawn(move || {
                left_sp
                    .detect_with_downscale(left_ref, downscale)
                    .map(|d| d.top_k(max_keypoints))
            });
            let right_handle = scope.spawn(move || {
                right_sp
                    .detect_with_downscale(right_ref, downscale)
                    .map(|d| d.top_k(max_keypoints))
            });

            (left_handle.join(), right_handle.join())
        });

        let left_det = left_det
            .map_err(|_| InferenceError::Domain("left superpoint thread panicked".to_string()))??;
        let right_det = right_det
            .map_err(|_| InferenceError::Domain("right superpoint thread panicked".to_string()))??;

        let left_arc = Arc::new(left_det);
        let right_arc = Arc::new(right_det);

        let matches = if left_arc.is_empty() || right_arc.is_empty() {
            return Err(TrackerError::KeyframeRejected { landmarks: 0 });
        } else {
            self.lightglue
                .match_these(left_arc.clone(), right_arc.clone())?
        };

        let result = self.triangulator.triangulate(&matches)?;
        let landmarks = result.keyframe.landmarks().len();
        if landmarks < self.config.min_keyframe_points {
            return Err(TrackerError::KeyframeRejected { landmarks });
        }

        let keyframe = Arc::new(result.keyframe);
        let keyframe_id =
            insert_keyframe_into_map(&mut self.map, &keyframe, left.timestamp(), pose_world)?;

        Ok((
            TrackerOutput {
                pose: None,
                inliers: 0,
                keyframe: Some(keyframe),
                stereo_matches: Some(matches),
                frame_id,
            },
            keyframe_id,
        ))
    }
}

fn insert_keyframe_into_map(
    map: &mut SlamMap,
    keyframe: &Arc<Keyframe>,
    timestamp: Timestamp,
    pose_world: Pose,
) -> Result<KeyframeId, TrackerError> {
    let keyframe_id = map.add_keyframe_from_detections(
        keyframe.detections().as_ref(),
        timestamp,
        pose_world,
    )?;

    for (landmark, &det_idx) in keyframe
        .landmarks()
        .iter()
        .zip(keyframe.landmark_indices().iter())
    {
        let keypoint_ref = map.keyframe_keypoint(keyframe_id, det_idx)?;
        let descriptor = keyframe.detections().descriptors()[det_idx];
        map.add_map_point(*landmark, descriptor, keypoint_ref)?;
    }
    Ok(keyframe_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Detections, Descriptor, Keypoint, SensorId, Point3, Timestamp};

    fn make_descriptor() -> Descriptor {
        Descriptor([0.0; 256])
    }

    #[test]
    fn keyframe_insertion_populates_map_points() {
        let keypoints = vec![
            Keypoint { x: 1.0, y: 2.0 },
            Keypoint { x: 3.0, y: 4.0 },
            Keypoint { x: 5.0, y: 6.0 },
        ];
        let scores = vec![1.0, 1.0, 1.0];
        let descriptors = vec![make_descriptor(), make_descriptor(), make_descriptor()];

        let detections = Detections::new(
            SensorId::StereoLeft,
            FrameId::new(10),
            640,
            480,
            keypoints,
            scores,
            descriptors,
        )
        .expect("detections");

        let landmarks = vec![
            Point3 { x: 0.0, y: 0.0, z: 1.0 },
            Point3 { x: 1.0, y: 0.0, z: 1.5 },
        ];
        let landmark_indices = vec![0, 2];
        let keyframe = Arc::new(
            Keyframe::from_arc(Arc::new(detections), landmarks, landmark_indices)
                .expect("keyframe"),
        );

        let mut map = SlamMap::new();
        let keyframe_id = insert_keyframe_into_map(
            &mut map,
            &keyframe,
            Timestamp::from_nanos(42),
            Pose::identity(),
        )
        .expect("insert keyframe");

        assert_eq!(map.num_keyframes(), 1);
        assert_eq!(map.num_points(), keyframe.landmarks().len());

        for &det_idx in keyframe.landmark_indices() {
            let kp_ref = map.keyframe_keypoint(keyframe_id, det_idx).expect("kp ref");
            let point_id = map
                .map_point_for_keypoint(kp_ref)
                .expect("map lookup")
                .expect("point id");
            let point = map.point(point_id).expect("point");
            let landmark = keyframe
                .landmark_for_detection(det_idx)
                .expect("landmark");
            let Point3 { x, y, z } = point.position();
            assert_eq!(x, landmark.x);
            assert_eq!(y, landmark.y);
            assert_eq!(z, landmark.z);
        }
    }
}

fn median_parallax_px(
    matches: &Matches<Verified>,
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
        let Some(&(li, ki)) = matches.indices().get(idx) else {
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
