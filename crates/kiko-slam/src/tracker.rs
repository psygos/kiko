use std::sync::Arc;

use crate::{
    solve_pnp_ransac, DownscaleFactor, Detections, Frame, FrameId, Keyframe, KeypointLimit,
    LightGlue, LocalBaConfig, LocalBundleAdjuster, Matches, MapObservation, ObservationSet,
    PinholeIntrinsics, Point3, Pose, Raw, RansacConfig, RectifiedStereo, StereoPair, SuperPoint, Timestamp,
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
    pub redundancy: Option<RedundancyPolicy>,
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

#[derive(Clone, Copy, Debug)]
pub struct RedundancyPolicy {
    max_covisibility: CovisibilityRatio,
}

#[derive(Debug)]
pub enum KeyframePolicyError {
    ZeroInliers,
    NonPositiveParallax { value: f32 },
    CovisibilityOutOfRange { value: f32 },
}

#[derive(Debug)]
pub enum RedundancyPolicyError {
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

impl std::fmt::Display for RedundancyPolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RedundancyPolicyError::CovisibilityOutOfRange { value } => write!(
                f,
                "redundancy covisibility must be within [0, 1] (got {value})"
            ),
        }
    }
}

impl std::error::Error for RedundancyPolicyError {}

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

impl RedundancyPolicy {
    pub fn new(max_covisibility: f32) -> Result<Self, RedundancyPolicyError> {
        if !max_covisibility.is_finite() || max_covisibility < 0.0 || max_covisibility > 1.0 {
            return Err(RedundancyPolicyError::CovisibilityOutOfRange {
                value: max_covisibility,
            });
        }
        Ok(Self {
            max_covisibility: CovisibilityRatio(max_covisibility),
        })
    }

    pub fn max_covisibility(&self) -> f32 {
        self.max_covisibility.0
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackingHealth {
    Good,
    Degraded,
    Lost,
}

#[derive(Debug)]
pub struct TrackerOutput {
    pub pose: Option<Pose>,
    pub inliers: usize,
    pub keyframe: Option<Arc<Keyframe>>,
    pub stereo_matches: Option<Matches<Raw>>,
    pub frame_id: FrameId,
    pub health: TrackingHealth,
}

#[derive(Debug)]
enum TrackerState {
    NeedKeyframe,
    Tracking {
        keyframe: Arc<Keyframe>,
        keyframe_id: KeyframeId,
    },
}

#[derive(Debug)]
struct SharedMatches {
    keyframe_id: KeyframeId,
    pairs: Vec<(usize, usize)>,
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
    consecutive_tracking_failures: usize,
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
            consecutive_tracking_failures: 0,
        }
    }

    pub fn process(&mut self, pair: StereoPair) -> Result<TrackerOutput, TrackerError> {
        let tracking = match &self.state {
            TrackerState::NeedKeyframe => None,
            TrackerState::Tracking {
                keyframe,
                keyframe_id,
            } => Some((Arc::clone(keyframe), *keyframe_id)),
        };

        if let Some((keyframe, keyframe_id)) = tracking {
            self.track(pair, &keyframe, keyframe_id)
        } else {
            self.create_keyframe(pair, Pose::identity())
        }
    }

    pub fn covisibility_snapshot(&self) -> crate::map::CovisibilitySnapshot {
        self.map.covisibility_snapshot()
    }

    fn tracking_failure_health(&mut self) -> TrackingHealth {
        const LOST_AFTER_CONSECUTIVE_FAILURES: usize = 3;
        self.consecutive_tracking_failures = self.consecutive_tracking_failures.saturating_add(1);
        if self.consecutive_tracking_failures >= LOST_AFTER_CONSECUTIVE_FAILURES {
            TrackingHealth::Lost
        } else {
            TrackingHealth::Degraded
        }
    }

    fn track(
        &mut self,
        pair: StereoPair,
        keyframe: &Arc<Keyframe>,
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
                health: self.tracking_failure_health(),
            });
        } else {
            self.lightglue
                .match_these(current.clone(), keyframe.detections().clone())?
        };

        let verified = match matches.with_landmarks(keyframe) {
            Ok(verified) => verified,
            Err(err) => return Err(TrackerError::Inference(InferenceError::Domain(format!("{err:?}")))),
        };

        let observations = match build_map_observations(
            &self.map,
            keyframe_id,
            &verified,
            current.as_ref(),
            self.intrinsics,
        ) {
            Ok(obs) => obs,
            Err(crate::PnpError::NotEnoughPoints { .. }) => {
                return Ok(TrackerOutput {
                    pose: None,
                    inliers: 0,
                    keyframe: None,
                    stereo_matches: None,
                    frame_id,
                    health: self.tracking_failure_health(),
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
                    health: self.tracking_failure_health(),
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

        let pose_world = result.pose;
        let refined_world = ObservationSet::new(map_observations, self.ba.min_observations())
            .ok()
            .and_then(|set| self.ba.push_frame(&self.map, pose_world, set));

        let pose_world = refined_world.unwrap_or(pose_world);
        self.consecutive_tracking_failures = 0;

        let mut output = TrackerOutput {
            pose: Some(pose_world),
            inliers: result.inliers.len(),
            keyframe: None,
            stereo_matches: None,
            frame_id,
            health: TrackingHealth::Good,
        };

        let should_refresh = self
            .config
            .keyframe_policy
            .should_refresh(result.inliers.len(), parallax_px, covisibility);

        if should_refresh {
            let new_pose = pose_world;
            let shared = build_shared_matches(keyframe_id, &verified, &result.inliers);
            if let Ok((keyframe_output, keyframe_id)) = self.create_keyframe_internal(
                left,
                right,
                new_pose,
                Some(current.clone()),
                Some(shared),
            ) {
                if let Some(keyframe) = keyframe_output.keyframe {
                    let redundant = self
                        .config
                        .redundancy
                        .map(|policy| is_redundant(&self.map, keyframe_id, policy.max_covisibility()))
                        .transpose()?
                        .unwrap_or(false);
                    if redundant {
                        let _ = self.map.remove_keyframe(keyframe_id);
                    } else {
                        let window = self
                            .map
                            .covisible_window(keyframe_id, self.ba.window_size())?;
                        let _ = self.ba.optimize_keyframe_window(&mut self.map, &window);
                        self.state = TrackerState::Tracking {
                            keyframe: keyframe.clone(),
                            keyframe_id,
                        };
                        self.ba.reset();
                        output.keyframe = Some(keyframe);
                        output.stereo_matches = keyframe_output.stereo_matches;
                    }
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
        let (output, keyframe_id) =
            self.create_keyframe_internal(left, right, pose_world, None, None)?;
        let keyframe = output.keyframe.clone().expect("keyframe");
        self.state = TrackerState::Tracking {
            keyframe,
            keyframe_id,
        };
        self.ba.reset();
        self.consecutive_tracking_failures = 0;
        Ok(TrackerOutput {
            pose: Some(pose_world),
            inliers: 0,
            keyframe: output.keyframe,
            stereo_matches: output.stereo_matches,
            frame_id,
            health: TrackingHealth::Good,
        })
    }

fn create_keyframe_internal(
    &mut self,
    left: Frame,
    right: Frame,
    pose_world: Pose,
    left_det: Option<Arc<Detections>>,
    shared: Option<SharedMatches>,
) -> Result<(TrackerOutput, KeyframeId), TrackerError> {
        let frame_id = left.frame_id();
        let max_keypoints = self.config.max_keypoints();

        let (left_arc, right_arc) = match left_det {
            Some(left_arc) => {
                let right_det = self
                    .superpoint_right
                    .detect_with_downscale(&right, self.config.downscale)?
                    .top_k(max_keypoints);
                (left_arc, Arc::new(right_det))
            }
            None => {
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

                let left_det = left_det.map_err(|_| {
                    InferenceError::Domain("left superpoint thread panicked".to_string())
                })??;
                let right_det = right_det.map_err(|_| {
                    InferenceError::Domain("right superpoint thread panicked".to_string())
                })??;

                (Arc::new(left_det), Arc::new(right_det))
            }
        };

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
        let keyframe_id = insert_keyframe_into_map(
            &mut self.map,
            &keyframe,
            left.timestamp(),
            pose_world,
            shared.as_ref(),
        )?;

        Ok((
            TrackerOutput {
                pose: None,
                inliers: 0,
                keyframe: Some(keyframe),
                stereo_matches: Some(matches),
                frame_id,
                health: TrackingHealth::Good,
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
    shared: Option<&SharedMatches>,
) -> Result<KeyframeId, TrackerError> {
    let keyframe_id = map.add_keyframe_from_detections(
        keyframe.detections().as_ref(),
        timestamp,
        pose_world,
    )?;

    if let Some(shared) = shared {
        for &(current_idx, old_idx) in &shared.pairs {
            let old_kp = map.keyframe_keypoint(shared.keyframe_id, old_idx)?;
            let Some(point_id) = map.map_point_for_keypoint(old_kp)? else {
                continue;
            };
            let new_kp = map.keyframe_keypoint(keyframe_id, current_idx)?;
            if map.map_point_for_keypoint(new_kp)?.is_none() {
                map.add_observation(point_id, new_kp)?;
            }
        }
    }

    // Cull stale singleton landmarks from previous keyframes before adding
    // new landmarks from this keyframe.
    if map.num_points() > 0 {
        let _ = map.cull_points(2);
    }

    for (landmark, &det_idx) in keyframe
        .landmarks()
        .iter()
        .zip(keyframe.landmark_indices().iter())
    {
        let keypoint_ref = map.keyframe_keypoint(keyframe_id, det_idx)?;
        if map.map_point_for_keypoint(keypoint_ref)?.is_some() {
            continue;
        }
        let descriptor = keyframe.detections().descriptors()[det_idx];
        let world = camera_to_world(pose_world, *landmark);
        map.add_map_point(world, descriptor, keypoint_ref)?;
    }
    Ok(keyframe_id)
}

fn camera_to_world(pose_world: Pose, point: Point3) -> Point3 {
    let inv = pose_world.inverse();
    let v = crate::math::transform_point(
        inv.rotation(),
        inv.translation(),
        [point.x, point.y, point.z],
    );
    Point3 {
        x: v[0],
        y: v[1],
        z: v[2],
    }
}

fn build_shared_matches(
    keyframe_id: KeyframeId,
    matches: &Matches<Verified>,
    inliers: &[usize],
) -> SharedMatches {
    let mut pairs = Vec::with_capacity(inliers.len());
    for &idx in inliers {
        if let Some(&(ci, ki)) = matches.indices().get(idx) {
            pairs.push((ci, ki));
        }
    }
    SharedMatches { keyframe_id, pairs }
}

fn is_redundant(
    map: &SlamMap,
    keyframe_id: KeyframeId,
    max_covisibility: f32,
) -> Result<bool, TrackerError> {
    let Some(neighbors) = map.covisibility().neighbors(keyframe_id) else {
        return Ok(false);
    };
    for &neighbor in neighbors.keys() {
        let ratio = map.covisibility_ratio(keyframe_id, neighbor)?;
        if ratio >= max_covisibility {
            return Ok(true);
        }
    }
    Ok(false)
}

fn build_map_observations(
    map: &SlamMap,
    keyframe_id: KeyframeId,
    matches: &Matches<Verified>,
    current: &Detections,
    intrinsics: PinholeIntrinsics,
) -> Result<Vec<crate::Observation>, crate::PnpError> {
    let mut observations = Vec::with_capacity(matches.len());
    let current_len = current.len();

    for &(ci, ki) in matches.indices() {
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
    }

    if observations.len() < 4 {
        return Err(crate::PnpError::NotEnoughPoints {
            required: 4,
            actual: observations.len(),
        });
    }
    Ok(observations)
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
            None,
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
