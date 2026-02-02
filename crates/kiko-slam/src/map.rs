use std::collections::HashMap;
use std::num::NonZeroU32;

use slotmap::{new_key_type, SlotMap};

use crate::{Descriptor, Detections, FrameId, Keypoint, Point3, Pose, SensorId, Timestamp};

new_key_type! {
    pub struct MapPointId;
    pub struct KeyframeId;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageSize {
    width: u32,
    height: u32,
}

impl ImageSize {
    pub fn try_new(width: u32, height: u32) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }
        Some(Self { width, height })
    }

    pub fn width(self) -> u32 {
        self.width
    }

    pub fn height(self) -> u32 {
        self.height
    }

    pub fn max_dim(self) -> u32 {
        self.width.max(self.height)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct KeypointIndex(usize);

impl KeypointIndex {
    fn new(index: usize, len: usize) -> Option<Self> {
        if index < len {
            Some(Self(index))
        } else {
            None
        }
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KeyframeKeypoint {
    keyframe_id: KeyframeId,
    index: KeypointIndex,
}

impl KeyframeKeypoint {
    pub fn keyframe_id(self) -> KeyframeId {
        self.keyframe_id
    }

    pub fn index(self) -> usize {
        self.index.as_usize()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DescriptorBlend(f32);

#[derive(Debug)]
pub enum BlendError {
    OutOfRange { alpha: f32 },
}

impl std::fmt::Display for BlendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlendError::OutOfRange { alpha } => {
                write!(f, "descriptor blend must be in (0, 1], got {alpha}")
            }
        }
    }
}

impl std::error::Error for BlendError {}

impl DescriptorBlend {
    pub fn try_new(alpha: f32) -> Result<Self, BlendError> {
        if alpha > 0.0 && alpha <= 1.0 {
            Ok(Self(alpha))
        } else {
            Err(BlendError::OutOfRange { alpha })
        }
    }

    pub fn alpha(self) -> f32 {
        self.0
    }
}

#[derive(Debug)]
pub struct MapPoint {
    position: Point3,
    descriptor: Descriptor,
    observations: Vec<KeyframeKeypoint>,
}

impl MapPoint {
    pub fn position(&self) -> Point3 {
        self.position
    }

    pub fn descriptor(&self) -> &Descriptor {
        &self.descriptor
    }

    pub fn observations(&self) -> &[KeyframeKeypoint] {
        &self.observations
    }

    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    fn observes_keyframe(&self, keyframe_id: KeyframeId) -> bool {
        self.observations
            .iter()
            .any(|obs| obs.keyframe_id == keyframe_id)
    }

    fn add_observation(&mut self, obs: KeyframeKeypoint) {
        self.observations.push(obs);
    }

    fn remove_observation_for(&mut self, keyframe_id: KeyframeId) -> bool {
        let before = self.observations.len();
        self.observations.retain(|obs| obs.keyframe_id != keyframe_id);
        before != self.observations.len()
    }

    fn update_descriptor(&mut self, new_desc: &Descriptor, blend: DescriptorBlend) {
        let alpha = blend.alpha();
        let inv = 1.0 - alpha;
        for i in 0..256 {
            self.descriptor.0[i] = self.descriptor.0[i] * inv + new_desc.0[i] * alpha;
        }
        normalize_descriptor(&mut self.descriptor);
    }

    fn set_position(&mut self, pos: Point3) {
        self.position = pos;
    }
}

#[derive(Debug)]
pub struct KeyframeEntry {
    frame_id: FrameId,
    timestamp: Timestamp,
    pose: Pose,
    image_size: ImageSize,
    keypoints: Vec<Keypoint>,
    point_refs: Vec<Option<MapPointId>>,
}

impl KeyframeEntry {
    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    pub fn timestamp(&self) -> Timestamp {
        self.timestamp
    }

    pub fn pose(&self) -> Pose {
        self.pose
    }

    pub fn image_size(&self) -> ImageSize {
        self.image_size
    }

    pub fn keypoints(&self) -> &[Keypoint] {
        &self.keypoints
    }

    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }

    fn keypoint(&self, index: KeypointIndex) -> Keypoint {
        self.keypoints[index.as_usize()]
    }

    fn point_ref(&self, index: KeypointIndex) -> Option<MapPointId> {
        self.point_refs[index.as_usize()]
    }

    fn set_point_ref(&mut self, index: KeypointIndex, point_id: MapPointId) {
        self.point_refs[index.as_usize()] = Some(point_id);
    }

    fn clear_point_ref(&mut self, index: KeypointIndex) {
        self.point_refs[index.as_usize()] = None;
    }

    fn set_pose(&mut self, pose: Pose) {
        self.pose = pose;
    }

    fn map_point_ids(&self) -> impl Iterator<Item = MapPointId> + '_ {
        self.point_refs.iter().filter_map(|r| *r)
    }
}

#[derive(Debug, Default)]
pub struct CovisibilityGraph {
    edges: HashMap<KeyframeId, HashMap<KeyframeId, NonZeroU32>>,
}

impl CovisibilityGraph {
    fn increment_pair(&mut self, a: KeyframeId, b: KeyframeId) {
        if a == b {
            return;
        }
        self.increment_one(a, b);
        self.increment_one(b, a);
    }

    fn increment_one(&mut self, a: KeyframeId, b: KeyframeId) {
        let entry = self.edges.entry(a).or_default();
        if let Some(weight) = entry.get_mut(&b) {
            let next = weight.get() + 1;
            *weight = NonZeroU32::new(next).expect("non-zero");
        } else {
            entry.insert(b, NonZeroU32::new(1).expect("non-zero"));
        }
    }

    fn decrement_pair(&mut self, a: KeyframeId, b: KeyframeId) {
        if a == b {
            return;
        }
        self.decrement_one(a, b);
        self.decrement_one(b, a);
    }

    fn decrement_one(&mut self, a: KeyframeId, b: KeyframeId) {
        let remove_edge = if let Some(neighbors) = self.edges.get_mut(&a) {
            if let Some(weight) = neighbors.get(&b).copied() {
                let next = weight.get().saturating_sub(1);
                if next == 0 {
                    neighbors.remove(&b);
                } else {
                    neighbors.insert(b, NonZeroU32::new(next).expect("non-zero"));
                }
            }
            neighbors.is_empty()
        } else {
            false
        };

        if remove_edge {
            self.edges.remove(&a);
        }
    }

    fn remove_point_observations(&mut self, observations: &[KeyframeKeypoint]) {
        for i in 0..observations.len() {
            for j in (i + 1)..observations.len() {
                self.decrement_pair(
                    observations[i].keyframe_id,
                    observations[j].keyframe_id,
                );
            }
        }
    }

    pub fn neighbors(&self, kf_id: KeyframeId) -> Option<&HashMap<KeyframeId, NonZeroU32>> {
        self.edges.get(&kf_id)
    }

    pub fn remove_keyframe(&mut self, kf_id: KeyframeId) {
        if let Some(neighbors) = self.edges.remove(&kf_id) {
            for neighbor_id in neighbors.keys() {
                if let Some(their_edges) = self.edges.get_mut(neighbor_id) {
                    their_edges.remove(&kf_id);
                    if their_edges.is_empty() {
                        self.edges.remove(neighbor_id);
                    }
                }
            }
        }
    }

    pub fn covisibility_count(&self, a: KeyframeId, b: KeyframeId) -> u32 {
        self.edges
            .get(&a)
            .and_then(|m| m.get(&b))
            .map(|v| v.get())
            .unwrap_or(0)
    }
}

#[derive(Debug)]
pub enum MapError {
    KeyframeNotFound(KeyframeId),
    MapPointNotFound(MapPointId),
    FrameAlreadyKeyframed { frame_id: FrameId, existing: KeyframeId },
    KeypointIndexOutOfBounds { index: usize, len: usize },
    DetectionAlreadyAssociated { keyframe_id: KeyframeId, index: usize, existing: MapPointId },
    DuplicateObservation { point_id: MapPointId, keyframe_id: KeyframeId },
    InvalidImageSize { width: u32, height: u32 },
    EmptyKeyframe { frame_id: FrameId },
    SensorMismatch { expected: SensorId, actual: SensorId },
}

impl std::fmt::Display for MapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MapError::KeyframeNotFound(id) => write!(f, "keyframe not found: {id:?}"),
            MapError::MapPointNotFound(id) => write!(f, "map point not found: {id:?}"),
            MapError::FrameAlreadyKeyframed { frame_id, existing } => write!(
                f,
                "frame {frame_id:?} already has keyframe {existing:?}"
            ),
            MapError::KeypointIndexOutOfBounds { index, len } => {
                write!(f, "keypoint index {index} out of bounds (len={len})")
            }
            MapError::DetectionAlreadyAssociated {
                keyframe_id,
                index,
                existing,
            } => write!(
                f,
                "keypoint {index} on {keyframe_id:?} already maps to {existing:?}"
            ),
            MapError::DuplicateObservation { point_id, keyframe_id } => write!(
                f,
                "map point {point_id:?} already observed in keyframe {keyframe_id:?}"
            ),
            MapError::InvalidImageSize { width, height } => {
                write!(f, "invalid image size {width}x{height}")
            }
            MapError::EmptyKeyframe { frame_id } => {
                write!(f, "keyframe {frame_id:?} has no keypoints")
            }
            MapError::SensorMismatch { expected, actual } => write!(
                f,
                "keyframe detections must be from {expected:?}, got {actual:?}"
            ),
        }
    }
}

impl std::error::Error for MapError {}

#[derive(Debug)]
pub struct SlamMap {
    points: SlotMap<MapPointId, MapPoint>,
    keyframes: SlotMap<KeyframeId, KeyframeEntry>,
    covisibility: CovisibilityGraph,
    frame_to_keyframe: HashMap<FrameId, KeyframeId>,
}

impl SlamMap {
    pub fn new() -> Self {
        Self {
            points: SlotMap::with_key(),
            keyframes: SlotMap::with_key(),
            covisibility: CovisibilityGraph::default(),
            frame_to_keyframe: HashMap::new(),
        }
    }

    pub fn add_keyframe_from_detections(
        &mut self,
        detections: &Detections,
        timestamp: Timestamp,
        pose: Pose,
    ) -> Result<KeyframeId, MapError> {
        if detections.sensor_id() != SensorId::StereoLeft {
            return Err(MapError::SensorMismatch {
                expected: SensorId::StereoLeft,
                actual: detections.sensor_id(),
            });
        }

        let image_size = ImageSize::try_new(detections.width(), detections.height())
            .ok_or(MapError::InvalidImageSize {
                width: detections.width(),
                height: detections.height(),
            })?;

        let keypoints = detections.keypoints().to_vec();
        self.add_keyframe(
            detections.frame_id(),
            timestamp,
            pose,
            image_size,
            keypoints,
        )
    }

    pub fn add_keyframe(
        &mut self,
        frame_id: FrameId,
        timestamp: Timestamp,
        pose: Pose,
        image_size: ImageSize,
        keypoints: Vec<Keypoint>,
    ) -> Result<KeyframeId, MapError> {
        if let Some(existing) = self.frame_to_keyframe.get(&frame_id) {
            return Err(MapError::FrameAlreadyKeyframed {
                frame_id,
                existing: *existing,
            });
        }
        if keypoints.is_empty() {
            return Err(MapError::EmptyKeyframe { frame_id });
        }

        let entry = KeyframeEntry {
            frame_id,
            timestamp,
            pose,
            image_size,
            point_refs: vec![None; keypoints.len()],
            keypoints,
        };

        let kf_id = self.keyframes.insert(entry);
        self.frame_to_keyframe.insert(frame_id, kf_id);
        Ok(kf_id)
    }

    pub fn keyframe_keypoint(
        &self,
        keyframe_id: KeyframeId,
        index: usize,
    ) -> Result<KeyframeKeypoint, MapError> {
        let entry = self
            .keyframes
            .get(keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        let idx = KeypointIndex::new(index, entry.len())
            .ok_or(MapError::KeypointIndexOutOfBounds {
                index,
                len: entry.len(),
            })?;
        Ok(KeyframeKeypoint {
            keyframe_id,
            index: idx,
        })
    }

    pub fn keypoint(&self, keypoint: KeyframeKeypoint) -> Result<Keypoint, MapError> {
        let entry = self
            .keyframes
            .get(keypoint.keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keypoint.keyframe_id))?;
        Ok(entry.keypoint(keypoint.index))
    }

    pub fn add_map_point(
        &mut self,
        position: Point3,
        descriptor: Descriptor,
        first_obs: KeyframeKeypoint,
    ) -> Result<MapPointId, MapError> {
        let entry = self
            .keyframes
            .get(first_obs.keyframe_id)
            .ok_or(MapError::KeyframeNotFound(first_obs.keyframe_id))?;
        let idx = first_obs.index.as_usize();
        debug_assert!(idx < entry.point_refs.len(), "KeyframeKeypoint out of bounds");
        if let Some(existing) = entry.point_ref(first_obs.index) {
            return Err(MapError::DetectionAlreadyAssociated {
                keyframe_id: first_obs.keyframe_id,
                index: idx,
                existing,
            });
        }

        let mut desc = descriptor;
        normalize_descriptor(&mut desc);
        let point_id = self.points.insert(MapPoint {
            position,
            descriptor: desc,
            observations: vec![first_obs],
        });

        let entry = self
            .keyframes
            .get_mut(first_obs.keyframe_id)
            .expect("keyframe exists");
        entry.set_point_ref(first_obs.index, point_id);
        Ok(point_id)
    }

    pub fn add_observation(
        &mut self,
        point_id: MapPointId,
        obs: KeyframeKeypoint,
    ) -> Result<(), MapError> {
        let entry = self
            .keyframes
            .get(obs.keyframe_id)
            .ok_or(MapError::KeyframeNotFound(obs.keyframe_id))?;
        let idx = obs.index.as_usize();
        debug_assert!(idx < entry.point_refs.len(), "KeyframeKeypoint out of bounds");
        if let Some(existing) = entry.point_ref(obs.index) {
            return Err(MapError::DetectionAlreadyAssociated {
                keyframe_id: obs.keyframe_id,
                index: idx,
                existing,
            });
        }

        let other_keyframes: Vec<KeyframeId> = {
            let point = self
                .points
                .get(point_id)
                .ok_or(MapError::MapPointNotFound(point_id))?;
            if point.observes_keyframe(obs.keyframe_id) {
                return Err(MapError::DuplicateObservation {
                    point_id,
                    keyframe_id: obs.keyframe_id,
                });
            }
            point.observations
                .iter()
                .map(|o| o.keyframe_id)
                .collect()
        };

        for other in other_keyframes {
            self.covisibility.increment_pair(obs.keyframe_id, other);
        }

        let point = self
            .points
            .get_mut(point_id)
            .expect("map point exists");
        point.add_observation(obs);

        let entry = self
            .keyframes
            .get_mut(obs.keyframe_id)
            .expect("keyframe exists");
        entry.set_point_ref(obs.index, point_id);
        Ok(())
    }

    pub fn update_map_point_descriptor(
        &mut self,
        point_id: MapPointId,
        new_desc: &Descriptor,
        blend: DescriptorBlend,
    ) -> Result<(), MapError> {
        let point = self
            .points
            .get_mut(point_id)
            .ok_or(MapError::MapPointNotFound(point_id))?;
        point.update_descriptor(new_desc, blend);
        Ok(())
    }

    pub fn set_map_point_position(
        &mut self,
        point_id: MapPointId,
        position: Point3,
    ) -> Result<(), MapError> {
        let point = self
            .points
            .get_mut(point_id)
            .ok_or(MapError::MapPointNotFound(point_id))?;
        point.set_position(position);
        Ok(())
    }

    pub fn set_keyframe_pose(&mut self, keyframe_id: KeyframeId, pose: Pose) -> Result<(), MapError> {
        let entry = self
            .keyframes
            .get_mut(keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        entry.set_pose(pose);
        Ok(())
    }

    pub fn remove_map_point(&mut self, point_id: MapPointId) -> Result<(), MapError> {
        let point = self
            .points
            .remove(point_id)
            .ok_or(MapError::MapPointNotFound(point_id))?;

        for obs in &point.observations {
            if let Some(entry) = self.keyframes.get_mut(obs.keyframe_id) {
                entry.clear_point_ref(obs.index);
            }
        }
        self.covisibility.remove_point_observations(&point.observations);
        Ok(())
    }

    pub fn remove_keyframe(&mut self, keyframe_id: KeyframeId) -> Result<(), MapError> {
        let entry = self
            .keyframes
            .remove(keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        self.frame_to_keyframe.remove(&entry.frame_id);
        self.covisibility.remove_keyframe(keyframe_id);

        let mut to_remove = Vec::new();
        for point_id in entry.map_point_ids() {
            if let Some(point) = self.points.get_mut(point_id) {
                point.remove_observation_for(keyframe_id);
                if point.observations.is_empty() {
                    to_remove.push(point_id);
                }
            }
        }
        for point_id in to_remove {
            let _ = self.points.remove(point_id);
        }
        Ok(())
    }

    pub fn cull_points(&mut self, min_observations: usize) -> usize {
        let to_remove: Vec<MapPointId> = self
            .points
            .iter()
            .filter(|(_, p)| p.observation_count() < min_observations)
            .map(|(id, _)| id)
            .collect();
        let count = to_remove.len();
        for id in to_remove {
            let removed = self.remove_map_point(id);
            debug_assert!(removed.is_ok(), "map point missing during cull");
        }
        count
    }

    pub fn keyframe(&self, id: KeyframeId) -> Option<&KeyframeEntry> {
        self.keyframes.get(id)
    }

    pub fn map_point_for_keypoint(
        &self,
        keypoint: KeyframeKeypoint,
    ) -> Result<Option<MapPointId>, MapError> {
        let entry = self
            .keyframes
            .get(keypoint.keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keypoint.keyframe_id))?;
        Ok(entry.point_ref(keypoint.index))
    }

    pub fn point(&self, id: MapPointId) -> Option<&MapPoint> {
        self.points.get(id)
    }

    pub fn keyframe_by_frame(&self, frame_id: FrameId) -> Option<KeyframeId> {
        self.frame_to_keyframe.get(&frame_id).copied()
    }

    pub fn covisibility(&self) -> &CovisibilityGraph {
        &self.covisibility
    }

    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    pub fn num_keyframes(&self) -> usize {
        self.keyframes.len()
    }

    pub fn points(&self) -> impl Iterator<Item = (MapPointId, &MapPoint)> {
        self.points.iter()
    }

    pub fn keyframes(&self) -> impl Iterator<Item = (KeyframeId, &KeyframeEntry)> {
        self.keyframes.iter()
    }
}

impl Default for SlamMap {
    fn default() -> Self {
        Self::new()
    }
}

fn normalize_descriptor(desc: &mut Descriptor) {
    let norm_sq: f32 = desc.0.iter().map(|x| x * x).sum();
    if norm_sq > 0.0 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for v in &mut desc.0 {
            *v *= inv_norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Keypoint, Pose, Timestamp};

    fn make_keypoints(n: usize) -> Vec<Keypoint> {
        (0..n)
            .map(|i| Keypoint {
                x: i as f32,
                y: i as f32,
            })
            .collect()
    }

    fn make_descriptor() -> Descriptor {
        Descriptor([1.0; 256])
    }

    #[test]
    fn covisibility_increments_and_decrements_on_map_point_changes() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = Pose::identity();

        let kf1 = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                pose,
                size,
                make_keypoints(3),
            )
            .expect("keyframe1");
        let kf2 = map
            .add_keyframe(
                FrameId::new(2),
                Timestamp::from_nanos(2),
                pose,
                size,
                make_keypoints(3),
            )
            .expect("keyframe2");

        let obs1 = map.keyframe_keypoint(kf1, 0).expect("obs1");
        let point = map
            .add_map_point(Point3 { x: 0.0, y: 0.0, z: 1.0 }, make_descriptor(), obs1)
            .expect("map point");

        let obs2 = map.keyframe_keypoint(kf2, 0).expect("obs2");
        map.add_observation(point, obs2).expect("second observation");

        assert_eq!(map.covisibility().covisibility_count(kf1, kf2), 1);

        map.remove_map_point(point).expect("remove point");
        assert_eq!(map.covisibility().covisibility_count(kf1, kf2), 0);
    }

    #[test]
    fn duplicate_observation_is_rejected() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = Pose::identity();

        let kf1 = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                pose,
                size,
                make_keypoints(3),
            )
            .expect("keyframe1");
        let kf2 = map
            .add_keyframe(
                FrameId::new(2),
                Timestamp::from_nanos(2),
                pose,
                size,
                make_keypoints(3),
            )
            .expect("keyframe2");

        let obs1 = map.keyframe_keypoint(kf1, 0).expect("obs1");
        let point = map
            .add_map_point(Point3 { x: 0.0, y: 0.0, z: 1.0 }, make_descriptor(), obs1)
            .expect("map point");

        let obs2 = map.keyframe_keypoint(kf2, 0).expect("obs2");
        map.add_observation(point, obs2).expect("second observation");

        let obs2_alt = map.keyframe_keypoint(kf2, 1).expect("obs2_alt");
        let err = map.add_observation(point, obs2_alt).expect_err("duplicate observation");
        match err {
            MapError::DuplicateObservation { point_id, keyframe_id } => {
                assert_eq!(point_id, point);
                assert_eq!(keyframe_id, kf2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn remove_keyframe_removes_orphaned_points() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = Pose::identity();

        let kf1 = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                pose,
                size,
                make_keypoints(1),
            )
            .expect("keyframe1");

        let obs1 = map.keyframe_keypoint(kf1, 0).expect("obs1");
        let point = map
            .add_map_point(Point3 { x: 0.0, y: 0.0, z: 1.0 }, make_descriptor(), obs1)
            .expect("map point");
        assert!(map.point(point).is_some());

        map.remove_keyframe(kf1).expect("remove keyframe");
        assert_eq!(map.num_keyframes(), 0);
        assert_eq!(map.num_points(), 0);
    }

    #[test]
    fn covisibility_updates_for_shared_points() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = Pose::identity();

        let kf1 = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                pose,
                size,
                make_keypoints(3),
            )
            .expect("keyframe1");
        let kf2 = map
            .add_keyframe(
                FrameId::new(2),
                Timestamp::from_nanos(2),
                pose,
                size,
                make_keypoints(3),
            )
            .expect("keyframe2");
        let kf3 = map
            .add_keyframe(
                FrameId::new(3),
                Timestamp::from_nanos(3),
                pose,
                size,
                make_keypoints(3),
            )
            .expect("keyframe3");

        let obs1 = map.keyframe_keypoint(kf1, 0).expect("obs1");
        let point_a = map
            .add_map_point(Point3 { x: 0.0, y: 0.0, z: 1.0 }, make_descriptor(), obs1)
            .expect("point A");
        let obs2 = map.keyframe_keypoint(kf2, 0).expect("obs2");
        map.add_observation(point_a, obs2).expect("obs2 add");
        let obs3 = map.keyframe_keypoint(kf3, 0).expect("obs3");
        map.add_observation(point_a, obs3).expect("obs3 add");

        assert_eq!(map.covisibility().covisibility_count(kf1, kf2), 1);
        assert_eq!(map.covisibility().covisibility_count(kf1, kf3), 1);
        assert_eq!(map.covisibility().covisibility_count(kf2, kf3), 1);

        let obs1b = map.keyframe_keypoint(kf1, 1).expect("obs1b");
        let point_b = map
            .add_map_point(Point3 { x: 1.0, y: 0.0, z: 2.0 }, make_descriptor(), obs1b)
            .expect("point B");
        let obs2b = map.keyframe_keypoint(kf2, 1).expect("obs2b");
        map.add_observation(point_b, obs2b).expect("obs2b add");

        assert_eq!(map.covisibility().covisibility_count(kf1, kf2), 2);

        map.remove_map_point(point_b).expect("remove point B");
        assert_eq!(map.covisibility().covisibility_count(kf1, kf2), 1);
    }
}
