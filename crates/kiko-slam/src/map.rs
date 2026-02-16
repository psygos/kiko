use std::collections::{HashMap, HashSet};
use std::num::{NonZeroU32, NonZeroUsize};

use slotmap::{SlotMap, new_key_type};

use crate::{CompactDescriptor, Detections, FrameId, Keypoint, Point3, Pose, SensorId, Timestamp};

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
        if index < len { Some(Self(index)) } else { None }
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

#[derive(Clone, Debug)]
pub struct MapPoint {
    position: Point3,
    descriptor: CompactDescriptor,
    observations: Vec<KeyframeKeypoint>,
}

impl MapPoint {
    pub fn position(&self) -> Point3 {
        self.position
    }

    pub fn descriptor(&self) -> &CompactDescriptor {
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
        self.observations
            .retain(|obs| obs.keyframe_id != keyframe_id);
        before != self.observations.len()
    }

    fn update_descriptor(&mut self, new_desc: &CompactDescriptor, blend: DescriptorBlend) {
        // Use fixed-point blending so descriptor updates stay bounded and deterministic.
        let alpha_scaled = (blend.alpha() * 256.0).round().clamp(1.0, 256.0) as u16;
        let inv_scaled = 256u16.saturating_sub(alpha_scaled);
        for i in 0..256 {
            let prev = self.descriptor.0[i] as u32;
            let next = new_desc.0[i] as u32;
            let mixed = prev * inv_scaled as u32 + next * alpha_scaled as u32;
            self.descriptor.0[i] = ((mixed + 128) / 256) as u8;
        }
    }

    fn set_position(&mut self, pos: Point3) {
        self.position = pos;
    }
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug, Default)]
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
                self.decrement_pair(observations[i].keyframe_id, observations[j].keyframe_id);
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
    FrameAlreadyKeyframed {
        frame_id: FrameId,
        existing: KeyframeId,
    },
    KeypointIndexOutOfBounds {
        index: usize,
        len: usize,
    },
    DetectionAlreadyAssociated {
        keyframe_id: KeyframeId,
        index: usize,
        existing: MapPointId,
    },
    DuplicateObservation {
        point_id: MapPointId,
        keyframe_id: KeyframeId,
    },
    InvalidImageSize {
        width: u32,
        height: u32,
    },
    EmptyKeyframe {
        frame_id: FrameId,
    },
    SensorMismatch {
        expected: SensorId,
        actual: SensorId,
    },
}

impl std::fmt::Display for MapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MapError::KeyframeNotFound(id) => write!(f, "keyframe not found: {id:?}"),
            MapError::MapPointNotFound(id) => write!(f, "map point not found: {id:?}"),
            MapError::FrameAlreadyKeyframed { frame_id, existing } => {
                write!(f, "frame {frame_id:?} already has keyframe {existing:?}")
            }
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
            MapError::DuplicateObservation {
                point_id,
                keyframe_id,
            } => write!(
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MapGeneration(u64);

impl MapGeneration {
    fn initial() -> Self {
        Self(0)
    }

    fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct SlamMap {
    points: SlotMap<MapPointId, MapPoint>,
    keyframes: SlotMap<KeyframeId, KeyframeEntry>,
    covisibility: CovisibilityGraph,
    frame_to_keyframe: HashMap<FrameId, KeyframeId>,
    generation: MapGeneration,
}

#[derive(Debug, Clone)]
pub struct CovisibilityNode {
    pub id: KeyframeId,
    pub pose: Pose,
}

#[derive(Debug, Clone, Copy)]
pub struct CovisibilityEdge {
    pub a: KeyframeId,
    pub b: KeyframeId,
    pub weight: u32,
}

#[derive(Debug, Clone)]
pub struct CovisibilitySnapshot {
    pub nodes: Vec<CovisibilityNode>,
    pub edges: Vec<CovisibilityEdge>,
}

impl SlamMap {
    pub fn new() -> Self {
        Self {
            points: SlotMap::with_key(),
            keyframes: SlotMap::with_key(),
            covisibility: CovisibilityGraph::default(),
            frame_to_keyframe: HashMap::new(),
            generation: MapGeneration::initial(),
        }
    }

    fn bump_generation(&mut self) {
        self.generation = self.generation.next();
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

        let image_size = ImageSize::try_new(detections.width(), detections.height()).ok_or(
            MapError::InvalidImageSize {
                width: detections.width(),
                height: detections.height(),
            },
        )?;

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
        self.bump_generation();
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
        let idx =
            KeypointIndex::new(index, entry.len()).ok_or(MapError::KeypointIndexOutOfBounds {
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
        descriptor: CompactDescriptor,
        first_obs: KeyframeKeypoint,
    ) -> Result<MapPointId, MapError> {
        let entry = self
            .keyframes
            .get(first_obs.keyframe_id)
            .ok_or(MapError::KeyframeNotFound(first_obs.keyframe_id))?;
        let idx = first_obs.index.as_usize();
        debug_assert!(
            idx < entry.point_refs.len(),
            "KeyframeKeypoint out of bounds"
        );
        if let Some(existing) = entry.point_ref(first_obs.index) {
            return Err(MapError::DetectionAlreadyAssociated {
                keyframe_id: first_obs.keyframe_id,
                index: idx,
                existing,
            });
        }

        let point_id = self.points.insert(MapPoint {
            position,
            descriptor,
            observations: vec![first_obs],
        });

        let entry = self
            .keyframes
            .get_mut(first_obs.keyframe_id)
            .expect("keyframe exists");
        entry.set_point_ref(first_obs.index, point_id);
        self.bump_generation();
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
        debug_assert!(
            idx < entry.point_refs.len(),
            "KeyframeKeypoint out of bounds"
        );
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
            point.observations.iter().map(|o| o.keyframe_id).collect()
        };

        for other in other_keyframes {
            self.covisibility.increment_pair(obs.keyframe_id, other);
        }

        let point = self.points.get_mut(point_id).expect("map point exists");
        point.add_observation(obs);

        let entry = self
            .keyframes
            .get_mut(obs.keyframe_id)
            .expect("keyframe exists");
        entry.set_point_ref(obs.index, point_id);
        self.bump_generation();
        Ok(())
    }

    pub fn update_map_point_descriptor(
        &mut self,
        point_id: MapPointId,
        new_desc: &CompactDescriptor,
        blend: DescriptorBlend,
    ) -> Result<(), MapError> {
        let point = self
            .points
            .get_mut(point_id)
            .ok_or(MapError::MapPointNotFound(point_id))?;
        point.update_descriptor(new_desc, blend);
        self.bump_generation();
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
        self.bump_generation();
        Ok(())
    }

    pub fn set_keyframe_pose(
        &mut self,
        keyframe_id: KeyframeId,
        pose: Pose,
    ) -> Result<(), MapError> {
        let entry = self
            .keyframes
            .get_mut(keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        entry.set_pose(pose);
        self.bump_generation();
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
        self.covisibility
            .remove_point_observations(&point.observations);
        self.bump_generation();
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
        self.bump_generation();
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

    pub fn keyframe_observation_pixels(
        &self,
        keyframe_id: KeyframeId,
    ) -> Result<Vec<(KeyframeKeypoint, Keypoint)>, MapError> {
        let entry = self
            .keyframes
            .get(keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        let mut observations = Vec::new();
        for (idx, point_id) in entry.point_refs.iter().enumerate() {
            if point_id.is_some() {
                let index =
                    KeypointIndex::new(idx, entry.len()).expect("keypoint index within bounds");
                let keypoint_ref = KeyframeKeypoint { keyframe_id, index };
                observations.push((keypoint_ref, entry.keypoints[idx]));
            }
        }
        Ok(observations)
    }

    pub fn keyframe_point_descriptors(
        &self,
        keyframe_id: KeyframeId,
    ) -> Result<Vec<(KeyframeKeypoint, CompactDescriptor)>, MapError> {
        let entry = self
            .keyframes
            .get(keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        let mut descriptors = Vec::new();
        for (idx, point_ref) in entry.point_refs.iter().enumerate() {
            let Some(point_id) = point_ref else {
                continue;
            };
            let point = self
                .points
                .get(*point_id)
                .ok_or(MapError::MapPointNotFound(*point_id))?;
            let index =
                KeypointIndex::new(idx, entry.len()).expect("point refs length matches keypoints");
            descriptors.push((
                KeyframeKeypoint { keyframe_id, index },
                point.descriptor().clone(),
            ));
        }
        Ok(descriptors)
    }

    pub fn keyframe_point_count(&self, keyframe_id: KeyframeId) -> Result<usize, MapError> {
        let entry = self
            .keyframes
            .get(keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        Ok(entry.map_point_ids().count())
    }

    pub fn covisible_window(
        &self,
        seed: KeyframeId,
        max: NonZeroUsize,
    ) -> Result<Vec<KeyframeId>, MapError> {
        if !self.keyframes.contains_key(seed) {
            return Err(MapError::KeyframeNotFound(seed));
        }
        let mut window = Vec::new();
        window.push(seed);

        let neighbors = match self.covisibility.neighbors(seed) {
            Some(neighbors) => neighbors,
            None => return Ok(window),
        };

        let mut sorted: Vec<(KeyframeId, NonZeroU32)> =
            neighbors.iter().map(|(&id, &w)| (id, w)).collect();
        sorted.sort_by(|a, b| b.1.get().cmp(&a.1.get()));

        let limit = max.get().saturating_sub(1);
        for (id, _) in sorted.into_iter().take(limit) {
            window.push(id);
        }
        Ok(window)
    }

    pub fn covisibility_ratio(&self, a: KeyframeId, b: KeyframeId) -> Result<f32, MapError> {
        let count = self.covisibility.covisibility_count(a, b) as f32;
        if count == 0.0 {
            return Ok(0.0);
        }
        let a_points = self.keyframe_point_count(a)? as f32;
        let b_points = self.keyframe_point_count(b)? as f32;
        if a_points == 0.0 || b_points == 0.0 {
            return Ok(0.0);
        }
        Ok(count / a_points.min(b_points))
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

    pub fn covisibility_snapshot(&self) -> CovisibilitySnapshot {
        let nodes: Vec<CovisibilityNode> = self
            .keyframes
            .iter()
            .map(|(id, entry)| CovisibilityNode {
                id,
                pose: entry.pose(),
            })
            .collect();

        let mut edges = Vec::new();
        let mut seen: HashSet<(KeyframeId, KeyframeId)> = HashSet::new();
        for (&a, neighbors) in &self.covisibility.edges {
            for (&b, weight) in neighbors {
                if a == b {
                    continue;
                }
                if seen.contains(&(b, a)) {
                    continue;
                }
                seen.insert((a, b));
                edges.push(CovisibilityEdge {
                    a,
                    b,
                    weight: weight.get(),
                });
            }
        }

        CovisibilitySnapshot { nodes, edges }
    }

    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    pub fn num_keyframes(&self) -> usize {
        self.keyframes.len()
    }

    pub fn generation(&self) -> MapGeneration {
        self.generation
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

#[cfg(test)]
#[derive(Debug)]
pub(crate) enum MapInvariantError {
    FrameIndexLenMismatch {
        keyframes: usize,
        frame_to_keyframe: usize,
    },
    FrameIndexMissingKeyframe {
        frame_id: FrameId,
        keyframe_id: KeyframeId,
    },
    FrameIndexMismatchedFrameId {
        frame_id: FrameId,
        keyframe_id: KeyframeId,
        stored_frame_id: FrameId,
    },
    EmptyKeyframe {
        keyframe_id: KeyframeId,
    },
    KeypointPointRefLenMismatch {
        keyframe_id: KeyframeId,
        keypoints: usize,
        point_refs: usize,
    },
    KeyframeReferencesMissingPoint {
        keyframe_id: KeyframeId,
        index: usize,
        point_id: MapPointId,
    },
    KeyframePointBackrefMissing {
        keyframe_id: KeyframeId,
        index: usize,
        point_id: MapPointId,
    },
    DuplicatePointInKeyframe {
        keyframe_id: KeyframeId,
        point_id: MapPointId,
    },
    EmptyMapPoint {
        point_id: MapPointId,
    },
    MapPointDuplicateObservation {
        point_id: MapPointId,
        keyframe_id: KeyframeId,
    },
    MapPointObservationMissingKeyframe {
        point_id: MapPointId,
        keyframe_id: KeyframeId,
    },
    MapPointObservationIndexOutOfBounds {
        point_id: MapPointId,
        keyframe_id: KeyframeId,
        index: usize,
        keyframe_len: usize,
    },
    MapPointBackrefMismatch {
        point_id: MapPointId,
        keyframe_id: KeyframeId,
        index: usize,
        found: Option<MapPointId>,
    },
    CovisibilitySelfEdge {
        keyframe_id: KeyframeId,
    },
    CovisibilityMissingReverseEdge {
        a: KeyframeId,
        b: KeyframeId,
    },
    CovisibilityAsymmetricWeight {
        a: KeyframeId,
        b: KeyframeId,
        ab: u32,
        ba: u32,
    },
    CovisibilityUnexpectedWeight {
        a: KeyframeId,
        b: KeyframeId,
        actual: u32,
        expected: u32,
    },
}

#[cfg(test)]
impl std::fmt::Display for MapInvariantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MapInvariantError::FrameIndexLenMismatch {
                keyframes,
                frame_to_keyframe,
            } => write!(
                f,
                "frame index mismatch: keyframes={keyframes}, frame_to_keyframe={frame_to_keyframe}"
            ),
            MapInvariantError::FrameIndexMissingKeyframe {
                frame_id,
                keyframe_id,
            } => write!(
                f,
                "frame index points to missing keyframe: frame={frame_id:?}, keyframe={keyframe_id:?}"
            ),
            MapInvariantError::FrameIndexMismatchedFrameId {
                frame_id,
                keyframe_id,
                stored_frame_id,
            } => write!(
                f,
                "frame index mismatch: frame={frame_id:?}, keyframe={keyframe_id:?}, stored_frame={stored_frame_id:?}"
            ),
            MapInvariantError::EmptyKeyframe { keyframe_id } => {
                write!(f, "keyframe has no keypoints: keyframe={keyframe_id:?}")
            }
            MapInvariantError::KeypointPointRefLenMismatch {
                keyframe_id,
                keypoints,
                point_refs,
            } => write!(
                f,
                "keyframe keypoint/point_ref length mismatch: keyframe={keyframe_id:?}, keypoints={keypoints}, point_refs={point_refs}"
            ),
            MapInvariantError::KeyframeReferencesMissingPoint {
                keyframe_id,
                index,
                point_id,
            } => write!(
                f,
                "keyframe references missing map point: keyframe={keyframe_id:?}, index={index}, point={point_id:?}"
            ),
            MapInvariantError::KeyframePointBackrefMissing {
                keyframe_id,
                index,
                point_id,
            } => write!(
                f,
                "keyframe->point reference missing backref: keyframe={keyframe_id:?}, index={index}, point={point_id:?}"
            ),
            MapInvariantError::DuplicatePointInKeyframe {
                keyframe_id,
                point_id,
            } => write!(
                f,
                "same map point referenced by multiple keypoints in keyframe={keyframe_id:?}, point={point_id:?}"
            ),
            MapInvariantError::EmptyMapPoint { point_id } => {
                write!(f, "map point has zero observations: point={point_id:?}")
            }
            MapInvariantError::MapPointDuplicateObservation {
                point_id,
                keyframe_id,
            } => write!(
                f,
                "map point observes same keyframe twice: point={point_id:?}, keyframe={keyframe_id:?}"
            ),
            MapInvariantError::MapPointObservationMissingKeyframe {
                point_id,
                keyframe_id,
            } => write!(
                f,
                "map point observation references missing keyframe: point={point_id:?}, keyframe={keyframe_id:?}"
            ),
            MapInvariantError::MapPointObservationIndexOutOfBounds {
                point_id,
                keyframe_id,
                index,
                keyframe_len,
            } => write!(
                f,
                "map point observation index out of bounds: point={point_id:?}, keyframe={keyframe_id:?}, index={index}, keyframe_len={keyframe_len}"
            ),
            MapInvariantError::MapPointBackrefMismatch {
                point_id,
                keyframe_id,
                index,
                found,
            } => write!(
                f,
                "map point backref mismatch: point={point_id:?}, keyframe={keyframe_id:?}, index={index}, found={found:?}"
            ),
            MapInvariantError::CovisibilitySelfEdge { keyframe_id } => {
                write!(
                    f,
                    "covisibility self edge present: keyframe={keyframe_id:?}"
                )
            }
            MapInvariantError::CovisibilityMissingReverseEdge { a, b } => {
                write!(f, "covisibility missing reverse edge: {a:?} -> {b:?}")
            }
            MapInvariantError::CovisibilityAsymmetricWeight { a, b, ab, ba } => write!(
                f,
                "covisibility asymmetric weights: {a:?}->{b:?}={ab}, {b:?}->{a:?}={ba}"
            ),
            MapInvariantError::CovisibilityUnexpectedWeight {
                a,
                b,
                actual,
                expected,
            } => write!(
                f,
                "covisibility weight mismatch: {a:?}<->{b:?}, actual={actual}, expected={expected}"
            ),
        }
    }
}

#[cfg(test)]
impl std::error::Error for MapInvariantError {}

#[cfg(test)]
pub(crate) fn assert_map_invariants(map: &SlamMap) -> Result<(), MapInvariantError> {
    if map.keyframes.len() != map.frame_to_keyframe.len() {
        return Err(MapInvariantError::FrameIndexLenMismatch {
            keyframes: map.keyframes.len(),
            frame_to_keyframe: map.frame_to_keyframe.len(),
        });
    }

    for (&frame_id, &keyframe_id) in &map.frame_to_keyframe {
        let Some(entry) = map.keyframes.get(keyframe_id) else {
            return Err(MapInvariantError::FrameIndexMissingKeyframe {
                frame_id,
                keyframe_id,
            });
        };
        if entry.frame_id() != frame_id {
            return Err(MapInvariantError::FrameIndexMismatchedFrameId {
                frame_id,
                keyframe_id,
                stored_frame_id: entry.frame_id(),
            });
        }
    }

    for (keyframe_id, entry) in map.keyframes.iter() {
        if entry.is_empty() {
            return Err(MapInvariantError::EmptyKeyframe { keyframe_id });
        }
        if entry.keypoints.len() != entry.point_refs.len() {
            return Err(MapInvariantError::KeypointPointRefLenMismatch {
                keyframe_id,
                keypoints: entry.keypoints.len(),
                point_refs: entry.point_refs.len(),
            });
        }

        let mut seen_points = HashSet::new();
        for (index, maybe_point_id) in entry.point_refs.iter().enumerate() {
            let Some(point_id) = *maybe_point_id else {
                continue;
            };

            let Some(point) = map.points.get(point_id) else {
                return Err(MapInvariantError::KeyframeReferencesMissingPoint {
                    keyframe_id,
                    index,
                    point_id,
                });
            };

            let backref_exists = point
                .observations
                .iter()
                .any(|obs| obs.keyframe_id == keyframe_id && obs.index.as_usize() == index);
            if !backref_exists {
                return Err(MapInvariantError::KeyframePointBackrefMissing {
                    keyframe_id,
                    index,
                    point_id,
                });
            }

            if !seen_points.insert(point_id) {
                return Err(MapInvariantError::DuplicatePointInKeyframe {
                    keyframe_id,
                    point_id,
                });
            }
        }
    }

    let mut expected_covisibility: HashMap<(KeyframeId, KeyframeId), u32> = HashMap::new();
    for (point_id, point) in map.points.iter() {
        if point.observations.is_empty() {
            return Err(MapInvariantError::EmptyMapPoint { point_id });
        }

        let mut seen_keyframes = HashSet::new();
        for obs in &point.observations {
            if !seen_keyframes.insert(obs.keyframe_id) {
                return Err(MapInvariantError::MapPointDuplicateObservation {
                    point_id,
                    keyframe_id: obs.keyframe_id,
                });
            }

            let Some(entry) = map.keyframes.get(obs.keyframe_id) else {
                return Err(MapInvariantError::MapPointObservationMissingKeyframe {
                    point_id,
                    keyframe_id: obs.keyframe_id,
                });
            };

            let index = obs.index.as_usize();
            if index >= entry.len() {
                return Err(MapInvariantError::MapPointObservationIndexOutOfBounds {
                    point_id,
                    keyframe_id: obs.keyframe_id,
                    index,
                    keyframe_len: entry.len(),
                });
            }

            let found = entry.point_ref(obs.index);
            if found != Some(point_id) {
                return Err(MapInvariantError::MapPointBackrefMismatch {
                    point_id,
                    keyframe_id: obs.keyframe_id,
                    index,
                    found,
                });
            }
        }

        for i in 0..point.observations.len() {
            for j in (i + 1)..point.observations.len() {
                let a = point.observations[i].keyframe_id;
                let b = point.observations[j].keyframe_id;
                *expected_covisibility.entry((a, b)).or_insert(0) += 1;
                *expected_covisibility.entry((b, a)).or_insert(0) += 1;
            }
        }
    }

    for (&a, neighbors) in &map.covisibility.edges {
        for (&b, &weight) in neighbors {
            if a == b {
                return Err(MapInvariantError::CovisibilitySelfEdge { keyframe_id: a });
            }

            let Some(reverse_neighbors) = map.covisibility.edges.get(&b) else {
                return Err(MapInvariantError::CovisibilityMissingReverseEdge { a, b });
            };
            let Some(reverse_weight) = reverse_neighbors.get(&a) else {
                return Err(MapInvariantError::CovisibilityMissingReverseEdge { a, b });
            };
            if reverse_weight.get() != weight.get() {
                return Err(MapInvariantError::CovisibilityAsymmetricWeight {
                    a,
                    b,
                    ab: weight.get(),
                    ba: reverse_weight.get(),
                });
            }

            let expected = expected_covisibility.get(&(a, b)).copied().unwrap_or(0);
            if expected != weight.get() {
                return Err(MapInvariantError::CovisibilityUnexpectedWeight {
                    a,
                    b,
                    actual: weight.get(),
                    expected,
                });
            }
        }
    }

    for ((a, b), expected) in expected_covisibility {
        let actual = map.covisibility.covisibility_count(a, b);
        if actual != expected {
            return Err(MapInvariantError::CovisibilityUnexpectedWeight {
                a,
                b,
                actual,
                expected,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompactDescriptor, Keypoint, Pose, Timestamp};

    fn make_keypoints(n: usize) -> Vec<Keypoint> {
        (0..n)
            .map(|i| Keypoint {
                x: i as f32,
                y: i as f32,
            })
            .collect()
    }

    fn make_descriptor() -> CompactDescriptor {
        CompactDescriptor([128; 256])
    }

    #[test]
    fn descriptor_blend_uses_u8_weighted_average() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = Pose::identity();
        let kf = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                pose,
                size,
                make_keypoints(1),
            )
            .expect("keyframe");
        let kp = map.keyframe_keypoint(kf, 0).expect("keypoint");
        let point_id = map
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                CompactDescriptor([0; 256]),
                kp,
            )
            .expect("point");

        let blend = DescriptorBlend::try_new(0.25).expect("blend");
        map.update_map_point_descriptor(point_id, &CompactDescriptor([255; 256]), blend)
            .expect("update");
        let stored = map.point(point_id).expect("point").descriptor();
        assert_eq!(stored.0[0], 64);
        assert_eq!(stored.0[255], 64);
    }

    #[test]
    fn descriptor_update_preserves_map_invariants() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = Pose::identity();
        let kf = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                pose,
                size,
                make_keypoints(1),
            )
            .expect("keyframe");
        let kp = map.keyframe_keypoint(kf, 0).expect("keypoint");
        let point_id = map
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                CompactDescriptor([10; 256]),
                kp,
            )
            .expect("point");
        map.update_map_point_descriptor(
            point_id,
            &CompactDescriptor([240; 256]),
            DescriptorBlend::try_new(0.5).expect("blend"),
        )
        .expect("update");
        assert_map_invariants(&map).expect("invariants");
    }

    #[test]
    fn map_generation_increments_on_mutation() {
        let mut map = SlamMap::new();
        assert_eq!(map.generation().as_u64(), 0);

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
        assert_eq!(map.generation().as_u64(), 1);

        let obs1 = map.keyframe_keypoint(kf1, 0).expect("obs1");
        map.add_map_point(
            Point3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            make_descriptor(),
            obs1,
        )
        .expect("map point");
        assert_eq!(map.generation().as_u64(), 2);
    }

    #[test]
    fn map_clone_preserves_generation() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = Pose::identity();
        let _ = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                pose,
                size,
                make_keypoints(1),
            )
            .expect("keyframe1");

        let cloned = map.clone();
        assert_eq!(cloned.generation(), map.generation());
    }

    #[test]
    fn covisibility_increments_and_decrements_on_map_point_changes() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = Pose::identity();
        assert_map_invariants(&map).expect("empty map invariants");

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
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                obs1,
            )
            .expect("map point");

        let obs2 = map.keyframe_keypoint(kf2, 0).expect("obs2");
        map.add_observation(point, obs2)
            .expect("second observation");

        assert_eq!(map.covisibility().covisibility_count(kf1, kf2), 1);
        assert_map_invariants(&map).expect("after shared observation");

        map.remove_map_point(point).expect("remove point");
        assert_eq!(map.covisibility().covisibility_count(kf1, kf2), 0);
        assert_map_invariants(&map).expect("after point removal");
    }

    #[test]
    fn duplicate_observation_is_rejected() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = Pose::identity();
        assert_map_invariants(&map).expect("empty map invariants");

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
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                obs1,
            )
            .expect("map point");

        let obs2 = map.keyframe_keypoint(kf2, 0).expect("obs2");
        map.add_observation(point, obs2)
            .expect("second observation");
        assert_map_invariants(&map).expect("after shared observation");

        let obs2_alt = map.keyframe_keypoint(kf2, 1).expect("obs2_alt");
        let err = map
            .add_observation(point, obs2_alt)
            .expect_err("duplicate observation");
        match err {
            MapError::DuplicateObservation {
                point_id,
                keyframe_id,
            } => {
                assert_eq!(point_id, point);
                assert_eq!(keyframe_id, kf2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
        assert_map_invariants(&map).expect("after duplicate rejection");
    }

    #[test]
    fn remove_keyframe_removes_orphaned_points() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = Pose::identity();
        assert_map_invariants(&map).expect("empty map invariants");

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
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                obs1,
            )
            .expect("map point");
        assert!(map.point(point).is_some());
        assert_map_invariants(&map).expect("after point insertion");

        map.remove_keyframe(kf1).expect("remove keyframe");
        assert_eq!(map.num_keyframes(), 0);
        assert_eq!(map.num_points(), 0);
        assert_map_invariants(&map).expect("after keyframe removal");
    }

    #[test]
    fn covisibility_updates_for_shared_points() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = Pose::identity();
        assert_map_invariants(&map).expect("empty map invariants");

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
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                obs1,
            )
            .expect("point A");
        let obs2 = map.keyframe_keypoint(kf2, 0).expect("obs2");
        map.add_observation(point_a, obs2).expect("obs2 add");
        let obs3 = map.keyframe_keypoint(kf3, 0).expect("obs3");
        map.add_observation(point_a, obs3).expect("obs3 add");
        assert_map_invariants(&map).expect("after first shared point");

        assert_eq!(map.covisibility().covisibility_count(kf1, kf2), 1);
        assert_eq!(map.covisibility().covisibility_count(kf1, kf3), 1);
        assert_eq!(map.covisibility().covisibility_count(kf2, kf3), 1);

        let obs1b = map.keyframe_keypoint(kf1, 1).expect("obs1b");
        let point_b = map
            .add_map_point(
                Point3 {
                    x: 1.0,
                    y: 0.0,
                    z: 2.0,
                },
                make_descriptor(),
                obs1b,
            )
            .expect("point B");
        let obs2b = map.keyframe_keypoint(kf2, 1).expect("obs2b");
        map.add_observation(point_b, obs2b).expect("obs2b add");
        assert_map_invariants(&map).expect("after second shared point");

        assert_eq!(map.covisibility().covisibility_count(kf1, kf2), 2);

        map.remove_map_point(point_b).expect("remove point B");
        assert_eq!(map.covisibility().covisibility_count(kf1, kf2), 1);
        assert_map_invariants(&map).expect("after point B removal");
    }
}
