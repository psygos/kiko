use std::collections::{HashMap, HashSet};
use std::num::{NonZeroU16, NonZeroU32, NonZeroUsize};
use std::sync::atomic::{AtomicU64, Ordering};

use slotmap::{SlotMap, new_key_type};

use crate::{
    CompactDescriptor, Detections, FrameId, Keypoint, SensorId, Timestamp, WorldPoint3,
    WorldToCamera,
};

/// Fixed-point scale factor for descriptor blending (8-bit precision).
const BLEND_SCALE: u16 = 256;
/// Smallest requested blend that rounds to a non-zero fixed-point weight.
const MIN_BLEND_ALPHA: f32 = 0.5 / BLEND_SCALE as f32;
/// Rounding bias for fixed-point descriptor blending (half of BLEND_SCALE).
const BLEND_ROUND: u32 = (BLEND_SCALE / 2) as u32;
static NEXT_MAP_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);

new_key_type! {
    struct RawMapPointId;
    struct RawKeyframeId;
}

/// A map point identifier scoped to the [`SlamMap`] that created it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MapPointId {
    instance_id: MapInstanceId,
    key: RawMapPointId,
}

impl MapPointId {
    fn new(instance_id: MapInstanceId, key: RawMapPointId) -> Self {
        Self { instance_id, key }
    }

    fn raw_for(self, instance_id: MapInstanceId) -> Option<RawMapPointId> {
        (self.instance_id == instance_id).then_some(self.key)
    }

    pub fn map_instance_id(self) -> MapInstanceId {
        self.instance_id
    }
}

impl Default for MapPointId {
    fn default() -> Self {
        Self::new(MapInstanceId(0), RawMapPointId::default())
    }
}

impl std::fmt::Debug for MapPointId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapPointId")
            .field("map", &self.instance_id.0)
            .field("key", &self.key)
            .finish()
    }
}

/// A keyframe identifier scoped to the [`SlamMap`] that created it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KeyframeId {
    instance_id: MapInstanceId,
    key: RawKeyframeId,
}

impl KeyframeId {
    fn new(instance_id: MapInstanceId, key: RawKeyframeId) -> Self {
        Self { instance_id, key }
    }

    fn raw_for(self, instance_id: MapInstanceId) -> Option<RawKeyframeId> {
        (self.instance_id == instance_id).then_some(self.key)
    }

    pub fn map_instance_id(self) -> MapInstanceId {
        self.instance_id
    }

    #[cfg(test)]
    pub(crate) fn for_test(index: usize) -> Self {
        let mut slots = SlotMap::<RawKeyframeId, ()>::with_key();
        let mut key = RawKeyframeId::default();
        for _ in 0..=index {
            key = slots.insert(());
        }
        Self::new(MapInstanceId(0), key)
    }
}

impl Default for KeyframeId {
    fn default() -> Self {
        Self::new(MapInstanceId(0), RawKeyframeId::default())
    }
}

impl std::fmt::Debug for KeyframeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyframeId")
            .field("map", &self.instance_id.0)
            .field("key", &self.key)
            .finish()
    }
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

/// A descriptor blend parsed into a nonzero fixed-point weight with `1 / 256` resolution.
///
/// [`DescriptorBlend::try_new`] rounds a requested coefficient to the nearest
/// multiple of `1 / 256`; exact half steps round upward because coefficients
/// are positive. [`DescriptorBlend::alpha`] reports that effective coefficient,
/// not the pre-quantization request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorBlend(NonZeroU16);

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlendError {
    OutOfRange { alpha: f32 },
    BelowResolution { alpha: f32, minimum: f32 },
}

impl std::fmt::Display for BlendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlendError::OutOfRange { alpha } => {
                write!(f, "descriptor blend must be in (0, 1], got {alpha}")
            }
            BlendError::BelowResolution { alpha, minimum } => write!(
                f,
                "descriptor blend {alpha} is below the fixed-point rounding threshold {minimum}"
            ),
        }
    }
}

impl std::error::Error for BlendError {}

impl DescriptorBlend {
    pub fn try_new(alpha: f32) -> Result<Self, BlendError> {
        if !(alpha > 0.0 && alpha <= 1.0) {
            return Err(BlendError::OutOfRange { alpha });
        }

        let weight = (alpha * f32::from(BLEND_SCALE)).round() as u16;
        let weight = NonZeroU16::new(weight).ok_or(BlendError::BelowResolution {
            alpha,
            minimum: MIN_BLEND_ALPHA,
        })?;
        Ok(Self(weight))
    }

    /// Returns the effective coefficient after fixed-point quantization.
    pub fn alpha(self) -> f32 {
        f32::from(self.0.get()) / f32::from(BLEND_SCALE)
    }

    fn weight(self) -> u16 {
        self.0.get()
    }
}

#[derive(Clone, Debug)]
pub struct MapPoint {
    position: WorldPoint3,
    descriptor: CompactDescriptor,
    observations: Vec<KeyframeKeypoint>,
}

impl MapPoint {
    pub fn position(&self) -> WorldPoint3 {
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
        let alpha_scaled = blend.weight();
        let inv_scaled = BLEND_SCALE - alpha_scaled;
        for (dst, &src) in self.descriptor.0.iter_mut().zip(new_desc.0.iter()) {
            let prev = *dst as u32;
            let next = src as u32;
            let mixed = prev * inv_scaled as u32 + next * alpha_scaled as u32;
            *dst = ((mixed + BLEND_ROUND) / BLEND_SCALE as u32) as u8;
        }
    }

    fn set_position(&mut self, pos: WorldPoint3) {
        self.position = pos;
    }
}

#[derive(Clone, Debug)]
pub struct KeyframeEntry {
    frame_id: FrameId,
    timestamp: Timestamp,
    pose: WorldToCamera,
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

    pub fn pose(&self) -> WorldToCamera {
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

    fn set_pose(&mut self, pose: WorldToCamera) {
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
            *weight = weight.saturating_add(1);
        } else {
            entry.insert(b, NonZeroU32::MIN);
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
                match NonZeroU32::new(weight.get().saturating_sub(1)) {
                    Some(next) => {
                        neighbors.insert(b, next);
                    }
                    None => {
                        neighbors.remove(&b);
                    }
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
        for (i, obs_a) in observations.iter().enumerate() {
            for obs_b in &observations[i + 1..] {
                self.decrement_pair(obs_a.keyframe_id, obs_b.keyframe_id);
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
    InvalidMapPointPosition(crate::Point3Error),
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
            MapError::InvalidMapPointPosition(err) => {
                write!(f, "invalid map point position: {err}")
            }
        }
    }
}

impl std::error::Error for MapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidMapPointPosition(err) => Some(err),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MapGeneration(u64);

impl MapGeneration {
    fn initial() -> Self {
        Self(0)
    }

    fn next(self) -> Self {
        Self(
            self.0
                .checked_add(1)
                .expect("map generation space exhausted"),
        )
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MapInstanceId(u64);

impl MapInstanceId {
    fn next() -> Self {
        let value = NEXT_MAP_INSTANCE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("map instance ID space exhausted");
        Self(value)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MapSnapshot {
    instance_id: MapInstanceId,
    generation: MapGeneration,
}

impl MapSnapshot {
    pub fn instance_id(self) -> MapInstanceId {
        self.instance_id
    }

    pub fn generation(self) -> MapGeneration {
        self.generation
    }
}

#[derive(Clone, Debug)]
pub struct SlamMap {
    instance_id: MapInstanceId,
    points: SlotMap<RawMapPointId, MapPoint>,
    keyframes: SlotMap<RawKeyframeId, KeyframeEntry>,
    covisibility: CovisibilityGraph,
    frame_to_keyframe: HashMap<FrameId, KeyframeId>,
    generation: MapGeneration,
}

#[derive(Debug, Clone)]
pub struct CovisibilityNode {
    pub id: KeyframeId,
    pub pose: WorldToCamera,
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
            instance_id: MapInstanceId::next(),
            points: SlotMap::with_key(),
            keyframes: SlotMap::with_key(),
            covisibility: CovisibilityGraph::default(),
            frame_to_keyframe: HashMap::new(),
            generation: MapGeneration::initial(),
        }
    }

    fn raw_keyframe_id(&self, id: KeyframeId) -> Result<RawKeyframeId, MapError> {
        id.raw_for(self.instance_id)
            .ok_or(MapError::KeyframeNotFound(id))
    }

    fn raw_point_id(&self, id: MapPointId) -> Result<RawMapPointId, MapError> {
        id.raw_for(self.instance_id)
            .ok_or(MapError::MapPointNotFound(id))
    }

    pub fn add_keyframe_from_detections(
        &mut self,
        detections: &Detections,
        timestamp: Timestamp,
        pose: WorldToCamera,
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
        pose: WorldToCamera,
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

        let next_generation = self.generation.next();
        let kf_id = KeyframeId::new(self.instance_id, self.keyframes.insert(entry));
        self.frame_to_keyframe.insert(frame_id, kf_id);
        self.generation = next_generation;
        Ok(kf_id)
    }

    pub fn keyframe_keypoint(
        &self,
        keyframe_id: KeyframeId,
        index: usize,
    ) -> Result<KeyframeKeypoint, MapError> {
        let raw_keyframe_id = self.raw_keyframe_id(keyframe_id)?;
        let entry = self
            .keyframes
            .get(raw_keyframe_id)
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
        let raw_keyframe_id = self.raw_keyframe_id(keypoint.keyframe_id)?;
        let entry = self
            .keyframes
            .get(raw_keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keypoint.keyframe_id))?;
        Ok(entry.keypoint(keypoint.index))
    }

    pub fn add_map_point(
        &mut self,
        position: WorldPoint3,
        descriptor: CompactDescriptor,
        first_obs: KeyframeKeypoint,
    ) -> Result<MapPointId, MapError> {
        let position = position
            .validate()
            .map_err(MapError::InvalidMapPointPosition)?;
        let raw_keyframe_id = self.raw_keyframe_id(first_obs.keyframe_id)?;
        let entry = self
            .keyframes
            .get(raw_keyframe_id)
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

        let next_generation = self.generation.next();
        let point_id = MapPointId::new(
            self.instance_id,
            self.points.insert(MapPoint {
                position,
                descriptor,
                observations: vec![first_obs],
            }),
        );

        let entry = self
            .keyframes
            .get_mut(raw_keyframe_id)
            .ok_or(MapError::KeyframeNotFound(first_obs.keyframe_id))?;
        entry.set_point_ref(first_obs.index, point_id);
        self.generation = next_generation;
        Ok(point_id)
    }

    pub fn add_observation(
        &mut self,
        point_id: MapPointId,
        obs: KeyframeKeypoint,
    ) -> Result<(), MapError> {
        let raw_keyframe_id = self.raw_keyframe_id(obs.keyframe_id)?;
        let entry = self
            .keyframes
            .get(raw_keyframe_id)
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

        let raw_point_id = self.raw_point_id(point_id)?;
        let other_keyframes: Vec<KeyframeId> = {
            let point = self
                .points
                .get(raw_point_id)
                .ok_or(MapError::MapPointNotFound(point_id))?;
            if point.observes_keyframe(obs.keyframe_id) {
                return Err(MapError::DuplicateObservation {
                    point_id,
                    keyframe_id: obs.keyframe_id,
                });
            }
            point.observations.iter().map(|o| o.keyframe_id).collect()
        };

        let next_generation = self.generation.next();
        for other in other_keyframes {
            self.covisibility.increment_pair(obs.keyframe_id, other);
        }

        let point = self
            .points
            .get_mut(raw_point_id)
            .ok_or(MapError::MapPointNotFound(point_id))?;
        point.add_observation(obs);

        let entry = self
            .keyframes
            .get_mut(raw_keyframe_id)
            .ok_or(MapError::KeyframeNotFound(obs.keyframe_id))?;
        entry.set_point_ref(obs.index, point_id);
        self.generation = next_generation;
        Ok(())
    }

    pub fn update_map_point_descriptor(
        &mut self,
        point_id: MapPointId,
        new_desc: &CompactDescriptor,
        blend: DescriptorBlend,
    ) -> Result<(), MapError> {
        let raw_point_id = self.raw_point_id(point_id)?;
        let point = self
            .points
            .get_mut(raw_point_id)
            .ok_or(MapError::MapPointNotFound(point_id))?;
        let next_generation = self.generation.next();
        point.update_descriptor(new_desc, blend);
        self.generation = next_generation;
        Ok(())
    }

    pub fn set_map_point_position(
        &mut self,
        point_id: MapPointId,
        position: WorldPoint3,
    ) -> Result<(), MapError> {
        let position = position
            .validate()
            .map_err(MapError::InvalidMapPointPosition)?;
        let raw_point_id = self.raw_point_id(point_id)?;
        let point = self
            .points
            .get_mut(raw_point_id)
            .ok_or(MapError::MapPointNotFound(point_id))?;
        let next_generation = self.generation.next();
        point.set_position(position);
        self.generation = next_generation;
        Ok(())
    }

    pub fn set_keyframe_pose(
        &mut self,
        keyframe_id: KeyframeId,
        pose: WorldToCamera,
    ) -> Result<(), MapError> {
        let raw_keyframe_id = self.raw_keyframe_id(keyframe_id)?;
        let entry = self
            .keyframes
            .get_mut(raw_keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        let next_generation = self.generation.next();
        entry.set_pose(pose);
        self.generation = next_generation;
        Ok(())
    }

    pub fn remove_map_point(&mut self, point_id: MapPointId) -> Result<(), MapError> {
        let raw_point_id = self.raw_point_id(point_id)?;
        if !self.points.contains_key(raw_point_id) {
            return Err(MapError::MapPointNotFound(point_id));
        }
        let next_generation = self.generation.next();
        let point = self
            .points
            .remove(raw_point_id)
            .expect("map point existence validated before mutation");

        for obs in &point.observations {
            let Some(raw_keyframe_id) = obs.keyframe_id.raw_for(self.instance_id) else {
                continue;
            };
            if let Some(entry) = self.keyframes.get_mut(raw_keyframe_id) {
                entry.clear_point_ref(obs.index);
            }
        }
        self.covisibility
            .remove_point_observations(&point.observations);
        self.generation = next_generation;
        Ok(())
    }

    pub fn remove_keyframe(&mut self, keyframe_id: KeyframeId) -> Result<(), MapError> {
        let raw_keyframe_id = self.raw_keyframe_id(keyframe_id)?;
        if !self.keyframes.contains_key(raw_keyframe_id) {
            return Err(MapError::KeyframeNotFound(keyframe_id));
        }
        let next_generation = self.generation.next();
        let entry = self
            .keyframes
            .remove(raw_keyframe_id)
            .expect("keyframe existence validated before mutation");
        self.frame_to_keyframe.remove(&entry.frame_id);
        self.covisibility.remove_keyframe(keyframe_id);

        let mut to_remove = Vec::new();
        for point_id in entry.map_point_ids() {
            let Some(raw_point_id) = point_id.raw_for(self.instance_id) else {
                continue;
            };
            if let Some(point) = self.points.get_mut(raw_point_id) {
                point.remove_observation_for(keyframe_id);
                if point.observations.is_empty() {
                    to_remove.push(raw_point_id);
                }
            }
        }
        for point_id in to_remove {
            let removed = self.points.remove(point_id);
            debug_assert!(
                removed.is_some(),
                "point scheduled for removal was missing from map"
            );
        }
        self.generation = next_generation;
        Ok(())
    }

    pub fn cull_points(&mut self, min_observations: usize) -> usize {
        let to_remove: Vec<MapPointId> = self
            .points
            .iter()
            .filter(|(_, p)| p.observation_count() < min_observations)
            .map(|(id, _)| MapPointId::new(self.instance_id, id))
            .collect();
        let count = to_remove.len();
        for id in to_remove {
            let removed = self.remove_map_point(id);
            debug_assert!(removed.is_ok(), "map point missing during cull");
        }
        count
    }

    pub fn keyframe(&self, id: KeyframeId) -> Option<&KeyframeEntry> {
        id.raw_for(self.instance_id)
            .and_then(|raw_id| self.keyframes.get(raw_id))
    }

    pub fn keyframe_observation_pixels(
        &self,
        keyframe_id: KeyframeId,
    ) -> Result<Vec<(KeyframeKeypoint, Keypoint)>, MapError> {
        let raw_keyframe_id = self.raw_keyframe_id(keyframe_id)?;
        let entry = self
            .keyframes
            .get(raw_keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        entry
            .point_refs
            .iter()
            .enumerate()
            .filter_map(|(idx, pid)| pid.map(|_| idx))
            .map(|idx| {
                let index = KeypointIndex::new(idx, entry.len()).ok_or(
                    MapError::KeypointIndexOutOfBounds {
                        index: idx,
                        len: entry.len(),
                    },
                )?;
                let keypoint_ref = KeyframeKeypoint { keyframe_id, index };
                Ok((keypoint_ref, entry.keypoints[idx]))
            })
            .collect()
    }

    pub fn keyframe_point_descriptors(
        &self,
        keyframe_id: KeyframeId,
    ) -> Result<Vec<(KeyframeKeypoint, CompactDescriptor)>, MapError> {
        let raw_keyframe_id = self.raw_keyframe_id(keyframe_id)?;
        let entry = self
            .keyframes
            .get(raw_keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        let mut descriptors = Vec::new();
        for (idx, point_ref) in entry.point_refs.iter().enumerate() {
            let Some(point_id) = point_ref else {
                continue;
            };
            let raw_point_id = self.raw_point_id(*point_id)?;
            let point = self
                .points
                .get(raw_point_id)
                .ok_or(MapError::MapPointNotFound(*point_id))?;
            let index =
                KeypointIndex::new(idx, entry.len()).ok_or(MapError::KeypointIndexOutOfBounds {
                    index: idx,
                    len: entry.len(),
                })?;
            descriptors.push((
                KeyframeKeypoint { keyframe_id, index },
                point.descriptor().clone(),
            ));
        }
        Ok(descriptors)
    }

    pub fn keyframe_point_count(&self, keyframe_id: KeyframeId) -> Result<usize, MapError> {
        let raw_keyframe_id = self.raw_keyframe_id(keyframe_id)?;
        let entry = self
            .keyframes
            .get(raw_keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        Ok(entry.map_point_ids().count())
    }

    pub fn covisible_window(
        &self,
        seed: KeyframeId,
        max: NonZeroUsize,
    ) -> Result<Vec<KeyframeId>, MapError> {
        let raw_seed = self.raw_keyframe_id(seed)?;
        if !self.keyframes.contains_key(raw_seed) {
            return Err(MapError::KeyframeNotFound(seed));
        }
        let mut window = Vec::with_capacity(max.get());
        window.push(seed);

        let neighbors = match self.covisibility.neighbors(seed) {
            Some(neighbors) => neighbors,
            None => return Ok(window),
        };

        let mut sorted: Vec<(KeyframeId, NonZeroU32)> =
            neighbors.iter().map(|(&id, &w)| (id, w)).collect();
        sorted.sort_by_key(|entry| std::cmp::Reverse(entry.1.get()));

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
        let raw_keyframe_id = self.raw_keyframe_id(keypoint.keyframe_id)?;
        let entry = self
            .keyframes
            .get(raw_keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keypoint.keyframe_id))?;
        Ok(entry.point_ref(keypoint.index))
    }

    pub fn point(&self, id: MapPointId) -> Option<&MapPoint> {
        id.raw_for(self.instance_id)
            .and_then(|raw_id| self.points.get(raw_id))
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
                id: KeyframeId::new(self.instance_id, id),
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

    pub fn snapshot(&self) -> MapSnapshot {
        MapSnapshot {
            instance_id: self.instance_id,
            generation: self.generation,
        }
    }

    pub fn points(&self) -> impl Iterator<Item = (MapPointId, &MapPoint)> {
        self.points
            .iter()
            .map(|(id, point)| (MapPointId::new(self.instance_id, id), point))
    }

    pub fn keyframes(&self) -> impl Iterator<Item = (KeyframeId, &KeyframeEntry)> {
        self.keyframes
            .iter()
            .map(|(id, keyframe)| (KeyframeId::new(self.instance_id, id), keyframe))
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
        let Some(entry) = map.keyframe(keyframe_id) else {
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

    for (keyframe_id, entry) in map.keyframes() {
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

            let Some(point) = map.point(point_id) else {
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
    for (point_id, point) in map.points() {
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

            let Some(entry) = map.keyframe(obs.keyframe_id) else {
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

        for (i, obs_a) in point.observations.iter().enumerate() {
            for obs_b in &point.observations[i + 1..] {
                *expected_covisibility
                    .entry((obs_a.keyframe_id, obs_b.keyframe_id))
                    .or_insert(0) += 1;
                *expected_covisibility
                    .entry((obs_b.keyframe_id, obs_a.keyframe_id))
                    .or_insert(0) += 1;
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
    use crate::{CompactDescriptor, Keypoint, Point3, Timestamp, WorldToCamera};

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

    fn assert_generation_exhaustion<T>(operation: impl FnOnce() -> T) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
        let Err(payload) = result else {
            panic!("generation exhaustion must panic before mutation");
        };
        assert_eq!(
            crate::panic_payload_to_string(payload.as_ref()),
            "map generation space exhausted"
        );
    }

    struct GenerationExhaustionFixture {
        map: SlamMap,
        snapshot: MapSnapshot,
        first_keyframe: KeyframeId,
        second_keyframe: KeyframeId,
        point_id: MapPointId,
        first_observation: KeyframeKeypoint,
        first_free_keypoint: KeyframeKeypoint,
        second_free_keypoint: KeyframeKeypoint,
        original_position: WorldPoint3,
        original_descriptor: CompactDescriptor,
        original_pose: WorldToCamera,
    }

    impl GenerationExhaustionFixture {
        fn new() -> Self {
            let mut map = SlamMap::new();
            let size = ImageSize::try_new(640, 480).expect("valid size");
            let original_pose = WorldToCamera::identity();
            let first_keyframe = map
                .add_keyframe(
                    FrameId::new(1),
                    Timestamp::from_nanos(1),
                    original_pose,
                    size,
                    make_keypoints(2),
                )
                .expect("first keyframe");
            let second_keyframe = map
                .add_keyframe(
                    FrameId::new(2),
                    Timestamp::from_nanos(2),
                    original_pose,
                    size,
                    make_keypoints(1),
                )
                .expect("second keyframe");
            let first_observation = map
                .keyframe_keypoint(first_keyframe, 0)
                .expect("first observation");
            let first_free_keypoint = map
                .keyframe_keypoint(first_keyframe, 1)
                .expect("first free keypoint");
            let second_free_keypoint = map
                .keyframe_keypoint(second_keyframe, 0)
                .expect("second free keypoint");
            let original_position = WorldPoint3::new(0.0, 0.0, 1.0);
            let original_descriptor = make_descriptor();
            let point_id = map
                .add_map_point(
                    original_position,
                    original_descriptor.clone(),
                    first_observation,
                )
                .expect("map point");

            map.generation = MapGeneration(u64::MAX);
            let snapshot = map.snapshot();
            Self {
                map,
                snapshot,
                first_keyframe,
                second_keyframe,
                point_id,
                first_observation,
                first_free_keypoint,
                second_free_keypoint,
                original_position,
                original_descriptor,
                original_pose,
            }
        }

        fn assert_unchanged(&self) {
            assert_eq!(self.map.snapshot(), self.snapshot);
            assert_eq!(self.map.num_keyframes(), 2);
            assert_eq!(self.map.num_points(), 1);
            assert_eq!(
                self.map.keyframe_by_frame(FrameId::new(1)),
                Some(self.first_keyframe)
            );
            assert_eq!(
                self.map.keyframe_by_frame(FrameId::new(2)),
                Some(self.second_keyframe)
            );

            let point = self.map.point(self.point_id).expect("original map point");
            assert_eq!(point.position(), self.original_position);
            assert_eq!(point.descriptor(), &self.original_descriptor);
            assert_eq!(point.observations(), &[self.first_observation]);
            assert_eq!(
                self.map
                    .map_point_for_keypoint(self.first_observation)
                    .expect("associated keypoint"),
                Some(self.point_id)
            );
            assert_eq!(
                self.map
                    .map_point_for_keypoint(self.first_free_keypoint)
                    .expect("first free keypoint"),
                None
            );
            assert_eq!(
                self.map
                    .map_point_for_keypoint(self.second_free_keypoint)
                    .expect("second free keypoint"),
                None
            );
            assert_eq!(
                self.map
                    .covisibility()
                    .covisibility_count(self.first_keyframe, self.second_keyframe),
                0
            );

            let pose = self
                .map
                .keyframe(self.first_keyframe)
                .expect("first keyframe")
                .pose();
            assert_eq!(pose.rotation(), self.original_pose.rotation());
            assert_eq!(pose.translation(), self.original_pose.translation());
            assert_map_invariants(&self.map).expect("map invariants after rejected mutation");
        }
    }

    fn reference_blend_byte(previous: u8, next: u8, weight: u16) -> u8 {
        let inverse = BLEND_SCALE - weight;
        let numerator = u64::from(previous) * u64::from(inverse)
            + u64::from(next) * u64::from(weight)
            + u64::from(BLEND_SCALE / 2);
        u8::try_from(numerator / u64::from(BLEND_SCALE))
            .expect("a convex combination of bytes remains a byte")
    }

    #[test]
    fn descriptor_blend_rejects_non_finite_and_out_of_range_requests() {
        let requests = [
            f32::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            f32::from_bits(1.0_f32.to_bits() + 1),
            f32::INFINITY,
            f32::NAN,
        ];

        for requested in requests {
            let BlendError::OutOfRange { alpha } =
                DescriptorBlend::try_new(requested).expect_err("request must be rejected")
            else {
                panic!("unexpected blend error for {requested:?}");
            };
            assert_eq!(alpha.to_bits(), requested.to_bits());
        }
    }

    #[test]
    fn descriptor_blend_rejects_weights_below_fixed_point_resolution() {
        let just_below = f32::from_bits(MIN_BLEND_ALPHA.to_bits() - 1);
        for requested in [f32::from_bits(1), just_below] {
            let BlendError::BelowResolution { alpha, minimum } =
                DescriptorBlend::try_new(requested).expect_err("request must round to zero")
            else {
                panic!("unexpected blend error for {requested:?}");
            };
            assert_eq!(alpha.to_bits(), requested.to_bits());
            assert_eq!(minimum, MIN_BLEND_ALPHA);
        }
    }

    #[test]
    fn descriptor_blend_rounds_once_and_reports_its_effective_alpha() {
        let minimum = DescriptorBlend::try_new(MIN_BLEND_ALPHA).expect("half step rounds upward");
        assert_eq!(minimum.weight(), 1);
        assert_eq!(minimum.alpha(), 1.0 / f32::from(BLEND_SCALE));

        let second_half_step = 1.5 / f32::from(BLEND_SCALE);
        let just_below = f32::from_bits(second_half_step.to_bits() - 1);
        assert_eq!(
            DescriptorBlend::try_new(just_below)
                .expect("positive representable blend")
                .weight(),
            1
        );
        assert_eq!(
            DescriptorBlend::try_new(second_half_step)
                .expect("exact half step rounds upward")
                .weight(),
            2
        );

        let off_grid = DescriptorBlend::try_new(0.1).expect("in-range blend");
        assert_eq!(off_grid.weight(), 26);
        assert_eq!(off_grid.alpha(), 26.0 / f32::from(BLEND_SCALE));

        let full = DescriptorBlend::try_new(1.0).expect("full replacement");
        assert_eq!(full.weight(), BLEND_SCALE);
        assert_eq!(full.alpha(), 1.0);
    }

    #[test]
    fn descriptor_blend_matches_integer_reference_for_every_fixed_point_weight() {
        let previous = CompactDescriptor(std::array::from_fn(|index| {
            u8::try_from(index).expect("descriptor index fits in a byte")
        }));
        let next = CompactDescriptor(std::array::from_fn(|index| {
            let value = (index * 73 + 19) % 256;
            u8::try_from(value).expect("value is reduced modulo 256")
        }));

        for weight in 1..=BLEND_SCALE {
            let blend = DescriptorBlend::try_new(f32::from(weight) / f32::from(BLEND_SCALE))
                .expect("fixed-point grid value");
            assert_eq!(blend.weight(), weight);

            let expected = std::array::from_fn(|index| {
                reference_blend_byte(previous.0[index], next.0[index], weight)
            });
            let mut point = MapPoint {
                position: Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                descriptor: previous.clone(),
                observations: Vec::new(),
            };
            point.update_descriptor(&next, blend);
            assert_eq!(point.descriptor.0, expected, "weight {weight}");
        }
    }

    #[test]
    fn descriptor_blend_uses_u8_weighted_average() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = WorldToCamera::identity();
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
        let pose = WorldToCamera::identity();
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
        let pose = WorldToCamera::identity();
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
    fn nonfinite_map_point_positions_are_rejected_transactionally() {
        let mut map = SlamMap::new();
        let keyframe_id = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                WorldToCamera::identity(),
                ImageSize::try_new(640, 480).expect("valid size"),
                make_keypoints(1),
            )
            .expect("keyframe");
        let observation = map
            .keyframe_keypoint(keyframe_id, 0)
            .expect("keyframe keypoint");
        let before_insert = map.snapshot();

        let malformed = WorldPoint3::new(0.0, f32::NAN, 1.0);
        assert!(matches!(
            map.add_map_point(malformed, make_descriptor(), observation),
            Err(MapError::InvalidMapPointPosition(crate::Point3Error::NonFinite {
                axis: 1,
                value,
            })) if value.is_nan()
        ));
        assert_eq!(map.snapshot(), before_insert);
        assert_eq!(map.num_points(), 0);
        assert_eq!(
            map.map_point_for_keypoint(observation)
                .expect("valid keypoint reference"),
            None
        );

        let original = WorldPoint3::new(1.0, 2.0, 3.0);
        let point_id = map
            .add_map_point(original, make_descriptor(), observation)
            .expect("valid map point");
        let before_update = map.snapshot();
        assert!(matches!(
            map.set_map_point_position(point_id, WorldPoint3::new(f32::INFINITY, 2.0, 3.0),),
            Err(MapError::InvalidMapPointPosition(
                crate::Point3Error::NonFinite {
                    axis: 0,
                    value: f32::INFINITY,
                }
            ))
        ));
        assert_eq!(map.snapshot(), before_update);
        assert_eq!(map.point(point_id).expect("map point").position(), original);
    }

    #[test]
    fn map_generation_exhaustion_precedes_every_map_mutation() {
        let mut fixture = GenerationExhaustionFixture::new();

        assert_generation_exhaustion(|| {
            fixture.map.add_keyframe(
                FrameId::new(3),
                Timestamp::from_nanos(3),
                WorldToCamera::identity(),
                ImageSize::try_new(640, 480).expect("valid size"),
                make_keypoints(1),
            )
        });
        fixture.assert_unchanged();

        let first_free_keypoint = fixture.first_free_keypoint;
        assert_generation_exhaustion(|| {
            fixture.map.add_map_point(
                WorldPoint3::new(1.0, 2.0, 3.0),
                CompactDescriptor([255; 256]),
                first_free_keypoint,
            )
        });
        fixture.assert_unchanged();

        let point_id = fixture.point_id;
        let second_free_keypoint = fixture.second_free_keypoint;
        assert_generation_exhaustion(|| {
            fixture.map.add_observation(point_id, second_free_keypoint)
        });
        fixture.assert_unchanged();

        assert_generation_exhaustion(|| {
            fixture.map.update_map_point_descriptor(
                point_id,
                &CompactDescriptor([255; 256]),
                DescriptorBlend::try_new(1.0).expect("full replacement"),
            )
        });
        fixture.assert_unchanged();

        assert_generation_exhaustion(|| {
            fixture
                .map
                .set_map_point_position(point_id, WorldPoint3::new(1.0, 2.0, 3.0))
        });
        fixture.assert_unchanged();

        let first_keyframe = fixture.first_keyframe;
        let changed_pose = WorldToCamera::from_legacy_pose(crate::Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 2.0, 3.0],
        ));
        assert_generation_exhaustion(|| {
            fixture.map.set_keyframe_pose(first_keyframe, changed_pose)
        });
        fixture.assert_unchanged();

        assert_generation_exhaustion(|| fixture.map.remove_map_point(point_id));
        fixture.assert_unchanged();

        assert_generation_exhaustion(|| fixture.map.remove_keyframe(first_keyframe));
        fixture.assert_unchanged();

        assert_generation_exhaustion(|| fixture.map.cull_points(2));
        fixture.assert_unchanged();
    }

    #[test]
    fn map_clone_preserves_generation() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = WorldToCamera::identity();
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
        assert_eq!(cloned.snapshot(), map.snapshot());
        assert_ne!(
            SlamMap::new().snapshot().instance_id(),
            map.snapshot().instance_id()
        );
    }

    #[test]
    fn ids_from_another_map_cannot_resolve_even_when_slots_match() {
        let mut map_a = SlamMap::new();
        let mut map_b = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");

        let keyframe_a = map_a
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                WorldToCamera::identity(),
                size,
                make_keypoints(1),
            )
            .expect("keyframe A");
        let keyframe_b = map_b
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                WorldToCamera::identity(),
                size,
                make_keypoints(1),
            )
            .expect("keyframe B");

        assert_eq!(keyframe_a.key, keyframe_b.key, "slot keys should collide");
        assert_ne!(keyframe_a, keyframe_b, "scoped IDs must remain distinct");
        assert!(map_b.keyframe(keyframe_a).is_none());
        assert!(matches!(
            map_b.keyframe_keypoint(keyframe_a, 0),
            Err(MapError::KeyframeNotFound(id)) if id == keyframe_a
        ));

        let observation_a = map_a
            .keyframe_keypoint(keyframe_a, 0)
            .expect("observation A");
        let observation_b = map_b
            .keyframe_keypoint(keyframe_b, 0)
            .expect("observation B");
        assert!(matches!(
            map_b.add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                observation_a,
            ),
            Err(MapError::KeyframeNotFound(id)) if id == keyframe_a
        ));
        let point_a = map_a
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                observation_a,
            )
            .expect("point A");
        let point_b = map_b
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                observation_b,
            )
            .expect("point B");

        assert_eq!(point_a.key, point_b.key, "slot keys should collide");
        assert_ne!(point_a, point_b, "scoped IDs must remain distinct");
        assert!(map_b.point(point_a).is_none());
        assert!(matches!(
            map_b.set_map_point_position(
                point_a,
                Point3 {
                    x: 1.0,
                    y: 0.0,
                    z: 1.0,
                }
            ),
            Err(MapError::MapPointNotFound(id)) if id == point_a
        ));
    }

    #[test]
    fn covisibility_increments_and_decrements_on_map_point_changes() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let pose = WorldToCamera::identity();
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
        let pose = WorldToCamera::identity();
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
        let pose = WorldToCamera::identity();
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
        let pose = WorldToCamera::identity();
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
