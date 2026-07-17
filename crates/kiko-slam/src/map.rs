use std::collections::{HashMap, HashSet};
use std::num::{NonZeroU16, NonZeroU32, NonZeroUsize};
use std::sync::atomic::{AtomicU64, Ordering};

use slotmap::{SlotMap, new_key_type};

use crate::{
    CompactDescriptor, Detections, FrameDimensions, FrameId, Keypoint, Point3, Pose, Pose64,
    Pose64Error, SensorId, Timestamp,
};

/// Fixed-point scale factor for descriptor blending (8-bit precision).
const BLEND_SCALE: u16 = 256;
/// Smallest requested blend that rounds to a non-zero fixed-point weight.
const MIN_BLEND_ALPHA: f32 = 0.5 / BLEND_SCALE as f32;
/// Rounding bias for fixed-point descriptor blending (half of BLEND_SCALE).
const BLEND_ROUND: u32 = (BLEND_SCALE / 2) as u32;
static NEXT_MAP_INSTANCE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_MAP_LINEAGE_ID: AtomicU64 = AtomicU64::new(1);

fn allocate_monotonic_id(counter: &AtomicU64) -> Option<u64> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .ok()
}

new_key_type! {
    pub struct MapPointId;
    pub struct KeyframeId;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MapInstanceId(u64);

impl MapInstanceId {
    fn fresh() -> Self {
        Self(allocate_monotonic_id(&NEXT_MAP_INSTANCE_ID).expect("map instance ID space exhausted"))
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MapLineageId(u64);

impl MapLineageId {
    fn fresh() -> Self {
        Self(allocate_monotonic_id(&NEXT_MAP_LINEAGE_ID).expect("map lineage ID space exhausted"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct KeypointIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeypointIndexError {
    OutOfBounds { index: usize, len: usize },
}

impl KeypointIndex {
    fn new(index: usize, len: usize) -> Result<Self, KeypointIndexError> {
        if index < len {
            Ok(Self(index))
        } else {
            Err(KeypointIndexError::OutOfBounds { index, len })
        }
    }

    fn as_usize(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KeyframeKeypoint {
    map_instance_id: MapInstanceId,
    keyframe_id: KeyframeId,
    index: KeypointIndex,
}

impl KeyframeKeypoint {
    pub fn map_instance_id(self) -> MapInstanceId {
        self.map_instance_id
    }

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
    position: FiniteMapPoint,
    descriptor: CompactDescriptor,
    observations: Vec<KeyframeKeypoint>,
}

impl MapPoint {
    pub fn position(&self) -> Point3 {
        self.position.get()
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

    fn set_position(&mut self, pos: FiniteMapPoint) {
        self.position = pos;
    }
}

#[derive(Clone, Copy, Debug)]
struct FiniteMapPoint(Point3);

impl FiniteMapPoint {
    fn try_new(point: Point3) -> Result<Self, MapError> {
        for (axis, value) in [("x", point.x), ("y", point.y), ("z", point.z)] {
            if !value.is_finite() {
                return Err(MapError::NonFiniteMapPoint { axis, value });
            }
        }
        Ok(Self(point))
    }

    fn get(self) -> Point3 {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct KeyframeEntry {
    frame_id: FrameId,
    timestamp: Timestamp,
    pose: Pose,
    image_size: FrameDimensions,
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

    pub fn image_size(&self) -> FrameDimensions {
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

    fn clear_point_ref(&mut self, index: KeypointIndex) -> Option<MapPointId> {
        self.point_refs[index.as_usize()].take()
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

fn covisibility_ratio_from_counts(shared: u32, denominator: usize) -> f32 {
    debug_assert_ne!(shared, 0);
    debug_assert!(shared as usize <= denominator);
    (f64::from(shared) / denominator as f64) as f32
}

#[derive(Debug)]
pub enum MapError {
    ForeignKeypoint {
        expected: MapInstanceId,
        actual: MapInstanceId,
    },
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
    EmptyKeyframe {
        frame_id: FrameId,
    },
    InvalidKeypoint {
        index: usize,
        axis: &'static str,
        value: f32,
        upper_bound: u32,
    },
    NonFiniteMapPoint {
        axis: &'static str,
        value: f32,
    },
    InvalidPose(Pose64Error),
    SensorMismatch {
        expected: SensorId,
        actual: SensorId,
    },
    InconsistentCovisibility {
        a: KeyframeId,
        b: KeyframeId,
        shared: u32,
        a_points: usize,
        b_points: usize,
    },
}

impl std::fmt::Display for MapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MapError::ForeignKeypoint { expected, actual } => write!(
                f,
                "keypoint belongs to map instance {}, not {}",
                actual.as_u64(),
                expected.as_u64()
            ),
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
            MapError::EmptyKeyframe { frame_id } => {
                write!(f, "keyframe {frame_id:?} has no keypoints")
            }
            MapError::InvalidKeypoint {
                index,
                axis,
                value,
                upper_bound,
            } => write!(
                f,
                "keyframe keypoint {index} has invalid {axis} coordinate {value}; expected finite value in [0, {upper_bound})"
            ),
            MapError::NonFiniteMapPoint { axis, value } => {
                write!(f, "map point {axis} coordinate must be finite, got {value}")
            }
            MapError::InvalidPose(source) => write!(f, "invalid map pose: {source}"),
            MapError::SensorMismatch { expected, actual } => write!(
                f,
                "keyframe detections must be from {expected:?}, got {actual:?}"
            ),
            MapError::InconsistentCovisibility {
                a,
                b,
                shared,
                a_points,
                b_points,
            } => write!(
                f,
                "covisibility count {shared} between {a:?} and {b:?} exceeds the smaller keyframe point count ({a_points}, {b_points})"
            ),
        }
    }
}

impl std::error::Error for MapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPose(source) => Some(source),
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
        self.advance_by(1)
    }

    fn advance_by(self, steps: usize) -> Self {
        let steps = u64::try_from(steps).expect("map generation space exhausted");
        Self(
            self.0
                .checked_add(steps)
                .expect("map generation space exhausted"),
        )
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// A process-local freshness token for one exact [`SlamMap`] revision.
///
/// Clones compare equal until either copy mutates. Generation remains monotonic
/// within one branch, while lineage distinguishes independently mutated clones
/// at the same generation. This is deliberately not a structural content hash.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MapSnapshot {
    instance_id: MapInstanceId,
    generation: MapGeneration,
    lineage: MapLineageId,
}

impl MapSnapshot {
    pub fn instance_id(self) -> MapInstanceId {
        self.instance_id
    }

    pub fn generation(self) -> MapGeneration {
        self.generation
    }

    pub fn is_same_or_older_than(self, current: Self) -> bool {
        self.instance_id == current.instance_id
            && self.lineage == current.lineage
            && self.generation <= current.generation
    }
}

impl std::fmt::Display for MapSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{}:{}",
            self.instance_id.as_u64(),
            self.generation.as_u64(),
            self.lineage.0
        )
    }
}

#[derive(Debug)]
pub struct SlamMap {
    instance_id: MapInstanceId,
    points: SlotMap<MapPointId, MapPoint>,
    keyframes: SlotMap<KeyframeId, KeyframeEntry>,
    covisibility: CovisibilityGraph,
    frame_to_keyframe: HashMap<FrameId, KeyframeId>,
    generation: MapGeneration,
    lineage: MapLineageId,
    mutation_lineage: MapLineageId,
}

impl Clone for SlamMap {
    fn clone(&self) -> Self {
        self.clone_with_mutation_lineage(MapLineageId::fresh())
    }
}

impl SlamMap {
    /// Clone this revision for an all-or-nothing mutation that will replace it.
    ///
    /// The caller must keep the source revision immutable and promote at most
    /// one candidate. Successful candidate mutations then remain a canonical
    /// continuation, so older asynchronous work can still be recognized as
    /// belonging to the same history. Ordinary [`Clone`] remains a true fork.
    pub(crate) fn clone_for_transaction(&self) -> Self {
        self.clone_with_mutation_lineage(self.mutation_lineage)
    }

    fn clone_with_mutation_lineage(&self, mutation_lineage: MapLineageId) -> Self {
        Self {
            instance_id: self.instance_id,
            points: self.points.clone(),
            keyframes: self.keyframes.clone(),
            covisibility: self.covisibility.clone(),
            frame_to_keyframe: self.frame_to_keyframe.clone(),
            generation: self.generation,
            lineage: self.lineage,
            mutation_lineage,
        }
    }
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
        let lineage = MapLineageId::fresh();
        Self {
            instance_id: MapInstanceId::fresh(),
            points: SlotMap::with_key(),
            keyframes: SlotMap::with_key(),
            covisibility: CovisibilityGraph::default(),
            frame_to_keyframe: HashMap::new(),
            generation: MapGeneration::initial(),
            lineage,
            mutation_lineage: lineage,
        }
    }

    pub fn instance_id(&self) -> MapInstanceId {
        self.instance_id
    }

    pub fn snapshot(&self) -> MapSnapshot {
        MapSnapshot {
            instance_id: self.instance_id,
            generation: self.generation,
            lineage: self.lineage,
        }
    }

    fn ensure_local_keypoint(&self, keypoint: KeyframeKeypoint) -> Result<(), MapError> {
        if keypoint.map_instance_id != self.instance_id {
            return Err(MapError::ForeignKeypoint {
                expected: self.instance_id,
                actual: keypoint.map_instance_id,
            });
        }
        Ok(())
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

        let keypoints = detections.keypoints().to_vec();
        self.add_keyframe(
            detections.frame_id(),
            timestamp,
            pose,
            detections.dimensions(),
            keypoints,
        )
    }

    pub fn add_keyframe(
        &mut self,
        frame_id: FrameId,
        timestamp: Timestamp,
        pose: Pose,
        image_size: FrameDimensions,
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
        validate_map_pose(pose)?;
        validate_keypoints(&keypoints, image_size)?;

        let entry = KeyframeEntry {
            frame_id,
            timestamp,
            pose,
            image_size,
            point_refs: vec![None; keypoints.len()],
            keypoints,
        };

        let next_generation = self.generation.next();
        let kf_id = self.keyframes.insert(entry);
        self.frame_to_keyframe.insert(frame_id, kf_id);
        self.lineage = self.mutation_lineage;
        self.generation = next_generation;
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
        let idx = KeypointIndex::new(index, entry.len()).map_err(|_| {
            MapError::KeypointIndexOutOfBounds {
                index,
                len: entry.len(),
            }
        })?;
        Ok(KeyframeKeypoint {
            map_instance_id: self.instance_id,
            keyframe_id,
            index: idx,
        })
    }

    pub fn keypoint(&self, keypoint: KeyframeKeypoint) -> Result<Keypoint, MapError> {
        self.ensure_local_keypoint(keypoint)?;
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
        self.ensure_local_keypoint(first_obs)?;
        let entry = self
            .keyframes
            .get_mut(first_obs.keyframe_id)
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

        let position = FiniteMapPoint::try_new(position)?;
        let next_generation = self.generation.next();

        let point_id = self.points.insert(MapPoint {
            position,
            descriptor,
            observations: vec![first_obs],
        });

        entry.set_point_ref(first_obs.index, point_id);
        self.lineage = self.mutation_lineage;
        self.generation = next_generation;
        Ok(point_id)
    }

    pub fn add_observation(
        &mut self,
        point_id: MapPointId,
        obs: KeyframeKeypoint,
    ) -> Result<(), MapError> {
        self.ensure_local_keypoint(obs)?;
        let entry = self
            .keyframes
            .get_mut(obs.keyframe_id)
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

        let point = self
            .points
            .get_mut(point_id)
            .ok_or(MapError::MapPointNotFound(point_id))?;
        if point.observes_keyframe(obs.keyframe_id) {
            return Err(MapError::DuplicateObservation {
                point_id,
                keyframe_id: obs.keyframe_id,
            });
        }

        let next_generation = self.generation.next();
        for other in point
            .observations
            .iter()
            .map(|existing| existing.keyframe_id)
        {
            self.covisibility.increment_pair(obs.keyframe_id, other);
        }

        point.add_observation(obs);
        entry.set_point_ref(obs.index, point_id);
        self.lineage = self.mutation_lineage;
        self.generation = next_generation;
        Ok(())
    }

    pub fn update_map_point_descriptor(
        &mut self,
        point_id: MapPointId,
        new_desc: &CompactDescriptor,
        blend: DescriptorBlend,
    ) -> Result<(), MapError> {
        if !self.points.contains_key(point_id) {
            return Err(MapError::MapPointNotFound(point_id));
        }
        let next_generation = self.generation.next();
        let point = self
            .points
            .get_mut(point_id)
            .expect("map point existence validated before mutation");
        point.update_descriptor(new_desc, blend);
        self.lineage = self.mutation_lineage;
        self.generation = next_generation;
        Ok(())
    }

    pub fn set_map_point_position(
        &mut self,
        point_id: MapPointId,
        position: Point3,
    ) -> Result<(), MapError> {
        let position = FiniteMapPoint::try_new(position)?;
        if !self.points.contains_key(point_id) {
            return Err(MapError::MapPointNotFound(point_id));
        }
        let next_generation = self.generation.next();
        let point = self
            .points
            .get_mut(point_id)
            .expect("map point existence validated before mutation");
        point.set_position(position);
        self.lineage = self.mutation_lineage;
        self.generation = next_generation;
        Ok(())
    }

    pub fn set_keyframe_pose(
        &mut self,
        keyframe_id: KeyframeId,
        pose: Pose,
    ) -> Result<(), MapError> {
        validate_map_pose(pose)?;
        if !self.keyframes.contains_key(keyframe_id) {
            return Err(MapError::KeyframeNotFound(keyframe_id));
        }
        let next_generation = self.generation.next();
        let entry = self
            .keyframes
            .get_mut(keyframe_id)
            .expect("keyframe existence validated before mutation");
        entry.set_pose(pose);
        self.lineage = self.mutation_lineage;
        self.generation = next_generation;
        Ok(())
    }

    pub(crate) fn apply_geometry_updates(
        &mut self,
        pose_updates: &[(KeyframeId, Pose)],
        point_updates: &[(MapPointId, Point3)],
    ) -> Result<(), MapError> {
        for &(keyframe_id, pose) in pose_updates {
            validate_map_pose(pose)?;
            if !self.keyframes.contains_key(keyframe_id) {
                return Err(MapError::KeyframeNotFound(keyframe_id));
            }
        }

        let mut validated_points = Vec::with_capacity(point_updates.len());
        for &(point_id, position) in point_updates {
            let position = FiniteMapPoint::try_new(position)?;
            if !self.points.contains_key(point_id) {
                return Err(MapError::MapPointNotFound(point_id));
            }
            validated_points.push((point_id, position));
        }

        let next_generation =
            (!pose_updates.is_empty() || !point_updates.is_empty()).then(|| self.generation.next());
        // Every fallible condition is checked above while the map is unchanged.
        for &(keyframe_id, pose) in pose_updates {
            let entry = self
                .keyframes
                .get_mut(keyframe_id)
                .expect("keyframe existence validated before mutation");
            entry.set_pose(pose);
        }
        for (point_id, position) in validated_points {
            let point = self
                .points
                .get_mut(point_id)
                .expect("map point existence validated before mutation");
            point.set_position(position);
        }

        if let Some(next_generation) = next_generation {
            self.lineage = self.mutation_lineage;
            self.generation = next_generation;
        }
        Ok(())
    }

    pub fn remove_map_point(&mut self, point_id: MapPointId) -> Result<(), MapError> {
        self.validate_map_point_removal(point_id)?;
        let next_generation = self.generation.next();
        self.remove_map_point_without_generation(point_id);
        self.lineage = self.mutation_lineage;
        self.generation = next_generation;
        Ok(())
    }

    fn validate_map_point_removal(&self, point_id: MapPointId) -> Result<(), MapError> {
        let point = self
            .points
            .get(point_id)
            .ok_or(MapError::MapPointNotFound(point_id))?;
        for (observation_index, obs) in point.observations.iter().enumerate() {
            assert!(
                point.observations[..observation_index]
                    .iter()
                    .all(|previous| previous.keyframe_id != obs.keyframe_id),
                "map point contains duplicate keyframe observation"
            );
            let entry = self
                .keyframes
                .get(obs.keyframe_id)
                .expect("map point observation keyframe is missing");
            let index = obs.index.as_usize();
            assert!(
                index < entry.keypoints.len(),
                "map point observation index is out of keyframe bounds"
            );
            assert!(
                entry.point_refs.get(index).copied().flatten() == Some(point_id),
                "map point observation backreference mismatch"
            );
        }
        Ok(())
    }

    fn remove_map_point_without_generation(&mut self, point_id: MapPointId) {
        let point = self
            .points
            .remove(point_id)
            .expect("map point existence validated before mutation");

        for obs in &point.observations {
            let entry = self
                .keyframes
                .get_mut(obs.keyframe_id)
                .expect("map point observation keyframe validated before mutation");
            assert_eq!(
                entry.clear_point_ref(obs.index),
                Some(point_id),
                "map point backreference changed after validation"
            );
        }
        self.covisibility
            .remove_point_observations(&point.observations);
    }

    pub fn remove_keyframe(&mut self, keyframe_id: KeyframeId) -> Result<(), MapError> {
        let entry = self
            .keyframes
            .get(keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        let next_generation = self.generation.next();

        assert_eq!(
            entry.keypoints.len(),
            entry.point_refs.len(),
            "keyframe keypoint and point-reference lengths differ"
        );
        assert!(
            self.frame_to_keyframe.get(&entry.frame_id).copied() == Some(keyframe_id),
            "keyframe frame index is missing or mismatched"
        );
        for (index, maybe_point_id) in entry.point_refs.iter().enumerate() {
            let Some(point_id) = *maybe_point_id else {
                continue;
            };
            let point = self
                .points
                .get(point_id)
                .expect("keyframe map point is missing");
            let mut observations = point
                .observations
                .iter()
                .filter(|obs| obs.keyframe_id == keyframe_id);
            let observation = observations
                .next()
                .expect("keyframe map point has no reciprocal observation");
            assert_eq!(
                observation.index.as_usize(),
                index,
                "keyframe map point observation index mismatch"
            );
            assert!(
                observations.next().is_none(),
                "keyframe map point has duplicate reciprocal observations"
            );
        }

        let entry = self
            .keyframes
            .remove(keyframe_id)
            .expect("keyframe existence validated before mutation");
        assert_eq!(
            self.frame_to_keyframe.remove(&entry.frame_id),
            Some(keyframe_id),
            "keyframe frame index changed after validation"
        );
        self.covisibility.remove_keyframe(keyframe_id);

        for point_id in entry.map_point_ids() {
            let orphaned = {
                let point = self
                    .points
                    .get_mut(point_id)
                    .expect("keyframe map point existence validated before mutation");
                assert!(
                    point.remove_observation_for(keyframe_id),
                    "keyframe reciprocal observation changed after validation"
                );
                point.observations.is_empty()
            };
            if orphaned {
                self.points
                    .remove(point_id)
                    .expect("orphaned map point disappeared during keyframe removal");
            }
        }
        self.lineage = self.mutation_lineage;
        self.generation = next_generation;
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
        if count == 0 {
            return 0;
        }
        let final_generation = self.generation.advance_by(count);
        for &id in &to_remove {
            self.validate_map_point_removal(id)
                .expect("map point collected for culling must still exist");
        }
        for id in to_remove {
            self.remove_map_point_without_generation(id);
        }
        self.lineage = self.mutation_lineage;
        self.generation = final_generation;
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
        entry
            .point_refs
            .iter()
            .enumerate()
            .filter_map(|(idx, pid)| pid.map(|_| idx))
            .map(|idx| {
                let index = KeypointIndex::new(idx, entry.len()).map_err(|_| {
                    MapError::KeypointIndexOutOfBounds {
                        index: idx,
                        len: entry.len(),
                    }
                })?;
                let keypoint_ref = KeyframeKeypoint {
                    map_instance_id: self.instance_id,
                    keyframe_id,
                    index,
                };
                Ok((keypoint_ref, entry.keypoints[idx]))
            })
            .collect()
    }

    pub fn for_each_keyframe_point_descriptor(
        &self,
        keyframe_id: KeyframeId,
        mut visit: impl FnMut(KeyframeKeypoint, &CompactDescriptor),
    ) -> Result<(), MapError> {
        let entry = self
            .keyframes
            .get(keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        for (idx, point_ref) in entry.point_refs.iter().enumerate() {
            let Some(point_id) = point_ref else {
                continue;
            };
            let point = self
                .points
                .get(*point_id)
                .ok_or(MapError::MapPointNotFound(*point_id))?;
            let index = KeypointIndex::new(idx, entry.len()).map_err(|_| {
                MapError::KeypointIndexOutOfBounds {
                    index: idx,
                    len: entry.len(),
                }
            })?;
            visit(
                KeyframeKeypoint {
                    map_instance_id: self.instance_id,
                    keyframe_id,
                    index,
                },
                point.descriptor(),
            );
        }
        Ok(())
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
        let mut window = Vec::with_capacity(max.get());
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

    /// Returns the shared-point count divided by the smaller associated-point count.
    ///
    /// Two valid keyframes without a graph edge, including a self-comparison, have ratio zero.
    /// Missing keyframes and graph counts that exceed either keyframe's point count are errors.
    pub fn covisibility_ratio(&self, a: KeyframeId, b: KeyframeId) -> Result<f32, MapError> {
        let a_entry = self.keyframe(a).ok_or(MapError::KeyframeNotFound(a))?;
        let b_entry = self.keyframe(b).ok_or(MapError::KeyframeNotFound(b))?;
        let shared = self.covisibility.covisibility_count(a, b);
        if shared == 0 {
            return Ok(0.0);
        }

        let a_points = a_entry.map_point_ids().count();
        let b_points = b_entry.map_point_ids().count();
        let denominator = a_points.min(b_points);
        if shared as usize > denominator {
            return Err(MapError::InconsistentCovisibility {
                a,
                b,
                shared,
                a_points,
                b_points,
            });
        }
        Ok(covisibility_ratio_from_counts(shared, denominator))
    }

    pub fn map_point_for_keypoint(
        &self,
        keypoint: KeyframeKeypoint,
    ) -> Result<Option<MapPointId>, MapError> {
        self.ensure_local_keypoint(keypoint)?;
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

fn validate_map_pose(pose: Pose) -> Result<(), MapError> {
    Pose64::try_from_pose32(pose)
        .map(|_| ())
        .map_err(MapError::InvalidPose)
}

fn validate_keypoints(keypoints: &[Keypoint], image_size: FrameDimensions) -> Result<(), MapError> {
    for (index, keypoint) in keypoints.iter().enumerate() {
        for (axis, value, upper_bound) in [
            ("x", keypoint.x, image_size.width()),
            ("y", keypoint.y, image_size.height()),
        ] {
            if !value.is_finite() || value < 0.0 || f64::from(value) >= f64::from(upper_bound) {
                return Err(MapError::InvalidKeypoint {
                    index,
                    axis,
                    value,
                    upper_bound,
                });
            }
        }
    }
    Ok(())
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
    use crate::{CompactDescriptor, Descriptor, Keypoint, Pose, SensorId, Timestamp};

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

    fn assert_panics_with<T>(expected: &str, operation: impl FnOnce() -> T) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
        let Err(payload) = result else {
            panic!("operation must panic with {expected:?}");
        };
        assert_eq!(crate::panic_payload_to_string(payload.as_ref()), expected);
    }

    fn assert_generation_exhaustion(operation: impl FnOnce()) {
        assert_panics_with("map generation space exhausted", operation);
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
                position: FiniteMapPoint::try_new(Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                })
                .expect("finite map point"),
                descriptor: previous.clone(),
                observations: Vec::new(),
            };
            point.update_descriptor(&next, blend);
            assert_eq!(point.descriptor.0, expected, "weight {weight}");
        }
    }

    #[test]
    fn keyframes_reuse_validated_detection_dimensions() {
        let dimensions = FrameDimensions::try_new(7, 5).expect("dimensions");
        let detections = Detections::new(
            SensorId::StereoLeft,
            FrameId::new(1),
            dimensions.width(),
            dimensions.height(),
            vec![Keypoint { x: 3.0, y: 2.0 }],
            vec![1.0],
            vec![Descriptor::ZERO],
        )
        .expect("detections");
        let mut map = SlamMap::new();

        let keyframe_id = map
            .add_keyframe_from_detections(&detections, Timestamp::from_nanos(1), Pose::identity())
            .expect("keyframe");

        assert_eq!(
            map.keyframe(keyframe_id).expect("entry").image_size(),
            dimensions
        );
    }

    #[test]
    fn keyframe_construction_rejects_invalid_keypoints_without_mutation() {
        let size = FrameDimensions::try_new(640, 480).expect("image size");
        for (axis, keypoint) in [
            (
                "x",
                Keypoint {
                    x: f32::NAN,
                    y: 1.0,
                },
            ),
            ("x", Keypoint { x: 640.0, y: 1.0 }),
            ("y", Keypoint { x: 1.0, y: -1.0 }),
            (
                "y",
                Keypoint {
                    x: 1.0,
                    y: f32::INFINITY,
                },
            ),
        ] {
            let mut map = SlamMap::new();
            let generation = map.generation();
            let error = map
                .add_keyframe(
                    FrameId::new(1),
                    Timestamp::from_nanos(1),
                    Pose::identity(),
                    size,
                    vec![keypoint],
                )
                .expect_err("invalid keypoint must be rejected");
            assert!(matches!(
                error,
                MapError::InvalidKeypoint {
                    index: 0,
                    axis: actual,
                    ..
                } if actual == axis
            ));
            assert_eq!(map.num_keyframes(), 0);
            assert_eq!(map.generation(), generation);
        }
    }

    #[test]
    fn map_point_writes_reject_non_finite_positions_transactionally() {
        let size = FrameDimensions::try_new(640, 480).expect("image size");
        let mut map = SlamMap::new();
        let keyframe = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                Pose::identity(),
                size,
                make_keypoints(2),
            )
            .expect("keyframe");
        let first = map.keyframe_keypoint(keyframe, 0).expect("first keypoint");
        let generation = map.generation();
        assert!(matches!(
            map.add_map_point(
                Point3 {
                    x: f32::NAN,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                first,
            ),
            Err(MapError::NonFiniteMapPoint { axis: "x", .. })
        ));
        assert_eq!(map.num_points(), 0);
        assert_eq!(map.generation(), generation);
        assert_eq!(
            map.map_point_for_keypoint(first).expect("association"),
            None
        );

        let point_id = map
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                first,
            )
            .expect("valid point");
        let original = map.point(point_id).expect("point").position();
        let generation = map.generation();
        assert!(matches!(
            map.set_map_point_position(
                point_id,
                Point3 {
                    x: 0.0,
                    y: f32::INFINITY,
                    z: 1.0,
                },
            ),
            Err(MapError::NonFiniteMapPoint { axis: "y", .. })
        ));
        let stored = map.point(point_id).expect("point").position();
        assert_eq!(
            [stored.x, stored.y, stored.z],
            [original.x, original.y, original.z]
        );
        assert_eq!(map.generation(), generation);
    }

    #[test]
    fn map_pose_writes_reject_invalid_se3_with_source_context() {
        let size = FrameDimensions::try_new(640, 480).expect("image size");
        let invalid = Pose::from_rt(
            [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        );
        let mut map = SlamMap::new();
        let error = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                invalid,
                size,
                make_keypoints(1),
            )
            .expect_err("invalid pose must be rejected");
        assert!(matches!(error, MapError::InvalidPose(_)));
        assert!(std::error::Error::source(&error).is_some());
        assert_eq!(map.num_keyframes(), 0);
        assert_eq!(map.generation().as_u64(), 0);

        let keyframe = map
            .add_keyframe(
                FrameId::new(2),
                Timestamp::from_nanos(2),
                Pose::identity(),
                size,
                make_keypoints(1),
            )
            .expect("valid keyframe");
        let generation = map.generation();
        assert!(matches!(
            map.set_keyframe_pose(keyframe, invalid),
            Err(MapError::InvalidPose(_))
        ));
        let stored = map.keyframe(keyframe).expect("keyframe").pose();
        assert_eq!(stored.rotation(), Pose::identity().rotation());
        assert_eq!(stored.translation(), Pose::identity().translation());
        assert_eq!(map.generation(), generation);
    }

    #[test]
    fn geometry_batch_preflights_every_update_before_mutation() {
        let size = FrameDimensions::try_new(640, 480).expect("image size");
        let mut map = SlamMap::new();
        let keyframe = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                Pose::identity(),
                size,
                make_keypoints(1),
            )
            .expect("keyframe");
        let keypoint = map
            .keyframe_keypoint(keyframe, 0)
            .expect("keypoint reference");
        let point = map
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                keypoint,
            )
            .expect("map point");
        let generation = map.generation();
        let updated_pose = Pose::from_rt(Pose::identity().rotation(), [1.0, 2.0, 3.0]);

        let error = map
            .apply_geometry_updates(
                &[(keyframe, updated_pose)],
                &[(
                    point,
                    Point3 {
                        x: f32::NAN,
                        y: 0.0,
                        z: 1.0,
                    },
                )],
            )
            .expect_err("invalid point must reject the complete batch");

        assert!(matches!(
            error,
            MapError::NonFiniteMapPoint { axis: "x", .. }
        ));
        let stored_pose = map.keyframe(keyframe).expect("keyframe").pose();
        assert_eq!(stored_pose.rotation(), Pose::identity().rotation());
        assert_eq!(stored_pose.translation(), Pose::identity().translation());
        let stored_point = map.point(point).expect("point").position();
        assert_eq!(
            [stored_point.x, stored_point.y, stored_point.z],
            [0.0, 0.0, 1.0]
        );
        assert_eq!(map.generation(), generation);
    }

    #[test]
    fn keyframe_keypoints_are_scoped_to_their_map_instance() {
        let size = FrameDimensions::try_new(640, 480).expect("image size");
        let mut map_a = SlamMap::new();
        let keyframe_a = map_a
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                Pose::identity(),
                size,
                make_keypoints(1),
            )
            .expect("keyframe a");
        let keypoint_a = map_a.keyframe_keypoint(keyframe_a, 0).expect("keypoint a");

        let mut map_b = SlamMap::new();
        map_b
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                Pose::identity(),
                size,
                make_keypoints(1),
            )
            .expect("keyframe b");

        assert_ne!(map_a.instance_id(), map_b.instance_id());
        assert_eq!(map_a.clone().instance_id(), map_a.instance_id());
        assert_eq!(keypoint_a.map_instance_id(), map_a.instance_id());
        assert!(matches!(
            map_b.keypoint(keypoint_a),
            Err(MapError::ForeignKeypoint { expected, actual })
                if expected == map_b.instance_id() && actual == map_a.instance_id()
        ));
        assert!(matches!(
            map_b.map_point_for_keypoint(keypoint_a),
            Err(MapError::ForeignKeypoint { .. })
        ));
    }

    #[test]
    fn map_snapshots_order_generations_only_within_the_same_instance() {
        let size = FrameDimensions::try_new(640, 480).expect("image size");
        let mut map = SlamMap::new();
        let before = map.snapshot();
        map.add_keyframe(
            FrameId::new(1),
            Timestamp::from_nanos(1),
            Pose::identity(),
            size,
            make_keypoints(1),
        )
        .expect("keyframe");
        let after = map.snapshot();

        assert!(before.is_same_or_older_than(after));
        assert!(!after.is_same_or_older_than(before));
        assert!(!before.is_same_or_older_than(SlamMap::new().snapshot()));
    }

    #[test]
    fn descriptor_blend_uses_u8_weighted_average() {
        let mut map = SlamMap::new();
        let size = FrameDimensions::try_new(640, 480).expect("valid size");
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
        let size = FrameDimensions::try_new(640, 480).expect("valid size");
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

        let size = FrameDimensions::try_new(640, 480).expect("valid size");
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
    fn map_generation_exhaustion_precedes_every_map_mutation() {
        let dimensions = FrameDimensions::try_new(640, 480).expect("dimensions");
        let mut map = SlamMap::new();
        let first = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                Pose::identity(),
                dimensions,
                make_keypoints(2),
            )
            .expect("first keyframe");
        let second = map
            .add_keyframe(
                FrameId::new(2),
                Timestamp::from_nanos(2),
                Pose::identity(),
                dimensions,
                make_keypoints(1),
            )
            .expect("second keyframe");
        let first_observation = map.keyframe_keypoint(first, 0).expect("observation");
        let first_free = map.keyframe_keypoint(first, 1).expect("free keypoint");
        let second_free = map.keyframe_keypoint(second, 0).expect("free keypoint");
        let point = map
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                first_observation,
            )
            .expect("map point");
        map.generation = MapGeneration(u64::MAX);
        let snapshot = map.snapshot();

        assert_generation_exhaustion(|| {
            let _ = map.add_keyframe(
                FrameId::new(3),
                Timestamp::from_nanos(3),
                Pose::identity(),
                dimensions,
                make_keypoints(1),
            );
        });
        assert_generation_exhaustion(|| {
            let _ = map.add_map_point(
                Point3 {
                    x: 1.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                first_free,
            );
        });
        assert_generation_exhaustion(|| {
            let _ = map.add_observation(point, second_free);
        });
        assert_generation_exhaustion(|| {
            let _ = map.update_map_point_descriptor(
                point,
                &CompactDescriptor([255; 256]),
                DescriptorBlend::try_new(1.0).expect("blend"),
            );
        });
        assert_generation_exhaustion(|| {
            let _ = map.set_map_point_position(
                point,
                Point3 {
                    x: 1.0,
                    y: 2.0,
                    z: 3.0,
                },
            );
        });
        assert_generation_exhaustion(|| {
            let _ = map.set_keyframe_pose(
                first,
                Pose::from_rt(
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [1.0, 0.0, 0.0],
                ),
            );
        });
        assert_generation_exhaustion(|| {
            let _ = map.apply_geometry_updates(
                &[(first, Pose::identity())],
                &[(
                    point,
                    Point3 {
                        x: 0.0,
                        y: 0.0,
                        z: 2.0,
                    },
                )],
            );
        });
        assert_generation_exhaustion(|| {
            let _ = map.remove_map_point(point);
        });
        assert_generation_exhaustion(|| {
            let _ = map.remove_keyframe(first);
        });
        assert_generation_exhaustion(|| {
            let _ = map.cull_points(2);
        });

        assert_eq!(map.snapshot(), snapshot);
        assert_eq!(map.num_keyframes(), 2);
        assert_eq!(map.num_points(), 1);
        assert_eq!(
            map.map_point_for_keypoint(first_observation)
                .expect("association"),
            Some(point)
        );
        assert_eq!(
            map.map_point_for_keypoint(first_free)
                .expect("free keypoint"),
            None
        );
        assert_eq!(
            map.map_point_for_keypoint(second_free)
                .expect("free keypoint"),
            None
        );
        let stored = map.point(point).expect("point retained");
        assert_eq!(stored.position().z, 1.0);
        assert_eq!(stored.descriptor(), &make_descriptor());
        assert_map_invariants(&map).expect("unchanged map invariants");
    }

    #[test]
    fn removal_backreference_failures_precede_mutation() {
        let mut map = SlamMap::new();
        let size = FrameDimensions::try_new(640, 480).expect("valid size");
        let first_keyframe = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                Pose::identity(),
                size,
                make_keypoints(2),
            )
            .expect("first keyframe");
        let second_keyframe = map
            .add_keyframe(
                FrameId::new(2),
                Timestamp::from_nanos(2),
                Pose::identity(),
                size,
                make_keypoints(1),
            )
            .expect("second keyframe");
        let first_shared = map
            .keyframe_keypoint(first_keyframe, 0)
            .expect("first shared observation");
        let second_shared = map
            .keyframe_keypoint(second_keyframe, 0)
            .expect("second shared observation");
        let orphan_observation = map
            .keyframe_keypoint(first_keyframe, 1)
            .expect("orphan observation");
        let shared_point = map
            .add_map_point(
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                first_shared,
            )
            .expect("shared point");
        map.add_observation(shared_point, second_shared)
            .expect("second shared observation");
        let orphan_point = map
            .add_map_point(
                Point3 {
                    x: 1.0,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                orphan_observation,
            )
            .expect("orphan point");
        assert_map_invariants(&map).expect("valid removal fixture");

        let mut broken_point_backref = map.clone();
        assert_eq!(
            broken_point_backref
                .keyframes
                .get_mut(second_keyframe)
                .expect("second keyframe")
                .clear_point_ref(second_shared.index),
            Some(shared_point)
        );
        let point_snapshot = broken_point_backref.snapshot();
        assert_panics_with("map point observation backreference mismatch", || {
            broken_point_backref.remove_map_point(shared_point)
        });
        assert_eq!(broken_point_backref.snapshot(), point_snapshot);
        assert!(broken_point_backref.point(shared_point).is_some());
        assert_eq!(
            broken_point_backref
                .map_point_for_keypoint(first_shared)
                .expect("first backreference"),
            Some(shared_point)
        );
        assert_eq!(
            broken_point_backref
                .map_point_for_keypoint(second_shared)
                .expect("corrupt second backreference"),
            None
        );

        let mut broken_point_observation = map.clone();
        broken_point_observation
            .points
            .get_mut(orphan_point)
            .expect("orphan point")
            .observations
            .clear();
        let keyframe_snapshot = broken_point_observation.snapshot();
        assert_panics_with("keyframe map point has no reciprocal observation", || {
            broken_point_observation.remove_keyframe(first_keyframe)
        });
        assert_eq!(broken_point_observation.snapshot(), keyframe_snapshot);
        assert!(broken_point_observation.keyframe(first_keyframe).is_some());
        assert!(broken_point_observation.point(shared_point).is_some());
        assert!(broken_point_observation.point(orphan_point).is_some());

        let mut broken_frame_index = map;
        assert_eq!(
            broken_frame_index
                .frame_to_keyframe
                .remove(&FrameId::new(1)),
            Some(first_keyframe)
        );
        let frame_snapshot = broken_frame_index.snapshot();
        assert_panics_with("keyframe frame index is missing or mismatched", || {
            broken_frame_index.remove_keyframe(first_keyframe)
        });
        assert_eq!(broken_frame_index.snapshot(), frame_snapshot);
        assert!(broken_frame_index.keyframe(first_keyframe).is_some());
        assert!(broken_frame_index.point(shared_point).is_some());
        assert!(broken_frame_index.point(orphan_point).is_some());
    }

    #[test]
    fn map_clone_preserves_revision_until_branches_diverge() {
        let mut map = SlamMap::new();
        let size = FrameDimensions::try_new(640, 480).expect("valid size");
        let pose = Pose::identity();
        let keyframe = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                pose,
                size,
                make_keypoints(1),
            )
            .expect("keyframe1");

        let shared_snapshot = map.snapshot();
        let mut cloned = map.clone();
        assert_eq!(cloned.generation(), map.generation());
        assert_eq!(cloned.snapshot(), shared_snapshot);

        map.set_keyframe_pose(
            keyframe,
            Pose::from_rt(Pose::identity().rotation(), [1.0, 0.0, 0.0]),
        )
        .expect("mutate source branch");
        cloned
            .set_keyframe_pose(
                keyframe,
                Pose::from_rt(Pose::identity().rotation(), [-1.0, 0.0, 0.0]),
            )
            .expect("mutate clone branch");

        assert_eq!(cloned.generation(), map.generation());
        assert_ne!(cloned.snapshot(), map.snapshot());
        assert!(shared_snapshot.is_same_or_older_than(map.snapshot()));
        assert!(!shared_snapshot.is_same_or_older_than(cloned.snapshot()));

        let transaction_source = map.snapshot();
        let mut transaction = map.clone_for_transaction();
        transaction
            .set_keyframe_pose(
                keyframe,
                Pose::from_rt(Pose::identity().rotation(), [2.0, 0.0, 0.0]),
            )
            .expect("mutate canonical transaction");
        assert!(transaction_source.is_same_or_older_than(transaction.snapshot()));
        assert_ne!(transaction_source, transaction.snapshot());
    }

    #[test]
    fn monotonic_id_allocator_exhausts_without_wrapping() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(allocate_monotonic_id(&counter), Some(u64::MAX - 1));
        assert_eq!(allocate_monotonic_id(&counter), None);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
    }

    #[test]
    fn covisibility_increments_and_decrements_on_map_point_changes() {
        let mut map = SlamMap::new();
        let size = FrameDimensions::try_new(640, 480).expect("valid size");
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
        assert_eq!(map.covisibility_ratio(kf1, kf2).expect("ratio"), 1.0);

        for (keyframe_id, keypoint_index) in [(kf1, 1), (kf2, 1)] {
            let observation = map
                .keyframe_keypoint(keyframe_id, keypoint_index)
                .expect("unique observation");
            map.add_map_point(
                Point3 {
                    x: keypoint_index as f32,
                    y: 0.0,
                    z: 1.0,
                },
                make_descriptor(),
                observation,
            )
            .expect("unique map point");
        }
        for ratio in [
            map.covisibility_ratio(kf1, kf2).expect("forward ratio"),
            map.covisibility_ratio(kf2, kf1).expect("reverse ratio"),
        ] {
            assert_eq!(ratio, 0.5);
            assert!((0.0..=1.0).contains(&ratio));
        }
        assert_map_invariants(&map).expect("after shared observation");

        map.remove_map_point(point).expect("remove point");
        assert_eq!(map.covisibility().covisibility_count(kf1, kf2), 0);
        assert_eq!(map.covisibility_ratio(kf1, kf2).expect("zero ratio"), 0.0);
        assert_map_invariants(&map).expect("after point removal");
    }

    #[test]
    fn covisibility_ratio_preserves_f32_precision_at_the_integer_boundary() {
        assert_eq!(
            covisibility_ratio_from_counts(16_777_216, 16_777_217),
            f32::from_bits(0x3f7f_ffff)
        );
    }

    #[test]
    fn covisibility_ratio_validates_stale_keyframes_before_returning_zero() {
        let size = FrameDimensions::try_new(640, 480).expect("valid size");
        let mut map = SlamMap::new();
        let first = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                Pose::identity(),
                size,
                make_keypoints(1),
            )
            .expect("first keyframe");
        let second = map
            .add_keyframe(
                FrameId::new(2),
                Timestamp::from_nanos(2),
                Pose::identity(),
                size,
                make_keypoints(1),
            )
            .expect("second keyframe");

        assert_eq!(
            map.covisibility_ratio(first, second)
                .expect("two live non-neighbors"),
            0.0
        );
        assert_eq!(
            map.covisibility_ratio(first, first)
                .expect("self-comparison has no graph edge"),
            0.0
        );

        map.remove_keyframe(second).expect("remove second keyframe");
        let replacement = map
            .add_keyframe(
                FrameId::new(3),
                Timestamp::from_nanos(3),
                Pose::identity(),
                size,
                make_keypoints(1),
            )
            .expect("replacement keyframe");
        assert!(map.keyframe(replacement).is_some());
        for (a, b) in [(second, first), (first, second)] {
            assert!(matches!(
                map.covisibility_ratio(a, b),
                Err(MapError::KeyframeNotFound(id)) if id == second
            ));
        }
    }

    #[test]
    fn covisibility_ratio_reports_inconsistent_graph_counts() {
        let size = FrameDimensions::try_new(640, 480).expect("valid size");
        let mut map = SlamMap::new();
        let first = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                Pose::identity(),
                size,
                make_keypoints(1),
            )
            .expect("first keyframe");
        let second = map
            .add_keyframe(
                FrameId::new(2),
                Timestamp::from_nanos(2),
                Pose::identity(),
                size,
                make_keypoints(1),
            )
            .expect("second keyframe");

        map.covisibility.increment_pair(first, second);
        assert!(matches!(
            map.covisibility_ratio(first, second),
            Err(MapError::InconsistentCovisibility {
                a,
                b,
                shared: 1,
                a_points: 0,
                b_points: 0,
            }) if a == first && b == second
        ));
    }

    #[test]
    fn duplicate_observation_is_rejected() {
        let mut map = SlamMap::new();
        let size = FrameDimensions::try_new(640, 480).expect("valid size");
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
        let size = FrameDimensions::try_new(640, 480).expect("valid size");
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
        let size = FrameDimensions::try_new(640, 480).expect("valid size");
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
