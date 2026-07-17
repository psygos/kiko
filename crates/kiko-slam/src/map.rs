use std::collections::HashMap;
use std::num::{NonZeroU16, NonZeroU32, NonZeroUsize};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(test)]
use std::collections::HashSet;

use slotmap::{SlotMap, new_key_type};

use crate::{
    CompactDescriptor, Detections, FrameDimensions, FrameId, Keypoint, SensorId, Timestamp,
    WorldPoint3, WorldToCamera,
};

/// Fixed-point scale factor for descriptor blending (8-bit precision).
const BLEND_SCALE: u16 = 256;
/// Smallest requested blend that rounds to a non-zero fixed-point weight.
const MIN_BLEND_ALPHA: f32 = 0.5 / BLEND_SCALE as f32;
/// Rounding bias for fixed-point descriptor blending (half of BLEND_SCALE).
const BLEND_ROUND: u32 = (BLEND_SCALE / 2) as u32;
/// SlotMap reserves one `u32` key for its sentinel.
const MAX_COVISIBILITY_COUNT: u32 = u32::MAX - 1;
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

impl From<FrameDimensions> for ImageSize {
    fn from(dimensions: FrameDimensions) -> Self {
        Self {
            width: dimensions.width(),
            height: dimensions.height(),
        }
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

    fn clear_point_ref(&mut self, index: KeypointIndex) -> Option<MapPointId> {
        self.point_refs[index.as_usize()].take()
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
    fn symmetric_pair_weight(&self, a: KeyframeId, b: KeyframeId) -> Option<NonZeroU32> {
        assert_ne!(a, b, "covisibility graph contains a self edge");

        let forward_neighbors = self.edges.get(&a);
        if let Some(neighbors) = forward_neighbors {
            assert!(
                !neighbors.is_empty(),
                "covisibility graph contains an empty adjacency bucket"
            );
        }
        let reverse_neighbors = self.edges.get(&b);
        if let Some(neighbors) = reverse_neighbors {
            assert!(
                !neighbors.is_empty(),
                "covisibility graph contains an empty adjacency bucket"
            );
        }

        let forward = forward_neighbors.and_then(|neighbors| neighbors.get(&b).copied());
        let reverse = reverse_neighbors.and_then(|neighbors| neighbors.get(&a).copied());
        match (forward, reverse) {
            (None, None) => None,
            (Some(weight), Some(reverse_weight)) => {
                assert!(
                    weight == reverse_weight,
                    "covisibility graph contains asymmetric pair weights"
                );
                assert!(
                    weight.get() <= MAX_COVISIBILITY_COUNT,
                    "covisibility count exceeds the map point key space"
                );
                Some(weight)
            }
            _ => panic!("covisibility graph contains a missing reverse edge"),
        }
    }

    fn preflight_increment_pair(&self, a: KeyframeId, b: KeyframeId) {
        if let Some(weight) = self.symmetric_pair_weight(a, b) {
            // SlotMap reserves one u32 key for its sentinel, and a map point
            // contributes at most once to a keyframe pair. A valid count is
            // therefore strictly below u32::MAX.
            weight
                .checked_add(1)
                .filter(|next| next.get() <= MAX_COVISIBILITY_COUNT)
                .expect("covisibility count exceeds the map point key space");
        }
    }

    fn preflight_decrement_pair(&self, a: KeyframeId, b: KeyframeId) {
        self.symmetric_pair_weight(a, b)
            .expect("covisibility graph is missing an observed keyframe pair");
    }

    fn set_pair_weight(&mut self, a: KeyframeId, b: KeyframeId, weight: Option<NonZeroU32>) {
        if let Some(weight) = weight {
            self.edges.entry(a).or_default().insert(b, weight);
            self.edges.entry(b).or_default().insert(a, weight);
            return;
        }

        for (from, to) in [(a, b), (b, a)] {
            let remove_bucket = {
                let neighbors = self
                    .edges
                    .get_mut(&from)
                    .expect("preflight proved the covisibility adjacency bucket exists");
                neighbors
                    .remove(&to)
                    .expect("preflight proved the covisibility edge exists");
                neighbors.is_empty()
            };
            if remove_bucket {
                self.edges.remove(&from);
            }
        }
    }

    fn increment_pair_after_preflight(&mut self, a: KeyframeId, b: KeyframeId) {
        let next = self
            .edges
            .get(&a)
            .and_then(|neighbors| neighbors.get(&b).copied())
            .map_or(NonZeroU32::MIN, |weight| {
                weight
                    .checked_add(1)
                    .filter(|next| next.get() <= MAX_COVISIBILITY_COUNT)
                    .expect("preflight proved the covisibility count has capacity")
            });
        self.set_pair_weight(a, b, Some(next));
    }

    fn decrement_pair_after_preflight(&mut self, a: KeyframeId, b: KeyframeId) {
        let weight = self
            .edges
            .get(&a)
            .and_then(|neighbors| neighbors.get(&b).copied())
            .expect("preflight proved the covisibility edge exists");
        self.set_pair_weight(a, b, NonZeroU32::new(weight.get() - 1));
    }

    #[cfg(test)]
    fn increment_pair(&mut self, a: KeyframeId, b: KeyframeId) {
        if a == b {
            return;
        }
        self.preflight_increment_pair(a, b);
        self.increment_pair_after_preflight(a, b);
    }

    #[cfg(test)]
    fn decrement_pair(&mut self, a: KeyframeId, b: KeyframeId) {
        if a == b {
            return;
        }
        self.preflight_decrement_pair(a, b);
        self.decrement_pair_after_preflight(a, b);
    }

    fn preflight_increment_observation_pairs(
        &self,
        keyframe_id: KeyframeId,
        observations: &[KeyframeKeypoint],
    ) {
        for (index, observation) in observations.iter().enumerate() {
            assert!(
                observation.keyframe_id != keyframe_id
                    && observations[..index]
                        .iter()
                        .all(|previous| previous.keyframe_id != observation.keyframe_id),
                "map point contains duplicate keyframe observations"
            );
            self.preflight_increment_pair(keyframe_id, observation.keyframe_id);
        }
    }

    fn increment_observation_pairs_after_preflight(
        &mut self,
        keyframe_id: KeyframeId,
        observations: &[KeyframeKeypoint],
    ) {
        for observation in observations {
            self.increment_pair_after_preflight(keyframe_id, observation.keyframe_id);
        }
    }

    fn preflight_remove_point_observations(&self, observations: &[KeyframeKeypoint]) {
        for (i, obs_a) in observations.iter().enumerate() {
            for obs_b in &observations[i + 1..] {
                self.preflight_decrement_pair(obs_a.keyframe_id, obs_b.keyframe_id);
            }
        }
    }

    fn remove_point_observations_after_preflight(&mut self, observations: &[KeyframeKeypoint]) {
        for (i, obs_a) in observations.iter().enumerate() {
            for obs_b in &observations[i + 1..] {
                self.decrement_pair_after_preflight(obs_a.keyframe_id, obs_b.keyframe_id);
            }
        }
    }

    /// Returns the non-empty neighbor set for a keyframe with incident edges.
    pub fn neighbors(&self, kf_id: KeyframeId) -> Option<&HashMap<KeyframeId, NonZeroU32>> {
        self.edges.get(&kf_id).inspect(|neighbors| {
            assert!(
                !neighbors.is_empty(),
                "covisibility graph contains an empty adjacency bucket"
            );
        })
    }

    fn preflight_remove_keyframe(&self, kf_id: KeyframeId) {
        let outgoing = self.edges.get(&kf_id);
        if let Some(neighbors) = outgoing {
            assert!(
                !neighbors.is_empty(),
                "covisibility graph contains an empty adjacency bucket"
            );
            for &neighbor_id in neighbors.keys() {
                self.symmetric_pair_weight(kf_id, neighbor_id)
                    .expect("outgoing covisibility edge disappeared during immutable preflight");
            }
        }

        for (&neighbor_id, neighbors) in &self.edges {
            if neighbor_id == kf_id {
                continue;
            }
            if neighbors.contains_key(&kf_id) {
                assert!(
                    outgoing.is_some_and(|our_edges| our_edges.contains_key(&neighbor_id)),
                    "covisibility graph contains an incoming-only edge"
                );
            }
        }
    }

    fn preflight_keyframe_counts(
        &self,
        kf_id: KeyframeId,
        expected: &HashMap<KeyframeId, NonZeroU32>,
    ) {
        let actual = self.edges.get(&kf_id);
        for (&neighbor_id, &weight) in expected {
            assert!(
                actual.and_then(|edges| edges.get(&neighbor_id)).copied() == Some(weight),
                "covisibility graph is missing or miscounts an incident pair"
            );
        }
        if let Some(actual) = actual {
            assert!(
                actual.len() == expected.len(),
                "covisibility graph contains an unexpected incident pair"
            );
        }
    }

    fn remove_keyframe_after_preflight(&mut self, kf_id: KeyframeId) {
        let Some(neighbors) = self.edges.remove(&kf_id) else {
            return;
        };
        for (&neighbor_id, &weight) in &neighbors {
            let remove_bucket = {
                let their_edges = self
                    .edges
                    .get_mut(&neighbor_id)
                    .expect("preflight proved the reverse adjacency bucket exists");
                assert_eq!(
                    their_edges.remove(&kf_id),
                    Some(weight),
                    "preflight proved the reverse covisibility edge matches"
                );
                their_edges.is_empty()
            };
            if remove_bucket {
                self.edges.remove(&neighbor_id);
            }
        }
    }

    /// Removes every edge incident to `kf_id`.
    ///
    /// This is a no-op when the keyframe has no incident edges. It panics if
    /// the graph's private symmetric-edge invariant has been violated.
    pub fn remove_keyframe(&mut self, kf_id: KeyframeId) {
        self.preflight_remove_keyframe(kf_id);
        self.remove_keyframe_after_preflight(kf_id);
    }

    /// Returns the exact shared-point count, or zero for an absent or self pair.
    ///
    /// Panics if the graph's private symmetric-edge invariant has been violated.
    pub fn covisibility_count(&self, a: KeyframeId, b: KeyframeId) -> u32 {
        if a == b {
            if let Some(neighbors) = self.edges.get(&a) {
                assert!(
                    !neighbors.is_empty(),
                    "covisibility graph contains an empty adjacency bucket"
                );
                assert!(
                    !neighbors.contains_key(&a),
                    "covisibility graph contains a self edge"
                );
            }
            return 0;
        }
        self.symmetric_pair_weight(a, b).map_or(0, NonZeroU32::get)
    }
}

fn covisibility_ratio_from_counts(shared: u32, denominator: usize) -> f32 {
    debug_assert_ne!(shared, 0);
    debug_assert!(shared as usize <= denominator);
    (f64::from(shared) / denominator as f64) as f32
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
    EmptyKeyframe {
        frame_id: FrameId,
    },
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
            MapError::EmptyKeyframe { frame_id } => {
                write!(f, "keyframe {frame_id:?} has no keypoints")
            }
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
        self.checked_advance_by(1)
            .expect("map generation space exhausted")
    }

    fn checked_advance_by(self, mutations: usize) -> Option<Self> {
        let mutations = u64::try_from(mutations).ok()?;
        self.0.checked_add(mutations).map(Self)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MapInstanceId(u64);

impl MapInstanceId {
    fn next() -> Self {
        let value =
            allocate_monotonic_id(&NEXT_MAP_INSTANCE_ID).expect("map instance ID space exhausted");
        Self(value)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MapLineageId(u64);

impl MapLineageId {
    fn next() -> Self {
        let value =
            allocate_monotonic_id(&NEXT_MAP_LINEAGE_ID).expect("map lineage ID space exhausted");
        Self(value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct MapVersion {
    generation: MapGeneration,
    lineage: MapLineageId,
}

impl MapVersion {
    fn initial(lineage: MapLineageId) -> Self {
        Self {
            generation: MapGeneration::initial(),
            lineage,
        }
    }

    fn next(self, mutation_lineage: MapLineageId) -> Self {
        Self {
            generation: self.generation.next(),
            lineage: mutation_lineage,
        }
    }
}

/// A process-local freshness token for one exact [`SlamMap`] revision.
///
/// Clones compare equal until either copy mutates. The generation remains a
/// useful monotonic counter within one branch, but snapshot equality also
/// distinguishes independently mutated branches at the same generation. This
/// is not a structural content hash: equivalent mutations can produce unequal
/// snapshots.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MapSnapshot {
    instance_id: MapInstanceId,
    version: MapVersion,
}

impl MapSnapshot {
    pub fn instance_id(self) -> MapInstanceId {
        self.instance_id
    }

    pub fn generation(self) -> MapGeneration {
        self.version.generation
    }

    /// Whether two revisions belong to the same concrete mutation branch.
    ///
    /// Generation order alone cannot establish ancestry after a [`SlamMap`]
    /// clone: the clone retains the exact source snapshot but reserves a
    /// distinct lineage for its future mutations. This process-local predicate
    /// lets internal consumers reject a newer revision from that sibling
    /// branch without exposing the opaque lineage identifier as public data.
    pub(crate) fn shares_mutation_lineage_with(self, other: Self) -> bool {
        self.instance_id == other.instance_id && self.version.lineage == other.version.lineage
    }
}

#[derive(Debug)]
pub struct SlamMap {
    instance_id: MapInstanceId,
    points: SlotMap<RawMapPointId, MapPoint>,
    keyframes: SlotMap<RawKeyframeId, KeyframeEntry>,
    covisibility: CovisibilityGraph,
    frame_to_keyframe: HashMap<FrameId, KeyframeId>,
    version: MapVersion,
    mutation_lineage: MapLineageId,
}

impl Clone for SlamMap {
    fn clone(&self) -> Self {
        let mutation_lineage = MapLineageId::next();
        Self {
            instance_id: self.instance_id,
            points: self.points.clone(),
            keyframes: self.keyframes.clone(),
            covisibility: self.covisibility.clone(),
            frame_to_keyframe: self.frame_to_keyframe.clone(),
            version: self.version,
            // The clone still represents exactly the source state. Reserve a
            // distinct lineage only for its future successful mutations so
            // asynchronous snapshots of an unmodified clone remain equal.
            mutation_lineage,
        }
    }
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

fn covisibility_window_rank(
    &(id, weight): &(KeyframeId, NonZeroU32),
) -> (std::cmp::Reverse<u32>, KeyframeId) {
    (std::cmp::Reverse(weight.get()), id)
}

impl SlamMap {
    pub fn new() -> Self {
        let lineage = MapLineageId::next();
        Self {
            instance_id: MapInstanceId::next(),
            points: SlotMap::with_key(),
            keyframes: SlotMap::with_key(),
            covisibility: CovisibilityGraph::default(),
            frame_to_keyframe: HashMap::new(),
            version: MapVersion::initial(lineage),
            mutation_lineage: lineage,
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

        let image_size = ImageSize::from(detections.dimensions());

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

        let next_version = self.version.next(self.mutation_lineage);
        let kf_id = KeyframeId::new(self.instance_id, self.keyframes.insert(entry));
        self.frame_to_keyframe.insert(frame_id, kf_id);
        self.version = next_version;
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
            .get_mut(raw_keyframe_id)
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

        let next_version = self.version.next(self.mutation_lineage);
        let point_id = MapPointId::new(
            self.instance_id,
            self.points.insert(MapPoint {
                position,
                descriptor,
                observations: vec![first_obs],
            }),
        );

        entry.set_point_ref(first_obs.index, point_id);
        self.version = next_version;
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
            .get_mut(raw_keyframe_id)
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

        let raw_point_id = point_id
            .raw_for(self.instance_id)
            .ok_or(MapError::MapPointNotFound(point_id))?;
        let point = self
            .points
            .get_mut(raw_point_id)
            .ok_or(MapError::MapPointNotFound(point_id))?;
        if point.observes_keyframe(obs.keyframe_id) {
            return Err(MapError::DuplicateObservation {
                point_id,
                keyframe_id: obs.keyframe_id,
            });
        }

        let next_version = self.version.next(self.mutation_lineage);
        self.covisibility
            .preflight_increment_observation_pairs(obs.keyframe_id, &point.observations);
        self.covisibility
            .increment_observation_pairs_after_preflight(obs.keyframe_id, &point.observations);

        point.add_observation(obs);
        entry.set_point_ref(obs.index, point_id);
        self.version = next_version;
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
        let next_version = self.version.next(self.mutation_lineage);
        point.update_descriptor(new_desc, blend);
        self.version = next_version;
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
        let next_version = self.version.next(self.mutation_lineage);
        point.set_position(position);
        self.version = next_version;
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
        let next_version = self.version.next(self.mutation_lineage);
        entry.set_pose(pose);
        self.version = next_version;
        Ok(())
    }

    pub fn remove_map_point(&mut self, point_id: MapPointId) -> Result<(), MapError> {
        let raw_point_id = self.raw_point_id(point_id)?;
        let point = self
            .points
            .get(raw_point_id)
            .ok_or(MapError::MapPointNotFound(point_id))?;
        let next_version = self.version.next(self.mutation_lineage);

        for (observation_index, obs) in point.observations.iter().enumerate() {
            assert!(
                point.observations[..observation_index]
                    .iter()
                    .all(|previous| previous.keyframe_id != obs.keyframe_id),
                "map point contains duplicate keyframe observation"
            );
            let raw_keyframe_id = obs
                .keyframe_id
                .raw_for(self.instance_id)
                .expect("map point observation keyframe belongs to another map");
            let entry = self
                .keyframes
                .get(raw_keyframe_id)
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
        self.covisibility
            .preflight_remove_point_observations(&point.observations);

        let point = self
            .points
            .remove(raw_point_id)
            .expect("map point existence validated before mutation");

        for obs in &point.observations {
            let raw_keyframe_id = obs
                .keyframe_id
                .raw_for(self.instance_id)
                .expect("map point observation scope validated before mutation");
            let entry = self
                .keyframes
                .get_mut(raw_keyframe_id)
                .expect("map point observation keyframe validated before mutation");
            assert_eq!(
                entry.clear_point_ref(obs.index),
                Some(point_id),
                "map point backreference changed after validation"
            );
        }
        self.covisibility
            .remove_point_observations_after_preflight(&point.observations);
        self.version = next_version;
        Ok(())
    }

    pub fn remove_keyframe(&mut self, keyframe_id: KeyframeId) -> Result<(), MapError> {
        let raw_keyframe_id = self.raw_keyframe_id(keyframe_id)?;
        let entry = self
            .keyframes
            .get(raw_keyframe_id)
            .ok_or(MapError::KeyframeNotFound(keyframe_id))?;
        let next_version = self.version.next(self.mutation_lineage);

        assert_eq!(
            entry.keypoints.len(),
            entry.point_refs.len(),
            "keyframe keypoint and point-reference lengths differ"
        );
        assert!(
            self.frame_to_keyframe.get(&entry.frame_id).copied() == Some(keyframe_id),
            "keyframe frame index is missing or mismatched"
        );
        let mut expected_covisibility = HashMap::new();
        for (index, maybe_point_id) in entry.point_refs.iter().enumerate() {
            let Some(point_id) = *maybe_point_id else {
                continue;
            };
            let raw_point_id = point_id
                .raw_for(self.instance_id)
                .expect("keyframe map point belongs to another map");
            let point = self
                .points
                .get(raw_point_id)
                .expect("keyframe map point is missing");
            let mut reciprocal_index = None;
            for (observation_index, observation) in point.observations.iter().enumerate() {
                if point.observations[..observation_index]
                    .iter()
                    .any(|previous| previous.keyframe_id == observation.keyframe_id)
                {
                    if observation.keyframe_id == keyframe_id {
                        panic!("keyframe map point has duplicate reciprocal observations");
                    }
                    panic!("keyframe map point contains duplicate keyframe observations");
                }
                if observation.keyframe_id == keyframe_id {
                    reciprocal_index = Some(observation.index.as_usize());
                    continue;
                }

                let raw_observation_keyframe = observation
                    .keyframe_id
                    .raw_for(self.instance_id)
                    .expect("keyframe map point observation belongs to another map");
                let observation_keyframe = self
                    .keyframes
                    .get(raw_observation_keyframe)
                    .expect("keyframe map point observation keyframe is missing");
                assert!(
                    observation_keyframe
                        .point_refs
                        .get(observation.index.as_usize())
                        .copied()
                        .flatten()
                        == Some(point_id),
                    "keyframe map point observation backreference mismatch"
                );

                expected_covisibility
                    .entry(observation.keyframe_id)
                    .and_modify(|count: &mut NonZeroU32| {
                        *count = count
                            .checked_add(1)
                            .filter(|next| next.get() <= MAX_COVISIBILITY_COUNT)
                            .expect("covisibility count exceeds the map point key space");
                    })
                    .or_insert(NonZeroU32::MIN);
            }
            let Some(reciprocal_index) = reciprocal_index else {
                panic!("keyframe map point has no reciprocal observation");
            };
            assert_eq!(
                reciprocal_index, index,
                "keyframe map point observation index mismatch"
            );
        }
        self.covisibility.preflight_remove_keyframe(keyframe_id);
        self.covisibility
            .preflight_keyframe_counts(keyframe_id, &expected_covisibility);

        let entry = self
            .keyframes
            .remove(raw_keyframe_id)
            .expect("keyframe existence validated before mutation");
        assert_eq!(
            self.frame_to_keyframe.remove(&entry.frame_id),
            Some(keyframe_id),
            "keyframe frame index changed after validation"
        );
        self.covisibility
            .remove_keyframe_after_preflight(keyframe_id);

        for point_id in entry.map_point_ids() {
            let raw_point_id = point_id
                .raw_for(self.instance_id)
                .expect("keyframe map point scope validated before mutation");
            let orphaned = {
                let point = self
                    .points
                    .get_mut(raw_point_id)
                    .expect("keyframe map point existence validated before mutation");
                assert!(
                    point.remove_observation_for(keyframe_id),
                    "keyframe reciprocal observation changed after validation"
                );
                point.observations.is_empty()
            };
            if orphaned {
                self.points
                    .remove(raw_point_id)
                    .expect("orphaned map point disappeared during keyframe removal");
            }
        }
        self.version = next_version;
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
        // Each removal advances the generation once. Prove that the complete
        // batch fits before the first mutation so exhaustion cannot expose a
        // partially culled map to a caller that catches the panic.
        let expected_generation = self
            .version
            .generation
            .checked_advance_by(count)
            .expect("map generation space exhausted");
        for id in to_remove {
            self.remove_map_point(id)
                .expect("map point collected for culling must still exist");
        }
        debug_assert_eq!(self.generation(), expected_generation);
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

    /// Returns the seed followed by at most `max - 1` covisible keyframes.
    ///
    /// Neighbors are ordered by descending shared-point count, then by keyframe identity.
    pub fn covisible_window(
        &self,
        seed: KeyframeId,
        max: NonZeroUsize,
    ) -> Result<Vec<KeyframeId>, MapError> {
        let raw_seed = self.raw_keyframe_id(seed)?;
        if !self.keyframes.contains_key(raw_seed) {
            return Err(MapError::KeyframeNotFound(seed));
        }

        let limit = max.get() - 1;
        if limit == 0 {
            return Ok(vec![seed]);
        }

        let neighbors = match self.covisibility.neighbors(seed) {
            Some(neighbors) => neighbors,
            None => return Ok(vec![seed]),
        };

        let mut sorted: Vec<(KeyframeId, NonZeroU32)> =
            neighbors.iter().map(|(&id, &w)| (id, w)).collect();
        sorted.sort_unstable_by_key(covisibility_window_rank);

        let mut window = Vec::with_capacity(limit.min(sorted.len()) + 1);
        window.push(seed);
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
        for (&a, neighbors) in &self.covisibility.edges {
            assert!(
                !neighbors.is_empty(),
                "covisibility graph contains an empty adjacency bucket"
            );
            for (&b, weight) in neighbors {
                assert_eq!(
                    self.covisibility.symmetric_pair_weight(a, b),
                    Some(*weight),
                    "covisibility snapshot pair changed during immutable traversal"
                );
                if a > b {
                    continue;
                }
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
        self.version.generation
    }

    pub fn snapshot(&self) -> MapSnapshot {
        MapSnapshot {
            instance_id: self.instance_id,
            version: self.version,
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
    CovisibilityEmptyAdjacency {
        keyframe_id: KeyframeId,
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
            MapInvariantError::CovisibilityEmptyAdjacency { keyframe_id } => write!(
                f,
                "covisibility adjacency bucket is empty: keyframe={keyframe_id:?}"
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
                let forward = expected_covisibility
                    .entry((obs_a.keyframe_id, obs_b.keyframe_id))
                    .or_insert(0);
                *forward = forward
                    .checked_add(1)
                    .expect("covisibility count exceeds the map point key space");
                let reverse = expected_covisibility
                    .entry((obs_b.keyframe_id, obs_a.keyframe_id))
                    .or_insert(0);
                *reverse = reverse
                    .checked_add(1)
                    .expect("covisibility count exceeds the map point key space");
            }
        }
    }

    for (&a, neighbors) in &map.covisibility.edges {
        if neighbors.is_empty() {
            return Err(MapInvariantError::CovisibilityEmptyAdjacency { keyframe_id: a });
        }
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

    fn add_test_keyframe(map: &mut SlamMap, sequence: u64, keypoint_count: usize) -> KeyframeId {
        map.add_keyframe(
            FrameId::new(sequence),
            Timestamp::from_nanos(i64::try_from(sequence).expect("test sequence fits i64")),
            WorldToCamera::identity(),
            ImageSize::try_new(640, 480).expect("valid image size"),
            make_keypoints(keypoint_count),
        )
        .expect("test keyframe")
    }

    fn add_test_keyframes<const N: usize>(
        map: &mut SlamMap,
        first_sequence: u64,
        keypoint_counts: [usize; N],
    ) -> [KeyframeId; N] {
        std::array::from_fn(|index| {
            add_test_keyframe(
                map,
                first_sequence + u64::try_from(index).expect("test index fits u64"),
                keypoint_counts[index],
            )
        })
    }

    fn test_observations<const N: usize>(
        map: &SlamMap,
        keyframes: [KeyframeId; N],
        keypoint_index: usize,
    ) -> [KeyframeKeypoint; N] {
        keyframes.map(|keyframe_id| {
            map.keyframe_keypoint(keyframe_id, keypoint_index)
                .expect("test observation")
        })
    }

    fn add_shared_test_point(
        map: &mut SlamMap,
        observations: &[KeyframeKeypoint],
        x: f32,
    ) -> MapPointId {
        let (&first, remaining) = observations
            .split_first()
            .expect("shared test point needs an observation");
        let point_id = map
            .add_map_point(WorldPoint3::new(x, 0.0, 1.0), make_descriptor(), first)
            .expect("shared test point");
        for &observation in remaining {
            map.add_observation(point_id, observation)
                .expect("shared test observation");
        }
        point_id
    }

    fn assert_panics_with<T>(expected: &str, operation: impl FnOnce() -> T) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation));
        let Err(payload) = result else {
            panic!("operation must panic with {expected:?}");
        };
        assert_eq!(crate::panic_payload_to_string(payload.as_ref()), expected);
    }

    fn assert_generation_exhaustion<T>(operation: impl FnOnce() -> T) {
        assert_panics_with("map generation space exhausted", operation);
    }

    #[test]
    fn monotonic_id_allocator_exhausts_without_wrapping() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(allocate_monotonic_id(&counter), Some(u64::MAX - 1));
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        assert_eq!(allocate_monotonic_id(&counter), None);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
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

            map.version.generation = MapGeneration(u64::MAX);
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
    fn map_errors_precede_generation_exhaustion_without_mutation() {
        let mut fixture = GenerationExhaustionFixture::new();
        let malformed = WorldPoint3::new(f32::NAN, 0.0, 1.0);
        assert!(matches!(
            fixture.map.add_map_point(
                malformed,
                make_descriptor(),
                fixture.first_observation,
            ),
            Err(MapError::InvalidMapPointPosition(
                crate::Point3Error::NonFinite { axis: 0, value }
            )) if value.is_nan()
        ));
        fixture.assert_unchanged();

        assert!(matches!(
            fixture.map.add_map_point(
                WorldPoint3::new(1.0, 2.0, 3.0),
                make_descriptor(),
                fixture.first_observation,
            ),
            Err(MapError::DetectionAlreadyAssociated { existing, .. })
                if existing == fixture.point_id
        ));
        fixture.assert_unchanged();

        let missing_point = MapPointId::default();
        assert!(matches!(
            fixture
                .map
                .add_observation(missing_point, fixture.first_observation),
            Err(MapError::DetectionAlreadyAssociated { existing, .. })
                if existing == fixture.point_id
        ));
        fixture.assert_unchanged();

        let stale_point = MapPointId::new(fixture.map.instance_id, RawMapPointId::default());
        assert!(fixture.map.point(stale_point).is_none());
        assert!(matches!(
            fixture
                .map
                .add_observation(stale_point, fixture.second_free_keypoint),
            Err(MapError::MapPointNotFound(id)) if id == stale_point
        ));
        fixture.assert_unchanged();

        assert!(matches!(
            fixture
                .map
                .add_observation(missing_point, fixture.second_free_keypoint),
            Err(MapError::MapPointNotFound(id)) if id == missing_point
        ));
        fixture.assert_unchanged();

        assert!(matches!(
            fixture
                .map
                .add_observation(fixture.point_id, fixture.first_free_keypoint),
            Err(MapError::DuplicateObservation {
                point_id,
                keyframe_id,
            }) if point_id == fixture.point_id && keyframe_id == fixture.first_keyframe
        ));
        fixture.assert_unchanged();

        assert!(matches!(
            fixture.map.remove_map_point(missing_point),
            Err(MapError::MapPointNotFound(id)) if id == missing_point
        ));
        fixture.assert_unchanged();
        assert!(matches!(
            fixture.map.remove_map_point(stale_point),
            Err(MapError::MapPointNotFound(id)) if id == stale_point
        ));
        fixture.assert_unchanged();

        let missing_keyframe = KeyframeId::default();
        assert!(matches!(
            fixture.map.remove_keyframe(missing_keyframe),
            Err(MapError::KeyframeNotFound(id)) if id == missing_keyframe
        ));
        fixture.assert_unchanged();
    }

    #[test]
    fn removal_backreference_failures_precede_mutation() {
        let mut map = SlamMap::new();
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let first_keyframe = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                WorldToCamera::identity(),
                size,
                make_keypoints(2),
            )
            .expect("first keyframe");
        let second_keyframe = map
            .add_keyframe(
                FrameId::new(2),
                Timestamp::from_nanos(2),
                WorldToCamera::identity(),
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
                WorldPoint3::new(0.0, 0.0, 1.0),
                make_descriptor(),
                first_shared,
            )
            .expect("shared point");
        map.add_observation(shared_point, second_shared)
            .expect("second shared observation");
        let orphan_point = map
            .add_map_point(
                WorldPoint3::new(1.0, 0.0, 1.0),
                make_descriptor(),
                orphan_observation,
            )
            .expect("orphan point");
        assert_map_invariants(&map).expect("valid removal fixture");

        let mut broken_point_backref = map.clone();
        let raw_second_keyframe = second_keyframe
            .raw_for(broken_point_backref.instance_id)
            .expect("same-map keyframe");
        assert_eq!(
            broken_point_backref
                .keyframes
                .get_mut(raw_second_keyframe)
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
        assert_eq!(
            broken_point_backref
                .covisibility()
                .covisibility_count(first_keyframe, second_keyframe),
            1
        );

        let mut broken_point_observation = map.clone();
        let raw_orphan_point = orphan_point
            .raw_for(broken_point_observation.instance_id)
            .expect("same-map point");
        broken_point_observation
            .points
            .get_mut(raw_orphan_point)
            .expect("orphan point")
            .observations
            .clear();
        let keyframe_snapshot = broken_point_observation.snapshot();
        assert_panics_with("keyframe map point has no reciprocal observation", || {
            broken_point_observation.remove_keyframe(first_keyframe)
        });
        assert_eq!(broken_point_observation.snapshot(), keyframe_snapshot);
        assert!(broken_point_observation.keyframe(first_keyframe).is_some());
        assert_eq!(
            broken_point_observation
                .frame_to_keyframe
                .get(&FrameId::new(1)),
            Some(&first_keyframe)
        );
        assert_eq!(
            broken_point_observation
                .point(shared_point)
                .expect("shared point")
                .observation_count(),
            2
        );
        assert!(broken_point_observation.point(orphan_point).is_some());
        assert_eq!(
            broken_point_observation
                .covisibility()
                .covisibility_count(first_keyframe, second_keyframe),
            1
        );

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
    fn cull_points_preflights_the_complete_generation_range() {
        let mut map = SlamMap::new();
        let keyframe_id = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                WorldToCamera::identity(),
                ImageSize::try_new(640, 480).expect("valid size"),
                make_keypoints(2),
            )
            .expect("keyframe");
        let observations = [
            map.keyframe_keypoint(keyframe_id, 0)
                .expect("first observation"),
            map.keyframe_keypoint(keyframe_id, 1)
                .expect("second observation"),
        ];
        let points = [
            map.add_map_point(
                WorldPoint3::new(0.0, 0.0, 1.0),
                make_descriptor(),
                observations[0],
            )
            .expect("first point"),
            map.add_map_point(
                WorldPoint3::new(1.0, 0.0, 1.0),
                make_descriptor(),
                observations[1],
            )
            .expect("second point"),
        ];
        let mut exact_fit = map.clone();

        map.version.generation = MapGeneration(u64::MAX - 1);
        let before = map.snapshot();
        assert_generation_exhaustion(|| map.cull_points(2));
        assert_eq!(map.snapshot(), before);
        assert_eq!(map.num_points(), 2);
        for (&point_id, &observation) in points.iter().zip(&observations) {
            assert!(map.point(point_id).is_some());
            assert_eq!(
                map.map_point_for_keypoint(observation)
                    .expect("keypoint association"),
                Some(point_id)
            );
        }
        assert_map_invariants(&map).expect("rejected cull preserves invariants");

        exact_fit.version.generation = MapGeneration(u64::MAX - 2);
        assert_eq!(exact_fit.cull_points(2), 2);
        assert_eq!(exact_fit.generation(), MapGeneration(u64::MAX));
        assert_eq!(exact_fit.num_points(), 0);
        for (&point_id, &observation) in points.iter().zip(&observations) {
            assert!(exact_fit.point(point_id).is_none());
            assert_eq!(
                exact_fit
                    .map_point_for_keypoint(observation)
                    .expect("cleared keypoint association"),
                None
            );
        }
        assert_map_invariants(&exact_fit).expect("exact-fit cull preserves invariants");

        let exhausted = exact_fit.snapshot();
        assert_eq!(exact_fit.cull_points(2), 0);
        assert_eq!(exact_fit.snapshot(), exhausted);
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
    fn independently_mutated_clones_have_distinct_snapshots() {
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
        let source_snapshot = map.snapshot();
        let mut first = map.clone();
        let mut second = map.clone();

        assert_eq!(first.snapshot(), source_snapshot);
        assert_eq!(second.snapshot(), source_snapshot);
        assert!(
            first
                .snapshot()
                .shares_mutation_lineage_with(source_snapshot)
        );
        assert!(
            second
                .snapshot()
                .shares_mutation_lineage_with(source_snapshot)
        );

        let first_pose = WorldToCamera::from_legacy_pose(crate::Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [1.0, 0.0, 0.0],
        ));
        let second_pose = WorldToCamera::from_legacy_pose(crate::Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 1.0, 0.0],
        ));
        first
            .set_keyframe_pose(keyframe_id, first_pose)
            .expect("mutate first branch");
        second
            .set_keyframe_pose(keyframe_id, second_pose)
            .expect("mutate second branch");

        assert_eq!(first.generation(), second.generation());
        assert_eq!(
            first.snapshot().instance_id(),
            second.snapshot().instance_id()
        );
        assert_ne!(first.snapshot(), second.snapshot());
        assert!(
            !first
                .snapshot()
                .shares_mutation_lineage_with(source_snapshot)
        );
        assert!(
            !second
                .snapshot()
                .shares_mutation_lineage_with(source_snapshot)
        );
        assert!(
            !first
                .snapshot()
                .shares_mutation_lineage_with(second.snapshot())
        );

        map.set_keyframe_pose(keyframe_id, first_pose)
            .expect("mutate the source branch");
        assert!(map.snapshot().shares_mutation_lineage_with(source_snapshot));
        assert_ne!(
            first
                .keyframe(keyframe_id)
                .expect("first keyframe")
                .pose()
                .translation(),
            second
                .keyframe(keyframe_id)
                .expect("second keyframe")
                .pose()
                .translation()
        );
    }

    #[test]
    fn concurrent_clone_mutations_have_unique_snapshots() {
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

        let snapshots: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|index| {
                    let source = &map;
                    scope.spawn(move || {
                        let mut branch = source.clone();
                        let pose = WorldToCamera::from_legacy_pose(crate::Pose::from_rt(
                            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            [index as f32 + 1.0, 0.0, 0.0],
                        ));
                        branch
                            .set_keyframe_pose(keyframe_id, pose)
                            .expect("mutate branch");
                        branch.snapshot()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("clone worker"))
                .collect()
        });

        assert!(snapshots.iter().all(|snapshot| {
            snapshot.instance_id() == map.snapshot().instance_id()
                && snapshot.generation().as_u64() == map.generation().as_u64() + 1
        }));
        assert_eq!(
            snapshots.iter().copied().collect::<HashSet<_>>().len(),
            snapshots.len()
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
    fn covisibility_pair_updates_are_exact_and_symmetric() {
        let first = KeyframeId::for_test(0);
        let second = KeyframeId::for_test(1);
        let mut graph = CovisibilityGraph::default();

        graph.increment_pair(first, second);
        assert_eq!(graph.covisibility_count(first, second), 1);
        assert_eq!(graph.covisibility_count(second, first), 1);

        graph.increment_pair(first, second);
        assert_eq!(graph.covisibility_count(first, second), 2);
        assert_eq!(graph.covisibility_count(second, first), 2);

        graph.decrement_pair(first, second);
        assert_eq!(graph.covisibility_count(first, second), 1);
        assert_eq!(graph.covisibility_count(second, first), 1);
        graph.decrement_pair(first, second);
        assert_eq!(graph.covisibility_count(first, second), 0);
        assert!(graph.edges.is_empty());

        graph.set_pair_weight(first, second, NonZeroU32::new(MAX_COVISIBILITY_COUNT - 1));
        graph.increment_pair(first, second);
        assert_eq!(
            graph.covisibility_count(first, second),
            MAX_COVISIBILITY_COUNT
        );
        assert_eq!(
            graph.covisibility_count(second, first),
            MAX_COVISIBILITY_COUNT
        );

        let before = graph.edges.clone();
        assert_panics_with("covisibility count exceeds the map point key space", || {
            graph.increment_pair(first, second)
        });
        assert_eq!(graph.edges, before);

        graph.set_pair_weight(first, second, NonZeroU32::new(u32::MAX));
        let before = graph.edges.clone();
        assert_panics_with("covisibility count exceeds the map point key space", || {
            graph.increment_pair(first, second)
        });
        assert_eq!(graph.edges, before);
    }

    #[test]
    fn covisibility_pair_updates_reject_corruption_without_mutation() {
        let first = KeyframeId::for_test(0);
        let second = KeyframeId::for_test(1);
        let one = NonZeroU32::MIN;
        let two = NonZeroU32::new(2).expect("nonzero weight");

        let mut missing_reverse = CovisibilityGraph::default();
        missing_reverse
            .edges
            .entry(first)
            .or_default()
            .insert(second, one);

        let mut asymmetric = CovisibilityGraph::default();
        asymmetric
            .edges
            .entry(first)
            .or_default()
            .insert(second, one);
        asymmetric
            .edges
            .entry(second)
            .or_default()
            .insert(first, two);

        let mut empty_adjacency = CovisibilityGraph::default();
        empty_adjacency.edges.insert(first, HashMap::new());

        for (expected, graph) in [
            (
                "covisibility graph contains a missing reverse edge",
                missing_reverse,
            ),
            (
                "covisibility graph contains asymmetric pair weights",
                asymmetric,
            ),
            (
                "covisibility graph contains an empty adjacency bucket",
                empty_adjacency,
            ),
        ] {
            let mut increment = graph.clone();
            let before = increment.edges.clone();
            assert_panics_with(expected, || increment.increment_pair(first, second));
            assert_eq!(increment.edges, before);

            let mut decrement = graph;
            let before = decrement.edges.clone();
            assert_panics_with(expected, || decrement.decrement_pair(first, second));
            assert_eq!(decrement.edges, before);
        }
    }

    #[test]
    fn covisibility_batch_failures_precede_map_mutation() {
        let mut insertion_map = SlamMap::new();
        let insertion_keyframes = add_test_keyframes(&mut insertion_map, 1, [1; 3]);
        let insertion_observations = test_observations(&insertion_map, insertion_keyframes, 0);
        let insertion_point =
            add_shared_test_point(&mut insertion_map, &insertion_observations[..2], 0.0);
        let third_observation = insertion_observations[2];
        insertion_map
            .covisibility
            .edges
            .entry(insertion_keyframes[2])
            .or_default()
            .insert(insertion_keyframes[1], NonZeroU32::MIN);

        let insertion_snapshot = insertion_map.snapshot();
        let insertion_edges = insertion_map.covisibility.edges.clone();
        assert_panics_with("covisibility graph contains a missing reverse edge", || {
            insertion_map.add_observation(insertion_point, third_observation)
        });
        assert_eq!(insertion_map.snapshot(), insertion_snapshot);
        assert_eq!(insertion_map.covisibility.edges, insertion_edges);
        assert_eq!(
            insertion_map
                .point(insertion_point)
                .expect("insertion point remains")
                .observations,
            insertion_observations[..2]
        );
        for observation in &insertion_observations[..2] {
            assert_eq!(
                insertion_map
                    .map_point_for_keypoint(*observation)
                    .expect("existing insertion backreference remains readable"),
                Some(insertion_point)
            );
        }
        assert_eq!(
            insertion_map
                .map_point_for_keypoint(third_observation)
                .expect("third backreference remains readable"),
            None
        );

        let mut removal_map = SlamMap::new();
        let removal_keyframes = add_test_keyframes(&mut removal_map, 11, [1; 3]);
        let removal_observations = test_observations(&removal_map, removal_keyframes, 0);
        let removal_point = add_shared_test_point(&mut removal_map, &removal_observations, 0.0);
        removal_map
            .covisibility
            .edges
            .get_mut(&removal_keyframes[2])
            .expect("third adjacency")
            .remove(&removal_keyframes[1])
            .expect("reverse edge to corrupt");

        let removal_snapshot = removal_map.snapshot();
        let removal_edges = removal_map.covisibility.edges.clone();
        assert_panics_with("covisibility graph contains a missing reverse edge", || {
            removal_map.remove_map_point(removal_point)
        });
        assert_eq!(removal_map.snapshot(), removal_snapshot);
        assert_eq!(removal_map.covisibility.edges, removal_edges);
        assert_eq!(
            removal_map
                .point(removal_point)
                .expect("removal point remains")
                .observations,
            removal_observations
        );
        for observation in removal_observations {
            assert_eq!(
                removal_map
                    .map_point_for_keypoint(observation)
                    .expect("removal backreference remains readable"),
                Some(removal_point)
            );
        }
    }

    #[test]
    fn keyframe_removal_preflights_incident_covisibility() {
        let mut map = SlamMap::new();
        let keyframes = add_test_keyframes(&mut map, 21, [1, 2, 2, 1]);
        let shared_observations = test_observations(&map, keyframes, 0);
        let shared_point = add_shared_test_point(&mut map, &shared_observations[..3], 0.0);
        let second_pair_observations = [
            map.keyframe_keypoint(keyframes[1], 1)
                .expect("second pair observation"),
            map.keyframe_keypoint(keyframes[2], 1)
                .expect("third pair observation"),
        ];
        let second_pair_point = add_shared_test_point(&mut map, &second_pair_observations, 1.0);
        assert_eq!(
            map.covisibility
                .covisibility_count(keyframes[1], keyframes[2]),
            2
        );
        assert_map_invariants(&map).expect("valid keyframe-removal fixture");

        let mut incoming_only = map.clone();
        incoming_only
            .covisibility
            .edges
            .get_mut(&keyframes[0])
            .expect("first adjacency")
            .remove(&keyframes[2])
            .expect("outgoing edge to corrupt");

        let mut outgoing_only = map.clone();
        outgoing_only
            .covisibility
            .edges
            .get_mut(&keyframes[2])
            .expect("third adjacency")
            .remove(&keyframes[0])
            .expect("reverse edge to corrupt");

        let mut asymmetric = map.clone();
        *asymmetric
            .covisibility
            .edges
            .get_mut(&keyframes[2])
            .expect("third adjacency")
            .get_mut(&keyframes[0])
            .expect("reverse edge") = NonZeroU32::new(2).expect("nonzero weight");

        let mut missing_pair = map.clone();
        for (from, to) in [(keyframes[0], keyframes[2]), (keyframes[2], keyframes[0])] {
            missing_pair
                .covisibility
                .edges
                .get_mut(&from)
                .expect("pair adjacency")
                .remove(&to)
                .expect("pair edge to corrupt");
        }

        let mut wrong_count = map.clone();
        for (from, to) in [(keyframes[0], keyframes[2]), (keyframes[2], keyframes[0])] {
            *wrong_count
                .covisibility
                .edges
                .get_mut(&from)
                .expect("pair adjacency")
                .get_mut(&to)
                .expect("pair edge") = NonZeroU32::new(2).expect("nonzero weight");
        }

        let mut unexpected_pair = map.clone();
        unexpected_pair.covisibility.set_pair_weight(
            keyframes[0],
            keyframes[3],
            Some(NonZeroU32::MIN),
        );

        for (expected, mut corrupt) in [
            (
                "covisibility graph contains an incoming-only edge",
                incoming_only,
            ),
            (
                "covisibility graph contains a missing reverse edge",
                outgoing_only,
            ),
            (
                "covisibility graph contains asymmetric pair weights",
                asymmetric,
            ),
            (
                "covisibility graph is missing or miscounts an incident pair",
                missing_pair,
            ),
            (
                "covisibility graph is missing or miscounts an incident pair",
                wrong_count,
            ),
            (
                "covisibility graph contains an unexpected incident pair",
                unexpected_pair,
            ),
        ] {
            let corrupt_snapshot = corrupt.snapshot();
            let corrupt_edges = corrupt.covisibility.edges.clone();
            assert_panics_with(expected, || corrupt.remove_keyframe(keyframes[0]));
            assert_eq!(corrupt.snapshot(), corrupt_snapshot);
            assert_eq!(corrupt.covisibility.edges, corrupt_edges);
            assert_eq!(corrupt.num_keyframes(), 4);
            assert_eq!(corrupt.num_points(), 2);
            assert_eq!(
                corrupt.keyframe_by_frame(FrameId::new(21)),
                Some(keyframes[0])
            );
            assert_eq!(
                corrupt
                    .point(shared_point)
                    .expect("shared point remains")
                    .observations,
                shared_observations[..3]
            );
            for observation in &shared_observations[..3] {
                assert_eq!(
                    corrupt
                        .map_point_for_keypoint(*observation)
                        .expect("shared backreference remains readable"),
                    Some(shared_point)
                );
            }
        }

        map.remove_keyframe(keyframes[0])
            .expect("valid keyframe removal");
        assert!(map.keyframe(keyframes[0]).is_none());
        assert_eq!(map.keyframe_by_frame(FrameId::new(21)), None);
        assert_eq!(
            map.point(shared_point)
                .expect("shared point remains observed")
                .observation_count(),
            2
        );
        assert!(map.point(second_pair_point).is_some());
        assert_eq!(
            map.covisibility
                .covisibility_count(keyframes[1], keyframes[2]),
            2
        );
        assert!(map.covisibility.neighbors(keyframes[0]).is_none());
        assert_map_invariants(&map).expect("after valid keyframe removal");
    }

    #[test]
    fn covisibility_ratio_preserves_f32_precision_at_the_integer_boundary() {
        assert_eq!(
            covisibility_ratio_from_counts(16_777_216, 16_777_217),
            f32::from_bits(0x3f7f_ffff)
        );
    }

    #[test]
    fn covisibility_ratio_validates_both_keyframes_before_returning_zero() {
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let mut map = SlamMap::new();
        let first = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                WorldToCamera::identity(),
                size,
                make_keypoints(1),
            )
            .expect("first keyframe");
        let second = map
            .add_keyframe(
                FrameId::new(2),
                Timestamp::from_nanos(2),
                WorldToCamera::identity(),
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

        let mut other_map = SlamMap::new();
        let foreign = other_map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                WorldToCamera::identity(),
                size,
                make_keypoints(1),
            )
            .expect("foreign keyframe");
        for (a, b, missing) in [(foreign, first, foreign), (first, foreign, foreign)] {
            assert!(matches!(
                map.covisibility_ratio(a, b),
                Err(MapError::KeyframeNotFound(id)) if id == missing
            ));
        }

        let mut third_map = SlamMap::new();
        let second_foreign = third_map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                WorldToCamera::identity(),
                size,
                make_keypoints(1),
            )
            .expect("second foreign keyframe");
        assert!(matches!(
            map.covisibility_ratio(foreign, second_foreign),
            Err(MapError::KeyframeNotFound(id)) if id == foreign
        ));

        map.remove_keyframe(second).expect("remove second keyframe");
        let replacement = map
            .add_keyframe(
                FrameId::new(3),
                Timestamp::from_nanos(3),
                WorldToCamera::identity(),
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
        let size = ImageSize::try_new(640, 480).expect("valid size");
        let mut map = SlamMap::new();
        let first = map
            .add_keyframe(
                FrameId::new(1),
                Timestamp::from_nanos(1),
                WorldToCamera::identity(),
                size,
                make_keypoints(1),
            )
            .expect("first keyframe");
        let second = map
            .add_keyframe(
                FrameId::new(2),
                Timestamp::from_nanos(2),
                WorldToCamera::identity(),
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
    fn covisible_window_bounds_capacity_by_available_results() {
        let mut map = SlamMap::new();
        let seed = add_test_keyframe(&mut map, 1, 1);

        assert_eq!(
            map.covisible_window(
                seed,
                NonZeroUsize::new(usize::MAX).expect("non-zero maximum"),
            )
            .expect("finite covisible window"),
            vec![seed]
        );
    }

    #[test]
    fn covisible_window_orders_equal_weights_by_keyframe_id() {
        let mut map = SlamMap::new();
        let [seed, high_a, high_b, low_a, low_b] = add_test_keyframes(&mut map, 1, [6, 2, 2, 1, 1]);
        let weighted_neighbors = [(high_a, 2_usize), (high_b, 2), (low_a, 1), (low_b, 1)];

        let mut seed_keypoint_index = 0;
        for (neighbor, weight) in weighted_neighbors {
            for neighbor_keypoint_index in 0..weight {
                let observations = [
                    map.keyframe_keypoint(seed, seed_keypoint_index)
                        .expect("seed observation"),
                    map.keyframe_keypoint(neighbor, neighbor_keypoint_index)
                        .expect("neighbor observation"),
                ];
                add_shared_test_point(&mut map, &observations, seed_keypoint_index as f32);
                seed_keypoint_index += 1;
            }
        }

        let one = NonZeroU32::new(1).expect("non-zero weight");
        let two = NonZeroU32::new(2).expect("non-zero weight");
        let mut high_ids = [high_a, high_b];
        high_ids.sort_unstable();
        let mut low_ids = [low_a, low_b];
        low_ids.sort_unstable();
        let expected_rank = vec![
            (high_ids[0], two),
            (high_ids[1], two),
            (low_ids[0], one),
            (low_ids[1], one),
        ];

        let mut forward = vec![(high_a, two), (high_b, two), (low_a, one), (low_b, one)];
        let mut reverse = forward.clone();
        reverse.reverse();
        forward.sort_unstable_by_key(covisibility_window_rank);
        reverse.sort_unstable_by_key(covisibility_window_rank);
        assert_eq!(forward, expected_rank);
        assert_eq!(reverse, expected_rank);

        assert_eq!(
            map.covisible_window(seed, NonZeroUsize::new(1).expect("non-zero window"))
                .expect("seed-only covisible window"),
            vec![seed]
        );
        assert_eq!(
            map.covisible_window(seed, NonZeroUsize::new(4).expect("non-zero window"))
                .expect("truncated covisible window"),
            vec![seed, high_ids[0], high_ids[1], low_ids[0]]
        );
        assert_eq!(
            map.covisible_window(
                seed,
                NonZeroUsize::new(usize::MAX).expect("non-zero maximum"),
            )
            .expect("complete covisible window"),
            vec![seed, high_ids[0], high_ids[1], low_ids[0], low_ids[1]]
        );
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
