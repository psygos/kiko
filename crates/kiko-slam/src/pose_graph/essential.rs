use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;

use crate::Pose64;
use crate::map::{KeyframeId, SlamMap};

use super::{PoseGraphEdge, PoseGraphEdgeError, scaled_identity6};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EssentialEdgeKind {
    SpanningTree,
    StrongCovisibility,
    Loop,
}

#[derive(Clone, Debug)]
pub struct EssentialEdge {
    a: KeyframeId,
    b: KeyframeId,
    kind: EssentialEdgeKind,
    /// Camera-`a` to camera-`b` transform for world-to-camera keyframe poses.
    relative_pose: Pose64,
    information: [[f64; 6]; 6],
}

impl EssentialEdge {
    pub fn try_new(
        a: KeyframeId,
        b: KeyframeId,
        kind: EssentialEdgeKind,
        relative_pose: Pose64,
        information: [[f64; 6]; 6],
    ) -> Result<Self, EssentialGraphError> {
        if a == b {
            return Err(EssentialGraphError::SelfEdge { keyframe_id: a });
        }
        PoseGraphEdge::try_new(0, 1, relative_pose, information)
            .map_err(EssentialGraphError::InvalidEdge)?;
        Ok(Self {
            a,
            b,
            kind,
            relative_pose,
            information,
        })
    }

    pub fn a(&self) -> KeyframeId {
        self.a
    }

    pub fn b(&self) -> KeyframeId {
        self.b
    }

    pub fn kind(&self) -> EssentialEdgeKind {
        self.kind
    }

    pub fn relative_pose(&self) -> Pose64 {
        self.relative_pose
    }

    pub fn information(&self) -> [[f64; 6]; 6] {
        self.information
    }
}

#[derive(Clone, Debug)]
pub struct EssentialGraphSnapshot {
    pub parent: HashMap<KeyframeId, KeyframeId>,
    pub order: Vec<KeyframeId>,
    pub spanning_edges: Vec<EssentialEdge>,
    pub strong_covis_edges: Vec<EssentialEdge>,
    pub loop_edges: Vec<EssentialEdge>,
    pub strong_threshold: u32,
}

#[derive(Clone, Debug)]
pub struct PoseGraphInput {
    pub keyframe_ids: Vec<KeyframeId>,
    pub edges: Vec<PoseGraphEdge>,
}

#[derive(Clone, Debug)]
pub struct EssentialGraph {
    parent: HashMap<KeyframeId, KeyframeId>,
    order: Vec<KeyframeId>,
    spanning_edges: Vec<EssentialEdge>,
    strong_covis_edges: Vec<EssentialEdge>,
    loop_edges: Vec<EssentialEdge>,
    strong_threshold: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EssentialGraphError {
    KeyframeNotFound { keyframe_id: KeyframeId },
    RootRemovalDenied { keyframe_id: KeyframeId },
    SelfEdge { keyframe_id: KeyframeId },
    InvalidEdge(PoseGraphEdgeError),
}

impl std::fmt::Display for EssentialGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EssentialGraphError::KeyframeNotFound { keyframe_id } => {
                write!(f, "essential graph keyframe not found: {keyframe_id:?}")
            }
            EssentialGraphError::RootRemovalDenied { keyframe_id } => {
                write!(
                    f,
                    "cannot remove essential graph root keyframe: {keyframe_id:?}"
                )
            }
            EssentialGraphError::SelfEdge { keyframe_id } => {
                write!(
                    f,
                    "essential graph edge cannot be a self-edge: {keyframe_id:?}"
                )
            }
            EssentialGraphError::InvalidEdge(err) => {
                write!(f, "invalid essential graph edge: {err}")
            }
        }
    }
}

impl std::error::Error for EssentialGraphError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidEdge(err) => Some(err),
            Self::KeyframeNotFound { .. }
            | Self::RootRemovalDenied { .. }
            | Self::SelfEdge { .. } => None,
        }
    }
}

impl EssentialGraph {
    pub fn new(strong_threshold: u32) -> Self {
        Self {
            parent: HashMap::new(),
            order: Vec::new(),
            spanning_edges: Vec::new(),
            strong_covis_edges: Vec::new(),
            loop_edges: Vec::new(),
            strong_threshold,
        }
    }

    pub fn parent_of(&self, keyframe_id: KeyframeId) -> Option<KeyframeId> {
        self.parent.get(&keyframe_id).copied()
    }

    pub fn add_keyframe(
        &mut self,
        keyframe_id: KeyframeId,
        covisibility: Option<&HashMap<KeyframeId, NonZeroU32>>,
        map: &SlamMap,
    ) {
        if self.parent.contains_key(&keyframe_id) {
            return;
        }
        self.order.push(keyframe_id);
        if self.parent.is_empty() {
            self.parent.insert(keyframe_id, keyframe_id);
            return;
        }

        let Some(neighbors) = covisibility else {
            self.parent.insert(keyframe_id, keyframe_id);
            return;
        };
        if neighbors.is_empty() {
            self.parent.insert(keyframe_id, keyframe_id);
            return;
        }

        let mut strongest = None;
        for (&neighbor, &weight) in neighbors {
            let neighbor_is_registered = self.parent.contains_key(&neighbor);
            if neighbor_is_registered
                && strongest
                    .as_ref()
                    .is_none_or(|(_, best_w): &(KeyframeId, u32)| weight.get() > *best_w)
            {
                strongest = Some((neighbor, weight.get()));
            }

            if neighbor_is_registered
                && weight.get() >= self.strong_threshold
                && !contains_edge(&self.strong_covis_edges, keyframe_id, neighbor)
                && let Some(relative_pose) = relative_pose(map, keyframe_id, neighbor)
            {
                self.strong_covis_edges.push(
                    EssentialEdge::try_new(
                        keyframe_id,
                        neighbor,
                        EssentialEdgeKind::StrongCovisibility,
                        relative_pose,
                        scaled_identity6(weight.get() as f64),
                    )
                    .expect("nonzero covisibility weight must produce a valid edge"),
                );
            }
        }

        let Some((parent, weight)) = strongest else {
            self.parent.insert(keyframe_id, keyframe_id);
            return;
        };
        // The parent relationship is already represented by the spanning edge.
        // Keeping the same pair as a strong-covisibility edge double-weights it.
        self.strong_covis_edges
            .retain(|edge| !same_endpoints(edge, parent, keyframe_id));
        self.parent.insert(keyframe_id, parent);
        if let Some(relative_pose) = relative_pose(map, parent, keyframe_id) {
            self.spanning_edges.push(
                EssentialEdge::try_new(
                    parent,
                    keyframe_id,
                    EssentialEdgeKind::SpanningTree,
                    relative_pose,
                    scaled_identity6(weight as f64),
                )
                .expect("nonzero spanning weight must produce a valid edge"),
            );
        }
    }

    pub fn add_loop_edge(&mut self, edge: EssentialEdge) {
        if let std::collections::hash_map::Entry::Vacant(entry) = self.parent.entry(edge.a) {
            entry.insert(edge.a);
            self.order.push(edge.a);
        }
        if let std::collections::hash_map::Entry::Vacant(entry) = self.parent.entry(edge.b) {
            entry.insert(edge.b);
            self.order.push(edge.b);
        }
        self.loop_edges.push(edge);
    }

    pub fn remove_keyframe(
        &mut self,
        keyframe_id: KeyframeId,
        map: &SlamMap,
    ) -> Result<(), EssentialGraphError> {
        let parent = self
            .parent
            .get(&keyframe_id)
            .copied()
            .ok_or(EssentialGraphError::KeyframeNotFound { keyframe_id })?;
        if parent == keyframe_id {
            return Err(EssentialGraphError::RootRemovalDenied { keyframe_id });
        }

        let children: Vec<KeyframeId> = self
            .parent
            .iter()
            .filter_map(|(&child, &child_parent)| {
                if child_parent == keyframe_id && child != keyframe_id {
                    Some(child)
                } else {
                    None
                }
            })
            .collect();

        for child in &children {
            if let Some(entry) = self.parent.get_mut(child) {
                *entry = parent;
            }
        }

        self.parent.remove(&keyframe_id);
        self.order.retain(|&id| id != keyframe_id);
        self.spanning_edges
            .retain(|edge| edge.a != keyframe_id && edge.b != keyframe_id);
        self.strong_covis_edges
            .retain(|edge| edge.a != keyframe_id && edge.b != keyframe_id);
        self.loop_edges
            .retain(|edge| edge.a != keyframe_id && edge.b != keyframe_id);

        for child in children {
            if contains_edge(&self.spanning_edges, parent, child) {
                continue;
            }
            if let Some(relative_pose) = relative_pose(map, parent, child) {
                self.spanning_edges.push(EssentialEdge::try_new(
                    parent,
                    child,
                    EssentialEdgeKind::SpanningTree,
                    relative_pose,
                    scaled_identity6(1.0),
                )?);
            }
        }

        Ok(())
    }

    /// Iterate over all essential edges (spanning tree + strong covisibility + loop).
    fn iter_all_edges(&self) -> impl Iterator<Item = &EssentialEdge> {
        self.spanning_edges
            .iter()
            .chain(self.strong_covis_edges.iter())
            .chain(self.loop_edges.iter())
    }

    pub fn snapshot(&self) -> EssentialGraphSnapshot {
        EssentialGraphSnapshot {
            parent: self.parent.clone(),
            order: self.order.clone(),
            spanning_edges: self.spanning_edges.clone(),
            strong_covis_edges: self.strong_covis_edges.clone(),
            loop_edges: self.loop_edges.clone(),
            strong_threshold: self.strong_threshold,
        }
    }

    pub fn pose_graph_input(&self) -> PoseGraphInput {
        let mut keyframe_ids = self.order.clone();
        let mut seen: HashSet<KeyframeId> = keyframe_ids.iter().copied().collect();
        for edge in self.iter_all_edges() {
            if seen.insert(edge.a) {
                keyframe_ids.push(edge.a);
            }
            if seen.insert(edge.b) {
                keyframe_ids.push(edge.b);
            }
        }

        let mut id_to_idx = HashMap::new();
        for (idx, &id) in keyframe_ids.iter().enumerate() {
            id_to_idx.insert(id, idx);
        }

        let edges = self
            .iter_all_edges()
            .map(|edge| {
                let from = *id_to_idx
                    .get(&edge.a)
                    .expect("essential edge endpoint must be indexed");
                let to = *id_to_idx
                    .get(&edge.b)
                    .expect("essential edge endpoint must be indexed");
                PoseGraphEdge::try_new(from, to, edge.relative_pose, edge.information)
                    .expect("essential graph must contain validated edge information")
            })
            .collect();

        PoseGraphInput {
            keyframe_ids,
            edges,
        }
    }

    pub fn all_edges(&self) -> Vec<PoseGraphEdge> {
        self.pose_graph_input().edges
    }
}

fn contains_edge(edges: &[EssentialEdge], a: KeyframeId, b: KeyframeId) -> bool {
    edges.iter().any(|edge| same_endpoints(edge, a, b))
}

fn same_endpoints(edge: &EssentialEdge, a: KeyframeId, b: KeyframeId) -> bool {
    (edge.a == a && edge.b == b) || (edge.a == b && edge.b == a)
}

fn relative_pose(map: &SlamMap, from: KeyframeId, to: KeyframeId) -> Option<Pose64> {
    let from_pose = map.keyframe(from)?.pose();
    let to_pose = map.keyframe(to)?.pose();
    let from_64 = Pose64::from_pose32(from_pose.into_legacy_pose());
    let to_64 = Pose64::from_pose32(to_pose.into_legacy_pose());
    Some(to_64.compose(from_64.inverse()))
}
