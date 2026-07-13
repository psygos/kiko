use std::collections::HashMap;
use std::num::NonZeroU32;

use crate::Pose64;
use crate::map::{KeyframeId, SlamMap};

use super::{PoseGraphEdge, PoseGraphError, scaled_identity6};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EssentialEdgeKind {
    SpanningTree,
    StrongCovisibility,
    Odometry,
    Loop,
}

#[derive(Clone, Debug)]
pub struct EssentialEdge {
    pub a: KeyframeId,
    pub b: KeyframeId,
    pub kind: EssentialEdgeKind,
    pub relative_pose: Pose64,
    pub information: [[f64; 6]; 6],
}

#[derive(Clone, Debug)]
pub struct EssentialGraphSnapshot {
    pub parent: HashMap<KeyframeId, KeyframeId>,
    pub order: Vec<KeyframeId>,
    pub spanning_edges: Vec<EssentialEdge>,
    pub strong_covis_edges: Vec<EssentialEdge>,
    pub odometry_edges: Vec<EssentialEdge>,
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
    odometry_edges: Vec<EssentialEdge>,
    loop_edges: Vec<EssentialEdge>,
    strong_threshold: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EssentialGraphError {
    KeyframeNotFound { keyframe_id: KeyframeId },
    RootRemovalDenied { keyframe_id: KeyframeId },
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
        }
    }
}

impl std::error::Error for EssentialGraphError {}

impl EssentialGraph {
    pub fn new(strong_threshold: u32) -> Self {
        Self {
            parent: HashMap::new(),
            order: Vec::new(),
            spanning_edges: Vec::new(),
            strong_covis_edges: Vec::new(),
            odometry_edges: Vec::new(),
            loop_edges: Vec::new(),
            strong_threshold,
        }
    }

    pub fn parent_of(&self, keyframe_id: KeyframeId) -> Option<KeyframeId> {
        self.parent.get(&keyframe_id).copied()
    }

    /// Add one keyframe in graph order. Covisibility entries for keyframes not
    /// yet registered in this graph are intentionally ineligible as parents
    /// or strong edges; this preserves an acyclic spanning tree even when the
    /// supplied map already contains later keyframes.
    pub fn add_keyframe(
        &mut self,
        keyframe_id: KeyframeId,
        covisibility: Option<&HashMap<KeyframeId, NonZeroU32>>,
        map: &SlamMap,
    ) {
        if self.parent.contains_key(&keyframe_id) {
            return;
        }
        let fallback_parent = self.order.last().copied();
        self.order.push(keyframe_id);
        if self.parent.is_empty() {
            self.parent.insert(keyframe_id, keyframe_id);
            return;
        }

        let Some(neighbors) = covisibility else {
            if let Some(parent) = fallback_parent {
                self.attach_parent(keyframe_id, parent, 1.0, map);
            }
            return;
        };
        if neighbors.is_empty() {
            if let Some(parent) = fallback_parent {
                self.attach_parent(keyframe_id, parent, 1.0, map);
            }
            return;
        }

        let mut strongest = None;
        for (&neighbor, &weight) in neighbors {
            if neighbor == keyframe_id || !self.parent.contains_key(&neighbor) {
                continue;
            }
            if strongest
                .as_ref()
                .is_none_or(|(_, best_w): &(KeyframeId, u32)| weight.get() > *best_w)
            {
                strongest = Some((neighbor, weight.get()));
            }
        }

        let Some((parent, parent_weight)) = strongest else {
            if let Some(parent) = fallback_parent {
                self.attach_parent(keyframe_id, parent, 1.0, map);
            }
            return;
        };

        for (&neighbor, &weight) in neighbors {
            if weight.get() >= self.strong_threshold
                && neighbor != parent
                && self.parent.contains_key(&neighbor)
                && !contains_edge(&self.strong_covis_edges, keyframe_id, neighbor)
                && let Some(relative_pose) = relative_pose(map, keyframe_id, neighbor)
            {
                self.strong_covis_edges.push(EssentialEdge {
                    a: keyframe_id,
                    b: neighbor,
                    kind: EssentialEdgeKind::StrongCovisibility,
                    relative_pose,
                    information: scaled_identity6(weight.get() as f64),
                });
            }
        }
        self.attach_parent(keyframe_id, parent, parent_weight as f64, map);
    }

    pub fn add_loop_edge(&mut self, edge: EssentialEdge) {
        let root = self.order.first().copied();
        if let std::collections::hash_map::Entry::Vacant(entry) = self.parent.entry(edge.a) {
            entry.insert(root.unwrap_or(edge.a));
            self.order.push(edge.a);
        }
        if let std::collections::hash_map::Entry::Vacant(entry) = self.parent.entry(edge.b) {
            entry.insert(root.unwrap_or(edge.a));
            self.order.push(edge.b);
        }
        self.loop_edges
            .retain(|existing| !same_endpoints(existing.a, existing.b, edge.a, edge.b));
        self.loop_edges.push(edge);
    }

    pub fn add_odometry_edge(&mut self, edge: EssentialEdge) {
        let root = self.order.first().copied();
        if let std::collections::hash_map::Entry::Vacant(entry) = self.parent.entry(edge.a) {
            entry.insert(root.unwrap_or(edge.a));
            self.order.push(edge.a);
        }
        if let std::collections::hash_map::Entry::Vacant(entry) = self.parent.entry(edge.b) {
            entry.insert(root.unwrap_or(edge.a));
            self.order.push(edge.b);
        }
        self.odometry_edges
            .retain(|existing| !same_endpoints(existing.a, existing.b, edge.a, edge.b));
        self.odometry_edges.push(edge);
    }

    fn attach_parent(
        &mut self,
        keyframe_id: KeyframeId,
        parent: KeyframeId,
        information_scale: f64,
        map: &SlamMap,
    ) {
        self.parent.insert(keyframe_id, parent);
        if let Some(relative_pose) = relative_pose(map, parent, keyframe_id) {
            self.spanning_edges.push(EssentialEdge {
                a: parent,
                b: keyframe_id,
                kind: EssentialEdgeKind::SpanningTree,
                relative_pose,
                information: scaled_identity6(information_scale),
            });
        }
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
        self.odometry_edges
            .retain(|edge| edge.a != keyframe_id && edge.b != keyframe_id);
        self.loop_edges
            .retain(|edge| edge.a != keyframe_id && edge.b != keyframe_id);

        for child in children {
            if contains_edge(&self.spanning_edges, parent, child) {
                continue;
            }
            if let Some(relative_pose) = relative_pose(map, parent, child) {
                self.spanning_edges.push(EssentialEdge {
                    a: parent,
                    b: child,
                    kind: EssentialEdgeKind::SpanningTree,
                    relative_pose,
                    information: scaled_identity6(1.0),
                });
            }
        }

        Ok(())
    }

    /// Iterate over all essential edges: spanning-tree, strong-covisibility,
    /// odometry, and loop constraints.
    fn iter_all_edges(&self) -> impl Iterator<Item = &EssentialEdge> {
        self.spanning_edges
            .iter()
            .chain(self.strong_covis_edges.iter())
            .chain(self.odometry_edges.iter())
            .chain(self.loop_edges.iter())
    }

    pub fn snapshot(&self) -> EssentialGraphSnapshot {
        EssentialGraphSnapshot {
            parent: self.parent.clone(),
            order: self.order.clone(),
            spanning_edges: self.spanning_edges.clone(),
            strong_covis_edges: self.strong_covis_edges.clone(),
            odometry_edges: self.odometry_edges.clone(),
            loop_edges: self.loop_edges.clone(),
            strong_threshold: self.strong_threshold,
        }
    }

    pub fn pose_graph_input(&self) -> Result<PoseGraphInput, PoseGraphError> {
        let mut keyframe_ids = self.order.clone();
        let mut id_to_idx = HashMap::with_capacity(keyframe_ids.len());
        for (idx, &id) in keyframe_ids.iter().enumerate() {
            id_to_idx.insert(id, idx);
        }

        let mut edges = Vec::with_capacity(self.iter_all_edges().size_hint().0);
        for (edge_index, edge) in self.iter_all_edges().enumerate() {
            let from = keyframe_index(&mut keyframe_ids, &mut id_to_idx, edge.a);
            let to = keyframe_index(&mut keyframe_ids, &mut id_to_idx, edge.b);
            edges.push(
                PoseGraphEdge::try_new(from, to, edge.relative_pose, edge.information)
                    .map_err(|source| PoseGraphError::EdgeConstruction { edge_index, source })?,
            );
        }

        Ok(PoseGraphInput {
            keyframe_ids,
            edges,
        })
    }

    pub fn all_edges(&self) -> Result<Vec<PoseGraphEdge>, PoseGraphError> {
        self.pose_graph_input().map(|input| input.edges)
    }
}

fn keyframe_index(
    keyframe_ids: &mut Vec<KeyframeId>,
    id_to_idx: &mut HashMap<KeyframeId, usize>,
    keyframe_id: KeyframeId,
) -> usize {
    *id_to_idx.entry(keyframe_id).or_insert_with(|| {
        let index = keyframe_ids.len();
        keyframe_ids.push(keyframe_id);
        index
    })
}

fn contains_edge(edges: &[EssentialEdge], a: KeyframeId, b: KeyframeId) -> bool {
    edges
        .iter()
        .any(|edge| (edge.a == a && edge.b == b) || (edge.a == b && edge.b == a))
}

fn same_endpoints(a0: KeyframeId, b0: KeyframeId, a1: KeyframeId, b1: KeyframeId) -> bool {
    (a0 == a1 && b0 == b1) || (a0 == b1 && b0 == a1)
}

fn relative_pose(map: &SlamMap, from: KeyframeId, to: KeyframeId) -> Option<Pose64> {
    let from_pose = map.keyframe(from)?.pose();
    let to_pose = map.keyframe(to)?.pose();
    let from_64 = Pose64::from_pose32(from_pose);
    let to_64 = Pose64::from_pose32(to_pose);
    Some(from_64.inverse().compose(to_64))
}
