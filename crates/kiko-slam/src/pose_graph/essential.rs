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
    KeyframeNotFound {
        keyframe_id: KeyframeId,
    },
    KeyframeNotRegistered {
        keyframe_id: KeyframeId,
    },
    DisconnectedKeyframe {
        keyframe_id: KeyframeId,
    },
    DuplicateKeyframe {
        keyframe_id: KeyframeId,
    },
    InvalidRootCount {
        count: usize,
    },
    InvalidSpanningEdgeCount {
        expected: usize,
        actual: usize,
    },
    MissingSpanningEdge {
        child: KeyframeId,
        parent: KeyframeId,
    },
    DuplicateEdge {
        a: KeyframeId,
        b: KeyframeId,
        kind: EssentialEdgeKind,
    },
    ConflictingEdgeKinds {
        a: KeyframeId,
        b: KeyframeId,
        first: EssentialEdgeKind,
        second: EssentialEdgeKind,
    },
    UnexpectedEdgeKind {
        expected: EssentialEdgeKind,
        actual: EssentialEdgeKind,
    },
    RootRemovalDenied {
        keyframe_id: KeyframeId,
    },
    SelfEdge {
        keyframe_id: KeyframeId,
    },
    InvalidEdge(PoseGraphEdgeError),
}

impl std::fmt::Display for EssentialGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EssentialGraphError::KeyframeNotFound { keyframe_id } => {
                write!(f, "essential graph keyframe not found: {keyframe_id:?}")
            }
            EssentialGraphError::KeyframeNotRegistered { keyframe_id } => write!(
                f,
                "keyframe is not registered in the essential graph: {keyframe_id:?}"
            ),
            EssentialGraphError::DisconnectedKeyframe { keyframe_id } => write!(
                f,
                "keyframe has no connection to the essential graph root: {keyframe_id:?}"
            ),
            EssentialGraphError::DuplicateKeyframe { keyframe_id } => write!(
                f,
                "essential graph order contains duplicate keyframe: {keyframe_id:?}"
            ),
            EssentialGraphError::InvalidRootCount { count } => write!(
                f,
                "non-empty essential graph must have exactly one root, found {count}"
            ),
            EssentialGraphError::InvalidSpanningEdgeCount { expected, actual } => write!(
                f,
                "essential graph requires {expected} spanning edges, found {actual}"
            ),
            EssentialGraphError::MissingSpanningEdge { child, parent } => write!(
                f,
                "essential graph is missing the spanning edge from {parent:?} to {child:?}"
            ),
            EssentialGraphError::DuplicateEdge { a, b, kind } => write!(
                f,
                "essential graph contains duplicate {kind:?} edge between {a:?} and {b:?}"
            ),
            EssentialGraphError::ConflictingEdgeKinds {
                a,
                b,
                first,
                second,
            } => write!(
                f,
                "essential graph pair {a:?} to {b:?} is both {first:?} and {second:?}"
            ),
            EssentialGraphError::UnexpectedEdgeKind { expected, actual } => write!(
                f,
                "essential graph expected a {expected:?} edge, got {actual:?}"
            ),
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
            | Self::KeyframeNotRegistered { .. }
            | Self::DisconnectedKeyframe { .. }
            | Self::DuplicateKeyframe { .. }
            | Self::InvalidRootCount { .. }
            | Self::InvalidSpanningEdgeCount { .. }
            | Self::MissingSpanningEdge { .. }
            | Self::DuplicateEdge { .. }
            | Self::ConflictingEdgeKinds { .. }
            | Self::UnexpectedEdgeKind { .. }
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
    ) -> Result<(), EssentialGraphError> {
        self.validate_registered_keyframes(map)?;
        require_map_keyframe(map, keyframe_id)?;
        if self.parent.contains_key(&keyframe_id) {
            return Ok(());
        }
        if self.parent.is_empty() {
            self.order.push(keyframe_id);
            self.parent.insert(keyframe_id, keyframe_id);
            return Ok(());
        }

        let neighbors =
            covisibility.ok_or(EssentialGraphError::DisconnectedKeyframe { keyframe_id })?;

        let mut strongest = None;
        for (&neighbor, &weight) in neighbors {
            if neighbor == keyframe_id {
                return Err(EssentialGraphError::SelfEdge { keyframe_id });
            }
            require_map_keyframe(map, neighbor)?;
            let neighbor_is_registered = self.parent.contains_key(&neighbor);
            if neighbor_is_registered
                && strongest
                    .as_ref()
                    .is_none_or(|(best_id, best_weight): &(KeyframeId, u32)| {
                        weight.get() > *best_weight
                            || (weight.get() == *best_weight && neighbor < *best_id)
                    })
            {
                strongest = Some((neighbor, weight.get()));
            }
        }

        let (parent, weight) =
            strongest.ok_or(EssentialGraphError::DisconnectedKeyframe { keyframe_id })?;
        let spanning_edge = EssentialEdge::try_new(
            parent,
            keyframe_id,
            EssentialEdgeKind::SpanningTree,
            relative_pose(map, parent, keyframe_id)?,
            scaled_identity6(weight as f64),
        )?;
        let mut strong_edges = Vec::new();
        for (&neighbor, &weight) in neighbors {
            if neighbor != parent
                && self.parent.contains_key(&neighbor)
                && weight.get() >= self.strong_threshold
                && !contains_edge(&self.strong_covis_edges, keyframe_id, neighbor)
            {
                strong_edges.push(EssentialEdge::try_new(
                    keyframe_id,
                    neighbor,
                    EssentialEdgeKind::StrongCovisibility,
                    relative_pose(map, keyframe_id, neighbor)?,
                    scaled_identity6(weight.get() as f64),
                )?);
            }
        }

        self.order.push(keyframe_id);
        self.parent.insert(keyframe_id, parent);
        self.spanning_edges.push(spanning_edge);
        self.strong_covis_edges.extend(strong_edges);
        Ok(())
    }

    /// Register a keyframe whose connection was established outside the local
    /// covisibility graph, such as a geometrically verified relocalization.
    ///
    /// The caller supplies the information matrix for that verification. The
    /// spanning measurement itself is always rebuilt from the two map poses so
    /// its direction remains parent-camera to child-camera.
    pub(crate) fn add_keyframe_with_verified_parent(
        &mut self,
        keyframe_id: KeyframeId,
        parent: KeyframeId,
        information: [[f64; 6]; 6],
        map: &SlamMap,
    ) -> Result<(), EssentialGraphError> {
        self.validate_registered_keyframes(map)?;
        require_map_keyframe(map, keyframe_id)?;
        require_map_keyframe(map, parent)?;
        if !self.parent.contains_key(&parent) {
            return Err(EssentialGraphError::KeyframeNotRegistered {
                keyframe_id: parent,
            });
        }
        if self.parent.contains_key(&keyframe_id) {
            return Ok(());
        }
        if keyframe_id == parent {
            return Err(EssentialGraphError::SelfEdge { keyframe_id });
        }

        let spanning_edge = EssentialEdge::try_new(
            parent,
            keyframe_id,
            EssentialEdgeKind::SpanningTree,
            relative_pose(map, parent, keyframe_id)?,
            information,
        )?;
        self.order.push(keyframe_id);
        self.parent.insert(keyframe_id, parent);
        self.spanning_edges.push(spanning_edge);
        Ok(())
    }

    pub fn add_loop_edge(
        &mut self,
        edge: EssentialEdge,
        map: &SlamMap,
    ) -> Result<(), EssentialGraphError> {
        self.validate_registered_keyframes(map)?;
        if edge.kind != EssentialEdgeKind::Loop {
            return Err(EssentialGraphError::UnexpectedEdgeKind {
                expected: EssentialEdgeKind::Loop,
                actual: edge.kind,
            });
        }
        for keyframe_id in [edge.a, edge.b] {
            require_map_keyframe(map, keyframe_id)?;
            if !self.parent.contains_key(&keyframe_id) {
                return Err(EssentialGraphError::KeyframeNotRegistered { keyframe_id });
            }
        }
        if contains_edge(&self.loop_edges, edge.a, edge.b) {
            return Err(EssentialGraphError::DuplicateEdge {
                a: edge.a.min(edge.b),
                b: edge.a.max(edge.b),
                kind: EssentialEdgeKind::Loop,
            });
        }
        self.loop_edges.push(edge);
        Ok(())
    }

    pub fn remove_keyframe(
        &mut self,
        keyframe_id: KeyframeId,
        map: &SlamMap,
    ) -> Result<(), EssentialGraphError> {
        self.validate_registered_keyframes(map)?;
        require_map_keyframe(map, keyframe_id)?;
        let parent = self
            .parent
            .get(&keyframe_id)
            .copied()
            .ok_or(EssentialGraphError::KeyframeNotFound { keyframe_id })?;
        if parent == keyframe_id {
            return Err(EssentialGraphError::RootRemovalDenied { keyframe_id });
        }

        let children: Vec<KeyframeId> = self
            .order
            .iter()
            .copied()
            .filter(|&child| child != keyframe_id && self.parent.get(&child) == Some(&keyframe_id))
            .collect();
        let reparent_edges = children
            .iter()
            .copied()
            .map(|child| {
                EssentialEdge::try_new(
                    parent,
                    child,
                    EssentialEdgeKind::SpanningTree,
                    relative_pose(map, parent, child)?,
                    scaled_identity6(1.0),
                )
                .map(|edge| (child, edge))
            })
            .collect::<Result<Vec<_>, _>>()?;

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

        for (child, edge) in reparent_edges {
            self.strong_covis_edges
                .retain(|edge| !same_endpoints(edge, parent, child));
            self.spanning_edges.push(edge);
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

    /// Graph fields are private and every mutator commits only after its new
    /// edges have been constructed successfully. A caller can still mutate the
    /// separately owned map, so mutation preflight must reject stale registered
    /// IDs without re-walking every structural edge on the hot path.
    fn validate_registered_keyframes(&self, map: &SlamMap) -> Result<(), EssentialGraphError> {
        for &keyframe_id in &self.order {
            require_map_keyframe(map, keyframe_id)?;
        }
        Ok(())
    }

    fn validate_topology(&self, map: &SlamMap) -> Result<(), EssentialGraphError> {
        let mut registered = HashSet::with_capacity(self.order.len());
        for &keyframe_id in &self.order {
            if !registered.insert(keyframe_id) {
                return Err(EssentialGraphError::DuplicateKeyframe { keyframe_id });
            }
            require_map_keyframe(map, keyframe_id)?;
            if !self.parent.contains_key(&keyframe_id) {
                return Err(EssentialGraphError::KeyframeNotRegistered { keyframe_id });
            }
        }
        if let Some(keyframe_id) = self
            .parent
            .keys()
            .copied()
            .filter(|keyframe_id| !registered.contains(keyframe_id))
            .min()
        {
            return Err(EssentialGraphError::KeyframeNotRegistered { keyframe_id });
        }
        for &keyframe_id in &self.order {
            let parent_id = *self
                .parent
                .get(&keyframe_id)
                .ok_or(EssentialGraphError::KeyframeNotRegistered { keyframe_id })?;
            if !registered.contains(&parent_id) {
                return Err(EssentialGraphError::KeyframeNotRegistered {
                    keyframe_id: parent_id,
                });
            }
        }

        if registered.is_empty() {
            if let Some(edge) = self.iter_all_edges().next() {
                return Err(EssentialGraphError::KeyframeNotRegistered {
                    keyframe_id: edge.a,
                });
            }
            return Ok(());
        }

        let mut root = None;
        let mut root_count = 0;
        for &child in &self.order {
            let parent = self.parent[&child];
            if child == parent {
                root = Some(child);
                root_count += 1;
            }
        }
        if root_count != 1 {
            return Err(EssentialGraphError::InvalidRootCount { count: root_count });
        }
        let root = root.ok_or(EssentialGraphError::InvalidRootCount { count: root_count })?;
        let mut connected = HashSet::with_capacity(registered.len());
        connected.insert(root);
        let mut path = Vec::with_capacity(registered.len());
        for &start in &self.order {
            let mut current = start;
            path.clear();
            while !connected.contains(&current) && path.len() < registered.len() {
                path.push(current);
                current = *self.parent.get(&current).ok_or(
                    EssentialGraphError::KeyframeNotRegistered {
                        keyframe_id: current,
                    },
                )?;
            }
            if !connected.contains(&current) {
                return Err(EssentialGraphError::DisconnectedKeyframe { keyframe_id: start });
            }
            connected.extend(path.iter().copied());
        }

        let mut structural_pairs =
            HashMap::with_capacity(self.spanning_edges.len() + self.strong_covis_edges.len());
        Self::validate_edges(
            &self.spanning_edges,
            EssentialEdgeKind::SpanningTree,
            &registered,
            &mut structural_pairs,
        )?;
        Self::validate_edges(
            &self.strong_covis_edges,
            EssentialEdgeKind::StrongCovisibility,
            &registered,
            &mut structural_pairs,
        )?;
        let mut loop_pairs = HashMap::with_capacity(self.loop_edges.len());
        Self::validate_edges(
            &self.loop_edges,
            EssentialEdgeKind::Loop,
            &registered,
            &mut loop_pairs,
        )?;
        let expected_spanning_edges = registered.len() - 1;
        if self.spanning_edges.len() != expected_spanning_edges {
            return Err(EssentialGraphError::InvalidSpanningEdgeCount {
                expected: expected_spanning_edges,
                actual: self.spanning_edges.len(),
            });
        }
        for &child in &self.order {
            let parent = self.parent[&child];
            if child != parent && !contains_directed_edge(&self.spanning_edges, parent, child) {
                return Err(EssentialGraphError::MissingSpanningEdge { child, parent });
            }
        }
        Ok(())
    }

    fn validate_edges(
        edges: &[EssentialEdge],
        expected_kind: EssentialEdgeKind,
        registered: &HashSet<KeyframeId>,
        seen: &mut HashMap<(KeyframeId, KeyframeId), EssentialEdgeKind>,
    ) -> Result<(), EssentialGraphError> {
        for edge in edges {
            if edge.kind != expected_kind {
                return Err(EssentialGraphError::UnexpectedEdgeKind {
                    expected: expected_kind,
                    actual: edge.kind,
                });
            }
            for keyframe_id in [edge.a, edge.b] {
                if !registered.contains(&keyframe_id) {
                    return Err(EssentialGraphError::KeyframeNotRegistered { keyframe_id });
                }
            }
            let endpoints = canonical_endpoints(edge.a, edge.b);
            if let Some(previous_kind) = seen.insert(endpoints, expected_kind) {
                return Err(if previous_kind == expected_kind {
                    EssentialGraphError::DuplicateEdge {
                        a: endpoints.0,
                        b: endpoints.1,
                        kind: expected_kind,
                    }
                } else {
                    EssentialGraphError::ConflictingEdgeKinds {
                        a: endpoints.0,
                        b: endpoints.1,
                        first: previous_kind,
                        second: expected_kind,
                    }
                });
            }
        }
        Ok(())
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

    pub fn pose_graph_input(&self, map: &SlamMap) -> Result<PoseGraphInput, EssentialGraphError> {
        self.validate_topology(map)?;
        let keyframe_ids = self.order.clone();
        let mut id_to_idx = HashMap::with_capacity(keyframe_ids.len());
        for (idx, &id) in keyframe_ids.iter().enumerate() {
            id_to_idx.insert(id, idx);
        }

        let edges =
            self.iter_all_edges()
                .map(|edge| {
                    let from = *id_to_idx.get(&edge.a).ok_or(
                        EssentialGraphError::KeyframeNotRegistered {
                            keyframe_id: edge.a,
                        },
                    )?;
                    let to = *id_to_idx.get(&edge.b).ok_or(
                        EssentialGraphError::KeyframeNotRegistered {
                            keyframe_id: edge.b,
                        },
                    )?;
                    PoseGraphEdge::try_new(from, to, edge.relative_pose, edge.information)
                        .map_err(EssentialGraphError::InvalidEdge)
                })
                .collect::<Result<Vec<_>, _>>()?;

        Ok(PoseGraphInput {
            keyframe_ids,
            edges,
        })
    }

    pub fn all_edges(&self, map: &SlamMap) -> Result<Vec<PoseGraphEdge>, EssentialGraphError> {
        Ok(self.pose_graph_input(map)?.edges)
    }

    #[cfg(test)]
    pub(crate) fn reverse_spanning_edge_for_test(&mut self, child: KeyframeId) -> bool {
        let Some(edge) = self.spanning_edges.iter_mut().find(|edge| edge.b == child) else {
            return false;
        };
        std::mem::swap(&mut edge.a, &mut edge.b);
        true
    }
}

fn contains_edge(edges: &[EssentialEdge], a: KeyframeId, b: KeyframeId) -> bool {
    edges.iter().any(|edge| same_endpoints(edge, a, b))
}

fn contains_directed_edge(edges: &[EssentialEdge], from: KeyframeId, to: KeyframeId) -> bool {
    edges.iter().any(|edge| edge.a == from && edge.b == to)
}

fn same_endpoints(edge: &EssentialEdge, a: KeyframeId, b: KeyframeId) -> bool {
    (edge.a == a && edge.b == b) || (edge.a == b && edge.b == a)
}

fn canonical_endpoints(a: KeyframeId, b: KeyframeId) -> (KeyframeId, KeyframeId) {
    (a.min(b), a.max(b))
}

fn require_map_keyframe(map: &SlamMap, keyframe_id: KeyframeId) -> Result<(), EssentialGraphError> {
    map.keyframe(keyframe_id)
        .map(|_| ())
        .ok_or(EssentialGraphError::KeyframeNotFound { keyframe_id })
}

fn relative_pose(
    map: &SlamMap,
    from: KeyframeId,
    to: KeyframeId,
) -> Result<Pose64, EssentialGraphError> {
    let from_pose = map
        .keyframe(from)
        .ok_or(EssentialGraphError::KeyframeNotFound { keyframe_id: from })?
        .pose();
    let to_pose = map
        .keyframe(to)
        .ok_or(EssentialGraphError::KeyframeNotFound { keyframe_id: to })?
        .pose();
    let from_64 = Pose64::from_pose32(from_pose.into_legacy_pose());
    let to_64 = Pose64::from_pose32(to_pose.into_legacy_pose());
    Ok(to_64.compose(from_64.inverse()))
}
