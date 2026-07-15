use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;

use crate::map::{KeyframeId, SlamMap};
use crate::{Pose64, Pose64Error};

use super::{
    PoseGraphEdge, PoseGraphError, PoseGraphInformation, PoseGraphInformationError,
    scaled_identity6,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EssentialEdgeKind {
    SpanningTree,
    StrongCovisibility,
    Odometry,
    Loop,
}

#[derive(Clone, Debug)]
pub struct EssentialEdge {
    a: KeyframeId,
    b: KeyframeId,
    kind: EssentialEdgeKind,
    relative_pose: Pose64,
    information: PoseGraphInformation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EssentialEdgeError {
    SelfEdge { keyframe_id: KeyframeId },
    RelativePose { source: Pose64Error },
    Information { source: PoseGraphInformationError },
}

impl std::fmt::Display for EssentialEdgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SelfEdge { keyframe_id } => write!(
                f,
                "essential graph edge endpoints must differ, both were {keyframe_id:?}"
            ),
            Self::RelativePose { source } => {
                write!(f, "invalid essential graph relative pose: {source}")
            }
            Self::Information { source } => {
                write!(f, "invalid essential graph edge information: {source}")
            }
        }
    }
}

impl std::error::Error for EssentialEdgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::RelativePose { source } => Some(source),
            Self::Information { source } => Some(source),
            Self::SelfEdge { .. } => None,
        }
    }
}

impl EssentialEdge {
    /// Construct one validated constraint whose measurement uses the graph's
    /// exact algebraic convention `pose(a).inverse().compose(pose(b))`.
    fn try_new(
        a: KeyframeId,
        b: KeyframeId,
        kind: EssentialEdgeKind,
        relative_pose: Pose64,
        information: [[f64; 6]; 6],
    ) -> Result<Self, EssentialEdgeError> {
        if a == b {
            return Err(EssentialEdgeError::SelfEdge { keyframe_id: a });
        }
        let relative_pose =
            Pose64::try_from_rt(relative_pose.rotation(), relative_pose.translation())
                .map_err(|source| EssentialEdgeError::RelativePose { source })?;
        let information = PoseGraphInformation::try_new(information)
            .map_err(|source| EssentialEdgeError::Information { source })?;
        Ok(Self {
            a,
            b,
            kind,
            relative_pose,
            information,
        })
    }

    pub fn endpoint_a(&self) -> KeyframeId {
        self.a
    }

    pub fn endpoint_b(&self) -> KeyframeId {
        self.b
    }

    pub fn kind(&self) -> EssentialEdgeKind {
        self.kind
    }

    pub fn relative_pose(&self) -> Pose64 {
        self.relative_pose
    }

    pub fn information(&self) -> &PoseGraphInformation {
        &self.information
    }
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EssentialGraphError {
    KeyframeNotFound {
        keyframe_id: KeyframeId,
    },
    RootRemovalDenied {
        keyframe_id: KeyframeId,
    },
    InvalidPose {
        keyframe_id: KeyframeId,
        source: Pose64Error,
    },
    PoseComputation {
        operation: &'static str,
        source: Pose64Error,
    },
    MissingInsertionFallback {
        registered_keyframes: usize,
    },
    KeyframeNotRegistered {
        keyframe_id: KeyframeId,
    },
    DuplicateKeyframe {
        keyframe_id: KeyframeId,
    },
    DisconnectedKeyframe {
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
    EdgeConstruction {
        kind: EssentialEdgeKind,
        source: EssentialEdgeError,
    },
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
            EssentialGraphError::InvalidPose {
                keyframe_id,
                source,
            } => write!(
                f,
                "essential graph keyframe {keyframe_id:?} has an invalid pose: {source}"
            ),
            EssentialGraphError::PoseComputation { operation, source } => {
                write!(f, "essential graph {operation} failed: {source}")
            }
            EssentialGraphError::MissingInsertionFallback {
                registered_keyframes,
            } => write!(
                f,
                "essential graph has {registered_keyframes} registered keyframes but no insertion-order fallback"
            ),
            EssentialGraphError::KeyframeNotRegistered { keyframe_id } => write!(
                f,
                "keyframe is not registered in the essential graph: {keyframe_id:?}"
            ),
            EssentialGraphError::DuplicateKeyframe { keyframe_id } => write!(
                f,
                "essential graph order contains duplicate keyframe: {keyframe_id:?}"
            ),
            EssentialGraphError::DisconnectedKeyframe { keyframe_id } => write!(
                f,
                "keyframe has no connection to the essential graph root: {keyframe_id:?}"
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
            EssentialGraphError::EdgeConstruction { kind, source } => {
                write!(
                    f,
                    "failed to construct {kind:?} essential graph edge: {source}"
                )
            }
        }
    }
}

impl std::error::Error for EssentialGraphError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidPose { source, .. } => Some(source),
            Self::PoseComputation { source, .. } => Some(source),
            Self::EdgeConstruction { source, .. } => Some(source),
            Self::KeyframeNotFound { .. }
            | Self::RootRemovalDenied { .. }
            | Self::MissingInsertionFallback { .. }
            | Self::KeyframeNotRegistered { .. }
            | Self::DuplicateKeyframe { .. }
            | Self::DisconnectedKeyframe { .. }
            | Self::InvalidRootCount { .. }
            | Self::InvalidSpanningEdgeCount { .. }
            | Self::MissingSpanningEdge { .. }
            | Self::DuplicateEdge { .. }
            | Self::ConflictingEdgeKinds { .. }
            | Self::UnexpectedEdgeKind { .. } => None,
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
    ) -> Result<(), EssentialGraphError> {
        let keyframe_pose = map_pose64(map, keyframe_id)?;
        if self.parent.contains_key(&keyframe_id) {
            return Ok(());
        }
        let fallback_parent = self.order.last().copied();
        if self.parent.is_empty() {
            self.order.push(keyframe_id);
            self.parent.insert(keyframe_id, keyframe_id);
            return Ok(());
        }

        let mut strongest = None;
        if let Some(neighbors) = covisibility {
            for (&neighbor, &weight) in neighbors {
                if neighbor == keyframe_id || !self.parent.contains_key(&neighbor) {
                    continue;
                }
                if strongest
                    .as_ref()
                    .is_none_or(|(best_id, best_w): &(KeyframeId, u32)| {
                        weight.get() > *best_w || (weight.get() == *best_w && neighbor < *best_id)
                    })
                {
                    strongest = Some((neighbor, weight.get()));
                }
            }
        }

        let (parent, parent_weight, selected_by_covisibility) = match strongest {
            Some((parent, weight)) => (parent, f64::from(weight), true),
            None => {
                let parent =
                    fallback_parent.ok_or(EssentialGraphError::MissingInsertionFallback {
                        registered_keyframes: self.parent.len(),
                    })?;
                (parent, 1.0, false)
            }
        };

        let parent_pose = map_pose64(map, parent)?;
        let spanning_relative_pose = parent_pose
            .try_inverse()
            .and_then(|inverse| inverse.try_compose(keyframe_pose))
            .map_err(|source| EssentialGraphError::PoseComputation {
                operation: "spanning-edge relative-pose construction",
                source,
            })?;
        let spanning_edge = parse_essential_edge(
            parent,
            keyframe_id,
            EssentialEdgeKind::SpanningTree,
            spanning_relative_pose,
            scaled_identity6(parent_weight),
        )?;
        let mut new_strong_edges = Vec::new();
        if selected_by_covisibility && let Some(neighbors) = covisibility {
            let keyframe_pose_inverse = keyframe_pose.try_inverse().map_err(|source| {
                EssentialGraphError::PoseComputation {
                    operation: "strong-covisibility source-pose inversion",
                    source,
                }
            })?;
            for (&neighbor, &weight) in neighbors {
                if weight.get() >= self.strong_threshold
                    && neighbor != parent
                    && self.parent.contains_key(&neighbor)
                    && !contains_edge(&self.strong_covis_edges, keyframe_id, neighbor)
                {
                    let neighbor_pose = map_pose64(map, neighbor)?;
                    new_strong_edges.push(parse_essential_edge(
                        keyframe_id,
                        neighbor,
                        EssentialEdgeKind::StrongCovisibility,
                        keyframe_pose_inverse
                            .try_compose(neighbor_pose)
                            .map_err(|source| EssentialGraphError::PoseComputation {
                                operation: "strong-covisibility relative-pose construction",
                                source,
                            })?,
                        scaled_identity6(f64::from(weight.get())),
                    )?);
                }
            }
        }

        // All map lookups and pose conversions succeeded; commit together.
        self.order.push(keyframe_id);
        self.parent.insert(keyframe_id, parent);
        self.strong_covis_edges.extend(new_strong_edges);
        self.spanning_edges.push(spanning_edge);
        Ok(())
    }

    /// Register a keyframe using a geometrically verified parent rather than
    /// inferred covisibility. The edge is constructed before graph mutation,
    /// so an invalid pose or information matrix leaves this graph unchanged.
    pub(crate) fn add_keyframe_with_verified_parent(
        &mut self,
        keyframe_id: KeyframeId,
        parent: KeyframeId,
        information: [[f64; 6]; 6],
        map: &SlamMap,
    ) -> Result<(), EssentialGraphError> {
        let child_pose = map_pose64(map, keyframe_id)?;
        let parent_pose = map_pose64(map, parent)?;
        if !self.parent.contains_key(&parent) {
            return Err(EssentialGraphError::KeyframeNotRegistered {
                keyframe_id: parent,
            });
        }
        if self.parent.contains_key(&keyframe_id) {
            return Ok(());
        }
        if keyframe_id == parent {
            return Err(EssentialGraphError::EdgeConstruction {
                kind: EssentialEdgeKind::SpanningTree,
                source: EssentialEdgeError::SelfEdge { keyframe_id },
            });
        }
        let spanning_edge = parse_essential_edge(
            parent,
            keyframe_id,
            EssentialEdgeKind::SpanningTree,
            parent_pose
                .try_inverse()
                .and_then(|inverse| inverse.try_compose(child_pose))
                .map_err(|source| EssentialGraphError::PoseComputation {
                    operation: "verified-parent relative-pose construction",
                    source,
                })?,
            information,
        )?;
        self.order.push(keyframe_id);
        self.parent.insert(keyframe_id, parent);
        self.spanning_edges.push(spanning_edge);
        Ok(())
    }

    pub fn add_loop_edge(
        &mut self,
        a: KeyframeId,
        b: KeyframeId,
        relative_pose: Pose64,
        information: [[f64; 6]; 6],
    ) -> Result<(), EssentialGraphError> {
        let edge = parse_essential_edge(a, b, EssentialEdgeKind::Loop, relative_pose, information)?;
        self.validate_registered_endpoints(&edge)?;
        self.loop_edges
            .retain(|existing| !same_endpoints(existing.a, existing.b, edge.a, edge.b));
        self.loop_edges.push(edge);
        Ok(())
    }

    pub fn add_odometry_edge(
        &mut self,
        a: KeyframeId,
        b: KeyframeId,
        relative_pose: Pose64,
        information: [[f64; 6]; 6],
    ) -> Result<(), EssentialGraphError> {
        let edge = parse_essential_edge(
            a,
            b,
            EssentialEdgeKind::Odometry,
            relative_pose,
            information,
        )?;
        self.validate_registered_endpoints(&edge)?;
        self.odometry_edges
            .retain(|existing| !same_endpoints(existing.a, existing.b, edge.a, edge.b));
        self.odometry_edges.push(edge);
        Ok(())
    }

    fn validate_registered_endpoints(
        &self,
        edge: &EssentialEdge,
    ) -> Result<(), EssentialGraphError> {
        for keyframe_id in [edge.a, edge.b] {
            if !self.parent.contains_key(&keyframe_id) {
                return Err(EssentialGraphError::KeyframeNotFound { keyframe_id });
            }
        }
        Ok(())
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
            .order
            .iter()
            .copied()
            .filter(|&child| child != keyframe_id && self.parent.get(&child) == Some(&keyframe_id))
            .collect();

        let mut replacement_edges = Vec::with_capacity(children.len());
        for &child in &children {
            let retained_edge_exists = self.spanning_edges.iter().any(|edge| {
                edge.a != keyframe_id
                    && edge.b != keyframe_id
                    && same_endpoints(edge.a, edge.b, parent, child)
            });
            if !retained_edge_exists {
                replacement_edges.push(parse_essential_edge(
                    parent,
                    child,
                    EssentialEdgeKind::SpanningTree,
                    relative_pose(map, parent, child)?,
                    scaled_identity6(1.0),
                )?);
            }
        }

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

        for edge in replacement_edges {
            self.strong_covis_edges
                .retain(|strong| !same_endpoints(strong.a, strong.b, edge.a, edge.b));
            self.spanning_edges.push(edge);
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

    fn validate_topology(&self) -> Result<(), EssentialGraphError> {
        let mut registered = HashSet::with_capacity(self.order.len());
        for &keyframe_id in &self.order {
            if !registered.insert(keyframe_id) {
                return Err(EssentialGraphError::DuplicateKeyframe { keyframe_id });
            }
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
        if registered.is_empty() {
            if let Some(edge) = self.iter_all_edges().next() {
                return Err(EssentialGraphError::KeyframeNotRegistered {
                    keyframe_id: edge.a,
                });
            }
            return Ok(());
        }

        let roots = self
            .order
            .iter()
            .copied()
            .filter(|id| self.parent.get(id) == Some(id))
            .collect::<Vec<_>>();
        if roots.len() != 1 {
            return Err(EssentialGraphError::InvalidRootCount { count: roots.len() });
        }
        let root = roots[0];
        for &keyframe_id in &self.order {
            let parent = *self
                .parent
                .get(&keyframe_id)
                .ok_or(EssentialGraphError::KeyframeNotRegistered { keyframe_id })?;
            if !registered.contains(&parent) {
                return Err(EssentialGraphError::KeyframeNotRegistered {
                    keyframe_id: parent,
                });
            }
            let mut cursor = keyframe_id;
            let mut remaining = registered.len();
            while cursor != root && remaining > 0 {
                cursor = *self.parent.get(&cursor).ok_or(
                    EssentialGraphError::KeyframeNotRegistered {
                        keyframe_id: cursor,
                    },
                )?;
                remaining -= 1;
            }
            if cursor != root {
                return Err(EssentialGraphError::DisconnectedKeyframe { keyframe_id });
            }
        }

        let mut structural_pairs = HashMap::with_capacity(
            self.spanning_edges
                .len()
                .saturating_add(self.strong_covis_edges.len()),
        );
        validate_edge_set(
            &self.spanning_edges,
            EssentialEdgeKind::SpanningTree,
            &registered,
            &mut structural_pairs,
        )?;
        validate_edge_set(
            &self.strong_covis_edges,
            EssentialEdgeKind::StrongCovisibility,
            &registered,
            &mut structural_pairs,
        )?;
        let mut odometry_pairs = HashMap::with_capacity(self.odometry_edges.len());
        validate_edge_set(
            &self.odometry_edges,
            EssentialEdgeKind::Odometry,
            &registered,
            &mut odometry_pairs,
        )?;
        let mut loop_pairs = HashMap::with_capacity(self.loop_edges.len());
        validate_edge_set(
            &self.loop_edges,
            EssentialEdgeKind::Loop,
            &registered,
            &mut loop_pairs,
        )?;

        let expected = registered.len() - 1;
        if self.spanning_edges.len() != expected {
            return Err(EssentialGraphError::InvalidSpanningEdgeCount {
                expected,
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
        self.validate_topology()
            .map_err(|source| PoseGraphError::EssentialTopology { source })?;
        let keyframe_ids = self.order.clone();
        let mut id_to_idx = HashMap::with_capacity(keyframe_ids.len());
        for (idx, &id) in keyframe_ids.iter().enumerate() {
            id_to_idx.insert(id, idx);
        }

        let mut edges = Vec::with_capacity(self.iter_all_edges().size_hint().0);
        for (edge_index, edge) in self.iter_all_edges().enumerate() {
            let from = *id_to_idx.get(&edge.a).ok_or(
                PoseGraphError::UnregisteredEssentialEdgeEndpoint {
                    edge_index,
                    endpoint: "a",
                    keyframe_id: edge.a,
                },
            )?;
            let to = *id_to_idx.get(&edge.b).ok_or(
                PoseGraphError::UnregisteredEssentialEdgeEndpoint {
                    edge_index,
                    endpoint: "b",
                    keyframe_id: edge.b,
                },
            )?;
            edges.push(
                PoseGraphEdge::try_from_validated_information(
                    from,
                    to,
                    edge.relative_pose,
                    edge.information,
                )
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

    #[cfg(test)]
    pub(crate) fn reverse_spanning_edge_for_test(&mut self, child: KeyframeId) -> bool {
        let Some(edge) = self.spanning_edges.iter_mut().find(|edge| edge.b == child) else {
            return false;
        };
        std::mem::swap(&mut edge.a, &mut edge.b);
        true
    }
}

fn validate_edge_set(
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
        if let Some(first) = seen.insert(endpoints, expected_kind) {
            return Err(if first == expected_kind {
                EssentialGraphError::DuplicateEdge {
                    a: endpoints.0,
                    b: endpoints.1,
                    kind: expected_kind,
                }
            } else {
                EssentialGraphError::ConflictingEdgeKinds {
                    a: endpoints.0,
                    b: endpoints.1,
                    first,
                    second: expected_kind,
                }
            });
        }
    }
    Ok(())
}

fn parse_essential_edge(
    a: KeyframeId,
    b: KeyframeId,
    kind: EssentialEdgeKind,
    relative_pose: Pose64,
    information: [[f64; 6]; 6],
) -> Result<EssentialEdge, EssentialGraphError> {
    EssentialEdge::try_new(a, b, kind, relative_pose, information)
        .map_err(|source| EssentialGraphError::EdgeConstruction { kind, source })
}

fn contains_edge(edges: &[EssentialEdge], a: KeyframeId, b: KeyframeId) -> bool {
    edges
        .iter()
        .any(|edge| (edge.a == a && edge.b == b) || (edge.a == b && edge.b == a))
}

fn contains_directed_edge(edges: &[EssentialEdge], from: KeyframeId, to: KeyframeId) -> bool {
    edges.iter().any(|edge| edge.a == from && edge.b == to)
}

fn canonical_endpoints(a: KeyframeId, b: KeyframeId) -> (KeyframeId, KeyframeId) {
    (a.min(b), a.max(b))
}

fn same_endpoints(a0: KeyframeId, b0: KeyframeId, a1: KeyframeId, b1: KeyframeId) -> bool {
    (a0 == a1 && b0 == b1) || (a0 == b1 && b0 == a1)
}

fn relative_pose(
    map: &SlamMap,
    from: KeyframeId,
    to: KeyframeId,
) -> Result<Pose64, EssentialGraphError> {
    let from_64 = map_pose64(map, from)?;
    let to_64 = map_pose64(map, to)?;
    from_64
        .try_inverse()
        .and_then(|inverse| inverse.try_compose(to_64))
        .map_err(|source| EssentialGraphError::PoseComputation {
            operation: "map relative-pose construction",
            source,
        })
}

fn map_pose64(map: &SlamMap, keyframe_id: KeyframeId) -> Result<Pose64, EssentialGraphError> {
    let pose = map
        .keyframe(keyframe_id)
        .ok_or(EssentialGraphError::KeyframeNotFound { keyframe_id })?
        .pose();
    Pose64::try_from_pose32(pose).map_err(|source| EssentialGraphError::InvalidPose {
        keyframe_id,
        source,
    })
}
