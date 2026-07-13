use std::collections::HashMap;
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
    MissingInsertionFallback {
        registered_keyframes: usize,
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
            EssentialGraphError::MissingInsertionFallback {
                registered_keyframes,
            } => write!(
                f,
                "essential graph has {registered_keyframes} registered keyframes but no insertion-order fallback"
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
            Self::EdgeConstruction { source, .. } => Some(source),
            Self::KeyframeNotFound { .. }
            | Self::RootRemovalDenied { .. }
            | Self::MissingInsertionFallback { .. } => None,
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
                    .is_none_or(|(_, best_w): &(KeyframeId, u32)| weight.get() > *best_w)
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
        let spanning_relative_pose = parent_pose.inverse().compose(keyframe_pose);
        let spanning_edge = parse_essential_edge(
            parent,
            keyframe_id,
            EssentialEdgeKind::SpanningTree,
            spanning_relative_pose,
            scaled_identity6(parent_weight),
        )?;
        let mut new_strong_edges = Vec::new();
        if selected_by_covisibility && let Some(neighbors) = covisibility {
            let keyframe_pose_inverse = keyframe_pose.inverse();
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
                        keyframe_pose_inverse.compose(neighbor_pose),
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

        self.spanning_edges.extend(replacement_edges);

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
    Ok(from_64.inverse().compose(to_64))
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
