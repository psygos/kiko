use crate::map::{
    CovisibilityGraph, CovisibilitySnapshot, KeyframeEntry, KeyframeId, KeyframeKeypoint, MapError,
    MapPoint, MapPointId, SlamMap,
};
#[cfg(feature = "vio")]
use crate::pose_graph::EssentialEdge;
use crate::pose_graph::{EssentialGraph, EssentialGraphError};
use crate::{Point3, Pose};
use std::num::NonZeroUsize;

#[allow(dead_code)]
pub(crate) struct GlobalMap {
    map: SlamMap,
    essential_graph: EssentialGraph,
}

impl GlobalMap {
    pub(crate) fn new(strong_threshold: u32) -> Self {
        Self {
            map: SlamMap::new(),
            essential_graph: EssentialGraph::new(strong_threshold),
        }
    }

    pub(crate) fn map(&self) -> &SlamMap {
        &self.map
    }

    pub(crate) fn map_mut(&mut self) -> &mut SlamMap {
        &mut self.map
    }

    #[allow(dead_code)]
    pub(crate) fn essential_graph(&self) -> &EssentialGraph {
        &self.essential_graph
    }

    #[allow(dead_code)]
    pub(crate) fn essential_graph_mut(&mut self) -> &mut EssentialGraph {
        &mut self.essential_graph
    }

    pub(crate) fn split_mut(&mut self) -> (&mut SlamMap, &mut EssentialGraph) {
        (&mut self.map, &mut self.essential_graph)
    }

    pub(crate) fn clone_map(&self) -> SlamMap {
        self.map.clone()
    }

    pub(crate) fn covisibility_snapshot(&self) -> CovisibilitySnapshot {
        self.map.covisibility_snapshot()
    }

    pub(crate) fn num_points(&self) -> usize {
        self.map.num_points()
    }

    pub(crate) fn num_keyframes(&self) -> usize {
        self.map.num_keyframes()
    }

    pub(crate) fn keyframe(&self, id: KeyframeId) -> Option<&KeyframeEntry> {
        self.map.keyframe(id)
    }

    pub(crate) fn keyframe_keypoint(
        &self,
        keyframe_id: KeyframeId,
        index: usize,
    ) -> Result<KeyframeKeypoint, MapError> {
        self.map.keyframe_keypoint(keyframe_id, index)
    }

    #[allow(dead_code)]
    pub(crate) fn map_point_for_keypoint(
        &self,
        keypoint: KeyframeKeypoint,
    ) -> Result<Option<MapPointId>, MapError> {
        self.map.map_point_for_keypoint(keypoint)
    }

    #[allow(dead_code)]
    pub(crate) fn point(&self, id: MapPointId) -> Option<&MapPoint> {
        self.map.point(id)
    }

    #[allow(dead_code)]
    pub(crate) fn points(&self) -> impl Iterator<Item = (MapPointId, &MapPoint)> {
        self.map.points()
    }

    #[allow(dead_code)]
    pub(crate) fn covisibility(&self) -> &CovisibilityGraph {
        self.map.covisibility()
    }

    pub(crate) fn covisible_window(
        &self,
        seed: KeyframeId,
        max: NonZeroUsize,
    ) -> Result<Vec<KeyframeId>, MapError> {
        self.map.covisible_window(seed, max)
    }

    #[allow(dead_code)]
    pub(crate) fn covisibility_ratio(&self, a: KeyframeId, b: KeyframeId) -> Result<f32, MapError> {
        self.map.covisibility_ratio(a, b)
    }

    #[allow(dead_code)]
    pub(crate) fn set_keyframe_pose(
        &mut self,
        keyframe_id: KeyframeId,
        pose: Pose,
    ) -> Result<(), MapError> {
        self.map.set_keyframe_pose(keyframe_id, pose)
    }

    #[allow(dead_code)]
    pub(crate) fn set_map_point_position(
        &mut self,
        point_id: MapPointId,
        position: Point3,
    ) -> Result<(), MapError> {
        self.map.set_map_point_position(point_id, position)
    }

    pub(crate) fn add_keyframe_to_graph(&mut self, keyframe_id: KeyframeId) {
        let neighbors = self.map.covisibility().neighbors(keyframe_id).cloned();
        self.essential_graph
            .add_keyframe(keyframe_id, neighbors.as_ref(), &self.map);
    }

    #[cfg(feature = "vio")]
    #[allow(dead_code)]
    pub(crate) fn add_odometry_edge(&mut self, edge: EssentialEdge) {
        self.essential_graph.add_odometry_edge(edge);
    }

    pub(crate) fn remove_keyframe_from_graph(
        &mut self,
        keyframe_id: KeyframeId,
    ) -> Result<(), EssentialGraphError> {
        self.essential_graph.remove_keyframe(keyframe_id, &self.map)
    }

    pub(crate) fn remove_keyframe(&mut self, keyframe_id: KeyframeId) -> Result<(), MapError> {
        self.map.remove_keyframe(keyframe_id)
    }

    #[cfg(test)]
    pub(crate) fn from_parts(map: SlamMap, essential_graph: EssentialGraph) -> Self {
        Self {
            map,
            essential_graph,
        }
    }
}
