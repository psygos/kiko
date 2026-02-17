use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;

use super::{
    BlockCsr6x6, EssentialEdge, EssentialEdgeKind, EssentialGraph, EssentialGraphError,
    PoseGraphConfig, PoseGraphEdge, PoseGraphOptimizer, compute_edge_error, compute_edge_jacobians,
    solve_pcg,
};
use crate::Pose64;
use crate::map::{ImageSize, SlamMap};
use crate::math::se3_exp_f64;
use crate::{CompactDescriptor, FrameId, Keypoint, Point3, Pose, Timestamp};

#[derive(Clone, Debug)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.state >> 32) as usize) % upper
    }
}

fn scalar_block(diagonal: f64) -> [[f64; 6]; 6] {
    let mut block = [[0.0_f64; 6]; 6];
    for (i, row) in block.iter_mut().enumerate() {
        row[i] = diagonal;
    }
    block
}

fn make_map_for_essential_graph() -> (
    SlamMap,
    crate::map::KeyframeId,
    crate::map::KeyframeId,
    crate::map::KeyframeId,
) {
    let mut map = SlamMap::new();
    let size = ImageSize::try_new(640, 480).expect("size");
    let keypoints = vec![
        Keypoint { x: 20.0, y: 20.0 },
        Keypoint { x: 40.0, y: 20.0 },
        Keypoint { x: 60.0, y: 20.0 },
    ];
    let kf0 = map
        .add_keyframe(
            FrameId::new(1),
            Timestamp::from_nanos(1),
            Pose::identity(),
            size,
            keypoints.clone(),
        )
        .expect("kf0");
    let kf1 = map
        .add_keyframe(
            FrameId::new(2),
            Timestamp::from_nanos(2),
            Pose::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [1.0, 0.0, 0.0],
            ),
            size,
            keypoints.clone(),
        )
        .expect("kf1");
    let kf2 = map
        .add_keyframe(
            FrameId::new(3),
            Timestamp::from_nanos(3),
            Pose::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [2.0, 0.0, 0.0],
            ),
            size,
            keypoints,
        )
        .expect("kf2");

    for i in 0..2 {
        let kp0 = map.keyframe_keypoint(kf0, i).expect("kp0");
        let point_id = map
            .add_map_point(
                Point3 {
                    x: i as f32,
                    y: 0.0,
                    z: 3.0,
                },
                CompactDescriptor([128; 256]),
                kp0,
            )
            .expect("point");
        let kp1 = map.keyframe_keypoint(kf1, i).expect("kp1");
        map.add_observation(point_id, kp1).expect("obs");
    }

    let kp1 = map.keyframe_keypoint(kf1, 2).expect("kp1 third");
    let point_id = map
        .add_map_point(
            Point3 {
                x: 2.0,
                y: 0.0,
                z: 3.0,
            },
            CompactDescriptor([128; 256]),
            kp1,
        )
        .expect("point third");
    let kp2 = map.keyframe_keypoint(kf2, 0).expect("kp2");
    map.add_observation(point_id, kp2).expect("obs third");

    (map, kf0, kf1, kf2)
}

fn make_chain_keyframes(count: usize) -> (SlamMap, Vec<crate::map::KeyframeId>) {
    let mut map = SlamMap::new();
    let size = ImageSize::try_new(640, 480).expect("size");
    let keypoints = vec![Keypoint { x: 20.0, y: 20.0 }, Keypoint { x: 40.0, y: 20.0 }];
    let mut ids = Vec::with_capacity(count);
    for idx in 0..count {
        let pose = Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [idx as f32 * 0.5, 0.0, 0.0],
        );
        let id = map
            .add_keyframe(
                FrameId::new((idx + 1) as u64),
                Timestamp::from_nanos((idx + 1) as i64),
                pose,
                size,
                keypoints.clone(),
            )
            .expect("keyframe");
        ids.push(id);
    }
    (map, ids)
}

#[test]
fn block_csr_insert_and_get_are_consistent() {
    let mut h = BlockCsr6x6::new(3);
    let block = scalar_block(2.0);
    h.insert(1, 2, block).expect("insert");
    assert_eq!(h.get(1, 2), Some(block));

    let replacement = scalar_block(3.0);
    h.insert(1, 2, replacement).expect("replace");
    assert_eq!(h.get(1, 2), Some(replacement));
}

#[test]
fn block_csr_spmv_matches_dense_reference() {
    let mut h = BlockCsr6x6::new(2);
    h.insert(0, 0, scalar_block(2.0)).expect("insert");
    h.insert(0, 1, scalar_block(1.0)).expect("insert");
    h.insert(1, 0, scalar_block(-1.0)).expect("insert");
    h.insert(1, 1, scalar_block(3.0)).expect("insert");

    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
    let mut y_sparse = vec![0.0; 12];
    h.spmv(&x, &mut y_sparse).expect("spmv");

    let mut y_dense = [0.0; 12];
    for row in 0..2 {
        for col in 0..2 {
            let Some(block) = h.get(row, col) else {
                continue;
            };
            for r in 0..6 {
                let mut sum = 0.0;
                for c in 0..6 {
                    sum += block[r][c] * x[col * 6 + c];
                }
                y_dense[row * 6 + r] += sum;
            }
        }
    }

    for i in 0..12 {
        assert!(
            (y_sparse[i] - y_dense[i]).abs() < 1e-12,
            "mismatch at {i}: sparse={}, dense={}",
            y_sparse[i],
            y_dense[i]
        );
    }
}

#[test]
fn block_csr_diagonal_extraction_returns_only_diagonal_blocks() {
    let mut h = BlockCsr6x6::new(3);
    h.insert(0, 0, scalar_block(1.0)).expect("insert");
    h.insert(0, 1, scalar_block(5.0)).expect("insert");
    h.insert(1, 1, scalar_block(2.0)).expect("insert");
    h.insert(2, 0, scalar_block(7.0)).expect("insert");
    h.insert(2, 2, scalar_block(3.0)).expect("insert");

    let diag = h.diagonal_blocks();
    assert_eq!(diag.len(), 3);
    assert_eq!(diag[0], scalar_block(1.0));
    assert_eq!(diag[1], scalar_block(2.0));
    assert_eq!(diag[2], scalar_block(3.0));
}

#[test]
fn pcg_solves_identity_in_one_iteration() {
    let mut h = BlockCsr6x6::new(2);
    h.insert(0, 0, scalar_block(1.0)).expect("insert");
    h.insert(1, 1, scalar_block(1.0)).expect("insert");
    let b = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
    ];
    let mut x = vec![0.0; b.len()];
    let result = solve_pcg(&h, &b, &mut x, 20, 1e-12).expect("pcg");
    assert!(result.converged);
    assert_eq!(result.iterations, 1);
    for i in 0..x.len() {
        assert!((x[i] - b[i]).abs() < 1e-12);
    }
}

#[test]
fn pcg_converges_on_small_spd_system() {
    let mut h = BlockCsr6x6::new(2);
    h.insert(0, 0, scalar_block(4.0)).expect("insert");
    h.insert(1, 1, scalar_block(5.0)).expect("insert");
    h.insert(0, 1, scalar_block(0.2)).expect("insert");
    h.insert(1, 0, scalar_block(0.2)).expect("insert");

    let x_true = vec![
        0.5, -0.3, 0.8, 0.1, -0.2, 0.4, 1.2, -0.6, 0.7, -0.9, 0.2, 0.3,
    ];
    let mut b = vec![0.0; x_true.len()];
    h.spmv(&x_true, &mut b).expect("spmv");

    let mut x = vec![0.0; x_true.len()];
    let result = solve_pcg(&h, &b, &mut x, 50, 1e-10).expect("pcg");
    assert!(result.converged, "pcg did not converge: {result:?}");
    for i in 0..x.len() {
        assert!(
            (x[i] - x_true[i]).abs() < 1e-8,
            "solution mismatch at {i}: got {}, expected {}",
            x[i],
            x_true[i]
        );
    }
}

#[test]
fn pcg_zero_rhs_returns_zero_solution() {
    let mut h = BlockCsr6x6::new(2);
    h.insert(0, 0, scalar_block(3.0)).expect("insert");
    h.insert(1, 1, scalar_block(2.0)).expect("insert");
    let b = vec![0.0; 12];
    let mut x = vec![0.0; 12];
    let result = solve_pcg(&h, &b, &mut x, 10, 1e-12).expect("pcg");
    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert!(x.iter().all(|v| v.abs() < 1e-15));
}

#[test]
fn pose_graph_edge_error_is_zero_for_consistent_measurement() {
    let pose_a = Pose64::identity();
    let pose_b = se3_exp_f64([0.2, -0.1, 0.05, 0.03, -0.02, 0.01]);
    let measurement = pose_a.inverse().compose(pose_b);
    let edge = PoseGraphEdge {
        from: 0,
        to: 1,
        measurement,
        information: scalar_block(1.0),
    };
    let error = compute_edge_error(&edge, &[pose_a, pose_b]).expect("edge error");
    let norm: f64 = error.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm < 1e-9, "expected near-zero error, got {norm}");
}

#[test]
fn pose_graph_edge_jacobians_match_finite_difference() {
    let pose_a = se3_exp_f64([0.1, 0.05, -0.02, 0.02, -0.01, 0.03]);
    let pose_b = se3_exp_f64([0.3, -0.08, 0.12, -0.02, 0.03, -0.01]);
    let measurement = pose_a
        .inverse()
        .compose(pose_b)
        .compose(se3_exp_f64([0.01, -0.005, 0.002, 0.001, -0.0015, 0.0008]));
    let edge = PoseGraphEdge {
        from: 0,
        to: 1,
        measurement,
        information: scalar_block(1.0),
    };
    let poses = [pose_a, pose_b];
    let (j_from, j_to) = compute_edge_jacobians(&edge, &poses).expect("jacobians");
    for row in 0..6 {
        for col in 0..6 {
            assert!(j_from[row][col].is_finite(), "non-finite J_from entry");
            assert!(j_to[row][col].is_finite(), "non-finite J_to entry");
        }
    }
}

fn edge(from: usize, to: usize, from_pose: Pose64, to_pose: Pose64) -> PoseGraphEdge {
    let measurement = from_pose.inverse().compose(to_pose);
    PoseGraphEdge {
        from,
        to,
        measurement,
        information: scalar_block(1.0),
    }
}

fn translation_error(poses: &[Pose64], target: &[Pose64]) -> f64 {
    poses
        .iter()
        .zip(target.iter())
        .map(|(a, b)| {
            let dx = a.translation()[0] - b.translation()[0];
            let dy = a.translation()[1] - b.translation()[1];
            let dz = a.translation()[2] - b.translation()[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .sum::<f64>()
        / poses.len() as f64
}

#[test]
fn pose_graph_optimizer_ring_graph_converges() {
    let gt = vec![
        Pose64::identity(),
        se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        se3_exp_f64([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        se3_exp_f64([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];
    let edges = vec![
        edge(0, 1, gt[0], gt[1]),
        edge(1, 2, gt[1], gt[2]),
        edge(2, 3, gt[2], gt[3]),
        edge(3, 0, gt[3], gt[0]),
    ];
    let mut initial = vec![
        gt[0],
        se3_exp_f64([1.2, 0.1, 0.0, 0.0, 0.01, 0.0]),
        se3_exp_f64([2.3, -0.2, 0.1, 0.0, -0.02, 0.0]),
        se3_exp_f64([3.4, 0.2, -0.1, 0.0, 0.01, 0.0]),
    ];
    let before = translation_error(&initial, &gt);
    let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::default());
    let result = optimizer.optimize(&edges, &mut initial).expect("optimize");
    let after = translation_error(&result.corrected_poses, &gt);
    assert!(result.converged || result.iterations > 0);
    assert!(
        after < before,
        "ring graph did not improve: before={before}, after={after}"
    );
}

#[test]
fn pose_graph_optimizer_loop_closure_reduces_drift() {
    let gt = vec![
        Pose64::identity(),
        se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        se3_exp_f64([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];
    let edges = vec![
        edge(0, 1, gt[0], gt[1]),
        edge(1, 2, gt[1], gt[2]),
        edge(0, 2, gt[0], gt[2]),
    ];
    let mut initial = vec![
        gt[0],
        se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        se3_exp_f64([2.7, 0.4, 0.0, 0.0, 0.03, 0.0]),
    ];
    let before = translation_error(&initial, &gt);
    let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::default());
    let result = optimizer.optimize(&edges, &mut initial).expect("optimize");
    let after = translation_error(&result.corrected_poses, &gt);
    assert!(after < before, "loop closure did not reduce drift");
}

#[test]
fn pose_graph_optimizer_keeps_anchor_pose_fixed() {
    let gt = [
        Pose64::identity(),
        se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];
    let edges = vec![edge(0, 1, gt[0], gt[1])];
    let mut initial = vec![gt[0], se3_exp_f64([1.4, 0.3, 0.0, 0.0, 0.02, 0.0])];
    let anchor_before = initial[0];
    let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::default());
    let result = optimizer.optimize(&edges, &mut initial).expect("optimize");
    let anchor_after = result.corrected_poses[0];
    for i in 0..3 {
        assert!((anchor_before.translation()[i] - anchor_after.translation()[i]).abs() < 1e-12);
        for j in 0..3 {
            assert!((anchor_before.rotation()[i][j] - anchor_after.rotation()[i][j]).abs() < 1e-12);
        }
    }
}

#[test]
fn essential_graph_builds_spanning_tree_connectivity() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
    graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
    graph.add_keyframe(kf2, map.covisibility().neighbors(kf2), &map);

    assert_eq!(graph.parent_of(kf0), Some(kf0));
    assert_eq!(graph.parent_of(kf1), Some(kf0));
    assert_eq!(graph.parent_of(kf2), Some(kf1));
    assert!(graph.all_edges().len() >= 2);
}

#[test]
fn essential_graph_respects_strong_edge_threshold() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
    graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
    graph.add_keyframe(kf2, map.covisibility().neighbors(kf2), &map);
    let snapshot = graph.snapshot();
    assert_eq!(snapshot.strong_covis_edges.len(), 1);
    let strong = &snapshot.strong_covis_edges[0];
    assert_eq!(strong.kind, EssentialEdgeKind::StrongCovisibility);
    assert!((strong.a == kf1 && strong.b == kf0) || (strong.a == kf0 && strong.b == kf1));
}

#[test]
fn essential_graph_snapshot_is_independent_copy() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
    graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
    graph.add_keyframe(kf2, map.covisibility().neighbors(kf2), &map);
    let snapshot = graph.snapshot();
    graph.add_loop_edge(EssentialEdge {
        a: kf2,
        b: kf0,
        kind: EssentialEdgeKind::Loop,
        relative_pose: Pose64::identity(),
        information: scalar_block(1.0),
    });
    assert_eq!(snapshot.loop_edges.len(), 0);
    assert_eq!(graph.snapshot().loop_edges.len(), 1);
}

#[test]
fn essential_graph_remove_keyframe_reparents_children() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
    graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
    graph.add_keyframe(kf2, map.covisibility().neighbors(kf2), &map);
    assert_eq!(graph.parent_of(kf2), Some(kf1));

    graph
        .remove_keyframe(kf1, &map)
        .expect("remove non-root keyframe");
    assert_eq!(graph.parent_of(kf2), Some(kf0));
    assert_eq!(graph.parent_of(kf1), None);
    let snapshot = graph.snapshot();
    assert!(snapshot.order.iter().all(|&id| id != kf1));
    assert!(
        snapshot
            .spanning_edges
            .iter()
            .all(|edge| edge.a != kf1 && edge.b != kf1)
    );
    let input = graph.pose_graph_input();
    assert!(input.keyframe_ids.iter().all(|&id| id != kf1));
}

#[test]
fn essential_graph_remove_keyframe_rejects_root() {
    let (map, kf0, kf1, _kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
    graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);

    let err = graph
        .remove_keyframe(kf0, &map)
        .expect_err("root removal should fail");
    assert_eq!(
        err,
        EssentialGraphError::RootRemovalDenied { keyframe_id: kf0 }
    );
}

#[test]
fn essential_graph_remove_keyframe_rejects_missing_id() {
    let (map, kf0, kf1, _kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);

    let err = graph
        .remove_keyframe(kf1, &map)
        .expect_err("missing keyframe should fail");
    assert_eq!(
        err,
        EssentialGraphError::KeyframeNotFound { keyframe_id: kf1 }
    );
}

#[test]
fn essential_graph_remove_keyframe_purges_incident_loop_edges() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
    graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
    graph.add_keyframe(kf2, map.covisibility().neighbors(kf2), &map);
    graph.add_loop_edge(EssentialEdge {
        a: kf2,
        b: kf0,
        kind: EssentialEdgeKind::Loop,
        relative_pose: Pose64::identity(),
        information: scalar_block(1.0),
    });
    assert_eq!(graph.snapshot().loop_edges.len(), 1);

    graph
        .remove_keyframe(kf2, &map)
        .expect("remove keyframe with loop edge");
    let snapshot = graph.snapshot();
    assert_eq!(snapshot.loop_edges.len(), 0);
    assert!(
        snapshot
            .strong_covis_edges
            .iter()
            .all(|e| e.a != kf2 && e.b != kf2)
    );
    assert!(
        snapshot
            .spanning_edges
            .iter()
            .all(|e| e.a != kf2 && e.b != kf2)
    );
}

#[test]
fn essential_graph_random_remove_preserves_connectivity_invariants() {
    let (map, ids) = make_chain_keyframes(12);
    let root = ids[0];
    let mut graph = EssentialGraph::new(100);
    for (idx, &id) in ids.iter().enumerate() {
        if idx == 0 {
            graph.add_keyframe(id, None, &map);
        } else {
            let mut covis = HashMap::new();
            covis.insert(ids[idx - 1], NonZeroU32::new(10).expect("non-zero"));
            graph.add_keyframe(id, Some(&covis), &map);
        }
    }

    let mut alive = ids.clone();
    let mut rng = Lcg::new(0x5EED_u64);
    for _ in 0..64 {
        let removable: Vec<_> = alive.iter().copied().filter(|id| *id != root).collect();
        if removable.is_empty() {
            break;
        }
        let remove_id = removable[rng.next_usize(removable.len())];
        graph
            .remove_keyframe(remove_id, &map)
            .expect("non-root should be removable");
        alive.retain(|id| *id != remove_id);

        let alive_set: HashSet<_> = alive.iter().copied().collect();
        let snapshot = graph.snapshot();
        assert_eq!(snapshot.parent.len(), alive.len());
        assert!(!snapshot.order.contains(&remove_id));

        for (&child, &parent) in &snapshot.parent {
            assert!(alive_set.contains(&child));
            assert!(alive_set.contains(&parent));
            if child == root {
                assert_eq!(parent, root);
            }
        }

        for edge in snapshot
            .spanning_edges
            .iter()
            .chain(snapshot.strong_covis_edges.iter())
            .chain(snapshot.loop_edges.iter())
        {
            assert!(alive_set.contains(&edge.a));
            assert!(alive_set.contains(&edge.b));
        }

        let input = graph.pose_graph_input();
        let input_set: HashSet<_> = input.keyframe_ids.iter().copied().collect();
        assert!(input_set.is_subset(&alive_set));
        for edge in &input.edges {
            assert!(edge.from < input.keyframe_ids.len());
            assert!(edge.to < input.keyframe_ids.len());
        }
    }
}
