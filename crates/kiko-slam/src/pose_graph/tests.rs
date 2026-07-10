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
use crate::{CompactDescriptor, FrameId, Keypoint, Point3, Pose, Timestamp, WorldToCamera};

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

fn transform_point(pose: Pose64, point: [f64; 3]) -> [f64; 3] {
    let rotation = pose.rotation();
    let translation = pose.translation();
    [
        rotation[0][0] * point[0]
            + rotation[0][1] * point[1]
            + rotation[0][2] * point[2]
            + translation[0],
        rotation[1][0] * point[0]
            + rotation[1][1] * point[1]
            + rotation[1][2] * point[2]
            + translation[1],
        rotation[2][0] * point[0]
            + rotation[2][1] * point[1]
            + rotation[2][2] * point[2]
            + translation[2],
    ]
}

fn point_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter()
        .zip(b)
        .map(|(lhs, rhs)| (lhs - rhs) * (lhs - rhs))
        .sum::<f64>()
        .sqrt()
}

fn raw_left_increment_central_difference(
    edge: &PoseGraphEdge,
    poses: &[Pose64],
    pose_idx: usize,
    eps: f64,
) -> [[f64; 6]; 6] {
    let mut jacobian = [[0.0_f64; 6]; 6];
    for axis in 0..6 {
        let mut plus = poses.to_vec();
        let mut minus = poses.to_vec();
        let mut delta = [0.0_f64; 6];
        delta[axis] = eps;
        plus[pose_idx] = se3_exp_f64(delta).compose(plus[pose_idx]);
        delta[axis] = -eps;
        minus[pose_idx] = se3_exp_f64(delta).compose(minus[pose_idx]);
        let error_plus = compute_edge_error(edge, &plus).expect("positive perturbation");
        let error_minus = compute_edge_error(edge, &minus).expect("negative perturbation");
        for row in 0..6 {
            jacobian[row][axis] = (error_plus[row] - error_minus[row]) / (2.0 * eps);
        }
    }
    jacobian
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
            WorldToCamera::identity(),
            size,
            keypoints.clone(),
        )
        .expect("kf0");
    let kf1 = map
        .add_keyframe(
            FrameId::new(2),
            Timestamp::from_nanos(2),
            WorldToCamera::from_legacy_pose(Pose::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [1.0, 0.0, 0.0],
            )),
            size,
            keypoints.clone(),
        )
        .expect("kf1");
    let kf2 = map
        .add_keyframe(
            FrameId::new(3),
            Timestamp::from_nanos(3),
            WorldToCamera::from_legacy_pose(Pose::from_rt(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [2.0, 0.0, 0.0],
            )),
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
                WorldToCamera::from_legacy_pose(pose),
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
fn pcg_handles_tiny_scale_aware_curvature() {
    let mut h = BlockCsr6x6::new(1);
    h.insert(0, 0, scalar_block(1e20)).expect("insert");
    let b = vec![1.0; 6];
    let mut x = vec![0.0; 6];

    let result = solve_pcg(&h, &b, &mut x, 10, 1e-12).expect("pcg");

    assert!(
        result.converged,
        "scaled system did not converge: {result:?}"
    );
    assert_eq!(result.iterations, 1);
    for value in x {
        assert!((value - 1e-20).abs() < 1e-30);
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
    let measurement = pose_b.compose(pose_a.inverse());
    let edge = PoseGraphEdge::try_new(0, 1, measurement, scalar_block(1.0)).expect("edge");
    let error = compute_edge_error(&edge, &[pose_a, pose_b]).expect("edge error");
    let norm: f64 = error.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm < 1e-9, "expected near-zero error, got {norm}");
}

#[test]
fn pose_graph_edge_jacobians_match_finite_difference() {
    let pose_a = se3_exp_f64([0.1, 0.05, -0.02, 0.02, -0.01, 0.03]);
    let pose_b = se3_exp_f64([0.3, -0.08, 0.12, -0.02, 0.03, -0.01]);
    let measurement = pose_b
        .compose(pose_a.inverse())
        .compose(se3_exp_f64([0.1, -0.05, 0.02, 0.25, -0.18, 0.12]));
    let edge = PoseGraphEdge::try_new(0, 1, measurement, scalar_block(1.0)).expect("edge");
    let poses = [pose_a, pose_b];
    let (j_from, j_to) = compute_edge_jacobians(&edge, &poses).expect("jacobians");
    let expected_from = raw_left_increment_central_difference(&edge, &poses, 0, 1e-6);
    let expected_to = raw_left_increment_central_difference(&edge, &poses, 1, 1e-6);
    for row in 0..6 {
        for col in 0..6 {
            assert!(
                (j_from[row][col] - expected_from[row][col]).abs() < 1e-9,
                "J_from mismatch at ({row}, {col}): actual={}, expected={}",
                j_from[row][col],
                expected_from[row][col]
            );
            assert!(
                (j_to[row][col] - expected_to[row][col]).abs() < 1e-9,
                "J_to mismatch at ({row}, {col}): actual={}, expected={}",
                j_to[row][col],
                expected_to[row][col]
            );
        }
    }
}

#[test]
fn pose_graph_edge_rejects_invalid_information_and_self_edges() {
    let pose = Pose64::identity();
    assert!(PoseGraphEdge::try_new(0, 1, pose, scalar_block(1e-20)).is_ok());
    assert!(matches!(
        PoseGraphEdge::try_new(0, 0, pose, scalar_block(1.0)),
        Err(super::PoseGraphEdgeError::SelfEdge { index: 0 })
    ));

    let mut nonfinite = scalar_block(1.0);
    nonfinite[2][3] = f64::NAN;
    assert!(matches!(
        PoseGraphEdge::try_new(0, 1, pose, nonfinite),
        Err(super::PoseGraphEdgeError::NonFiniteInformation { row: 2, col: 3 })
    ));

    let mut asymmetric = scalar_block(1.0);
    asymmetric[0][1] = 0.5;
    assert!(matches!(
        PoseGraphEdge::try_new(0, 1, pose, asymmetric),
        Err(super::PoseGraphEdgeError::NonSymmetricInformation { .. })
    ));

    let mut indefinite = scalar_block(1.0);
    indefinite[4][4] = -1.0;
    assert!(matches!(
        PoseGraphEdge::try_new(0, 1, pose, indefinite),
        Err(super::PoseGraphEdgeError::NonPositiveDefiniteInformation { .. })
    ));
}

#[test]
fn pose_graph_config_rejects_invalid_solver_limits() {
    assert!(matches!(
        PoseGraphConfig::try_new(0, 10, 1e-6, 1.0),
        Err(super::PoseGraphConfigError::ZeroIterations {
            field: "max_iterations"
        })
    ));
    assert!(matches!(
        PoseGraphConfig::try_new(10, 0, 1e-6, 1.0),
        Err(super::PoseGraphConfigError::ZeroIterations {
            field: "pcg_max_iters"
        })
    ));
    for (tol, huber) in [(0.0, 1.0), (f64::NAN, 1.0), (1e-6, f64::INFINITY)] {
        assert!(PoseGraphConfig::try_new(10, 10, tol, huber).is_err());
    }
}

fn edge(from: usize, to: usize, from_pose: Pose64, to_pose: Pose64) -> PoseGraphEdge {
    let measurement = to_pose.compose(from_pose.inverse());
    PoseGraphEdge::try_new(from, to, measurement, scalar_block(1.0)).expect("edge")
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
fn pose_graph_optimizer_rejects_unconverged_pcg_step() {
    let target = [
        Pose64::identity(),
        se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];
    let edges = vec![edge(0, 1, target[0], target[1])];
    let mut initial = vec![target[0], se3_exp_f64([2.0, 0.5, 0.0, 0.0, 0.0, 0.0])];
    let before = initial.clone();
    let defaults = PoseGraphConfig::default();
    let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::new_unchecked_for_test(
        defaults.max_iterations(),
        0,
        defaults.pcg_tol(),
        defaults.huber_delta(),
    ));

    let error = optimizer
        .optimize(&edges, &mut initial)
        .expect_err("zero PCG iterations must not be reported as convergence");

    assert!(matches!(
        error,
        super::PoseGraphError::PcgDidNotConverge { iterations: 0 }
    ));
    for (actual, expected) in initial.iter().zip(before.iter()) {
        assert_eq!(actual.translation(), expected.translation());
        assert_eq!(actual.rotation(), expected.rotation());
    }
}

#[test]
fn pose_graph_optimizer_rejects_invalid_edge_endpoints() {
    let poses = [
        Pose64::identity(),
        se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];
    let bad_edge = PoseGraphEdge::try_new(
        0,
        poses.len(),
        poses[1].compose(poses[0].inverse()),
        scalar_block(1.0),
    )
    .expect("edge structure is valid independently of pose array bounds");
    let mut initial = poses.to_vec();

    let error = PoseGraphOptimizer::new(PoseGraphConfig::default())
        .optimize(&[bad_edge], &mut initial)
        .expect_err("out-of-range endpoint must not be skipped");

    assert!(matches!(
        error,
        super::PoseGraphError::EdgeToOutOfBounds {
            to: 2,
            pose_count: 2
        }
    ));
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
fn essential_measurement_maps_from_camera_point_to_to_camera_point() {
    let (mut map, kf0, kf1, _kf2) = make_map_for_essential_graph();
    let from_pose = se3_exp_f64([0.8, -0.3, 0.2, 0.25, -0.15, 0.1]);
    let to_pose = se3_exp_f64([-0.4, 0.7, 0.5, -0.2, 0.12, 0.3]);
    map.set_keyframe_pose(kf0, WorldToCamera::from_legacy_pose(from_pose.to_pose32()))
        .expect("from pose");
    map.set_keyframe_pose(kf1, WorldToCamera::from_legacy_pose(to_pose.to_pose32()))
        .expect("to pose");

    let mut graph = EssentialGraph::new(2);
    graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
    graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
    let input = graph.pose_graph_input();
    let from_idx = input
        .keyframe_ids
        .iter()
        .position(|&id| id == kf0)
        .expect("from keyframe index");
    let to_idx = input
        .keyframe_ids
        .iter()
        .position(|&id| id == kf1)
        .expect("to keyframe index");
    let edge = input
        .edges
        .iter()
        .find(|edge| edge.from() == from_idx && edge.to() == to_idx)
        .expect("directed spanning edge");

    let world_point = [1.3, -0.6, 4.2];
    let point_in_from_camera = transform_point(from_pose, world_point);
    let expected_in_to_camera = transform_point(to_pose, world_point);
    let measured_in_to_camera = transform_point(edge.measurement(), point_in_from_camera);

    assert!(
        point_distance(measured_in_to_camera, expected_in_to_camera) < 2e-6,
        "edge measurement must map camera-from coordinates into camera-to coordinates"
    );
    let error = compute_edge_error(edge, &[from_pose, to_pose]).expect("edge error");
    assert!(
        error.iter().map(|value| value * value).sum::<f64>().sqrt() < 2e-6,
        "physical relative measurement must have zero graph residual"
    );
}

#[test]
fn essential_graph_respects_strong_edge_threshold() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(1);
    graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
    graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
    let mut neighbors = HashMap::new();
    neighbors.insert(kf0, NonZeroU32::new(1).expect("nonzero"));
    neighbors.insert(kf1, NonZeroU32::new(2).expect("nonzero"));
    graph.add_keyframe(kf2, Some(&neighbors), &map);
    let snapshot = graph.snapshot();
    assert_eq!(snapshot.strong_covis_edges.len(), 1);
    let strong = &snapshot.strong_covis_edges[0];
    assert_eq!(strong.kind(), EssentialEdgeKind::StrongCovisibility);
    assert!((strong.a() == kf0 && strong.b() == kf2) || (strong.a() == kf2 && strong.b() == kf0));
    assert_eq!(graph.parent_of(kf2), Some(kf1));
}

#[test]
fn essential_graph_does_not_duplicate_spanning_pairs_as_strong_edges() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(1);
    graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
    graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
    graph.add_keyframe(kf2, map.covisibility().neighbors(kf2), &map);
    let snapshot = graph.snapshot();

    for spanning in &snapshot.spanning_edges {
        assert!(snapshot.strong_covis_edges.iter().all(|strong| {
            !((strong.a() == spanning.a() && strong.b() == spanning.b())
                || (strong.a() == spanning.b() && strong.b() == spanning.a()))
        }));
    }
}

#[test]
fn essential_graph_snapshot_is_independent_copy() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    graph.add_keyframe(kf0, map.covisibility().neighbors(kf0), &map);
    graph.add_keyframe(kf1, map.covisibility().neighbors(kf1), &map);
    graph.add_keyframe(kf2, map.covisibility().neighbors(kf2), &map);
    let snapshot = graph.snapshot();
    graph.add_loop_edge(
        EssentialEdge::try_new(
            kf2,
            kf0,
            EssentialEdgeKind::Loop,
            Pose64::identity(),
            scalar_block(1.0),
        )
        .expect("loop edge"),
    );
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
            .all(|edge| edge.a() != kf1 && edge.b() != kf1)
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
    graph.add_loop_edge(
        EssentialEdge::try_new(
            kf2,
            kf0,
            EssentialEdgeKind::Loop,
            Pose64::identity(),
            scalar_block(1.0),
        )
        .expect("loop edge"),
    );
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
            .all(|e| e.a() != kf2 && e.b() != kf2)
    );
    assert!(
        snapshot
            .spanning_edges
            .iter()
            .all(|e| e.a() != kf2 && e.b() != kf2)
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
            assert!(alive_set.contains(&edge.a()));
            assert!(alive_set.contains(&edge.b()));
        }

        let input = graph.pose_graph_input();
        let input_set: HashSet<_> = input.keyframe_ids.iter().copied().collect();
        assert!(input_set.is_subset(&alive_set));
        for edge in &input.edges {
            assert!(edge.from() < input.keyframe_ids.len());
            assert!(edge.to() < input.keyframe_ids.len());
        }
    }
}
