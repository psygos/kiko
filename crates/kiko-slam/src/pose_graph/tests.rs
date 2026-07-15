use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;

use super::{
    BlockCsr6x6, EssentialEdge, EssentialEdgeKind, EssentialGraph, EssentialGraphError,
    PoseGraphConfig, PoseGraphEdge, PoseGraphOptimizer, compute_edge_error, compute_edge_jacobians,
    solve_pcg,
};
use crate::map::{ImageSize, KeyframeId, SlamMap};
use crate::math::se3_exp_f64;
use crate::{CompactDescriptor, FrameId, Keypoint, Point3, Pose, Timestamp, WorldToCamera};
use crate::{Pose64, Pose64Error};

use super::optimizer::clamp_step;

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
        plus[pose_idx] = se3_exp_f64(delta)
            .try_compose(plus[pose_idx])
            .expect("positive perturbation must stay valid");
        delta[axis] = -eps;
        minus[pose_idx] = se3_exp_f64(delta)
            .try_compose(minus[pose_idx])
            .expect("negative perturbation must stay valid");
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

fn register_keyframe(
    graph: &mut EssentialGraph,
    keyframe_id: crate::map::KeyframeId,
    covisibility: Option<&HashMap<crate::map::KeyframeId, NonZeroU32>>,
    map: &SlamMap,
) {
    graph
        .add_keyframe(keyframe_id, covisibility, map)
        .expect("register keyframe in essential graph");
}

fn contains_pair(
    edges: &[EssentialEdge],
    a: crate::map::KeyframeId,
    b: crate::map::KeyframeId,
) -> bool {
    edges
        .iter()
        .any(|edge| (edge.a() == a && edge.b() == b) || (edge.a() == b && edge.b() == a))
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
    let measurement = pose_b
        .try_compose(pose_a.try_inverse().expect("inverse"))
        .expect("measurement");
    let edge = PoseGraphEdge::try_new(0, 1, measurement, scalar_block(1.0)).expect("edge");
    let error = compute_edge_error(&edge, &[pose_a, pose_b]).expect("edge error");
    let norm: f64 = error.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm < 1e-9, "expected near-zero error, got {norm}");
}

#[test]
fn pose_graph_edge_error_reports_pose_arithmetic_overflow() {
    let from_pose = Pose64::from_rt(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [-f64::MAX, 0.0, 0.0],
    )
    .expect("finite from pose");
    let to_pose = Pose64::from_rt(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [f64::MAX, 0.0, 0.0],
    )
    .expect("finite to pose");
    let edge =
        PoseGraphEdge::try_new(0, 1, Pose64::identity(), scalar_block(1.0)).expect("valid edge");

    assert!(matches!(
        compute_edge_error(&edge, &[from_pose, to_pose]),
        Err(super::PoseGraphError::PoseComputation(
            Pose64Error::ComposeTranslationNonFinite { axis: 0 }
        ))
    ));
    assert!(
        compute_edge_jacobians(&edge, &[from_pose, to_pose]).is_err(),
        "Jacobian computation must not return non-finite values"
    );
}

#[test]
fn pose_graph_edge_error_rejects_nonfinite_log_residual() {
    let to_pose = Pose64::from_rt(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [f64::MAX, f64::MAX, 0.0],
    )
    .expect("finite pose");
    let edge =
        PoseGraphEdge::try_new(0, 1, Pose64::identity(), scalar_block(1.0)).expect("valid edge");

    assert_eq!(
        compute_edge_error(&edge, &[Pose64::identity(), to_pose])
            .expect_err("SE(3) log overflow must be typed"),
        super::PoseGraphError::NonFiniteEdgeResidual { component: 0 }
    );
}

#[test]
fn pose_graph_edge_jacobians_match_finite_difference() {
    let pose_a = se3_exp_f64([0.1, 0.05, -0.02, 0.02, -0.01, 0.03]);
    let pose_b = se3_exp_f64([0.3, -0.08, 0.12, -0.02, 0.03, -0.01]);
    let measurement = pose_b
        .try_compose(pose_a.try_inverse().expect("inverse"))
        .expect("relative pose")
        .try_compose(se3_exp_f64([0.1, -0.05, 0.02, 0.25, -0.18, 0.12]))
        .expect("perturbed measurement");
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

#[test]
fn pose_graph_step_clamps_large_finite_components_without_square_overflow() {
    let mut step = [1e300, -1e300, 5e299, 0.0, 0.0, 0.0];

    let reported_norm = clamp_step(&mut step, 7).expect("finite step must be clamped");
    let actual_norm = step.iter().fold(0.0_f64, |norm, value| norm.hypot(*value));

    assert_eq!(reported_norm, 1.0);
    assert!(
        (actual_norm - 1.0).abs() < 1e-15,
        "actual norm={actual_norm}"
    );
    assert!(step.iter().all(|value| value.is_finite()));
}

fn edge(from: usize, to: usize, from_pose: Pose64, to_pose: Pose64) -> PoseGraphEdge {
    let measurement = to_pose
        .try_compose(from_pose.try_inverse().expect("inverse"))
        .expect("measurement");
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
fn pose_graph_optimizer_accepts_one_pose_without_constraints() {
    let pose = se3_exp_f64([0.4, -0.2, 0.1, 0.03, -0.04, 0.02]);
    let mut initial = [pose];

    let result = PoseGraphOptimizer::new(PoseGraphConfig::default())
        .optimize(&[], &mut initial)
        .expect("one pose is a fully anchored trivial graph");

    assert!(result.converged);
    assert_eq!(result.iterations, 0);
    assert_eq!(result.corrected_poses.len(), 1);
    assert_eq!(result.corrected_poses[0].translation(), pose.translation());
    assert_eq!(result.corrected_poses[0].rotation(), pose.rotation());
}

#[test]
fn pose_graph_optimizer_rejects_multiple_poses_without_constraints() {
    let mut initial = [Pose64::identity(), Pose64::identity()];

    let error = PoseGraphOptimizer::new(PoseGraphConfig::default())
        .optimize(&[], &mut initial)
        .expect_err("multiple poses without constraints are underconstrained");

    assert_eq!(
        error,
        super::PoseGraphError::UnconstrainedPoseGraph { pose_count: 2 }
    );
}

#[test]
fn pose_graph_optimizer_rejects_disconnected_components() {
    let poses = [
        Pose64::identity(),
        se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        se3_exp_f64([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        se3_exp_f64([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];
    let constraints = [
        edge(0, 1, poses[0], poses[1]),
        edge(2, 3, poses[2], poses[3]),
    ];
    let mut initial = poses;

    let error = PoseGraphOptimizer::new(PoseGraphConfig::default())
        .optimize(&constraints, &mut initial)
        .expect_err("each disconnected component has an independent gauge freedom");

    assert_eq!(
        error,
        super::PoseGraphError::DisconnectedPoseGraph {
            pose_count: 4,
            component_count: 2,
            anchor_component_size: 2,
        }
    );
}

#[test]
fn pose_graph_optimizer_accepts_connected_constraint_tree() {
    let poses = [
        Pose64::identity(),
        se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        se3_exp_f64([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        se3_exp_f64([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ];
    let constraints = [
        edge(1, 0, poses[1], poses[0]),
        edge(1, 2, poses[1], poses[2]),
        edge(3, 2, poses[3], poses[2]),
    ];
    let mut initial = poses;

    let result = PoseGraphOptimizer::new(PoseGraphConfig::default())
        .optimize(&constraints, &mut initial)
        .expect("edge direction must not affect topology connectivity");

    assert!(result.converged);
    assert_eq!(result.iterations, 1);
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
fn pose_graph_optimizer_uses_a_hard_anchor_at_high_information() {
    let anchor = Pose64::identity();
    let target = se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let measurement = target
        .try_compose(anchor.try_inverse().expect("inverse"))
        .expect("measurement");
    let high_information = scalar_block(1e12);
    let edge = PoseGraphEdge::try_new(0, 1, measurement, high_information)
        .expect("positive definite high-information edge");
    let mut initial = vec![anchor, se3_exp_f64([1.1, 0.0, 0.0, 0.0, 0.0, 0.0])];
    let defaults = PoseGraphConfig::default();
    let one_iteration = PoseGraphConfig::new_unchecked_for_test(
        1,
        defaults.pcg_max_iters(),
        defaults.pcg_tol(),
        defaults.huber_delta(),
    );

    let result = PoseGraphOptimizer::new(one_iteration)
        .optimize(&[edge], &mut initial)
        .expect("hard-anchored optimization");

    assert_eq!(
        result.corrected_poses[0].translation(),
        anchor.translation()
    );
    assert_eq!(result.corrected_poses[0].rotation(), anchor.rotation());
    let corrected_x = result.corrected_poses[1].translation()[0];
    assert!(
        (corrected_x - 1.0).abs() < 1e-6,
        "hard-anchor step should recover x=1 without soft-prior overshoot, got {corrected_x}"
    );
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
        poses[1]
            .try_compose(poses[0].try_inverse().expect("inverse"))
            .expect("measurement"),
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
fn pose_graph_optimizer_validates_edges_before_empty_graph_shortcut() {
    let edge = PoseGraphEdge::try_new(0, 1, Pose64::identity(), scalar_block(1.0))
        .expect("edge structure is valid independently of pose array bounds");
    let mut initial = [];

    let error = PoseGraphOptimizer::new(PoseGraphConfig::default())
        .optimize(&[edge], &mut initial)
        .expect_err("an empty pose array must not bypass endpoint validation");

    assert_eq!(
        error,
        super::PoseGraphError::EdgeFromOutOfBounds {
            from: 0,
            pose_count: 0,
        }
    );
}

#[test]
fn essential_graph_builds_spanning_tree_connectivity() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    register_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    register_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);

    assert_eq!(graph.parent_of(kf0), Some(kf0));
    assert_eq!(graph.parent_of(kf1), Some(kf0));
    assert_eq!(graph.parent_of(kf2), Some(kf1));
    assert!(graph.all_edges(&map).expect("pose graph input").len() >= 2);
}

#[test]
fn verified_parent_connects_uncovisible_keyframe_in_parent_to_child_direction() {
    let (map, ids) = make_chain_keyframes(2);
    let parent = ids[0];
    let child = ids[1];
    assert!(map.covisibility().neighbors(child).is_none());
    let mut graph = EssentialGraph::new(2);
    register_keyframe(&mut graph, parent, None, &map);

    graph
        .add_keyframe_with_verified_parent(child, parent, scalar_block(7.0), &map)
        .expect("verified attachment supplies a truthful spanning connection");

    assert_eq!(graph.parent_of(child), Some(parent));
    let snapshot = graph.snapshot();
    assert_eq!(snapshot.spanning_edges.len(), 1);
    assert_eq!(snapshot.spanning_edges[0].a(), parent);
    assert_eq!(snapshot.spanning_edges[0].b(), child);
    assert_eq!(snapshot.spanning_edges[0].information(), scalar_block(7.0));
    graph
        .pose_graph_input(&map)
        .expect("verified parent topology remains connected");
}

#[test]
fn essential_graph_rejects_reversed_spanning_direction() {
    let (map, ids) = make_chain_keyframes(2);
    let parent = ids[0];
    let child = ids[1];
    let mut graph = EssentialGraph::new(2);
    register_keyframe(&mut graph, parent, None, &map);
    graph
        .add_keyframe_with_verified_parent(child, parent, scalar_block(1.0), &map)
        .expect("verified child");
    assert!(graph.reverse_spanning_edge_for_test(child));

    assert_eq!(
        graph
            .pose_graph_input(&map)
            .expect_err("unordered endpoints cannot satisfy directed parentage"),
        EssentialGraphError::MissingSpanningEdge { child, parent }
    );
}

#[test]
fn essential_graph_rejects_map_missing_keyframe_without_mutation() {
    let (map, _kf0, _kf1, _kf2) = make_map_for_essential_graph();
    let foreign_id = KeyframeId::default();
    let mut graph = EssentialGraph::new(2);

    let error = graph
        .add_keyframe(foreign_id, None, &map)
        .expect_err("a keyframe ID from outside the map cannot become the graph root");

    assert_eq!(
        error,
        EssentialGraphError::KeyframeNotFound {
            keyframe_id: foreign_id,
        }
    );
    assert!(graph.snapshot().order.is_empty());
}

#[test]
fn essential_graph_rejects_a_second_unconnected_root_without_mutation() {
    let (map, kf0, kf1, _kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    register_keyframe(&mut graph, kf0, None, &map);

    let error = graph
        .add_keyframe(kf1, None, &map)
        .expect_err("only the first keyframe may establish a root");

    assert_eq!(
        error,
        EssentialGraphError::DisconnectedKeyframe { keyframe_id: kf1 }
    );
    let snapshot = graph.snapshot();
    assert_eq!(snapshot.order, vec![kf0]);
    assert_eq!(snapshot.parent.len(), 1);
    assert_eq!(snapshot.parent.get(&kf0), Some(&kf0));
}

#[test]
fn essential_graph_rejects_unregistered_loop_endpoint_without_auto_registration() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    register_keyframe(&mut graph, kf0, None, &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    let edge = EssentialEdge::try_new(
        kf0,
        kf2,
        EssentialEdgeKind::Loop,
        Pose64::identity(),
        scalar_block(1.0),
    )
    .expect("valid loop edge payload");

    let error = graph
        .add_loop_edge(edge, &map)
        .expect_err("loop endpoints must already belong to the essential graph");

    assert_eq!(
        error,
        EssentialGraphError::KeyframeNotRegistered { keyframe_id: kf2 }
    );
    let snapshot = graph.snapshot();
    assert!(!snapshot.order.contains(&kf2));
    assert!(snapshot.loop_edges.is_empty());
}

#[test]
fn essential_graph_rejects_self_loop_and_mislabeled_loop_edge() {
    let (map, kf0, kf1, _kf2) = make_map_for_essential_graph();
    assert!(matches!(
        EssentialEdge::try_new(
            kf0,
            kf0,
            EssentialEdgeKind::Loop,
            Pose64::identity(),
            scalar_block(1.0),
        ),
        Err(EssentialGraphError::SelfEdge { keyframe_id }) if keyframe_id == kf0
    ));

    let mut graph = EssentialGraph::new(2);
    register_keyframe(&mut graph, kf0, None, &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    let mislabeled = EssentialEdge::try_new(
        kf0,
        kf1,
        EssentialEdgeKind::StrongCovisibility,
        Pose64::identity(),
        scalar_block(1.0),
    )
    .expect("valid non-loop edge payload");

    assert_eq!(
        graph
            .add_loop_edge(mislabeled, &map)
            .expect_err("loop collection must preserve its typed edge kind"),
        EssentialGraphError::UnexpectedEdgeKind {
            expected: EssentialEdgeKind::Loop,
            actual: EssentialEdgeKind::StrongCovisibility,
        }
    );
}

#[test]
fn essential_graph_rejects_duplicate_unordered_loop_pair() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    register_keyframe(&mut graph, kf0, None, &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    register_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);
    let loop_edge = |a, b| {
        EssentialEdge::try_new(
            a,
            b,
            EssentialEdgeKind::Loop,
            Pose64::identity(),
            scalar_block(1.0),
        )
        .expect("valid loop edge")
    };
    graph
        .add_loop_edge(loop_edge(kf2, kf0), &map)
        .expect("first loop pair");

    let error = graph
        .add_loop_edge(loop_edge(kf0, kf2), &map)
        .expect_err("reversed endpoints identify the same loop pair");

    assert_eq!(
        error,
        EssentialGraphError::DuplicateEdge {
            a: kf0.min(kf2),
            b: kf0.max(kf2),
            kind: EssentialEdgeKind::Loop,
        }
    );
    assert_eq!(graph.snapshot().loop_edges.len(), 1);
}

#[test]
fn pose_graph_input_rejects_stale_map_keyframe_before_indexing_edges() {
    let (mut map, kf0, kf1, _kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    register_keyframe(&mut graph, kf0, None, &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    map.remove_keyframe(kf1).expect("inject stale graph ID");

    let error = graph
        .pose_graph_input(&map)
        .expect_err("stale map identity must fail before optimizer input is produced");

    assert_eq!(
        error,
        EssentialGraphError::KeyframeNotFound { keyframe_id: kf1 }
    );
}

#[test]
fn essential_measurement_maps_from_camera_point_to_to_camera_point() {
    let (mut map, kf0, kf1, _kf2) = make_map_for_essential_graph();
    let from_pose = se3_exp_f64([0.8, -0.3, 0.2, 0.25, -0.15, 0.1]);
    let to_pose = se3_exp_f64([-0.4, 0.7, 0.5, -0.2, 0.12, 0.3]);
    map.set_keyframe_pose(
        kf0,
        WorldToCamera::from_legacy_pose(
            from_pose
                .try_to_pose32()
                .expect("test pose should fit in f32"),
        ),
    )
    .expect("from pose");
    map.set_keyframe_pose(
        kf1,
        WorldToCamera::from_legacy_pose(
            to_pose
                .try_to_pose32()
                .expect("test pose should fit in f32"),
        ),
    )
    .expect("to pose");

    let mut graph = EssentialGraph::new(2);
    register_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    let input = graph.pose_graph_input(&map).expect("pose graph input");
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
fn essential_graph_reports_malformed_map_pose_without_mutation() {
    let (mut map, kf0, kf1, _kf2) = make_map_for_essential_graph();
    map.set_keyframe_pose(
        kf1,
        WorldToCamera::from_legacy_pose(Pose::from_rt(
            [[1.0, 0.0, 0.0], [0.0, f32::NAN, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
        )),
    )
    .expect("inject malformed legacy pose");
    let mut graph = EssentialGraph::new(2);
    register_keyframe(&mut graph, kf0, None, &map);

    let error = graph
        .add_keyframe(kf1, map.covisibility().neighbors(kf1), &map)
        .expect_err("malformed legacy pose must not enter the graph");

    assert_eq!(
        error,
        EssentialGraphError::PoseComputation(Pose64Error::NonFinite)
    );
    assert_eq!(graph.parent_of(kf1), None);
}

#[test]
fn essential_graph_respects_strong_edge_threshold() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(1);
    register_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    let mut neighbors = HashMap::new();
    neighbors.insert(kf0, NonZeroU32::new(1).expect("nonzero"));
    neighbors.insert(kf1, NonZeroU32::new(2).expect("nonzero"));
    register_keyframe(&mut graph, kf2, Some(&neighbors), &map);
    let snapshot = graph.snapshot();
    assert_eq!(snapshot.strong_covis_edges.len(), 1);
    let strong = &snapshot.strong_covis_edges[0];
    assert_eq!(strong.kind(), EssentialEdgeKind::StrongCovisibility);
    assert!((strong.a() == kf0 && strong.b() == kf2) || (strong.a() == kf2 && strong.b() == kf0));
    assert_eq!(graph.parent_of(kf2), Some(kf1));
}

#[test]
fn essential_graph_breaks_equal_parent_weights_by_keyframe_identity() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(100);
    register_keyframe(&mut graph, kf0, None, &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    let mut neighbors = HashMap::new();
    neighbors.insert(kf0, NonZeroU32::new(1).expect("nonzero"));
    neighbors.insert(kf1, NonZeroU32::new(1).expect("nonzero"));

    register_keyframe(&mut graph, kf2, Some(&neighbors), &map);

    assert_eq!(graph.parent_of(kf2), Some(kf0.min(kf1)));
}

#[test]
fn essential_graph_does_not_duplicate_spanning_pairs_as_strong_edges() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(1);
    register_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    register_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);
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
    register_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    register_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);
    let snapshot = graph.snapshot();
    graph
        .add_loop_edge(
            EssentialEdge::try_new(
                kf2,
                kf0,
                EssentialEdgeKind::Loop,
                Pose64::identity(),
                scalar_block(1.0),
            )
            .expect("loop edge"),
            &map,
        )
        .expect("register loop edge");
    assert_eq!(snapshot.loop_edges.len(), 0);
    assert_eq!(graph.snapshot().loop_edges.len(), 1);
}

#[test]
fn essential_graph_remove_keyframe_reparents_children() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    register_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    register_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);
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
    let input = graph.pose_graph_input(&map).expect("pose graph input");
    assert!(input.keyframe_ids.iter().all(|&id| id != kf1));
}

#[test]
fn essential_graph_reparenting_replaces_conflicting_strong_edge() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(1);
    register_keyframe(&mut graph, kf0, None, &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    let mut neighbors = HashMap::new();
    neighbors.insert(kf0, NonZeroU32::new(1).expect("nonzero"));
    neighbors.insert(kf1, NonZeroU32::new(2).expect("nonzero"));
    register_keyframe(&mut graph, kf2, Some(&neighbors), &map);
    assert!(contains_pair(
        &graph.snapshot().strong_covis_edges,
        kf0,
        kf2
    ));

    graph
        .remove_keyframe(kf1, &map)
        .expect("reparent through existing strong pair");

    let snapshot = graph.snapshot();
    assert_eq!(graph.parent_of(kf2), Some(kf0));
    assert!(contains_pair(&snapshot.spanning_edges, kf0, kf2));
    assert!(!contains_pair(&snapshot.strong_covis_edges, kf0, kf2));
    graph
        .pose_graph_input(&map)
        .expect("reparented topology remains valid");
}

#[test]
fn essential_graph_remove_keyframe_rejects_root() {
    let (map, kf0, kf1, _kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    register_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);

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
    register_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);

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
    register_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    register_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    register_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);
    graph
        .add_loop_edge(
            EssentialEdge::try_new(
                kf2,
                kf0,
                EssentialEdgeKind::Loop,
                Pose64::identity(),
                scalar_block(1.0),
            )
            .expect("loop edge"),
            &map,
        )
        .expect("register loop edge");
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
            register_keyframe(&mut graph, id, None, &map);
        } else {
            let mut covis = HashMap::new();
            covis.insert(ids[idx - 1], NonZeroU32::new(10).expect("non-zero"));
            register_keyframe(&mut graph, id, Some(&covis), &map);
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

        let input = graph.pose_graph_input(&map).expect("pose graph input");
        let input_set: HashSet<_> = input.keyframe_ids.iter().copied().collect();
        assert!(input_set.is_subset(&alive_set));
        for edge in &input.edges {
            assert!(edge.from() < input.keyframe_ids.len());
            assert!(edge.to() < input.keyframe_ids.len());
        }
    }
}
