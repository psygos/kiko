use std::collections::{HashMap, HashSet};
use std::error::Error as _;
use std::num::NonZeroU32;

use super::optimizer::scale_vector_to_norm;
use super::{
    BlockCsr6x6, EssentialEdgeError, EssentialEdgeKind, EssentialGraph, EssentialGraphError,
    PcgStopReason, PoseGraphConfig, PoseGraphEdge, PoseGraphOptimizer, PoseGraphTermination,
    compute_edge_error, compute_edge_jacobians, solve_pcg,
};
use crate::map::SlamMap;
use crate::math::se3_exp_f64;
use crate::{
    CompactDescriptor, FrameDimensions, FrameId, Keypoint, Point3, Pose, Pose64, Timestamp,
};

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

fn add_graph_keyframe(
    graph: &mut EssentialGraph,
    keyframe_id: crate::map::KeyframeId,
    covisibility: Option<&HashMap<crate::map::KeyframeId, NonZeroU32>>,
    map: &SlamMap,
) {
    graph
        .add_keyframe(keyframe_id, covisibility, map)
        .expect("test keyframe and pose must be valid for the essential graph");
}

fn make_map_for_essential_graph() -> (
    SlamMap,
    crate::map::KeyframeId,
    crate::map::KeyframeId,
    crate::map::KeyframeId,
) {
    let mut map = SlamMap::new();
    let size = FrameDimensions::try_new(640, 480).expect("size");
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
    let size = FrameDimensions::try_new(640, 480).expect("size");
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
fn block_csr_exact_anchor_removes_all_anchor_coupling() {
    let mut h = BlockCsr6x6::new(2);
    h.insert(0, 0, scalar_block(4.0)).expect("anchor diagonal");
    h.insert(0, 1, scalar_block(2.0)).expect("forward coupling");
    h.insert(1, 0, scalar_block(2.0)).expect("reverse coupling");
    h.insert(1, 1, scalar_block(3.0)).expect("free diagonal");

    h.fix_block_to_zero_increment(0).expect("fix anchor");

    assert_eq!(h.get(0, 0), Some(scalar_block(1.0)));
    assert_eq!(h.get(0, 1), Some([[0.0; 6]; 6]));
    assert_eq!(h.get(1, 0), Some([[0.0; 6]; 6]));
    assert_eq!(h.get(1, 1), Some(scalar_block(3.0)));
    h.validate_symmetric()
        .expect("fixed system remains symmetric");
}

#[test]
fn block_csr_rejects_nonfinite_updates_transactionally() {
    let mut h = BlockCsr6x6::new(1);
    h.insert(0, 0, scalar_block(1.0)).expect("insert");
    let before = h.get(0, 0).expect("existing block");

    let mut nonfinite = scalar_block(1.0);
    nonfinite[2][3] = f64::NAN;
    let error = h
        .add_to(0, 0, nonfinite)
        .expect_err("non-finite update must fail");
    assert!(matches!(
        error,
        super::PoseGraphError::NonFiniteCsrBlockValue {
            row: 0,
            col: 0,
            block_row: 2,
            block_col: 3,
            ..
        }
    ));
    assert_eq!(h.get(0, 0), Some(before));
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
    assert!(result.converged());
    assert_eq!(result.iterations, 1);
    for i in 0..x.len() {
        assert!((x[i] - b[i]).abs() < 1e-12);
    }
}

#[test]
fn pcg_rejects_invalid_tolerance_and_singular_preconditioner() {
    let mut identity = BlockCsr6x6::new(1);
    identity.insert(0, 0, scalar_block(1.0)).expect("insert");
    let b = vec![1.0; 6];
    let mut x = vec![0.0; 6];
    assert!(matches!(
        solve_pcg(&identity, &b, &mut x, 10, f64::NAN),
        Err(super::PoseGraphError::InvalidPcgTolerance { .. })
    ));
    assert!(matches!(
        solve_pcg(&identity, &b, &mut x, 10, 1.01),
        Err(super::PoseGraphError::InvalidPcgTolerance { .. })
    ));

    let singular = BlockCsr6x6::new(1);
    let before = x.clone();
    let error =
        solve_pcg(&singular, &b, &mut x, 10, 1e-6).expect_err("singular diagonal must fail");
    assert!(matches!(
        error,
        super::PoseGraphError::InvalidPcgDiagonalBlock { block_index: 0 }
    ));
    assert_eq!(x, before);
}

#[test]
fn pcg_rejects_asymmetric_matrix_before_mutating_solution() {
    let mut h = BlockCsr6x6::new(2);
    h.insert(0, 0, scalar_block(1.0)).expect("insert");
    h.insert(1, 1, scalar_block(1.0)).expect("insert");
    h.insert(0, 1, scalar_block(0.25)).expect("insert");
    let b = vec![1.0; 12];
    let mut x = vec![0.0; 12];
    let before = x.clone();

    let error =
        solve_pcg(&h, &b, &mut x, 10, 1e-6).expect_err("asymmetric system must be rejected");
    assert!(matches!(
        error,
        super::PoseGraphError::AsymmetricPcgMatrix { row: 0, col: 1, .. }
    ));
    assert_eq!(x, before);
}

#[test]
fn pcg_reports_nonpositive_curvature_without_mutating_solution() {
    let mut h = BlockCsr6x6::new(2);
    h.insert(0, 0, scalar_block(1.0)).expect("insert");
    h.insert(1, 1, scalar_block(1.0)).expect("insert");
    h.insert(0, 1, scalar_block(2.0)).expect("insert");
    h.insert(1, 0, scalar_block(2.0)).expect("insert");
    let b = [vec![1.0; 6], vec![-1.0; 6]].concat();
    let mut x = vec![0.0; 12];
    let result = solve_pcg(&h, &b, &mut x, 10, 1e-6).expect("pcg outcome");
    assert_eq!(result.stop_reason, PcgStopReason::NonPositiveCurvature);
    assert_eq!(x, vec![0.0; 12]);
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
    assert!(result.converged(), "pcg did not converge: {result:?}");
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
    assert!(result.converged());
    assert_eq!(result.iterations, 0);
    assert!(x.iter().all(|v| v.abs() < 1e-15));
}

#[test]
fn pose_graph_edge_error_is_zero_for_consistent_measurement() {
    let pose_a = Pose64::identity();
    let pose_b = se3_exp_f64([0.2, -0.1, 0.05, 0.03, -0.02, 0.01]);
    let measurement = pose_a.inverse().compose(pose_b);
    let edge = PoseGraphEdge::try_new(0, 1, measurement, scalar_block(1.0))
        .expect("valid pose graph edge");
    let error = compute_edge_error(&edge, &[pose_a, pose_b]).expect("edge error");
    let norm: f64 = error.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm < 1e-9, "expected near-zero error, got {norm}");
}

#[test]
fn pose_graph_edge_error_reports_pose_arithmetic_overflow() {
    let from_pose = Pose64::try_from_rt(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [-f64::MAX, 0.0, 0.0],
    )
    .expect("finite source pose");
    let to_pose = Pose64::try_from_rt(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [f64::MAX, 0.0, 0.0],
    )
    .expect("finite destination pose");
    let edge =
        PoseGraphEdge::try_new(0, 1, Pose64::identity(), scalar_block(1.0)).expect("valid edge");

    assert!(matches!(
        compute_edge_error(&edge, &[from_pose, to_pose]),
        Err(super::PoseGraphError::PoseComputation { .. })
    ));
    assert!(compute_edge_jacobians(&edge, &[from_pose, to_pose]).is_err());
}

#[test]
fn pose_graph_edge_error_rejects_nonfinite_log_residual() {
    let to_pose = Pose64::try_from_rt(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [f64::MAX, f64::MAX, 0.0],
    )
    .expect("finite pose");
    let edge =
        PoseGraphEdge::try_new(0, 1, Pose64::identity(), scalar_block(1.0)).expect("valid edge");

    assert!(matches!(
        compute_edge_error(&edge, &[Pose64::identity(), to_pose]),
        Err(super::PoseGraphError::NonFiniteEdgeResidual { component: 0, .. })
    ));
}

#[test]
fn pose_graph_step_clamp_handles_extreme_finite_components() {
    let mut step = [1e300, -1e300, 5e299];
    scale_vector_to_norm(&mut step, 1.0);
    let actual_norm = step.into_iter().fold(0.0_f64, f64::hypot);

    assert!((actual_norm - 1.0).abs() < 1e-15);
    assert!(step.into_iter().all(f64::is_finite));
}

#[test]
fn pose_graph_edge_jacobians_match_finite_difference() {
    let pose_a = se3_exp_f64([0.1, 0.05, -0.02, 0.02, -0.01, 0.03]);
    let pose_b = se3_exp_f64([0.3, -0.08, 0.12, -0.02, 0.03, -0.01]);
    let measurement = pose_a
        .inverse()
        .compose(pose_b)
        .compose(se3_exp_f64([0.01, -0.005, 0.002, 0.001, -0.0015, 0.0008]));
    let edge = PoseGraphEdge::try_new(0, 1, measurement, scalar_block(1.0))
        .expect("valid pose graph edge");
    let poses = [pose_a, pose_b];
    let (j_from, j_to) = compute_edge_jacobians(&edge, &poses).expect("jacobians");
    let eps = 1e-6;
    for (pose_idx, jacobian) in [(edge.from(), j_from), (edge.to(), j_to)] {
        for col in 0..6 {
            let mut plus = poses;
            let mut minus = poses;
            let mut delta = [0.0_f64; 6];
            delta[col] = eps;
            plus[pose_idx] = se3_exp_f64(delta).compose(poses[pose_idx]);
            delta[col] = -eps;
            minus[pose_idx] = se3_exp_f64(delta).compose(poses[pose_idx]);
            let error_plus = compute_edge_error(&edge, &plus).expect("positive perturbation");
            let error_minus = compute_edge_error(&edge, &minus).expect("negative perturbation");

            for row in 0..6 {
                let expected = (error_plus[row] - error_minus[row]) / (2.0 * eps);
                assert!(
                    (jacobian[row][col] - expected).abs() < 1e-9,
                    "jacobian mismatch at pose={pose_idx}, row={row}, col={col}: actual={}, expected={expected}",
                    jacobian[row][col]
                );
            }
        }
    }
}

fn edge(from: usize, to: usize, from_pose: Pose64, to_pose: Pose64) -> PoseGraphEdge {
    let measurement = from_pose.inverse().compose(to_pose);
    PoseGraphEdge::try_new(from, to, measurement, scalar_block(1.0)).expect("valid pose graph edge")
}

fn optimizer_config(max_iterations: usize, pcg_max_iters: usize, pcg_tol: f64) -> PoseGraphConfig {
    PoseGraphConfig::try_new(max_iterations, pcg_max_iters, pcg_tol, 1.0).expect("optimizer config")
}

#[test]
fn pose_graph_config_rejects_invalid_numeric_values() {
    assert!(matches!(
        PoseGraphConfig::try_new(0, 100, 1e-6, 1.0),
        Err(super::PoseGraphConfigError::ZeroOuterIterations)
    ));
    assert!(matches!(
        PoseGraphConfig::try_new(20, 0, 1e-6, 1.0),
        Err(super::PoseGraphConfigError::ZeroPcgIterations)
    ));
    for invalid in [0.0, -1.0, 1.01, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            PoseGraphConfig::try_new(20, 100, invalid, 1.0),
            Err(super::PoseGraphConfigError::InvalidPcgTolerance { .. })
        ));
    }
    for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            PoseGraphConfig::try_new(20, 100, 1e-6, invalid),
            Err(super::PoseGraphConfigError::InvalidNormalizedResidualHuberDelta { .. })
        ));
    }
    let config = optimizer_config(7, 11, 1e-4);
    assert_eq!(config.max_outer_iterations(), 7);
    assert_eq!(config.max_pcg_iterations(), 11);
    assert_eq!(config.pcg_tol(), 1e-4);
    assert_eq!(config.huber_delta_normalized_residual(), 1.0);
}

#[test]
fn pose_graph_edge_parses_measurement_and_information_once() {
    assert!(matches!(
        PoseGraphEdge::try_new(2, 2, Pose64::identity(), scalar_block(1.0)),
        Err(super::PoseGraphEdgeError::SelfEdge { pose_index: 2 })
    ));

    let huge_translation = Pose64::try_from_rt(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [f64::MAX, 0.0, 0.0],
    )
    .expect("finite test pose");
    let measurement_error = huge_translation
        .try_compose(huge_translation)
        .expect_err("arithmetic-overflowed measurement must be rejected at its source");
    assert!(matches!(
        measurement_error,
        crate::Pose64Error::ComposeTranslationNonFinite { axis: 0, .. }
    ));

    let mut nonfinite = scalar_block(1.0);
    nonfinite[1][4] = f64::NAN;
    let nonfinite_error =
        PoseGraphEdge::try_new(0, 1, Pose64::identity(), nonfinite).expect_err("non-finite info");
    assert!(matches!(
        nonfinite_error,
        super::PoseGraphEdgeError::Information {
            source: super::PoseGraphInformationError::NonFiniteEntry {
                row: 1,
                col: 4,
                value,
            },
        } if value.is_nan()
    ));
    assert!(nonfinite_error.source().is_some());

    let mut asymmetric = scalar_block(1.0);
    asymmetric[0][1] = 0.25;
    assert!(matches!(
        PoseGraphEdge::try_new(0, 1, Pose64::identity(), asymmetric),
        Err(super::PoseGraphEdgeError::Information {
            source: super::PoseGraphInformationError::Asymmetric { row: 0, col: 1, .. },
        })
    ));

    let mut indefinite = scalar_block(1.0);
    indefinite[0][1] = 2.0;
    indefinite[1][0] = 2.0;
    assert!(matches!(
        PoseGraphEdge::try_new(0, 1, Pose64::identity(), indefinite),
        Err(super::PoseGraphEdgeError::Information {
            source: super::PoseGraphInformationError::NotPositiveDefinite { .. },
        })
    ));
}

#[test]
fn pose_graph_information_normalization_is_invariant_to_diagonal_scale() {
    let mut information = [[0.0_f64; 6]; 6];
    for (axis, value) in [1e-300, 1e-180, 1e-60, 1e60, 1e180, 1e300]
        .into_iter()
        .enumerate()
    {
        information[axis][axis] = value;
    }
    let parsed = super::PoseGraphInformation::try_new(information)
        .expect("positive diagonal information across scales");
    assert_eq!(parsed.matrix(), &information);
    assert!(!parsed.was_symmetrized());
}

#[test]
fn pose_graph_reports_within_tolerance_information_symmetrization() {
    let mut information = scalar_block(2.0);
    information[0][1] = 0.5;
    information[1][0] = 0.5 + 4.0 * f64::EPSILON;
    let edge = PoseGraphEdge::try_new(0, 1, Pose64::identity(), information)
        .expect("within-tolerance asymmetry");
    assert!(edge.information().was_symmetrized());
    let parsed = edge.information().matrix();
    assert_eq!(parsed[0][1], parsed[1][0]);

    let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::default());
    let mut poses = vec![Pose64::identity(), Pose64::identity()];
    let result = optimizer.optimize(&[edge], &mut poses).expect("optimize");
    assert_eq!(result.symmetrized_edge_information_count, 1);
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

    assert_eq!(result.outer_iterations, 0);
    assert_eq!(result.termination, PoseGraphTermination::NoConstraints);
    assert_eq!(result.corrected_poses, vec![pose]);
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

    PoseGraphOptimizer::new(PoseGraphConfig::default())
        .optimize(&constraints, &mut initial)
        .expect("edge direction must not affect topology connectivity");
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
    assert!(matches!(
        result.termination,
        PoseGraphTermination::Converged {
            criterion: super::PoseGraphConvergenceCriterion::TranslationAndRotationStepNorms
        }
    ));
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
    let measurement = target.compose(anchor.inverse());
    let high_information = scalar_block(1e12);
    let edge = PoseGraphEdge::try_new(0, 1, measurement, high_information)
        .expect("positive definite high-information edge");
    let mut initial = vec![anchor, se3_exp_f64([1.1, 0.0, 0.0, 0.0, 0.0, 0.0])];
    let optimizer = PoseGraphOptimizer::new(optimizer_config(1, 100, 1e-12));

    let result = optimizer
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
fn pose_graph_reports_translation_and_rotation_step_clamps_separately() {
    let target = [Pose64::identity(), Pose64::identity()];
    let edges = vec![edge(0, 1, target[0], target[1])];
    let mut initial = vec![target[0], se3_exp_f64([100.0, 0.0, 0.0, 2.5, 0.0, 0.0])];
    let optimizer = PoseGraphOptimizer::new(optimizer_config(1, 100, 1e-6));
    let result = optimizer.optimize(&edges, &mut initial).expect("optimize");

    assert_eq!(result.termination, PoseGraphTermination::IterationLimit);
    assert_eq!(result.clamped_translation_step_count, 1);
    assert_eq!(result.clamped_rotation_step_count, 1);
    assert!((result.last_max_translation_step_m - 1.0).abs() < 1e-12);
    assert!((result.last_max_rotation_step_rad - 1.0).abs() < 1e-12);
    assert!(result.last_linear_solve_residual_norm.is_finite());
}

#[test]
fn pose_graph_optimizer_rejects_invalid_edges() {
    let poses = vec![Pose64::identity()];
    let mut initial = poses.clone();
    let optimizer = PoseGraphOptimizer::new(PoseGraphConfig::default());
    let err = optimizer
        .optimize(
            &[
                PoseGraphEdge::try_new(0, 1, Pose64::identity(), scalar_block(1.0))
                    .expect("valid edge with unresolved endpoint"),
            ],
            &mut initial,
        )
        .expect_err("invalid edge should fail");
    assert!(matches!(
        err,
        super::PoseGraphError::EdgeToOutOfBounds {
            edge_index: 0,
            to: 1,
            pose_count: 1,
        }
    ));
}

#[test]
fn pose_graph_configuration_cannot_represent_zero_iteration_limits() {
    assert_eq!(
        PoseGraphConfig::try_new(0, 1, 1e-6, 1.0).expect_err("zero outer iterations"),
        super::PoseGraphConfigError::ZeroOuterIterations
    );
    assert_eq!(
        PoseGraphConfig::try_new(1, 0, 1e-6, 1.0).expect_err("zero PCG iterations"),
        super::PoseGraphConfigError::ZeroPcgIterations
    );
}

#[test]
fn pose_graph_optimizer_rejects_partial_pcg_step_without_mutating_input() {
    let gt = [
        Pose64::identity(),
        se3_exp_f64([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        se3_exp_f64([2.0, 0.2, 0.0, 0.0, 0.03, 0.0]),
    ];
    let edges = vec![
        edge(0, 1, gt[0], gt[1]),
        edge(1, 2, gt[1], gt[2]),
        edge(0, 2, gt[0], gt[2]),
    ];
    let mut initial = vec![
        gt[0],
        se3_exp_f64([1.4, 0.3, 0.0, 0.0, 0.02, 0.0]),
        se3_exp_f64([2.8, -0.4, 0.2, 0.01, -0.04, 0.02]),
    ];
    let before = initial.clone();
    let optimizer = PoseGraphOptimizer::new(optimizer_config(1, 1, 1e-15));

    let error = optimizer
        .optimize(&edges, &mut initial)
        .expect_err("partial inner solve must not be applied");
    assert!(matches!(
        error,
        super::PoseGraphError::PcgDidNotConverge {
            outer_iteration: 1,
            pcg_iterations: 1,
            stop_reason: PcgStopReason::IterationLimit,
            ..
        }
    ));
    assert_eq!(initial, before, "failed optimization mutated caller poses");
}

#[test]
fn essential_graph_builds_spanning_tree_connectivity() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    add_graph_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    add_graph_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);

    assert_eq!(graph.parent_of(kf0), Some(kf0));
    assert_eq!(graph.parent_of(kf1), Some(kf0));
    assert_eq!(graph.parent_of(kf2), Some(kf1));
    assert!(graph.all_edges().expect("valid graph edges").len() >= 2);
}

#[test]
fn verified_parent_connects_uncovisible_keyframe_in_parent_to_child_direction() {
    let (map, ids) = make_chain_keyframes(2);
    let parent = ids[0];
    let child = ids[1];
    let mut graph = EssentialGraph::new(2);
    add_graph_keyframe(&mut graph, parent, None, &map);

    graph
        .add_keyframe_with_verified_parent(child, parent, scalar_block(7.0), &map)
        .expect("verified attachment supplies a truthful spanning connection");

    assert_eq!(graph.parent_of(child), Some(parent));
    let snapshot = graph.snapshot();
    assert_eq!(snapshot.spanning_edges.len(), 1);
    assert_eq!(snapshot.spanning_edges[0].endpoint_a(), parent);
    assert_eq!(snapshot.spanning_edges[0].endpoint_b(), child);
    assert_eq!(
        snapshot.spanning_edges[0].information().matrix(),
        &scalar_block(7.0)
    );
    graph
        .pose_graph_input()
        .expect("verified parent topology remains connected");
}

#[test]
fn essential_graph_rejects_reversed_spanning_direction() {
    let (map, ids) = make_chain_keyframes(2);
    let parent = ids[0];
    let child = ids[1];
    let mut graph = EssentialGraph::new(2);
    add_graph_keyframe(&mut graph, parent, None, &map);
    graph
        .add_keyframe_with_verified_parent(child, parent, scalar_block(1.0), &map)
        .expect("verified child");
    assert!(graph.reverse_spanning_edge_for_test(child));

    assert!(matches!(
        graph
            .pose_graph_input()
            .expect_err("reversed endpoints cannot satisfy directed parentage"),
        super::PoseGraphError::EssentialTopology {
            source: EssentialGraphError::MissingSpanningEdge {
                child: actual_child,
                parent: actual_parent,
            },
        } if actual_child == child && actual_parent == parent
    ));
}

#[test]
fn essential_graph_add_rejects_missing_map_keyframe_without_mutation() {
    let (map, kf0, _kf1, _kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    let before = graph.snapshot();
    let missing = crate::map::KeyframeId::default();

    let error = graph
        .add_keyframe(missing, None, &map)
        .expect_err("a graph keyframe absent from the map must fail");

    assert_eq!(
        error,
        EssentialGraphError::KeyframeNotFound {
            keyframe_id: missing
        }
    );
    let after = graph.snapshot();
    assert_eq!(after.parent, before.parent);
    assert_eq!(after.order, before.order);
    assert_eq!(after.spanning_edges.len(), before.spanning_edges.len());
    assert_eq!(
        after.strong_covis_edges.len(),
        before.strong_covis_edges.len()
    );
}

#[test]
fn essential_graph_pose_error_preserves_conversion_source() {
    let source = crate::Pose64Error::RotationNotOrthonormal { max_error: 0.5 };
    let error = EssentialGraphError::InvalidPose {
        keyframe_id: crate::map::KeyframeId::default(),
        source,
    };

    assert_eq!(
        error.source().expect("pose conversion source").to_string(),
        source.to_string()
    );
}

#[test]
fn essential_graph_never_selects_a_future_keyframe_as_parent() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(1);
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    add_graph_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    add_graph_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);

    let snapshot = graph.snapshot();
    let order_index = snapshot
        .order
        .iter()
        .copied()
        .enumerate()
        .map(|(index, keyframe_id)| (keyframe_id, index))
        .collect::<HashMap<_, _>>();
    for (&child, &parent) in &snapshot.parent {
        if child == parent {
            assert_eq!(child, kf0, "only the root may be its own parent");
        } else {
            assert!(order_index[&parent] < order_index[&child]);
        }
    }
    assert!(graph.pose_graph_input().is_ok());
}

#[test]
fn essential_graph_does_not_duplicate_the_spanning_parent_as_a_strong_edge() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    add_graph_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    add_graph_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);
    let snapshot = graph.snapshot();
    assert!(snapshot.strong_covis_edges.is_empty());

    for spanning in &snapshot.spanning_edges {
        assert!(snapshot.strong_covis_edges.iter().all(|strong| {
            !((spanning.endpoint_a() == strong.endpoint_a()
                && spanning.endpoint_b() == strong.endpoint_b())
                || (spanning.endpoint_a() == strong.endpoint_b()
                    && spanning.endpoint_b() == strong.endpoint_a()))
        }));
    }
}

#[test]
fn essential_graph_retains_strong_non_parent_neighbors() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    add_graph_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);

    let neighbors = HashMap::from([
        (kf0, NonZeroU32::new(2).expect("non-zero weight")),
        (kf1, NonZeroU32::new(3).expect("non-zero weight")),
    ]);
    add_graph_keyframe(&mut graph, kf2, Some(&neighbors), &map);

    let snapshot = graph.snapshot();
    assert_eq!(snapshot.parent.get(&kf2), Some(&kf1));
    assert_eq!(snapshot.strong_covis_edges.len(), 1);
    let strong = &snapshot.strong_covis_edges[0];
    assert_eq!(strong.kind(), EssentialEdgeKind::StrongCovisibility);
    assert!(
        (strong.endpoint_a() == kf2 && strong.endpoint_b() == kf0)
            || (strong.endpoint_a() == kf0 && strong.endpoint_b() == kf2)
    );
}

#[test]
fn essential_graph_snapshot_is_independent_copy() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    add_graph_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    add_graph_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);
    let snapshot = graph.snapshot();
    graph
        .add_loop_edge(kf2, kf0, Pose64::identity(), scalar_block(1.0))
        .expect("registered loop endpoints");
    assert_eq!(snapshot.loop_edges.len(), 0);
    assert_eq!(graph.snapshot().loop_edges.len(), 1);
}

#[test]
fn essential_graph_without_covisibility_attaches_to_previous_keyframe() {
    let (map, ids) = make_chain_keyframes(2);
    let mut graph = EssentialGraph::new(10);
    add_graph_keyframe(&mut graph, ids[0], None, &map);
    add_graph_keyframe(&mut graph, ids[1], None, &map);

    assert_eq!(graph.parent_of(ids[0]), Some(ids[0]));
    assert_eq!(graph.parent_of(ids[1]), Some(ids[0]));
    assert_eq!(graph.snapshot().spanning_edges.len(), 1);
}

#[test]
fn essential_graph_deduplicates_loop_edges_by_keyframe_pair() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    add_graph_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    add_graph_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);

    graph
        .add_loop_edge(kf2, kf0, Pose64::identity(), scalar_block(1.0))
        .expect("registered loop endpoints");
    graph
        .add_loop_edge(kf2, kf0, Pose64::identity(), scalar_block(3.0))
        .expect("registered loop endpoints");

    let snapshot = graph.snapshot();
    assert_eq!(snapshot.loop_edges.len(), 1);
    assert_eq!(
        *snapshot.loop_edges[0].information().matrix(),
        scalar_block(3.0)
    );
}

#[test]
fn essential_graph_parses_external_edge_payload_before_mutation() {
    let (map, kf0, kf1, _kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    add_graph_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);

    let mut asymmetric = scalar_block(1.0);
    asymmetric[0][1] = 0.5;
    let error = graph
        .add_loop_edge(kf0, kf1, Pose64::identity(), asymmetric)
        .expect_err("invalid information must fail at the graph insertion boundary");
    assert!(matches!(
        error,
        EssentialGraphError::EdgeConstruction {
            kind: EssentialEdgeKind::Loop,
            source: EssentialEdgeError::Information {
                source: super::PoseGraphInformationError::Asymmetric { .. },
            },
        }
    ));
    assert!(error.source().is_some());
    assert!(error.source().and_then(std::error::Error::source).is_some());
    assert!(graph.snapshot().loop_edges.is_empty());
    graph
        .pose_graph_input()
        .expect("a rejected edge cannot poison later graph conversion");

    let huge_translation = Pose64::try_from_rt(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [f64::MAX, 0.0, 0.0],
    )
    .expect("finite test pose");
    let pose_error = huge_translation
        .try_compose(huge_translation)
        .expect_err("arithmetic-overflowed relative pose must be rejected at its source");
    assert!(matches!(
        pose_error,
        crate::Pose64Error::ComposeTranslationNonFinite { axis: 0, .. }
    ));
    assert!(graph.snapshot().loop_edges.is_empty());
}

#[test]
fn essential_graph_external_edges_require_registered_distinct_endpoints() {
    let (map, kf0, kf1, _kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    let before = graph.snapshot();

    let missing = graph
        .add_loop_edge(kf0, kf1, Pose64::identity(), scalar_block(1.0))
        .expect_err("unregistered endpoint must fail before mutation");
    assert_eq!(
        missing,
        EssentialGraphError::KeyframeNotFound { keyframe_id: kf1 }
    );
    assert_eq!(graph.snapshot().order, before.order);
    assert!(graph.snapshot().loop_edges.is_empty());

    add_graph_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    let self_edge = graph
        .add_odometry_edge(kf0, kf0, Pose64::identity(), scalar_block(1.0))
        .expect_err("self edge must fail before mutation");
    assert!(matches!(
        self_edge,
        EssentialGraphError::EdgeConstruction {
            kind: EssentialEdgeKind::Odometry,
            source: EssentialEdgeError::SelfEdge { keyframe_id },
        } if keyframe_id == kf0
    ));
    assert!(graph.snapshot().loop_edges.is_empty());
    assert!(graph.snapshot().odometry_edges.is_empty());

    graph
        .add_odometry_edge(kf0, kf1, Pose64::identity(), scalar_block(1.0))
        .expect("registered distinct odometry endpoints");
    assert_eq!(graph.snapshot().odometry_edges.len(), 1);
    assert_eq!(
        graph.snapshot().odometry_edges[0].kind(),
        EssentialEdgeKind::Odometry
    );
}

#[test]
fn essential_graph_remove_keyframe_reparents_children() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    add_graph_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    add_graph_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);
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
            .all(|edge| edge.endpoint_a() != kf1 && edge.endpoint_b() != kf1)
    );
    let input = graph.pose_graph_input().expect("valid graph input");
    assert!(input.keyframe_ids.iter().all(|&id| id != kf1));
}

#[test]
fn essential_graph_remove_preflights_reparenting_before_mutation() {
    let (map, kf0, kf1, kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    add_graph_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    add_graph_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);
    let before = graph.snapshot();
    let mut incomplete_map = map.clone();
    incomplete_map
        .remove_keyframe(kf0)
        .expect("remove prospective replacement parent from test map");

    let error = graph
        .remove_keyframe(kf1, &incomplete_map)
        .expect_err("missing replacement-parent pose must fail before graph mutation");

    assert_eq!(
        error,
        EssentialGraphError::KeyframeNotFound { keyframe_id: kf0 }
    );
    let after = graph.snapshot();
    assert_eq!(after.parent, before.parent);
    assert_eq!(after.order, before.order);
    assert_eq!(after.spanning_edges.len(), before.spanning_edges.len());
    assert_eq!(
        after.strong_covis_edges.len(),
        before.strong_covis_edges.len()
    );
    assert_eq!(after.loop_edges.len(), before.loop_edges.len());
}

#[test]
fn essential_graph_remove_keyframe_rejects_root() {
    let (map, kf0, kf1, _kf2) = make_map_for_essential_graph();
    let mut graph = EssentialGraph::new(2);
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    add_graph_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);

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
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);

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
    add_graph_keyframe(&mut graph, kf0, map.covisibility().neighbors(kf0), &map);
    add_graph_keyframe(&mut graph, kf1, map.covisibility().neighbors(kf1), &map);
    add_graph_keyframe(&mut graph, kf2, map.covisibility().neighbors(kf2), &map);
    graph
        .add_loop_edge(kf2, kf0, Pose64::identity(), scalar_block(1.0))
        .expect("registered loop endpoints");
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
            .all(|edge| edge.endpoint_a() != kf2 && edge.endpoint_b() != kf2)
    );
    assert!(
        snapshot
            .spanning_edges
            .iter()
            .all(|edge| edge.endpoint_a() != kf2 && edge.endpoint_b() != kf2)
    );
}

#[test]
fn essential_graph_random_remove_preserves_connectivity_invariants() {
    let (map, ids) = make_chain_keyframes(12);
    let root = ids[0];
    let mut graph = EssentialGraph::new(100);
    for (idx, &id) in ids.iter().enumerate() {
        if idx == 0 {
            add_graph_keyframe(&mut graph, id, None, &map);
        } else {
            let mut covis = HashMap::new();
            covis.insert(ids[idx - 1], NonZeroU32::new(10).expect("non-zero"));
            add_graph_keyframe(&mut graph, id, Some(&covis), &map);
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
            assert!(alive_set.contains(&edge.endpoint_a()));
            assert!(alive_set.contains(&edge.endpoint_b()));
        }

        let input = graph.pose_graph_input().expect("valid graph input");
        let input_set: HashSet<_> = input.keyframe_ids.iter().copied().collect();
        assert!(input_set.is_subset(&alive_set));
        for edge in &input.edges {
            assert!(edge.from() < input.keyframe_ids.len());
            assert!(edge.to() < input.keyframe_ids.len());
        }
    }
}
