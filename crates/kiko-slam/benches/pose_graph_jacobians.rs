use std::hint::black_box;
use std::time::Instant;

use kiko_slam::Pose64;
use kiko_slam::pose_graph::{PoseGraphEdge, compute_edge_jacobians};

const SAMPLE_COUNT: usize = 7;

fn identity_information() -> [[f64; 6]; 6] {
    let mut information = [[0.0; 6]; 6];
    for (index, row) in information.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    information
}

fn run_iterations(
    edge: &PoseGraphEdge,
    poses: &[Pose64],
    iterations: usize,
    clone_full_slice: bool,
) {
    for _ in 0..iterations {
        let owned;
        let input = if clone_full_slice {
            owned = black_box(poses).to_vec();
            black_box(owned.as_slice())
        } else {
            black_box(poses)
        };
        black_box(compute_edge_jacobians(black_box(edge), input).expect("benchmark Jacobians"));
    }
}

fn median_nanos_per_call(
    edge: &PoseGraphEdge,
    poses: &[Pose64],
    iterations: usize,
    clone_full_slice: bool,
) -> f64 {
    run_iterations(edge, poses, 1_000, clone_full_slice);
    let mut samples = [0.0_f64; SAMPLE_COUNT];
    for sample in &mut samples {
        let start = Instant::now();
        run_iterations(edge, poses, iterations, clone_full_slice);
        *sample = start.elapsed().as_nanos() as f64 / iterations as f64;
    }
    samples.sort_by(f64::total_cmp);
    samples[SAMPLE_COUNT / 2]
}

fn benchmark_case(pose_count: usize, iterations: usize) {
    let poses = vec![Pose64::identity(); pose_count];
    let edge = PoseGraphEdge::try_new(
        0,
        pose_count - 1,
        Pose64::identity(),
        identity_information(),
    )
    .expect("benchmark edge");

    let cloned = median_nanos_per_call(&edge, &poses, iterations, true);
    let endpoints = median_nanos_per_call(&edge, &poses, iterations, false);
    let speedup = cloned / endpoints;
    println!(
        "pose_count={pose_count} iterations={iterations} samples={SAMPLE_COUNT} cloned_ns={cloned:.1} endpoint_ns={endpoints:.1} speedup={speedup:.3}x"
    );
}

fn main() {
    benchmark_case(2, 100_000);
    benchmark_case(64, 50_000);
    benchmark_case(1_024, 10_000);
}
