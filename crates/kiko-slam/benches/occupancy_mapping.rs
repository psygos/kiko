//! Reproducible, dependency-free benchmark for deterministic occupancy mapping.
//!
//! Run the full sample set with
//! `cargo bench -p kiko-slam --bench occupancy_mapping`.
//! Set `KIKO_OCCUPANCY_BENCH_SHORT=1` for a compile-and-behavior smoke run.

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use kiko_slam::dense::occupancy::{
    DepthCameraModel, DepthRangeMeters, DepthToTrackingCamera, HeightRangeMeters, OccupancyConfig,
    OccupancyEvidenceModel, OccupancyGridGeometry, OccupancyMapper, OccupancyRemoveOutcome,
    WorldToOccupancy,
};
use kiko_slam::map::{ImageSize, KeyframeId, SlamMap};
use kiko_slam::{
    DepthImage, FrameDimensions, FrameId, Keypoint, PinholeIntrinsics, Timestamp, WorldToCamera,
};

const DEPTH_WIDTH: u32 = 160;
const DEPTH_HEIGHT: u32 = 120;
const SAMPLE_BLOCK_PX: u32 = 4;
const GRID_WIDTH: u32 = 400;
const GRID_HEIGHT: u32 = 400;
const DEFAULT_SAMPLES: usize = 7;
const DEFAULT_WARMUP_ROUNDS: usize = 2;
const DEFAULT_INTEGRATION_ITERATIONS: usize = 40;
const DEFAULT_SNAPSHOT_ITERATIONS: usize = 200;
const SHORT_SAMPLES: usize = 3;

struct Fixture {
    mapper: OccupancyMapper,
    keyframe_id: KeyframeId,
    depth: DepthImage,
    sampled_blocks: usize,
}

fn benchmark_keyframe() -> KeyframeId {
    let mut map = SlamMap::new();
    map.add_keyframe(
        FrameId::new(1),
        Timestamp::from_nanos(1),
        WorldToCamera::identity(),
        ImageSize::try_new(DEPTH_WIDTH, DEPTH_HEIGHT).expect("benchmark image size"),
        vec![Keypoint { x: 0.0, y: 0.0 }],
    )
    .expect("benchmark keyframe")
}

fn build_fixture() -> Fixture {
    let dimensions =
        FrameDimensions::try_new(DEPTH_WIDTH, DEPTH_HEIGHT).expect("benchmark dimensions");
    let camera = DepthCameraModel::new(
        PinholeIntrinsics::try_new(
            120.0,
            120.0,
            (DEPTH_WIDTH as f32 - 1.0) * 0.5,
            (DEPTH_HEIGHT as f32 - 1.0) * 0.5,
        )
        .expect("benchmark intrinsics"),
        dimensions,
        DepthToTrackingCamera::identity(),
    );
    let config = OccupancyConfig::try_new(
        OccupancyGridGeometry::try_new(
            0.05,
            [-10.0, -5.0],
            GRID_WIDTH,
            GRID_HEIGHT,
            usize::try_from(GRID_WIDTH * GRID_HEIGHT).expect("benchmark cell count"),
        )
        .expect("benchmark geometry"),
        WorldToOccupancy::level_optical_world(0.6).expect("benchmark occupancy frame"),
        camera,
        HeightRangeMeters::try_new(0.05, 1.8).expect("benchmark height range"),
        DepthRangeMeters::try_new(0.2, 10.0).expect("benchmark depth range"),
        SAMPLE_BLOCK_PX,
        OccupancyEvidenceModel::try_new(-1, 3, -2, 2).expect("benchmark evidence"),
        1,
    )
    .expect("benchmark occupancy config");
    let depth_values = (0..dimensions.area())
        .map(|index| {
            if index % 97 == 0 {
                0.0
            } else {
                2.0 + (index % 301) as f32 * 0.01
            }
        })
        .collect();
    let depth = DepthImage::new(
        FrameId::new(1),
        Timestamp::from_nanos(1),
        DEPTH_WIDTH,
        DEPTH_HEIGHT,
        depth_values,
    )
    .expect("benchmark depth");
    let mut mapper = OccupancyMapper::try_new(config).expect("benchmark mapper");
    let keyframe_id = benchmark_keyframe();
    let sampled_blocks = mapper
        .integrate(keyframe_id, WorldToCamera::identity(), &depth)
        .expect("benchmark behavior probe")
        .sampled_blocks;
    assert_eq!(
        mapper.remove(keyframe_id).expect("benchmark probe removal"),
        OccupancyRemoveOutcome::Removed {
            retained_keyframes: 0,
            revision: 2,
        }
    );
    assert!(
        sampled_blocks > 0,
        "benchmark must exercise valid depth rays"
    );

    Fixture {
        mapper,
        keyframe_id,
        depth,
        sampled_blocks,
    }
}

fn run_integration_cycles(fixture: &mut Fixture, iterations: usize) -> usize {
    let mut touched_cells = 0_usize;
    for _ in 0..iterations {
        let outcome = fixture
            .mapper
            .integrate(
                black_box(fixture.keyframe_id),
                black_box(WorldToCamera::identity()),
                black_box(&fixture.depth),
            )
            .expect("benchmark integration");
        assert_eq!(outcome.sampled_blocks, fixture.sampled_blocks);
        touched_cells = touched_cells
            .checked_add(outcome.free_cells_touched)
            .and_then(|value| value.checked_add(outcome.occupied_cells_touched))
            .expect("benchmark touched-cell count");
        assert!(matches!(
            fixture
                .mapper
                .remove(fixture.keyframe_id)
                .expect("benchmark removal"),
            OccupancyRemoveOutcome::Removed {
                retained_keyframes: 0,
                ..
            }
        ));
    }
    touched_cells
}

fn run_snapshots(fixture: &Fixture, iterations: usize) -> usize {
    let mut output_cells = 0_usize;
    for _ in 0..iterations {
        let snapshot = fixture.mapper.snapshot().expect("benchmark snapshot");
        assert_eq!(
            snapshot.class_ids().len(),
            (GRID_WIDTH * GRID_HEIGHT) as usize
        );
        output_cells = output_cells
            .checked_add(snapshot.class_ids().len())
            .expect("benchmark output-cell count");
        black_box(snapshot.class_ids());
        black_box(snapshot);
    }
    output_cells
}

fn median_nanos(samples: usize, mut operation: impl FnMut() -> usize, iterations: usize) -> f64 {
    let mut observations = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        black_box(operation());
        observations.push(start.elapsed().as_secs_f64() * 1.0e9 / iterations as f64);
    }
    observations.sort_by(f64::total_cmp);
    let median = observations[samples / 2];
    assert!(
        median.is_finite() && median > 0.0,
        "benchmark timer must produce a positive finite duration"
    );
    median
}

fn short_mode() -> bool {
    let Some(value) = std::env::var_os("KIKO_OCCUPANCY_BENCH_SHORT") else {
        return false;
    };
    match value
        .to_str()
        .expect("KIKO_OCCUPANCY_BENCH_SHORT must be valid UTF-8")
    {
        "1" | "true" => true,
        "0" | "false" => false,
        _ => panic!("KIKO_OCCUPANCY_BENCH_SHORT must be one of: 0, 1, false, true"),
    }
}

fn command_version(program: &std::ffi::OsStr, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.trim().to_owned())
        .filter(|output| !output.is_empty())
        .unwrap_or_else(|| "unavailable".to_owned())
}

fn git_worktree_state() -> &'static str {
    match Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
    {
        Ok(output) if output.status.success() && output.stdout.is_empty() => "clean",
        Ok(output) if output.status.success() => "dirty",
        _ => "unavailable",
    }
}

fn main() {
    let short = short_mode();
    let samples = if short {
        SHORT_SAMPLES
    } else {
        DEFAULT_SAMPLES
    };
    let warmup_rounds = if short { 1 } else { DEFAULT_WARMUP_ROUNDS };
    let integration_iterations = if short {
        1
    } else {
        DEFAULT_INTEGRATION_ITERATIONS
    };
    let snapshot_iterations = if short {
        1
    } else {
        DEFAULT_SNAPSHOT_ITERATIONS
    };
    let mut fixture = build_fixture();

    for _ in 0..warmup_rounds {
        black_box(run_integration_cycles(&mut fixture, integration_iterations));
        black_box(run_snapshots(&fixture, snapshot_iterations));
    }

    let integration_median_ns = median_nanos(
        samples,
        || run_integration_cycles(&mut fixture, integration_iterations),
        integration_iterations,
    );
    let snapshot_median_ns = median_nanos(
        samples,
        || run_snapshots(&fixture, snapshot_iterations),
        snapshot_iterations,
    );
    let integration_remove_cycles_per_second = 1.0e9 / integration_median_ns;
    let snapshot_output_cells_per_second =
        (GRID_WIDTH * GRID_HEIGHT) as f64 * 1.0e9 / snapshot_median_ns;
    let git_commit = command_version("git".as_ref(), &["rev-parse", "HEAD"]);
    let git_worktree = git_worktree_state();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let rustc_version = command_version(&rustc, &["--version"]);

    println!(
        "occupancy benchmark: git_commit={git_commit} git_worktree={git_worktree} rustc={rustc_version:?} os={} arch={} short={short} depth={}x{} sample_block_px={} sampled_input_blocks_per_integrate={} grid={}x{} resolution_m=0.05 samples={} warmup_rounds={} integration_remove_cycles_per_sample={} integration_remove_cycle_median_ns={:.1} integration_remove_cycles_per_s={:.1} snapshot_calls_per_sample={} snapshot_call_median_ns={:.1} snapshot_output_cells_per_s={:.1}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        DEPTH_WIDTH,
        DEPTH_HEIGHT,
        SAMPLE_BLOCK_PX,
        fixture.sampled_blocks,
        GRID_WIDTH,
        GRID_HEIGHT,
        samples,
        warmup_rounds,
        integration_iterations,
        integration_median_ns,
        integration_remove_cycles_per_second,
        snapshot_iterations,
        snapshot_median_ns,
        snapshot_output_cells_per_second,
    );
}
