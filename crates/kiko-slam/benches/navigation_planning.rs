//! Reproducible, dependency-free benchmark for deterministic global planning.
//!
//! Run the full sample set with
//! `cargo bench -p kiko-slam --bench navigation_planning`.
//! Set `KIKO_NAVIGATION_BENCH_SHORT=1` for a compile-and-behavior smoke run.

use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use kiko_slam::dense::occupancy::{
    DepthCameraModel, DepthRangeMeters, DepthToTrackingCamera, HeightRangeMeters, OccupancyCell,
    OccupancyConfig, OccupancyEvidenceModel, OccupancyGridGeometry, OccupancyGridSnapshot,
    OccupancyMapper, WorldToOccupancy,
};
use kiko_slam::map::{ImageSize, SlamMap};
use kiko_slam::navigation::{
    GlobalPlanError, GlobalPlanner, GlobalPlannerConfig, MapPoint, PlanStart, PointGoal,
    UnknownSpacePolicy,
};
use kiko_slam::{
    DepthImage, FrameDimensions, FrameId, Keypoint, PinholeIntrinsics, Timestamp, WorldToCamera,
};

const GRID_WIDTH: u32 = 400;
const GRID_HEIGHT: u32 = 400;
const GRID_RESOLUTION_M: f64 = 0.05;
const GRID_LOWER_BOUND_M: [f64; 2] = [-10.0, -10.0];
const DEFAULT_SAMPLES: usize = 7;
const DEFAULT_WARMUP_ROUNDS: usize = 2;
const DEFAULT_CONSTRUCTION_ITERATIONS: usize = 5;
const DEFAULT_PLANNING_ITERATIONS: usize = 100;
const SHORT_SAMPLES: usize = 3;
const SHORT_PLANNING_ITERATIONS: usize = 100;

struct Fixtures {
    unknown: OccupancyGridSnapshot,
    sparse: OccupancyGridSnapshot,
    start: MapPoint,
    goal: MapPoint,
}

#[derive(Clone, Copy)]
enum ExpectedBehavior {
    Reachable,
    StartBlocked,
}

struct Scenario<'a> {
    name: &'static str,
    snapshot: &'a OccupancyGridSnapshot,
    config: GlobalPlannerConfig,
    expected: ExpectedBehavior,
}

#[derive(Clone, Copy)]
enum BehaviorProbe {
    Path {
        point_count: usize,
        point_digest: u64,
    },
    StartBlocked,
}

#[derive(Clone, Copy)]
struct CellCounts {
    unknown: usize,
    free: usize,
    occupied: usize,
}

fn build_fixtures() -> Fixtures {
    let dimensions = FrameDimensions::try_new(1, 1).expect("benchmark dimensions");
    let camera = DepthCameraModel::new(
        PinholeIntrinsics::try_new(1.0, 1.0, 0.0, 0.0).expect("benchmark intrinsics"),
        dimensions,
        DepthToTrackingCamera::identity(),
    );
    let geometry = OccupancyGridGeometry::try_new(
        GRID_RESOLUTION_M,
        GRID_LOWER_BOUND_M,
        GRID_WIDTH,
        GRID_HEIGHT,
        usize::try_from(GRID_WIDTH * GRID_HEIGHT).expect("benchmark cell count"),
    )
    .expect("benchmark geometry");
    let config = OccupancyConfig::try_new(
        geometry,
        WorldToOccupancy::level_optical_world(0.6).expect("benchmark occupancy frame"),
        camera,
        HeightRangeMeters::try_new(0.05, 1.8).expect("benchmark height range"),
        DepthRangeMeters::try_new(0.2, 10.0).expect("benchmark depth range"),
        1,
        OccupancyEvidenceModel::try_new(-1, 3, -1, 1).expect("benchmark evidence"),
        1,
    )
    .expect("benchmark occupancy config");

    let mut map = SlamMap::new();
    let keyframe_id = map
        .add_keyframe(
            FrameId::new(1),
            Timestamp::from_nanos(1),
            WorldToCamera::identity(),
            ImageSize::try_new(1, 1).expect("benchmark image size"),
            vec![Keypoint { x: 0.0, y: 0.0 }],
        )
        .expect("benchmark keyframe");
    let map_instance_id = map.snapshot().instance_id();
    let mut mapper = OccupancyMapper::try_new(config).expect("benchmark mapper");
    mapper
        .reset_to_map(map_instance_id)
        .expect("benchmark map reset");
    let unknown = mapper.snapshot().expect("unknown benchmark snapshot");
    let depth = DepthImage::new(FrameId::new(1), Timestamp::from_nanos(1), 1, 1, vec![2.0])
        .expect("benchmark depth");
    let integration = mapper
        .integrate(keyframe_id, WorldToCamera::identity(), &depth)
        .expect("benchmark sparse-ray integration");
    assert!(integration.free_cells_touched > 0);
    assert!(integration.occupied_cells_touched > 0);
    let sparse = mapper.snapshot().expect("sparse benchmark snapshot");

    let unknown_counts = cell_counts(&unknown);
    assert_eq!(unknown_counts.unknown, geometry.cell_count());
    assert_eq!(unknown_counts.free, 0);
    assert_eq!(unknown_counts.occupied, 0);
    let sparse_counts = cell_counts(&sparse);
    assert!(sparse_counts.unknown > sparse_counts.free);
    assert!(sparse_counts.free > 0);
    assert!(sparse_counts.occupied > 0);
    assert_eq!(
        sparse_counts.unknown + sparse_counts.free + sparse_counts.occupied,
        geometry.cell_count()
    );

    Fixtures {
        unknown,
        sparse,
        start: MapPoint::try_new(0.0, -8.0).expect("benchmark start"),
        goal: MapPoint::try_new(0.0, 8.0).expect("benchmark goal"),
    }
}

fn cell_counts(snapshot: &OccupancyGridSnapshot) -> CellCounts {
    let mut counts = CellCounts {
        unknown: 0,
        free: 0,
        occupied: 0,
    };
    for row in 0..snapshot.height() {
        for column in 0..snapshot.width() {
            match snapshot.cell(column, row).expect("benchmark snapshot cell") {
                OccupancyCell::Unknown => counts.unknown += 1,
                OccupancyCell::Free => counts.free += 1,
                OccupancyCell::Occupied => counts.occupied += 1,
            }
        }
    }
    counts
}

fn path_digest(points: &[MapPoint]) -> u64 {
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    for point in points {
        for byte in point
            .x_m()
            .to_bits()
            .to_le_bytes()
            .into_iter()
            .chain(point.y_m().to_bits().to_le_bytes())
        {
            digest ^= u64::from(byte);
            digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    digest
}

fn probe_behavior(
    scenario: &Scenario<'_>,
    planner: &GlobalPlanner,
    start: PlanStart,
    goal: PointGoal,
    start_point: MapPoint,
    goal_point: MapPoint,
) -> BehaviorProbe {
    assert_eq!(planner.map_instance_id(), start.map_instance_id());
    assert_eq!(planner.map_revision(), start.map_revision());
    assert_eq!(planner.safety_profile(), scenario.config);
    match scenario.expected {
        ExpectedBehavior::Reachable => {
            let first = planner.plan(start, goal).expect("benchmark path probe");
            let second = planner
                .plan(start, goal)
                .expect("deterministic benchmark path probe");
            assert_eq!(first, second);
            assert_eq!(first.points().first(), Some(&start_point));
            assert_eq!(first.points().last(), Some(&goal_point));
            assert!(first.points().len() >= 2);
            assert!(first.is_current_for(scenario.snapshot));
            assert_eq!(first.safety_profile(), scenario.config);
            for point in first.points() {
                assert!(
                    scenario
                        .snapshot
                        .geometry()
                        .point_index(point.as_array())
                        .is_some()
                );
            }
            BehaviorProbe::Path {
                point_count: first.points().len(),
                point_digest: path_digest(first.points()),
            }
        }
        ExpectedBehavior::StartBlocked => {
            for _ in 0..2 {
                assert_eq!(
                    planner.plan(start, goal),
                    Err(GlobalPlanError::StartBlocked { point: start_point })
                );
            }
            BehaviorProbe::StartBlocked
        }
    }
}

fn probe_sparse_obstacle_changes_route(fixtures: &Fixtures) {
    let config = GlobalPlannerConfig::try_new(0.10, UnknownSpacePolicy::Traversable)
        .expect("sparse-route comparison config");
    let unknown_planner =
        GlobalPlanner::try_new(&fixtures.unknown, config).expect("open comparison planner");
    let sparse_planner =
        GlobalPlanner::try_new(&fixtures.sparse, config).expect("sparse comparison planner");
    let unknown_path = unknown_planner
        .plan(
            PlanStart::for_snapshot(fixtures.start, &fixtures.unknown)
                .expect("open comparison start"),
            PointGoal::for_snapshot(fixtures.goal, &fixtures.unknown)
                .expect("open comparison goal"),
        )
        .expect("open comparison path");
    let sparse_path = sparse_planner
        .plan(
            PlanStart::for_snapshot(fixtures.start, &fixtures.sparse)
                .expect("sparse comparison start"),
            PointGoal::for_snapshot(fixtures.goal, &fixtures.sparse)
                .expect("sparse comparison goal"),
        )
        .expect("sparse comparison path");
    assert_ne!(
        unknown_path.points(),
        sparse_path.points(),
        "sparse benchmark obstacle must alter the selected route"
    );
}

fn run_constructions(
    scenario: &Scenario<'_>,
    expected_map_revision: u64,
    iterations: usize,
) -> usize {
    let mut checksum = 0_usize;
    for _ in 0..iterations {
        let planner = GlobalPlanner::try_new(black_box(scenario.snapshot), scenario.config)
            .expect("benchmark planner construction");
        assert_eq!(planner.map_revision(), expected_map_revision);
        assert_eq!(planner.safety_profile(), scenario.config);
        checksum = checksum.wrapping_add(planner.map_revision() as usize);
        black_box(planner);
    }
    checksum
}

fn run_plans(
    planner: &GlobalPlanner,
    start: PlanStart,
    goal: PointGoal,
    probe: BehaviorProbe,
    start_point: MapPoint,
    iterations: usize,
) -> usize {
    let mut output_points = 0_usize;
    for _ in 0..iterations {
        match probe {
            BehaviorProbe::Path { point_count, .. } => {
                let path = planner
                    .plan(black_box(start), black_box(goal))
                    .expect("benchmark A* plan");
                assert_eq!(path.points().len(), point_count);
                output_points = output_points
                    .checked_add(path.points().len())
                    .expect("benchmark path-point count");
                black_box(path);
            }
            BehaviorProbe::StartBlocked => {
                assert_eq!(
                    planner.plan(black_box(start), black_box(goal)),
                    Err(GlobalPlanError::StartBlocked { point: start_point })
                );
            }
        }
    }
    output_points
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
    let Some(value) = std::env::var_os("KIKO_NAVIGATION_BENCH_SHORT") else {
        return false;
    };
    match value
        .to_str()
        .expect("KIKO_NAVIGATION_BENCH_SHORT must be valid UTF-8")
    {
        "1" | "true" => true,
        "0" | "false" => false,
        _ => panic!("KIKO_NAVIGATION_BENCH_SHORT must be one of: 0, 1, false, true"),
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

fn cpu_model() -> String {
    #[cfg(target_os = "macos")]
    {
        let sysctl = if std::path::Path::new("/usr/sbin/sysctl").is_file() {
            std::ffi::OsStr::new("/usr/sbin/sysctl")
        } else {
            std::ffi::OsStr::new("sysctl")
        };
        let brand = command_version(sysctl, &["-n", "machdep.cpu.brand_string"]);
        if brand != "unavailable" {
            return brand;
        }
        command_version(sysctl, &["-n", "hw.model"])
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for key in ["model name", "Hardware", "Processor"] {
                if let Some(value) = cpuinfo.lines().find_map(|line| {
                    let (candidate, value) = line.split_once(':')?;
                    (candidate.trim() == key).then(|| value.trim())
                }) && !value.is_empty()
                {
                    return value.to_owned();
                }
            }
        }
        let lscpu = command_version("lscpu".as_ref(), &[]);
        if lscpu != "unavailable"
            && let Some(value) = lscpu.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key.trim() == "Model name").then(|| value.trim())
            })
            && !value.is_empty()
        {
            return value.to_owned();
        }
        "unavailable".to_owned()
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "unavailable".to_owned()
    }
}

fn environment_value(name: &str) -> String {
    std::env::var_os(name)
        .map(|value| value.to_string_lossy().into_owned())
        .filter(|value| !value.is_empty())
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
    let construction_iterations = if short {
        1
    } else {
        DEFAULT_CONSTRUCTION_ITERATIONS
    };
    let planning_iterations = if short {
        SHORT_PLANNING_ITERATIONS
    } else {
        DEFAULT_PLANNING_ITERATIONS
    };
    let fixtures = build_fixtures();
    probe_sparse_obstacle_changes_route(&fixtures);
    let scenarios = [
        Scenario {
            name: "open_unknown_point",
            snapshot: &fixtures.unknown,
            config: GlobalPlannerConfig::try_new(0.0, UnknownSpacePolicy::Traversable)
                .expect("explicit point-robot scenario"),
            expected: ExpectedBehavior::Reachable,
        },
        Scenario {
            name: "open_unknown_clearance",
            snapshot: &fixtures.unknown,
            config: GlobalPlannerConfig::try_new(0.35, UnknownSpacePolicy::Traversable)
                .expect("open clearance scenario"),
            expected: ExpectedBehavior::Reachable,
        },
        Scenario {
            name: "sparse_ray_clearance_0_10",
            snapshot: &fixtures.sparse,
            config: GlobalPlannerConfig::try_new(0.10, UnknownSpacePolicy::Traversable)
                .expect("sparse clearance scenario"),
            expected: ExpectedBehavior::Reachable,
        },
        Scenario {
            name: "sparse_ray_clearance_0_35",
            snapshot: &fixtures.sparse,
            config: GlobalPlannerConfig::try_new(0.35, UnknownSpacePolicy::Traversable)
                .expect("sparse clearance scenario"),
            expected: ExpectedBehavior::Reachable,
        },
        Scenario {
            name: "dense_unknown_clearance_0_10",
            snapshot: &fixtures.unknown,
            config: GlobalPlannerConfig::try_new(0.10, UnknownSpacePolicy::Blocked)
                .expect("dense unknown scenario"),
            expected: ExpectedBehavior::StartBlocked,
        },
        Scenario {
            name: "dense_unknown_clearance_0_35",
            snapshot: &fixtures.unknown,
            config: GlobalPlannerConfig::try_new(0.35, UnknownSpacePolicy::Blocked)
                .expect("dense unknown scenario"),
            expected: ExpectedBehavior::StartBlocked,
        },
    ];

    let git_commit = command_version("git".as_ref(), &["rev-parse", "HEAD"]);
    let git_worktree = git_worktree_state();
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let rustc_version = command_version(&rustc, &["--version"]);
    let cargo_version = command_version(&cargo, &["--version"]);
    let cpu_model = cpu_model();
    let logical_parallelism = std::thread::available_parallelism()
        .map(|value| value.get().to_string())
        .unwrap_or_else(|_| "unavailable".to_owned());
    let compilation_profile = if cfg!(debug_assertions) {
        "debug-assertions"
    } else {
        "optimized"
    };
    let rustflags = environment_value("RUSTFLAGS");
    let cargo_encoded_rustflags = environment_value("CARGO_ENCODED_RUSTFLAGS");
    let mut cargo_build_target = environment_value("CARGO_BUILD_TARGET");
    if cargo_build_target == "unavailable" {
        cargo_build_target = environment_value("TARGET");
    }
    println!(
        "navigation planning benchmark metadata: git_commit={git_commit} git_worktree={git_worktree} rustc={rustc_version:?} cargo={cargo_version:?} os={} arch={} cpu_model={cpu_model:?} logical_parallelism={} compilation_profile={} debug_assertions={} rustflags={rustflags:?} cargo_encoded_rustflags={cargo_encoded_rustflags:?} cargo_build_target={cargo_build_target:?} short={short} grid={}x{} resolution_m={} samples={} warmup_rounds={} construction_iterations_per_sample={} planning_iterations_per_sample={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        logical_parallelism,
        compilation_profile,
        cfg!(debug_assertions),
        GRID_WIDTH,
        GRID_HEIGHT,
        GRID_RESOLUTION_M,
        samples,
        warmup_rounds,
        construction_iterations,
        planning_iterations,
    );

    for scenario in &scenarios {
        let planner = GlobalPlanner::try_new(scenario.snapshot, scenario.config)
            .expect("benchmark behavior-probe planner");
        let start = PlanStart::for_snapshot(fixtures.start, scenario.snapshot)
            .expect("benchmark bound start");
        let goal = PointGoal::for_snapshot(fixtures.goal, scenario.snapshot)
            .expect("benchmark bound goal");
        let probe = probe_behavior(
            scenario,
            &planner,
            start,
            goal,
            fixtures.start,
            fixtures.goal,
        );

        for _ in 0..warmup_rounds {
            black_box(run_constructions(
                scenario,
                planner.map_revision(),
                construction_iterations,
            ));
            black_box(run_plans(
                &planner,
                start,
                goal,
                probe,
                fixtures.start,
                planning_iterations,
            ));
        }
        let construction_median_ns = median_nanos(
            samples,
            || run_constructions(scenario, planner.map_revision(), construction_iterations),
            construction_iterations,
        );
        let planning_median_ns = median_nanos(
            samples,
            || {
                run_plans(
                    &planner,
                    start,
                    goal,
                    probe,
                    fixtures.start,
                    planning_iterations,
                )
            },
            planning_iterations,
        );
        let counts = cell_counts(scenario.snapshot);
        let (behavior, path_points, path_digest) = match probe {
            BehaviorProbe::Path {
                point_count,
                point_digest,
            } => ("path", point_count, format!("0x{point_digest:016x}")),
            BehaviorProbe::StartBlocked => ("start_blocked", 0, "none".to_owned()),
        };
        println!(
            "navigation planning benchmark: scenario={} map_revision={} unknown_policy={:?} clearance_radius_m={} cells_unknown={} cells_free={} cells_occupied={} behavior={} path_points={} path_digest={} construction_median_ns={:.1} constructions_per_s={:.1} planning_median_ns={:.1} plans_per_s={:.1}",
            scenario.name,
            scenario.snapshot.revision(),
            scenario.config.unknown_space(),
            scenario.config.clearance_radius_m(),
            counts.unknown,
            counts.free,
            counts.occupied,
            behavior,
            path_points,
            path_digest,
            construction_median_ns,
            1.0e9 / construction_median_ns,
            planning_median_ns,
            1.0e9 / planning_median_ns,
        );
    }
}
