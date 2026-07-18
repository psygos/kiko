//! Reproducible, dependency-free host benchmark for shadow-navigation admission.
//!
//! Run the representative sample set with
//! `cargo bench -p kiko-slam --bench navigation_shadow`.
//! Set `KIKO_NAVIGATION_SHADOW_BENCH_SHORT=1` for a short behavior and timing run.
//!
//! This executable uses only public production APIs. Those APIs deliberately do
//! not let an external benchmark forge `TimeAlignedOdomPose` or
//! `LocalCostmapProvenance`: both require tracker-produced localization followed
//! by `PlanarOdometry`. The synthetic fixture therefore proves and measures the
//! exact ready-but-unproven-depth fail-closed path. It does not claim to time a
//! successful collision-checked MPC solve. Successful solver behavior and final
//! revalidation remain covered by focused in-crate tests, where real typed
//! provenance can be established without weakening the public boundary.

use std::ffi::OsStr;
use std::hint::black_box;
use std::process::Command;
use std::time::{Duration, Instant};

use kiko_slam::dense::occupancy::{
    DepthCameraModel, DepthRangeMeters, DepthToTrackingCamera, HeightRangeMeters, OccupancyConfig,
    OccupancyEvidenceModel, OccupancyGridGeometry, OccupancyMapper, WorldToOccupancy,
};
use kiko_slam::map::SlamMap;
use kiko_slam::navigation::mpc::{
    CollisionProvenanceError, HostMonotonicClock, MPC_CONFIG_V1, MpcConfigV1, MpcConfigV1Dto,
    MpcSolver, NavigationEpochV1, ODOM_MOTION_STATE_V1, OdomMotionStateV1, OdomMotionStateV1Dto,
    OdomPoseV1, PLANT_MODEL_V1, PlantEvidenceV1Dto, PlantModelV1, PlantModelV1Dto,
    PlantValidityEnvelopeV1Dto, WheelPlantV1Dto,
};
use kiko_slam::navigation::{
    FORWARD_MOST_NEAREST_SEGMENT_V1, GlobalPath, GlobalPlanner, GlobalPlannerConfig, LocalCostmap,
    LocalCostmapConfig, LocalCostmapFreshness, MapPoint, MapToOdom, OdomSegmentId,
    PATH_REFERENCE_CONFIG_V1, PathReferenceBuilderV1, PathReferenceConfigV1,
    PathReferenceConfigV1Dto, PlanStart, PointGoal, SafetyDecisionOutcome, SafetyReadyTick,
    SafetyStopCause, SafetyTickInput, ShadowCommandConfig, ShadowCommandConfigDto,
    ShadowCommandDisposition, ShadowSafetySupervisor, SolverBudgetNs, TrackingCameraToBase,
    UnknownSpacePolicy,
};
use kiko_slam::{
    DepthImage, DepthObservation, DeviceSessionId, FrameDimensions, FrameId,
    HostMonotonicTimestamp, PinholeIntrinsics, Pose, Timestamp,
};

const SYNTHETIC_EVIDENCE_LABEL: &str = "synthetic-public-api-fail-closed-v1";
const DEVICE_SESSION_ID: u64 = 73;
const ODOM_SEGMENT_ID: u64 = 41;
const DEPTH_FRAME_ID: u64 = 9_001;
const DEPTH_DEVICE_TIMESTAMP_NS: i64 = 5_000_000;
const DEPTH_HOST_ARRIVAL_NS: u64 = 10_000_000;
const FIRST_TICK_NS: u64 = 10_100_000;
const TICK_INCREMENT_NS: u64 = 1_000;
const SOLVER_BUDGET_NS: u64 = 2_000_000;
const RETAINED_SHADOW_RECORDS: usize = 64;

const HORIZON_STEPS: u16 = 8;
const INTEGRATION_SUBSTEPS: u16 = 2;
const OPTIMIZATION_ITERATIONS: u8 = 2;
const CANDIDATES_PER_WHEEL: u8 = 5;
const STEP_PERIOD_S: f64 = 0.1;

const DEFAULT_SAMPLES: usize = 9;
const DEFAULT_WARMUP_ROUNDS: usize = 3;
const DEFAULT_ITERATIONS: usize = 10_000;
const SHORT_SAMPLES: usize = 3;
const SHORT_WARMUP_ROUNDS: usize = 1;
const SHORT_ITERATIONS: usize = 500;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

struct Fixtures {
    path: GlobalPath,
    epoch: NavigationEpochV1,
    local_costmap: LocalCostmap,
    model: PlantModelV1,
    mpc_config: MpcConfigV1,
    reference_config: PathReferenceConfigV1,
    parsed_depth: DepthObservation,
}

#[derive(Clone, Copy)]
struct BehaviorProbe {
    map_instance_id: u64,
    map_revision: u64,
    path_point_count: usize,
    reference_point_count: usize,
    behavior_digest: u64,
}

struct RunState {
    reference_builder: PathReferenceBuilderV1,
    supervisor: ShadowSafetySupervisor,
    next_tick_ns: u64,
}

struct PanicClock;

impl HostMonotonicClock for PanicClock {
    fn now(&mut self) -> HostMonotonicTimestamp {
        panic!("missing collision provenance must stop before the MPC clock is queried")
    }
}

fn camera() -> DepthCameraModel {
    let dimensions = FrameDimensions::try_new(9, 5).expect("synthetic depth dimensions");
    DepthCameraModel::new(
        PinholeIntrinsics::try_new(4.0, 4.0, 4.0, 2.0).expect("synthetic depth intrinsics"),
        dimensions,
        DepthToTrackingCamera::identity(),
    )
}

fn optical_to_base() -> TrackingCameraToBase {
    TrackingCameraToBase::new(
        Pose::try_from_rt(
            [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            [-0.5, 0.0, 0.5],
        )
        .expect("synthetic calibrated optical-to-base transform"),
    )
}

fn plant_model() -> PlantModelV1 {
    PlantModelV1::parse(PlantModelV1Dto {
        schema_version: PLANT_MODEL_V1,
        model_id: "navigation-shadow-bench-plant".into(),
        model_version: 1,
        sample_period_s: STEP_PERIOD_S,
        wheelbase_m: 0.42,
        left: WheelPlantV1Dto {
            velocity_gain_mps_per_pwm_percent: 0.008,
            time_constant_s: 0.35,
        },
        right: WheelPlantV1Dto {
            velocity_gain_mps_per_pwm_percent: 0.0082,
            time_constant_s: 0.39,
        },
        validity: PlantValidityEnvelopeV1Dto {
            left_pwm_min_percent: -70,
            left_pwm_max_percent: 70,
            right_pwm_min_percent: -70,
            right_pwm_max_percent: 70,
            left_velocity_min_mps: -0.7,
            left_velocity_max_mps: 0.7,
            right_velocity_min_mps: -0.7,
            right_velocity_max_mps: 0.7,
            max_abs_yaw_rate_rad_s: 3.0,
            max_abs_lateral_velocity_mps: 0.2,
        },
        evidence: PlantEvidenceV1Dto::SyntheticFixture {
            fixture_id: SYNTHETIC_EVIDENCE_LABEL.into(),
            generator_id: "dependency-free-hand-fixture-v1".into(),
        },
    })
    .expect("synthetic typed plant model")
}

fn mpc_config() -> MpcConfigV1 {
    MpcConfigV1::parse(MpcConfigV1Dto {
        schema_version: MPC_CONFIG_V1,
        horizon_steps: HORIZON_STEPS,
        step_period_s: STEP_PERIOD_S,
        integration_substeps: INTEGRATION_SUBSTEPS,
        optimization_iterations: OPTIMIZATION_ITERATIONS,
        candidates_per_wheel: CANDIDATES_PER_WHEEL,
        max_rollout_evaluations: 10_000,
        initial_search_radius_percent: 20,
        search_radius_decay_numerator: 1,
        search_radius_decay_denominator: 2,
        left_pwm_min_percent: -60,
        left_pwm_max_percent: 60,
        right_pwm_min_percent: -60,
        right_pwm_max_percent: 60,
        left_max_slew_percent_per_step: 30,
        right_max_slew_percent_per_step: 30,
        max_integration_tube_radius_m: 0.5,
        position_cost_per_m2: 1_000.0,
        heading_cost_per_rad2: 50.0,
        forward_velocity_cost_s2_per_m2: 500.0,
        yaw_rate_cost_s2_per_rad2: 20.0,
        pwm_cost_per_percent2: 0.02,
        slew_cost_per_percent2: 0.1,
        terminal_state_cost_multiplier: 3.0,
    })
    .expect("synthetic typed MPC config")
}

fn reference_config() -> PathReferenceConfigV1 {
    PathReferenceConfigV1::parse(PathReferenceConfigV1Dto {
        schema_version: PATH_REFERENCE_CONFIG_V1,
        maximum_path_points: 128,
        minimum_segment_length_m: 0.01,
        maximum_path_length_m: 100.0,
        maximum_projection_distance_m: 2.0,
        target_forward_speed_mps: 0.35,
        goal_stop_distance_m: 0.5,
        maximum_abs_yaw_rate_rad_s: 2.0,
        nearest_segment_tie_policy: FORWARD_MOST_NEAREST_SEGMENT_V1,
    })
    .expect("synthetic typed path-reference config")
}

fn build_fixtures() -> Fixtures {
    let map_snapshot = SlamMap::new().snapshot();
    let occupancy_config = OccupancyConfig::try_new(
        OccupancyGridGeometry::try_new(0.25, [-4.0, -4.0], 32, 32, 1_024)
            .expect("synthetic global geometry"),
        WorldToOccupancy::level_optical_world(0.75).expect("synthetic occupancy frame"),
        camera(),
        HeightRangeMeters::try_new(0.05, 1.8).expect("synthetic global height range"),
        DepthRangeMeters::try_new(0.1, 8.0).expect("synthetic global depth range"),
        1,
        OccupancyEvidenceModel::try_new(-1, 3, -1, 1).expect("synthetic occupancy evidence"),
        1,
    )
    .expect("synthetic global occupancy config");
    let mut mapper = OccupancyMapper::try_new(occupancy_config).expect("synthetic mapper");
    mapper
        .reset_to_map(map_snapshot.instance_id())
        .expect("synthetic map identity");
    let occupancy = mapper.snapshot().expect("synthetic global snapshot");
    let mut planner = GlobalPlanner::try_new(
        &occupancy,
        GlobalPlannerConfig::try_new(0.2, UnknownSpacePolicy::Traversable)
            .expect("explicit synthetic global planning policy"),
    )
    .expect("synthetic global planner");
    let start_point = MapPoint::try_new(0.0, 0.0).expect("synthetic map-frame start");
    let goal_point = MapPoint::try_new(3.0, 0.0).expect("synthetic map-frame goal");
    let path = planner
        .plan(
            PlanStart::for_snapshot(start_point, &occupancy)
                .expect("revision-bound synthetic start"),
            PointGoal::for_snapshot(goal_point, &occupancy).expect("map-bound synthetic goal"),
        )
        .expect("synthetic global path");
    assert!(path.is_current_for(&occupancy));
    assert_eq!(path.map_revision(), occupancy.revision());

    let session = DeviceSessionId::try_new(DEVICE_SESSION_ID).expect("synthetic device session");
    let segment = OdomSegmentId::try_new(ODOM_SEGMENT_ID).expect("synthetic odom segment");
    let epoch = NavigationEpochV1::from_runtime(session, segment, map_snapshot, &path)
        .expect("synthetic navigation epoch");
    assert_eq!(epoch.global_plan_identity(), path.identity());

    let local_config = LocalCostmapConfig::try_new(
        OccupancyGridGeometry::try_new(0.1, [-2.0, -1.5], 40, 30, 1_200)
            .expect("synthetic local geometry"),
        camera(),
        optical_to_base(),
        HeightRangeMeters::try_new(0.05, 1.8).expect("synthetic local height range"),
        DepthRangeMeters::try_new(0.1, 8.0).expect("synthetic local depth range"),
        1,
        0.18,
        0.12,
        Duration::from_secs(1),
    )
    .expect("synthetic local costmap config");
    let local_costmap =
        LocalCostmap::try_new(local_config, session).expect("synthetic local costmap");

    let depth = DepthImage::new(
        FrameId::new(DEPTH_FRAME_ID),
        Timestamp::from_nanos(DEPTH_DEVICE_TIMESTAMP_NS),
        9,
        5,
        vec![2.0; 45],
    )
    .expect("synthetic metric depth image");
    let parsed_depth = DepthObservation::parse(
        session,
        HostMonotonicTimestamp::from_nanos(DEPTH_HOST_ARRIVAL_NS),
        depth,
    )
    .expect("single parsed synthetic depth boundary");

    Fixtures {
        path,
        epoch,
        local_costmap,
        model: plant_model(),
        mpc_config: mpc_config(),
        reference_config: reference_config(),
        parsed_depth,
    }
}

fn supervisor(fixtures: &Fixtures) -> ShadowSafetySupervisor {
    let solver = MpcSolver::new(fixtures.model, fixtures.mpc_config).expect("bounded MPC solver");
    let shadow_config = ShadowCommandConfig::parse(ShadowCommandConfigDto {
        lease_ms: 150,
        retained_records: RETAINED_SHADOW_RECORDS,
        initial_sequence: 0,
    })
    .expect("bounded shadow command config");
    ShadowSafetySupervisor::try_new(solver, shadow_config).expect("shadow safety supervisor")
}

fn run_state(fixtures: &Fixtures, first_tick_ns: u64) -> RunState {
    RunState {
        reference_builder: PathReferenceBuilderV1::new(fixtures.reference_config),
        supervisor: supervisor(fixtures),
        next_tick_ns: first_tick_ns,
    }
}

fn motion_state(fixtures: &Fixtures, tick_ns: u64) -> OdomMotionStateV1 {
    OdomMotionStateV1::parse(
        OdomMotionStateV1Dto {
            schema_version: ODOM_MOTION_STATE_V1,
            observed_at_host_ns: tick_ns,
            x_m: 0.0,
            y_m: 0.0,
            yaw_rad: 0.0,
            odom_velocity_x_mps: 0.0,
            odom_velocity_y_mps: 0.0,
            yaw_rate_rad_s: 0.0,
        },
        fixtures.epoch,
    )
    .expect("fresh synthetic odom motion state")
}

fn hash_byte(digest: &mut u64, byte: u8) {
    *digest ^= u64::from(byte);
    *digest = digest.wrapping_mul(FNV_PRIME);
}

fn hash_u64(digest: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        hash_byte(digest, byte);
    }
}

fn hash_reference(digest: &mut u64, reference: &kiko_slam::navigation::mpc::MpcReferenceV1<'_>) {
    hash_u64(digest, reference.points().len() as u64);
    for point in reference.points() {
        let pose = point.pose();
        for value in [
            pose.position().x_m(),
            pose.position().y_m(),
            pose.yaw_rad(),
            point.forward_velocity_mps(),
            point.yaw_rate_rad_s(),
        ] {
            hash_u64(digest, value.to_bits());
        }
    }
}

fn execute_iteration(fixtures: &Fixtures, state: &mut RunState) -> u64 {
    let tick_ns = state.next_tick_ns;
    state.next_tick_ns = tick_ns
        .checked_add(TICK_INCREMENT_NS)
        .expect("benchmark host timestamp domain");
    let tick = HostMonotonicTimestamp::from_nanos(tick_ns);
    let reference = state
        .reference_builder
        .build(
            fixtures.epoch,
            &fixtures.path,
            MapToOdom::try_new(0.0, 0.0, 0.0).expect("identity map-to-odom transform"),
            OdomPoseV1::try_new(0.0, 0.0, 0.0).expect("synthetic odom pose"),
            fixtures.mpc_config,
            tick,
        )
        .expect("time-parameterized synthetic reference");
    assert_eq!(reference.epoch(), fixtures.epoch);
    assert_eq!(reference.global_plan_identity(), fixtures.path.identity());
    assert_eq!(reference.created_at(), tick);

    let local_view = fixtures
        .local_costmap
        .view_at(tick)
        .expect("monotonic immutable local collision view");
    assert_eq!(local_view.freshness(), LocalCostmapFreshness::NoObservation);
    assert_eq!(local_view.provenance(), None);
    let ready = SafetyReadyTick::new(
        fixtures.epoch,
        motion_state(fixtures, tick_ns),
        &reference,
        local_view,
        SolverBudgetNs::try_new(SOLVER_BUDGET_NS).expect("nonzero solver budget"),
    );
    let decision = state
        .supervisor
        .decide(tick, SafetyTickInput::Ready(ready), &mut PanicClock)
        .expect("fail-closed shadow record");
    assert_eq!(decision.record().recorded_at(), tick);
    assert_eq!(
        decision.record().disposition(),
        ShadowCommandDisposition::FailClosedStop
    );
    assert!(decision.record().pwm().is_stop());
    assert_eq!(decision.motor_packets_sent().get(), 0);
    assert_eq!(decision.record().motor_packets_sent().get(), 0);
    assert_eq!(state.supervisor.motor_packets_sent().get(), 0);
    assert_eq!(
        state.supervisor.shadow_session().motor_packets_sent().get(),
        0
    );
    assert!(state.supervisor.last_success_trajectory().is_none());
    match decision.outcome() {
        SafetyDecisionOutcome::Stopped(stopped) => assert!(matches!(
            stopped.cause(),
            SafetyStopCause::CollisionProvenance(CollisionProvenanceError::NoObservation)
        )),
        SafetyDecisionOutcome::Controller(controller) => panic!(
            "unproven local depth must not reach final revalidation: request={} validation={:?}",
            controller.request_id(),
            controller.final_validation()
        ),
    }

    let mut digest = FNV_OFFSET;
    hash_reference(&mut digest, &reference);
    for value in [
        decision.record().pwm().left().get(),
        decision.record().pwm().right().get(),
    ] {
        hash_byte(&mut digest, value.to_le_bytes()[0]);
    }
    hash_byte(&mut digest, 1); // fail-closed STOP
    hash_byte(&mut digest, 0); // no forgeable final-validation value
    hash_byte(&mut digest, 0); // motor packets sent
    black_box(decision);
    digest
}

fn behavior_probe(fixtures: &Fixtures) -> BehaviorProbe {
    assert_eq!(
        fixtures.epoch.device_session_id().as_u64(),
        DEVICE_SESSION_ID
    );
    assert_eq!(fixtures.epoch.odom_segment_id().as_u64(), ODOM_SEGMENT_ID);
    assert_eq!(
        fixtures.epoch.global_plan_identity(),
        fixtures.path.identity()
    );
    assert_eq!(
        fixtures.path.map_instance_id(),
        fixtures.epoch.map_snapshot().instance_id()
    );
    assert_eq!(
        fixtures.path.map_revision(),
        fixtures.epoch.global_plan_identity().map_revision()
    );
    assert_eq!(
        fixtures.parsed_depth.session_id().as_u64(),
        DEVICE_SESSION_ID
    );
    assert_eq!(fixtures.parsed_depth.frame_id().as_u64(), DEPTH_FRAME_ID);
    assert_eq!(
        fixtures.parsed_depth.device_timestamp().as_nanos(),
        DEPTH_DEVICE_TIMESTAMP_NS as u64
    );
    assert_eq!(
        fixtures.parsed_depth.host_arrival().as_nanos(),
        DEPTH_HOST_ARRIVAL_NS
    );
    assert_eq!(
        fixtures.local_costmap.session_id(),
        fixtures.epoch.device_session_id()
    );
    assert_eq!(fixtures.local_costmap.provenance(), None);

    let mut first = run_state(fixtures, FIRST_TICK_NS);
    let first_digest = execute_iteration(fixtures, &mut first);
    let first_record = first
        .supervisor
        .shadow_session()
        .latest()
        .expect("first behavior record");
    assert_eq!(first.supervisor.shadow_session().retained_len(), 1);

    let mut repeated = run_state(fixtures, FIRST_TICK_NS);
    let repeated_digest = execute_iteration(fixtures, &mut repeated);
    let repeated_record = repeated
        .supervisor
        .shadow_session()
        .latest()
        .expect("repeated behavior record");
    assert_eq!(first_digest, repeated_digest);
    assert_eq!(first_record, repeated_record);

    let reference_point_count = fixtures.mpc_config.horizon_steps();
    BehaviorProbe {
        map_instance_id: fixtures.path.map_instance_id().as_u64(),
        map_revision: fixtures.path.map_revision(),
        path_point_count: fixtures.path.points().len(),
        reference_point_count,
        behavior_digest: first_digest,
    }
}

fn run_iterations(fixtures: &Fixtures, state: &mut RunState, iterations: usize) -> u64 {
    let mut digest = FNV_OFFSET;
    for _ in 0..iterations {
        hash_u64(&mut digest, execute_iteration(fixtures, state));
    }
    assert!(state.supervisor.shadow_session().retained_len() <= RETAINED_SHADOW_RECORDS);
    assert_eq!(state.supervisor.motor_packets_sent().get(), 0);
    digest
}

fn measure_samples(
    fixtures: &Fixtures,
    samples: usize,
    iterations: usize,
    first_tick_ns: u64,
) -> (f64, u64, usize) {
    let mut observations = Vec::with_capacity(samples);
    let mut stable_digest = None;
    let mut state = run_state(fixtures, first_tick_ns);
    for _ in 0..samples {
        let start = Instant::now();
        let digest = black_box(run_iterations(fixtures, &mut state, iterations));
        let elapsed_ns_per_iteration = start.elapsed().as_secs_f64() * 1.0e9 / iterations as f64;
        assert!(
            elapsed_ns_per_iteration.is_finite() && elapsed_ns_per_iteration > 0.0,
            "benchmark timer must produce a positive finite duration"
        );
        if let Some(expected) = stable_digest {
            assert_eq!(
                digest, expected,
                "timed behavior digest changed between samples"
            );
        } else {
            stable_digest = Some(digest);
        }
        observations.push(elapsed_ns_per_iteration);
    }
    observations.sort_by(f64::total_cmp);
    (
        observations[samples / 2],
        stable_digest.expect("nonempty sample set"),
        state.supervisor.shadow_session().retained_len(),
    )
}

fn short_mode() -> bool {
    let Some(value) = std::env::var_os("KIKO_NAVIGATION_SHADOW_BENCH_SHORT") else {
        return false;
    };
    match value
        .to_str()
        .expect("KIKO_NAVIGATION_SHADOW_BENCH_SHORT must be valid UTF-8")
    {
        "1" | "true" => true,
        "0" | "false" => false,
        _ => panic!("KIKO_NAVIGATION_SHADOW_BENCH_SHORT must be one of: 0, 1, false, true"),
    }
}

fn command_version(program: &OsStr, arguments: &[&str]) -> String {
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
            OsStr::new("/usr/sbin/sysctl")
        } else {
            OsStr::new("sysctl")
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
        let lscpu = command_version(OsStr::new("lscpu"), &[]);
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
    let warmup_rounds = if short {
        SHORT_WARMUP_ROUNDS
    } else {
        DEFAULT_WARMUP_ROUNDS
    };
    let iterations = if short {
        SHORT_ITERATIONS
    } else {
        DEFAULT_ITERATIONS
    };
    let fixtures = build_fixtures();
    let probe = behavior_probe(&fixtures);

    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let logical_parallelism = std::thread::available_parallelism()
        .map(|value| value.get().to_string())
        .unwrap_or_else(|_| "unavailable".to_owned());
    let compilation_profile = if cfg!(debug_assertions) {
        "debug-assertions"
    } else {
        "optimized"
    };
    let mut cargo_build_target = environment_value("CARGO_BUILD_TARGET");
    if cargo_build_target == "unavailable" {
        cargo_build_target = environment_value("TARGET");
    }
    println!(
        "navigation shadow benchmark metadata: evidence_label={SYNTHETIC_EVIDENCE_LABEL} timing_scope=reference_build_plus_ready_fail_closed_admission git_commit={} git_worktree={} rustc={:?} cargo={:?} os={} arch={} cpu_model={:?} logical_parallelism={} compilation_profile={} debug_assertions={} rustflags={:?} cargo_encoded_rustflags={:?} cargo_build_target={:?} short={} samples={} warmup_rounds={} iterations_per_sample={} horizon_steps={} step_period_s={} integration_substeps={} optimization_iterations={} candidates_per_wheel={} max_rollout_evaluations={} shadow_retained_records={} solver_budget_ns={} allocations=not_instrumented successful_mpc_timing=false",
        command_version(OsStr::new("git"), &["rev-parse", "HEAD"]),
        git_worktree_state(),
        command_version(&rustc, &["--version"]),
        command_version(&cargo, &["--version"]),
        std::env::consts::OS,
        std::env::consts::ARCH,
        cpu_model(),
        logical_parallelism,
        compilation_profile,
        cfg!(debug_assertions),
        environment_value("RUSTFLAGS"),
        environment_value("CARGO_ENCODED_RUSTFLAGS"),
        cargo_build_target,
        short,
        samples,
        warmup_rounds,
        iterations,
        fixtures.mpc_config.horizon_steps(),
        fixtures.mpc_config.step_period_s(),
        fixtures.mpc_config.integration_substeps(),
        OPTIMIZATION_ITERATIONS,
        CANDIDATES_PER_WHEEL,
        fixtures.mpc_config.max_rollout_evaluations(),
        RETAINED_SHADOW_RECORDS,
        SOLVER_BUDGET_NS,
    );
    println!(
        "navigation shadow behavior: map_instance_id={} map_revision={} path_points={} reference_points={} device_session_id={} odom_segment_id={} depth_frame_id={} depth_device_timestamp_ns={} depth_host_arrival_ns={} local_collision_view=no_observation decision=fail_closed_stop requested_pwm=0,0 solver_deadline=not_formed_without_collision_provenance final_revalidation=not_reached_without_collision_provenance predicted_trajectory=none motor_packets_sent=0 behavior_digest=0x{:016x}",
        probe.map_instance_id,
        probe.map_revision,
        probe.path_point_count,
        probe.reference_point_count,
        DEVICE_SESSION_ID,
        ODOM_SEGMENT_ID,
        DEPTH_FRAME_ID,
        DEPTH_DEVICE_TIMESTAMP_NS,
        DEPTH_HOST_ARRIVAL_NS,
        probe.behavior_digest,
    );

    let warmup_iterations = iterations.max(1);
    let mut warmup = run_state(&fixtures, FIRST_TICK_NS + 1_000_000_000);
    let mut warmup_digest = None;
    for _ in 0..warmup_rounds {
        let digest = black_box(run_iterations(&fixtures, &mut warmup, warmup_iterations));
        if let Some(expected) = warmup_digest {
            assert_eq!(digest, expected, "warmup behavior digest changed");
        } else {
            warmup_digest = Some(digest);
        }
    }

    let (median_ns, timed_digest, final_retained_records) = measure_samples(
        &fixtures,
        samples,
        iterations,
        FIRST_TICK_NS + 10_000_000_000,
    );
    println!(
        "navigation shadow benchmark: behavior=ready_but_unproven_depth_fail_closed median_ns_per_iteration={:.1} iterations_per_second={:.1} samples={} iterations_per_sample={} total_timed_iterations={} stable_timed_digest=0x{:016x} final_retained_records={} motor_packets_sent=0",
        median_ns,
        1.0e9 / median_ns,
        samples,
        iterations,
        samples * iterations,
        timed_digest,
        final_retained_records,
    );
}
