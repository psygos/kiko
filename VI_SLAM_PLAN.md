# VI-SLAM Commit Plan

Rust-native, geometry-authoritative VI-SLAM.

Core shape:

`CaptureBundle -> EstimatorService -> LocalVio(odom) -> KeyframeMap(map) -> PlaceRecognition -> LoopVerifier -> GlobalPoseGraph(map) -> MapFromOdom -> TrackingPose`

This plan is intentionally narrow:
- local estimation is a fixed-lag visual-inertial factor graph
- global correction is a separate pose-only graph
- ML proposes correspondences and places; geometry decides
- no ROS runtime
- no generic factor-graph framework
- no EKF/MSCKF

## Non-Negotiables

1. `CaptureBundle` is the authoritative estimator input.
2. `MapFromOdom` exists before VIO is wired into the tracker.
3. `NavState` is body/IMU-centric, not camera-centric.
4. `LocalVio` updates every capture, not only on keyframes.
5. Global graph remains pose-only.
6. Loop closure is visually certified; IMU only guides or vetoes.
7. VIO math is `f64` only.
8. Every commit compiles and passes tests.

## Service Model

Crossbeam services stay explicit and small:

- `CaptureService`
- `EstimatorService`
- `PlaceService`
- `LoopService`
- `GlobalGraphService`
- `VizService`
- `WriterService`

`EstimatorService` remains single-thread authoritative and owns:
- stereo frontend
- local estimator
- keyframe decisions

Do not split local VIO across services.

## Anti-Goals

1. No generic `FactorGraph<Node, Factor>` engine.
2. No quaternions as the public rotation representation.
3. No online camera-IMU extrinsic calibration in the first implementation.
4. No velocity/bias state in the long-horizon global graph.
5. No destructive loop correction that mutates local estimator state.
6. No giant mechanical rename pass before semantics are true.

## Naming Rules

Use explicit transforms:

- `pose_body_from_odom`
- `pose_cam_from_odom`
- `pose_cam_from_map`
- `map_from_odom`

Do not use new `world` names for any new VIO code.

## Commit Sequence

### Commit 1: Add Always-On Time Arithmetic, Capture, and IMU Types

Goal:
- establish exact estimator input types before touching runtime behavior

Files:
- `crates/kiko-slam/src/lib.rs`
- `crates/kiko-slam/src/capture.rs`
- `crates/kiko-slam/src/imu.rs`

Add:
- `Timestamp::delta_ns`
- `Timestamp::seconds_since`
- `Timestamp::midpoint`
- `CaptureId`
- `CaptureBundle { pair: StereoPair, imu: ImuBatch }`
- `ImuSample`
- `ImuBatch`
- `ImuBias`
- `ImuNoiseModel`
- `ImuExtrinsics`

Rules:
- `capture` and `imu` types are always compiled
- only solver code will be feature-gated later
- `ImuBatch` constructor enforces sorted, finite samples

Tests:
- timestamp arithmetic
- `ImuSample` rejects NaN/Inf
- `ImuBatch` rejects empty and unsorted data
- `CaptureId` monotonicity

Gate:
- `cargo check -p kiko-slam`
- `cargo test -p kiko-slam`

### Commit 2: Record and Replay IMU in Dataset IO

Goal:
- make IMU a real data plane, not metadata fiction

Files:
- `crates/kiko-slam/src/dataset/mod.rs`
- `crates/kiko-slam/src/dataset/reader.rs`
- `crates/kiko-slam/src/bin/kiko_slam/record.rs`
- `crates/kiko-slam/src/bin/kiko_slam/live.rs`
- `crates/kiko-slam/src/oak.rs`

Add:
- persisted `imu.bin`
- dataset writer path for IMU samples
- dataset reader path for IMU samples
- `next_bundle()` returning `CaptureBundle`
- OAK record/live IMU enable path

Rules:
- replay must produce deterministic `CaptureBundle` streams
- visual-only datasets produce empty `ImuBatch`

Tests:
- write/read round-trip for IMU
- bundle assembly between stereo timestamps
- no IMU dataset still replays cleanly

Gate:
- visual-only behavior unchanged
- dataset replay tests pass

### Commit 3: Add CalibrationBundle and Enforce Offline IMU Calibration

Goal:
- keep bad inertial calibration out of the estimator

Files:
- `crates/kiko-slam/src/calibration.rs`
- `crates/kiko-slam/src/lib.rs`
- tracker and bin call sites

Add:
- `CalibrationBundle`
- validated IMU noise and extrinsics
- gravity magnitude
- explicit `has_imu()`

Change:
- `SlamTracker::try_new(...)` takes `CalibrationBundle`

Rules:
- visual-only constructor remains easy
- IMU mode refuses invalid calibration

Tests:
- valid visual-only path
- invalid noise rejected
- invalid extrinsics / offset rejected

Gate:
- all existing tests updated, no behavior change

### Commit 4: Introduce MapFromOdom and TrackingPose Before VIO

Goal:
- create the local/global authority seam while behavior stays identical

Files:
- `crates/kiko-slam/src/map_from_odom.rs`
- `crates/kiko-slam/src/tracker.rs`
- `crates/kiko-slam/src/lib.rs`
- bin consumers

Add:
- `MapFromOdom(Pose64)`
- `TrackingPose { cam_from_odom, cam_from_map }`

Change:
- `TrackerOutput.pose` becomes `Option<TrackingPose>`
- tracker initializes with `map_from_odom = identity`
- current visual output sets `cam_from_odom == cam_from_map`

Rules:
- no pose renaming lie yet
- semantics become true before names change

Tests:
- identity bridge passthrough
- correction changes map pose without changing odom pose
- downstream bin code still works

Gate:
- zero behavior change in visual-only mode

### Commit 5: Split Tracker Into Explicit Components

Goal:
- remove the god object before VIO arrives

Files:
- `crates/kiko-slam/src/frontend.rs`
- `crates/kiko-slam/src/global_map.rs`
- `crates/kiko-slam/src/place_recognition.rs`
- `crates/kiko-slam/src/loop_manager.rs`
- `crates/kiko-slam/src/tracker.rs`

Extract:
- `StereoFrontend`
- `GlobalMap`
- `PlaceRecognition`
- `LoopManager`

Rules:
- single-threaded orchestrator remains
- no new concurrency here
- pure refactor, no semantic drift

Tests:
- differential frontend-vs-old-tracker behavior tests
- all existing tracker and loop tests remain green

Gate:
- `cargo test -p kiko-slam` unchanged except additive tests

### Commit 6: Add Missing SO(3) / SE(3) Math and Small Dense Solvers

Goal:
- support VI factors with exact math already aligned to the codebase

Files:
- `crates/kiko-slam/src/math.rs`

Add:
- `so3_left_jacobian_f64`
- `so3_left_jacobian_inv_f64`
- `so3_right_jacobian_inv_f64`
- small dense Cholesky helpers

Tests:
- Jacobian identities
- inverse round-trip
- finite-difference checks
- Cholesky solve checks

Gate:
- all math tests pass

### Commit 7: Add IMU Preintegration

Goal:
- standalone, heavily tested inertial math

Files:
- `crates/kiko-slam/src/vio/mod.rs`
- `crates/kiko-slam/src/vio/preintegration.rs`

Add:
- `PreintegratedImu`
- covariance propagation
- first-order bias correction

Rules:
- body/IMU frame only
- standard gravity-compensated preintegration semantics
- tests reflect preintegrated delta behavior, not naive world propagation

Tests:
- zero-motion stationary case under correct gravity handling
- constant rotation
- covariance PSD
- SO(3) validity
- bias Jacobian finite difference
- reintegration agreement for small bias updates

Gate:
- `cargo test -p kiko-slam --features vio`

### Commit 8: Add Local Visual Track Source for the Window

Goal:
- give the smoother a proper visual measurement source

Files:
- `crates/kiko-slam/src/vio/tracks.rs`
- frontend/tracker integration points

Add one of:
- `TrackManager`, or
- explicit `WindowLandmarks` / tracked feature observations

Rules:
- the VI graph cannot rely only on active-keyframe PnP
- visual residuals must be defined on window states

Tests:
- track continuity across short windows
- landmark association invariants
- deterministic track construction on replay

Gate:
- track source exists before VIO solver lands

### Commit 9: Add Body-Centric NavState and VI Factors

Goal:
- define the actual local estimator state and residuals

Files:
- `crates/kiko-slam/src/vio/state.rs`
- `crates/kiko-slam/src/vio/factors.rs`

Add:
- `NavState { pose_body_from_odom, velocity_odom, bias }`
- `Gravity`
- IMU factor
- bias random-walk factor
- visual reprojection factor

Rules:
- camera pose derived via extrinsics
- analytic Jacobians only

Tests:
- every Jacobian block against finite difference
- zero residual at ground truth
- reprojection factor correctness

Gate:
- factor tests must be exhaustive before smoother implementation

### Commit 10: Add LocalVio Fixed-Lag Smoother

Goal:
- local visual-inertial estimation in `odom`

Files:
- `crates/kiko-slam/src/vio/smoother.rs`
- `crates/kiko-slam/src/vio/marginalization.rs`

Add:
- `LocalVio`
- LM solve over fixed lag
- per-capture propagation and update
- marginalization support

Rules:
- runs every `CaptureBundle`
- keyframes affect anchoring / map insertion, not whether the local state updates

Tests:
- static bias convergence
- gravity stability
- accepted-step cost monotonicity
- window slide behavior

Gate:
- replay-only first
- no tracker wiring yet

### Commit 11: Export Pose-Only OdomConstraint From Marginalization

Goal:
- make the local/global handoff mathematically explicit

Files:
- `crates/kiko-slam/src/vio/marginalization.rs`
- `crates/kiko-slam/src/vio/odometry_factor.rs`

Add:
- `OdomConstraint { from, to, relative_pose, information }`

Rules:
- separate:
  - full marginalized local prior
  - exported 6x6 pose-only relative factor
- document and test the reduction

Tests:
- information matrix symmetry / SPD
- relative-pose reduction consistency at linearization point
- known synthetic window -> sensible pose-only constraint

Gate:
- do not feed the global graph until this derivation is justified by tests

### Commit 12: Wire LocalEstimator Into Tracker

Goal:
- tracker can run either visual-only or visual-inertial locally

Files:
- `crates/kiko-slam/src/tracker.rs`
- config/bin wiring

Add:
- `enum LocalEstimator { Visual(LocalBundleAdjuster), Inertial(LocalVio) }`

Change:
- tracker accepts `CaptureBundle`
- visual-only path wraps `StereoPair` into empty-IMU bundle
- inertial path updates every capture
- tracker output uses `TrackingPose`

Rules:
- zero regressions when IMU absent

Tests:
- all old tests pass in visual mode
- synthetic IMU replay produces odom poses in VI mode

Gate:
- both visual and vio test suites pass

### Commit 13: Replace Frozen Global Odometry Edges With OdomConstraint

Goal:
- stop using stale visual edges as global odometry truth

Files:
- `crates/kiko-slam/src/pose_graph/essential.rs`
- `crates/kiko-slam/src/global_map.rs`
- tracker/loop manager wiring

Add:
- `EssentialEdgeKind::Odometry`

Change:
- VIO-emitted `OdomConstraint` becomes global odometry edge
- old spanning-tree semantics get demoted / replaced for those pairs

Tests:
- replacement behavior
- global graph convergence with odometry edges

Gate:
- graph uses live local-estimator output, not frozen insertion-time snapshots

### Commit 14: Make Loop Retrieval API Explicit

Goal:
- remove hidden query identity assumptions

Files:
- `crates/kiko-slam/src/loop_closure.rs`
- place recognition call sites

Change:
- `KeyframeDatabase::query(...)` takes explicit query keyframe identity

Tests:
- explicit query keyframe tests
- determinism tests

Gate:
- no hidden “last inserted entry is query” behavior

### Commit 15: Harden Loop Verification

Goal:
- accepted loops become extremely hard to falsify

Files:
- `crates/kiko-slam/src/loop_closure.rs`
- `crates/kiko-slam/src/loop_verifier.rs`

Add:
- guided matching under local prior
- robust pose refinement
- measurement-aware loop information
- IMU gravity plausibility veto

Rules:
- IMU never certifies
- visual geometry remains proof

Tests:
- gravity-consistent vs inconsistent cases
- better weighting with stronger measurements
- adversarial false-loop rejection cases

Gate:
- verification quality rises before any fancy loop-set machinery

### Commit 16: Apply Loop Closure Through MapFromOdom Only

Goal:
- global correction stops touching local odom state

Files:
- `crates/kiko-slam/src/loop_manager.rs`
- `crates/kiko-slam/src/tracker.rs`
- map/global graph plumbing

Change:
- loop optimization updates global keyframe map poses
- compute correction into `MapFromOdom`
- local estimator state remains continuous

Tests:
- local odom continuity across loop correction
- map trajectory corrects globally
- no local state reset on loop application

Gate:
- this is the architectural keystone

### Commit 17: Add Robust Loop-Set Selection

Goal:
- one bad loop cannot poison the global map

Files:
- `crates/kiko-slam/src/loop_consensus.rs`
- loop manager wiring

Add:
- pairwise-consistency or equivalent accepted-loop subset selection

Tests:
- all-good set accepted
- one-bad-loop rejected
- empty input behavior

Gate:
- loops are committed as a consistent set, not one-by-one blind trust

### Commit 18: Add VI-Aware Relocalization

Goal:
- relocalization uses inertial continuity without becoming inertial fantasy

Files:
- `crates/kiko-slam/src/tracker.rs`
- relocalization support code

Change:
- dead-reckoned odom prior continues during loss
- visual relocalization checked against gravity plausibility
- recovery seeds local VI from recovered map pose plus current inertial state

Tests:
- short loss preserves velocity/bias
- gravity-inconsistent hypothesis rejected
- recovery latency bounded

Gate:
- relocalization stops rebooting the estimator unless absolutely necessary

### Commit 19: Add Jetson Diagnostics and Default Profile

Goal:
- keep the design honest on target hardware

Files:
- `crates/kiko-slam/src/diagnostics.rs`
- `crates/kiko-slam/src/bin/kiko_slam/config.rs`
- `bench.rs`

Add diagnostics:
- IMU propagation time
- preintegration sample count
- VI solve time
- marginalization time

Add Nano-safe defaults:
- small VI window
- small LM iteration cap
- reduced keypoints
- more conservative loop candidate count

Gate:
- benchmark labels remain exact
- timing data proves the budget instead of hand-waving it

## Verification At Every Commit

Required:
1. `cargo check -p kiko-slam`
2. `cargo test -p kiko-slam`

When `vio` code exists:
3. `cargo check -p kiko-slam --features vio`
4. `cargo test -p kiko-slam --features vio`

Optional but expected on larger milestones:
5. `cargo clippy -p kiko-slam --features vio`

Rules:
- no test deletions
- additive only
- every residual gets finite-difference Jacobian tests
- every commit must be bisect-friendly

## Staging Policy

When this branch is used:
- stage only files intentionally changed for the current milestone
- never stage `mkrspc/`
- check `git status --short` before every commit
- check `git diff --cached --stat` before every commit

## Immediate Next Commit After This Plan

Start with:
- Commit 1: `add capture and imu core types`

Do not start with tracker renames.
Do not start with VIO solver code.
Do not start with loop-closure hardening.
