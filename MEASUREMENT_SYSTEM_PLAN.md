# Measurement System Plan

Geometry-authoritative, type-first, invariant-first measurement architecture for `kiko-slam`.

This document is the measurement and verification companion to [`VI_SLAM_PLAN.md`](./VI_SLAM_PLAN.md).
`VI_SLAM_PLAN.md` defines the high-level estimator architecture.
This document defines how measurements, uncertainty, metrics, and dense reconstruction must be represented so the architecture remains mathematically honest.

This plan is intentionally strict:
- the source of truth is geometry plus an explicit measurement model
- compile-time guarantees are preferred over runtime checks
- uncertainty must survive every transformation that claims to be estimator-authoritative
- derived artifacts are never allowed to masquerade as measurements
- metrics must name their support set, units, and statistical meaning
- policy decisions must be layered above typed statistical quantities

If this document conflicts with expedient local shortcuts, this document wins.

## Problem Statement

The current codebase has strong estimator components and weak measurement semantics.
The strongest math already exists in:
- stereo geometry
- PnP reprojection residuals
- visual bundle adjustment
- IMU preintegration
- pose-graph information matrices

The weakest semantics exist in the dense path:
- interpolated dense depth can be mistaken for measured depth
- persistent dense points carry no uncertainty or provenance
- TSDF updates use equal incremental weights
- metrics are partly real and partly operational, but not always labeled as such
- map and odom authority are explicit in some places and implicit in others

The net effect is category error:
- measurement
- inference
- posterior belief
- cached artifact
- renderable debug output

are not separated strongly enough in types or APIs.

The purpose of this plan is to remove those category errors.

## Non-Negotiables

1. No authoritative geometry crosses subsystem boundaries as an untyped `[f32; 3]`, `Pose64`, or `Vec<f32>` unless it is at an adapter boundary to an external library or viewer.
2. No dense fusion path may consume interpolated depth as though it were measured depth.
3. No solver objective is reported as a product KPI without explicit normalization and support-set semantics.
4. No metric name may omit its support set if multiple support sets are plausible.
5. No fusion update may discard uncertainty unless the state is explicitly downgraded to visualization-only.
6. No frame ambiguity is tolerated: map, odom, body, camera, voxel, and pixel spaces must be explicit.
7. No learned model score is treated as probability or information unless it is calibrated and tested as such.
8. No threshold belongs inside the measurement-model kernel unless it is part of the model definition itself.
9. No stringly typed library errors in the estimator path.
10. No dense map or TSDF is the source of truth; they are materialized views over typed observations and state.
11. No `f32` in authoritative geometry, covariance, information, or solver-facing measurement code without a documented reason.
12. No raw support count is ever treated as independent evidence count; correlated evidence must be modeled, bounded, or conservatively grouped.
13. No reported uncertainty may be labeled posterior unless its gauge treatment, anchor treatment, damping treatment, and robust-loss linearization semantics are explicitly declared.
14. No interpolated depth may enter authoritative TSDF before predictive calibration and held-out verification exist.
15. No commit merges unless its new invariants, metrics, and failure modes are explicitly reviewed.

## Semantic Ontology

Every type must belong to exactly one semantic category.

### Observation

A value directly tied to a sensing event, plus enough metadata to model its uncertainty.

Examples:
- rectified stereo feature correspondence
- IMU sample
- calibrated depth pixel from a real depth sensor

Observation rules:
- must retain acquisition frame and timestamp
- must retain provenance
- must retain or derive uncertainty
- must be impossible to construct if physically invalid

### Derived Observation

A value inferred from observations but not itself directly measured.

Examples:
- barycentrically interpolated disparity
- temporally resampled signal
- fused candidate correspondence score

Derived observation rules:
- must never share the same type as a direct observation
- must carry explicit provenance and uncertainty inflation
- must be banned from estimator-authoritative sinks unless there is an explicit conversion step

### Latent State

A variable optimized or propagated by the estimator.

Examples:
- `NavState`
- map point
- local plane
- surfel mean and covariance
- pose graph state

Latent state rules:
- must name its frame
- must be impossible to construct with invalid covariance or information
- must retain the state ordering required to interpret solver outputs

### Posterior Belief

A latent state paired with quantified uncertainty after fusion.

Examples:
- posterior pose with information matrix
- surface belief with support count and innovation statistics

Posterior rules:
- means without uncertainty are not posterior beliefs
- support count is auxiliary, not a replacement for information
- support count must never be interpreted as independent evidence count
- posterior covariance must not include LM damping or arbitrary stabilizers when reported as confidence
- every reported posterior must declare gauge treatment, anchor treatment, and robust-loss semantics

### Cached Artifact

A materialized view derived from authoritative state to accelerate rendering or downstream access.

Examples:
- TSDF voxel grid
- mesh
- append-only Rerun point cloud

Cache rules:
- cache invalidation and rebuild semantics must be explicit
- cache correctness is subordinate to authoritative state correctness
- cache APIs must say what state version they were derived from

### Diagnostic Metric

A value produced to observe system behavior, not to serve as a state or loss.

Metric rules:
- must declare support set
- must declare units
- must declare whether it is a residual, objective, posterior uncertainty, or policy signal

### Policy Signal

A quantity consumed by decision logic.

Policy rules:
- may be derived from metrics
- must not redefine the metric itself
- thresholds must be documented in statistical terms when possible

## Rust Design Laws

### Parse, Do Not Validate

Domain objects should be parsed into lawful forms instead of created broadly and checked later.

Required patterns:
- `TryFrom` for data entering the system
- sealed constructors for internally lawful types
- newtypes for positive scalars, finite values, and SPD matrices
- typestates or provenance markers where the state space branches semantically

Forbidden patterns:
- public structs with raw primitive fields plus a later `validate()` call
- `Option` fields that encode invalid temporary states which should never exist once constructed
- late validation of frame identity or covariance shape after the value has already crossed subsystem boundaries

### Illegal States Must Be Unrepresentable

Required examples:
- `Transform<To, From>`
- `Point3<F>`
- `UnitRay<F>`
- `Variance`
- `StdDev`
- `Cov2`, `Cov3`, `Cov6`, `Cov9`
- `Info2`, `Info3`, `Info6`, `Info9`
- `MeasuredDepthImage<Cam>`
- `InterpolatedDepthField<Cam>`
- `SurfaceObservation<F, Provenance>`
- `SurfaceBelief<F>`

### Error Propagation

Library code uses typed errors.
Binary edges may wrap into `anyhow` with context.

Required error layers:
- `GeometryError`
- `MeasurementError`
- `CalibrationError`
- `FusionError`
- `MetricError`
- `SolverError`
- `PolicyError`

Required properties:
- exact structured cause
- `source()` chaining where appropriate
- contextual fields, not only display strings
- no loss of the lower-level cause at intermediate layers

### Frame Safety

Any transform or point crossing more than one module boundary must be frame-tagged.

The minimum required frame kinds:
- `Map`
- `Odom`
- `Body`
- `CamL`
- `CamR`
- `Voxel`
- `Image`

### Provenance Safety

Measurement provenance is part of the type system, not an annotation in a comment.

Minimum provenance markers:
- `MeasuredStereo`
- `InterpolatedStereo`
- `MeasuredDepthSensor`
- `ReprojectedPrediction`
- `PosteriorSurface`

### Numeric Discipline

Authoritative estimator numerics are `f64`.
Visualization and image buffers may remain `f32` or integer-typed where required.

### Metrics Discipline

Every metric type must answer:
- what random variable is being summarized
- over which support set
- in what units
- under what model
- whether lower is better, higher is better, or comparison is invalid without context

## Measurement Truth Charter

The system must stop optimizing for numbers that are not semantically comparable.

### Metrics That Are Real But Partial Today

- PnP inlier reprojection RMSE
- PnP inlier reprojection max
- inlier ratio
- covisibility ratio
- pose-graph edge residuals
- VIO and BA solver objectives

### Why They Are Partial

- PnP residuals are over accepted inliers, not all candidate observations
- inlier ratio is a selection outcome, not a reconstruction quality metric
- BA and VIO `final_cost` are optimization objectives, not normalized predictive metrics
- covisibility is a structural property, not a correctness metric

### Metrics The System Must Eventually Trust

- all-observation reprojection residual distributions
- held-out predictive reprojection residuals
- normalized innovation squared by observation class
- posterior covariance calibration curves
- posterior surface thickness orthogonal to local surface model
- coverage under bounded uncertainty
- information gain and entropy reduction over an explicitly named state

## Current Repo Starting Point

This plan is not written against a blank slate.
It is written against the current tree.

### Already Present And Useful

- `MapFromOdom` already exists and names the local-global bridge.
- tracker output already exposes odom and map pose semantics.
- PnP inlier reprojection error is already a real partial metric.
- the dense module already explicitly declares itself visualization-only.

### Already Present But Semantically Incomplete

- the tracker seam is incomplete because `realign_map_from_odom` is still a placeholder.
- VIO reprojection still contains identity-bridge assumptions in parts of the current solver path.
- diagnostics contain real signals but do not yet encode support-set truth strongly enough.
- TSDF still accepts raw depth buffers with equal incremental weighting.

### Not Yet Present As First-Class Truth

- provenance-typed observation objects
- support-set-safe metric types
- solver-exposed uncertainty with lawful semantics
- posterior surface belief state
- predictive calibration and held-out quality harnesses
- uncertainty-aware authoritative TSDF input types

## Commit Track Overview

This plan is an overlay on the estimator roadmap in [`VI_SLAM_PLAN.md`](./VI_SLAM_PLAN.md).

The dependency order is:
- Stage A: semantics and compile-time safety
- Stage B: authoritative solver exposure
- Stage C: surface-state and dense-fusion redesign
- Stage D: verifiability and policy cleanup
- Stage E: removal of misleading legacy paths

Do not start Stage C before Stage A is substantially complete.
Do not declare dense reconstruction solved before Stage D is in place.

## Interlock With VI_SLAM_PLAN

The repo is already partway into the architectural work described in [`VI_SLAM_PLAN.md`](./VI_SLAM_PLAN.md), but unevenly.
This measurement plan therefore interlocks with the existing codebase as follows:

- M0 is immediate and applies to all later work.
- M1 through M5 should land before any large rewrite of `local_ba.rs`, `diagnostics.rs`, `dense_cloud.rs`, `tsdf.rs`, or solver result types.
- M6 is not a fresh abstraction pass; it is the completion of seams that already exist in `map_from_odom.rs` and `tracker.rs` but remain semantically incomplete.
- M7 through M10 must be implemented against the completed M6 seam, not against an identity `map_from_odom` shortcut.
- M10 is explicitly blocked on M11 if the input is interpolated depth rather than measured depth.
- M11 through M14 are the gate before claiming that dense reconstruction quality is solved or that performance tuning is worthwhile.

## Commit Sequence

### Commit M0: Measurement Charter And Naming Freeze

Goal:
- establish the semantic contract before implementation work changes APIs

Files:
- `MEASUREMENT_SYSTEM_PLAN.md`
- `COMMIT_META_REVIEW.md`
- `reviews/meta/README.md`
- optional small pointer additions in `README.md` or `VI_SLAM_PLAN.md`

Add:
- semantic ontology
- non-negotiables
- naming rules for measurements, posteriors, and metrics
- support-set naming rules
- review artifact storage convention

Invariants:
- no new metric or type is added without a semantic category
- names like `world` are not reintroduced if `map` and `odom` are the true authorities

Tests:
- none beyond documentation review

Gate:
- no code merges that violate the charter after this point

### Commit M1: Frame-Safe Geometry Kernel

Goal:
- move geometry and transforms onto compile-time-safe rails

Primary files:
- `crates/kiko-slam/src/math.rs`
- `crates/kiko-slam/src/lib.rs`
- new `crates/kiko-slam/src/geometry.rs` if separation becomes cleaner

Add:
- `Point3<F>`
- `Vec3<F>`
- `Transform<To, From>`
- `UnitRay<F>`
- positive finite scalar wrappers
- SPD covariance and information wrappers

Change:
- adapt existing pose and point helpers through explicit conversion shims

Invariants:
- map and odom values cannot be mixed accidentally
- covariance and information matrices cannot be built without shape and finiteness checks

Tests:
- transform inverse and composition round-trip
- compile-fail tests for frame mismatch misuse
- SPD constructor acceptance and rejection
- finite-difference checks for key transform Jacobians if exposed

Gate:
- default build and `--features vio` build remain green

### Commit M2: Typed Error Taxonomy And Parse-First Constructors

Goal:
- make errors precise and make invalid domain objects impossible to construct silently

Primary files:
- `crates/kiko-slam/src/calibration.rs`
- `crates/kiko-slam/src/capture.rs`
- `crates/kiko-slam/src/imu.rs`
- `crates/kiko-slam/src/triangulation.rs`
- `crates/kiko-slam/src/local_ba.rs`

Add:
- subsystem error enums
- `TryFrom` and `new(...) -> Result<...>` constructors for domain types
- explicit `InvariantViolation` only for truly impossible internal states

Change:
- library internals stop returning generic string errors
- binary entrypoints add context at call boundaries

Invariants:
- no authoritative measurement or calibration object can exist in a nonsensical state
- all failure modes preserve structured causes

Tests:
- constructor rejection tests for NaN, Inf, unsorted samples, invalid calibration, non-SPD covariance
- error source-chain tests
- feature-gated path tests preserving exact error causes

Gate:
- estimator-path errors are typed and context-rich

### Commit M3: Observation And Provenance Domain Model

Goal:
- distinguish measured, interpolated, and predicted observations in the type system

Primary files:
- `crates/kiko-slam/src/triangulation.rs`
- `crates/kiko-slam/src/depth.rs`
- `crates/kiko-slam/src/dense_cloud.rs`
- `crates/kiko-slam/src/tsdf.rs`

Add:
- `StereoFeatureObservation<CamL, CamR>`
- `MeasuredDepthPixel<Cam>`
- `MeasuredDepthImage<Cam>`
- `InterpolatedDepthPixel<Cam>`
- `InterpolatedDepthField<Cam>`
- `SurfaceObservation<F, P>`

Change:
- legacy `SparseStereoSample` becomes an adapter or disappears
- raw dense depth buffers are no longer authoritative internal types

Invariants:
- interpolated depth cannot be passed to sinks that require measured depth
- provenance is explicit at API boundaries

Tests:
- compile-fail tests for provenance misuse
- lawful constructor tests for disparity and depth uncertainty
- round-trip adapters for legacy callers during migration

Gate:
- dense and TSDF paths compile through explicit adapters only

### Commit M4: Metric Taxonomy And Support-Set-Safe Diagnostics

Goal:
- stop conflating residuals, objectives, and operational counters

Primary files:
- `crates/kiko-slam/src/diagnostics.rs`
- `crates/kiko-slam/src/observability.rs`
- `crates/kiko-slam/src/pnp.rs`
- `crates/kiko-slam/src/tracker.rs`

Add:
- typed metric wrappers such as `ResidualRmsePx`, `ResidualMaxPx`, `InnovationNis`, `SolverObjectiveValue`, `SupportCount`
- support-set descriptors such as `AllObservations`, `AcceptedInliers`, `HeldOutObservations`

Change:
- diagnostics field names become explicit about support sets
- ambiguous scalars are either renamed or wrapped

Invariants:
- no metric can be exported without support-set semantics
- solver objective cannot be misreported as predictive accuracy

Tests:
- serialization and logging tests for metric naming
- regression tests for diagnostics on fixed small datasets

Gate:
- dashboard and logs remain readable while becoming semantically precise

### Commit M5: Solver Posterior Exposure

Goal:
- expose the estimator information already being built instead of discarding it

Primary files:
- `crates/kiko-slam/src/local_ba.rs`
- `crates/kiko-slam/src/vio/preintegration.rs`
- `crates/kiko-slam/src/pose_graph/*.rs`

Add:
- posterior or reduced-information summaries to BA results
- posterior information blocks or calibrated covariance views for VIO results
- graph solve information summaries keyed by pose ordering
- explicit uncertainty-report contract covering gauge, anchor, damping, and robust-loss semantics

Change:
- result types stop ending at `{iterations, final_cost}`
- uncertainty objects that cannot satisfy the full contract are named as local linearization information, not posterior covariance

Invariants:
- any reported covariance or information names its state ordering and excludes LM damping when labeled posterior
- reported uncertainty is tied to the final accepted state, not a stale linearization
- gauge handling, anchor treatment, and robust-loss treatment are declared on every reported uncertainty object

Tests:
- synthetic system covariance consistency
- symmetry and SPD checks on exposed information blocks
- final-state rescoring tests
- gauge-aware uncertainty reporting tests
- robust-loss semantics tests preventing false posterior labeling

Gate:
- downstream code can consume uncertainty explicitly without statistical mislabeling

### Commit M6: Complete The Map/Odom Authority Seam

Goal:
- make local and global pose authority consistent before dense fusion is trusted

Primary files:
- `crates/kiko-slam/src/map_from_odom.rs`
- `crates/kiko-slam/src/tracker.rs`
- `crates/kiko-slam/src/vio/factors.rs`
- `crates/kiko-slam/src/local_ba.rs`

Add:
- explicit versioning or invalidation semantics for map corrections
- final implementation of `realign_map_from_odom`

Change:
- remove identity-bridge assumptions from VIO reprojection and any dense consumers
- make authoritative frame semantics uniform across tracking, VIO, loop closure, and dense outputs

Invariants:
- local estimator owns odom continuity
- global graph owns map correction
- dense caches know which authority they were built against

Tests:
- map correction changes map pose without discontinuity in odom
- loop closure invalidates or reintegrates affected dense caches
- frame-tagged compile-fail tests

Gate:
- no authoritative path remains frame-ambiguous

### Commit M7: Stereo Uncertainty Model

Goal:
- derive 3D observation uncertainty from geometry and calibration, not intuition

Primary files:
- `crates/kiko-slam/src/triangulation.rs`
- `crates/kiko-slam/src/calibration.rs`
- `crates/kiko-slam/src/pnp.rs`

Add:
- disparity uncertainty model
- epipolar residual accounting
- rectification residual accounting
- 3D covariance propagation through stereo Jacobians

Change:
- measured stereo observations now carry uncertainty into subsequent stages

Invariants:
- every authoritative 3D stereo observation has a covariance or information representation
- learned matching scores remain auxiliary unless calibrated

Tests:
- analytic versus finite-difference covariance propagation
- synthetic Monte Carlo agreement
- epipolar residual sanity checks

Gate:
- downstream fusion can weight observations by actual modeled information

### Commit M8: Derived Dense Field Redesign

Goal:
- keep interpolated dense geometry useful for visualization while preventing category mistakes

Primary files:
- `crates/kiko-slam/src/dense_cloud.rs`
- `crates/kiko-slam/src/viz.rs`

Add:
- `InterpolatedDepthField<Cam>` with propagated variance and triangle support provenance
- deterministic within-frame ownership or fusion rule for overlapping triangle contributions

Change:
- debug dense views are generated from typed derived observations
- append-only dense cloud logging is marked as visualization-only

Invariants:
- interpolated dense products are clearly derived, not measured
- debug views cannot silently become authoritative fusion inputs

Tests:
- duplicate suppression tests on overlapping triangles
- barycentric variance propagation tests
- deterministic output tests on replay

Gate:
- `dense_cloud.rs` is semantically incapable of masquerading as the reconstruction truth path

### Commit M9: Surface Belief Map

Goal:
- replace naked persistent dense points with posterior surface beliefs

Primary files:
- new `crates/kiko-slam/src/surface_map.rs`
- `crates/kiko-slam/src/viz.rs`
- `crates/kiko-slam/src/tsdf.rs`

Add:
- `SurfaceBelief<F>`
- `SurfaceBeliefMap<F>`
- support count, posterior information, and innovation statistics
- explicit quarantine for inconsistent updates
- conservative grouping or correlation classes for duplicated evidence

Change:
- persistent dense map views are rendered from `SurfaceBeliefMap`

Invariants:
- every persistent surface element has uncertainty
- repeated observations reduce uncertainty only according to information addition
- support count is never treated as independent evidence count
- inconsistency raises diagnostics rather than silently thickening geometry

Tests:
- covariance shrinks under repeated consistent observations
- inconsistent observations trip innovation statistics
- correlation-aware tests prevent false confidence from duplicate evidence

Gate:
- persistent point cloud output is no longer append-only truth

### Commit M10: Uncertainty-Aware TSDF

Goal:
- make volumetric fusion estimator-honest

Primary files:
- `crates/kiko-slam/src/tsdf.rs`
- `crates/kiko-slam/src/bin/kiko_slam/slam.rs`

Add:
- typed TSDF input messages that require lawful provenance
- inverse-variance or information-weighted updates
- uncertainty-aware truncation semantics

Change:
- authoritative TSDF path no longer accepts raw `Vec<f32>` depth buffers
- equal-weight `+1` updates disappear from authoritative fusion paths

Rules:
- measured depth may become authoritative TSDF input before M11 if its uncertainty path is lawful
- interpolated depth is banned from authoritative TSDF before M11 predictive calibration and held-out verification exist
- after M11, any probabilistic conversion from interpolated depth remains opt-in and must be empirically justified in review

Invariants:
- TSDF is a posterior fusion surface, not a bag of equally trusted rays
- derived dense depth requires an explicit probabilistic conversion step and empirical calibration before authoritative use

Tests:
- synthetic plane and wall convergence tests
- uncertainty-weighting differentiation tests
- reintegration correctness under pose-map corrections
- pre-M11 guard tests preventing interpolated-depth authority

Gate:
- volumetric fusion cannot lie about the authority of its input

### Commit M11: Predictive Metric And Calibration Harness

Goal:
- measure what future observations say about current estimates

Primary files:
- new `crates/kiko-slam/src/eval/` module or equivalent
- `crates/kiko-slam/src/diagnostics.rs`
- dataset CLI entrypoints

Add:
- held-out predictive reprojection metrics
- normalized innovation squared metrics by observation class
- covariance calibration curves
- posterior surface thickness metrics
- bounded-uncertainty coverage metrics

Change:
- top-line quality reporting shifts from operational counters to predictive consistency

Invariants:
- a good reconstruction predicts future data well
- confidence claims must match empirical error

Tests:
- synthetic truth-scene evaluation
- Monte Carlo calibration tests
- deterministic dataset regression outputs

Gate:
- CI or dataset regression fails when predictive quality regresses materially

### Commit M12: Policy Layer Over Statistical Quantities

Goal:
- separate decision logic from measurement kernels

Primary files:
- new `crates/kiko-slam/src/policy.rs`
- `crates/kiko-slam/src/tracker.rs`
- `crates/kiko-slam/src/loop_closure.rs`
- `crates/kiko-slam/src/local_ba.rs`

Add:
- typed policy inputs built from metrics and posteriors
- explicit decision structs for keyframe insertion, observation rejection, loop acceptance, dense fusion admission

Change:
- magic thresholds are lifted out of lower-level measurement code where possible

Invariants:
- metric computation is stable under policy changes
- decision thresholds become explicit and auditable

Tests:
- policy unit tests over synthetic metric inputs
- regression tests proving kernel outputs do not change when policy-only thresholds move

Gate:
- heuristic logic is concentrated in one thin layer

### Commit M13: Verification Matrix And Compile-Time Misuse Tests

Goal:
- make semantic correctness continuously enforceable

Primary files:
- new test modules
- CI configuration

Add:
- `trybuild` compile-fail tests for frame and provenance misuse
- property tests for transform algebra and SPD invariants
- finite-difference Jacobian checks
- dataset regression harness
- Monte Carlo calibration harness

Invariants:
- illegal API use fails either at compile time or at parse-time construction
- measurement-model math is regression-tested against numerical truth

Tests:
- this commit is mostly tests and harnesses

Gate:
- CI matrix covers default, `--features vio`, and dataset-eval where practical

### Commit M14: Remove Legacy Misleading Paths

Goal:
- finish the migration by deleting semantics that encourage wrong reasoning

Primary files:
- `crates/kiko-slam/src/dense_cloud.rs`
- `crates/kiko-slam/src/tsdf.rs`
- `crates/kiko-slam/src/diagnostics.rs`
- any adapter shims introduced earlier

Remove or quarantine:
- ambiguous diagnostics names
- raw dense depth handoff into authoritative fusion
- append-only dense cloud as anything other than debug output
- library APIs that accept untyped authoritative geometry

Invariants:
- the easy path and the correct path are now the same path

Tests:
- deletion-focused regression tests
- old misuse sites removed or compile-fail

Gate:
- after this commit, the codebase can no longer accidentally optimize the wrong artifact as though it were truth

## Commit Dependency Matrix

Hard dependencies:
- M1 depends on M0
- M2 depends on M1 in any module that adopts the new types
- M3 depends on M1 and M2
- M4 depends on M0 and should begin before or alongside M5
- M5 depends on M1 and M4
- M6 depends on the architectural seam in `VI_SLAM_PLAN.md`
- M7 depends on M1 and M3
- M8 depends on M3 and M7
- M9 depends on M5, M6, and M7
- M10 depends on M3, M6, M7, and M9
- M11 depends on M4, M5, M7, and M9
- M12 depends on M4 and M11
- M13 spans all earlier commits
- M14 depends on the successful adoption of the new pathways

Soft dependencies:
- M4 can start early and migrate in phases
- M13 should begin in skeletal form as soon as M1 exists

## Review Artifact Record

Every reviewed commit must leave an auditable artifact.
The default convention is:
- reviewer output stored at `reviews/meta/<yyyy-mm-dd>-<short-topic>.md`
- commit message includes a trailer `Meta-Review: reviews/meta/<yyyy-mm-dd>-<short-topic>.md`

A commit is not review-complete if the sign-off exists only in chat.

## Review Gates Per Commit

Every commit in this plan must satisfy all of the following:
- semantic category declared for every new exported type
- frame authority declared for every new geometry-bearing value
- provenance declared for every new measurement-bearing value
- uncertainty retained or explicitly downgraded to visualization-only
- metrics labeled with support set and units
- no raw support count treated as independent evidence count
- tests added at the layer where the new invariant lives
- migration path does not silently preserve old misleading semantics
- reviewer artifact stored using the convention in [`reviews/meta/README.md`](./reviews/meta/README.md)

## Definition Of Done

This plan is complete only when the following become true:
- dense reconstruction paths cannot consume untyped or provenance-free depth
- posterior uncertainty is available from the major solvers and used downstream with lawful semantics
- predictive metrics are first-class and can fail CI or dataset regression
- local odom and corrected map semantics are consistent across tracking and dense fusion
- the type system catches the common geometry and provenance errors before runtime
- library errors preserve exact cause and context instead of flattening into strings
- dense visualizations remain useful, but they are no longer confusable with estimator truth

At that point the system stops being a stack of clever modules and becomes a measurement system.
