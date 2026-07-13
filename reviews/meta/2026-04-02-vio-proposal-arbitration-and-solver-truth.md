# Meta Review

> Historical API note (2026-07-14): this review records the names used by the
> commit it evaluated. The current API uses `VioObjectiveBreakdown`,
> `VioSolveResult::objective_breakdown()`, and
> `VioSolveResult::final_mixed_objective()`. The objective combines a robust
> pixel-squared reprojection term with dimensionless Mahalanobis terms, so it
> does not carry one physical unit. References to `VioCostBreakdown`,
> `cost_breakdown`, and `final_cost` below are historical rather than current
> API guidance.

## Commit Goal

Make the VIO refinement path truthful and transactional. The solver must rescore candidate states and only accept a genuinely lower-cost state, proposal adoption must not mutate authoritative runtime or local-BA state until the proposal is accepted, and the diagnostics/observability surface must expose proposal support sets and VIO cost composition honestly enough to explain why a frame was or was not adopted.

## Current Repo Starting Point

This change starts from the `measurement-system` branch after the stable-surface work: typed measured-vs-interpolated depth exists, TSDF is blocked from consuming interpolated stereo depth, surface fusion is pose-gated and support-gated, and surface visibility/debug logging is already in place. The remaining failure was that VIO could worsen reprojection yet still be adopted, and the existing VIO `final_cost` was not a trustworthy post-update acceptance metric.

## Previous Invariants

- Surface pose gating used tracked reprojection metrics, but the VIO proposal path could still mutate runtime state before adoption.
- `VioSolveResult.final_cost` existed, but the LM loop did not truthfully compare candidate-state cost against the current accepted state.
- Proposal metrics for visual, VIO, and shared support existed conceptually, but the code path still needed proposal-specific types and observability to prevent silent support-set confusion.
- Visual BA fallback in VIO mode could still mutate the local BA window before the proposal was judged against the visual baseline.

## New Invariants Claimed

- `optimize_vio(...)` rescored the retracted candidate state and only accepts it if the candidate cost is lower than the current accepted state cost.
- `VioSolveResult.final_cost` and `VioSolveResult.cost_breakdown` describe the final accepted state, not a stale pre-update linearization.
- VIO proposals do not mutate `predicted_state`, `last_optimized_state`, `vio_window`, or pending-IMU ownership until the proposal is explicitly committed.
- Visual BA fallback proposals do not mutate the authoritative `LocalBundleAdjuster` window until the proposal is explicitly committed.
- VIO adoption compares only exact shared projectable tracked support; changed support sets are rejected instead of being compared as if they were the same evidence.
- Proposal diagnostics are typed by support set and logged with proposal/disposition/source context, so downstream consumers can tell visual, VIO, and shared-support metrics apart.

## Touched Files

- `crates/kiko-slam/src/local_ba.rs`
- `crates/kiko-slam/src/tracker.rs`
- `crates/kiko-slam/src/diagnostics.rs`
- `crates/kiko-slam/src/observability.rs`
- `crates/kiko-slam/src/lib.rs`

## New Or Changed Metrics

- `TrackingPoseSource`
- `VioProposalDisposition`
- `VisualProposalProjectableTrackedObservationCountMetric`
- `VisualProposalProjectableTrackedObservationPixelResidualMetric`
- `VioProposalProjectableTrackedObservationCountMetric`
- `VioProposalProjectableTrackedObservationPixelResidualMetric`
- `VisualVsVioSharedProjectableTrackedObservationCountMetric`
- `VisualVsVioSharedProjectableTrackedObservationPixelResidualMetric`
- `VisualVsVioSharedProjectableTrackedObservationReprojectionMsePerAxisPx2Metric`
- observability scalars for proposal-ran/adopted/reject-reason one-hots and proposal/shared RMSE counts

## New Or Changed Solver Outputs

- `VioCostBreakdown`
- `VioSolveResult.cost_breakdown`
- `VioSolveResult.final_cost` now reflects the accepted state rather than a stale pre-acceptance cost

## Tests Added

- `local_ba::tests::local_bundle_adjuster_clone_supports_transactional_candidate_updates`
- `tracker::tests::vio_pose_adoption_rejects_changed_projectable_support_even_when_counts_match`
- `tracker::tests::visual_ba_adoption_requires_exact_projectable_support_match`

## Tests Run

- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`

## Known Risks Or Deferred Follow-Ups

- There is still no direct tracker-level integration test proving that a rejected VIO proposal leaves `predicted_state`/`vio_window` untouched across successive frames; the code structure now enforces this, but the proof is still indirect.
- The VIO cost is still a solver objective, not a normalized predictive metric. It is now honest about the accepted state, but it is still not the top-line quality metric for map acceptance.
- Calibration ingestion, gravity initialization, and IMU interval-boundary handling are still follow-up work. This change makes those future fixes measurable rather than silently masked.
- Existing repo warnings outside this tranche remain, especially in `vio/solve.rs`, `frontend.rs`, `global_map.rs`, `tracker.rs`, and `bench.rs`.

## Findings

- none blocking in this tranche after the transactional visual-BA fix and exact shared-support adoption checks

## Invariant Verdict

- strengthened: truthful accepted-state VIO cost reporting; transactional VIO proposal commit; transactional visual-BA fallback commit; exact shared-support comparison before VIO adoption; typed proposal-specific metric supports
- weakened or ambiguous: tracker-level proof of non-mutation on rejected VIO proposals is still indirect rather than covered by a dedicated integration test

## Metric Verdict

- trustworthy: proposal/shared tracked counts and RMSEs are support-set-specific; proposal source and disposition are explicit; decomposed VIO solver costs are logged for the accepted state
- partial or misleading: VIO `final_cost` remains an optimizer objective rather than a predictive invariant metric; tracked reprojection is still frame-local and not yet a held-out predictive score

## Test Verdict

- covered: default and `vio` test matrices; support-set metric typing; shared-support rejection logic; visual-BA exact-support adoption rule; BA clone transactional semantics
- missing: end-to-end tracker test proving rejected VIO proposals preserve authoritative runtime state across frames

## Merge Decision

`accept with follow-up`
