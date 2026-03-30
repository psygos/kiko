# Meta Review

## Commit Goal

Add a measurement-system roadmap, a standing meta-review protocol, and a durable in-repo review record path so future estimator work is type-first, invariant-first, metric-honest, and continuously audited.

## Current Repo Starting Point

The repo already had `VI_SLAM_PLAN.md`, an explicit `MapFromOdom` seam, real but partial PnP inlier reprojection metrics, and a dense path that remained semantically weak: `dense_cloud.rs` explicitly described itself as visualization-only while `tsdf.rs` still accepted raw depth buffers and equal-weight updates.

## Previous Invariants

- `MapFromOdom` exists as the local-global bridge.
- tracker output already distinguishes odom and map pose semantics.
- PnP inlier reprojection RMSE is real but partial.
- dense reconstruction semantics were not yet type-safe or provenance-safe.

## New Invariants Claimed

- future measurement-system commits must declare semantic category, frame authority, provenance, uncertainty retention, support-set-safe metrics, and invariant-layer tests.
- support count must never be treated as independent evidence count.
- posterior uncertainty must not be labeled without gauge, anchor, damping, and robust-loss semantics.
- interpolated depth is banned from authoritative TSDF before predictive calibration exists.
- every reviewed commit must leave a durable artifact under `reviews/meta/`.

## Touched Files

- `MEASUREMENT_SYSTEM_PLAN.md`
- `COMMIT_META_REVIEW.md`
- `reviews/meta/README.md`
- `reviews/meta/TEMPLATE.md`
- `scripts/check_meta_review.sh`
- `.github/workflows/ci.yml`

## New Or Changed Metrics

- none implemented in runtime code
- added a metric taxonomy and support-set rules in the roadmap and review protocol

## New Or Changed Solver Outputs

- none implemented in runtime code
- added roadmap constraints for future uncertainty reporting semantics

## Tests Added

- `scripts/check_meta_review.sh` validates stored review artifacts for filename stability and required section headings
- CI job runs the validator on every push and pull request

## Tests Run

- `bash scripts/check_meta_review.sh`
- manual readback of `MEASUREMENT_SYSTEM_PLAN.md`
- manual readback of `COMMIT_META_REVIEW.md`
- manual readback of `reviews/meta/README.md`
- `git status --short`

## Known Risks Or Deferred Follow-Ups

- the validator currently enforces required headings, not semantic quality of the content
- commit-message trailer enforcement is documented but not yet mechanically checked in CI
- no Rust runtime code was changed in this commit, so estimator behavior is unchanged

## Findings

- major risk resolved: posterior semantics now explicitly require gauge, anchor, damping, and robust-loss disclosure before anything is called posterior uncertainty
- major risk resolved: the roadmap is now tied to the current repo state instead of describing a greenfield architecture
- major risk resolved: interpolated depth is explicitly blocked from authoritative TSDF before predictive calibration and held-out verification exist
- process risk resolved: review artifacts now have a stable pre-commit filename convention and a durable schema under `reviews/meta/`
- remaining process limitation: the stored review validator checks structure, not judgment quality

## Invariant Verdict

- strengthened: semantic categories, frame authority, provenance, uncertainty labeling, support-set-safe metrics, durable review artifacts
- weakened or ambiguous: none remaining at blocker level after the final review pass

## Metric Verdict

- trustworthy: existing PnP inlier reprojection metrics remain trustworthy as partial metrics; operational counts remain trustworthy only as operational statistics
- partial or misleading: BA/VIO `final_cost` remains only a solver objective; dense visualization outputs remain non-authoritative until future commits land

## Test Verdict

- covered: documentation/process invariants now have a mechanical validator and CI hook
- missing: semantic correctness of future runtime commits still depends on the per-commit reviewer and the future runtime tests defined in the roadmap

## Merge Decision

`accept`
