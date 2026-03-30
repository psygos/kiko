# Meta Review

## Commit Goal

Add a predictive-consistency gate to the stable surface belief map so novel support views that disagree with an existing voxel belief do not enter the low-resolution stable map, while exposing honest rejection accounting and confirmed-surface consistency metrics for observability.

## Current Repo Starting Point

This tranche starts after `a9de99e` on `measurement-system`. At the start of the change, the stable surface path already used measured sparse stereo observations with conservative scalar positional variance, grouped same-batch evidence conservatively, required novel support views, enforced posterior-sigma and residual-consistency confirmation gates, and required tracked-pose quality before surface fusion. But a novel outlier view could still strengthen a voxel belief and only show up later as elevated residual energy.

## Previous Invariants

- Stable surface observations were measured sparse stereo points with conservative positional variance.
- Repeated same-view evidence did not count as new support.
- Stable surface confirmation required enough support views, low residual consistency score, and low posterior sigma.
- Tracked pose quality was checked before stable surface integration, but there was no predictive check against the voxel belief itself.

## New Invariants Claimed

- A novel support view must now satisfy an explicit predictive consistency score against the existing voxel belief before it can strengthen that belief.
- Predictive rejections do not mutate support count, posterior mean, posterior sigma, or belief raw-observation count.
- Predictive rejection accounting is exported separately from grouped support and grouped redundancy counts.
- Confirmed-surface summaries now expose consistency metrics explicitly.
- The predictive threshold is explicit and discoverable via `KIKO_SURFACE_MAX_PREDICTIVE_CONSISTENCY_SCORE`.

## Touched Files

- `crates/kiko-slam/src/surface_map.rs`
- `crates/kiko-slam/src/viz.rs`
- `crates/kiko-slam/src/bin/kiko_slam/slam.rs`

## New Or Changed Metrics

- `diagnostics/surface/predictive_grouped_views_rejected`
- `diagnostics/surface/rejected_predictive_grouped_views_mean_consistency_score`
- `diagnostics/surface/rejected_predictive_grouped_views_max_consistency_score`
- `diagnostics/surface/mean_confirmed_consistency_score`
- `diagnostics/surface/max_confirmed_consistency_score`

## New Or Changed Solver Outputs

- none

## Tests Added

- `surface_map::tests::predictive_gate_rejects_novel_outlier_view_without_polluting_belief`
- `surface_map::tests::summary_reports_confirmed_consistency_metrics`
- `viz::tests::surface_summary_scalars_export_confirmed_consistency_metrics`
- `slam::tests::slam_env_help_mentions_surface_predictive_consistency_env`

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/surface_map.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/viz.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/bin/kiko_slam/slam.rs`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `bash /home/makerspace/kiko-vio/scripts/check_meta_review.sh`
- reviewer agent `Euclid`: `accept with follow-up`

## Known Risks Or Deferred Follow-Ups

- The predictive threshold is mathematically scoped and explicitly named, but it still lives in `SurfaceMapConfig` rather than a dedicated policy module.
- The predictive score is lawful under the current isotropic scalar-variance model, but it is not a full anisotropic NIS and must not be renamed as such until the covariance model improves.
- Viewing-angle effects are still captured indirectly through stereo uncertainty growth and multi-view predictive disagreement rather than an explicit surface-normal model.

## Findings

- `minor risk`: the predictive threshold remains coupled to the belief-map module instead of a separate policy layer. This is explicit and reviewable, but it is still one policy seam inside posterior fusion.
- `major`: the surface map is materially harder to pollute because novel outlier views are rejected before they can strengthen support or shift the posterior state.
- `major`: the new predictive metrics are semantically honest because they are scoped to grouped predictive rejections and are kept separate from support-view counts and confirmed-belief summaries.

## Invariant Verdict

- strengthened: novel views now require both geometric novelty and predictive consistency; predictive rejections leave posterior mean, posterior sigma, support count, and belief raw-observation count unchanged; confirmed-surface consistency is now observable explicitly.
- weakened or ambiguous: the predictive threshold still belongs to `SurfaceMapConfig` instead of a dedicated policy layer.

## Metric Verdict

- trustworthy: `predictive_consistency_score` is explicitly not mislabeled as NIS; predictive rejection count and rejected-score summaries are grouped-view honest; confirmed consistency summaries are exported and pinned in tests.
- partial or misleading: confirmed consistency summaries describe accepted confirmed beliefs only and are not whole-map conflict metrics.

## Test Verdict

- covered: predictive rejection behavior at the surface-map layer; unchanged posterior and raw-observation invariants after rejection; summary scalar exports; env help discoverability; default and `vio` test matrices.
- missing: no blocker-level gap remains for this tranche.

## Merge Decision

`accept with follow-up`
