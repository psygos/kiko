# Meta Review

## Commit Goal

Remove dead weight introduced around the stable-surface debug visibility work without changing runtime behavior. The cleanup deletes the now-redundant `is_confirmed_belief` path in `SurfaceBeliefMap` so confirmation classification has a single authority, and it removes duplicated in-test entity-path extraction logic in `viz.rs`.

## Current Repo Starting Point

This tranche starts immediately after `5f555c8` on `measurement-system`, where stable-surface debug visibility was added successfully. That commit intentionally prioritized explicit visibility and tests, but it left one redundant confirmation helper in `surface_map.rs` and duplicated test-only entity-path extraction code in `viz.rs`.

## Previous Invariants

- `SurfaceBeliefMap` had two independent confirmation code paths:
  - `is_confirmed_belief`
  - `classify_belief(...) == Confirmed`
- `viz.rs` tests duplicated Arrow-message entity-path extraction logic between `log_frames_emits_left_and_right_view_entities` and the new stable-surface debug test.

## New Invariants Claimed

- `SurfaceBeliefMap` uses `classify_belief` as the single source of truth for confirmed-vs-nonconfirmed classification.
- Test-only entity-path extraction in `viz.rs` goes through a single helper.

## Touched Files

- `crates/kiko-slam/src/surface_map.rs`
- `crates/kiko-slam/src/viz.rs`

## New Or Changed Metrics

- none

## New Or Changed Solver Outputs

- none

## Tests Added

- none

## Tests Run

- `rustfmt --edition 2024 /home/makerspace/kiko-vio/crates/kiko-slam/src/surface_map.rs /home/makerspace/kiko-vio/crates/kiko-slam/src/viz.rs`
- `cargo test -p kiko-slam --manifest-path /home/makerspace/kiko-vio/Cargo.toml`
- `cargo test -p kiko-slam --features vio --manifest-path /home/makerspace/kiko-vio/Cargo.toml`

## Known Risks Or Deferred Follow-Ups

- This is intentionally a narrow cleanup. It does not address the broader repository warning set outside the stable-surface tranche.
- `main.rs` remains dirty in the worktree but is unrelated to this commit and was left untouched.

## Findings

- none: the cleanup removes duplication without changing visible behavior or weakening the stable-surface observability guarantees.

## Invariant Verdict

- strengthened: confirmed-voxel classification now has a single authority path in `surface_map.rs`.
- weakened or ambiguous: none.

## Metric Verdict

- trustworthy: unchanged.
- partial or misleading: unchanged.

## Test Verdict

- covered: full default and `vio` crate suites remain green after the cleanup.
- missing: no additional coverage needed for this narrow dead-code removal.

## Merge Decision

`accept`
