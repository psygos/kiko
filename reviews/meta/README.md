# Meta Review Records

Store one reviewer artifact per reviewed commit in this directory.
Use [`TEMPLATE.md`](./TEMPLATE.md) as the canonical shape.

Filename convention:
- `YYYY-MM-DD-<short-topic>.md`

Rules:
- choose a stable filename before the commit is finalized
- do not rename the artifact to a commit SHA later
- if the review is revised before merge, keep the same base name and append `-v2`, `-v3`, and so on only if necessary

Minimum contents:
- `## Commit Goal`
- `## Current Repo Starting Point`
- `## Previous Invariants`
- `## New Invariants Claimed`
- `## Touched Files`
- `## New Or Changed Metrics`
- `## New Or Changed Solver Outputs`
- `## Tests Added`
- `## Tests Run`
- `## Known Risks Or Deferred Follow-Ups`
- `## Findings`
- `## Invariant Verdict`
- `## Metric Verdict`
- `## Test Verdict`
- `## Merge Decision`

Commit message convention:
- add a trailer `Meta-Review: reviews/meta/<filename>.md`

A review recorded only in chat is not considered durable.
