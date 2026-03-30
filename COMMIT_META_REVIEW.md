# Commit Meta Review Protocol

This document defines the standing review loop for every commit that touches the measurement system, estimator semantics, metrics, or dense reconstruction.

The purpose of this review is not style enforcement.
The purpose is to detect semantic drift, metric dishonesty, invariant leakage, and commit-to-commit regressions before they land.

## Reviewer Role

The reviewer agent is a meta-judge, not a patch generator.

Its job is to compare:
- the previous semantic state of the codebase
- the intended effect of the current commit
- the actual changes in the current diff

and then answer:
- did the commit preserve or improve the declared invariants
- did it accidentally weaken type safety
- did it introduce new heuristic authority where math should have stayed authoritative
- did it mislabel or misreport any metric
- did it create a cache or artifact that can be mistaken for source truth

The reviewer is expected to be adversarial toward ambiguity and optimistic metric narratives.

## Review Timing

Run the reviewer:
- after the commit diff exists
- before the commit is finalized
- after tests finish, so the reviewer can see both the diff and the verification evidence

Run it again if:
- the patch changes substantially after review
- the support set of a metric changes
- a new public type is introduced
- a cache becomes authoritative
- a solver result begins exporting uncertainty

## Review Artifact Storage

Reviewer output must be durable.
Store it at:
- `reviews/meta/<yyyy-mm-dd>-<short-topic>.md`

Rules:
- the filename must be stable before the commit is finalized
- do not rely on a future commit SHA to name the file
- if the review is revised before merge, keep the same base name and append `-v2`, `-v3`, and so on only if necessary

The commit message must include:
- `Meta-Review: reviews/meta/<yyyy-mm-dd>-<short-topic>.md`

A review that exists only in chat is not considered recorded.

## Required Input Packet

The commit author must provide the reviewer with:
- the commit goal in one paragraph
- the current repo starting point relevant to this change
- the previous invariant set that this commit depends on
- the new invariant set the commit claims to establish
- the exact file list touched
- the diff or a summarized diff if very large
- new or changed exported types
- new or changed metrics
- new or changed solver outputs
- tests added
- tests run
- known risks and deferred follow-ups

If this packet is incomplete, the reviewer should block the commit.

## Review Axes

### 1. Semantic Category Integrity

Questions:
- Is each new type clearly an observation, derived observation, latent state, posterior belief, cached artifact, diagnostic metric, or policy signal?
- Did any type cross categories without a new explicit wrapper or conversion step?
- Did any debug or visualization type gain estimator authority by convenience?

Hard blockers:
- interpolated data flowing into a measured-data API without explicit conversion
- solver objective exported as though it were predictive quality
- cached artifact treated as source truth

### 2. Frame And Authority Integrity

Questions:
- Are new geometry-bearing values frame-tagged or otherwise semantically explicit?
- Did any path collapse `map` and `odom` authority?
- Does loop correction leave local odom continuity intact?
- Can any new API accidentally mix camera, body, map, or voxel frames?

Hard blockers:
- new authoritative geometry API with ambiguous frame semantics
- hidden identity-map assumptions in new VIO or dense code

### 3. Provenance Integrity

Questions:
- Can the system still tell whether a depth value is measured, interpolated, or predicted?
- Did a conversion preserve or intentionally inflate uncertainty?
- Is provenance explicit at API boundaries?

Hard blockers:
- provenance erased on an authoritative path
- a derived value shares the exact type of a measured value
- interpolated depth reaches authoritative TSDF before predictive calibration exists

### 4. Uncertainty Integrity

Questions:
- Does every authoritative fused quantity retain covariance or information?
- Are solver-reported uncertainties actually posterior-like, or contaminated by gauge fixing, anchors, LM damping, or robust-loss linearization?
- Does repeated evidence add information in a mathematically lawful way?
- Could support count be misread as independent evidence count?

Hard blockers:
- equal-weight fusion where unequal uncertainty is available
- covariance reported without state ordering or provenance
- posterior labeling without declared gauge, anchor, damping, and robust-loss semantics
- confidence increase from duplicate correlated evidence without qualification
- support count used as independent evidence count

### 5. Error-Handling Integrity

Questions:
- Are new failure modes represented as typed errors?
- Is contextual cause preserved?
- Did any library API flatten structured failures into strings or generic `anyhow` too early?

Hard blockers:
- stringly typed library errors in estimator code
- lost lower-level cause on the main failure path

### 6. Metric Truthfulness

Questions:
- Does every new metric name its support set and units?
- Is the metric a residual, objective, posterior uncertainty, or policy score?
- Are before and after comparisons actually valid?
- Could an operator misread the metric as reconstruction quality when it is only an optimization or selection statistic?
- If information gain or entropy reduction is reported, is the state it is measured over explicitly named?

Hard blockers:
- ambiguous metric names
- incomparable metrics presented as direct quality comparisons
- acceptance criteria built on an uncalibrated or semantically partial metric without disclosure

### 7. Heuristic Creep

Questions:
- Did a threshold move into the measurement kernel instead of the policy layer?
- Did a fallback path silently become the main path?
- Did prose like "good enough", "confidence", or "quality" hide missing mathematical definitions?

Hard blockers:
- hard-coded magic thresholds in measurement or posterior fusion kernels without a model-level justification
- matcher score treated as probability without calibration evidence

### 8. Test Adequacy

Questions:
- Does the new invariant have a test at the same semantic layer?
- Are compile-time guarantees backed by compile-fail tests where appropriate?
- Are Jacobians checked numerically if new analytic math landed?
- Are metrics checked on deterministic fixtures or synthetic truth cases?
- Are gauge-aware and robust-loss-aware uncertainty claims tested if uncertainty reporting changed?

Hard blockers:
- untested new invariant
- no misuse test for a type-level safety claim
- no calibration or predictive test for a new quality metric
- no gauge or robust-loss verification when posterior semantics changed

### 9. Commit-To-Commit Regression Risk

Questions:
- Relative to the previous state, what truthful capability improved?
- Relative to the previous state, what truthful capability may have regressed?
- Did the commit add abstraction that hides an unresolved semantic bug?
- Did migration shims preserve old misleading semantics longer than intended?
- Did the commit actually bind itself to the current repo seams, or only describe an abstract future?

Hard blockers:
- net decrease in semantic honesty hidden behind cleaner APIs
- migration shim that keeps the old wrong path as the default path
- roadmap or implementation that duplicates an already-present seam instead of completing it

## Reviewer Output Contract

The reviewer response must be structured as:

1. Findings
- ordered by severity
- each finding must include file references when possible
- each finding must say whether it is a blocker, major risk, or minor risk

2. Invariant Verdict
- list invariants strengthened
- list invariants weakened or left ambiguous

3. Metric Verdict
- name any metric that is partial, misleading, or incomparable
- confirm which metrics remain trustworthy after the commit

4. Test Verdict
- confirm whether the tests actually cover the new semantic claims
- list missing compile-fail, property, synthetic, gauge, robust-loss, or dataset regression tests

5. Merge Decision
- `accept`
- `accept with follow-up`
- `block`

The reviewer must not lead with summary praise.
Findings come first.

## Prompt Template For The Reviewer Agent

Use the following prompt, filling in the commit-specific packet:

```text
You are the measurement-system meta-reviewer for kiko-vio.
Your job is to judge the semantic effect of the current commit relative to the immediately previous code state.

Review priorities, in order:
1. Invariant preservation and strengthening
2. Metric truthfulness and support-set honesty
3. Frame and provenance safety
4. Error propagation quality
5. Heuristic creep and hidden semantic regressions
6. Test adequacy for the exact claims made by the commit

You are not a patch author here. You are an adversarial reviewer.
Assume the cost of a mistaken merge is high.
Do not reward architectural prose unless the diff actually enforces it.
If a metric could be misread, treat that as a defect.
If an illegal state is still representable after the commit, say so plainly.
If uncertainty is dropped anywhere on an authoritative path, block the commit.
If a cache or debug artifact can still masquerade as estimator truth, block the commit.
If a threshold lives in a kernel and should live in policy, flag it.
If posterior uncertainty is reported without declared gauge, anchor, damping, and robust-loss semantics, block the commit.
If support count is acting as independent evidence count, block the commit.

Required output order:
1. Findings, highest severity first, with file references
2. Invariants strengthened
3. Invariants weakened or still ambiguous
4. Metrics that remain trustworthy
5. Metrics that are misleading or partial
6. Missing tests
7. Merge decision: accept / accept with follow-up / block

Review packet:
- Commit goal: <fill>
- Current repo starting point: <fill>
- Previous invariants: <fill>
- New invariants claimed: <fill>
- Touched files: <fill>
- Diff summary: <fill>
- New exported types: <fill>
- New or changed metrics: <fill>
- New or changed solver outputs: <fill>
- Tests added: <fill>
- Tests run: <fill>
- Known risks / deferred follow-ups: <fill>
```

## Commit Acceptance Checklist

Before merging, the author and reviewer must be able to answer "yes" to all of these:
- Does each new exported type have an unambiguous semantic category?
- Does each new geometry-bearing value have an explicit frame authority?
- Does each new measurement-bearing value have explicit provenance?
- Does each new posterior or fused value retain uncertainty?
- Does each new uncertainty report declare gauge, anchor, damping, and robust-loss semantics?
- Does each new metric say what it is and over what support set?
- Does any support count avoid claiming independent evidence count?
- Do tests exist exactly where the new invariant lives?
- Has the commit made the truthful path easier than the misleading path?
- Is the review artifact stored under `reviews/meta/` and linked from the commit message?

If any answer is "no", the commit is not ready.

## Operating Rule

The standing principle is:

Do not ask whether the code got cleaner.
Ask whether the code got truer.
