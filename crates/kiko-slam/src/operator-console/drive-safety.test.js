"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { driveSafety } = require("./view-model.js");

function transition(state, event) {
  return driveSafety.reduce(state, event);
}

function openedState() {
  return transition(
    driveSafety.createState(),
    { kind: "session_opened" },
  ).state;
}

function liveState() {
  return transition(openedState(), {
    kind: "snapshot_observed",
    revision: 1n,
    now_milliseconds: 100,
    stale_after_milliseconds: 1_500,
    local_safety_latched: false,
    server_safety_latched: false,
  }).state;
}

test("key release either releases authority or services the remaining intent", () => {
  const state = liveState();
  assert.deepEqual(
    transition(state, {
      kind: "key_released",
      held_direction_count: 0,
      desired_intent_present: false,
    }).effects,
    [driveSafety.effects.releaseManual],
  );
  assert.deepEqual(
    transition(state, {
      kind: "key_released",
      held_direction_count: 2,
      desired_intent_present: false,
    }).effects,
    [driveSafety.effects.releaseManual],
    "opposing held directions must remain released",
  );
  assert.deepEqual(
    transition(state, {
      kind: "key_released",
      held_direction_count: 1,
      desired_intent_present: true,
    }).effects,
    [driveSafety.effects.ensureDriveLoop],
  );
});

test("every browser lifecycle loss requests the best-effort release fallback", () => {
  const state = liveState();
  for (const source of [
    "blur",
    "offline",
    "pagehide",
    "visibility_hidden",
  ]) {
    const result = transition(state, { kind: "lifecycle_loss", source });
    assert.equal(result.state, state, `${source} must not invent telemetry state`);
    assert.deepEqual(
      result.effects,
      [driveSafety.effects.releaseManualBestEffort],
      source,
    );
  }
});

test("request and response timeouts disconnect and inhibit manual reacquisition", () => {
  const requestTimeout = driveSafety.requestTimeoutMessage("request", 750);
  const responseTimeout = driveSafety.requestTimeoutMessage("response", 750);
  assert.equal(requestTimeout, "request timed out after 750 ms");
  assert.equal(
    responseTimeout,
    "request timed out after 750 ms while reading response",
  );

  for (const detail of [requestTimeout, responseTimeout]) {
    const failed = transition(liveState(), {
      kind: "transport_failed",
      detail,
    });
    assert.equal(failed.state.connectionKind, "disconnected", detail);
    assert.equal(failed.state.localInhibit, true, detail);
    assert.equal(failed.state.snapshotFresh, false, detail);
    assert.deepEqual(
      failed.effects,
      [driveSafety.effects.releaseManualBestEffort],
      detail,
    );
  }
});

test("unchanged telemetry crosses the exact stale threshold once", () => {
  const live = liveState();
  const atThreshold = transition(live, {
    kind: "snapshot_observed",
    revision: 1n,
    now_milliseconds: 1_600,
    stale_after_milliseconds: 1_500,
    local_safety_latched: false,
    server_safety_latched: false,
  });
  assert.equal(atThreshold.state.connectionKind, "live");
  assert.equal(atThreshold.state.localInhibit, false);
  assert.deepEqual(atThreshold.effects, []);

  const stale = transition(atThreshold.state, {
    kind: "snapshot_observed",
    revision: 1n,
    now_milliseconds: 1_601,
    stale_after_milliseconds: 1_500,
    local_safety_latched: false,
    server_safety_latched: false,
  });
  assert.equal(stale.state.connectionKind, "stale");
  assert.equal(stale.state.localInhibit, true);
  assert.equal(stale.state.snapshotFresh, false);
  assert.deepEqual(
    stale.effects,
    [driveSafety.effects.releaseManualBestEffort],
  );

  const stillStale = transition(stale.state, {
    kind: "snapshot_observed",
    revision: 1n,
    now_milliseconds: 1_700,
    stale_after_milliseconds: 1_500,
    local_safety_latched: false,
    server_safety_latched: false,
  });
  assert.deepEqual(
    stillStale.effects,
    [],
    "an already stale observation must not schedule duplicate releases",
  );
});

test("reconnect stays inhibited until an acceptably fresh observation", () => {
  const disconnected = transition(liveState(), {
    kind: "transport_failed",
    detail: "request timed out after 750 ms",
  }).state;
  const unchangedAndStale = transition(disconnected, {
    kind: "snapshot_observed",
    revision: 1n,
    now_milliseconds: 1_701,
    stale_after_milliseconds: 1_500,
    local_safety_latched: false,
    server_safety_latched: false,
  }).state;
  assert.equal(unchangedAndStale.connectionKind, "stale");
  assert.equal(unchangedAndStale.localInhibit, true);

  const advanced = transition(unchangedAndStale, {
    kind: "snapshot_observed",
    revision: 2n,
    now_milliseconds: 1_702,
    stale_after_milliseconds: 1_500,
    local_safety_latched: false,
    server_safety_latched: false,
  });
  assert.equal(advanced.snapshotAdvanced, true);
  assert.equal(advanced.state.connectionKind, "live");
  assert.equal(advanced.state.localInhibit, false);

  const latched = transition(advanced.state, {
    kind: "snapshot_observed",
    revision: 3n,
    now_milliseconds: 1_703,
    stale_after_milliseconds: 1_500,
    local_safety_latched: false,
    server_safety_latched: true,
  }).state;
  assert.equal(latched.connectionKind, "live");
  assert.equal(latched.localInhibit, true);
});

test("a regressed monotonic observation is rejected", () => {
  assert.throws(
    () => transition(liveState(), {
      kind: "snapshot_observed",
      revision: 2n,
      now_milliseconds: 99,
      stale_after_milliseconds: 1_500,
      local_safety_latched: false,
      server_safety_latched: false,
    }),
    /snapshot observation clock regressed/,
  );
});

test("terminal stop confirms release first and only then uses stop fallback", () => {
  const releaseOnlySteps = [];
  let releaseOnly = driveSafety.createTerminalStopState();
  releaseOnlySteps.push(releaseOnly.nextStep);
  releaseOnly = driveSafety.reduceTerminalStop(
    releaseOnly,
    "release_confirmed",
  );
  assert.equal(releaseOnly.completed, true);
  assert.deepEqual(releaseOnlySteps, ["release_manual_exact_zero"]);

  const fallbackSteps = [];
  let fallback = driveSafety.createTerminalStopState();
  fallbackSteps.push(fallback.nextStep);
  fallback = driveSafety.reduceTerminalStop(
    fallback,
    "release_unavailable",
  );
  fallbackSteps.push(fallback.nextStep);
  fallback = driveSafety.reduceTerminalStop(
    fallback,
    "terminal_stop_confirmed",
  );
  assert.equal(fallback.completed, true);
  assert.deepEqual(fallbackSteps, [
    "release_manual_exact_zero",
    "terminal_stop_exact_zero",
  ]);

  assert.throws(
    () => driveSafety.reduceTerminalStop(
      driveSafety.createTerminalStopState(),
      "terminal_stop_confirmed",
    ),
    /release outcome is unsupported/,
  );
});
