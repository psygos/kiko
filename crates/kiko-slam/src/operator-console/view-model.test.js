"use strict";

const assert = require("node:assert/strict");
const model = require("./view-model.js");

function snapshot() {
  return {
    schema_version: 4,
    authority_kind: "production_external_interlocks",
    revision: "9",
    telemetry_observed_at_host_monotonic_ns: "1000",
    runtime: { kind: "active", mode: "frontier_explore" },
    requested_owner: {
      kind: "autonomous",
      session_id: "7",
      authority_generation: "11",
      mode: "frontier_explore",
    },
    actual_authority: {
      source: "agent",
      mode: "frontier_explore",
      authority_lease_id: "19",
      console_downstream_request_id: "11",
    },
    map: {
      map_epoch_id: "3",
      revision: "4",
      localization: "localized",
      grid: {
        width: 2,
        height: 2,
        resolution_m_per_cell: 0.1,
        origin_x_m: -0.1,
        origin_y_m: -0.1,
        cell_encoding: "unknown0_free1_occupied2",
        linearization: "row_major_x_fast_rows_increase_positive_map_y",
        origin_convention: "minimum_x_y_corner_of_cell00",
        map_axes: "right_handed_x_right_y_up",
      },
    },
    navigation: {
      pose: { x_m: 0, y_m: 0, yaw_rad: 0 },
      path: [{ x_m: 0, y_m: 0 }],
      goal: { x_m: 0.05, y_m: 0.05 },
      mpc_predicted_path: [{ x_m: 0, y_m: 0 }, { x_m: 0.01, y_m: 0 }],
      solver_duration_ns: "1500000",
      control_tick_lateness_ns: "250000",
    },
    last_requested_actuation: {
      downstream_request_id: "11",
      decision_id: "22",
      left_timer_pwm_percent: 4,
      right_timer_pwm_percent: 5,
    },
    last_applied: {
      sequence: 8,
      result_code: "applied_new",
      applied_left_timer_pwm_percent: 4,
      applied_right_timer_pwm_percent: 5,
      output_state: "nonzero_pwm",
      controller_fault_bits: 0,
    },
    stop_certainty: "uncertain",
    health: {
      stm32: "ready",
      head: "ready",
      eyes: "ready",
      oak: "ready",
    },
    software_safety_stop_latched: false,
    software_safety_signal_state: "not_latched",
    physical_emergency_stop_state: "released",
    rerun_diagnostics_url: null,
  };
}

assert.equal(model.parseConsoleSnapshot(snapshot()).revision, "9");

function qualificationSnapshot(motionAuthorityEnabled) {
  const value = snapshot();
  value.authority_kind = "wheels_off_qualification";
  value.wheels_off_qualification = {
    schema_version: 2,
    motion_authority_enabled: motionAuthorityEnabled,
    frontend_state: "connected",
    stop_barrier_pending: false,
    software_safety_stop_latched: false,
    runtime_ingress_state: "connected",
  };
  return value;
}

{
  const retired = qualificationSnapshot(false);
  retired.wheels_off_qualification.schema_version = 1;
  assert.throws(
    () => model.parseConsoleSnapshot(retired),
    /wheels_off_qualification.schema_version is unsupported/,
  );
}

{
  const parsed = model.parseConsoleSnapshot(qualificationSnapshot(false));
  assert.deepEqual(parsed.qualification_motion_gate, {
    motionAuthorityEnabled: false,
    frontendState: "connected",
    runtimeIngressState: "connected",
    stopBarrierPending: false,
    softwareSafetyStopLatched: false,
    ready: false,
  });
  assert(Object.isFrozen(parsed.qualification_motion_gate));
  assert.deepEqual(model.qualificationMotionView(parsed), {
    ready: false,
    ownerLabel: "MOTION ATTESTATION PENDING",
    requestedOwnerLabel: "manual motion locked pending attended attestation",
    modeLabel:
      "Manual qualification motion is locked pending attended startup attestation.",
    readinessLabel: "manual motion locked · attended attestation pending",
    className: "warn",
  });
}

{
  const parsed = model.parseConsoleSnapshot(qualificationSnapshot(true));
  assert.equal(parsed.qualification_motion_gate.ready, true);
  assert.equal(model.qualificationMotionView(parsed).ready, true);
}

for (const invalid of [undefined, null, "false", 0, 1]) {
  const value = qualificationSnapshot(false);
  value.wheels_off_qualification.motion_authority_enabled = invalid;
  assert.throws(
    () => model.parseConsoleSnapshot(value),
    /motion_authority_enabled must be boolean/,
  );
}

for (const invalid of [undefined, null, "false", 0, 1]) {
  const value = qualificationSnapshot(true);
  value.wheels_off_qualification.stop_barrier_pending = invalid;
  assert.throws(
    () => model.parseConsoleSnapshot(value),
    /stop_barrier_pending must be boolean/,
  );
}

for (const invalid of [undefined, null, "false", 0, 1]) {
  const value = qualificationSnapshot(true);
  value.wheels_off_qualification.software_safety_stop_latched = invalid;
  assert.throws(
    () => model.parseConsoleSnapshot(value),
    /software_safety_stop_latched must be boolean/,
  );
}

{
  const value = qualificationSnapshot(true);
  value.wheels_off_qualification.runtime_ingress_state = "reconnecting";
  assert.throws(
    () => model.parseConsoleSnapshot(value),
    /runtime_ingress_state is unsupported/,
  );
}

{
  const value = qualificationSnapshot(true);
  value.wheels_off_qualification.frontend_state = "reconnecting";
  assert.throws(
    () => model.parseConsoleSnapshot(value),
    /frontend_state is unsupported/,
  );
}

{
  const value = qualificationSnapshot(false);
  value.wheels_off_qualification.stop_barrier_pending = true;
  value.wheels_off_qualification.runtime_ingress_state =
    "disconnected_stop_unconfirmed";
  const view = model.qualificationMotionView(model.parseConsoleSnapshot(value));
  assert.equal(view.ready, false);
  assert.match(view.readinessLabel, /attestation pending/);
  assert.match(view.readinessLabel, /runtime ingress disconnected stop unconfirmed/);
  assert.match(view.readinessLabel, /stop barrier pending/);
}

{
  const value = qualificationSnapshot(true);
  value.wheels_off_qualification.stop_barrier_pending = true;
  const view = model.qualificationMotionView(model.parseConsoleSnapshot(value));
  assert.equal(view.ready, false);
  assert.equal(view.ownerLabel, "QUALIFICATION STOP BARRIER PENDING");
}

{
  const value = qualificationSnapshot(true);
  value.wheels_off_qualification.software_safety_stop_latched = true;
  const parsed = model.parseConsoleSnapshot(value);
  assert.equal(parsed.qualification_motion_gate.ready, false);
  const view = model.qualificationMotionView(parsed);
  assert.equal(view.ownerLabel, "QUALIFICATION SAFETY STOP LATCHED");
  assert.equal(view.className, "fault");
}

{
  const value = qualificationSnapshot(false);
  value.wheels_off_qualification.frontend_state = "awaiting_connection";
  const awaiting =
    model.qualificationMotionView(model.parseConsoleSnapshot(value));
  assert.equal(awaiting.ownerLabel, "QUALIFICATION FRONTEND STARTING");
  assert.equal(awaiting.className, "warn");

  value.wheels_off_qualification.frontend_state = "disconnected";
  const disconnected =
    model.qualificationMotionView(model.parseConsoleSnapshot(value));
  assert.equal(
    disconnected.ownerLabel,
    "QUALIFICATION FRONTEND DISCONNECTED",
  );
  assert.equal(disconnected.className, "fault");
}

{
  const value = qualificationSnapshot(true);
  value.wheels_off_qualification.runtime_ingress_state =
    "disconnected_stop_unconfirmed";
  const view = model.qualificationMotionView(model.parseConsoleSnapshot(value));
  assert.equal(view.ready, false);
  assert.equal(view.ownerLabel, "QUALIFICATION INGRESS DISCONNECTED");
  assert.equal(view.className, "fault");
}

{
  const legacy = snapshot();
  legacy.schema_version = 3;
  assert.throws(
    () => model.parseConsoleSnapshot(legacy),
    /unsupported snapshot schema/,
    "V3 omitted the authority class and must not be inferred as production",
  );
}

{
  const missing = snapshot();
  delete missing.authority_kind;
  assert.throws(
    () => model.parseConsoleSnapshot(missing),
    /snapshot.authority_kind is unsupported/,
  );
}

{
  const contradictory = qualificationSnapshot(true);
  contradictory.authority_kind = "production_external_interlocks";
  assert.throws(
    () => model.parseConsoleSnapshot(contradictory),
    /contradicts snapshot.authority_kind/,
  );
}

{
  const missingEvidence = snapshot();
  missingEvidence.authority_kind = "wheels_off_qualification";
  assert.throws(
    () => model.parseConsoleSnapshot(missingEvidence),
    /requires wheels_off_qualification evidence/,
  );
}

{
  const value = snapshot();
  value.rerun_diagnostics_url = "rerun+http://127.0.0.1:9876/proxy";
  const parsed = model.parseConsoleSnapshot(value);
  assert.deepEqual(parsed.rerun_diagnostics_url, {
    connectUri: "rerun+http://127.0.0.1:9876/proxy",
    forwardedPort: 9876,
  });
  assert(Object.isFrozen(parsed.rerun_diagnostics_url));
}

for (const invalid of [
  "http://127.0.0.1:9876/proxy",
  "rerun+https://127.0.0.1:9876/proxy",
  "rerun+http://192.168.50.2:9876/proxy",
  "rerun+http://localhost:9876/proxy",
  "rerun+http://127.0.0.1:9876/",
  "rerun+http://127.0.0.1:9876/proxy?token=secret",
  "rerun+http://user@127.0.0.1:9876/proxy",
  "rerun+http://127.0.0.1:0/proxy",
]) {
  const value = snapshot();
  value.rerun_diagnostics_url = invalid;
  assert.throws(
    () => model.parseConsoleSnapshot(value),
    /rerun_diagnostics_url/,
  );
}

{
  const invalid = snapshot();
  invalid.map.grid.linearization = "row_major_unspecified";
  assert.throws(
    () => model.parseConsoleSnapshot(invalid),
    /geometry contract is unsupported/,
  );
}

{
  const invalid = snapshot();
  invalid.navigation.pose.x_m = Number.POSITIVE_INFINITY;
  assert.throws(() => model.parseConsoleSnapshot(invalid), /must be finite/);
}

{
  const invalid = snapshot();
  invalid.actual_authority.authority_lease_id = "01";
  assert.throws(() => model.parseConsoleSnapshot(invalid), /exact decimal identity/);
}

{
  const view = model.authorityView(snapshot(), "7");
  assert.equal(view.actualLabel, "agent · frontier explore · lease 19");
  assert.equal(view.requestedLabel, "this session · frontier explore · generation 11");
  assert.equal(view.activeIntent, "autonomous_frontier_explore");
}

{
  const view = model.readinessView(snapshot());
  assert.equal(view.className, "ready");
  assert.equal(view.readinessLabel, "runtime active / frontier explore · STM32 ready · OAK ready");
}

{
  const terminal = snapshot();
  terminal.runtime = { kind: "shutting_down" };
  terminal.terminal = {
    kind: "control_ending",
    reason: "finalizing_warm_restart_checkpoint",
    current_camera_localization: "not_claimed",
  };
  assert.equal(model.parseConsoleSnapshot(terminal).terminal.kind, "control_ending");
  assert.match(
    model.readinessView(terminal).readinessLabel,
    /capture stop requested.*streams draining/,
  );

  terminal.runtime = { kind: "ready_stopped" };
  assert.throws(
    () => model.parseConsoleSnapshot(terminal),
    /requires a shutting-down runtime/,
  );
}

{
  const view = model.mpcView(snapshot());
  assert.equal(
    view.label,
    "last successful solve 1.50 ms · rollout 2 points",
  );
  assert.match(view.timingLabel, /tick lateness 0.25 ms$/);
}

{
  const value = snapshot();
  value.software_safety_stop_latched = true;
  value.physical_emergency_stop_state = "engaged";
  value.health.oak = "faulted";
  value.last_applied.controller_fault_bits = 5;
  const view = model.faultView(value);
  assert.equal(view.className, "fault");
  assert.match(view.label, /oak faulted/);
  assert.match(view.label, /software safety stop latched/);
  assert.match(view.label, /physical E-stop engaged/);
  assert.match(view.label, /controller fault bits 0x5/);
}

assert.equal(model.connectionView("stale").telemetryPrefix, "stale");
assert.match(model.connectionView("disconnected").overlay, /MAP FROZEN/);

{
  assert.deepEqual(
    model.qualificationMotionSessionDecision(true, true, false),
    {
      freshSessionRequired: true,
      clearHeldInputBeforeHandshake: true,
    },
    "the pending-generation session must be replaced only after held input is cleared",
  );
  for (const inputs of [
    [true, false, false],
    [true, true, true],
    [false, true, false],
  ]) {
    assert.equal(
      model.qualificationMotionSessionDecision(...inputs).freshSessionRequired,
      false,
    );
  }
}

{
  const previous = {
    sessionId: "7",
    sessionCapability: "11".repeat(32),
  };
  const replacement = {
    sessionId: "8",
    sessionCapability: "22".repeat(32),
  };
  const swap = model.qualificationMotionSessionSwap(previous, replacement);
  assert.deepEqual(swap.activeSession, replacement);
  assert.deepEqual(swap.retiringSession, previous);
  assert.notEqual(swap.activeSession, replacement);
  assert.notEqual(swap.retiringSession, previous);
  assert(Object.isFrozen(swap));
  assert(Object.isFrozen(swap.activeSession));
  assert(Object.isFrozen(swap.retiringSession));
  assert.throws(
    () => model.qualificationMotionSessionSwap(previous, previous),
    /replacement session must be fresh/,
  );
}

{
  const decision = (overrides) => model.softwareSafetyStopRetryDecision({
    attempt: 1,
    maximumAttempts: 4,
    confirmed: false,
    retryable: true,
    consoleUnlocked: true,
    ...overrides,
  });
  assert.equal(decision({ confirmed: true }), "confirmed");
  assert.equal(decision({}), "retry");
  assert.equal(decision({ attempt: 4 }), "unavailable");
  assert.equal(decision({ retryable: false }), "unavailable");
  assert.equal(decision({ consoleUnlocked: false }), "unavailable");
  assert.throws(
    () => decision({ attempt: 0 }),
    /retry attempt is invalid/,
  );
}

{
  // Wire row zero is minimum map Y. The model deliberately does not invert Y;
  // inversion is only a canvas presentation concern.
  const cells = new Uint8Array([1, 2, 0, 1]);
  const metadata = { width: 2, height: 2 };
  assert.equal(model.cellAt(cells, metadata, 0.25, 0.25), "free");
  assert.equal(model.cellAt(cells, metadata, 1.25, 0.25), "occupied");
  assert.equal(model.cellAt(cells, metadata, 0.25, 1.25), "unknown");
  assert.equal(model.cellAt(cells, metadata, 2, 0), "outside");
}

console.log("operator-console view-model tests passed");
