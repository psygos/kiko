"use strict";

const assert = require("node:assert/strict");
const model = require("./view-model.js");

function snapshot() {
  return {
    schema_version: 2,
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
