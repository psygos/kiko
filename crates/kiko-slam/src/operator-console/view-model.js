(() => {
  "use strict";

  const DECIMAL_ID = /^[1-9][0-9]*$/;
  const DECIMAL_U64 = /^(0|[1-9][0-9]*)$/;
  const HEALTH = new Set(["ready", "degraded", "faulted", "unavailable"]);
  const LOCALIZATION = new Set(["localized", "lost", "unavailable"]);
  const RUNTIME = new Set([
    "booting",
    "inventory",
    "disarmed",
    "awaiting_zero",
    "ready_stopped",
    "active",
    "faulted",
    "shutting_down",
  ]);
  const TERMINAL_KIND = new Set(["control_ending"]);
  const TERMINAL_REASON = new Set(["finalizing_warm_restart_checkpoint"]);
  const CHECKPOINT_LOCALIZATION = new Set(["not_claimed"]);
  const ACTIVE_MODES = new Set([
    "map_only",
    "commissioning",
    "manual",
    "frontier_explore",
    "point_goal",
  ]);
  const AUTHORITY_MODES = new Set(["manual", "frontier_explore", "point_goal"]);
  const AUTHORITY_SOURCES = new Set(["operator", "agent"]);
  const PHYSICAL_STOPS = new Set(["released", "engaged", "unavailable", "faulted"]);
  const STOP_CERTAINTY = new Set([
    "confirmed_applied_zero",
    "controller_reported_safe",
    "uncertain",
  ]);
  const GRID_ENCODING = "unknown0_free1_occupied2";
  const GRID_LINEARIZATION = "row_major_x_fast_rows_increase_positive_map_y";
  // This is the exact serde `snake_case` spelling of
  // `MinimumXYCornerOfCell00`; the wire header has a separately explicit
  // spelling and is checked by app.js.
  const GRID_ORIGIN = "minimum_x_y_corner_of_cell00";
  const GRID_AXES = "right_handed_x_right_y_up";
  const MAX_GRID_CELLS = 2_000_000;
  const MAX_PATH_POINTS = 16_384;

  function object(value, field) {
    if (value == null || typeof value !== "object" || Array.isArray(value)) {
      throw new Error(`${field} must be an object`);
    }
    return value;
  }

  function optionalObject(value, field) {
    return value == null ? null : object(value, field);
  }

  function exactString(value, pattern, field) {
    if (typeof value !== "string" || !pattern.test(value)) {
      throw new Error(`${field} is not an exact decimal identity`);
    }
    return value;
  }

  function finite(value, field) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`${field} must be finite`);
    }
    return value;
  }

  function integer(value, minimum, maximum, field) {
    if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
      throw new Error(`${field} is outside its integer domain`);
    }
    return value;
  }

  function enumValue(value, admitted, field) {
    if (typeof value !== "string" || !admitted.has(value)) {
      throw new Error(`${field} is unsupported`);
    }
    return value;
  }

  function parseRuntime(value) {
    const runtime = optionalObject(value, "runtime");
    if (!runtime) return null;
    enumValue(runtime.kind, RUNTIME, "runtime.kind");
    if (runtime.kind === "active") {
      enumValue(runtime.mode, ACTIVE_MODES, "runtime.mode");
    } else if (runtime.mode != null) {
      throw new Error("inactive runtime must not carry a mode");
    }
    return runtime;
  }

  function parseTerminal(value) {
    const terminal = optionalObject(value, "terminal");
    if (!terminal) return null;
    enumValue(terminal.kind, TERMINAL_KIND, "terminal.kind");
    enumValue(terminal.reason, TERMINAL_REASON, "terminal.reason");
    enumValue(
      terminal.current_camera_localization,
      CHECKPOINT_LOCALIZATION,
      "terminal.current_camera_localization",
    );
    return terminal;
  }

  function parseHealth(value) {
    const health = object(value, "health");
    for (const component of ["stm32", "head", "eyes", "oak"]) {
      if (health[component] != null) {
        enumValue(health[component], HEALTH, `health.${component}`);
      }
    }
    return health;
  }

  function parseGrid(value) {
    const grid = optionalObject(value, "map.grid");
    if (!grid) return null;
    const width = integer(grid.width, 1, MAX_GRID_CELLS, "map.grid.width");
    const height = integer(grid.height, 1, MAX_GRID_CELLS, "map.grid.height");
    const cells = width * height;
    if (!Number.isSafeInteger(cells) || cells > MAX_GRID_CELLS) {
      throw new Error("map.grid exceeds the console cell bound");
    }
    const resolution = finite(
      grid.resolution_m_per_cell,
      "map.grid.resolution_m_per_cell",
    );
    if (resolution <= 0) {
      throw new Error("map.grid.resolution_m_per_cell must be positive");
    }
    const originX = finite(grid.origin_x_m, "map.grid.origin_x_m");
    const originY = finite(grid.origin_y_m, "map.grid.origin_y_m");
    const maximumX = originX + width * resolution;
    const maximumY = originY + height * resolution;
    if (!Number.isFinite(maximumX) || !Number.isFinite(maximumY)
      || maximumX <= originX || maximumY <= originY) {
      throw new Error("map.grid has an unrepresentable metric extent");
    }
    if (grid.cell_encoding !== GRID_ENCODING
      || grid.linearization !== GRID_LINEARIZATION
      || grid.origin_convention !== GRID_ORIGIN
      || grid.map_axes !== GRID_AXES) {
      throw new Error("map.grid geometry contract is unsupported");
    }
    return grid;
  }

  function parseMap(value) {
    const map = optionalObject(value, "map");
    if (!map) return null;
    exactString(map.map_epoch_id, DECIMAL_ID, "map.map_epoch_id");
    exactString(map.revision, DECIMAL_U64, "map.revision");
    enumValue(map.localization, LOCALIZATION, "map.localization");
    parseGrid(map.grid);
    return map;
  }

  function parsePoint(value, field, pose = false) {
    const point = object(value, field);
    finite(point.x_m, `${field}.x_m`);
    finite(point.y_m, `${field}.y_m`);
    if (pose) finite(point.yaw_rad, `${field}.yaw_rad`);
    return point;
  }

  function parsePath(value, field) {
    if (value == null) return null;
    if (!Array.isArray(value) || value.length > MAX_PATH_POINTS) {
      throw new Error(`${field} exceeds its path bound`);
    }
    value.forEach((point, index) => parsePoint(point, `${field}[${index}]`));
    return value;
  }

  function parseNavigation(value) {
    const navigation = optionalObject(value, "navigation");
    if (!navigation) return null;
    if (navigation.pose != null) parsePoint(navigation.pose, "navigation.pose", true);
    if (navigation.goal != null) parsePoint(navigation.goal, "navigation.goal");
    parsePath(navigation.path, "navigation.path");
    parsePath(navigation.mpc_predicted_path, "navigation.mpc_predicted_path");
    for (const field of ["solver_duration_ns", "control_tick_lateness_ns"]) {
      if (navigation[field] != null) {
        exactString(navigation[field], DECIMAL_U64, `navigation.${field}`);
      }
    }
    return navigation;
  }

  function parseRequestedOwner(value) {
    const owner = optionalObject(value, "requested_owner");
    if (!owner) return null;
    if (owner.kind !== "manual" && owner.kind !== "autonomous") {
      throw new Error("requested_owner.kind is unsupported");
    }
    exactString(owner.session_id, DECIMAL_ID, "requested_owner.session_id");
    exactString(
      owner.authority_generation,
      DECIMAL_ID,
      "requested_owner.authority_generation",
    );
    if (owner.kind === "manual") {
      exactString(
        owner.deadman_deadline_host_monotonic_ns,
        DECIMAL_U64,
        "requested_owner.deadman_deadline_host_monotonic_ns",
      );
    } else {
      enumValue(owner.mode, new Set(["frontier_explore", "point_goal"]), "requested_owner.mode");
    }
    return owner;
  }

  function parseActualAuthority(value) {
    const authority = optionalObject(value, "actual_authority");
    if (!authority) return null;
    enumValue(authority.source, AUTHORITY_SOURCES, "actual_authority.source");
    enumValue(authority.mode, AUTHORITY_MODES, "actual_authority.mode");
    exactString(authority.authority_lease_id, DECIMAL_ID, "actual_authority.authority_lease_id");
    if (authority.console_downstream_request_id != null) {
      exactString(
        authority.console_downstream_request_id,
        DECIMAL_ID,
        "actual_authority.console_downstream_request_id",
      );
    }
    return authority;
  }

  function parseRequestedActuation(value) {
    const actuation = optionalObject(value, "last_requested_actuation");
    if (!actuation) return null;
    if (actuation.downstream_request_id != null) {
      exactString(
        actuation.downstream_request_id,
        DECIMAL_ID,
        "last_requested_actuation.downstream_request_id",
      );
    }
    if (actuation.decision_id != null) {
      exactString(actuation.decision_id, DECIMAL_ID, "last_requested_actuation.decision_id");
    }
    integer(
      actuation.left_timer_pwm_percent,
      -100,
      100,
      "last_requested_actuation.left_timer_pwm_percent",
    );
    integer(
      actuation.right_timer_pwm_percent,
      -100,
      100,
      "last_requested_actuation.right_timer_pwm_percent",
    );
    return actuation;
  }

  function parseReceipt(value) {
    const receipt = optionalObject(value, "last_applied");
    if (!receipt) return null;
    integer(receipt.sequence, 0, 0xffff_ffff, "last_applied.sequence");
    enumValue(
      receipt.result_code,
      new Set(["applied_new", "duplicate_cached", "stopped"]),
      "last_applied.result_code",
    );
    enumValue(
      receipt.output_state,
      new Set(["disabled", "zero_pwm", "nonzero_pwm"]),
      "last_applied.output_state",
    );
    integer(
      receipt.applied_left_timer_pwm_percent,
      -100,
      100,
      "last_applied.applied_left_timer_pwm_percent",
    );
    integer(
      receipt.applied_right_timer_pwm_percent,
      -100,
      100,
      "last_applied.applied_right_timer_pwm_percent",
    );
    integer(
      receipt.controller_fault_bits,
      0,
      0xffff_ffff,
      "last_applied.controller_fault_bits",
    );
    return receipt;
  }

  function parseRerunDiagnosticsUrl(value) {
    if (value == null) return null;
    if (typeof value !== "string") {
      throw new Error("rerun_diagnostics_url must be a string or null");
    }
    let url;
    try {
      url = new URL(value);
    } catch (_) {
      throw new Error("rerun_diagnostics_url must be an exact Rerun proxy URI");
    }
    const port = Number(url.port);
    const canonical = `rerun+http://${url.host}/proxy`;
    if (url.protocol !== "rerun+http:"
      || url.hostname !== "127.0.0.1"
      || !Number.isInteger(port)
      || port < 1
      || port > 65_535
      || url.username !== ""
      || url.password !== ""
      || url.pathname !== "/proxy"
      || url.search !== ""
      || url.hash !== ""
      || value !== canonical) {
      throw new Error(
        "rerun_diagnostics_url must be a canonical unencrypted 127.0.0.1 proxy URI",
      );
    }
    return Object.freeze({
      connectUri: canonical,
      forwardedPort: port,
    });
  }

  function parseConsoleSnapshot(raw) {
    const snapshot = object(raw, "snapshot");
    if (snapshot.schema_version !== 3) {
      throw new Error("unsupported snapshot schema");
    }
    exactString(snapshot.revision, DECIMAL_ID, "snapshot.revision");
    if (snapshot.telemetry_observed_at_host_monotonic_ns != null) {
      exactString(
        snapshot.telemetry_observed_at_host_monotonic_ns,
        DECIMAL_U64,
        "snapshot.telemetry_observed_at_host_monotonic_ns",
      );
    }
    const runtime = parseRuntime(snapshot.runtime);
    const terminal = parseTerminal(snapshot.terminal);
    if (terminal && runtime?.kind !== "shutting_down") {
      throw new Error("terminal checkpoint requires a shutting-down runtime");
    }
    parseRequestedOwner(snapshot.requested_owner);
    parseActualAuthority(snapshot.actual_authority);
    parseMap(snapshot.map);
    parseNavigation(snapshot.navigation);
    parseRequestedActuation(snapshot.last_requested_actuation);
    parseReceipt(snapshot.last_applied);
    parseHealth(snapshot.health);
    if (typeof snapshot.software_safety_stop_latched !== "boolean") {
      throw new Error("software_safety_stop_latched must be boolean");
    }
    enumValue(
      snapshot.physical_emergency_stop_state,
      PHYSICAL_STOPS,
      "physical_emergency_stop_state",
    );
    if (snapshot.stop_certainty != null) {
      enumValue(snapshot.stop_certainty, STOP_CERTAINTY, "stop_certainty");
    }
    return {
      ...snapshot,
      rerun_diagnostics_url: parseRerunDiagnosticsUrl(
        snapshot.rerun_diagnostics_url,
      ),
    };
  }

  function words(value) {
    return typeof value === "string" ? value.replaceAll("_", " ") : "unknown";
  }

  function formatNsMilliseconds(raw) {
    if (typeof raw !== "string" || !DECIMAL_U64.test(raw)) return "unknown";
    const hundredths = (BigInt(raw) + 5000n) / 10000n;
    return `${hundredths / 100n}.${(hundredths % 100n).toString().padStart(2, "0")} ms`;
  }

  function connectionView(kind) {
    switch (kind) {
      case "live":
        return {
          pill: "LOCAL / AUTHENTICATED",
          className: "pill ready",
          overlay: "",
          telemetryPrefix: "fresh",
        };
      case "stale":
        return {
          pill: "STATE STALE",
          className: "pill fault",
          overlay: "STALE VIEW · MOTION INHIBITED",
          telemetryPrefix: "stale",
        };
      case "disconnected":
        return {
          pill: "CONNECTION LOST",
          className: "pill fault",
          overlay: "CONNECTION LOST · MAP FROZEN · MOTION INHIBITED",
          telemetryPrefix: "disconnected",
        };
      default:
        return {
          pill: "LOCKED",
          className: "pill unknown",
          overlay: "LOCKED · NO CONTROL SESSION",
          telemetryPrefix: "locked",
        };
    }
  }

  function authorityView(snapshot, sessionId) {
    const actual = snapshot.actual_authority;
    const requested = snapshot.requested_owner;
    const actualLabel = actual
      ? `${words(actual.source)} · ${words(actual.mode)} · lease ${actual.authority_lease_id}`
      : "NO ACTIVE AUTHORITY";
    let requestedLabel = "none";
    if (requested) {
      const ownership = requested.session_id === sessionId ? "this session" : "another session";
      const mode = requested.kind === "manual" ? "manual" : words(requested.mode);
      requestedLabel =
        `${ownership} · ${mode} · generation ${requested.authority_generation}`;
    }
    const runtime = snapshot.runtime;
    const activeMode = actual?.mode
      || (runtime?.kind === "active" ? runtime.mode : null);
    const activeIntent = {
      map_only: "autonomous_map_only",
      frontier_explore: "autonomous_frontier_explore",
      point_goal: "autonomous_point_goal",
      manual: "manual",
      commissioning: "commissioning",
    }[activeMode] || null;
    return { actualLabel, requestedLabel, activeMode, activeIntent };
  }

  function readinessView(snapshot) {
    const runtime = snapshot.runtime;
    const terminal = snapshot.terminal;
    const runtimeLabel = runtime
      ? runtime.kind === "active"
        ? `active / ${words(runtime.mode)}`
        : words(runtime.kind)
      : "unknown";
    const stm32 = snapshot.health?.stm32 || "unknown";
    const oak = snapshot.health?.oak || "unknown";
    const faulted = runtime?.kind === "faulted"
      || stm32 === "faulted"
      || oak === "faulted";
    const ready = ["ready_stopped", "active"].includes(runtime?.kind)
      && stm32 === "ready"
      && oak === "ready";
    return {
      runtimeLabel,
      readinessLabel: terminal
        ? "terminal checkpoint · capture stop requested · sensor/map streams draining · finalizing exact warm-replay inputs · current-camera localization not claimed"
        : `runtime ${runtimeLabel} · STM32 ${stm32} · OAK ${oak}`,
      className: faulted ? "fault" : ready ? "ready" : "warn",
    };
  }

  function mpcView(snapshot) {
    const navigation = snapshot.navigation;
    const duration = formatNsMilliseconds(navigation?.solver_duration_ns);
    const rolloutPoints = Array.isArray(navigation?.mpc_predicted_path)
      ? navigation.mpc_predicted_path.length
      : 0;
    const lateness = formatNsMilliseconds(navigation?.control_tick_lateness_ns);
    const label = duration === "unknown"
      ? "no successful solve published"
      : `last successful solve ${duration} · rollout ${rolloutPoints} points`;
    return {
      label,
      timingLabel: `${label} · tick lateness ${lateness}`,
      className: duration === "unknown" ? "unknown" : "ready",
    };
  }

  function faultView(snapshot) {
    const faults = [];
    if (snapshot.runtime?.kind === "faulted") faults.push("runtime faulted");
    for (const component of ["stm32", "oak", "head", "eyes"]) {
      if (snapshot.health?.[component] === "faulted") {
        faults.push(`${component} faulted`);
      }
    }
    if (snapshot.software_safety_stop_latched) faults.push("software safety stop latched");
    if (Number.isInteger(snapshot.last_applied?.controller_fault_bits)
      && snapshot.last_applied.controller_fault_bits !== 0) {
      faults.push(
        `controller fault bits 0x${snapshot.last_applied.controller_fault_bits.toString(16)}`,
      );
    }
    if (snapshot.physical_emergency_stop_state === "engaged") {
      faults.push("physical E-stop engaged");
    } else if (snapshot.physical_emergency_stop_state === "faulted") {
      faults.push("physical E-stop input faulted");
    }
    return {
      label: faults.length ? faults.join(" · ") : "none reported",
      className: faults.length ? "fault" : "ready",
    };
  }

  function physicalStopView(snapshot) {
    const state = snapshot.physical_emergency_stop_state;
    return {
      label: words(state),
      className: state === "released" ? "ready"
        : state === "unavailable" ? "warn"
          : "fault",
    };
  }

  function cellAt(cells, metadata, xCell, yCell) {
    if (!(cells instanceof Uint8Array)
      || !Number.isFinite(xCell) || !Number.isFinite(yCell)
      || xCell < 0 || yCell < 0
      || xCell >= metadata.width || yCell >= metadata.height) {
      return "outside";
    }
    const x = Math.floor(xCell);
    const y = Math.floor(yCell);
    const value = cells[y * metadata.width + x];
    return value === 1 ? "free" : value === 2 ? "occupied" : "unknown";
  }

  const api = Object.freeze({
    parseConsoleSnapshot,
    parseRerunDiagnosticsUrl,
    formatNsMilliseconds,
    connectionView,
    authorityView,
    readinessView,
    mpcView,
    faultView,
    physicalStopView,
    cellAt,
  });
  globalThis.KikoOperatorConsoleModel = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})();
