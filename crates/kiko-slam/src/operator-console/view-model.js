(() => {
  "use strict";

  const DECIMAL_ID = /^[1-9][0-9]*$/;
  const DECIMAL_U64 = /^(0|[1-9][0-9]*)$/;
  const HEALTH = new Set(["ready", "degraded", "faulted", "unavailable"]);
  const REQUESTED_INFERENCE_BACKENDS = new Set([
    "auto", "cpu", "coreml_gpu", "cuda", "tensorrt",
  ]);
  const SELECTED_INFERENCE_BACKENDS = new Set([
    "cpu", "coreml_gpu", "cuda", "tensorrt",
  ]);
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
  const RUNTIME_AUTHORITY_KINDS = new Set([
    "production_external_interlocks",
    "attended_navigation_trial",
    "wheels_off_qualification",
  ]);
  const PHYSICAL_STOPS = new Set(["released", "engaged", "unavailable", "faulted"]);
  const STOP_CERTAINTY = new Set([
    "confirmed_applied_zero",
    "controller_reported_safe",
    "uncertain",
  ]);
  const QUALIFICATION_RUNTIME_INGRESS = new Set([
    "connected",
    "disconnected_stop_confirmed",
    "disconnected_stop_unconfirmed",
  ]);
  const QUALIFICATION_FRONTEND_STATE = new Set([
    "awaiting_connection",
    "connected",
    "disconnected",
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
    for (const component of ["stm32", "head", "eyes", "oak", "slam"]) {
      if (health[component] != null) {
        enumValue(health[component], HEALTH, `health.${component}`);
      }
    }
    return health;
  }

  function parseSlam(value) {
    const slam = optionalObject(value, "slam");
    if (!slam) return null;
    const inference = object(slam.inference, "slam.inference");
    for (const component of ["superpoint", "lightglue"]) {
      const selection = object(
        inference[component],
        `slam.inference.${component}`,
      );
      enumValue(
        selection.requested,
        REQUESTED_INFERENCE_BACKENDS,
        `slam.inference.${component}.requested`,
      );
      enumValue(
        selection.selected,
        SELECTED_INFERENCE_BACKENDS,
        `slam.inference.${component}.selected`,
      );
    }
    const counters = {};
    for (const field of [
      "started_pairs",
      "successful_pairs",
      "recoverable_failures",
      "fatal_failures",
    ]) {
      counters[field] = BigInt(exactString(slam[field], DECIMAL_U64, `slam.${field}`));
    }
    const completed = counters.successful_pairs
      + counters.recoverable_failures
      + counters.fatal_failures;
    if (completed > counters.started_pairs || counters.started_pairs - completed > 1n) {
      throw new Error("slam counters do not describe at most one in-flight pair");
    }
    const sourceArrival = slam.last_successful_source_arrival_host_monotonic_ns;
    const completion = slam.last_successful_completion_host_monotonic_ns;
    if ((sourceArrival == null) !== (completion == null)) {
      throw new Error("slam successful timestamps must be present together");
    }
    if (sourceArrival != null) {
      const source = BigInt(exactString(
        sourceArrival,
        DECIMAL_U64,
        "slam.last_successful_source_arrival_host_monotonic_ns",
      ));
      const completedAt = BigInt(exactString(
        completion,
        DECIMAL_U64,
        "slam.last_successful_completion_host_monotonic_ns",
      ));
      if (completedAt < source) {
        throw new Error("slam completion precedes its source arrival");
      }
    } else if (counters.successful_pairs !== 0n) {
      throw new Error("slam successful pairs require successful timestamps");
    }
    const rateWindow = optionalObject(slam.rate_window, "slam.rate_window");
    if (rateWindow) {
      const count = integer(
        rateWindow.successful_completions,
        2,
        64,
        "slam.rate_window.successful_completions",
      );
      const span = BigInt(exactString(
        rateWindow.span_ns,
        DECIMAL_U64,
        "slam.rate_window.span_ns",
      ));
      if (span === 0n || BigInt(count) > counters.successful_pairs) {
        throw new Error("slam rate window is inconsistent with successful pairs");
      }
    }
    return slam;
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

  function parseQualificationMotionGate(value) {
    const qualification = optionalObject(value, "wheels_off_qualification");
    if (!qualification) return null;
    if (qualification.schema_version !== 2) {
      throw new Error("wheels_off_qualification.schema_version is unsupported");
    }
    if (typeof qualification.motion_authority_enabled !== "boolean") {
      throw new Error(
        "wheels_off_qualification.motion_authority_enabled must be boolean",
      );
    }
    if (typeof qualification.stop_barrier_pending !== "boolean") {
      throw new Error(
        "wheels_off_qualification.stop_barrier_pending must be boolean",
      );
    }
    if (typeof qualification.software_safety_stop_latched !== "boolean") {
      throw new Error(
        "wheels_off_qualification.software_safety_stop_latched must be boolean",
      );
    }
    const runtimeIngressState = enumValue(
      qualification.runtime_ingress_state,
      QUALIFICATION_RUNTIME_INGRESS,
      "wheels_off_qualification.runtime_ingress_state",
    );
    const frontendState = enumValue(
      qualification.frontend_state,
      QUALIFICATION_FRONTEND_STATE,
      "wheels_off_qualification.frontend_state",
    );
    const ready = qualification.motion_authority_enabled
      && frontendState === "connected"
      && runtimeIngressState === "connected"
      && !qualification.stop_barrier_pending
      && !qualification.software_safety_stop_latched;
    return Object.freeze({
      motionAuthorityEnabled: qualification.motion_authority_enabled,
      frontendState,
      runtimeIngressState,
      stopBarrierPending: qualification.stop_barrier_pending,
      softwareSafetyStopLatched:
        qualification.software_safety_stop_latched,
      ready,
    });
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
    if (snapshot.schema_version !== 5) {
      throw new Error("unsupported snapshot schema");
    }
    const authorityKind = enumValue(
      snapshot.authority_kind,
      RUNTIME_AUTHORITY_KINDS,
      "snapshot.authority_kind",
    );
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
    const slam = parseSlam(snapshot.slam);
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
    const qualificationMotionGate = parseQualificationMotionGate(
      snapshot.wheels_off_qualification,
    );
    if (authorityKind === "wheels_off_qualification"
      && qualificationMotionGate == null) {
      throw new Error(
        "wheels-off authority requires wheels_off_qualification evidence",
      );
    }
    if (authorityKind !== "wheels_off_qualification"
      && qualificationMotionGate != null) {
      throw new Error(
        "wheels_off_qualification evidence contradicts snapshot.authority_kind",
      );
    }
    return {
      ...snapshot,
      slam,
      authority_kind: authorityKind,
      rerun_diagnostics_url: parseRerunDiagnosticsUrl(
        snapshot.rerun_diagnostics_url,
      ),
      qualification_motion_gate: qualificationMotionGate,
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
    const slam = snapshot.health?.slam || "unknown";
    const faulted = runtime?.kind === "faulted"
      || stm32 === "faulted"
      || oak === "faulted"
      || slam === "faulted";
    const ready = ["ready_stopped", "active"].includes(runtime?.kind)
      && stm32 === "ready"
      && oak === "ready"
      && slam === "ready";
    return {
      runtimeLabel,
      readinessLabel: terminal
        ? "terminal checkpoint · capture stop requested · sensor/map streams draining · finalizing exact warm-replay inputs · current-camera localization not claimed"
        : `runtime ${runtimeLabel} · STM32 ${stm32} · OAK ${oak} · SLAM ${slam}`,
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
    for (const component of ["stm32", "oak", "slam", "head", "eyes"]) {
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

  function qualificationMotionView(snapshot) {
    const gate = snapshot.qualification_motion_gate;
    if (!gate) return null;
    if (gate.softwareSafetyStopLatched) {
      return {
        ready: false,
        ownerLabel: "QUALIFICATION SAFETY STOP LATCHED",
        requestedOwnerLabel: "manual motion permanently locked for this process",
        modeLabel:
          "Manual qualification motion is locked by the one-way software safety stop.",
        readinessLabel: "manual motion locked · software safety stop latched",
        className: "fault",
      };
    }
    if (gate.frontendState !== "connected") {
      const awaiting = gate.frontendState === "awaiting_connection";
      return {
        ready: false,
        ownerLabel: awaiting
          ? "QUALIFICATION FRONTEND STARTING"
          : "QUALIFICATION FRONTEND DISCONNECTED",
        requestedOwnerLabel: awaiting
          ? "manual motion locked until frontend readiness"
          : "manual motion locked after frontend loss",
        modeLabel: awaiting
          ? "Manual qualification motion is locked until the loopback frontend is ready."
          : "Manual qualification motion is locked because the loopback frontend exited.",
        readinessLabel: awaiting
          ? "manual motion locked · frontend awaiting connection"
          : "manual motion locked · frontend disconnected",
        className: awaiting ? "warn" : "fault",
      };
    }
    if (!gate.motionAuthorityEnabled) {
      const additionalBlockers = [];
      if (gate.runtimeIngressState !== "connected") {
        additionalBlockers.push(
          `runtime ingress ${words(gate.runtimeIngressState)}`,
        );
      }
      if (gate.stopBarrierPending) additionalBlockers.push("stop barrier pending");
      const suffix = additionalBlockers.length
        ? ` · ${additionalBlockers.join(" · ")}`
        : "";
      return {
        ready: false,
        ownerLabel: "MOTION ATTESTATION PENDING",
        requestedOwnerLabel: "manual motion locked pending attended attestation",
        modeLabel:
          "Manual qualification motion is locked pending attended startup attestation.",
        readinessLabel:
          `manual motion locked · attended attestation pending${suffix}`,
        className: "warn",
      };
    }
    if (gate.runtimeIngressState !== "connected") {
      return {
        ready: false,
        ownerLabel: "QUALIFICATION INGRESS DISCONNECTED",
        requestedOwnerLabel: "manual motion locked by runtime ingress state",
        modeLabel:
          "Manual qualification motion is locked because runtime ingress is disconnected.",
        readinessLabel:
          `manual motion locked · runtime ingress ${words(gate.runtimeIngressState)}`,
        className: gate.runtimeIngressState === "disconnected_stop_unconfirmed"
          ? "fault"
          : "warn",
      };
    }
    if (gate.stopBarrierPending) {
      return {
        ready: false,
        ownerLabel: "QUALIFICATION STOP BARRIER PENDING",
        requestedOwnerLabel: "manual motion locked by stop barrier",
        modeLabel:
          "Manual qualification motion is locked until the stop barrier completes.",
        readinessLabel: "manual motion locked · stop barrier pending",
        className: "warn",
      };
    }
    return {
      ready: true,
      ownerLabel: null,
      requestedOwnerLabel: null,
      modeLabel: null,
      readinessLabel: null,
      className: "ready",
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

  const DRIVE_SAFETY_CONNECTIONS = new Set([
    "locked",
    "live",
    "stale",
    "disconnected",
  ]);
  const DRIVE_SAFETY_LIFECYCLE_LOSSES = new Set([
    "blur",
    "offline",
    "pagehide",
    "visibility_hidden",
  ]);
  const DRIVE_SAFETY_EFFECTS = Object.freeze({
    ensureDriveLoop: "ensure_drive_loop",
    releaseManual: "release_manual",
    releaseManualBestEffort: "release_manual_best_effort",
  });
  const DRIVE_SAFETY_STATES = new WeakSet();
  const TERMINAL_STOP_STATES = new WeakSet();

  function frozenDriveSafetyState({
    connectionKind,
    localInhibit,
    snapshotFresh,
    lastSnapshotRevision,
    lastSnapshotAdvanceAtMilliseconds,
  }) {
    enumValue(
      connectionKind,
      DRIVE_SAFETY_CONNECTIONS,
      "drive safety state.connectionKind",
    );
    if (typeof localInhibit !== "boolean"
      || typeof snapshotFresh !== "boolean") {
      throw new Error("drive safety state flags must be boolean");
    }
    if (typeof lastSnapshotRevision !== "bigint"
      || lastSnapshotRevision < 0n) {
      throw new Error("drive safety state revision must be an unsigned bigint");
    }
    if (lastSnapshotAdvanceAtMilliseconds != null
      && (!Number.isFinite(lastSnapshotAdvanceAtMilliseconds)
        || lastSnapshotAdvanceAtMilliseconds < 0)) {
      throw new Error("drive safety state advance time must be monotonic");
    }
    const state = Object.freeze({
      connectionKind,
      localInhibit,
      snapshotFresh,
      lastSnapshotRevision,
      lastSnapshotAdvanceAtMilliseconds,
    });
    DRIVE_SAFETY_STATES.add(state);
    return state;
  }

  function createDriveSafetyState() {
    return frozenDriveSafetyState({
      connectionKind: "locked",
      localInhibit: true,
      snapshotFresh: false,
      lastSnapshotRevision: 0n,
      lastSnapshotAdvanceAtMilliseconds: null,
    });
  }

  function driveSafetyResult(state, effects = [], detail = {}) {
    return Object.freeze({
      state,
      effects: Object.freeze(effects),
      ...detail,
    });
  }

  function driveSafetyStateWith(state, changes) {
    return frozenDriveSafetyState({ ...state, ...changes });
  }

  function reduceDriveSafety(state, event) {
    if (!DRIVE_SAFETY_STATES.has(state)) {
      throw new Error("drive safety transition requires a parsed state");
    }
    object(event, "drive safety event");
    switch (event.kind) {
      case "key_released": {
        const heldDirectionCount = integer(
          event.held_direction_count,
          0,
          4,
          "drive safety event.held_direction_count",
        );
        if (typeof event.desired_intent_present !== "boolean") {
          throw new Error(
            "drive safety event.desired_intent_present must be boolean",
          );
        }
        const effect = heldDirectionCount === 0
          || !event.desired_intent_present
          ? DRIVE_SAFETY_EFFECTS.releaseManual
          : DRIVE_SAFETY_EFFECTS.ensureDriveLoop;
        return driveSafetyResult(state, [effect]);
      }
      case "lifecycle_loss":
        enumValue(
          event.source,
          DRIVE_SAFETY_LIFECYCLE_LOSSES,
          "drive safety event.source",
        );
        return driveSafetyResult(
          state,
          [DRIVE_SAFETY_EFFECTS.releaseManualBestEffort],
        );
      case "local_inhibit":
        return driveSafetyResult(
          driveSafetyStateWith(state, { localInhibit: true }),
        );
      case "session_opened":
        return driveSafetyResult(
          driveSafetyStateWith(state, { localInhibit: false }),
        );
      case "locked":
        return driveSafetyResult(driveSafetyStateWith(state, {
          connectionKind: "locked",
          localInhibit: true,
          snapshotFresh: false,
        }));
      case "transport_failed":
        if (typeof event.detail !== "string" || event.detail.length === 0) {
          throw new Error("drive safety transport failure requires detail");
        }
        return driveSafetyResult(
          driveSafetyStateWith(state, {
            connectionKind: "disconnected",
            localInhibit: true,
            snapshotFresh: false,
          }),
          [DRIVE_SAFETY_EFFECTS.releaseManualBestEffort],
        );
      case "snapshot_observed": {
        if (typeof event.revision !== "bigint" || event.revision < 0n) {
          throw new Error("snapshot revision must be an unsigned bigint");
        }
        const now = finite(
          event.now_milliseconds,
          "drive safety event.now_milliseconds",
        );
        const staleAfter = finite(
          event.stale_after_milliseconds,
          "drive safety event.stale_after_milliseconds",
        );
        if (now < 0 || staleAfter <= 0) {
          throw new Error("snapshot timing bounds must be positive");
        }
        for (const field of [
          "local_safety_latched",
          "server_safety_latched",
        ]) {
          if (typeof event[field] !== "boolean") {
            throw new Error(`drive safety event.${field} must be boolean`);
          }
        }
        if (event.revision < state.lastSnapshotRevision) {
          throw new Error("snapshot revision regressed");
        }
        if (state.lastSnapshotAdvanceAtMilliseconds != null
          && now < state.lastSnapshotAdvanceAtMilliseconds) {
          throw new Error("snapshot observation clock regressed");
        }

        const snapshotAdvanced =
          event.revision > state.lastSnapshotRevision;
        let next = snapshotAdvanced
          ? driveSafetyStateWith(state, {
            lastSnapshotRevision: event.revision,
            lastSnapshotAdvanceAtMilliseconds: now,
            snapshotFresh: true,
          })
          : state;
        const lastAdvance = next.lastSnapshotAdvanceAtMilliseconds;
        const stale = lastAdvance == null
          || now - lastAdvance > staleAfter;
        if (stale) {
          const becameStale = next.snapshotFresh;
          next = driveSafetyStateWith(next, {
            connectionKind: "stale",
            localInhibit: true,
            snapshotFresh: false,
          });
          return driveSafetyResult(
            next,
            becameStale
              ? [DRIVE_SAFETY_EFFECTS.releaseManualBestEffort]
              : [],
            { snapshotAdvanced },
          );
        }
        next = driveSafetyStateWith(next, {
          connectionKind: "live",
          localInhibit:
            event.local_safety_latched || event.server_safety_latched,
          snapshotFresh: true,
        });
        return driveSafetyResult(next, [], { snapshotAdvanced });
      }
      default:
        throw new Error("drive safety event kind is unsupported");
    }
  }

  function requestTimeoutMessage(phase, timeoutMilliseconds) {
    enumValue(
      phase,
      new Set(["request", "response"]),
      "request timeout phase",
    );
    integer(
      timeoutMilliseconds,
      1,
      60_000,
      "request timeout milliseconds",
    );
    return phase === "request"
      ? `request timed out after ${timeoutMilliseconds} ms`
      : `request timed out after ${timeoutMilliseconds} ms while reading response`;
  }

  function qualificationMotionSessionDecision(
    qualification,
    motionReady,
    sessionMotionAuthorityCurrent,
  ) {
    for (const [field, value] of [
      ["qualification", qualification],
      ["motionReady", motionReady],
      ["sessionMotionAuthorityCurrent", sessionMotionAuthorityCurrent],
    ]) {
      if (typeof value !== "boolean") {
        throw new Error(`qualification motion session ${field} must be boolean`);
      }
    }
    const freshSessionRequired =
      qualification && motionReady && !sessionMotionAuthorityCurrent;
    return Object.freeze({
      freshSessionRequired,
      clearHeldInputBeforeHandshake: freshSessionRequired,
    });
  }

  function parsedConsoleSession(value, label) {
    if (value == null || typeof value !== "object" || Array.isArray(value)
      || typeof value.sessionId !== "string"
      || !DECIMAL_ID.test(value.sessionId)
      || typeof value.sessionCapability !== "string"
      || !/^[0-9a-f]{64}$/.test(value.sessionCapability)) {
      throw new Error(`${label} console session is invalid`);
    }
    return Object.freeze({
      sessionId: value.sessionId,
      sessionCapability: value.sessionCapability,
    });
  }

  function qualificationMotionSessionSwap(previous, replacement) {
    const retiringSession = parsedConsoleSession(previous, "previous");
    const activeSession = parsedConsoleSession(replacement, "replacement");
    if (retiringSession.sessionId === activeSession.sessionId) {
      throw new Error("qualification replacement session must be fresh");
    }
    return Object.freeze({ activeSession, retiringSession });
  }

  function softwareSafetyStopRetryDecision({
    attempt,
    maximumAttempts,
    confirmed,
    retryable,
    consoleUnlocked,
  }) {
    if (!Number.isInteger(attempt) || attempt < 1
      || !Number.isInteger(maximumAttempts) || maximumAttempts < 1
      || attempt > maximumAttempts) {
      throw new Error("software safety-stop retry attempt is invalid");
    }
    for (const [field, value] of [
      ["confirmed", confirmed],
      ["retryable", retryable],
      ["consoleUnlocked", consoleUnlocked],
    ]) {
      if (typeof value !== "boolean") {
        throw new Error(`software safety-stop ${field} must be boolean`);
      }
    }
    if (confirmed) return "confirmed";
    if (!consoleUnlocked || !retryable || attempt === maximumAttempts) {
      return "unavailable";
    }
    return "retry";
  }

  function frozenTerminalStopState(nextStep, completed) {
    const state = Object.freeze({ nextStep, completed });
    TERMINAL_STOP_STATES.add(state);
    return state;
  }

  function createTerminalStopState() {
    return frozenTerminalStopState("release_manual_exact_zero", false);
  }

  function reduceTerminalStop(state, outcome) {
    if (!TERMINAL_STOP_STATES.has(state) || state.completed) {
      throw new Error("terminal stop transition requires an active parsed state");
    }
    if (state.nextStep === "release_manual_exact_zero") {
      if (outcome === "release_confirmed") {
        return frozenTerminalStopState(null, true);
      }
      if (outcome === "release_unavailable") {
        return frozenTerminalStopState("terminal_stop_exact_zero", false);
      }
      throw new Error("release outcome is unsupported");
    }
    if (state.nextStep === "terminal_stop_exact_zero"
      && outcome === "terminal_stop_confirmed") {
      return frozenTerminalStopState(null, true);
    }
    throw new Error("terminal stop outcome is out of order");
  }

  const driveSafety = Object.freeze({
    effects: DRIVE_SAFETY_EFFECTS,
    createState: createDriveSafetyState,
    reduce: reduceDriveSafety,
    requestTimeoutMessage,
    createTerminalStopState,
    reduceTerminalStop,
  });

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
    qualificationMotionView,
    qualificationMotionSessionDecision,
    qualificationMotionSessionSwap,
    softwareSafetyStopRetryDecision,
    cellAt,
    driveSafety,
  });
  globalThis.KikoOperatorConsoleModel = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})();
