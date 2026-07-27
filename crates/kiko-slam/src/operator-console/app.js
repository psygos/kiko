(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));
  const model = globalThis.KikoOperatorConsoleModel;
  if (!model) throw new Error("operator-console view model is unavailable");
  const driveSafety = model.driveSafety;
  if (!driveSafety) throw new Error("operator-console drive safety model is unavailable");
  const API_REQUEST_TIMEOUT_MILLISECONDS = 750;
  const PRODUCTION_CONTROL_PROFILE_KIND = "production_body_frame_si";
  const QUALIFICATION_CONTROL_PROFILE_KIND =
    "wheels_off_raw_timer_pwm_qualification";
  const QUALIFICATION_INTENT_ENDPOINT =
    "/api/v1/wheels-off-qualification/intents";
  const QUALIFICATION_BANNER =
    "WHEELS-OFF QUALIFICATION — RAW TIMER DUTY ONLY — AUTONOMOUS ACTUATION DISABLED";
  // Production publishes every control period and the browser polls every
  // 300 ms. Five unchanged polls are treated as stale observational state.
  const SNAPSHOT_STALE_AFTER_MILLISECONDS = 1500;
  class ApiError extends Error {
    constructor(status, code) {
      super(code);
      this.name = "ApiError";
      this.status = status;
      this.code = code;
    }
  }
  const state = {
    capability: null,
    sessionId: null,
    sessionCapability: null,
    sequence: 0n,
    idempotency: 0n,
    held: new Set(),
    driveGeneration: 0,
    driveLoopRunning: false,
    manualBegun: false,
    driveSafety: driveSafety.createState(),
    localSafetyLatched: false,
    serverSafetyLatched: false,
    controlProfile: null,
    controlProfileFingerprint: null,
    snapshot: null,
    grid: null,
    gridRaster: null,
    gridKey: null,
    gridLoadGeneration: 0,
    gridError: null,
    gridFailureKey: null,
    gridRetryAfterPerformanceMilliseconds: 0,
    lastResponseId: null,
    polling: false,
  };

  const canvas = $("map");
  const ctx = canvas.getContext("2d", { alpha: false });
  const pendingResponseBodies = new WeakMap();
  let toastTimer;

  function applyDriveSafetyEvent(event) {
    const transition = driveSafety.reduce(state.driveSafety, event);
    state.driveSafety = transition.state;
    return transition;
  }

  async function executeDriveSafetyEffects(effects) {
    for (const effect of effects) {
      if (effect === driveSafety.effects.ensureDriveLoop) {
        ensureDriveLoop();
      } else if (effect === driveSafety.effects.releaseManual) {
        await releaseManual();
      } else if (effect === driveSafety.effects.releaseManualBestEffort) {
        await releaseManual({ bestEffort: true });
      } else {
        throw new Error("unsupported drive safety effect");
      }
    }
  }

  function errorDetail(error) {
    return error instanceof Error && error.message.length > 0
      ? error.message
      : "unknown console transport failure";
  }

  async function failClosedForTransport(error) {
    const detail = errorDetail(error);
    const transition = applyDriveSafetyEvent({
      kind: "transport_failed",
      detail,
    });
    setConnectionView(transition.state.connectionKind, detail);
    updateControlAvailability();
    await executeDriveSafetyEffects(transition.effects);
  }

  function toast(message, fault = false) {
    const node = $("toast");
    node.textContent = message;
    node.style.borderColor = fault ? "var(--red)" : "var(--line)";
    node.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => node.classList.remove("show"), 2800);
  }

  function setEvidenceClass(node, className) {
    node.classList.remove("ready", "warn", "fault");
    const normalized = ["degraded", "unavailable"].includes(className)
      ? "warn"
      : className;
    if (["ready", "warn", "fault"].includes(normalized)) {
      node.classList.add(normalized);
    }
  }

  function setConnectionView(kind, detail = null) {
    const view = model.connectionView(kind);
    const pill = $("connection-pill");
    pill.textContent = view.pill;
    pill.className = view.className;
    const overlay = $("map-freshness");
    overlay.textContent = view.overlay;
    overlay.classList.toggle("hidden", !view.overlay);
    $("console").classList.toggle("telemetry-stale", kind !== "live");
    const revision = state.snapshot?.revision || "—";
    const observed = state.snapshot?.telemetry_observed_at_host_monotonic_ns || "unknown";
    $("telemetry-state").textContent = detail
      ? `${view.telemetryPrefix} · ${detail}`
      : `${view.telemetryPrefix} · snapshot ${revision} · host monotonic ${observed} ns`;
    setEvidenceClass(
      $("telemetry-state"),
      kind === "live" ? "ready" : kind === "locked" ? "warn" : "fault",
    );
  }

  function sessionHeaders(headers = {}) {
    const output = new Headers(headers);
    if (state.sessionCapability) {
      output.set("X-Kiko-Session-Capability", state.sessionCapability);
    }
    if (state.sessionId) output.set("X-Kiko-Session-Id", state.sessionId);
    return output;
  }

  async function api(path, options = {}) {
    if (!state.capability) throw new Error("console is locked");
    const {
      timeoutMilliseconds = API_REQUEST_TIMEOUT_MILLISECONDS,
      ...fetchOptions
    } = options;
    const headers = sessionHeaders(fetchOptions.headers);
    headers.set("X-Kiko-Console-Capability", state.capability);
    if (fetchOptions.body) headers.set("Content-Type", "application/json");
    const abort = new AbortController();
    const timeout = setTimeout(() => abort.abort(), timeoutMilliseconds);
    let responseBodyOwnsTimeout = false;
    try {
      const response = await fetch(path, {
        ...fetchOptions,
        headers,
        cache: "no-store",
        signal: abort.signal,
      });
      if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try { detail = (await response.json()).error || detail; } catch (_) {}
        throw new ApiError(response.status, detail);
      }
      // `fetch()` resolves after headers, not after the response body. Keep the
      // abort armed so snapshot/grid/response decoding is bounded as well.
      pendingResponseBodies.set(response, { abort, timeout, timeoutMilliseconds });
      responseBodyOwnsTimeout = true;
      return response;
    } catch (error) {
      if (abort.signal.aborted) {
        throw new Error(
          driveSafety.requestTimeoutMessage("request", timeoutMilliseconds),
        );
      }
      throw error;
    } finally {
      if (!responseBodyOwnsTimeout) clearTimeout(timeout);
    }
  }

  async function consumeResponse(response, consume) {
    const pending = pendingResponseBodies.get(response);
    try {
      return await consume();
    } catch (error) {
      if (pending?.abort.signal.aborted) {
        throw new Error(
          driveSafety.requestTimeoutMessage(
            "response",
            pending.timeoutMilliseconds,
          ),
        );
      }
      throw error;
    } finally {
      if (pending) {
        clearTimeout(pending.timeout);
        pendingResponseBodies.delete(response);
      }
    }
  }

  function responseJson(response) {
    return consumeResponse(response, () => response.json());
  }

  function responseArrayBuffer(response) {
    return consumeResponse(response, () => response.arrayBuffer());
  }

  async function openSession() {
    const response = await api("/api/v1/sessions", {
      method: "POST",
      body: JSON.stringify({ schema_version: 1, source: "operator" }),
    });
    const body = await responseJson(response);
    if (typeof body.session_id !== "string"
      || !/^[1-9][0-9]*$/.test(body.session_id)
      || typeof body.session_capability !== "string"
      || !/^[0-9a-f]{64}$/.test(body.session_capability)) {
      throw new Error("invalid typed session response");
    }
    state.sessionId = body.session_id;
    state.sessionCapability = body.session_capability;
    state.sequence = 0n;
    state.idempotency = 0n;
  }

  function qualificationProfile() {
    return state.controlProfile?.kind === QUALIFICATION_CONTROL_PROFILE_KIND
      ? state.controlProfile
      : null;
  }

  function intentEndpoint() {
    return qualificationProfile()?.intent_endpoint || "/api/v1/intents";
  }

  function nextEnvelope(intent) {
    state.sequence += 1n;
    state.idempotency += 1n;
    const envelope = {
      schema_version: 1,
      session_id: state.sessionId,
      source_sequence: state.sequence.toString(),
      idempotency_key: state.idempotency.toString(),
      intent,
    };
    if (qualificationProfile()) {
      envelope.control_profile = QUALIFICATION_CONTROL_PROFILE_KIND;
    }
    return envelope;
  }

  async function submit(intent, { quiet = false, keepalive = false } = {}) {
    if (!state.sessionId || !state.sessionCapability) {
      throw new Error("no active console session");
    }
    const response = await api(intentEndpoint(), {
      method: "POST",
      keepalive,
      body: JSON.stringify(nextEnvelope(intent)),
    });
    const body = await responseJson(response);
    const requestId = typeof body.downstream_request_id === "string"
      ? body.downstream_request_id
      : body.event_id;
    if (typeof requestId !== "string" || !/^[1-9][0-9]*$/.test(requestId)) {
      throw new Error("invalid typed request identity");
    }
    state.lastResponseId = qualificationProfile() ? null : requestId;
    $("request-state").textContent =
      `#${requestId} ${body.state || "accepted"} · NOT APPLIED`;
    if (!quiet) {
      toast(`Request #${requestId} accepted; not yet applied.`);
    }
    return { ...body, request_id: requestId };
  }

  function clearDriveUi(clearHeld = true) {
    state.driveGeneration += 1;
    if (clearHeld) state.held.clear();
    document.querySelectorAll("[data-drive]")
      .forEach((button) => button.classList.remove("active"));
  }

  async function responseRecord(id) {
    const response = await api(`/api/v1/responses/${id}`);
    return responseJson(response);
  }

  async function awaitExactStop(id, timeoutMilliseconds = 3500) {
    const deadline = performance.now() + timeoutMilliseconds;
    while (performance.now() < deadline) {
      if (qualificationProfile()) {
        const response = await api("/api/v1/snapshot");
        const snapshot = await responseJson(response);
        applyControlProfile(snapshot);
        const qualification = snapshot.wheels_off_qualification;
        const completion = qualification?.last_terminal_completion;
        if (completion?.event_id === id
          && completion.observed_applied_zero?.applied_left_timer_pwm_percent === 0
          && completion.observed_applied_zero?.applied_right_timer_pwm_percent === 0) {
          return completion;
        }
        if (qualification?.runtime_ingress_state === "disconnected_stop_unconfirmed") {
          throw new Error("qualification runtime disconnected; applied zero is unconfirmed");
        }
        await delay(50);
        continue;
      }
      const record = await responseRecord(id);
      if (record.state === "completed" && record.applied === true
        && record.exact_receipt != null) {
        return record;
      }
      if (record.state === "rejected") {
        throw new Error(`stop rejected: ${record.rejection_code || "unknown"}`);
      }
      await delay(50);
    }
    throw new Error("timed out waiting for exact applied-zero evidence");
  }

  async function sendRelease({ quiet = true, keepalive = false } = {}) {
    const response = await submit(
      { kind: "release_manual" },
      { quiet, keepalive },
    );
    state.manualBegun = false;
    return response;
  }

  async function releaseManual({
    bestEffort = false,
    awaitAppliedZero = false,
    clearHeld = true,
  } = {}) {
    clearDriveUi(clearHeld);
    const couldOwnManual = state.manualBegun || state.driveLoopRunning;
    if (!couldOwnManual || !state.sessionId) return null;
    try {
      const response = await sendRelease({ quiet: true, keepalive: bestEffort });
      if (awaitAppliedZero && !bestEffort) {
        await awaitExactStop(response.request_id);
      }
      return response;
    } catch (error) {
      if (!bestEffort) toast(error.message, true);
      return null;
    }
  }

  function admittedManualCommand() {
    const qualification = qualificationProfile();
    if (qualification) {
      return {
        kind: "raw_timer_pwm",
        patterns: qualification.patterns,
      };
    }
    const envelope = state.snapshot?.manual_command_envelope;
    if (!envelope) return null;
    const forward = envelope.command_forward_velocity_mps;
    const yaw = envelope.command_yaw_rate_rad_s;
    if (!Number.isFinite(forward) || forward <= 0
      || !Number.isFinite(yaw) || yaw <= 0) {
      return null;
    }
    return { kind: "body_frame_si", forward, yaw };
  }

  function desiredManualIntent() {
    const admitted = admittedManualCommand();
    if (!admitted) return null;
    if (admitted.kind === "raw_timer_pwm") {
      const direction = Array.from(state.held).at(-1);
      const patternName = {
        forward: "both_positive",
        backward: "both_negative",
        left: "left_negative_right_positive",
        right: "left_positive_right_negative",
      }[direction];
      const pattern = admitted.patterns[patternName];
      if (!pattern) return null;
      return {
        kind: "manual_pwm",
        left_timer_pwm_percent: pattern.left_timer_pwm_percent,
        right_timer_pwm_percent: pattern.right_timer_pwm_percent,
      };
    }
    let forward = 0;
    let yaw = 0;
    if (state.held.has("forward")) forward += admitted.forward;
    if (state.held.has("backward")) forward -= admitted.forward;
    if (state.held.has("left")) yaw += admitted.yaw;
    if (state.held.has("right")) yaw -= admitted.yaw;
    if (forward === 0 && yaw === 0) return null;
    return {
      kind: "manual_velocity",
      forward_velocity_mps: forward,
      yaw_rate_rad_s: yaw,
    };
  }

  async function driveLoop(generation) {
    state.driveLoopRunning = true;
    try {
      const begin = await submit({ kind: "begin_manual" }, { quiet: true });
      if (generation !== state.driveGeneration || state.driveSafety.localInhibit) {
        state.manualBegun = true;
        await sendRelease({ quiet: true });
        return;
      }
      state.manualBegun = true;
      let unused = begin;
      while (generation === state.driveGeneration
        && !state.driveSafety.localInhibit
        && qualificationMotionReady()
        && state.held.size) {
        const manualIntent = desiredManualIntent();
        if (!manualIntent) {
          clearDriveUi(false);
          await sendRelease({ quiet: true });
          return;
        }
        unused = await submit(manualIntent, { quiet: true });
        if (generation !== state.driveGeneration) break;
        await delay(50);
      }
      void unused;
    } catch (error) {
      toast(errorDetail(error), true);
      if (error instanceof ApiError) {
        if (generation === state.driveGeneration) clearDriveUi();
      } else {
        await failClosedForTransport(error);
      }
    } finally {
      state.driveLoopRunning = false;
      if (generation !== state.driveGeneration && state.manualBegun) {
        try { await sendRelease({ quiet: true }); } catch (_) {}
      }
      // A release/re-grab can arrive while this generation is still awaiting
      // HTTP completion. Once the old release is ordered, immediately service
      // the newly held desired state without requiring another key press.
      if (state.held.size && !state.driveSafety.localInhibit) ensureDriveLoop();
    }
  }

  function ensureDriveLoop() {
    if (state.driveLoopRunning
      || state.driveSafety.localInhibit
      || !qualificationMotionReady()
      || !state.held.size) return;
    if (!admittedManualCommand()) {
      toast(
        qualificationProfile()
          ? "Raw qualification patterns are unavailable or invalid."
          : "Manual drive is unavailable until an admitted SI envelope is published.",
        true,
      );
      clearDriveUi();
      return;
    }
    // Opposing held directions can cancel to an exact zero intention. Stay
    // released until the next key/pointer transition instead of repeatedly
    // acquiring and releasing manual authority while the input is unchanged.
    if (!desiredManualIntent()) return;
    const generation = ++state.driveGeneration;
    void driveLoop(generation);
  }

  function startDrive(direction) {
    if (state.driveSafety.localInhibit || !qualificationMotionReady()) return;
    state.held.add(direction);
    document.querySelector(`[data-drive="${direction}"]`)?.classList.add("active");
    ensureDriveLoop();
  }

  const keyDirection = {
    ArrowUp: "forward", KeyW: "forward",
    ArrowDown: "backward", KeyS: "backward",
    ArrowLeft: "left", KeyA: "left",
    ArrowRight: "right", KeyD: "right",
  };

  document.addEventListener("keydown", (event) => {
    const direction = keyDirection[event.code];
    if (!direction || event.repeat || event.target instanceof HTMLInputElement) return;
    event.preventDefault();
    startDrive(direction);
  });
  document.addEventListener("keyup", (event) => {
    const direction = keyDirection[event.code];
    if (!direction) return;
    event.preventDefault();
    state.held.delete(direction);
    document.querySelector(`[data-drive="${direction}"]`)?.classList.remove("active");
    const transition = applyDriveSafetyEvent({
      kind: "key_released",
      held_direction_count: state.held.size,
      desired_intent_present:
        state.held.size > 0 && desiredManualIntent() != null,
    });
    if (transition.effects[0] === driveSafety.effects.releaseManual
      && state.held.size > 0) {
      void releaseManual({ clearHeld: false });
      return;
    }
    void executeDriveSafetyEffects(transition.effects);
  });

  function releaseForLifecycleLoss(source) {
    const transition = applyDriveSafetyEvent({ kind: "lifecycle_loss", source });
    void executeDriveSafetyEffects(transition.effects);
  }

  window.addEventListener("blur", () => releaseForLifecycleLoss("blur"));
  window.addEventListener("offline", () => releaseForLifecycleLoss("offline"));
  window.addEventListener("pagehide", () => releaseForLifecycleLoss("pagehide"));
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") {
      releaseForLifecycleLoss("visibility_hidden");
    }
  });

  document.querySelectorAll("[data-drive]").forEach((button) => {
    const direction = button.dataset.drive;
    button.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      button.setPointerCapture(event.pointerId);
      startDrive(direction);
    });
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) {
      button.addEventListener(name, () => void releaseManual());
    }
  });
  $("manual-release").addEventListener("click", () => void releaseManual());

  async function ensureExactStopped() {
    let terminalStop = driveSafety.createTerminalStopState();
    while (!terminalStop.completed) {
      if (terminalStop.nextStep === "release_manual_exact_zero") {
        const release = await releaseManual({ awaitAppliedZero: true });
        terminalStop = driveSafety.reduceTerminalStop(
          terminalStop,
          release ? "release_confirmed" : "release_unavailable",
        );
      } else if (terminalStop.nextStep === "terminal_stop_exact_zero") {
        const stop = await submit({ kind: "stop" }, { quiet: true });
        await awaitExactStop(stop.request_id);
        terminalStop = driveSafety.reduceTerminalStop(
          terminalStop,
          "terminal_stop_confirmed",
        );
      } else {
        throw new Error("unsupported terminal stop step");
      }
    }
  }

  document.querySelectorAll("[data-intent]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        const intent = button.dataset.intent;
        if (state.driveSafety.localInhibit
          && ["arm", "autonomous_frontier_explore", "save_map"].includes(intent)) {
          throw new Error("fresh, non-inhibited runtime state is required");
        }
        if (intent === "stop") {
          await ensureExactStopped();
        } else if (intent === "disarm") {
          clearDriveUi();
          const response = await submit({ kind: "disarm" });
          await awaitExactStop(response.request_id);
        } else {
          await ensureExactStopped();
          await submit({ kind: intent });
        }
      } catch (error) {
        toast(error.message, true);
      }
    });
  });

  function applyLocalSafetyInhibit() {
    state.localSafetyLatched = true;
    applyDriveSafetyEvent({ kind: "local_inhibit" });
    clearDriveUi();
    $("software-stop").classList.add("latched");
    $("software-stop").disabled = true;
    updateControlAvailability();
  }

  function renderSafetyState(snapshot = state.snapshot) {
    const signal = snapshot?.software_safety_signal_state;
    if (state.serverSafetyLatched || snapshot?.software_safety_stop_latched === true) {
      state.serverSafetyLatched = true;
      state.manualBegun = false;
      const detail = {
        pending_runtime_drain: "server latch accepted; waiting for runtime drain",
        runtime_drained_awaiting_completion:
          "runtime drained; waiting for exact controller completion",
        completed_fault_latched: "runtime completion recorded; fault remains latched",
        runtime_adapter_disconnected:
          "server latched, but runtime-adapter completion is unavailable",
      }[signal] || "server latch accepted; runtime completion not yet observed";
      $("safety-state").textContent =
        `SOFTWARE STOP: ${detail}. No remote reset exists. Physical E-stop remains separate.`;
    } else if (state.localSafetyLatched) {
      $("safety-state").textContent =
        "LOCAL BROWSER CONTROLS INHIBITED. Server latch is unconfirmed; release fallback and the server deadman remain authoritative. Physical E-stop remains separate.";
    } else {
      $("safety-state").textContent =
        "Not latched. The independent physical emergency stop remains separate.";
    }
  }

  $("software-stop").addEventListener("click", async () => {
    applyLocalSafetyInhibit();
    renderSafetyState();
    try {
      await submit({ kind: "software_safety_stop" });
      state.serverSafetyLatched = true;
      state.manualBegun = false;
      renderSafetyState();
    } catch (error) {
      if (error instanceof ApiError
        && error.status === 423
        && error.code === "software_safety_stop_latched") {
        state.serverSafetyLatched = true;
        state.manualBegun = false;
        state.lastResponseId = null;
        $("request-state").textContent =
          "software safety stop already latched outside this session · no session-scoped receipt";
        renderSafetyState();
        toast("Software safety stop was already latched outside this session.");
        return;
      }
      // Delivery may have failed before the server latch was reached. Preserve
      // the one-way local inhibit, but still attempt the ordinary manual
      // release path; if the latch did arrive, the server will reject it.
      await releaseManual({ bestEffort: true });
      renderSafetyState();
      toast(
        `Local controls inhibited; server latch status unknown: ${error.message}`,
        true,
      );
    }
  });

  function health(value) {
    return value == null ? "unknown" : value;
  }

  function exactQualificationPattern(pattern, left, right, maximum) {
    return pattern != null
      && Number.isInteger(pattern.left_timer_pwm_percent)
      && Number.isInteger(pattern.right_timer_pwm_percent)
      && Math.abs(pattern.left_timer_pwm_percent) <= maximum
      && Math.abs(pattern.right_timer_pwm_percent) <= maximum
      && pattern.left_timer_pwm_percent === left
      && pattern.right_timer_pwm_percent === right;
  }

  function parsedControlProfile(snapshot) {
    const raw = snapshot.control_profile;
    if (raw == null) {
      // Compatibility with the already-published production schema. A
      // qualification process is never allowed to omit its explicit profile.
      return Object.freeze({ kind: PRODUCTION_CONTROL_PROFILE_KIND });
    }
    if (raw.kind === PRODUCTION_CONTROL_PROFILE_KIND) {
      return Object.freeze({ kind: PRODUCTION_CONTROL_PROFILE_KIND });
    }
    if (raw.kind !== QUALIFICATION_CONTROL_PROFILE_KIND
      || raw.banner !== QUALIFICATION_BANNER
      || raw.command_units !== "signed_timer_duty_percent"
      || raw.required_wheel_state !== "removed"
      || raw.autonomous_actuation !== "disabled_shadow_only"
      || raw.intent_endpoint !== QUALIFICATION_INTENT_ENDPOINT
      || !Number.isInteger(raw.maximum_abs_timer_pwm_percent)
      || raw.maximum_abs_timer_pwm_percent < 1
      || raw.maximum_abs_timer_pwm_percent > 100
      || !Number.isInteger(raw.manual_test_magnitude_timer_pwm_percent)
      || raw.manual_test_magnitude_timer_pwm_percent < 1
      || raw.manual_test_magnitude_timer_pwm_percent
        > raw.maximum_abs_timer_pwm_percent
      || !Number.isInteger(raw.manual_deadman_ms)
      || raw.manual_deadman_ms < 1
      || raw.manual_deadman_ms > 5000) {
      throw new Error("invalid or unsupported qualification control profile");
    }
    const magnitude = raw.manual_test_magnitude_timer_pwm_percent;
    const maximum = raw.maximum_abs_timer_pwm_percent;
    const patterns = raw.patterns;
    if (!exactQualificationPattern(
      patterns?.both_positive,
      magnitude,
      magnitude,
      maximum,
    ) || !exactQualificationPattern(
      patterns?.both_negative,
      -magnitude,
      -magnitude,
      maximum,
    ) || !exactQualificationPattern(
      patterns?.left_negative_right_positive,
      -magnitude,
      magnitude,
      maximum,
    ) || !exactQualificationPattern(
      patterns?.left_positive_right_negative,
      magnitude,
      -magnitude,
      maximum,
    )) {
      throw new Error("qualification test patterns do not match their typed magnitude");
    }
    const qualification = snapshot.wheels_off_qualification;
    if (qualification?.schema_version !== 2
      || qualification.control_profile?.kind !== QUALIFICATION_CONTROL_PROFILE_KIND
      || snapshot.qualification_motion_gate == null) {
      throw new Error("qualification snapshot projection is absent or mismatched");
    }
    return Object.freeze(raw);
  }

  function renderControlProfile(profile) {
    const qualification = profile.kind === QUALIFICATION_CONTROL_PROFILE_KIND;
    const banner = $("control-profile-banner");
    const domain = $("manual-domain-label");
    const hint = $("manual-control-hint");
    if (!qualification) {
      banner.classList.add("hidden");
      domain.textContent = "BODY-FRAME SI CONTROL";
      hint.textContent =
        "Hold Arrow/WASD. Browser release is best-effort; the server's monotonic deadman is authoritative.";
      const labels = { forward: "W", backward: "S", left: "A", right: "D" };
      document.querySelectorAll("[data-drive]").forEach((button) => {
        button.querySelector("small").textContent = labels[button.dataset.drive];
        button.removeAttribute("title");
      });
      document.querySelectorAll("[data-intent]").forEach((button) => {
        button.classList.remove("hidden");
      });
      return;
    }
    banner.classList.remove("hidden");
    $("control-profile-title").textContent = profile.banner;
    $("control-profile-detail").textContent =
      `±${profile.maximum_abs_timer_pwm_percent}% admitted · ${profile.manual_test_magnitude_timer_pwm_percent}% raw patterns · ${profile.manual_deadman_ms} ms deadman · MPC is observational only`;
    domain.textContent = "SIGNED TIMER DUTY / QUALIFICATION ONLY";
    hint.textContent =
      "Wheels must remain removed. Arrow/WASD select explicit left/right electrical sign patterns; they do not claim calibrated travel direction. Release and connection loss enter the server stop barrier.";
    const labels = {
      forward: "L+ R+",
      backward: "L− R−",
      left: "L− R+",
      right: "L+ R−",
    };
    document.querySelectorAll("[data-drive]").forEach((button) => {
      button.querySelector("small").textContent = labels[button.dataset.drive];
      button.title =
        `${labels[button.dataset.drive]} · ${profile.manual_test_magnitude_timer_pwm_percent}% signed timer duty`;
    });
    document.querySelectorAll("[data-intent]").forEach((button) => {
      button.classList.toggle("hidden", button.dataset.intent !== "stop");
    });
  }

  function applyControlProfile(snapshot) {
    const profile = parsedControlProfile(snapshot);
    const fingerprint = JSON.stringify(profile);
    if (state.controlProfileFingerprint != null
      && state.controlProfileFingerprint !== fingerprint) {
      applyDriveSafetyEvent({ kind: "local_inhibit" });
      throw new Error("control profile changed during this authenticated boot");
    }
    if (state.controlProfileFingerprint == null) {
      state.controlProfile = profile;
      state.controlProfileFingerprint = fingerprint;
      renderControlProfile(profile);
    }
  }

  function qualificationMotionReady() {
    if (!qualificationProfile()) return true;
    return state.snapshot?.qualification_motion_gate?.ready === true;
  }

  function updateControlAvailability() {
    const qualificationReady = qualificationMotionReady();
    const terminal = state.snapshot?.terminal != null;
    const production = qualificationProfile() == null;
    const sensorMotionReady = !production
      || state.snapshot?.health?.oak === "ready";
    const enabled = !terminal
      && !state.driveSafety.localInhibit
      && qualificationReady
      && sensorMotionReady
      && admittedManualCommand() != null;
    document.querySelectorAll("[data-drive]").forEach((button) => {
      button.disabled = !enabled;
    });
    const localizedMap = state.snapshot?.map?.localization === "localized"
      && state.snapshot?.map?.grid != null;
    const mapAvailable = state.snapshot?.map?.grid != null;
    // Stop, disarm, and map-only remain available during stale telemetry
    // because they reduce motion. New authority/motion and persistence require
    // a fresh observation; Save Map also requires an exact published grid.
    const availability = {
      arm: production && !terminal && !state.driveSafety.localInhibit,
      disarm: production && !terminal,
      autonomous_map_only: production && !terminal,
      autonomous_frontier_explore:
        production
        && !terminal
        && !state.driveSafety.localInhibit
        && sensorMotionReady
        && localizedMap,
      save_map:
        production && !terminal && !state.driveSafety.localInhibit && mapAvailable,
      stop: !terminal,
    };
    document.querySelectorAll("[data-intent]").forEach((button) => {
      button.disabled = availability[button.dataset.intent] !== true;
    });
    if (state.localSafetyLatched) {
      document.querySelectorAll("[data-intent]").forEach((button) => {
        button.disabled = true;
      });
    }
    canvas.classList.toggle(
      "control-inhibited",
      state.driveSafety.localInhibit || !qualificationReady,
    );
  }

  function renderGlobalRequestEvidence(snapshot) {
    const qualification = snapshot.wheels_off_qualification;
    const requested = snapshot.last_requested;
    const displayedRequested = qualificationProfile()
      ? qualification?.last_requested
      : requested;
    const requestedId =
      displayedRequested?.downstream_request_id || displayedRequested?.event_id;
    $("global-request-state").textContent =
      displayedRequested && typeof requestedId === "string"
        && typeof displayedRequested.kind === "string"
        ? `#${requestedId} · ${displayedRequested.kind.replaceAll("_", " ")}`
        : "none";

    const actuation = snapshot.last_requested_actuation;
    if (!qualificationProfile()) {
      if (!actuation
        || !Number.isInteger(actuation.left_timer_pwm_percent)
        || !Number.isInteger(actuation.right_timer_pwm_percent)) {
        $("actuation-request-state").textContent = "none / no checked actuation";
        return;
      }
      const request = typeof actuation.downstream_request_id === "string"
        ? `#${actuation.downstream_request_id}`
        : "periodic";
      const decision = typeof actuation.decision_id === "string"
        ? `decision ${actuation.decision_id}`
        : "decision unknown";
      $("actuation-request-state").textContent =
        `${request} · ${decision} · [${actuation.left_timer_pwm_percent}, ${actuation.right_timer_pwm_percent}]%`;
      return;
    }
    const displayedActuation = displayedRequested?.requested_pwm;
    if (!displayedActuation
      || !Number.isInteger(displayedActuation.left_timer_pwm_percent)
      || !Number.isInteger(displayedActuation.right_timer_pwm_percent)) {
      $("actuation-request-state").textContent = "none / no checked actuation";
      return;
    }
    const request = typeof displayedActuation.downstream_request_id === "string"
      ? `#${displayedActuation.downstream_request_id}`
      : typeof requestedId === "string" ? `#${requestedId}` : "periodic";
    const decision = typeof displayedActuation.decision_id === "string"
      ? `decision ${displayedActuation.decision_id}`
      : "decision unknown";
    $("actuation-request-state").textContent =
      `${request} · ${decision} · [${displayedActuation.left_timer_pwm_percent}, ${displayedActuation.right_timer_pwm_percent}]%`;
  }

  function renderAuthorityAndPipeline(snapshot) {
    const readiness = model.readinessView(snapshot);
    const authority = model.authorityView(snapshot, state.sessionId);
    const qualificationAuthority =
      snapshot.wheels_off_qualification?.manual_authority;
    const qualificationMotion = qualificationProfile()
      ? model.qualificationMotionView(snapshot)
      : null;
    const ownerPill = $("owner-pill");
    if (qualificationProfile()) {
      if (qualificationMotion && !qualificationMotion.ready) {
        ownerPill.textContent = qualificationMotion.ownerLabel;
        ownerPill.className = `pill ${qualificationMotion.className}`;
        $("requested-owner-state").textContent =
          qualificationMotion.requestedOwnerLabel;
      } else {
        ownerPill.textContent = qualificationAuthority
          ? `${qualificationAuthority.source} · raw PWM · generation ${qualificationAuthority.authority_generation}`
          : "NO QUALIFICATION AUTHORITY";
        ownerPill.className = `pill ${qualificationAuthority ? "warn" : "unknown"}`;
        $("requested-owner-state").textContent = qualificationAuthority
          ? "qualification-only manual authority"
          : "none";
      }
    } else {
      ownerPill.textContent = authority.actualLabel.toUpperCase();
      ownerPill.className = `pill ${snapshot.actual_authority ? "ready" : "unknown"}`;
      $("requested-owner-state").textContent = authority.requestedLabel;
    }
    const activeMode = authority.activeMode;
    $("mode-state").textContent = snapshot.terminal
      ? "FINALIZING RESTART CHECKPOINT: capture stop is requested, sensor/map streams are draining, control is ending, and the exact final occupancy plus finalized session will be selected after drain. This does not claim current-camera localization."
      : qualificationMotion && !qualificationMotion.ready
        ? qualificationMotion.modeLabel
        : activeMode
          ? `Published active mode: ${activeMode.replaceAll("_", " ")}.`
          : `Runtime: ${readiness.runtimeLabel}; no active motion mode is published.`;
    document.querySelectorAll("[data-intent]").forEach((button) => {
      button.classList.toggle(
        "mode-active",
        button.dataset.intent === authority.activeIntent,
      );
    });
    $("readiness-state").textContent =
      qualificationMotion && !qualificationMotion.ready
        ? qualificationMotion.readinessLabel
        : readiness.readinessLabel;
    setEvidenceClass(
      $("readiness-state"),
      qualificationMotion && !qualificationMotion.ready
        ? qualificationMotion.className
        : readiness.className,
    );

    const runtimePill = $("runtime-pill");
    runtimePill.textContent = readiness.runtimeLabel.toUpperCase();
    runtimePill.className = `pill ${readiness.className}`;

    const mpc = model.mpcView(snapshot);
    $("timing-state").textContent = mpc.timingLabel;
    setEvidenceClass($("timing-state"), mpc.className);

    const physicalStop = model.physicalStopView(snapshot);
    $("physical-estop-state").textContent = physicalStop.label;
    setEvidenceClass($("physical-estop-state"), physicalStop.className);

    const faults = model.faultView(snapshot);
    $("fault-state").textContent = faults.label;
    setEvidenceClass($("fault-state"), faults.className);
  }

  function renderSnapshot(snapshot) {
    state.snapshot = snapshot;
    renderAuthorityAndPipeline(snapshot);
    renderGlobalRequestEvidence(snapshot);
    const qualificationStep =
      snapshot.wheels_off_qualification?.last_applied_step;
    if (qualificationProfile()) {
      const target = qualificationStep?.requested_target;
      const receipt = qualificationStep?.receipt;
      $("receipt-state").textContent =
        qualificationStep
          && typeof qualificationStep.event_id === "string"
          && typeof qualificationStep.navigation_ingress_sequence === "string"
          && Number.isInteger(receipt?.sequence)
          && Number.isInteger(target?.left_timer_pwm_percent)
          && Number.isInteger(target?.right_timer_pwm_percent)
          && Number.isInteger(receipt?.applied_left_timer_pwm_percent)
          && Number.isInteger(receipt?.applied_right_timer_pwm_percent)
          ? `event #${qualificationStep.event_id} · journal #${qualificationStep.navigation_ingress_sequence} · controller seq ${receipt.sequence} · target [${target.left_timer_pwm_percent}, ${target.right_timer_pwm_percent}]% · applied [${receipt.applied_left_timer_pwm_percent}, ${receipt.applied_right_timer_pwm_percent}]% · ${receipt.output_state || "output unknown"} · fault bits ${Number.isInteger(receipt.controller_fault_bits) ? `0x${receipt.controller_fault_bits.toString(16)}` : "unknown"}`
          : "none / no journaled qualification receipt";
    } else {
      $("receipt-state").textContent = snapshot.last_applied
        ? `seq ${snapshot.last_applied.sequence} · [${snapshot.last_applied.applied_left_timer_pwm_percent}, ${snapshot.last_applied.applied_right_timer_pwm_percent}]% · ${snapshot.last_applied.output_state} · ${snapshot.last_applied.result_code} · fault bits 0x${snapshot.last_applied.controller_fault_bits.toString(16)}`
        : "unknown / no exact receipt";
    }
    const stopCertainty = snapshot.stop_certainty || "unknown";
    $("stop-certainty").textContent = stopCertainty;
    setEvidenceClass(
      $("stop-certainty"),
      ["confirmed_applied_zero", "controller_reported_safe"].includes(stopCertainty)
        ? "ready"
        : "warn",
    );
    $("stm-health").textContent = health(snapshot.health?.stm32);
    setEvidenceClass($("stm-health"), snapshot.health?.stm32);
    $("oak-health").textContent =
      `${health(snapshot.health?.oak)} / ${snapshot.map?.localization || "unknown"}`;
    setEvidenceClass(
      $("oak-health"),
      snapshot.health?.oak === "faulted"
        ? "fault"
        : snapshot.health?.oak === "ready"
          && snapshot.map?.localization === "localized"
          ? "ready"
          : "warn",
    );
    $("expression-health").textContent =
      `${health(snapshot.health?.head)} / ${health(snapshot.health?.eyes)}`;
    setEvidenceClass(
      $("expression-health"),
      [snapshot.health?.head, snapshot.health?.eyes].includes("faulted")
        ? "fault"
        : snapshot.health?.head === "ready" && snapshot.health?.eyes === "ready"
          ? "ready"
          : "warn",
    );
    const map = snapshot.map;
    $("map-binding").textContent =
      map ? `epoch ${map.map_epoch_id} · rev ${map.revision}` : "epoch — · rev —";
    $("localization").textContent = (map?.localization || "unknown").toUpperCase();
    $("localization").className =
      `pill ${map?.localization === "localized" ? "ready" : map ? "warn" : "unknown"}`;
    const rerun = $("rerun-diagnostics");
    const rerunUrl = snapshot.rerun_diagnostics_url;
    if (rerunUrl) {
      rerun.textContent =
        `Rerun tunnel: rerun --connect ${rerunUrl.connectUri}`;
      rerun.classList.remove("hidden");
    } else {
      rerun.textContent = "";
      rerun.classList.add("hidden");
    }
    if (snapshot.software_safety_stop_latched) {
      state.serverSafetyLatched = true;
      applyLocalSafetyInhibit();
    }
    renderSafetyState(snapshot);
    updateControlAvailability();
    void loadGridIfNeeded(map)
      .then(drawMap)
      .catch((error) => {
        state.gridError = error.message;
        state.gridFailureKey = map ? `${map.map_epoch_id}:${map.revision}` : null;
        state.gridRetryAfterPerformanceMilliseconds = performance.now() + 1000;
        clearGrid();
        toast(error.message, true);
      });
  }

  function clearGrid() {
    state.grid = null;
    state.gridRaster = null;
    state.gridKey = null;
    drawMap();
  }

  function buildGridRaster(cells, metadata) {
    const image = document.createElement("canvas");
    image.width = metadata.width;
    image.height = metadata.height;
    const imageCtx = image.getContext("2d");
    if (!imageCtx) throw new Error("browser cannot allocate a map raster context");
    const pixels = imageCtx.createImageData(metadata.width, metadata.height);
    // Wire row zero is minimum map Y. Canvas row zero is top: invert exactly here.
    for (let y = 0; y < metadata.height; y += 1) {
      const canvasY = metadata.height - 1 - y;
      for (let x = 0; x < metadata.width; x += 1) {
        const cell = cells[y * metadata.width + x];
        if (cell > 2) {
          throw new Error("grid contains an unsupported occupancy class");
        }
        const shade = cell === 0 ? 24 : cell === 1 ? 215 : 18;
        const offset = (canvasY * metadata.width + x) * 4;
        pixels.data[offset] = shade;
        pixels.data[offset + 1] = cell === 1 ? 227 : shade + 5;
        pixels.data[offset + 2] = cell === 1 ? 222 : shade + 3;
        pixels.data[offset + 3] = 255;
      }
    }
    imageCtx.putImageData(pixels, 0, 0);
    return image;
  }

  async function loadGridIfNeeded(map) {
    if (!map?.grid) {
      state.gridLoadGeneration += 1;
      state.gridError = null;
      state.gridFailureKey = null;
      clearGrid();
      return;
    }
    const key = `${map.map_epoch_id}:${map.revision}`;
    if (state.gridKey === key && state.grid) return;
    if (state.gridFailureKey === key
      && performance.now() < state.gridRetryAfterPerformanceMilliseconds) {
      drawMap();
      return;
    }
    const generation = ++state.gridLoadGeneration;
    state.gridError = null;
    clearGrid();
    const response = await api(
      `/api/v1/maps/${map.map_epoch_id}/revisions/${map.revision}/grid`,
    );
    if (response.headers.get("x-kiko-map-epoch") !== map.map_epoch_id
      || response.headers.get("x-kiko-map-revision") !== map.revision
      || response.headers.get("x-kiko-grid-width") !== String(map.grid.width)
      || response.headers.get("x-kiko-grid-height") !== String(map.grid.height)
      || response.headers.get("x-kiko-grid-encoding")
        !== "u8_unknown0_free1_occupied2"
      || response.headers.get("x-kiko-grid-row-order")
        !== "row_major_x_fast_rows_increase_positive_map_y"
      || response.headers.get("x-kiko-grid-origin")
        !== "minimum_xy_corner_of_cell_0_0") {
      throw new Error("grid response contract does not match the displayed map");
    }
    const cells = new Uint8Array(await responseArrayBuffer(response));
    const expected = map.grid.width * map.grid.height;
    if (!Number.isSafeInteger(expected) || cells.length !== expected) {
      throw new Error("grid byte count does not match typed geometry");
    }
    if (generation !== state.gridLoadGeneration) return;
    state.grid = cells;
    state.gridRaster = buildGridRaster(cells, map.grid);
    state.gridKey = key;
    state.gridError = null;
    state.gridFailureKey = null;
  }

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  function mapTransform(metadata) {
    const scale = Math.min(
      canvas.width / metadata.width,
      canvas.height / metadata.height,
    );
    const displayWidth = metadata.width * scale;
    const displayHeight = metadata.height * scale;
    return {
      scale,
      displayWidth,
      displayHeight,
      offsetX: (canvas.width - displayWidth) / 2,
      offsetY: (canvas.height - displayHeight) / 2,
    };
  }

  function mapProject(point, metadata, transform) {
    const xCell = (point.x_m - metadata.origin_x_m)
      / metadata.resolution_m_per_cell;
    const yCell = (point.y_m - metadata.origin_y_m)
      / metadata.resolution_m_per_cell;
    return {
      x: transform.offsetX + xCell * transform.scale,
      y: transform.offsetY + transform.displayHeight - yCell * transform.scale,
    };
  }

  function drawPolyline(points, metadata, transform, color, width, dash = []) {
    if (!points?.length) return;
    ctx.beginPath();
    points.forEach((point, index) => {
      const projected = mapProject(point, metadata, transform);
      if (index) ctx.lineTo(projected.x, projected.y);
      else ctx.moveTo(projected.x, projected.y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function drawMap() {
    resizeCanvas();
    ctx.fillStyle = "#08100f";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const map = state.snapshot?.map;
    const metadata = map?.grid;
    if (!metadata || !state.grid || !state.gridRaster
      || state.gridKey !== `${map.map_epoch_id}:${map.revision}`) {
      const empty = $("map-empty");
      empty.textContent = state.gridError
        ? `Map grid unavailable: ${state.gridError}`
        : "Waiting for a known map grid…";
      empty.classList.remove("hidden");
      return;
    }
    $("map-empty").classList.add("hidden");
    const transform = mapTransform(metadata);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(
      state.gridRaster,
      transform.offsetX,
      transform.offsetY,
      transform.displayWidth,
      transform.displayHeight,
    );
    const nav = state.snapshot.navigation;
    drawPolyline(nav?.path, metadata, transform, "#78f0c2", 3);
    drawPolyline(nav?.mpc_predicted_path, metadata, transform, "#63d8ff", 2, [7, 5]);
    if (nav?.goal) {
      const goal = mapProject(nav.goal, metadata, transform);
      ctx.strokeStyle = "#ffc76a";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(goal.x, goal.y, 9, 0, Math.PI * 2);
      ctx.stroke();
    }
    if (nav?.pose) {
      const pose = mapProject(nav.pose, metadata, transform);
      ctx.save();
      ctx.translate(pose.x, pose.y);
      // Map +yaw is CCW in Y-up; canvas Y points down, hence the minus sign.
      ctx.rotate(-nav.pose.yaw_rad);
      ctx.fillStyle = "#fffbef";
      ctx.beginPath();
      ctx.moveTo(16, 0);
      ctx.lineTo(-10, -8);
      ctx.lineTo(-6, 0);
      ctx.lineTo(-10, 8);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }

  canvas.addEventListener("click", async (event) => {
    if (qualificationProfile()) {
      toast(
        "Qualification profile keeps autonomous actuation disabled; map, path, and MPC remain observational.",
        true,
      );
      return;
    }
    const map = state.snapshot?.map;
    const metadata = map?.grid;
    const sensorMotionReady = state.snapshot?.health?.oak === "ready";
    if (state.driveSafety.localInhibit
      || !sensorMotionReady || !metadata || !state.grid
      || state.gridKey !== `${map.map_epoch_id}:${map.revision}`
      || map.localization !== "localized") {
      toast(
        state.driveSafety.localInhibit
          ? "Fresh, non-inhibited control state is required for a point goal."
          : !sensorMotionReady
            ? "Fresh visual, depth, and motion-estimation evidence is required for a point goal."
          : "A localized, revision-bound map is required for a point goal.",
        true,
      );
      return;
    }
    resizeCanvas();
    const transform = mapTransform(metadata);
    const rect = canvas.getBoundingClientRect();
    if (!(rect.width > 0) || !(rect.height > 0)) {
      toast("Map canvas has no measurable display area.", true);
      return;
    }
    const canvasX = (event.clientX - rect.left) / rect.width * canvas.width;
    const canvasY = (event.clientY - rect.top) / rect.height * canvas.height;
    // These are continuous cell coordinates. The click position already
    // supplies its within-cell fraction and the grid origin is the minimum
    // corner, so adding 0.5 here would incorrectly shift every goal.
    const xCell = (canvasX - transform.offsetX) / transform.scale;
    const yCell = (transform.offsetY + transform.displayHeight - canvasY)
      / transform.scale;
    if (xCell < 0 || xCell >= metadata.width || yCell < 0 || yCell >= metadata.height) {
      toast("Click inside the rendered map bounds.", true);
      return;
    }
    const selectedCell = model.cellAt(state.grid, metadata, xCell, yCell);
    if (selectedCell !== "free") {
      toast(
        `Point goals require a currently free cell; selected cell is ${selectedCell}.`,
        true,
      );
      return;
    }
    try {
      await ensureExactStopped();
      await submit({
        kind: "autonomous_point_goal",
        map_epoch_id: map.map_epoch_id,
        displayed_revision: map.revision,
        x_m: metadata.origin_x_m + xCell * metadata.resolution_m_per_cell,
        y_m: metadata.origin_y_m + yCell * metadata.resolution_m_per_cell,
      });
    } catch (error) {
      toast(error.message, true);
    }
  });

  async function pollOnce() {
    const response = await api("/api/v1/snapshot");
    const snapshot = model.parseConsoleSnapshot(await responseJson(response));
    applyControlProfile(snapshot);
    const serverSafetyLatched = snapshot.software_safety_stop_latched === true;
    if (serverSafetyLatched) {
      state.serverSafetyLatched = true;
      applyLocalSafetyInhibit();
    }
    const revision = BigInt(snapshot.revision);
    const now = performance.now();
    const transition = applyDriveSafetyEvent({
      kind: "snapshot_observed",
      revision,
      now_milliseconds: now,
      stale_after_milliseconds: SNAPSHOT_STALE_AFTER_MILLISECONDS,
      local_safety_latched: state.localSafetyLatched,
      server_safety_latched: serverSafetyLatched,
    });
    if (transition.snapshotAdvanced) {
      renderSnapshot(snapshot);
    } else {
      // Requested ownership and the process-wide software-stop state are live
      // overlays and can change without claiming a new telemetry revision.
      state.snapshot = snapshot;
      renderAuthorityAndPipeline(snapshot);
      renderSafetyState(snapshot);
    }
    if (state.lastResponseId) {
      const record = await responseRecord(state.lastResponseId);
      $("request-state").textContent =
        `#${record.downstream_request_id} ${record.state} · ${record.applied ? "EXACT APPLIED RECEIPT" : "NOT APPLIED"}`;
    }
    setConnectionView(transition.state.connectionKind);
    updateControlAvailability();
    await executeDriveSafetyEffects(transition.effects);
  }

  async function pollLoop() {
    if (state.polling) return;
    state.polling = true;
    while (state.capability && state.sessionId) {
      try {
        await pollOnce();
      } catch (error) {
        await failClosedForTransport(error);
      }
      await delay(300);
    }
    state.polling = false;
  }

  $("unlock-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = $("capability");
    const candidate = input.value.trim().toLowerCase();
    input.value = "";
    if (!/^[0-9a-f]{64}$/.test(candidate)) {
      toast("Capability must be exactly 64 hexadecimal characters.", true);
      return;
    }
    state.capability = candidate;
    try {
      await openSession();
      $("unlock-panel").classList.add("hidden");
      $("console").classList.remove("hidden");
      applyDriveSafetyEvent({ kind: "session_opened" });
      await pollOnce();
      void pollLoop();
    } catch (error) {
      state.capability = null;
      state.sessionId = null;
      state.sessionCapability = null;
      applyDriveSafetyEvent({ kind: "locked" });
      setConnectionView("locked", error.message);
      toast(error.message, true);
    }
  });

  window.addEventListener("resize", drawMap);
  setConnectionView("locked");
  updateControlAvailability();
})();
