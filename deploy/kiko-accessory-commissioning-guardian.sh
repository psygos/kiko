#!/usr/bin/env bash
# Attended Nano owner for the proven full expression/head runtime.
#
# This deliberately does not start the STM32, base, navigation, or SLAM. One
# process owns OAK RGB, KEP2 eyes, and the STS head bus so its face tracking and
# four-axis character decision cannot be split across competing owners.
#
# v2 (2026-08-02): the 21-hour incident proved `kill -0` cannot tell a hung
# child from a live one — the child sat in futex_wait with the head frozen
# while the guardian saw a healthy PID. This version supervises on a
# heartbeat file the engine's main loop touches every second, escalates
# SIGTERM -> SIGKILL on a stale heartbeat, and restarts with backoff instead
# of latching on the first exit. It latches only after repeated rapid faults.

set -u

readonly KIKO_ROOT=/home/makerspace/kiko
readonly EXPRESSION_ROOT="${KIKO_ROOT}/deploy/expression"
readonly LOCK_FILE=/tmp/kiko-accessory-commissioning-guardian.lock
readonly GUARDIAN_LOG=/tmp/kiko-accessory-commissioning-guardian.log
readonly EXPRESSION_LOG=/tmp/kiko-follow-track.log
readonly HEARTBEAT_FILE=/tmp/kiko-expression-heartbeat

# The engine needs camera boot (up to 30 s) plus head admission before its
# main loop starts beating; give it a generous cold-start allowance.
readonly STARTUP_GRACE_S=120
# Main loop beats every 5 ticks (~0.25 s nominal). Worst credible live tick
# is ~2 s (a bus retrying every read), so a live loop beats within ~11 s.
# 60 s of silence is a hung loop, not a slow one. Note: this compares the
# file mtime against wall clock, so a large NTP step can cause one false
# recycle — acceptable, the child restarts cleanly.
readonly HEARTBEAT_STALE_S=60
# SIGTERM triggers park-and-release: current tick + eye release + park ramp
# (14 s) + settle poll (up to ~6.6 s) + camera join (6 s) can approach 30 s.
# Grant more than that before concluding TERM cannot land.
readonly TERM_GRACE_S=45
# Latch after this many consecutive faults with no intervening healthy run.
# (A wall-clock window fails here: backoff + TERM grace + startup grace
# stretch the recycle period past any sane window, so a child that hangs at
# minute 3 every time would recycle forever and never demand a human.)
# A run counts as healthy once it stays up HEALTHY_UPTIME_S.
readonly FAULT_LATCH_COUNT=5
readonly HEALTHY_UPTIME_S=600

accessory_pid=
accessory_started_at=0
accessory_fault_latched=0
stop_requested=0
restart_backoff_s=5
fault_streak=0

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"${GUARDIAN_LOG}"
}

child_is_live() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

heartbeat_age_s() {
  # Prints seconds since the engine last touched the heartbeat file, or a
  # huge number if the file does not exist yet.
  local mtime
  if ! mtime=$(stat -c %Y "${HEARTBEAT_FILE}" 2>/dev/null); then
    printf '999999'
    return
  fi
  printf '%d' "$(( $(date +%s) - mtime ))"
}

stop_child() {
  # Bounded escalation: TERM (park path), wait TERM_GRACE_S, then KILL.
  # KILL leaves the servos holding their last goal (torque stays on inside
  # the servos) — that is the operator hard rule's preferred failure mode.
  local role="$1"
  local pid="$2"
  if ! child_is_live "${pid}"; then
    wait "${pid}" 2>/dev/null || true  # reap if it already exited
    return
  fi
  log "${role} stop requested pid=${pid} (TERM, grace ${TERM_GRACE_S}s)"
  kill -TERM "${pid}" 2>/dev/null || true
  local waited=0
  while (( waited < TERM_GRACE_S )) && child_is_live "${pid}"; do
    sleep 1
    waited=$(( waited + 1 ))
  done
  if child_is_live "${pid}"; then
    log "${role} ignored TERM for ${TERM_GRACE_S}s; escalating to KILL pid=${pid}"
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  # Bounded reap: a child stuck in uninterruptible D-state (wedged USB-serial
  # ioctl) survives even KILL, and a bare `wait` would freeze the supervisor
  # with it. Poll, then abandon supervision of the corpse if it will not die.
  local reap_waited=0
  while (( reap_waited < 10 )) && child_is_live "${pid}"; do
    sleep 1
    reap_waited=$(( reap_waited + 1 ))
  done
  if child_is_live "${pid}"; then
    # Explicit containment: the corpse still owns the OAK and both tty
    # handles, so any replacement child would just die at device open.
    # Latch instead of relaunching into a guaranteed single-ownership
    # violation; a human must resolve a D-state serial hang anyway.
    log "${role} unkillable after KILL (D-state?) pid=${pid}; latching"
    accessory_fault_latched=1
    return
  fi
  wait "${pid}" 2>/dev/null || true
  log "${role} stopped pid=${pid}"
}

request_stop() {
  stop_requested=1
}

cleanup() {
  trap - EXIT INT TERM HUP
  stop_child accessory "${accessory_pid}"
  log "guardian stopped pid=$$"
}

record_fault() {
  local reason="$1"
  fault_streak=$(( fault_streak + 1 ))
  log "accessory fault (${reason}); consecutive ${fault_streak}/${FAULT_LATCH_COUNT}"
  if (( fault_streak >= FAULT_LATCH_COUNT )); then
    accessory_fault_latched=1
    log "accessory fault latched after ${fault_streak} consecutive faults; deliberate guardian restart required"
  fi
}

apply_backoff() {
  log "restart in ${restart_backoff_s}s"
  local slept=0
  while (( slept < restart_backoff_s && stop_requested == 0 )); do
    sleep 1
    slept=$(( slept + 1 ))
  done
  restart_backoff_s=$(( restart_backoff_s * 3 ))
  (( restart_backoff_s > 180 )) && restart_backoff_s=180
}

start_accessory() {
  if [[ ! -f "${EXPRESSION_ROOT}/kiko_face_follow.py" ]]; then
    log "accessory start refused: ${EXPRESSION_ROOT}/kiko_face_follow.py is missing"
    return 1
  fi
  rm -f "${HEARTBEAT_FILE}"
  (
    cd "${EXPRESSION_ROOT}" || exit 1
    exec python3 kiko_face_follow.py --duration-s 864000 \
      --heartbeat-file "${HEARTBEAT_FILE}"
  ) >>"${EXPRESSION_LOG}" 2>&1 < /dev/null &
  accessory_pid=$!
  accessory_started_at=$(date +%s)
  log "accessory started pid=${accessory_pid} authority=oak,eyes,head"
}

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  exit 0
fi

trap request_stop INT TERM HUP
trap cleanup EXIT
log "guardian started pid=$$"

while (( stop_requested == 0 )); do
  if (( accessory_fault_latched == 0 )); then
    if ! child_is_live "${accessory_pid}"; then
      if [[ -n "${accessory_pid}" ]]; then
        accessory_status=0
        wait "${accessory_pid}" 2>/dev/null || accessory_status=$?
        log "accessory exited status=${accessory_status} pid=${accessory_pid}"
        accessory_pid=
        record_fault "exit status=${accessory_status}"
        if (( accessory_fault_latched == 0 )); then
          apply_backoff
        fi
      fi
      if (( accessory_fault_latched == 0 && stop_requested == 0 )); then
        if ! start_accessory; then
          record_fault "start refused"
        fi
      fi
    else
      # Liveness: a PID that answers kill -0 can still be a corpse with a
      # frozen main loop. Judge by heartbeat freshness after cold start.
      now=$(date +%s)
      age=$(heartbeat_age_s)
      uptime_s=$(( now - accessory_started_at ))
      if (( uptime_s > STARTUP_GRACE_S && age > HEARTBEAT_STALE_S )); then
        log "accessory heartbeat stale (age=${age}s uptime=${uptime_s}s); recycling pid=${accessory_pid}"
        stop_child accessory "${accessory_pid}"
        accessory_pid=
        record_fault "stale heartbeat age=${age}s"
        if (( accessory_fault_latched == 0 )); then
          apply_backoff
        fi
      elif (( uptime_s > HEALTHY_UPTIME_S )); then
        # A child that has run healthy this long earns a clean slate: the
        # backoff resets and the consecutive-fault streak clears.
        restart_backoff_s=5
        fault_streak=0
      fi
    fi
  fi

  sleep 5 &
  wait $! 2>/dev/null || true
done
