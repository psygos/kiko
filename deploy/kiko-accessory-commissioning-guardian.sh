#!/usr/bin/env bash
# Transitional Nano owner for attended expression/head commissioning.
#
# This deliberately does not start the STM32, base, navigation, or SLAM. The
# legacy expression process is retained only for OAK-derived eyes and is
# required to release the head with --no-head. The typed Rust commissioning
# binary is the sole head-bus owner.

set -u

readonly KIKO_ROOT=/home/makerspace/kiko
readonly FOLLOW_ROOT=/home/makerspace/kiko-follow
readonly HEAD_BINARY="${KIKO_ROOT}/target/release/kiko-head-commission"
readonly HEAD_CONFIG="${KIKO_ROOT}/configs/nano-head-compliant-commissioning-v1.json"
readonly LOCK_FILE=/tmp/kiko-accessory-commissioning-guardian.lock
readonly GUARDIAN_LOG=/tmp/kiko-accessory-commissioning-guardian.log
readonly FOLLOW_LOG=/tmp/kiko-follow-track.log
readonly HEAD_LOG=/tmp/kiko-head-compliant.log

follow_pid=
head_pid=
head_fault_latched=0
stop_requested=0

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"${GUARDIAN_LOG}"
}

child_is_live() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

stop_child() {
  local role="$1"
  local pid="$2"
  if ! child_is_live "${pid}"; then
    return
  fi
  log "${role} stop requested pid=${pid}"
  kill -TERM "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  log "${role} stopped pid=${pid}"
}

request_stop() {
  stop_requested=1
}

cleanup() {
  trap - EXIT INT TERM HUP
  stop_child head "${head_pid}"
  stop_child expression "${follow_pid}"
  log "guardian stopped pid=$$"
}

start_expression() {
  if [[ ! -f "${FOLLOW_ROOT}/kiko_face_follow.py" ]]; then
    log "expression start refused: ${FOLLOW_ROOT}/kiko_face_follow.py is missing"
    return 1
  fi
  (
    cd "${FOLLOW_ROOT}" || exit 1
    exec python3 kiko_face_follow.py --duration-s 864000 --no-head
  ) >>"${FOLLOW_LOG}" 2>&1 < /dev/null &
  follow_pid=$!
  log "expression started pid=${follow_pid} head_authority=disabled"
}

start_head() {
  if [[ ! -x "${HEAD_BINARY}" ]]; then
    log "head start refused: ${HEAD_BINARY} is not executable"
    return 1
  fi
  if [[ ! -r "${HEAD_CONFIG}" ]]; then
    log "head start refused: ${HEAD_CONFIG} is not readable"
    return 1
  fi
  "${HEAD_BINARY}" \
    --config "${HEAD_CONFIG}" \
    --compliant-hold \
    --physical-torque-consent \
    --physical-motion-consent \
    >>"${HEAD_LOG}" 2>&1 < /dev/null &
  head_pid=$!
  log "head compliant owner started pid=${head_pid}"
}

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  exit 0
fi

trap request_stop INT TERM HUP
trap cleanup EXIT
log "guardian started pid=$$"

while (( stop_requested == 0 )); do
  if ! child_is_live "${follow_pid}"; then
    if [[ -n "${follow_pid}" ]]; then
      wait "${follow_pid}" 2>/dev/null
      log "expression exited status=$? pid=${follow_pid}"
    fi
    follow_pid=
    start_expression || true
  fi

  if (( head_fault_latched == 0 )) && ! child_is_live "${head_pid}"; then
    if [[ -n "${head_pid}" ]]; then
      head_status=0
      wait "${head_pid}" 2>/dev/null || head_status=$?
      log "head compliant owner exited status=${head_status} pid=${head_pid}"
      head_fault_latched=1
      log "head compliant fault latched; deliberate guardian restart required"
    fi
    head_pid=
    if (( head_fault_latched == 0 )) && ! start_head; then
      head_fault_latched=1
      log "head compliant start fault latched; deliberate guardian restart required"
    fi
  fi

  sleep 5 &
  wait $! 2>/dev/null || true
done
