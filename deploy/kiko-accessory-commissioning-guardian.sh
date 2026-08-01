#!/usr/bin/env bash
# Attended Nano owner for the proven full expression/head runtime.
#
# This deliberately does not start the STM32, base, navigation, or SLAM. One
# process owns OAK RGB, KEP2 eyes, and the STS head bus so its face tracking and
# four-axis character decision cannot be split across competing owners.

set -u

readonly KIKO_ROOT=/home/makerspace/kiko
readonly EXPRESSION_ROOT="${KIKO_ROOT}/deploy/expression"
readonly LOCK_FILE=/tmp/kiko-accessory-commissioning-guardian.lock
readonly GUARDIAN_LOG=/tmp/kiko-accessory-commissioning-guardian.log
readonly EXPRESSION_LOG=/tmp/kiko-follow-track.log

accessory_pid=
accessory_fault_latched=0
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
  stop_child accessory "${accessory_pid}"
  log "guardian stopped pid=$$"
}

start_accessory() {
  if [[ ! -f "${EXPRESSION_ROOT}/kiko_face_follow.py" ]]; then
    log "accessory start refused: ${EXPRESSION_ROOT}/kiko_face_follow.py is missing"
    return 1
  fi
  (
    cd "${EXPRESSION_ROOT}" || exit 1
    exec python3 kiko_face_follow.py --duration-s 864000
  ) >>"${EXPRESSION_LOG}" 2>&1 < /dev/null &
  accessory_pid=$!
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
  if (( accessory_fault_latched == 0 )) && ! child_is_live "${accessory_pid}"; then
    if [[ -n "${accessory_pid}" ]]; then
      accessory_status=0
      wait "${accessory_pid}" 2>/dev/null || accessory_status=$?
      log "accessory exited status=${accessory_status} pid=${accessory_pid}"
      accessory_fault_latched=1
      log "accessory fault latched; deliberate guardian restart required"
    fi
    accessory_pid=
    if (( accessory_fault_latched == 0 )) && ! start_accessory; then
      accessory_fault_latched=1
      log "accessory start fault latched; deliberate guardian restart required"
    fi
  fi

  sleep 5 &
  wait $! 2>/dev/null || true
done
