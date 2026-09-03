#!/bin/bash
# container_monitor_run.sh <logfile> <cmd> [args...]
#
# Runs <cmd> in a new session (its own process group), writes output to <logfile>,
# and watches the log while the command runs:
#   - prints a progress summary every PROGRESS_INTERVAL seconds
#   - on fatal error patterns, kills the WHOLE process group immediately and dumps context
# Intended to run inside the cann container.
set -u

LOG=$1; shift
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-30}
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

FATAL_PATTERNS='FAILED: \[code=|CMake Error|Traceback \(most recent call last\)|ModuleNotFoundError|ImportError:|make\[[0-9]+\]: \*\*\* \[|ninja: build stopped|^ERROR: '

setsid bash -c '"$@"' _ "$@" >> "$LOG" 2>&1 &
PGID=$!
echo "[monitor] started pgid=$PGID: $*"
echo "[monitor] log: $LOG"

kill_tree() {
  kill -TERM -- -"$PGID" 2>/dev/null
  sleep 2
  kill -KILL -- -"$PGID" 2>/dev/null
  # sweep any stragglers spawned by the build
  for pat in "opc_tool" "ccec" "bisheng"; do
    pkill -KILL -f "$pat" 2>/dev/null
  done
}

SECS=0
while kill -0 "$PGID" 2>/dev/null; do
  if grep -qE "$FATAL_PATTERNS" "$LOG"; then
    echo ""
    echo "[monitor] *** FATAL error detected — killing build (pgid=$PGID) ***"
    kill_tree
    echo "[monitor] ===== fatal context (last 60 lines) ====="
    tail -60 "$LOG"
    exit 1
  fi
  sleep 5
  SECS=$((SECS + 5))
  if [ "$SECS" -ge "$PROGRESS_INTERVAL" ]; then
    SECS=0
    echo "[monitor] still running (${PROGRESS_INTERVAL}s elapsed) — last 3 log lines:"
    tail -3 "$LOG" | sed 's/^/    /'
  fi
done

wait "$PGID"; RC=$?
echo "[monitor] command finished, rc=$RC"
if [ "$RC" -ne 0 ]; then
  echo "[monitor] ===== last 60 log lines ====="
  tail -60 "$LOG"
fi
exit "$RC"
