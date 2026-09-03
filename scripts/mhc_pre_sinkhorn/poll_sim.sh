#!/bin/bash
# Check background sim status: process count + log tail + output products.
set -u
C=${CANN_CONTAINER:-cann_container}
docker start "$C" >/dev/null 2>&1 || true
docker exec "$C" bash -c '
echo "msprof procs: $(ps aux | grep -v grep | grep -cE msprof)"
L=/root/HcPre/logs/mhcs_rerun.log
[ -f /root/HcPre/logs/mhcs_sim.log ] && L=/root/HcPre/logs/mhcs_sim.log
echo "log: $L ($(stat -c%s $L 2>/dev/null) bytes)"
grep -aE "RUN_FINISHED|EXIT_CODE|Profiling results saved|Model Stop|signal|mhcs-ex" "$L" 2>/dev/null | tail -8
echo "--- last 3 lines ---"
tail -3 "$L" 2>/dev/null
'
