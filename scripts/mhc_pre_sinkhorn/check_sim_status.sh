#!/bin/bash
# Robust status check: container uptime, sim processes, output age
set -u
C=${CANN_CONTAINER:-cann_container}
docker start "$C" >/dev/null 2>&1 || true
sleep 1
docker exec "$C" bash -c 'NOW=$(date +%s); echo "age_of_last_write_sec=$((NOW-1788402259))"; echo "msprof procs: $(ps aux | grep -v grep | grep -c msprof)"; echo "sim procs: $(ps aux | grep -v grep | grep -cE "msopprof|sinkhorn_hcshape|camodel")"'
