#!/bin/bash
# Copy sim_out results from the cann container to Windows D:\proj\HcPre\sim_out
# Usage: wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/fetch_sim_out.sh [--rm]
set -euo pipefail

C=${CANN_CONTAINER:-cann_container}
SRC=/root/HcPre/sim_out
DST=/mnt/d/proj/HcPre

docker start "$C" >/dev/null 2>&1 || true

echo "[fetch] container sim_out size:"
docker exec "$C" bash -c "du -sh $SRC 2>/dev/null || { echo 'sim_out not found in container; run sim first' >&2; exit 1; }"

rm -rf "$DST/sim_out"
docker cp "$C:$SRC" "$DST/sim_out"

echo "[fetch] done:"
du -sh "$DST/sim_out"
find "$DST/sim_out" -name "*_instr_exe.csv" | wc -l | xargs echo "instr_exe.csv count:"
ls "$DST/sim_out"

if [ "${1:-}" = "--rm" ]; then
  docker exec "$C" rm -rf "$SRC"
  echo "[fetch] removed from container"
fi
