#!/bin/bash
# Copy mhc_pre_sinkhorn sim results from container to Windows sim_out/<name>.
# Usage: bash fetch.sh [dirname]   (default mhc_pre_sinkhorn_1_512_4_4096)
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}
NAME=${1:-mhc_pre_sinkhorn_1_512_4_4096}
SRC=/root/HcPre/sim_out_mhcs
DST=/mnt/d/proj/HcPre/sim_out/$NAME

docker exec "$C" chmod -R a+rX "$SRC" 2>/dev/null || true
rm -rf "$DST" 2>/dev/null || true
mkdir -p "$DST"

echo "[fetch] copying $SRC -> $DST"
docker cp "$C:$SRC/." "$DST/"
docker cp "$C:/root/HcPre/mhcs_output_bs512.bin" "$DST/mhcs_output_bs512.bin"

echo "[fetch] done:"
du -sh "$DST"
echo "trace.json: $(find "$DST" -name "trace.json" | wc -l)"
echo "instr csv:  $(find "$DST" -name "*_instr_exe.csv" | wc -l)"
ls "$DST"
