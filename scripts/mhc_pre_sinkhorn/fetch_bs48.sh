#!/bin/bash
# Fetch bs=48 debug sim results (incl visualize_data.bin) to Windows sim_out.
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}
SRC=/root/HcPre/sim_out_mhcs/OPPROF_20260903031603_UMYSBWALUAKZZTZK
DST=/mnt/d/proj/HcPre/sim_out/mhc_pre_sinkhorn_1_48_4_4096

docker exec "$C" chmod -R a+rX "$SRC" 2>/dev/null || true
echo "[fetch] size:"
docker exec "$C" du -sh "$SRC"

rm -rf "$DST" 2>/dev/null || true
mkdir -p "$DST"

docker cp "$C:$SRC/." "$DST/"
docker cp "$C:/root/HcPre/mhcs_output_bs48.bin" "$DST/mhcs_output_bs48.bin"
docker cp "$C:/root/HcPre/mhcs_output_bs48.bin" /mnt/d/proj/HcPre/golden_refs/mhcs_sim_bs48.bin

echo "[fetch] result on Windows:"
du -sh "$DST"
echo "instr_csv: $(find "$DST" -name '*_instr_exe.csv' | wc -l)"
echo "trace_json: $(find "$DST" -name 'trace.json' | wc -l)"
ls -la "$DST/simulator/visualize_data.bin" 2>/dev/null || ls "$DST"
