#!/bin/bash
# Copy debug-run mhc_pre_sinkhorn sim results to Windows sim_out (overwrite previous release copy)
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}
SRC=/root/HcPre/sim_out_mhcs/OPPROF_20260903012809_BDIQSFXYLMKDXVHA
DST=/mnt/d/proj/HcPre/sim_out/mhc_pre_sinkhorn_1_512_4_4096

docker exec "$C" chmod -R a+rX /root/HcPre/sim_out_mhcs

echo "[fetch-mhcs] source size:"
docker exec "$C" du -sh "$SRC"

rm -rf "$DST" 2>/dev/null || docker exec "$C" rm -rf /root/HcPre/sim_out_mhcs_win_old 2>/dev/null
# stale root-owned files from earlier docker cp: remove via container (bind mount sees same fs)
docker exec "$C" rm -rf /mnt/d/proj/HcPre/sim_out/mhc_pre_sinkhorn_1_512_4_4096 2>/dev/null || true
rm -rf "$DST" 2>/dev/null || true
mkdir -p "$DST"

echo "[fetch-mhcs] copying..."
docker cp "$C:$SRC/." "$DST/"
docker cp "$C:/root/HcPre/mhcs_output_bs512.bin" "$DST/mhcs_output_bs512.bin"
docker cp "$C:/root/HcPre/mhcs_output_bs512.bin" /mnt/d/proj/HcPre/golden_refs/mhcs_sim_bs512.bin

echo "[fetch-mhcs] result on Windows:"
du -sh "$DST"
echo "trace.json count: $(find "$DST" -name "trace.json" | wc -l)"
ls "$DST"
