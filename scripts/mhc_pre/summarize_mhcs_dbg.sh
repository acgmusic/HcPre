#!/bin/bash
# Debug-run summary: pipe utilization on top cores + hot lines + precision recheck
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}
docker exec "$C" bash -c '
chmod -R a+rX /root/HcPre/sim_out_mhcs 2>/dev/null
find /root/HcPre/sim_out_mhcs -name "*_instr_exe.csv" | wc -l
python3 /root/HcPre/scripts/parse_sim_csv.py /root/HcPre/sim_out_mhcs 2>/dev/null | head -32
'
