#!/bin/bash
# Clean ALL simulation result artifacts inside the WSL cann container ONLY
# (Windows local D:\proj\HcPre\sim_out etc. are untouched).
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}

echo "=== before cleanup ==="
docker exec "$C" bash -c '
du -sh /root/HcPre/sim_out* /root/HcPre/mhc_output_bs512.bin /root/HcPre/mhc_fault.log \
        /root/HcPre/sim_result_small.pt /tmp/mhcsim*.log /tmp/var*.log /tmp/mhc_stage.log /tmp/csv_path 2>/dev/null
du -sh /root/HcPre 2>/dev/null
'

docker exec "$C" bash -c '
rm -rf /root/HcPre/sim_out
rm -rf /root/HcPre/sim_out_mhc
rm -rf /root/HcPre/sim_out_mhc_orig
rm -rf /root/HcPre/sim_out_mhc_var
rm -f  /root/HcPre/mhc_output_bs512.bin
rm -f  /root/HcPre/mhc_fault.log
rm -f  /root/HcPre/sim_result_small.pt
rm -f  /tmp/mhcsim*.log /tmp/var*.log /tmp/mhc_stage.log /tmp/p.csv /tmp/csv_path
# simulator scratch dirs created by msprof under / (from earlier runs)
rm -rf /root/mindstudio_sanitizer_log/* 2>/dev/null
'

echo
echo "=== after cleanup ==="
docker exec "$C" bash -c '
ls /root/HcPre/
echo ---
du -sh /root/HcPre 2>/dev/null
df -h / | tail -1
'
