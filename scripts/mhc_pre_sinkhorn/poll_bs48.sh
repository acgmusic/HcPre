#!/bin/bash
# Poll bs=48 sim: log tail + process + products
set -u
C=${CANN_CONTAINER:-cann_container}
docker exec -i "$C" bash <<'INNER'
L=/root/HcPre/logs/mhcs_sim_bs48.log
echo "procs: $(ps aux | grep -v grep | grep -cE msprof)"
echo "log bytes: $(stat -c%s $L 2>/dev/null)"
grep -aE "mhcs-ex|Model Stop|Start parse|Profiling results saved|EXIT_CODE|RUN_FINISHED|signal" "$L" 2>/dev/null | tail -8
echo "--- products ---"
echo "instr_csv: $(find /root/HcPre/sim_out_mhcs -name '*_instr_exe.csv' 2>/dev/null | wc -l)"
echo "trace_json: $(find /root/HcPre/sim_out_mhcs -name 'trace.json' 2>/dev/null | wc -l)"
echo "visualize: $(find /root/HcPre/sim_out_mhcs -name 'visualize_data.bin' 2>/dev/null)"
INNER
