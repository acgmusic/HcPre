#!/bin/bash
# Inspect rerun log for profiling completion + count products (quoting-safe)
set -u
C=${CANN_CONTAINER:-cann_container}
docker exec -i "$C" bash <<'INNER'
L=/root/HcPre/logs/mhcs_rerun.log
echo "=== log key lines ==="
grep -aE "Parse|Model Stop|Profiling results|Start parse|Core operator|addr2line" "$L" | head -8
echo "=== products ==="
echo "instr_csv: $(find /root/HcPre/sim_out_mhcs -name '*_instr_exe.csv' 2>/dev/null | wc -l)"
echo "trace_json: $(find /root/HcPre/sim_out_mhcs -name 'trace.json' 2>/dev/null | wc -l)"
echo "visualize: $(find /root/HcPre/sim_out_mhcs -name 'visualize_data.bin' 2>/dev/null | wc -l)"
echo "=== full log size/tail ==="
wc -l "$L"
tail -25 "$L" | head -30
INNER
