#!/bin/bash
# 全开配置 check_state(dump) — 长超时版
docker exec cann_container bash -c 'rm -rf /root/HcPre/sim_out/* /root/HcPre/sim_dump_out /root/HcPre/comb_dump.pt /tmp/dbi_run.log; timeout 2200 bash /root/HcPre/scripts/simopt/check_state.sh > /tmp/dbi_run.log 2>&1; true'
docker exec cann_container bash -c '
echo "DBI count: $(grep -c "DBI" /tmp/dbi_run.log)"
grep -E "Model RUN TIME" /tmp/dbi_run.log
tail -3 /tmp/dbi_run.log
ls -la /root/HcPre/comb_dump.pt 2>/dev/null || echo NO_DUMP
'
