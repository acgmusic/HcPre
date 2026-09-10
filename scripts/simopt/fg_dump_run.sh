#!/bin/bash
# 前台阻塞式 dump 仿真(单次策略): 清场 -> check_state -> 报告
set -e
docker start cann_container >/dev/null
docker exec cann_container bash -c 'rm -rf /root/HcPre/sim_out/* /root/HcPre/sim_dump_out /root/HcPre/comb_dump.pt'
docker exec cann_container bash -c 'timeout 2400 bash /root/HcPre/scripts/simopt/check_state.sh > /tmp/dbi_run.log 2>&1 || true'
docker exec cann_container bash -c '
echo "DBI=$(grep -c DBI /tmp/dbi_run.log)"
grep -E "Model RUN TIME|compare|overall" /tmp/dbi_run.log | head -5
ls -la /root/HcPre/comb_dump.pt 2>/dev/null || echo NO_DUMP
'
