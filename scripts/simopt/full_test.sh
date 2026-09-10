#!/bin/bash
# 全开配置: build + check_state(dump), 输出 DBI 数与 dump 状态
set -e
cd /mnt/d/proj/HcPre
bash scripts/hc_pre/run.sh build > /mnt/d/proj/HcPre/simopt_logs/opt10_build7.log 2>&1
docker exec cann_container bash -c 'rm -rf /root/HcPre/sim_out/* /root/HcPre/sim_dump_out /root/HcPre/comb_dump.pt /tmp/dbi_run.log; timeout 1400 bash /root/HcPre/scripts/simopt/check_state.sh > /tmp/dbi_run.log 2>&1; true'
docker exec cann_container bash -c '
echo "DBI count: $(grep -c "DBI" /tmp/dbi_run.log)"
grep -E "Model RUN TIME|compare|PASS|FAIL" /tmp/dbi_run.log | head -6
ls -la /root/HcPre/comb_dump.pt 2>/dev/null || echo NO_DUMP
'
