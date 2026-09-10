#!/bin/bash
# 完整复现: 跑 check_state, 收首个 DBI 上下文
set -e
docker exec cann_container bash -c 'rm -rf /root/HcPre/sim_out/* /root/HcPre/sim_dump_out /tmp/dbi_run.log; timeout 500 bash /root/HcPre/scripts/simopt/check_state.sh > /tmp/dbi_run.log 2>&1; true'
docker exec cann_container bash -c '
N=$(grep -c "DBI" /tmp/dbi_run.log)
echo "DBI count: $N"
L=$(grep -n -m1 "DBI" /tmp/dbi_run.log | cut -d: -f1)
if [ -n "$L" ]; then
  S=$((L-30)); [ $S -lt 1 ] && S=1
  echo "== 首个 DBI 上下文 =="
  sed -n "${S},$((L+3))p" /tmp/dbi_run.log
else
  echo "== 无 DBI, 尾部日志 =="; tail -12 /tmp/dbi_run.log
fi
'
