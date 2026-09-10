#!/bin/bash
# 后台启动单次 check_state(dump), 带 40min 硬超时; 用 status 文件轮询
docker exec -d cann_container bash -c '
rm -rf /root/HcPre/sim_out/* /root/HcPre/sim_dump_out /root/HcPre/comb_dump.pt /tmp/dbi_run.log /tmp/dbi_status.txt
timeout 2400 bash /root/HcPre/scripts/simopt/check_state.sh > /tmp/dbi_run.log 2>&1
echo "EXIT=$? DBI=$(grep -c DBI /tmp/dbi_run.log) MODEL=$(grep -oE "Model RUN TIME: [0-9.e+]+" /tmp/dbi_run.log | head -1)" > /tmp/dbi_status.txt
ls /root/HcPre/comb_dump.pt >/dev/null 2>&1 && echo "DUMP=YES" >> /tmp/dbi_status.txt || echo "DUMP=NO" >> /tmp/dbi_status.txt
'
echo "launched background sim; poll: docker exec cann_container cat /tmp/dbi_status.txt"
