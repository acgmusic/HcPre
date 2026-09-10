#!/bin/bash
docker exec cann_container bash -c '
echo "== 容器内仿真进程"; ps aux | grep -E "msprof|hcpre_sim|check_state" | grep -v grep | awk "{print \$2, \$11, \$12}" | head -5
echo "== status 文件"; cat /tmp/dbi_status.txt 2>/dev/null || echo "不存在(后台进程没跑完或已死)"
echo "== dbi_run.log 尾部"; tail -4 /tmp/dbi_run.log 2>/dev/null
echo "== log 大小/修改时间"; ls -la /tmp/dbi_run.log 2>/dev/null
echo "== dump 产物"; ls /root/HcPre/comb_dump.pt 2>/dev/null || echo NO_DUMP
date
'
