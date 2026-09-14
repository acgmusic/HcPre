#!/bin/bash
# simopt8 debug 模式构建 + 仿真 + 指标
set -e
cd /mnt/d/proj/HcPre
bash scripts/hc_pre/run.sh build --debug > /mnt/d/proj/HcPre/simopt_logs/opt8_dbg_build.log 2>&1
grep -E "debug_sections|build done" /mnt/d/proj/HcPre/simopt_logs/opt8_dbg_build.log | tail -3
bash scripts/hc_pre/run.sh sim 1 512 > /mnt/d/proj/HcPre/simopt_logs/opt8_dbg_sim.log 2>&1
grep -E "compare" /mnt/d/proj/HcPre/simopt_logs/opt8_dbg_sim.log | tail -4
docker exec cann_container python3 /root/HcPre/scripts/simopt/metrics.py /root/HcPre/sim_out 2>&1 | head -2
