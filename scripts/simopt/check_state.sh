#!/bin/bash
tail -3 /mnt/d/proj/HcPre/simopt_logs/opt4_build.log
echo ===SIM_TAIL===
tail -6 /mnt/d/proj/HcPre/simopt_logs/opt4_sim.log 2>/dev/null
echo ===PROC===
docker exec cann_container bash -c 'ps aux | grep -E "msprof|hcpre_sim|msop" | grep -v grep | head -5'
echo ===BUILD_TAIL_RAW===
tail -3 /mnt/d/proj/HcPre/simopt_logs/opt4_build.log
