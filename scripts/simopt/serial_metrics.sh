#!/bin/bash
# 从串行 dump 仿真产物提取 WALL 指标
docker exec cann_container bash -c '
ls -t /root/HcPre/sim_dump_out/ | head -2
python3 /root/HcPre/scripts/simopt/metrics.py /root/HcPre/sim_dump_out 2>&1 | head -8
'
