#!/bin/bash
# simopt10 全开配置: build + 串行模式 dump 仿真 (判决实验)
set -e
docker start cann_container >/dev/null
bash /mnt/d/proj/HcPre/scripts/simopt/make_serial_cfg.sh > /dev/null
cd /mnt/d/proj/HcPre
bash scripts/hc_pre/run.sh build > /mnt/d/proj/HcPre/simopt_logs/opt10_serial_build.log 2>&1
tail -1 /mnt/d/proj/HcPre/simopt_logs/opt10_serial_build.log
bash /mnt/d/proj/HcPre/scripts/simopt/serial_dump_run.sh
