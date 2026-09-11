#!/bin/bash
# 在容器内跑 part2_split 检查 cubecore 最后活动 (区分 opt8 vs B+C 的行为签名)
docker cp /mnt/c/Users/wang/AppData/Local/Temp/opencode/part2_split.py cann_container:/tmp/part2_split.py
docker exec cann_container bash -c '
B=$(ls /root/HcPre/sim_dump_out/OPPROF_*/simulator/visualize_data.bin | head -1)
echo "BIN: $B"
python3 /tmp/part2_split.py "$B" 2>&1 | tail -6
'
