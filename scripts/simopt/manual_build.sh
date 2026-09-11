#!/bin/bash
# 手动全量跑 kernel 编译, 完整输出到文件
docker exec cann_container bash -c '
source /root/HcPre/scripts/container_env.sh
cd /root/HcPre/vllm-ascend/csrc
bash build.sh --pkg --ops=hc_pre --soc=ascend910_93 -j8 > /tmp/kernel_build_full.log 2>&1 || true
echo "exit=$?"
grep -n -iE "^.*error" /tmp/kernel_build_full.log | head -25
'
