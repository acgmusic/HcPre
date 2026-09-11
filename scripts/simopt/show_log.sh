#!/bin/bash
docker exec cann_container bash -c '
cd /root/HcPre/vllm-ascend/csrc
ls build.sh 2>/dev/null || ls ../build.sh 2>/dev/null
grep -rn "build.sh" /root/HcPre/scripts/hc_pre/run.sh | grep -m2 "pkg"
'
