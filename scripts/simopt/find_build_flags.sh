#!/bin/bash
docker exec cann_container bash -c 'find /root/HcPre/vllm-ascend \( -name flags.make -o -name compile_commands.json -o -name build.ninja \) 2>/dev/null | grep -i -E "hc_pre|kernel" | head -8'
