#!/bin/bash
# 串行模式 dump 仿真 v2: --timeout 提到 115 分钟
set -e
docker start cann_container >/dev/null
docker exec cann_container bash -c 'rm -rf /root/HcPre/sim_out/* /root/HcPre/sim_dump_out /root/HcPre/comb_dump.pt'
docker exec cann_container bash -c '
source /root/HcPre/scripts/container_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/simulator/Ascend910_9382/lib:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/root/HcPre/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export HC_PRE_PYBIND_SO=$(ls /root/HcPre/vllm-ascend/vllm_ascend/vllm_ascend_C*.so | head -1)
export LD_LIBRARY_PATH="$(dirname $HC_PRE_PYBIND_SO):$LD_LIBRARY_PATH"
export HC_PRE_SIZE=512 HC_PRE_BATCH=1 HC_PRE_COMPARE=0 NPU_ID=0
export HC_PRE_DUMP=/root/HcPre/comb_dump.pt
export CAMODEL_CONFIG_PATH=/root/HcPre/simopt_serial_cfg
rm -rf /root/HcPre/sim_dump_out; mkdir -p /root/HcPre/sim_dump_out
timeout 6800 msprof op simulator \
  --application="python3 /root/HcPre/scripts/hc_pre/hcpre_sim_app.py" \
  --output=/root/HcPre/sim_dump_out \
  --kernel-name=HcPre \
  --launch-count=1 \
  --soc-version=Ascend910_9382 \
  --timeout=115 > /tmp/dbi_run.log 2>&1 || true
echo "DBI=$(grep -c DBI /tmp/dbi_run.log)"
grep -E "Model RUN TIME|Total tick|compare|sim-app" /tmp/dbi_run.log | head -6
ls -la /root/HcPre/comb_dump.pt 2>/dev/null || echo NO_DUMP
'
