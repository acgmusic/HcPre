#!/bin/bash
# 手动仿真(带 HC_PRE_DUMP 导出 NPU 输出) — 复刻 run.sh sim 的调用并加 dump
set -e
source /root/HcPre/scripts/container_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/simulator/Ascend910_9382/lib:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/root/HcPre/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export HC_PRE_PYBIND_SO=$(ls /root/HcPre/vllm-ascend/vllm_ascend/vllm_ascend_C*.so | head -1)
export LD_LIBRARY_PATH="$(dirname $HC_PRE_PYBIND_SO):$LD_LIBRARY_PATH"
export HC_PRE_SIZE=512 HC_PRE_BATCH=1 HC_PRE_COMPARE=0 NPU_ID=0
export HC_PRE_DUMP=/root/HcPre/comb_dump.pt
rm -rf /root/HcPre/sim_out_dump; mkdir -p /root/HcPre/sim_dump_out
msprof op simulator \
  --application="python3 /root/HcPre/scripts/hc_pre/hcpre_sim_app.py" \
  --output=/root/HcPre/sim_dump_out \
  --kernel-name=HcPre \
  --launch-count=1 \
  --soc-version=Ascend910_9382 \
  --timeout=30 || true
ls -la /root/HcPre/comb_dump.pt
