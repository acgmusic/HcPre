#!/bin/bash
# 直接在仿真器环境运行 hcpre_sim_app（无 profiler），输出精度对比
set -e
source /root/HcPre/scripts/container_env.sh >/dev/null 2>&1
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/simulator/Ascend910_9382/lib:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/root/HcPre/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export HC_PRE_PYBIND_SO=$(ls /root/HcPre/vllm-ascend/vllm_ascend/vllm_ascend_C*.so | head -1)
export LD_LIBRARY_PATH="$(dirname $HC_PRE_PYBIND_SO):$LD_LIBRARY_PATH"
export HC_PRE_SIZE=${HC_PRE_SIZE:-512} HC_PRE_BATCH=1 HC_PRE_COMPARE=1 NPU_ID=0
python3 /root/HcPre/scripts/hc_pre/hcpre_sim_app.py 2>&1 | grep -E "sim-app|compare|DONE|rror"
