#!/bin/bash
# 全清产物 -> opt10 全新构建 -> 串行仿真 -> 精度+指标+AIC签名 一次完成
set -e
docker start cann_container >/dev/null
cd /mnt/d/proj/HcPre

# 1. 确保工作区在 simopt10 且开关全开
git -C vllm-ascend checkout simopt10-bc-y-offload 2>/dev/null || true
BR=$(git -C vllm-ascend branch --show-current)
echo "== 分支: $BR"
grep -E "ENABLE_XT_TRANSPOSE = |ENABLE_Y_MM = " vllm-ascend/csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h | sed 's/^ *//'

# 2. 删除容器内全部编译产物
docker exec cann_container bash -c '
rm -rf /root/HcPre/vllm-ascend/vllm_ascend/_cann_ops_custom
rm -rf /root/HcPre/vllm-ascend/csrc/build /root/HcPre/vllm-ascend/csrc/output /root/HcPre/vllm-ascend/csrc/build_out
rm -rf /root/HcPre/vllm-ascend/build-pybind /root/HcPre/vllm-ascend/.deps
find /root/HcPre -maxdepth 4 -name "CMakeCache.txt" -delete 2>/dev/null
echo "== 编译产物已全删"
'

# 3. 全新构建
bash scripts/hc_pre/run.sh build > /mnt/d/proj/HcPre/simopt_logs/opt10_clean_build.log 2>&1
tail -1 /mnt/d/proj/HcPre/simopt_logs/opt10_clean_build.log
docker exec cann_container bash -c 'stat -c "kernel .o: %s bytes, %y" /root/HcPre/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/hc_pre/HcPre_*.o | head -1'

# 4. 串行模式仿真 (判决配置: CAMODEL_CONFIG_PATH=parsim 0)
bash /mnt/d/proj/HcPre/scripts/simopt/make_serial_cfg.sh > /dev/null
docker exec cann_container bash -c '
source /root/HcPre/scripts/container_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/simulator/Ascend910_9382/lib:$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=/root/HcPre/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export HC_PRE_PYBIND_SO=$(ls /root/HcPre/vllm-ascend/vllm_ascend/vllm_ascend_C*.so | head -1)
export LD_LIBRARY_PATH="$(dirname $HC_PRE_PYBIND_SO):$LD_LIBRARY_PATH"
export HC_PRE_SIZE=512 HC_PRE_BATCH=1 HC_PRE_COMPARE=0 NPU_ID=0
export HC_PRE_DUMP=/root/HcPre/comb_dump.pt
export CAMODEL_CONFIG_PATH=/root/HcPre/simopt_serial_cfg
rm -rf /root/HcPre/sim_dump_out /root/HcPre/sim_out; mkdir -p /root/HcPre/sim_dump_out /root/HcPre/sim_out
rm -f /root/HcPre/comb_dump.pt
timeout 6800 msprof op simulator \
  --application="python3 /root/HcPre/scripts/hc_pre/hcpre_sim_app.py" \
  --output=/root/HcPre/sim_dump_out \
  --kernel-name=HcPre \
  --launch-count=1 \
  --soc-version=Ascend910_9382 \
  --timeout=115 > /tmp/dbi_run.log 2>&1 || true
echo "== DBI=$(grep -c DBI /tmp/dbi_run.log)"
grep -E "Model RUN TIME|Total tick" /tmp/dbi_run.log
ls -la /root/HcPre/comb_dump.pt 2>/dev/null || echo NO_DUMP
'

# 5. 指标 + AIC 行为签名
docker exec cann_container bash -c 'python3 /root/HcPre/scripts/simopt/metrics.py /root/HcPre/sim_dump_out 2>&1 | head -8'
docker cp /mnt/c/Users/wang/AppData/Local/Temp/opencode/part2_split.py cann_container:/tmp/part2_split.py > /dev/null
docker exec cann_container bash -c 'B=$(ls /root/HcPre/sim_dump_out/OPPROF_*/simulator/visualize_data.bin | head -1); python3 /tmp/part2_split.py "$B" 2>&1 | tail -3'
