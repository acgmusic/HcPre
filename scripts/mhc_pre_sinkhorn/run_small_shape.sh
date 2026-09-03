#!/bin/bash
# Debug-mode mhc_pre_sinkhorn sim with a SMALL shape (bs=48) to obtain visualize_data.bin.
# Smaller kernel dump => msprof post-processing (CSV + visualize packaging) can finish.
# Usage: bash run_small_shape.sh [bs]     (default 48)
set -euo pipefail

C=${CANN_CONTAINER:-cann_container}
BS=${1:-48}

echo "===== [small-shape] 1/4 dump bs=$BS input via golden ====="
docker exec -e MHCS_GOLDEN_SIZE=$BS -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e PATH=/usr/local/python3.11.15/bin:/usr/bin:/bin \
  "$C" python3 /root/HcPre/scripts/mhc_pre_sinkhorn/run_golden.py \
  --dump-input /root/HcPre/mhc_input_bs${BS}.bin \
  --save /root/HcPre/golden_refs/mhcs_golden_bs${BS}.pt 2>&1 | grep -E "golden.*(dumped|saved|outputs)"

echo "===== [small-shape] 2/4 sync (debug CMakeLists) + rebuild + example ====="
bash /mnt/d/proj/HcPre/scripts/sync_repo.sh | tail -1
bash /mnt/d/proj/HcPre/scripts/mhc_pre_sinkhorn/build.sh all

echo "===== [small-shape] 3/4 verify debug .o ====="
docker exec "$C" bash <<INNER
OBJDUMP=/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/llvm-objdump
V=/root/HcPre/mhc_pre_install/vendors/custom_transformer
for OBJ in \$(find \$V/op_impl/ai_core/tbe/kernel/ascend910_93/mhc_pre_sinkhorn -name "*.o" | head -2); do
  n=\$("\$OBJDUMP" --section-headers "\$OBJ" 2>/dev/null | grep -c "\.debug_")
  echo "\$(basename \$OBJ): size=\$(stat -c%s "\$OBJ") debug_sections=\$n"
done
INNER

echo "===== [small-shape] 4/4 launch sim (background) ====="
RUNNER=/root/run_mhcs_sim_small.sh
docker exec "$C" bash -c "cat > $RUNNER" <<EOS
#!/bin/bash
source /root/HcPre/scripts/container_env.sh
V=/root/HcPre/mhc_pre_install/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH=\$V
export LD_LIBRARY_PATH=\$ASCEND_HOME_PATH/x86_64-linux/simulator/Ascend910_9382/lib:\$V/op_api/lib:\$LD_LIBRARY_PATH
cd /root/HcPre/ops-transformer/build
rm -rf /root/HcPre/sim_out_mhcs
mkdir -p /root/HcPre/logs
msprof op simulator \\
  --application="\$PWD/test_aclnn_mhc_pre_sinkhorn_hcshape mhc_input_bs${BS}.bin /root/HcPre/mhcs_output_bs${BS}.bin" \\
  --output=/root/HcPre/sim_out_mhcs \\
  --kernel-name=MhcPreSinkhorn \\
  --launch-count=1 \\
  --soc-version=Ascend910_9382 \\
  --timeout=\${MHCS_SIM_TIMEOUT_MIN:-120} > /root/HcPre/logs/mhcs_sim_bs${BS}.log 2>&1
echo "EXIT_CODE=\$?" >> /root/HcPre/logs/mhcs_sim_bs${BS}.log
echo "RUN_FINISHED \$(date)" >> /root/HcPre/logs/mhcs_sim_bs${BS}.log
EOS
docker exec "$C" chmod +x "$RUNNER"
docker exec -d "$C" setsid "$RUNNER"
echo "[small-shape] background sim launched (bs=$BS), log: /root/HcPre/logs/mhcs_sim_bs${BS}.log"
