#!/bin/bash
# Debug: why docker exec -d didn't run? Try nohup approach with explicit bash -c file
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}
# write a runner script inside container first (avoids all quoting layers)
docker exec "$C" bash -c 'cat > /root/run_mhcs_sim.sh <<EOF
#!/bin/bash
source /root/HcPre/scripts/container_env.sh
V=/root/HcPre/mhc_pre_install/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH=\$V
export LD_LIBRARY_PATH=\$ASCEND_HOME_PATH/x86_64-linux/simulator/Ascend910_9382/lib:\$V/op_api/lib:\$LD_LIBRARY_PATH
cd /root/HcPre/ops-transformer/build
rm -rf /root/HcPre/sim_out_mhcs
mkdir -p /root/HcPre/logs
msprof op simulator \
  --application="\$PWD/test_aclnn_mhc_pre_sinkhorn_hcshape mhc_input_bs512.bin /root/HcPre/mhcs_output_bs512.bin" \
  --output=/root/HcPre/sim_out_mhcs \
  --kernel-name=MhcPreSinkhorn \
  --launch-count=1 \
  --soc-version=Ascend910_9382 \
  --timeout=240 > /root/HcPre/logs/mhcs_rerun.log 2>&1
echo "EXIT_CODE=\$?" >> /root/HcPre/logs/mhcs_rerun.log
echo "RUN_FINISHED \$(date)" >> /root/HcPre/logs/mhcs_rerun.log
EOF
chmod +x /root/run_mhcs_sim.sh'
# launch detached via setsid nohup
docker exec -d "$C" setsid /root/run_mhcs_sim.sh
sleep 8
docker exec "$C" bash -c 'ps aux | grep -v grep | grep -cE msprof || true; ls -la /root/HcPre/logs/mhcs_rerun.log 2>/dev/null'
