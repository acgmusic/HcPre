#!/bin/bash
# Run mhc_pre_sinkhorn hcshape example under msprof op simulator (A3).
# Usage: wsl -d Ubuntu-22.04 -- bash .../sim.sh [--bg]
#   default: foreground with output to console
#   --bg:    background (detached), log at /root/HcPre/logs/mhcs_sim.log
set -euo pipefail

C=${CANN_CONTAINER:-cann_container}
ENVSH=/root/HcPre/scripts/container_env.sh
REPO=/root/HcPre/ops-transformer
INSTALL_DIR=/root/HcPre/mhc_pre_install
SIM_SOC=Ascend910_9382
LOG=/root/HcPre/logs/mhcs_sim.log
OUT_BIN=/root/HcPre/mhcs_output_bs512.bin
BG=0
[ "${1:-}" = "--bg" ] && BG=1

RUNNER=/root/run_mhcs_sim.sh
docker exec "$C" bash -c "cat > $RUNNER" <<'EOS'
#!/bin/bash
source /root/HcPre/scripts/container_env.sh
V=/root/HcPre/mhc_pre_install/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH=$V
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/simulator/Ascend910_9382/lib:$V/op_api/lib:$LD_LIBRARY_PATH
cd /root/HcPre/ops-transformer/build
rm -rf /root/HcPre/sim_out_mhcs
mkdir -p /root/HcPre/logs
msprof op simulator \
  --application="$PWD/test_aclnn_mhc_pre_sinkhorn_hcshape mhc_input_bs512.bin /root/HcPre/mhcs_output_bs512.bin" \
  --output=/root/HcPre/sim_out_mhcs \
  --kernel-name=MhcPreSinkhorn \
  --launch-count=1 \
  --soc-version=Ascend910_9382 \
  --timeout=${MHCS_SIM_TIMEOUT_MIN:-240} > /root/HcPre/logs/mhcs_sim.log 2>&1
echo "EXIT_CODE=$?" >> /root/HcPre/logs/mhcs_sim.log
echo "RUN_FINISHED $(date)" >> /root/HcPre/logs/mhcs_sim.log
EOS
docker exec "$C" chmod +x "$RUNNER"

docker start "$C" >/dev/null 2>&1 || true

if [ "$BG" = "1" ]; then
  docker exec -d "$C" setsid "$RUNNER"
  echo "[sim] launched in background; poll: docker exec $C tail -5 $LOG"
else
  docker exec -i "$C" "$RUNNER"
fi
