#!/bin/bash
# Run the ORIGINAL mhc_pre example (default shape) under msprof to bisect:
# crash = package issue in sim env; ok = my wrapper's fault
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}
docker exec -i "$C" bash -s <<'EOS'
set -e
source /root/HcPre/scripts/container_env.sh
V=/root/HcPre/mhc_pre_install/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH=$V
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/x86_64-linux/simulator/Ascend910_9382/lib:$V/op_api/lib:$LD_LIBRARY_PATH
cd /root/HcPre/ops-transformer/build
rm -rf /root/HcPre/sim_out_mhc_orig; mkdir -p /root/HcPre/sim_out_mhc_orig
set +e
msprof op simulator \
  --application="$PWD/test_aclnn_mhc_pre" \
  --output=/root/HcPre/sim_out_mhc_orig \
  --kernel-name=mhc_pre \
  --launch-count=1 \
  --soc-version=Ascend910_9382 \
  --timeout=30
echo "RC=$?"
EOS
