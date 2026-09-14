#!/bin/bash
# Bisect mhc_pre segfault via the variant example under msprof.
# usage: bash test_variant.sh <bs> <d> <mode>
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}
BS=${1:-8}
D=${2:-2560}
MODE=${3:-0}
docker exec -i "$C" bash -s <<EOS
set -e
source /root/HcPre/scripts/container_env.sh
V=/root/HcPre/mhc_pre_install/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH=\$V
export LD_LIBRARY_PATH=\$ASCEND_HOME_PATH/x86_64-linux/simulator/Ascend910_9382/lib:\$V/op_api/lib:\$LD_LIBRARY_PATH
cd /root/HcPre/ops-transformer/build
rm -rf /root/HcPre/sim_out_mhc_var; mkdir -p /root/HcPre/sim_out_mhc_var
set +e
msprof op simulator \\
  --application="\$PWD/test_aclnn_mhc_pre_variant $BS $D $MODE" \\
  --output=/root/HcPre/sim_out_mhc_var \\
  --kernel-name=MhcPre \\
  --launch-count=1 \\
  --soc-version=Ascend910_9382 \\
  --timeout=20
echo "RC=\$?"
EOS
