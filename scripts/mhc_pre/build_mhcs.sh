#!/bin/bash
# Build + run mhc_pre_sinkhorn with hc_pre-equivalent inputs (bs=512, seed 1024) under A3 simulator.
# Usage: bash build_mhcs.sh build|example|sim|all
set -euo pipefail

C=${CANN_CONTAINER:-cann_container}
ENVSH=/root/HcPre/scripts/container_env.sh
REPO=/root/HcPre/ops-transformer
INSTALL_DIR=/root/HcPre/mhc_pre_install
BUILD_SOC=ascend910_93
SIM_SOC=Ascend910_9382

step() { echo -e "\n===== [mhcs] $* ====="; }

build() {
  step "build mhc_pre_sinkhorn op package"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
cd $REPO
rm -rf output build_out build/binary build/autogen
bash /root/HcPre/scripts/container_monitor_run.sh /root/HcPre/logs/mhcs_op_build.log \
  bash build.sh --pkg --ops=mhc_pre_sinkhorn --soc=$BUILD_SOC -j\$(nproc) 2>&1 | tail -3 || {
    tail -40 /root/HcPre/logs/mhcs_op_build.log; exit 1; }
EOS

  step "install run package"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
cd $REPO
mkdir -p $INSTALL_DIR
shopt -s nullglob
installers=(./build/cann-ops-transformer*.run)
shopt -u nullglob
[[ \${#installers[@]} -ge 1 ]] || { echo "no installer"; exit 1; }
chmod +x "\${installers[0]}"
"\${installers[0]}" --install-path=$INSTALL_DIR
ls $INSTALL_DIR/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/config/ascend910_93/ | grep mhc
EOS
}

example() {
  step "compile hcshape example"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
cd $REPO
mkdir -p build
CUST_LIB="$INSTALL_DIR/vendors/custom_transformer/op_api/lib"
CUST_INC="$INSTALL_DIR/vendors/custom_transformer/op_api/include/aclnnop"
COMMON_INC="$REPO/common/include/external"
g++ mhc/mhc_pre_sinkhorn/examples/test_aclnn_mhc_pre_sinkhorn_hcshape.cpp \\
  -I \$CUST_INC -I \$COMMON_INC -I \$ASCEND_HOME_PATH/include -I \$ASCEND_HOME_PATH/include/aclnn \\
  -L \$CUST_LIB -L \$ASCEND_HOME_PATH/lib64 \\
  -lopapi_math -lcust_opapi -lascendcl -lnnopbase -lc_sec \\
  -o build/test_aclnn_mhc_pre_sinkhorn_hcshape \\
  -Wl,-rpath=\$CUST_LIB
ls -la build/test_aclnn_mhc_pre_sinkhorn_hcshape
EOS
}

sim() {
  step "run under msprof op simulator"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
V=$INSTALL_DIR/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH=\$V
export LD_LIBRARY_PATH=\$ASCEND_HOME_PATH/x86_64-linux/simulator/$SIM_SOC/lib:\$V/op_api/lib:\$LD_LIBRARY_PATH
cp -f /root/HcPre/mhc_input_bs512.bin $REPO/build/
cd $REPO/build
rm -rf /root/HcPre/sim_out_mhcs; mkdir -p /root/HcPre/sim_out_mhcs
set +e
msprof op simulator \\
  --application="\$PWD/test_aclnn_mhc_pre_sinkhorn_hcshape mhc_input_bs512.bin /root/HcPre/mhcs_output_bs512.bin" \\
  --output=/root/HcPre/sim_out_mhcs \\
  --kernel-name=MhcPreSinkhorn \\
  --launch-count=1 \\
  --soc-version=$SIM_SOC \\
  --timeout=\${MHCS_SIM_TIMEOUT_MIN:-120}
rc=\$?
set -e
echo "msprof exit code: \$rc"
ls -la /root/HcPre/mhcs_output_bs512.bin 2>/dev/null || { echo "no output bin"; exit 1; }
EOS
}

case "${1:-}" in
  build) build ;;
  example) example ;;
  sim) sim ;;
  all) build; example; sim ;;
  *) grep '^#' "$0" | head -3; exit 1 ;;
esac
