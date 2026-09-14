#!/bin/bash
# Build + install mhc_pre (ops-transformer) for A3 in the cann container.
# Usage: wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/mhc_pre/build_mhcpre.sh build
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C=${CANN_CONTAINER:-cann_container}
REPO=/root/HcPre/ops-transformer          # synced local copy (has 910_93 support)
ENVSH=/root/HcPre/scripts/container_env.sh
INSTALL_DIR=/root/HcPre/mhc_pre_install
VENDOR_DIR=$INSTALL_DIR/vendors/ops_transformer
BUILD_SOC=${MHC_BUILD_SOC:-ascend910_93}

step() { echo -e "\n===== [build_mhcpre] $* ====="; }

build_example() {
  step "compile custom hcshape example (g++ direct, mirroring build.sh run_example)"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
cd $REPO
mkdir -p build
CUST_LIB="$INSTALL_DIR/vendors/custom_transformer/op_api/lib"
CUST_INC="$INSTALL_DIR/vendors/custom_transformer/op_api/include/aclnnop"
COMMON_INC="$REPO/common/include/external"
for src in mhc/mhc_pre/examples/test_aclnn_mhc_pre_hcshape.cpp mhc/mhc_pre/examples/test_aclnn_mhc_pre.cpp mhc/mhc_pre/examples/test_aclnn_mhc_pre_variant.cpp; do
  name=\$(basename "\$src" .cpp)
  g++ \$src \\
    -I \$CUST_INC -I \$COMMON_INC -I \$ASCEND_HOME_PATH/include -I \$ASCEND_HOME_PATH/include/aclnn \\
    -L \$CUST_LIB -L \$ASCEND_HOME_PATH/lib64 \\
    -lopapi_math -lcust_opapi -lascendcl -lnnopbase -lc_sec \\
    -o build/\$name \\
    -Wl,-rpath=\$CUST_LIB
done
ls -la build/test_aclnn_mhc_pre_hcshape build/test_aclnn_mhc_pre
EOS
}

build() {
  step "1/2 build mhc_pre op package (soc=$BUILD_SOC)"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
cd $REPO
rm -rf output build_out build/binary build/autogen
bash /root/HcPre/scripts/container_monitor_run.sh /root/HcPre/logs/mhc_op_build.log \
  bash build.sh --pkg --ops=mhc_pre --soc=$BUILD_SOC -j\$(nproc) 2>&1 | tail -5 || {
    tail -40 /root/HcPre/logs/mhc_op_build.log; exit 1; }
EOS

  step "2/2 install run package"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
cd $REPO
mkdir -p $INSTALL_DIR
shopt -s nullglob
installers=(./build/cann-ops-transformer*.run)
shopt -u nullglob
echo "installers: \${installers[@]}"
[[ \${#installers[@]} -ge 1 ]] || { echo "no installer found"; ls ./build/ | head; exit 1; }
chmod +x "\${installers[0]}"
"\${installers[0]}" --install-path=$INSTALL_DIR
echo "installed:"
ls $INSTALL_DIR/vendors/*/op_impl/ai_core/tbe/kernel/config/ascend910_93/ 2>/dev/null | head
EOS
}

sim() {
  step "run mhc_pre hcshape example under msprof op simulator"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
V=$INSTALL_DIR/vendors/custom_transformer
export ASCEND_CUSTOM_OPP_PATH=\$V
export LD_LIBRARY_PATH=\$ASCEND_HOME_PATH/x86_64-linux/simulator/Ascend910_9382/lib:\$V/op_api/lib:\$LD_LIBRARY_PATH
cp -f /root/HcPre/mhc_input_bs512.bin $REPO/build/
cd $REPO/build
rm -rf /root/HcPre/sim_out_mhc; mkdir -p /root/HcPre/sim_out_mhc
set +e
msprof op simulator \\
  --application="\$PWD/test_aclnn_mhc_pre_hcshape mhc_input_bs512.bin /root/HcPre/mhc_output_bs512.bin" \\
  --output=/root/HcPre/sim_out_mhc \\
  --kernel-name=MhcPre \\
  --launch-count=1 \\
  --soc-version=Ascend910_9382 \\
  --timeout=\${MHC_SIM_TIMEOUT_MIN:-90}
rc=\$?
set -e
echo "msprof exit code: \$rc"
ls -la /root/HcPre/mhc_output_bs512.bin 2>/dev/null || { echo "no output bin"; exit 1; }
EOS
}

case "${1:-}" in
  build) build ;;
  example) build_example ;;
  sim) sim ;;
  all) build; build_example; sim ;;
  *) grep '^#' "$0" | head -3; exit 1 ;;
esac
