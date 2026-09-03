#!/bin/bash
# mhc_pre_sinkhorn: build op package + install + compile example, in the cann container.
# Usage: wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/mhc_pre_sinkhorn/build.sh build|example|all
set -euo pipefail

C=${CANN_CONTAINER:-cann_container}
ENVSH=/root/HcPre/scripts/container_env.sh
REPO=/root/HcPre/ops-transformer
INSTALL_DIR=/root/HcPre/mhc_pre_install
BUILD_SOC=${MHCS_BUILD_SOC:-ascend910_93}
MONITOR=/root/HcPre/scripts/container_monitor_run.sh

step() { echo -e "\n===== [mhcs-build] $* ====="; }

build() {
  step "build mhc_pre_sinkhorn op package (soc=$BUILD_SOC)"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
cd $REPO
rm -rf output build_out build/binary build/autogen
bash $MONITOR /root/HcPre/logs/mhcs_op_build.log \
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
  step "compile hcshape example (g++)"
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

case "${1:-}" in
  build) build ;;
  example) example ;;
  all) build; example ;;
  *) grep '^#' "$0" | head -3; exit 1 ;;
esac
