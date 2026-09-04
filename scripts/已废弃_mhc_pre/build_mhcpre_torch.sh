#!/bin/bash
# Build the mhc_pre_torch pybind module in the container + run sim app via msprof.
# Usage: bash build_mhcpre_torch.sh build|sim
set -euo pipefail

C=${CANN_CONTAINER:-cann_container}
ENVSH=/root/HcPre/scripts/container_env.sh
INSTALL_DIR=/root/HcPre/mhc_pre_install
CUST="$INSTALL_DIR/vendors/custom_transformer"
SCRIPTS=/root/HcPre/scripts/mhc_pre
OUTSO=/root/HcPre/scripts/mhc_pre/mhc_pre_torch.so

step() { echo -e "\n===== [mhc_torch] $* ====="; }

build() {
  step "compile pybind wrapper"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
PYBIN=\$(which python3)
TNPU_LOC=\$(python3 -m pip show torch-npu | awk '/^Location:/ {print \$2}')
g++ $SCRIPTS/mhc_pre_torch.cpp -shared -fPIC -o $OUTSO \\
  -I \$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \\
  -I \$TNPU_LOC/torch_npu/include \\
  -I \$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))") \\
  -I \$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'include/torch/csrc/api/include'))") \\
  -I \$ASCEND_HOME_PATH/include \\
  -L \$ASCEND_HOME_PATH/lib64 -L \$ASCEND_HOME_PATH/x86_64-linux/lib64 \\
  -lascendcl -lnnopbase -lc_sec \\
  -L \$TNPU_LOC/torch_npu/lib -ltorch_npu \\
  -Wl,-rpath=$CUST/op_api/lib
ls -la $OUTSO
EOS
}

sim() {
  step "run mhc_pre sim app under msprof (torch path)"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
export LD_LIBRARY_PATH=\$ASCEND_HOME_PATH/x86_64-linux/simulator/Ascend910_9382/lib:\$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=$CUST
export MHC_PRE_FAULTLOG=/root/HcPre/mhc_fault.log
export PYTHONUNBUFFERED=1
rm -rf /root/HcPre/sim_out_mhc; mkdir -p /root/HcPre/sim_out_mhc
set +e
msprof op simulator \\
  --application="python3 $SCRIPTS/mhcpre_sim_app.py" \\
  --output=/root/HcPre/sim_out_mhc \\
  --kernel-name=mhc_pre \\
  --launch-count=1 \\
  --soc-version=Ascend910_9382 \\
  --timeout=\${MHC_SIM_TIMEOUT_MIN:-60}
rc=\$?
set -e
echo "msprof exit code: \$rc"
echo "--- fault log ---"
cat /root/HcPre/mhc_fault.log 2>/dev/null | head -30
ls -la /root/HcPre/mhc_output_bs512.bin 2>/dev/null || { echo "no output bin"; exit 1; }
EOS
}

case "${1:-}" in
  build) build ;;
  sim) sim ;;
  all) build; sim ;;
  *) grep '^#' "$0" | head -3; exit 1 ;;
esac
