#!/bin/bash
# hc_pre operator: build + install + pybind + msprof op simulator, inside the local docker cann container.
#
# Usage (from Windows via WSL):
#   wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/build_hcpre.sh deps   # one-time python deps (torch etc.)
#   wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/build_hcpre.sh build  # sync + build op pkg + install + pybind
#   wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/build_hcpre.sh sim    # msprof op simulator (A3 / Ascend910_9382)
#   wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/build_hcpre.sh all    # deps + build + sim
#
# Env overrides:
#   CANN_CONTAINER=cann_container   HC_BUILD_SOC=ascend910_93   HC_SIM_SOC=Ascend910_9382
#   HC_SIM_TIMEOUT_MIN=30           HC_PRE_COMPARE=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C=${CANN_CONTAINER:-cann_container}
WS=/root/HcPre
REPO=$WS/vllm-ascend
ENVSH=$WS/scripts/container_env.sh
OP_INSTALL_DIR=$REPO/vllm_ascend/_cann_ops_custom
VENDOR_DIR=$OP_INSTALL_DIR/vendors/custom_transformer
SIM_OUT=$WS/sim_out
BUILD_SOC=${HC_BUILD_SOC:-ascend910_93}
PYBIND_SOC=${HC_PYBIND_SOC:-ascend910_9382}
SIM_SOC=${HC_SIM_SOC:-Ascend910_9382}

step() { echo -e "\n===== [build_hcpre] $* ====="; }

deps() {
  step "install python deps in container (torch/torch-npu/pybind11/cmake/ninja)"
  docker exec "$C" bash -c '
    set -e
    export PATH=/usr/local/python3.11.15/bin:$PATH
    if python3 -c "import torch, torch_npu, pybind11, regex, scipy, attr, cloudpickle, tornado" 2>/dev/null; then
      echo "python deps already present"
    else
      # torch-npu requires the +cpu build of torch
      python3 -m pip install torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu
      python3 -m pip install torch-npu==2.10.0.post4 pybind11 "cmake>=3.26" ninja numpy pyyaml psutil decorator regex scipy attrs cloudpickle tornado
    fi
  '
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
python3 -c "import torch, torch_npu; print(\"torch\", torch.__version__, \"| torch_npu\", torch_npu.__version__)"
echo "cmake: \$(cmake --version | head -1)"
EOS
}

build() {
  step "1/3 sync workspace into container"
  bash "$SCRIPT_DIR/sync_repo.sh"

  step "2/3 build hc_pre op run package (soc=$BUILD_SOC)"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
cd $REPO/csrc
rm -rf output build_out
# force kernel recompile: ninja caches binary .done stamps and does not track the opc-options ini
rm -rf build/binary build/autogen
bash $WS/scripts/container_monitor_run.sh $WS/logs/op_build.log \\
  bash build.sh --pkg --ops=hc_pre --soc=$BUILD_SOC -j\$(nproc)

# ---- install run package into vllm_ascend/_cann_ops_custom (mirrors csrc/build_aclnn.sh) ----
custom_ops_install_dir=$OP_INSTALL_DIR
mkdir -p "\$custom_ops_install_dir"
find "\$custom_ops_install_dir" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf -- {} +
shopt -s nullglob
installer_candidates=(./build/cann-ops-transformer*.run)
shopt -u nullglob
echo "installer candidates: \${installer_candidates[@]}"
[[ \${#installer_candidates[@]} -eq 1 ]] || { echo "ERROR: expected 1 installer" >&2; exit 1; }
chmod +x "\${installer_candidates[0]}"
"\${installer_candidates[0]}" --install-path="\$custom_ops_install_dir"
[[ -d "\$custom_ops_install_dir/vendors/custom_transformer/scripts" ]] && chmod u+w "\$custom_ops_install_dir/vendors/custom_transformer/scripts"
echo "op package installed to \$custom_ops_install_dir"
EOS

  step "3/3 build pybind module vllm_ascend_C.so"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
PYBIN=\$(which python3)
TNPU_LOC=\$(python3 -m pip show torch-npu | awk '/^Location:/ {print \$2}')
cmake -S $REPO -B $REPO/build-pybind \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \\
  -DASCEND_HOME_PATH=\$ASCEND_HOME_PATH \\
  -DPYTHON_EXECUTABLE=\$PYBIN \\
  -DPYTHON_INCLUDE_PATH=\$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \\
  -DCMAKE_INSTALL_PREFIX=$REPO/vllm_ascend \\
  -DCMAKE_PREFIX_PATH=\$(python3 -m pybind11 --cmakedir) \\
  -DSOC_VERSION=$PYBIND_SOC \\
  -DFETCHCONTENT_BASE_DIR=$REPO/.deps \\
  -DTORCH_NPU_PATH=\$TNPU_LOC/torch_npu
bash $WS/scripts/container_monitor_run.sh $WS/logs/pybind_build.log \\
  cmake --build $REPO/build-pybind --target install -j\$(nproc)
PYBIND_SO_REAL=\$(ls $REPO/vllm_ascend/vllm_ascend_C*.so | head -1)
echo "pybind module: \$PYBIND_SO_REAL"
ls -la "\$PYBIND_SO_REAL"
EOS

  step "build done. artifacts:"
  docker exec "$C" bash -c "ls -la $REPO/vllm_ascend/vllm_ascend_C*.so $VENDOR_DIR/op_api/lib/ 2>/dev/null | head"
}

sim() {
  step "run msprof op simulator (soc=$SIM_SOC, kernel=hc_pre)"
  docker exec -i "$C" bash -s <<EOS
set -e
source $ENVSH
export LD_LIBRARY_PATH=\$ASCEND_HOME_PATH/x86_64-linux/simulator/$SIM_SOC/lib:\$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=$VENDOR_DIR
PYBIND_SO_REAL=\$(ls $REPO/vllm_ascend/vllm_ascend_C*.so | head -1)
export HC_PRE_PYBIND_SO=\$PYBIND_SO_REAL
export HC_PRE_COMPARE=${HC_PRE_COMPARE:-1}
export HC_PRE_DUMP=${HC_PRE_DUMP:-}
export HC_PRE_SIZE=${HC_PRE_SIZE:-512}
[[ -f "\$PYBIND_SO_REAL" ]] || { echo "vllm_ascend_C.so missing; run build first" >&2; exit 1; }
[[ -d "$VENDOR_DIR" ]] || { echo "custom op package missing; run build first" >&2; exit 1; }

rm -rf $SIM_OUT; mkdir -p $SIM_OUT
set +e
msprof op simulator \\
  --application="python3 $WS/scripts/hc_pre/hcpre_sim_app.py" \\
  --output=$SIM_OUT \\
  --kernel-name=HcPre \\
  --launch-count=1 \\
  --soc-version=$SIM_SOC \\
  --timeout=${HC_SIM_TIMEOUT_MIN:-30}
rc=\$?
set -e
echo "msprof exit code: \$rc"

csv_count=\$(find $SIM_OUT -name "*_instr_exe.csv" 2>/dev/null | wc -l)
if [[ "\$csv_count" -eq 0 ]]; then
  echo "ERROR: no *_instr_exe.csv produced under $SIM_OUT (simulation failed)"
  find $SIM_OUT -name "msprof.log" -exec tail -30 {} \; 2>/dev/null || true
  exit "\$rc"
fi
echo "simulation finished, \$csv_count core CSV files produced"

python3 $WS/scripts/parse_sim_csv.py $SIM_OUT
EOS

  step "sim outputs kept in container at $SIM_OUT"
}

case "${1:-}" in
  deps) deps ;;
  build) build ;;
  sim) sim ;;
  all) deps; build; sim ;;
  *) grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -20; exit 1 ;;
esac
