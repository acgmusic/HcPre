#!/bin/bash
# ============================================================================
# hc_pre 一键运行脚本（编译/上板精度/仿真/上板性能）
#
# 用法:
#   bash run.sh build  [--debug|--release]        # 编译+安装算子包+pybind(默认 release)
#   bash run.sh board  [b] [bs]                   # 上板跑精度用例并对比 golden
#   bash run.sh sim    [b] [bs]                   # msprof 仿真(精度对比内嵌)
#   bash run.sh perf   [b] [bs]                   # 上板 msprof 性能, 打印 Task Duration + op_summary csv 路径
#   bash run.sh all    [--debug|--release] [b] [bs]   # build + board + sim + perf
#
# 环境变量:
#   HC_B / HC_BS        等效于位置参数 b / bs (默认 b=1, bs=2)
#   HC_MODE             debug|release (默认 release, 可被 --debug/--release 覆盖)
#   HC_SIM_TIMEOUT_MIN  仿真超时分钟数 (默认 30)
#   CONTAINER           容器名 (默认 cann_container; 无 docker 的裸机场景设 CONTAINER=none)
#
# 备注:
#   - 所有随机种子固定 1024, golden 与上板/仿真用同一组数据
#   - b*bs 合并为算子内部的 bs 维 (x shape = (b, bs, 4, 4096))
#   - board/perf 需要真实 A3 NPU; 在无 NPU 的容器上会失败, 这是预期行为
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C=${CONTAINER:-cann_container}
WS=/root/HcPre
REPO=$WS/vllm-ascend
ENVSH=$WS/scripts/container_env.sh
VENDOR_DIR=$REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
BUILD_SOC=${HC_BUILD_SOC:-ascend910_93}
PYBIND_SOC=${HC_PYBIND_SOC:-ascend910_9382}
SIM_SOC=${HC_SIM_SOC:-Ascend910_9382}
MODE=${HC_MODE:-release}
B=${HC_B:-1}
BS=${HC_BS:-2}
OPNAME=hc_pre

step() { echo -e "\n===== [hc_pre.run] $* ====="; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# 无容器模式: 直接在当前机器执行 (裸机 A3 部署)
if [ "$C" = "none" ]; then
  run() { bash -c "$*"; }
  runi() { bash -c "$*"; }
  # 裸机路径约定: 脚本仓在 ~/HcPre
  WS=${HCPRE_HOME:-$HOME/HcPre}
  REPO=$WS/vllm-ascend
  ENVSH=$WS/scripts/container_env.sh
  VENDOR_DIR=$REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
else
  docker start "$C" >/dev/null 2>&1 || true
  run()  { docker exec "$C" bash -c "$*"; }
  runi() { docker exec -i "$C" bash -s; }
fi

# ---------------------------------------------------------------------------
# 1. 编译/安装 (debug / release 由 CMakeLists 中 ccec_g 行控制)
# ---------------------------------------------------------------------------
do_build() {
  step "build $OPNAME ($MODE, soc=$BUILD_SOC)"

  # 确保代码同步到容器(容器模式)
  if [ "$C" != "none" ]; then
    bash "$SCRIPT_DIR/../sync_repo.sh" >/dev/null
  fi

  # debug/release 由 CMakeLists 的 CMAKE_BUILD_TYPE 条件块控制
  BUILD_TYPE_FLAG=""
  [ "$MODE" = "debug" ] && BUILD_TYPE_FLAG="--build-type=Debug"

  # 编译算子包 (容器内跑)
  runi <<EOS
set -e
source $ENVSH
cd $REPO/csrc
rm -rf output build_out build/binary build/autogen
bash $WS/scripts/container_monitor_run.sh $WS/logs/hc_op_build.log \\
  bash build.sh --pkg $BUILD_TYPE_FLAG --ops=$OPNAME --soc=$BUILD_SOC -j\$(nproc) 2>&1 | tail -3 || {
    tail -40 $WS/logs/hc_op_build.log; exit 1; }
# verify debug info
if [ "$MODE" = "debug" ]; then
  OBJDUMP=\$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump
  OBJ=\$(find build/binary/ascend910_93/bin/hc_pre -name "*.o" ! -name "*relocatable*" | head -1)
  n=\$(\$OBJDUMP --section-headers "\$OBJ" 2>/dev/null | grep -c "\.debug_")
  echo "[build] kernel .o: \$(basename \$OBJ) size=\$(stat -c%s "\$OBJ") debug_sections=\$n"
  [ "\$n" -gt 0 ] || { echo "ERROR: debug build requested but .o has no debug sections"; exit 1; }
else
  OBJ=\$(find build/binary/ascend910_93/bin/hc_pre -name "*.o" ! -name "*relocatable*" | head -1)
  echo "[build] kernel .o: \$(basename \$OBJ) size=\$(stat -c%s "\$OBJ") (release)"
fi
EOS

  # 安装 run 包
  runi <<EOS
set -e
source $ENVSH
cd $REPO/csrc
mkdir -p $REPO/vllm_ascend/_cann_ops_custom
shopt -s nullglob
installers=(./build/cann-ops-transformer*.run)
shopt -u nullglob
[[ \${#installers[@]} -eq 1 ]] || die "expected 1 installer"
chmod +x "\${installers[0]}"
"\${installers[0]}" --install-path=$REPO/vllm_ascend/_cann_ops_custom
[[ -d $VENDOR_DIR/scripts ]] && chmod u+w $VENDOR_DIR/scripts
echo "installed to $VENDOR_DIR"
EOS

  # pybind 编译 (与 build_hcpre.sh 相同流程)
  runi <<EOS
set -e
source $ENVSH
PYBIN=\$(which python3)
TNPU_LOC=\$(python3 -m pip show torch-npu | awk '/^Location:/ {print \$2}')
cmake -S $REPO -B $REPO/build-pybind \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DASCEND_HOME_PATH=\$ASCEND_HOME_PATH \\
  -DPYTHON_EXECUTABLE=\$PYBIN \\
  -DPYTHON_INCLUDE_PATH=\$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \\
  -DCMAKE_INSTALL_PREFIX=$REPO/vllm_ascend \\
  -DCMAKE_PREFIX_PATH=\$(python3 -m pybind11 --cmakedir) \\
  -DSOC_VERSION=$PYBIND_SOC \\
  -DFETCHCONTENT_BASE_DIR=$REPO/.deps \\
  -DTORCH_NPU_PATH=\$TNPU_LOC/torch_npu > /dev/null
cmake --build $REPO/build-pybind --target install -j\$(nproc) > /dev/null 2>&1 || \\
  cmake --build $REPO/build-pybind --target install -j\$(nproc)
ls -la $REPO/vllm_ascend/vllm_ascend_C*.so
EOS
  step "build done ($MODE)"
}

# ---------------------------------------------------------------------------
# 2. 上板精度 (调 torch.ops._C_ascend.npu_hc_pre_v2, 对比 CPU golden)
# ---------------------------------------------------------------------------
do_board() {
  step "board accuracy test (b=$B, bs=$BS)"
  runi <<EOS
set -e
source $ENVSH
export ASCEND_CUSTOM_OPP_PATH=$VENDOR_DIR
PYBIND_SO=\$(ls $REPO/vllm_ascend/vllm_ascend_C*.so | head -1)
export HC_PRE_PYBIND_SO=\$PYBIND_SO
export HC_PRE_SIZE=\$(( $B * $BS ))
export HC_PRE_BATCH=$B
export HC_PRE_COMPARE=1
python3 $WS/scripts/hc_pre/hcpre_sim_app.py
EOS
}

# ---------------------------------------------------------------------------
# 3. msprof 仿真 (精度对比内嵌在 sim app)
# ---------------------------------------------------------------------------
do_sim() {
  step "msprof simulator (b=$B, bs=$BS, soc=$SIM_SOC)"
  runi <<EOS
set -e
source $ENVSH
export LD_LIBRARY_PATH=\$ASCEND_HOME_PATH/x86_64-linux/simulator/$SIM_SOC/lib:\$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=$VENDOR_DIR
PYBIND_SO=\$(ls $REPO/vllm_ascend/vllm_ascend_C*.so | head -1)
export HC_PRE_PYBIND_SO=\$PYBIND_SO
export HC_PRE_SIZE=\$(( $B * $BS ))
export HC_PRE_BATCH=$B
export HC_PRE_COMPARE=1
rm -rf $WS/sim_out; mkdir -p $WS/sim_out
msprof op simulator \\
  --application="python3 $WS/scripts/hc_pre/hcpre_sim_app.py" \\
  --output=$WS/sim_out \\
  --kernel-name=HcPre \\
  --launch-count=1 \\
  --soc-version=$SIM_SOC \\
  --timeout=${HC_SIM_TIMEOUT_MIN:-30} || true
echo "--- sim csv summary ---"
python3 $WS/scripts/parse_sim_csv.py $WS/sim_out 2>/dev/null | head -30 || echo "(no csv)"
EOS
}

# ---------------------------------------------------------------------------
# 4. 上板性能 (msprof 板端模式, 打印 Task Duration + op_summary csv 路径)
# ---------------------------------------------------------------------------
do_perf() {
  step "board performance (msprof, b=$B, bs=$BS)"
  runi <<EOS
set -e
source $ENVSH
which msprof >/dev/null 2>&1 || { echo "msprof not found"; exit 1; }
export ASCEND_CUSTOM_OPP_PATH=$VENDOR_DIR
PYBIND_SO=\$(ls $REPO/vllm_ascend/vllm_ascend_C*.so | head -1)
export HC_PRE_PYBIND_SO=\$PYBIND_SO
export HC_PRE_SIZE=\$(( $B * $BS ))
export HC_PRE_BATCH=$B
export HC_PRE_COMPARE=0    # 性能跑不对比精度
rm -rf $WS/perf_out; mkdir -p $WS/perf_out
msprof --application="python3 $WS/scripts/hc_pre/hcpre_sim_app.py" \\
  --output=$WS/perf_out \\
  --aic-metrics=PipeUtilization 2>&1 | tail -5 || true
echo
echo "=== op_summary csv ==="
find $WS/perf_out -name "op_summary_*.csv" -exec ls -la {} \; 2>/dev/null || echo "(none found)"
CSV=\$(find $WS/perf_out -name "op_summary_*.csv" | head -1)
if [ -n "\$CSV" ]; then
  echo
  echo "=== Task Duration (us) ==="
  python3 - "\$CSV" <<'PYEOF'
import csv, sys
with open(sys.argv[1]) as f:
    rows = list(csv.DictReader(f))
for r in rows:
    name = r.get("Op Name") or r.get("Name") or "?"
    dur = r.get("Task Duration(us)") or r.get("Task Duration") or "?"
    print(f"  {name:<40} Task Duration = {dur} us")
PYEOF
fi
EOS
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
CMD=${1:-}
shift || true
# 解析 --debug/--release 与位置参数 b bs
while [ $# -gt 0 ]; do
  case "$1" in
    --debug)   MODE=debug ;;
    --release) MODE=release ;;
    *) if [ -z "${POS_B:-}" ]; then POS_B=$1; else POS_BS=$1; fi ;;
  esac
  shift
done
B=${POS_B:-$B}
BS=${POS_BS:-$BS}

case "$CMD" in
  build) do_build ;;
  board) do_board ;;
  sim)   do_sim ;;
  perf)  do_perf ;;
  all)
    do_build
    do_board || echo "(board skipped/failed - no NPU?)"
    do_sim
    do_perf || echo "(perf skipped/failed - no NPU?)"
    ;;
  *) sed -n '2,25p' "$0"; exit 1 ;;
esac
