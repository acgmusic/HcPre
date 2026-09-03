#!/bin/bash
# ============================================================================
# mhc_pre_sinkhorn 一键运行脚本（编译/上板精度/仿真/上板性能）
#
# 用法:
#   bash run.sh build  [--debug|--release]        # 编译+安装算子包+编译 example(默认 release)
#   bash run.sh board  [b] [bs]                   # 上板跑精度用例并对比 golden (aclnn example)
#   bash run.sh sim    [b] [bs]                   # msprof 仿真 (example 跑完自动比对 golden)
#   bash run.sh perf   [b] [bs]                   # 上板 msprof 性能, 打印 Task Duration + op_summary csv 路径
#   bash run.sh all    [--debug|--release] [b] [bs]
#
# 环境变量:
#   MHCS_B / MHCS_BS    等效位置参数 (默认 b=1, bs=2)
#   MHCS_MODE           debug|release
#   MHCS_SIM_TIMEOUT_MIN 仿真超时分钟 (默认 30)
#   CONTAINER           容器名 (默认 cann_container; 裸机 A3 设 CONTAINER=none)
#
# 备注:
#   - 随机种子固定 1024; golden dump 的输入 bin 与上板/仿真共用
#   - x shape = (b, bs, 4, 4096), 算子 api 校验要求 4D (A3 上 3D 被拒)
#   - board/perf 需要真实 NPU; example 直接调 aclnnMhcPreSinkhorn (纯 C++ 无 torch 依赖)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
C=${CONTAINER:-cann_container}
WS=/root/HcPre
REPO=$WS/ops-transformer
ENVSH=$WS/scripts/container_env.sh
INSTALL_DIR=$WS/mhc_pre_install
VENDOR_DIR=$INSTALL_DIR/vendors/custom_transformer
BUILD_SOC=${MHCS_BUILD_SOC:-ascend910_93}
SIM_SOC=${MHCS_SIM_SOC:-Ascend910_9382}
MODE=${MHCS_MODE:-release}
B=${MHCS_B:-1}
BS=${MHCS_BS:-2}
EXE=$REPO/build/test_aclnn_mhc_pre_sinkhorn_hcshape
IN_BIN=$WS/mhcs_input_run.bin
OUT_BIN=$WS/mhcs_output_run.bin

step() { echo -e "\n===== [mhcs.run] $* ====="; }
die()  { echo "ERROR: $*" >&2; exit 1; }

if [ "$C" = "none" ]; then
  run()  { bash -c "$*"; }
  runi() { bash -s; }
  WS=${HCPRE_HOME:-$HOME/HcPre}
  REPO=$WS/ops-transformer
  ENVSH=$WS/scripts/container_env.sh
  INSTALL_DIR=$WS/mhc_pre_install
  VENDOR_DIR=$INSTALL_DIR/vendors/custom_transformer
  EXE=$REPO/build/test_aclnn_mhc_pre_sinkhorn_hcshape
else
  docker start "$C" >/dev/null 2>&1 || true
  run()  { docker exec "$C" bash -c "$*"; }
  runi() { docker exec -i "$C" bash -s; }
fi

TOTAL_BS=$(( B * BS ))   # recomputed in main after arg parsing; kept for early defaults

# ---------------------------------------------------------------------------
# 生成输入 bin + golden 参考 (4 元头: b, bs, n, d)
# ---------------------------------------------------------------------------
gen_inputs() {
  runi <<EOS
set -e
export PATH=/usr/local/python3.11.15/bin:\$PATH 2>/dev/null || true
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
MHCS_GOLDEN_BATCH=$B MHCS_GOLDEN_SIZE=$BS python3 $WS/scripts/mhc_pre_sinkhorn/run_golden.py \\
  --dump-input $IN_BIN --save $WS/golden_refs/mhcs_golden_b${B}_bs${BS}.pt 2>&1 | grep -E "golden.*(dumped|saved|outputs)"
EOS
}

# ---------------------------------------------------------------------------
# 1. build: 编译算子包(debug/release) + 安装 + 编译 example
# ---------------------------------------------------------------------------
do_build() {
  step "build mhc_pre_sinkhorn ($MODE, soc=$BUILD_SOC)"

  if [ "$C" != "none" ]; then
    bash "$SCRIPT_DIR/../sync_repo.sh" >/dev/null
  fi

  # debug/release 由 CMakeLists 的 CMAKE_BUILD_TYPE 条件块控制
  # (op_host/CMakeLists.txt: ccec_g enabled when CMAKE_BUILD_TYPE STREQUAL "Debug")
  BUILD_TYPE_FLAG=""
  [ "$MODE" = "debug" ] && BUILD_TYPE_FLAG="--build-type=Debug"

  # 编译算子包
  runi <<EOS
set -e
source $ENVSH
cd $REPO
rm -rf output build_out build/binary build/autogen
bash $WS/scripts/container_monitor_run.sh $WS/logs/mhcs_op_build.log \\
  bash build.sh --pkg $BUILD_TYPE_FLAG --ops=mhc_pre_sinkhorn --soc=$BUILD_SOC -j\$(nproc) 2>&1 | tail -3 || {
    tail -40 $WS/logs/mhcs_op_build.log; exit 1; }
# verify debug info if requested
if [ "$MODE" = "debug" ]; then
  OBJDUMP=\$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/llvm-objdump
  OBJ=\$(find build/binary/ascend910_93/bin/mhc_pre_sinkhorn -name "*.o" | head -1)
  n=\$(\$OBJDUMP --section-headers "\$OBJ" 2>/dev/null | grep -c "\.debug_")
  sz=\$(stat -c%s "\$OBJ")
  echo "[build] kernel .o: \$(basename \$OBJ) size=\$sz debug_sections=\$n"
  [ "\$n" -gt 0 ] || { echo "ERROR: debug build requested but .o has no debug sections"; exit 1; }
else
  OBJ=\$(find build/binary/ascend910_93/bin/mhc_pre_sinkhorn -name "*.o" | head -1)
  echo "[build] kernel .o: \$(basename \$OBJ) size=\$(stat -c%s "\$OBJ") (release)"
fi
EOS

  # 安装
  runi <<EOS
set -e
source $ENVSH
cd $REPO
mkdir -p $INSTALL_DIR
shopt -s nullglob
installers=(./build/cann-ops-transformer*.run)
shopt -u nullglob
[[ \${#installers[@]} -ge 1 ]] || die "no installer"
chmod +x "\${installers[0]}"
"\${installers[0]}" --install-path=$INSTALL_DIR
ls $VENDOR_DIR/op_impl/ai_core/tbe/kernel/config/ascend910_93/ | grep mhc
EOS

  # 编译 example
  runi <<EOS
set -e
source $ENVSH
cd $REPO
mkdir -p build
CUST_LIB="$VENDOR_DIR/op_api/lib"
CUST_INC="$VENDOR_DIR/op_api/include/aclnnop"
COMMON_INC="$REPO/common/include/external"
g++ mhc/mhc_pre_sinkhorn/examples/test_aclnn_mhc_pre_sinkhorn_hcshape.cpp \\
  -I \$CUST_INC -I \$COMMON_INC -I \$ASCEND_HOME_PATH/include -I \$ASCEND_HOME_PATH/include/aclnn \\
  -L \$CUST_LIB -L \$ASCEND_HOME_PATH/lib64 \\
  -lopapi_math -lcust_opapi -lascendcl -lnnopbase -lc_sec \\
  -o $EXE \\
  -Wl,-rpath=\$CUST_LIB
ls -la $EXE
EOS
  step "build done ($MODE)"
}

# ---------------------------------------------------------------------------
# 2. board 精度: example 直跑 + golden 比对
# ---------------------------------------------------------------------------
do_board() {
  step "board accuracy (b=$B, bs=$BS, total_bs=$TOTAL_BS)"
  gen_inputs
  runi <<EOS
set -e
source $ENVSH
export ASCEND_CUSTOM_OPP_PATH=$VENDOR_DIR
$EXE $IN_BIN $OUT_BIN
EOS
  runi <<EOS
set -e
export PATH=/usr/local/python3.11.15/bin:\$PATH 2>/dev/null || true
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
# out bin header is flat (bs,n,d); golden compare flattens b too
MHCS_GOLDEN_BATCH=1 MHCS_GOLDEN_SIZE=$TOTAL_BS python3 $WS/scripts/mhc_pre_sinkhorn/run_golden.py \\
  --result $OUT_BIN 2>&1 | grep golden
EOS
}

# ---------------------------------------------------------------------------
# 3. msprof 仿真
# ---------------------------------------------------------------------------
do_sim() {
  step "msprof simulator (b=$B, bs=$BS)"
  gen_inputs
  runi <<EOS
set -e
source $ENVSH
V=$VENDOR_DIR
export ASCEND_CUSTOM_OPP_PATH=\$V
export LD_LIBRARY_PATH=\$ASCEND_HOME_PATH/x86_64-linux/simulator/$SIM_SOC/lib:\$V/op_api/lib:\$LD_LIBRARY_PATH
cp -f $IN_BIN $REPO/build/ 2>/dev/null || true
cd $REPO/build
rm -rf $WS/sim_out_mhcs; mkdir -p $WS/sim_out_mhcs
msprof op simulator \\
  --application="$EXE $IN_BIN $OUT_BIN" \\
  --output=$WS/sim_out_mhcs \\
  --kernel-name=MhcPreSinkhorn \\
  --launch-count=1 \\
  --soc-version=$SIM_SOC \\
  --timeout=${MHCS_SIM_TIMEOUT_MIN:-30} || true
echo "--- sim csv summary ---"
python3 $WS/scripts/parse_sim_csv.py $WS/sim_out_mhcs 2>/dev/null | head -30 || echo "(no csv)"
EOS
  # 仿真后比对
  runi <<EOS
set -e
export PATH=/usr/local/python3.11.15/bin:\$PATH 2>/dev/null || true
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
[ -f $OUT_BIN ] && MHCS_GOLDEN_BATCH=1 MHCS_GOLDEN_SIZE=$TOTAL_BS \\
  python3 $WS/scripts/mhc_pre_sinkhorn/run_golden.py --result $OUT_BIN 2>&1 | grep golden || echo "(no output bin to compare)"
EOS
}

# ---------------------------------------------------------------------------
# 4. board 性能: msprof 板端 + Task Duration 统计 + op_summary csv 路径
# ---------------------------------------------------------------------------
do_perf() {
  step "board performance (msprof, b=$B, bs=$BS)"
  gen_inputs
  runi <<EOS
set -e
source $ENVSH
which msprof >/dev/null 2>&1 || { echo "msprof not found"; exit 1; }
export ASCEND_CUSTOM_OPP_PATH=$VENDOR_DIR
rm -rf $WS/perf_out_mhcs; mkdir -p $WS/perf_out_mhcs
msprof --application="$EXE $IN_BIN $OUT_BIN" \\
  --output=$WS/perf_out_mhcs 2>&1 | tail -3 || true
echo
echo "=== op_summary csv ==="
find $WS/perf_out_mhcs -name "op_summary_*.csv" -exec ls -la {} \; 2>/dev/null || echo "(none)"
CSV=\$(find $WS/perf_out_mhcs -name "op_summary_*.csv" | head -1)
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
    print(f"  {name:<50} Task Duration = {dur} us")
PYEOF
fi
EOS
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
CMD=${1:-}
shift || true
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
TOTAL_BS=$(( B * BS ))   # recompute after positional args override B/BS

case "$CMD" in
  build) do_build ;;
  board) do_board ;;
  sim)   do_sim ;;
  perf)  do_perf ;;
  all)
    do_build
    do_board || echo "(board failed - no NPU?)"
    do_sim
    do_perf || echo "(perf failed - no NPU?)"
    ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
