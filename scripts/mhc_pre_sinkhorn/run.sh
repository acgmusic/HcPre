#!/bin/bash
# ============================================================================
# mhc_pre_sinkhorn 一键运行脚本（编译/上板精度/仿真/上板性能）— 环境无关版
#
# 自动适配三种运行环境 (与 hc_pre/run.sh 相同的探测逻辑):
#   A. 裸机/容器内直接运行: 路径取脚本实际位置, 全部本地执行
#   B. WSL + docker: 检测 docker 可用性, 自动同步代码进容器
#   C. 远程 A3 容器 (工程整体拷入): 同 A
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
#   MHCS_SIM_TIMEOUT_MIN 仿真超时分钟数 (默认 30)
#   NPU_ID              上板(board/perf)测试使用的 NPU 卡号 (默认 0; sim 为纯仿真, 固定 0)
#   CONTAINER           强制指定容器名; 设 CONTAINER=none 强制本地; 未设置时自动探测
#
# 备注:
#   - 随机种子固定 1024; golden dump 的输入 bin 与上板/仿真共用
#   - x shape = (b, bs, 4, 4096), 算子 api 校验要求 4D (A3 上 3D 被拒)
#   - board/perf 需要真实 NPU; example 直接调 aclnnMhcPreSinkhorn (纯 C++ 无 torch 依赖)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
WS="$(cd "$SCRIPT_DIR/../.." && pwd)"

detect_mode() {
  if [ -n "${CONTAINER:-}" ]; then
    echo "${CONTAINER}"
    return
  fi
  command -v docker >/dev/null 2>&1 || { echo none; return; }
  local def=${CANN_CONTAINER:-cann_container}
  docker inspect "$def" >/dev/null 2>&1 || { echo none; return; }
  echo "$def"
}
C=$(detect_mode)

REPO=$WS/ops-transformer
BUILD_SOC=${MHCS_BUILD_SOC:-ascend910_93}
SIM_SOC=${MHCS_SIM_SOC:-Ascend910_9382}
MODE=${MHCS_MODE:-release}
B=${MHCS_B:-1}
BS=${MHCS_BS:-2}
NPU_ID=${NPU_ID:-0}

step() { echo -e "\n===== [mhcs.run] $* ====="; }
die()  { echo "ERROR: $*" >&2; exit 1; }

xec() {
  if [ "$C" = "none" ]; then
    bash -s
  else
    docker exec -i "$C" bash -s
  fi
}

# 容器内工程路径 (docker 模式固定 /root/HcPre; 本地模式 = 实际脚本位置)
lws()  { [ "$C" = "none" ] && echo "$WS" || echo /root/HcPre; }

TOTAL_BS=$(( B * BS ))

# ---------------------------------------------------------------------------
# 生成输入 bin + golden 参考 (4 元头: b, bs, n, d)
# ---------------------------------------------------------------------------
gen_inputs() {
  local LWS; LWS=$(lws)
  xec <<EOS
set -e
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
MHCS_GOLDEN_BATCH=$B MHCS_GOLDEN_SIZE=$BS python3 $LWS/scripts/mhc_pre_sinkhorn/run_golden.py \\
  --dump-input $LWS/mhcs_input_run.bin --save $LWS/golden_refs/mhcs_golden_b${B}_bs${BS}.pt 2>&1 | grep -E "golden.*(dumped|saved|outputs)"
EOS
}

# ---------------------------------------------------------------------------
# 1. build
# ---------------------------------------------------------------------------
do_build() {
  step "build mhc_pre_sinkhorn ($MODE, soc=$BUILD_SOC) [mode=$C]"

  if [ "$C" != "none" ]; then
    bash "$SCRIPT_DIR/../sync_repo.sh" >/dev/null
  fi
  [ -d "$REPO" ] || die "repo not found at $REPO"

  local LWS LREPO LENVSH LINSTALL LVENDOR LEXE
  LWS=$(lws); LREPO=$LWS/ops-transformer; LENVSH=$LWS/scripts/container_env.sh
  LINSTALL=$LWS/mhc_pre_install; LVENDOR=$LINSTALL/vendors/custom_transformer
  LEXE=$LREPO/build/test_aclnn_mhc_pre_sinkhorn_hcshape

  BUILD_TYPE_FLAG=""
  [ "$MODE" = "debug" ] && BUILD_TYPE_FLAG="--build-type=Debug"

  # 编译算子包
  xec <<EOS
set -e
source $LENVSH
cd $LREPO
rm -rf output build_out build/binary build/autogen
bash $LWS/scripts/container_monitor_run.sh $LWS/logs/mhcs_op_build.log \\
  bash build.sh --pkg $BUILD_TYPE_FLAG --ops=mhc_pre_sinkhorn --soc=$BUILD_SOC -j\$(nproc) 2>&1 | tail -3 || {
    tail -40 $LWS/logs/mhcs_op_build.log; exit 1; }
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
  xec <<EOS
set -e
source $LENVSH
cd $LREPO
mkdir -p $LINSTALL
shopt -s nullglob
installers=(./build/cann-ops-transformer*.run)
shopt -u nullglob
[[ \${#installers[@]} -ge 1 ]] || { echo "ERROR: no installer"; exit 1; }
chmod +x "\${installers[0]}"
"\${installers[0]}" --install-path=$LINSTALL
ls $LVENDOR/op_impl/ai_core/tbe/kernel/config/ascend910_93/ | grep mhc
EOS

  # 编译 example
  xec <<EOS
set -e
source $LENVSH
cd $LREPO
mkdir -p build
CUST_LIB="$LVENDOR/op_api/lib"
CUST_INC="$LVENDOR/op_api/include/aclnnop"
COMMON_INC="$LREPO/common/include/external"
g++ mhc/mhc_pre_sinkhorn/examples/test_aclnn_mhc_pre_sinkhorn_hcshape.cpp \\
  -I \$CUST_INC -I \$COMMON_INC -I \$ASCEND_HOME_PATH/include -I \$ASCEND_HOME_PATH/include/aclnn \\
  -L \$CUST_LIB -L \$ASCEND_HOME_PATH/lib64 \\
  -lopapi_math -lcust_opapi -lascendcl -lnnopbase -lc_sec \\
  -o $LEXE \\
  -Wl,-rpath=\$CUST_LIB
ls -la $LEXE
EOS
  step "build done ($MODE)"
}

# ---------------------------------------------------------------------------
# 2. board 精度
# ---------------------------------------------------------------------------
do_board() {
  step "board accuracy (b=$B, bs=$BS, total_bs=$TOTAL_BS, npu=$NPU_ID) [mode=$C]"
  local LWS LREPO LENVSH LVENDOR LEXE
  LWS=$(lws); LREPO=$LWS/ops-transformer; LENVSH=$LWS/scripts/container_env.sh
  LVENDOR=$LWS/mhc_pre_install/vendors/custom_transformer
  LEXE=$LREPO/build/test_aclnn_mhc_pre_sinkhorn_hcshape
  gen_inputs
  xec <<EOS
set -e
source $LENVSH
export ASCEND_CUSTOM_OPP_PATH=$LVENDOR
export NPU_ID=$NPU_ID
$LEXE $LWS/mhcs_input_run.bin $LWS/mhcs_output_run.bin
EOS
  xec <<EOS
set -e
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
MHCS_GOLDEN_BATCH=1 MHCS_GOLDEN_SIZE=$TOTAL_BS python3 $LWS/scripts/mhc_pre_sinkhorn/run_golden.py \\
  --result $LWS/mhcs_output_run.bin 2>&1 | grep golden
EOS
}

# ---------------------------------------------------------------------------
# 3. msprof 仿真
# ---------------------------------------------------------------------------
do_sim() {
  step "msprof simulator (b=$B, bs=$BS) [mode=$C]"
  local LWS LREPO LENVSH LVENDOR LEXE
  LWS=$(lws); LREPO=$LWS/ops-transformer; LENVSH=$LWS/scripts/container_env.sh
  LVENDOR=$LWS/mhc_pre_install/vendors/custom_transformer
  LEXE=$LREPO/build/test_aclnn_mhc_pre_sinkhorn_hcshape
  gen_inputs
  xec <<EOS
set -e
source $LENVSH
export ASCEND_CUSTOM_OPP_PATH=$LVENDOR
export NPU_ID=0
export LD_LIBRARY_PATH=\$ASCEND_HOME_PATH/x86_64-linux/simulator/$SIM_SOC/lib:\$LD_LIBRARY_PATH
rm -rf $LWS/sim_out_mhcs; mkdir -p $LWS/sim_out_mhcs
msprof op simulator \\
  --application="$LEXE $LWS/mhcs_input_run.bin $LWS/mhcs_output_run.bin" \\
  --output=$LWS/sim_out_mhcs \\
  --kernel-name=MhcPreSinkhorn \\
  --launch-count=1 \\
  --soc-version=$SIM_SOC \\
  --timeout=${MHCS_SIM_TIMEOUT_MIN:-30} || true
echo "--- sim csv summary ---"
python3 $LWS/scripts/parse_sim_csv.py $LWS/sim_out_mhcs 2>/dev/null | head -30 || echo "(no csv)"
EOS
  xec <<EOS
set -e
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
[ -f $LWS/mhcs_output_run.bin ] && MHCS_GOLDEN_BATCH=1 MHCS_GOLDEN_SIZE=$TOTAL_BS \\
  python3 $LWS/scripts/mhc_pre_sinkhorn/run_golden.py --result $LWS/mhcs_output_run.bin 2>&1 | grep golden || echo "(no output bin to compare)"
EOS
}

# ---------------------------------------------------------------------------
# 4. board 性能
# ---------------------------------------------------------------------------
do_perf() {
  step "board performance (msprof, b=$B, bs=$BS, npu=$NPU_ID) [mode=$C]"
  local LWS LREPO LENVSH LVENDOR LEXE
  LWS=$(lws); LREPO=$LWS/ops-transformer; LENVSH=$LWS/scripts/container_env.sh
  LVENDOR=$LWS/mhc_pre_install/vendors/custom_transformer
  LEXE=$LREPO/build/test_aclnn_mhc_pre_sinkhorn_hcshape
  gen_inputs
  xec <<EOS
set -e
source $LENVSH
which msprof >/dev/null 2>&1 || { echo "msprof not found"; exit 1; }
export ASCEND_CUSTOM_OPP_PATH=$LVENDOR
export NPU_ID=$NPU_ID
rm -rf $LWS/perf_out_mhcs; mkdir -p $LWS/perf_out_mhcs
msprof --application="$LEXE $LWS/mhcs_input_run.bin $LWS/mhcs_output_run.bin" \\
  --output=$LWS/perf_out_mhcs 2>&1 | tail -3 || true
echo
echo "=== op_summary csv ==="
find $LWS/perf_out_mhcs -name "op_summary_*.csv" -exec ls -la {} \; 2>/dev/null || echo "(none)"
CSV=\$(find $LWS/perf_out_mhcs -name "op_summary_*.csv" | head -1)
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
TOTAL_BS=$(( B * BS ))

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
