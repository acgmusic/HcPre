#!/bin/bash
# ============================================================================
# hc_pre 一键运行脚本（编译/上板精度/仿真/上板性能）— 环境无关版
#
# 自动适配三种运行环境:
#   A. 裸机/容器内直接运行 (本脚本就在工程内): 全部本地执行, 路径取脚本实际位置
#   B. WSL + docker (脚本在 Windows 侧, 代码需同步进容器): 自动检测 docker 可用性
#   C. 远程 A3 容器 (工程整体拷入, 如 /data/xxx/HcPre): 同 A
#
# 用法:
#   bash run.sh build  [--debug|--release]        # 编译+安装算子包+pybind(默认 release)
#   bash run.sh board  [b] [bs]                   # 上板跑精度用例并对比 golden
#   bash run.sh verify [N | shapes.csv]           # 上板泛化精度验证: N 个随机 shape
#                                                 #   (默认 1000, b∈[1,4] 均匀 / s 对数均匀
#                                                 #   [1,8192] / d=4096, 固定种子
#                                                 #   HC_PRE_VERIFY_SEED=1024) 或 csv 指定;
#                                                 #   逐 shape 比对 golden, 首个 FAIL 立即停止
#                                                 #   并报告(设 HC_PRE_DUMP=path 可落盘失败现场)
#   bash run.sh sim    [b] [bs]                   # msprof 仿真(精度对比内嵌)
#   bash run.sh perf   [b] [bs]                   # 上板 msprof 性能, 打印 Task Duration + op_summary csv 路径
#   bash run.sh perf   test_shapes.csv [rounds]   # 多 shape 批量: rounds 轮独立 msprof 进程(默认 50,
#                                                 #   可用 HC_PRE_ROUNDS 覆盖), 每轮进程内 HC_PRE_ITERS 次
#                                                 #   (默认 50)遍历全部 shape; 每 shape 输出
#                                                 #   rounds 行 x iters 列时间序列(perf_extreme_eval 格式:
#                                                 #   行=独立进程块, 列=进程内采样), 及全量统计
#   环境变量 NPU_ID 指定非零物理卡号时, perf 会自动用 ASCEND_RT_VISIBLE_DEVICES
#   钉卡(profiling 跟随), app 内逻辑卡号归 0
#   bash run.sh all    [--debug|--release] [b] [bs]
#
# 环境变量:
#   HC_B / HC_BS        等效于位置参数 b / bs (默认 b=1, bs=2)
#   HC_MODE             debug|release (默认 release, 可被 --debug/--release 覆盖)
#   HC_SIM_TIMEOUT_MIN  仿真超时分钟数 (默认 30)
#   NPU_ID              上板(board/perf)测试使用的 NPU 卡号 (默认 0; sim 为纯仿真, 固定 0)
#   HC_PRE_ITERS        perf 上板时算子 launch 次数 (默认 50, 输出 50 次采样统计)
#   HC_PRE_DUMP         设置时 hc_pre board 额外把 NPU 输出 torch.save 到该路径 (对比脚本用)
#   CONTAINER           强制指定容器名; 设 CONTAINER=none 强制本地执行;
#                       未设置时自动探测 (docker 可用且默认容器在运行则用 docker, 否则本地)
#
# 备注:
#   - 所有随机种子固定 1024, golden 与上板/仿真用同一组数据
#   - b*bs 合并为算子内部的 bs 维 (x shape = (b, bs, 4, 4096))
#   - board/perf 需要真实 A3 NPU; 在无 NPU 的环境上会失败, 这是预期行为
# ============================================================================
set -euo pipefail

# ---- 环境探测 ------------------------------------------------------------
# 脚本实际所在目录 (兼容 bash run.sh / sh run.sh / source 后调用)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# 工程根 = scripts/hc_pre 的上两级 (相对路径, 与安装位置无关)
WS="$(cd "$SCRIPT_DIR/../.." && pwd)"

detect_mode() {
  if [ -n "${CONTAINER:-}" ]; then
    echo "${CONTAINER}"
    return
  fi
  # docker 不可用 -> 本地
  command -v docker >/dev/null 2>&1 || { echo none; return; }
  # docker 可用但默认容器不存在 -> 本地 (典型: 远程 A3 容器内装了 docker CLI 但无 daemon)
  local def=${CANN_CONTAINER:-cann_container}
  docker inspect "$def" >/dev/null 2>&1 || { echo none; return; }
  echo "$def"
}
C=$(detect_mode)

# 路径 (本地/远程统一用相对脚本位置的绝对路径; 本地与 docker 模式共享同一 WS 定义)
REPO=$WS/vllm-ascend
ENVSH=$WS/scripts/container_env.sh
VENDOR_DIR=$REPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
BUILD_SOC=${HC_BUILD_SOC:-ascend910_93}
PYBIND_SOC=${HC_PYBIND_SOC:-ascend910_9382}
SIM_SOC=${HC_SIM_SOC:-Ascend910_9382}
MODE=${HC_MODE:-release}
B=${HC_B:-1}
BS=${HC_BS:-2}
NPU_ID=${NPU_ID:-0}
ITERS=${HC_PRE_ITERS:-50}
OPNAME=hc_pre

step() { echo -e "\n===== [hc_pre.run] $* ====="; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ---- 执行封装 ------------------------------------------------------------
if [ "$C" = "none" ]; then
  # 本地模式 (裸机 / 容器内直接运行): heredoc 走 stdin, 路径原样使用
  run()  { bash -c "$*"; }
  runi() { bash -s; }
  # docker 模式下容器内路径可能不同于宿主机路径, 由 sync 决定; 本地模式直接用
  C_WS=$WS
else
  docker start "$C" >/dev/null 2>&1 || true
  run()  { docker exec "$C" bash -c "$*"; }
  runi() { bash -s; }   # 由调用方通过管道喂给 docker exec -i
  # docker 模式: 容器内工程固定在 /root/HcPre (sync_repo.sh 的约定)
  C_WS=/root/HcPre
fi

# runi 的统一入口: 本地直接执行 heredoc; docker 模式喂给容器
xec() {
  if [ "$C" = "none" ]; then
    bash -s
  else
    docker exec -i "$C" bash -s
  fi
}

# ---------------------------------------------------------------------------
# 1. 编译/安装
# ---------------------------------------------------------------------------
do_build() {
  step "build $OPNAME ($MODE, soc=$BUILD_SOC) [mode=$C]"

  # docker 模式: 先把宿主机工程同步进容器
  if [ "$C" != "none" ]; then
    bash "$SCRIPT_DIR/../sync_repo.sh" >/dev/null
  fi
  # 本地模式: 确保脚本/代码就位 (相对路径, 无操作也无害)
  [ -d "$REPO" ] || die "repo not found at $REPO"

  # docker 模式下容器内使用 C_WS 路径; 本地使用 WS. 统一变量:
  LWS=$([ "$C" = "none" ] && echo "$WS" || echo "$C_WS")
  LREPO=$LWS/vllm-ascend
  LENVSH=$LWS/scripts/container_env.sh
  LVENDOR=$LREPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer

  # debug/release 由 CMakeLists 的 CMAKE_BUILD_TYPE 条件块控制
  BUILD_TYPE_FLAG=""
  [ "$MODE" = "debug" ] && BUILD_TYPE_FLAG="--build-type=Debug"

  # 编译算子包
  xec <<EOS
set -e
source $LENVSH
cd $LREPO/csrc
rm -rf output build_out build/binary build/autogen
bash $LWS/scripts/container_monitor_run.sh $LWS/logs/hc_op_build.log \\
  bash build.sh --pkg $BUILD_TYPE_FLAG --ops=$OPNAME --soc=$BUILD_SOC -j\$(nproc) 2>&1 | tail -3 || {
    tail -40 $LWS/logs/hc_op_build.log; exit 1; }
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
  xec <<EOS
set -e
source $LENVSH
cd $LREPO/csrc
mkdir -p $LREPO/vllm_ascend/_cann_ops_custom
shopt -s nullglob
installers=(./build/cann-ops-transformer*.run)
shopt -u nullglob
[[ \${#installers[@]} -eq 1 ]] || { echo "ERROR: expected 1 installer, got \${#installers[@]}"; exit 1; }
chmod +x "\${installers[0]}"
"\${installers[0]}" --install-path=$LREPO/vllm_ascend/_cann_ops_custom
[[ -d $LVENDOR/scripts ]] && chmod u+w $LVENDOR/scripts
echo "installed to $LVENDOR"
EOS

  # pybind 编译
  xec <<EOS
set -e
source $LENVSH
PYBIN=\$(which python3)
TNPU_LOC=\$(python3 -m pip show torch-npu | awk '/^Location:/ {print \$2}')
cmake -S $LREPO -B $LREPO/build-pybind \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DASCEND_HOME_PATH=\$ASCEND_HOME_PATH \\
  -DPYTHON_EXECUTABLE=\$PYBIN \\
  -DPYTHON_INCLUDE_PATH=\$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \\
  -DCMAKE_INSTALL_PREFIX=$LREPO/vllm_ascend \\
  -DCMAKE_PREFIX_PATH=\$(python3 -m pybind11 --cmakedir) \\
  -DSOC_VERSION=$PYBIND_SOC \\
  -DFETCHCONTENT_BASE_DIR=$LREPO/.deps \\
  -DTORCH_NPU_PATH=\$TNPU_LOC/torch_npu > /dev/null
cmake --build $LREPO/build-pybind --target install -j\$(nproc) > /dev/null 2>&1 || \\
  cmake --build $LREPO/build-pybind --target install -j\$(nproc)
ls -la $LREPO/vllm_ascend/vllm_ascend_C*.so
EOS
  step "build done ($MODE)"
}

# ---------------------------------------------------------------------------
# 2. 上板精度
# ---------------------------------------------------------------------------
do_board() {
  step "board accuracy test (b=$B, bs=$BS, npu=$NPU_ID) [mode=$C]"
  LWS=$([ "$C" = "none" ] && echo "$WS" || echo "$C_WS")
  LREPO=$LWS/vllm-ascend
  LENVSH=$LWS/scripts/container_env.sh
  LVENDOR=$LREPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
  xec <<EOS
set -e
source $LENVSH
export ASCEND_CUSTOM_OPP_PATH=$LVENDOR
PYBIND_SO=\$(ls $LREPO/vllm_ascend/vllm_ascend_C*.so | head -1)
export HC_PRE_PYBIND_SO=\$PYBIND_SO
# libvllm_ascend_kernels.so sits next to the pybind so; \$ORIGIN rpath is not
# always honored by torch.ops.load_library (ctypes dlopen), so prepend explicitly.
export LD_LIBRARY_PATH="\$(dirname "\$PYBIND_SO"):\${LD_LIBRARY_PATH:-}"
export HC_PRE_SIZE=\$(( $B * $BS ))
export HC_PRE_BATCH=$B
export HC_PRE_COMPARE=1
export HC_PRE_DUMP=${HC_PRE_DUMP:-}
export NPU_ID=$NPU_ID
python3 $LWS/scripts/hc_pre/hcpre_sim_app.py
EOS
}

# ---------------------------------------------------------------------------
# 2b. 上板泛化精度验证 (多 shape, 固定种子, 首错即停)
# ---------------------------------------------------------------------------
do_verify() {
  VARG=${POS_B:-}
  VCSV=""
  VCNT=""
  if [ -n "$VARG" ]; then
    case "$VARG" in
      *.csv) VCSV="$VARG" ;;
      *)
        VCNT="$VARG"
        case "$VCNT" in
          ''|*[!0-9]*) die "verify: expected a positive count or a *.csv path, got: '$VARG'" ;;
        esac
        [ "$VCNT" -ge 1 ] || die "verify: count must be >= 1, got: $VCNT"
        ;;
    esac
  fi
  [ -z "$VCSV" ] || [ -z "$VCNT" ] || die "verify: give either a count or a csv, not both"

  if [ "$C" != "none" ]; then
    bash "$SCRIPT_DIR/../sync_repo.sh" >/dev/null
  fi
  LWS=$([ "$C" = "none" ] && echo "$WS" || echo "$C_WS")
  LREPO=$LWS/vllm-ascend
  LENVSH=$LWS/scripts/container_env.sh
  LVENDOR=$LREPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer

  if [ -n "$VCSV" ]; then
    # csv 模式: 解析为绝对路径(绝对直接用; 相对依次尝试 当前目录/工程根/脚本目录)
    case "$VCSV" in
      /*) VABS="$VCSV" ;;
      *)
        VABS=""
        for _cand in "$PWD/$VCSV" "$WS/$VCSV" "$SCRIPT_DIR/$VCSV"; do
          [ -f "$_cand" ] && { VABS="$_cand"; break; }
        done
        ;;
    esac
    [ -n "$VABS" ] && [ -f "$VABS" ] || \
      die "verify: shapes csv not found: $VCSV (tried \$PWD, $WS, $SCRIPT_DIR)"
    VREL=${VABS#"$WS"/}
    [ "$VREL" != "$VABS" ] || die "verify: shapes csv must live under $WS (got $VABS)"
    LSHAPES=$LWS/$VREL
    VDESC="csv=$VREL"
  else
    LSHAPES="auto"
    VDESC="auto count=${VCNT:-1000} seed=${HC_PRE_VERIFY_SEED:-1024}"
  fi

  step "board generalization accuracy verify ($VDESC, npu=$NPU_ID) [mode=$C]"
  xec <<EOS
set -e
source $LENVSH
export ASCEND_CUSTOM_OPP_PATH=$LVENDOR
PYBIND_SO=\$(ls $LREPO/vllm_ascend/vllm_ascend_C*.so | head -1)
export HC_PRE_PYBIND_SO=\$PYBIND_SO
export LD_LIBRARY_PATH="\$(dirname "\$PYBIND_SO"):\${LD_LIBRARY_PATH:-}"
export HC_PRE_SHAPES=$LSHAPES
export HC_PRE_VERIFY_COUNT=${VCNT:-1000}
export HC_PRE_VERIFY_SEED=${HC_PRE_VERIFY_SEED:-1024}
export HC_PRE_COMPARE=1
export NPU_ID=$NPU_ID
export HC_PRE_DUMP=${HC_PRE_DUMP:-}
python3 $LWS/scripts/hc_pre/hcpre_sim_app.py
EOS
}

# ---------------------------------------------------------------------------
# 3. msprof 仿真
# ---------------------------------------------------------------------------
do_sim() {
  step "msprof simulator (b=$B, bs=$BS, soc=$SIM_SOC) [mode=$C]"
  LWS=$([ "$C" = "none" ] && echo "$WS" || echo "$C_WS")
  LREPO=$LWS/vllm-ascend
  LENVSH=$LWS/scripts/container_env.sh
  LVENDOR=$LREPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
  xec <<EOS
set -e
source $LENVSH
export LD_LIBRARY_PATH=\$ASCEND_HOME_PATH/x86_64-linux/simulator/$SIM_SOC/lib:\$LD_LIBRARY_PATH
export ASCEND_CUSTOM_OPP_PATH=$LVENDOR
PYBIND_SO=\$(ls $LREPO/vllm_ascend/vllm_ascend_C*.so | head -1)
export HC_PRE_PYBIND_SO=\$PYBIND_SO
export LD_LIBRARY_PATH="\$(dirname "\$PYBIND_SO"):\$LD_LIBRARY_PATH"
export HC_PRE_SIZE=\$(( $B * $BS ))
export HC_PRE_BATCH=$B
export HC_PRE_COMPARE=1
export NPU_ID=0
rm -rf $LWS/sim_out; mkdir -p $LWS/sim_out
msprof op simulator \\
  --application="python3 $LWS/scripts/hc_pre/hcpre_sim_app.py" \\
  --output=$LWS/sim_out \\
  --kernel-name=HcPre \\
  --launch-count=1 \\
  --soc-version=$SIM_SOC \\
  --timeout=${HC_SIM_TIMEOUT_MIN:-30} || true
echo "--- sim csv summary ---"
python3 $LWS/scripts/parse_sim_csv.py $LWS/sim_out 2>/dev/null | head -30 || echo "(no csv)"
EOS
}

# ---------------------------------------------------------------------------
# 4. 上板性能
# ---------------------------------------------------------------------------
do_perf() {
  step "board performance (msprof, b=$B, bs=$BS, npu=$NPU_ID, iters=$ITERS) [mode=$C]"
  LWS=$([ "$C" = "none" ] && echo "$WS" || echo "$C_WS")
  LREPO=$LWS/vllm-ascend
  LENVSH=$LWS/scripts/container_env.sh
  LVENDOR=$LREPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
  xec <<EOS
set -e
source $LENVSH
which msprof >/dev/null 2>&1 || { echo "msprof not found"; exit 1; }
export ASCEND_CUSTOM_OPP_PATH=$LVENDOR
PYBIND_SO=\$(ls $LREPO/vllm_ascend/vllm_ascend_C*.so | head -1)
export HC_PRE_PYBIND_SO=\$PYBIND_SO
export LD_LIBRARY_PATH="\$(dirname "\$PYBIND_SO"):\${LD_LIBRARY_PATH:-}"
export HC_PRE_SIZE=\$(( $B * $BS ))
export HC_PRE_BATCH=$B
export HC_PRE_COMPARE=0
export NPU_ID=$NPU_ID
export HC_PRE_ITERS=$ITERS
# [NPU 钉卡] 非零卡号: msprof profiling 层需要 ASCEND_RT_VISIBLE_DEVICES 物理钉卡,
# 单靠 app 内 set_device 不够; 钉卡后进程可见设备集只有该卡, 卡内逻辑号归 0
if [ "\$NPU_ID" != "0" ] && [ -z "\${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
  export ASCEND_RT_VISIBLE_DEVICES=\$NPU_ID
  echo "[perf] pinned to physical NPU \$NPU_ID (ASCEND_RT_VISIBLE_DEVICES=\$NPU_ID, app 逻辑卡号=0)"
  export NPU_ID=0
fi
rm -rf $LWS/perf_out; mkdir -p $LWS/perf_out
msprof --application="python3 $LWS/scripts/hc_pre/hcpre_sim_app.py" \\
  --output=$LWS/perf_out \\
  --aic-metrics=PipeUtilization 2>&1 | tail -5 || true
echo
echo "=== op_summary csv ==="
find $LWS/perf_out -name "op_summary_*.csv" -exec ls -la {} \; 2>/dev/null || echo "(none found)"
CSV=\$(find $LWS/perf_out -name "op_summary_*.csv" | head -1)
if [ -n "\$CSV" ]; then
  python3 $LWS/scripts/perf_stats.py "\$CSV" HcPre || true
else
  echo "(no op_summary csv found)"
fi
EOS
}

# ---------------------------------------------------------------------------
# 4b. 上板性能 (多 shape 批量: 一次 msprof 跑完 csv 内全部 shape)
# ---------------------------------------------------------------------------
do_perf_multi() {
  SHAPES_CSV_ARG=${POS_B:?}
  # rounds: 独立 msprof 进程数(外层, 状态清空); iters: 每轮进程内遍历全部 shape 的次数
  ROUNDS=${POS_BS:-${HC_PRE_ROUNDS:-50}}
  ITERS_INPROC=${HC_PRE_ITERS:-50}

  # rounds/iters 必须为正整数(防误传导致 seq 静默失败)
  for _v in "$ROUNDS" "$ITERS_INPROC"; do
    case "$_v" in
      ''|*[!0-9]*) die "rounds/iters must be positive integers, got: '$_v'" ;;
    esac
    [ "$_v" -ge 1 ] || die "rounds/iters must be >= 1, got: $_v"
  done

  # 解析为绝对路径: 绝对路径直接用; 相对路径依次尝试 当前目录/工程根/脚本目录
  # (docker 模式需位于工程内以随 sync_repo 进容器)
  case "$SHAPES_CSV_ARG" in
    /*) SHAPES_ABS="$SHAPES_CSV_ARG" ;;
    *)
      SHAPES_ABS=""
      for _cand in "$PWD/$SHAPES_CSV_ARG" "$WS/$SHAPES_CSV_ARG" "$SCRIPT_DIR/$SHAPES_CSV_ARG"; do
        [ -f "$_cand" ] && { SHAPES_ABS="$_cand"; break; }
      done
      ;;
  esac
  [ -n "$SHAPES_ABS" ] && [ -f "$SHAPES_ABS" ] || \
    die "shapes csv not found: $SHAPES_CSV_ARG (tried \$PWD, $WS, $SCRIPT_DIR)"
  SHAPES_REL=${SHAPES_ABS#"$WS"/}
  [ "$SHAPES_REL" != "$SHAPES_ABS" ] || die "shapes csv must live under $WS (got $SHAPES_ABS)"

  if [ "$C" != "none" ]; then
    bash "$SCRIPT_DIR/../sync_repo.sh" >/dev/null
  fi
  LWS=$([ "$C" = "none" ] && echo "$WS" || echo "$C_WS")
  LREPO=$LWS/vllm-ascend
  LENVSH=$LWS/scripts/container_env.sh
  LVENDOR=$LREPO/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
  LSHAPES=$LWS/$SHAPES_REL

  step "board perf multi-shape (csv=$SHAPES_REL, rounds=$ROUNDS x iters=$ITERS_INPROC, npu=$NPU_ID) [mode=$C]"
  xec <<EOS
set -e
source $LENVSH
which msprof >/dev/null 2>&1 || { echo "msprof not found"; exit 1; }
export ASCEND_CUSTOM_OPP_PATH=$LVENDOR
PYBIND_SO=\$(ls $LREPO/vllm_ascend/vllm_ascend_C*.so | head -1)
export HC_PRE_PYBIND_SO=\$PYBIND_SO
export LD_LIBRARY_PATH="\$(dirname "\$PYBIND_SO"):\${LD_LIBRARY_PATH:-}"
export HC_PRE_SHAPES=$LSHAPES
export HC_PRE_COMPARE=0
# 轮次语义: 每一轮 = 一次独立 msprof 进程(清空状态); 进程内 iter 次遍历全部 shape
# (iter 外层, shape 内层), 每 shape 每轮得到 iters 个进程内采样
export HC_PRE_ITERS=$ITERS_INPROC
export NPU_ID=$NPU_ID
# [NPU 钉卡] 非零卡号: msprof profiling 层需要 ASCEND_RT_VISIBLE_DEVICES 物理钉卡,
# 单靠 app 内 set_device 不够; 钉卡后进程可见设备集只有该卡, 卡内逻辑号归 0
if [ "\$NPU_ID" != "0" ] && [ -z "\${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
  export ASCEND_RT_VISIBLE_DEVICES=\$NPU_ID
  echo "[perf-multi] pinned to physical NPU \$NPU_ID (ASCEND_RT_VISIBLE_DEVICES=\$NPU_ID, app 逻辑卡号=0)"
  export NPU_ID=0
fi
rm -rf $LWS/perf_out_multi; mkdir -p $LWS/perf_out_multi
CSVS=""
for R in \$(seq 1 $ROUNDS); do
  printf -- '--- round %s / %s ---\n' "\$R" "$ROUNDS"
  RD=$LWS/perf_out_multi/round_\$(printf '%03d' "\$R")
  rm -rf "\$RD"; mkdir -p "\$RD"
  msprof --application="python3 $LWS/scripts/hc_pre/hcpre_sim_app.py" \\
    --output="\$RD" \\
    --aic-metrics=PipeUtilization > "\$RD/msprof.log" 2>&1 \\
    || { echo "  round \$R msprof FAILED:"; tail -5 "\$RD/msprof.log"; }
  CSV=\$(find "\$RD" -name "op_summary_*.csv" 2>/dev/null | head -1)
  if [ -n "\$CSV" ]; then
    CSVS="\$CSVS \$CSV"
    echo "  ok"
  else
    echo "  no op_summary csv (round skipped)"
  fi
done
echo
if [ -z "\$CSVS" ]; then
  echo "ERROR: no op_summary csv collected from any round"
  exit 1
fi
echo "=== aggregate: \$ROUNDS rounds requested, \$(echo \$CSVS | wc -w) collected ==="
python3 $LWS/scripts/perf_stats_multi.py "$LSHAPES" $LWS/perf_out_multi \$CSVS || true
du -sh $LWS/perf_out_multi 2>/dev/null || true
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

case "$CMD" in
  build) do_build ;;
  board) do_board ;;
  verify) do_verify ;;
  sim)   do_sim ;;
  perf)
    case "${POS_B:-}" in
      *.csv) do_perf_multi ;;
      *)     do_perf ;;
    esac
    ;;
  all)
    do_build
    do_board || echo "(board skipped/failed - no NPU?)"
    do_sim
    do_perf || echo "(perf skipped/failed - no NPU?)"
    ;;
  *) sed -n '2,25p' "$0"; exit 1 ;;
esac
