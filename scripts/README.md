# HcPre 容器验证工作流

本地 Windows 开发（`D:\proj\HcPre`）→ WSL docker `cann_container`（openEuler 24.03, CANN 9.1.0, A3 ops）内编译/仿真。

## 一次性环境准备

```powershell
# 1. 安装 CANN（toolkit + A3 ops，从 C:\Users\wang\Downloads 读取 .run 包）
wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/install_cann.sh

# 2. 安装 python 依赖（torch 2.10.0+cpu / torch-npu / pybind11 / tbe 依赖等）
wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/build_hcpre.sh deps
```

## 日常开发循环

```powershell
# 编译 + 安装 hc_pre 算子包 + 编译 pybind（自动先同步本地代码到容器）
wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/build_hcpre.sh build

# A3 (Ascend910_9382) 仿真：运行 + 精度对比 + per-core pipe 统计
wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/build_hcpre.sh sim

# 一条龙
wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/build_hcpre.sh all
```

仿真用例（`hcpre_sim_app.py`，与算子定义一致）：
- 输入 `x(1,2,4,4096) BF16`、`hc_fn(24,16384) FP32`、`hc_scale(3,) FP32`、`hc_base(24,) FP32`（x 必须为 BF16，算子 def 只接受 DT_BF16）
- 输出 `y(1,2,4096) BF16`、`post(1,2,4) FP32`、`comb_frag(1,2,4,4) FP32`
- 与 CPU golden（HF32 仿真）比对，`HC_PRE_COMPARE=0` 可关闭

## 关键产物路径（容器内）

| 产物 | 路径 |
|---|---|
| pybind 模块 | `/root/HcPre/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-311-x86_64-linux-gnu.so` |
| 自定义算子包 | `/root/HcPre/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer` |
| 构建日志（实时监控） | `/root/HcPre/logs/{op_build,pybind_build}.log` |
| 仿真结果 | `/root/HcPre/sim_out/OPPROF_*/`（instr_exe.csv / code_exe.csv per core） |

## 脚本说明

| 脚本 | 用途 |
|---|---|
| `install_cann.sh` | CANN 安装（toolkit / ops 包，支持 `--toolkit-only` / `--ops-only`） |
| `build_hcpre.sh` | 主入口：deps / build / sim / all |
| `sync_repo.sh` | tar 管道同步本地仓库到容器（排除构建产物，CRLF→LF 归一化） |
| `container_env.sh` | 容器内环境预置（清掉镜像自带 CANN 9.0 ENV 污染，固定 9.1.0） |
| `container_monitor_run.sh` | 构建进程监控：后台运行 + 轮询日志，致命错误立即杀进程组 |
| `hcpre_sim_app.py` | msprof 仿真应用（构造用例 → 调 `torch.ops._C_ascend.npu_hc_pre_v2` → CPU 比对） |
| `parse_sim_csv.py` | 汇总仿真 CSV：per-core pipe 占比 + top 指令 + 热源码行 |
| `kill_stale.sh` | 清理容器内残留构建进程 |

## 常用 shape 的精度基准库（golden_refs）

三个常用 shape 的仿真结果与 CPU golden 已固化为二进制归档（`golden_refs/`，本地与容器 `/root/HcPre/golden_refs/` 同步）：

| 文件 | 内容 |
|---|---|
| `sim_bs{2,240,512}.pt` | A3 仿真输出（y/post/comb_frag，固定 seed=1024） |
| `golden_bs{2,240,512}.pt` | CPU golden 输入+输出全量（可作后续任意比对的基准） |

当前精度结论（仿真 vs golden，混合容差判定全部 PASS）：

| bs | y max_diff (BF16 量化级) | y 通过率 | post/comb_frag |
|---|---|---|---|
| 2 | 3.13e-02 | 99.39% | 100% / 100% |
| 240 | 3.13e-02 | 99.96% | 100% / 100% |
| 512 | 3.13e-02 | 99.31% | 100% / 100% |

一键重跑/比对/归档（shape 可任意指定）：

```powershell
wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/run_common_shapes.sh 2 240 512
```

单 shape 手动流程：

```powershell
# 仿真 + dump（shape 由 HC_PRE_SIZE 控制，seed 固定 1024）
wsl -d Ubuntu-22.04 -- bash -c 'HC_PRE_SIZE=64 HC_PRE_COMPARE=0 HC_PRE_DUMP=/root/HcPre/golden_refs/sim_bs64.pt HC_SIM_TIMEOUT_MIN=60 bash /mnt/d/proj/HcPre/scripts/build_hcpre.sh sim'
# golden 离线比对
wsl -d Ubuntu-22.04 -- bash -c 'docker exec -e HC_GOLDEN_SIZE=64 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e PATH=/usr/local/python3.11.15/bin:/usr/bin:/bin cann_container python3 /root/HcPre/scripts/run_golden.py --result /root/HcPre/golden_refs/sim_bs64.pt'
```

## 已知注意事项

- 容器镜像自带 CANN 9.0.0-beta.1 的 ENV（ASCEND_HOME_PATH/PYTHONPATH 等），所有命令必须先 `source container_env.sh` 清理，否则 torch_npu/opc 工具链加载错库。
- pybind 的 `SOC_VERSION` 必须是具体子型号（`ascend910_9382`），而 ops 仓 `build.sh --soc` 用组名（`ascend910_93`）。
- `--kernel-name=HcPre` 用 CamelCase 子串匹配（实际 kernel 名形如 `HcPre_<hash>_<key>_mix_aic`）。
- 仿真默认 timeout 30 分钟（`HC_SIM_TIMEOUT_MIN` 可调）。

## Kernel debug 编译（code_exe 源码行定位）

默认 kernel 二进制不带 debug line info，仿真只有 per-pipe 统计（instr_exe.csv），code_exe.csv 为空。开启方式：

1. `csrc/cmake/scripts/util/ascendc_gen_options.py` 已支持 `--op_debug_config=` 选项透传（本地仓库改动，同步脚本自动带入容器）。
2. `csrc/moe/hc_pre/op_host/CMakeLists.txt` 的 `add_ops_compile_options` OPTIONS 中含 `--op_debug_config=ccec_g`（ccec 编译带 `-g`，与 `--op_relocatable_kernel_binary=true` 走不同通道可共存）。
3. `build_hcpre.sh build` 每次强制清理 `csrc/build/{binary,autogen}` 重编 kernel（ninja 的 `.done` 戳不跟踪 ini 变化）。

验证方式（`scripts/check_debug_info.sh`）：
- `custom_opc_options.ini` 应含 `HcPre@@--op_debug_config=--op_relocatable_kernel_binary=true;ccec_g`
- gen 脚本 opc 命令带 `--op_debug_config=ccec_g`
- `build/binary/ascend910_93/bin/hc_pre/HcPre_*_relocatable.o` 用 bisheng `llvm-objdump --section-headers` 应看到 `.debug_info/.debug_line/.debug_line_str` 段（约 33KB → 1MB）

开启后仿真多打印 `Parse N addr2line relations`，code_exe.csv 直指源文件行（如 `hc_pre.cpp:84`、`hc_pre_m_k_split_core.h:477`）；`HcPre_*_kernel.cpp` 行是 opc 生成的中间产物，需按变量名回溯到 op_kernel 源码。
