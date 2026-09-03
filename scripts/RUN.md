# hc_pre / mhc_pre_sinkhorn 一键运行脚本

两个算子各有一个统一入口 `run.sh`，覆盖：编译安装（debug/release 可选）、上板精度、仿真、上板性能。

## 快速上手（A3 机器）

```bash
# hc_pre
bash scripts/hc_pre/run.sh build --debug        # 或 --release（默认）
bash scripts/hc_pre/run.sh board 1 512           # 上板精度 (b=1, bs=512)
bash scripts/hc_pre/run.sh sim 1 8               # 仿真（小 shape 快速验证）
bash scripts/hc_pre/run.sh perf 1 512            # 上板性能 (Task Duration + op_summary csv 路径)
bash scripts/hc_pre/run.sh all --release 1 512   # 全流程

# mhc_pre_sinkhorn
bash scripts/mhc_pre_sinkhorn/run.sh build --debug
bash scripts/mhc_pre_sinkhorn/run.sh board 1 512
bash scripts/mhc_pre_sinkhorn/run.sh sim 1 8
bash scripts/mhc_pre_sinkhorn/run.sh perf 1 512
bash scripts/mhc_pre_sinkhorn/run.sh all --release 1 512
```

## 参数说明

| 项 | 说明 |
|---|---|
| `b` / `bs` | 位置参数 2、3；等效环境变量 `HC_B/HC_BS`（hc_pre）与 `MHCS_B/MHCS_BS`（mhc_pre_sinkhorn）。x shape = (b, bs, 4, 4096)，算子内部折叠 b*bs |
| `--debug` / `--release` | kernel 编译模式（build 子命令的 flag，默认 release）。debug 产 DWARF（仿真可定位源码行），release 性能最优 |
| `CONTAINER` | 默认 `cann_container`（WSL docker）；裸机 A3 部署设 `CONTAINER=none`（脚本仓需位于 `~/HcPre`，可通过 `HCPRE_HOME` 覆盖） |
| seed | 固定 1024，golden/上板/仿真三方同数据 |

## debug/release 实现机制

`--build-type=Debug` 传给 ops 仓 `build.sh`，由 op_host CMakeLists 中
`if (CMAKE_BUILD_TYPE STREQUAL "Debug") → --op_debug_config=ccec_g` 条件块控制。
build 完自动校验 .o 的 debug 段数（debug 模式 15 段，失败即报错退出）。

## 上板性能输出示例

```
=== op_summary csv ===
-rw-r--r-- 1 root root 12345 ... /root/HcPre/perf_out/PROF_xxx/device0/op_summary_*.csv
=== Task Duration (us) ===
  HcPre_xxx_0_mix_aic        Task Duration = 123.4 us
```

## 注意事项

- **board/perf 需要真实 NPU**；容器内跑会报 `aclInit ret=500000`，属预期
- **仿真建议用小 shape**（bs≤48）；debug kernel + 大 shape 时 msprof 后处理会静默中断（CSV/visualize 不产出，trace.json 仍可用）
- mhc_pre_sinkhorn 的 example 内部把 4D (b,bs,n,d) 折叠为 (1, b*bs, n, d)——kernel 4D b>1 直通路径输出有误（已实测 b=2 全错），数学上折叠等价，规避之
- hc_pre 的 b*bs 折叠发生在 torch binding 层（`check_hc_pre_shape_and_dtype` 之后），b>1 原生支持
