# hc_pre bs=512 仿真流水优化实验报告

> 实验环境：本地 WSL `cann_container`（CANN 9.1.0，msprof op simulator，Ascend910_9382 仿真模型 ~1.85GHz）
> 验证流程：每分支 `run.sh build` + `run.sh sim 1 512`（内嵌 golden 精度对比）+ visualize_data.bin 流水指标提取
> 基线：master @ `0ef93962`（tag `sim-baseline`）

## 结果总表

| 优化点 | 分支 | 最终状态 | 精度 | WALL(µs) | Δ vs 基线 98.697 | 结论 |
|---|---|---|---|---|---|---|
| 基线 | master | — | PASS (y 99.31%/post 100%/comb 100%) | 98.697 | — | — |
| 1. AIC 初始 flag 前移 | `simopt1-early-flag` | **保留** | PASS（逐位一致） | **98.469** | **−0.23 (−0.23%)** | ✅ 采纳（小而稳） |
| 2. MTE2 预取/发射优化 | `simopt2-mte2-prefetch` | 已还原 | 4 个变体见下 | 102.07 / 103.93 | +3.4 ~ +5.2 | ❌ 单独不可行 |
| 3. AIC 提前启动 | 无代码（纯分析） | — | — | — | — | ❌ 数据依赖锁死，证明见下 |
| 4. AIV 错峰 | `simopt4-aiv-stagger` | 已还原 | PASS（逐位一致） | 98.8 / ~115 | 0 ~ +16 | ❌ 无收益 |

## 优化点 1：AIC 初始 flag 前移（唯一采纳项）

**改动**（`hc_pre_m_k_split_core.h`）：把 Process 顶部的两条 `CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_TO_AIV_FLAG)` 移到 `Init()` 最前面（cubeCompute_.Init 之前）。flag 是纯计数信号，与 AIC 的 L1/L0 InitBuffer 无依赖。

**验证**：
- 基线 trace 证实 AIV 空等 0.47µs（t=2.94 处 `WAIT_FLAG_DEVI`）等 AIC 走完 Init 前奏
- 优化后早期等待 0.47→0.29µs（残余为 FIX pipe 启动延迟），首个 MTE2 CopyIn 3.463→3.415µs
- 精度 pass_rate 逐位一致；WALL −0.23µs
- 收益上限即初始等待本身（~0.5µs），属"白捡"，无风险

## 优化点 2：MTE2 喂数优化——4 个变体全部失败（详细过程）

靶子：基线中 MTE2 相邻搬运间 102~510ns 空隙（V 队列 WAIT_FLAG 卡住 → in-order SCALAR 被背压 → 下一条 CopyIn 发不出去）。

| 变体 | 做法 | 结果 | 根因 |
|---|---|---|---|
| A | EnQue(i+1) 提到 DeQue(i) 之前 | **确定性死锁**（仿真 5min 超时被杀） | SCALAR 永久自旋于 `TQueBind::AllocBuffer` 状态扫描环（kernel_tquebind_impl.h:503-510，trace 实锤 pc 映射）；队列内 2 个在飞块的交织触发 TQue 实现边界行为 |
| B | DeQue(i) 后、Cast(i) 发射前预取 i+1 | 精度 PASS，**WALL +3.1~3.4µs**（两次运行 102.07/101.83，可复现；MTE3 max 5.2→9~12） | 喂数变快移除了 V 队列背压对 CopyOut 的"天然限流"，同步 MTE3 herd 放大 |
| C | 单条多 repeat VCONV 压缩 Cast 发射 | **精度 FAIL**（y 0%） | VCONV repeat 模式每 repeat 的 mask 宽度覆盖不了 1024 元素，结果错乱 |
| D | 预取仅在 cvLoop0（无 AIC 争用阶段） | 精度 PASS，**双峰**：103.93 / 98.49（两次运行） | 首次测得 +5.2µs 未复现——cvLoop0 尾部 herd 放大是否触发取决于跨核事件次序（见下"非确定性"） |

> **仿真器非确定性（重要）**：msprof 并行仿真（ModelParsim 20 线程）的跨核事件到达次序不可复现，争用类指标呈双峰/波动：同一 commit 的变体 D 两次运行 WALL 相差 5.4µs（MTE3 max 12.0 vs 5.7）。单次运行的争用类结论有 ±3~5µs 不确定度；变体 B 的回退经复跑确认稳定，变体 D 属"从未好于基线"的双峰。**后续任何针对 GM 争用的优化验证必须至少跑 2~3 次取分布。**

**结论**：MTE2 断流是真的（合计 ~2-4µs 潜力），但 AIV 喂数速度处在一个耦合的争用平衡点上——任何单侧提速都会通过 GM 带宽优先级（AIC 读 > AIV MTE3 写）放大拥塞，净效果为负。提交历史 `d997e7d1..23f389ab` 完整保留 4 个实验。

## 优化点 3：让 AIC 更早启动——结构性不可行（证明）

AIC 消费 xCast workspace 的方式（`CopyInA1` Nd2Nz，ComputeDecode 内）：每个 L1A 装载块 = **全部 256 行 × K 的 128 列片**。即 AIC 的第一个装载就需要所有 9 片 CopyOut（每片 15/8 行 × 全 K）的前 512B——AIV 侧每一片都贡献数据，**任何"部分就绪即可启动"的细粒度 flag 都无济于事**：

1. 按 CO 拆 flag：每个 L1A 块横跨所有 CO → 必须等全部 CO
2. 按 AIV 半边拆 L1A：两半 AIV 本就锁步完成（同时起步+相同节拍），拆了也白拆
3. 按 K 优先重排 workspace（先写 K[0:128) 全行）：AIV 的 CopyIn 变成 512B 粒度的跨步读，MTE2 突发效率崩溃
4. AIC 的 phi 预载（CopyInB1）本就在等待 flag 之前完成，已是最优重叠

AIC 启动时刻 = 最慢 CO 落地时刻，**纯数据依赖锁死**。唯一杠杆是让 AIV 阶段 1 更快——即优化点 2，已证明单独不可行。

## 优化点 4：AIV 错峰——无收益

| 档位 | 精度 | WALL | 判读 |
|---|---|---|---|
| %8（~1.4µs 展开） | PASS | ~115µs（+16） | 错峰本身扰动耦合流水，远超 herd 收益 |
| %2（1 个 dummy） | PASS | ~98.8µs（持平） | dummy MTE2 开销仅 0.14µs——证明 %8 的回退不是搬运开销而是"错峰"本身 |

**结论**：cvLoop 边界的 herd（MTE3 max 5µs 的拥塞）**不是 wall time 的约束项**——wall 由 Part2（~70µs，最忙核 VECTOR 82% busy、其中 BAR/SyncAll 停顿占大头条目：627 次 BAR 37.9 万周期）和 AIV→AIC→AIV 串行依赖链主导。实验历史 `87a1db7a..b716943a`。

## 附：真正值得投入的下一个方向

本次 4 个优化点全部针对 Part1 喂数流水（仅占 wall ~30%），而实测瓶颈图景是：
1. **Part2 的 VECTOR/BAR 停顿**（~70µs，82% busy 但大量 BAR 等待）——Sinkhorn 循环内 19 轮 × 6 步向量操作每步 `PipeBarrier<PIPE_V>()` + `ReduceSumARAPerf` 的 DataCopy+Add 结构，是最大的可优化池
2. Part1→Part2 的 `SyncAll`（8.16µs 空转）与 Part1 末尾两条 `CrossCoreWaitFlag` 的串行链
3. 若上板（非仿真）：MTE3 herd 的真实带宽行为需重新评测（仿真器的读写优先级模型可能与真机不同）

## 工件清单

- 分支：`simopt1-early-flag`（可合入）、`simopt2-mte2-prefetch`（实验记录，tip=原始代码）、`simopt4-aiv-stagger`（实验记录，tip=原始代码）
- **本地仿真产物（2026-09-08 复跑，各 ~13GB，含 README.txt 指标说明）**：
  - `sim_out/opt2b_prefetch/` — 变体 B（commit bcf071c8），WALL 101.83，回退可复现
  - `sim_out/opt2d_prefetch_cvloop0/` — 变体 D（commit 23f389ab），WALL 98.49（原测 103.93 未复现，双峰）
  - 各含 visualize_data.bin（自研脚本解析）+ trace.json（Perfetto/chrome://tracing 可直接打开）+ 72 核 csv + dump
- 辅助脚本：`scripts/simopt/`（metrics/hang_diag/hang_pc/hang_queue，在 simopt1-early-flag 分支 git 历史中；本地直跑版 `%TEMP%\opencode\metrics_local.py`）
- 运行日志：`simopt_logs/`（各分支 build/sim 全量日志，含 opt2b/opt2d 复跑日志）
- 容器内 master 基线已重建恢复
