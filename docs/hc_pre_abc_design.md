# hc_pre A+B+C 组合优化设计（bs=512 仿真）

> 基线：opt8 (C 方案) `simopt8-comb-batching` @ 28b48df6，WALL 3 runs 80.4/82.2/88.4µs（median 82.2）
> 对照：master 98.6µs；阶段拆分（opt8 dump, core0）：stage-1 ≈31.3 → y段 15~23 → comb-batched ≈23.7
> 日期：2026-09-09

## 0. 关键代码事实（已核实）

| 事实 | 出处 |
|---|---|
| stage-1 = mixes mm（A1=x fp32 来自 xCast ws 乒乓, B1=hcFn Nd2Nz, HF32）+ Gram(x·xᵀ diag, 非 HF32) | `hc_pre_cube_compute.h` ComputeDecode/MmadA2/MmadAB |
| AIV 在 stage-1 逐 chunk 把 x bf16 cast 成 fp32 写 xCast ws（32MB 总量，乒乓区域仅 5.5MB） | Part1::Process AIV 分支 |
| mmOut 布局：每 K 分片一片，行距 128 float（512B 对齐），pre=列 0..3, post=列 4..7, comb=列 8..23 | Fixp CopyOut nz2nd, dstStride=128 |
| Fixpipe 可 nz2nd 直写 ND；nSize=24 先例（Mmad n=24 合法）；fp32→bf16 直转未验证 | CopyOut, QuantMode_t |
| 公开转置 API：`Transpose` 仅 b16；**`TransDataTo5HD` 支持 float**（dav_l300 `TRANSDATA_LIST_IMPL(float,32)` + `scatter_vnchwconv_b8`，16 源/目指针列表形式） | kernel_operator_vec_transpose_intf_impl.h |
| workspace 默认 208MB（192MB 余量），xFull(32MB)+xT(32MB)+mmOut2(2.75MB)+yFp32(8MB) 均可容纳 | hc_pre_tiling.cpp GetWorkspaceSize |
| tiling（bs=512）：mL1Size=256, splitKSize=1536, cubeBlockDimK=11, cvLoopKSize=1024, stage1MFactor=16, rowOfFormerBlock=11, d=4096, hcMult=4 | hc_pre_tiling.cpp |

## 1. 三方案时序推演（核心结论：A 与 B 在关键路径上互斥）

```
C-only (opt8 实测):  stage1 31.3 → y段 15~23 → comb 23.7 → ~80.4-82.2

A+C (comb 先行):     pass1(Gram+comb列) ~26.5 → [bar] → AIV comb 26.5→50.2
                     AIC pass2(pre/post列, A1 从 xFull 重扫) 26.5→~37 (并行)
                     → AIV y/post 50.2→~66-74
                     ≈ 66-74µs  (−8~16 vs C)

B+C (y 卸载):        stage1 31.3 (AIV 顺路产 xT) → AIV pre/post+A组装 34.3 → [bar2]
                     AIV comb 34.3→58.3 ‖ AIC y-mm 35→~48 (并行, 吃 xT)
                     → [bar3] → AIV y-cast → ≈ 59-60µs  (−22 vs C)

A+B+C:               pass1 26.5 → [bar] → AIC pass2 →37 ; AIV 等 pre/post →39.3
                     → AIV comb →63 ‖ AIC y-mm 39.3→52 → y-cast → ≈ 64-65µs
                     （A 使 pre/post 列延后 ~10µs，反噬 B 的 y-mm 门控）
```

**结论**：B+C 最优；A 在全栈中为负贡献（~5µs）。按用户要求全部实现并实测对比三配置。

## 2. 实施计划（分支 simopt9-abc-combined，自 opt8 tip）

### Commit 1：方案 A（两遍 K 扫描, comb 先行）
- xCast 乒乓 → **xFull**（`(t, k)` 全量 fp32, 32MB）：AIV CopyOut 目标偏移改 `(mOffset)×k + kGmBase`；MTE3 总量不变
- pass1：B1=`fnGm + 8×kGmSize`（comb 16 行），nValue=16；Fixpipe AB → **mmOut2 独立区域**（行距 128, 列 0..15；避免与 pass2 列重叠/对齐问题）；Gram 照旧
- pass2（AIC-only，AIV 不参与、不 WaitFlag(8)）：同 (m,k) 网格，A1=Nd2Nz from xFull（srcDValue=kGmSize），B1=`fnGm`（pre/post 8 行），nValue=8 → 原 mmOut 区（列 0..7）
- 同步：pass1 后 SyncAll#1（全员）→ AIV 进 Part2 comb-first（读 mmOut2, 列偏移 hcMult*2→0）→ comb 完 SyncAll#2（与 AIC pass2 完成汇合）→ loop1（y/post）
- rsqrt 暂存段（squareSum 归约+rsqrtAllLocal）移到 comb 之前
- Part2 pre/post 读偏移不变（pass2 写原位）

### Commit 2：方案 B（ProcessY 卸载为 AIC 块稀疏 Mmad）
- **xT**：AIV stage-1 每 (16t × 128d) 子块 cast 后 `TransDataTo5HD`（fp32, b8 scatter）→ (128d, 16t) 8KB → CopyOut 至 `xT[h][d][t]`（(d, t) 主序, 行距 512 float, 32MB）
- **A-stage**：AIV pre/post 段每 token 写 4 行（h=0..3）×16 float 对角行（行内仅 [t_local] 处 = pre[t,h]，其余 0）→ `A-stage[(g×4+h)×16 + t_local]` 行 × 16（k=16 对角阵）；行所有权唯一（无跨核竞争）
- **AIC y-mm**（Part2 AIC 分支, SyncAll#2 后）：每 (group, h)：A1=diag(pre_h) 16×16，B1=Nd2Nz from `xT[h][d-tile][16g..+16]`（n=d-tile, k=16, srcDValue=512）→ Mmad(16, n, 16) 按 h 累加（cmatrixInitVal 首 h）→ Fixpipe nz2nd → yFp32Ws（(t, d) fp32, 8MB）；d 按 1024 列分块（L0C 64KB）
- **AIV 重排**：pre/post 段 + A-stage 写 → SyncAll#2 → comb → SyncAll#3 → y-cast 尾（CopyIn yFp32 行 → Cast → CopyOut yGm bf16）
- loop1 的 x 读/ProcessY/y 输出全部删除；xQue/xCast/yCast/yQue 释放
- 精度注意：y-mm 不开 HF32（对齐 AIV fp32 乘加序：k 序 = h 序 0..3 ✓ 逐位等价预期）

### Commit 3：A-off 变体（B+C 对照）
- 单遍全 24 列（等价回退 pass 拆分），保留 xFull/xT/B/C → 实测 B+C

## 3. 风险与对策
| 风险 | 对策 |
|---|---|
| vnchwconv(TransDataTo5HD fp32) 参数语义（b8 16 指针） | 从 dav_l300 impl + rmsnorm 用例推导；精度门禁立即暴露；失败则 B 降级为 A+C |
| xT 转置增加 stage-1 AIV vector 负载（~32MB vnchwconv + 32MB 额外 MTE3） | 转置置于 flag-8 之后（不阻塞 AIC）；首跑对比 stage-1 尾时刻（基线 31.3） |
| Fixpipe nSize/对齐（mmOut2 新区域、y 512 行距） | 区域基址 512B 对齐；nSize=16/8 有 n=24 先例 |
| Mmad n 上限（y-mm 大 n 分块） | n=16/24 有先例；先 n≤64，若 Mmad 次数过多再升 |
| SyncAll 次序跨核一致（3 个 SyncAll） | 每核调用序严格 #1/#2/#3；sim 卡死即查（hang_*.py 工具在） |
| y 输出 fp32→bf16（Fixpipe 直转未验证） | 首版走 yFp32Ws + AIV cast 尾（+~1µs）；验证后可优化 |

## 4. 验收
- 精度：y/post/comb_frag 三项 PASS（阈值同基线）
- 性能：3-run WALL；对比 C-alone (82.2) / A+C / A+B+C / B+C 四配置
- 产物：sim_out/opt9_* + README；报告更新 docs/hc_pre_simopt_report.md
