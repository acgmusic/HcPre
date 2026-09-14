# hc_pre golden 逐行 → kernel 映射与详解

> 分析对象：
> - **golden**：`scripts/hc_pre/run_golden.py` 的 `hc_pre_cpu()`（run_golden.py:56-73，CPU 参考实现，与 `hcpre_sim_app.py` 的 `_hc_pre_cpu` 逐行一致）
> - **kernel**：`vllm-ascend/csrc/moe/hc_pre/op_kernel/` 下的 AscendC 实现
>
> 本文逐行回答"golden 每一行对应 kernel 哪些代码"，并详解这些 kernel 代码。
> 分核逻辑详见同目录《hcpre_shape_and_core_split.md》，本文聚焦算子语义映射。

---

## 1. 文件清单与架构背景

### 1.1 kernel 文件角色表

| 文件 | 角色 |
|---|---|
| `hc_pre.cpp` | kernel 入口：按架构/tilingKey 分派 Part1/Part2 两个阶段类 |
| `hc_pre_m_k_split_core.h` | **arch22 主路径**（A3/910_9382，TILING_KEY=0）：`HcPreMembaseKSplitCorePart1`（Stage1）+ `HcPreMembaseKSplitCorePart2`（Stage2） |
| `hc_pre_cube_compute.h` | arch22 cube 侧计算：`HcCubeCompute<enableSquareSum>`，Gram + 主 matmul 的三级双缓冲流水 |
| `hc_pre_base.h` | arch22 vector 侧算子原语库（ProcessPre/Post/Y、SoftmaxFP32Perf、ReduceSumARAPerf、各类 BrcInline 广播指令封装等） |
| `hc_pre_m_k_split_core_arch35.h` | **arch35 主路径**（`__DAV_C310__`/950，TILING_KEY=1000）：`HcPreNs::HcPreMKSplitCorePart1/Part2` |
| `hc_pre_m_split_core_arch35.h` | arch35 的 M-only 切分变体（TILING_KEY=1001） |
| `hc_pre_base_arch35.h` | arch35 MicroAPI 原语库（VFProcess* 系列，寄存器级实现） |
| `hc_pre_cube_compute_arch35.h` | arch35 cube 侧：L0 ping-pong + MmadBase |
| `hc_pre_tiling.h/.cpp` | host 侧 tiling（分核/UB 容量推导，见分核文档） |

### 1.2 入口编排（hc_pre.cpp）

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);      // :37  1 AIC : 2 AIV 配对启动
...
#if defined(__DAV_C310__)                                // :19  arch35 (950)
    if (TILING_KEY_IS(1000)) { ... Part1; Part2; }       // :50-62 m_k_split 路径
    else if (TILING_KEY_IS(1001)) { ... MSplitCorePart1; } // :63-69 m_split 路径
#else                                                    // arch22 (A3)
    if (TILING_KEY_IS(0)) {                              // :72
        HcPreMembaseKSplitCorePart1 op; op.Init(...); op.Process();   // Stage1
        HcPreMembaseKSplitCorePart2 op2; op2.Init(...); op2.Process(); // Stage2
    }
#endif
```

**两阶段设计**：Stage1 用 AIC（cube）做两笔 matmul、AIV 做喂数与 cast；Stage2 全部落在 AIV 做 Σx² 归约、逐 token 的 pre/post/comb_frag/y 后处理。两阶段通过 **GM workspace** 解耦（arch22 三区：xCast / mmOut / squareSum，m_k_split_core.h:48-60；arch35 两区：mm / rms，m_k_split_core_arch35.h:271-272）。

**arch22 与 arch35 的关键架构差异**（影响下文所有映射）：

| | arch22（A3） | arch35（950） |
|---|---|---|
| Σx² 计算位置 | **AIC**：Gram matmul（MmadA2） | **AIV**：cast 顺带 Mul+ReduceSum |
| x 的 cast 结果去向 | AIV → GM workspace → AIC 从 GM 搬 L1 | AIV → UB → **直接 CopyToL1**（配对核共享 L1，不经 GM） |
| vector API | 传统 AscendC（UB LocalTensor + repeat 参数） | MicroAPI（RegTensor 寄存器 + Mask） |
| cube | 手写 LoadData3D/2D + Mmad 三级双缓冲 | L0 ping-pong + MmadBase |

---

## 2. 逐行映射总表

以 `hc_pre_cpu` 的行号为准（run_golden.py）：

| golden 行 | 语义 | arch22 位置 | arch35 位置 |
|---|---|---|---|
| :57 `x.float()` | bf16→fp32 | Stage1 AIV `CastTwoDim`（m_k_split:151-152）→ xCastWs | Part1 AIV `VFProcessCastAndInvRmsPart1` 内 `LoadInputData+StoreOutputData`（base_arch35:267-270） |
| :58 `flatten(-2)` | 布局重排 | 无独立算子（行寻址 `row*k+kOff`，m_k_split:146-148） | 同左 |
| :59a `x_flat.square()` | Σx² | AIC `MmadA2`（cube:418-437，块对角 Gram）+ `Fixp`（:233-251） | AIV `Mul(x1,x,x)+Add(sum,sum,x1)`（base_arch35:268-269） |
| :59b `.mean(-1)` | ÷K | Stage2 `Muls(coeff=1/k)`（m_k_split:363） | Part1 `Muls(sum,coeff)`（base_arch35:272，coeff 见 m_k_split_arch35:160） |
| :59c `+ NORM_EPS` | 加 eps | `Adds(normEps)`（m_k_split:366） | `Adds(sum1,eps)`（base_arch35:466） |
| :59d `torch.rsqrt` | 1/√ | `Sqrt` + `Div(1/√)`（m_k_split:368-371） | `Sqrt` + `Div(one,sum)`（base_arch35:467-468） |
| :60a `to_hf32` 双操作数 | HF32 截断 | `MmadAB` 内 `SetHF32Mode(1)` 硬件隐式截断（cube:462-463） | cube Init `SetHF32Mode(1)`（cube_arch35:88-89，:237-239 恢复） |
| :60b `F.linear` | 主 matmul | `CopyInB1`(phi, cube:194-208) + `LoadAToL0A/LoadBToL0B`(:359-454) + `MmadAB`(:456-477) + `Fixp` NZ→ND(:210-231) | cube `CopyInA2/CopyInB2/MmadBase` ping-pong（cube_arch35:140-236） |
| :60c K 部分和归约 | Σ over k分片 | Stage2 `ReduceSumARAPerf`（pre/post 段 m_k_split:391；comb 段 :449） | Part2 装载 + `VFProcessInvRmsPart3WithGroupReduce` 内 `Add(sumM)`（base_arch35:463-464） |
| :60d `* inv_rms` | 乘 rms | 折进三个消费者开头 `MulABLastDimBrcInline(rsqrt)`（base:434/:522；comb 段 m_k_split:453） | `Mul(y, sum2, rsqrt)`（base_arch35:470） |
| :61 `split` | 列切片 [0:4)/[4:8)/[8:24) | `CopyInWithOuterFor` 列偏移 0 与 4（m_k_split:380-387）；comb 偏移 `hcMult*2`（:438-442） | `mixesLocal[0]/[hcMult]/[hcMult*2]` 列偏移（m_k_split_arch35:339-341/:365；comb 调用 `mixesLocal[hcMult*2]`） |
| :63 `unflatten` | (n²)→(n,n) | 布局解释（(rows,n,n) 寻址） | 同左 |
| :64 `pre` | σ(s·x+b)+ε | `ProcessPre`（base:428-441） | `VFProcessPre`（base_arch35:591-643） |
| :65 `post` | 2σ(s·x+b) | `ProcessPost`（base:516-529，sigmoid 后 `Muls(2)`） | `VFProcessPost`（base_arch35:645-703，`VFSigmoid` 后 `Muls(2.0)`） |
| :66 comb 仿射 | s·x+b₂ | 内联三步（m_k_split:453-460） | `VFProcessCombFragPacked` 系列内 `Muls+Add`（如 RLessVL :736-737） |
| :67 `softmax(-1)+ε` | 行 softmax | `SoftmaxFP32Perf`（base:550-564） | 内联（RLessVL :738-745） |
| :68 首次列归一 | /Σ₋₂+ε | `ReduceSumARAPerf+Adds+DivABABrcInline`（m_k_split:463-467） | `Add(sum1)+Div`（RLessVL :746-755） |
| :69-71 Sinkhorn ×19 | 行/列交替归一 | 循环体（m_k_split:468-484） | `iters` 循环（RLessVL :757-778，调用点传 `iterTimes-1`） |
| :72 `y` | Σₙ pre·x → bf16 | `ProcessY`（base:503-513） | `VFProcessY`（base_arch35:1013-1064） |
| :73 `return` | 写回 GM | `CopyOut` y/post/combFrag（m_k_split:412-416/:428-431/:488+） | Part2 `CopyOut`（m_k_split_arch35:358/:369+/:377+） |

---

## 3. 逐行详解

### 3.1 `x_float = x.float()`（run_golden.py:57）

**语义**：bf16 → fp32，逐元素无损转换（bf16 是 fp32 的尾数截断子集）。

**arch22**：发生在 Stage1 的 AIV 侧（m_k_split_core.h:145-162）：

```cpp
xLocal = xQue.template DeQue<T>();                 // :150  UB 内 bf16
xCastLocal = mmInQue.AllocTensor<float>();         // :151
CastTwoDim(xCastLocal, xLocal, curUbMFactor, realKGmSize);  // :152
...
CopyOut(xCastLocal, xCastFp32WsGm[cutMmInOffset], ...);     // :159  写 GM workspace
```

`CastTwoDim`（base.h:484-500）对 bf16→float 用 `RoundMode::CAST_NONE`（无损方向），逐行 `Cast`；随后 `CopyOut` 把 fp32 x 写进该 AIC 配对核专属的 xCast workspace 分片（双缓冲，:156-158 计算 double-buffer 偏移）。AIC 在 `ComputeDecode` 里经 `CopyInA1`（cube:177-192，`DataCopy` Nd2Nz，把 ND 行主序转成 cube 需要的 NZ fractal 布局）搬进 L1A。

**arch35**：cast 与平方和**融合**在同一次遍历里（`VFProcessCastAndInvRmsPart1`，详见 3.3），cast 后的 x 在 UB 内经 `VFTransND2NZ`（base_arch35:142，ND→NZ）转布局后 **`CopyToL1` 直接进 L1**（m_k_split_arch35:176-194）——不经过 GM workspace，省一次 GM 往返（arch35 配对 AIV/AIC 共享 L1）。

### 3.2 `x_flat = x_float.flatten(-2)`（run_golden.py:58）

**纯布局操作，kernel 无对应算子**。golden 把 (b,s,4,4096) 折成 (b,s,16384) 行主序矩阵；kernel 从一开始就按行主序寻址：`curGlobalxOffset = (mVectorOffset + i*stage1MFactor) * k + kGmBaseOffset`（m_k_split:146-148，`k = hcMult*d = 16384`）。flatten 的语义被"行×K 的扁平寻址"自然吸收。

### 3.3 `inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)`（run_golden.py:59）

这一行在 kernel 里被拆成 4 段，且**两个架构的切分位置完全不同**。

#### 3.3.1 square 段（Σx²）

**arch22 —— cube 上的块对角 Gram**：

`MmadA2`（hc_pre_cube_compute.h:417-437）：

```cpp
for (uint64_t mL0Offset = 0; mL0Offset < mmParams.curML1; mL0Offset += BLOCK_CUBE) { // 16行一块
    Mmad(l0c_[... + mL0Offset * BLOCK_CUBE],        // C_i：M×16 带上第 i 块
         l0a_[... + mL0Offset * ...], l0b_[... + mL0Offset * ...],  // A_i 与 B_i 是同一块 x（B 转置装载）
         {m=16, n=16, k=32});
}
```

- **A/B 同源**：`LoadAToL0A`（:359-395，用 LoadData3D + `padList[3]=255` 借卷积 3D 装载器一次产出 ZZ fractal）与 `LoadAToL0B`（:397-415，LoadData2D + 步长技巧转置装载）读的是**同一块 L1A 的 x**，因此算出的是 `x_i·x_iᵀ` 的 16×16 块，块间配对从不发生——等效于只算完整 Gram 的**块对角带（1/32 计算量）**，每块对角线 16 个元素即该 16 行各自的 Σx²（K 部分和）。
- **为什么用 16×16 而非只算对角**：Mmad 的 N 最小粒度是 16（一个 fractal），16×16 是能容纳 16 个对角值的最小 cube 可计算对象。
- **复用动机**：x 分片本来就要为主 matmul 装载，Gram 只多一次 `LoadAToL0B` + 一条 Mmad，数据零额外搬运。
- 结果经 `Fixp`（:233-251）`CopyOut` 到 squareSum workspace（NZ→NZ，(M,16)）。

**arch35 —— AIV 上的融合平方和**（`VFProcessCastAndInvRmsPart1`，base_arch35:237-315）：

```cpp
for (j = 0; j < loopCount; j++) {
    LoadInputData(x, xLocalAddr, pregLoop, i * curColNumAlign + j * VL_FP32); // 取一行（一段K）
    Mul(x1, x, x, pregLoop);       // :268  x²
    Add(sum, sum, x1, pregMain);   // :269  行内累加（寄存器）
    StoreOutputData(xCastLocalAddr, x, ...);  // :270  顺带完成 bf16→fp32 cast（golden :57 在此融合）
}
Muls(sum, sum, coeff, pregMain);   // :272  ×(1/k) —— golden 的 mean 也在此融合
ReduceSum(sum, sum, pregMain);     // :273  寄存器内行归约 → 每行一个 Σx²
DataCopy<..., DIST_FIRST_ELEMENT_B32>(rmsNormLocalAddr + i, sum, ...); // 取首元素存每行结果
```

`WithUbReduce` 模板参数（:261-263/:274-276）控制多 K 块之间是"累加"还是"覆盖"——第一个 cvLoop 用 false 初始化、后续用 true 累加（m_k_split_arch35:168-172），实现 **K-split 部分和**。

#### 3.3.2 mean 段（÷K）

golden 的 `.mean(-1)` = Σx²/k。arch22 在 Stage2 补上：`Muls(rsqrtLocal, rsqrtLocal, coeff)`，`coeff = 1.0f/tilingData->k`（m_k_split:363-364）；arch35 在 Part1 融合（见上 `Muls(sum, sum, coeff)`，coeff 定义在 m_k_split_arch35:160）。

#### 3.3.3 K 部分和归约（K-split 的拼装）

**arch22**（m_k_split:347-358）：Stage2 把 `cubeBlockDimK`（如 bs=512 时为 11）份 (rows,16) Gram 逐元素相加：

```cpp
CopyIn(workspaceGm[... + stage2BlockIdx * ... + rowOuterIdx * ...], squareSumOutLocal,
       stage1UsedCoreNum, curRowFactor * SQUARE_SUM_SIZE, ...);  // :349-354 装 11 份
ReduceSumARAPerf(squareReduceLocal, squareSumOutLocal, 1, stage1UsedCoreNum,
                 curRowFactor * SQUARE_SUM_SIZE);                 // :357 dim1=11 逐元素加
```

**arch35**：Part1 期间 `WithUbReduce` 已在同核内累加（见上），跨 K 组的累加在 Part2 的 `VFProcessInvRmsPart3WithGroupReduce`（base_arch35:431-560）用**广播标量累加**完成：`LoadInputDataWithBrc(x, xLocalAddr, ...)` 把 rmsGm 里某个 (k组,token) 的标量广播到整条寄存器，`Add(sum1, sum1, x)` 累加——同时同一循环里也在累加 mm 的 K 部分和（`Add(sumM...)`，对应 golden :60 的 K 归约）。

#### 3.3.4 对角提取 + eps + rsqrt（arch22 特有的收尾）

```cpp
GatherMaskByDiagonal(rsqrtLocal, squareReduceLocal,
                     maskPatternLocal[(curBsIdxForAll % SQUARE_SUM_SIZE) * 8], curRowFactor); // :361-362
Muls(rsqrtLocal, rsqrtLocal, coeff, curRowFactor);   // :364  ×(1/k)
Adds(rsqrtLocal, rsqrtLocal, tilingData->normEps, curRowFactor);  // :366  +eps
Sqrt(rsqrtLocal, rsqrtLocal, curRowFactor);          // :368
Duplicate(rowBrcbLocal0, 1.0f, curRowFactor);       // :369
Div(rsqrtLocal, rowBrcbLocal0, rsqrtLocal, curRowFactor);  // :371  1/√
```

- `GatherMaskByDiagonal`（base.h:72-88）：`GatherMask` 指令按预生成掩码（`SetGatherMaskPattern`，base.h:62-70，16 组 `1<<i` 位模式）从每个 256 元素（16×16 NZ 列优先）块中抽 16 个对角元素；掩码基址带 `(curBsIdxForAll % 16) * 8` 的**相位旋转**——token 全局行号在 16 行块组内的相位决定对角位置。
- **rsqrt 用 `Sqrt` + `Div` 两步实现**（而非一条 Rsqrt 指令）：两步各自的舍入行为与 golden 的 `torch.rsqrt` 对齐策略一致（容忍阈值内）。
- **注意精度细节**：Σx² 全程 fp32，无 HF32 截断——对应 golden 该行不做 `to_hf32`（x 来自 bf16 本就无损，且 golden 的 square 不经 HF32）。

### 3.4 `mixes = F.linear(to_hf32(x_flat), to_hf32(hc_fn)) * inv_rms`（run_golden.py:60）

#### 3.4.1 `to_hf32` —— 硬件隐式 HF32 截断

golden 的 `to_hf32`（run_golden.py:50-53）用位掩码把 fp32 尾数截到 10 位，模拟 cube 的 HF32 输入。kernel 里**没有任何显式 cast 指令**：

```cpp
AscendC::SetHF32Mode(1);          // cube:462（arch35 cube_arch35:88-89）
AscendC::SetHF32TransMode(1);
Mmad(...);                        // 操作数在 MAC 输入端被硬件截成 HF32，累加保持 fp32
AscendC::SetHF32Mode(0);          // cube:476
```

**HF32 只包裹主 matmul（MmadAB）**，Gram（MmadA2）不开启——与 golden 的 `to_hf32` 只出现在 `F.linear` 参数上完全一致。数值上：x 侧 bf16(8位有效)→HF32(11位) 无损，`hc_fn` 侧 fp32(24位)→HF32 真截断，故 golden 对两个操作数都调 `to_hf32` 只是统一模拟硬件行为。

#### 3.4.2 `F.linear` —— 主 matmul 流水

**arch22**（`ComputeDecode`，cube:266-344，三级双缓冲软件流水）：

```
MTE2(GM→L1):  [装L1A_0][装L1A_1][装L1A_0']…   CopyInA1(x, Nd2Nz) / CopyInB1(phi, Nd2Nz→L1B)
MTE1(L1→L0):           [x→L0A/L0B][phi→L0B]…  LoadAToL0A / LoadAToL0B / LoadBToL0B
M(Mmad):                       [A2+AB 累加]…    MmadA2(Gram) + MmadAB(主matmul)
Fixp:                                            …最后 CopyOut 双结果
```

- 每 32-K 子步（`K_L0_SIZE=32`）执行：`LoadAToL0A` → `LoadAToL0B` → `MmadA2` → `LoadBToL0B`（L0B 被 xᵀ/phi 复用，ping-pong 两次）→ `MmadAB`。
- `cmatrixInitVal = isFirstK && kGmOffset==0`（:428/:468）：仅全局首个 K 片清零 L0C，其余累加——**K-split 累加语义**。
- `unitFlag`：非最后片 =2（保持 L0C 累加窗口常开），最后片 =3（自动关窗，之后才能 Fixp）（:430/:470）。
- 结果 `Fixp`（:233-251）：mixes 经 `CopyOut` **NZ→ND**（`nz2ndEn=true` + `SetFixpipeNz2ndFlag`，dstStride=nOutSize，cube:220-223）写入 mmOut workspace，方便 Stage2 按 ND 消费。

**arch35**：cube 侧 L0 ping-pong（aL0Ping_/Pong_ 等，cube_arch35:44-78）+ `MmadBase`（:176-188）；x 由 AIV 直达 L1（见 3.1），phi 经 `CopyInB2` 进 L0B。

#### 3.4.3 K 部分和归约（×inv_rms 之前）

**arch22**（m_k_split:373-392）：pre/post 两段（列 [0:4)+[4:8)）一起装载、一次归约：

```cpp
CopyInWithOuterFor(workspaceGm[mixBaseOffset], mixes01Local, stage1UsedCoreNum,
                   curRowFactor, tilingData->hcMult, tilingData->bs, ...);          // :380  pre 段
CopyInWithOuterFor(workspaceGm[mixBaseOffset + tilingData->hcMult],                 // :383  post 段（列偏移4）
                   mixes01Local[stage1UsedCoreNum * ... * hcMultAlign], ...);
ReduceSumARAPerf(mixes01ReduceLocal, mixes01Local, NUM_TWO, stage1UsedCoreNum,
                 curRowFactor * tilingData->hcMultAlign);                            // :391  dim0=2 同时归约两段
```

comb 段（16 列）单独装载（:435-446，逐 (核,行) `CopyIn` 偏移 `hcMult*2`）后 `ReduceSumARAPerf(mixes02ReduceLocal, mixes2Local, 1, k份数, row*4*8)`（:449）。

**arch35**：见 3.3.3——`VFProcessInvRmsPart3WithGroupReduce` 把 mm 的 K 累加（`Add(sumM...)`）与 rms 的 K 累加（`Add(sumX...)`）放进同一循环，随后一次完成 rsqrt 与乘法：

```cpp
Adds(sum1, sum1, eps, pregMerge);  Sqrt(sum1, sum1, ...);  Div(rsqrt, one, sum1, ...);  // :466-468
Duplicate(rsqrt, rsqrt, pregLoop);                                                        // :469  每 token 标量广播到 24 列
Mul(y, sum2, rsqrt, pregLoop);                                                            // :470  mixes × inv_rms ← golden :60d
```

#### 3.4.4 `* inv_rms` —— 折进三个消费者

golden 里乘在 mixes 上；arch22 把它**分布到 pre/post/comb 三个处理函数的开头**（数学等价，per-token 标量）：

- `ProcessPre`：`MulABLastDimBrcInline(mixLocal, mixLocal, rsqrtLocal, ...)`（base:434）
- `ProcessPost`：同（base:522）
- comb 段：`MulABLastDimBrcInline<float,false>(mixes02ReduceLocal, ..., rsqrtLocal, ...)`（m_k_split:453）

`MulABLastDimBrcInline`（base:91-137）是 broadcast 乘的封装：`Brcb` 把 (rows,1) 的 rsqrt 块广播，或直接用 `src1RepStride=0` 让第二操作数在 repeat 间"原地不动"（:117/:132）——不物化广播张量。arch35 则在 3.4.3 的 `Mul(y, sum2, rsqrt)` 一次完成（rsqrt 已被 `Duplicate` 广播到全寄存器）。

### 3.5 `pre, post, comb_frag = mixes.split(...)` / `unflatten`（run_golden.py:61/:63）

split/unflatten 是纯视图操作，kernel 用**列偏移寻址**实现：mixes 每行 24 列（`hcMix=24`，workspace 侧对齐到 128），pre=[0:4)、post=[4:8)、comb=[8:24)。arch22 的两处 `CopyInWithOuterFor` 列偏移（0 / hcMult）与 comb 的 `hcMult*2` 偏移（m_k_split:380-387/:438-442）即 split 的物化；unflatten 体现为 comb 段按 (rows, 4, 4) 三维寻址（`ReduceSumARAPerf(..., hcMult, hcMult)` 的 dim1/dim2）。arch35 直接用 `mixesLocal[0]/[hcMult]/[hcMult*2]` 起址（m_k_split_arch35:339-341/:365）。

### 3.6 `pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS`（run_golden.py:64）

**arch22 `ProcessPre`**（base:428-441）：

```cpp
MulABLastDimBrcInline<float,true>(mixLocal, mixLocal, rsqrtLocal, tmpBuffer0, ...);  // ×inv_rms（golden :60d 折入）
Muls(mixLocal, mixLocal, scale, curRowNum * curColNumAlign);                         // ×hc_scale[0]
AddBAFirstDimBrcInline<float>(mixLocal, mixLocal, hcBaseLocal, ...);                 // +hc_base[:4]（首维广播）
SigmoidPerf(preLocal, mixLocal, tmpBuffer1, ...);                                    // sigmoid
Adds(preLocal, preLocal, eps, ...);                                                  // +HC_EPS
```

- `AddBAFirstDimBrcInline`（base:324-388）：`Add` + `src1RepStride=0`（:356）实现 (1,col)→(row,col) 的**首维广播加**，分支处理 255 repeat-stride 上限与尾块。
- `SigmoidPerf`（base:403-426）= `1/(1+exp(-x))`：`Duplicate(tmp,1)` → `CalcDenominator`（base:390-400：`Muls(-1)→Exp→Adds(1)`）→ `Div(1, denom)`。**三步各自舍入**，与 golden 的 torch.sigmoid 数值路径对齐（也解释了为何不用融合 FMA 类指令）。
- **为什么 Muls+Add 两条而不是一条 Axpy**：Axpy 的加数只能是 dst 自身且是单舍入 FMA，无法表达带 broadcast 的第三操作数；两步舍入与 golden 逐算子舍入对齐。

**arch35 `VFProcessPre`**（base_arch35:591-643，寄存器版）：

```cpp
LoadInputData<float>(base, hcBaseLocalAddr, pregLoop, ...);  // base 预载寄存器
Muls(mix, mix, scale, pregLoop);      // :615
Add(mix, mix, base, pregLoop);        // :616
VFSigmoid(mix, mix, one, pregLoop);   // :617  σ(x)=one/(1+e⁻ˣ)，one 预置 1.0
Adds(mix, mix, eps, pregLoop);        // :618
```

`VFSigmoid` 的第三操作数 `one` 是分子系数——`Duplicate(one, 1.0)` 时算普通 sigmoid；这个设计让同一指令可表达 `c·σ(x)` 形态。

### 3.7 `post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[4:8])`（run_golden.py:65）

**arch22 `ProcessPost`**（base:516-529）：与 ProcessPre 同构，差异仅两点——输入列偏移 `mixes01ReduceLocal[stage2RowFactor * hcMultAlign]`（m_k_split:422，即 post 段），以及 sigmoid 后多一步 `Muls(postLocal, postLocal, 2.0f)`（base:527）实现 golden 的 `2·`。

**arch35 `VFProcessPost`**（base_arch35:645-703）：`Muls(scale) → Add(base) → VFSigmoid(one=1) → Muls(2.0)`（:673-676），另用 `DataCopyUnAlignPre/LoadInputDataUnalign`（:666/:672）处理 mix 列段起始地址非对齐的装载。

**语义注**：post 无 `+HC_EPS`（golden 有意为之——post 是纯门控，0 是合法工作点，详见算子语义讨论）。

### 3.8 `comb_frag = comb_frag * hc_scale[2] + hc_base[8:].view(4,4)`（run_golden.py:66）

**arch22**：直接内联在 Part2 主循环（m_k_split:453-460），与 3.4.4 的 ×inv_rms 连成四连：

```cpp
MulABLastDimBrcInline<float,false>(mixes02ReduceLocal, ..., rsqrtLocal, ...);   // ×inv_rms
Muls(mixes02ReduceLocal, ..., hcScaleGm.GetValue(NUM_TWO), ...);                // ×hc_scale[2]
AddBAFirstDimBrcInline<float>(mixes02ReduceLocal, ..., hcBase2Local, ...);      // +hc_base[8:]（16 元素按 (4,4) 广播）
```

`hcBase2Local` 是 (4,4) 的 base（Init 时 `CopyIn(hcBaseGm[hcMult*2], hcBase2Local, hcMult, hcMult)`，m_k_split:330），首维广播语义恰好等价于 golden 的 `.view(4,4)` 广播。

**arch35**：在 `VFProcessCombFragRLessVL`（base_arch35:706-780）等三个变体函数（`RLessVL`/`UseFourUnfold`/`Packed`，调用点见 m_k_split_arch35:373-375 传 `iterTimes-1`）内：`Muls(mix, mix, scale)` + `Add(mix, mix, base)`（:736-737）。

### 3.9 `comb_frag.softmax(-1) + HC_EPS`（run_golden.py:67）

**arch22 `SoftmaxFP32Perf`**（base:550-564，注释注明"R 轴小于 64"——n=4 满足）：

```cpp
LastDimReduceMaxPerf(tmpReduceBuffer, input, curRowNum, curColNum);   // :555 行 max
SubABLastDimBrcInline<float,true>(output, input, tmpReduceBuffer, ...); // :556 x - max（行广播减）
Exp(output, output, curRowNum * curColNumAlign);                      // :558
LastDimReduceSumPerf(tmpReduceBuffer, output, ...);                   // :560 行 Σexp
DivABLastDimBrcInline<float,true>(output, output, tmpReduceBuffer, ...); // :561 /Σ
Adds(output, output, eps, ...);                                       // :562 +HC_EPS
```

- `LastDimReduceMaxPerf`（:531-538）用 **`WholeReduceMax(..., ReduceOrder::ORDER_ONLY_VALUE)`**——注意这里（"长行、少输出"场景：行内 4 元素…实际是 rows×4 的每行归约）用的是归约指令家族；与 Σx² 的选择（`ReduceSumARAPerf` 加法树）相反——因为这里归约轴=列(4)、输出=每行 1 个，归约指令产出匹配。`LastDimReduceSumPerf`（:540-546）同构用 `WholeReduceSum`。
- 与 torch softmax 的数值路径一致（exp(x−max)/Σ），`+eps` 防后续 Sinkhorn 除法退化。

**arch35**：寄存器内联（RLessVL :738-745）：`ReduceMax → Duplicate(max) → Sub → Exp → ReduceSum → Duplicate(sum) → Div → Adds(eps)`。`Duplicate`（寄存器标量广播）替代了 arch22 的 Brcb/repeat-stride 广播机制。

### 3.10 `comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + HC_EPS)`（run_golden.py:68，首次列归一）

**arch22**（m_k_split:463-467）：

```cpp
ReduceSumARAPerf(reduceLocal, mixes02ReduceLocal, curRowFactor, hcMult, hcMult); // 列和：(rows,4,4) 沿 dim1(-2) 求和 → (rows,1,4)
Adds(reduceLocal, reduceLocal, hcEps, curRowFactor * hcMult);                    // +eps
DivABABrcInline(combFragLocal, mixes02ReduceLocal, reduceLocal, curRowFactor, hcMult, hcMult); // 每列除以自己的列和
```

`DivABABrcInline`（base:568-616）实现 (d0,d1,d2)/(d0,1,d2) 的**中维广播除**：`src1RepStride=0` 让列和在每个 d1 重复间保持不动（:587），按 d1≥d2RepeatTimes 与否分两个 repeat 组织方向。

**arch35**（RLessVL :746-755）：softmax 循环里顺带用 `Add(sum1, sum1, mix)` 按列累加出列和（:746），`LocalMemBar` 后 `Div(mix, mix, sum1)` 逐列除（:751-755）——**把列和计算折叠进 softmax 的行循环**，一遍数据两用。

### 3.11 `for _ in range(HC_SINKHORN_ITERS - 1):` 行/列交替归一（run_golden.py:69-71）

**arch22**（m_k_split:468-484）：

```cpp
for (iter = 0; iter < tilingData->iterTimes - 1; iter++) {
    LastDimReduceSumPerf(reduceLocal, combFragLocal, rows * hcMult, hcMult);      // :469 行和（-1 轴）
    Adds(reduceLocal, reduceLocal, hcEps, ...);                                    // :471
    DivABLastDimBrcInline<float,true>(combFragLocal, combFragLocal, reduceLocal, ...); // :474 行归一（末维广播除）
    ReduceSumARAPerf(reduceLocal, combFragLocal, curRowFactor, hcMult, hcMult);    // :477 列和（-2 轴）
    Adds(reduceLocal, reduceLocal, hcEps, ...);                                    // :479
    DivABABrcInline(combFragLocal, combFragLocal, reduceLocal, ...);               // :482 列归一（中维广播除）
}
```

每轮 6 步向量操作（行和、+eps、行除、列和、+eps、列除），19 轮全在 UB 内完成——数据不出片上。`DivABLastDimBrcInline`（base:247）是**末维**广播除（(r,n,n)/(r,n,1)），与 `DivABABrcInline`（中维）互补；两者与 `SubABLastDimBrcInline`（softmax 用）构成 arch22 的三个广播二元模板族。

**arch35**：RLessVL :757-778 的 `iters` 循环（`VFProcessIteration`，:782-790，是行归一四连的提取：`ReduceSum→Duplicate→Adds→Div→Add(sum1)`——同样把列和折叠进行归一）；`UseFourUnfold`（:792+）是 4 路展开的加速变体，`Packed`（:925+）是打包布局变体，三者按 shape/布局选一（m_k_split_arch35:373 调用 `Packed`，传 `iterTimes-1`）。

### 3.12 `y = (pre.unsqueeze(-1) * x_float).sum(dim=-2).to(x.dtype)`（run_golden.py:72）

**arch22 `ProcessY`**（base:503-513）：

```cpp
CastTwoDim(xCastLocal, xLocal, dim0 * dim1, dim2);                              // bf16→fp32（再次 cast，Stage2 本地）
MulABLastDimBrcInline<float,true>(xCastLocal, xCastLocal, mix01Local, hcBrcbLocal1, dim0 * dim1, dim2); // ×pre
ReduceSumARAPerf(yCastLocal, xCastLocal, dim0, dim1, dim2);                     // 沿 dim1(-2, hc=4) 加法树求和
CastTwoDim(yLocal, yCastLocal, dim0, dim2);                                     // fp32→bf16（CAST_RINT）
```

- `dim0=rows, dim1=hcMult(4), dim2=d`；`mix01Local` 即 ProcessPre 输出的 pre（含 +eps）。
- **加法树而非 ReduceSum 指令**：`ReduceSumARAPerf`（base:443-482）= "拷首行 + 逐行全长 Add"（:456 DataCopy 行0，:470 for j=1..3 Add）。归约轴只有 4、保留轴 4096 长——"短归约轴+长保留轴"下加法树满宽并行，pattern 归约指令（如 WholeReduceSum）产出吞吐低不适合。累加顺序 k=0..3 与 golden `sum(-2)` 一致。
- y 的 d 维可能分 `dLoop` 轮（UB 放不下时，tiling 的 dFactor 自适应；d=4096 通常一轮全载）。

**arch35 `VFProcessY`**（base_arch35:1013-1064）：寄存器加法树，**乘 pre 融合进归约循环**：

```cpp
Duplicate(sum, 0.0f, ...);
for (k = 0; k < hcMult; k++) {
    LoadInputDataWithBrc<float>(mix, mixLocalAddr, ..., i * hcMixAlign + k);  // pre[token,k] 标量广播
    LoadInputData<T>(x, xLocalAddr, ..., i * hcMult * dAlign + ... + k * dAlign);  // 以 T(bf16) 源装载进 float 寄存器（MicroAPI 装载内升精度，见文件头 castTraitB162B32 路径）
    Mul(x, mix, x, ...);      // :1037  pre_k × x_k
    Add(sum, sum, x, ...);    // :1038  顺序累加
}
StoreOutputData(yLocalAddr, sum, ...);  // 存回 bf16
```

`LoadInputDataWithBrc`（标量广播装载）让乘法无需单独的广播指令——arch35 MicroAPI 的典型风格。

### 3.13 `return y, post, comb_frag`（run_golden.py:73）

Stage2 每个 AIV 算完后经 OutQue 走 `CopyOut` 写回 GM：y（m_k_split:412-416，dLoop 轮）、post（:428-431）、combFrag（:488+，含 (rows,4,4)→(rows,16) 的落盘布局）。arch35 对应 m_k_split_arch35:358/:369+/:377+。

---

## 4. 精度对齐要点汇总

| # | 要点 | 证据 |
|---|---|---|
| 1 | **HF32 截断只作用于主 matmul**，Gram/Σx²/所有后处理全程 fp32 | cube:462-463 只包 MmadAB；golden to_hf32 只在 F.linear 参数 |
| 2 | x 侧 HF32 无损（bf16 8 位有效 ⊂ HF32 11 位），hc_fn 侧真截断 | bf16→fp32 cast 用 CAST_NONE（base:490-493） |
| 3 | **rsqrt = Sqrt + Div 两步**，不用单条指令 | m_k_split:368-371；base_arch35:467-468 |
| 4 | **sigmoid = 1/(1+exp(-x)) 三步**（Muls/Exp/Adds/Div 各自舍入），不用融合 FMA | base:390-426；VFSigmoid 同构 |
| 5 | **broadcast 不物化**：repeat-stride=0（arch22）/ Duplicate 标量广播（arch35），既省带宽也保数值路径 | base:117/:356；base_arch35:469 |
| 6 | **归约顺序对齐**：y 的 4 分支加法树按 k=0..3 顺序累加，与 golden sum(-2) 一致 | base:470；base_arch35:1034-1038 |
| 7 | **乘加分步**（Muls+Add 而非 Axpy）：两步舍入匹配 golden 逐算子语义；Axpy 是单舍入 FMA 且不支持 broadcast 加数 | base:434-437（讨论见会话记录） |
| 8 | Sinkhorn 数值顺序 kernel-exact：softmax→+eps→列归一→(行归一→列归一)×19，每步分母 +eps | m_k_split:461-484 与 golden :67-71 逐步对应 |

---

## 5. 附录

### 5.1 关键常量（base.h:23-40 / cube_compute.h）

| 常量 | 值 | 含义 |
|---|---|---|
| `BLOCK_CUBE` | 16 | cube fractal 边长（Mmad 最小 m/n） |
| `K_L0_SIZE` | 32 | L0 内 K 推进粒度 |
| `SQUARE_SUM_SIZE` | 16 | Gram 输出列宽（块对角带） |
| `N_SIZE` | 24 | mixes 列数（hcMix） |
| `CV_RATIO` | 2 | AIV:AIC 配对比 |
| `M_L1_MAX_SIZE` | 256 | M 方向每核行上限（tiling） |
| `L1_BUF_OFFSET` 等 | 128×256/32×256/64×256/256×16 | L1/L0/L0C/L0C-A2 双缓冲分片偏移 |
| `UNIT_FLAG_ENABLE(_AUTO_CLOSE)` | 2 / 3 | L0C 累加窗口 开 / 末片自动关 |

### 5.2 关键 tiling 字段（hc_pre_tiling.cpp 推导）

| 字段 | 含义 | bs=512 示例（24 AIC） |
|---|---|---|
| `cubeBlockDimM/K` | AIC 网格 M×K | 2 × 11 |
| `multCoreSplitMSize/KSize` | 每核行段 / K 段 | 256 / 1536 |
| `mL1Size / kL1Size` | L1 装载 M / K 粒度 | 256 / 128 |
| `cvLoopKSize` | AIV 喂数 K 节拍 | 1024 |
| `stage1MFactor` | AIV 每 UB 轮行数 | ≈16 |
| `rowOfFormerBlock/TailBlock` | Stage2 AIV 行段 | 11 / 6 |
| `stage2RowFactor` | Stage2 UB 行数 | 1 |
| `secondUsedCoreNum` | Stage2 有效 AIV 数 | 47 |
| `dLoop/dFactor` | y 的 d 维分轮 | 1 / 4096 |
| `iterTimes` | Sinkhorn 轮数 | 20（kernel 循环 19） |

### 5.3 相关文档

- 《hcpre_shape_and_core_split.md》：分核逻辑（M×K 网格、AIV 配对、两阶段核分配）
- golden：`scripts/hc_pre/run_golden.py`、`scripts/hc_pre/hcpre_sim_app.py`
- mhc_pre_sinkhorn 的对照实现：`ops-transformer/mhc/mhc_pre_sinkhorn/`（语义相同，tiling/文件组织独立）
