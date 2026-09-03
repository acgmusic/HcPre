# HcPre 算子：用例 shape 解析与分核逻辑分析

> 依据：`csrc/moe/hc_pre/op_host/hc_pre_tiling.cpp`、`csrc/moe/hc_pre/op_kernel/hc_pre.cpp`、`csrc/moe/hc_pre/op_kernel/hc_pre_m_k_split_core.h`、`csrc/moe/hc_pre/op_kernel/hc_pre_base.h`、`csrc/moe/hc_pre/op_host/hc_pre_def.cpp`、`tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_hc_pre.py`
>
> 仿真环境：A3（Ascend910_9382），24 AIC + 48 AIV

---

## 一、用例 shape 解析

用例（ST 标准三段式格式：shape / dtype / format）：

```
输入：
"1,2,4,4096;24,16384;3;24"     ← 4 个输入的 shape
FLOAT;FLOAT;FLOAT;FLOAT          ← dtype
ND;ND;ND;ND                      ← format

输出：
"1,2,4096;1,2,4;1,2,4,4"        ← 3 个输出的 shape
DT_BF16;FLOAT;FLOAT              ← dtype
ND;ND;ND                         ← format
```

### 1.1 输入张量

| 张量 | shape | 含义 |
|---|---|---|
| `x` | `1,2,4,4096` | 主输入：4D = (batch=1, size=2, hc=4, d=4096)。batch 批大小；size token 数；**hc=4 超连接流数**（算子硬约束 `HC_PRE_HC_LIMIT=4`，见 `torch_binding.cpp` 的 `check_hc_pre_shape_and_dtype`）；d=4096 隐藏维（约束 4096 或 7168，即 DSV4 flash hidden size） |
| `hc_fn` | `24,16384` | 混合矩阵：24 = MIX_HC（4 pre + 4 post + 4×4 comb_frag 共 24 行）；16384 = hc×d = 4×4096（matmul 的 fan-in） |
| `hc_scale` | `3` | 3 个缩放因子（1-D），分别作用于 pre / post / comb_frag 三段 |
| `hc_base` | `24` | 24 个偏置（1-D），与 hc_fn 的 24 行一一对应 |

tiling 侧解析（`GetShapeAttrsInfoInner`，hc_pre_tiling.cpp:97）：
- 4D 输入折叠为 `bs = b*s = 2`、`hcMult = 4`、`d = 4096`（3D 输入则 `bs = dim0`）
- 校验：`hc_fn.shape[1] == d * hcMult`、`hc_scale.shape[0] == 3`、`hc_base.shape[0] == hcMix`

### 1.2 输出张量

| 张量 | shape | 含义 |
|---|---|---|
| `y` | `1,2,4096` | 融合输出：(batch, size, d)，4 条 hc 流经 pre 门控加权求和后落回隐藏维 |
| `post` | `1,2,4` | 后置门控：(batch, size, hc_mult)，`2·sigmoid(post·scale[1]+base)` |
| `comb_frag` | `1,2,4,4` | 双随机矩阵：(batch, size, 4, 4)，经 softmax + 20 轮 Sinkhorn 归一化的流间组合片段 |

### 1.3 数值关系

```
mixes = RMSnorm(x) · hc_fnᵀ        # (bs, k=hc·d) × (k, 24) → (bs, 24)
pre, post, comb_frag = split(mixes) # 按 4 / 4 / 16 切分
pre       = sigmoid(pre·scale[0] + base) + eps
post      = 2·sigmoid(post·scale[1] + base)
comb_frag = Sinkhorn_iter20(comb_frag·scale[2] + base)   # 4×4 双随机化
y         = Σ_hc(pre ⊙ x)           # (bs, 4, d) → (bs, d)
```

### 1.4 一处已说明的偏差

用例中 `x` 标注 FLOAT，但算子定义（`hc_pre_def.cpp`）只接受 `DT_BF16`（输出段 `y=DT_BF16` 也印证了这点），所以仿真脚本中 x 按 BF16 构造，FLOAT 只对 hc_fn/hc_scale/hc_base 生效。

---

## 二、分核逻辑

以本用例（bs=2, d=4096, k=hc×d=16384）代入。kernel 声明为 `KERNEL_TYPE_MIX_AIC_1_2`：
**每个 block = 1 个 AIC + 2 个 AIV**（A3 上 24 AIC + 48 AIV）。`PostTiling()` 设
`blockDim = aicCoreNum = 24`。算子内部分两个阶段串行（Part1 → workspace → Part2），中间 `SyncAll`。

### 2.1 Stage 1（Part1：`mixes = RMSnorm(x)·hc_fnᵀ` 的 Cube 阶段）—— M×K 二维切核

`CalcOpTiling()`（hc_pre_tiling.cpp:277）把 matmul `[bs, k]×[k, 24]` 的 **M 轴和 K 轴**
切到 AIC 网格：

```
mDimNum     = min(aicCoreNum, ceil(bs / 256))       # M 方向核数（M_L1_MAX_SIZE=256）
kDimNum     = aicCoreNum / mDimNum                  # K 方向核数
splitKSize  = RoundUp(ceil(k / kDimNum), 256)       # 每核 K 段，256 对齐（K_MULIT_CORE_SPLIT_BASE_SIZE）
```

kernel 侧（hc_pre_m_k_split_core.h:87-99）：

```
mBlkDimIdx = blockIdx / cubeBlockDimK
kBlkDimIdx = blockIdx % cubeBlockDimK
```

各核处理自己 的 K 切片，部分和写 workspace（`mmOutFp32Ws` / `squareSumFp32Ws`
按 `kBlkDimIdx` 分片）。

**本例代入**：bs=2 → `mDimNum = min(24, 1) = 1`，`kDimNum = 24`，
`splitKSize = RoundUp(⌈16384/24⌉=683, 256) = 768`，`cubeBlockDimK = ⌈16384/768⌉ = 22`。
即 **M 用 1 个核、K 切 22 个核，共 22 个 AIC 干活，blockIdx 22/23 的 M 偏移越界空转**。

**核内再切**：
- M 按 `mL1Size = min(256, singleCoreM)` 进 L1（本例 singleCoreM = RoundUp(2,16) = 16）
- K 按 `kL1Size`（`A_L1_SIZE / mL1Size` 封顶 1024，128 对齐）和 `cvLoopKSize = 1024` 分批循环

**AIC/AIV 配合**（同一 block 内）：
- 2 个 AIV 把 x 的 bf16→fp32 cast 到 workspace，按行对半分
  （`mVectorLength = (realM+1)/2`，奇偶 `curVectorBlockIdx % 2` 区分）
- 每轮 cv-loop 用 `CrossCoreSetFlag / CrossCoreWaitFlag`（flag 8/9，MODE2 双缓冲）
  与 AIC 的 MMAD 乒乓同步

### 2.2 Stage 2（Part2：归约 + 激活 + Sinkhorn + y 融合）—— 纯 Vector，按 bs 行切核

`CalcMKSplitCoreMembasePart2Tiling()`（hc_pre_tiling.cpp:145）：

```
rowOfFormerBlock = ceil(bs / aivCoreNum)            # 每核行数（aivCoreNum = 48）
usedAivCoreNums  = min(ceil(bs / rowOfFormerBlock), 48)
```

kernel 侧（hc_pre_m_k_split_core.h:317-321）：`stage2BlockIdx = GetBlockIdx()`，
**≥ usedAivCoreNums 的 AIV 直接 return**。

活跃核的工作：
1. 把 `cubeBlockDimK`（本例 22）份 K 部分和归约（workspace 维 reduce）
2. rsqrt 归一化（RMSnorm）
3. 切出 pre / post / comb_frag 并做激活
4. 20 轮 Sinkhorn（4×4 双随机化）
5. `y = Σ_hc(pre ⊙ x)`（d 维按 `dFactor` 分批，UB 放得下则全载）

**本例代入**：bs=2 → `rowOfFormerBlock = 1`，`usedAivCoreNums = 2` →
**只有 blockIdx 0/1 两个 AIV 干活，46 个 AIV 空转**。

### 2.3 与仿真数据互相印证

| 仿真现象 | 解释 |
|---|---|
| 24 个 cubecore 全部 ~6.34us | 22 个算 MMAD，2 个越界空转，但被 `SyncAll` 拖到同长 |
| 46 个 veccore ~6.6us | Stage2 直接 return，时间耗在 Stage1 同步等待 |
| **core0.veccore0/1 = 14.9us**（其他核 2.3 倍） | 它俩是 Stage2 仅有的 2 个活跃核（bs=2 恰好全落在这） |

code_exe 热点佐证：`hc_pre.cpp:84`（Part2 主循环）、`kernel_operator_block_sync_intf_impl.h:111`
+ `kernel_reg.h:98`（BlockSync 同步开销）、`hc_pre_base.h:478`、`hc_pre_m_k_split_core.h:477/474/391`。

### 2.4 结论

这个 shape（bs=2）远小于硬件规模：
- Stage1 只用 22/24 AIC，且每核 K 段仅 768（远小于 k=16384 的均分）
- Stage2 只用 2/48 AIV

分核逻辑本身是标准的"Cube 按 M×K 网格 + Vector 按行"两段式；瓶颈是
**小 bs 下 Stage2 行切分几乎单核化**，这也是 Part2 主循环与 BlockSync 开销集中的根因。
