# hc_pre 算子 cube-offload 优化思路可行性分析（bs=512 仿真基线）

> 分析对象：master 基线（`sim_out/hc_pre_sim_1_512_4_4096/simulator/visualize_data.bin`，WALL 98.6µs）
> 参考技能：`cannbot-skills/ops/ascendc-cube-offload`（判定框架 / tricks-catalog / fixpipe-dequant / gating-economics）
> 日期：2026-09-08

---

## 0. 基线事实（trace 实测，一切结论的前提）

| 量 | 实测值 |
|---|---|
| AIC 空闲窗口 | **31.3µs → 99.8µs = 68.5µs 全闲**（22 个 AIC，Part2 期间零活动） |
| AIV Part2 总长 | ~70µs（30.9 → 99.8），是 wall 的绝对大头 |
| Part2 内部拆分 | **y 段 ≈ 14.7µs**（30.9→45.6：x 读 + ProcessY + 每 token 归约/pre/post）；**comb 段 ≈ 54.2µs**（45.6→99.8：装载 + K 分片归约 + affine + softmax + 19 轮 Sinkhorn） |
| comb 段画像 | SCALAR 事件密度 ~700 条/µs 持续饱和（LD/ADD/ZEROEXT/SHL 地址组装为主）；向量操作仅 16 元素宽（n×n=16）、每 token ~4.9µs——**标量发射/延迟受限，非吞吐受限** |
| 每 token MTE2 | ~16 次 strided 小搬运（K 分片归集）；Part2 共 162 次 25.9µs busy |
| AIC 吞吐参照 | 本 kernel Part1 实测 ≈ 0.8M MAC/µs/核；AIC GM 读 ≈ 171GB/s/核 |

**结论性观察**：comb 段确实是 AIV 的大头（54/70µs），AIC 确实全程空闲——卸载的"门禁"（mac_ratio 低 + vec/scalar 饱和 + 融合 kernel 内）满足。

---

## 1. 原始方案回顾

```
AIC: 先算 comb 列 [b,s,nd]@[nd,n²] → 通知 AIV → 再算 pre/post 列 [b,s,nd]@[nd,2n]
AIV: 收到 comb 通知后立刻开始 comb 后处理直到完成
AIC: comb 通知后，用 cube-offload 技术帮 AIV 完成 pre/post 后处理
```

---

## 2. 方案 A：列拆分 + comb 先行 —— ✅ 可行，预期 ~3-5µs

### 2.1 关键坑：单遍交错拆分无效

不能在每个 K-chunk 内先 Mmad comb 列再 Mmad pre/post 列——**L0C 累加跨全部 14 个 K-chunk**（22 核 × K 分片），comb 的 16 列要扫完最后一个 chunk 才累加完成，单遍交错只能提前最后一个 chunk 的 ~0.3µs。

### 2.2 正确做法：两遍 K 扫描

- **pass1**：只跑 Gram + comb 列（MAC 占比 12.2/15.3 ≈ 80%）→ pass1 结束 ≈ 26-27µs（vs 原 FIXP 完成 29.5-31.3µs）→ Fixp comb → flag AIV
- **pass2**：重扫 K 只算 pre/post 列

**代价与吸收**：pass2 需重装 L1A 的 x。L1A 256KB 放不下全 K 的 1.8MB（fp32），每 AIC 从 xCast workspace 重读 ~1.8MB ≈ 10.5µs（@171GB/s）——全部落在 AIC 空闲期，不占关键路径。

**数据自洽性**：AIV comb 段的输入 = comb mm（pass1 产出）+ inv_rms（Gram 产出，pass1 同步产出）+ kernel 原始输入（hc_scale/hc_base）——自洽，可先行。

### 2.3 收益上限

AIV 在 ~26.5µs 开始 comb 段（原 45.6µs），提前 ~19µs；但 y 段（依赖 pre/post 列，须等 pass2）排到 comb 后 → 净收益 ≈ comb 段被 AIC pass2 挤掉的重叠量，预估 **~3-5µs**。

---

## 3. 方案 B：pre/post 卸载到 AIC —— 逐项判定（大部分不成立）

| 计算块 | 判定 | 依据 |
|---|---|---|
| **sigmoid**（pre/post 核心） | ❌ **cube 红线** | 非线性激活不可写成乘积和 Σa·b（技能判定框架第一条，同 FlashAttention softmax 停 AIV 的能力边界）。且仅 bs×4=2048 元素，留 AIV 是 ns 级成本 |
| affine（×scale + base） | ❌ 无意义 | 理论可上（Mmad bias per-N / diag 构造），但 2048 元素，AIV 免费 |
| **随路量化（Fixpipe dequant）** | ❌ **路径不适用** | s322f16 随路反量化挂在 **int8×int8→int32→fp16** 专用路径；本算子是 HF32 matmul、L0C 为 fp32，硬件通道不在路径上（fixpipe-dequant.md 第 4 节坑表第一条的同源问题）。**且** inv_rms 是跨 11 核 K 分片归约后的量，AIC fix 时刻不可知——数据依赖也封死 |
| **ProcessY（y = Σₙ pre·x）** | ✅ **唯一值得卸载** | 见下 |

### 3.1 ProcessY 的 cube 表达

按 16 token 一组构造块稀疏矩阵：

```
A(16×64) = per-token pre 对角块（每行 4 个非零）   ← AIV 算完 sigmoid 后 scatter 组装
x_stacked(64×4096) = 16 token × 4 分支展平        ← 复用 CopyInB1 的 Nd2Nz 装载
y = A @ x_stacked  →  L0C → Fixp → GM（最终输出）
```

- **零回路红利**：y 是最终输出，L0C 经 Fixpipe 直写 GM y 区——技能 gating §2.3 的"L0C 回 AIV 回路成本"整条消失
- **经济学**：K=64 只有 4 个有效 → 16× MAC 冗余（134M vs AIV 直算 8.4M），**违反纯吞吐 break-even（K>16 后 AIV 更便宜）**——但这是教科书式的**核间配平**场景：134M MAC ÷ 22 核 ÷ 0.8M MAC/µs ≈ 7.6µs，全部埋在 68.5µs 空闲 cube 里，换 AIV 关键路径上 ~10-12µs（y 段的 x 读+cast+mul+reduce）
- **前置条件**：AIV 先完成 pre 的 sigmoid 并组装 A（128KB，scatter 小操作）→ 新增 1 条 AIV→AIC flag；x 装载每 AIC ~750KB bf16
- **精度**：走纯 fp32 Mmad（本 kernel MmadA2 Gram 已有不开 HF32 的先例）→ 无额外截断；fp32-L0C→bf16 的 Fixpipe dtype 支持【待验证】，不支持则 AIV 补一次 cast（成本可忽略，仅 y 段输出量）

---

## 4. 组合预期（时间线推演）

```
基线:    AIV 29.6 ──── y段 14.7 ──── comb段 54.2 ────────> 99.8

方案A:   AIV 26.5 ──── comb段 54.2 ──── y段 14.7 ────> ~95.5     (−4µs)
         AIC 26.5 flag ── pass2 重扫 10.5 ──> 37 (并行, 空闲期)

A+B:     AIV 26.5 ──── comb 54.2 + pre/post 收尾 3 ──> ~84
         AIC 26.5 ── pass2 10.5 ── A组装等待 ── y-mmad ~10.5 ──> ~50 (并行)
                                                        (−16µs, −16%)
```

与已验证的 opt7（Stage2 初始化前移，−1.1µs）相互独立、可叠加。

---

## 5. 重要提醒：有一条便宜得多的路线，建议先做

### 5.1 comb 段的病根是标量发射受限，不是向量吞吐不足

comb 段 54µs 中，每条 16 元素宽的向量操作前需要 5-10 条标量指令做地址/repeat 参数组装（LD/ADD/ZEROEXT/SHL 为主，~700 条/µs），19 轮 Sinkhorn × 11 token 逐 token 串行——**向量 pipe 利用率极低**。

### 5.2 批处理改造（bat­ching）

11 个 token 的 Sinkhorn 本就相互独立（逐 token 的 4×4 矩阵归一化）：
- 批成 `(11×4, 11×4)` 块对角 或 `(11, 4, 4)` 三维张量后，向量操作变宽 11×、标量/barrier 开销摊薄不变
- 归约维度从 4 变 4（不变），但 repeat 参数组装一次服务 11 token
- 预估 comb 段 **54 → 20-25µs**，工程量远小于 cube-offload
- 工程前置：`stage2RowFactor` 从 1 提到 3-4（x/xCast/yCast 每行 ~112KB → 3 行 ~336KB，UB 184KB 放不下 → 需 ProcessY 卸载或减小 dFactor 配合——**两者正交互补**：若 ProcessY 先卸载，x/xCast/yCast 的 UB 预算全部解放，批处理天然可行）

### 5.3 建议实施顺序

```
1. comb 批处理（直击 54µs 大头, 中等工程量）→ 复测
2. 若 y 段 14.7µs 成为新大头 → ProcessY 卸载（方案 B 存活部分）
3. 方案 A（列拆分两遍扫描）独立可并行推进, 但收益（3-5µs）小于 1/2, 优先级最低
```

---

## 6. 落地硬约束（三项都要过）

1. **同步链**：从 1 条 SyncAll 变 3-4 条 flag（comb-ready / prepost-ready / A-built / y-done）；flagId 8/9 已被 Part1 占用，同一 flagId 计数上限 15 次（CANN 8.5.0 约束）——改完必跑 ascendc-sync-audit（Set/Wait 配对、方向、flagId 冲突）
2. **验证协议**：争用类收益必须 3 次运行取样（仿真器跨核事件次序非确定，±3-5µs 双峰分布——opt2D 的教训）；精度 y/post/comb_frag 三项 pass_rate 不降
3. **workspace 重排**：mm 分区改 comb/prepost 两区 + y 的 AIC 直写区；x 的 Nd2Nz 装载布局

---

## 7. 结论速查

| 子方案 | 判定 | 预期 |
|---|---|---|
| A. 列拆分 comb 先行（两遍 K 扫描） | ✅ 可行 | −3~5µs |
| B1. sigmoid 上 AIC | ❌ 非线性红线 | — |
| B2. affine 上 AIC | ❌ 无意义（2048 元素） | — |
| B3. 随路量化做 ×inv_rms | ❌ s322f16 路径不符 + 数据依赖封死 | — |
| B4. ProcessY 卸载（块稀疏 A @ x_stacked） | ✅ 可行（大改造） | −10~12µs（配平红利） |
| C. comb 批处理（未在原方案中, 本分析补充） | ✅ 强烈推荐先做 | −29~34µs（54→20-25） |
| 叠加 opt7（已验证 −1.1µs） | ✅ 独立 | 全部可叠加 |
