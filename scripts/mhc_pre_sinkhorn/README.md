# mhc_pre_sinkhorn 工作流脚本

`ops-transformer/mhc/mhc_pre_sinkhorn` 算子的构建/仿真/比对/归档脚本（与 `scripts/hc_pre`、`scripts/mhc_pre` 平级）。

## 常用命令（Windows PowerShell）

```powershell
# 1. golden（CPU，无需 NPU；--dump-input 产 C++ example 输入）
wsl -d Ubuntu-22.04 -- bash -c 'docker exec -e MHCS_GOLDEN_SIZE=512 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e PATH=/usr/local/python3.11.15/bin:/usr/bin:/bin cann_container python3 /root/HcPre/scripts/mhc_pre_sinkhorn/run_golden.py'

# 2. 构建（算子包 + 安装）与编译 example
wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/mhc_pre_sinkhorn/build.sh all

# 3. 仿真（前台）或后台
wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/mhc_pre_sinkhorn/sim.sh
wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/mhc_pre_sinkhorn/sim.sh --bg

# 4. 后台仿真状态轮询
wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/mhc_pre_sinkhorn/poll_sim.sh

# 5. 精度比对（仿真输出 vs CPU golden）
wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/mhc_pre_sinkhorn/compare.sh

# 6. 结果拷回 Windows sim_out/
wsl -d Ubuntu-22.04 -- bash /mnt/d/proj/HcPre/scripts/mhc_pre_sinkhorn/fetch.sh
```

## 脚本清单

| 脚本 | 用途 |
|---|---|
| `run_golden.py` | **CPU golden**：RMSnorm → HF32 matmul → split → sigmoid → Sinkhorn(numIters) 全流程；`--save/--dump-input/--result`(.bin/.pt)；seed=1024 与 hc_pre/mhc_pre golden 同数据 |
| `build.sh` | 算子包构建安装（build）+ hcshape example 编译（example） |
| `sim.sh` | msprof A3 仿真，前台/后台（`--bg`，日志 `/root/HcPre/logs/mhcs_sim.log`） |
| `compare.sh` | 仿真输出 vs golden 三路比对（hIn/hPost/hRes） |
| `fetch.sh` | 仿真结果拷回 `D:\proj\HcPre\sim_out\` |
| `poll_sim.sh` / `check_sim_status.sh` | 后台仿真状态检查 |
| `launch_sim_bg.sh` | 后台启动的底层实现（写容器内 runner 再 detach） |

## golden 数值语义（与 kernel 逐条对应）

```
invRms   = rsqrt(mean(xFlat²)+normEps)                    # m_k_split_core.h:210-222（kernel 用 1/sqrt 除法实现）
w        = (xFlat @ phiᵀ)_hf32 * invRms
hPre     = sigmoid(pPre·α₀+b₀) + hcEps                    # ProcessPre：sigmoid 后加 eps
hPost    = 2·sigmoid(pPost·α₁+b₁)                          # ProcessPost：不加 eps
hIn      = Σ_n(hPre_i · x_i)                               # ProcessY
hRes     = Sinkhorn(pRes·α₂+b₂, hcEps, numIters=20)       # softmax+eps → 列归一 → 19×[行归一、列归一]
```

## 注意事项

- **A3 上 x 必须 4D `(1,bs,n,d)`**：api 层拒绝 3D（TND 仅 950），与 mhc_pre 不同
- **optional 输出（hPre/hcBeforeNorm/invRms/sumOut/normOut）必须传齐**，传 nullptr 会段错误（同 mhc_pre 的坑）；`needBackward=false` 时不写这些输出但 tensor 仍需创建
- `num_iters` 仅支持 20（api 校验 `NUM_ITERS_DEFAULT_VALUE`）
- debug 编译：op_host/CMakeLists.txt 的 OPTIONS 已含 `--op_debug_config=ccec_g`
- debug kernel 的 msprof 后处理（dump→CSV 解析）可能静默中断，trace.json 仍完整；如需 CSV/可视化数据建议用 release 编译跑
