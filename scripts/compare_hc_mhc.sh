#!/bin/bash
# ============================================================================
# [临时脚本] hc_pre vs mhc_pre_sinkhorn 同 shape 输出精度交叉对比
#
# 两个算子的 golden 使用同一输入配方 (seed 1024, 依次生成 x/phi|hc_fn/alpha|
# hc_scale/bias|hc_base, 逐行相同), 相同 b/bs 时两侧输入 bit 级一致, 因此两组
# NPU 输出可直接对比:
#   hc_pre  : y(b,bs,4096) bf16   post(b,bs,4) fp32   comb_frag(b,bs,4,4) fp32
#   mhc_pre : hIn(totalBs,4096) bf16, hPost(totalBs,4) fp32, hRes(totalBs,4,4) fp32
# 映射: y<->hIn, post<->hPost, comb_frag<->hRes; 阈值沿用各自 golden 标准
#   (hIn/y: 4e-3 & 98%, hPost/hRes: 1e-4 & 99.5%)
#
# 流程:
#   1. HC_PRE_DUMP=<WS>/hc_pre_out_compare.pt bash hc_pre/run.sh board b bs
#   2. bash mhc_pre_sinkhorn/run.sh board b bs   (example 落盘 mhcs_output_run.bin)
#   3. 容器/本地内 python 交叉对比, 输出逐输出 pass_rate / max_abs_diff
#
# 用法:
#   bash compare_hc_mhc.sh [b] [bs]     # 默认 b=1 bs=512
#
# 产物 (留在工作区根目录):
#   hc_pre_out_compare.pt               # hc_pre NPU 输出 (torch.save: y/post/comb_frag)
#   mhcs_output_run.bin                 # mhc_pre_sinkhorn NPU 输出 (C++ example)
#   golden_refs/mhcs_golden_b{B}_bs{BS}.pt
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
WS="$(cd "$SCRIPT_DIR/.." && pwd)"

detect_mode() {
  if [ -n "${CONTAINER:-}" ]; then echo "${CONTAINER}"; return; fi
  command -v docker >/dev/null 2>&1 || { echo none; return; }
  local def=${CANN_CONTAINER:-cann_container}
  docker inspect "$def" >/dev/null 2>&1 || { echo none; return; }
  echo "$def"
}
C=$(detect_mode)
lws() { [ "$C" = "none" ] && echo "$WS" || echo /root/HcPre; }
xec() {
  if [ "$C" = "none" ]; then bash -s; else docker exec -i "$C" bash -s; fi
}

B=${1:-1}
BS=${2:-512}
LWS=$(lws)
HC_DUMP=$LWS/hc_pre_out_compare.pt
MHC_BIN=$LWS/mhcs_output_run.bin

step() { echo -e "\n===== [xcmp] $* ====="; }

step "0/3 clean stale dumps [mode=$C]"
xec <<EOS
rm -f $HC_DUMP $MHC_BIN
EOS

step "1/3 hc_pre board (b=$B, bs=$BS, dump=$HC_DUMP)"
HC_PRE_DUMP=$HC_DUMP bash "$SCRIPT_DIR/hc_pre/run.sh" board "$B" "$BS"

step "2/3 mhc_pre_sinkhorn board (b=$B, bs=$BS, bin=$MHC_BIN)"
bash "$SCRIPT_DIR/mhc_pre_sinkhorn/run.sh" board "$B" "$BS"

step "3/3 cross-op output diff"
xec <<EOS
set -e
ls -la $HC_DUMP $MHC_BIN
python3 - <<'PYEOF'
import numpy as np
import torch

HC_PT = "$HC_DUMP"
MHC_BIN = "$MHC_BIN"

Y_THR, Y_REQ = 4e-3, 0.98
AUX_THR, AUX_REQ = 1e-4, 0.995


def load_mhc_bin(path):
    with open(path, "rb") as f:
        raw = f.read()
    code, totalBs, n, d = np.frombuffer(raw[:32], dtype=np.int64)
    if code != 0:
        raise RuntimeError(f"bad output bin code={code}")
    off = 32
    hin = torch.from_numpy(
        np.frombuffer(raw[off:off + totalBs * d * 2], dtype=np.uint16).copy()
    ).view(torch.bfloat16).float().reshape(totalBs, d)
    off += totalBs * d * 2
    hpost = torch.from_numpy(
        np.frombuffer(raw[off:off + totalBs * n * 4], dtype=np.float32).copy()
    ).reshape(totalBs, n)
    off += totalBs * n * 4
    hres = torch.from_numpy(
        np.frombuffer(raw[off:off + totalBs * n * n * 4], dtype=np.float32).copy()
    ).reshape(totalBs, n, n)
    return hin, hpost, hres


def diff_stats(a, b, name, thr, req):
    if a.shape != b.shape:
        raise RuntimeError(f"{name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    abs_diff = (a - b).abs()
    magnitude = torch.maximum(a.abs(), b.abs())
    close = (abs_diff <= thr) | (
        abs_diff / magnitude.clamp_min(torch.finfo(torch.float32).tiny) <= thr
    )
    pass_rate = close.float().mean().item()
    status = "PASS" if pass_rate >= req else "FAIL"
    print(f"[xcmp] {name:<12}: {status} pass_rate={pass_rate:.2%} (thr={thr:g}, req {req:.2%}) "
          f"max_abs_diff={abs_diff.max().item():.3e} mean_abs_diff={abs_diff.mean().item():.3e} "
          f"nan_mhc={torch.isnan(a).sum().item()} nan_hc={torch.isnan(b).sum().item()}")
    return status == "PASS"


hc = torch.load(HC_PT, map_location="cpu")
y = hc["y"].float().reshape(-1, hc["y"].shape[-1])
post = hc["post"].float().reshape(-1, hc["post"].shape[-1])
comb = hc["comb_frag"].float().reshape(-1, hc["comb_frag"].shape[-2], hc["comb_frag"].shape[-1])
print(f"[xcmp] hc_pre outputs loaded: y={tuple(y.shape)}, post={tuple(post.shape)}, comb={tuple(comb.shape)}")

hin, hpost, hres = load_mhc_bin(MHC_BIN)
print(f"[xcmp] mhc outputs loaded: hIn={tuple(hin.shape)}, hPost={tuple(hpost.shape)}, hRes={tuple(hres.shape)}")

ok = [
    diff_stats(hin, y, "hIn~y", Y_THR, Y_REQ),
    diff_stats(hpost, post, "hPost~post", AUX_THR, AUX_REQ),
    diff_stats(hres, comb, "hRes~comb", AUX_THR, AUX_REQ),
]
print(f"[xcmp] overall: {'PASS' if all(ok) else 'FAIL'}")
PYEOF
EOS
step "done (dumps kept: $HC_DUMP, $MHC_BIN)"
