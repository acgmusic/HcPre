"""hc_pre simulation application, run under `msprof op simulator` (A3 / Ascend910_9382).

Case (per requirement, ND format):
  inputs : x(1,2,4,4096) BF16, hc_fn(24,16384) FP32, hc_scale(3,) FP32, hc_base(24,) FP32
  outputs: y(1,2,4096) BF16, post(1,2,4) FP32, comb_frag(1,2,4,4) FP32

Note: hc_pre op def (op_host/hc_pre_def.cpp) accepts x=DT_BF16 only. The FLOAT dtype in
the original case description applies to hc_fn / hc_scale / hc_base; x is created as BF16
so the operator accepts it (same as tests/e2e/.../test_npu_hc_pre.py).

Env:
  HC_PRE_PYBIND_SO   path to vllm_ascend_C.so (required)
  HC_PRE_COMPARE     "1" (default) to compare against CPU golden
  NPU_ID             device index used for on-board runs (default 0)
  HC_PRE_ITERS       number of op launches per run (default 1; perf uses 50)
  HC_PRE_SHAPES      path to a shapes csv (multi-shape perf mode): header
                     "b,bs,d" (d column optional, default 4096), '#'-comments
                     allowed. Overrides HC_PRE_BATCH/HC_PRE_SIZE; launches are
                     round-major: outer loop = ITERS rounds, inner loop = shapes
                     (all inputs pre-moved to device, outside the timed region).
                     Compare/dump are skipped in this mode (perf only).
"""

import os

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

torch_npu.npu.config.allow_internal_format = True

NPU_ID = int((os.environ.get("NPU_ID") or "").strip() or 0)
torch.npu.set_device(NPU_ID)

PERF_ITERS = max(1, int((os.environ.get("HC_PRE_ITERS") or "").strip() or 1))
HC_PRE_SHAPES = (os.environ.get("HC_PRE_SHAPES") or "").strip()

HC_MULT = 4
HIDDEN_SIZE = 4096
MIX_HC = 24
HC_SINKHORN_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6
HF32_MANTISSA_BITS = 10
FP32_MANTISSA_BITS = 23
Y_DIFF_THRESHOLD = 4e-3
Y_REQUIRED_PASS_RATE = 0.98
AUX_DIFF_THRESHOLD = 1e-4
AUX_REQUIRED_PASS_RATE = 0.995

X_SHAPE = (int(os.environ.get("HC_PRE_BATCH", "1")), int(os.environ.get("HC_PRE_SIZE", "512")),
           HC_MULT, HIDDEN_SIZE)  # 4D: (b, bs, hc=4, d=4096); b*bs folds into operator bs dim


def _parse_shapes_csv(path):
    """解析 shapes csv: 跳过 '#' 注释与空行; 表头列名 b, bs|s|seqlen, d|hidden|hidden_size
    (d 列可缺省=4096); 数据行 2 列 => (b,bs,d=4096), 3 列 => (b,bs,d)。返回 [(b,bs,d), ...]"""
    col_alias = {"b": "b", "bs": "bs", "s": "bs", "seqlen": "bs", "seq_len": "bs",
                 "d": "d", "hidden": "d", "hidden_size": "d"}
    shapes = []
    ndim = None
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            toks = [t.strip() for t in s.split(",") if t.strip() != ""]
            if not toks:
                continue
            try:
                vals = [int(t) for t in toks]
            except ValueError:
                low = [t.lower() for t in toks]
                names = [col_alias.get(t) for t in low]
                if "b" in names and "bs" in names:
                    ndim = 3 if "d" in names else 2
                    continue
                raise ValueError(f"[sim-app] bad line in shapes csv: {s!r}")
            if ndim is None:
                ndim = len(vals)
            if len(vals) < 2 or len(vals) > 3:
                raise ValueError(f"[sim-app] bad shape row (need b,bs[,d]): {s!r}")
            b, bs = vals[0], vals[1]
            d = vals[2] if len(vals) >= 3 else HIDDEN_SIZE
            if b < 1 or bs < 1 or d < 1:
                raise ValueError(f"[sim-app] bad shape values: {s!r}")
            shapes.append((b, bs, d))
    if not shapes:
        raise ValueError(f"[sim-app] no shapes found in {path}")
    return shapes


def _make_hc_pre_inputs(b=None, bs=None, d=None):
    torch.manual_seed(1024)  # fixed seed: keeps inputs identical across sim runs and golden
    if b is None:
        x_shape = X_SHAPE
    else:
        x_shape = (b, bs, HC_MULT, d)
    fan_in = HC_MULT * x_shape[-1]
    x = (torch.rand(x_shape, dtype=torch.float32) * 2).to(torch.bfloat16)
    hc_fn = torch.rand(MIX_HC, fan_in, dtype=torch.float32) / fan_in
    hc_scale = torch.rand(3, dtype=torch.float32) * 2
    hc_base = torch.rand(MIX_HC, dtype=torch.float32) * 2
    return x, hc_fn, hc_scale, hc_base


def _to_hf32(tensor: torch.Tensor) -> torch.Tensor:
    dropped_mantissa_bits = FP32_MANTISSA_BITS - HF32_MANTISSA_BITS
    mantissa_mask = ~((1 << dropped_mantissa_bits) - 1)
    bits = tensor.contiguous().view(torch.int32)
    return (bits & mantissa_mask).view(torch.float32)


def _hc_pre_cpu(x, hc_fn, hc_scale, hc_base):
    x_float = x.float()
    x_flat = x_float.flatten(-2)
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(_to_hf32(x_flat), _to_hf32(hc_fn)) * inv_rms
    pre, post, comb_frag = mixes.split([HC_MULT, HC_MULT, HC_MULT * HC_MULT], dim=-1)
    comb_frag = comb_frag.unflatten(-1, (HC_MULT, HC_MULT))
    pre = torch.sigmoid(pre * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post = 2 * torch.sigmoid(post * hc_scale[1] + hc_base[HC_MULT : 2 * HC_MULT])
    comb_frag = comb_frag * hc_scale[2] + hc_base[2 * HC_MULT :].view(HC_MULT, HC_MULT)
    comb_frag = comb_frag.softmax(-1) + HC_EPS
    comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITERS - 1):
        comb_frag = comb_frag / (comb_frag.sum(-1, keepdim=True) + HC_EPS)
        comb_frag = comb_frag / (comb_frag.sum(-2, keepdim=True) + HC_EPS)
    y = (pre.unsqueeze(-1) * x_float).sum(dim=-2).to(x.dtype)
    return y, post, comb_frag


def _assert_close_with_pass_rate(actual, expected, *, name, diff_threshold, required_pass_rate):
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    abs_diff = (actual - expected).abs()
    magnitude = torch.maximum(actual.abs(), expected.abs())
    close = (abs_diff <= diff_threshold) | (
        abs_diff / magnitude.clamp_min(torch.finfo(torch.float32).tiny) <= diff_threshold
    )
    pass_rate = close.float().mean().item()
    max_abs_diff = abs_diff.max().item()
    status = "PASS" if pass_rate >= required_pass_rate else "FAIL"
    print(f"[compare] {name}: {status} pass_rate={pass_rate:.2%} "
          f"(required {required_pass_rate:.2%}), max_abs_diff={max_abs_diff:.3e}")
    return status == "PASS"


def _run_multi_shape_perf():
    """多 shape 上板性能: 所有 shape 在同一次进程(同一次 msprof)内完成。
    发射顺序 round-major: 外层 PERF_ITERS 轮, 内层遍历 shapes —— op_summary
    中第 r*nShapes+s 条记录即 shape s 的第 r 轮。所有输入一次性驻留 device,
    计时区间内只有算子 launch。"""
    shapes = _parse_shapes_csv(HC_PRE_SHAPES)
    print(f"[sim-app] multi-shape perf: {len(shapes)} shapes x {PERF_ITERS} iters "
          f"(round-major: outer={PERF_ITERS} rounds, inner={len(shapes)} shapes)")
    dev_inputs = []
    for (b, bs, d) in shapes:
        x, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs(b, bs, d)
        dev_inputs.append((x.npu(), hc_fn.npu(), hc_scale.npu(), hc_base.npu()))
        print(f"[sim-app]   shape b={b} bs={bs} d={d}: x={tuple(x.shape)}")
    last = None
    for _round in range(PERF_ITERS):
        for (x_d, fn_d, scale_d, base_d) in dev_inputs:
            last = torch.ops._C_ascend.npu_hc_pre_v2(
                x_d, fn_d, scale_d, base_d,
                HC_MULT, HC_SINKHORN_ITERS, NORM_EPS, HC_EPS,
            )
    torch.npu.synchronize()
    print(f"[sim-app] multi-shape perf done: {PERF_ITERS * len(shapes)} launches, "
          f"last y={tuple(last.shape)} {last.dtype}")


def main():
    print(f"[sim-app] using NPU device {NPU_ID}")
    so_path = os.environ["HC_PRE_PYBIND_SO"]
    torch.ops.load_library(so_path)

    if HC_PRE_SHAPES:
        _run_multi_shape_perf()
        print("DONE")
        return

    x, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs()
    print(f"[sim-app] x={tuple(x.shape)} bf16, hc_fn={tuple(hc_fn.shape)} fp32, "
          f"hc_scale={tuple(hc_scale.shape)}, hc_base={tuple(hc_base.shape)}")

    y, post, comb_frag = None, None, None
    for _ in range(PERF_ITERS):
        y, post, comb_frag = torch.ops._C_ascend.npu_hc_pre_v2(
            x.npu(),
            hc_fn.npu(),
            hc_scale.npu(),
            hc_base.npu(),
            HC_MULT,
            HC_SINKHORN_ITERS,
            NORM_EPS,
            HC_EPS,
        )
    torch.npu.synchronize()
    print(f"[sim-app] outputs (iters={PERF_ITERS}): y={tuple(y.shape)} {y.dtype}, "
          f"post={tuple(post.shape)} {post.dtype}, comb_frag={tuple(comb_frag.shape)} {comb_frag.dtype}")

    dump_path = os.environ.get("HC_PRE_DUMP")
    if dump_path:
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        torch.save({"y": y.cpu(), "post": post.cpu(), "comb_frag": comb_frag.cpu()}, dump_path)
        print(f"[sim-app] outputs dumped to {dump_path}")

    if os.environ.get("HC_PRE_COMPARE", "1") == "1":
        expected_y, expected_post, expected_comb_frag = _hc_pre_cpu(x, hc_fn, hc_scale, hc_base)
        ok = [
            _assert_close_with_pass_rate(y, expected_y, name="y",
                                         diff_threshold=Y_DIFF_THRESHOLD,
                                         required_pass_rate=Y_REQUIRED_PASS_RATE),
            _assert_close_with_pass_rate(post, expected_post, name="post",
                                         diff_threshold=AUX_DIFF_THRESHOLD,
                                         required_pass_rate=AUX_REQUIRED_PASS_RATE),
            _assert_close_with_pass_rate(comb_frag, expected_comb_frag, name="comb_frag",
                                         diff_threshold=AUX_DIFF_THRESHOLD,
                                         required_pass_rate=AUX_REQUIRED_PASS_RATE),
        ]
        print(f"[compare] overall: {'PASS' if all(ok) else 'FAIL'}")

    print("DONE")


if __name__ == "__main__":
    main()
