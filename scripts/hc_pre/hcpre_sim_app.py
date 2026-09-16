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
  HC_PRE_SHAPES      multi-shape mode: a shapes csv path, or "auto[:N][:BMAX]" to
                     generate N random shapes with HC_PRE_VERIFY_SEED (default 1024):
                     b ∈ [1,BMAX] uniform (default 4, HC_PRE_VERIFY_BMAX), s log-uniform
                     in [1,SMAX] (default 8192, HC_PRE_VERIFY_SMAX), d fixed 4096.
                     N defaults to HC_PRE_VERIFY_COUNT (1000). BMAX/SMAX cap the
                     per-shape workload (sim Model RUN TIME ~ linear in b*s*d) so a
                     bounded sweep can finish within a time budget, e.g. in the
                     simulator. csv format: header "b,bs|s[,d|hidden]" ('#'-comments;
                     d default 4096). With HC_PRE_COMPARE=1 this runs the
                     generalization accuracy sweep: each shape is launched once and
                     compared against the CPU golden; the FIRST failure stops the run
                     (exit 1) with a detailed report; set HC_PRE_DUMP to also save
                     the failing shape's inputs/outputs. With HC_PRE_COMPARE=0 it is
                     the perf mode: PERF_ITERS passes over all shapes (iter-major),
                     no compare/dump.
  HC_PRE_VERIFY_COUNT  shapes to generate in "auto" mode (default 1000)
  HC_PRE_VERIFY_SEED   shape-generation seed for "auto" mode (default 1024)
  HC_PRE_VERIFY_BMAX   max b in "auto" mode (default 4)
  HC_PRE_VERIFY_SMAX   max s in "auto" mode (default 8192)
"""

import math
import os
import random
import sys
import time

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

torch_npu.npu.config.allow_internal_format = True

NPU_ID = int((os.environ.get("NPU_ID") or "").strip() or 0)
torch.npu.set_device(NPU_ID)

PERF_ITERS = max(1, int((os.environ.get("HC_PRE_ITERS") or "").strip() or 1))
HC_PRE_SHAPES = (os.environ.get("HC_PRE_SHAPES") or "").strip()
HC_PRE_VERIFY_COUNT = max(1, int((os.environ.get("HC_PRE_VERIFY_COUNT") or "").strip() or 1000))
HC_PRE_VERIFY_SEED = int((os.environ.get("HC_PRE_VERIFY_SEED") or "").strip() or 1024)
HC_PRE_VERIFY_BMAX = max(1, int((os.environ.get("HC_PRE_VERIFY_BMAX") or "").strip() or 4))
HC_PRE_VERIFY_SMAX = max(1, int((os.environ.get("HC_PRE_VERIFY_SMAX") or "").strip() or 8192))

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
    (d 列可缺省=4096); 数据行列数必须与表头一致(无表头时按首行数据定), 拒绝重复 shape。
    返回 [(b,bs,d), ...]"""
    col_alias = {"b": "b", "bs": "bs", "s": "bs", "seqlen": "bs", "seq_len": "bs",
                 "d": "d", "hidden": "d", "hidden_size": "d"}
    shapes = []
    seen = set()
    ndim = None
    with open(path, encoding="utf-8-sig") as f:
        for lineno, raw in enumerate(f, 1):
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
                raise ValueError(f"[sim-app] {path}:{lineno}: bad line: {s!r}")
            if ndim is None:
                ndim = len(vals)
            if len(vals) != ndim:
                raise ValueError(f"{path}:{lineno}: expected {ndim} columns, got {len(vals)}: {s!r}")
            if len(vals) < 2 or len(vals) > 3:
                raise ValueError(f"{path}:{lineno}: bad shape row (need b,bs[,d]): {s!r}")
            b, bs = vals[0], vals[1]
            d = vals[2] if ndim == 3 else HIDDEN_SIZE
            if b < 1 or bs < 1 or d < 1:
                raise ValueError(f"{path}:{lineno}: bad shape values: {s!r}")
            if (b, bs, d) in seen:
                raise ValueError(f"{path}:{lineno}: duplicate shape b={b} bs={bs} d={d}")
            seen.add((b, bs, d))
            shapes.append((b, bs, d))
    if not shapes:
        raise ValueError(f"[sim-app] no shapes found in {path}")
    return shapes


def _gen_shapes(count, seed, b_max=4, s_max=8192):
    """固定种子的随机 shape 生成(泛化精度验证): b ∈ [1,b_max] 均匀, s 对数均匀
    [1,s_max] (等权覆盖每个倍频程, 与手工扫描清单 1,2,4,...,8970 的分布一致),
    d 固定 4096。保证 count 个互不相同的 (b, s, d)。
    b_max/s_max 可由 HC_PRE_VERIFY_BMAX / HC_PRE_VERIFY_SMAX 收紧, 用于把
    仿真工作量压到可承受范围(bs_total = b*s ≤ b_max*s_max, Model RUN TIME
    与 bs_total×d 大致线性)。"""
    if b_max < 1 or s_max < 1:
        raise ValueError(f"[sim-app] b_max/s_max must be >= 1, got {b_max}/{s_max}")
    if b_max * s_max < count:
        raise ValueError(f"[sim-app] cannot draw {count} distinct shapes from "
                         f"b∈[1,{b_max}] x s∈[1,{s_max}] (only {b_max * s_max} combos)")
    s_log2 = math.log2(s_max)
    rng = random.Random(seed)
    seen = set()
    shapes = []
    while len(shapes) < count:
        b = rng.randint(1, b_max)
        s = max(1, min(s_max, int(2 ** rng.uniform(0.0, s_log2))))
        if (b, s) in seen:
            continue
        seen.add((b, s))
        shapes.append((b, s, HIDDEN_SIZE))
    return shapes


def _load_shapes(spec):
    """HC_PRE_SHAPES 取值解析: 'auto[:N][:b_max][:s_max]' -> 随机生成(各段可
    独立省略, 缺省取 HC_PRE_VERIFY_COUNT / HC_PRE_VERIFY_BMAX /
    HC_PRE_VERIFY_SMAX); 其它 -> shapes csv 路径"""
    if spec == "auto" or spec.startswith("auto:"):
        parts = spec.split(":", 2)
        count = int(parts[1]) if len(parts) > 1 and parts[1] else HC_PRE_VERIFY_COUNT
        b_max = int(parts[2]) if len(parts) > 2 and parts[2] else HC_PRE_VERIFY_BMAX
        s_max = HC_PRE_VERIFY_SMAX
        return _gen_shapes(max(1, count), HC_PRE_VERIFY_SEED, max(1, b_max), max(1, s_max))
    return _parse_shapes_csv(spec)


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


def _compare_one(actual, expected, *, name, diff_threshold, required_pass_rate):
    """静默比对: 返回 (ok, pass_rate, max_abs_diff), 不打印"""
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    abs_diff = (actual - expected).abs()
    magnitude = torch.maximum(actual.abs(), expected.abs())
    close = (abs_diff <= diff_threshold) | (
        abs_diff / magnitude.clamp_min(torch.finfo(torch.float32).tiny) <= diff_threshold
    )
    pass_rate = close.float().mean().item()
    max_abs_diff = abs_diff.max().item()
    return pass_rate >= required_pass_rate, pass_rate, max_abs_diff


def _assert_close_with_pass_rate(actual, expected, *, name, diff_threshold, required_pass_rate):
    ok, pass_rate, max_abs_diff = _compare_one(actual, expected, name=name,
                                               diff_threshold=diff_threshold,
                                               required_pass_rate=required_pass_rate)
    status = "PASS" if ok else "FAIL"
    print(f"[compare] {name}: {status} pass_rate={pass_rate:.2%} "
          f"(required {required_pass_rate:.2%}), max_abs_diff={max_abs_diff:.3e}")
    return ok


def _run_multi_shape_perf(shapes):
    """多 shape 上板性能: 每进程跑 PERF_ITERS 遍全部 shape(iter 外层, shape 内层)。
    run.sh 的 rounds = 独立 msprof 进程数(状态轮间清空), 进程内 iter 次遍历由
    HC_PRE_ITERS 控制 —— 每 shape 每轮得到 iters 个进程内采样, 构成
    rounds x iters 时间序列(perf_extreme_eval 的块格式)。
    所有输入一次性驻留 device, 计时区间内只有算子 launch。"""
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


def _run_multi_shape_verify(shapes):
    """泛化精度验证: 逐 shape 独立生成输入(固定种子) -> 上板运行一次 -> 比对
    CPU golden -> 单行结果; 首个 FAIL 立即停止(exit 1)并输出详细报告
    (shape 索引/种子/各项 pass rate/max diff; 若设 HC_PRE_DUMP 同时落盘现场)。
    单 shape 复现: HC_PRE_BATCH=b HC_PRE_SIZE=bs*... (见报告中的提示)。"""
    total = len(shapes)
    print(f"[verify] generalization accuracy sweep: {total} shapes, "
          f"shape_seed={HC_PRE_VERIFY_SEED}, input_seed=1024, "
          f"thresholds y={Y_DIFF_THRESHOLD}/{Y_REQUIRED_PASS_RATE:.0%} "
          f"aux={AUX_DIFF_THRESHOLD}/{AUX_REQUIRED_PASS_RATE:.0%}")
    t_sweep = time.time()
    for idx, (b, bs, d) in enumerate(shapes, 1):
        t0 = time.time()
        x, hc_fn, hc_scale, hc_base = _make_hc_pre_inputs(b, bs, d)
        y, post, comb_frag = torch.ops._C_ascend.npu_hc_pre_v2(
            x.npu(), hc_fn.npu(), hc_scale.npu(), hc_base.npu(),
            HC_MULT, HC_SINKHORN_ITERS, NORM_EPS, HC_EPS,
        )
        expected_y, expected_post, expected_comb_frag = _hc_pre_cpu(x, hc_fn, hc_scale, hc_base)
        checks = [
            ("y", _compare_one(y, expected_y, name="y",
                               diff_threshold=Y_DIFF_THRESHOLD,
                               required_pass_rate=Y_REQUIRED_PASS_RATE)),
            ("post", _compare_one(post, expected_post, name="post",
                                  diff_threshold=AUX_DIFF_THRESHOLD,
                                  required_pass_rate=AUX_REQUIRED_PASS_RATE)),
            ("comb_frag", _compare_one(comb_frag, expected_comb_frag, name="comb_frag",
                                       diff_threshold=AUX_DIFF_THRESHOLD,
                                       required_pass_rate=AUX_REQUIRED_PASS_RATE)),
        ]
        del expected_y, expected_post, expected_comb_frag
        dt = time.time() - t0
        elapsed = time.time() - t_sweep
        eta = elapsed / idx * (total - idx)
        all_ok = all(ok for _, (ok, _, _) in checks)
        rates = " ".join(f"{name}={rate:.2%}" for name, (_, rate, _) in checks)
        print(f"[verify {idx}/{total}] b={b} bs={bs} d={d}: "
              f"{'PASS' if all_ok else 'FAIL'} ({rates}) "
              f"[{dt:.1f}s eta {eta:.0f}s]", flush=True)
        if not all_ok:
            print()
            print("=" * 68)
            print(f"[verify] FAILED at shape {idx}/{total}: b={b} bs={bs} d={d} "
                  f"(folded bs={b * bs})")
            print(f"[verify] shape_seed={HC_PRE_VERIFY_SEED} input_seed=1024 "
                  f"shape_index={idx}")
            print(f"[verify] 单 shape 复现: HC_PRE_BATCH={b} HC_PRE_SIZE={b * bs} "
                  f"bash scripts/hc_pre/run.sh board")
            for name, (ok, rate, mad) in checks:
                status = "PASS" if ok else "FAIL"
                print(f"[verify]   {name}: {status} pass_rate={rate:.2%}, max_abs_diff={mad:.3e}")
            dump_path = os.environ.get("HC_PRE_DUMP")
            if dump_path:
                os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
                torch.save({"shape": (b, bs, d), "x": x, "hc_fn": hc_fn,
                            "hc_scale": hc_scale, "hc_base": hc_base,
                            "y": y.cpu(), "post": post.cpu(),
                            "comb_frag": comb_frag.cpu()}, dump_path)
                print(f"[verify]   failing case dumped to {dump_path}")
            print("=" * 68)
            sys.exit(1)
    print(f"[verify] ALL {total} shapes PASSED "
          f"({time.time() - t_sweep:.0f}s total)")


def main():
    print(f"[sim-app] using NPU device {NPU_ID}")
    so_path = os.environ["HC_PRE_PYBIND_SO"]
    torch.ops.load_library(so_path)

    if HC_PRE_SHAPES:
        shapes = _load_shapes(HC_PRE_SHAPES)
        if os.environ.get("HC_PRE_COMPARE", "1") == "1":
            _run_multi_shape_verify(shapes)
        else:
            _run_multi_shape_perf(shapes)
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
