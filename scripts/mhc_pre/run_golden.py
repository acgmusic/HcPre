"""Standalone CPU golden runner for MhcPre (ops-transformer) — no NPU required.

Mirrors scripts/hc_pre/run_golden.py conventions (fixed seed=1024, same input recipe)
so the two operators can be compared on identical data.

MhcPre semantics (from op_host/op_api + A3 kernel mhc_pre_m_split_core.h):
    mixes = RMSnorm(x) @ phi^T           # phi: (matN, matK), matK = N*D; HF32 matmul
    pre, post, res = split(mixes)        # N / N / N*N
    h_pre  = sigmoid(pre * alpha[0] + bias[:N]) + hc_eps          # == hc_pre "pre"
    h_post = 2 * sigmoid(post * alpha[1] + bias[N:2N])            # == hc_pre "post"
    h_res  = res * inv_rms? NO: res(alpha-scaled, biased) RAW fragment, no softmax/sinkhorn
             = (res * alpha[2] + bias[2N:])                        # == hc_pre pre-sinkhorn comb_frag
    hin    = sum_N(h_pre * x)                                       # == hc_pre "y"
    NOTE: h_res here uses raw mm_res (NOT rms-normalized), matching hcBeforeNorm? ->
    kernel reads hcBeforeNormGm (pre-normalization mm output), then multiplies inv_rms
    (MulABLastDimBrcInline with squareSum) -> so h_res = res_norm * alpha[2] + bias.

Outputs: hin (x dtype), h_post (fp32), h_res (fp32) + optional debug outs.

Usage:
    python3 run_golden.py                       # print summary
    python3 run_golden.py --save out.pt         # save inputs+outputs
    python3 run_golden.py --result sim.pt       # compare vs sim dump
Env: MHC_GOLDEN_SIZE (default 512)
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

HC_MULT = 4            # N: hyper-connection flows (same as hc_pre case)
HIDDEN_SIZE = 4096     # D
MIX_HC = 24            # matN = 4 + 4 + 4*4
NORM_EPS = 1e-6
HC_EPS = 1e-6
HF32_MANTISSA_BITS = 10
FP32_MANTISSA_BITS = 23
Y_DIFF_THRESHOLD = 4e-3
Y_REQUIRED_PASS_RATE = 0.98
AUX_DIFF_THRESHOLD = 1e-4
AUX_REQUIRED_PASS_RATE = 0.995

X_SHAPE = (1, int(os.environ.get("MHC_GOLDEN_SIZE", "512")), HC_MULT, HIDDEN_SIZE)


def make_inputs():
    """Identical recipe to hc_pre/run_golden.py (seed 1024, same tensors):
    hc_fn -> phi (transposed layout check below), hc_scale -> alpha, hc_base -> bias."""
    torch.manual_seed(1024)
    fan_in = HC_MULT * HIDDEN_SIZE
    x = (torch.rand(X_SHAPE, dtype=torch.float32) * 2).to(torch.bfloat16)
    phi = torch.rand(MIX_HC, fan_in, dtype=torch.float32) / fan_in
    alpha = torch.rand(3, dtype=torch.float32) * 2
    bias = torch.rand(MIX_HC, dtype=torch.float32) * 2
    return x, phi, alpha, bias


def to_hf32(t):
    dropped = FP32_MANTISSA_BITS - HF32_MANTISSA_BITS
    mask = ~((1 << dropped) - 1)
    return (t.contiguous().view(torch.int32) & mask).view(torch.float32)


def mhc_pre_cpu(x, phi, alpha, bias, gamma=None):
    x_float = x.float()
    x_flat = x_float.flatten(-2)
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(to_hf32(x_flat), to_hf32(phi)) * inv_rms
    pre, post, res = mixes.split([HC_MULT, HC_MULT, HC_MULT * HC_MULT], dim=-1)
    res = res.unflatten(-1, (HC_MULT, HC_MULT))

    h_pre = torch.sigmoid(pre * alpha[0] + bias[:HC_MULT]) + HC_EPS
    h_post = 2 * torch.sigmoid(post * alpha[1] + bias[HC_MULT : 2 * HC_MULT])
    # h_res: rms-normalized res, scaled+bias — NO softmax / sinkhorn (stops here)
    h_res = res * alpha[2] + bias[2 * HC_MULT :].view(HC_MULT, HC_MULT)

    hin = (h_pre.unsqueeze(-1) * x_float).sum(dim=-2).to(x.dtype)
    return hin, h_post, h_res, inv_rms


def assert_close(actual, expected, name, diff_threshold, required_pass_rate):
    actual = actual.cpu().float()
    expected = expected.cpu().float()
    abs_diff = (actual - expected).abs()
    magnitude = torch.maximum(actual.abs(), expected.abs())
    close = (abs_diff <= diff_threshold) | (
        abs_diff / magnitude.clamp_min(torch.finfo(torch.float32).tiny) <= diff_threshold
    )
    pass_rate = close.float().mean().item()
    status = "PASS" if pass_rate >= required_pass_rate else "FAIL"
    print(f"[golden] {name}: {status} pass_rate={pass_rate:.2%} "
          f"(required {required_pass_rate:.2%}), max_abs_diff={abs_diff.max().item():.3e}")
    return status == "PASS"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default=None,
                        help="optional .pt file with {'hin','h_post','h_res'} from simulation")
    parser.add_argument("--save", default=None)
    parser.add_argument("--hc-compat", action="store_true",
                        help="also compute hc_pre comb_frag (with sinkhorn) for cross-op comparison")
    parser.add_argument("--dump-input", default=None,
                        help="dump raw input .bin for the C++ example (bs,n,d,x_bf16,phi,alpha,bias)")
    args = parser.parse_args()

    x, phi, alpha, bias = make_inputs()
    print(f"[golden] inputs: x={tuple(x.shape)} bf16, phi={tuple(phi.shape)} fp32, "
          f"alpha={tuple(alpha.shape)}, bias={tuple(bias.shape)}")

    if args.dump_input:
        import struct
        bs = x.shape[0] * x.shape[1]
        n = HC_MULT
        d = HIDDEN_SIZE
        x3 = x.reshape(bs, n, d)  # C++ example uses 3D (bs, n, d)
        with open(args.dump_input, "wb") as f:
            f.write(struct.pack("<qqq", bs, n, d))
            f.write(x3.view(torch.uint16).numpy().tobytes())
            f.write(phi.contiguous().numpy().tobytes())
            f.write(alpha.contiguous().numpy().tobytes())
            f.write(bias.contiguous().numpy().tobytes())
        print(f"[golden] input .bin dumped to {args.dump_input}")

    hin, h_post, h_res, inv_rms = mhc_pre_cpu(x, phi, alpha, bias)
    print(f"[golden] outputs: hin={tuple(hin.shape)} {hin.dtype}, h_post={tuple(h_post.shape)} fp32, "
          f"h_res={tuple(h_res.shape)} fp32")
    print(f"[golden] stats: hin[{hin.float().min():.4f},{hin.float().max():.4f}] "
          f"h_post[{h_post.min():.4f},{h_post.max():.4f}] "
          f"h_res[{h_res.min():.4f},{h_res.max():.4f}]")

    saved = {"x": x, "phi": phi, "alpha": alpha, "bias": bias,
             "hin": hin, "h_post": h_post, "h_res": h_res, "inv_rms": inv_rms}
    if args.hc_compat:
        hc_eps = HC_EPS
        comb = h_res.softmax(-1) + hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
        for _ in range(20 - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
        saved["comb_frag"] = comb
        print(f"[golden] hc-compat comb_frag (post-sinkhorn) "
              f"[{comb.min():.4f},{comb.max():.4f}] row-sum={comb.sum(-1).mean():.4f}")

    if args.save:
        torch.save(saved, args.save)
        print(f"[golden] saved to {args.save}")

    if args.result:
        if not os.path.exists(args.result):
            print(f"[golden] result file not found: {args.result}")
            return 1
        # result may be raw .bin from the C++ example or .pt
        if args.result.endswith(".bin"):
            import struct
            with open(args.result, "rb") as f:
                (code,) = struct.unpack("<q", f.read(8))
                (bs, n, d) = struct.unpack("<qqq", f.read(24))
                hin = torch.from_numpy(
                    __import__("numpy").frombuffer(f.read(bs * d * 2), dtype=__import__("numpy").uint16)
                ).view(torch.bfloat16).reshape(bs, d)
                hp = torch.from_numpy(
                    __import__("numpy").frombuffer(f.read(bs * n * 4), dtype=__import__("numpy").float32)
                ).reshape(bs, n)
                hr = torch.from_numpy(
                    __import__("numpy").frombuffer(f.read(bs * n * n * 4), dtype=__import__("numpy").float32)
                ).reshape(bs, n, n)
            r = {"hin": hin, "h_post": hp, "h_res": hr}
        else:
            r = torch.load(args.result, map_location="cpu")
        ok = [assert_close(r["hin"], hin, "hin", Y_DIFF_THRESHOLD, Y_REQUIRED_PASS_RATE),
              assert_close(r["h_post"], h_post, "h_post", AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
              assert_close(r["h_res"], h_res, "h_res", AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE)]
        print(f"[golden] overall: {'PASS' if all(ok) else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
