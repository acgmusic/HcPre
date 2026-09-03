"""Standalone CPU golden runner for MhcPreSinkhorn (ops-transformer) — no NPU required.

Mirrors scripts/hc_pre & scripts/mhc_pre golden conventions: fixed seed 1024,
identical input recipe, so all three operators (hc_pre / mhc_pre / mhc_pre_sinkhorn)
are comparable on the same data.

Semantics (from docs/aclnnMhcPreSinkhorn.md + op_kernel/mhc_pre_sinkhorn_m_k_split_core.h):

    invRms  = 1/sqrt(mean(xFlat^2) + normEps)               # per token
    hMix    = xFlat @ phi^T                                  # HF32 matmul
    w       = hMix * invRms
    (pPre, pPost, pRes) = split(w, (n, n, n^2))              # alpha=(3,) mode
    hPre    = sigmoid(pPre * alpha[0] + bias[:n])  + hcEps   # ProcessPre: sigmoid + eps
    hPost   = 2 * sigmoid(pPost * alpha[1] + bias[n:2n])     # ProcessPost: NO eps
    pHRes   = pRes * alpha[2] + bias[2n:]                    # raw fragment
    hIn     = sum_n(hPre_i * x_i)                            # weighted branch sum

    Sinkhorn (numIters=20, hcEps):                           # identical to hc_pre
        norm[0] = softmax(pHRes, dim=-1) + eps
        s       = sum(norm[0], dim=-2, keepdim) + eps;  norm[1] = norm[0] / s
        repeat (numIters-1) times:
            s = sum(norm, dim=-1, keepdim) + eps;  norm = norm / s
            s = sum(norm, dim=-2, keepdim) + eps;  norm = norm / s
        hRes = final norm

Outputs: hIn (x dtype), hPost (fp32), hRes (fp32, doubly stochastic).
Optional needBackward outputs (hPre/hcBeforeNorm/invRms/sumOut/normOut) are NOT
emulated here (forward-only golden).

Usage:
    python3 run_golden.py                        # print summary
    python3 run_golden.py --save out.pt          # save inputs+outputs
    python3 run_golden.py --dump-input x.bin     # dump C++ example input bin
    python3 run_golden.py --result sim.bin|.pt   # compare vs sim dump
Env: MHCS_GOLDEN_SIZE (default 512)
"""

import argparse
import os
import struct
import sys

import torch
import torch.nn.functional as F

HC_MULT = 4
HIDDEN_SIZE = 4096
MIX_HC = 24
NUM_ITERS = 20
NORM_EPS = 1e-6
HC_EPS = 1e-6
HF32_MANTISSA_BITS = 10
FP32_MANTISSA_BITS = 23
Y_DIFF_THRESHOLD = 4e-3
Y_REQUIRED_PASS_RATE = 0.98
AUX_DIFF_THRESHOLD = 1e-4
AUX_REQUIRED_PASS_RATE = 0.995

X_SHAPE = (1, int(os.environ.get("MHCS_GOLDEN_SIZE", "512")), HC_MULT, HIDDEN_SIZE)


def make_inputs():
    """Same recipe as hc_pre/mhc_pre goldens (seed 1024)."""
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


def sinkhorn(res, eps, num_iters):
    """Softmax + alternating row/col normalization (kernel-exact order)."""
    norm = res.softmax(-1) + eps
    norm = norm / (norm.sum(-2, keepdim=True) + eps)
    for _ in range(num_iters - 1):
        norm = norm / (norm.sum(-1, keepdim=True) + eps)
        norm = norm / (norm.sum(-2, keepdim=True) + eps)
    return norm


def mhc_pre_sinkhorn_cpu(x, phi, alpha, bias):
    n = HC_MULT
    x_float = x.float()
    x_flat = x_float.flatten(-2)
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(to_hf32(x_flat), to_hf32(phi)) * inv_rms
    p_pre, p_post, p_res = mixes.split([n, n, n * n], dim=-1)
    p_res = p_res.unflatten(-1, (n, n))

    h_pre = torch.sigmoid(p_pre * alpha[0] + bias[:n]) + HC_EPS
    h_post = 2 * torch.sigmoid(p_post * alpha[1] + bias[n : 2 * n])
    ph_res = p_res * alpha[2] + bias[2 * n :].view(n, n)
    h_res = sinkhorn(ph_res, HC_EPS, NUM_ITERS)
    h_in = (h_pre.unsqueeze(-1) * x_float).sum(dim=-2).to(x.dtype)
    return h_in, h_post, h_res, inv_rms


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
                        help="sim dump: .bin (C++ example) or .pt")
    parser.add_argument("--save", default=None)
    parser.add_argument("--dump-input", default=None)
    args = parser.parse_args()

    x, phi, alpha, bias = make_inputs()
    print(f"[golden] inputs: x={tuple(x.shape)} bf16, phi={tuple(phi.shape)} fp32, "
          f"alpha={tuple(alpha.shape)}, bias={tuple(bias.shape)}")

    if args.dump_input:
        bs = x.shape[0] * x.shape[1]
        n, d = HC_MULT, HIDDEN_SIZE
        x3 = x.reshape(bs, n, d)
        with open(args.dump_input, "wb") as f:
            f.write(struct.pack("<qqq", bs, n, d))
            f.write(x3.view(torch.uint16).numpy().tobytes())
            f.write(phi.contiguous().numpy().tobytes())
            f.write(alpha.contiguous().numpy().tobytes())
            f.write(bias.contiguous().numpy().tobytes())
        print(f"[golden] input .bin dumped to {args.dump_input}")

    h_in, h_post, h_res, inv_rms = mhc_pre_sinkhorn_cpu(x, phi, alpha, bias)
    print(f"[golden] outputs: hIn={tuple(h_in.shape)} {h_in.dtype}, hPost={tuple(h_post.shape)} fp32, "
          f"hRes={tuple(h_res.shape)} fp32")
    print(f"[golden] stats: hIn[{h_in.float().min():.4f},{h_in.float().max():.4f}] "
          f"hPost[{h_post.min():.4f},{h_post.max():.4f}] "
          f"hRes[{h_res.min():.4f},{h_res.max():.4f}] "
          f"(row-sum={h_res.sum(-1).mean():.4f}, col-sum={h_res.sum(-2).mean():.4f})")

    if args.save:
        torch.save({"x": x, "phi": phi, "alpha": alpha, "bias": bias,
                    "hIn": h_in, "hPost": h_post, "hRes": h_res, "invRms": inv_rms}, args.save)
        print(f"[golden] saved to {args.save}")

    if args.result:
        if not os.path.exists(args.result):
            print(f"[golden] result file not found: {args.result}")
            return 1
        if args.result.endswith(".bin"):
            import numpy as np
            with open(args.result, "rb") as f:
                (code,) = struct.unpack("<q", f.read(8))
                (bs, n, d) = struct.unpack("<qqq", f.read(24))
                hin_s = torch.from_numpy(
                    np.frombuffer(f.read(bs * d * 2), dtype=np.uint16).copy()
                ).view(torch.bfloat16).reshape(bs, d)
                hp_s = torch.from_numpy(
                    np.frombuffer(f.read(bs * n * 4), dtype=np.float32).copy()
                ).reshape(bs, n)
                hr_s = torch.from_numpy(
                    np.frombuffer(f.read(bs * n * n * 4), dtype=np.float32).copy()
                ).reshape(bs, n, n)
            r = {"hIn": hin_s, "hPost": hp_s, "hRes": hr_s}
        else:
            r = torch.load(args.result, map_location="cpu")
        ok = [assert_close(r["hIn"], h_in, "hIn", Y_DIFF_THRESHOLD, Y_REQUIRED_PASS_RATE),
              assert_close(r["hPost"], h_post, "hPost", AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
              assert_close(r["hRes"], h_res, "hRes", AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE)]
        print(f"[golden] overall: {'PASS' if all(ok) else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
