"""Standalone golden runner for HcPre (CPU only, no NPU required).

Computes the CPU reference for the configured case and (optionally) compares
against a result .pt file dumped by hcpre_sim_app.py.

Usage (inside the cann container, python3):
    # 1) print golden outputs summary only
    python3 /root/HcPre/scripts/run_golden.py

    2) compare golden vs simulation result dumped from hcpre_sim_app.py
    python3 /root/HcPre/scripts/run_golden.py --result /root/HcPre/sim_result.pt

Set HC_PRE_COMPARE=0 in build_hcpre.sh sim *and* uncomment the dump block in
hcpre_sim_app.py to produce sim_result.pt, then use mode 2.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

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

X_SHAPE = (1, int(os.environ.get("HC_GOLDEN_SIZE", "512")), HC_MULT, HIDDEN_SIZE)  # keep in sync with hcpre_sim_app.py


def make_inputs():
    torch.manual_seed(1024)  # fixed seed: keep in sync with hcpre_sim_app.py
    fan_in = HC_MULT * HIDDEN_SIZE
    x = (torch.rand(X_SHAPE, dtype=torch.float32) * 2).to(torch.bfloat16)
    hc_fn = torch.rand(MIX_HC, fan_in, dtype=torch.float32) / fan_in
    hc_scale = torch.rand(3, dtype=torch.float32) * 2
    hc_base = torch.rand(MIX_HC, dtype=torch.float32) * 2
    return x, hc_fn, hc_scale, hc_base


def to_hf32(t):
    dropped = FP32_MANTISSA_BITS - HF32_MANTISSA_BITS
    mask = ~((1 << dropped) - 1)
    return (t.contiguous().view(torch.int32) & mask).view(torch.float32)


def hc_pre_cpu(x, hc_fn, hc_scale, hc_base):
    x_float = x.float()
    x_flat = x_float.flatten(-2)
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(to_hf32(x_flat), to_hf32(hc_fn)) * inv_rms
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
                        help="optional .pt file with {'y','post','comb_frag'} from simulation")
    parser.add_argument("--save", default=None, help="save golden outputs to this .pt file")
    args = parser.parse_args()

    x, hc_fn, hc_scale, hc_base = make_inputs()
    print(f"[golden] inputs: x={tuple(x.shape)} bf16, hc_fn={tuple(hc_fn.shape)}, "
          f"hc_scale={tuple(hc_scale.shape)}, hc_base={tuple(hc_base.shape)}")

    y, post, comb_frag = hc_pre_cpu(x, hc_fn, hc_scale, hc_base)
    print(f"[golden] outputs: y={tuple(y.shape)} {y.dtype}, post={tuple(post.shape)} {post.dtype}, "
          f"comb_frag={tuple(comb_frag.shape)} {comb_frag.dtype}")
    print(f"[golden] stats: y[{y.float().min():.4f},{y.float().max():.4f}] "
          f"post[{post.min():.4f},{post.max():.4f}] "
          f"comb_frag[{comb_frag.min():.4f},{comb_frag.max():.4f}] "
          f"(comb_frag row-sum≈1: {comb_frag.sum(-1).mean():.4f})")

    if args.save:
        torch.save({"x": x, "hc_fn": hc_fn, "hc_scale": hc_scale, "hc_base": hc_base,
                    "y": y, "post": post, "comb_frag": comb_frag}, args.save)
        print(f"[golden] saved inputs+outputs to {args.save}")

    if args.result:
        if not os.path.exists(args.result):
            print(f"[golden] result file not found: {args.result}")
            return 1
        r = torch.load(args.result, map_location="cpu")
        ok = [assert_close(r["y"], y, "y", Y_DIFF_THRESHOLD, Y_REQUIRED_PASS_RATE),
              assert_close(r["post"], post, "post", AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE),
              assert_close(r["comb_frag"], comb_frag, "comb_frag",
                           AUX_DIFF_THRESHOLD, AUX_REQUIRED_PASS_RATE)]
        print(f"[golden] overall: {'PASS' if all(ok) else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
