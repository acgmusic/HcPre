"""Three-way precision comparison: mhc_pre_sinkhorn sim vs hc_pre sim vs CPU golden.

Inputs (all bs=512, n=4, d=4096, seed 1024):
  - mhcs bin : /root/HcPre/mhcs_output_bs512.bin  (hin bf16, hPost fp32, hRes fp32, 4D flattened)
  - hc sim   : /root/HcPre/golden_refs/sim_bs512.pt (y, post, comb_frag)
  - golden   : recomputed CPU reference (y, post, comb_frag incl. sinkhorn)
"""
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/HcPre/scripts/mhc_pre")
from run_golden import HC_MULT as N, make_inputs  # noqa: E402
import torch.nn.functional as F  # noqa: E402

NORM_EPS = 1e-6
HC_EPS = 1e-6


def hc_pre_cpu(x, phi, alpha, bias):
    x_float = x.float()
    x_flat = x_float.flatten(-2)
    from run_golden import to_hf32
    inv_rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes = F.linear(to_hf32(x_flat), to_hf32(phi)) * inv_rms
    pre, post, res = mixes.split([N, N, N * N], dim=-1)
    res = res.unflatten(-1, (N, N))
    h_pre = torch.sigmoid(pre * alpha[0] + bias[:N]) + HC_EPS
    h_post = 2 * torch.sigmoid(post * alpha[1] + bias[N : 2 * N])
    comb = res * alpha[2] + bias[2 * N :].view(N, N)
    comb = comb.softmax(-1) + HC_EPS
    comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(20 - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + HC_EPS)
        comb = comb / (comb.sum(-2, keepdim=True) + HC_EPS)
    y = (h_pre.unsqueeze(-1) * x_float).sum(dim=-2).to(x.dtype)
    return y, h_post, comb


def report(name, a, b, threshold, required):
    a = a.cpu().float().reshape(-1)
    b = b.cpu().float().reshape(-1)
    d = (a - b).abs()
    mag = torch.maximum(a.abs(), b.abs())
    rel = d / mag.clamp_min(torch.finfo(torch.float32).tiny)
    ok = (d <= threshold) | (rel <= threshold)
    rate = ok.float().mean().item()
    status = "PASS" if rate >= required else "FAIL"
    print(f"[cmp] {name}: {status} pass_rate={rate:.2%} (req {required:.0%}) "
          f"max_abs={d.max():.3e} mean_abs={d.mean():.3e}")
    return rate >= required


def main():
    bs = 512
    with open("/root/HcPre/mhcs_output_bs512.bin", "rb") as f:
        (code,) = struct.unpack("<q", f.read(8))
        (b0, n, d) = struct.unpack("<qqq", f.read(24))
        assert b0 == 512 and n == N and d == 4096, f"unexpected dims {b0},{n},{d}"
        hin = torch.from_numpy(np.frombuffer(f.read(bs * d * 2), dtype=np.uint16).copy()
                               ).view(torch.bfloat16).reshape(bs, d)
        hpost = torch.from_numpy(np.frombuffer(f.read(bs * n * 4), dtype=np.float32).copy()).reshape(bs, n)
        hres = torch.from_numpy(np.frombuffer(f.read(bs * n * n * 4), dtype=np.float32).copy()
                                ).reshape(bs, n, n)

    hc = torch.load("/root/HcPre/golden_refs/sim_bs512.pt", map_location="cpu")
    x, phi, alpha, bias = make_inputs()
    gy, gpost, gcomb = hc_pre_cpu(x, phi, alpha, bias)

    print("=== mhc_pre_sinkhorn(sim) vs hc_pre(sim) — the replacement claim ===")
    r1 = report("hin vs y      ", hin, hc["y"], 4e-3, 0.98)
    r2 = report("hPost vs post ", hpost, hc["post"], 1e-4, 0.995)
    r3 = report("hRes vs comb_frag", hres, hc["comb_frag"], 1e-4, 0.995)

    print()
    print("=== mhc_pre_sinkhorn(sim) vs CPU golden ===")
    r4 = report("hin vs y*     ", hin, gy, 4e-3, 0.98)
    r5 = report("hPost vs post*", hpost, gpost, 1e-4, 0.995)
    r6 = report("hRes vs comb*", hres, gcomb, 1e-4, 0.995)

    print()
    exact = (hin.cpu().float().reshape(-1) == hc["y"].cpu().float().reshape(-1)).float().mean().item()
    print(f"[cmp] hin vs y bit-exact ratio: {exact:.2%}")
    print(f"[cmp] OVERALL REPLACEMENT: {'VALID' if all([r1, r2, r3, r4, r5, r6]) else 'CHECK ABOVE'}")
    # extra: hRes row-sum property (doubly stochastic)
    print(f"[cmp] hRes row-sum mean={hres.sum(-1).mean():.4f} col-sum mean={hres.sum(-2).mean():.4f} (should ~1)")


if __name__ == "__main__":
    main()
