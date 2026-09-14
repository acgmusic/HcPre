"""mhc_pre simulation application (torch path), mirroring hc_pre/hcpre_sim_app.py.

Loads the mhc_pre_torch pybind module, builds the SAME inputs as hc_pre's
size=512 case (seed 1024), runs aclnnMhcPre via torch_npu, dumps outputs to
/root/HcPre/mhc_output_bs512.bin (same binary layout the C++ example writes).
"""

import faulthandler
import os
import struct

faulthandler.enable()
if os.environ.get("MHC_PRE_FAULTLOG"):
    f = open(os.environ["MHC_PRE_FAULTLOG"], "w")
    faulthandler.enable(f)

import torch
import torch_npu  # noqa: F401

torch_npu.npu.config.allow_internal_format = True

HC_MULT = 4
HIDDEN_SIZE = 4096
MIX_HC = 24
X_SHAPE = (1, 512, HC_MULT, HIDDEN_SIZE)  # same as hc_pre size=512 case
OUT_BIN = "/root/HcPre/mhc_output_bs512.bin"


def main():
    torch.manual_seed(1024)
    fan_in = HC_MULT * HIDDEN_SIZE
    x = (torch.rand(X_SHAPE, dtype=torch.float32) * 2).to(torch.bfloat16)
    phi = torch.rand(MIX_HC, fan_in, dtype=torch.float32) / fan_in
    alpha = torch.rand(3, dtype=torch.float32) * 2
    bias = torch.rand(MIX_HC, dtype=torch.float32) * 2

    import mhc_pre_torch
    hin, h_post, h_res = mhc_pre_torch.npu_mhc_pre(
        x.npu().reshape(512, 4, 4096), phi.npu(), alpha.npu(), bias.npu(), 1e-6, 1e-6
    )
    torch.npu.synchronize()
    print(f"[mhc-sim] hin={tuple(hin.shape)} {hin.dtype}, h_post={tuple(h_post.shape)}, "
          f"h_res={tuple(h_res.shape)}")

    hin_c = hin.cpu()
    hp_c = h_post.cpu()
    hr_c = h_res.cpu()
    bs, n, d = 512, HC_MULT, HIDDEN_SIZE
    with open(OUT_BIN, "wb") as f:
        f.write(struct.pack("<q", 0))
        f.write(struct.pack("<qqq", bs, n, d))
        f.write(hin_c.reshape(bs, d).view(torch.uint16).numpy().tobytes())
        f.write(hp_c.reshape(bs, n).numpy().tobytes())
        f.write(hr_c.reshape(bs, n, n).numpy().tobytes())
    print(f"[mhc-sim] outputs dumped to {OUT_BIN}")
    print("DONE")


if __name__ == "__main__":
    main()
