#!/bin/bash
docker exec cann_container bash -c '
CC=/root/HcPre/vllm-ascend/build-pybind/vllm_ascend_kernels_preprocess-prefix/src/vllm_ascend_kernels_preprocess-build/compile_commands.json
python3 - <<"EOF"
import json
with open("/root/HcPre/vllm-ascend/build-pybind/vllm_ascend_kernels_preprocess-prefix/src/vllm_ascend_kernels_preprocess-build/compile_commands.json") as f:
    cmds = json.load(f)
for c in cmds[:3]:
    cmd = c["command"]
    arch = [tok for tok in cmd.split() if "ARCH" in tok.upper() or "soc" in tok.lower() or "aic" in tok.lower() or "cpu" in tok.lower()]
    print(c["file"].split("/")[-1], "::", arch[:6])
EOF
'
