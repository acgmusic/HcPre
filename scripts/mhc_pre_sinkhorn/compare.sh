#!/bin/bash
# Compare mhc_pre_sinkhorn sim output vs CPU golden (and optionally vs hc_pre sim).
# Usage: bash compare.sh [sim_output.bin]   (default /root/HcPre/mhcs_output_bs512.bin)
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}
BIN=${1:-/root/HcPre/mhcs_output_bs512.bin}

docker exec -e MHCS_GOLDEN_SIZE=512 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e PATH=/usr/local/python3.11.15/bin:/usr/bin:/bin \
  "$C" python3 /root/HcPre/scripts/mhc_pre_sinkhorn/run_golden.py --result "$BIN" 2>&1 | grep golden
