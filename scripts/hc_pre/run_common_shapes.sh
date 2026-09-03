#!/bin/bash
# Run sims for the common shapes, dump results, then compare all against golden.
# Usage: bash /mnt/d/proj/HcPre/scripts/run_common_shapes.sh [size ...]
set -euo pipefail

SIZES=${@:-2 240 512}
LOG=/tmp/hcpre_common_shapes.log

for s in $SIZES; do
  echo "=== [common] simulating bs=$s ==="
  HC_PRE_SIZE=$s \
  HC_PRE_COMPARE=0 \
  HC_PRE_DUMP=/root/HcPre/golden_refs/sim_bs${s}.pt \
  HC_SIM_TIMEOUT_MIN=120 \
  bash /mnt/d/proj/HcPre/scripts/hc_pre/build_hcpre.sh sim > "$LOG" 2>&1 || { tail -20 "$LOG"; exit 1; }
  grep -E "sim-app|dumped|DONE" "$LOG"
done

echo
echo "=== [common] golden comparison for all archived shapes ==="
for s in $SIZES; do
  echo "--- bs=$s ---"
  docker exec -e HC_GOLDEN_SIZE=$s -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    -e PATH=/usr/local/python3.11.15/bin:/usr/bin:/bin \
    cann_container python3 /root/HcPre/scripts/hc_pre/run_golden.py \
    --result /root/HcPre/golden_refs/sim_bs${s}.pt 2>&1 | grep "\[golden\]"
done

echo
echo "=== [common] archiving: also save pure-golden refs ==="
for s in $SIZES; do
  docker exec -e HC_GOLDEN_SIZE=$s -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    -e PATH=/usr/local/python3.11.15/bin:/usr/bin:/bin \
    cann_container python3 /root/HcPre/scripts/hc_pre/run_golden.py \
    --save /root/HcPre/golden_refs/golden_bs${s}.pt 2>&1 | grep "saved"
done
ls -la /root/HcPre/golden_refs/
