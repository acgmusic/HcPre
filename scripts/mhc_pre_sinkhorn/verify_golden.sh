#!/bin/bash
# Verify new golden vs archived sim bin (quick, avoids sync_repo slowness)
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}
# copy the single new file directly (faster than full sync)
docker cp /mnt/d/proj/HcPre/scripts/mhc_pre_sinkhorn/run_golden.py "$C":/root/HcPre/scripts/mhc_pre_sinkhorn/run_golden.py 2>/dev/null || \
  docker cp "D:\\proj\\HcPre\\scripts\\mhc_pre_sinkhorn\\run_golden.py" "$C":/root/HcPre/scripts/mhc_pre_sinkhorn/run_golden.py
docker exec -e MHCS_GOLDEN_SIZE=512 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  -e PATH=/usr/local/python3.11.15/bin:/usr/bin:/bin \
  "$C" python3 /root/HcPre/scripts/mhc_pre_sinkhorn/run_golden.py \
  --result /root/HcPre/golden_refs/mhcs_sim_bs512.bin 2>&1 | grep golden
