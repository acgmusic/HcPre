#!/bin/bash
# Sync the local HcPre workspace (D:\proj\HcPre) into the docker cann container.
# Local Windows files are the single source of truth; container copy is overwritten each sync.
# CRLF -> LF normalization is applied (Windows checkout breaks bash scripts otherwise).
set -euo pipefail

C=${CANN_CONTAINER:-cann_container}
SRC_ROOT=/mnt/d/proj          # Windows D:\proj
WS=HcPre

docker start "$C" >/dev/null 2>&1 || true

echo "[sync] packing $SRC_ROOT/$WS (excl. build artifacts) ..."
tar -C "$SRC_ROOT" -cf - \
    --exclude="$WS/sim_out" \
    --exclude="$WS/logs" \
    --exclude="$WS/vllm-ascend/.deps" \
    --exclude="$WS/vllm-ascend/build-pybind" \
    --exclude="$WS/vllm-ascend/csrc/build" \
    --exclude="$WS/vllm-ascend/csrc/output" \
    --exclude="$WS/vllm-ascend/csrc/build_out" \
    "$WS" | docker exec -i "$C" tar -xf - -C /root

echo "[sync] normalizing CRLF -> LF (text files only) ..."
docker exec "$C" bash -c "grep -rlI \$'\r' /root/$WS --exclude-dir=.git 2>/dev/null | xargs -r sed -i 's/\r\$//' || true"

echo "[sync] done. container copy:"
docker exec "$C" bash -c "du -sh /root/$WS; ls /root/$WS"
