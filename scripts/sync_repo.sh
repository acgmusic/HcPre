#!/bin/bash
# Sync the local HcPre workspace into the docker cann container.
# Only meaningful when docker + the target container are available (WSL host).
# In bare-metal / remote-container scenarios this script is a no-op (docker
# missing or container absent), because the code is already in place.
set -euo pipefail

C=${CANN_CONTAINER:-cann_container}
SRC_ROOT=/mnt/d/proj          # Windows D:\proj (only exists on the WSL host)
WS=HcPre

# --- docker unavailable: no-op with a clear message (code already local) ---
if ! command -v docker >/dev/null 2>&1; then
    echo "[sync] docker not available - skipping sync (code should already be in place)"
    exit 0
fi
# --- container not present: no-op ---
if ! docker inspect "$C" >/dev/null 2>&1; then
    echo "[sync] container '$C' not found - skipping sync"
    exit 0
fi
# --- source tree not mounted (not on the WSL host): no-op ---
if [ ! -d "$SRC_ROOT/$WS" ]; then
    echo "[sync] $SRC_ROOT/$WS not found (not the WSL host) - skipping sync"
    exit 0
fi

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
