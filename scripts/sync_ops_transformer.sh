#!/bin/bash
# Replace container ops-transformer with the local Windows copy (plain rm + full copy).
set -euo pipefail

C=${CANN_CONTAINER:-cann_container}
SRC=/mnt/d/proj/HcPre/ops-transformer
DST=/root/cann/ops-transformer

docker start "$C" >/dev/null 2>&1 || true

echo "[sync-ops] removing container copy ..."
docker exec "$C" rm -rf "$DST"

echo "[sync-ops] copying local repo (incl .git, excl build artifacts) ..."
tar -C "$(dirname "$SRC")" -cf - \
    --exclude="ops-transformer/build" \
    --exclude="ops-transformer/output" \
    --exclude="ops-transformer/build_out" \
    "ops-transformer" | docker exec -i "$C" tar -xf - -C /root/cann

docker exec "$C" bash -c "git config --global --add safe.directory $DST 2>/dev/null || true"

echo "[sync-ops] normalizing CRLF -> LF ..."
docker exec "$C" bash -c "grep -rlI \$'\r' $DST --exclude-dir=.git 2>/dev/null | xargs -r sed -i 's/\r\$//' || true"

echo "[sync-ops] verify:"
docker exec "$C" bash -c "
  cd $DST && git log --oneline -1
  grep -n 'AddConfig' mhc/mhc_pre/op_host/mhc_pre_def.cpp
  ls mhc/mhc_pre/op_kernel/
"
