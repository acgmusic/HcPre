#!/bin/bash
# Verify installed mhc_pre_sinkhorn kernel is the debug build (.o size), then rerun sim
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}
docker exec "$C" bash -c '
OBJDUMP=/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/llvm-objdump
V=/root/HcPre/mhc_pre_install/vendors/custom_transformer
KDIR=$V/op_impl/ai_core/tbe/kernel/ascend910_93/mhc_pre_sinkhorn
ls "$KDIR" 2>/dev/null | head -4
for OBJ in $(find "$KDIR" -name "*.o" 2>/dev/null | head -2); do
  n=$("$OBJDUMP" --section-headers "$OBJ" 2>/dev/null | grep -c "\.debug_")
  sz=$(stat -c%s "$OBJ")
  echo "$(basename $OBJ): size=$sz debug_sections=$n"
done
'
