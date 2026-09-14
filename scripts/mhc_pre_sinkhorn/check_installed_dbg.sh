#!/bin/bash
# Verify installed mhc_pre_sinkhorn .o is debug build
set -u
C=${CANN_CONTAINER:-cann_container}
docker exec -i "$C" bash <<'INNER'
OBJDUMP=/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/llvm-objdump
OBJ=$(find /root/HcPre/mhc_pre_install/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/mhc_pre_sinkhorn -name "*.o" | head -1)
echo "obj: $OBJ"
echo "debug_sections: $($OBJDUMP --section-headers "$OBJ" 2>/dev/null | grep -c debug_)"
echo "size: $(stat -c%s "$OBJ")"
INNER
