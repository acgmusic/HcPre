#!/bin/bash
# Verify mhc_pre debug build: ini injection -> gen script -> .o debug sections
set -u
C=${CANN_CONTAINER:-cann_container}
docker exec "$C" bash -c '
OBJDUMP=/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/llvm-objdump
REPO=/root/HcPre/ops-transformer

echo "=== [1] custom_opc_options.ini ==="
cat $REPO/build/autogen/custom_opc_options.ini

echo
echo "=== [2] gen script opc flags ==="
for f in $REPO/build/binary/ascend910_93/gen/MhcPre-mhc_pre-*.sh; do
  grep -o "op_debug_config=[a-z_0-9;]*" "$f" | head -1 && break
done

echo
echo "=== [3] kernel .o debug sections ==="
for OBJ in $(find $REPO/build/binary/ascend910_93/bin/mhc_pre -name "*.o" | head -4); do
  n=$("$OBJDUMP" --section-headers "$OBJ" 2>/dev/null | grep -c "\.debug_")
  sz=$(stat -c%s "$OBJ")
  echo "$(basename $OBJ): size=$sz debug_sections=$n"
done

echo
echo "=== [4] source refs in .debug_str ==="
OBJ=$(find $REPO/build/binary/ascend910_93/bin/mhc_pre -name "*.o" | head -1)
strings -a "$OBJ" | grep -E "mhc_pre.*\.(cpp|h)$" | sort -u | head -6
'
