#!/bin/bash
# Rebuild mhc_pre_sinkhorn in RELEASE mode + rerun sim (to get full profiling products
# incl. visualize_data.bin, which the debug build's msprof post-processing fails to emit).
set -euo pipefail
C=${CANN_CONTAINER:-cann_container}

# 1. sync repo (CMakeLists back to release)
wsl_out=$(bash /mnt/d/proj/HcPre/scripts/sync_repo.sh) || { echo "$wsl_out"; exit 1; }
echo "$wsl_out" | tail -2

# 2. rebuild + install + example
bash /mnt/d/proj/HcPre/scripts/mhc_pre_sinkhorn/build.sh all

# 3. verify release .o (no debug sections, small size)
docker exec "$C" bash <<'INNER'
OBJDUMP=/usr/local/Ascend/cann-9.1.0/tools/bisheng_compiler/bin/llvm-objdump
V=/root/HcPre/mhc_pre_install/vendors/custom_transformer
for OBJ in $(find $V/op_impl/ai_core/tbe/kernel/ascend910_93/mhc_pre_sinkhorn -name "*.o" | head -2); do
  n=$("$OBJDUMP" --section-headers "$OBJ" 2>/dev/null | grep -c "\.debug_")
  echo "$(basename $OBJ): size=$(stat -c%s "$OBJ") debug_sections=$n"
done
INNER

# 4. launch sim in background
bash /mnt/d/proj/HcPre/scripts/mhc_pre_sinkhorn/sim.sh --bg
echo "[release-rerun] background sim launched"
