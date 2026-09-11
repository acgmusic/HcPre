#!/bin/bash
# .o 指纹对比: simopt10(刚建好) vs opt8
set -e
docker start cann_container >/dev/null
OBJ=/root/HcPre/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/hc_pre/HcPre_c12a9d608e5abdbd1728e849208025ff_relocatable.o
echo "== simopt10 构建的 .o:"
docker exec cann_container bash -c "md5sum $OBJ; stat -c '%s bytes, %y' $OBJ"
echo "== objdump -d 实际输出样例:"
docker exec cann_container bash -c '/usr/local/Ascend/cann-9.1.0/x86_64-linux/ccec_compiler/bin/objdump -d $OBJ 2>/dev/null | head -15 || objdump -d $OBJ 2>/dev/null | head -15'
