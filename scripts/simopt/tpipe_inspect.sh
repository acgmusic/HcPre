#!/bin/bash
# 公共 Cast API 的 repeat 重载 + UnaryRepeatParams 结构
I=/usr/local/Ascend/cann-9.1.0/x86_64-linux/asc/impl/basic_api/kernel_operator_vec_vconv_intf_impl.h
echo "=== Cast 公共重载 ==="
grep -n "void Cast(" $I | head
echo
echo "=== UnaryRepeatParams ==="
sed -n '25,60p' /usr/local/Ascend/cann-9.1.0/x86_64-linux/asc/include/basic_api/kernel_struct_unary.h
echo
echo "=== Cast 重载体(含 repeat 的) ==="
grep -n -A 12 "const int32_t calCount, const int32_t repeatTimes" $I | head -30
