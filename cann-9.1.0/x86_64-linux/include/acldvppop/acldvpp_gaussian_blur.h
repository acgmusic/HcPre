/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ACLDVPP_GAUSSIAN_BLUR_H_
#define ACLDVPP_GAUSSIAN_BLUR_H_

#include "acldvpp_base.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* @brief acldvppGaussianBlurGetWorkspaceSize 的第一段接口，根据具体的计算流程，计算workspace大小。
* @param [in] self: npu device侧的aclTensor，仅支持连续的Tensor，数据类型支持 UINT8 和 FLOAT，
*                   数据格式支持NCHW、NHWC，C轴支持1和3。
* @param [in] kernelSize: 要使用的高斯核的大小，大小只支持1，3，5。长度为2的数组，则它必须是表示（宽度、高度）的 2 个值。
* @param [in] sigma: 要使用的高斯核的标准差，该值必须是正数。长度为2的数组，它必须是表示（宽度、高度）的 2 个值。
* @param [in] paddingMode: uint32_t 类型，该变量值与对应填充模式对应关系为 0：CONSTANT 默认填充0， 2: REFLECT。
* @param [in] out: npu device侧的aclTensor，仅支持连续的Tensor，数据类型支持 UINT8 和 FLOAT，
*                  数据格式支持NCHW、NHWC，且数据格式、数据类型、shape需要与self一致。
* @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
* @param [out] executor: 返回op执行器，包含了算子计算流程。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppGaussianBlurGetWorkspaceSize(const aclTensor* self, const aclIntArray* kernelSize,
    const aclFloatArray* sigma, uint32_t paddingMode, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

/**
* @brief acldvppGaussianBlur 的第二段接口，用于执行计算。
* @param [in] workspace: 在npu device侧申请的workspace内存地址。
* @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口 acldvppGaussianBlurGetWorkspaceSize 获取。
* @param [in] executor: op执行器，包含了算子计算流程。
* @param [in] stream: acl stream流。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppGaussianBlur(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
#ifdef __cplusplus
}
#endif

#endif // ACLDVPP_GAUSSIAN_BLUR_H_