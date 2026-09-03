/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ACLDVPP_AUTO_CONTRAST_H
#define ACLDVPP_AUTO_CONTRAST_H

#include "acldvpp_base.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* @brief acldvppAutoContrast 的第一段接口，根据具体的计算流程，计算workspace大小。
* @param [in] self: npu device侧的aclTensor，仅支持连续的Tensor，数据类型支持 UINT8 和 FLOAT，
*                   数据格式支持NCHW、NHWC，C轴支持1和3。当数据类型为FLOAT时，期望其值的范围为[0, 1]。
* @param [in] cutoff: aclFloatArray类型，长度为2。输入图像直方图中需要剔除的最暗和最亮像素的百分比。
*                     该值必须在 [0.0, 50.0) 范围内。如果传入空指针，则两个百分比都设置为默认值：0.0。
* @param [in] ignore: aclIntArray类型，长度<=256，要忽略的背景像素值，忽略值必须在 [0, 255] 范围内。
*                     如果传入空指针，则默认不忽略像素值。
* @param [in] out: npu device侧的aclTensor，仅支持连续的Tensor，数据类型支持 UINT8 和 FLOAT，
*                  数据格式支持NCHW、NHWC，且数据格式、数据类型、shape需要与self一致。
* @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
* @param [out] executor: 返回op执行器，包含了算子计算流程。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppAutoContrastGetWorkspaceSize(const aclTensor *self, const aclFloatArray *cutoff,
    const aclIntArray *ignore, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
* @brief acldvppAutoContrast 的第二段接口，用于执行计算。
* @param [in] workspace: 在npu device侧申请的workspace内存地址。
* @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口 acldvppAutoContrastGetWorkspaceSize 获取。
* @param [in] executor: op执行器，包含了算子计算流程。
* @param [in] stream: acl stream流。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppAutoContrast(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLDVPP_AUTO_CONTRAST_H