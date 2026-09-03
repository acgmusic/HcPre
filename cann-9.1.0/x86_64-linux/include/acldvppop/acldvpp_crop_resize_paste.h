/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ACLDVPP_CROP_RESIZE_PASTE_H_
#define ACLDVPP_CROP_RESIZE_PASTE_H_

#include "acldvpp_base.h"
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
* @brief acldvppCropResizePaste 的第一段接口，根据具体的计算流程，计算workspace大小。
* @param [in] self: npu device侧的aclTensor，仅支持连续的Tensor，数据类型支持 UINT8 和 FLOAT，
*                   数据格式支持NCHW、NHWC，C轴支持1和3。
* @param [in] top: uint32_t，抠图的上边界位置。
* @param [in] left: uint32_t，抠图的左边界位置。
* @param [in] height: uint32_t，抠图的高度。
* @param [in] width: uint32_t，抠图的宽度。
* @param [in] size: aclIntArray，表示缩放之后的高、宽。
* @param [in] interpolationMode: uint32_t，缩放插值算法
                                 取值与对应缩放插值算法对应关系为：0: LINEAR/BILINEAR，1: NEAREST，2: BICUBIC
* @param [in] topOffset: uint32_t, 贴图到目的图的顶部偏移
* @param [in] leftOffset: uint32_t, 贴图到目的图的左部偏移
* @param [in] out: npu device侧的aclTensor，仅支持连续的Tensor，数据类型支持 UINT8 和 FLOAT，数据格式支持NCHW、NHWC。
*                  且数据格式、数据类型和通道数需要与self保持一致。宽高需与缩放后宽高保持一致。
* @param [out] workspaceSize: 返回用户需要在npu device侧申请的workspace大小。
* @param [out] executor: 返回op执行器，包含了算子计算流程
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppCropResizePasteGetWorkspaceSize(const aclTensor *self, uint32_t top, uint32_t left, uint32_t height,
    uint32_t width, const aclIntArray* size, uint32_t interpolationMode, uint32_t topOffset, uint32_t leftOffset, 
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
* @brief acldvppCropResizePaste 的第二段接口，用于执行计算。
* @param [in] workspace: 在npu device侧申请的workspace内存地址。
* @param [in] workspaceSize: 在npu device侧申请的workspace大小，由第一段接口 acldvppCropResizePasteGetWorkspaceSize 获取。
* @param [in] executor: op执行器，包含了算子计算流程。
* @param [in] stream: acl stream流。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppCropResizePaste(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLDVPP_CROP_RESIZE_PASTE_H_