/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ACLDVPP_BASE_H_
#define ACLDVPP_BASE_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t acldvppStatus;

#define ACLDVPP_SUCCESS 0
#define ACLDVPP_ERR_PARAM_NULLPTR 106001
#define ACLDVPP_ERR_PARAM_INVALID 106002
#define ACLDVPP_ERR_UNINITIALIZE 106101
#define ACLDVPP_ERR_REPEAT_INITIALIZE 106102
#define ACLDVPP_ERR_API_NOT_SUPPORT 206001
#define ACLDVPP_ERR_RUNTIME_ERROR 306001

#define ACLDVPP_ERR_INNER 506000
#define ACLDVPP_ERR_INNER_CREATE_EXECUTOR 506101
#define ACLDVPP_ERR_INNER_NOT_TRANS_EXECUTOR 506102
#define ACLDVPP_ERR_INNER_NULLPTR 506103

enum acldvppConvertMode {
    COLOR_BGR2BGRA     = 0, // 新增alpha通道到RGB或BGR图像
    COLOR_RGB2RGBA     = COLOR_BGR2BGRA,

    COLOR_BGRA2BGR     = 1, // 从RGB或BGR图像删除alpha通道
    COLOR_RGBA2RGB     = COLOR_BGRA2BGR,

    COLOR_BGR2RGBA     = 2, // 在RGB和BGR空间之间转换，同时新增alpha通道
    COLOR_RGB2BGRA     = COLOR_BGR2RGBA,

    COLOR_RGBA2BGR     = 3, // 在RGB和BGR空间之间转换，同时删除alpha通道
    COLOR_BGRA2RGB     = COLOR_RGBA2BGR,

    COLOR_BGR2RGB      = 4,
    COLOR_RGB2BGR      = COLOR_BGR2RGB,

    COLOR_BGRA2RGBA    = 5,
    COLOR_RGBA2BGRA    = COLOR_BGRA2RGBA,

    COLOR_BGR2GRAY     = 6,
    COLOR_RGB2GRAY     = 7,
    COLOR_GRAY2BGR     = 8,
    COLOR_GRAY2RGB     = COLOR_GRAY2BGR,
    COLOR_GRAY2BGRA    = 9,
    COLOR_GRAY2RGBA    = COLOR_GRAY2BGRA,
    COLOR_BGRA2GRAY    = 10,
    COLOR_RGBA2GRAY    = 11,
};

#ifdef __cplusplus
}
#endif
#endif // ACLDVPP_BASE_H_