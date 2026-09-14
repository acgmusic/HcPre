/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef ACLDVPP_INIT_H_
#define ACLDVPP_INIT_H_

#include "acldvpp_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
* @brief DVPP初始化函数，同步接口。
* @param [in] configPath: 预留参数，配置文件所在路径的指针，包含文件名，当前需要配置为null
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppInit(const char *configPath);

/**
* @brief DVPP去初始化函数，同步接口。
* @return acldvppStatus: 返回状态码。
*/
acldvppStatus acldvppFinalize();

#ifdef __cplusplus
}
#endif

#endif // ACLDVPP_INIT_H_