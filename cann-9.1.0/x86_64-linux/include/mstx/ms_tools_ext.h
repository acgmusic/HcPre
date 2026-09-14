/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */

#ifndef MSTX_TOOLS_EXT
#define MSTX_TOOLS_EXT

#if defined(MSTX_NO_IMPL)
#define MSTX_DECLSPEC
#else
#define MSTX_DECLSPEC inline static
#endif

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define MSTX_SUCCESS 0
#define MSTX_FAIL 1

#define MSTX_TOOL_INVALID_ID 0x0
#define MSTX_TOOL_MSPROF_ID 0x1000
#define MSTX_TOOL_MSOPPROF_ID 0x1001
#define MSTX_TOOL_MSSANITIZER_ID 0x1002
#define MSTX_TOOL_MSLEAKS_ID 0x1003

typedef uint64_t mstxRangeId;
typedef void* aclrtStream;

struct mstxDomainRegistration_st;
typedef struct mstxDomainRegistration_st mstxDomainRegistration_t;
typedef mstxDomainRegistration_t* mstxDomainHandle_t;

/**
 * @ingroup MSTX
 * @brief mstx mark
 *
 * @param message [IN]    message for mark, cannot be null
 * @param stream  [IN]    stream used by mark, which can be set to null if not needed.
 */
MSTX_DECLSPEC void mstxMarkA(const char *message, aclrtStream stream);

/**
 * @ingroup MSTX
 * @brief mstx range start
 *
 * @param message [IN]    message to mark range, cannot be null
 * @param stream  [IN]    stream used by range, which can be set to null if not needed.
 * @retval mstxRangeId  the range id used for mstxRangeEnd
 * @retval return 0 if range start failed
 */
MSTX_DECLSPEC mstxRangeId mstxRangeStartA(const char *message, aclrtStream stream);

/**
 * @ingroup MSTX
 * @brief mstx range end
 *
 * @param id [IN]    the range id return by range start api
 */
MSTX_DECLSPEC void mstxRangeEnd(mstxRangeId id);

/**
 * @ingroup MSTX
 * @brief mstx create a domain
 *
 * @param name [IN]    a unique string representing the domain.
 * @retval mstxDomainHandle_t  a handle representing the domain.
*/
MSTX_DECLSPEC mstxDomainHandle_t mstxDomainCreateA(const char *name);

/**
 * @ingroup MSTX
 * @brief mstx destroy a domain
 *
 * @param mstxDomainHandle_t [IN]    the domain handle to be destroyed.
*/
MSTX_DECLSPEC void mstxDomainDestroy(mstxDomainHandle_t domain);

/**
 * @ingroup MSTX
 * @brief mstx mark for specific domain
 *
 * @param domain  [IN]    the domain of scoping the category
 * @param message [IN]    message for mark, cannot be null
 * @param stream  [IN]    stream used by mark, which can be set to null if not needed.
 */
MSTX_DECLSPEC void mstxDomainMarkA(mstxDomainHandle_t domain, const char *message, aclrtStream stream);

/**
 * @ingroup MSTX
 * @brief mstx range start for specific domain
 *
 * @param domain  [IN]    the domain of scoping the category
 * @param message [IN]    message to mark range, cannot be null
 * @param stream  [IN]    stream used by range, which can be set to null if not needed.
 * @retval mstxRangeId  the range id used for mstxRangeEnd
 * @retval return 0 if range start failed or input invalid domain handle
 */
MSTX_DECLSPEC mstxRangeId mstxDomainRangeStartA(mstxDomainHandle_t domain, const char *message, aclrtStream stream);

/**
 * @ingroup MSTX
 * @brief mstx range end for specific domain
 *
 * @param domain  [IN]    the domain of scoping the category
 * @param id [IN]    the range id return by range start api
 */
MSTX_DECLSPEC void mstxDomainRangeEnd(mstxDomainHandle_t domain, mstxRangeId id);

/**
 * @ingroup MSTX
 * @brief get id of the registered tool
 * @param id [OUT] Tool registration identifier. If the tool is not registered
 *                 or this function is not registered by tool, `id` will be set to MSTX_TOOL_INVALID_ID.
 */
MSTX_DECLSPEC void mstxGetToolId(uint64_t *id);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#define MSTX_IMPL_GUARD // Ensure other headers cannot include directly

#include "ms_tools_ext_mem.h"
#include "mstx_detail/mstx_types.h"

#ifndef MSTX_NO_IMPL
#include "mstx_detail/mstx_impl.h"
#endif // MSTX_NO_IMPL

#undef MSTX_IMPL_GUARD
#endif // MSTX_TOOLS_EXT
