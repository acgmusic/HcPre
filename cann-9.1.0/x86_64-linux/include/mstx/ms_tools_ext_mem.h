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

#ifndef MSTX_TOOLS_EXT_MEM
#define MSTX_TOOLS_EXT_MEM

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct mstxDomainRegistration_st;
typedef struct mstxDomainRegistration_st mstxDomainRegistration_t;
typedef mstxDomainRegistration_t* mstxDomainHandle_t;

struct mstxMemHeap_st;
typedef struct mstxMemHeap_st mstxMemHeap_t;
typedef mstxMemHeap_t* mstxMemHeapHandle_t;

struct mstxMemRegion_st;
typedef struct mstxMemRegion_st mstxMemRegion_t;
typedef mstxMemRegion_t* mstxMemRegionHandle_t;

/** @brief Structure to describe a memory virtual range
 * @member deviceId - device id which the memory belongs to
 * @member ptr - memory pointer
 * @member size - memory size
 */
typedef struct mstxMemVirtualRangeDesc_t {
    uint32_t deviceId;
    void const *ptr;
    uint64_t size;
} mstxMemVirtualRangeDesc_t;

/** @brief Usage characteristics of the heap
 * Usage characteristics help tools like memcheckers, sanitizer,
 * as well as other debugging and profiling tools to determine some
 * special behaviors they should apply to the heap and it's regions.
 */
typedef enum mstxMemHeapUsageType {
     /** @brief This heap is a sub-allocator
     * Heap created with this usage should not be accessed by the user until regions are registered.
     */
    MSTX_MEM_HEAP_USAGE_TYPE_SUB_ALLOCATOR = 0,
} mstxMemHeapUsageType;

/** @brief Memory type characteristics of the heap
 * The 'type' indicates how to interpret the ptr field of the heapDesc.
 * This is intended to support many additional types of memory, beyond
 * standard process virtual memory, such as API specific memory only
 * addressed by handles or multi-dimensional memory requiring more complex
 * descriptions to handle features like strides, tiling, or interlace.
 */
typedef enum mstxMemType {
     /** @brief Standard process userspace virtual addresses for linear allocations.
     * mstxMemHeapRegister receives a heapDesc of type mstxMemVirtualRangeDesc_t
     */
    MSTX_MEM_TYPE_VIRTUAL_ADDRESS = 0,
} mstxMemType;

/** @brief Structure to describe a memory heap register description
 * @member usage - usage characteristics of the heap
 * @member type - memory type characteristics of the heap
 * @member typeSpecificDesc - memory heap description for specific memory type
 */
typedef struct mstxMemHeapDesc_t {
    mstxMemHeapUsageType usage;
    mstxMemType type;
    void const *typeSpecificDesc;
} mstxMemHeapDesc_t;

/** @brief Description for a batch of memory regions
 * @member heap - the memory pool which the memory regions belong to
 * @member regionType - memory type characteristics of the heap
 * @member regionCount - amount of memory region descriptions
 * @member regionDescArray - array of memory region descriptions
 * @member regionHandleArrayOut - [out] array of handles that stands for registered memory regions
 */
typedef struct mstxMemRegionsRegisterBatch_t {
    mstxMemHeapHandle_t heap;
    mstxMemType regionType;
    size_t regionCount;
    void const *regionDescArray;
    mstxMemRegionHandle_t* regionHandleArrayOut;
} mstxMemRegionsRegisterBatch_t;


typedef enum mstxMemRegionRefType {
    MSTX_MEM_REGION_REF_TYPE_POINTER = 0,
    MSTX_MEM_REGION_REF_TYPE_HANDLE
} mstxMemRegionRefType;

/** @brief Memory region reference
  * @member refType - the way the reference is described
  * @member pointer - reference is described via pointer, and the pointer is saved
  * @member handle - reference is described via handle, and the handle is saved
  */
typedef struct mstxMemRegionRef_t {
    mstxMemRegionRefType refType;
    union {
        void const *pointer;
        mstxMemRegionHandle_t handle;
    };
} mstxMemRegionRef_t;

typedef struct mstxMemRegionsUnregisterBatch_t {
    size_t refCount;
    mstxMemRegionRef_t const *refArray;
} mstxMemRegionsUnregisterBatch_t;

/** @brief There are no permissions for this memory
 */
#define MSTX_MEM_PERMISSIONS_REGION_FLAGS_NONE 0x00

/** @brief The memory is readable
 */
#define MSTX_MEM_PERMISSIONS_REGION_FLAGS_READ 0x01

/** @brief The memory is writable
 */
#define MSTX_MEM_PERMISSIONS_REGION_FLAGS_WRITE 0x02

/** @brief The memory is accessable among multiple devices
 */
#define MSTX_MEM_PERMISSIONS_REGION_FLAGS_SHARED 0x04

/** @brief Description for memory permissions assigned to region
  * @member flags - permission flags from MSTX_MEM_PERMISSIONS_REGION_FLAGS_*
  * @member region - region reference to registered virtual memory region
  */
typedef struct mstxMemPermissionsAssignRegionsDesc_t {
    uint32_t flags;
    mstxMemRegionRef_t region;
} mstxMemPermissionsAssignRegionsDesc_t;

/** @brief Description for a Batch of memory permissions assigned to regions
  * @member regionCount - count of elements in regionDescArray
  * @member regionDescArray - array of memory permissions assignment descriptions
  */
typedef struct mstxMemPermissionsAssignBatch_t {
    size_t regionCount;
    mstxMemPermissionsAssignRegionsDesc_t const *regionDescArray;
} mstxMemPermissionsAssignBatch_t;

static mstxDomainHandle_t const globalDomain = NULL;

/**
 * @ingroup MSTX
 * @brief Register a memory heap as a memory pool
 *
 * @param domain [IN]    domain the memory pool belongs to
 * @param desc   [IN]    memory range description for memory pool
 * @retval mstxMemHeapHandle_t Return memory pool
 */
MSTX_DECLSPEC mstxMemHeapHandle_t mstxMemHeapRegister(mstxDomainHandle_t domain, mstxMemHeapDesc_t const *desc);

/**
 * @ingroup MSTX
 * @brief Unregister a memory pool
 *
 * @param domain [IN]    domain the memory pool belongs to
 * @param heap   [IN]    memory pool to unregister
 */
MSTX_DECLSPEC void mstxMemHeapUnregister(mstxDomainHandle_t domain, mstxMemHeapHandle_t heap);

/**
 * @ingroup MSTX
 * @brief Register suballocations within a pool
 *
 * @param domain - domain the memory pool belongs to
 * @param desc - description of a batch of memory regions
 */
MSTX_DECLSPEC void mstxMemRegionsRegister(mstxDomainHandle_t domain, mstxMemRegionsRegisterBatch_t const *desc);

/**
 * @ingroup MSTX
 * @brief Unregister suballocations within a pool
 * @param domain - domain the suballocations belongs to
 * @param desc - description of a batch of region references
 */
MSTX_DECLSPEC void mstxMemRegionsUnregister(mstxDomainHandle_t domain, mstxMemRegionsUnregisterBatch_t const *desc);

/**
 * @ingroup MSTX
 * @brief Change the permissions of a region of process virtual memory
 * @param domain - domain the suballocations belongs to
 * @param desc - description of a batch of permission descriptions assigned to region references
 */
MSTX_DECLSPEC void mstxMemPermissionsAssign(mstxDomainHandle_t domain, mstxMemPermissionsAssignBatch_t const *desc);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // MSTX_TOOLS_EXT_MEM
