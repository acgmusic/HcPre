
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_STRIDED_SLICE_ASSIGN_V2_H_
#define ACLNN_STRIDED_SLICE_ASSIGN_V2_H_

#include "aclnn/acl_meta.h"
#warning "This file is scheduled to be deprecated. Please use the file with the same name under include/aclnnop in the CANN package installation path instead."

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnStridedSliceAssignV2GetWorkspaceSize
 * parameters :
 * varRef : required
 * inputValue : required
 * begin : required
 * end : required
 * strides : required
 * axesOptional : optional
 * varRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnStridedSliceAssignV2GetWorkspaceSize(
    aclTensor *varRef,
    const aclTensor *inputValue,
    const aclIntArray *begin,
    const aclIntArray *end,
    const aclIntArray *strides,
    const aclIntArray *axesOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnStridedSliceAssignV2TensorGetWorkspaceSize(
    aclTensor *varRef,
    const aclTensor *inputValue,
    const aclTensor *begin,
    const aclTensor *end,
    const aclTensor *strides,
    const aclTensor *axesOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnStridedSliceAssignV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnStridedSliceAssignV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
