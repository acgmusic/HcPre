
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MASKED_CAUSAL_CONV1D_H_
#define ACLNN_MASKED_CAUSAL_CONV1D_H_

#include "aclnn/acl_meta.h"
#warning "This file is scheduled to be deprecated. Please use the file with the same name under include/aclnnop in the CANN package installation path instead."

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMaskedCausalConv1dGetWorkspaceSize
 * parameters :
 * x : required
 * weight : required
 * maskOptional : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMaskedCausalConv1dGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *maskOptional,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMaskedCausalConv1d
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMaskedCausalConv1d(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
