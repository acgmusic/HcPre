
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MRGBA_CUSTOM_H_
#define ACLNN_MRGBA_CUSTOM_H_

#include "aclnn/acl_meta.h"
#warning "This file is scheduled to be deprecated. Please use the file with the same name under include/aclnnop in the CANN package installation path instead."

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMrgbaCustomGetWorkspaceSize
 * parameters :
 * rgb : required
 * alpha : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMrgbaCustomGetWorkspaceSize(
    const aclTensor *rgb,
    const aclTensor *alpha,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMrgbaCustom
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMrgbaCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
