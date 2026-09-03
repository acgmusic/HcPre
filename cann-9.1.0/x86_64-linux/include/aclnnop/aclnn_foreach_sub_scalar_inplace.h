
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FOREACH_SUB_SCALAR_INPLACE_H_
#define ACLNN_FOREACH_SUB_SCALAR_INPLACE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnForeachSubScalarInplaceGetWorkspaceSize
 * parameters :
 * x : dynamic
 * scalar : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachSubScalarInplaceGetWorkspaceSize(
    const aclTensorList *x,
    const aclTensor *scalar,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnForeachSubScalarInplace
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachSubScalarInplace(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
