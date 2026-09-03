
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FOREACH_MUL_SCALAR_INPLACE_H_
#define ACLNN_FOREACH_MUL_SCALAR_INPLACE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnForeachMulScalarInplaceGetWorkspaceSize
 * parameters :
 * x : dynamic
 * scalar : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachMulScalarInplaceGetWorkspaceSize(
    const aclTensorList *x,
    const aclTensor *scalar,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnForeachMulScalarInplace
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachMulScalarInplace(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
