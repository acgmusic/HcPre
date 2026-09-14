
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FOREACH_MUL_LIST_INPLACE_H_
#define ACLNN_FOREACH_MUL_LIST_INPLACE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnForeachMulListInplaceGetWorkspaceSize
 * parameters :
 * x1 : dynamic
 * x2 : dynamic
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachMulListInplaceGetWorkspaceSize(
    const aclTensorList *x1,
    const aclTensorList *x2,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnForeachMulListInplace
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachMulListInplace(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
