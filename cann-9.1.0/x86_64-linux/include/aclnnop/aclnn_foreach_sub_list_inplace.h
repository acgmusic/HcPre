
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FOREACH_SUB_LIST_INPLACE_H_
#define ACLNN_FOREACH_SUB_LIST_INPLACE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnForeachSubListInplaceGetWorkspaceSize
 * parameters :
 * x1 : dynamic
 * x2 : dynamic
 * alpha : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachSubListInplaceGetWorkspaceSize(
    const aclTensorList *x1,
    const aclTensorList *x2,
    const aclTensor *alpha,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnForeachSubListInplace
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachSubListInplace(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
