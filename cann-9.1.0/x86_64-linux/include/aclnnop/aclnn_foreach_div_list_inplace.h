
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FOREACH_DIV_LIST_INPLACE_H_
#define ACLNN_FOREACH_DIV_LIST_INPLACE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnForeachDivListInplaceGetWorkspaceSize
 * parameters :
 * x1 : dynamic
 * x2 : dynamic
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachDivListInplaceGetWorkspaceSize(
    const aclTensorList *x1,
    const aclTensorList *x2,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnForeachDivListInplace
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnForeachDivListInplace(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
