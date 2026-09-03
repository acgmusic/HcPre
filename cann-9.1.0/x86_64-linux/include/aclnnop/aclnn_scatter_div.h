
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SCATTER_DIV_H_
#define ACLNN_SCATTER_DIV_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnScatterDivGetWorkspaceSize
 * parameters :
 * varRef : required
 * indices : required
 * updates : required
 * useLocking : optional
 * varRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterDivGetWorkspaceSize(
    aclTensor *varRef,
    const aclTensor *indices,
    const aclTensor *updates,
    bool useLocking,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnScatterDiv
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnScatterDiv(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
