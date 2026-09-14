
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_APPLY_ADAM_H_
#define ACLNN_APPLY_ADAM_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnApplyAdamGetWorkspaceSize
 * parameters :
 * varRef : required
 * m : required
 * v : required
 * beta1Power : required
 * beta2Power : required
 * lr : required
 * beta1 : required
 * beta2 : required
 * epsilon : required
 * grad : required
 * useLocking : optional
 * useNesterov : optional
 * varRef : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnApplyAdamGetWorkspaceSize(
    aclTensor *varRef,
    const aclTensor *m,
    const aclTensor *v,
    const aclTensor *beta1Power,
    const aclTensor *beta2Power,
    const aclTensor *lr,
    const aclTensor *beta1,
    const aclTensor *beta2,
    const aclTensor *epsilon,
    const aclTensor *grad,
    bool useLocking,
    bool useNesterov,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnApplyAdam
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnApplyAdam(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
