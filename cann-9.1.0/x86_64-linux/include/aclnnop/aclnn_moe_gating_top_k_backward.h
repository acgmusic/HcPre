
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MOE_GATING_TOP_KBACKWARD_H_
#define ACLNN_MOE_GATING_TOP_KBACKWARD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeGatingTopKBackwardGetWorkspaceSize
 * parameters :
 * xNorm : required
 * gradY : required
 * expertIdx : required
 * renorm : optional
 * normType : optional
 * routedScalingFactor : optional
 * eps : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeGatingTopKBackwardGetWorkspaceSize(
    const aclTensor *xNorm,
    const aclTensor *gradY,
    const aclTensor *expertIdx,
    int64_t renorm,
    int64_t normType,
    double routedScalingFactor,
    double eps,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMoeGatingTopKBackward
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeGatingTopKBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
