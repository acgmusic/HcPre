
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SWIGLU_GROUP_QUANT_GRAD_H_
#define ACLNN_SWIGLU_GROUP_QUANT_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSwigluGroupQuantGradGetWorkspaceSize
 * parameters :
 * gradY : required
 * x : required
 * weightOptional : optional
 * yOriginOptional : optional
 * groupIndexOptional : optional
 * clampLimit : optional
 * gradXOut : required
 * gradWeightOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwigluGroupQuantGradGetWorkspaceSize(
    const aclTensor *gradY,
    const aclTensor *x,
    const aclTensor *weightOptional,
    const aclTensor *yOriginOptional,
    const aclTensor *groupIndexOptional,
    double clampLimit,
    const aclTensor *gradXOut,
    const aclTensor *gradWeightOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSwigluGroupQuantGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwigluGroupQuantGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
