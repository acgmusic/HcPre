
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_AMP_UPDATE_SCALE_H_
#define ACLNN_AMP_UPDATE_SCALE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAmpUpdateScaleGetWorkspaceSize
 * parameters :
 * currentScale : required
 * growthTracker : required
 * foundInf : required
 * growthFactor : required
 * backoffFactor : required
 * growthInterval : required
 * updatedScaleOut : required
 * updatedGrowthTrackerOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAmpUpdateScaleGetWorkspaceSize(
    const aclTensor *currentScale,
    const aclTensor *growthTracker,
    const aclTensor *foundInf,
    double growthFactor,
    double backoffFactor,
    int64_t growthInterval,
    const aclTensor *updatedScaleOut,
    const aclTensor *updatedGrowthTrackerOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAmpUpdateScale
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAmpUpdateScale(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
