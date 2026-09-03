
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DYNAMIC_DUAL_LEVEL_MX_QUANT_H_
#define ACLNN_DYNAMIC_DUAL_LEVEL_MX_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDynamicDualLevelMxQuantGetWorkspaceSize
 * parameters :
 * x : required
 * smoothScaleOptional : optional
 * roundModeOptional : optional
 * level0BlockSize : optional
 * level1BlockSize : optional
 * yOut : required
 * level0ScaleOut : required
 * level1ScaleOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicDualLevelMxQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *smoothScaleOptional,
    char *roundModeOptional,
    int64_t level0BlockSize,
    int64_t level1BlockSize,
    const aclTensor *yOut,
    const aclTensor *level0ScaleOut,
    const aclTensor *level1ScaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDynamicDualLevelMxQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicDualLevelMxQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
