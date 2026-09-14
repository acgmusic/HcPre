
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SWIGLU_GROUP_QUANT_H_
#define ACLNN_SWIGLU_GROUP_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSwigluGroupQuantGetWorkspaceSize
 * parameters :
 * x : required
 * weightOptional : optional
 * groupIndexOptional : optional
 * scaleOptional : optional
 * dstType : optional
 * quantMode : optional
 * blockSize : optional
 * roundScale : optional
 * clampLimit : optional
 * dstTypeMax : optional
 * outputOrigin : optional
 * yOut : required
 * yScaleOut : required
 * yOriginOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwigluGroupQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weightOptional,
    const aclTensor *groupIndexOptional,
    const aclTensor *scaleOptional,
    int64_t dstType,
    int64_t quantMode,
    int64_t blockSize,
    bool roundScale,
    double clampLimit,
    double dstTypeMax,
    bool outputOrigin,
    const aclTensor *yOut,
    const aclTensor *yScaleOut,
    const aclTensor *yOriginOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSwigluGroupQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwigluGroupQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
