
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DYNAMIC_BLOCK_QUANT_V2_H_
#define ACLNN_DYNAMIC_BLOCK_QUANT_V2_H_
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDynamicBlockQuantV2GetWorkspaceSize
 * parameters :
 * x : required
 * minScale : optional
 * roundModeOptional : optional
 * dstType : optional
 * rowBlockSize : optional
 * colBlockSize : optional
 * dstTypeMax : optional
 * yOut : required
 * scaleOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicBlockQuantV2GetWorkspaceSize(
    const aclTensor *x,
    double minScale,
    char *roundModeOptional,
    int64_t dstType,
    int64_t rowBlockSize,
    int64_t colBlockSize,
    double dstTypeMax,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDynamicBlockQuantV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicBlockQuantV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
