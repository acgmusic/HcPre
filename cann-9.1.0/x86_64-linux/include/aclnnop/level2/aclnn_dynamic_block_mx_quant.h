
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DYNAMIC_BLOCK_MX_QUANT_H_
#define ACLNN_DYNAMIC_BLOCK_MX_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDynamicBlockMxQuantGetWorkspaceSize
 * parameters :
 * x : required
 * roundModeOptional : optional
 * dstType : optional
 * scaleAlg : optional
 * dstTypeMax : optional
 * yOut : required
 * scale1Out : required
 * scale2Out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicBlockMxQuantGetWorkspaceSize(
    const aclTensor *x,
    char *roundModeOptional,
    int64_t dstType,
    int64_t scaleAlg,
    double dstTypeMax,
    const aclTensor *yOut,
    const aclTensor *scale1Out,
    const aclTensor *scale2Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDynamicBlockMxQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicBlockMxQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
