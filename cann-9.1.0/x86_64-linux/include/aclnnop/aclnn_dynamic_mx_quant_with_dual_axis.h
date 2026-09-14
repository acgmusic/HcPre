
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H_
#define ACLNN_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize
 * parameters :
 * x : required
 * roundModeOptional : optional
 * dstType : optional
 * scaleAlg : optional
 * y1Out : required
 * mxscale1Out : required
 * y2Out : required
 * mxscale2Out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize(
    const aclTensor *x,
    char *roundModeOptional,
    int64_t dstType,
    int64_t scaleAlg,
    const aclTensor *y1Out,
    const aclTensor *mxscale1Out,
    const aclTensor *y2Out,
    const aclTensor *mxscale2Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDynamicMxQuantWithDualAxis
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDynamicMxQuantWithDualAxis(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
