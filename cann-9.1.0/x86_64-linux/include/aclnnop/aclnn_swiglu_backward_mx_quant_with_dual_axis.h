
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SWIGLU_BACKWARD_MX_QUANT_WITH_DUAL_AXIS_H_
#define ACLNN_SWIGLU_BACKWARD_MX_QUANT_WITH_DUAL_AXIS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSwigluBackwardMxQuantWithDualAxisGetWorkspaceSize
 * parameters :
 * x : required
 * yGrad : required
 * groupIndexOptional : optional
 * activateLeft : optional
 * roundModeOptional : optional
 * scaleAlg : optional
 * dstDtype : optional
 * maxDtypeValue : optional
 * y1OutOut : required
 * mxscale1OutOut : required
 * y2OutOut : required
 * mxscale2OutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwigluBackwardMxQuantWithDualAxisGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *yGrad,
    const aclTensor *groupIndexOptional,
    bool activateLeft,
    char *roundModeOptional,
    int64_t scaleAlg,
    int64_t dstDtype,
    double maxDtypeValue,
    const aclTensor *y1OutOut,
    const aclTensor *mxscale1OutOut,
    const aclTensor *y2OutOut,
    const aclTensor *mxscale2OutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSwigluBackwardMxQuantWithDualAxis
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwigluBackwardMxQuantWithDualAxis(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
