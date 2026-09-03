
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SWIGLU_MX_QUANT_WITH_DUAL_AXIS_H_
#define ACLNN_SWIGLU_MX_QUANT_WITH_DUAL_AXIS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize
 * parameters :
 * x : required
 * groupIndexOptional : optional
 * activateLeft : optional
 * roundModeOptional : optional
 * scaleAlg : optional
 * dstType : optional
 * maxDtypeValue : optional
 * y1Out : required
 * mxScale1Out : required
 * y2Out : required
 * mxScale2Out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *groupIndexOptional,
    bool activateLeft,
    char *roundModeOptional,
    int64_t scaleAlg,
    int64_t dstType,
    double maxDtypeValue,
    const aclTensor *y1Out,
    const aclTensor *mxScale1Out,
    const aclTensor *y2Out,
    const aclTensor *mxScale2Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSwigluMxQuantWithDualAxis
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwigluMxQuantWithDualAxis(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
