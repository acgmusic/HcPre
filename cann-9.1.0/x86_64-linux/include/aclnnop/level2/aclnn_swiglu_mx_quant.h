
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SWIGLU_MX_QUANT_H_
#define ACLNN_SWIGLU_MX_QUANT_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSwigluMxQuantGetWorkspaceSize
 * parameters :
 * x : required
 * groupIndexOptional : optional
 * activateDim : optional
 * activateLeft : optional
 * swigluMode : optional
 * clampLimit : optional
 * gluAlpha : optional
 * gluBias : optional
 * groupMode : optional
 * axis : optional
 * dstType : optional
 * roundModeOptional : optional
 * scaleAlg : optional
 * maxDtypeValue : optional
 * yOut : required
 * mxscaleOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwigluMxQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *groupIndexOptional,
    int64_t activateDim,
    bool activateLeft,
    int64_t swigluMode,
    double clampLimit,
    double gluAlpha,
    double gluBias,
    int64_t groupMode,
    int64_t axis,
    int64_t dstType,
    char *roundModeOptional,
    int64_t scaleAlg,
    double maxDtypeValue,
    const aclTensor *yOut,
    const aclTensor *mxscaleOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSwigluMxQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSwigluMxQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
