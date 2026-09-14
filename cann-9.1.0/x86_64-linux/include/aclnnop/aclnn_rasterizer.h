
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_RASTERIZER_H_
#define ACLNN_RASTERIZER_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnRasterizerGetWorkspaceSize
 * parameters :
 * v : required
 * f : required
 * dOptional : optional
 * width : required
 * height : required
 * occlusionTruncation : required
 * useDepthPrior : required
 * findicesOut : required
 * barycentricOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRasterizerGetWorkspaceSize(
    const aclTensor *v,
    const aclTensor *f,
    const aclTensor *dOptional,
    int64_t width,
    int64_t height,
    double occlusionTruncation,
    int64_t useDepthPrior,
    const aclTensor *findicesOut,
    const aclTensor *barycentricOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnRasterizer
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnRasterizer(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
