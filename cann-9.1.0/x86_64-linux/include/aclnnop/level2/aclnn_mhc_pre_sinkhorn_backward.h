
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MHC_PRE_SINKHORN_BACKWARD_H_
#define ACLNN_MHC_PRE_SINKHORN_BACKWARD_H_

#include "aclnn/acl_meta.h"
#warning "This file is scheduled to be deprecated. Please use the file with the same name under include/aclnnop in the CANN package installation path instead."

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMhcPreSinkhornBackwardGetWorkspaceSize
 * parameters :
 * gradHin : required
 * gradHPost : required
 * gradHRes : required
 * x : required
 * phi : required
 * alpha : required
 * bias : required
 * hPre : required
 * hcBeforeNorm : required
 * invRms : required
 * sumOut : required
 * normOut : required
 * hcEps : optional
 * gradXOut : required
 * gradPhiOut : required
 * gradAlphaOut : required
 * gradBiasOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMhcPreSinkhornBackwardGetWorkspaceSize(
    const aclTensor *gradHin,
    const aclTensor *gradHPost,
    const aclTensor *gradHRes,
    const aclTensor *x,
    const aclTensor *phi,
    const aclTensor *alpha,
    const aclTensor *bias,
    const aclTensor *hPre,
    const aclTensor *hcBeforeNorm,
    const aclTensor *invRms,
    const aclTensor *sumOut,
    const aclTensor *normOut,
    double hcEps,
    const aclTensor *gradXOut,
    const aclTensor *gradPhiOut,
    const aclTensor *gradAlphaOut,
    const aclTensor *gradBiasOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMhcPreSinkhornBackward
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMhcPreSinkhornBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
