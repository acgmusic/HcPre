
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SPARSE_LIGHTNING_INDEXER_KLLOSS_GRAD_H_
#define ACLNN_SPARSE_LIGHTNING_INDEXER_KLLOSS_GRAD_H_

#include "aclnn/acl_meta.h"
#warning "This file is scheduled to be deprecated. Please use the file with the same name under include/aclnnop in the CANN package installation path instead."

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize
 * parameters :
 * q : required
 * k : required
 * w : required
 * sparseIndices : required
 * attnSoftmaxL1Norm : required
 * cuSeqlensQOptional : optional
 * cuSeqlensKOptional : optional
 * sequsedQOptional : optional
 * sequsedKOptional : optional
 * cmpResidualKOptional : optional
 * metadataOptional : optional
 * layoutQOptional : optional
 * layoutKOptional : optional
 * maskMode : optional
 * cmpRatio : optional
 * dqOut : required
 * dkOut : required
 * dwOut : required
 * softmaxOutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseLightningIndexerKLLossGradGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *sparseIndices,
    const aclTensor *attnSoftmaxL1Norm,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional,
    const aclTensor *cmpResidualKOptional,
    const aclTensor *metadataOptional,
    char *layoutQOptional,
    char *layoutKOptional,
    int64_t maskMode,
    int64_t cmpRatio,
    const aclTensor *dqOut,
    const aclTensor *dkOut,
    const aclTensor *dwOut,
    const aclTensor *softmaxOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSparseLightningIndexerKLLossGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseLightningIndexerKLLossGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
