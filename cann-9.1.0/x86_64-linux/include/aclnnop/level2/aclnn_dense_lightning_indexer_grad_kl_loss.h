
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DENSE_LIGHTNING_INDEXER_GRAD_KLLOSS_H_
#define ACLNN_DENSE_LIGHTNING_INDEXER_GRAD_KLLOSS_H_

#include "aclnn/acl_meta.h"
#warning "This file is scheduled to be deprecated. Please use the file with the same name under include/aclnnop in the CANN package installation path instead."

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * queryIndex : required
 * keyIndex : required
 * weight : required
 * softmaxMax : required
 * softmaxSum : required
 * softmaxMaxIndex : required
 * softmaxSumIndex : required
 * queryRopeOptional : optional
 * keyRopeOptional : optional
 * actualSeqLengthsQueryOptional : optional
 * actualSeqLengthsKeyOptional : optional
 * scaleValue : required
 * layoutOptional : optional
 * sparseMode : optional
 * preTokens : optional
 * nextTokens : optional
 * dQueryIndexOut : required
 * dKeyIndexOut : required
 * dWeightOut : required
 * lossOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *queryIndex,
    const aclTensor *keyIndex,
    const aclTensor *weight,
    const aclTensor *softmaxMax,
    const aclTensor *softmaxSum,
    const aclTensor *softmaxMaxIndex,
    const aclTensor *softmaxSumIndex,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    const aclIntArray *actualSeqLengthsQueryOptional,
    const aclIntArray *actualSeqLengthsKeyOptional,
    double scaleValue,
    char *layoutOptional,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    const aclTensor *dQueryIndexOut,
    const aclTensor *dKeyIndexOut,
    const aclTensor *dWeightOut,
    const aclTensor *lossOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnDenseLightningIndexerGradKLLossTensorGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *queryIndex,
    const aclTensor *keyIndex,
    const aclTensor *weight,
    const aclTensor *softmaxMax,
    const aclTensor *softmaxSum,
    const aclTensor *softmaxMaxIndex,
    const aclTensor *softmaxSumIndex,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    double scaleValue,
    char *layoutOptional,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    const aclTensor *dQueryIndexOut,
    const aclTensor *dKeyIndexOut,
    const aclTensor *dWeightOut,
    const aclTensor *lossOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDenseLightningIndexerGradKLLoss
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDenseLightningIndexerGradKLLoss(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
