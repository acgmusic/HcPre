
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_H_
#define ACLNN_DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_H_

#include "aclnn/acl_meta.h"
#warning "This file is scheduled to be deprecated. Please use the file with the same name under include/aclnnop in the CANN package installation path instead."

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize
 * parameters :
 * queryIndex : required
 * keyIndex : required
 * weight : required
 * actualSeqLengthsQueryOptional : optional
 * actualSeqLengthsKeyOptional : optional
 * layoutOptional : optional
 * sparseMode : optional
 * preTokens : optional
 * nextTokens : optional
 * softmaxMaxOut : required
 * softmaxSumOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(
    const aclTensor *queryIndex,
    const aclTensor *keyIndex,
    const aclTensor *weight,
    const aclIntArray *actualSeqLengthsQueryOptional,
    const aclIntArray *actualSeqLengthsKeyOptional,
    char *layoutOptional,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

__attribute__((visibility("default")))
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseTensorGetWorkspaceSize(
    const aclTensor *queryIndex,
    const aclTensor *keyIndex,
    const aclTensor *weight,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    char *layoutOptional,
    int64_t sparseMode,
    int64_t preTokens,
    int64_t nextTokens,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnDenseLightningIndexerSoftmaxLse
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnDenseLightningIndexerSoftmaxLse(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
