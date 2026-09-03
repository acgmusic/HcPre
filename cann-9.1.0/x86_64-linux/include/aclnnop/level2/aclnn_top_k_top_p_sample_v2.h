
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_TOP_KTOP_PSAMPLE_V2_H_
#define ACLNN_TOP_KTOP_PSAMPLE_V2_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnTopKTopPSampleV2GetWorkspaceSize
 * parameters :
 * logits : required
 * topK : required
 * topP : required
 * qOptional : optional
 * minPsOptional : optional
 * eps : optional
 * isNeedLogits : optional
 * topKGuess : optional
 * ksMax : optional
 * inputIsLogits : optional
 * isNeedSampleResult : optional
 * logitsSelectIdxOut : required
 * logitsTopKpSelectOut : required
 * logitsIdxOut : required
 * logitsSortMaskedOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTopKTopPSampleV2GetWorkspaceSize(
    const aclTensor *logits,
    const aclTensor *topK,
    const aclTensor *topP,
    const aclTensor *qOptional,
    const aclTensor *minPsOptional,
    double eps,
    bool isNeedLogits,
    int64_t topKGuess,
    int64_t ksMax,
    bool inputIsLogits,
    bool isNeedSampleResult,
    const aclTensor *logitsSelectIdxOut,
    const aclTensor *logitsTopKpSelectOut,
    const aclTensor *logitsIdxOut,
    const aclTensor *logitsSortMaskedOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnTopKTopPSampleV2
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnTopKTopPSampleV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
