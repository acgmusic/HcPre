
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FUSED_FLOYD_ATTENTION_H_
#define ACLNN_FUSED_FLOYD_ATTENTION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFusedFloydAttentionGetWorkspaceSize
 * parameters :
 * query : required
 * key1 : required
 * value1 : required
 * key2 : required
 * value2 : required
 * attenMaskOptional : optional
 * scaleValue : optional
 * softmaxMaxOut : required
 * softmaxSumOut : required
 * attentionOutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFusedFloydAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key1,
    const aclTensor *value1,
    const aclTensor *key2,
    const aclTensor *value2,
    const aclTensor *attenMaskOptional,
    double scaleValue,
    const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFusedFloydAttention
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFusedFloydAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
