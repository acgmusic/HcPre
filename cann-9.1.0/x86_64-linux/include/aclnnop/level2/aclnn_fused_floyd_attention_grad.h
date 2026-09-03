
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FUSED_FLOYD_ATTENTION_GRAD_H_
#define ACLNN_FUSED_FLOYD_ATTENTION_GRAD_H_

#include "aclnn/acl_meta.h"
#warning "This file is scheduled to be deprecated. Please use the file with the same name under include/aclnnop in the CANN package installation path instead."

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFusedFloydAttentionGradGetWorkspaceSize
 * parameters :
 * query : required
 * key1 : required
 * value1 : required
 * key2 : required
 * value2 : required
 * dy : required
 * attenMaskOptional : optional
 * softmaxMaxOptional : optional
 * softmaxSumOptional : optional
 * attentionInOptional : optional
 * scaleValue : optional
 * dqOut : required
 * dk1Out : required
 * dv1Out : required
 * dk2Out : required
 * dv2Out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFusedFloydAttentionGradGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key1,
    const aclTensor *value1,
    const aclTensor *key2,
    const aclTensor *value2,
    const aclTensor *dy,
    const aclTensor *attenMaskOptional,
    const aclTensor *softmaxMaxOptional,
    const aclTensor *softmaxSumOptional,
    const aclTensor *attentionInOptional,
    double scaleValue,
    const aclTensor *dqOut,
    const aclTensor *dk1Out,
    const aclTensor *dv1Out,
    const aclTensor *dk2Out,
    const aclTensor *dv2Out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFusedFloydAttentionGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFusedFloydAttentionGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
