
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ATTENTION_WORKER_COMBINE_H_
#define ACLNN_ATTENTION_WORKER_COMBINE_H_

#include "aclnn/acl_meta.h"
#warning "This file is scheduled to be deprecated. Please use the file with the same name under include/aclnnop in the CANN package installation path instead."

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAttentionWorkerCombineGetWorkspaceSize
 * parameters :
 * scheduleContext : required
 * expertScales : required
 * layerId : required
 * hiddenSize : required
 * tokenDtype : optional
 * needSchedule : optional
 * yOut : required
 * nextLayerIdOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAttentionWorkerCombineGetWorkspaceSize(
    const aclTensor *scheduleContext,
    const aclTensor *expertScales,
    const aclTensor *layerId,
    int64_t hiddenSize,
    int64_t tokenDtype,
    int64_t needSchedule,
    const aclTensor *yOut,
    const aclTensor *nextLayerIdOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAttentionWorkerCombine
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAttentionWorkerCombine(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
