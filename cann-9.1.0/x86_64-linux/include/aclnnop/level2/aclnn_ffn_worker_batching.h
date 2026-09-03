
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_FFN_WORKER_BATCHING_H_
#define ACLNN_FFN_WORKER_BATCHING_H_

#include "aclnn/acl_meta.h"
#warning "This file is scheduled to be deprecated. Please use the file with the same name under include/aclnnop in the CANN package installation path instead."

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnFfnWorkerBatchingGetWorkspaceSize
 * parameters :
 * scheduleContext : required
 * expertNum : required
 * maxOutShape : required
 * tokenDtype : optional
 * needSchedule : optional
 * layerNum : optional
 * yOut : required
 * groupListOut : required
 * sessionIdsOut : required
 * microBatchIdsOut : required
 * tokenIdsOut : required
 * expertOffsetsOut : required
 * dynamicScaleOut : required
 * actualTokenNumOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFfnWorkerBatchingGetWorkspaceSize(
    const aclTensor *scheduleContext,
    int64_t expertNum,
    const aclIntArray *maxOutShape,
    int64_t tokenDtype,
    int64_t needSchedule,
    int64_t layerNum,
    const aclTensor *yOut,
    const aclTensor *groupListOut,
    const aclTensor *sessionIdsOut,
    const aclTensor *microBatchIdsOut,
    const aclTensor *tokenIdsOut,
    const aclTensor *expertOffsetsOut,
    const aclTensor *dynamicScaleOut,
    const aclTensor *actualTokenNumOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnFfnWorkerBatching
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnFfnWorkerBatching(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
