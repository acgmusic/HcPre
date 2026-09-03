
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MOE_INIT_ROUTING_QUANT_H_
#define ACLNN_MOE_INIT_ROUTING_QUANT_H_

#include "aclnn/acl_meta.h"
#warning "This file is scheduled to be deprecated. Please use the file with the same name under include/aclnnop in the CANN package installation path instead."

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeInitRoutingQuantGetWorkspaceSize
 * parameters :
 * x : required
 * rowIdx : required
 * expertIdx : required
 * activeNum : required
 * scale : required
 * offset : required
 * expandedXOut : required
 * expandedRowIdxOut : required
 * expandedExpertIdxOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeInitRoutingQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *rowIdx,
    const aclTensor *expertIdx,
    int64_t activeNum,
    double scale,
    double offset,
    const aclTensor *expandedXOut,
    const aclTensor *expandedRowIdxOut,
    const aclTensor *expandedExpertIdxOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMoeInitRoutingQuant
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeInitRoutingQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
