
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SPARSE_FLASH_MLA_GRAD_H_
#define ACLNN_SPARSE_FLASH_MLA_GRAD_H_

#include "aclnn/acl_meta.h"
#warning "This file is scheduled to be deprecated. Please use the file with the same name under include/aclnnop in the CANN package installation path instead."

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSparseFlashMlaGradGetWorkspaceSize
 * parameters :
 * query : required
 * dOut : required
 * out : required
 * lse : required
 * oriKvOptional : optional
 * cmpKvOptional : optional
 * oriSparseIndicesOptional : optional
 * cmpSparseIndicesOptional : optional
 * cuSeqlensQOptional : optional
 * cuSeqlensOriKvOptional : optional
 * cuSeqlensCmpKvOptional : optional
 * sequsedQOptional : optional
 * sequsedOriKvOptional : optional
 * sequsedCmpKvOptional : optional
 * cmpResidualKvOptional : optional
 * oriTopkLengthOptional : optional
 * cmpTopkLengthOptional : optional
 * sinksOptional : optional
 * metadataOptional : optional
 * scaleValue : optional
 * cmpRatio : optional
 * oriMaskMode : optional
 * cmpMaskMode : optional
 * oriWinLeft : optional
 * oriWinRight : optional
 * layoutQOptional : optional
 * layoutKvOptional : optional
 * dQueryOut : required
 * dOriKvOutOptional : optional
 * dCmpKvOutOptional : optional
 * dSinksOutOptional : optional
 * oriSoftmaxL1NormOutOptional : optional
 * cmpSoftmaxL1NormOutOptional : optional
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseFlashMlaGradGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *dOut,
    const aclTensor *out,
    const aclTensor *lse,
    const aclTensor *oriKvOptional,
    const aclTensor *cmpKvOptional,
    const aclTensor *oriSparseIndicesOptional,
    const aclTensor *cmpSparseIndicesOptional,
    const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional,
    const aclTensor *sequsedQOptional,
    const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional,
    const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional,
    const aclTensor *cmpTopkLengthOptional,
    const aclTensor *sinksOptional,
    const aclTensor *metadataOptional,
    double scaleValue,
    int64_t cmpRatio,
    int64_t oriMaskMode,
    int64_t cmpMaskMode,
    int64_t oriWinLeft,
    int64_t oriWinRight,
    char *layoutQOptional,
    char *layoutKvOptional,
    const aclTensor *dQueryOut,
    const aclTensor *dOriKvOutOptional,
    const aclTensor *dCmpKvOutOptional,
    const aclTensor *dSinksOutOptional,
    const aclTensor *oriSoftmaxL1NormOutOptional,
    const aclTensor *cmpSoftmaxL1NormOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSparseFlashMlaGrad
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseFlashMlaGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
