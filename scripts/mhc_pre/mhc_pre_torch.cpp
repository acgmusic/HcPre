/**
 * Minimal pybind11 wrapper exposing aclnnMhcPre to torch via dlopen (vllm EXEC_NPU_CMD style).
 * Avoids direct -lcust_opapi linking: resolves symbols from the loaded opapi lib at runtime,
 * which is the proven-working mechanism in vllm_ascend (csrc/aclnn_torch_adapter).
 */
#include <torch/extension.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

#include <dlfcn.h>

namespace {

using aclCreateTensorFn = aclTensor *(*)(const int64_t *, uint64_t, aclDataType,
                                         const int64_t *, int64_t, aclFormat,
                                         const int64_t *, uint64_t, void *);
using aclDestroyTensorFn = int (*)(const aclTensor *);

void *LoadLib()
{
    // cust_opapi.so is already loadable via LD_LIBRARY_PATH (vendor op_api/lib)
    void *h = dlopen("libcust_opapi.so", RTLD_NOW | RTLD_GLOBAL);
    if (h == nullptr) {
        h = dlopen("libopapi.so", RTLD_NOW | RTLD_GLOBAL);
    }
    TORCH_CHECK(h != nullptr, "cannot dlopen opapi lib: ", dlerror());
    return h;
}

void *GetSym(void *h, const char *name)
{
    void *s = dlsym(h, name);
    TORCH_CHECK(s != nullptr, "missing symbol ", name);
    return s;
}

aclTensor *MakeAclTensor(void *h, const at::Tensor &t, aclDataType dt)
{
    static auto fn = reinterpret_cast<aclCreateTensorFn>(GetSym(h, "aclCreateTensor"));
    auto sizes = t.sizes().vec();
    auto strides = t.strides().vec();
    std::vector<int64_t> storageDims{t.storage().nbytes() / t.itemsize()};
    return fn(sizes.data(), sizes.size(), dt, strides.data(), t.storage_offset(),
              aclFormat::ACL_FORMAT_ND, storageDims.data(), storageDims.size(),
              const_cast<void *>(t.storage().data()));
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_mhc_pre(
    const at::Tensor &x, const at::Tensor &phi, const at::Tensor &alpha,
    const at::Tensor &bias, double norm_eps, double hc_eps)
{
    int64_t bs = x.size(0);
    int64_t n = x.size(1);
    int64_t d = x.size(2);

    at::Tensor hin = at::empty({bs, d}, x.options());
    at::Tensor h_post = at::empty({bs, n}, x.options().dtype(at::kFloat));
    at::Tensor h_res = at::empty({bs, n, n}, x.options().dtype(at::kFloat));

    static void *h = LoadLib();
    fprintf(stderr, "[mhc-torch] lib=%p\n", h); fflush(stderr);
    static auto getWs = reinterpret_cast<int64_t (*)(const aclTensor *, const aclTensor *,
        const aclTensor *, const aclTensor *, const aclTensor *, double, double,
        aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *, aclTensor *,
        uint64_t *, aclOpExecutor **)>(GetSym(h, "aclnnMhcPreGetWorkspaceSize"));
    static auto runOp = reinterpret_cast<int64_t (*)(void *, uint64_t, aclOpExecutor *,
                                                      aclrtStream)>(GetSym(h, "aclnnMhcPre"));
    fprintf(stderr, "[mhc-torch] syms ok\n"); fflush(stderr);

    aclTensor *x_t = MakeAclTensor(h, x, ACL_BF16);
    aclTensor *phi_t = MakeAclTensor(h, phi, ACL_FLOAT);
    aclTensor *alpha_t = MakeAclTensor(h, alpha, ACL_FLOAT);
    aclTensor *bias_t = MakeAclTensor(h, bias, ACL_FLOAT);
    aclTensor *hin_t = MakeAclTensor(h, hin, ACL_BF16);
    aclTensor *hp_t = MakeAclTensor(h, h_post, ACL_FLOAT);
    aclTensor *hr_t = MakeAclTensor(h, h_res, ACL_FLOAT);
    fprintf(stderr, "[mhc-torch] tensors built\n"); fflush(stderr);

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    int64_t rc = getWs(x_t, phi_t, alpha_t, bias_t, nullptr, norm_eps, hc_eps,
                       hin_t, hp_t, hr_t, nullptr, nullptr, nullptr, &wsSize, &exec);
    fprintf(stderr, "[mhc-torch] getWs rc=%ld ws=%lu\n", (long)rc, (unsigned long)wsSize); fflush(stderr);
    TORCH_CHECK(rc == 0, "aclnnMhcPreGetWorkspaceSize failed, code=", rc);

    at::Tensor ws = at::empty({static_cast<int64_t>(wsSize)}, x.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    rc = runOp(ws.data_ptr(), wsSize, exec, stream);
    fprintf(stderr, "[mhc-torch] runOp rc=%ld\n", (long)rc); fflush(stderr);
    TORCH_CHECK(rc == 0, "aclnnMhcPre failed, code=", rc);
    return {hin, h_post, h_res};
}

PYBIND11_MODULE(mhc_pre_torch, m)
{
    m.def("npu_mhc_pre", &npu_mhc_pre, "MhcPre via aclnn (dlopen)");
}
