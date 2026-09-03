/**
 * Variant sweep example for mhc_pre: run one case selected by argv[3] to bisect the segfault.
 * case 0: 3D x (bs,n,d), gamma=null, alpha(3)  -- original failing posture, d configurable
 * case 1: 4D x (1,bs,n,d), gamma given, alpha(3), d=2560 (known-good posture)
 * Data: x/phi/alpha/bias all constant (like the original example) to remove data variance.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "acl/acl.h"
#include "aclnn_mhc_pre.h"

#define CHECK_RET(cond, msg) \
    do { if (!(cond)) { printf("ERROR: %s (ret=%d)\n", msg, static_cast<int>(ret)); return -1; } } while (0)

namespace {
int64_t ReadI64(FILE *f) { int64_t v = 0; size_t r = fread(&v, 8, 1, f); (void)r; return v; }
void WriteI64(FILE *f, int64_t v) { size_t r = fwrite(&v, 8, 1, f); (void)r; }

int MakeTensorF32(const std::vector<float> &host, const std::vector<int64_t> &shape,
                  void *&dev, aclTensor *&t)
{
    size_t n = 1;
    for (auto d : shape) { n *= static_cast<size_t>(d); }
    aclrtMalloc(&dev, n * 4, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(dev, n * 4, host.data(), n * 4, ACL_MEMCPY_HOST_TO_DEVICE);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    t = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                        aclFormat::ACL_FORMAT_ND, strides.data(), shape.size(), dev);
    return 0;
}

int MakeTensorF16(const std::vector<uint16_t> &host, const std::vector<int64_t> &shape,
                   void *&dev, aclTensor *&t)
{
    size_t n = 1;
    for (auto d : shape) { n *= static_cast<size_t>(d); }
    aclrtMalloc(&dev, n * 2, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(dev, n * 2, host.data(), n * 2, ACL_MEMCPY_HOST_TO_DEVICE);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    t = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, strides.data(), 0,
                        aclFormat::ACL_FORMAT_ND, strides.data(), shape.size(), dev);
    return 0;
}

void MakeOut(const std::vector<int64_t> &shape, aclDataType dt, size_t itemsize,
             void *&dev, aclTensor *&t)
{
    size_t bytes = 1;
    for (auto dd : shape) { bytes *= static_cast<size_t>(dd); }
    bytes *= itemsize;
    aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    t = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0,
                        aclFormat::ACL_FORMAT_ND, strides.data(), shape.size(), dev);
}
}  // namespace

int main(int argc, char **argv)
{
    int64_t bs = (argc > 1) ? atoll(argv[1]) : 8;
    int64_t d = (argc > 2) ? atoll(argv[2]) : 2560;
    int mode = (argc > 3) ? atoi(argv[3]) : 0;   // 0: 3D+no gamma, 1: 4D+gamma
    aclnnStatus ret = ACL_SUCCESS;
    int64_t n = 4, mixHc = n * n + 2 * n;

    ret = static_cast<aclnnStatus>(aclInit(nullptr));
    CHECK_RET(ret == ACL_SUCCESS, "aclInit");
    aclrtContext ctx; aclrtStream strm;
    aclrtCreateContext(&ctx, 0);
    aclrtCreateStream(&strm);

    std::vector<uint16_t> xHost(static_cast<size_t>(bs * n * d), 0x3c00);  // fp16 1.0
    std::vector<float> phiHost(static_cast<size_t>(mixHc * n * d), 1e-4f);
    std::vector<float> alphaHost(3, 1.0f);
    std::vector<float> biasHost(static_cast<size_t>(mixHc), 0.5f);

    void *xDev = nullptr, *phiDev = nullptr, *alphaDev = nullptr, *biasDev = nullptr, *gammaDev = nullptr;
    void *hinDev = nullptr, *hpDev = nullptr, *hrDev = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr, *gamma = nullptr;
    aclTensor *hin = nullptr, *hPost = nullptr, *hRes = nullptr;
    aclTensor *invRms = nullptr, *hMix = nullptr, *hPre = nullptr;

    std::vector<int64_t> phiShape = {mixHc, n * d}, alphaShape = {3}, biasShape = {mixHc};
    std::vector<int64_t> xShape, hinShape, hpShape, hrShape;
    if (mode == 1) {
        xShape = {1, bs, n, d}; hinShape = {1, bs, d}; hpShape = {1, bs, n}; hrShape = {1, bs, n, n};
    } else {
        xShape = {bs, n, d}; hinShape = {bs, d}; hpShape = {bs, n}; hrShape = {bs, n, n};
    }
    MakeTensorF16(xHost, xShape, xDev, x);
    MakeTensorF32(phiHost, phiShape, phiDev, phi);
    MakeTensorF32(alphaHost, alphaShape, alphaDev, alpha);
    MakeTensorF32(biasHost, biasShape, biasDev, bias);
    if (mode == 1) {
        std::vector<float> gammaHost(static_cast<size_t>(n * d), 1.0f);
        MakeTensorF32(gammaHost, {n, d}, gammaDev, gamma);
    }
    MakeOut(hinShape, ACL_FLOAT16, 2, hinDev, hin);
    MakeOut(hpShape, ACL_FLOAT, 4, hpDev, hPost);
    MakeOut(hrShape, ACL_FLOAT, 4, hrDev, hRes);
    // mirror original example: provide ALL optional outputs (null optionals segfault in A3 path)
    void *irDev = nullptr, *hmDev = nullptr, *hp2Dev = nullptr;
    std::vector<int64_t> irShape = (mode == 1) ? std::vector<int64_t>{1, bs} : std::vector<int64_t>{bs};
    std::vector<int64_t> hmShape = (mode == 1) ? std::vector<int64_t>{1, bs, mixHc} : std::vector<int64_t>{bs, mixHc};
    std::vector<int64_t> hp2Shape = (mode == 1) ? std::vector<int64_t>{1, bs, n} : std::vector<int64_t>{bs, n};
    MakeOut(irShape, ACL_FLOAT, 4, irDev, invRms);
    MakeOut(hmShape, ACL_FLOAT, 4, hmDev, hMix);
    MakeOut(hp2Shape, ACL_FLOAT, 4, hp2Dev, hPre);

    printf("[mhc-var] bs=%ld d=%ld mode=%d\n", (long)bs, (long)d, mode);
    fflush(stdout);

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    ret = aclnnMhcPreGetWorkspaceSize(x, phi, alpha, bias, gamma, 1e-6, 1e-6,
                                      hin, hPost, hRes, invRms, hMix, hPre, &wsSize, &exec);
    printf("[mhc-var] getWs rc=%d ws=%lu\n", (int)ret, (unsigned long)wsSize);
    fflush(stdout);
    CHECK_RET(ret == ACL_SUCCESS, "GetWorkspaceSize");

    void *ws = nullptr;
    if (wsSize > 0) { aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_NORMAL_ONLY); }
    ret = aclnnMhcPre(ws, wsSize, exec, strm);
    printf("[mhc-var] run rc=%d\n", (int)ret);
    fflush(stdout);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnMhcPre");
    aclrtSynchronizeStream(strm);
    printf("[mhc-var] DONE\n");
    return 0;
}
