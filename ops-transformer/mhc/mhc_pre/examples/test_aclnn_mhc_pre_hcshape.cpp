/**
 * Custom mhc_pre example: load inputs from a .pt-ish raw binary, run aclnnMhcPre, dump outputs.
 * Built alongside the standard example via the ops-transformer example build system
 * (add mhc_pre_hc_shape example), driven under msprof op simulator.
 *
 * Data contract with scripts/mhc_pre/run_golden.py --save-dump:
 *   input .bin  (little endian, raw):
 *     int64 x_bs, x_n, x_d;  then x (bf16 as uint16), phi fp32 (24, n*d),
 *     alpha fp32 (3,), bias fp32 (24,)
 *   output .bin:
 *     int64 code; then hin (bf16 uint16), h_post fp32, h_res fp32 (contiguous)
 */
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn_mhc_pre.h"

#define CHECK_RET(cond, msg) \
    do { if (!(cond)) { printf("ERROR: %s (ret=%d)\n", msg, static_cast<int>(ret)); return -1; } } while (0)

namespace {
int64_t ReadI64(FILE *f) { int64_t v = 0; size_t r = fread(&v, 8, 1, f); (void)r; return v; }
void WriteI64(FILE *f, int64_t v) { size_t r = fwrite(&v, 8, 1, f); (void)r; }

int MakeTensor(const std::vector<float> &host, const std::vector<int64_t> &shape,
               aclDataType dt, void *&dev, aclTensor *&t)
{
    size_t n = 1;
    for (auto d : shape) { n *= static_cast<size_t>(d); }
    aclrtMalloc(&dev, n * 4, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(dev, n * 4, host.data(), n * 4, ACL_MEMCPY_HOST_TO_DEVICE);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    t = aclCreateTensor(shape.data(), shape.size(), dt, strides.data(), 0,
                        aclFormat::ACL_FORMAT_ND, strides.data(), shape.size(), dev);
    return 0;
}

int MakeBf16TensorFromU16(const std::vector<uint16_t> &host, const std::vector<int64_t> &shape,
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
    t = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_BF16, strides.data(), 0,
                        aclFormat::ACL_FORMAT_ND, strides.data(), shape.size(), dev);
    return 0;
}
}  // namespace

int main(int argc, char **argv)
{
    const char *inPath = (argc > 1) ? argv[1] : "mhc_pre_input.bin";
    const char *outPath = (argc > 2) ? argv[2] : "mhc_pre_output.bin";
    aclnnStatus ret = ACL_SUCCESS;

    FILE *fin = fopen(inPath, "rb");
    if (fin == nullptr) { printf("cannot open %s\n", inPath); return -1; }
    int64_t bs = ReadI64(fin), n = ReadI64(fin), d = ReadI64(fin);
    int64_t mixHc = n * n + 2 * n;
    std::vector<uint16_t> xHost(static_cast<size_t>(bs * n * d));
    fread(xHost.data(), 2, xHost.size(), fin);
    std::vector<float> phiHost(static_cast<size_t>(mixHc * n * d));
    fread(phiHost.data(), 4, phiHost.size(), fin);
    std::vector<float> alphaHost(3);
    fread(alphaHost.data(), 4, 3, fin);
    std::vector<float> biasHost(static_cast<size_t>(mixHc));
    fread(biasHost.data(), 4, biasHost.size(), fin);
    fclose(fin);
    printf("[mhc-ex] bs=%ld n=%ld d=%ld mixHc=%ld\n", static_cast<long>(bs), static_cast<long>(n),
           static_cast<long>(d), static_cast<long>(mixHc));

    ret = static_cast<aclnnStatus>(aclInit(nullptr));
    CHECK_RET(ret == ACL_SUCCESS, "aclInit");
    aclrtContext ctx; aclrtStream strm;
    aclrtCreateContext(&ctx, 0);
    aclrtCreateStream(&strm);

    void *xDev = nullptr, *phiDev = nullptr, *alphaDev = nullptr, *biasDev = nullptr;
    void *hinDev = nullptr, *hpDev = nullptr, *hrDev = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr, *gamma = nullptr;
    aclTensor *gammaT_ = nullptr;
    aclTensor *hin = nullptr, *hPost = nullptr, *hRes = nullptr;
    aclTensor *invRms = nullptr, *hMix = nullptr, *hPre = nullptr;  // optional: keep null

    std::vector<int64_t> xShape = {bs, n, d}, phiShape = {mixHc, n * d}, alphaShape = {3},
                         biasShape = {mixHc}, hinShape = {bs, d}, hpShape = {bs, n}, hrShape = {bs, n, n};
    printf("[mhc-ex] using 4D x + gamma to mirror the known-good original example path\n");
    std::vector<int64_t> xShape4 = {1, bs, n, d}, hinShape4 = {1, bs, d}, hpShape4 = {1, bs, n},
                         hrShape4 = {1, bs, n, n}, gammaShape = {n, d};
    std::vector<float> gammaHost(static_cast<size_t>(n * d), 1.0f);
    MakeBf16TensorFromU16(xHost, xShape4, xDev, x);
    MakeTensor(phiHost, phiShape, ACL_FLOAT, phiDev, phi);
    MakeTensor(alphaHost, alphaShape, ACL_FLOAT, alphaDev, alpha);
    MakeTensor(biasHost, biasShape, ACL_FLOAT, biasDev, bias);
    void *gammaDev2 = nullptr;
    MakeTensor(gammaHost, gammaShape, ACL_FLOAT, gammaDev2, gammaT_);
    gamma = gammaT_;

    auto MakeOutFp32 = [&](const std::vector<int64_t> &shape, void *&dev, aclTensor *&t) {
        size_t bytes = 1;
        for (auto dd : shape) { bytes *= static_cast<size_t>(dd); }
        bytes *= 4;
        aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        t = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, strides.data(), shape.size(), dev);
    };
    auto MakeOutBf16 = [&](const std::vector<int64_t> &shape, void *&dev, aclTensor *&t) {
        size_t bytes = 1;
        for (auto dd : shape) { bytes *= static_cast<size_t>(dd); }
        bytes *= 2;
        aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        t = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, strides.data(), 0,
                            aclFormat::ACL_FORMAT_ND, strides.data(), shape.size(), dev);
    };
    MakeOutBf16(hinShape4, hinDev, hin);
    MakeOutFp32(hpShape4, hpDev, hPost);
    MakeOutFp32(hrShape4, hrDev, hRes);
    // A3 path segfaults when optional outputs are null; provide all of them (original example does too)
    void *irDev = nullptr, *hmDev = nullptr, *hp2Dev = nullptr;
    MakeOutFp32({1, bs}, irDev, invRms);
    MakeOutFp32({1, bs, mixHc}, hmDev, hMix);
    MakeOutFp32({1, bs, n}, hp2Dev, hPre);

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    printf("[mhc-ex] before getWs\n"); fflush(stdout);
    ret = aclnnMhcPreGetWorkspaceSize(x, phi, alpha, bias, gamma, 1e-6 /*normEps*/, 1e-6 /*hcEps*/,
                                      hin, hPost, hRes, invRms, hMix, hPre, &wsSize, &exec);
    printf("[mhc-ex] getWs rc=%d ws=%lu\n", (int)ret, (unsigned long)wsSize); fflush(stdout);
    CHECK_RET(ret == ACL_SUCCESS, "GetWorkspaceSize");
    void *ws = nullptr;
    if (wsSize > 0) { aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_NORMAL_ONLY); }
    ret = aclnnMhcPre(ws, wsSize, exec, strm);
    printf("[mhc-ex] run rc=%d\n", (int)ret); fflush(stdout);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnMhcPre");
    aclrtSynchronizeStream(strm);
    printf("[mhc-ex] synced\n"); fflush(stdout);

    std::vector<uint16_t> hinOut(static_cast<size_t>(bs * d));
    std::vector<float> hpOut(static_cast<size_t>(bs * n));
    std::vector<float> hrOut(static_cast<size_t>(bs * n * n));
    aclrtMemcpy(hinOut.data(), hinOut.size() * 2, hinDev, hinOut.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hpOut.data(), hpOut.size() * 4, hpDev, hpOut.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hrOut.data(), hrOut.size() * 4, hrDev, hrOut.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST);

    FILE *fout = fopen(outPath, "wb");
    WriteI64(fout, 0);
    WriteI64(fout, bs); WriteI64(fout, n); WriteI64(fout, d);
    fwrite(hinOut.data(), 2, hinOut.size(), fout);
    fwrite(hpOut.data(), 4, hpOut.size(), fout);
    fwrite(hrOut.data(), 4, hrOut.size(), fout);
    fclose(fout);
    printf("[mhc-ex] outputs dumped to %s\nDONE\n", outPath);
    return 0;
}
