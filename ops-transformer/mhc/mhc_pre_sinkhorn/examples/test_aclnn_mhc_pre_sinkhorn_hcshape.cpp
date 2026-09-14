/**
 * hc_pre-equivalent example for mhc_pre_sinkhorn: same inputs as vllm hc_pre's bs=512 case
 * (reads mhc_input_bs512.bin produced by run_golden.py --dump-input, seed 1024),
 * runs aclnnMhcPreSinkhorn with matching attrs (hc_mult=4, num_iters=20, eps=1e-6),
 * dumps hin/hPost/hRes for comparison against hc_pre's y/post/comb_frag.
 *
 * Output bin layout: [code=0:int64][bs,n,d:int64x3][hin bf16][hPost fp32][hRes fp32 (n*n per row)]
 */
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn_mhc_pre_sinkhorn.h"

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

int MakeTensorBf16(const std::vector<uint16_t> &host, const std::vector<int64_t> &shape,
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
    t = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, strides.data(), 0,
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
    const char *inPath = (argc > 1) ? argv[1] : "mhc_input_bs512.bin";
    const char *outPath = (argc > 2) ? argv[2] : "/root/HcPre/mhcs_output_bs512.bin";
    aclnnStatus ret = ACL_SUCCESS;

    FILE *fin = fopen(inPath, "rb");
    if (fin == nullptr) { printf("cannot open %s\n", inPath); return -1; }
    // header: (b, bs, n, d) 4-field (new) or (bs, n, d) 3-field (legacy, b=1)
    int64_t b = 1, bs = 0, n = 0, d = 0;
    {
        std::vector<int64_t> hdr;
        for (int i = 0; i < 4; ++i) {
            int64_t v = 0;
            size_t r = fread(&v, 8, 1, fin);
            if (r != 1) { printf("bad header\n"); return -1; }
            hdr.push_back(v);
        }
        // heuristic: legacy 3-field header has n=4 (small), 4-field has b,bs,n=4,d=4096
        if (hdr[2] == 4 && hdr[3] >= 16 && hdr[1] >= 1 && hdr[0] >= 1) {
            b = hdr[0]; bs = hdr[1]; n = hdr[2]; d = hdr[3];
        } else {
            // legacy: bs,n,d were the first 3; push back the 4th read (it is part of data? no —
            // legacy header is exactly 3 fields, so the 4th read consumed x data. Reopen.
            fclose(fin);
            fin = fopen(inPath, "rb");
            int64_t v0 = 0, v1 = 0, v2 = 0;
            fread(&v0, 8, 1, fin); fread(&v1, 8, 1, fin); fread(&v2, 8, 1, fin);
            b = 1; bs = v0; n = v1; d = v2;
        }
    }
    int64_t mixHc = n * n + 2 * n;
    int64_t numIters = 20;
    std::vector<uint16_t> xHost(static_cast<size_t>(b * bs * n * d));
    fread(xHost.data(), 2, xHost.size(), fin);
    std::vector<float> phiHost(static_cast<size_t>(mixHc * n * d));
    fread(phiHost.data(), 4, phiHost.size(), fin);
    std::vector<float> alphaHost(3);
    fread(alphaHost.data(), 4, 3, fin);
    std::vector<float> biasHost(static_cast<size_t>(mixHc));
    fread(biasHost.data(), 4, biasHost.size(), fin);
    fclose(fin);
    printf("[mhcs-ex] b=%ld bs=%ld n=%ld d=%ld mix=%ld iters=%ld\n", (long)b, (long)bs, (long)n, (long)d,
           (long)mixHc, (long)numIters);
    fflush(stdout);

    ret = static_cast<aclnnStatus>(aclInit(nullptr));
    CHECK_RET(ret == ACL_SUCCESS, "aclInit");
    aclrtContext ctx; aclrtStream strm;
    aclrtCreateContext(&ctx, 0);
    aclrtCreateStream(&strm);

    void *xDev = nullptr, *phiDev = nullptr, *alphaDev = nullptr, *biasDev = nullptr;
    void *hinDev = nullptr, *hpDev = nullptr, *hrDev = nullptr;
    void *hpreDev = nullptr, *hbnDev = nullptr, *irDev = nullptr, *soDev = nullptr, *noDev = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr;
    aclTensor *hin = nullptr, *hPost = nullptr, *hRes = nullptr;
    aclTensor *hPre = nullptr, *hcBeforeNorm = nullptr, *invRms = nullptr, *sumOut = nullptr, *normOut = nullptr;

    // NOTE: kernel 4D (b>1) path produces wrong results (verified b=2 vs golden);
    // fold to (1, b*bs, n, d) which is mathematically identical and known-good.
    std::vector<int64_t> xShape = {1, b * bs, n, d}, phiShape = {mixHc, n * d},
                         alphaShape = {3}, biasShape = {mixHc};
    std::vector<int64_t>                          hinShape = {1, b * bs, d}, hpShape = {1, b * bs, n}, hrShape = {1, b * bs, n * n},
                         hpreShape = {1, b * bs, n}, hbnShape = {1, b * bs, mixHc}, irShape = {1, b * bs, 1},
                         soShape = {numIters * 2, 1, b * bs, n}, noShape = {numIters * 2, 1, b * bs, n, n};

    MakeTensorBf16(xHost, xShape, xDev, x);
    MakeTensorF32(phiHost, phiShape, phiDev, phi);
    MakeTensorF32(alphaHost, alphaShape, alphaDev, alpha);
    MakeTensorF32(biasHost, biasShape, biasDev, bias);
    MakeOut(hinShape, ACL_BF16, 2, hinDev, hin);
    MakeOut(hpShape, ACL_FLOAT, 4, hpDev, hPost);
    MakeOut(hrShape, ACL_FLOAT, 4, hrDev, hRes);
    MakeOut(hpreShape, ACL_FLOAT, 4, hpreDev, hPre);
    MakeOut(hbnShape, ACL_FLOAT, 4, hbnDev, hcBeforeNorm);
    MakeOut(irShape, ACL_FLOAT, 4, irDev, invRms);
    MakeOut(soShape, ACL_FLOAT, 4, soDev, sumOut);
    MakeOut(noShape, ACL_FLOAT, 4, noDev, normOut);

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    printf("[mhcs-ex] before getWs\n"); fflush(stdout);
    ret = aclnnMhcPreSinkhornGetWorkspaceSize(x, phi, alpha, bias,
                                              n /*hcMult*/, numIters, 1e-6 /*hcEps*/, 1e-6 /*normEps*/,
                                              false /*needBackward*/,
                                              hin, hPost, hRes, hPre, hcBeforeNorm, invRms, sumOut, normOut,
                                              &wsSize, &exec);
    printf("[mhcs-ex] getWs rc=%d ws=%lu\n", (int)ret, (unsigned long)wsSize); fflush(stdout);
    CHECK_RET(ret == ACL_SUCCESS, "GetWorkspaceSize");

    void *ws = nullptr;
    if (wsSize > 0) { aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_NORMAL_ONLY); }
    ret = aclnnMhcPreSinkhorn(ws, wsSize, exec, strm);
    printf("[mhcs-ex] run rc=%d\n", (int)ret); fflush(stdout);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnMhcPreSinkhorn");
    aclrtSynchronizeStream(strm);
    printf("[mhcs-ex] synced\n"); fflush(stdout);

    int64_t totalBs = b * bs;
    std::vector<uint16_t> hinOut(static_cast<size_t>(totalBs * d));
    std::vector<float> hpOut(static_cast<size_t>(totalBs * n));
    std::vector<float> hrOut(static_cast<size_t>(totalBs * n * n));
    aclrtMemcpy(hinOut.data(), hinOut.size() * 2, hinDev, hinOut.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hpOut.data(), hpOut.size() * 4, hpDev, hpOut.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(hrOut.data(), hrOut.size() * 4, hrDev, hrOut.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST);

    FILE *fout = fopen(outPath, "wb");
    WriteI64(fout, 0);
    WriteI64(fout, totalBs); WriteI64(fout, n); WriteI64(fout, d);  // out bin header: flat (b*bs, n, d)
    fwrite(hinOut.data(), 2, hinOut.size(), fout);
    fwrite(hpOut.data(), 4, hpOut.size(), fout);
    fwrite(hrOut.data(), 4, hrOut.size(), fout);
    fclose(fout);
    printf("[mhcs-ex] outputs dumped to %s\nDONE\n", outPath);
    return 0;
}
