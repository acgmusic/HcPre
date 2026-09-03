"""Reproduce HcPre stage-1 tiling loop counts per tiling.cpp formulas (A3: 24 AIC)."""
import math

AIC = 24
M_L1_MAX = 256
K_SPLIT_BASE = 256
A_L1_SIZE = 128 * 256
K_L1_MAX = 1024
CV_LOOP_K = 1024
BLOCK_CUBE = 16


def ceil_div(a, b):
    return -(-a // b)


def round_up(a, b):
    return ceil_div(a, b) * b


def stage1(bs, k=16384):
    m_dim = min(AIC, ceil_div(bs, M_L1_MAX))          # tiling.cpp:281
    single_core_m = round_up(ceil_div(bs, m_dim), BLOCK_CUBE)
    k_dim = AIC // m_dim                               # :283
    split_k = round_up(ceil_div(k, k_dim), K_SPLIT_BASE)  # :284
    m_l1 = min(M_L1_MAX, single_core_m)                # :289
    k_l1 = min(A_L1_SIZE // m_l1, K_L1_MAX) // 128 * 128  # :291
    cv_loops = ceil_div(split_k, CV_LOOP_K)            # kernel :104-105
    m_l1_loops = ceil_div(single_core_m, m_l1)         # kernel :100-101
    l1_loops = ceil_div(split_k, k_l1)                 # cube_compute :279
    return m_dim, k_dim, split_k, m_l1, k_l1, cv_loops, m_l1_loops, l1_loops


print(f"{'bs':>6} {'mDim':>4} {'kDim':>4} {'每核K段':>7} {'mL1':>4} {'kL1':>4} "
      f"{'cv轮数':>6} {'M-L1轮数':>8} {'L1轮数':>6}")
for bs in [2, 240, 256, 512, 600, 1024, 1200, 2048, 3072, 6144, 12288, 24576]:
    m, kd, sk, ml1, kl1, cv, ml, l1 = stage1(bs)
    print(f"{bs:>6} {m:>4} {kd:>4} {sk:>7} {ml1:>4} {kl1:>4} {cv:>6} {ml:>8} {l1:>6}")
