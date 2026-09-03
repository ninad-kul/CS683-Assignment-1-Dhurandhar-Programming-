// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

// Horizontal sum function for 256 bit AVX2 registers
inline float hsum_avx2(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum   = _mm_add_ps(vlow, vhigh);
    __m128 half  = _mm_movehl_ps(sum, sum);
    sum          = _mm_add_ps(sum, half);
    __m128 final = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
    return _mm_cvtss_f32(final);
}

// Optimized values for My PC Intel Core i5 7300U (4 core) L1 is 128KB (32KB per core) / L2 512KB (256KB per core)
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 256;

void matmul_prefetch(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {

    // L2 cache tiling loops (i0, j0)
    for (int i0 = 0; i0 < M; i0 += BM) {
        int i_max = std::min(i0 + BM, M);

        for (int j0 = 0; j0 < N; j0 += BN) {
            int j_max = std::min(j0 + BN, N);

            // register tiling, micro kernel (2x4 Blocks)
            int i = i0;
            for (; i <= i_max - 2; i += 2) {
                int j = j0;
                for (; j <= j_max - 4; j += 4) {

                    // Accumulators held in YMM registers across K blocks
                    __m256 c00 = _mm256_setzero_ps();
                    __m256 c01 = _mm256_setzero_ps();
                    __m256 c02 = _mm256_setzero_ps();
                    __m256 c03 = _mm256_setzero_ps();
                    __m256 c10 = _mm256_setzero_ps();
                    __m256 c11 = _mm256_setzero_ps();
                    __m256 c12 = _mm256_setzero_ps();
                    __m256 c13 = _mm256_setzero_ps();

                    const float* a0_ptr = A + static_cast<long>(i) * lda;
                    const float* a1_ptr = A + static_cast<long>(i + 1) * lda;
                    const float* b0_ptr = B + static_cast<long>(j) * ldb;
                    const float* b1_ptr = B + static_cast<long>(j + 1) * ldb;
                    const float* b2_ptr = B + static_cast<long>(j + 2) * ldb;
                    const float* b3_ptr = B + static_cast<long>(j + 3) * ldb;

                    // K dimension cache blocking
                    for (int k0 = 0; k0 < K; k0 += BK) {
                        int k_max = std::min(k0 + BK, K);

                        int p = k0;
                        for (; p <= k_max - 8; p += 8) {
                            // prefetch 16 floats (64 bytes = 1 cache line) ahead into L1d
                            _mm_prefetch((const char*)(a0_ptr + p + 16), _MM_HINT_T0);
                            _mm_prefetch((const char*)(a1_ptr + p + 16), _MM_HINT_T0);
                            _mm_prefetch((const char*)(b0_ptr + p + 16), _MM_HINT_T0);
                            _mm_prefetch((const char*)(b1_ptr + p + 16), _MM_HINT_T0);
                            _mm_prefetch((const char*)(b2_ptr + p + 16), _MM_HINT_T0);
                            _mm_prefetch((const char*)(b3_ptr + p + 16), _MM_HINT_T0);

                            // Load SIMD vectors
                            __m256 a0 = _mm256_loadu_ps(a0_ptr + p);
                            __m256 a1 = _mm256_loadu_ps(a1_ptr + p);
                            __m256 b0 = _mm256_loadu_ps(b0_ptr + p);
                            __m256 b1 = _mm256_loadu_ps(b1_ptr + p);
                            __m256 b2 = _mm256_loadu_ps(b2_ptr + p);
                            __m256 b3 = _mm256_loadu_ps(b3_ptr + p);

                            // Fused Multiply-Add
                            c00 = _mm256_fmadd_ps(a0, b0, c00);
                            c01 = _mm256_fmadd_ps(a0, b1, c01);
                            c02 = _mm256_fmadd_ps(a0, b2, c02);
                            c03 = _mm256_fmadd_ps(a0, b3, c03);

                            c10 = _mm256_fmadd_ps(a1, b0, c10);
                            c11 = _mm256_fmadd_ps(a1, b1, c11);
                            c12 = _mm256_fmadd_ps(a1, b2, c12);
                            c13 = _mm256_fmadd_ps(a1, b3, c13);
                        }

                        // K fringe within tile
                        for (; p < k_max; ++p) {
                            float a0_v = a0_ptr[p], a1_v = a1_ptr[p];
                            float b0_v = b0_ptr[p], b1_v = b1_ptr[p], b2_v = b2_ptr[p], b3_v = b3_ptr[p];

                            c00[0] += a0_v * b0_v; c01[0] += a0_v * b1_v;
                            c02[0] += a0_v * b2_v; c03[0] += a0_v * b3_v;
                            c10[0] += a1_v * b0_v; c11[0] += a1_v * b1_v;
                            c12[0] += a1_v * b2_v; c13[0] += a1_v * b3_v;
                        }
                    }

                    // Write back reduced horizontal sum to matrix C
                    C[static_cast<long>(i)     * ldc + j]     = hsum_avx2(c00);
                    C[static_cast<long>(i)     * ldc + j + 1] = hsum_avx2(c01);
                    C[static_cast<long>(i)     * ldc + j + 2] = hsum_avx2(c02);
                    C[static_cast<long>(i)     * ldc + j + 3] = hsum_avx2(c03);

                    C[static_cast<long>(i + 1) * ldc + j]     = hsum_avx2(c10);
                    C[static_cast<long>(i + 1) * ldc + j + 1] = hsum_avx2(c11);
                    C[static_cast<long>(i + 1) * ldc + j + 2] = hsum_avx2(c12);
                    C[static_cast<long>(i + 1) * ldc + j + 3] = hsum_avx2(c13);
                }

                // N boundary fringe
                for (; j < j_max; ++j) {
                    float acc0 = 0.0f, acc1 = 0.0f;
                    const float* a0_ptr = A + static_cast<long>(i) * lda;
                    const float* a1_ptr = A + static_cast<long>(i + 1) * lda;
                    const float* b_ptr = B + static_cast<long>(j) * ldb;

                    for (int p = 0; p < K; ++p) {
                        acc0 += a0_ptr[p] * b_ptr[p];
                        acc1 += a1_ptr[p] * b_ptr[p];
                    }
                    C[static_cast<long>(i) * ldc + j] = acc0;
                    C[static_cast<long>(i + 1) * ldc + j] = acc1;
                }
            }

            // M boundary fringe
            for (; i < i_max; ++i) {
                for (int j = j0; j < j_max; ++j) {
                    float acc = 0.0f;
                    const float* a_ptr = A + static_cast<long>(i) * lda;
                    const float* b_ptr = B + static_cast<long>(j) * ldb;
                    for (int p = 0; p < K; ++p) {
                        acc += a_ptr[p] * b_ptr[p];
                    }
                    C[static_cast<long>(i) * ldc + j] = acc;
                }
            }
        }
    }
}
