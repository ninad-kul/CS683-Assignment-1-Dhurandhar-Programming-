// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.


#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

// Horizontal sum helper for 256 bit AVX2 registers
inline float hsum_avx2(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum   = _mm_add_ps(vlow, vhigh);
    __m128 half  = _mm_movehl_ps(sum, sum);
    sum          = _mm_add_ps(sum, half);
    __m128 final = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
    return _mm_cvtss_f32(final);
}

// for My PC Intel Core i5 7300U L2 Cache (256 KB per core)
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 256;

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {

    // L2 Cache Tiling Loops
    for (int i0 = 0; i0 < M; i0 += BM) {
        int i_max = std::min(i0 + BM, M);

        for (int j0 = 0; j0 < N; j0 += BN) {
            int j_max = std::min(j0 + BN, N);

            // Register Tiled Micro Kernel (2x4 Blocks)
            int i = i0;
            for (; i <= i_max - 2; i += 2) {
                int j = j0;
                for (; j <= j_max - 4; j += 4) {

                    // YMM accumulators held in registers across all K tiles
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

                    int p = 0;

                    // K Cache-Blocking
                    for (int k0 = 0; k0 < K; k0 += BK) {
                        int k_max = std::min(k0 + BK, K);

                        p = k0;
                        // 2x Unrolled Inner Loop (16 floats / 64 bytes per iteration)
                        for (; p <= k_max - 16; p += 16) {
                            // Prefetch 2 cache lines (32 floats = 128 bytes) ahead
                            _mm_prefetch((const char*)(a0_ptr + p + 32), _MM_HINT_T0);
                            _mm_prefetch((const char*)(a1_ptr + p + 32), _MM_HINT_T0);
                            _mm_prefetch((const char*)(b0_ptr + p + 32), _MM_HINT_T0);
                            _mm_prefetch((const char*)(b1_ptr + p + 32), _MM_HINT_T0);
                            _mm_prefetch((const char*)(b2_ptr + p + 32), _MM_HINT_T0);
                            _mm_prefetch((const char*)(b3_ptr + p + 32), _MM_HINT_T0);

                            // First 8 floats 
                            __m256 a0_0 = _mm256_loadu_ps(a0_ptr + p);
                            __m256 a1_0 = _mm256_loadu_ps(a1_ptr + p);
                            __m256 b0_0 = _mm256_loadu_ps(b0_ptr + p);
                            __m256 b1_0 = _mm256_loadu_ps(b1_ptr + p);
                            __m256 b2_0 = _mm256_loadu_ps(b2_ptr + p);
                            __m256 b3_0 = _mm256_loadu_ps(b3_ptr + p);

                            c00 = _mm256_fmadd_ps(a0_0, b0_0, c00);
                            c01 = _mm256_fmadd_ps(a0_0, b1_0, c01);
                            c02 = _mm256_fmadd_ps(a0_0, b2_0, c02);
                            c03 = _mm256_fmadd_ps(a0_0, b3_0, c03);

                            c10 = _mm256_fmadd_ps(a1_0, b0_0, c10);
                            c11 = _mm256_fmadd_ps(a1_0, b1_0, c11);
                            c12 = _mm256_fmadd_ps(a1_0, b2_0, c12);
                            c13 = _mm256_fmadd_ps(a1_0, b3_0, c13);

                            // Second 8 floats 
                            __m256 a0_1 = _mm256_loadu_ps(a0_ptr + p + 8);
                            __m256 a1_1 = _mm256_loadu_ps(a1_ptr + p + 8);
                            __m256 b0_1 = _mm256_loadu_ps(b0_ptr + p + 8);
                            __m256 b1_1 = _mm256_loadu_ps(b1_ptr + p + 8);
                            __m256 b2_1 = _mm256_loadu_ps(b2_ptr + p + 8);
                            __m256 b3_1 = _mm256_loadu_ps(b3_ptr + p + 8);

                            c00 = _mm256_fmadd_ps(a0_1, b0_1, c00);
                            c01 = _mm256_fmadd_ps(a0_1, b1_1, c01);
                            c02 = _mm256_fmadd_ps(a0_1, b2_1, c02);
                            c03 = _mm256_fmadd_ps(a0_1, b3_1, c03);

                            c10 = _mm256_fmadd_ps(a1_1, b0_1, c10);
                            c11 = _mm256_fmadd_ps(a1_1, b1_1, c11);
                            c12 = _mm256_fmadd_ps(a1_1, b2_1, c12);
                            c13 = _mm256_fmadd_ps(a1_1, b3_1, c13);
                        }

                        // Single 8 float SIMD residual loop
                        for (; p <= k_max - 8; p += 8) {
                            __m256 a0 = _mm256_loadu_ps(a0_ptr + p);
                            __m256 a1 = _mm256_loadu_ps(a1_ptr + p);
                            __m256 b0 = _mm256_loadu_ps(b0_ptr + p);
                            __m256 b1 = _mm256_loadu_ps(b1_ptr + p);
                            __m256 b2 = _mm256_loadu_ps(b2_ptr + p);
                            __m256 b3 = _mm256_loadu_ps(b3_ptr + p);

                            c00 = _mm256_fmadd_ps(a0, b0, c00);
                            c01 = _mm256_fmadd_ps(a0, b1, c01);
                            c02 = _mm256_fmadd_ps(a0, b2, c02);
                            c03 = _mm256_fmadd_ps(a0, b3, c03);

                            c10 = _mm256_fmadd_ps(a1, b0, c10);
                            c11 = _mm256_fmadd_ps(a1, b1, c11);
                            c12 = _mm256_fmadd_ps(a1, b2, c12);
                            c13 = _mm256_fmadd_ps(a1, b3, c13);
                        }
                    }

                    // Deferred Horizontal Reduction (executed once per 2x4 block output)
                    float acc00 = hsum_avx2(c00);
                    float acc01 = hsum_avx2(c01);
                    float acc02 = hsum_avx2(c02);
                    float acc03 = hsum_avx2(c03);

                    float acc10 = hsum_avx2(c10);
                    float acc11 = hsum_avx2(c11);
                    float acc12 = hsum_avx2(c12);
                    float acc13 = hsum_avx2(c13);

                    // Scalar loop for remaining non vectorized floats
                    for (; p < K; ++p) {
                        float a0_v = a0_ptr[p], a1_v = a1_ptr[p];
                        acc00 += a0_v * b0_ptr[p];
                        acc01 += a0_v * b1_ptr[p];
                        acc02 += a0_v * b2_ptr[p];
                        acc03 += a0_v * b3_ptr[p];

                        acc10 += a1_v * b0_ptr[p];
                        acc11 += a1_v * b1_ptr[p];
                        acc12 += a1_v * b2_ptr[p];
                        acc13 += a1_v * b3_ptr[p];
                    }

                    // Write out final results
                    C[static_cast<long>(i)     * ldc + j]     = acc00;
                    C[static_cast<long>(i)     * ldc + j + 1] = acc01;
                    C[static_cast<long>(i)     * ldc + j + 2] = acc02;
                    C[static_cast<long>(i)     * ldc + j + 3] = acc03;

                    C[static_cast<long>(i + 1) * ldc + j]     = acc10;
                    C[static_cast<long>(i + 1) * ldc + j + 1] = acc11;
                    C[static_cast<long>(i + 1) * ldc + j + 2] = acc12;
                    C[static_cast<long>(i + 1) * ldc + j + 3] = acc13;
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
