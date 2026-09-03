// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics

#include <immintrin.h>
#include "matmul.h"

// function to horizontally sum 8 float AVX2 register into 1 float
inline float hsum_avx2(__m256 v) {
    // split 256 bit vector into two 128 bit vectors and add them
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 sum   = _mm_add_ps(vlow, vhigh);
    

    __m128 half  = _mm_movehl_ps(sum, sum);
    sum          = _mm_add_ps(sum, half);
    
    __m128 final = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
    return _mm_cvtss_f32(final);
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    
    // We register tile C into 2x4 blocks (M_r = 2, N_r = 4)
    // This requires 8 accumulator registers, 2 registers for A, and 4 registers for B,
    // now we have 14 registers used (AVX2 has 16 YMM registers total, avoiding register spilling)
    
    int i = 0;
    for (; i <= M - 2; i += 2) {
        int j = 0;
        for (; j <= N - 4; j += 4) {
            
            // Initialize 2x4 tile accumulators to zero
            __m256 c00 = _mm256_setzero_ps();
            __m256 c01 = _mm256_setzero_ps();
            __m256 c02 = _mm256_setzero_ps();
            __m256 c03 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps();
            __m256 c11 = _mm256_setzero_ps();
            __m256 c12 = _mm256_setzero_ps();
            __m256 c13 = _mm256_setzero_ps();

            // Pointers to the start of the 2 rows of A and 4 rows (columns) of B
            const float* a0_ptr = A + static_cast<long>(i) * lda;
            const float* a1_ptr = A + static_cast<long>(i + 1) * lda;
            const float* b0_ptr = B + static_cast<long>(j) * ldb;
            const float* b1_ptr = B + static_cast<long>(j + 1) * ldb;
            const float* b2_ptr = B + static_cast<long>(j + 2) * ldb;
            const float* b3_ptr = B + static_cast<long>(j + 3) * ldb;

            int p = 0;
            // 8 wide FMA loop over K
            for (; p <= K - 8; p += 8) {
                // Load 8 elements from the two rows of A
                __m256 a0 = _mm256_loadu_ps(a0_ptr + p);
                __m256 a1 = _mm256_loadu_ps(a1_ptr + p);
                
                // Load 8 elements from the four rows of B
                __m256 b0 = _mm256_loadu_ps(b0_ptr + p);
                __m256 b1 = _mm256_loadu_ps(b1_ptr + p);
                __m256 b2 = _mm256_loadu_ps(b2_ptr + p);
                __m256 b3 = _mm256_loadu_ps(b3_ptr + p);

                // Accumulate dot products using FMA (a * b + c)
                c00 = _mm256_fmadd_ps(a0, b0, c00);
                c01 = _mm256_fmadd_ps(a0, b1, c01);
                c02 = _mm256_fmadd_ps(a0, b2, c02);
                c03 = _mm256_fmadd_ps(a0, b3, c03);

                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c11 = _mm256_fmadd_ps(a1, b1, c11);
                c12 = _mm256_fmadd_ps(a1, b2, c12);
                c13 = _mm256_fmadd_ps(a1, b3, c13);
            }
            
            // Horizontally sum 8 wide SIMD registers to scalar partial sums
            float acc00 = hsum_avx2(c00);
            float acc01 = hsum_avx2(c01);
            float acc02 = hsum_avx2(c02);
            float acc03 = hsum_avx2(c03);
            
            float acc10 = hsum_avx2(c10);
            float acc11 = hsum_avx2(c11);
            float acc12 = hsum_avx2(c12);
            float acc13 = hsum_avx2(c13);

            // Cleanup loop for the K dimension (safety if K is not a multiple of 8)
            for (; p < K; ++p) {
                float a0_v = a0_ptr[p];
                float a1_v = a1_ptr[p];
                float b0_v = b0_ptr[p];
                float b1_v = b1_ptr[p];
                float b2_v = b2_ptr[p];
                float b3_v = b3_ptr[p];

                acc00 += a0_v * b0_v;
                acc01 += a0_v * b1_v;
                acc02 += a0_v * b2_v;
                acc03 += a0_v * b3_v;

                acc10 += a1_v * b0_v;
                acc11 += a1_v * b1_v;
                acc12 += a1_v * b2_v;
                acc13 += a1_v * b3_v;
            }

            // Write computed 2x4 block out to C
            C[static_cast<long>(i)     * ldc + j]     = acc00;
            C[static_cast<long>(i)     * ldc + j + 1] = acc01;
            C[static_cast<long>(i)     * ldc + j + 2] = acc02;
            C[static_cast<long>(i)     * ldc + j + 3] = acc03;

            C[static_cast<long>(i + 1) * ldc + j]     = acc10;
            C[static_cast<long>(i + 1) * ldc + j + 1] = acc11;
            C[static_cast<long>(i + 1) * ldc + j + 2] = acc12;
            C[static_cast<long>(i + 1) * ldc + j + 3] = acc13;
        }

        // safe edges if N is not a multiple of 4
        for (; j < N; ++j) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            const float* a0_ptr = A + static_cast<long>(i) * lda;
            const float* a1_ptr = A + static_cast<long>(i + 1) * lda;
            const float* b_ptr = B + static_cast<long>(j) * ldb;

            int p = 0;
            __m256 c0 = _mm256_setzero_ps();
            __m256 c1 = _mm256_setzero_ps();
            
            for (; p <= K - 8; p += 8) {
                __m256 a0 = _mm256_loadu_ps(a0_ptr + p);
                __m256 a1 = _mm256_loadu_ps(a1_ptr + p);
                __m256 b0 = _mm256_loadu_ps(b_ptr + p);
                c0 = _mm256_fmadd_ps(a0, b0, c0);
                c1 = _mm256_fmadd_ps(a1, b0, c1);
            }
            
            acc0 = hsum_avx2(c0);
            acc1 = hsum_avx2(c1);

            for (; p < K; ++p) {
                acc0 += a0_ptr[p] * b_ptr[p];
                acc1 += a1_ptr[p] * b_ptr[p];
            }

            C[static_cast<long>(i) * ldc + j] = acc0;
            C[static_cast<long>(i + 1) * ldc + j] = acc1;
        }
    }
    
    // safe edges when M is not a multiple of 2
    for (; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a_ptr = A + static_cast<long>(i) * lda;
            const float* b_ptr = B + static_cast<long>(j) * ldb;

            int p = 0;
            __m256 c = _mm256_setzero_ps();
            
            for (; p <= K - 8; p += 8) {
                __m256 a = _mm256_loadu_ps(a_ptr + p);
                __m256 b = _mm256_loadu_ps(b_ptr + p);
                c = _mm256_fmadd_ps(a, b, c);
            }
            
            acc = hsum_avx2(c);

            for (; p < K; ++p) {
                acc += a_ptr[p] * b_ptr[p];
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
