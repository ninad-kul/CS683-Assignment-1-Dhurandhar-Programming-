// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics

#include <immintrin.h>
#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker, int H, int W, int K) {
    int pad = K / 2;
    int stride = W + 2 * pad; // True stride of the zero-padded halo buffer

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; x += 8) {
            __m256 acc = _mm256_setzero_ps();
            
            for (int ky = 0; ky < K; ++ky) {
                int in_row_offset = (y + ky) * stride;
                const float* ker_row = &ker[ky * K];
                
                for (int kx = 0; kx < K; ++kx) {
                    __m256 k_vec = _mm256_set1_ps(ker_row[kx]);
                    __m256 img = _mm256_loadu_ps(&in[in_row_offset + x + kx]);
                    acc = _mm256_fmadd_ps(img, k_vec, acc);
                }
            }
            
            _mm256_storeu_ps(&out[y * W + x], acc);
        }
    }
}
