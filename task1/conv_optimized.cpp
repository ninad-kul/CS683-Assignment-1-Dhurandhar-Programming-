// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER

#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    int pad = K / 2;
    int in_stride = W + 2 * pad; // input is padded

    for (int y = 0; y < H; ++y) {
        int x = 0;
        
      
        for (; x <= W - 32; x += 32) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
          
                int in_offset = (y + ky) * in_stride;
                const float* ker_row = &ker[ky * K];
                
                for (int kx = 0; kx < K; ++kx) {
                    __m256 k_vec = _mm256_set1_ps(ker_row[kx]);
                    const float* in_ptr = &in[in_offset + x + kx];

       
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(in_ptr), k_vec, acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(in_ptr + 8), k_vec, acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(in_ptr + 16), k_vec, acc2);
                    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(in_ptr + 24), k_vec, acc3);
                }
            }
            
            float* out_ptr = &out[y * W + x];
            _mm256_storeu_ps(out_ptr, acc0);
            _mm256_storeu_ps(out_ptr + 8, acc1);
            _mm256_storeu_ps(out_ptr + 16, acc2);
            _mm256_storeu_ps(out_ptr + 24, acc3);
        }
        
        
        for (; x <= W - 8; x += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                int in_offset = (y + ky) * in_stride;
                for (int kx = 0; kx < K; ++kx) {
                    __m256 k_vec = _mm256_set1_ps(ker[ky * K + kx]);
                    __m256 img = _mm256_loadu_ps(&in[in_offset + x + kx]);
                    acc = _mm256_fmadd_ps(img, k_vec, acc);
                }
            }
            _mm256_storeu_ps(&out[y * W + x], acc);
        }
        

    }
}
