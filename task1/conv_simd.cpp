// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics

#include <immintrin.h>
#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {

int pad = K / 2;

    for (int y = 0; y < H; ++y) {
        // as W is divisible by 8 hence I kept step 8
        for (int x = 0; x < W; x += 8) {

            // from the inner loop
            if (y >= pad && y < H - pad && x >= pad && x + 7 < W - pad) {
                
                // Initialize accumulator for 8 pixels
                __m256 acc = _mm256_setzero_ps();
                
                for (int ky = 0; ky < K; ++ky) {
                    int img_y = y + ky - pad;
                    int in_offset = img_y * W; // precompute row offset
                    
                    for (int kx = 0; kx < K; ++kx) {
                        int img_x = x + kx - pad;
                        
                        // load 8 contiguous pixels
                        __m256 img_vec = _mm256_loadu_ps(&in[in_offset + img_x]);
                        
                        // provide 1 kernel weight to 8 lanes
                        __m256 ker_vec = _mm256_set1_ps(ker[ky * K + kx]);
                        
                        // multiple add : acc = (img * ker) + acc
                        acc = _mm256_fmadd_ps(img_vec, ker_vec, acc);
                    }
                }
                
                // store 8 finished pixels to memory
                _mm256_storeu_ps(&out[y * W + x], acc);
                
            } else {
                // handle zero-padding boundaries safely without vector masks
                for (int i = 0; i < 8; ++i) {
                    int curr_x = x + i;
                    float sum = 0.0f;
                    
                    for (int ky = 0; ky < K; ++ky) {
                        int img_y = y + ky - pad;
                        if (img_y >= 0 && img_y < H) {
                            for (int kx = 0; kx < K; ++kx) {
                                int img_x = curr_x + kx - pad;
                                if (img_x >= 0 && img_x < W) {
                                    sum += in[img_y * W + img_x] * ker[ky * K + kx];
                                }
                            }
                        }
                    }
                    out[y * W + curr_x] = sum;
                }
            }
        }
    }
}
