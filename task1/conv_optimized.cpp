// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    int pad = K / 2;

    for (int y = 0; y < H; ++y) {
        // fast Path: check if the entire row is safely away from boundaries
        if (y >= pad && y < H - pad) {
            int x = 0;
            
            // left edge Scalar
            for (; x < pad; ++x) {
                float sum = 0.0f;
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        int img_x = x + kx - pad;
                        if (img_x >= 0 && img_x < W) {
                            sum += in[(y + ky - pad) * W + img_x] * ker[ky * K + kx];
                        }
                    }
                }
                out[y * W + x] = sum;
            }
            
            // main loop: register-unrolling SIMD (32 pixels)
            int safe_W = W - pad;
            for (; x <= safe_W - 32; x += 32) {
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();

                for (int ky = 0; ky < K; ++ky) {
                    int in_offset = (y + ky - pad) * W;
                    const float* ker_row = &ker[ky * K];
                    
                    for (int kx = 0; kx < K; ++kx) {
                        __m256 k_vec = _mm256_set1_ps(ker_row[kx]);
                        
                        int base_img_x = x + kx - pad;
                        const float* in_ptr = &in[in_offset + base_img_x];

                        // load 32 pixels and execute 4 multiply-adds independently
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
            
            // remainder: Standard SIMD (8 pixels)
            for (; x <= safe_W - 8; x += 8) {
                __m256 acc = _mm256_setzero_ps();
                for (int ky = 0; ky < K; ++ky) {
                    int in_offset = (y + ky - pad) * W;
                    for (int kx = 0; kx < K; ++kx) {
                        __m256 k_vec = _mm256_set1_ps(ker[ky * K + kx]);
                        __m256 img = _mm256_loadu_ps(&in[in_offset + x + kx - pad]);
                        acc = _mm256_fmadd_ps(img, k_vec, acc);
                    }
                }
                _mm256_storeu_ps(&out[y * W + x], acc);
            }
            
            // right edge scalar 
            for (; x < W; ++x) {
                float sum = 0.0f;
                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {
                        int img_x = x + kx - pad;
                        if (img_x >= 0 && img_x < W) {
                            sum += in[(y + ky - pad) * W + img_x] * ker[ky * K + kx];
                        }
                    }
                }
                out[y * W + x] = sum;
            }
            
        } else {
            // slow path: top and bottom edge rows
            for (int x = 0; x < W; ++x) {
                float sum = 0.0f;
                for (int ky = 0; ky < K; ++ky) {
                    int img_y = y + ky - pad;
                    if (img_y >= 0 && img_y < H) {
                        for (int kx = 0; kx < K; ++kx) {
                            int img_x = x + kx - pad;
                            if (img_x >= 0 && img_x < W) {
                                sum += in[img_y * W + img_x] * ker[ky * K + kx];
                            }
                        }
                    }
                }
                out[y * W + x] = sum;
            }
        }
    }
}
