// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker, int H, int W, int K) {
    int pad = K / 2;
    int stride = W + 2 * pad;

    for (int y = 0; y < H; ++y) {
        float* out_row = &out[y * W];
        
        for (int x = 0; x < W; ++x) {
            float sum = 0.0f;
            
            // Reorder
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = &in[(y + ky) * stride + x];
                const float* ker_row = &ker[ky * K];
                
                for (int kx = 0; kx < K; ++kx) {
                    sum += in_row[kx] * ker_row[kx];
                }
            }
            out_row[x] = sum;
        }
    }
}

