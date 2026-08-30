// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.

#include "convolution.h"
#include <cstring> // for using std::memset

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
                    
const int p = K / 2;
    const int in_stride = W + 2 * p;

    std::memset(out, 0, H * W * sizeof(float));

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            
            float w = ker[ky * K + kx];

            for (int oy = 0; oy < H; ++oy) {
                for (int ox = 0; ox < W; ++ox) {
                    out[oy * W + ox] += w * in[(oy + ky) * in_stride + (ox + kx)];
                }
            }
        }
    }
}

