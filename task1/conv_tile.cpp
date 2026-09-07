// conv_tile.cpp  STAGE 3: CACHE TILING

#include <algorithm>
#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker, int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // 32x256 tile fits comfortably within L1/L2 caches
    constexpr int TILE_H = 32;
    constexpr int TILE_W = 256;

    for (int by = 0; by < H; by += TILE_H) {
        const int y_max = std::min(by + TILE_H, H);

        for (int bx = 0; bx < W; bx += TILE_W) {
            const int x_max = std::min(bx + TILE_W, W);

          
            for (int oy = by; oy < y_max; ++oy) {
                float* out_row = &out[oy * W];

           
                for (int ox = bx; ox < x_max; ++ox) {
                    out_row[ox] = 0.0f;
                }

     
                for (int ky = 0; ky < K; ++ky) {
                    const float* in_row = &in[(oy + ky) * in_stride];
                    const float* ker_row = &ker[ky * K];

                    for (int kx = 0; kx < K; ++kx) {
                        const float k_val = ker_row[kx];
                        const float* in_ptr = in_row + kx;

               
                        for (int ox = bx; ox < x_max; ++ox) {
                            out_row[ox] += in_ptr[ox] * k_val;
                        }
                    }
                }
            }
        }
    }
}
