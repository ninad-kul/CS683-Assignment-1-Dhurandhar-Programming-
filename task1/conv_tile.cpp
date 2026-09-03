// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker, int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // 32x256 tile uses ~34KB memory, fit in L1/L2 
    // reduces outer loop iterations
    constexpr int TILE_H = 32;
    constexpr int TILE_W = 256;

    for (int by = 0; by < H; by += TILE_H) {
        for (int bx = 0; bx < W; bx += TILE_W) {

            // Process output pixels within the tile
            for (int oy = by; oy < by + TILE_H; ++oy) {
                float* out_row = &out[oy * W];
                
                for (int ox = bx; ox < bx + TILE_W; ++ox) {
                    float acc = 0.0f;
                    
                    // row pointer math outside the loop
                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_row = &in[(oy + ky) * in_stride];
                        const float* ker_row = &ker[ky * K];
                        
                        for (int kx = 0; kx < K; ++kx) {
                            acc += in_row[ox + kx] * ker_row[kx];
                        }
                    }
                    out_row[ox] = acc;
                }
            }
        }
    }
}
