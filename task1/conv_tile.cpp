// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {

   const int p = K / 2;
    const int in_stride = W + 2 * p;

    constexpr int TILE_H = 32;
    constexpr int TILE_W = 32;

    for (int by = 0; by < H; by += TILE_H) {
        for (int bx = 0; bx < W; bx += TILE_W) {

            // processing output pixel within this tile completely
            for (int oy = by; oy < by + TILE_H; ++oy) {
                for (int ox = bx; ox < bx + TILE_W; ++ox) {
                    
                    float acc = 0.0f; // register accumulation (for faster operation)
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = acc; // single store per pixel
                }
            }

        }
    }
}
