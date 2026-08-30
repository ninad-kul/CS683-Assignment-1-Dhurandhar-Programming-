// conv_unroll.cpp  STAGE 2: LOOP UNROLLING

#include "convolution.h"
#include <cstring> 

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // initialize to 0
    std::memset(out, 0, H * W * sizeof(float));

    // made the outside loop for kernel
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            float w = ker[ky * K + kx];

            // inner loop for image (spatial concious)
            for (int oy = 0; oy < H; ++oy) {
                int out_row = oy * W;
                int in_row = (oy + ky) * in_stride + kx;

                // inner loop - 4 unrolling
                for (int ox = 0; ox < W; ox += 4) {
                    out[out_row + ox + 0] += w * in[in_row + ox + 0];
                    out[out_row + ox + 1] += w * in[in_row + ox + 1];
                    out[out_row + ox + 2] += w * in[in_row + ox + 2];
                    out[out_row + ox + 3] += w * in[in_row + ox + 3];
                }
            }
        }
    }
}
