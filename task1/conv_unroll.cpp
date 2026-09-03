// conv_unroll.cpp  STAGE 2: LOOP UNROLLING

#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker, int H, int W, int K) {
    int pad = K / 2;
    int stride = W + 2 * pad;

    for (int y = 0; y < H; ++y) {
        float* out_row = &out[y * W];
        int x = 0;

        // Unrolling - Processing 4 adjacent pixels concurrently
        for (; x <= W - 4; x += 4) {
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            float sum3 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = &in[(y + ky) * stride + x];
                const float* ker_row = &ker[ky * K];

                for (int kx = 0; kx < K; ++kx) {
                    float k_val = ker_row[kx];
                    sum0 += in_row[kx + 0] * k_val;
                    sum1 += in_row[kx + 1] * k_val;
                    sum2 += in_row[kx + 2] * k_val;
                    sum3 += in_row[kx + 3] * k_val;
                }
            }

            out_row[x + 0] = sum0;
            out_row[x + 1] = sum1;
            out_row[x + 2] = sum2;
            out_row[x + 3] = sum3;
        }

        // cleanup loop for remaining edge pixels
        for (; x < W; ++x) {
            float sum = 0.0f;
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
