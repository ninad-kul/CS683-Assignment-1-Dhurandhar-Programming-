#include "matmul.h"
#include <xmmintrin.h> // for _mm_prefetch


const int PREFETCH_DISTANCE = 128; // 16, 32, 64, 128
const _mm_hint PREFETCH_HINT = _MM_HINT_T0; 


void matmul_prefetch(const float* A, const float* B, float* C, 
                      int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            
            for (int p = 0; p < K; ++p) {
                // Prefetch data ahead of the current index
                if (p + PREFETCH_DISTANCE < K) {
                    _mm_prefetch(reinterpret_cast<const char*>(a + p + PREFETCH_DISTANCE), PREFETCH_HINT);
                    _mm_prefetch(reinterpret_cast<const char*>(b + p + PREFETCH_DISTANCE), PREFETCH_HINT);
                }
                
                acc += a[p] * b[p];
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
