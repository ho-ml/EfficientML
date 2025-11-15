#include "conv.h"
#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <algorithm>
#include <xmmintrin.h>

#define NUM_THREAD 4
#define BLOCK_SIZE 32

struct thread_args {
    const float *A;
    const float *B;
    float *C;
    int start_i, end_i;
    int M, N, K;
};

namespace conv {
    void transpose_b(const float *B, float *BT, int K, int N) {
        for (int k = 0; k < K; k++) {
            for (int n = 0; n < N; n++) {
                BT[n * K + k] = B[k * N + n];
            }
        }
    }

    inline void gemm_kernel(
        const float *a, const float *b, float *c,
        int block_m, int block_n, int K, int N
    ) {
        for (int i = 0; i < block_m; i++) {
            // unrolling
            for (int j = 0; j < block_n; j += 4) {
                // excpetion (block_n % 4 != 0)
                if (j + 3 >= block_n) {
                    for (int jr = j; jr < block_n; jr++) {
                        float acc = 0.0f;

                        for (int k = 0; k < K; k++)
                            acc += a[i * K + k] * b[jr * K + k];
                        
                        c[i * N + jr] = acc;
                    }

                    break;
                }

                // Intel SIMD
                float Cij0[4] = {}, Cij1[4] = {}, Cij2[4] = {}, Cij3[4] = {};
                int k = 0;

                __m128 *acc0 = (__m128 *)Cij0;
                __m128 *acc1 = (__m128 *)Cij1;
                __m128 *acc2 = (__m128 *)Cij2;
                __m128 *acc3 = (__m128 *)Cij3;
                
                // Unrolling
                for (k = 0; k + 3 < K; k += 4) {
                    __m128 val;
                    __m128 aik = _mm_load_ps(&a[i * K + k]);

                    val = _mm_mul_ps(aik, _mm_load_ps(&b[j * K + k]));
                    *acc0 = _mm_add_ps(*acc0, val);

                    val = _mm_mul_ps(aik, _mm_load_ps(&b[(j + 1) * K + k]));
                    *acc1 = _mm_add_ps(*acc1, val);

                    val = _mm_mul_ps(aik, _mm_load_ps(&b[(j + 2) * K + k]));
                    *acc2 = _mm_add_ps(*acc2, val);

                    val = _mm_mul_ps(aik, _mm_load_ps(&b[(j + 3) * K + k]));
                    *acc3 = _mm_add_ps(*acc3, val);

                }
                
                // exception
                for (; k < K; k++) {
                    float aik = a[i * K + k];
                    Cij0[0] += aik * b[j * K + k];
                    Cij1[0] += aik * b[(j + 1) * K + k];
                    Cij2[0] += aik * b[(j + 2) * K + k];
                    Cij3[0] += aik * b[(j + 3) * K + k];
                }

                // output
                c[i * N + j] = Cij0[0] + Cij0[1] + Cij0[2] + Cij0[3];
                c[i * N + j + 1] = Cij1[0] + Cij1[1] + Cij1[2] + Cij1[3];
                c[i * N + j + 2] = Cij2[0] + Cij2[1] + Cij2[2] + Cij2[3];
                c[i * N + j + 3] = Cij3[0] + Cij3[1] + Cij3[2] + Cij3[3];

            }
        }

    }
        
    void *thread_func(void *args) {
        // initialize
        struct thread_args *mat_args = (struct thread_args*)args;
        const float *A = mat_args->A;
        const float *B = mat_args->B;
        float *C = mat_args->C;
        int M = mat_args->M, N = mat_args->N, K = mat_args->K;
        int start_i = mat_args->start_i, end_i = mat_args->end_i;
        int tm, tn, endm, endn;

        // tiling
        for (tm = start_i; tm < end_i; tm += BLOCK_SIZE) {
            endm = std::min(BLOCK_SIZE, end_i - tm);

            for (tn = 0; tn < N; tn += BLOCK_SIZE) {
                endn = std::min(BLOCK_SIZE, N - tn);

                gemm_kernel(
                    &A[tm * K], &B[tn * K], &C[tm * N + tn],
                    endm, endn, K, N
                );
            }
        }

        return NULL;
    }

    void gemm(const float *A, const float *B, float *C, int M, int N, int K) {
        // initialize
        int j;
        int num_thread = NUM_THREAD;
        int nrows = (M + num_thread - 1) / num_thread;
        pthread_t thread_pool[num_thread];
        struct thread_args thread_args[num_thread];
        
        // transpose B (need to optimize)
        float *BT = new float[N * K];
        transpose_b(B, BT, K, N);

        // create threads
        for (j = 0; j < num_thread; j++) {
            thread_args[j].start_i = j * nrows;
            thread_args[j].end_i = std::min((j + 1) * nrows, M);

            thread_args[j].A = A;
            thread_args[j].B = BT;
            thread_args[j].C = C;
            thread_args[j].M = M;
            thread_args[j].N = N;
            thread_args[j].K = K;

            if (thread_args[j].start_i < thread_args[j].end_i)
                pthread_create(&thread_pool[j], NULL, thread_func, &thread_args[j]);
            else
                thread_pool[j] = 0;
        }

        // join threads
        for (j = 0; j < num_thread; j++) {
            if (thread_pool[j] != 0)
                pthread_join(thread_pool[j], NULL);
        }

        delete[] BT;
    }
}