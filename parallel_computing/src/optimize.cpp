#include "matmul.h"
#include <stdio.h>
#include <assert.h>
#include <pthread.h>

#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace matmul {
    inline void matmul_func(
        const float *a, const float *b, float *c,
        int col_a, int col_b, int col_c, int block_size
    ) {
        for (int i = 0; i < block_size; i++) {
            // Unrolling
            for (int j = 0; j < block_size; j+=4) {
                float Cij0[4] = {}, Cij1[4] = {}, Cij2[4] = {}, Cij3[4] = {};

                // SIMD Programming (Intel)
                #ifdef __SSE__
                
                __m128 *acc0 = (__m128 *)Cij0;
                __m128 *acc1 = (__m128 *)Cij1;
                __m128 *acc2 = (__m128 *)Cij2;
                __m128 *acc3 = (__m128 *)Cij3;

                // Unrolling
                for (int k = 0; k < col_a; k+=4) {
                    __m128 val;
                    __m128 aik = _mm_load_ps(&a[i * col_a + k]);

                    val = _mm_mul_ps(aik, _mm_load_ps(&b[j * col_b + k]));
                    *acc0 = _mm_add_ps(*acc0, val);

                    val = _mm_mul_ps(aik, _mm_load_ps(&b[(j + 1) * col_b + k]));
                    *acc1 = _mm_add_ps(*acc1, val);

                    val = _mm_mul_ps(aik, _mm_load_ps(&b[(j + 2) * col_b + k]));
                    *acc2 = _mm_add_ps(*acc2, val);

                    val = _mm_mul_ps(aik, _mm_load_ps(&b[(j + 3) * col_b + k]));
                    *acc3 = _mm_add_ps(*acc3, val);
                }

                #endif

                // SIMD Programming (Arm)
                #ifdef __ARM_NEON
                
                float32x4_t *acc0 = (float32x4_t *)Cij0;
                float32x4_t *acc1 = (float32x4_t *)Cij1;
                float32x4_t *acc2 = (float32x4_t *)Cij2;
                float32x4_t *acc3 = (float32x4_t *)Cij3;
                
                // Unrolling
                for (int k = 0; k < col_a; k+=4) {
                    float32x4_t val;
                    float32x4_t aik = vld1q_f32(&a[i * col_a + k]);

                    val = vmulq_f32(aik, vld1q_f32(&b[j * col_b + k]));
                    *acc0 = vaddq_f32(*acc0, val);

                    val = vmulq_f32(aik, vld1q_f32(&b[(j + 1) * col_b + k]));
                    *acc1 = vaddq_f32(*acc1, val);

                    val = vmulq_f32(aik, vld1q_f32(&b[(j + 2) * col_b + k]));
                    *acc2 = vaddq_f32(*acc2, val);

                    val = vmulq_f32(aik, vld1q_f32(&b[(j + 3) * col_b + k]));
                    *acc3 = vaddq_f32(*acc3, val);
                }

                #endif
                
                c[i * col_c + j] = Cij0[0] + Cij0[1] + Cij0[2] + Cij0[3];
                c[i * col_c + j + 1] = Cij1[0] + Cij1[1] + Cij1[2] + Cij1[3];
                c[i * col_c + j + 2] = Cij2[0] + Cij2[1] + Cij2[2] + Cij2[3];
                c[i * col_c + j + 3] = Cij3[0] + Cij3[1] + Cij3[2] + Cij3[3];
            }
        }
    }
    
    void *fast_thread_func(void *args) {
        struct thread_args *mat_args = (struct thread_args*)args;

        // initialize
        int ti, tj;
        int start_i = mat_args->start_i, end_i = mat_args->end_i;
        int BLOCK_SIZE = mat_args->block_size;
        const struct matrix *A = mat_args->A, *B = mat_args->B, *C = mat_args->C;
        float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
        
        // sanity check
        assert((end_i - start_i) % BLOCK_SIZE == 0);
        assert(C->column % BLOCK_SIZE == 0);
        assert(BLOCK_SIZE % 4 == 0);

        // Tiling (i, j)
        for (ti = start_i; ti < end_i; ti += BLOCK_SIZE) {
            for (tj = 0; tj < C->column; tj += BLOCK_SIZE) {
                matmul_func(
                    &data_A[ti * A->column],
                    &data_B[tj * B->column],
                    &data_C[ti * C->column + tj],
                    A->column,
                    B->column,
                    C->column,
                    BLOCK_SIZE
                );
            }
        }

        return NULL;
    }


    void MatmulOperator::mat_mul_fast(const struct matmul_params *params) {
        // initialize
        int j, num_thread = params->opt_params.num_thread;
        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        pthread_t thread_pool[num_thread];
        struct thread_args thread_args[num_thread];

        // sanity check (Note. B is transposed original B)
        assert(A->column == B->column);
        assert(C->row == A->row);
        assert(C->column == B->row);
        assert(num_thread != 0);
        assert(C->row % num_thread == 0);

        // create threads
        for (j = 0; j < num_thread; j++) {
            thread_args[j].start_i = j * (C->row / num_thread);
            thread_args[j].end_i = (j + 1) * (C->row / num_thread);
            thread_args[j].block_size = params->opt_params.block_size;

            thread_args[j].A = A;
            thread_args[j].B = B;
            thread_args[j].C = C;

            pthread_create(&thread_pool[j], NULL, fast_thread_func, &thread_args[j]);
        }

        // join threads
        for (j = 0; j < num_thread; j++)
            pthread_join(thread_pool[j], NULL);
    }
}