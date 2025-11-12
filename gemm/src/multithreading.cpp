#include "matmul.h"
#include <stdio.h>
#include <assert.h>
#include <pthread.h>

namespace matmul {
    void *thread_func(void *args) {
        struct thread_args *mat_args = (struct thread_args*)args;

        // initialize
        int i, j, k;
        int start_i = mat_args->start_i, end_i = mat_args->end_i;
        const struct matrix *A = mat_args->A, *B = mat_args->B, *C = mat_args->C;
        float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

        // implementation
        for (i = start_i; i < end_i; i++)
            for (j = 0; j < C->column; j++) {
                float Cij = 0;

                for (k = 0; k < B->row; k++)
                    Cij += data_A[i * A->column + k] * data_B[k * B->column + j];

                data_C[i * C->column + j] = Cij;
            }
        
        return NULL;
    }

    void MatmulOperator::mat_mul_multithreading(const struct matmul_params *params) {
        // initialize
        int j, num_thread = params->opt_params.num_thread;
        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        pthread_t thread_pool[num_thread];
        struct thread_args thread_args[num_thread];

        // sanity check
        CHECK_MATRICES(A, B, C);
        assert(num_thread != 0);
        assert(C->row % num_thread == 0);

        // create threads
        for (j = 0; j < num_thread; j++) {
            thread_args[j].start_i = j * (C->row / num_thread);
            thread_args[j].end_i = (j + 1) * (C->row / num_thread);

            thread_args[j].A = A;
            thread_args[j].B = B;
            thread_args[j].C = C;

            pthread_create(&thread_pool[j], NULL, thread_func, &thread_args[j]);
        }

        // join threads (모든 thread가 끝날 때까지 대기)
        for (j = 0; j < num_thread; j++)
            pthread_join(thread_pool[j], NULL);
    }
}