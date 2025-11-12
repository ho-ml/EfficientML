#include "matmul.h"
#include <stdio.h>
#include <assert.h>

namespace matmul {
    void MatmulOperator::mat_mul_tiling(const struct matmul_params *params) {
        // initialize
        int i, j, k, ti, tj, tk;
        int BLOCK_SIZE = params->opt_params.block_size;
        float Aik;
        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;
        
        // sanity check
        CHECK_MATRICES(A, B, C);
        assert(C->row % BLOCK_SIZE == 0);
        assert(C->column % BLOCK_SIZE == 0);
        assert(B->row % BLOCK_SIZE == 0);

        // clear output matrix C
        for (i = 0; i < C->row; i++)
            for (j = 0; j < C->column; j++)
                data_C[i * C->column + j] = 0;

        // implementation
        // Tiling
        for (ti = 0; ti < C->row; ti += BLOCK_SIZE) {
            for (tk = 0; tk < B->row; tk += BLOCK_SIZE) {
                for (tj = 0; tj < C->column; tj += BLOCK_SIZE) {
                    
                    // matrix multiplication (BLOCK_SIZE * BLOCK_SIZE)
                    for (i = ti; i < ti + BLOCK_SIZE; i++)
                        for (k = tk; k < tk + BLOCK_SIZE; k++) {
                            Aik = data_A[i * A->column + k];
                            for (j = tj; j < tj + BLOCK_SIZE; j++)
                                data_C[i * C->column + j] += Aik * data_B[k * B->column + j];
                        }

                }
            }
        }
        
    }
}