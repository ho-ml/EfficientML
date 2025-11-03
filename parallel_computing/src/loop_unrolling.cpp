#include "matmul.h"
#include <stdio.h>

namespace matmul {
    void MatmulOperator::mat_mul_unrolling(const struct matmul_params *params) {
        // initialize
        int i, j, k;
        const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
        float *data_A = A->data_ptr, *data_B = B->data_ptr, *data_C = C->data_ptr;

        // sanity check
        CHECK_MATRICES(A, B, C);
        
        // implementation
        for (i = 0; i < C->row; i++)
            for (j = 0; j < C->column; j += 4) {
                float Cij0 = 0, Cij1 = 0, Cij2 = 0, Cij3 = 0;

                for (k = 0; k < B->row; k += 4) {
                    float Aik0 = data_A[i * A->column + k];
                    float Aik1 = data_A[i * A->column + k + 1];
                    float Aik2 = data_A[i * A->column + k + 2];
                    float Aik3 = data_A[i * A->column + k + 3];

                    Cij0 += Aik0 * data_B[k * B->column + j];
                    Cij0 += Aik1 * data_B[(k + 1) * B->column + j];
                    Cij0 += Aik2 * data_B[(k + 2) * B->column + j];
                    Cij0 += Aik3 * data_B[(k + 3) * B->column + j];

                    Cij1 += Aik0 * data_B[k * B->column + j + 1];
                    Cij1 += Aik1 * data_B[(k + 1) * B->column + j + 1];
                    Cij1 += Aik2 * data_B[(k + 2) * B->column + j + 1];
                    Cij1 += Aik3 * data_B[(k + 3) * B->column + j + 1];

                    Cij2 += Aik0 * data_B[k * B->column + j + 2];
                    Cij2 += Aik1 * data_B[(k + 1) * B->column + j + 2];
                    Cij2 += Aik2 * data_B[(k + 2) * B->column + j + 2];
                    Cij2 += Aik3 * data_B[(k + 3) * B->column + j + 2];

                    Cij3 += Aik0 * data_B[k * B->column + j + 3];
                    Cij3 += Aik1 * data_B[(k + 1) * B->column + j + 3];
                    Cij3 += Aik2 * data_B[(k + 2) * B->column + j + 3];
                    Cij3 += Aik3 * data_B[(k + 3) * B->column + j + 3];
                }

                data_C[i * C->column + j] = Cij0;
                data_C[i * C->column + j + 1] = Cij1;
                data_C[i * C->column + j + 2] = Cij2;
                data_C[i * C->column + j + 3] = Cij3;
            }
    }
}