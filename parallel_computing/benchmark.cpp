#include "matmul.h"

#include <stdio.h>
#include <math.h>
#include <iostream>

#define BLOCK_SIZE 32
#define MAX_PRECISION_ERROR 0.01

#define A_ROW 640
#define A_COLUMN 12800
#define B_ROW 12800
#define B_COLUMN 640
#define C_ROW 640
#define C_COLUMN 640
#define NUM_THREAD 4

float matrix_A[A_ROW * A_COLUMN];
float matrix_B[B_ROW * B_COLUMN];
float matrix_BT[B_COLUMN * B_ROW];
float native_C[C_ROW * C_COLUMN];
float output_C[C_ROW * C_COLUMN];

bool check_identical(float matA[], float matB[], int size) {
    // matA 배열과 matB 배열의 값이 동일한지 비교 (오차: 1%)
    for (int i = 0; i < size; i++) {
        if (abs((matA[i] - matB[i]) / matA[i]) > MAX_PRECISION_ERROR) {
            printf("%f, %f", matA[i], matB[i]);
            return false;
        }
    }

    return true;
}

void initialize_matrix(float A[], int size) {
    // 0.0 ~ 1.0 사이의 랜덤 float 값으로 초기화
    for (int i = 0; i < size; i++) {
        A[i] = (float)(rand()) / (float)(RAND_MAX);
    }
}

using namespace matmul;

bool run_switch(std::string target, std::string type) {
    // 최적화 기법 설정
    if (target == "ALL" || target == type)
        return true;
    
    return false;
}

int main(int argc, char* argv[]) {
    auto target = "ALL";
    if (argc == 2)
        target = argv[1];

    // 행렬 랜덤 값으로 초기화
    initialize_matrix(matrix_A, A_ROW * A_COLUMN);
    initialize_matrix(matrix_B, B_ROW * B_COLUMN);
    initialize_matrix(native_C, C_ROW * C_COLUMN);
    
    // 객체 생성
    MatmulOperator matmul_op = MatmulOperator();

    // 행렬 곱 파라미터 설정
    struct matmul_params params;
    params.A.row = A_ROW; params.A.column = A_COLUMN; params.A.data_ptr = matrix_A;
    params.B.row = B_ROW; params.B.column = B_COLUMN; params.B.data_ptr = matrix_B;
    params.C.row = C_ROW; params.C.column = C_COLUMN;

    // 최적화 옵션 설정
    params.opt_params.block_size = BLOCK_SIZE; params.opt_params.num_thread = NUM_THREAD;

    // baseline
    params.C.data_ptr = native_C;
    matmul_op.evaluate(MatmulOperator::NAIVE, &params);
    params.C.data_ptr = output_C;

    // reordering
    if (run_switch(target, "loop_reordering")) {
        matmul_op.evaluate(MatmulOperator::REORDER, &params);
        if (!check_identical(native_C, output_C, C_ROW * C_COLUMN))
            printf("incorrect output of mat_mul_reordering\n");
    }

    // tiliing
    if (run_switch(target, "loop_tiling")) {
        matmul_op.evaluate(MatmulOperator::TILING, &params);
        if (!check_identical(native_C, output_C, C_ROW * C_COLUMN))
            printf("incorrect output of mat_mul_tiling\n");
    }
    
    // unrolling
    if (run_switch(target, "loop_unrolling")) {
        matmul_op.evaluate(MatmulOperator::UNROLL, &params);
        if (!check_identical(native_C, output_C, C_ROW * C_COLUMN))
            printf("incorrect output of mat_mul_unrolling\n");
    }

    // simd
    if (run_switch(target, "SIMD_programming")){
        matmul_op.evaluate(MatmulOperator::TRANSPOSE_SIMD, &params);
        if (!check_identical(native_C, output_C, C_ROW * C_COLUMN))
            printf("incorrect output of mat_mul_transpose_simd\n");
    }

    // multi-threading
    if (run_switch(target, "multithreading")){
        matmul_op.evaluate(MatmulOperator::MULTITHREAD, &params);
        if (!check_identical(native_C, output_C, C_ROW * C_COLUMN))
            printf("incorrect output of mat_mul_multithreading\n");
    }

    #ifdef CUDA_ENABLE
        // cuda
        if (run_switch(target, "CUDA")){
            matmul_op.evaluate(MatmulOperator::CUDA, &params);
            if (!check_identical(native_C, output_C, C_ROW * C_COLUMN))
                printf("incorrect output of mat_mul_cuda\n");
        }
    #endif
        // transpose B (spatial locality 향상)
        for (int i = 0; i < B_COLUMN; i++)
            for (int j = 0; j < B_ROW; j++)
                matrix_BT[i * B_ROW + j] = matrix_B[j * B_COLUMN + i];
        params.B.row = B_COLUMN; params.B.column = B_ROW; params.B.data_ptr = matrix_BT;
        params.opt_params.block_size = 4; params.opt_params.num_thread = 4;

        // fast
        if (run_switch(target, "fast")){
            matmul_op.evaluate(MatmulOperator::FAST, &params);
            if (!check_identical(native_C, output_C, C_ROW * C_COLUMN))
                printf("incorrect output of mat_mul_fast\n");
        }

    return 0;
}